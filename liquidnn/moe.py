"""
Mixture of Experts Router — Dinamik Katman Seçimi

Her token için hangi expert katmanların (LiquidODECell)
kullanılacağına karar verir. Top-k routing ile kolay tokenler
sadece hızlı katmanlardan, zor tokenler derin katmanlardan geçer.

Load balancing aux loss ile tüm expert'lerin dengeli
kullanılması sağlanır (Switch Transformer, 2021).

v0.3.4: Expert Capacity Limiting — her expert'in işleyebileceği
        token sayısını sınırlayarak dengeli dağılım zorunlu kılar.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertRouter(nn.Module):
    """
    Token seviyesinde expert seçici (top-k gating).

    Her token vektörü için num_experts arasından top_k tanesini seçer
    ve ağırlıklandırılmış çıktı üretir.

    Args:
        embed_dim:        Gömme boyutu
        num_experts:      Toplam expert (katman) sayısı
        top_k:            Her token için seçilecek expert sayısı
        noise_std:        Routing'e eklenen gürültü (eğitimde keşif için)
        capacity_factor:  Expert kapasite çarpanı (0 = sınırsız, >0 = aktif)
    """

    def __init__(self, embed_dim: int, num_experts: int,
                 top_k: int = 2, noise_std: float = 0.1,
                 capacity_factor: float = 0.0):
        super().__init__()
        assert top_k <= num_experts, \
            f"top_k ({top_k}) num_experts'ten ({num_experts}) büyük olamaz"

        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.capacity_factor = capacity_factor

        # Router MLP: token vektöründen expert skorlarına
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.SiLU(),
            nn.Linear(embed_dim // 2, num_experts),
        )

    def forward(self, x: torch.Tensor):
        """
        Token vektöründen top-k expert seçimi.

        Args:
            x: [B, D] token gömmesi

        Returns:
            weights:      [B, top_k] seçilen expertlerin ağırlıkları (softmax)
            indices:      [B, top_k] seçilen expert indeksleri
            aux_loss:     skaler load balancing kaybı
            dropped_mask: [B, top_k] bool, True = token bu expert'e gönderilmeyecek
                          (capacity_factor=0 ise None)
        """
        B = x.size(0)

        # Router skorları
        logits = self.gate(x)  # [B, num_experts]

        # Eğitimde gürültü ekle (keşif için)
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        # Top-k seçim
        top_vals, top_idx = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(top_vals, dim=-1)  # [B, top_k]

        # Load balancing auxiliary loss
        # (her expert'in eşit kullanılmasını teşvik eder)
        aux_loss = self._load_balance_loss(logits)

        # ── Expert Capacity Limiting ──────────────────────────────
        dropped_mask = None
        if self.capacity_factor > 0 and B > 0:
            capacity = max(1, math.ceil(
                B * self.top_k / self.num_experts * self.capacity_factor))
            expert_counts = torch.zeros(self.num_experts, dtype=torch.long,
                                        device=x.device)
            dropped_mask = torch.zeros(B, self.top_k, dtype=torch.bool,
                                       device=x.device)
            for k_idx in range(self.top_k):
                for b in range(B):
                    eidx = top_idx[b, k_idx].item()
                    if expert_counts[eidx] >= capacity:
                        dropped_mask[b, k_idx] = True
                    else:
                        expert_counts[eidx] += 1

        return weights, top_idx, aux_loss, dropped_mask

    def _load_balance_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Switch Transformer tarzı load balancing kaybı.

        L_balance = N * Σ_i (f_i * P_i)
        f_i: expert i'ye yönlendirilen tokenların oranı
        P_i: tüm tokenlerden expert i'nin ortalama olasılığı
        """
        probs = F.softmax(logits, dim=-1)     # [B, num_experts]
        # Her expert'e atanan tokenların oranı
        # (top-1 ile basitleştirilmiş)
        top1_idx = logits.argmax(dim=-1)       # [B]
        freq = torch.zeros(self.num_experts, device=logits.device)
        for i in range(self.num_experts):
            freq[i] = (top1_idx == i).float().mean()

        # Her expert'in ortalama routing olasılığı
        avg_prob = probs.mean(dim=0)           # [num_experts]

        # Load balance loss (düşük = dengeli)
        loss = self.num_experts * (freq * avg_prob).sum()
        return loss
