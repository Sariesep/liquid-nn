"""
Mixture of Experts Router — Dinamik Katman Seçimi

Her token için hangi expert katmanların (LiquidODECell)
kullanılacağına karar verir. Top-k routing ile kolay tokenler
sadece hızlı katmanlardan, zor tokenler derin katmanlardan geçer.

Load balancing aux loss ile tüm expert'lerin dengeli
kullanılması sağlanır (Switch Transformer, 2021).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertRouter(nn.Module):
    """
    Token seviyesinde expert seçici (top-k gating).

    Her token vektörü için num_experts arasından top_k tanesini seçer
    ve ağırlıklandırılmış çıktı üretir.

    Args:
        embed_dim:    Gömme boyutu
        num_experts:  Toplam expert (katman) sayısı
        top_k:        Her token için seçilecek expert sayısı
        noise_std:    Routing'e eklenen gürültü (eğitimde keşif için)
    """

    def __init__(self, embed_dim: int, num_experts: int,
                 top_k: int = 2, noise_std: float = 0.1):
        super().__init__()
        assert top_k <= num_experts, \
            f"top_k ({top_k}) num_experts'ten ({num_experts}) büyük olamaz"

        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std

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
            weights:  [B, top_k] seçilen expertlerin ağırlıkları (softmax)
            indices:  [B, top_k] seçilen expert indeksleri
            aux_loss: skaler load balancing kaybı
        """
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

        return weights, top_idx, aux_loss

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
