"""
MiniLiquidGPT — Sıvı Nöral Ağ Dil Modeli

4 katmanlı hibrit yapı:
  Katman 0-1: ode_steps=1 → hızlı algı (Euler, plastisite OFF)
  Katman 2-3: ode_steps=3 → derin düşünce (RK2, Hebb ON)

Ağırlık paylaşımı: embed ↔ lm_head
Pozisyon kodlaması: Sinüzoidal (sabit)

Eklentiler:
  - Sliding-Window Attention (isteğe bağlı)
  - Mixture of Experts Router (isteğe bağlı)
  - Speculative Decoding (adaptive gamma + stochastic acceptance)
"""

import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ode_cell import LiquidODECell
from .attention import SlidingWindowAttention
from .moe import ExpertRouter
from .rmsnorm import RMSNorm


def _greedy_pick(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Logits'ten greedy (argmax) token seçimi.

    Args:
        logits: [B, V] logit tensörü
        temperature: Sıcaklık (burada greedy olduğu için sadece scaling)

    Returns:
        [B, 1] seçilen token ID'leri
    """
    if temperature != 1.0 and temperature > 0:
        logits = logits / temperature
    return logits.argmax(dim=-1, keepdim=True)


def _speculative_accept(main_logits: torch.Tensor,
                        draft_logits: torch.Tensor,
                        draft_token: torch.Tensor,
                        temperature: float = 0.8):
    """
    Speculative Sampling kabul/red kararı (Leviathan et al., 2023).

    p(x) >= q(x) → her zaman kabul
    p(x) <  q(x) → p(x)/q(x) olasılıkla kabul
    Reject → max(0, p - q) dağılımından düzeltme tokeni seç

    Args:
        main_logits:  [B, V] ana modelin logit'leri
        draft_logits: [B, V] draft modelin logit'leri
        draft_token:  [B, 1] draft'ın seçtiği token
        temperature:  Sıcaklık

    Returns:
        (accepted: bool, correction_token: [B,1] or None)
    """
    p = F.softmax(main_logits / temperature, dim=-1)    # ana model dağılımı
    q = F.softmax(draft_logits / temperature, dim=-1)    # draft dağılımı

    tok_idx = draft_token.squeeze(-1)                     # [B]
    p_x = p[0, tok_idx[0]]                                # p(draft_token)
    q_x = q[0, tok_idx[0]]                                # q(draft_token)

    # Kabul olasılığı
    if q_x.item() == 0:
        accept_prob = 1.0
    else:
        accept_prob = min(1.0, p_x.item() / q_x.item())

    r = torch.rand(1).item()
    if r < accept_prob:
        return True, None
    else:
        # Reject → düzeltilmiş dağılımdan sample
        corrected = torch.clamp(p - q, min=0)
        corrected_sum = corrected.sum()
        if corrected_sum > 0:
            corrected = corrected / corrected_sum
        else:
            corrected = p  # fallback
        correction_token = torch.multinomial(corrected, 1)  # [B, 1]
        return False, correction_token


class AdaptiveGammaScheduler:
    """
    Kabul oranına göre gamma'yı otomatik ayarlayan scheduler.

    Yüksek kabul oranı → gamma artır (daha agresif drafting)
    Düşük kabul oranı  → gamma azalt (daha temkinli)

    EMA (Exponential Moving Average) ile yumuşak geçiş sağlar.

    Args:
        initial:    Başlangıç gamma değeri
        min_gamma:  Minimum gamma
        max_gamma:  Maksimum gamma
        ema_alpha:  EMA düzleştirme katsayısı (0-1, yüksek = hızlı tepki)
        target_low: Bu oranın altında gamma azalır
        target_high: Bu oranın üstünde gamma artar
    """

    def __init__(self, initial: int = 4, min_gamma: int = 1,
                 max_gamma: int = 8, ema_alpha: float = 0.3,
                 target_low: float = 0.3, target_high: float = 0.7):
        self._gamma = float(initial)
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.ema_alpha = ema_alpha
        self.target_low = target_low
        self.target_high = target_high
        self._acceptance_ema = 0.5  # başlangıç tahmini

    def update(self, n_accepted: int, n_proposed: int):
        """Kabul oranıyla gamma'yı güncelle."""
        if n_proposed == 0:
            return
        rate = n_accepted / n_proposed
        self._acceptance_ema = (self.ema_alpha * rate +
                                (1 - self.ema_alpha) * self._acceptance_ema)

        if self._acceptance_ema > self.target_high:
            self._gamma = min(self._gamma + 1.0, self.max_gamma)
        elif self._acceptance_ema < self.target_low:
            self._gamma = max(self._gamma - 1.0, self.min_gamma)

    @property
    def gamma(self) -> int:
        return int(self._gamma)

    @property
    def acceptance_rate(self) -> float:
        return self._acceptance_ema

class MiniLiquidGPT(nn.Module):
    """
    Sıvı Nöral Ağ Dil Modeli.

    Args:
        vocab_size:     Kelime hazinesi boyutu (tiktoken gpt2 = 50257)
        embed_dim:      Gömme boyutu
        num_fast:       Hızlı katman sayısı (steps=1, plast OFF)
        num_deep:       Derin katman sayısı (steps=3, plast ON)
        fast_steps:     Hızlı katman ODE adım sayısı
        deep_steps:     Derin katman ODE adım sayısı
        dropout:        Dropout oranı
        max_seq:        Maksimum sekans uzunluğu (pozisyon kodlaması)
        use_attention:  Sliding-Window Attention aktif mi
        attn_heads:     Attention başı sayısı
        attn_window:    Attention pencere boyutu
        use_moe:        Mixture of Experts Router aktif mi
        moe_top_k:      MoE: her token için seçilecek expert sayısı
    """

    def __init__(self, vocab_size: int = 50257, embed_dim: int = 256,
                 num_fast: int = 2, num_deep: int = 2,
                 fast_steps: int = 1, deep_steps: int = 3,
                 dropout: float = 0.1, max_seq: int = 512,
                 use_attention: bool = False, attn_heads: int = 4,
                 attn_window: int = 64,
                 use_moe: bool = False, moe_top_k: int = 2,
                 use_rmsnorm: bool = False,
                 adaptive_ode: bool = False,
                 early_exit_threshold: float = 0.0,
                 use_rope: bool = False,
                 use_flash: bool = True,
                 use_multiscale: bool = False,
                 tau_gate: bool = False):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_fast + num_deep
        self.use_attention = use_attention
        self.use_moe = use_moe
        self.use_rmsnorm = use_rmsnorm
        self.adaptive_ode = adaptive_ode
        self.early_exit_threshold = early_exit_threshold
        self.use_multiscale = use_multiscale
        self.tau_gate = tau_gate

        # Norm sınıfı seçimi
        NormClass = RMSNorm if use_rmsnorm else nn.LayerNorm

        # Embedding + pozisyon
        self.embed = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.embed_drop = nn.Dropout(dropout)

        pe = self._make_sinusoidal_pe(max_seq, embed_dim)
        self.register_buffer('pos_enc', pe)

        # Katmanlar
        self.cells = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.drops = nn.ModuleList()

        for _ in range(num_fast):
            self.cells.append(LiquidODECell(embed_dim, embed_dim,
                                            ode_steps=fast_steps,
                                            use_plasticity=False))
            self.norms.append(NormClass(embed_dim))
            self.drops.append(nn.Dropout(dropout))

        for _ in range(num_deep):
            self.cells.append(LiquidODECell(embed_dim, embed_dim,
                                            ode_steps=deep_steps,
                                            use_plasticity=True))
            self.norms.append(NormClass(embed_dim))
            self.drops.append(nn.Dropout(dropout))

        self.out_norm = NormClass(embed_dim)

        # Sliding-Window Attention (isteğe bağlı)
        self.attn = None
        if use_attention:
            self.attn = SlidingWindowAttention(
                embed_dim, num_heads=attn_heads,
                window_size=attn_window, dropout=dropout,
                use_rope=use_rope, use_flash=use_flash
            )

        # MoE Router (isteğe bağlı)
        self.router = None
        self._aux_loss = torch.tensor(0.0)  # MoE load balance loss
        if use_moe and self.num_layers > 1:
            self.router = ExpertRouter(
                embed_dim, num_experts=self.num_layers,
                top_k=min(moe_top_k, self.num_layers)
            )

        self._hiddens: Optional[List[torch.Tensor]] = None

        # Multi-Scale Fusion: fast ve deep çıktılarını birleştiren gate
        self._fusion_gate = None
        if use_multiscale and num_fast > 0 and num_deep > 0:
            self._fusion_gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Sigmoid()
            )

    @staticmethod
    def _make_sinusoidal_pe(max_len: int, dim: int) -> torch.Tensor:
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float)
                        * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)  # [1, max_len, D]

    def init_hidden(self, batch_size: int, device: torch.device):
        """Gizli durumları ve attention buffer'ı sıfırla."""
        self._hiddens = [
            torch.zeros(batch_size, self.embed_dim, device=device)
            for _ in range(self.num_layers)
        ]
        if self.attn is not None:
            self.attn.reset_buffer()
        self._aux_loss = torch.tensor(0.0, device=device)

    def forward_token(self, token_id: torch.Tensor, pos: int,
                      enable_plasticity: bool = True) -> torch.Tensor:
        """
        Tek token işle.

        Args:
            token_id: [B] token ID'leri
            pos: Sekans pozisyonu
            enable_plasticity: Hebb güncellemesi aktif mi

        Returns:
            [B, V] logits
        """
        if token_id.dim() > 1:
            token_id = token_id.squeeze(-1)

        B = token_id.size(0)
        if self._hiddens is None:
            self.init_hidden(B, token_id.device)

        x = self.embed(token_id)
        if pos < self.pos_enc.size(1):
            x = x + self.pos_enc[0, pos]

        if self.router is not None:
            # MoE: top-k expert seçimi
            weights, indices, aux_loss = self.router(x)
            self._aux_loss = self._aux_loss + aux_loss
            out = torch.zeros_like(x)
            for k_idx in range(weights.size(1)):
                for b in range(B):
                    eidx = indices[b, k_idx].item()
                    cell = self.cells[eidx]
                    norm = self.norms[eidx]
                    drop = self.drops[eidx]
                    h_new = cell(x[b:b+1], self._hiddens[eidx][b:b+1],
                                 enable_plasticity,
                                 adaptive_steps=self.adaptive_ode,
                                 tau_gate_residual=self.tau_gate,
                                 moe_weight=weights[b, k_idx].item())
                    h_new = norm(h_new + x[b:b+1])
                    h_new = drop(h_new)

                    # In-place hatasını önlemek için clone() kullan
                    new_h = self._hiddens[eidx].clone()
                    new_h[b:b+1] = h_new
                    self._hiddens[eidx] = new_h

                    out[b:b+1] += weights[b, k_idx] * h_new
            x = out
        else:
            # Multi-Scale Fusion veya standart sıralı katmanlar
            if self.use_multiscale and self._fusion_gate is not None:
                # Fast ve deep katmanları paralel çalıştır
                fast_cells = [(i, c, n, d) for i, (c, n, d) in enumerate(
                    zip(self.cells, self.norms, self.drops))
                    if not c.use_plasticity]
                deep_cells = [(i, c, n, d) for i, (c, n, d) in enumerate(
                    zip(self.cells, self.norms, self.drops))
                    if c.use_plasticity]

                # Fast path
                x_fast = x
                for i, cell, norm, drop in fast_cells:
                    h_new = cell(x_fast, self._hiddens[i], enable_plasticity,
                                 adaptive_steps=self.adaptive_ode,
                                 tau_gate_residual=self.tau_gate)
                    h_new = norm(h_new + x_fast)
                    h_new = drop(h_new)
                    self._hiddens[i] = h_new
                    x_fast = h_new

                # Deep path
                x_deep = x
                for i, cell, norm, drop in deep_cells:
                    h_new = cell(x_deep, self._hiddens[i], enable_plasticity,
                                 adaptive_steps=self.adaptive_ode,
                                 tau_gate_residual=self.tau_gate)
                    h_new = norm(h_new + x_deep)
                    h_new = drop(h_new)
                    self._hiddens[i] = h_new
                    x_deep = h_new

                # Fusion: fast ve deep çıktılarını birleştir
                gate = self._fusion_gate(torch.cat([x_fast, x_deep], dim=-1))
                x = gate * x_fast + (1 - gate) * x_deep

            else:
                # Standart sıralı katmanlar
                for i, (cell, norm, drop) in enumerate(
                        zip(self.cells, self.norms, self.drops)):
                    h_new = cell(x, self._hiddens[i], enable_plasticity,
                                 adaptive_steps=self.adaptive_ode,
                                 tau_gate_residual=self.tau_gate)
                    h_new = norm(h_new + x)
                    h_new = drop(h_new)
                    self._hiddens[i] = h_new
                    x = h_new

                    # Early Exit
                    if (self.early_exit_threshold > 0 and
                            not self.training and
                            i < self.num_layers - 1):
                        early_logits = F.linear(self.out_norm(x),
                                                self.embed.weight)
                        confidence = F.softmax(early_logits, dim=-1).max(dim=-1).values.mean()
                        if confidence.item() > self.early_exit_threshold:
                            break

        # Sliding-Window Attention (etkinse)
        if self.attn is not None:
            x = self.attn(x, pos=pos)

        x = self.out_norm(x)
        logits = F.linear(x, self.embed.weight)  # Weight tying
        return logits

    def _forward_token_for_ckpt(self, token_id, pos, enable_plasticity):
        """Gradient checkpointing uyumlu forward_token wrapper."""
        return self.forward_token(token_id, pos, enable_plasticity)

    def forward(self, input_ids: torch.Tensor,
                enable_plasticity: bool = True,
                chunk_size: int = 16,
                use_checkpointing: bool = False) -> torch.Tensor:
        """
        Sekans işle (eğitim modu).

        Args:
            input_ids: [B, T] token ID'leri
            enable_plasticity: Hebb aktif mi
            chunk_size: Truncated BPTT parça boyutu
            use_checkpointing: Agresif bellek optimizasyonu
                               (daha sık detach → VRAM %40-60 düşüş,
                                eğitim hızında %10-15 kayıp)

        Returns:
            [B, T, V] logits
        """
        B, T = input_ids.shape
        self.init_hidden(B, input_ids.device)

        # Checkpointing: daha sık detach = daha az VRAM tüketimi
        detach_freq = max(1, chunk_size // 2) if use_checkpointing else chunk_size

        all_logits = []
        for t in range(T):
            # Truncated BPTT (agresif mod: her detach_freq adımda)
            if t > 0 and t % detach_freq == 0:
                self._hiddens = [h.detach() for h in self._hiddens]
                for cell in self.cells:
                    cell.detach_hebb()

            logits = self.forward_token(input_ids[:, t], t, enable_plasticity)
            all_logits.append(logits)

        return torch.stack(all_logits, dim=1)

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new: int = 80,
                 temperature: float = 0.8, top_k: int = 40,
                 top_p: float = 0.0, repetition_penalty: float = 1.0,
                 enable_plasticity: bool = True) -> torch.Tensor:
        """
        Metin üret.

        Args:
            prompt_ids: [T] veya [1, T] token ID'leri
            max_new: Üretilecek token sayısı
            temperature: Örnekleme sıcaklığı
            top_k: Top-k filtreleme (0 = devre dışı)
            top_p: Top-p (nucleus) filtreleme (0 = devre dışı)
            repetition_penalty: Tekrar cezası (1.0 = devre dışı, >1 = ceza)
            enable_plasticity: Hebb aktif mi

        Returns:
            [1, T+max_new] tüm token ID'leri
        """
        self.eval()
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)

        B = prompt_ids.size(0)
        self.init_hidden(B, prompt_ids.device)
        self.reset_hebb()

        ids = prompt_ids.clone()

        # Prompt'ı işle
        for t in range(ids.size(1)):
            logits = self.forward_token(ids[:, t], t, enable_plasticity)

        # Üret
        for step in range(max_new):
            pos = ids.size(1)

            # Repetition Penalty: önceki tokenlerin logit'lerini cezalandır
            if repetition_penalty != 1.0:
                for b in range(B):
                    prev_tokens = ids[b].unique()
                    for tok_id in prev_tokens:
                        if logits[b, tok_id] > 0:
                            logits[b, tok_id] /= repetition_penalty
                        else:
                            logits[b, tok_id] *= repetition_penalty

            scaled = logits / temperature

            # Top-k filtreleme
            if top_k > 0:
                v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                scaled[scaled < v[:, [-1]]] = -float('inf')

            # Top-p (nucleus) filtreleme
            if top_p > 0.0:
                sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
                cum_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1)
                # Eşik üzerindeki tokenleri kaldır
                remove_mask = cum_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[remove_mask] = -float('inf')
                # Orijinal sıralamaya geri dön
                scaled = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(scaled, dim=-1)
            next_id = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_id], dim=1)
            logits = self.forward_token(next_id.squeeze(-1), pos,
                                        enable_plasticity)

        return ids

    def reset_hebb(self):
        """Tüm plastik izleri sıfırla."""
        for cell in self.cells:
            cell.reset_hebb()

    # ── State Management (Speculative Decoding için) ──────────────

    def save_state(self) -> dict:
        """
        Modelin tüm mutable durumunu snapshot olarak döndür.

        Kopyalanan durumlar:
          - self._hiddens  (katman başına gizli durum tensörleri)
          - Her LiquidODECell'in syn_ih.Hebb ve syn_hh.Hebb matrisleri
          - Attention buffer (etkinse)

        Tüm tensörler .detach().clone() ile hesaplama grafiğinden koparılır;
        böylece geri yükleme sırasında VRAM sızıntısı önlenir.
        """
        saved_hiddens = None
        if self._hiddens is not None:
            saved_hiddens = [h.detach().clone() for h in self._hiddens]

        saved_hebbs = []
        for cell in self.cells:
            pair = {}
            pair['ih'] = (cell.syn_ih.Hebb.detach().clone()
                          if cell.syn_ih.Hebb is not None else None)
            pair['hh'] = (cell.syn_hh.Hebb.detach().clone()
                          if cell.syn_hh.Hebb is not None else None)
            saved_hebbs.append(pair)

        saved_attn = None
        if self.attn is not None:
            saved_attn = self.attn.get_buffer_state()

        return {'hiddens': saved_hiddens, 'hebbs': saved_hebbs,
                'attn': saved_attn}

    def restore_state(self, state_dict: dict):
        """
        save_state() ile kaydedilmiş durumu geri yükle.

        Reject durumunda ana modeli kabul edilen son tokenin
        state'ine geri döndürmek için kullanılır.
        """
        if state_dict['hiddens'] is not None:
            self._hiddens = [h.detach().clone()
                             for h in state_dict['hiddens']]
        else:
            self._hiddens = None

        for cell, pair in zip(self.cells, state_dict['hebbs']):
            cell.syn_ih.Hebb = (pair['ih'].detach().clone()
                                if pair['ih'] is not None else None)
            cell.syn_hh.Hebb = (pair['hh'].detach().clone()
                                if pair['hh'] is not None else None)

        if self.attn is not None and state_dict.get('attn') is not None:
            self.attn.set_buffer_state(state_dict['attn'])

    # ── Draft Model Factory ───────────────────────────────────────

    @classmethod
    def create_draft_model(cls, target_model: 'MiniLiquidGPT',
                           device: torch.device = None) -> 'MiniLiquidGPT':
        """
        Ana modelden hafif bir draft model oluştur.

        - 1 hızlı katman (Euler, PlasticSynapse OFF)
        - Embedding ağırlıkları ana modelden paylaşılır (VRAM tasarrufu)

        Args:
            target_model: Kaynak ana model
            device: Hedef cihaz (None → target_model.embed.weight.device)

        Returns:
            Hafif MiniLiquidGPT draft modeli
        """
        if device is None:
            device = target_model.embed.weight.device

        draft = cls(
            vocab_size=target_model.vocab_size,
            embed_dim=target_model.embed_dim,
            num_fast=1,
            num_deep=0,            # plastisite tamamen kapalı
            fast_steps=1,
            deep_steps=1,
            dropout=0.0,           # draft model'de dropout gereksiz
        ).to(device)

        # Embedding ağırlıklarını paylaş (aynı tensör, kopya değil)
        draft.embed.weight = target_model.embed.weight
        return draft

    # ── Speculative Decoding ──────────────────────────────────────

    @torch.no_grad()
    def generate_speculative(self, draft_model: 'MiniLiquidGPT',
                             prompt_ids: torch.Tensor,
                             max_new: int = 80, gamma: int = 4,
                             temperature: float = 0.8,
                             adaptive_gamma: bool = False,
                             use_stochastic: bool = False) -> torch.Tensor:
        """
        Speculative Decoding ile metin üret.

        Args:
            draft_model:     Hızlı taslak model (plastisitesiz, 1 katman)
            prompt_ids:      [T] veya [1, T] token ID'leri
            max_new:         Üretilecek maksimum yeni token sayısı
            gamma:           Draft modelin her turda üreteceği token sayısı
            temperature:     Örnekleme sıcaklığı
            adaptive_gamma:  True ise gamma kabul oranına göre otomatik ayarlanır
            use_stochastic:  True ise greedy yerine olasılıksal kabul kullanılır
                             (Leviathan et al., 2023)

        Returns:
            [1, T + max_new] tüm token ID'leri
        """
        self.eval()
        draft_model.eval()

        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)

        B = prompt_ids.size(0)
        device = prompt_ids.device

        # Adaptive gamma scheduler
        scheduler = None
        if adaptive_gamma:
            scheduler = AdaptiveGammaScheduler(initial=gamma)

        # ── Adım A: Prompt Sync ──────────────────────────────────
        self.init_hidden(B, device)
        self.reset_hebb()
        draft_model.init_hidden(B, device)
        draft_model.reset_hebb()

        ids = prompt_ids.clone()

        # Prompt'u her iki modele de besle
        for t in range(ids.size(1)):
            main_logits = self.forward_token(ids[:, t], t,
                                             enable_plasticity=True)
            draft_model.forward_token(ids[:, t], t,
                                      enable_plasticity=False)

        generated_count = 0

        # ── Ana Döngü ─────────────────────────────────────────────
        while generated_count < max_new:
            cur_pos = ids.size(1)

            # Gamma belirle
            cur_gamma_val = scheduler.gamma if scheduler else gamma
            remaining = max_new - generated_count
            cur_gamma = min(cur_gamma_val, remaining)

            # ── Adım B: Drafting ──────────────────────────────────
            draft_tokens = []
            draft_logits_list = []   # stochastic kabul için gerekli

            # İlk draft token: main_logits'ten greedy
            first_draft_tok = _greedy_pick(main_logits, temperature)
            draft_tokens.append(first_draft_tok)
            draft_logits_list.append(main_logits.clone())  # ilk token için

            # Draft modelin state'ini ilerlet + geri kalan tokenler
            d_logits = draft_model.forward_token(
                first_draft_tok.squeeze(-1),
                cur_pos - 1, enable_plasticity=False
            )

            for g in range(1, cur_gamma):
                draft_logits_list.append(d_logits.clone())
                next_tok = _greedy_pick(d_logits, temperature)
                draft_tokens.append(next_tok)
                d_logits = draft_model.forward_token(
                    next_tok.squeeze(-1),
                    cur_pos + g - 1,
                    enable_plasticity=False
                )

            # ── Adım C: Verification ─────────────────────────────
            saved_state = self.save_state()

            verified_logits_list = []
            for g in range(cur_gamma):
                v_logits = self.forward_token(
                    draft_tokens[g].squeeze(-1),
                    cur_pos + g,
                    enable_plasticity=True
                )
                verified_logits_list.append(v_logits)

            # ── Adım D: Accept / Reject ──────────────────────────
            n_accepted = 0
            stochastic_correction = None

            for g in range(cur_gamma):
                # Ana modelin bu pozisyondaki logits'i
                if g == 0:
                    target_logits = main_logits
                else:
                    target_logits = verified_logits_list[g - 1]

                if use_stochastic:
                    # Olasılıksal kabul
                    accepted, correction = _speculative_accept(
                        target_logits, draft_logits_list[g],
                        draft_tokens[g], temperature
                    )
                    if accepted:
                        n_accepted += 1
                    else:
                        stochastic_correction = correction
                        break
                else:
                    # Greedy kabul
                    main_choice = _greedy_pick(target_logits, temperature)
                    if main_choice.item() == draft_tokens[g].item():
                        n_accepted += 1
                    else:
                        break

            # Adaptive gamma güncelle
            if scheduler:
                scheduler.update(n_accepted, cur_gamma)

            # ── Adım E: Rollback veya Commit ─────────────────────
            self.restore_state(saved_state)

            # Kabul edilen tokenler için state'i yeniden ilerlet
            accepted_ids = []
            replay_logits = None
            for g in range(n_accepted):
                tok = draft_tokens[g]
                accepted_ids.append(tok)
                replay_logits = self.forward_token(
                    tok.squeeze(-1),
                    cur_pos + g,
                    enable_plasticity=True
                )

            # Bonus / düzeltme tokeni
            if n_accepted < cur_gamma:
                if use_stochastic and stochastic_correction is not None:
                    # Stochastic: düzeltilmiş dağılımdan seçilmiş token
                    bonus = stochastic_correction
                else:
                    # Greedy: ana modelin kendi seçimi
                    if n_accepted == 0:
                        correction_logits = main_logits
                    else:
                        correction_logits = replay_logits
                    bonus = _greedy_pick(correction_logits, temperature)
            else:
                # Tümü kabul → son verified logits'ten bonus token
                if use_stochastic:
                    p = F.softmax(verified_logits_list[-1] / temperature,
                                  dim=-1)
                    bonus = torch.multinomial(p, 1)
                else:
                    bonus = _greedy_pick(verified_logits_list[-1], temperature)

            accepted_ids.append(bonus)
            # Bonus tokeni state'e commit et
            self.forward_token(
                bonus.squeeze(-1),
                cur_pos + n_accepted,
                enable_plasticity=True
            )

            # Kabul edilen tokenları sekansa ekle
            new_tokens = torch.cat(accepted_ids, dim=-1)
            if new_tokens.dim() == 1:
                new_tokens = new_tokens.unsqueeze(0)
            ids = torch.cat([ids, new_tokens], dim=1)
            generated_count += new_tokens.size(1)

            # main_logits güncelle
            main_logits = self.forward_token(
                ids[:, -1], ids.size(1) - 1,
                enable_plasticity=True
            )

            # Draft modeli senkronize et
            for tok_idx in range(new_tokens.size(1)):
                draft_model.forward_token(
                    new_tokens[:, tok_idx],
                    cur_pos + tok_idx,
                    enable_plasticity=False
                )

        # max_new aşımını kes
        total_len = prompt_ids.size(1) + max_new
        ids = ids[:, :total_len]
        return ids

    def hebb_stats(self) -> dict:
        """Tüm Hebb normlarını döndür."""
        stats = {}
        for i, cell in enumerate(self.cells):
            info = cell.hebb_info
            stats[f'L{i}_ih'] = info['ih']
            stats[f'L{i}_hh'] = info['hh']
        return stats

    def count_params(self) -> dict:
        """Parametre sayımı."""
        total = sum(p.numel() for p in self.parameters())
        embed = self.embed.weight.numel()
        cells = sum(sum(p.numel() for p in c.parameters())
                    for c in self.cells)
        return {'total': total, 'embed': embed, 'cells': cells,
                'other': total - embed - cells}
