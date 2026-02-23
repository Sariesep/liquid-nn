"""
Diferansiyel Plastik Sinapslar — Hebbian Öğrenme

W_eff = W_base + α ⊙ Hebb

W_base: Eğitimle öğrenilen sabit ağırlıklar
Hebb:   Her forward pass'te güncellenen plastik iz
α:      Hangi sinapsların ne kadar plastik olduğunu belirler

v0.3.4 eklentileri:
  - Çift Hızlı Hebb (fast + slow timescale)
  - Sinaptik Konsolidasyon (önemli izleri koruma)
  - Nöromodülasyon desteği (mod_signal ile eta ölçekleme)

"Birlikte ateşleyen nöronlar birbirine bağlanır" — Donald Hebb, 1949
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlasticSynapse(nn.Module):
    """
    Diferansiyel plastik lineer katman.

    Eğitim sırasında: α gradient alır, hangi sinapsların plastik olacağını öğrenir
    İnference sırasında: Hebb matrisi her token'da güncellenir → gerçek zamanlı öğrenme

    Args:
        in_dim:                Girdi boyutu
        out_dim:               Çıktı boyutu
        sparse_k:              Top-k sparse Hebb (0 = tam yoğun)
        use_dual_hebb:         Çift hızlı Hebb (fast + slow) aktif mi
        use_consolidation:     Sinaptik konsolidasyon aktif mi
        consolidation_strength: Konsolidasyon gücü (yüksek → daha dirençli)
    """

    def __init__(self, in_dim: int, out_dim: int, sparse_k: int = 0,
                 use_dual_hebb: bool = False,
                 use_consolidation: bool = False,
                 consolidation_strength: float = 1.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sparse_k = sparse_k  # 0 = tam yoğun, >0 = top-k sparse
        self.use_dual_hebb = use_dual_hebb
        self.use_consolidation = use_consolidation
        self.consolidation_strength = consolidation_strength

        # Sabit ağırlıklar (tüm modellerde var)
        self.W = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.kaiming_normal_(self.W, mode='fan_in', nonlinearity='linear')
        self.W.data *= 0.5
        self.b = nn.Parameter(torch.zeros(out_dim))

        # Plastisite kontrolleri — Fast Hebb (eğitimle öğrenilir)
        self.alpha = nn.Parameter(0.01 * torch.randn(out_dim, in_dim))
        self.log_eta = nn.Parameter(torch.tensor(-3.0))
        self.logit_decay = nn.Parameter(torch.tensor(3.0))
        self.hebb_capacity = nn.Parameter(torch.tensor(1.0))

        # Plastik iz (fast)
        self.register_buffer('Hebb', None)

        # ── Çift Hızlı Hebb (Slow timescale) ──────────────────────
        if use_dual_hebb:
            self.alpha_slow = nn.Parameter(
                0.005 * torch.randn(out_dim, in_dim))
            self.log_eta_slow = nn.Parameter(torch.tensor(-5.0))
            self.logit_decay_slow = nn.Parameter(torch.tensor(5.0))
            self.register_buffer('Hebb_slow', None)

        # ── Sinaptik Konsolidasyon ─────────────────────────────────
        if use_consolidation:
            self.register_buffer('_importance', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.Hebb is None:
            self.Hebb = torch.zeros(self.out_dim, self.in_dim,
                                    device=x.device, dtype=x.dtype)

        W_eff = self.W + self.alpha * self.Hebb

        # Slow Hebb katkısı
        if self.use_dual_hebb:
            if self.Hebb_slow is None:
                self.Hebb_slow = torch.zeros(self.out_dim, self.in_dim,
                                             device=x.device, dtype=x.dtype)
            W_eff = W_eff + self.alpha_slow * self.Hebb_slow

        return F.linear(x, W_eff, self.b)

    @torch.no_grad()
    def update_hebb(self, pre: torch.Tensor, post: torch.Tensor,
                    moe_weight: float = 1.0, mod_signal: float = 1.0):
        """
        Hebbian güncelleme (opsiyonel top-k sparsification ile).

        pre:  Presinaptik aktivasyon [B, in_dim]
        post: Postsinaptik aktivasyon [B, out_dim]
        moe_weight: Bu expert'in seçilme ağırlığı (MoE router'dan gelir)
        mod_signal: Nöromodülasyon sinyali (meta-plasticity)
        """
        decay = torch.sigmoid(self.logit_decay)
        eta = F.softplus(self.log_eta) * 0.03 * mod_signal

        if pre.dim() == 2:
            outer = torch.einsum('bi,bj->ij', post, pre) / max(pre.size(0), 1)
        else:
            outer = torch.outer(post.squeeze(), pre.squeeze())

        # MoE ağırlığı ile plastisiteyi ölçeklendir
        outer = outer * moe_weight

        if self.Hebb is None:
            self.Hebb = torch.zeros(self.out_dim, self.in_dim,
                                    device=pre.device, dtype=pre.dtype)

        # ── Sinaptik Konsolidasyon: önemli izleri koru ─────────────
        if self.use_consolidation:
            if self._importance is None:
                self._importance = torch.zeros_like(outer)
            # EMA importance: tutarlı büyük Hebb*alpha değerleri önemli
            self._importance = (0.99 * self._importance +
                                0.01 * (self.Hebb * self.alpha).abs())
            # Update mask: önemli → düşük güncelleme
            update_mask = 1.0 / (1.0 + self._importance *
                                 self.consolidation_strength)
            outer = outer * update_mask

        self.Hebb = decay * self.Hebb + eta * outer

        # Öğrenilebilir norm sınırı (fast)
        h_norm = self.Hebb.norm()
        max_norm = F.softplus(self.hebb_capacity)
        if h_norm > max_norm:
            self.Hebb = self.Hebb * (max_norm / (h_norm + 1e-8))

        # Top-k sparsification (fast)
        if self.sparse_k > 0:
            flat = self.Hebb.abs().view(-1)
            total = flat.numel()
            k = min(self.sparse_k, total)
            if k < total:
                threshold = torch.topk(flat, k).values[-1]
                mask = self.Hebb.abs() >= threshold
                self.Hebb = self.Hebb * mask

        # ── Slow Hebb güncellemesi ─────────────────────────────────
        if self.use_dual_hebb:
            decay_slow = torch.sigmoid(self.logit_decay_slow)
            eta_slow = F.softplus(self.log_eta_slow) * 0.01 * mod_signal

            if self.Hebb_slow is None:
                self.Hebb_slow = torch.zeros(self.out_dim, self.in_dim,
                                             device=pre.device,
                                             dtype=pre.dtype)

            # Aynı outer product (konsolidasyon uygulanmış), farklı hız
            self.Hebb_slow = decay_slow * self.Hebb_slow + eta_slow * outer * moe_weight

            # Aynı kapasite sınırı
            hs_norm = self.Hebb_slow.norm()
            if hs_norm > max_norm:
                self.Hebb_slow = self.Hebb_slow * (max_norm / (hs_norm + 1e-8))

    def reset_hebb(self):
        """Plastik izleri sıfırla."""
        self.Hebb = None
        if self.use_dual_hebb:
            self.Hebb_slow = None

    def detach_hebb(self):
        """Hebb'i hesaplama grafiğinden ayır (truncated BPTT için)."""
        if self.Hebb is not None:
            self.Hebb = self.Hebb.detach().clone()
        if self.use_dual_hebb and hasattr(self, 'Hebb_slow'):
            if self.Hebb_slow is not None:
                self.Hebb_slow = self.Hebb_slow.detach().clone()

    @property
    def hebb_norm(self) -> float:
        """Hebb matrisinin normu (fast)."""
        return self.Hebb.norm().item() if self.Hebb is not None else 0.0

    @property
    def hebb_slow_norm(self) -> float:
        """Hebb_slow matrisinin normu."""
        if self.use_dual_hebb and hasattr(self, 'Hebb_slow'):
            return self.Hebb_slow.norm().item() if self.Hebb_slow is not None else 0.0
        return 0.0

    def extra_repr(self) -> str:
        parts = [f'in={self.in_dim}, out={self.out_dim}',
                 f'hebb_norm={self.hebb_norm:.4f}']
        if self.use_dual_hebb:
            parts.append(f'hebb_slow_norm={self.hebb_slow_norm:.4f}')
        if self.use_consolidation:
            parts.append(f'consolidation={self.consolidation_strength}')
        return ', '.join(parts)
