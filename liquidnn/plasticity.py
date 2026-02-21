"""
Diferansiyel Plastik Sinapslar — Hebbian Öğrenme

W_eff = W_base + α ⊙ Hebb

W_base: Eğitimle öğrenilen sabit ağırlıklar
Hebb:   Her forward pass'te güncellenen plastik iz
α:      Hangi sinapsların ne kadar plastik olduğunu belirler

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
        in_dim:  Girdi boyutu
        out_dim: Çıktı boyutu
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Sabit ağırlıklar (tüm modellerde var)
        self.W = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.kaiming_normal_(self.W, mode='fan_in', nonlinearity='linear')
        self.W.data *= 0.5
        self.b = nn.Parameter(torch.zeros(out_dim))

        # Plastisite kontrolleri (eğitimle öğrenilir)
        self.alpha = nn.Parameter(0.01 * torch.randn(out_dim, in_dim))
        self.log_eta = nn.Parameter(torch.tensor(-3.0))
        self.logit_decay = nn.Parameter(torch.tensor(3.0))

        # Plastik iz
        self.register_buffer('Hebb', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.Hebb is None:
            self.Hebb = torch.zeros(self.out_dim, self.in_dim,
                                    device=x.device, dtype=x.dtype)
        W_eff = self.W + self.alpha * self.Hebb
        return F.linear(x, W_eff, self.b)

    @torch.no_grad()
    def update_hebb(self, pre: torch.Tensor, post: torch.Tensor):
        """
        Hebbian güncelleme.

        pre:  Presinaptik aktivasyon [B, in_dim]
        post: Postsinaptik aktivasyon [B, out_dim]
        """
        decay = torch.sigmoid(self.logit_decay)
        eta = F.softplus(self.log_eta) * 0.03

        if pre.dim() == 2:
            outer = torch.einsum('bi,bj->ij', post, pre) / max(pre.size(0), 1)
        else:
            outer = torch.outer(post.squeeze(), pre.squeeze())

        self.Hebb = decay * self.Hebb + eta * outer

        # Norm sınırı
        h_norm = self.Hebb.norm()
        max_norm = 0.3 * self.W.data.norm()
        if h_norm > max_norm:
            self.Hebb = self.Hebb * (max_norm / (h_norm + 1e-8))

    def reset_hebb(self):
        """Plastik izleri sıfırla."""
        self.Hebb = None

    def detach_hebb(self):
        """Hebb'i hesaplama grafiğinden ayır (truncated BPTT için)."""
        if self.Hebb is not None:
            self.Hebb = self.Hebb.detach().clone()

    @property
    def hebb_norm(self) -> float:
        """Hebb matrisinin normu."""
        return self.Hebb.norm().item() if self.Hebb is not None else 0.0

    def extra_repr(self) -> str:
        return (f'in={self.in_dim}, out={self.out_dim}, '
                f'hebb_norm={self.hebb_norm:.4f}')
