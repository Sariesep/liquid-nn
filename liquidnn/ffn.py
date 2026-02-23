"""
SwiGLU Feed-Forward Network

Transformer FFN katmanı — SwiGLU aktivasyonu ile.
Her ODE cell'den sonra uygulanarak modelin kapasitesini artırır.

SwiGLU: FFN(x) = (xW₁ ⊙ SiLU(xV)) W₂
(Shazeer, 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    Args:
        dim:     Girdi/çıktı boyutu
        mult:    Gizli katman çarpanı (dim * mult * 2/3)
        dropout: Dropout oranı
    """

    def __init__(self, dim: int, mult: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mult * 2 / 3)
        # En yakın 8'in katına yuvarla (GPU verimliliği)
        hidden = ((hidden + 7) // 8) * 8

        self.w1 = nn.Linear(dim, hidden, bias=False)   # gate
        self.v = nn.Linear(dim, hidden, bias=False)     # value
        self.w2 = nn.Linear(hidden, dim, bias=False)    # output
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., dim]
        Returns:
            [..., dim]
        """
        return self.drop(self.w2(F.silu(self.w1(x)) * self.v(x)))
