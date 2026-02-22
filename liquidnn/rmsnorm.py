"""
RMSNorm — Root Mean Square Layer Normalization

LayerNorm'a göre ~%30 daha hızlı: mean hesaplamaz,
sadece RMS (root mean square) ile normalize eder.

Zhang & Sennrich, 2019: "Root Mean Square Layer Normalization"
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    LayerNorm'dan farkı:
      - Mean çıkarma yok (sadece scale)
      - Daha az hesaplama, benzer kalite

    Args:
        dim: Normalize edilecek boyut
        eps: Sayısal kararlılık için epsilon
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight
