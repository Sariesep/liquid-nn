"""
Nöromodülasyon — Meta-Plasticity Controller

Prediction error sinyaline göre Hebbian öğrenme hızını global modüle eder.

Düşük hata → mod_signal ≈ baseline  (konsolidasyon modu)
Yüksek hata → mod_signal > baseline  (hızlı öğrenme modu)
Sürpriz   → mod_signal >> baseline (yeni bağlam algılandı)

Biyolojik karşılık: Dopaminerjik nöromodülasyon (Schultz, 1998)
"""

import torch
import torch.nn as nn


class Neuromodulator(nn.Module):
    """
    Prediction error → öğrenme hızı modülatörü.

    EMA ile geçmiş hataları takip eder; beklenmedik bir hata artışı
    (surprise) tespit ettiğinde mod_signal'i yükseltir.

    Args:
        embed_dim:    Girdi boyutu (prediction error vektör boyutu)
        baseline:     Mod sinyalinin taban değeri
        sensitivity:  Sürpriz duyarlılığı (yüksek → daha agresif modülasyon)
        ema_decay:    Hata EMA decay katsayısı (yüksek → yavaş adaptasyon)
    """

    def __init__(self, embed_dim: int, baseline: float = 1.0,
                 sensitivity: float = 1.0, ema_decay: float = 0.95):
        super().__init__()
        self.baseline = baseline
        self.sensitivity = sensitivity
        self.ema_decay = ema_decay

        # Prediction error → scalar surprise
        self.error_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.SiLU(),
            nn.Linear(embed_dim // 4, 1),
        )

        # EMA takibi (gradient gerektirmez)
        self.register_buffer('_error_ema', torch.tensor(0.0))
        self.register_buffer('_step_count', torch.tensor(0))

    def forward(self, prediction_error: torch.Tensor) -> float:
        """
        Prediction error vektöründen modülasyon sinyali üret.

        Args:
            prediction_error: [B, D] veya [D] — logit hatası veya
                              embedding uzayında hata vektörü

        Returns:
            float: mod_signal ∈ [baseline - sensitivity, baseline + sensitivity]
                   (pratikte ~[0.2, 2.5] aralığında)
        """
        if prediction_error.dim() == 1:
            prediction_error = prediction_error.unsqueeze(0)

        # Skaler hata büyüklüğü
        with torch.no_grad():
            error_magnitude = prediction_error.abs().mean().item()

            # EMA güncelle
            if self._step_count.item() == 0:
                self._error_ema.fill_(error_magnitude)
            else:
                self._error_ema = (self.ema_decay * self._error_ema +
                                   (1 - self.ema_decay) * error_magnitude)
            self._step_count += 1

            # Surprise: mevcut hata ile beklenen hata arasındaki fark
            surprise = abs(error_magnitude - self._error_ema.item())
            # Normalize: EMA'ya göre göreceli sürpriz
            relative_surprise = surprise / (self._error_ema.item() + 1e-6)

        # Mod signal: baseline ± sensitivity * tanh(surprise)
        mod_signal = (self.baseline +
                      self.sensitivity * torch.tanh(
                          torch.tensor(relative_surprise)).item())

        return max(0.1, mod_signal)  # asla tamamen sıfır olmasın

    def reset(self):
        """EMA durumunu sıfırla."""
        self._error_ema.zero_()
        self._step_count.zero_()
