"""
Sliding-Window Attention — Yerel Dikkat Mekanizması

RNN step uyumlu kayar pencere dikkati.
Her forward_token çağrısında yeni token'ı ring buffer'a ekler
ve son W token üzerinde multi-head self-attention uygular.

Bu, saf RNN'in zayıf kaldığı uzun mesafe bağımlılıklarını yakalamaya yardımcı olur.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SlidingWindowAttention(nn.Module):
    """
    Kayar pencere çok başlı dikkat mekanizması (RNN-uyumlu).

    Her adımda:
      1. Yeni token vektörünü ring buffer'a ekle
      2. Buffer'daki tokenler üzerinde causal attention uygula
      3. Sonucu döndür

    Args:
        embed_dim:   Gömme boyutu
        num_heads:   Dikkat başı sayısı
        window_size: Pencere boyutu (son kaç tokeni hatırla)
        dropout:     Dikkat dropout oranı
    """

    def __init__(self, embed_dim: int, num_heads: int = 4,
                 window_size: int = 64, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) num_heads'e ({num_heads}) bölünmeli"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5

        # Q, K, V projeksiyonları
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

        # Gated residual: attention çıktısını ne kadar kullanacağını öğrenir
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )

        self.attn_drop = nn.Dropout(dropout)

        # Ring buffer (register_buffer ile save/load uyumlu)
        self.register_buffer('_buffer',
                             torch.zeros(1, window_size, embed_dim))
        self._buf_len = 0   # buffer'daki geçerli eleman sayısı

    def reset_buffer(self):
        """Buffer'ı sıfırla (yeni sekans başlangıcında)."""
        self._buffer.zero_()
        self._buf_len = 0

    def get_buffer_state(self) -> dict:
        """save_state için buffer durumunu döndür."""
        return {
            'buffer': self._buffer.detach().clone(),
            'buf_len': self._buf_len,
        }

    def set_buffer_state(self, state: dict):
        """restore_state için buffer durumunu geri yükle."""
        self._buffer = state['buffer'].detach().clone()
        self._buf_len = state['buf_len']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tek token'a sliding-window attention uygula.

        Args:
            x: [B, D] mevcut token gömmesi

        Returns:
            [B, D] attention ile zenginleştirilmiş çıktı (gated residual)
        """
        B, D = x.shape

        # Buffer'ı batch boyutuna göre genişlet (gerekirse)
        if self._buffer.size(0) != B:
            self._buffer = torch.zeros(
                B, self.window_size, D,
                device=x.device, dtype=x.dtype
            )
            self._buf_len = 0

        # Yeni token'ı buffer'a ekle (FIFO)
        if self._buf_len < self.window_size:
            self._buffer[:, self._buf_len] = x
            self._buf_len += 1
        else:
            # Buffer dolu → en eskiyi at, yeniyi sona ekle
            self._buffer = torch.roll(self._buffer, -1, dims=1)
            self._buffer[:, -1] = x

        # Buffer'daki geçerli tokenler
        seq_len = self._buf_len
        kv_seq = self._buffer[:, :seq_len]  # [B, S, D]

        # Q: sadece mevcut token, K/V: tüm pencere
        q = self.W_q(x).unsqueeze(1)           # [B, 1, D]
        k = self.W_k(kv_seq)                    # [B, S, D]
        v = self.W_v(kv_seq)                    # [B, S, D]

        # Multi-head reshape
        H = self.num_heads
        Dh = self.head_dim
        q = q.view(B, 1, H, Dh).transpose(1, 2)      # [B, H, 1, Dh]
        k = k.view(B, seq_len, H, Dh).transpose(1, 2) # [B, H, S, Dh]
        v = v.view(B, seq_len, H, Dh).transpose(1, 2) # [B, H, S, Dh]

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,1,S]
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)              # [B, H, 1, Dh]
        out = out.transpose(1, 2).contiguous().view(B, D)  # [B, D]
        out = self.W_o(out)

        # Gated residual: x ve attention çıktısını birleştir
        gate_input = torch.cat([x, out], dim=-1)  # [B, 2D]
        g = self.gate(gate_input)                   # [B, D]  (0-1 arası)
        return x + g * out
