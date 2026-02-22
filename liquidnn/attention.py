"""
Sliding-Window Attention — Yerel Dikkat Mekanizması (GQA destekli)

RNN step uyumlu kayar pencere dikkati.
Her forward_token çağrısında yeni token'ı ring buffer'a ekler
ve son W token üzerinde multi-head self-attention uygular.

Grouped Query Attention (GQA): num_kv_heads < num_heads ise
KV head sayısı azaltılır → bellek ve hesaplama tasarrufu.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SlidingWindowAttention(nn.Module):
    """
    Kayar pencere çok başlı dikkat mekanizması (RNN-uyumlu, GQA destekli).

    Args:
        embed_dim:    Gömme boyutu
        num_heads:    Q (query) dikkat başı sayısı
        num_kv_heads: KV dikkat başı sayısı (None = num_heads, yani MHA)
                      1 → GQA, çok düşük bellek kullanımı
        window_size:  Pencere boyutu (son kaç tokeni hatırla)
        dropout:      Dikkat dropout oranı
    """

    def __init__(self, embed_dim: int, num_heads: int = 4,
                 num_kv_heads: int = None,
                 window_size: int = 64, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        assert embed_dim % num_heads == 0
        assert num_heads % self.num_kv_heads == 0, \
            f"num_heads ({num_heads}) num_kv_heads'e ({self.num_kv_heads}) bölünmeli"

        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.kv_dim = self.head_dim * self.num_kv_heads
        self.groups = num_heads // self.num_kv_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5

        # Q: tam boyut, K/V: kv_dim boyut (GQA'da daha küçük)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, self.kv_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, self.kv_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

        # Gated residual
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )

        self.attn_drop = nn.Dropout(dropout)

        # Ring buffer
        self.register_buffer('_buffer',
                             torch.zeros(1, window_size, embed_dim))
        self._buf_len = 0

    def reset_buffer(self):
        """Buffer'ı sıfırla."""
        self._buffer.zero_()
        self._buf_len = 0

    def get_buffer_state(self) -> dict:
        return {
            'buffer': self._buffer.detach().clone(),
            'buf_len': self._buf_len,
        }

    def set_buffer_state(self, state: dict):
        self._buffer = state['buffer'].detach().clone()
        self._buf_len = state['buf_len']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tek token'a sliding-window attention uygula (GQA destekli).

        Args:
            x: [B, D] mevcut token gömmesi

        Returns:
            [B, D] attention ile zenginleştirilmiş çıktı (gated residual)
        """
        B, D = x.shape

        if self._buffer.size(0) != B:
            self._buffer = torch.zeros(
                B, self.window_size, D,
                device=x.device, dtype=x.dtype
            )
            self._buf_len = 0

        # FIFO buffer güncelle
        if self._buf_len < self.window_size:
            self._buffer[:, self._buf_len] = x
            self._buf_len += 1
        else:
            self._buffer = torch.roll(self._buffer, -1, dims=1)
            self._buffer[:, -1] = x

        seq_len = self._buf_len
        kv_seq = self._buffer[:, :seq_len]

        # Projeksiyonlar
        H = self.num_heads
        Hkv = self.num_kv_heads
        Dh = self.head_dim

        q = self.W_q(x).unsqueeze(1)
        k = self.W_k(kv_seq)
        v = self.W_v(kv_seq)

        # Multi-head reshape
        q = q.view(B, 1, H, Dh).transpose(1, 2)
        k = k.view(B, seq_len, Hkv, Dh).transpose(1, 2)
        v = v.view(B, seq_len, Hkv, Dh).transpose(1, 2)

        # GQA: KV head'lerini Q head sayısına genişlet
        if self.groups > 1:
            k = k.repeat_interleave(self.groups, dim=1)
            v = v.repeat_interleave(self.groups, dim=1)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, D)
        out = self.W_o(out)

        # Gated residual
        gate_input = torch.cat([x, out], dim=-1)
        g = self.gate(gate_input)
        return x + g * out
