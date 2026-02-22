"""
Sliding-Window Attention v2 — RoPE + KV Cache + Flash Attention

v0.3.1 iyileştirmeleri:
  - RoPE (Rotary Position Encoding): göreceli pozisyon bilgisi
  - KV Cache: önceki tokenler için K/V yeniden hesaplanmaz
  - Flash Attention: F.scaled_dot_product_attention ile 2-4x hızlanma
  - GQA (Grouped Query Attention): devam ediyor
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── RoPE (Rotary Position Encoding) ──────────────────────────────

def _precompute_freqs(head_dim: int, max_len: int = 4096,
                      theta: float = 10000.0) -> torch.Tensor:
    """RoPE frekans tablosu [max_len, head_dim//2, 2]."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len, dtype=torch.float)
    angles = torch.outer(t, freqs)  # [max_len, head_dim//2]
    return torch.stack([angles.cos(), angles.sin()], dim=-1)  # [L, D/2, 2]


def _apply_rope(x: torch.Tensor, pos: int,
                freqs: torch.Tensor) -> torch.Tensor:
    """
    RoPE uygula.

    Args:
        x:     [B, H, 1, Dh] veya [B, H, S, Dh]
        pos:   başlangıç pozisyonu (tek token → pos, KV cache → offset)
        freqs: [max_len, Dh//2, 2] precomputed cos/sin

    Returns:
        aynı şekilde rotated x
    """
    B, H, S, Dh = x.shape
    # x'i çiftler halinde böl
    x_pair = x.view(B, H, S, Dh // 2, 2)  # [..., (x0, x1)]

    # İlgili pozisyonların frekanslarını al
    positions = torch.arange(pos, pos + S, device=x.device)
    f = freqs[positions].to(x.device, x.dtype)  # [S, Dh//2, 2]
    cos_f = f[..., 0].unsqueeze(0).unsqueeze(0)  # [1, 1, S, Dh//2]
    sin_f = f[..., 1].unsqueeze(0).unsqueeze(0)

    # Rotary: (x0 * cos - x1 * sin, x0 * sin + x1 * cos)
    x0 = x_pair[..., 0]  # [B, H, S, Dh//2]
    x1 = x_pair[..., 1]
    out0 = x0 * cos_f - x1 * sin_f
    out1 = x0 * sin_f + x1 * cos_f

    return torch.stack([out0, out1], dim=-1).view(B, H, S, Dh)


class SlidingWindowAttention(nn.Module):
    """
    v2: RoPE + KV Cache + Flash Attention + GQA.

    Args:
        embed_dim:    Gömme boyutu
        num_heads:    Q başı sayısı
        num_kv_heads: KV başı sayısı (None = MHA, 1 = GQA)
        window_size:  Pencere boyutu
        dropout:      Dikkat dropout oranı
        use_rope:     RoPE aktif mi
        use_flash:    Flash Attention aktif mi
    """

    def __init__(self, embed_dim: int, num_heads: int = 4,
                 num_kv_heads: int = None,
                 window_size: int = 64, dropout: float = 0.0,
                 use_rope: bool = True, use_flash: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        assert embed_dim % num_heads == 0
        assert num_heads % self.num_kv_heads == 0

        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.kv_dim = self.head_dim * self.num_kv_heads
        self.groups = num_heads // self.num_kv_heads
        self.window_size = window_size
        self.use_rope = use_rope
        self.use_flash = use_flash

        # Q, K, V, O projeksiyonları
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, self.kv_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, self.kv_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

        # Gated residual
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )

        self.attn_drop_p = dropout
        self.attn_drop = nn.Dropout(dropout)

        # RoPE frekansları
        if use_rope:
            rope_freqs = _precompute_freqs(self.head_dim, max_len=4096)
            self.register_buffer('_rope_freqs', rope_freqs)

        # KV Cache (ring buffer): head boyutunda cache'le
        # [B, Hkv, window_size, Dh] — pre-projected
        self.register_buffer('_k_cache',
                             torch.zeros(1, self.num_kv_heads,
                                         window_size, self.head_dim))
        self.register_buffer('_v_cache',
                             torch.zeros(1, self.num_kv_heads,
                                         window_size, self.head_dim))
        # Eski raw buffer (save/restore uyumu için)
        self.register_buffer('_buffer',
                             torch.zeros(1, window_size, embed_dim))
        self._buf_len = 0

    def reset_buffer(self):
        """Buffer ve cache'i sıfırla."""
        self._buffer.zero_()
        self._k_cache.zero_()
        self._v_cache.zero_()
        self._buf_len = 0

    def get_buffer_state(self) -> dict:
        return {
            'buffer': self._buffer.detach().clone(),
            'k_cache': self._k_cache.detach().clone(),
            'v_cache': self._v_cache.detach().clone(),
            'buf_len': self._buf_len,
        }

    def set_buffer_state(self, state: dict):
        self._buffer = state['buffer'].detach().clone()
        self._k_cache = state['k_cache'].detach().clone()
        self._v_cache = state['v_cache'].detach().clone()
        self._buf_len = state['buf_len']

    def forward(self, x: torch.Tensor, pos: int = 0) -> torch.Tensor:
        """
        Tek token'a attention uygula (RoPE + KV Cache + Flash).

        Args:
            x:   [B, D] mevcut token gömmesi
            pos: sekans pozisyonu (RoPE için gerekli)

        Returns:
            [B, D] attention ile zenginleştirilmiş çıktı
        """
        B, D = x.shape
        H = self.num_heads
        Hkv = self.num_kv_heads
        Dh = self.head_dim

        # Batch boyutu değiştiyse cache'leri yeniden oluştur
        if self._k_cache.size(0) != B:
            self._k_cache = torch.zeros(
                B, Hkv, self.window_size, Dh,
                device=x.device, dtype=x.dtype)
            self._v_cache = torch.zeros(
                B, Hkv, self.window_size, Dh,
                device=x.device, dtype=x.dtype)
            self._buffer = torch.zeros(
                B, self.window_size, D,
                device=x.device, dtype=x.dtype)
            self._buf_len = 0

        # ── Yeni token'ın Q, K, V projeksiyonları ────────────────
        q = self.W_q(x).view(B, 1, H, Dh).transpose(1, 2)    # [B, H, 1, Dh]
        k_new = self.W_k(x).view(B, 1, Hkv, Dh).transpose(1, 2)  # [B, Hkv, 1, Dh]
        v_new = self.W_v(x).view(B, 1, Hkv, Dh).transpose(1, 2)

        # ── RoPE ────────────────────────────────────────────────
        if self.use_rope:
            q = _apply_rope(q, pos, self._rope_freqs)
            k_new = _apply_rope(k_new, pos, self._rope_freqs)

        # ── KV Cache güncelle (FIFO) ────────────────────────────
        if self._buf_len < self.window_size:
            self._k_cache[:, :, self._buf_len] = k_new.squeeze(2)
            self._v_cache[:, :, self._buf_len] = v_new.squeeze(2)
            self._buffer[:, self._buf_len] = x
            self._buf_len += 1
        else:
            self._k_cache = torch.roll(self._k_cache, -1, dims=2)
            self._v_cache = torch.roll(self._v_cache, -1, dims=2)
            self._buffer = torch.roll(self._buffer, -1, dims=1)
            self._k_cache[:, :, -1] = k_new.squeeze(2)
            self._v_cache[:, :, -1] = v_new.squeeze(2)
            self._buffer[:, -1] = x

        # ── Cache'ten K/V al ────────────────────────────────────
        seq_len = self._buf_len
        k = self._k_cache[:, :, :seq_len]  # [B, Hkv, S, Dh]
        v = self._v_cache[:, :, :seq_len]

        # GQA: KV head'lerini Q sayısına genişlet
        if self.groups > 1:
            k = k.repeat_interleave(self.groups, dim=1)
            v = v.repeat_interleave(self.groups, dim=1)

        # ── Attention hesapla ───────────────────────────────────
        if self.use_flash and hasattr(F, 'scaled_dot_product_attention'):
            # Flash Attention (PyTorch 2.0+)
            drop_p = self.attn_drop_p if self.training else 0.0
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=drop_p,
                is_causal=False  # window zaten causal mantık sağlıyor
            )  # [B, H, 1, Dh]
        else:
            # Fallback: manual attention
            scale = Dh ** -0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, D)
        out = self.W_o(out)

        # Gated residual
        gate_input = torch.cat([x, out], dim=-1)
        g = self.gate(gate_input)
        return x + g * out
