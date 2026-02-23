"""
Sliding-Window Attention v3 — Training + Inference Dual Mode

v0.3.5 iyileştirmeleri:
  - Training Mode: Full-sequence causal attention (cache yok, autograd güvenli)
  - Inference Mode: KV Cache + detached yaklaşım (tek token)
  - RoPE, GQA, Flash Attention devam ediyor
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
        x:     [B, H, S, Dh]
        pos:   başlangıç pozisyonu
        freqs: [max_len, Dh//2, 2] precomputed cos/sin
    """
    B, H, S, Dh = x.shape
    x_pair = x.view(B, H, S, Dh // 2, 2)

    positions = torch.arange(pos, pos + S, device=x.device)
    f = freqs[positions].to(x.device, x.dtype)
    cos_f = f[..., 0].unsqueeze(0).unsqueeze(0)
    sin_f = f[..., 1].unsqueeze(0).unsqueeze(0)

    x0 = x_pair[..., 0]
    x1 = x_pair[..., 1]
    out0 = x0 * cos_f - x1 * sin_f
    out1 = x0 * sin_f + x1 * cos_f

    return torch.stack([out0, out1], dim=-1).view(B, H, S, Dh)


class SlidingWindowAttention(nn.Module):
    """
    v3: Dual-mode — Training (full-seq) + Inference (KV cache).

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

        # KV Cache (inference only)
        self.register_buffer('_k_cache',
                             torch.zeros(1, self.num_kv_heads,
                                         window_size, self.head_dim))
        self.register_buffer('_v_cache',
                             torch.zeros(1, self.num_kv_heads,
                                         window_size, self.head_dim))
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

    # ═══ Training Mode: Full-Sequence Causal Attention ═══════════

    def forward_train(self, x_seq: torch.Tensor,
                      start_pos: int = 0) -> torch.Tensor:
        """
        Eğitim modu: tam sekans üzerinde causal self-attention.
        KV cache KULLANMAZ — tamamen autograd-safe.

        Args:
            x_seq:     [B, T, D] token gömme sekansı
            start_pos: RoPE için başlangıç pozisyonu

        Returns:
            [B, T, D] attention çıktısı
        """
        B, T, D = x_seq.shape
        H = self.num_heads
        Hkv = self.num_kv_heads
        Dh = self.head_dim

        # Q, K, V projeksiyonları — tüm sekans birden
        q = self.W_q(x_seq).view(B, T, H, Dh).transpose(1, 2)      # [B, H, T, Dh]
        k = self.W_k(x_seq).view(B, T, Hkv, Dh).transpose(1, 2)    # [B, Hkv, T, Dh]
        v = self.W_v(x_seq).view(B, T, Hkv, Dh).transpose(1, 2)

        # RoPE
        if self.use_rope:
            q = _apply_rope(q, start_pos, self._rope_freqs)
            k = _apply_rope(k, start_pos, self._rope_freqs)

        # GQA genişletme
        if self.groups > 1:
            k = k.repeat_interleave(self.groups, dim=1)
            v = v.repeat_interleave(self.groups, dim=1)

        # Causal attention (sliding window)
        if self.use_flash and hasattr(F, 'scaled_dot_product_attention'):
            drop_p = self.attn_drop_p if self.training else 0.0
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=drop_p,
                is_causal=True
            )  # [B, H, T, Dh]
        else:
            scale = Dh ** -0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale

            # Causal mask
            causal_mask = torch.triu(
                torch.ones(T, T, device=x_seq.device, dtype=torch.bool),
                diagonal=1
            )
            attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0),
                                     float('-inf'))
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, T, D)   # [B, T, D]
        out = self.W_o(out)

        # Gated residual (sekans boyunca)
        gate_input = torch.cat([x_seq, out], dim=-1)   # [B, T, 2D]
        g = self.gate(gate_input)
        return x_seq + g * out

    # ═══ Inference Mode: Single-Token + KV Cache ═════════════════

    def forward(self, x: torch.Tensor, pos: int = 0) -> torch.Tensor:
        """
        Çift modlu forward:
        - training=True  → forward_train (x: [B, T, D] veya [B, D])
        - training=False → single-token KV cache (x: [B, D])
        """
        # Eğitim modunda ve 3D girdi → full-sequence
        if self.training and x.dim() == 3:
            return self.forward_train(x, start_pos=pos)

        # Aşağıdaki kısım sadece inference (single-token) için
        if x.dim() == 3:
            # Inference'da 3D girdi → her token için tek tek çağır
            B, T, D = x.shape
            outputs = []
            for t in range(T):
                out_t = self._forward_single(x[:, t], pos=pos + t)
                outputs.append(out_t)
            return torch.stack(outputs, dim=1)  # [B, T, D]

        return self._forward_single(x, pos=pos)

    def _forward_single(self, x: torch.Tensor, pos: int = 0) -> torch.Tensor:
        """Tek token inference — KV cache ile."""
        B, D = x.shape
        H = self.num_heads
        Hkv = self.num_kv_heads
        Dh = self.head_dim

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

        q = self.W_q(x).view(B, 1, H, Dh).transpose(1, 2)
        k_new = self.W_k(x).view(B, 1, Hkv, Dh).transpose(1, 2)
        v_new = self.W_v(x).view(B, 1, Hkv, Dh).transpose(1, 2)

        if self.use_rope:
            q = _apply_rope(q, pos, self._rope_freqs)
            k_new = _apply_rope(k_new, pos, self._rope_freqs)

        # KV Cache güncelle (detached)
        k_new_det = k_new.detach().squeeze(2)
        v_new_det = v_new.detach().squeeze(2)
        x_det = x.detach()

        if self._buf_len < self.window_size:
            self._k_cache[:, :, self._buf_len] = k_new_det
            self._v_cache[:, :, self._buf_len] = v_new_det
            self._buffer[:, self._buf_len] = x_det
            self._buf_len += 1
        else:
            self._k_cache = torch.roll(self._k_cache, -1, dims=2)
            self._v_cache = torch.roll(self._v_cache, -1, dims=2)
            self._buffer = torch.roll(self._buffer, -1, dims=1)
            self._k_cache[:, :, -1] = k_new_det
            self._v_cache[:, :, -1] = v_new_det
            self._buffer[:, -1] = x_det

        seq_len = self._buf_len
        if seq_len > 1:
            k_past = self._k_cache[:, :, :seq_len - 1]
            v_past = self._v_cache[:, :, :seq_len - 1]
            k = torch.cat([k_past, k_new], dim=2)
            v = torch.cat([v_past, v_new], dim=2)
        else:
            k = k_new
            v = v_new

        if self.groups > 1:
            k = k.repeat_interleave(self.groups, dim=1)
            v = v.repeat_interleave(self.groups, dim=1)

        if self.use_flash and hasattr(F, 'scaled_dot_product_attention'):
            drop_p = self.attn_drop_p if self.training else 0.0
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=drop_p,
                is_causal=False
            )
        else:
            scale = Dh ** -0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, D)
        out = self.W_o(out)

        gate_input = torch.cat([x, out], dim=-1)
        g = self.gate(gate_input)
        return x + g * out
