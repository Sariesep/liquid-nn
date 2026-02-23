"""v0.3.5 testleri — Attention Training, FFN, Adaptif Hebb."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn.functional as F
from liquidnn import MiniLiquidGPT, SwiGLUFFN
from liquidnn.attention import SlidingWindowAttention
from liquidnn.plasticity import PlasticSynapse


# ═══ 1. Training-Mode Attention ═════════════════════════════════

def test_attention_training_backward():
    """Attention eğitim modunda backward çalışmalı."""
    attn = SlidingWindowAttention(32, num_heads=4, window_size=16)
    attn.train()
    x = torch.randn(2, 10, 32, requires_grad=True)  # [B, T, D]
    out = attn(x)
    assert out.shape == (2, 10, 32), f"Beklenen (2,10,32), alınan {out.shape}"
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "Gradient akmalı"
    print("✅ attention_training_backward")


def test_attention_inference_single_token():
    """Attention inference modunda tek token KV cache ile çalışmalı."""
    attn = SlidingWindowAttention(32, num_heads=4, window_size=16)
    attn.eval()
    attn.reset_buffer()
    for t in range(5):
        x = torch.randn(1, 32)
        out = attn(x, pos=t)
        assert out.shape == (1, 32)
    print("✅ attention_inference_single_token")


def test_model_attention_training():
    """Model attention AÇIK + eğitim, backward sorunsuz çalışmalı."""
    model = MiniLiquidGPT(
        vocab_size=100, embed_dim=32,
        num_fast=1, num_deep=1,
        use_attention=True, attn_heads=4,
        use_rmsnorm=True,
    )
    model.train()
    x = torch.randint(0, 100, (2, 16))
    logits = model(x, chunk_size=8)
    assert logits.shape == (2, 16, 100)
    loss = logits.sum()
    loss.backward()
    assert model.embed.weight.grad is not None
    print("✅ model_attention_training")


# ═══ 2. SwiGLU FFN ══════════════════════════════════════════════

def test_ffn_output_shape():
    """FFN girdi/çıktı boyutu aynı olmalı."""
    ffn = SwiGLUFFN(64, mult=4.0)
    x = torch.randn(2, 10, 64)
    y = ffn(x)
    assert y.shape == (2, 10, 64), f"Beklenen (2,10,64), alınan {y.shape}"
    print("✅ ffn_output_shape")


def test_ffn_in_model():
    """FFN katmanları modele parametre eklemeli."""
    model_no = MiniLiquidGPT(vocab_size=100, embed_dim=32,
                              num_fast=1, num_deep=1, use_ffn=False)
    model_yes = MiniLiquidGPT(vocab_size=100, embed_dim=32,
                               num_fast=1, num_deep=1, use_ffn=True)
    params_no = sum(p.numel() for p in model_no.parameters())
    params_yes = sum(p.numel() for p in model_yes.parameters())
    assert params_yes > params_no, \
        f"FFN parametre eklemeli: no={params_no}, yes={params_yes}"
    
    # Forward çalışmalı
    x = torch.randint(0, 100, (2, 10))
    logits = model_yes(x)
    assert logits.shape == (2, 10, 100)
    print(f"✅ ffn_in_model (no={params_no:,} → yes={params_yes:,}, "
          f"+{params_yes-params_no:,})")


# ═══ 3. Adaptif Hebb Kapasitesi ═════════════════════════════════

def test_adaptive_hebb_capacity_grows():
    """Hebb kapasitesi adım sayısıyla büyümeli."""
    syn = PlasticSynapse(16, 16)
    pre = torch.randn(1, 16)
    post = torch.randn(1, 16)

    # 100 adım sonra kapasite artmış olmalı
    for _ in range(100):
        syn.update_hebb(pre, post)

    # softplus(2.0) * growth > softplus(2.0) * 1.0
    growth = 1.0 + 0.1 * torch.log1p(syn._hebb_steps.float()).item()
    assert growth > 1.0, f"Growth > 1 olmalı: {growth:.3f}"
    
    max_norm = F.softplus(syn.hebb_capacity) * growth
    base_norm = F.softplus(torch.tensor(2.0))
    assert max_norm > base_norm, \
        f"Kapasite büyümeli: base={base_norm:.3f}, current={max_norm:.3f}"
    print(f"✅ adaptive_hebb_capacity_grows (growth={growth:.3f})")


# ═══ Combined ════════════════════════════════════════════════════

def test_all_v035_combined():
    """Tüm v0.3.5 özellikleri açık, forward + backward çalışmalı."""
    model = MiniLiquidGPT(
        vocab_size=100, embed_dim=32,
        num_fast=1, num_deep=1,
        use_attention=True, attn_heads=4,
        use_ffn=True, ffn_mult=2.0,
        use_neuromod=True,
        use_homeostasis=True,
        use_dual_hebb=True,
        use_consolidation=True,
        use_rmsnorm=True,
    )
    model.train()
    x = torch.randint(0, 100, (2, 12))
    logits = model(x, chunk_size=6)
    loss = logits.sum()
    loss.backward()
    assert model.embed.weight.grad is not None
    print("✅ all_v035_combined")


def test_backward_compat():
    """Varsayılan parametrelerle v0.3.4 davranışı korunmalı."""
    model = MiniLiquidGPT(vocab_size=100, embed_dim=32,
                           num_fast=1, num_deep=1)
    assert model.use_ffn == False
    x = torch.randint(0, 100, (2, 10))
    logits = model(x)
    assert logits.shape == (2, 10, 100)
    print("✅ backward_compat")


if __name__ == "__main__":
    test_attention_training_backward()
    test_attention_inference_single_token()
    test_model_attention_training()

    test_ffn_output_shape()
    test_ffn_in_model()

    test_adaptive_hebb_capacity_grows()

    test_all_v035_combined()
    test_backward_compat()

    print("\n🏆 Tüm v0.3.5 testleri geçti!")
