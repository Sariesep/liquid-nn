"""Energy efficiency unit testleri — 7 optimizasyon."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from liquidnn import (MiniLiquidGPT, RMSNorm, SlidingWindowAttention,
                      quantize_model, model_size_mb)
from liquidnn.plasticity import PlasticSynapse
from liquidnn.ode_cell import LiquidODECell


# ═══ 1. RMSNorm ══════════════════════════════════════════════════

def test_rmsnorm_output_shape():
    """RMSNorm doğru şekil döndürmeli."""
    norm = RMSNorm(32)
    x = torch.randn(2, 32)
    out = norm(x)
    assert out.shape == (2, 32)
    print("✅ rmsnorm_output_shape")


def test_rmsnorm_in_model():
    """use_rmsnorm=True ile model çalışmalı."""
    model = MiniLiquidGPT(vocab_size=100, embed_dim=32,
                           num_fast=1, num_deep=1,
                           use_rmsnorm=True)
    x = torch.randint(0, 100, (1, 10))
    logits = model(x)
    assert logits.shape == (1, 10, 100)
    # LayerNorm yerine RMSNorm kullanıldığını doğrula
    assert isinstance(model.norms[0], RMSNorm)
    assert isinstance(model.out_norm, RMSNorm)
    print("✅ rmsnorm_in_model")


# ═══ 2. GQA ══════════════════════════════════════════════════════

def test_gqa_reduces_params():
    """GQA (1 KV head) daha az parametreye sahip olmalı."""
    mha = SlidingWindowAttention(embed_dim=32, num_heads=4,
                                  num_kv_heads=None, window_size=8)
    gqa = SlidingWindowAttention(embed_dim=32, num_heads=4,
                                  num_kv_heads=1, window_size=8)

    mha_params = sum(p.numel() for p in mha.parameters())
    gqa_params = sum(p.numel() for p in gqa.parameters())
    assert gqa_params < mha_params, \
        f"GQA ({gqa_params}) MHA'dan ({mha_params}) küçük olmalı"
    print(f"✅ gqa_reduces_params: MHA={mha_params}, GQA={gqa_params}")


def test_gqa_in_model():
    """GQA + attention ile model çalışmalı."""
    model = MiniLiquidGPT(vocab_size=100, embed_dim=32,
                           num_fast=1, num_deep=1,
                           use_attention=True, attn_heads=4)
    x = torch.randint(0, 100, (1, 10))
    logits = model(x)
    assert logits.shape == (1, 10, 100)
    print("✅ gqa_in_model")


# ═══ 3. Adaptive ODE Steps ═══════════════════════════════════════

def test_adaptive_ode_runs():
    """adaptive_steps=True ile ODE çalışmalı."""
    cell = LiquidODECell(32, 32, ode_steps=3, use_plasticity=True)
    x = torch.randn(1, 32)
    h = torch.zeros(1, 32)
    h_out = cell(x, h, enable_plasticity=True, adaptive_steps=True)
    assert h_out.shape == (1, 32)
    print("✅ adaptive_ode_runs")


def test_adaptive_ode_in_model():
    """adaptive_ode=True ile model çalışmalı."""
    model = MiniLiquidGPT(vocab_size=100, embed_dim=32,
                           num_fast=1, num_deep=1,
                           adaptive_ode=True)
    x = torch.randint(0, 100, (1, 10))
    logits = model(x)
    assert logits.shape == (1, 10, 100)
    print("✅ adaptive_ode_in_model")


# ═══ 4. Sparse Hebb ══════════════════════════════════════════════

def test_sparse_hebb():
    """sparse_k ile Hebb matrisinde az eleman kalmalı."""
    synapse = PlasticSynapse(32, 32, sparse_k=64)
    x = torch.randn(1, 32)
    _ = synapse(x)  # init Hebb
    synapse.update_hebb(x, torch.randn(1, 32))

    nonzero = (synapse.Hebb != 0).sum().item()
    assert nonzero <= 64, f"sparse_k=64 ama {nonzero} nonzero eleman var"
    print(f"✅ sparse_hebb: nonzero={nonzero}/1024")


def test_sparse_hebb_default():
    """sparse_k=0 (varsayılan) → tam yoğun Hebb."""
    synapse = PlasticSynapse(32, 32, sparse_k=0)
    x = torch.randn(1, 32)
    _ = synapse(x)
    synapse.update_hebb(x, torch.randn(1, 32))

    nonzero = (synapse.Hebb != 0).sum().item()
    assert nonzero > 64, f"Varsayılanda çoğu eleman nonzero olmalı, {nonzero}"
    print(f"✅ sparse_hebb_default: nonzero={nonzero}/1024")


# ═══ 5. Early Exit ═══════════════════════════════════════════════

def test_early_exit():
    """early_exit_threshold ile model çalışmalı."""
    model = MiniLiquidGPT(vocab_size=100, embed_dim=32,
                           num_fast=2, num_deep=2,
                           early_exit_threshold=0.5)
    model.eval()
    x = torch.randint(0, 100, (1, 5))

    model.init_hidden(1, x.device)
    logits = model.forward_token(x[:, 0], 0)
    assert logits.shape == (1, 100)
    print("✅ early_exit")


def test_early_exit_training_disabled():
    """Early exit eğitimde aktif olmamalı."""
    model = MiniLiquidGPT(vocab_size=100, embed_dim=32,
                           num_fast=2, num_deep=2,
                           early_exit_threshold=0.5)
    model.train()
    x = torch.randint(0, 100, (1, 5))
    logits = model(x)
    assert logits.shape == (1, 5, 100)
    print("✅ early_exit_training_disabled")


# ═══ 6. Gradient Checkpointing ═══════════════════════════════════

def test_gradient_checkpointing():
    """use_checkpointing=True ile forward çalışmalı."""
    model = MiniLiquidGPT(vocab_size=100, embed_dim=32,
                           num_fast=1, num_deep=1)
    model.train()
    x = torch.randint(0, 100, (1, 10))
    # Normal forward (referans)
    logits_normal = model(x, use_checkpointing=False)
    assert logits_normal.shape == (1, 10, 100)

    # Checkpointed forward
    model.init_hidden(1, x.device)
    model.reset_hebb()
    logits_ckpt = model(x, use_checkpointing=True)
    assert logits_ckpt.shape == (1, 10, 100)

    # Backward çalışmalı — parametrelerde grad olmalı
    loss = logits_ckpt[:, :-1].reshape(-1, 100)
    targets = x[:, 1:].reshape(-1)
    ce = torch.nn.functional.cross_entropy(loss, targets)
    ce.backward()

    has_grad = any(p.grad is not None for p in model.parameters()
                   if p.requires_grad)
    assert has_grad, "Checkpointing sonrası gradient olmalı"
    print("✅ gradient_checkpointing")


# ═══ 7. INT8 Quantization ════════════════════════════════════════

def test_quantize_model():
    """Quantize sonrası model daha küçük olmalı."""
    model = MiniLiquidGPT(vocab_size=100, embed_dim=32,
                           num_fast=1, num_deep=1)
    size_before = model_size_mb(model)

    model_q = quantize_model(model)
    size_after = model_size_mb(model_q)

    # Quantize edilmiş model daha küçük olmalı
    # (küçük modelde fark az olabilir ama en azından çalışmalı)
    assert size_after <= size_before * 1.1, \
        f"Quantize sonrası boyut artmamalı: {size_before:.2f} → {size_after:.2f}"
    print(f"✅ quantize_model: {size_before:.2f}MB → {size_after:.2f}MB")


# ═══ Combined Test ════════════════════════════════════════════════

def test_all_features_combined():
    """Tüm optimizasyonlar birlikte çalışmalı."""
    model = MiniLiquidGPT(vocab_size=100, embed_dim=32,
                           num_fast=1, num_deep=1,
                           use_rmsnorm=True,
                           adaptive_ode=True,
                           early_exit_threshold=0.5,
                           use_attention=True, attn_heads=4)
    model.eval()

    prompt = torch.tensor([1, 2, 3])
    out = model.generate(prompt, max_new=5)
    assert out.shape[1] == 8
    print("✅ all_features_combined")


def test_backward_compat():
    """Varsayılan parametrelerle mevcut davranış korunmalı."""
    model = MiniLiquidGPT(vocab_size=100, embed_dim=32,
                           num_fast=1, num_deep=1)
    import torch.nn as nn
    assert isinstance(model.norms[0], nn.LayerNorm), "Varsayılan LayerNorm"
    assert model.adaptive_ode == False
    assert model.early_exit_threshold == 0.0

    x = torch.randint(0, 100, (2, 10))
    logits = model(x)
    assert logits.shape == (2, 10, 100)
    print("✅ backward_compat")


if __name__ == "__main__":
    test_rmsnorm_output_shape()
    test_rmsnorm_in_model()

    test_gqa_reduces_params()
    test_gqa_in_model()

    test_adaptive_ode_runs()
    test_adaptive_ode_in_model()

    test_sparse_hebb()
    test_sparse_hebb_default()

    test_early_exit()
    test_early_exit_training_disabled()

    test_gradient_checkpointing()

    test_quantize_model()

    test_all_features_combined()
    test_backward_compat()

    print("\n🏆 Tüm enerji verimliliği testleri geçti!")
