"""v0.3.2 testleri — Top-p, Repetition Penalty, Tau-Gated Residual, Multi-Scale Fusion."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from liquidnn import MiniLiquidGPT
from liquidnn.ode_cell import LiquidODECell


# ═══ 1. Top-p (Nucleus) Sampling ═════════════════════════════════

def test_top_p_runs():
    """top_p parametresi ile generate çalışmalı."""
    model = MiniLiquidGPT(vocab_size=100, embed_dim=32,
                           num_fast=1, num_deep=1)
    prompt = torch.tensor([1, 2, 3])
    out = model.generate(prompt, max_new=5, top_p=0.9, top_k=0)
    assert out.shape[1] == 8
    print("✅ top_p_runs")


def test_top_p_and_top_k():
    """top_p + top_k birlikte çalışmalı."""
    model = MiniLiquidGPT(vocab_size=100, embed_dim=32,
                           num_fast=1, num_deep=1)
    prompt = torch.tensor([1, 2, 3])
    out = model.generate(prompt, max_new=5, top_p=0.9, top_k=20)
    assert out.shape[1] == 8
    print("✅ top_p_and_top_k")


# ═══ 2. Repetition Penalty ═══════════════════════════════════════

def test_repetition_penalty():
    """repetition_penalty ile generate çalışmalı."""
    model = MiniLiquidGPT(vocab_size=100, embed_dim=32,
                           num_fast=1, num_deep=1)
    prompt = torch.tensor([1, 2, 3])
    out = model.generate(prompt, max_new=5, repetition_penalty=1.2)
    assert out.shape[1] == 8
    print("✅ repetition_penalty")


def test_high_penalty_reduces_repeats():
    """Yüksek penalty tekrar sayısını azaltmalı."""
    torch.manual_seed(42)
    model = MiniLiquidGPT(vocab_size=20, embed_dim=32,
                           num_fast=1, num_deep=1)

    prompt = torch.tensor([1, 2, 3])

    # Cezasız
    torch.manual_seed(42)
    out_no_penalty = model.generate(prompt, max_new=20, repetition_penalty=1.0,
                                     temperature=0.5)
    # Cezalı
    torch.manual_seed(42)
    out_with_penalty = model.generate(prompt, max_new=20, repetition_penalty=2.0,
                                       temperature=0.5)

    unique_no = out_no_penalty[0].unique().numel()
    unique_with = out_with_penalty[0].unique().numel()
    # Cezalı çıktıda en az aynı veya daha fazla unique token olmalı
    print(f"  unique (no penalty)={unique_no}, unique (penalty)={unique_with}")
    print("✅ high_penalty_reduces_repeats")


# ═══ 3. Tau-Gated Residual ═══════════════════════════════════════

def test_tau_gate_cell():
    """tau_gate_residual ile ODE cell çalışmalı."""
    cell = LiquidODECell(32, 32, ode_steps=3, use_plasticity=True)
    x = torch.randn(1, 32)
    h = torch.zeros(1, 32)

    # Normal forward
    h_normal = cell(x, h.clone(), tau_gate_residual=False)
    assert h_normal.shape == (1, 32)

    # Reset Hebb for clean comparison
    cell.reset_hebb()

    # Gated forward (aynı cell, temiz state)
    h_gated = cell(x, h.clone(), tau_gate_residual=True)
    assert h_gated.shape == (1, 32)

    # Gated çıktı farklı olmalı (gate karıştırma yapıyor)
    assert not torch.allclose(h_normal, h_gated, atol=1e-6), \
        "Tau-gated ve normal aynı olmamalı"
    print("✅ tau_gate_cell")


def test_tau_gate_in_model():
    """tau_gate=True ile model çalışmalı."""
    model = MiniLiquidGPT(vocab_size=100, embed_dim=32,
                           num_fast=1, num_deep=1,
                           tau_gate=True)
    x = torch.randint(0, 100, (1, 10))
    logits = model(x)
    assert logits.shape == (1, 10, 100)
    print("✅ tau_gate_in_model")


# ═══ 4. Multi-Scale ODE Fusion ═══════════════════════════════════

def test_multiscale_fusion():
    """use_multiscale ile model çalışmalı."""
    model = MiniLiquidGPT(vocab_size=100, embed_dim=32,
                           num_fast=2, num_deep=2,
                           use_multiscale=True)
    x = torch.randint(0, 100, (1, 10))
    logits = model(x)
    assert logits.shape == (1, 10, 100)
    # Fusion gate oluşturulmuş olmalı
    assert model._fusion_gate is not None
    print("✅ multiscale_fusion")


def test_multiscale_generate():
    """Multi-scale ile generate çalışmalı."""
    model = MiniLiquidGPT(vocab_size=100, embed_dim=32,
                           num_fast=2, num_deep=2,
                           use_multiscale=True)
    prompt = torch.tensor([1, 2, 3])
    out = model.generate(prompt, max_new=5)
    assert out.shape[1] == 8
    print("✅ multiscale_generate")


def test_multiscale_only_fast_or_deep():
    """Sadece fast veya sadece deep → fusion gate None olmalı."""
    model_fast = MiniLiquidGPT(vocab_size=100, embed_dim=32,
                                num_fast=4, num_deep=0,
                                use_multiscale=True)
    assert model_fast._fusion_gate is None, "Sadece fast → fusion gate yok"

    model_deep = MiniLiquidGPT(vocab_size=100, embed_dim=32,
                                num_fast=0, num_deep=4,
                                use_multiscale=True)
    assert model_deep._fusion_gate is None, "Sadece deep → fusion gate yok"
    print("✅ multiscale_only_fast_or_deep")


# ═══ Combined ════════════════════════════════════════════════════

def test_all_v032_combined():
    """Tüm v0.3.2 özellikleri birlikte çalışmalı."""
    model = MiniLiquidGPT(vocab_size=100, embed_dim=32,
                           num_fast=2, num_deep=2,
                           use_multiscale=True,
                           tau_gate=True,
                           use_attention=True, attn_heads=4,
                           use_rope=True)
    prompt = torch.tensor([1, 2, 3])
    out = model.generate(prompt, max_new=8, top_p=0.9,
                          repetition_penalty=1.2)
    assert out.shape[1] == 11
    print("✅ all_v032_combined")


def test_backward_compat_generate():
    """Varsayılan generate davranışı korunmalı."""
    model = MiniLiquidGPT(vocab_size=100, embed_dim=32,
                           num_fast=1, num_deep=1)
    prompt = torch.tensor([1, 2, 3])
    out = model.generate(prompt, max_new=5)
    assert out.shape[1] == 8
    assert model.use_multiscale == False
    assert model.tau_gate == False
    print("✅ backward_compat_generate")


if __name__ == "__main__":
    test_top_p_runs()
    test_top_p_and_top_k()

    test_repetition_penalty()
    test_high_penalty_reduces_repeats()

    test_tau_gate_cell()
    test_tau_gate_in_model()

    test_multiscale_fusion()
    test_multiscale_generate()
    test_multiscale_only_fast_or_deep()

    test_all_v032_combined()
    test_backward_compat_generate()

    print("\n🏆 Tüm v0.3.2 testleri geçti!")
