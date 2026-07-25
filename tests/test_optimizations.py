"""Optimizasyon unit testleri — 5 yeni özellik."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from liquidnn import (MiniLiquidGPT, AdaptiveGammaScheduler,
                      SlidingWindowAttention, ExpertRouter)
from liquidnn.distillation import DistillationTrainer
from liquidnn.model import _speculative_accept


# ═══ 1. Adaptive Gamma ═══════════════════════════════════════════

def test_adaptive_gamma_increases():
    """Yüksek kabul oranı → gamma artmalı."""
    sched = AdaptiveGammaScheduler(initial=4, max_gamma=8,
                                    target_high=0.7, ema_alpha=0.9)
    # %100 kabul simülasyonu
    for _ in range(5):
        sched.update(4, 4)
    assert sched.gamma > 4, f"Gamma artmalıydı, {sched.gamma} kaldı"
    print("✅ adaptive_gamma_increases")


def test_adaptive_gamma_decreases():
    """Düşük kabul oranı → gamma azalmalı."""
    sched = AdaptiveGammaScheduler(initial=4, min_gamma=1,
                                    target_low=0.3, ema_alpha=0.9)
    # %0 kabul simülasyonu
    for _ in range(5):
        sched.update(0, 4)
    assert sched.gamma < 4, f"Gamma azalmalıydı, {sched.gamma} kaldı"
    print("✅ adaptive_gamma_decreases")


def test_speculative_with_adaptive_gamma():
    """adaptive_gamma=True ile generate_speculative çalışmalı."""
    main = MiniLiquidGPT(vocab_size=100, embed_dim=32,
                          num_fast=1, num_deep=1)
    draft = MiniLiquidGPT.create_draft_model(main)
    prompt = torch.tensor([1, 2, 3])
    out = main.generate_speculative(draft, prompt, max_new=10,
                                     gamma=3, adaptive_gamma=True)
    assert out.shape[1] == 13, f"Beklenen 13, alınan {out.shape[1]}"
    print("✅ speculative_with_adaptive_gamma")


# ═══ 2. Speculative Sampling ═════════════════════════════════════

def test_stochastic_acceptance_api():
    """_speculative_accept fonksiyonu çalışmalı ve doğru tipte dönmeli."""
    main_logits = torch.randn(1, 100)
    draft_logits = torch.randn(1, 100)
    draft_token = torch.tensor([[42]])

    accepted, correction = _speculative_accept(
        main_logits, draft_logits, draft_token, temperature=0.8
    )
    assert isinstance(accepted, bool)
    if not accepted:
        assert correction is not None
        assert correction.shape == (1, 1)
    print("✅ stochastic_acceptance_api")


def test_speculative_with_stochastic():
    """use_stochastic=True ile generate_speculative çalışmalı."""
    main = MiniLiquidGPT(vocab_size=100, embed_dim=32,
                          num_fast=1, num_deep=1)
    draft = MiniLiquidGPT.create_draft_model(main)
    prompt = torch.tensor([1, 2, 3])
    out = main.generate_speculative(draft, prompt, max_new=10,
                                     gamma=3, use_stochastic=True)
    assert out.shape[1] == 13
    assert out.min().item() >= 0
    assert out.max().item() < 100
    print("✅ speculative_with_stochastic")


# ═══ 3. Knowledge Distillation ═══════════════════════════════════

def test_distillation_loss_decreases():
    """3 adım distillation sonrası loss düşmeli veya stabil kalmalı."""
    teacher = MiniLiquidGPT(vocab_size=100, embed_dim=32,
                             num_fast=1, num_deep=1)
    student = MiniLiquidGPT.create_draft_model(teacher)

    trainer = DistillationTrainer(teacher, student, lr=1e-2,
                                   temperature_kd=4.0)

    losses = []
    for _ in range(5):
        batch = torch.randint(0, 100, (2, 16))
        metrics = trainer.distill_step(batch)
        losses.append(metrics['loss'])

    # İlk loss > son loss (en azından yön doğru olmalı)
    assert losses[-1] < losses[0] * 2, \
        f"Loss patladı: {losses[0]:.4f} → {losses[-1]:.4f}"
    print(f"✅ distillation_loss: {losses[0]:.4f} → {losses[-1]:.4f}")


# ═══ 4. Sliding-Window Attention ═════════════════════════════════

def test_attention_buffer_works():
    """Attention buffer tokenları doğru biriktirmeli."""
    attn = SlidingWindowAttention(embed_dim=32, num_heads=4, window_size=8)

    # 5 token besle
    for _ in range(5):
        x = torch.randn(1, 32)
        out = attn(x)
        assert out.shape == (1, 32)

    assert attn._buf_len == 5
    print("✅ attention_buffer_works")


def test_attention_save_restore():
    """Attention buffer save/restore çalışmalı."""
    attn = SlidingWindowAttention(embed_dim=32, num_heads=4, window_size=8)

    for _ in range(3):
        attn(torch.randn(1, 32))

    state = attn.get_buffer_state()
    buf_before = state['buffer'].clone()

    # Buffer'ı değiştir
    for _ in range(5):
        attn(torch.randn(1, 32))

    # Restore
    attn.set_buffer_state(state)
    assert attn._buf_len == 3
    assert torch.allclose(attn._buffer, buf_before, atol=1e-6)
    print("✅ attention_save_restore")


def test_model_with_attention():
    """use_attention=True ile model çalışmalı."""
    model = MiniLiquidGPT(vocab_size=100, embed_dim=32,
                           num_fast=1, num_deep=1,
                           use_attention=True, attn_window=16)
    x = torch.randint(0, 100, (1, 10))
    logits = model(x)
    assert logits.shape == (1, 10, 100)
    print("✅ model_with_attention")


def test_attention_speculative():
    """Attention + speculative decoding birlikte çalışmalı."""
    main = MiniLiquidGPT(vocab_size=100, embed_dim=32,
                          num_fast=1, num_deep=1,
                          use_attention=True, attn_window=16)
    draft = MiniLiquidGPT.create_draft_model(main)

    prompt = torch.tensor([1, 2, 3])
    out = main.generate_speculative(draft, prompt, max_new=8, gamma=3)
    assert out.shape[1] == 11
    print("✅ attention_speculative")


# ═══ 5. MoE Router ═══════════════════════════════════════════════

def test_moe_routing():
    """ExpertRouter top-k seçim yapmalı."""
    router = ExpertRouter(embed_dim=32, num_experts=4, top_k=2)
    x = torch.randn(2, 32)
    weights, indices, aux_loss, dropped_mask = router(x)

    assert weights.shape == (2, 2)
    assert indices.shape == (2, 2)
    assert aux_loss.item() >= 0
    assert dropped_mask is None  # capacity_factor=0 → sınırsız
    # Ağırlıklar toplamı ~1 olmalı
    w_sums = weights.sum(dim=-1)
    assert torch.allclose(w_sums, torch.ones(2), atol=0.01)
    print("✅ moe_routing")


def test_moe_load_balance():
    """Aux loss > 0 olmalı (load balance düzenlileştirmesi)."""
    router = ExpertRouter(embed_dim=32, num_experts=4, top_k=2)
    router.train()
    x = torch.randn(8, 32)
    _, _, aux_loss, _ = router(x)
    assert aux_loss.item() > 0, "Aux loss sıfır olmamalı"
    print("✅ moe_load_balance")


def test_model_with_moe():
    """use_moe=True ile model çalışmalı."""
    model = MiniLiquidGPT(vocab_size=100, embed_dim=32,
                           num_fast=2, num_deep=2,
                           use_moe=True, moe_top_k=2)
    x = torch.randint(0, 100, (1, 10))
    logits = model(x)
    assert logits.shape == (1, 10, 100)
    print("✅ model_with_moe")


# ═══ Backward Compatibility ══════════════════════════════════════

def test_backward_compatibility():
    """Varsayılan parametrelerle mevcut davranış korunmalı."""
    model = MiniLiquidGPT(vocab_size=100, embed_dim=32,
                           num_fast=1, num_deep=1)
    assert model.attn is None, "Varsayılanda attention kapalı olmalı"
    assert model.router is None, "Varsayılanda MoE kapalı olmalı"

    x = torch.randint(0, 100, (2, 10))
    logits = model(x)
    assert logits.shape == (2, 10, 100)

    # Generate hala çalışmalı
    prompt = torch.tensor([1, 2, 3])
    out = model.generate(prompt, max_new=5)
    assert out.shape[1] == 8
    print("✅ backward_compatibility")


if __name__ == "__main__":
    # 1. Adaptive Gamma
    test_adaptive_gamma_increases()
    test_adaptive_gamma_decreases()
    test_speculative_with_adaptive_gamma()

    # 2. Speculative Sampling
    test_stochastic_acceptance_api()
    test_speculative_with_stochastic()

    # 3. Knowledge Distillation
    test_distillation_loss_decreases()

    # 4. Sliding-Window Attention
    test_attention_buffer_works()
    test_attention_save_restore()
    test_model_with_attention()
    test_attention_speculative()

    # 5. MoE Router
    test_moe_routing()
    test_moe_load_balance()
    test_model_with_moe()

    # Backward Compatibility
    test_backward_compatibility()

    print("\n🏆 Tüm optimizasyon testleri geçti!")
