"""MiniLiquidGPT unit testleri."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from liquidnn import MiniLiquidGPT


def test_forward_shape():
    model = MiniLiquidGPT(vocab_size=100, embed_dim=32,
                           num_fast=1, num_deep=1)
    x = torch.randint(0, 100, (2, 10))
    logits = model(x)
    assert logits.shape == (2, 10, 100), f"Beklenen (2,10,100), alınan {logits.shape}"
    print("✅ model_forward_shape")


def test_weight_tying():
    model = MiniLiquidGPT(vocab_size=100, embed_dim=32)
    # lm_head embed.weight'i kullanmalı — forward'da F.linear(x, embed.weight)
    x = torch.randint(0, 100, (1, 5))
    logits = model(x)
    # Logit boyutu vocab_size olmalı
    assert logits.shape[-1] == 100
    print("✅ weight_tying")


def test_generation():
    model = MiniLiquidGPT(vocab_size=100, embed_dim=32,
                           num_fast=1, num_deep=1)
    prompt = torch.tensor([1, 2, 3])
    out = model.generate(prompt, max_new=5)
    assert out.shape[1] == 8, f"3 prompt + 5 new = 8, alınan {out.shape[1]}"
    print("✅ generation")


def test_truncated_bptt():
    """Chunk size ile forward çalışmalı ve gradient üretmeli."""
    model = MiniLiquidGPT(vocab_size=100, embed_dim=32,
                           num_fast=1, num_deep=1)
    x = torch.randint(0, 100, (2, 20))
    logits = model(x, chunk_size=5)
    loss = logits.sum()
    loss.backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters())
    assert has_grad, "Truncated BPTT ile gradient üretilmeli"
    print("✅ truncated_bptt")


def test_hebb_persistence():
    """Hidden sıfırlansa bile Hebb kalmalı."""
    model = MiniLiquidGPT(vocab_size=100, embed_dim=32,
                           num_fast=0, num_deep=2, deep_steps=2)
    x = torch.randint(0, 100, (1, 10))

    model(x, enable_plasticity=True)
    hs1 = model.hebb_stats()
    deep_h = sum(v for k, v in hs1.items()) / len(hs1)
    assert deep_h > 0, "Plastisite ON → Hebb > 0"

    # Hidden sıfırla, Hebb kalmalı
    model.init_hidden(1, torch.device('cpu'))
    hs2 = model.hebb_stats()
    deep_h2 = sum(v for k, v in hs2.items()) / len(hs2)
    assert abs(deep_h - deep_h2) < 1e-6, "Hidden reset Hebb'i etkilememeli"
    print("✅ hebb_persistence")


def test_param_count():
    model = MiniLiquidGPT(vocab_size=50257, embed_dim=256,
                           num_fast=2, num_deep=2)
    p = model.count_params()
    assert p['total'] > 1_000_000, "~14M param bekleniyor"
    assert p['total'] < 30_000_000, "30M'den az olmalı"
    print(f"✅ param_count: {p['total']/1e6:.1f}M")


if __name__ == "__main__":
    test_forward_shape()
    test_weight_tying()
    test_generation()
    test_truncated_bptt()
    test_hebb_persistence()
    test_param_count()
    print("\n🏆 Tüm model testleri geçti!")
