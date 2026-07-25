"""
Toplu logit yolu (fused head) testleri.

forward() artık per-token vokab matmul'u yerine sekans sonunda tek
matmul yapıyor — sonuç eski per-token yolla aynı olmalı.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from liquidnn import MiniLiquidGPT


def _per_token_logits(model, x, enable_plasticity, chunk_size=16):
    """Eski yol: her token'da logit hesapla, sonra yığ."""
    B, T = x.shape
    model.init_hidden(B, x.device)
    outs = []
    for t in range(T):
        if t > 0 and t % chunk_size == 0:
            model._hiddens = [h.detach() for h in model._hiddens]
            for cell in model.cells:
                cell.detach_hebb()
        outs.append(model.forward_token(x[:, t], t, enable_plasticity))
    return torch.stack(outs, dim=1)


def test_fused_head_matches_per_token():
    """Çıplak baseline: fused ve per-token yol aynı logit üretmeli."""
    torch.manual_seed(0)
    model = MiniLiquidGPT(vocab_size=120, embed_dim=32,
                          num_fast=1, num_deep=1)
    model.eval()  # dropout kapalı — iki yol karşılaştırılabilir olsun
    x = torch.randint(0, 120, (2, 20))

    model.reset_hebb()
    fused = model(x, enable_plasticity=True, chunk_size=16)

    model.reset_hebb()
    ref = _per_token_logits(model, x, True)

    assert fused.shape == ref.shape == (2, 20, 120)
    assert torch.allclose(fused, ref, atol=1e-5), \
        f"max fark: {(fused - ref).abs().max().item()}"
    print("✅ fused_head_matches_per_token")


def test_fused_head_with_features():
    """FFN + RMSNorm + tau-gate + dual-hebb ile de eşleşmeli."""
    torch.manual_seed(1)
    model = MiniLiquidGPT(vocab_size=120, embed_dim=32,
                          num_fast=1, num_deep=1,
                          use_ffn=True, use_rmsnorm=True, tau_gate=True,
                          use_dual_hebb=True)
    model.eval()
    x = torch.randint(0, 120, (2, 12))

    model.reset_hebb()
    fused = model(x, enable_plasticity=True, chunk_size=16)

    model.reset_hebb()
    ref = _per_token_logits(model, x, True)

    assert torch.allclose(fused, ref, atol=1e-5), \
        f"max fark: {(fused - ref).abs().max().item()}"
    print("✅ fused_head_with_features")


def test_neuromod_uses_per_token_path():
    """Nöromodülasyon per-token logit ister — fused yol devre dışı kalmalı."""
    torch.manual_seed(2)
    model = MiniLiquidGPT(vocab_size=120, embed_dim=32,
                          num_fast=1, num_deep=1, use_neuromod=True)
    model.eval()
    x = torch.randint(0, 120, (1, 8))
    model.reset_hebb()
    logits = model(x, enable_plasticity=True, chunk_size=16)
    assert logits.shape == (1, 8, 120)
    print("✅ neuromod_uses_per_token_path")


def test_fused_head_backward():
    """Fused yol üzerinden gradyan akmalı."""
    torch.manual_seed(3)
    model = MiniLiquidGPT(vocab_size=120, embed_dim=32,
                          num_fast=1, num_deep=1)
    model.train()
    x = torch.randint(0, 120, (2, 12))
    model.reset_hebb()
    logits = model(x, enable_plasticity=True, chunk_size=8)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 120), x.reshape(-1))
    loss.backward()
    assert model.embed.weight.grad is not None
    assert not torch.isnan(model.embed.weight.grad).any()
    print("✅ fused_head_backward")


if __name__ == "__main__":
    test_fused_head_matches_per_token()
    test_fused_head_with_features()
    test_neuromod_uses_per_token_path()
    test_fused_head_backward()
    print("\n🎉 Tüm fused head testleri geçti!")
