"""PlasticSynapse unit testleri."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from liquidnn import PlasticSynapse


def test_forward_shape():
    syn = PlasticSynapse(64, 32)
    x = torch.randn(4, 64)
    y = syn(x)
    assert y.shape == (4, 32), f"Beklenen (4,32), alınan {y.shape}"
    print("✅ forward_shape")


def test_hebb_update():
    syn = PlasticSynapse(64, 32)
    x = torch.randn(4, 64)
    y = syn(x)
    assert syn.hebb_norm == 0.0, "Hebb başlangıçta 0 olmalı"

    syn.update_hebb(x, y)
    assert syn.hebb_norm > 0.0, "Hebb güncellemeden sonra > 0 olmalı"
    print("✅ hebb_update")


def test_hebb_reset():
    syn = PlasticSynapse(64, 32)
    x = torch.randn(4, 64)
    y = syn(x)
    syn.update_hebb(x, y)
    assert syn.hebb_norm > 0

    syn.reset_hebb()
    assert syn.Hebb is None, "Reset sonrası Hebb None olmalı"
    print("✅ hebb_reset")


def test_hebb_norm_bounded():
    """Hebb normu öğrenilebilir kapasiteyi aşmamalı."""
    syn = PlasticSynapse(64, 32)
    x = torch.randn(4, 64)
    for _ in range(100):
        y = syn(x)
        syn.update_hebb(x, y)

    max_allowed = F.softplus(syn.hebb_capacity).item()
    assert syn.hebb_norm <= max_allowed * 1.01, \
        f"Hebb norm {syn.hebb_norm:.4f} > limit {max_allowed:.4f}"
    print("✅ hebb_norm_bounded")


def test_hebb_detach():
    syn = PlasticSynapse(64, 32)
    x = torch.randn(4, 64)
    y = syn(x)
    syn.update_hebb(x, y)
    old_norm = syn.hebb_norm

    syn.detach_hebb()
    assert abs(syn.hebb_norm - old_norm) < 1e-6, "Detach norm değiştirmemeli"
    assert not syn.Hebb.requires_grad, "Detach sonrası grad olmamalı"
    print("✅ hebb_detach")


if __name__ == "__main__":
    test_forward_shape()
    test_hebb_update()
    test_hebb_reset()
    test_hebb_norm_bounded()
    test_hebb_detach()
    print("\n🏆 Tüm plastisite testleri geçti!")
