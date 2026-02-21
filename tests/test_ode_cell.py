"""LiquidODECell unit testleri."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from liquidnn import LiquidODECell


def test_forward_shape():
    cell = LiquidODECell(64, 32, ode_steps=3)
    x = torch.randn(4, 64)
    h = torch.zeros(4, 32)
    h_new = cell(x, h)
    assert h_new.shape == (4, 32), f"Beklenen (4,32), alınan {h_new.shape}"
    print("✅ ode_forward_shape")


def test_euler_vs_rk2():
    """steps=1 (Euler) ve steps=3 (RK2) farklı sonuç vermeli."""
    cell_e = LiquidODECell(32, 32, ode_steps=1, use_plasticity=False)
    cell_r = LiquidODECell(32, 32, ode_steps=3, use_plasticity=False)
    # Aynı ağırlıkları paylaş
    cell_r.load_state_dict(cell_e.state_dict())

    x = torch.randn(1, 32)
    h = torch.zeros(1, 32)
    h_e = cell_e(x, h, enable_plasticity=False)
    h_r = cell_r(x, h, enable_plasticity=False)
    assert not torch.allclose(h_e, h_r, atol=1e-3), \
        "Euler ve RK2 aynı sonuç vermemeli"
    print("✅ euler_vs_rk2")


def test_plasticity_toggle():
    cell = LiquidODECell(32, 32, ode_steps=2, use_plasticity=True)
    x = torch.randn(2, 32)
    h = torch.zeros(2, 32)

    # Plast OFF
    cell(x, h, enable_plasticity=False)
    assert cell.hebb_info['ih'] == 0.0, "Plast OFF → Hebb 0 olmalı"

    # Plast ON
    cell.reset_hebb()
    cell(x, h, enable_plasticity=True)
    assert cell.hebb_info['ih'] > 0.0, "Plast ON → Hebb > 0 olmalı"
    print("✅ plasticity_toggle")


def test_sequential_accumulation():
    """Birden fazla forward'da Hebb birikir."""
    cell = LiquidODECell(32, 32, ode_steps=2, use_plasticity=True)
    h = torch.zeros(1, 32)

    norms = []
    for _ in range(5):
        x = torch.randn(1, 32)
        h = cell(x, h, enable_plasticity=True)
        norms.append(cell.hebb_info['ih'])

    # Norm artmalı (en azından ilk birkaç adımda)
    assert norms[-1] > norms[0], "Hebb birikimi artmalı"
    print("✅ sequential_accumulation")


if __name__ == "__main__":
    test_forward_shape()
    test_euler_vs_rk2()
    test_plasticity_toggle()
    test_sequential_accumulation()
    print("\n🏆 Tüm ODE cell testleri geçti!")
