"""v0.3.3 testleri — Teorik ve Biyolojik İyileştirmeler."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn.functional as F
from liquidnn import MiniLiquidGPT
from liquidnn.ode_cell import LiquidODECell
from liquidnn.plasticity import PlasticSynapse


# ═══ 1. Decoupled Hebbian Capacity ══════════════════════════════

def test_hebb_capacity_independent_of_w():
    """Hebb_capacity'nin W normundan bağımsız çalıştığını doğrula."""
    syn = PlasticSynapse(10, 10)
    syn.W.data.zero_()  # W normu 0 olursa eski mantıkta plastisite de 0 olurdu
    assert syn.W.data.norm() == 0.0

    # Sınır formülü büyüme faktörü içerir: cap * (1 + 0.1*log1p(adım))
    import math
    growth = 1.0 + 0.1 * math.log1p(1)
    expected_cap = F.softplus(syn.hebb_capacity).item() * growth

    pre = torch.randn(1, 10)
    post = torch.randn(1, 10) * 100  # Çok büyük güncelleme

    syn.update_hebb(pre, post)

    # Norm sıfır olmamalı, kapasitede kırpılmış olmalı
    actual_norm = syn.Hebb.norm().item()
    assert actual_norm > 0.0, "Hebb güncellenmeli (W=0 olsa bile)"
    assert actual_norm <= expected_cap * 1.01, "Kapasiteyi aşmamalı"
    print("✅ hebb_capacity_independent_of_w")


# ═══ 2. MoE Unbalanced Updates ══════════════════════════════════

def test_plasticity_moe_weight_scaling():
    """update_hebb fonksiyonunun moe_weight ile ölçeklendiğini doğrula."""
    syn1 = PlasticSynapse(10, 10)
    syn2 = PlasticSynapse(10, 10)
    # Parametreleri eşitle
    syn2.load_state_dict(syn1.state_dict())

    pre = torch.randn(1, 10)
    post = torch.randn(1, 10)

    # weight=1.0 (tam güncelleme)
    syn1.update_hebb(pre, post, moe_weight=1.0)
    norm1 = syn1.hebb_norm

    # weight=0.1 (çok küçük güncelleme)
    syn2.update_hebb(pre, post, moe_weight=0.1)
    norm2 = syn2.hebb_norm

    assert norm1 > norm2, "moe_weight=1.0 daha çok güncellemeli"
    # Oran decay dahil olduğu için tam 10x olmayabilir, ama ciddi fark etmeli
    assert norm1 / norm2 > 2.0, "moe_weight ciddi fark yaratmalı"
    print("✅ plasticity_moe_weight_scaling")


def test_model_with_moe_passes_weight():
    """Modelin MoE modunda forward_token'ın çökmediğini doğrula."""
    model = MiniLiquidGPT(vocab_size=100, embed_dim=32,
                           num_fast=2, num_deep=2, use_moe=True)
    x = torch.randint(0, 100, (1, 10))
    logits = model(x)
    assert logits.shape == (1, 10, 100)
    print("✅ model_with_moe_passes_weight")


# ═══ 3. Biological RK2 Plausibility ═════════════════════════════

def test_rk2_biological_timing():
    """ode_cell içindeki update_hebb çağrısının steps'e uyumunu doğrula."""
    cell1 = LiquidODECell(32, 32, ode_steps=1, use_plasticity=True)
    cell3 = LiquidODECell(32, 32, ode_steps=3, use_plasticity=True)
    
    # Parametreleri eşitle (tau net dahil)
    cell3.load_state_dict(cell1.state_dict())
    
    x = torch.randn(1, 32)
    h = torch.zeros(1, 32)
    
    h1 = cell1(x, h.clone(), adaptive_steps=False)
    # RK2 loop'ta hebb 3 kez güncellenecek, ama step=3 olduğu için 1/3 gücünde
    h3 = cell3(x, h.clone(), adaptive_steps=False)
    
    assert cell1.hebb_info['ih'] > 0
    assert cell3.hebb_info['ih'] > 0
    
    # 3 adımlı hücrenin hebb normu 1 adımlıdan aşırı büyük olmamalı (normalize edildiği için)
    ratio = cell3.hebb_info['ih'] / (cell1.hebb_info['ih'] + 1e-8)
    # İdealde ~1 civarı olmalı, RK2 entegrasyon farkından sapma olabilir ama 3x olmamalı
    assert ratio < 2.0, f"RK2 Hebb birikimi anormal: {ratio:.2f}x"
    print("✅ rk2_biological_timing")


# ═══ Combined ════════════════════════════════════════════════════

def test_all_v033_combined():
    """Gelişmiş ODE, MoE ve Plastisite özelliklerinin stabil çalışması."""
    model = MiniLiquidGPT(vocab_size=100, embed_dim=32,
                           num_fast=2, num_deep=2,
                           use_moe=True, 
                           tau_gate=True,
                           use_multiscale=True)
    # Training step
    model.train()
    x = torch.randint(0, 100, (1, 10))
    logits = model(x)
    loss = logits.sum() + model._aux_loss
    loss.backward()
    
    assert model.embed.weight.grad is not None
    print("✅ all_v033_combined")


if __name__ == "__main__":
    test_hebb_capacity_independent_of_w()
    
    test_plasticity_moe_weight_scaling()
    test_model_with_moe_passes_weight()
    
    test_rk2_biological_timing()
    
    test_all_v033_combined()

    print("\n🏆 Tüm v0.3.3 testleri geçti!")
