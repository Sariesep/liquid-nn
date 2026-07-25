"""
Türevlenebilir plastisite testleri (v0.4).

İlk ablasyonun tanısı: update_hebb @torch.no_grad ile işaretliydi,
eta/decay/hebb_capacity hiç gradyan almıyordu — model izleri yazmayı
öğrenemiyordu. Bu testler yazma yolunun artık öğrenilebilir olduğunu
garanti eder.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from liquidnn import MiniLiquidGPT, PlasticSynapse


def _train_step_grads(model, x):
    model.train()
    model.reset_hebb()
    logits = model(x, enable_plasticity=True, chunk_size=8)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           x.reshape(-1))
    loss.backward()


def test_eta_decay_receive_gradients():
    """Derin katmanların eta/decay/capacity'si gradyan almalı."""
    torch.manual_seed(0)
    model = MiniLiquidGPT(vocab_size=120, embed_dim=32,
                          num_fast=1, num_deep=1)
    x = torch.randint(0, 120, (2, 16))
    _train_step_grads(model, x)

    deep = model.cells[1]  # plastisiteli katman
    for syn_name in ('syn_ih', 'syn_hh'):
        syn = getattr(deep, syn_name)
        for pname in ('log_eta', 'logit_decay', 'alpha'):
            g = getattr(syn, pname).grad
            assert g is not None, f"{syn_name}.{pname} gradyan almadı"
            assert g.abs().sum() > 0, f"{syn_name}.{pname} gradyanı sıfır"
        # capacity yalnızca norm sınırı bağlayınca sıfırdan farklı gradyan
        # alır (clamp doygunken türev 0) — grafiğe katılması yeterli
        assert syn.hebb_capacity.grad is not None, \
            f"{syn_name}.hebb_capacity grafiğe katılmadı"
    print("✅ eta_decay_receive_gradients")


def test_fast_layer_plast_params_stay_gradless():
    """Hızlı katmanlar (plastisite kapalı) iz yazmaz — eta gradyansız."""
    torch.manual_seed(1)
    model = MiniLiquidGPT(vocab_size=120, embed_dim=32,
                          num_fast=1, num_deep=1)
    x = torch.randint(0, 120, (2, 16))
    _train_step_grads(model, x)

    fast = model.cells[0]
    g = fast.syn_ih.log_eta.grad
    assert g is None or g.abs().sum() == 0
    print("✅ fast_layer_plast_params_stay_gradless")


def test_consolidation_importance_stays_out_of_graph():
    """Konsolidasyon EMA'sı grafiğe girmemeli (bellek sızıntısı önlemi)."""
    torch.manual_seed(2)
    syn = PlasticSynapse(8, 8, use_consolidation=True)
    pre = torch.randn(4, 8, requires_grad=True)
    post = syn(pre)
    syn.update_hebb(pre, post)
    assert syn._importance is not None
    assert not syn._importance.requires_grad
    assert syn.Hebb.requires_grad  # iz ise grafikte olmalı
    print("✅ consolidation_importance_stays_out_of_graph")


def test_inference_path_builds_no_graph():
    """torch.no_grad altında (çıkarım) iz grafiksiz kalmalı."""
    torch.manual_seed(3)
    model = MiniLiquidGPT(vocab_size=120, embed_dim=32,
                          num_fast=1, num_deep=1)
    model.eval()
    x = torch.randint(0, 120, (1, 12))
    model.reset_hebb()
    with torch.no_grad():
        model(x, enable_plasticity=True, chunk_size=8)
    deep = model.cells[1]
    assert deep.syn_ih.Hebb is not None
    assert not deep.syn_ih.Hebb.requires_grad
    print("✅ inference_path_builds_no_graph")


def test_two_consecutive_batches_backward():
    """reset_hebb + detach zinciri: ardışık batch'ler çift-backward
    hatası vermemeli."""
    torch.manual_seed(4)
    model = MiniLiquidGPT(vocab_size=120, embed_dim=32,
                          num_fast=1, num_deep=1)
    for _ in range(2):
        x = torch.randint(0, 120, (2, 20))
        _train_step_grads(model, x)  # chunk=8 → sekans içi detach da test edilir
        model.zero_grad(set_to_none=True)
    print("✅ two_consecutive_batches_backward")


if __name__ == "__main__":
    test_eta_decay_receive_gradients()
    test_fast_layer_plast_params_stay_gradless()
    test_consolidation_importance_stays_out_of_graph()
    test_inference_path_builds_no_graph()
    test_two_consecutive_batches_backward()
    print("\n🎉 Tüm türevlenebilir plastisite testleri geçti!")
