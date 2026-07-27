"""
Delta kuralı testleri (v0.5) — DeltaNet / Kimi Delta Attention ailesi.

Test edilen asıl iddia: delta kuralı hata düzeltmelidir. Aynı anahtara
yeni bir değer yazınca eski çağrışımı SİLER; Hebbian kural ise ikisini
üst üste biriktirip karıştırır. Bu fark plastik hafızanın kapasitesini
belirler ve ilk ablasyonun null çıkmasının hipotezlerinden biridir.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from liquidnn import MiniLiquidGPT, PlasticSynapse


def _write(syn, k, v, n=1):
    """Bir çağrışımı n kez yaz."""
    for _ in range(n):
        syn.update_hebb(k, v)


def _recall(syn, k):
    """İzden (W hariç, saf plastik bileşen) geri çağır."""
    return F.linear(F.normalize(k, dim=-1), syn.Hebb)


def test_delta_overwrites_instead_of_accumulating():
    """Aynı anahtara iki farklı değer: delta son değeri hatırlamalı."""
    torch.manual_seed(0)
    delta = PlasticSynapse(16, 16, update_rule='delta')
    hebb = PlasticSynapse(16, 16, update_rule='hebb')

    k = torch.randn(1, 16)
    v1 = torch.randn(1, 16)
    v2 = torch.randn(1, 16)

    for syn in (delta, hebb):
        _write(syn, k, v1, n=20)
        _write(syn, k, v2, n=20)

    # Delta: son yazılan v2'ye v1'den daha yakın olmalı
    r = _recall(delta, k)
    d_to_v2 = F.cosine_similarity(r, v2).item()
    d_to_v1 = F.cosine_similarity(r, v1).item()
    assert d_to_v2 > d_to_v1, \
        f"delta eski değeri unutmadı: v2={d_to_v2:.3f} v1={d_to_v1:.3f}"
    print(f"✅ delta_overwrites (v2={d_to_v2:.3f} > v1={d_to_v1:.3f})")


def test_delta_converges_to_target():
    """Tek çağrışımı tekrar tekrar yazınca iz hedefe yakınsamalı."""
    torch.manual_seed(1)
    syn = PlasticSynapse(16, 16, update_rule='delta')
    k = torch.randn(1, 16)
    v = torch.randn(1, 16)

    _write(syn, k, v, n=1)
    err1 = (_recall(syn, k) - v).norm().item()
    _write(syn, k, v, n=30)
    err2 = (_recall(syn, k) - v).norm().item()

    assert err2 < err1, f"delta yakınsamadı: {err1:.4f} -> {err2:.4f}"
    print(f"✅ delta_converges (hata {err1:.4f} -> {err2:.4f})")


def test_delta_retains_both_when_keys_orthogonal():
    """Dik anahtarlar: delta her iki çağrışımı da tam tutmalı.

    Bu test aynı zamanda norm-sınırı regresyonunu yakalar: sabit skaler
    sınır uygulanırsa her kırpma matrisi küçültür, delta son anahtarı
    tam güce geri yazar ve A sistematik olarak silinir (A → ~0.29).
    """
    torch.manual_seed(2)
    syn = PlasticSynapse(32, 32, update_rule='delta')
    ka, kb = torch.randn(1, 32), torch.randn(1, 32)
    va, vb = torch.randn(1, 32), torch.randn(1, 32)
    _write(syn, ka, va, n=15)
    _write(syn, kb, vb, n=15)

    sa = F.cosine_similarity(_recall(syn, ka), va).item()
    sb = F.cosine_similarity(_recall(syn, kb), vb).item()
    assert sa > 0.9, f"dik anahtarda A unutuldu: {sa:.3f} (norm sınırı mı bağlıyor?)"
    assert sb > 0.9, f"B hatırlanmıyor: {sb:.3f}"
    print(f"✅ delta_retains_both_orthogonal (A={sa:.3f}, B={sb:.3f})")


def test_delta_recalls_recent_association_better_than_hebb():
    """Korelasyonlu anahtarlar: delta EN SON çağrışımı tam verir.

    Kuralların farklı hafıza politikası: delta üzerine yazar (son
    çağrışım kesin), Hebbian harmanlar (ikisi de yaklaşık). Delta'nın
    ayırt edici özelliği budur — "daha az karışma" değil, son yazılanın
    kesinliği.
    """
    torch.manual_seed(2)
    recent = {}
    for rule in ('delta', 'hebb'):
        syn = PlasticSynapse(32, 32, update_rule=rule)
        ka = torch.randn(1, 32)
        kb = 0.7 * ka + 0.3 * torch.randn(1, 32)  # korelasyonlu
        va, vb = torch.randn(1, 32), torch.randn(1, 32)
        _write(syn, ka, va, n=15)
        _write(syn, kb, vb, n=15)
        recent[rule] = F.cosine_similarity(_recall(syn, kb), vb).item()

    assert recent['delta'] > recent['hebb'], \
        f"delta son çağrışımda daha kesin olmalı: {recent}"
    print(f"✅ delta_recalls_recent_better "
          f"(delta={recent['delta']:.3f} > hebb={recent['hebb']:.3f})")


def test_delta_norm_stays_bounded():
    """Delta kendi kendini sınırlar: uzun koşuda norm patlamamalı."""
    torch.manual_seed(6)
    syn = PlasticSynapse(32, 32, update_rule='delta')
    for _ in range(300):
        syn.update_hebb(torch.randn(4, 32), torch.randn(4, 32))
    assert syn.hebb_norm < 50, f"delta normu patladı: {syn.hebb_norm:.2f}"
    assert syn.hebb_norm > 0.5, f"delta izi söndü: {syn.hebb_norm:.2f}"
    print(f"✅ delta_norm_bounded (‖H‖={syn.hebb_norm:.2f})")


def test_channel_gate_shape_and_grad():
    """Kanal başına kapı [in_dim] olmalı ve gradyan almalı.

    Not: ilk güncellemede iz sıfır olduğu için decay'in etkisi de
    sıfırdır (matematiksel olarak doğru) — gradyan ancak iz doluyken
    oluşur, o yüzden iki güncelleme gerekir.
    """
    torch.manual_seed(3)
    syn = PlasticSynapse(12, 8, update_rule='delta', channel_gate=True)
    assert syn.logit_decay.shape == (12,)

    pre = torch.randn(2, 12)
    syn.update_hebb(pre, syn(pre))          # izi doldur
    syn.update_hebb(pre, syn(pre))          # decay artık dolu ize etki ediyor
    syn.Hebb.sum().backward()
    assert syn.logit_decay.grad is not None
    assert syn.logit_decay.grad.abs().sum() > 0
    print("✅ channel_gate_shape_and_grad")


def test_delta_params_receive_gradients_in_model():
    """Model içinde delta kuralı: beta/decay/alpha gradyan almalı."""
    torch.manual_seed(4)
    model = MiniLiquidGPT(vocab_size=120, embed_dim=32,
                          num_fast=1, num_deep=1,
                          plast_rule='delta', plast_channel_gate=True)
    model.train()
    x = torch.randint(0, 120, (2, 16))
    model.reset_hebb()
    logits = model(x, enable_plasticity=True, chunk_size=8)
    F.cross_entropy(logits.reshape(-1, 120), x.reshape(-1)).backward()

    syn = model.cells[1].syn_ih
    for pname in ('logit_beta', 'logit_decay', 'alpha'):
        g = getattr(syn, pname).grad
        assert g is not None, f"{pname} gradyan almadı"
        assert g.abs().sum() > 0, f"{pname} gradyanı sıfır"
    print("✅ delta_params_receive_gradients_in_model")


def test_invalid_rule_rejected():
    """Bilinmeyen kural adı hata vermeli (sessizce hebb'e düşmemeli)."""
    try:
        PlasticSynapse(8, 8, update_rule='ttt')
        assert False, "geçersiz kural kabul edildi"
    except ValueError:
        pass
    print("✅ invalid_rule_rejected")


def test_delta_model_inference_stable():
    """Delta kuralıyla çıkarım: NaN/patlama olmamalı."""
    torch.manual_seed(5)
    model = MiniLiquidGPT(vocab_size=120, embed_dim=32,
                          num_fast=1, num_deep=1, plast_rule='delta')
    model.eval()
    x = torch.randint(0, 120, (1, 40))
    model.reset_hebb()
    with torch.no_grad():
        out = model(x, enable_plasticity=True, chunk_size=16)
    assert not torch.isnan(out).any() and not torch.isinf(out).any()
    assert model.cells[1].syn_ih.hebb_norm > 0
    print("✅ delta_model_inference_stable")


if __name__ == "__main__":
    test_delta_overwrites_instead_of_accumulating()
    test_delta_converges_to_target()
    test_delta_retains_both_when_keys_orthogonal()
    test_delta_recalls_recent_association_better_than_hebb()
    test_delta_norm_stays_bounded()
    test_channel_gate_shape_and_grad()
    test_delta_params_receive_gradients_in_model()
    test_invalid_rule_rejected()
    test_delta_model_inference_stable()
    print("\n🎉 Tüm delta kuralı testleri geçti!")
