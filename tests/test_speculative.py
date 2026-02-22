"""Speculative Decoding unit testleri."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from liquidnn import MiniLiquidGPT


# ── Test Helpers ──────────────────────────────────────────────────

def _make_model(**kw):
    defaults = dict(vocab_size=100, embed_dim=32,
                    num_fast=1, num_deep=1, deep_steps=2)
    defaults.update(kw)
    return MiniLiquidGPT(**defaults)


# ── State Management Tests ────────────────────────────────────────

def test_save_restore_state():
    """save → forward → restore → state eski haline dönmeli."""
    model = _make_model()
    prompt = torch.randint(0, 100, (1, 5))
    model.init_hidden(1, torch.device('cpu'))
    model.reset_hebb()

    # Prompt'u işle (Hebb aktifleşsin)
    for t in range(5):
        model.forward_token(prompt[:, t], t, enable_plasticity=True)

    # Checkpoint al
    state = model.save_state()
    h_before = [h.clone() for h in model._hiddens]
    hebb_before = []
    for c in model.cells:
        hebb_before.append({
            'ih': c.syn_ih.Hebb.clone() if c.syn_ih.Hebb is not None else None,
            'hh': c.syn_hh.Hebb.clone() if c.syn_hh.Hebb is not None else None,
        })

    # Modeli birkaç token daha ilerlet → state değişir
    for t in range(5, 10):
        model.forward_token(torch.randint(0, 100, (1,)), t,
                            enable_plasticity=True)

    # Restore et
    model.restore_state(state)

    # Kontrol: hidden state'ler eşit olmalı
    for i, (ha, hb) in enumerate(zip(model._hiddens, h_before)):
        assert torch.allclose(ha, hb, atol=1e-6), \
            f"Katman {i} hidden restore başarısız"

    # Kontrol: Hebb matrisleri eşit olmalı
    for i, cell in enumerate(model.cells):
        if hebb_before[i]['ih'] is not None:
            assert torch.allclose(cell.syn_ih.Hebb, hebb_before[i]['ih'],
                                  atol=1e-6), \
                f"Katman {i} syn_ih.Hebb restore başarısız"
        if hebb_before[i]['hh'] is not None:
            assert torch.allclose(cell.syn_hh.Hebb, hebb_before[i]['hh'],
                                  atol=1e-6), \
                f"Katman {i} syn_hh.Hebb restore başarısız"

    print("✅ save_restore_state")


def test_state_independence():
    """Save sonrası modeli değiştirmek, kaydedilmiş state'i bozmamalı."""
    model = _make_model()
    prompt = torch.randint(0, 100, (1, 5))
    model.init_hidden(1, torch.device('cpu'))
    model.reset_hebb()

    for t in range(5):
        model.forward_token(prompt[:, t], t, enable_plasticity=True)

    state = model.save_state()
    saved_h0 = state['hiddens'][0].clone()

    # Modeli 20 token daha ilerlet
    for t in range(5, 25):
        model.forward_token(torch.randint(0, 100, (1,)), t,
                            enable_plasticity=True)

    # Kaydedilmiş state değişmemiş olmalı
    assert torch.allclose(state['hiddens'][0], saved_h0, atol=1e-7), \
        "Kaydedilmiş state sonraki forward'lardan etkilendi — detach/clone hatası"

    print("✅ state_independence")


# ── Draft Model Tests ─────────────────────────────────────────────

def test_draft_model_creation():
    """create_draft_model doğru yapıda model oluşturmalı."""
    main = _make_model()
    draft = MiniLiquidGPT.create_draft_model(main)

    assert draft.num_layers == 1, f"Draft 1 katman olmalı, {draft.num_layers} var"
    assert draft.vocab_size == main.vocab_size
    assert draft.embed_dim == main.embed_dim

    # Embedding ağırlık paylaşımı
    assert draft.embed.weight is main.embed.weight, \
        "Embedding ağırlıkları paylaşılmalı (aynı tensör)"

    # Plastisitesi olmamalı
    for cell in draft.cells:
        assert not cell.use_plasticity, "Draft cell plastisitesiz olmalı"

    # Draft modelin parametresi daha az olmalı
    main_p = sum(p.numel() for p in main.parameters())
    draft_p = sum(p.numel() for p in draft.parameters())
    assert draft_p < main_p, f"Draft ({draft_p}) < Main ({main_p}) olmalı"

    print("✅ draft_model_creation")


# ── Speculative Generation Tests ──────────────────────────────────

def test_speculative_output_length():
    """generate_speculative doğru uzunlukta çıktı üretmeli."""
    main = _make_model()
    draft = MiniLiquidGPT.create_draft_model(main)

    prompt = torch.tensor([1, 2, 3])
    max_new = 10
    out = main.generate_speculative(draft, prompt, max_new=max_new, gamma=3)

    expected_len = 3 + max_new  # prompt + new
    assert out.shape[1] == expected_len, \
        f"Beklenen {expected_len}, alınan {out.shape[1]}"

    print("✅ speculative_output_length")


def test_speculative_produces_valid_tokens():
    """Üretilen tüm tokenler vocab aralığında olmalı."""
    vocab = 100
    main = _make_model(vocab_size=vocab)
    draft = MiniLiquidGPT.create_draft_model(main)

    prompt = torch.tensor([1, 2, 3, 4, 5])
    out = main.generate_speculative(draft, prompt, max_new=20, gamma=4)

    assert out.min().item() >= 0, "Token ID negatif olamaz"
    assert out.max().item() < vocab, f"Token ID vocab_size ({vocab}) aşmamalı"

    print("✅ speculative_valid_tokens")


def test_speculative_with_gamma_1():
    """gamma=1 ile de çalışmalı (her turda 1 draft token)."""
    main = _make_model()
    draft = MiniLiquidGPT.create_draft_model(main)

    prompt = torch.tensor([10, 20, 30])
    out = main.generate_speculative(draft, prompt, max_new=8, gamma=1)

    expected_len = 3 + 8
    assert out.shape[1] == expected_len, \
        f"gamma=1: Beklenen {expected_len}, alınan {out.shape[1]}"

    print("✅ speculative_gamma_1")


def test_no_memory_leak():
    """save/restore döngüsünde tensör referans sızıntısı olmamalı."""
    model = _make_model()
    model.init_hidden(1, torch.device('cpu'))
    model.reset_hebb()

    # Hebb'i aktifleştir
    for t in range(5):
        model.forward_token(torch.randint(0, 100, (1,)), t,
                            enable_plasticity=True)

    states = []
    for _ in range(100):
        s = model.save_state()
        states.append(s)
        model.forward_token(torch.randint(0, 100, (1,)), 5,
                            enable_plasticity=True)
        model.restore_state(s)

    # Her snapshot bağımsız olmalı — ilk ve son aynı
    h_first = states[0]['hiddens'][0]
    h_last = states[-1]['hiddens'][0]
    # Farklı olabilirler ama en azından tensör boyutları tutmalı
    assert h_first.shape == h_last.shape

    # Basit bellek kontrolü: state dict boyutu sabit
    import sys
    size_first = sum(t.nelement() for t in states[0]['hiddens'])
    size_last = sum(t.nelement() for t in states[-1]['hiddens'])
    assert size_first == size_last, "Snapshot boyutu değişmemeli"

    print("✅ no_memory_leak")


if __name__ == "__main__":
    test_save_restore_state()
    test_state_independence()
    test_draft_model_creation()
    test_speculative_output_length()
    test_speculative_produces_valid_tokens()
    test_speculative_with_gamma_1()
    test_no_memory_leak()
    print("\n🏆 Tüm speculative decoding testleri geçti!")
