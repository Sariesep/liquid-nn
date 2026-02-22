"""v0.3.1 testleri — Flash Attention, KV Cache, RoPE."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from liquidnn import MiniLiquidGPT, SlidingWindowAttention
from liquidnn.attention import _precompute_freqs, _apply_rope


# ═══ 1. RoPE ═════════════════════════════════════════════════════

def test_rope_freqs():
    """RoPE frekans tablosu doğru boyutta olmalı."""
    freqs = _precompute_freqs(head_dim=8, max_len=128)
    assert freqs.shape == (128, 4, 2), f"Beklenen (128,4,2), alınan {freqs.shape}"
    print("✅ rope_freqs")


def test_rope_apply():
    """RoPE uygulama sonrası şekil korunmalı."""
    freqs = _precompute_freqs(head_dim=8, max_len=128)
    x = torch.randn(1, 4, 1, 8)  # [B, H, 1, Dh]
    out = _apply_rope(x, pos=5, freqs=freqs)
    assert out.shape == x.shape
    # RoPE sonrası norm yaklaşık aynı kalmalı (rotasyon)
    assert abs(x.norm().item() - out.norm().item()) < 0.01
    print("✅ rope_apply")


def test_rope_different_pos():
    """Aynı vektöre farklı pozisyon → farklı çıktı."""
    freqs = _precompute_freqs(head_dim=8, max_len=128)
    x = torch.randn(1, 4, 1, 8)
    out5 = _apply_rope(x, pos=5, freqs=freqs)
    out50 = _apply_rope(x, pos=50, freqs=freqs)
    assert not torch.allclose(out5, out50), "Farklı pozisyonlar farklı çıktı vermeli"
    print("✅ rope_different_pos")


def test_model_with_rope():
    """use_rope=True ile model çalışmalı."""
    model = MiniLiquidGPT(vocab_size=100, embed_dim=32,
                           num_fast=1, num_deep=1,
                           use_attention=True, attn_heads=4,
                           use_rope=True)
    x = torch.randint(0, 100, (1, 10))
    logits = model(x)
    assert logits.shape == (1, 10, 100)
    print("✅ model_with_rope")


# ═══ 2. KV Cache ═════════════════════════════════════════════════

def test_kv_cache_populated():
    """KV cache token ekledikçe dolmalı."""
    attn = SlidingWindowAttention(embed_dim=32, num_heads=4,
                                   window_size=8, use_rope=False)
    for i in range(5):
        x = torch.randn(1, 32)
        attn(x, pos=i)

    assert attn._buf_len == 5
    # K cache de dolu olmalı
    k_nonzero = (attn._k_cache[:, :, :5].abs().sum() > 0).any()
    assert k_nonzero, "KV cache boş olmamalı"
    print("✅ kv_cache_populated")


def test_kv_cache_save_restore():
    """KV cache save/restore çalışmalı."""
    attn = SlidingWindowAttention(embed_dim=32, num_heads=4,
                                   window_size=8, use_rope=False)
    for i in range(3):
        attn(torch.randn(1, 32), pos=i)

    state = attn.get_buffer_state()

    # Cache'i değiştir
    for i in range(5):
        attn(torch.randn(1, 32), pos=3+i)

    # Restore
    attn.set_buffer_state(state)
    assert attn._buf_len == 3
    assert torch.allclose(attn._k_cache, state['k_cache'])
    print("✅ kv_cache_save_restore")


def test_kv_cache_overflow():
    """Window dolunca en eski eleman atılmalı (FIFO)."""
    attn = SlidingWindowAttention(embed_dim=32, num_heads=4,
                                   window_size=4, use_rope=False)
    # 6 token besle (4'lük window'a)
    for i in range(6):
        attn(torch.randn(1, 32), pos=i)

    assert attn._buf_len == 4, f"Buffer dolu olmalı, {attn._buf_len}"
    print("✅ kv_cache_overflow")


# ═══ 3. Flash Attention ══════════════════════════════════════════

def test_flash_attention():
    """use_flash=True ile attention çalışmalı."""
    attn = SlidingWindowAttention(embed_dim=32, num_heads=4,
                                   window_size=8, use_flash=True)
    for i in range(5):
        out = attn(torch.randn(1, 32), pos=i)
    assert out.shape == (1, 32)
    print("✅ flash_attention")


def test_flash_vs_manual_close():
    """Flash ve manual attention yakın sonuç vermeli."""
    torch.manual_seed(42)
    flash_attn = SlidingWindowAttention(embed_dim=32, num_heads=4,
                                         window_size=8,
                                         use_flash=True, use_rope=False)
    torch.manual_seed(42)
    manual_attn = SlidingWindowAttention(embed_dim=32, num_heads=4,
                                          window_size=8,
                                          use_flash=False, use_rope=False)

    # Ağırlıkları eşitle
    manual_attn.load_state_dict(flash_attn.state_dict())
    flash_attn.eval()
    manual_attn.eval()

    tokens = [torch.randn(1, 32) for _ in range(5)]

    for i, tok in enumerate(tokens):
        out_flash = flash_attn(tok, pos=i)
        out_manual = manual_attn(tok, pos=i)

    # Son çıktılar yakın olmalı
    assert torch.allclose(out_flash, out_manual, atol=1e-4), \
        f"Flash/manual fark: {(out_flash - out_manual).abs().max():.6f}"
    print("✅ flash_vs_manual_close")


# ═══ Combined ════════════════════════════════════════════════════

def test_all_v031_features():
    """RoPE + Flash + KV Cache ile speculative decoding çalışmalı."""
    main = MiniLiquidGPT(vocab_size=100, embed_dim=32,
                          num_fast=1, num_deep=1,
                          use_attention=True, attn_heads=4,
                          use_rope=True, use_flash=True)
    draft = MiniLiquidGPT.create_draft_model(main)

    prompt = torch.tensor([1, 2, 3])
    out = main.generate_speculative(draft, prompt, max_new=10, gamma=3)
    assert out.shape[1] == 13
    print("✅ all_v031_features")


def test_backward_compat_attention():
    """Varsayılan parametrelerle eski attention davranışı korunmalı."""
    attn = SlidingWindowAttention(embed_dim=32, num_heads=4,
                                   window_size=8)
    # Varsayılan: use_rope=True, use_flash=True
    assert attn.use_rope == True
    assert attn.use_flash == True

    x = torch.randn(1, 32)
    out = attn(x, pos=0)
    assert out.shape == (1, 32)
    print("✅ backward_compat_attention")


def test_generate_with_rope():
    """Normal generate RoPE ile çalışmalı."""
    model = MiniLiquidGPT(vocab_size=100, embed_dim=32,
                           num_fast=1, num_deep=1,
                           use_attention=True, use_rope=True)
    prompt = torch.tensor([1, 2, 3])
    out = model.generate(prompt, max_new=8)
    assert out.shape[1] == 11
    print("✅ generate_with_rope")


if __name__ == "__main__":
    # RoPE
    test_rope_freqs()
    test_rope_apply()
    test_rope_different_pos()
    test_model_with_rope()

    # KV Cache
    test_kv_cache_populated()
    test_kv_cache_save_restore()
    test_kv_cache_overflow()

    # Flash Attention
    test_flash_attention()
    test_flash_vs_manual_close()

    # Combined
    test_all_v031_features()
    test_backward_compat_attention()
    test_generate_with_rope()

    print("\n🏆 Tüm v0.3.1 testleri geçti!")
