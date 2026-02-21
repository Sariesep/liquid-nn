"""
Veri yükleme — Wikitext-2 veya Shakespeare
"""

import os
import torch


def load_data(tokenizer, max_tokens=200_000, seq_len=128, device='cpu'):
    """
    Eğitim/doğrulama verisi hazırla.

    Önce Wikitext-2 dener, başarısız olursa Shakespeare'e düşer.

    Returns:
        train_x, train_y, val_x, val_y — her biri [N, seq_len] tensor
    """
    print("\n📚 Veri yükleniyor...")
    text = _try_wikitext() or _try_shakespeare()

    if text is None:
        raise RuntimeError("Veri yüklenemedi!")

    # Tokenize
    all_ids = tokenizer.encode(text)
    if len(all_ids) > max_tokens:
        all_ids = all_ids[:max_tokens]
    print(f"   {len(all_ids):,} token")

    # Train/val split (%90/%10)
    split = int(0.9 * len(all_ids))
    train_ids = all_ids[:split]
    val_ids = all_ids[split:]

    train_x, train_y = _make_sequences(train_ids, seq_len, device)
    val_x, val_y = _make_sequences(val_ids, seq_len, device)
    print(f"   Eğitim: {train_x.size(0)} sekans × {seq_len} token")
    print(f"   Doğrulama: {val_x.size(0)} sekans")
    return train_x, train_y, val_x, val_y


def _try_wikitext() -> str:
    """Wikitext-2 yüklemeyi dene."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n".join([t for t in ds["text"] if len(t.strip()) > 50])
        print(f"   Wikitext-2: {len(text):,} karakter")
        return text
    except Exception as e:
        print(f"   Wikitext yüklenemedi: {e}")
        return None


def _try_shakespeare() -> str:
    """Shakespeare yüklemeyi dene."""
    import urllib.request
    url = ('https://raw.githubusercontent.com/karpathy/'
           'char-rnn/master/data/tinyshakespeare/input.txt')
    path = '/tmp/shakespeare.txt'
    try:
        if not os.path.exists(path):
            print("   Shakespeare indiriliyor...")
            urllib.request.urlretrieve(url, path)
        with open(path) as f:
            text = f.read()
        print(f"   Shakespeare: {len(text):,} karakter")
        return text
    except Exception as e:
        print(f"   Shakespeare yüklenemedi: {e}")
        return None


def _make_sequences(ids: list, seq_len: int, device) -> tuple:
    """Token listesini [N, seq_len] input/target çiftlerine çevir."""
    t = torch.tensor(ids, dtype=torch.long, device=device)
    n = len(ids) // (seq_len + 1)
    t = t[:n * (seq_len + 1)].view(n, seq_len + 1)
    return t[:, :-1], t[:, 1:]
