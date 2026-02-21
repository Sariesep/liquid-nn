"""
Tokenizer — tiktoken GPT-2 BPE sarmalayıcı
"""

import os
import torch


def get_tokenizer():
    """tiktoken GPT-2 tokenizer yükle."""
    try:
        import tiktoken
    except ImportError:
        os.system("pip install tiktoken -q")
        import tiktoken
    return tiktoken.get_encoding("gpt2")


class TokenizerWrapper:
    """tiktoken'ı basit bir arayüze sarar."""

    def __init__(self, enc=None):
        self.enc = enc or get_tokenizer()
        self.vocab_size = self.enc.n_vocab  # 50257

    def encode(self, text: str) -> list:
        return self.enc.encode(text, allowed_special={'<|endoftext|>'})

    def decode(self, ids) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.enc.decode(ids)

    def encode_tensor(self, text: str, device='cpu') -> torch.Tensor:
        """Direkt tensor döndür."""
        return torch.tensor(self.encode(text), dtype=torch.long, device=device)
