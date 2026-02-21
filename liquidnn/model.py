"""
MiniLiquidGPT — Sıvı Nöral Ağ Dil Modeli

4 katmanlı hibrit yapı:
  Katman 0-1: ode_steps=1 → hızlı algı (Euler, plastisite OFF)
  Katman 2-3: ode_steps=3 → derin düşünce (RK2, Hebb ON)

Ağırlık paylaşımı: embed ↔ lm_head
Pozisyon kodlaması: Sinüzoidal (sabit)
"""

import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ode_cell import LiquidODECell


class MiniLiquidGPT(nn.Module):
    """
    Sıvı Nöral Ağ Dil Modeli.

    Args:
        vocab_size: Kelime hazinesi boyutu (tiktoken gpt2 = 50257)
        embed_dim:  Gömme boyutu
        num_fast:   Hızlı katman sayısı (steps=1, plast OFF)
        num_deep:   Derin katman sayısı (steps=3, plast ON)
        fast_steps: Hızlı katman ODE adım sayısı
        deep_steps: Derin katman ODE adım sayısı
        dropout:    Dropout oranı
        max_seq:    Maksimum sekans uzunluğu (pozisyon kodlaması)
    """

    def __init__(self, vocab_size: int = 50257, embed_dim: int = 256,
                 num_fast: int = 2, num_deep: int = 2,
                 fast_steps: int = 1, deep_steps: int = 3,
                 dropout: float = 0.1, max_seq: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_fast + num_deep

        # Embedding + pozisyon
        self.embed = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.embed_drop = nn.Dropout(dropout)

        pe = self._make_sinusoidal_pe(max_seq, embed_dim)
        self.register_buffer('pos_enc', pe)

        # Katmanlar
        self.cells = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.drops = nn.ModuleList()

        for _ in range(num_fast):
            self.cells.append(LiquidODECell(embed_dim, embed_dim,
                                            ode_steps=fast_steps,
                                            use_plasticity=False))
            self.norms.append(nn.LayerNorm(embed_dim))
            self.drops.append(nn.Dropout(dropout))

        for _ in range(num_deep):
            self.cells.append(LiquidODECell(embed_dim, embed_dim,
                                            ode_steps=deep_steps,
                                            use_plasticity=True))
            self.norms.append(nn.LayerNorm(embed_dim))
            self.drops.append(nn.Dropout(dropout))

        self.out_norm = nn.LayerNorm(embed_dim)
        # lm_head: weight tying ile embed.weight kullanılır

        self._hiddens: Optional[List[torch.Tensor]] = None

    @staticmethod
    def _make_sinusoidal_pe(max_len: int, dim: int) -> torch.Tensor:
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float)
                        * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)  # [1, max_len, D]

    def init_hidden(self, batch_size: int, device: torch.device):
        """Gizli durumları sıfırla."""
        self._hiddens = [
            torch.zeros(batch_size, self.embed_dim, device=device)
            for _ in range(self.num_layers)
        ]

    def forward_token(self, token_id: torch.Tensor, pos: int,
                      enable_plasticity: bool = True) -> torch.Tensor:
        """
        Tek token işle.

        Args:
            token_id: [B] token ID'leri
            pos: Sekans pozisyonu
            enable_plasticity: Hebb güncellemesi aktif mi

        Returns:
            [B, V] logits
        """
        if token_id.dim() > 1:
            token_id = token_id.squeeze(-1)

        B = token_id.size(0)
        if self._hiddens is None:
            self.init_hidden(B, token_id.device)

        x = self.embed(token_id)
        if pos < self.pos_enc.size(1):
            x = x + self.pos_enc[0, pos]

        for i, (cell, norm, drop) in enumerate(
                zip(self.cells, self.norms, self.drops)):
            h_new = cell(x, self._hiddens[i], enable_plasticity)
            h_new = norm(h_new + x)  # Residual + LayerNorm
            h_new = drop(h_new)
            self._hiddens[i] = h_new
            x = h_new

        x = self.out_norm(x)
        logits = F.linear(x, self.embed.weight)  # Weight tying
        return logits

    def forward(self, input_ids: torch.Tensor,
                enable_plasticity: bool = True,
                chunk_size: int = 16) -> torch.Tensor:
        """
        Sekans işle (eğitim modu).

        Args:
            input_ids: [B, T] token ID'leri
            enable_plasticity: Hebb aktif mi
            chunk_size: Truncated BPTT parça boyutu

        Returns:
            [B, T, V] logits
        """
        B, T = input_ids.shape
        self.init_hidden(B, input_ids.device)

        all_logits = []
        for t in range(T):
            # Truncated BPTT
            if t > 0 and t % chunk_size == 0:
                self._hiddens = [h.detach() for h in self._hiddens]
                for cell in self.cells:
                    cell.detach_hebb()

            logits = self.forward_token(input_ids[:, t], t, enable_plasticity)
            all_logits.append(logits)

        return torch.stack(all_logits, dim=1)

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new: int = 80,
                 temperature: float = 0.8, top_k: int = 40,
                 enable_plasticity: bool = True) -> torch.Tensor:
        """
        Metin üret.

        Args:
            prompt_ids: [T] veya [1, T] token ID'leri
            max_new: Üretilecek token sayısı
            temperature: Örnekleme sıcaklığı
            top_k: Top-k filtreleme
            enable_plasticity: Hebb aktif mi

        Returns:
            [1, T+max_new] tüm token ID'leri
        """
        self.eval()
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)

        B = prompt_ids.size(0)
        self.init_hidden(B, prompt_ids.device)
        self.reset_hebb()

        ids = prompt_ids.clone()

        # Prompt'u işle
        for t in range(ids.size(1)):
            logits = self.forward_token(ids[:, t], t, enable_plasticity)

        # Üret
        for step in range(max_new):
            pos = ids.size(1)
            scaled = logits / temperature

            if top_k > 0:
                v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                scaled[scaled < v[:, [-1]]] = -float('inf')

            probs = F.softmax(scaled, dim=-1)
            next_id = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_id], dim=1)
            logits = self.forward_token(next_id.squeeze(-1), pos,
                                        enable_plasticity)

        return ids

    def reset_hebb(self):
        """Tüm plastik izleri sıfırla."""
        for cell in self.cells:
            cell.reset_hebb()

    def hebb_stats(self) -> dict:
        """Tüm Hebb normlarını döndür."""
        stats = {}
        for i, cell in enumerate(self.cells):
            info = cell.hebb_info
            stats[f'L{i}_ih'] = info['ih']
            stats[f'L{i}_hh'] = info['hh']
        return stats

    def count_params(self) -> dict:
        """Parametre sayımı."""
        total = sum(p.numel() for p in self.parameters())
        embed = self.embed.weight.numel()
        cells = sum(sum(p.numel() for p in c.parameters())
                    for c in self.cells)
        return {'total': total, 'embed': embed, 'cells': cells,
                'other': total - embed - cells}
