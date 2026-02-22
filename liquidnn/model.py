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


def _greedy_pick(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Logits'ten greedy (argmax) token seçimi.

    Args:
        logits: [B, V] logit tensörü
        temperature: Sıcaklık (burada greedy olduğu için sadece scaling)

    Returns:
        [B, 1] seçilen token ID'leri
    """
    if temperature != 1.0 and temperature > 0:
        logits = logits / temperature
    return logits.argmax(dim=-1, keepdim=True)


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

    # ── State Management (Speculative Decoding için) ──────────────

    def save_state(self) -> dict:
        """
        Modelin tüm mutable durumunu snapshot olarak döndür.

        Kopyalanan durumlar:
          - self._hiddens  (katman başına gizli durum tensörleri)
          - Her LiquidODECell'in syn_ih.Hebb ve syn_hh.Hebb matrisleri

        Tüm tensörler .detach().clone() ile hesaplama grafiğinden koparılır;
        böylece geri yükleme sırasında VRAM sızıntısı önlenir.
        """
        saved_hiddens = None
        if self._hiddens is not None:
            saved_hiddens = [h.detach().clone() for h in self._hiddens]

        saved_hebbs = []
        for cell in self.cells:
            pair = {}
            pair['ih'] = (cell.syn_ih.Hebb.detach().clone()
                          if cell.syn_ih.Hebb is not None else None)
            pair['hh'] = (cell.syn_hh.Hebb.detach().clone()
                          if cell.syn_hh.Hebb is not None else None)
            saved_hebbs.append(pair)

        return {'hiddens': saved_hiddens, 'hebbs': saved_hebbs}

    def restore_state(self, state_dict: dict):
        """
        save_state() ile kaydedilmiş durumu geri yükle.

        Reject durumunda ana modeli kabul edilen son tokenin
        state'ine geri döndürmek için kullanılır.
        """
        if state_dict['hiddens'] is not None:
            self._hiddens = [h.detach().clone()
                             for h in state_dict['hiddens']]
        else:
            self._hiddens = None

        for cell, pair in zip(self.cells, state_dict['hebbs']):
            cell.syn_ih.Hebb = (pair['ih'].detach().clone()
                                if pair['ih'] is not None else None)
            cell.syn_hh.Hebb = (pair['hh'].detach().clone()
                                if pair['hh'] is not None else None)

    # ── Draft Model Factory ───────────────────────────────────────

    @classmethod
    def create_draft_model(cls, target_model: 'MiniLiquidGPT',
                           device: torch.device = None) -> 'MiniLiquidGPT':
        """
        Ana modelden hafif bir draft model oluştur.

        - 1 hızlı katman (Euler, PlasticSynapse OFF)
        - Embedding ağırlıkları ana modelden paylaşılır (VRAM tasarrufu)

        Args:
            target_model: Kaynak ana model
            device: Hedef cihaz (None → target_model.embed.weight.device)

        Returns:
            Hafif MiniLiquidGPT draft modeli
        """
        if device is None:
            device = target_model.embed.weight.device

        draft = cls(
            vocab_size=target_model.vocab_size,
            embed_dim=target_model.embed_dim,
            num_fast=1,
            num_deep=0,            # plastisite tamamen kapalı
            fast_steps=1,
            deep_steps=1,
            dropout=0.0,           # draft model'de dropout gereksiz
        ).to(device)

        # Embedding ağırlıklarını paylaş (aynı tensör, kopya değil)
        draft.embed.weight = target_model.embed.weight
        return draft

    # ── Speculative Decoding ──────────────────────────────────────

    @torch.no_grad()
    def generate_speculative(self, draft_model: 'MiniLiquidGPT',
                             prompt_ids: torch.Tensor,
                             max_new: int = 80, gamma: int = 4,
                             temperature: float = 0.8) -> torch.Tensor:
        """
        Speculative Decoding ile metin üret.

        Args:
            draft_model:  Hızlı taslak model (plastisitesiz, 1 katman)
            prompt_ids:   [T] veya [1, T] token ID'leri
            max_new:      Üretilecek maksimum yeni token sayısı
            gamma:        Draft modelin her turda üreteceği token sayısı
            temperature:  Örnekleme sıcaklığı (verification greedy'dir)

        Returns:
            [1, T + generated] tüm token ID'leri
        """
        self.eval()
        draft_model.eval()

        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)

        B = prompt_ids.size(0)
        device = prompt_ids.device

        # ── Adım A: Prompt Sync ──────────────────────────────────
        self.init_hidden(B, device)
        self.reset_hebb()
        draft_model.init_hidden(B, device)
        draft_model.reset_hebb()

        ids = prompt_ids.clone()

        # Prompt'u her iki modele de besle
        for t in range(ids.size(1)):
            main_logits = self.forward_token(ids[:, t], t,
                                             enable_plasticity=True)
            draft_model.forward_token(ids[:, t], t,
                                      enable_plasticity=False)

        generated_count = 0

        # ── Ana Döngü ─────────────────────────────────────────────
        while generated_count < max_new:
            cur_pos = ids.size(1)

            # Kalan token sayısını kontrol et
            remaining = max_new - generated_count
            cur_gamma = min(gamma, remaining)

            # ── Adım B: Drafting ──────────────────────────────────
            draft_tokens = []
            draft_logits = main_logits  # son logits'ten başla
            # Draft modeli de son logits'ten başlatmak yerine
            # draft'ın kendi son logits'ini kullan
            # İlk draft token: ana modelin son logits'inden greedy al
            # (draft model de prompt sonundaki state'e sahip)

            # Draft model kendi state'iyle gamma token üretir
            # İlk draft girdisi: ana modelin son greedy tokeni
            d_logits = draft_model.forward_token(
                _greedy_pick(main_logits, temperature).squeeze(-1),
                cur_pos - 1, enable_plasticity=False
            )
            # Aslında draft modelin ilk inputu, ana modelden ilk aday
            # Ama daha doğru yaklaşım: draft kendi içinden üretsin
            # Önce draft'ün son prompttaki logits'i kullanılır
            # Draft zaten prompt sonundaki state'e sahip,
            # main_logits'ten greedy token alıp onu besliyoruz

            first_draft_tok = _greedy_pick(main_logits, temperature)
            draft_tokens.append(first_draft_tok)

            # Geriye kalan gamma-1 token draft kendi logits'inden
            for g in range(1, cur_gamma):
                d_logits = draft_model.forward_token(
                    draft_tokens[-1].squeeze(-1),
                    cur_pos + g - 1,
                    enable_plasticity=False
                )
                draft_tokens.append(_greedy_pick(d_logits, temperature))

            # draft_tokens: [tok0, tok1, ..., tok_{gamma-1}]  her biri [B,1]

            # ── Adım C: Verification ─────────────────────────────
            saved_state = self.save_state()

            verified_logits_list = []
            for g in range(cur_gamma):
                v_logits = self.forward_token(
                    draft_tokens[g].squeeze(-1),
                    cur_pos + g,
                    enable_plasticity=True
                )
                verified_logits_list.append(v_logits)

            # ── Adım D: Accept / Reject ──────────────────────────
            n_accepted = 0
            for g in range(cur_gamma):
                # Ana modelin draft token'ın GİRİLMESİNDEN ÖNCEKİ
                # logits'inden greedy seçim
                if g == 0:
                    target_logits = main_logits          # prompt-sonrası logits
                else:
                    target_logits = verified_logits_list[g - 1]

                main_choice = _greedy_pick(target_logits, temperature)

                if main_choice.item() == draft_tokens[g].item():
                    n_accepted += 1
                else:
                    break

            # ── Adım E: Rollback veya Commit ─────────────────────
            # Restore: checkpoint'a geri dön
            self.restore_state(saved_state)

            # Kabul edilen tokenler için state'i yeniden ilerlet
            accepted_ids = []
            for g in range(n_accepted):
                tok = draft_tokens[g]
                accepted_ids.append(tok)
                replay_logits = self.forward_token(
                    tok.squeeze(-1),
                    cur_pos + g,
                    enable_plasticity=True
                )

            # Ana modelin kendi seçtiği token (reddedilen ilk konum
            # veya tüm draft kabul edildiyse bonus token)
            if n_accepted < cur_gamma:
                # İlk reddedilen konumdaki ana model greedy tokeni
                if n_accepted == 0:
                    correction_logits = main_logits
                else:
                    correction_logits = replay_logits
                bonus = _greedy_pick(correction_logits, temperature)
                accepted_ids.append(bonus)
                # Bu bonus tokeni de state'e commit et
                self.forward_token(
                    bonus.squeeze(-1),
                    cur_pos + n_accepted,
                    enable_plasticity=True
                )
            else:
                # Tümü kabul → son verified logits'ten bonus token
                bonus = _greedy_pick(verified_logits_list[-1], temperature)
                accepted_ids.append(bonus)
                self.forward_token(
                    bonus.squeeze(-1),
                    cur_pos + n_accepted,
                    enable_plasticity=True
                )

            # Kabul edilen tokenları sekansa ekle
            new_tokens = torch.cat(accepted_ids, dim=-1)  # [B, n+1]
            if new_tokens.dim() == 1:
                new_tokens = new_tokens.unsqueeze(0)
            ids = torch.cat([ids, new_tokens], dim=1)
            generated_count += new_tokens.size(1)

            # main_logits güncelle: en son forward_token çıktısı
            main_logits = self.forward_token(
                ids[:, -1], ids.size(1) - 1,
                enable_plasticity=True
            )

            # Draft modeli de senkronize et:
            # yeni kabul edilen tokenları draft'a besle
            for tok_idx in range(new_tokens.size(1)):
                draft_model.forward_token(
                    new_tokens[:, tok_idx],
                    cur_pos + tok_idx,
                    enable_plasticity=False
                )

        # max_new aşımını kes
        total_len = prompt_ids.size(1) + max_new
        ids = ids[:, :total_len]
        return ids

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
