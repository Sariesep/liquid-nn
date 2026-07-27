"""
Diferansiyel Plastik Sinapslar — Öğrenilebilir Yazma Kuralları

W_eff = W_base + α ⊙ Hebb

W_base: Eğitimle öğrenilen sabit ağırlıklar
Hebb:   Her forward pass'te güncellenen plastik iz (hızlı ağırlık)
α:      Hangi sinapsların ne kadar plastik olduğunu belirler

v0.3.4 eklentileri:
  - Çift Hızlı Hebb (fast + slow timescale)
  - Sinaptik Konsolidasyon (önemli izleri koruma)
  - Nöromodülasyon desteği (mod_signal ile eta ölçekleme)

v0.5 — İki yazma kuralı (update_rule):

  'hebb'  : H ← decay·H + η·(post ⊗ pre)
            Saf toplamsal Hebbian birikim. Aynı anahtara ikinci kez
            yazınca eskisi silinmez; izler birbirine karışır ve tek
            çare tüm izi eşit oranda soldurmaktır.
            "Birlikte ateşleyen nöronlar birbirine bağlanır" — Hebb, 1949

  'delta' : H ← g⊙H + β·((post − (g⊙H)·k̂) ⊗ k̂),  k̂ = pre/‖pre‖
            Hata düzeltmeli delta kuralı (Widrow-Hoff 1960). Yazmadan
            önce o anahtarın MEVCUT karşılığını okur ve yalnızca farkı
            (hatayı) yazar → eski çağrışım hedefe doğru güncellenir,
            karışma yerine üzerine yazma olur. DeltaNet / Gated DeltaNet
            ve Kimi Delta Attention (KDA) bu aileden gelir; K3'ün
            hafızası da bu kuralla yazılıyor.

  channel_gate=True (KDA'nın katkısı): tek skaler decay yerine girdi
  kanalı başına öğrenilmiş unutma kapısı [in_dim] — her çağrışım kendi
  hızında solar, "hepsini birden unut" kısıtı kalkar.

Bu ikisi eş koşullarda karşılaştırılabilsin diye aynı sınıfta duruyor:
projenin açık araştırma sorusu "hangi yazma kuralı ne zaman kazanıyor".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlasticSynapse(nn.Module):
    """
    Diferansiyel plastik lineer katman.

    Eğitim sırasında: α gradient alır, hangi sinapsların plastik olacağını öğrenir
    İnference sırasında: Hebb matrisi her token'da güncellenir → gerçek zamanlı öğrenme

    Args:
        in_dim:                Girdi boyutu
        out_dim:               Çıktı boyutu
        sparse_k:              Top-k sparse Hebb (0 = tam yoğun)
        use_dual_hebb:         Çift hızlı Hebb (fast + slow) aktif mi
        use_consolidation:     Sinaptik konsolidasyon aktif mi
        consolidation_strength: Konsolidasyon gücü (yüksek → daha dirençli)
        update_rule:           'hebb' (toplamsal) | 'delta' (hata düzeltmeli)
        channel_gate:          Kanal başına unutma kapısı (KDA tarzı ince
                               taneli gating); False → tek skaler decay
    """

    VALID_RULES = ('hebb', 'delta')

    def __init__(self, in_dim: int, out_dim: int, sparse_k: int = 0,
                 use_dual_hebb: bool = False,
                 use_consolidation: bool = False,
                 consolidation_strength: float = 1.0,
                 update_rule: str = 'hebb',
                 channel_gate: bool = False):
        super().__init__()
        if update_rule not in self.VALID_RULES:
            raise ValueError(
                f"update_rule '{update_rule}' geçersiz; "
                f"geçerli değerler: {self.VALID_RULES}")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sparse_k = sparse_k  # 0 = tam yoğun, >0 = top-k sparse
        self.use_dual_hebb = use_dual_hebb
        self.use_consolidation = use_consolidation
        self.consolidation_strength = consolidation_strength
        self.update_rule = update_rule
        self.channel_gate = channel_gate

        # Sabit ağırlıklar (tüm modellerde var)
        self.W = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.kaiming_normal_(self.W, mode='fan_in', nonlinearity='linear')
        self.W.data *= 0.5
        self.b = nn.Parameter(torch.zeros(out_dim))

        # Plastisite kontrolleri — Fast Hebb (eğitimle öğrenilir)
        # v0.4 başlangıç değerleri: ilk ablasyonda eta≈0.0015 ve yarı ömür
        # ~5 token ile mekanizma fiilen ölü doğuyordu. Yeni init:
        # eta ≈ 0.0094, decay ≈ 0.989 (yarı ömür ~20 token).
        self.alpha = nn.Parameter(0.01 * torch.randn(out_dim, in_dim))
        self.log_eta = nn.Parameter(torch.tensor(-1.0))
        self.hebb_capacity = nn.Parameter(torch.tensor(2.0))
        self.register_buffer('_hebb_steps', torch.tensor(0))

        # Unutma kapısı: skaler (klasik) veya kanal başına (KDA tarzı).
        # İkisi de logit uzayında; sigmoid(4.5) ≈ 0.989 → yarı ömür ~20 token.
        if channel_gate:
            self.logit_decay = nn.Parameter(torch.full((in_dim,), 4.5))
        else:
            self.logit_decay = nn.Parameter(torch.tensor(4.5))

        # Delta kuralının yazma oranı β ∈ (0,1): 1 → çağrışımı tamamen
        # üzerine yaz, 0 → hiç yazma. sigmoid(0) = 0.5 ile başlar.
        if update_rule == 'delta':
            self.logit_beta = nn.Parameter(torch.tensor(0.0))

        # Norm sınırı ölçeği — delta'nın doğal çalışma normu √in_dim ile
        # büyür; Hebbian'da sınır regülatör olduğu için ölçek 1.0 kalır.
        self._norm_scale = in_dim ** 0.5 if update_rule == 'delta' else 1.0

        # Plastik iz (fast)
        self.register_buffer('Hebb', None)

        # ── Çift Hızlı Hebb (Slow timescale) ──────────────────────
        if use_dual_hebb:
            self.alpha_slow = nn.Parameter(
                0.005 * torch.randn(out_dim, in_dim))
            self.log_eta_slow = nn.Parameter(torch.tensor(-3.0))
            self.logit_decay_slow = nn.Parameter(torch.tensor(6.0))
            self.register_buffer('Hebb_slow', None)

        # ── Sinaptik Konsolidasyon ─────────────────────────────────
        if use_consolidation:
            self.register_buffer('_importance', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # İzler her zaman fp32 — AMP altında x fp16 gelse bile küçük
        # eta çarpanlarının hassasiyeti korunur (matmul'u autocast yönetir)
        if self.Hebb is None:
            self.Hebb = torch.zeros(self.out_dim, self.in_dim,
                                    device=x.device, dtype=torch.float32)

        W_eff = self.W + self.alpha * self.Hebb

        # Slow Hebb katkısı
        if self.use_dual_hebb:
            if self.Hebb_slow is None:
                self.Hebb_slow = torch.zeros(self.out_dim, self.in_dim,
                                             device=x.device,
                                             dtype=torch.float32)
            W_eff = W_eff + self.alpha_slow * self.Hebb_slow

        return F.linear(x, W_eff.to(x.dtype), self.b.to(x.dtype))

    def update_hebb(self, pre: torch.Tensor, post: torch.Tensor,
                    moe_weight: float = 1.0, mod_signal: float = 1.0):
        """
        Hebbian güncelleme (opsiyonel top-k sparsification ile) —
        v0.4'ten itibaren TÜREVLENEBİLİR (Miconi 2018 ile uyumlu).

        Eski @torch.no_grad() dekoratörü iz yazma yolunu grafikten
        koparıyordu: eta/decay/hebb_capacity hiç gradyan alamıyor,
        yalnızca alpha (okuma kazancı) öğreniliyordu — ilk ablasyonun
        "katkı yok" sonucunun kök nedeni. Artık iz zinciri chunk içinde
        grafiğe dahil; truncated BPTT sınırlarında detach_hebb() keser.
        Çıkarım zaten torch.no_grad() altında çağırdığı için ek maliyet
        yok.

        pre:  Presinaptik aktivasyon [B, in_dim]  (delta kuralında "anahtar")
        post: Postsinaptik aktivasyon [B, out_dim] (delta kuralında "değer")
        moe_weight: Bu expert'in seçilme ağırlığı (MoE router'dan gelir)
        mod_signal: Nöromodülasyon sinyali (meta-plasticity)
        """
        # decay: skaler veya [in_dim] — [out,in] ize son eksende yayılır
        decay = torch.sigmoid(self.logit_decay)

        # İz birikimi fp32'de yapılır (AMP altında pre/post fp16 gelebilir)
        pre = pre.float()
        post = post.float()
        if pre.dim() == 1:
            pre = pre.unsqueeze(0)
        if post.dim() == 1:
            post = post.unsqueeze(0)
        B = max(pre.size(0), 1)

        if self.Hebb is None:
            self.Hebb = torch.zeros(self.out_dim, self.in_dim,
                                    device=pre.device, dtype=torch.float32)

        if self.update_rule == 'delta':
            # ── Delta kuralı (DeltaNet / KDA ailesi) ───────────────
            # Anahtarı birim normla — delta kuralının kararlılığı
            # ‖k‖=1 varsayımına dayanır (aksi halde β etkin oranı
            # girdi büyüklüğüyle ölçeklenip patlar).
            k = F.normalize(pre, dim=-1, eps=1e-6)
            beta = torch.sigmoid(self.logit_beta) * mod_signal * moe_weight

            # Önce solma (gated delta rule), sonra hata düzeltmesi
            H_decayed = self.Hebb * decay
            # O anahtarın izde ŞU ANDA karşılığı ne? → hatayı ondan çıkar
            retrieved = F.linear(k, H_decayed)          # [B, out_dim]
            err = post - retrieved                      # yazılacak düzeltme
            outer = torch.einsum('bi,bj->ij', err, k) / B
            outer = self._consolidate(outer, pre.device)
            self.Hebb = H_decayed + beta * outer
        else:
            # ── Hebbian kural (toplamsal birikim) ─────────────────
            eta = F.softplus(self.log_eta) * 0.03 * mod_signal
            outer = torch.einsum('bi,bj->ij', post, pre) / B
            outer = outer * moe_weight
            outer = self._consolidate(outer, pre.device)
            self.Hebb = decay * self.Hebb + eta * outer

        # Adaptif norm sınırı: zaman içinde büyüyen kapasite
        # Branchless ölçekleme — .item()/bool karşılaştırması GPU'yu her
        # güncellemede senkronize ediyordu (token başına ~12 kez)
        #
        # Delta kuralında sınır REGÜLATÖR değil GÜVENLİK VALFİdir:
        # kural matematiksel olarak kendi kendini sınırlar (ölçüldü: norm
        # platoya oturuyor, ~√in_dim ile ölçekleniyor). Sabit skaler sınır
        # uygulanırsa her kırpma tüm matrisi küçültür, delta ise en son
        # anahtarı tam güce geri yazar → eski çağrışımlar sistematik
        # olarak silinir (dik anahtarlarda bile A-hatırlama 1.00→0.29).
        # Bu yüzden delta modunda sınır √in_dim ile ölçeklenir.
        self._hebb_steps += 1
        growth = 1.0 + 0.1 * torch.log1p(self._hebb_steps.float())
        h_norm = self.Hebb.norm()
        max_norm = F.softplus(self.hebb_capacity) * growth * self._norm_scale
        self.Hebb = self.Hebb * torch.clamp(max_norm / (h_norm + 1e-8),
                                            max=1.0)

        # Top-k sparsification (fast)
        if self.sparse_k > 0:
            flat = self.Hebb.abs().view(-1)
            total = flat.numel()
            k = min(self.sparse_k, total)
            if k < total:
                threshold = torch.topk(flat, k).values[-1]
                mask = self.Hebb.abs() >= threshold
                self.Hebb = self.Hebb * mask

        # ── Slow Hebb güncellemesi ─────────────────────────────────
        # Yavaş iz her zaman toplamsal Hebbian birikimdir: görevi uzun
        # vadeli istatistik tutmak, tekil çağrışımı düzeltmek değil.
        # update_rule yalnızca hızlı izi belirler — karşılaştırma o iz
        # üzerinde yapılıyor.
        if self.use_dual_hebb:
            decay_slow = torch.sigmoid(self.logit_decay_slow)
            eta_slow = F.softplus(self.log_eta_slow) * 0.01 * mod_signal

            if self.Hebb_slow is None:
                self.Hebb_slow = torch.zeros(self.out_dim, self.in_dim,
                                             device=pre.device,
                                             dtype=torch.float32)

            outer_slow = torch.einsum('bi,bj->ij', post, pre) / B
            outer_slow = self._consolidate(outer_slow, pre.device,
                                           update_ema=False)
            self.Hebb_slow = (decay_slow * self.Hebb_slow +
                              eta_slow * outer_slow * moe_weight)

            # Aynı kapasite sınırı (branchless)
            hs_norm = self.Hebb_slow.norm()
            self.Hebb_slow = self.Hebb_slow * torch.clamp(
                max_norm / (hs_norm + 1e-8), max=1.0)

    def _consolidate(self, outer: torch.Tensor, device,
                     update_ema: bool = True) -> torch.Tensor:
        """
        Sinaptik konsolidasyon maskesi: önemli izler değişime direnir.

        importance bir EMA istatistiği, öğrenilebilir yol değil —
        bilinçli olarak grafik DIŞINDA tutulur (aksi halde chunk boyunca
        hesap grafiği biriktirip bellek sızdırırdı).
        """
        if not self.use_consolidation:
            return outer

        with torch.no_grad():
            if self._importance is None:
                self._importance = torch.zeros(
                    self.out_dim, self.in_dim, device=device,
                    dtype=torch.float32)
            if update_ema:
                # Tutarlı büyük Hebb*alpha değerleri önemli sayılır
                self._importance = (
                    0.99 * self._importance +
                    0.01 * (self.Hebb.detach() * self.alpha.detach()).abs())

        # Önemli → düşük güncelleme (sabit katsayı, gradyan taşımaz)
        update_mask = 1.0 / (1.0 + self._importance *
                             self.consolidation_strength)
        return outer * update_mask

    def reset_hebb(self):
        """Plastik izleri sıfırla."""
        self.Hebb = None
        if self.use_dual_hebb:
            self.Hebb_slow = None

    def detach_hebb(self):
        """Hebb'i hesaplama grafiğinden ayır (truncated BPTT için)."""
        if self.Hebb is not None:
            self.Hebb = self.Hebb.detach().clone()
        if self.use_dual_hebb and hasattr(self, 'Hebb_slow'):
            if self.Hebb_slow is not None:
                self.Hebb_slow = self.Hebb_slow.detach().clone()

    @property
    def hebb_norm(self) -> float:
        """Hebb matrisinin normu (fast)."""
        return self.Hebb.norm().item() if self.Hebb is not None else 0.0

    @property
    def hebb_slow_norm(self) -> float:
        """Hebb_slow matrisinin normu."""
        if self.use_dual_hebb and hasattr(self, 'Hebb_slow'):
            return self.Hebb_slow.norm().item() if self.Hebb_slow is not None else 0.0
        return 0.0

    def extra_repr(self) -> str:
        parts = [f'in={self.in_dim}, out={self.out_dim}',
                 f'rule={self.update_rule}',
                 f'hebb_norm={self.hebb_norm:.4f}']
        if self.channel_gate:
            parts.append('channel_gate')
        if self.use_dual_hebb:
            parts.append(f'hebb_slow_norm={self.hebb_slow_norm:.4f}')
        if self.use_consolidation:
            parts.append(f'consolidation={self.consolidation_strength}')
        return ', '.join(parts)
