"""
Diferansiyel Plastik Sinapslar — Hebbian Öğrenme

W_eff = W_base + α ⊙ Hebb

W_base: Eğitimle öğrenilen sabit ağırlıklar
Hebb:   Her forward pass'te güncellenen plastik iz
α:      Hangi sinapsların ne kadar plastik olduğunu belirler

v0.3.4 eklentileri:
  - Çift Hızlı Hebb (fast + slow timescale)
  - Sinaptik Konsolidasyon (önemli izleri koruma)
  - Nöromodülasyon desteği (mod_signal ile eta ölçekleme)

"Birlikte ateşleyen nöronlar birbirine bağlanır" — Donald Hebb, 1949
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
    """

    def __init__(self, in_dim: int, out_dim: int, sparse_k: int = 0,
                 use_dual_hebb: bool = False,
                 use_consolidation: bool = False,
                 consolidation_strength: float = 1.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sparse_k = sparse_k  # 0 = tam yoğun, >0 = top-k sparse
        self.use_dual_hebb = use_dual_hebb
        self.use_consolidation = use_consolidation
        self.consolidation_strength = consolidation_strength

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
        self.logit_decay = nn.Parameter(torch.tensor(4.5))
        self.hebb_capacity = nn.Parameter(torch.tensor(2.0))
        self.register_buffer('_hebb_steps', torch.tensor(0))

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

        pre:  Presinaptik aktivasyon [B, in_dim]
        post: Postsinaptik aktivasyon [B, out_dim]
        moe_weight: Bu expert'in seçilme ağırlığı (MoE router'dan gelir)
        mod_signal: Nöromodülasyon sinyali (meta-plasticity)
        """
        decay = torch.sigmoid(self.logit_decay)
        eta = F.softplus(self.log_eta) * 0.03 * mod_signal

        # İz birikimi fp32'de yapılır (AMP altında pre/post fp16 gelebilir)
        pre = pre.float()
        post = post.float()

        if pre.dim() == 2:
            outer = torch.einsum('bi,bj->ij', post, pre) / max(pre.size(0), 1)
        else:
            outer = torch.outer(post.squeeze(), pre.squeeze())

        # MoE ağırlığı ile plastisiteyi ölçeklendir
        outer = outer * moe_weight

        if self.Hebb is None:
            self.Hebb = torch.zeros(self.out_dim, self.in_dim,
                                    device=pre.device, dtype=torch.float32)

        # ── Sinaptik Konsolidasyon: önemli izleri koru ─────────────
        # importance bir EMA istatistiği, öğrenilebilir yol değil —
        # bilinçli olarak grafik DIŞINDA tutulur (aksi halde epoch
        # boyunca hesap grafiği biriktirip bellek sızdırırdı)
        if self.use_consolidation:
            with torch.no_grad():
                if self._importance is None:
                    self._importance = torch.zeros(
                        self.out_dim, self.in_dim, device=pre.device,
                        dtype=torch.float32)
                # EMA importance: tutarlı büyük Hebb*alpha değerleri önemli
                self._importance = (
                    0.99 * self._importance +
                    0.01 * (self.Hebb.detach() * self.alpha.detach()).abs())
            # Update mask: önemli → düşük güncelleme (sabit katsayı)
            update_mask = 1.0 / (1.0 + self._importance *
                                 self.consolidation_strength)
            outer = outer * update_mask

        self.Hebb = decay * self.Hebb + eta * outer

        # Adaptif norm sınırı: zaman içinde büyüyen kapasite
        # Branchless ölçekleme — .item()/bool karşılaştırması GPU'yu her
        # güncellemede senkronize ediyordu (token başına ~12 kez)
        self._hebb_steps += 1
        growth = 1.0 + 0.1 * torch.log1p(self._hebb_steps.float())
        h_norm = self.Hebb.norm()
        max_norm = F.softplus(self.hebb_capacity) * growth
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
        if self.use_dual_hebb:
            decay_slow = torch.sigmoid(self.logit_decay_slow)
            eta_slow = F.softplus(self.log_eta_slow) * 0.01 * mod_signal

            if self.Hebb_slow is None:
                self.Hebb_slow = torch.zeros(self.out_dim, self.in_dim,
                                             device=pre.device,
                                             dtype=torch.float32)

            # Aynı outer product (konsolidasyon uygulanmış), farklı hız
            self.Hebb_slow = decay_slow * self.Hebb_slow + eta_slow * outer * moe_weight

            # Aynı kapasite sınırı (branchless)
            hs_norm = self.Hebb_slow.norm()
            self.Hebb_slow = self.Hebb_slow * torch.clamp(
                max_norm / (hs_norm + 1e-8), max=1.0)

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
                 f'hebb_norm={self.hebb_norm:.4f}']
        if self.use_dual_hebb:
            parts.append(f'hebb_slow_norm={self.hebb_slow_norm:.4f}')
        if self.use_consolidation:
            parts.append(f'consolidation={self.consolidation_strength}')
        return ', '.join(parts)
