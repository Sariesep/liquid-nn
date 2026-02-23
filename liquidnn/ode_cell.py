"""
Sıvı ODE Hücresi — Adaptif Zaman Sabiti ile Nöral ODE

dx/dt = (-x + tanh(syn_ih(input) + syn_hh(x))) / τ(input, x)

τ (tau): Girdi-bağımlı zaman sabiti
  Kolay girdi → τ küçük → hızlı güncelleme
  Zor girdi   → τ büyük → yavaş, dikkatli güncelleme

v0.3.4 eklentileri:
  - Homeostatik Plastisite (aktivasyon dengeleme)
  - Nöromodülasyon desteği (mod_signal iletimi)
"""

import torch
import torch.nn as nn

from .plasticity import PlasticSynapse


class LiquidODECell(nn.Module):
    """
    Sıvı zaman sabiti ile ODE tabanlı gizli durum güncellemesi.

    Args:
        input_dim:       Girdi boyutu
        hidden_dim:      Gizli durum boyutu
        ode_steps:       ODE integrasyon adım sayısı (1=Euler, 3+=RK2)
        use_plasticity:  Hebb güncellemesi aktif mi
        use_homeostasis: Homeostatik plastisite aktif mi
        target_avg:      Hedef ortalama aktivasyon büyüklüğü
        use_dual_hebb:         Çift hızlı Hebb aktif mi (PlasticSynapse'e iletilir)
        use_consolidation:     Sinaptik konsolidasyon aktif mi
        consolidation_strength: Konsolidasyon gücü
    """

    def __init__(self, input_dim: int, hidden_dim: int, ode_steps: int = 3,
                 use_plasticity: bool = True,
                 use_homeostasis: bool = False, target_avg: float = 0.5,
                 use_dual_hebb: bool = False,
                 use_consolidation: bool = False,
                 consolidation_strength: float = 1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ode_steps = ode_steps
        self.use_plasticity = use_plasticity
        self.use_homeostasis = use_homeostasis
        self.target_avg = target_avg

        self.syn_ih = PlasticSynapse(
            input_dim, hidden_dim,
            use_dual_hebb=use_dual_hebb,
            use_consolidation=use_consolidation,
            consolidation_strength=consolidation_strength,
        )
        self.syn_hh = PlasticSynapse(
            hidden_dim, hidden_dim,
            use_dual_hebb=use_dual_hebb,
            use_consolidation=use_consolidation,
            consolidation_strength=consolidation_strength,
        )

        self.tau_net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
        )
        self.tau_min = 0.2

        # ── Homeostatik Plastisite ─────────────────────────────────
        if use_homeostasis:
            self.register_buffer('_act_ema',
                                 torch.full((hidden_dim,), target_avg))

    def _dynamics(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """ODE sağ tarafı: dh/dt"""
        interaction = torch.tanh(self.syn_ih(x) + self.syn_hh(h))
        tau = self.tau_net(torch.cat([x, h], dim=-1)) + self.tau_min
        return (-h + interaction) / tau

    def forward(self, x: torch.Tensor, h: torch.Tensor,
                enable_plasticity: bool = True,
                adaptive_steps: bool = False,
                tau_gate_residual: bool = False,
                moe_weight: float = 1.0,
                mod_signal: float = 1.0) -> torch.Tensor:
        """
        Tek token işle.

        Args:
            x: [B, D_in]  girdi
            h: [B, H]     önceki gizli durum
            enable_plasticity: Hebb güncellemesi yapılsın mı
            adaptive_steps: True → tau değerine göre adım sayısı dinamik seç
            tau_gate_residual: True → tau değeri residual gating'i kontrol eder
            moe_weight: Bu hücrenin seçilme ağırlığı (MoE'dan gelir)
            mod_signal: Nöromodülasyon sinyali (meta-plasticity)

        Returns:
            [B, H] yeni gizli durum
        """
        # Tau hesapla (adaptive steps ve/veya gating için)
        tau = None
        if adaptive_steps or tau_gate_residual:
            tau = self.tau_net(torch.cat([x, h], dim=-1)) + self.tau_min
            mean_tau = tau.mean().item()

        # Adım sayısını belirle
        steps = self.ode_steps
        if adaptive_steps and self.ode_steps > 1 and tau is not None:
            if mean_tau < 0.5:
                steps = 1
            elif mean_tau < 1.0:
                steps = 2

        dt = 1.0 / max(steps, 1)

        h_input = h  # residual için sakla
        step_moe_weight = moe_weight / steps  # Toplam Hebbian birikimini normalleştir

        target_update = enable_plasticity and self.use_plasticity

        if steps <= 1:
            h = h + dt * self._dynamics(h, x)
            if target_update:
                self.syn_ih.update_hebb(x, h, moe_weight=step_moe_weight,
                                        mod_signal=mod_signal)
                self.syn_hh.update_hebb(h, h, moe_weight=step_moe_weight,
                                        mod_signal=mod_signal)
        else:
            for _ in range(steps):
                k1 = self._dynamics(h, x)
                h_mid = h + 0.5 * dt * k1

                # Sürekli Plastisite: Zaman entegrasyonu ortasında Hebb güncellemesi
                if target_update:
                    self.syn_ih.update_hebb(x, h_mid,
                                           moe_weight=step_moe_weight,
                                           mod_signal=mod_signal)
                    self.syn_hh.update_hebb(h_mid, h_mid,
                                           moe_weight=step_moe_weight,
                                           mod_signal=mod_signal)

                k2 = self._dynamics(h_mid, x)
                h = h + dt * k2

        # Tau-Gated Residual: kolay → güçlü skip, zor → tam ODE
        if tau_gate_residual and tau is not None:
            gate = torch.sigmoid(1.0 / (tau.mean(dim=-1, keepdim=True) + 1e-6) - 1.0)
            h = gate * h_input + (1 - gate) * h

        # ── Homeostatik Plastisite ─────────────────────────────────
        if self.use_homeostasis:
            with torch.no_grad():
                current_act = h.abs().mean(dim=0)
                self._act_ema = 0.99 * self._act_ema + 0.01 * current_act
            scale = self.target_avg / (self._act_ema + 1e-6)
            scale = scale.clamp(0.5, 2.0)  # güvenlik sınırları
            h = h * scale

        return h

    def reset_hebb(self):
        self.syn_ih.reset_hebb()
        self.syn_hh.reset_hebb()

    def detach_hebb(self):
        self.syn_ih.detach_hebb()
        self.syn_hh.detach_hebb()

    @property
    def hebb_info(self) -> dict:
        info = {
            'ih': self.syn_ih.hebb_norm,
            'hh': self.syn_hh.hebb_norm,
        }
        if self.syn_ih.use_dual_hebb:
            info['ih_slow'] = self.syn_ih.hebb_slow_norm
            info['hh_slow'] = self.syn_hh.hebb_slow_norm
        return info
