#!/usr/bin/env python3
"""
Yazma Kuralı Karakterizasyonu — Hebbian vs Delta

Amaç: PlasticSynapse'ın plastik izini (H) bir HAFIZA SİSTEMİ olarak
ölçmek. Eğitim yok, gradyan yok — sadece izin yazma/okuma/unutma
dinamiği. Eğitim ablasyonu bu karakterizasyonun üstüne kurulacak.

Yöntem notları:
  - İz saf haliyle incelenir: recall(k) = H·k̂ (W ve alpha devre dışı).
    Böylece ölçülen şey sabit ağırlıkların değil, plastik hafızanın
    kendisidir.
  - Benzerlik ölçüsü cosine (ölçek-bağımsız): iki kuralın yazma
    genlikleri çok farklı (eta≈0.0094 vs beta=0.5), bu yüzden mutlak
    büyüklük karşılaştırması yanıltıcı olurdu.
  - Kalıcılık (half-life) ölçümünde cosine işe yaramaz (ölçek-bağımsız,
    saf solmayı göremez) — orada izin hedefe izdüşüm BÜYÜKLÜĞÜ kullanılır.
  - Her ölçüm torch.no_grad() altında: update_hebb v0.4'ten beri
    türevlenebilir, grafik biriktirmesi 1000 adımda belleği patlatırdı.

═══════════════════════════════════════════════════════════════════
HİPOTEZLER (koşmadan önce yazıldı)
═══════════════════════════════════════════════════════════════════

H1 — Anahtar korelasyonu:
  Delta her zaman GÜNCEL anahtar için tam çözer, dolayısıyla son
  çağrışımın geri çağrılması korelasyondan bağımsız ~1.0 kalmalı.
  Hebbian harmanladığı için korelasyon arttıkça son çağrışım bozulmalı.
  ESKİ çağrışımda ise ters yön beklenir: delta ortak bileşeni üzerine
  yazdığı için hızla düşmeli, Hebbian daha yumuşak inmeli.
  → Beklenen: makas açılması, iki kuralın farklı politikası.

H2 — Ardışık üzerine yazma (aynı anahtar, v1..v100):
  Delta hata düzeltmeli olduğu için hata (1-β)^n ile geometrik sönmeli
  → son hedefe cosine ~1.0. Hebbian tüm değerlerin decay-ağırlıklı
  toplamını tuttuğu için son hedefe yakınsayamaz, plato yapmalı.

H3 — Kapasite (n çağrışım):
  Delta d boyutlu uzayda ~d bağımsız çağrışıma kadar tutabilmeli
  (rank doygunluğu), sonra düşmeli. Hebbian karışma yüzünden daha
  erken bozulmalı.
  DİKKAT — karıştırıcı değişken: decay=0.989 ile n=1000'de ilk yazılan
  çağrışım zaten 0.989^1000 ≈ 1.6e-5'e sönmüş olur, yani KURAL fark
  etmeksizin kaybolur. Bu yüzden kapasite hem decay AÇIK hem decay
  KAPALI (=1.0) koşulacak: "solmadan kaybetme" ile "karışmadan
  kaybetme" ayrıştırılmalı.

H4 — Norm dinamiği:
  Delta ~√d ölçeğinde platoya oturmalı (ön ölçüm: d=32→6.2, d=256→25).
  Hebbian denge normu η·‖outer‖/(1-decay) civarında olmalı. İkisi de
  sınırlı ama çok farklı ölçekte.

H5 — Yarı ömür:
  Sessiz koşulda (girdi yok) iki kural da aynı decay'i kullandığı için
  yarı ömür özdeş olmalı: ln0.5/ln(0.989) ≈ 62.7 adım. Gürültülü
  koşulda (araya başka çağrışımlar) delta daha hızlı kaybetmeli —
  üzerine yazma ek bir silme kanalıdır.

H6 — Norm sınırı:
  Hebbian'da sınır regülatördür ve bağlanması normaldir. Delta'da
  bağlanması patolojiktir: her kırpma tüm matrisi küçültür, delta en
  son anahtarı tam güce geri yazar → eski çağrışımlar sistematik
  silinir. √d ölçeklemesinin bunu çözdüğü doğrulanmalı.
"""

import os
import sys
import math
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import torch
import torch.nn.functional as F

from liquidnn import PlasticSynapse


# ═══════════════════════════════════════════════════════════════════
#  Yardımcılar
# ═══════════════════════════════════════════════════════════════════

def make_synapse(rule, dim, cap_on=True, decay_on=True, channel_gate=False):
    """Karşılaştırılabilir sinaps üret."""
    syn = PlasticSynapse(dim, dim, update_rule=rule,
                         channel_gate=channel_gate)
    if not cap_on:
        # softplus(20) ≈ 20 → tipik çalışma normunun çok üstünde, fiilen kapalı
        syn.hebb_capacity.data.fill_(20.0)
    if not decay_on:
        # sigmoid(20) ≈ 1.0 → solma yok
        syn.logit_decay.data.fill_(20.0)
    return syn


def recall(syn, k):
    """İzden saf geri çağırma: H·k̂ (W ve alpha hariç)."""
    return F.linear(F.normalize(k, dim=-1), syn.Hebb)


def cos(a, b):
    return F.cosine_similarity(a, b, dim=-1).mean().item()


def correlated_key(k, corr, gen):
    """k ile hedeflenen cosine benzerliğinde yeni anahtar üret."""
    d = k.size(-1)
    r = torch.randn(1, d, generator=gen)
    # Gram-Schmidt: r'yi k'ye dik yap, sonra hedef açıyla karıştır
    k_hat = F.normalize(k, dim=-1)
    r = r - (r * k_hat).sum(-1, keepdim=True) * k_hat
    r = F.normalize(r, dim=-1)
    return corr * k_hat + math.sqrt(max(1e-9, 1 - corr ** 2)) * r


def gen_for(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ═══════════════════════════════════════════════════════════════════
#  Deney 1 — Anahtar korelasyonu
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def exp_key_correlation(dim=256, n_writes=15, seeds=5):
    """A yaz → B yaz → ikisini de geri çağır. Korelasyon taranır."""
    corrs = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]
    out = {r: {'recent': [], 'old': [], 'recent_sd': [], 'old_sd': []}
           for r in ('hebb', 'delta')}

    for rule in ('hebb', 'delta'):
        for corr in corrs:
            rec, old = [], []
            for s in range(seeds):
                g = gen_for(1000 + s)
                syn = make_synapse(rule, dim)
                ka = torch.randn(1, dim, generator=g)
                kb = correlated_key(ka, corr, g)
                va = torch.randn(1, dim, generator=g)
                vb = torch.randn(1, dim, generator=g)
                for _ in range(n_writes):
                    syn.update_hebb(ka, va)
                for _ in range(n_writes):
                    syn.update_hebb(kb, vb)
                rec.append(cos(recall(syn, kb), vb))
                old.append(cos(recall(syn, ka), va))
            t_r, t_o = torch.tensor(rec), torch.tensor(old)
            out[rule]['recent'].append(t_r.mean().item())
            out[rule]['old'].append(t_o.mean().item())
            out[rule]['recent_sd'].append(t_r.std().item())
            out[rule]['old_sd'].append(t_o.std().item())
    return corrs, out


# ═══════════════════════════════════════════════════════════════════
#  Deney 2 — Ardışık üzerine yazma
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def exp_sequential_overwrite(dim=256, n_values=100, seeds=3):
    """Aynı anahtara v1..v100 yaz; her adımda GÜNCEL hedefe uyum ölç."""
    out = {}
    for rule in ('hebb', 'delta'):
        curves = []
        for s in range(seeds):
            g = gen_for(2000 + s)
            syn = make_synapse(rule, dim)
            k = torch.randn(1, dim, generator=g)
            traj = []
            for i in range(n_values):
                v = torch.randn(1, dim, generator=g)
                syn.update_hebb(k, v)           # her değer bir kez yazılır
                traj.append(cos(recall(syn, k), v))
            curves.append(traj)
        out[rule] = torch.tensor(curves).mean(0).tolist()
    return out


# ═══════════════════════════════════════════════════════════════════
#  Deney 3 — Bellek kapasitesi
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def exp_capacity(dim=256, counts=(10, 50, 100, 500, 1000), seeds=3):
    """
    n çağrışım yaz, hepsini geri çağır.

    Üç rejim — iki karıştırıcıyı ayrı ayrı devre dışı bırakmak için:
      'sınır+decay' : kodun gerçek hali
      'sınırsız'    : norm sınırı kapalı → KURALLARIN temiz karşılaştırması
      'sınırsız+decaysiz': solma da kapalı → saf karışma kapasitesi

    Gerekçe: 5. deney norm sınırının Hebbian izini gürültü altında 63
    adımdan 3 adıma düşürdüğünü gösterdi. Sınır açıkken ölçülen kapasite
    kuralın değil, sınırın kapasitesidir.
    """
    regimes = (('sınır+decay', True, True),
               ('sınırsız', False, True),
               ('sınırsız+decaysiz', False, False))
    out = {}
    for label, cap_on, decay_on in regimes:
        for rule in ('hebb', 'delta'):
            means = []
            for n in counts:
                per_seed = []
                for s in range(seeds):
                    g = gen_for(3000 + s)
                    syn = make_synapse(rule, dim, cap_on=cap_on,
                                       decay_on=decay_on)
                    ks = torch.randn(n, dim, generator=g)
                    vs = torch.randn(n, dim, generator=g)
                    for i in range(n):
                        syn.update_hebb(ks[i:i+1], vs[i:i+1])
                    sims = [cos(recall(syn, ks[i:i+1]), vs[i:i+1])
                            for i in range(n)]
                    per_seed.append(sum(sims) / len(sims))
                means.append(sum(per_seed) / len(per_seed))
            out[(rule, label)] = means
    return list(counts), out


# ═══════════════════════════════════════════════════════════════════
#  Deney 4 — Norm dinamiği
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def exp_norm_dynamics(dim=256, steps=2000, sample=25):
    """‖H‖'nin zaman içindeki seyri; sınır açık/kapalı."""
    out = {}
    for rule in ('hebb', 'delta'):
        for cap_on in (True, False):
            g = gen_for(4000)
            syn = make_synapse(rule, dim, cap_on=cap_on)
            xs, ys = [], []
            for t in range(1, steps + 1):
                syn.update_hebb(torch.randn(4, dim, generator=g),
                                torch.randn(4, dim, generator=g))
                if t % sample == 0:
                    xs.append(t)
                    ys.append(syn.hebb_norm)
            out[(rule, cap_on)] = (xs, ys)
    return out


# ═══════════════════════════════════════════════════════════════════
#  Deney 5 — Yarı ömür
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def exp_half_life(dim=256, horizon=400, seeds=3):
    """
    Bir çağrışım yaz, sonra T adım boyunca izle.

    sessiz:            sonraki adımlarda girdi yok → SAF solma
    gürültülü:         başka rastgele çağrışımlar → solma + karışma
    gürültülü-sınırsız: aynısı ama norm sınırı kapalı

    Üçüncü koşulun amacı bir karıştırıcıyı ayrıştırmak: gürültü altındaki
    kayıp kuralın KENDİSİNDEN mi geliyor, yoksa norm sınırının her yeni
    yazmada tüm matrisi küçültmesinden mi? Sınır kapalıyken kayıp
    kayboluyorsa suçlu kural değil sınırdır.

    Ölçüm cosine değil, hedefe İZDÜŞÜM BÜYÜKLÜĞÜ (t=0'a normalize):
    cosine ölçek-bağımsızdır ve saf solmayı göremez.
    """
    out, halves = {}, {}
    zero = torch.zeros(1, dim)
    conds = (('sessiz', True), ('gürültülü', True),
             ('gürültülü-sınırsız', False))
    for rule in ('hebb', 'delta'):
        for cond, cap_on in conds:
            curves = []
            for s in range(seeds):
                g = gen_for(5000 + s)
                syn = make_synapse(rule, dim, cap_on=cap_on)
                k = torch.randn(1, dim, generator=g)
                v = torch.randn(1, dim, generator=g)
                for _ in range(10):
                    syn.update_hebb(k, v)
                v_hat = F.normalize(v, dim=-1)
                base = (recall(syn, k) * v_hat).sum().item()
                traj = [1.0]
                for _ in range(horizon):
                    if cond.startswith('sessiz'):
                        syn.update_hebb(zero, zero)
                    else:
                        syn.update_hebb(torch.randn(1, dim, generator=g),
                                        torch.randn(1, dim, generator=g))
                    traj.append((recall(syn, k) * v_hat).sum().item() /
                                (base + 1e-12))
                curves.append(traj)
            mean = torch.tensor(curves).mean(0)
            out[(rule, cond)] = mean.tolist()
            below = (mean < 0.5).nonzero()
            halves[(rule, cond)] = (below[0].item() if below.numel()
                                    else float('inf'))
    return out, halves


# ═══════════════════════════════════════════════════════════════════
#  Deney 6 — Norm sınırı etkisi
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def exp_norm_cap(dim=256, n_assoc=50, seeds=3):
    """
    Norm sınırının hafızaya etkisi — üç rejim:

      açık   : mevcut kod (delta'da sınır √d ile ölçekli)
      kapalı : sınır fiilen devre dışı
      eski   : v0.5 öncesi skaler sınır (√d ölçeklemesi yok)

    'eski' rejimi bir regresyon tanığıdır: delta'da skaler sınır her
    kırpmada tüm matrisi küçültür, delta ise en son anahtarı tam güce
    geri yazar → eski çağrışımlar sistematik silinir.

    Tek çift yerine n_assoc çağrışım kullanılır: sınırın bağlanması
    için izin dolu olması gerekir, iki çağrışımla sınır hiç devreye
    girmez ve ölçüm boş çıkar.
    """
    out = {}
    for rule in ('hebb', 'delta'):
        for regime in ('açık', 'kapalı', 'eski'):
            first, last = [], []
            for s in range(seeds):
                g = gen_for(6000 + s)
                syn = make_synapse(rule, dim, cap_on=(regime != 'kapalı'))
                if regime == 'eski':
                    syn._norm_scale = 1.0     # v0.5 öncesi davranış
                ks = torch.randn(n_assoc, dim, generator=g)
                vs = torch.randn(n_assoc, dim, generator=g)
                for i in range(n_assoc):
                    syn.update_hebb(ks[i:i+1], vs[i:i+1])
                first.append(cos(recall(syn, ks[0:1]), vs[0:1]))
                last.append(cos(recall(syn, ks[-1:]), vs[-1:]))
            out[(rule, regime)] = (sum(first) / len(first),
                                   sum(last) / len(last))
    return out


# ═══════════════════════════════════════════════════════════════════
#  Çizim
# ═══════════════════════════════════════════════════════════════════

def plot_all(results, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    C = {'hebb': '#e07b39', 'delta': '#2e86ab'}
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1 — korelasyon
    corrs, r1 = results['corr']
    ax = axes[0, 0]
    for rule in ('hebb', 'delta'):
        ax.errorbar(corrs, r1[rule]['recent'], yerr=r1[rule]['recent_sd'],
                    marker='o', color=C[rule], label=f'{rule} — son çağrışım')
        ax.errorbar(corrs, r1[rule]['old'], yerr=r1[rule]['old_sd'],
                    marker='s', ls='--', color=C[rule], alpha=0.6,
                    label=f'{rule} — eski çağrışım')
    ax.set_xlabel('anahtar korelasyonu (cosine)')
    ax.set_ylabel('geri çağırma (cosine)')
    ax.set_title('1. Anahtar korelasyonu')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # 2 — ardışık üzerine yazma
    ax = axes[0, 1]
    for rule, traj in results['seq'].items():
        ax.plot(range(1, len(traj) + 1), traj, color=C[rule], label=rule)
    ax.set_xlabel('yazma sırası (aynı anahtar)')
    ax.set_ylabel('güncel hedefe uyum (cosine)')
    ax.set_title('2. Ardışık üzerine yazma')
    ax.legend(); ax.grid(alpha=0.3)

    # 3 — kapasite
    counts, r3 = results['cap']
    ax = axes[0, 2]
    ls_map = {'sınır+decay': '-', 'sınırsız': '--',
              'sınırsız+decaysiz': ':'}
    for (rule, label), ys in r3.items():
        ax.plot(counts, ys, marker='o', ms=4, color=C[rule],
                ls=ls_map[label], label=f"{rule} — {label}")
    ax.set_xscale('log')
    ax.set_xlabel('saklanan çağrışım sayısı')
    ax.set_ylabel('ortalama geri çağırma (cosine)')
    ax.set_title('3. Bellek kapasitesi (d=256)')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # 4 — norm dinamiği
    ax = axes[1, 0]
    for (rule, cap_on), (xs, ys) in results['norm'].items():
        ax.plot(xs, ys, color=C[rule], ls='-' if cap_on else '--',
                label=f"{rule} — sınır {'açık' if cap_on else 'kapalı'}")
    ax.set_yscale('log')
    ax.set_xlabel('güncelleme adımı'); ax.set_ylabel('‖H‖')
    ax.set_title('4. Norm dinamiği')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # 5 — yarı ömür
    ax = axes[1, 1]
    curves, halves = results['half']
    styles = {'sessiz': '-', 'gürültülü': ':', 'gürültülü-sınırsız': '-.'}
    for (rule, cond), traj in curves.items():
        ax.plot(traj, color=C[rule], ls=styles[cond], lw=1.4,
                label=f'{rule} — {cond}')
    ax.axhline(0.5, color='gray', lw=0.8, ls='--')
    ax.set_xlabel('sonraki adım sayısı')
    ax.set_ylabel('kalan iz (t=0\'a normalize)')
    ax.set_title('5. Bilginin yarı ömrü')
    ax.legend(fontsize=6); ax.grid(alpha=0.3)

    # 6 — norm sınırı
    ax = axes[1, 2]
    labels, firsts, lasts = [], [], []
    for (rule, regime), (f_, l_) in results['capgate'].items():
        labels.append(f"{rule}\nsınır: {regime}")
        firsts.append(f_); lasts.append(l_)
    xpos = range(len(labels))
    ax.bar([x - 0.2 for x in xpos], firsts, 0.4,
           label='ilk yazılan çağrışım', color='#888')
    ax.bar([x + 0.2 for x in xpos], lasts, 0.4,
           label='son yazılan çağrışım', color='#2e86ab')
    ax.set_xticks(list(xpos)); ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel('geri çağırma (cosine)')
    ax.set_title('6. Norm sınırı rejimleri (50 çağrışım)')
    ax.legend(fontsize=6); ax.grid(alpha=0.3, axis='y')

    fig.suptitle('PlasticSynapse yazma kuralı karakterizasyonu — '
                 'Hebbian vs Delta', fontsize=14)
    fig.tight_layout()
    path = os.path.join(outdir, 'write_rule_analysis.png')
    fig.savefig(path, dpi=130)
    print(f"\n📈 Grafik: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description='Yazma kuralı karakterizasyonu')
    p.add_argument('--dim', type=int, default=256)
    p.add_argument('--outdir', type=str, default='docs/analysis')
    p.add_argument('--no-plot', action='store_true')
    args = p.parse_args()

    torch.manual_seed(0)
    d = args.dim
    print("=" * 68)
    print(f"  YAZMA KURALI KARAKTERİZASYONU (d={d})")
    print("=" * 68)

    results = {}

    print("\n[1/6] Anahtar korelasyonu...")
    corrs, r1 = exp_key_correlation(dim=d)
    results['corr'] = (corrs, r1)
    print(f"  {'korelasyon':>10} │ {'hebb son':>9} {'hebb eski':>10} │"
          f" {'delta son':>10} {'delta eski':>11}")
    for i, c in enumerate(corrs):
        print(f"  {c:>10.2f} │ {r1['hebb']['recent'][i]:>9.3f} "
              f"{r1['hebb']['old'][i]:>10.3f} │ "
              f"{r1['delta']['recent'][i]:>10.3f} "
              f"{r1['delta']['old'][i]:>11.3f}")

    print("\n[2/6] Ardışık üzerine yazma...")
    results['seq'] = exp_sequential_overwrite(dim=d)
    for rule, traj in results['seq'].items():
        print(f"  {rule:>6}: yazma#1={traj[0]:.3f}  #10={traj[9]:.3f}  "
              f"#50={traj[49]:.3f}  #100={traj[-1]:.3f}")

    print("\n[3/6] Bellek kapasitesi...")
    counts, r3 = exp_capacity(dim=d)
    results['cap'] = (counts, r3)
    print(f"  {'n':>6} │ " + " ".join(
        f"{r}/{lab}".rjust(20) for (r, lab) in r3))
    for i, n in enumerate(counts):
        print(f"  {n:>6} │ " + " ".join(f"{v[i]:>20.3f}" for v in r3.values()))

    print("\n[4/6] Norm dinamiği...")
    results['norm'] = exp_norm_dynamics(dim=d)
    for (rule, cap_on), (xs, ys) in results['norm'].items():
        print(f"  {rule:>6} sınır={'açık ' if cap_on else 'kapalı'}: "
              f"t=100→{ys[3]:.2f}  t=1000→{ys[39]:.2f}  t=2000→{ys[-1]:.2f}")

    print("\n[5/6] Yarı ömür...")
    curves, halves = exp_half_life(dim=d)
    results['half'] = (curves, halves)
    theo = math.log(0.5) / math.log(torch.sigmoid(torch.tensor(4.5)).item())
    print(f"  Teorik saf-solma yarı ömrü: {theo:.1f} adım")
    for (rule, cond), h in halves.items():
        print(f"  {rule:>6} / {cond:<10}: {h} adım")

    print("\n[6/6] Norm sınırı etkisi...")
    results['capgate'] = exp_norm_cap(dim=d)
    for (rule, regime), (f_, l_) in results['capgate'].items():
        print(f"  {rule:>6} sınır={regime:<7}: "
              f"ilk={f_:.3f}  son={l_:.3f}")

    if not args.no_plot:
        plot_all(results, args.outdir)


if __name__ == "__main__":
    main()
