# Write-Rule Characterization: Hebbian vs Delta

**Date:** 2026-07-27 · **Code:** `scripts/analyze_write_rules.py` · **d = 256**
(matches `embed_dim`) · seeds: 3–5 per condition · no training, no gradients —
the plastic trace `H` is studied purely as a memory system.

Reproduce: `python scripts/analyze_write_rules.py`

## Headline

**The norm cap — not the write rule — is the dominant memory-destroying
mechanism in this codebase.** Under realistic (noisy) operation the Hebbian
trace's half-life collapses from 63 steps to **3 steps** when the cap is
active. Once that confound is removed, the two write rules have *comparable*
raw capacity; delta's apparent 6x capacity advantage was an artifact of the
cap penalising Hebbian only.

This is a second structural defect, independent of the `@torch.no_grad()` bug
fixed in v0.4.0. Both were present during the first ablation, whose null
result therefore says nothing about the merit of Hebbian plasticity.

## Method notes

- Recall is measured on the trace alone: `recall(k) = H·k̂`, excluding `W` and
  `alpha`, so what is measured is the plastic memory rather than the static
  weights.
- Similarity is cosine (scale-invariant): the two rules write at very
  different amplitudes (η ≈ 0.0094 vs β = 0.5), so magnitude comparisons would
  mislead.
- Half-life uses projection *magnitude* onto the target, not cosine — cosine is
  scale-invariant and cannot see pure decay.
- All measurements under `torch.no_grad()`; `update_hebb` is differentiable
  since v0.4.0 and would otherwise accumulate a 1000-step graph.

## Results

### 1. Key correlation — the rules implement different policies

| key corr | hebb recent | hebb old | delta recent | delta old |
|---|---|---|---|---|
| 0.00 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.40 | 0.916 | 0.938 | **1.000** | 0.868 |
| 0.80 | 0.748 | 0.801 | **1.000** | 0.336 |
| 0.95 | 0.687 | 0.747 | **1.000** | **0.065** |

Delta returns the most recent association *exactly*, at any correlation, and
pays for it by overwriting the shared component of the older one. Hebbian
blends: both associations degrade gracefully together. Neither is universally
better — this is overwrite vs. blend semantics, and which one wins is
task-dependent. **This difference is genuine, not a cap artifact** (the cap
never binds for delta).

### 2. Rapid overwrite of one key (v1 → v100, one write each)

| | write #1 | #10 | #50 | #100 |
|---|---|---|---|---|
| hebb | 1.000 | 0.676 | 0.636 | 0.627 |
| delta | 1.000 | 0.871 | 0.878 | **0.855** |

Delta tracks a changing value substantially better. With β = 0.5 a single
write moves the trace halfway to the new target, so it does not fully overwrite
in one shot — full convergence needs repeated writes (verified separately:
error 1.66 → 0.036 over 30 writes).

### 3. Capacity — the cap confound, revealed

Mean recall over all *n* stored associations (random keys):

| n | hebb (as-shipped) | delta (as-shipped) | hebb (cap off) | delta (cap off) | hebb (cap+decay off) | delta (cap+decay off) |
|---|---|---|---|---|---|---|
| 10 | 0.887 | 0.988 | 0.982 | 0.988 | 0.982 | 0.989 |
| 50 | **0.242** | 0.927 | **0.912** | 0.927 | 0.918 | 0.944 |
| 100 | **0.135** | 0.818 | **0.812** | 0.818 | 0.849 | 0.888 |
| 500 | 0.034 | 0.262 | 0.288 | 0.262 | 0.544 | 0.579 |
| 1000 | 0.017 | 0.131 | 0.145 | 0.131 | 0.351 | 0.385 |

Turning the cap off lifts Hebbian at n=100 from 0.135 to 0.812 — a 6x recovery
— while leaving delta bit-identical (the √d scaling means the cap never binds
for delta). **With the cap removed the two rules are within noise of each
other.** Error bars were not computed here (3 seeds); the small differences at
n ≥ 500 should not be read as a ranking.

Removing the cap then exposes decay as the next limit: at n = 1000, disabling
decay as well roughly doubles retention for both rules.

### 4. Norm dynamics

| | t=100 | t=1000 | t=2000 |
|---|---|---|---|
| hebb, cap on | 3.11 | 3.60 | 3.74 |
| hebb, cap off | 7.68 | 8.14 | **8.13** (plateau) |
| delta, cap on | 24.06 | 25.10 | 24.96 |
| delta, cap off | 24.06 | 25.10 | **24.96** (identical) |

Both rules are self-limiting: Hebbian plateaus at ≈8.1 (equilibrium
η·‖outer‖/(1−decay)), delta at ≈25 (≈√d scaling). The cap clamps Hebbian to
3.74, well below its natural equilibrium — it is regulating a quantity that
did not need regulating.

### 5. Half-life — the decisive experiment

Theoretical pure-decay half-life at decay = 0.989: **62.7 steps**.

| rule | silent (decay only) | noisy (decay + interference) | noisy, cap off |
|---|---|---|---|
| hebb | 63 | **3** | **63** |
| delta | 63 | 56 | 56 |

The Hebbian trace loses half its content in **3 steps** under interference —
and recovers fully to 63 steps with the cap disabled. The cap, not additive
accumulation, is the cause. Delta is unaffected (56 ≈ 63) because its cap never
binds; its small loss is genuine interference.

### 6. Norm-cap regimes (50 associations)

| rule | regime | first association | last association |
|---|---|---|---|
| hebb | as-shipped | **−0.020** | 0.995 |
| hebb | cap off | **0.829** | 0.950 |
| delta | as-shipped | 0.800 | 0.988 |
| delta | cap off | 0.800 | 0.988 |
| delta | legacy scalar cap | **0.014** | 1.000 |

The first-written association is *completely destroyed* for Hebbian as shipped
(−0.020 = noise) and recovers to 0.829 with the cap off. The legacy-scalar-cap
row reproduces the pre-v0.5 pathology on delta (0.014) that the √d scaling
fixed — a regression witness.

## Mechanism

The cap rescales the **entire matrix** by `max_norm/‖H‖` whenever the norm is
exceeded. Every write that trips the cap therefore attenuates *all previously
stored associations* by the same factor, while the association just written is
restored to full strength by the next write. Repeated over many steps this is
an exponential erasure channel stacked on top of decay, with a rate set by how
often the cap trips — not by anything the model can learn. For Hebbian the cap
trips constantly (natural equilibrium 8.13 vs cap 3.74), which is why its
effective half-life drops to 3 steps.

## What this means for the ablation

A trace with a 3-step half-life cannot contribute information the recurrent
hidden state does not already carry, so the v0.3.5 ablation was measuring a
memory that had been structurally disabled twice over: the write path was
outside autograd (`@torch.no_grad()`, fixed in v0.4.0) *and* the retained
content was being erased every few tokens. **Neither result is evidence for or
against Hebbian plasticity.**

## Recommended next step (not yet applied)

Apply the same treatment the delta rule received: make the cap a safety valve
rather than a regulator for the Hebbian rule too — scale `max_norm` so that it
sits above the natural equilibrium (≈8.1 at d=256) instead of below it, e.g. by
applying the existing `_norm_scale` to both rules or raising the capacity init.
Then re-run the three-arm ablation (control / hebb / delta) on identical seeds.

Until that is done, no training-level conclusion about write rules should be
drawn.
