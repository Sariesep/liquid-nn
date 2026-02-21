# Mimari Detaylar

## Genel Bakış

MiniLiquidGPT, Liquid Time-Constant Networks (LTC) ile Differentiable Hebbian Plasticity'yi birleştiren bir dil modelidir.

## Katman Yapısı

```
Token → Embed(50257, 256) + SinPosEnc
  → LiquidODE_0 (steps=1, Euler, plastisite OFF)  ← Hızlı algı
  → LiquidODE_1 (steps=1, Euler, plastisite OFF)  ← Basit gramer
  → LiquidODE_2 (steps=3, RK2,   plastisite ON)   ← Derin düşünce
  → LiquidODE_3 (steps=3, RK2,   plastisite ON)   ← Hebb öğrenme
  → LayerNorm → lm_head (embed weight tying)
```

### Hızlı Katmanlar (steps=1)
- Tek Euler adımı: `h = h + Δt · f(h, x)`
- Plastisite OFF — sadece statik ağırlıklar
- Amaç: Temel token işleme, gramer, basit örüntüler

### Derin Katmanlar (steps=3)
- RK2 Midpoint: Her adımda 2 dynamics evaluasyonu
- Plastisite ON — Hebb matrisi her token'da güncellenir
- Amaç: Anlam çıkarma, yeni kural öğrenme, adaptasyon

## ODE Dinamikleri

```
dh/dt = (-h + tanh(W_ih·x + W_hh·h)) / τ(x, h)
```

- `τ(x, h)`: Adaptif zaman sabiti — zor girdilerde yavaşlar
- `W_eff = W_base + α ⊙ Hebb`: Efektif ağırlık plastik bileşen içerir

## Hebbian Plastisite

```
Hebb ← decay · Hebb + η · (post ⊗ pre)
```

- `α`: Hangi sinapsların plastik olduğunu belirler (eğitimle öğrenilir)
- `η`: Öğrenme hızı (eğitimle öğrenilir)  
- `decay`: Unutma hızı (eğitimle öğrenilir)
- Norm sınırı: ||Hebb|| ≤ 0.3 · ||W_base||

## Truncated BPTT

Hesaplama grafiği her `chunk_size` token'da kesilir:
```python
if t % chunk_size == 0:
    hiddens = [h.detach() for h in hiddens]
    cells.detach_hebb()
```

Bu, bellek kullanımını O(T) → O(chunk_size) düşürür.

## Weight Tying

`lm_head` ayrı bir katman değil, `embed.weight` kullanılır:
```python
logits = F.linear(x, self.embed.weight)
```
Bu, ~13M parametre tasarrufu sağlar (50257 × 256).
