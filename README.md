# 🧠 Liquid Neural Networks — Plastik Sinapslarla Gerçek Zamanlı Öğrenme

> **GPT/Gemini statiktir. Eğitimi biter, donar. Bu model canlıdır — her token'da sinapslarını değiştirir.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## 🔬 Bu Nedir?

Sıfırdan yazılmış, Liquid Time-Constant Networks + Differentiable Hebbian Plasticity tabanlı bir dil modeli araştırma projesi.

### Transformer'lardan Farkımız

| | Transformer (GPT/Gemini) | Liquid Neural Network (Biz) |
|---|---|---|
| **Sinapslar** | Sabit (eğitim sonrası donar) | Plastik (her token'da güncellenir) |
| **Hafıza** | Context window (geçici) | Hebbian izler (kalıcı) |
| **Adaptasyon** | Fine-tune gerekir (saatler) | Gerçek zamanlı (milisaniyeler) |
| **Hesaplama** | Sabit derinlik | Adaptif ODE adımları (kolay→hızlı, zor→derin) |

### Mimari

```
Token → Embed(50257, 256) + SinPosEnc
  → LiquidODE × 2 (steps=1, Euler — hızlı algı)
  → LiquidODE × 2 (steps=3, RK2 + Hebb — derin düşünce)
  → Head (weight-tied) → Logits
```

## 🚀 Hızlı Başlangıç

```bash
# Klonla
git clone https://github.com/KULLANICI_ADIN/liquid-nn.git
cd liquid-nn

# Bağımlılıkları kur
pip install -r requirements.txt

# Eğit (Colab T4 veya yerel GPU)
python scripts/train.py --config configs/base.yaml

# Metin üret
python scripts/generate.py --checkpoint checkpoints/best_model.pt --prompt "The meaning of life"

# Plastisite testi
python scripts/plasticity_test.py --checkpoint checkpoints/best_model.pt
```

### Google Colab'da Çalıştır

```python
!git clone https://github.com/KULLANICI_ADIN/liquid-nn.git
%cd liquid-nn
!pip install -r requirements.txt
!python scripts/train.py --config configs/colab_t4.yaml
```

## 📁 Proje Yapısı

```
liquid-nn/
├── liquidnn/                # Ana kütüphane (pip install edilebilir)
│   ├── __init__.py
│   ├── plasticity.py        # PlasticSynapse — Hebbian öğrenme
│   ├── ode_cell.py          # LiquidODECell — Sıvı nöron
│   ├── model.py             # MiniLiquidGPT — Ana model
│   ├── tokenizer.py         # tiktoken sarmalayıcı
│   └── utils.py             # Yardımcı fonksiyonlar
├── configs/                 # Eğitim konfigürasyonları
│   ├── base.yaml            # Varsayılan ayarlar
│   ├── colab_t4.yaml        # Colab T4 optimize
│   ├── small.yaml           # Hızlı deney (~5M param)
│   └── large.yaml           # Büyük model (~50M param)
├── scripts/                 # Çalıştırılabilir scriptler
│   ├── train.py             # Eğitim
│   ├── generate.py          # Metin üretimi
│   ├── plasticity_test.py   # ZEPHYR / Bloop testi
│   └── benchmark.py         # Performans ölçümü
├── data/                    # Veri yükleme
│   └── loader.py
├── tests/                   # Unit testler
│   ├── test_plasticity.py
│   ├── test_ode_cell.py
│   └── test_model.py
├── notebooks/               # Jupyter notebook'lar
│   ├── 01_quickstart.ipynb
│   ├── 02_plasticity_demo.ipynb
│   └── 03_training.ipynb
├── docs/                    # Dokümantasyon
│   ├── architecture.md
│   └── plasticity.md
├── checkpoints/             # Model ağırlıkları (git'te yok)
├── requirements.txt
├── setup.py
├── pyproject.toml
├── .gitignore
├── LICENSE
└── README.md
```

## 📊 Sonuçlar

| Metrik | Değer |
|---|---|
| Parametreler | ~14M |
| Val Perplexity | ... |
| Plastisite ON vs OFF | ... |
| ZEPHYR Kalıcılık | ... |

*Sonuçlar eğitim tamamlandıkça güncellenecek.*

## 🔬 Araştırma Notları

Bu proje şu makalelerden ilham alır:
- [Liquid Time-constant Networks](https://arxiv.org/abs/2006.04439) (Hasani et al., 2020)
- [Differentiable Plasticity](https://arxiv.org/abs/1804.02464) (Miconi et al., 2018)
- [Neural ODEs](https://arxiv.org/abs/1806.07366) (Chen et al., 2018)

## 📝 Lisans

MIT License — İstediğin gibi kullan, geliştir, paylaş.

## 🤝 Katkı

Pull request'ler açıktır! Özellikle şu konularda yardım aranıyor:
- [ ] Daha büyük veri setleri (TinyStories, Cosmopedia)
- [ ] Multi-head plasticity
- [ ] Benchmark karşılaştırmaları (GPT-2 small vs Liquid)
- [ ] ONNX/TensorRT export
- [ ] Mobil deployment (CoreML, NNAPI)
