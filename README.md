# 🧠 Liquid Neural Networks — Real-Time Learning with Plastic Synapses

> **GPT/Gemini are static. Training ends, they freeze. This model is alive — it changes its synapses on every token.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## 🔬 What Is This?

A language model research project built from scratch, based on Liquid Time-Constant Networks + Differentiable Hebbian Plasticity.

### How We Differ from Transformers

| | Transformer (GPT/Gemini) | Liquid Neural Network (Ours) |
|---|---|---|
| **Synapses** | Fixed (freeze after training) | Plastic (updated on every token) |
| **Memory** | Context window (temporary) | Hebbian traces (persistent) |
| **Adaptation** | Requires fine-tuning (hours) | Real-time (milliseconds) |
| **Computation** | Fixed depth | Adaptive ODE steps (easy→fast, hard→deep) |

### Architecture (v0.3.5)

```
Token → Embed(50257, 256) + SinPosEnc
  → LiquidODE × 2 (steps=1, Euler — fast perception)
  → LiquidODE × 2 (steps=3, RK2 + Hebb — deep reasoning)
  → Head (weight-tied, fused) → Logits
```

Optional components, all off by default and enabled via config flags:
Sliding-Window Attention (RoPE, KV cache, Flash, GQA), SwiGLU FFN,
MoE router with capacity limiting, neuromodulation, homeostatic
plasticity, dual-rate Hebb, synaptic consolidation, RMSNorm,
tau-gated residual, speculative decoding, INT8 quantization.

- Bare baseline: **~14.7M params** (`configs/ablation_baseline.yaml`)
- Full v0.3.5: **~17.5M params** (`configs/base.yaml`)

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/Sariesep/liquid-nn.git
cd liquid-nn

# Install dependencies
pip install -r requirements.txt

# Train (Colab T4 or local GPU)
python scripts/train.py --config configs/base.yaml

# Generate text
python scripts/generate.py --checkpoint checkpoints/best_model.pt --prompt "The meaning of life"

# Plasticity test
python scripts/plasticity_test.py --checkpoint checkpoints/best_model.pt

# Training throughput benchmark
python scripts/benchmark.py
```

### Run on Google Colab

```python
!git clone https://github.com/Sariesep/liquid-nn.git
%cd liquid-nn
!pip install -r requirements.txt
!python scripts/train.py --config configs/colab_t4.yaml
```

## 📁 Project Structure

```
liquid-nn/
├── liquidnn/                  # Main library (pip installable)
│   ├── plasticity.py          # PlasticSynapse — Hebbian learning
│   ├── ode_cell.py            # LiquidODECell — Liquid neuron
│   ├── model.py               # MiniLiquidGPT — Main model
│   ├── attention.py           # Sliding-window attention (RoPE/KV/Flash/GQA)
│   ├── ffn.py                 # SwiGLU feed-forward
│   ├── moe.py                 # Mixture-of-Experts router
│   ├── neuromodulation.py     # Prediction-error neuromodulator
│   ├── rmsnorm.py             # RMSNorm
│   ├── distillation.py        # Teacher→student distillation
│   ├── quantize.py            # INT8 dynamic quantization
│   ├── tokenizer.py           # tiktoken wrapper
│   └── utils.py               # Save/load, device setup
├── configs/
│   ├── base.yaml              # Full v0.3.5 model (all flags on)
│   ├── colab_t4.yaml          # Same, checkpoints to Google Drive
│   ├── ablation_baseline.yaml # Bare Liquid ODE + Hebb (ablation)
│   ├── small.yaml             # Quick experiments
│   └── large.yaml             # Large model
├── scripts/
│   ├── train.py               # Training (dual-mode validation, AMP)
│   ├── generate.py            # Text generation
│   ├── plasticity_test.py     # ZEPHYR persistence test
│   └── benchmark.py           # Throughput/VRAM measurement
├── data/
│   └── loader.py              # Wikitext-2 (Shakespeare fallback)
├── notebooks/
│   ├── colab_demo_v034.py     # Colab demo cells
│   └── colab_train_2h.py      # 2-hour Colab training run
├── tests/                     # 100 unit tests (pytest)
├── docs/
│   └── architecture.md
└── checkpoints/               # Model weights (not in git)
```

## 📊 Results

| Metric | Value |
|---|---|
| Parameters | 14.7M (bare) / 17.5M (full) |
| Val Perplexity | *ablation in progress* |
| Plasticity ON vs OFF | *ablation in progress* |
| ZEPHYR Persistence | *pending trained checkpoint* |

*The plasticity ON/OFF ablation is currently running; this table will be
filled with measured numbers, not projections.*

## 🔬 Research Notes

This project is inspired by the following papers:
- [Liquid Time-constant Networks](https://arxiv.org/abs/2006.04439) (Hasani et al., 2020)
- [Differentiable Plasticity](https://arxiv.org/abs/1804.02464) (Miconi et al., 2018)
- [Neural ODEs](https://arxiv.org/abs/1806.07366) (Chen et al., 2018)

## 📝 License

MIT License — Use, modify, and share as you like.

## 🤝 Contributing

Pull requests are welcome! Help is especially needed on:
- [ ] Larger datasets (TinyStories, Cosmopedia)
- [ ] Matched-parameter transformer baseline for fair comparison
- [ ] Benchmark comparisons (GPT-2 small vs Liquid)
- [ ] ONNX/TensorRT export
- [ ] Mobile deployment (CoreML, NNAPI)
