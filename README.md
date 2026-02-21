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

### Architecture

```
Token → Embed(50257, 256) + SinPosEnc
  → LiquidODE × 2 (steps=1, Euler — fast perception)
  → LiquidODE × 2 (steps=3, RK2 + Hebb — deep reasoning)
  → Head (weight-tied) → Logits
```

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/liquid-nn.git
cd liquid-nn

# Install dependencies
pip install -r requirements.txt

# Train (Colab T4 or local GPU)
python scripts/train.py --config configs/base.yaml

# Generate text
python scripts/generate.py --checkpoint checkpoints/best_model.pt --prompt "The meaning of life"

# Plasticity test
python scripts/plasticity_test.py --checkpoint checkpoints/best_model.pt
```

### Run on Google Colab

```python
!git clone https://github.com/YOUR_USERNAME/liquid-nn.git
%cd liquid-nn
!pip install -r requirements.txt
!python scripts/train.py --config configs/colab_t4.yaml
```

## 📁 Project Structure

```
liquid-nn/
├── liquidnn/                # Main library (pip installable)
│   ├── __init__.py
│   ├── plasticity.py        # PlasticSynapse — Hebbian learning
│   ├── ode_cell.py          # LiquidODECell — Liquid neuron
│   ├── model.py             # MiniLiquidGPT — Main model
│   ├── tokenizer.py         # tiktoken wrapper
│   └── utils.py             # Utility functions
├── configs/                 # Training configurations
│   ├── base.yaml            # Default settings
│   ├── colab_t4.yaml        # Colab T4 optimized
│   ├── small.yaml           # Quick experiments (~5M params)
│   └── large.yaml           # Large model (~50M params)
├── scripts/                 # Executable scripts
│   ├── train.py             # Training
│   ├── generate.py          # Text generation
│   ├── plasticity_test.py   # ZEPHYR / Bloop test
│   └── benchmark.py         # Performance measurement
├── data/                    # Data loading
│   └── loader.py
├── tests/                   # Unit tests
│   ├── test_plasticity.py
│   ├── test_ode_cell.py
│   └── test_model.py
├── notebooks/               # Jupyter notebooks
│   ├── 01_quickstart.ipynb
│   ├── 02_plasticity_demo.ipynb
│   └── 03_training.ipynb
├── docs/                    # Documentation
│   ├── architecture.md
│   └── plasticity.md
├── checkpoints/             # Model weights (not in git)
├── requirements.txt
├── setup.py
├── pyproject.toml
├── .gitignore
├── LICENSE
└── README.md
```

## 📊 Results

| Metric | Value |
|---|---|
| Parameters | ~14M |
| Val Perplexity | ... |
| Plasticity ON vs OFF | ... |
| ZEPHYR Persistence | ... |

*Results will be updated as training completes.*

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
- [ ] Multi-head plasticity
- [ ] Benchmark comparisons (GPT-2 small vs Liquid)
- [ ] ONNX/TensorRT export
- [ ] Mobile deployment (CoreML, NNAPI)
