#!/usr/bin/env python3
"""
Eğitim verimi benchmark'ı — token/saniye ve VRAM ölçümü.

Fused head + AMP + batch ölçekleme kazancını gerçek donanımda ölçer.

Kullanım (Colab T4):
    python scripts/benchmark.py                       # varsayılan tarama
    python scripts/benchmark.py --batches 8,32,64,128
    python scripts/benchmark.py --no-amp              # AMP kapalı karşılaştırma
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Windows konsolu (cp1254) emoji basamıyor — UTF-8'e zorla
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import torch
import torch.nn.functional as F

from liquidnn import MiniLiquidGPT


def bench_config(model, batch_size, seq_len, device, use_amp,
                 steps=5, warmup=2):
    """Bir konfigürasyonu ölç: forward+backward, plastisite ON."""
    x = torch.randint(0, model.vocab_size, (batch_size, seq_len),
                      device=device)
    y = torch.randint(0, model.vocab_size, (batch_size, seq_len),
                      device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    def step():
        model.reset_hebb()
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(x, enable_plasticity=True, chunk_size=16)
            loss = F.cross_entropy(logits.reshape(-1, model.vocab_size),
                                   y.reshape(-1))
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    for _ in range(warmup):
        step()
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / steps

    tokens_per_sec = batch_size * seq_len / elapsed
    peak_mb = (torch.cuda.max_memory_allocated() / 1e6
               if device.type == 'cuda' else 0)
    return elapsed, tokens_per_sec, peak_mb


def main():
    parser = argparse.ArgumentParser(description='Eğitim verimi benchmark')
    parser.add_argument('--batches', type=str, default='8,32,64,128',
                        help='Denenecek batch boyutları (virgülle)')
    parser.add_argument('--seq', type=int, default=128)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--no-amp', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = (not args.no_amp) and device.type == 'cuda'

    if device.type == 'cuda':
        print(f"📱 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  GPU yok — CPU'da ölçülüyor (sonuçlar temsili değil)")
    print(f"🔧 AMP: {'aktif' if use_amp else 'kapalı'} │ "
          f"seq={args.seq} │ plastisite ON, chunk=16\n")

    # Çıplak baseline model (ablation_baseline.yaml ile aynı)
    model = MiniLiquidGPT().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model: {n_params/1e6:.1f}M param (çıplak baseline)\n")

    print(f"{'batch':>6} │ {'adım süresi':>12} │ {'token/sn':>10} │ "
          f"{'VRAM (MB)':>10}")
    print("─" * 50)

    base_tps = None
    for bs in [int(b) for b in args.batches.split(',')]:
        try:
            elapsed, tps, peak = bench_config(
                model, bs, args.seq, device, use_amp, steps=args.steps)
            rel = f"  ({tps/base_tps:.1f}x)" if base_tps else ""
            if base_tps is None:
                base_tps = tps
            print(f"{bs:>6} │ {elapsed:>10.2f}s │ {tps:>10,.0f}{rel} │ "
                  f"{peak:>10,.0f}")
        except torch.cuda.OutOfMemoryError:
            print(f"{bs:>6} │ {'OOM':>12} │ {'—':>10} │ {'—':>10}")
            torch.cuda.empty_cache()
            break

    print("\nNot: token/sn sütunundaki (Nx) değeri ilk batch boyutuna göre")
    print("ölçeklenmeyi gösterir. Eski kodla karşılaştırma için aynı komutu")
    print("main@539635e üzerinde çalıştırın.")


if __name__ == "__main__":
    main()
