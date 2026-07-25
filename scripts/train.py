#!/usr/bin/env python3
"""
MiniLiquidGPT Eğitim Scripti

Kullanım:
    python scripts/train.py --config configs/base.yaml
    python scripts/train.py --config configs/colab_t4.yaml
    python scripts/train.py  # varsayılan: configs/base.yaml
"""

import os
import sys
import json
import math
import time
import inspect
import argparse

# Proje kökünü path'e ekle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F

from liquidnn import MiniLiquidGPT, TokenizerWrapper, get_tokenizer
from liquidnn.utils import setup_device, setup_drive, save_model, save_history
from data.loader import load_data


def load_config(path: str) -> dict:
    """YAML config yükle."""
    try:
        import yaml
    except ImportError:
        os.system("pip install pyyaml -q")
        import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate(model, val_x, val_y, batch_size=8, max_batches=30,
             enable_plasticity=False):
    """Validation loss (tek modda)."""
    model.eval()
    total, n = 0.0, 0
    N = val_x.size(0)
    with torch.no_grad():
        for i in range(0, N, batch_size):
            if n >= max_batches:
                break
            x = val_x[i:i+batch_size]
            y = val_y[i:i+batch_size]
            model.reset_hebb()
            logits = model(x, enable_plasticity=enable_plasticity,
                           chunk_size=32)
            loss = F.cross_entropy(logits.reshape(-1, model.vocab_size),
                                   y.reshape(-1))
            total += loss.item()
            n += 1
    model.reset_hebb()
    return total / max(n, 1)


def evaluate_both(model, val_x, val_y, batch_size=8, max_batches=30):
    """
    Validation'ı iki modda ölç: plastisite OFF (statik) ve ON (plastik).

    Projenin ana tezi çıkarım anındaki plastisitenin faydası olduğundan
    yalnızca OFF ölçmek tezi görünmez kılar; ikisi de raporlanır.
    """
    val_off = evaluate(model, val_x, val_y, batch_size, max_batches,
                       enable_plasticity=False)
    val_on = evaluate(model, val_x, val_y, batch_size, max_batches,
                      enable_plasticity=True)
    return val_off, val_on


def train(model, train_x, train_y, val_x, val_y, cfg, save_dir):
    """Ana eğitim döngüsü."""
    tc = cfg['training']
    epochs = tc['epochs']
    batch_size = tc['batch_size']
    lr = tc['lr']
    chunk_size = tc['chunk_size']
    phase_split = tc.get('phase_split', 0.5)

    N = train_x.size(0)
    num_batches = N // batch_size
    device = train_x.device

    # AMP: cuda'da varsayılan açık (training.amp: false ile kapatılır)
    use_amp = tc.get('amp', True) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    print(f"\n🔬 Eğitim: {epochs} epoch × {num_batches} batch"
          f"{'  │  AMP aktif' if use_amp else ''}")

    # Optimizer
    plast_names = {'alpha', 'log_eta', 'logit_decay'}
    plast_p = [p for n, p in model.named_parameters()
               if any(pn in n for pn in plast_names)]
    other_p = [p for n, p in model.named_parameters()
               if not any(pn in n for pn in plast_names)]

    optimizer = torch.optim.AdamW([
        {'params': other_p, 'lr': lr, 'weight_decay': tc.get('weight_decay', 1e-4)},
        {'params': plast_p, 'lr': lr * tc.get('plast_lr_mult', 3.0), 'weight_decay': 0},
    ])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs * num_batches,
        pct_start=0.1, anneal_strategy='cos'
    )

    phase_b = int(epochs * phase_split)
    print(f"   Faz A (1-{phase_b}): Plastisite OFF")
    print(f"   Faz B ({phase_b+1}-{epochs}): Plastisite ON")
    print("-" * 70)

    best_val = float('inf')
    history = []

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        enable_plast = epoch >= phase_b
        total_loss, n_batch = 0.0, 0
        perm = torch.randperm(N, device=device)

        for bi in range(num_batches):
            idx = perm[bi * batch_size:(bi + 1) * batch_size]
            x, y = train_x[idx], train_y[idx]

            model.reset_hebb()
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(x, enable_plasticity=enable_plast,
                               chunk_size=chunk_size)
                loss = F.cross_entropy(logits.reshape(-1, model.vocab_size),
                                       y.reshape(-1))

            if torch.isnan(loss):
                print(f"  ⚠️ NaN! ep={epoch+1} batch={bi+1}")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=tc.get('grad_clip', 1.0))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            n_batch += 1
            del logits, loss

        # Epoch sonu — Hebb izleri eval'den ÖNCE okunmalı
        # (evaluate() izleri sıfırlar, sonra okunursa H̄ hep 0 çıkar)
        train_loss = total_loss / max(n_batch, 1)
        hs = model.hebb_stats()
        val_off, val_on = evaluate_both(model, val_x, val_y)
        ppl_off = math.exp(min(val_off, 20))
        ppl_on = math.exp(min(val_on, 20))
        elapsed = time.time() - t0
        deep_h = sum(v for k, v in hs.items()
                     if int(k[1]) >= cfg['model'].get('num_fast', 2)) / 4
        phase = "B" if enable_plast else "A"

        print(f"  [{phase}] Ep {epoch+1:2d}/{epochs} │ "
              f"train:{train_loss:.3f} │ "
              f"val OFF:{val_off:.3f} (ppl {ppl_off:.1f}) "
              f"ON:{val_on:.3f} (ppl {ppl_on:.1f}) │ "
              f"H̄={deep_h:.4f} │ {elapsed:.0f}s")

        history.append({
            'epoch': epoch + 1, 'train_loss': train_loss,
            'val_loss_off': val_off, 'ppl_off': ppl_off,
            'val_loss_on': val_on, 'ppl_on': ppl_on,
            'hebb': deep_h,
        })
        # Her epoch'ta kaydet — oturum düşerse eğriler kaybolmasın
        save_history(history, os.path.join(save_dir, 'history.json'))

        # En iyi model seçimi: iki modun iyisi (model hangi modda
        # kullanılacaksa o modda iyi olmalı)
        val_loss = min(val_off, val_on)
        if val_loss < best_val:
            best_val = val_loss
            save_model(model, os.path.join(save_dir, 'best_model.pt'), cfg)

        se = cfg['save'].get('save_every', 2)
        if (epoch + 1) % se == 0:
            save_model(model, os.path.join(save_dir, f'model_ep{epoch+1}.pt'), cfg)

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print(f"\n  ✅ En iyi val: {best_val:.4f} (ppl={math.exp(min(best_val,20)):.1f})")
    save_model(model, os.path.join(save_dir, 'final_model.pt'), cfg)
    save_history(history, os.path.join(save_dir, 'history.json'))
    return best_val, history


def main():
    parser = argparse.ArgumentParser(description='MiniLiquidGPT Eğitim')
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    args = parser.parse_args()

    print("=" * 70)
    print("  MİNİ-LİKİT-GPT — Sıvı Nöral Ağ Dil Modeli")
    print("=" * 70)

    # Config
    cfg = load_config(args.config)
    print(f"📋 Config: {args.config}")

    # Tekrarlanabilirlik: seed (training.seed, varsayılan 42)
    seed = cfg.get('training', {}).get('seed', 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"🎲 Seed: {seed}")

    # Ortam
    device = setup_device()
    save_dir = setup_drive(cfg['save'].get('dir', './checkpoints/'))

    # Koşu metadata'sı — hangi kod/config/seed ile koşulduğu kayda geçsin
    meta = {'seed': seed, 'config_file': args.config, 'config': cfg,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}
    try:
        import subprocess
        meta['git_commit'] = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], capture_output=True,
            text=True, timeout=5).stdout.strip()
    except Exception:
        pass
    with open(os.path.join(save_dir, 'run_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    # Tokenizer
    tokenizer = TokenizerWrapper()
    print(f"📝 Tokenizer: {tokenizer.vocab_size} token")

    # Model — config'teki tüm model anahtarlarını doğrudan geçir
    # (use_attention, use_ffn, use_neuromod vb. bayraklar dahil)
    mc = cfg['model']
    valid_keys = set(inspect.signature(MiniLiquidGPT.__init__).parameters) - {'self'}
    unknown = set(mc) - valid_keys
    if unknown:
        print(f"⚠️  Config'te bilinmeyen model anahtarları (yok sayıldı): {sorted(unknown)}")
    model = MiniLiquidGPT(
        **{k: v for k, v in mc.items() if k in valid_keys}
    ).to(device)

    flags = sorted(k for k, v in mc.items() if k.startswith('use_') and v)
    if flags:
        print(f"🔧 Aktif bayraklar: {', '.join(flags)}")
    else:
        print("🔧 Aktif bayrak yok — çıplak Liquid ODE + Hebb (baseline)")

    p = model.count_params()
    print(f"🧠 Model: {p['total']/1e6:.1f}M param "
          f"({mc['num_fast']} hızlı + {mc['num_deep']} derin)")

    # Veri
    dc = cfg['data']
    train_x, train_y, val_x, val_y = load_data(
        tokenizer, max_tokens=dc['max_tokens'],
        seq_len=dc['seq_len'], device=device
    )

    # Eğit
    best_val, history = train(model, train_x, train_y, val_x, val_y,
                               cfg, save_dir)

    print(f"\n💾 Dosyalar: {save_dir}")
    for fn in sorted(os.listdir(save_dir)):
        fp = os.path.join(save_dir, fn)
        if os.path.isfile(fp):
            print(f"   {fn} ({os.path.getsize(fp)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
