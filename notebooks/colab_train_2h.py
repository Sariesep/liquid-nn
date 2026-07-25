"""
LiquidNN v0.3.5 — 2 Saatlik Tam Eğitim (Google Colab)
=======================================================
GPU runtime seçin: Runtime → Change runtime type → T4 GPU
Eğitim ~2 saat sürer, checkpoint'lar Google Drive'a kaydedilir.
"""

# ╔═══════════════════════════════════════════════════════════════╗
# ║  CELL 1: Kurulum & Google Drive Bağlantısı                  ║
# ╚═══════════════════════════════════════════════════════════════╝

# Google Drive bağla — checkpoint'lar buraya kaydedilecek
from google.colab import drive
drive.mount('/content/drive')

import os
SAVE_DIR = '/content/drive/MyDrive/liquidnn_checkpoints'
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"💾 Checkpoint dizini: {SAVE_DIR}")

# Repo & bağımlılıklar
# !pip install tiktoken datasets -q
# !rm -rf liquid-nn
# !git clone https://github.com/Sariesep/liquid-nn.git
# %cd liquid-nn

# ╔═══════════════════════════════════════════════════════════════╗
# ║  CELL 2: Import & Cihaz                                     ║
# ╚═══════════════════════════════════════════════════════════════╝

import torch
import torch.nn.functional as F
import time
import math
import json
import sys, os

sys.path.insert(0, os.path.abspath('.'))
from liquidnn import MiniLiquidGPT
from liquidnn.tokenizer import TokenizerWrapper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Cihaz: {device}")
if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# ╔═══════════════════════════════════════════════════════════════╗
# ║  CELL 3: Dataset — WikiText veya Türkçe Veri                ║
# ╚═══════════════════════════════════════════════════════════════╝

tokenizer = TokenizerWrapper()

# ── Seçenek A: HuggingFace WikiText-103 (İngilizce, ~500MB) ───
# Küçük ama kaliteli, 2 saatte anlamlı sonuç verir
try:
    from datasets import load_dataset
    print("📥 WikiText-103 indiriliyor...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    raw_text = "\n".join([t for t in ds["text"] if len(t.strip()) > 50])
    # İlk ~10MB'ı al (2 saatlik eğitim için yeterli)
    MAX_CHARS = 10_000_000
    raw_text = raw_text[:MAX_CHARS]
    print(f"   Toplam metin: {len(raw_text):,} karakter")
except Exception as e:
    print(f"⚠️  HuggingFace dataset yüklenemedi: {e}")
    print("   Fallback: küçük dahili corpus kullanılıyor")
    raw_text = open("data/sample_corpus.txt", "r", encoding="utf-8").read() \
        if os.path.exists("data/sample_corpus.txt") else \
        ("Yapay zeka makinelerin insan benzeri davranış sergilemesidir. " * 5000)

# Tokenize
print("🔤 Tokenize ediliyor...")
all_tokens = tokenizer.encode(raw_text)
data = torch.tensor(all_tokens, dtype=torch.long, device=device)
print(f"   Token sayısı: {len(all_tokens):,}")
print(f"   Vocab boyutu: {tokenizer.vocab_size:,}")

# Eğitim/Validasyon bölme (%95 train, %5 val)
split = int(len(data) * 0.95)
train_data = data[:split]
val_data = data[split:]
print(f"   Train: {len(train_data):,} token, Val: {len(val_data):,} token")

# ╔═══════════════════════════════════════════════════════════════╗
# ║  CELL 4: Model Oluştur                                      ║
# ╚═══════════════════════════════════════════════════════════════╝

model = MiniLiquidGPT(
    vocab_size=tokenizer.vocab_size,
    embed_dim=256,
    num_fast=2,
    num_deep=2,
    fast_steps=1,
    deep_steps=3,
    dropout=0.1,
    max_seq=512,
    # ── Attention (şimdi eğitimde çalışıyor!) ──
    use_attention=True,
    attn_heads=4,
    attn_window=64,
    use_rope=True,
    use_flash=True,
    # ── FFN (v0.3.5 yeni!) ──
    use_ffn=True,
    ffn_mult=4.0,
    # ── v0.3.4 Özellikleri ──
    use_neuromod=True,
    use_homeostasis=True,
    homeostasis_target=0.5,
    use_dual_hebb=True,
    use_consolidation=True,
    consolidation_strength=1.0,
    # ── Diğer ──
    use_rmsnorm=True,
    tau_gate=True,
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n🧠 Model: MiniLiquidGPT v0.3.5")
print(f"   Parametreler: {total_params:,} ({total_params/1e6:.2f}M)")
print(f"   Katmanlar: {model.num_layers} (2 fast + 2 deep)")
print(f"   Özellikler: Attention ✅  FFN ✅  Neuromod ✅  DualHebb ✅")

# ╔═══════════════════════════════════════════════════════════════╗
# ║  CELL 5: Eğitim Ayarları                                    ║
# ╚═══════════════════════════════════════════════════════════════╝

# Hiperparametreler
SEQ_LEN = 128
BATCH_SIZE = 8
LR = 5e-4
CHUNK_SIZE = 32
WARMUP_STEPS = 200
MAX_HOURS = 2.0               # Maksimum eğitim süresi
CHECKPOINT_EVERY = 500        # Her N adımda checkpoint kaydet
VAL_EVERY = 250               # Her N adımda validasyon yap
LOG_EVERY = 50                # Her N adımda log bas
GRAD_ACCUM_STEPS = 4          # Gradient accumulation (effective batch = 32)

# Toplam adım tahmini
tokens_per_step = BATCH_SIZE * SEQ_LEN
total_train_tokens = len(train_data)
steps_per_epoch = total_train_tokens // tokens_per_step
estimated_steps = int(MAX_HOURS * 3600 / 0.7)  # ~0.7s/step tahmini
MAX_STEPS = min(estimated_steps, steps_per_epoch * 10)
print(f"\n⚙️  Eğitim Ayarları:")
print(f"   Sekans uzunluğu: {SEQ_LEN}")
print(f"   Batch boyutu: {BATCH_SIZE}")
print(f"   Adım başına token: {tokens_per_step:,}")
print(f"   Epoch başına adım: {steps_per_epoch:,}")
print(f"   Tahmini max adım: ~{MAX_STEPS:,}")
print(f"   Checkpoint: her {CHECKPOINT_EVERY} adım → Google Drive")

# Optimizer + Scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

def lr_schedule(step):
    """Warmup + Cosine decay."""
    if step < WARMUP_STEPS:
        return step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

def make_batch(data, seq_len, batch_size):
    """Rastgele batch üret."""
    max_start = len(data) - seq_len - 1
    starts = torch.randint(0, max(1, max_start), (batch_size,))
    x = torch.stack([data[s:s+seq_len] for s in starts])
    y = torch.stack([data[s+1:s+seq_len+1] for s in starts])
    return x, y

@torch.no_grad()
def evaluate(model, val_data, seq_len, n_batches=10,
             enable_plasticity=False):
    """Validasyon loss hesapla (tek modda).

    Her batch öncesi Hebb izleri sıfırlanır — eğitimden sızan izler
    val loss'u son eğitim batch'ine bağımlı kılıyordu.
    """
    model.eval()
    total_loss = 0
    for _ in range(n_batches):
        x, y = make_batch(val_data, seq_len, 4)
        model.reset_hebb()
        logits = model(x, enable_plasticity=enable_plasticity,
                       chunk_size=seq_len)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item()
    model.reset_hebb()
    model.train()
    return total_loss / n_batches


def evaluate_both(model, val_data, seq_len, n_batches=10):
    """Val'i iki modda ölç: plastisite OFF (statik) ve ON (plastik).

    Projenin ana tezi çıkarım anındaki plastisite olduğundan
    yalnızca OFF ölçmek tezi görünmez kılar."""
    val_off = evaluate(model, val_data, seq_len, n_batches,
                       enable_plasticity=False)
    val_on = evaluate(model, val_data, seq_len, n_batches,
                      enable_plasticity=True)
    return val_off, val_on

def save_checkpoint(model, optimizer, step, loss, path):
    """Checkpoint'ı Google Drive'a kaydet."""
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'model_config': {
            'vocab_size': model.vocab_size,
            'embed_dim': model.embed_dim,
            'num_layers': model.num_layers,
        }
    }, path)

# ╔═══════════════════════════════════════════════════════════════╗
# ║  CELL 6: EĞİTİM DÖNGÜSÜ (2 saat)                          ║
# ╚═══════════════════════════════════════════════════════════════╝

print("\n" + "═" * 65)
print("  EĞİTİM BAŞLIYOR (max 2 saat)")
print("═" * 65)

model.train()
start_time = time.time()
best_val_loss = float('inf')
train_losses = []
val_losses = []
step = 0

# Mixed Precision (AMP)
use_amp = device.type == 'cuda'
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
if use_amp:
    print("  ⚡ Mixed Precision (AMP) aktif")
print(f"  📦 Gradient Accumulation: {GRAD_ACCUM_STEPS}x "
      f"(effective batch = {BATCH_SIZE * GRAD_ACCUM_STEPS})")

optimizer.zero_grad()

try:
    while True:
        step += 1
        elapsed_hours = (time.time() - start_time) / 3600

        # Zaman limiti kontrolü
        if elapsed_hours >= MAX_HOURS:
            print(f"\n⏰ Zaman limiti ({MAX_HOURS} saat) doldu.")
            break
        if step > MAX_STEPS:
            print(f"\n📊 Maksimum adım ({MAX_STEPS}) ulaşıldı.")
            break

        # Forward + Backward (Gradient Accumulation)
        accum_loss = 0.0
        for accum_step in range(GRAD_ACCUM_STEPS):
            x_batch, y_batch = make_batch(train_data, SEQ_LEN, BATCH_SIZE)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x_batch, enable_plasticity=True,
                               chunk_size=CHUNK_SIZE)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y_batch.reshape(-1)
                ) / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()
            accum_loss += loss.item() * GRAD_ACCUM_STEPS

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        train_losses.append(accum_loss / GRAD_ACCUM_STEPS)

        # ── Log ───────────────────────────────────────────────
        if step % LOG_EVERY == 0:
            avg_loss = sum(train_losses[-LOG_EVERY:]) / LOG_EVERY
            ppl = math.exp(min(avg_loss, 20))
            lr_now = scheduler.get_last_lr()[0]
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed
            eta_hours = (MAX_HOURS - elapsed_hours)

            stats = model.hebb_stats()
            hebb_max = max(stats.values()) if stats else 0

            print(f"  Step {step:6d} │ Loss: {avg_loss:.4f} │ "
                  f"PPL: {ppl:8.1f} │ LR: {lr_now:.2e} │ "
                  f"Hebb: {hebb_max:.2f} │ "
                  f"{steps_per_sec:.1f} it/s │ ETA: {eta_hours:.1f}h")

        # ── Validasyon (iki modda: plastisite OFF ve ON) ──────
        if step % VAL_EVERY == 0:
            val_off, val_on = evaluate_both(model, val_data, SEQ_LEN)
            val_loss = min(val_off, val_on)
            ppl_off = math.exp(min(val_off, 20))
            ppl_on = math.exp(min(val_on, 20))
            val_losses.append((step, val_off, val_on))
            improved = "🏆 BEST" if val_loss < best_val_loss else ""
            print(f"  {'─' * 45}")
            print(f"  📋 VAL Step {step}: "
                  f"OFF={val_off:.4f} (PPL {ppl_off:.1f}) │ "
                  f"ON={val_on:.4f} (PPL {ppl_on:.1f}) {improved}")
            print(f"  {'─' * 45}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(SAVE_DIR, 'liquidnn_best.pt')
                save_checkpoint(model, optimizer, step, val_loss, best_path)
                print(f"  💾 Best model → {best_path}")

        # ── Periyodik Checkpoint ──────────────────────────────
        if step % CHECKPOINT_EVERY == 0:
            ckpt_path = os.path.join(SAVE_DIR, f'liquidnn_step{step}.pt')
            save_checkpoint(model, optimizer, step, loss.item(), ckpt_path)
            print(f"  💾 Checkpoint → {ckpt_path}")

except KeyboardInterrupt:
    print("\n\n⚠️  Eğitim kullanıcı tarafından durduruldu!")

# ── Son checkpoint kaydet ──────────────────────────────────────
total_time = time.time() - start_time
final_path = os.path.join(SAVE_DIR, 'liquidnn_final.pt')
save_checkpoint(model, optimizer, step, train_losses[-1], final_path)

print("\n" + "═" * 65)
print(f"  EĞİTİM TAMAMLANDI")
print(f"  Süre: {total_time/3600:.2f} saat ({total_time:.0f} saniye)")
print(f"  Toplam adım: {step:,}")
print(f"  Son Train Loss: {train_losses[-1]:.4f}")
print(f"  En İyi Val Loss: {best_val_loss:.4f}")
print(f"  Son model: {final_path}")
print("═" * 65)

# Eğitim log'unu kaydet
log_path = os.path.join(SAVE_DIR, 'training_log.json')
with open(log_path, 'w') as f:
    json.dump({
        'total_steps': step,
        'total_time_sec': total_time,
        'final_train_loss': train_losses[-1],
        'best_val_loss': best_val_loss,
        'train_losses_sampled': train_losses[::50],  # her 50. loss
        'val_losses': val_losses,
        'config': {
            'embed_dim': 256, 'seq_len': SEQ_LEN,
            'batch_size': BATCH_SIZE, 'lr': LR,
        }
    }, f, indent=2)
print(f"📄 Eğitim log'u: {log_path}")

# ╔═══════════════════════════════════════════════════════════════╗
# ║  CELL 7: Metin Üretimi (Eğitilmiş Model)                   ║
# ╚═══════════════════════════════════════════════════════════════╝

PROMPTS = [
    "The history of",
    "Neural networks are",
    "In the field of artificial intelligence",
    "The most important",
    "Scientists have discovered",
]

print("\n" + "═" * 65)
print("  METİN ÜRETİMİ (eğitilmiş model)")
print("═" * 65)

model.eval()
for prompt_text in PROMPTS:
    prompt_ids = torch.tensor(
        tokenizer.encode(prompt_text), dtype=torch.long, device=device
    )
    with torch.no_grad():
        out_ids = model.generate(
            prompt_ids, max_new=60,
            temperature=0.8, top_k=40, top_p=0.9,
            enable_plasticity=True
        )
    generated = tokenizer.decode(out_ids[0].tolist())
    print(f"\n  💬 \"{prompt_text}\" →")
    print(f"     {generated[:200]}")

# ╔═══════════════════════════════════════════════════════════════╗
# ║  CELL 8: Modeli Daha Sonra Yüklemek İçin                    ║
# ╚═══════════════════════════════════════════════════════════════╝

print("\n" + "═" * 65)
print("  MODEL YÜKLEME KODU (ileride kullanmak için kopyalayın)")
print("═" * 65)
print("""
# Modeli Google Drive'dan yüklemek için:

from liquidnn import MiniLiquidGPT
import torch

model = MiniLiquidGPT(
    vocab_size=50257, embed_dim=256,
    num_fast=2, num_deep=2,
    fast_steps=1, deep_steps=3,
    use_attention=True, attn_heads=4,
    use_ffn=True, ffn_mult=4.0,
    use_neuromod=True, use_homeostasis=True,
    use_dual_hebb=True, use_consolidation=True,
    use_rmsnorm=True, tau_gate=True,
)

ckpt = torch.load('/content/drive/MyDrive/liquidnn_checkpoints/liquidnn_best.pt')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"Model yüklendi! Step: {ckpt['step']}, Loss: {ckpt['loss']:.4f}")
""")

print("\n✅ Tüm checkpoint'lar Google Drive'da güvende!")
print(f"   📂 {SAVE_DIR}/")
print(f"      ├── liquidnn_best.pt      (en iyi val loss)")
print(f"      ├── liquidnn_final.pt     (son durum)")
print(f"      ├── liquidnn_step*.pt     (periyodik)")
print(f"      └── training_log.json     (eğitim metrikleri)")
