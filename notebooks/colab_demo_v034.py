"""
LiquidNN v0.3.4 — Google Colab Hızlı Eğitim Demo
==================================================
Bu script'i Colab'a kopyalayıp çalıştırın.
GPU runtime seçmeyi unutmayın: Runtime → Change runtime type → T4 GPU

Tahmini süre: ~3-5 dakika (T4 GPU ile)
"""

# ╔═══════════════════════════════════════════════════════════════╗
# ║  CELL 1: Kurulum                                            ║
# ╚═══════════════════════════════════════════════════════════════╝

# !pip install tiktoken -q
# !git clone https://github.com/Sariesep/liquid-nn.git
# %cd liquid-nn

# ╔═══════════════════════════════════════════════════════════════╗
# ║  CELL 2: Import & Cihaz                                     ║
# ╚═══════════════════════════════════════════════════════════════╝

import torch
import torch.nn.functional as F
import time
import math

# Eğer Colab'da "liquid-nn" dizinine cd yaptıysanız:
import sys, os
sys.path.insert(0, os.path.abspath('.'))

from liquidnn import MiniLiquidGPT, get_tokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Cihaz: {device}")
if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# ╔═══════════════════════════════════════════════════════════════╗
# ║  CELL 3: Eğitim Verisi (Küçük Türkçe Corpus)               ║
# ╚═══════════════════════════════════════════════════════════════╝

# Kısa demo metni — gerçek eğitimde daha büyük corpus kullanılmalı
CORPUS = """
Yapay zeka, makinelerin insan benzeri zeka sergilemesini sağlayan bir bilim dalıdır.
Derin öğrenme, yapay sinir ağlarının çok katmanlı yapılarla karmaşık kalıpları öğrenmesini sağlar.
Nöral ağlar, beyindeki nöronların çalışma prensibinden esinlenerek geliştirilmiştir.
Sıvı nöral ağlar, zaman değişkenli diferansiyel denklemlerle sürekli adaptasyon sağlar.
Hebbian öğrenme kuralı, birlikte ateşleyen nöronların bağlantılarının güçlendiğini söyler.
Transformer mimarisi, dikkat mekanizması ile uzun menzilli bağımlılıkları yakalar.
Öğrenme hızı, modelin ağırlıklarını ne kadar hızlı güncellediğini belirler.
Gradient iniş yöntemi, kayıp fonksiyonunu minimize etmek için kullanılır.
Aşırı öğrenme, modelin eğitim verisini ezberlemesi ve genelleme yapamaması durumudur.
Düzenlileştirme teknikleri, modelin genelleme kapasitesini artırmak için uygulanır.
Dikkat mekanizması, girdi sekansının farklı bölümlerine farklı ağırlıklar verir.
Geri yayılım algoritması, hata sinyalini ağ boyunca geriye doğru yayarak öğrenmeyi sağlar.
Batch normalizasyon, eğitim sürecini hızlandıran ve kararlı hale getiren bir tekniktir.
Konvolüsyonel ağlar, görüntü tanıma ve bilgisayarlı görü alanında devrim yaratmıştır.
Doğal dil işleme, bilgisayarların insan dilini anlamasını ve üretmesini hedefler.
Pekiştirmeli öğrenme, bir ajanın deneme yanılma yoluyla en iyi stratejiyi öğrenmesidir.
Transfer öğrenme, bir görevde öğrenilen bilginin başka bir göreve aktarılmasıdır.
Üretici çekişmeli ağlar, gerçekçi veri üretmek için iki ağın rekabet etmesini kullanır.
Otomatik kodlayıcılar, veriyi sıkıştırıp yeniden oluşturarak özellik çıkarmayı öğrenir.
Metin üretimi, dil modellerinin olasılıksal dağılımlardan yeni metinler oluşturmasıdır.
""".strip()

# Tokenize
tokenizer = get_tokenizer()
tokens = tokenizer.encode(CORPUS)
data = torch.tensor(tokens, dtype=torch.long, device=device)
print(f"📝 Corpus: {len(CORPUS)} karakter → {len(tokens)} token")

# ╔═══════════════════════════════════════════════════════════════╗
# ║  CELL 4: Model Oluştur (v0.3.4 tüm özellikler açık)        ║
# ╚═══════════════════════════════════════════════════════════════╝

# NOT: use_attention=False — Attention modülünün KV cache'i
# eğitim sırasında autograd in-place hatası verebilir.
# Inference'ta (generate) attention güvenle kullanılabilir.
# v0.3.4'ün asıl yenilikleri (neuromod, homeostasis, dual hebb,
# consolidation) ODE + plastisite katmanlarındadır.

model = MiniLiquidGPT(
    vocab_size=tokenizer.vocab_size,
    embed_dim=128,
    num_fast=2,
    num_deep=2,
    fast_steps=1,
    deep_steps=3,
    dropout=0.1,
    max_seq=512,
    # ── Attention (eğitimde KAPALI — KV cache autograd sorunu) ──
    use_attention=False,
    # ── MoE ──
    use_moe=False,          # küçük modelde MoE gereksiz
    # ── v0.3.4 ──
    use_neuromod=True,       # ✅ Nöromodülasyon
    use_homeostasis=True,    # ✅ Homeostatik Plastisite
    homeostasis_target=0.5,
    use_dual_hebb=True,      # ✅ Çift Hızlı Hebb
    use_consolidation=True,  # ✅ Sinaptik Konsolidasyon
    consolidation_strength=1.0,
    # ── Diğer ──
    use_rmsnorm=True,
    tau_gate=True,
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n🧠 Model: MiniLiquidGPT v0.3.4")
print(f"   Parametreler: {total_params:,} ({total_params/1e6:.2f}M)")
print(f"   Katmanlar: {model.num_layers} (2 fast + 2 deep)")
print(f"   Özellikler: Neuromod ✅  Homeostasis ✅  DualHebb ✅  Consolidation ✅")

# ╔═══════════════════════════════════════════════════════════════╗
# ║  CELL 5: Eğitim Döngüsü                                    ║
# ╚═══════════════════════════════════════════════════════════════╝

# Hiperparametreler
EPOCHS = 150
SEQ_LEN = 64
BATCH_SIZE = 4
LR = 3e-4
CHUNK_SIZE = 16

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

def make_batch(data, seq_len, batch_size):
    """Rastgele batch üret."""
    max_start = len(data) - seq_len - 1
    if max_start <= 0:
        starts = [0] * batch_size
    else:
        starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[s:s+seq_len] for s in starts])
    y = torch.stack([data[s+1:s+seq_len+1] for s in starts])
    return x, y

print("\n" + "═" * 60)
print("  EĞİTİM BAŞLIYOR")
print("═" * 60)

model.train()
start_time = time.time()
losses = []

for epoch in range(1, EPOCHS + 1):
    x_batch, y_batch = make_batch(data, SEQ_LEN, BATCH_SIZE)

    logits = model(x_batch, enable_plasticity=True, chunk_size=CHUNK_SIZE)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y_batch.reshape(-1)
    )

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    losses.append(loss.item())

    if epoch % 10 == 0 or epoch == 1:
        elapsed = time.time() - start_time
        avg_loss = sum(losses[-10:]) / len(losses[-10:])
        ppl = math.exp(min(avg_loss, 20))  # overflow koruması
        lr_now = scheduler.get_last_lr()[0]

        # Hebb istatistikleri
        stats = model.hebb_stats()
        hebb_summary = ""
        for k, v in stats.items():
            if v > 0:
                hebb_summary += f"{k}={v:.3f} "

        print(f"  Epoch {epoch:4d}/{EPOCHS} │ "
              f"Loss: {avg_loss:.4f} │ PPL: {ppl:8.1f} │ "
              f"LR: {lr_now:.2e} │ "
              f"⏱ {elapsed:.0f}s")
        if hebb_summary:
            print(f"           │ Hebb: {hebb_summary.strip()}")

total_time = time.time() - start_time
print("═" * 60)
print(f"  EĞİTİM TAMAMLANDI — {total_time:.1f} saniye")
print(f"  Son Loss: {losses[-1]:.4f} │ PPL: {math.exp(min(losses[-1], 20)):.1f}")
print("═" * 60)

# ╔═══════════════════════════════════════════════════════════════╗
# ║  CELL 6: Metin Üretimi                                      ║
# ╚═══════════════════════════════════════════════════════════════╝

PROMPTS = [
    "Yapay zeka",
    "Nöral ağlar",
    "Derin öğrenme",
    "Sıvı nöral",
]

print("\n" + "═" * 60)
print("  METİN ÜRETİMİ")
print("═" * 60)

model.eval()
for prompt_text in PROMPTS:
    prompt_ids = torch.tensor(
        tokenizer.encode(prompt_text), dtype=torch.long, device=device
    )
    with torch.no_grad():
        out_ids = model.generate(
            prompt_ids, max_new=40,
            temperature=0.8, top_k=30,
            enable_plasticity=True
        )
    generated = tokenizer.decode(out_ids[0].tolist())
    print(f"\n  💬 \"{prompt_text}\" →")
    print(f"     {generated}")

# ╔═══════════════════════════════════════════════════════════════╗
# ║  CELL 7: Model Diagnostik (v0.3.4 Özellikleri)              ║
# ╚═══════════════════════════════════════════════════════════════╝

print("\n" + "═" * 60)
print("  v0.3.4 DİAGNOSTİK")
print("═" * 60)

# Hebb istatistikleri
stats = model.hebb_stats()
print("\n  📊 Hebb Normları:")
for key, val in stats.items():
    bar = "█" * int(min(val * 50, 40))
    print(f"     {key:15s}: {val:.4f}  {bar}")

# Parametre sayımı
pcnt = model.count_params()
print(f"\n  📐 Parametre Dağılımı:")
print(f"     Toplam:    {pcnt['total']:>10,}")
print(f"     Embedding: {pcnt['embed']:>10,}")
print(f"     ODE Cells: {pcnt['cells']:>10,}")
print(f"     Diğer:     {pcnt['other']:>10,}")

print("\n✅ Demo tamamlandı!")
