"""
Yardımcı fonksiyonlar — kayıt/yükleme, ortam kurulumu
"""

import os
import gc
import json
import torch

from .model import MiniLiquidGPT


def setup_device():
    """GPU/CPU belirle ve belleği temizle."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gc.collect()
        torch.cuda.empty_cache()
        name = torch.cuda.get_device_name(0)
        try:
            mem = torch.cuda.mem_get_info()[1] / 1e9
            print(f"📱 GPU: {name} ({mem:.1f} GB)")
        except Exception:
            print(f"📱 GPU: {name}")
    else:
        device = torch.device('cpu')
        print("📱 CPU modu")
    return device


def setup_drive(local_fallback='./checkpoints/'):
    """Google Drive bağla (Colab'da) veya yerel dizin kullan."""
    save_dir = local_fallback
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        save_dir = '/content/drive/MyDrive/LiquidGPT_Models/'
        print(f"💾 Google Drive: {save_dir}")
    except Exception:
        print(f"💾 Yerel kayıt: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def save_model(model: MiniLiquidGPT, path: str, config: dict = None):
    """Model ağırlıklarını kaydet."""
    state = {
        'model_state_dict': model.state_dict(),
        'config': {k: v for k, v in (config or {}).items()
                   if not isinstance(v, torch.device)},
    }
    torch.save(state, path)
    size_mb = os.path.getsize(path) / 1e6
    print(f"💾 Kaydedildi: {path} ({size_mb:.1f} MB)")


def load_model(path: str, device='cpu') -> MiniLiquidGPT:
    """Kaydedilmiş modeli yükle."""
    state = torch.load(path, map_location=device, weights_only=True)
    cfg = state.get('config', {})
    model = MiniLiquidGPT(
        vocab_size=cfg.get('vocab_size', 50257),
        embed_dim=cfg.get('embed_dim', 256),
        num_fast=cfg.get('num_fast', 2),
        num_deep=cfg.get('num_deep', 2),
        fast_steps=cfg.get('fast_steps', 1),
        deep_steps=cfg.get('deep_steps', 3),
    ).to(device)
    model.load_state_dict(state['model_state_dict'])
    print(f"📂 Model yüklendi: {path}")
    return model


def save_history(history: list, path: str):
    """Eğitim geçmişini JSON olarak kaydet."""
    with open(path, 'w') as f:
        json.dump(history, f, indent=2)
