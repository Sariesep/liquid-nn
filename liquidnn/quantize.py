"""
INT8 Quantization Helpers — Enerji Verimli İnference

Dynamic INT8 quantization: ağırlıkları INT8'e dönüştürür,
model boyutunu ~%50 küçültür ve matris çarpımını hızlandırır.

Not: PlasticSynapse'ın Hebb matrisi runtime'da güncellenmesi
gerektiğinden, sadece sabit ağırlıklar quantize edilir.
"""

import time
import torch
import torch.nn as nn


def quantize_model(model: nn.Module) -> nn.Module:
    """
    Modele dynamic INT8 quantization uygula.

    Sadece Linear katmanları quantize eder.
    PlasticSynapse'ın custom forward'u korunur.

    Args:
        model: Quantize edilecek model

    Returns:
        Quantize edilmiş model (yeni kopya)
    """
    model_q = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},               # Sadece Linear katmanları
        dtype=torch.qint8
    )
    return model_q


def model_size_mb(model: nn.Module) -> float:
    """Model boyutunu MB cinsinden döndür."""
    total = 0
    for p in model.parameters():
        total += p.nelement() * p.element_size()
    for b in model.buffers():
        if b is not None:
            total += b.nelement() * b.element_size()
    return total / 1e6


def benchmark_model(model: nn.Module, input_ids: torch.Tensor,
                    n_runs: int = 10) -> dict:
    """
    Model hız ve bellek benchmark'ı.

    Args:
        model: Benchmark edilecek model
        input_ids: [B, T] test girdisi
        n_runs: Tekrar sayısı

    Returns:
        {'avg_time_ms': float, 'size_mb': float, 'tokens_per_sec': float}
    """
    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    B, T = input_ids.shape

    # Warmup
    with torch.no_grad():
        for _ in range(2):
            model(input_ids)

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            model.init_hidden(B, device)
            model.reset_hebb()
            start = time.perf_counter()
            model(input_ids, enable_plasticity=False)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

    avg_time = sum(times) / len(times)
    total_tokens = B * T
    tokens_per_sec = total_tokens / (avg_time / 1000)

    return {
        'avg_time_ms': round(avg_time, 2),
        'size_mb': round(model_size_mb(model), 2),
        'tokens_per_sec': round(tokens_per_sec, 1),
    }
