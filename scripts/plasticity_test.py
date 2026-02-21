#!/usr/bin/env python3
"""
MiniLiquidGPT Plastisite Testi (ZEPHYR / Bloop)

4 yollu karşılaştırma:
1. Baz çizgi:    Kural yok
2. Plast OFF:    Kural var, Hebb güncellenmiyor
3. Plast ON:     Kural var, Hebb güncelleniyor
4. Kalıcılık:    Hidden sıfır, Hebb korundu

Kullanım:
    python scripts/plasticity_test.py --checkpoint checkpoints/best_model.pt
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from liquidnn import TokenizerWrapper
from liquidnn.utils import load_model


def show_top5(probs, tokenizer):
    top = torch.topk(probs, 5)
    words = [tokenizer.decode([i]) for i in top.indices.tolist()]
    vals = [f"{p:.4f}" for p in top.values.tolist()]
    print(f"     Top-5: {list(zip(words, vals))}")


def run_test(model, tokenizer, device):
    """4 yollu plastisite testi."""
    print("\n" + "=" * 70)
    print("  🧪 PLASTİSİTE TESTİ — ZEPHYR")
    print("=" * 70)

    model.eval()

    rules = (
        "Important definition: ZEPHYR means sunset. "
        "ZEPHYR is the word for sunset. "
        "When someone says ZEPHYR they mean sunset. "
        "Remember: ZEPHYR equals sunset.\n"
    )
    test = "ZEPHYR is the"

    # Hedef tokenlar
    targets = {}
    for w in ["sunset", " sunset", "sun", " sun", " golden", " evening"]:
        ids = tokenizer.encode(w)
        if ids:
            targets[w.strip()] = ids[0]
    print(f"\n  Hedefler: {list(targets.keys())}")

    rule_ids = tokenizer.encode(rules)
    test_ids = tokenizer.encode(test)

    def get_probs(plast_rules, plast_test, show_rules=True, reset_h=False):
        model.reset_hebb()
        model.init_hidden(1, device)

        with torch.no_grad():
            if show_rules:
                for i, tid in enumerate(rule_ids):
                    model.forward_token(
                        torch.tensor([tid], device=device), i, plast_rules)

            if reset_h:
                model.init_hidden(1, device)

            off = len(rule_ids) if show_rules and not reset_h else 0
            for i, tid in enumerate(test_ids):
                logits = model.forward_token(
                    torch.tensor([tid], device=device), off + i, plast_test)

        return F.softmax(logits.squeeze(0), dim=-1)

    results = {}

    # Test 1: Baz çizgi
    print("\n  ── 1. Baz çizgi (kural yok) ──")
    p1 = get_probs(False, False, show_rules=False)
    avg1 = sum(p1[t].item() for t in targets.values()) / len(targets)
    results['baseline'] = avg1
    for w, t in targets.items():
        print(f"     '{w}': {p1[t].item():.5f}")
    show_top5(p1, tokenizer)

    # Test 2: Plast OFF
    print("\n  ── 2. Kural + Plastisite OFF ──")
    p2 = get_probs(False, False, show_rules=True)
    avg2 = sum(p2[t].item() for t in targets.values()) / len(targets)
    results['rules_plast_off'] = avg2
    for w, t in targets.items():
        print(f"     '{w}': {p2[t].item():.5f}")
    show_top5(p2, tokenizer)

    # Test 3: Plast ON
    print("\n  ── 3. Kural + Plastisite ON ──")
    p3 = get_probs(True, True, show_rules=True)
    avg3 = sum(p3[t].item() for t in targets.values()) / len(targets)
    results['rules_plast_on'] = avg3
    for w, t in targets.items():
        print(f"     '{w}': {p3[t].item():.5f}")
    hs = model.hebb_stats()
    print(f"     Hebb: {sum(hs.values())/len(hs):.5f}")
    show_top5(p3, tokenizer)

    # Test 4: Kalıcılık
    print("\n  ── 4. Kalıcılık (hidden sıfır, Hebb korundu) ──")
    # Önce kuralları plast ON ile göster
    model.reset_hebb()
    model.init_hidden(1, device)
    with torch.no_grad():
        for i, tid in enumerate(rule_ids):
            model.forward_token(
                torch.tensor([tid], device=device), i, True)
    # Hidden sıfır, Hebb kalsın
    model.init_hidden(1, device)
    with torch.no_grad():
        for i, tid in enumerate(test_ids):
            logits = model.forward_token(
                torch.tensor([tid], device=device), i, True)
    p4 = F.softmax(logits.squeeze(0), dim=-1)
    avg4 = sum(p4[t].item() for t in targets.values()) / len(targets)
    results['persistence'] = avg4
    for w, t in targets.items():
        print(f"     '{w}': {p4[t].item():.5f}")
    show_top5(p4, tokenizer)

    # Sonuç
    print("\n" + "─" * 55)
    print("  📊 SONUÇ")
    print("─" * 55)
    print(f"  1. Baz çizgi:         {results['baseline']:.5f}")
    print(f"  2. Kural + Plast OFF: {results['rules_plast_off']:.5f}")
    print(f"  3. Kural + Plast ON:  {results['rules_plast_on']:.5f}")
    print(f"  4. Kalıcılık:         {results['persistence']:.5f}")

    if avg3 > avg2 * 1.05:
        boost = (avg3 / max(avg2, 1e-10) - 1) * 100
        print(f"\n  🏆 PLASTİSİTE: +{boost:.1f}%")
    if avg4 > avg1 * 1.05:
        boost = (avg4 / max(avg1, 1e-10) - 1) * 100
        print(f"  🏆 KALICILIK: +{boost:.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description='Plastisite Testi')
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.checkpoint, device=device)
    tokenizer = TokenizerWrapper()
    run_test(model, tokenizer, device)


if __name__ == "__main__":
    main()
