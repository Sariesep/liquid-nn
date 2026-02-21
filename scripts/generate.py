#!/usr/bin/env python3
"""
MiniLiquidGPT Metin Üretimi

Kullanım:
    python scripts/generate.py --checkpoint checkpoints/best_model.pt \
                               --prompt "The meaning of life"
    python scripts/generate.py --checkpoint checkpoints/best_model.pt \
                               --interactive
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from liquidnn import TokenizerWrapper
from liquidnn.utils import load_model


def generate_text(model, tokenizer, prompt, device,
                  max_new=80, temperature=0.8, top_k=40,
                  enable_plasticity=True):
    """Tek prompt'tan metin üret."""
    ids = tokenizer.encode_tensor(prompt, device=device)
    out = model.generate(ids, max_new=max_new, temperature=temperature,
                         top_k=top_k, enable_plasticity=enable_plasticity)
    return tokenizer.decode(out[0])


def main():
    parser = argparse.ArgumentParser(description='MiniLiquidGPT Metin Üretimi')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--max_new', type=int, default=80)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--no_plasticity', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.checkpoint, device=device)
    tokenizer = TokenizerWrapper()
    plast = not args.no_plasticity

    if args.interactive:
        print("🧠 MiniLiquidGPT Interactive (çıkmak için 'quit')")
        print(f"   Plastisite: {'ON' if plast else 'OFF'}")
        print("-" * 50)
        while True:
            try:
                prompt = input("\n> ").strip()
                if prompt.lower() in ('quit', 'exit', 'q'):
                    break
                if prompt == '/reset':
                    model.reset_hebb()
                    model.init_hidden(1, device)
                    print("  🔄 Hebb + hidden sıfırlandı")
                    continue
                if prompt == '/hebb':
                    for k, v in model.hebb_stats().items():
                        print(f"  {k}: {v:.5f}")
                    continue
                text = generate_text(model, tokenizer, prompt, device,
                                     max_new=args.max_new,
                                     temperature=args.temperature,
                                     top_k=args.top_k,
                                     enable_plasticity=plast)
                print(text)
            except KeyboardInterrupt:
                break
    elif args.prompt:
        text = generate_text(model, tokenizer, args.prompt, device,
                             max_new=args.max_new,
                             temperature=args.temperature,
                             top_k=args.top_k,
                             enable_plasticity=plast)
        print(text)
    else:
        # Varsayılan demolar
        prompts = [
            "The meaning of life is",
            "Once upon a time, in a",
            "Scientists have discovered that",
        ]
        for p in prompts:
            model.reset_hebb()
            text = generate_text(model, tokenizer, p, device,
                                 enable_plasticity=plast)
            print(f"\n[{p}]")
            print(f"→ {text[len(p):][:200]}")


if __name__ == "__main__":
    main()
