"""
analyze.py
──────────
Demo analysis: load a trained model, generate one synthetic ECG per class,
classify each, and print a results table.

Usage
─────
    python analyze.py [--model cnn|transformer]

Prerequisites
─────────────
    python generate_synthetic_ecg.py   # create data
    python train.py                    # train models (saves best_cnn.pt / best_transformer.pt)
"""

import os
import sys
import argparse
import numpy as np
import torch

from config  import (
    NUM_CLASSES, CLASS_NAMES, CNN_WEIGHTS_PATH, TRANSFORMER_WEIGHTS_PATH,
    NUM_LEADS, SIGNAL_LENGTH, RANDOM_SEED,
)
from dataset import per_lead_normalise
from models  import ECG_CNN, ECG_Transformer
from generate_synthetic_ecg import generate_ecg


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_name: str, device: torch.device, script_dir: str) -> torch.nn.Module:
    if model_name == "cnn":
        model     = ECG_CNN()
        ckpt_path = os.path.join(script_dir, CNN_WEIGHTS_PATH)
    else:
        model     = ECG_Transformer()
        ckpt_path = os.path.join(script_dir, TRANSFORMER_WEIGHTS_PATH)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Weights not found: {ckpt_path}\n"
            "Run  python train.py  first."
        )

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="ECG demo analysis")
    parser.add_argument(
        "--model", type=str, default="cnn",
        choices=["cnn", "transformer"],
        help="Which trained model to use (default: cnn)",
    )
    parser.add_argument(
        "--samples_per_class", type=int, default=5,
        help="How many ECGs to generate per class (default: 5)",
    )
    return parser.parse_args()


def main():
    args       = parse_args()
    device     = get_device()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"Device    : {device}")
    print(f"Model     : {args.model.upper()}")
    print(f"Samples   : {args.samples_per_class} per class  ({args.samples_per_class * NUM_CLASSES} total)\n")

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model(args.model, device, script_dir)
    print(f"Loaded weights  ({sum(p.numel() for p in model.parameters()):,} params)\n")

    # ── Generate synthetic ECGs ───────────────────────────────────────────────
    rng       = np.random.default_rng(RANDOM_SEED + 999)
    signals   = []
    true_lbls = []

    for cls_idx in range(NUM_CLASSES):
        for _ in range(args.samples_per_class):
            ecg = generate_ecg(cls_idx)         # (12, 5000)
            signals.append(ecg)
            true_lbls.append(cls_idx)

    signals_np = np.stack(signals, axis=0)       # (N, 12, 5000)
    signals_np = per_lead_normalise(signals_np)  # z-score per lead

    # ── Classify ──────────────────────────────────────────────────────────────
    signals_t = torch.from_numpy(signals_np).to(device)
    with torch.no_grad():
        logits     = model(signals_t)
        probs      = torch.softmax(logits, dim=-1).cpu().numpy()
        pred_lbls  = probs.argmax(axis=-1)

    # ── Print results table ───────────────────────────────────────────────────
    col_w = 20
    header = (
        f"{'True Label':<{col_w}} "
        f"{'Predicted':<{col_w}} "
        f"{'Confidence':>12}  "
        f"{'Match':>6}"
    )
    print(header)
    print("-" * len(header))

    n_correct = 0
    for true_idx, pred_idx, prob in zip(true_lbls, pred_lbls, probs):
        true_name   = CLASS_NAMES[true_idx]
        pred_name   = CLASS_NAMES[pred_idx]
        confidence  = prob[pred_idx]
        match       = "✓" if true_idx == pred_idx else "✗"
        n_correct  += int(true_idx == pred_idx)
        print(
            f"{true_name:<{col_w}} "
            f"{pred_name:<{col_w}} "
            f"{confidence:>12.4f}  "
            f"{match:>6}"
        )

    total    = len(true_lbls)
    accuracy = n_correct / total
    print(f"\n{'─'*len(header)}")
    print(f"Overall accuracy: {n_correct}/{total} = {accuracy:.4f}")

    # ── Per-class accuracy ────────────────────────────────────────────────────
    print("\nPer-class accuracy:")
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask    = [t == cls_idx for t in true_lbls]
        correct = sum(
            1 for t, p, m in zip(true_lbls, pred_lbls, mask)
            if m and t == p
        )
        total_c = sum(mask)
        acc_c   = correct / total_c if total_c > 0 else 0.0
        print(f"  {cls_name:<20}: {correct}/{total_c}  ({acc_c:.2%})")


if __name__ == "__main__":
    main()
