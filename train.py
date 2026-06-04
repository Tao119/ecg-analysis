"""
train.py
────────
Training pipeline for ECG abnormality classification.

Trains both ECG_CNN and ECG_Transformer on the synthetic dataset,
reports per-class F1 scores, saves confusion matrix, and checkpoints
the best model weights.

Usage
─────
    # 1. Generate data first (if not done yet)
    python generate_synthetic_ecg.py

    # 2. Train
    python train.py [--model cnn|transformer|both]  [--epochs N]

Output files
────────────
    best_cnn.pt          — best CNN weights
    best_transformer.pt  — best Transformer weights
    confusion_matrix.png — confusion matrices for both models
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    BATCH_SIZE, EPOCHS, LR, VAL_SPLIT, NUM_CLASSES, CLASS_NAMES,
    CNN_WEIGHTS_PATH, TRANSFORMER_WEIGHTS_PATH, CONFUSION_PNG, RANDOM_SEED,
)
from dataset import load_split
from models  import ECG_CNN, ECG_Transformer


# ─────────────────────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cosine_lr(optimizer: torch.optim.Optimizer, epoch: int, total_epochs: int, lr_min: float = 1e-6) -> float:
    """Cosine annealing (no warmup)."""
    ratio = 0.5 * (1.0 + np.cos(np.pi * epoch / total_epochs))
    new_lr = lr_min + (LR - lr_min) * ratio
    for pg in optimizer.param_groups:
        pg["lr"] = new_lr
    return new_lr


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Compute per-class F1 score and macro-averaged F1 / accuracy.
    Returns (per_class_f1, macro_f1, accuracy, conf_mat).
    """
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        conf[t, p] += 1

    f1s = []
    for c in range(NUM_CLASSES):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1s.append(float(f1))

    macro_f1 = float(np.mean(f1s))
    accuracy = float(conf.diagonal().sum() / conf.sum())
    return np.array(f1s), macro_f1, accuracy, conf


def plot_confusion_matrix(conf: np.ndarray, title: str, ax: plt.Axes):
    """Render a confusion matrix on the given Axes."""
    im = ax.imshow(conf, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=range(NUM_CLASSES),
        yticks=range(NUM_CLASSES),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted",
        ylabel="True",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    thresh = conf.max() / 2.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(
                j, i, str(conf[i, j]),
                ha="center", va="center",
                color="white" if conf[i, j] > thresh else "black",
                fontsize=7,
            )


# ─────────────────────────────────────────────────────────────────────────────
#  Core training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    criterion:  nn.Module,
    optimizer:  torch.optim.Optimizer,
    device:     torch.device,
) -> float:
    """Return mean cross-entropy loss for the epoch."""
    model.train()
    total_loss = 0.0
    n_batches  = 0
    for signals, labels in loader:
        signals = signals.to(device, non_blocking=True)
        labels  = labels.to(device,  non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(signals)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model:   nn.Module,
    loader:  DataLoader,
    device:  torch.device,
):
    """Return (y_true, y_pred) numpy arrays for the full loader."""
    model.eval()
    all_true, all_pred = [], []
    for signals, labels in loader:
        signals = signals.to(device, non_blocking=True)
        logits  = model(signals)
        preds   = logits.argmax(dim=1).cpu().numpy()
        all_true.append(labels.numpy())
        all_pred.append(preds)
    return np.concatenate(all_true), np.concatenate(all_pred)


# ─────────────────────────────────────────────────────────────────────────────
#  Full training run for one model
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    model:       nn.Module,
    model_name:  str,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    device:       torch.device,
    save_path:    str,
    epochs:       int = EPOCHS,
) -> tuple:
    """
    Train a model for `epochs` epochs.
    Returns (best_conf_mat, per_class_f1_at_best, best_val_acc).
    """
    out_dir   = os.path.dirname(os.path.abspath(save_path))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    best_val_acc  = 0.0
    best_conf_mat = None
    best_f1       = None

    print(f"\n{'='*60}")
    print(f"  Training: {model_name}  |  params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"{'='*60}")

    for epoch in range(epochs):
        t0  = time.time()
        lr  = cosine_lr(optimizer, epoch, epochs)

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        y_true, y_pred = evaluate(model, val_loader, device)
        f1s, macro_f1, val_acc, conf_mat = compute_metrics(y_true, y_pred)

        elapsed = time.time() - t0
        print(
            f"  epoch {epoch+1:3d}/{epochs}: "
            f"lr={lr:.2e}  loss={train_loss:.4f}  "
            f"val_acc={val_acc:.4f}  macro_f1={macro_f1:.4f}  ({elapsed:.0f}s)"
        )

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_conf_mat = conf_mat.copy()
            best_f1       = f1s.copy()
            torch.save(model.state_dict(), save_path)
            print(f"    ✓ saved best weights → {os.path.basename(save_path)}")

    print(f"\n  Best val accuracy : {best_val_acc:.4f}")
    print(f"  Per-class F1      :")
    for c, (name, f1) in enumerate(zip(CLASS_NAMES, best_f1)):
        print(f"    {name:<20}: {f1:.4f}")

    return best_conf_mat, best_f1, best_val_acc


# ─────────────────────────────────────────────────────────────────────────────
#  Confusion-matrix plot
# ─────────────────────────────────────────────────────────────────────────────

def save_confusion_matrices(results: dict, out_path: str):
    """
    `results` = { model_name: conf_mat }
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]
    for ax, (name, conf) in zip(axes, results.items()):
        plot_confusion_matrix(conf, title=f"{name} — Confusion Matrix", ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\nSaved confusion matrix → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="ECG classification training")
    parser.add_argument("--model",  type=str, default="both",
                        choices=["cnn", "transformer", "both"],
                        help="Which model(s) to train (default: both)")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Number of epochs (default: {EPOCHS})")
    parser.add_argument("--batch",  type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    return parser.parse_args()


def main():
    args   = parse_args()
    device = get_device()
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path  = os.path.join(script_dir, "ecg_data.npz")

    train_ds, val_ds = load_split(data_path, val_split=VAL_SPLIT, seed=RANDOM_SEED)

    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"), generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch * 2, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    conf_results = {}
    run_cnn         = args.model in ("cnn",         "both")
    run_transformer = args.model in ("transformer", "both")

    if run_cnn:
        model_cnn = ECG_CNN().to(device)
        conf_cnn, f1_cnn, acc_cnn = train_model(
            model_cnn, "ECG_CNN",
            train_loader, val_loader, device,
            save_path=os.path.join(script_dir, CNN_WEIGHTS_PATH),
            epochs=args.epochs,
        )
        conf_results["ECG_CNN"] = conf_cnn

    if run_transformer:
        model_tr = ECG_Transformer().to(device)
        conf_tr, f1_tr, acc_tr = train_model(
            model_tr, "ECG_Transformer",
            train_loader, val_loader, device,
            save_path=os.path.join(script_dir, TRANSFORMER_WEIGHTS_PATH),
            epochs=args.epochs,
        )
        conf_results["ECG_Transformer"] = conf_tr

    # ── Save confusion matrices ───────────────────────────────────────────────
    if conf_results:
        save_confusion_matrices(
            conf_results,
            out_path=os.path.join(script_dir, CONFUSION_PNG),
        )

    print("\nTraining complete.")


if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    main()
