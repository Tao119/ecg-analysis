"""
evaluate_all.py — Complete ECG Evaluation Pipeline
────────────────────────────────────────────────────
Loads any available trained model weights (or falls back to random weights),
generates 100 fresh test ECG samples (20 per class), runs both CNN and
Transformer inference, and produces:

  • evaluation_results.json  — accuracy, F1 per class, confusion matrix
  • roc_curves.png           — one-vs-rest ROC curve for each class

Usage
─────
    cd ecg-analysis
    python evaluate_all.py
"""

import os
import sys
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Add ecg-analysis directory to path so local imports work regardless of cwd
# ---------------------------------------------------------------------------
ECG_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ECG_DIR)

from config import (
    NUM_CLASSES, CLASS_NAMES, NUM_LEADS, SIGNAL_LENGTH,
    CNN_WEIGHTS_PATH, TRANSFORMER_WEIGHTS_PATH, RANDOM_SEED,
)
from models import ECG_CNN, ECG_Transformer
from generate_synthetic_ecg import generate_ecg
from dataset import per_lead_normalise


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Generate 100 test samples (20 per class)
# ---------------------------------------------------------------------------

def generate_test_set(n_per_class: int = 20, seed: int = 999) -> tuple:
    """
    Generate n_per_class ECG samples for each of the 6 classes.

    Returns
    -------
    signals : (N, 12, 5000) float32  normalised
    labels  : (N,)           int64
    """
    rng_state = np.random.get_state()
    np.random.seed(seed)

    signals_list = []
    labels_list = []

    for cls_idx in range(NUM_CLASSES):
        for _ in range(n_per_class):
            ecg = generate_ecg(cls_idx)      # (12, 5000) float32
            signals_list.append(ecg)
            labels_list.append(cls_idx)

    # Restore original random state so callers are unaffected
    np.random.set_state(rng_state)

    signals = np.stack(signals_list, axis=0)   # (N, 12, 5000)
    labels = np.array(labels_list, dtype=np.int64)

    # Shuffle
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(labels))
    signals = signals[idx]
    labels = labels[idx]

    # Per-lead z-score normalisation
    signals = per_lead_normalise(signals)

    return signals, labels


# ---------------------------------------------------------------------------
# Load model — best weights if available, else random init
# ---------------------------------------------------------------------------

def load_model(model: torch.nn.Module, weights_path: str, device: torch.device) -> bool:
    """Load weights into model if the file exists. Returns True if loaded."""
    full_path = os.path.join(ECG_DIR, weights_path)
    if os.path.exists(full_path):
        state = torch.load(full_path, map_location=device)
        model.load_state_dict(state)
        print(f"  Loaded weights: {full_path}")
        return True
    print(f"  No weights found at {full_path} — using random initialisation")
    return False


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    signals: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
) -> tuple:
    """
    Run model inference over all test signals.

    Returns
    -------
    preds  : (N,)          int64 predicted class indices
    probs  : (N, C)        float32 softmax probabilities
    """
    model.eval()
    x = torch.from_numpy(signals).float()

    all_probs = []
    for i in range(0, len(x), batch_size):
        batch = x[i:i + batch_size].to(device)
        logits = model(batch)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)

    probs_all = np.concatenate(all_probs, axis=0)    # (N, C)
    preds = probs_all.argmax(axis=1).astype(np.int64)
    return preds, probs_all


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        conf[t, p] += 1
    return conf


def compute_per_class_f1(conf: np.ndarray) -> np.ndarray:
    f1s = []
    for c in range(NUM_CLASSES):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1s.append(float(f1))
    return np.array(f1s)


def compute_roc_ovr(y_true: np.ndarray, probs: np.ndarray):
    """
    Compute one-vs-rest ROC curve for each class.

    Returns
    -------
    roc_data : list of dicts, one per class
        Each dict has keys: 'fpr', 'tpr', 'auc'
    """
    roc_data = []
    for c in range(NUM_CLASSES):
        binary = (y_true == c).astype(int)
        score = probs[:, c]

        # Sort by descending score
        order = np.argsort(-score)
        binary_sorted = binary[order]
        score_sorted = score[order]

        # Compute TPR / FPR at each threshold
        n_pos = binary.sum()
        n_neg = len(binary) - n_pos

        tp_cum = np.cumsum(binary_sorted)
        fp_cum = np.cumsum(1 - binary_sorted)

        tpr = tp_cum / max(n_pos, 1)
        fpr = fp_cum / max(n_neg, 1)

        # Prepend (0, 0)
        tpr = np.concatenate([[0], tpr])
        fpr = np.concatenate([[0], fpr])

        # AUC via trapezoidal rule (np.trapezoid added in NumPy 2.0)
        _trapz = getattr(np, "trapezoid", None) or np.trapz
        auc = float(_trapz(tpr, fpr))

        roc_data.append({
            "class": CLASS_NAMES[c],
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": round(auc, 4),
        })
    return roc_data


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(conf: np.ndarray, title: str, ax: plt.Axes):
    im = ax.imshow(conf, cmap="Blues", interpolation="nearest")
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


def plot_roc_curves(roc_cnn: list, roc_tf: list, save_path: str):
    """
    Plot one-vs-rest ROC curves for both CNN and Transformer side by side.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("ROC Curves — One-vs-Rest (per class)", fontsize=14)
    colors_cnn = "#2980b9"
    colors_tf = "#e74c3c"

    for i, (roc_c, roc_t) in enumerate(zip(roc_cnn, roc_tf)):
        ax = axes[i // 3][i % 3]
        ax.plot(roc_c["fpr"], roc_c["tpr"],
                label=f"CNN  AUC={roc_c['auc']:.3f}", color=colors_cnn, lw=2)
        ax.plot(roc_t["fpr"], roc_t["tpr"],
                label=f"Transf AUC={roc_t['auc']:.3f}", color=colors_tf, lw=2, linestyle="--")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(roc_c["class"])
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC curves → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    device = get_device()
    print(f"Device: {device}")

    # ── 1. Generate test set ──────────────────────────────────────────────────
    print("\nGenerating 100 test ECG samples (20 per class)…")
    signals, labels = generate_test_set(n_per_class=20, seed=RANDOM_SEED + 1000)
    print(f"  Signals shape: {signals.shape}  |  Labels: {np.bincount(labels)}")

    # ── 2. Load models ────────────────────────────────────────────────────────
    print("\nLoading CNN…")
    cnn = ECG_CNN().to(device)
    cnn_loaded = load_model(cnn, CNN_WEIGHTS_PATH, device)

    print("Loading Transformer…")
    transformer = ECG_Transformer().to(device)
    tf_loaded = load_model(transformer, TRANSFORMER_WEIGHTS_PATH, device)

    # ── 3. Inference ──────────────────────────────────────────────────────────
    print("\nRunning inference…")
    preds_cnn, probs_cnn = run_inference(cnn, signals, device)
    preds_tf, probs_tf = run_inference(transformer, signals, device)

    # ── 4. Metrics ────────────────────────────────────────────────────────────
    conf_cnn = compute_confusion_matrix(labels, preds_cnn)
    conf_tf = compute_confusion_matrix(labels, preds_tf)

    f1_cnn = compute_per_class_f1(conf_cnn)
    f1_tf = compute_per_class_f1(conf_tf)

    acc_cnn = float(conf_cnn.diagonal().sum() / conf_cnn.sum())
    acc_tf = float(conf_tf.diagonal().sum() / conf_tf.sum())

    macro_f1_cnn = float(f1_cnn.mean())
    macro_f1_tf = float(f1_tf.mean())

    roc_cnn = compute_roc_ovr(labels, probs_cnn)
    roc_tf = compute_roc_ovr(labels, probs_tf)

    # ── 5. Print report ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  {'Metric':<30} {'CNN':>10} {'Transformer':>12}")
    print("-" * 60)
    print(f"  {'Accuracy':<30} {acc_cnn:>10.4f} {acc_tf:>12.4f}")
    print(f"  {'Macro F1':<30} {macro_f1_cnn:>10.4f} {macro_f1_tf:>12.4f}")
    print("-" * 60)
    for c, name in enumerate(CLASS_NAMES):
        print(f"  F1 {name:<26} {f1_cnn[c]:>10.4f} {f1_tf[c]:>12.4f}")
    print("-" * 60)
    for c, name in enumerate(CLASS_NAMES):
        auc_c = roc_cnn[c]["auc"]
        auc_t = roc_tf[c]["auc"]
        print(f"  AUC {name:<25} {auc_c:>10.4f} {auc_t:>12.4f}")
    print("=" * 60)
    print(f"\nWeights used: CNN={'pre-trained' if cnn_loaded else 'random'}, "
          f"Transformer={'pre-trained' if tf_loaded else 'random'}")

    # ── 6. Save results ───────────────────────────────────────────────────────
    results = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_test_samples": int(len(labels)),
        "n_per_class": 20,
        "classes": CLASS_NAMES,
        "cnn": {
            "weights": "pre-trained" if cnn_loaded else "random",
            "accuracy": round(acc_cnn, 4),
            "macro_f1": round(macro_f1_cnn, 4),
            "f1_per_class": {name: round(float(f), 4)
                             for name, f in zip(CLASS_NAMES, f1_cnn)},
            "confusion_matrix": conf_cnn.tolist(),
            "roc": [{
                "class": r["class"],
                "auc": r["auc"],
            } for r in roc_cnn],
        },
        "transformer": {
            "weights": "pre-trained" if tf_loaded else "random",
            "accuracy": round(acc_tf, 4),
            "macro_f1": round(macro_f1_tf, 4),
            "f1_per_class": {name: round(float(f), 4)
                             for name, f in zip(CLASS_NAMES, f1_tf)},
            "confusion_matrix": conf_tf.tolist(),
            "roc": [{
                "class": r["class"],
                "auc": r["auc"],
            } for r in roc_tf],
        },
    }

    results_path = os.path.join(ECG_DIR, "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved evaluation results → {results_path}")

    # ── 7. Plots ──────────────────────────────────────────────────────────────
    # Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    plot_confusion_matrix(conf_cnn, "ECG_CNN — Confusion Matrix", axes[0])
    plot_confusion_matrix(conf_tf, "ECG_Transformer — Confusion Matrix", axes[1])
    plt.tight_layout()
    cm_path = os.path.join(ECG_DIR, "evaluation_confusion_matrices.png")
    plt.savefig(cm_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrices → {cm_path}")

    # ROC curves
    roc_path = os.path.join(ECG_DIR, "roc_curves.png")
    plot_roc_curves(roc_cnn, roc_tf, roc_path)

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.1f}s")
    print("Done.")


if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    main()
