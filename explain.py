"""
explain.py — Gradient-based ECG Explanation (CAM-style / Vanilla Gradients)
─────────────────────────────────────────────────────────────────────────────
For a classified ECG, compute a saliency map using vanilla gradients.
Shows which time regions and leads the model focuses on by overlaying
gradient magnitudes as a heatmap on top of the raw ECG signal.

Output
------
    explanation_demo.png  — multi-panel figure:
        • Row per selected lead showing ECG + gradient heatmap overlay
        • Bottom panel: mean gradient magnitude across leads (global importance)

Usage
─────
    cd ecg-analysis
    python explain.py [--class CLASS_NAME] [--lead LEAD_IDX]

Examples
--------
    python explain.py
    python explain.py --class STEMI --lead 1
"""

import os
import sys
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ECG_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ECG_DIR)

from config import (
    NUM_CLASSES, CLASS_NAMES, NUM_LEADS, LEAD_NAMES, SIGNAL_LENGTH,
    SAMPLE_RATE, CNN_WEIGHTS_PATH, TRANSFORMER_WEIGHTS_PATH, RANDOM_SEED,
)
from models import ECG_CNN, ECG_Transformer
from generate_synthetic_ecg import generate_ecg
from dataset import per_lead_normalise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_best_effort(model, weights_path: str, device: torch.device) -> bool:
    full = os.path.join(ECG_DIR, weights_path)
    if os.path.exists(full):
        model.load_state_dict(torch.load(full, map_location=device))
        return True
    return False


def generate_and_normalise(class_idx: int, seed: int = 42) -> np.ndarray:
    """
    Generate one ECG of the given class and normalise it.
    Returns: (12, 5000) float32
    """
    np.random.seed(seed)
    ecg = generate_ecg(class_idx)           # (12, 5000)
    ecg = ecg[np.newaxis]                    # (1, 12, 5000)
    ecg = per_lead_normalise(ecg)[0]         # (12, 5000)
    return ecg


# ---------------------------------------------------------------------------
# Vanilla gradient saliency
# ---------------------------------------------------------------------------

def compute_saliency(
    model: torch.nn.Module,
    signal: np.ndarray,
    target_class: int,
    device: torch.device,
) -> np.ndarray:
    """
    Compute vanilla gradient saliency map.

    Parameters
    ----------
    model        : PyTorch model that accepts (1, 12, 5000) input
    signal       : (12, 5000) float32 ECG signal (normalised)
    target_class : class index to explain
    device       : torch device

    Returns
    -------
    saliency : (12, 5000) float32
        Absolute gradient w.r.t. input, averaged across batch dim.
    """
    model.eval()

    x = torch.from_numpy(signal).float().unsqueeze(0).to(device)  # (1, 12, 5000)
    x.requires_grad_(True)

    logits = model(x)                                # (1, C)
    # Score for target class
    score = logits[0, target_class]
    model.zero_grad()
    score.backward()

    # Gradient magnitude: |dL/dx|
    saliency = x.grad.abs().squeeze(0).detach().cpu().numpy()   # (12, 5000)
    return saliency


def smooth_gradient(
    model: torch.nn.Module,
    signal: np.ndarray,
    target_class: int,
    device: torch.device,
    n_samples: int = 30,
    noise_std: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    """
    SmoothGrad: average over n_samples noisy versions of the input.
    Produces cleaner saliency than vanilla gradients.

    Returns
    -------
    saliency : (12, 5000) float32
    """
    rng = np.random.default_rng(seed)
    model.eval()

    accum = np.zeros_like(signal)
    for _ in range(n_samples):
        noise = rng.standard_normal(signal.shape).astype(np.float32) * noise_std
        noisy = (signal + noise)
        s = compute_saliency(model, noisy, target_class, device)
        accum += s

    return accum / n_samples


def normalise_saliency(sal: np.ndarray) -> np.ndarray:
    """Scale saliency to [0, 1] for visualisation."""
    s_min, s_max = sal.min(), sal.max()
    if s_max - s_min < 1e-10:
        return np.zeros_like(sal)
    return (sal - s_min) / (s_max - s_min)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_explanation(
    signal: np.ndarray,
    saliency: np.ndarray,
    pred_class: int,
    true_class: int,
    selected_leads: list,
    save_path: str,
    model_name: str = "ECG_CNN",
):
    """
    Create explanation figure.

    signal   : (12, 5000) normalised ECG
    saliency : (12, 5000) gradient magnitude (normalised to [0,1])
    """
    sal_norm = normalise_saliency(saliency)
    n_leads_show = len(selected_leads)
    t = np.linspace(0, SIGNAL_LENGTH / SAMPLE_RATE, SIGNAL_LENGTH)  # seconds

    fig = plt.figure(figsize=(18, 3 * n_leads_show + 3))
    fig.suptitle(
        f"{model_name} — Gradient Saliency  |  "
        f"True: {CLASS_NAMES[true_class]}  |  "
        f"Predicted: {CLASS_NAMES[pred_class]}",
        fontsize=13,
    )

    gs = fig.add_gridspec(n_leads_show + 1, 1, hspace=0.45)

    cmap = plt.cm.hot

    for row_idx, lead_idx in enumerate(selected_leads):
        ax = fig.add_subplot(gs[row_idx, 0])
        ecg_sig = signal[lead_idx]
        sal_sig = sal_norm[lead_idx]

        # Plot ECG signal
        ax.plot(t, ecg_sig, color="#2c3e50", lw=0.9, zorder=3, label="ECG")

        # Overlay gradient as colour-coded fill using scatter trick
        # Divide signal into small windows and shade based on gradient
        window = 50
        for w_start in range(0, SIGNAL_LENGTH - window, window):
            w_end = w_start + window
            intensity = sal_sig[w_start:w_end].mean()
            colour = cmap(intensity)
            ax.axvspan(
                t[w_start], t[min(w_end, SIGNAL_LENGTH - 1)],
                alpha=intensity * 0.55 + 0.05,
                color=colour,
                zorder=1,
            )

        ax.set_ylabel(LEAD_NAMES[lead_idx], fontsize=9)
        ax.set_xlim(t[0], t[-1])

        # Annotate top gradient region
        top_idx = sal_norm[lead_idx].argmax()
        top_t = t[top_idx]
        ax.axvline(top_t, color="red", linestyle="--", lw=0.8, alpha=0.7, zorder=5)
        ax.text(top_t, ecg_sig.max() * 0.9, f"peak\n{top_t:.2f}s",
                color="red", fontsize=6.5, ha="center", zorder=6)

        if row_idx == n_leads_show - 1:
            ax.set_xlabel("Time (s)", fontsize=9)

    # Bottom panel: global gradient magnitude across all leads
    ax_global = fig.add_subplot(gs[n_leads_show, 0])
    mean_sal = sal_norm.mean(axis=0)   # (5000,)
    ax_global.fill_between(t, mean_sal, alpha=0.7, color="#e74c3c", label="Mean saliency")
    ax_global.plot(t, mean_sal, color="#c0392b", lw=0.8)
    ax_global.set_xlabel("Time (s)", fontsize=9)
    ax_global.set_ylabel("Mean |grad|", fontsize=9)
    ax_global.set_title("Global Gradient Magnitude (all leads)", fontsize=10)
    ax_global.set_xlim(t[0], t[-1])
    ax_global.legend(fontsize=8)
    ax_global.grid(axis="x", alpha=0.3)

    # Add colour bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.get_axes(), orientation="vertical",
                        fraction=0.015, pad=0.04, shrink=0.6)
    cbar.set_label("Gradient Importance", fontsize=8)

    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved explanation → {save_path}")


def plot_multi_model_comparison(
    signal: np.ndarray,
    sal_cnn: np.ndarray,
    sal_tf: np.ndarray,
    class_idx: int,
    save_path: str,
):
    """
    Side-by-side comparison of saliency from CNN vs Transformer.
    Shows lead II (index 1) as representative lead.
    """
    lead_idx = 1
    t = np.linspace(0, SIGNAL_LENGTH / SAMPLE_RATE, SIGNAL_LENGTH)

    sal_cnn_norm = normalise_saliency(sal_cnn)
    sal_tf_norm = normalise_saliency(sal_tf)

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    fig.suptitle(
        f"Saliency Comparison — {CLASS_NAMES[class_idx]} (Lead {LEAD_NAMES[lead_idx]})",
        fontsize=12,
    )

    ecg = signal[lead_idx]

    for ax, sal, name, color in [
        (axes[0], sal_cnn_norm, "ECG_CNN", "#2980b9"),
        (axes[1], sal_tf_norm, "ECG_Transformer", "#e74c3c"),
    ]:
        ax.plot(t, ecg, color="#2c3e50", lw=0.9, label="ECG signal")
        ax2 = ax.twinx()
        ax2.fill_between(t, sal[lead_idx], alpha=0.35, color=color,
                          label=f"{name} saliency")
        ax2.set_ylim(0, 1.2)
        ax2.set_ylabel("Grad magnitude (norm.)", fontsize=8)
        ax.set_ylabel("Amplitude", fontsize=8)
        ax.set_title(name, fontsize=10)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")

    axes[1].set_xlabel("Time (s)", fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="ECG gradient saliency explanation")
    parser.add_argument(
        "--class", dest="class_name", type=str, default="STEMI",
        choices=CLASS_NAMES,
        help="ECG class to explain (default: STEMI)"
    )
    parser.add_argument(
        "--leads", type=int, nargs="+", default=[0, 1, 6, 7],
        help="Lead indices to visualise (default: 0=I 1=II 6=V1 7=V2)"
    )
    parser.add_argument(
        "--smooth", action="store_true", default=True,
        help="Use SmoothGrad (default: True)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")

    class_idx = CLASS_NAMES.index(args.class_name)
    selected_leads = [l for l in args.leads if 0 <= l < NUM_LEADS]
    if not selected_leads:
        selected_leads = [0, 1, 6, 7]

    print(f"\nExplaining class: {CLASS_NAMES[class_idx]}  (index {class_idx})")
    print(f"Selected leads  : {[LEAD_NAMES[l] for l in selected_leads]}")

    # ── Generate one ECG sample ──────────────────────────────────────────────
    print("Generating ECG sample…")
    signal = generate_and_normalise(class_idx, seed=RANDOM_SEED + 42)

    # ── Load models ──────────────────────────────────────────────────────────
    cnn = ECG_CNN().to(device)
    cnn_loaded = load_model_best_effort(cnn, CNN_WEIGHTS_PATH, device)
    print(f"CNN weights: {'pre-trained' if cnn_loaded else 'random init'}")

    tf = ECG_Transformer().to(device)
    tf_loaded = load_model_best_effort(tf, TRANSFORMER_WEIGHTS_PATH, device)
    print(f"Transformer weights: {'pre-trained' if tf_loaded else 'random init'}")

    # ── Get predictions ──────────────────────────────────────────────────────
    x = torch.from_numpy(signal).float().unsqueeze(0).to(device)
    with torch.no_grad():
        pred_cnn = int(F.softmax(cnn(x), dim=-1).argmax(dim=1).item())
        pred_tf = int(F.softmax(tf(x), dim=-1).argmax(dim=1).item())

    print(f"CNN predicted    : {CLASS_NAMES[pred_cnn]}  (true: {CLASS_NAMES[class_idx]})")
    print(f"Transf predicted : {CLASS_NAMES[pred_tf]}  (true: {CLASS_NAMES[class_idx]})")

    # ── Compute saliency ─────────────────────────────────────────────────────
    print("\nComputing saliency…")
    t0 = time.time()

    if args.smooth:
        sal_cnn = smooth_gradient(cnn, signal, class_idx, device, n_samples=20)
        sal_tf = smooth_gradient(tf, signal, class_idx, device, n_samples=20)
    else:
        sal_cnn = compute_saliency(cnn, signal, class_idx, device)
        sal_tf = compute_saliency(tf, signal, class_idx, device)

    print(f"  Saliency computed in {time.time() - t0:.1f}s")

    # ── Main explanation figure ───────────────────────────────────────────────
    demo_path = os.path.join(ECG_DIR, "explanation_demo.png")
    plot_explanation(
        signal=signal,
        saliency=sal_cnn,
        pred_class=pred_cnn,
        true_class=class_idx,
        selected_leads=selected_leads,
        save_path=demo_path,
        model_name="ECG_CNN",
    )

    # ── Side-by-side comparison figure ───────────────────────────────────────
    comp_path = os.path.join(ECG_DIR, "explanation_comparison.png")
    plot_multi_model_comparison(
        signal=signal,
        sal_cnn=sal_cnn,
        sal_tf=sal_tf,
        class_idx=class_idx,
        save_path=comp_path,
    )

    # ── Per-lead saliency summary ─────────────────────────────────────────────
    sal_cnn_norm = normalise_saliency(sal_cnn)
    sal_tf_norm = normalise_saliency(sal_tf)

    print("\n=== Lead-level Saliency Summary (CNN) ===")
    lead_importance = sal_cnn_norm.mean(axis=1)   # (12,)
    ranked = np.argsort(-lead_importance)
    for rank, lead in enumerate(ranked, 1):
        print(f"  Rank {rank:>2}: {LEAD_NAMES[lead]:<6} "
              f"mean_saliency={lead_importance[lead]:.4f}")

    print("\n=== Top-5 Time Regions (CNN, Lead II) ===")
    lead1_sal = sal_cnn_norm[1]
    t = np.linspace(0, SIGNAL_LENGTH / SAMPLE_RATE, SIGNAL_LENGTH)
    window_size = 250   # 0.5 second windows at 500 Hz
    windows = [(i, i + window_size, lead1_sal[i:i + window_size].mean())
               for i in range(0, SIGNAL_LENGTH - window_size, window_size)]
    top_windows = sorted(windows, key=lambda x: -x[2])[:5]
    for i, (ws, we, mean_sal) in enumerate(top_windows, 1):
        print(f"  Top {i}: {t[ws]:.2f}s — {t[we]:.2f}s  (mean_sal={mean_sal:.4f})")

    print("\nDone.")


if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    main()
