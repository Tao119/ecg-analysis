"""
models/ecg_beats.py
────────────────────
Beat Segmentation and Per-Beat Classification from ECG signals.

Pipeline
────────
1. QRS detection  : find R-peaks using derivative + adaptive threshold
2. Beat windowing : extract 400 samples (±200 at 500 Hz = ±400 ms) around each peak
3. Classification : 1-D CNN per beat → Normal / PVC / PAC / Noise

Beat types (synthetic)
──────────────────────
- Normal  : clean QRS complex (sharp peak, narrow)
- PVC     : wide, bizarre QRS; no preceding P-wave; high amplitude
- PAC     : narrow QRS but preceded by ectopic (early) P-wave; slightly early timing
- Noise   : high-frequency random signal; no recognisable QRS

Usage
─────
  python models/ecg_beats.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAMPLE_RATE, RANDOM_SEED

# ─────────────────────────────────────────────────────────────────────────────
#  Beat type definitions
# ─────────────────────────────────────────────────────────────────────────────

BEAT_TYPES   = ["Normal", "PVC", "PAC", "Noise"]
BEAT_TO_IDX  = {b: i for i, b in enumerate(BEAT_TYPES)}
BEAT_WINDOW  = 200   # samples on each side of R-peak → total 400 samples
BEAT_LENGTH  = BEAT_WINDOW * 2   # 400 samples


# ─────────────────────────────────────────────────────────────────────────────
#  Synthetic beat generators
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian(t: np.ndarray, mu: float, sigma: float, amp: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((t - mu) / sigma) ** 2)


def generate_normal_beat(length: int = BEAT_LENGTH, fs: int = SAMPLE_RATE) -> np.ndarray:
    """
    Synthetic normal beat: P-wave + narrow QRS + T-wave.
    R-peak is centred at length//2.
    """
    t = np.arange(length) / fs          # time in seconds
    centre = length / 2 / fs            # R-peak time
    beat = (
        _gaussian(t, centre - 0.16, 0.025, 0.15)   # P-wave (160 ms before R)
        + _gaussian(t, centre - 0.02, 0.005, -0.1) # Q dip
        + _gaussian(t, centre,        0.010,  1.0)  # R peak (narrow, tall)
        + _gaussian(t, centre + 0.04, 0.008, -0.15) # S dip
        + _gaussian(t, centre + 0.20, 0.040,  0.3)  # T-wave
    )
    beat += np.random.randn(length) * 0.02
    return beat.astype(np.float32)


def generate_pvc_beat(length: int = BEAT_LENGTH, fs: int = SAMPLE_RATE) -> np.ndarray:
    """
    Premature Ventricular Complex: wide (>120 ms), bizarre, high amplitude.
    No P-wave; QRS starts early relative to expected timing.
    """
    t = np.arange(length) / fs
    centre = length / 2 / fs
    # Wide QRS (sigma ≈ 0.04 s ≈ 80 ms half-width → ~160 ms total)
    beat = (
        _gaussian(t, centre - 0.05, 0.040, -0.5)  # wide Q
        + _gaussian(t, centre,       0.025,  1.5)  # tall R (no preceding P)
        + _gaussian(t, centre + 0.08, 0.035, -0.6) # wide S
        + _gaussian(t, centre + 0.25, 0.060,  0.4) # discordant T
    )
    beat += np.random.randn(length) * 0.03
    return beat.astype(np.float32)


def generate_pac_beat(length: int = BEAT_LENGTH, fs: int = SAMPLE_RATE) -> np.ndarray:
    """
    Premature Atrial Complex: narrow QRS (like Normal) but with an ectopic
    P-wave that has a slightly different morphology and shorter PR interval.
    """
    t = np.arange(length) / fs
    centre = length / 2 / fs
    beat = (
        _gaussian(t, centre - 0.10, 0.020, 0.10)   # ectopic P (shorter PR)
        + _gaussian(t, centre - 0.015, 0.005, -0.08)
        + _gaussian(t, centre,          0.010,  0.9)  # narrow R
        + _gaussian(t, centre + 0.04,  0.008, -0.12)
        + _gaussian(t, centre + 0.18,  0.040,  0.25)
    )
    beat += np.random.randn(length) * 0.025
    return beat.astype(np.float32)


def generate_noise_beat(length: int = BEAT_LENGTH) -> np.ndarray:
    """High-frequency noise segment with no discernible QRS."""
    beat = np.random.randn(length) * 0.5
    # Add a few random high-frequency oscillations
    t = np.arange(length)
    for _ in range(3):
        freq  = np.random.uniform(20, 80)
        phase = np.random.uniform(0, 2 * np.pi)
        beat += np.random.uniform(0.1, 0.4) * np.sin(2 * np.pi * freq / SAMPLE_RATE * t + phase)
    return beat.astype(np.float32)


def generate_synthetic_beats(
    n_per_class: int = 300,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a balanced synthetic beat dataset.

    Returns
    -------
    beats  : (N, BEAT_LENGTH) float32
    labels : (N,) int64
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    generators = [
        generate_normal_beat,
        generate_pvc_beat,
        generate_pac_beat,
        lambda: generate_noise_beat(),
    ]

    all_beats  = []
    all_labels = []
    for cls_idx, gen in enumerate(generators):
        for _ in range(n_per_class):
            all_beats.append(gen())
            all_labels.append(cls_idx)

    beats  = np.stack(all_beats).astype(np.float32)
    labels = np.array(all_labels, dtype=np.int64)

    # Shuffle
    perm = rng.permutation(len(beats))
    return beats[perm], labels[perm]


# ─────────────────────────────────────────────────────────────────────────────
#  QRS Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_qrs_peaks(
    signal: np.ndarray,
    fs: int = SAMPLE_RATE,
    threshold_factor: float = 0.5,
    min_distance_ms: float = 300.0,
) -> np.ndarray:
    """
    Detect R-peak positions in a 1-D ECG signal.

    Algorithm
    ---------
    1. Compute absolute derivative |dx/dt|.
    2. Square it to emphasise large slopes.
    3. Apply a running-max envelope.
    4. Threshold at threshold_factor × global max.
    5. Enforce minimum inter-peak distance.

    Parameters
    ----------
    signal          : 1-D ECG array (one lead)
    fs              : sampling rate (Hz)
    threshold_factor: fraction of global max for threshold
    min_distance_ms : minimum distance between R-peaks (ms)

    Returns
    -------
    peaks : sorted array of sample indices
    """
    # Step 1-2: Derivative squared
    deriv  = np.diff(signal, prepend=signal[0])
    energy = deriv ** 2

    # Step 3: Smooth with a short moving average (~20 ms)
    win = max(1, int(0.020 * fs))
    kernel = np.ones(win) / win
    smoothed = np.convolve(energy, kernel, mode="same")

    # Step 4: Threshold
    threshold  = threshold_factor * smoothed.max()
    above      = smoothed > threshold

    # Step 5: Find rising edges (candidate peaks = local maxima of original signal in windows)
    min_dist = int(min_distance_ms / 1000 * fs)
    peaks    = []
    i        = 0
    while i < len(above):
        if above[i]:
            # Find the window extent
            j = i
            while j < len(above) and above[j]:
                j += 1
            # R-peak = argmax of |signal| in [i, j]
            segment = signal[i:j]
            if len(segment) > 0:
                local_peak = i + np.argmax(np.abs(segment))
                if not peaks or (local_peak - peaks[-1]) >= min_dist:
                    peaks.append(local_peak)
            i = j
        else:
            i += 1

    return np.array(peaks, dtype=np.int64)


def extract_beats(
    signal: np.ndarray,
    peaks: np.ndarray,
    window: int = BEAT_WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Window 2*window samples around each peak.
    Skips peaks too close to signal boundaries.

    Returns
    -------
    beats        : (M, 2*window) float32
    valid_peaks  : (M,) peak positions that were successfully windowed
    """
    L = len(signal)
    beats, valid = [], []
    for p in peaks:
        start, end = p - window, p + window
        if start >= 0 and end <= L:
            beats.append(signal[start:end].astype(np.float32))
            valid.append(p)
    if not beats:
        return np.empty((0, 2 * window), dtype=np.float32), np.array([], dtype=np.int64)
    return np.stack(beats), np.array(valid, dtype=np.int64)


# ─────────────────────────────────────────────────────────────────────────────
#  Per-beat 1-D CNN classifier
# ─────────────────────────────────────────────────────────────────────────────

class BeatCNN(nn.Module):
    """
    Compact 1-D CNN for single-beat classification.

    Input : (N, 1, BEAT_LENGTH)  — single-lead beat
    Output: (N, num_classes)     logits
    """

    def __init__(self, beat_length: int = BEAT_LENGTH, num_classes: int = len(BEAT_TYPES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=11, padding=5, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                          # L/2

            nn.Conv1d(32, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                          # L/4

            nn.Conv1d(64, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(8),                  # (N, 128, 8)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                             # (N, 128*8 = 1024)
            nn.Linear(128 * 8, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self(x).argmax(dim=1)


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────────────────────

class BeatDataset(Dataset):
    """Dataset of segmented beats (shape: N × BEAT_LENGTH)."""

    def __init__(self, beats: np.ndarray, labels: np.ndarray):
        # beats: (N, L) → add channel dim → (N, 1, L)
        self.beats  = torch.from_numpy(beats[:, None, :])    # (N, 1, L)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.beats)

    def __getitem__(self, idx):
        return self.beats[idx], self.labels[idx]


# ─────────────────────────────────────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────────────────────────────────────

def train_beat_classifier(
    beats: np.ndarray,
    labels: np.ndarray,
    val_frac: float = 0.2,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
) -> tuple[BeatCNN, list, list, float]:
    """
    Train BeatCNN on synthetic beats.

    Returns
    -------
    model        : trained BeatCNN
    train_losses : per-epoch training loss
    val_accs     : per-epoch validation accuracy
    final_acc    : final validation accuracy
    """
    rng = np.random.default_rng(RANDOM_SEED)
    N   = len(beats)
    idx = rng.permutation(N)
    n_val = int(N * val_frac)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    train_ds = BeatDataset(beats[train_idx], labels[train_idx])
    val_ds   = BeatDataset(beats[val_idx],   labels[val_idx])
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model     = BeatCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train_losses, val_accs = [], []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Validation accuracy
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_ld:
                x, y = x.to(device), y.to(device)
                preds  = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total   += len(y)
        val_acc = correct / total if total > 0 else 0.0

        train_losses.append(total_loss / len(train_ld))
        val_accs.append(val_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"loss={train_losses[-1]:.4f}  val_acc={val_acc*100:.1f}%")

    return model, train_losses, val_accs, val_accs[-1]


# ─────────────────────────────────────────────────────────────────────────────
#  End-to-end segmentation + classification demo
# ─────────────────────────────────────────────────────────────────────────────

def segment_and_classify(
    signal: np.ndarray,
    model: BeatCNN,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full pipeline: detect QRS peaks → window beats → classify.

    Parameters
    ----------
    signal : 1-D ECG (single lead)
    model  : trained BeatCNN

    Returns
    -------
    peaks      : R-peak positions (samples)
    beats      : extracted beat waveforms (M, BEAT_LENGTH)
    predictions: predicted class indices (M,)
    """
    peaks = detect_qrs_peaks(signal)
    beats, valid_peaks = extract_beats(signal, peaks)
    if len(beats) == 0:
        return np.array([]), np.empty((0, BEAT_LENGTH)), np.array([])
    x_tensor = torch.from_numpy(beats[:, None, :]).to(device)
    with torch.no_grad():
        preds = model(x_tensor).argmax(dim=1).cpu().numpy()
    return valid_peaks, beats, preds


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Generate synthetic beats ─────────────────────────────────────────────
    N_PER_CLASS = 300
    print(f"\nGenerating {N_PER_CLASS} beats per class ({len(BEAT_TYPES)} classes) …")
    beats, labels = generate_synthetic_beats(n_per_class=N_PER_CLASS)
    print(f"Total beats: {len(beats)}  shape: {beats.shape}")

    # ── Train beat classifier ────────────────────────────────────────────────
    print("\nTraining BeatCNN …")
    model, train_losses, val_accs, final_acc = train_beat_classifier(
        beats, labels, device=device, epochs=30,
    )
    print(f"\nFinal validation accuracy: {final_acc * 100:.2f}%")

    # ── Per-class accuracy ───────────────────────────────────────────────────
    model.eval()
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.permutation(len(beats))
    n_val = int(len(beats) * 0.2)
    val_idx = idx[:n_val]
    val_beats  = beats[val_idx]
    val_labels = labels[val_idx]

    x_t = torch.from_numpy(val_beats[:, None, :]).to(device)
    with torch.no_grad():
        preds = model(x_t).argmax(dim=1).cpu().numpy()

    print("\nPer-class accuracy:")
    for cls_idx, name in enumerate(BEAT_TYPES):
        mask = val_labels == cls_idx
        if mask.sum() == 0:
            continue
        acc  = (preds[mask] == val_labels[mask]).mean()
        print(f"  {name:8s}: {acc * 100:.1f}%  (n={mask.sum()})")

    # ── Demo: QRS detection on a synthetic rhythm strip ─────────────────────
    print("\nDemonstrating QRS detection on a synthetic rhythm strip …")
    # Build a 10-second strip with ~70 bpm beats of mixed types
    fs     = SAMPLE_RATE
    strip  = np.zeros(fs * 10, dtype=np.float32)
    rr_ms  = int(fs * 60 / 70)   # ~428 samples per beat
    types_cycle = [0, 0, 1, 0, 2, 0, 0, 3, 0, 0, 0, 1, 0, 0, 2]
    beat_funcs  = [generate_normal_beat, generate_pvc_beat,
                   generate_pac_beat, generate_noise_beat]
    true_types  = []
    positions   = []
    for k, cls in enumerate(types_cycle):
        centre = int(rr_ms * (k + 0.5))
        start  = centre - BEAT_WINDOW
        end    = centre + BEAT_WINDOW
        if start < 0 or end > len(strip):
            break
        beat_wave = beat_funcs[cls]()
        strip[start:end] += beat_wave
        true_types.append(cls)
        positions.append(centre)

    strip += np.random.randn(len(strip)) * 0.01

    det_peaks, det_beats, det_preds = segment_and_classify(strip, model, device=device)
    print(f"Detected {len(det_peaks)} peaks from {len(positions)} inserted beats")

    # ── Visualise ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Top: example beats per class
    colors = ["steelblue", "salmon", "seagreen", "orange"]
    t_beat = np.arange(BEAT_LENGTH) / SAMPLE_RATE * 1000  # ms
    ax0    = axes[0]
    for cls_idx, name in enumerate(BEAT_TYPES):
        ex = beats[labels == cls_idx][0]
        ax0.plot(t_beat + cls_idx * 450, ex + cls_idx * 2.0,
                 color=colors[cls_idx], linewidth=1.5, label=name)
    ax0.set_xlabel("Time (ms, stacked)")
    ax0.set_ylabel("Amplitude")
    ax0.set_title("Example Beats by Class")
    ax0.legend(loc="upper right")
    ax0.grid(True, alpha=0.3)

    # Middle: training curves
    axes[1].plot(train_losses, label="Train Loss", color="steelblue")
    ax1r = axes[1].twinx()
    ax1r.plot(val_accs, label="Val Acc", color="salmon", linestyle="--")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss", color="steelblue")
    ax1r.set_ylabel("Accuracy", color="salmon")
    axes[1].set_title("BeatCNN Training")
    axes[1].grid(True, alpha=0.3)

    # Bottom: rhythm strip with detected beats
    t_strip = np.arange(len(strip)) / SAMPLE_RATE
    axes[2].plot(t_strip, strip, color="grey", linewidth=0.7, alpha=0.8, label="ECG")
    if len(det_peaks) > 0:
        for p, pred in zip(det_peaks, det_preds):
            axes[2].axvline(p / SAMPLE_RATE, color=colors[pred], alpha=0.7,
                            linewidth=1.2, linestyle="--")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Amplitude")
    axes[2].set_title("Rhythm Strip with QRS Detection + Beat Classification")
    # Legend for beat types
    for cls_idx, name in enumerate(BEAT_TYPES):
        axes[2].plot([], [], color=colors[cls_idx], label=name, linewidth=2)
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = os.path.join(os.path.dirname(__file__), "..", "ecg_beats_results.png")
    plt.savefig(out_png, dpi=120)
    print(f"Plot saved to {out_png}")
    print("\nDone.")
