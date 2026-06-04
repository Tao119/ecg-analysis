"""
dataset.py
──────────
ECG data loader and preprocessing utilities.

Usage
─────
    from dataset import ECGDataset, load_split

    train_ds, val_ds = load_split("ecg_data.npz", val_split=0.2)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    for signals, labels in loader:
        ...   # signals: (B, 12, 5000), labels: (B,)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split

from config import SIGNAL_LENGTH, NUM_LEADS, NUM_CLASSES, CLASS_NAMES, DATA_FILE, RANDOM_SEED


# ─────────────────────────────────────────────────────────────────────────────
#  Normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def per_lead_normalise(signals: np.ndarray) -> np.ndarray:
    """
    Z-score normalise each lead independently.
    Input:  (N, 12, L) float32
    Output: (N, 12, L) float32
    """
    mean = signals.mean(axis=-1, keepdims=True)   # (N, 12, 1)
    std  = signals.std(axis=-1,  keepdims=True) + 1e-8
    return ((signals - mean) / std).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Augmentations (applied during training only)
# ─────────────────────────────────────────────────────────────────────────────

def augment_ecg(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Lightweight stochastic augmentations for a single ECG (12, L).
    Returns a copy — does not mutate the input.
    """
    out = signal.copy()

    # 1. Random amplitude scaling (±10 %)
    if rng.random() < 0.5:
        scale = rng.uniform(0.90, 1.10)
        out = out * scale

    # 2. Additive Gaussian noise
    if rng.random() < 0.5:
        noise = rng.normal(0, 0.02, out.shape).astype(np.float32)
        out = out + noise

    # 3. Baseline wander (slow sinusoidal drift)
    if rng.random() < 0.4:
        t       = np.linspace(0, 1, out.shape[-1], dtype=np.float32)
        freq    = rng.uniform(0.1, 0.5)
        phase   = rng.uniform(0, 2 * np.pi)
        amp     = rng.uniform(0.02, 0.08)
        wander  = amp * np.sin(2 * np.pi * freq * t + phase).astype(np.float32)
        out = out + wander[None, :]  # broadcast across leads

    # 4. Random time shift (±50 samples)
    if rng.random() < 0.5:
        shift = rng.integers(-50, 51)
        out   = np.roll(out, shift, axis=-1)

    return out.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset class
# ─────────────────────────────────────────────────────────────────────────────

class ECGDataset(Dataset):
    """
    PyTorch Dataset wrapping the .npz ECG archive.

    Parameters
    ──────────
    signals  : (N, 12, L) float32 — pre-normalised ECG signals
    labels   : (N,)       int64
    augment  : bool — apply random augmentations (training mode)
    seed     : int  — seed for per-worker RNG
    """

    def __init__(
        self,
        signals:  np.ndarray,
        labels:   np.ndarray,
        augment:  bool = False,
        seed:     int  = RANDOM_SEED,
    ):
        assert signals.ndim   == 3, f"Expected (N, C, L), got {signals.shape}"
        assert labels.ndim    == 1
        assert len(signals) == len(labels)

        self.signals = torch.from_numpy(signals)    # (N, 12, L)
        self.labels  = torch.from_numpy(labels)     # (N,)
        self.augment = augment
        self._rng    = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        sig = self.signals[idx].numpy()             # (12, L)
        lbl = self.labels[idx]

        if self.augment:
            sig = augment_ecg(sig, self._rng)
            sig = torch.from_numpy(sig)
        else:
            sig = torch.from_numpy(sig)

        return sig, lbl

    @property
    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for weighted loss."""
        counts = torch.bincount(self.labels, minlength=NUM_CLASSES).float()
        weights = 1.0 / (counts + 1e-6)
        return weights / weights.sum() * NUM_CLASSES


# ─────────────────────────────────────────────────────────────────────────────
#  Factory
# ─────────────────────────────────────────────────────────────────────────────

def load_split(
    data_path:  str   = None,
    val_split:  float = 0.2,
    seed:       int   = RANDOM_SEED,
):
    """
    Load .npz archive, normalise, and return (train_dataset, val_dataset).
    """
    if data_path is None:
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATA_FILE)

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            "Run  python generate_synthetic_ecg.py  first."
        )

    archive     = np.load(data_path, allow_pickle=True)
    signals_raw = archive["signals"].astype(np.float32)   # (N, 12, 5000)
    labels      = archive["labels"].astype(np.int64)       # (N,)

    # Normalise
    signals = per_lead_normalise(signals_raw)

    # Deterministic split
    n_total = len(labels)
    n_val   = int(n_total * val_split)
    n_train = n_total - n_val

    rng_np   = np.random.default_rng(seed)
    idx_shuf = rng_np.permutation(n_total)
    idx_train = idx_shuf[:n_train]
    idx_val   = idx_shuf[n_train:]

    train_ds = ECGDataset(signals[idx_train], labels[idx_train], augment=True,  seed=seed)
    val_ds   = ECGDataset(signals[idx_val],   labels[idx_val],   augment=False, seed=seed + 1)

    print(f"Dataset loaded — train: {len(train_ds)}, val: {len(val_ds)}")
    print(f"Signal shape: {signals.shape[1:]}  |  Classes: {CLASS_NAMES}")
    return train_ds, val_ds
