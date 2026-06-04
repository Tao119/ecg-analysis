"""
models/simclr_ecg.py
─────────────────────
SimCLR self-supervised contrastive pre-training for 12-lead ECG signals.

Architecture
────────────
Encoder  : same ResBlock1D CNN backbone as ecg_cnn.py → 256-dim representation
Proj head: MLP(256 → 128 → 64) with BatchNorm + ReLU
Loss     : NT-Xent (normalized temperature-scaled cross-entropy), τ = 0.1

Training flow
─────────────
1. For each ECG x, generate two augmented views v1, v2
2. Encode both: z1 = proj(encoder(v1)), z2 = proj(encoder(v2))
3. NT-Xent loss pulls (z1, z2) together, pushes all other pairs apart
4. 50 epochs on unlabeled data (no class labels used)
5. Save encoder weights for downstream use

Linear evaluation
─────────────────
Freeze encoder, train linear classifier on 10% labeled data.
Compare vs supervised baseline trained on the same 10% labels.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow running from the ecg-analysis root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    SAMPLE_RATE, SIGNAL_LENGTH, NUM_LEADS, NUM_CLASSES,
    CLASS_NAMES, DATA_FILE, RANDOM_SEED,
)
from models.ecg_cnn import ResBlock1d


# ─────────────────────────────────────────────────────────────────────────────
#  ECG Augmentations
# ─────────────────────────────────────────────────────────────────────────────

def gaussian_noise(x: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
    """Add zero-mean Gaussian noise to the signal."""
    return x + sigma * torch.randn_like(x)


def time_shift(x: torch.Tensor, max_shift: int = 50) -> torch.Tensor:
    """Shift the signal left or right by a random amount, zero-padding the gap."""
    shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
    if shift == 0:
        return x
    out = torch.zeros_like(x)
    if shift > 0:
        out[..., shift:] = x[..., :-shift]
    else:
        out[..., :shift] = x[..., -shift:]
    return out


def amplitude_scale(x: torch.Tensor, scale_range: tuple = (0.8, 1.2)) -> torch.Tensor:
    """Randomly scale the amplitude of the signal."""
    lo, hi = scale_range
    scale = lo + (hi - lo) * torch.rand(1).item()
    return x * scale


def lead_dropout(x: torch.Tensor, p: float = 0.2) -> torch.Tensor:
    """Randomly zero out entire leads with probability p."""
    # x shape: (..., n_leads, length)
    n_leads = x.shape[-2]
    mask = (torch.rand(n_leads) > p).float()
    # broadcast over length dimension
    return x * mask.view(*([1] * (x.dim() - 2)), n_leads, 1)


def baseline_wander(x: torch.Tensor, freq: float = 0.5) -> torch.Tensor:
    """Add a slow sinusoidal baseline drift (simulates respiration artifact)."""
    length = x.shape[-1]
    t = torch.linspace(0, 1, length, dtype=x.dtype, device=x.device)
    phase = 2 * torch.pi * freq * t + torch.rand(1).item() * 2 * torch.pi
    wander = 0.1 * torch.sin(phase)  # amplitude ~0.1 mV
    return x + wander


def bandpass_filter(
    x: torch.Tensor,
    low: float = 0.5,
    high: float = 40.0,
    fs: float = 500.0,
) -> torch.Tensor:
    """
    Simulate bandpass filtering (0.5–40 Hz) via FFT zeroing.
    Removes out-of-band frequency components.
    """
    length = x.shape[-1]
    freqs = torch.fft.rfftfreq(length, d=1.0 / fs)
    X_fft = torch.fft.rfft(x, dim=-1)
    mask = (freqs >= low) & (freqs <= high)
    X_fft = X_fft * mask.float()
    return torch.fft.irfft(X_fft, n=length, dim=-1)


def augment(x: torch.Tensor) -> torch.Tensor:
    """Apply a random subset of augmentations to produce one augmented view."""
    fns = [
        lambda t: gaussian_noise(t, sigma=0.05),
        lambda t: time_shift(t, max_shift=50),
        lambda t: amplitude_scale(t, scale_range=(0.8, 1.2)),
        lambda t: lead_dropout(t, p=0.2),
        lambda t: baseline_wander(t, freq=0.5),
        lambda t: bandpass_filter(t, low=0.5, high=40.0, fs=float(SAMPLE_RATE)),
    ]
    # Apply each augmentation independently with p=0.5
    for fn in fns:
        if torch.rand(1).item() < 0.5:
            x = fn(x)
    return x


# ─────────────────────────────────────────────────────────────────────────────
#  Encoder (shared backbone with ecg_cnn.py)
# ─────────────────────────────────────────────────────────────────────────────

class ECGEncoder(nn.Module):
    """
    ResBlock1D CNN backbone that outputs a 256-dim representation.
    Identical structure to ECG_CNN up to (and including) stage3.
    """

    def __init__(self, num_leads: int = NUM_LEADS):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(num_leads, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.stage1 = nn.Sequential(
            ResBlock1d(32, 64, kernel=5),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.stage2 = nn.Sequential(
            ResBlock1d(64, 128, kernel=5),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.stage3 = nn.Sequential(
            ResBlock1d(128, 256, kernel=3),
            nn.AdaptiveAvgPool1d(1),
        )
        self.flatten = nn.Flatten()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, num_leads, L) → h: (N, 256)"""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.flatten(x)


# ─────────────────────────────────────────────────────────────────────────────
#  Projection Head
# ─────────────────────────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """MLP: 256 → 128 → 64 with BatchNorm and ReLU between layers."""

    def __init__(self, in_dim: int = 256, hidden_dim: int = 128, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
#  NT-Xent Loss
# ─────────────────────────────────────────────────────────────────────────────

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross-Entropy Loss (NT-Xent).
    For a batch of N samples, produces 2N embeddings (two augmented views).
    Positive pairs: (z_i, z_j) from the same sample.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        z1, z2: (N, D) L2-normalized embeddings for two views.
        Returns scalar loss.
        """
        N = z1.shape[0]
        # Concatenate: [z1; z2] → (2N, D)
        z = torch.cat([z1, z2], dim=0)
        # Normalize
        z = F.normalize(z, dim=1)
        # Similarity matrix (2N, 2N)
        sim = torch.mm(z, z.T) / self.temperature
        # Mask out self-similarity
        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, -1e9)
        # Positive pairs: (i, i+N) and (i+N, i)
        labels = torch.cat([
            torch.arange(N, 2 * N, device=z.device),
            torch.arange(0, N, device=z.device),
        ])
        loss = F.cross_entropy(sim, labels)
        return loss


# ─────────────────────────────────────────────────────────────────────────────
#  SimCLR Model
# ─────────────────────────────────────────────────────────────────────────────

class SimCLR(nn.Module):
    """Full SimCLR: encoder + projection head."""

    def __init__(self, num_leads: int = NUM_LEADS, temperature: float = 0.1):
        super().__init__()
        self.encoder    = ECGEncoder(num_leads=num_leads)
        self.projector  = ProjectionHead()
        self.criterion  = NTXentLoss(temperature=temperature)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        x1, x2: (N, leads, L) — two augmented views of the same batch.
        Returns loss, (z1, z2).
        """
        h1, h2 = self.encoder(x1), self.encoder(x2)
        z1, z2 = self.projector(h1), self.projector(h2)
        loss = self.criterion(z1, z2)
        return loss, (z1, z2)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 256-dim representation (no projection head)."""
        self.eval()
        return self.encoder(x)


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset wrappers
# ─────────────────────────────────────────────────────────────────────────────

class UnlabeledECGDataset(Dataset):
    """Wraps ECG tensors; returns two augmented views (no labels)."""

    def __init__(self, signals: np.ndarray):
        # signals: (N, leads, length)
        self.signals = torch.from_numpy(signals.astype(np.float32))

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        x = self.signals[idx]
        return augment(x), augment(x)


class LabeledECGDataset(Dataset):
    """ECG dataset with class labels for supervised/linear eval."""

    def __init__(self, signals: np.ndarray, labels: np.ndarray):
        self.signals = torch.from_numpy(signals.astype(np.float32))
        self.labels  = torch.from_numpy(labels.astype(np.int64))

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]


# ─────────────────────────────────────────────────────────────────────────────
#  Training loops
# ─────────────────────────────────────────────────────────────────────────────

def pretrain_simclr(
    signals: np.ndarray,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 3e-4,
    device: str = "cpu",
    encoder_save_path: str = "simclr_encoder.pt",
) -> SimCLR:
    """Pre-train SimCLR on unlabeled ECG data."""
    dataset = UnlabeledECGDataset(signals)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model     = SimCLR().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n{'='*60}")
    print(f"SimCLR Pre-training  |  {epochs} epochs  |  {len(signals)} samples")
    print(f"{'='*60}")

    loss_history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x1, x2 in loader:
            x1, x2 = x1.to(device), x2.to(device)
            optimizer.zero_grad()
            loss, _ = model(x1, x2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}")

    # Save encoder weights only
    torch.save(model.encoder.state_dict(), encoder_save_path)
    print(f"\nEncoder saved to {encoder_save_path}")
    return model, loss_history


def train_linear_classifier(
    encoder: ECGEncoder,
    signals: np.ndarray,
    labels: np.ndarray,
    labeled_fraction: float = 0.1,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
) -> float:
    """
    Freeze encoder, train a linear head on labeled_fraction of data.
    Returns validation accuracy.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    N   = len(signals)
    idx = rng.permutation(N)
    n_labeled = max(int(N * labeled_fraction), batch_size)
    train_idx = idx[:n_labeled]
    val_idx   = idx[n_labeled:]

    train_ds = LabeledECGDataset(signals[train_idx], labels[train_idx])
    val_ds   = LabeledECGDataset(signals[val_idx],   labels[val_idx])
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    # Freeze encoder
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    head = nn.Linear(256, NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)

    for _ in range(epochs):
        head.train()
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                feat = encoder(x)
            logits = head(feat)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    head.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_ld:
            x, y = x.to(device), y.to(device)
            feat   = encoder(x)
            preds  = head(feat).argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += len(y)

    acc = correct / total if total > 0 else 0.0
    return acc


def train_supervised_baseline(
    signals: np.ndarray,
    labels: np.ndarray,
    labeled_fraction: float = 0.1,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
) -> float:
    """Train a full supervised model on labeled_fraction of data. Returns val accuracy."""
    from models.ecg_cnn import ECG_CNN

    rng = np.random.default_rng(RANDOM_SEED)
    N   = len(signals)
    idx = rng.permutation(N)
    n_labeled = max(int(N * labeled_fraction), batch_size)
    train_idx = idx[:n_labeled]
    val_idx   = idx[n_labeled:]

    train_ds = LabeledECGDataset(signals[train_idx], labels[train_idx])
    val_ds   = LabeledECGDataset(signals[val_idx],   labels[val_idx])
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model     = ECG_CNN(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_ld:
            x, y = x.to(device), y.to(device)
            preds  = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += len(y)

    return correct / total if total > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    torch.manual_seed(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load / generate data ─────────────────────────────────────────────────
    data_path = os.path.join(os.path.dirname(__file__), "..", DATA_FILE)
    if os.path.exists(data_path):
        data    = np.load(data_path)
        signals = data["signals"]   # (N, 12, 5000)
        labels  = data["labels"]    # (N,)
    else:
        print(f"Data file not found ({data_path}). Generating small synthetic dataset …")
        N       = 400
        signals = np.random.randn(N, NUM_LEADS, SIGNAL_LENGTH).astype(np.float32)
        labels  = np.random.randint(0, NUM_CLASSES, N)

    print(f"Dataset: {signals.shape}  labels: {labels.shape}")

    # ── SimCLR pre-training (no labels used) ────────────────────────────────
    encoder_path = os.path.join(os.path.dirname(__file__), "..", "simclr_encoder.pt")
    t0 = time.time()
    simclr_model, loss_hist = pretrain_simclr(
        signals,
        epochs=50,
        batch_size=min(64, len(signals) // 4),
        lr=3e-4,
        device=device,
        encoder_save_path=encoder_path,
    )
    pretrain_time = time.time() - t0
    print(f"Pre-training time: {pretrain_time:.1f}s")

    # ── Linear evaluation (10% labels) ──────────────────────────────────────
    print("\nLinear evaluation on 10% labeled data …")
    simclr_acc = train_linear_classifier(
        simclr_model.encoder,
        signals, labels,
        labeled_fraction=0.1,
        epochs=30,
        batch_size=min(64, len(signals) // 4),
        device=device,
    )

    # ── Supervised baseline (same 10% labels) ───────────────────────────────
    print("Supervised baseline on 10% labeled data …")
    supervised_acc = train_supervised_baseline(
        signals, labels,
        labeled_fraction=0.1,
        epochs=30,
        batch_size=min(64, len(signals) // 4),
        device=device,
    )

    # ── Report ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS (10% labeled data)")
    print("=" * 60)
    print(f"  SimCLR linear eval accuracy : {simclr_acc * 100:.2f}%")
    print(f"  Supervised baseline accuracy: {supervised_acc * 100:.2f}%")
    delta = (simclr_acc - supervised_acc) * 100
    sign  = "+" if delta >= 0 else ""
    print(f"  Delta (SimCLR - Supervised) : {sign}{delta:.2f}%")
    print("=" * 60)

    # ── Plot pre-training loss curve ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(range(1, len(loss_hist) + 1), loss_hist, color="steelblue", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("NT-Xent Loss")
    axes[0].set_title("SimCLR Pre-training Loss")
    axes[0].grid(True, alpha=0.3)

    methods = ["SimCLR\n(linear eval)", "Supervised\nbaseline"]
    accs    = [simclr_acc * 100, supervised_acc * 100]
    colors  = ["steelblue", "salmon"]
    bars    = axes[1].bar(methods, accs, color=colors, edgecolor="black", linewidth=0.8)
    for bar, acc in zip(bars, accs):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{acc:.1f}%",
            ha="center", va="bottom", fontsize=11,
        )
    axes[1].set_ylim(0, 110)
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("SimCLR vs Supervised (10% labels)")
    axes[1].grid(True, axis="y", alpha=0.3)

    plt.suptitle("SimCLR ECG Self-Supervised Pre-training", fontsize=13)
    plt.tight_layout()
    out_png = os.path.join(os.path.dirname(__file__), "..", "simclr_results.png")
    plt.savefig(out_png, dpi=120)
    print(f"\nPlot saved to {out_png}")
