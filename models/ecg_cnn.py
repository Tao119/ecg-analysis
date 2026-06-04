"""
models/ecg_cnn.py
─────────────────
1-D Residual CNN for 12-lead ECG classification.

Architecture
────────────
Input  : (N, 12, L)   where L = 5000

Stage 0 (stem)
  Conv1d(12→32, k=7, p=3) → BN → ReLU → MaxPool(2)        → (N, 32, 2500)

Stage 1
  ResBlock(32→64,  k=5) → MaxPool(2)                       → (N, 64,  1250)

Stage 2
  ResBlock(64→128, k=5) → MaxPool(2)                       → (N, 128,  625)

Stage 3
  ResBlock(128→256, k=3) → AdaptiveAvgPool(1)              → (N, 256,    1)

Head
  Flatten → Linear(256→64) → ReLU → Dropout(0.5)
  → Linear(64→num_classes)

Each ResBlock
  Conv1d(in, out, k, p=k//2) → BN → ReLU
  → Conv1d(out, out, k, p=k//2) → BN
  + skip (1×1 Conv if in≠out)
  → ReLU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_LEADS, NUM_CLASSES, DROPOUT


# ─────────────────────────────────────────────────────────────────────────────
#  Residual block
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock1d(nn.Module):
    """
    1-D residual block.

    Parameters
    ──────────
    in_ch, out_ch : channel sizes
    kernel        : kernel size (odd, so padding = kernel // 2 preserves length)
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 5):
        super().__init__()
        pad = kernel // 2

        self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)

        self.skip = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm1d(out_ch),
            )
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual, inplace=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Main model
# ─────────────────────────────────────────────────────────────────────────────

class ECG_CNN(nn.Module):
    """
    Residual 1-D CNN for multi-lead ECG classification.

    Parameters
    ──────────
    num_leads   : number of input channels (default 12)
    num_classes : number of output classes  (default 6)
    dropout     : dropout rate in the classification head
    """

    def __init__(
        self,
        num_leads:   int   = NUM_LEADS,
        num_classes: int   = NUM_CLASSES,
        dropout:     float = DROPOUT,
    ):
        super().__init__()

        # ── Stem ──────────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv1d(num_leads, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),          # L/2
        )

        # ── Residual stages ───────────────────────────────────────────────────
        self.stage1 = nn.Sequential(
            ResBlock1d(32,  64,  kernel=5),
            nn.MaxPool1d(kernel_size=2, stride=2),          # L/4
        )
        self.stage2 = nn.Sequential(
            ResBlock1d(64,  128, kernel=5),
            nn.MaxPool1d(kernel_size=2, stride=2),          # L/8
        )
        self.stage3 = nn.Sequential(
            ResBlock1d(128, 256, kernel=3),
            nn.AdaptiveAvgPool1d(1),                        # (N, 256, 1)
        )

        # ── Classification head ───────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Flatten(),                                   # (N, 256)
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes),
        )

        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ──────────
        x : (N, 12, L) float32

        Returns
        ───────
        logits : (N, num_classes) float32
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)

    # ── Convenience ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities (N, num_classes)."""
        self.eval()
        return torch.softmax(self(x), dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
#  Quick smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = ECG_CNN().to(device)
    dummy  = torch.randn(4, 12, 5000, device=device)
    logits = model(dummy)
    print(f"ECG_CNN | params: {model.count_parameters():,}")
    print(f"Input : {tuple(dummy.shape)}")
    print(f"Output: {tuple(logits.shape)}")   # (4, 6)
