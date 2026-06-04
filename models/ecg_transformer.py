"""
models/ecg_transformer.py
─────────────────────────
Positional-encoding Transformer encoder for 12-lead ECG classification.

Architecture
────────────
Input: (N, 12, L=5000)

1. Patch embedding
   • Split each lead into non-overlapping patches of size P=50
     → (N, 12, L//P) = (N, 12, 100) patches per lead
   • Linear projection: patch_dim=50*1=50 → d_model=64  (per lead)
   • After projection: (N, 12, 100, 64)

2. Multi-lead flattening
   • Reshape → (N, 12*100, 64) = (N, 1200, 64) tokens
   • Each token represents one spatial patch of one lead

3. Prepend learnable [CLS] token → (N, 1201, 64)

4. Add learnable positional encoding (N, 1201, 64)

5. Transformer Encoder: 4 layers, 8 heads, d_ff=256, dropout=0.1

6. Extract CLS token → (N, 64)

7. Linear(64 → num_classes)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NUM_LEADS, NUM_CLASSES, SIGNAL_LENGTH, DROPOUT


# ─────────────────────────────────────────────────────────────────────────────
#  Positional encoding
# ─────────────────────────────────────────────────────────────────────────────

class LearnablePositionalEncoding(nn.Module):
    """Learnable absolute positional embedding (max_len tokens)."""

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, T, d_model)"""
        return x + self.pe[:, :x.size(1), :]


# ─────────────────────────────────────────────────────────────────────────────
#  Patch embedding
# ─────────────────────────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    Convert a multi-lead ECG into a sequence of patch tokens.

    Each lead is partitioned into non-overlapping windows of width `patch_size`.
    A shared linear layer projects each patch to d_model dimensions.

    Parameters
    ──────────
    num_leads   : 12
    signal_len  : 5000
    patch_size  : 50  → 100 patches per lead
    d_model     : 64
    """

    def __init__(
        self,
        num_leads:   int = NUM_LEADS,
        signal_len:  int = SIGNAL_LENGTH,
        patch_size:  int = 50,
        d_model:     int = 64,
    ):
        super().__init__()
        assert signal_len % patch_size == 0, (
            f"signal_len {signal_len} must be divisible by patch_size {patch_size}"
        )
        self.patch_size  = patch_size
        self.num_patches = signal_len // patch_size          # 100
        self.num_leads   = num_leads
        self.d_model     = d_model

        # Each patch token: one lead × patch_size values → d_model
        # Implemented as Conv1d with kernel_size=stride=patch_size for efficiency
        self.proj = nn.Conv1d(
            in_channels  = num_leads,
            out_channels = d_model * num_leads,              # d_model per lead
            kernel_size  = patch_size,
            stride       = patch_size,
            groups       = num_leads,                        # lead-independent
            bias         = True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ──────────
        x : (N, 12, L)

        Returns
        ───────
        tokens : (N, num_leads * num_patches, d_model)
        """
        N = x.size(0)
        # Conv1d grouped: (N, 12*d_model, num_patches)
        out = self.proj(x)                                    # (N, 768, 100)
        # Reshape: (N, 12, d_model, 100) → (N, 12, 100, d_model)
        out = out.view(N, self.num_leads, self.d_model, self.num_patches)
        out = out.permute(0, 1, 3, 2)                        # (N, 12, 100, 64)
        # Flatten leads and patches: (N, 1200, 64)
        out = out.reshape(N, self.num_leads * self.num_patches, self.d_model)
        return out


# ─────────────────────────────────────────────────────────────────────────────
#  Transformer Encoder layer (standard pre-LN)
# ─────────────────────────────────────────────────────────────────────────────

class TransformerEncoderLayer(nn.Module):
    """Pre-LayerNorm Transformer block (more stable than post-LN)."""

    def __init__(
        self,
        d_model:  int   = 64,
        n_heads:  int   = 8,
        d_ff:     int   = 256,
        dropout:  float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0

        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention (pre-LN)
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop1(attn_out)

        # Feed-forward (pre-LN)
        x = x + self.ff(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
#  Main model
# ─────────────────────────────────────────────────────────────────────────────

class ECG_Transformer(nn.Module):
    """
    Patch-based Transformer for 12-lead ECG classification.

    Parameters
    ──────────
    num_leads   : number of input leads        (default 12)
    num_classes : number of output classes     (default 6)
    signal_len  : signal length                (default 5000)
    patch_size  : samples per patch            (default 50 → 100 patches/lead)
    d_model     : embedding dimension          (default 64)
    n_heads     : attention heads              (default 8)
    n_layers    : transformer encoder layers   (default 4)
    d_ff        : feed-forward hidden dim      (default 256)
    dropout     : dropout rate                 (default 0.1)
    """

    def __init__(
        self,
        num_leads:   int   = NUM_LEADS,
        num_classes: int   = NUM_CLASSES,
        signal_len:  int   = SIGNAL_LENGTH,
        patch_size:  int   = 50,
        d_model:     int   = 64,
        n_heads:     int   = 8,
        n_layers:    int   = 4,
        d_ff:        int   = 256,
        dropout:     float = 0.1,
    ):
        super().__init__()

        num_patches  = signal_len // patch_size        # 100
        num_tokens   = num_leads * num_patches + 1     # +1 for CLS  → 1201

        # ── Components ────────────────────────────────────────────────────────
        self.patch_embed = PatchEmbedding(
            num_leads, signal_len, patch_size, d_model
        )
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_enc     = LearnablePositionalEncoding(d_model, max_len=num_tokens)
        self.drop_emb    = nn.Dropout(p=dropout)

        self.encoder     = nn.Sequential(*[
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.norm        = nn.LayerNorm(d_model)
        self.head        = nn.Linear(d_model, num_classes)

        self._init_weights()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ──────────
        x : (N, 12, 5000) float32

        Returns
        ───────
        logits : (N, num_classes) float32
        """
        N = x.size(0)

        # Patch embedding → (N, 1200, 64)
        tokens = self.patch_embed(x)

        # Prepend CLS token → (N, 1201, 64)
        cls = self.cls_token.expand(N, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        # Positional encoding + dropout
        tokens = self.drop_emb(self.pos_enc(tokens))

        # Transformer encoder
        for layer in self.encoder:
            tokens = layer(tokens)

        # Extract CLS → (N, 64)
        cls_out = self.norm(tokens[:, 0, :])

        # Classification head
        return self.head(cls_out)

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
    model  = ECG_Transformer().to(device)
    dummy  = torch.randn(2, 12, 5000, device=device)
    logits = model(dummy)
    print(f"ECG_Transformer | params: {model.count_parameters():,}")
    print(f"Input : {tuple(dummy.shape)}")
    print(f"Output: {tuple(logits.shape)}")   # (2, 6)
