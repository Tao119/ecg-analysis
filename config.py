"""
ECG Analysis System - Global Configuration
"""

# ── Data ──────────────────────────────────────────────────────────────────────
SAMPLE_RATE    = 500          # Hz
DURATION_SEC   = 10           # seconds per recording
SIGNAL_LENGTH  = SAMPLE_RATE * DURATION_SEC   # 5000 samples
NUM_LEADS      = 12
LEAD_NAMES     = ["I", "II", "III", "aVR", "aVL", "aVF",
                  "V1", "V2", "V3", "V4", "V5", "V6"]

NUM_CLASSES    = 6
CLASS_NAMES    = ["Normal", "AFIB", "STEMI", "LBBB", "VT", "AV_Block"]
CLASS_MAP      = {name: i for i, name in enumerate(CLASS_NAMES)}

NUM_SAMPLES    = 2000         # synthetic ECG count (reduced for speed)
DATA_FILE      = "ecg_data.npz"
RANDOM_SEED    = 42

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE     = 32
EPOCHS         = 10
LR             = 1e-3
VAL_SPLIT      = 0.2
DROPOUT        = 0.5

# ── Paths ─────────────────────────────────────────────────────────────────────
CNN_WEIGHTS_PATH         = "best_cnn.pt"
TRANSFORMER_WEIGHTS_PATH = "best_transformer.pt"
CONFUSION_PNG            = "confusion_matrix.png"
