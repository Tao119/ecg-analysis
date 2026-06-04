# ECG Analysis System

12誘導心電図（ECG）異常分類システム。純粋な NumPy + PyTorch で実装した、合成 ECG による 6 クラス分類。

## アーキテクチャ図

```
入力 (N, 12, 5000)
       │
       ├──────────────────────────────────────────┐
       │  1D Residual CNN (ecg_cnn.py)            │  Patch Transformer (ecg_transformer.py)
       │                                          │
       │  Stem: Conv1d(12→32, k=7)               │  Patch Embedding: patch_size=50
       │        + BN + ReLU + MaxPool(2)          │  → 100 patches/lead → (N, 1200, 64)
       │                                          │
       │  Stage1: ResBlock(32→64, k=5)            │  [CLS] token → (N, 1201, 64)
       │          + MaxPool(2)                    │
       │                                          │  Learnable Positional Encoding
       │  Stage2: ResBlock(64→128, k=5)           │
       │          + MaxPool(2)                    │  4× TransformerEncoderLayer
       │                                          │    (8 heads, d_model=64, d_ff=256)
       │  Stage3: ResBlock(128→256, k=3)          │
       │          + AdaptiveAvgPool(1)            │  CLS token → Linear(64→6)
       │                                          │
       │  Linear(256→64) → ReLU                  │
       │  → Dropout(0.5) → Linear(64→6)          │
       │                                          │
       └──────────────────────────────────────────┘
                          │
                    Softmax → 6クラス
```

## 分類クラス

| クラス | 説明 | 生成パラメータ |
|--------|------|--------------|
| Normal | 正常洞調律 | 60–100 bpm, 正常 P/QRS/T 波 |
| AFIB | 心房細動 | 不整 RR 間隔、P 波消失、細動基線 |
| STEMI | ST 上昇型心筋梗塞 | ST 上昇 ≥2mm (下壁 or 前壁) |
| LBBB | 左脚ブロック | 幅広 QRS (≥120ms)、V5-V6 でノッチ R 波 |
| VT | 心室頻拍 | >150 bpm、幅広 QRS (>140ms) |
| AV_Block | 完全房室ブロック | P 波と QRS 独立 (解離) |

## 信号仕様

| 項目 | 値 |
|------|-----|
| 誘導数 | 12（I, II, III, aVR, aVL, aVF, V1-V6）|
| サンプリング周波数 | 500 Hz |
| 記録時間 | 10 秒 |
| 信号長 | 5,000 サンプル |
| データ形状 | (N, 12, 5000) float32 |
| データ件数 | 2,000 件（クラス均等） |

## ディレクトリ構成

```
ecg-analysis/
├── config.py                    # グローバル設定 (サンプリング周波数、クラス数など)
├── generate_synthetic_ecg.py    # 合成 ECG データ生成 → ecg_data.npz
├── dataset.py                   # ECGDataset (正規化・データ拡張)
├── models/
│   ├── ecg_cnn.py               # 1D Residual CNN
│   └── ecg_transformer.py       # Patch-based Transformer
├── train.py                     # 学習パイプライン (loss, checkpoint 保存)
├── analyze.py                   # デモ推論・可視化
├── evaluate_all.py              # 全評価 (ROC, confusion matrix, F1)
└── explain.py                   # Grad-CAM 的な重要度可視化
```

## 使い方

```bash
# 1. 合成データ生成 (2000件 × 6クラス)
python generate_synthetic_ecg.py
# → ecg_data.npz (X: 形状 (2000, 12, 5000), y: クラスラベル)

# 2. 学習 (CNN + Transformer、各 10 エポック)
python train.py --model both --epochs 10

# CNN のみ
python train.py --model cnn --epochs 20

# Transformer のみ
python train.py --model transformer --epochs 20

# 3. デモ推論
python analyze.py --model cnn
python analyze.py --model transformer

# 4. 全評価 (ROC 曲線 + 混同行列)
python evaluate_all.py
```

## モデル詳細

### ECG_CNN (`models/ecg_cnn.py`)

ResNet スタイルの 1D 畳み込みネットワーク。

```
パラメータ数: ~180K
入力: (N, 12, 5000)
出力: (N, 6) クラス確率

ResBlock 構造:
  Conv1d(C→C, k) → BN → ReLU
  Conv1d(C→2C, k) → BN
  + Shortcut Conv1d(C→2C, k=1)
  → ReLU
```

### ECG_Transformer (`models/ecg_transformer.py`)

パッチベースのビジョントランスフォーマー（ECG 版）。

```
パラメータ数: ~250K
パッチサイズ: 50 サンプル (100ms)
シーケンス長: 12 誘導 × 100 パッチ = 1200 + [CLS] = 1201
d_model: 64, heads: 8, layers: 4
```

## 学習結果

```
CNN (10 epochs):
  Train Acc: ~95%  Val Acc: ~88%
  Confusion: Normal/AFIB/STEMI 高精度、VT 判別が難しいケースあり

Transformer (10 epochs):
  Train Acc: ~92%  Val Acc: ~85%
  長距離依存性の高い AFIB パターンで優位
```

評価ファイル: `confusion_matrix.png`

## データ拡張

```python
# dataset.py 内で適用
augmentations = [
    GaussianNoise(std=0.01),
    BaselineWander(amp=0.05),
    AmplitudeScaling(range=(0.9, 1.1)),
    TemporalShift(max_shift=50),
]
```

## 依存関係

```
numpy >= 1.21
torch >= 2.0
matplotlib >= 3.5
```
