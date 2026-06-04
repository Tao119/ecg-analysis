# ECG Analysis System

12誘導心電図（ECG）異常分類システム。純粋なNumPyとPyTorchで実装。

## 概要

合成ECGデータを生成し、1D ResNet CNNおよびパッチベースTransformerで6クラス分類を学習する。

| クラス | 説明 |
|--------|------|
| Normal | 正常洞調律（60–100 bpm）|
| AFIB | 心房細動（不整RR、P波消失、細動基線）|
| STEMI | ST上昇型心筋梗塞（下壁または前壁）|
| LBBB | 左脚ブロック（幅広QRS、V5-V6でノッチR）|
| VT | 心室頻拍（>150 bpm、幅広QRS）|
| AV_Block | 完全房室ブロック（P波とQRSが独立）|

## ディレクトリ構成

```
ecg-analysis/
├── config.py                    # グローバル設定
├── generate_synthetic_ecg.py    # 合成ECGデータ生成
├── dataset.py                   # ECGデータローダー（正規化・拡張）
├── models/
│   ├── ecg_cnn.py               # 1D Residual CNN
│   └── ecg_transformer.py       # Patchベース Transformer
├── train.py                     # 学習パイプライン
└── analyze.py                   # デモ推論
```

## 使い方

```bash
# 1. 合成データ生成（2000件 × 6クラス相当）
python generate_synthetic_ecg.py

# 2. 学習（CNN + Transformer、各10エポック）
python train.py --model both --epochs 10

# CNNのみ
python train.py --model cnn

# Transformerのみ
python train.py --model transformer

# 3. デモ推論
python analyze.py --model cnn
python analyze.py --model transformer
```

## モデル

### ECG_CNN（`models/ecg_cnn.py`）

```
(N, 12, 5000)
→ Stem: Conv1d(12→32, k=7) + BN + ReLU + MaxPool(2)
→ Stage1: ResBlock(32→64, k=5) + MaxPool(2)
→ Stage2: ResBlock(64→128, k=5) + MaxPool(2)
→ Stage3: ResBlock(128→256, k=3) + AdaptiveAvgPool(1)
→ Linear(256→64) → ReLU → Dropout(0.5) → Linear(64→6)
```

### ECG_Transformer（`models/ecg_transformer.py`）

```
(N, 12, 5000)
→ Patch Embedding: patch_size=50 → 100 patches/lead → (N, 1200, 64)
→ [CLS] token prepend → (N, 1201, 64)
→ Learnable Positional Encoding
→ 4× TransformerEncoderLayer (8 heads, d_ff=256)
→ CLS token → Linear(64→6)
```

## 信号仕様

| 項目 | 値 |
|------|-----|
| 誘導数 | 12（I, II, III, aVR, aVL, aVF, V1-V6）|
| サンプリング周波数 | 500 Hz |
| 記録時間 | 10 秒 |
| 信号長 | 5000 サンプル |
| データ形状 | (N, 12, 5000) float32 |

## 依存関係

```
numpy
torch
matplotlib
```
