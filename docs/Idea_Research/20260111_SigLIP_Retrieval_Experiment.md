# SigLIP Embedding Retrieval 実験

**日付**: 2026-01-11
**目的**: SigLIP embeddingを使った検索ベースのアプローチで、StateやTargetがどの程度予測できるかを検証

---

## 実験設定

### 使用データ
- **Embedding**: `notebooks/HIGH_SCORE_NOTEBOOK_20251222/train_siglip_embeddings.csv`
- **モデル**: SigLIP (1152次元)
- **サンプル数**: 357画像

### CV戦略 (child-exp058と同一)
- **Stratify**: `(Dry_Clover_g > 0)_(Dry_Dead_g > 0)` の4カテゴリ
- **Group**: `day/month_State`
- **Folds**: 5

### 検索手法
- コサイン類似度によるkNN検索
- Target予測: Top-K画像のTarget値を類似度で重み付け平均

---

## 実験1: グローバルインデックス検索

全Trainデータを1つのインデックスとして検索。

### State予測精度 (K=1で最高)

| State | Accuracy | 件数 |
|-------|----------|------|
| **Vic** | **91.07%** | 102/112 |
| NSW | 76.00% | 57/75 |
| Tas | 65.94% | 91/138 |
| WA | 50.00% | 16/32 |
| **Overall** | **74.51%** | 266/357 |

**発見**:
- Vicは画像だけで91%識別可能（視覚的に特徴的）
- WAは50%と難しい（サンプル数も少ない: 32件）

### Target予測精度

| K | Global Weighted R² |
|---|-------------------|
| 1 | 0.3676 |
| 3 | 0.5500 |
| **5** | **0.5939** |
| 10 | 0.5881 |
| 20 | 0.5588 |

**Best**: K=5 で R² = 0.5939

### Per-Target R² (K=5)

| Target | R² | 備考 |
|--------|-----|------|
| Dry_Clover_g | 0.6459 | 最も予測しやすい |
| Dry_Green_g | 0.5052 | |
| GDM_g | 0.4691 | |
| Dry_Total_g | 0.4585 | |
| Dry_Dead_g | 0.2722 | 最も予測しにくい |

**発見**:
- Cloverが最も視覚的に特徴的（R²=0.65）
- Deadは草の下に隠れて見えにくい（R²=0.27）

---

## 実験2: State別インデックス検索

TrainデータをState別に分けてインデックスを構築し、検索。

### 比較手法

| Method | Global R² | 備考 |
|--------|-----------|------|
| **Global Index (Baseline)** | **0.5881** | 全Trainから検索 |
| True State Index (Oracle) | 0.5462 | 真のStateのインデックスのみ使用 |
| Best State (Max Sim) | 0.5168 | 最も類似度が高いStateを選択 |
| Weighted All States | 0.4100 | 全Stateの予測を重み付け平均 |

### Per-Target比較

| Target | Global | Oracle | 差分 |
|--------|--------|--------|------|
| Dry_Clover_g | **0.6550** | 0.3776 | **-0.277** |
| Dry_Green_g | 0.4912 | 0.4951 | +0.004 |
| Dry_Dead_g | 0.2672 | 0.2983 | +0.031 |
| GDM_g | 0.4530 | 0.3855 | -0.068 |
| Dry_Total_g | 0.4540 | 0.3979 | -0.056 |

---

## 重要な示唆

### 1. State別インデックスは効果なし（むしろ悪化）

**予想外の結果**: Stateを正しく予測できても（Oracle）、性能は -0.042 低下する。

**理由の考察**:
- サンプル数の減少により検索の多様性が失われる（特にWA: 32件のみ）
- **Cloverで顕著な低下** (0.655 → 0.378): 他Stateの類似画像も予測に有用
- State間で共通する視覚パターンが存在（枯れ草の見た目など）

### 2. 検索だけでR² 0.59達成

- CNNモデル（LB 0.73）と比較すると低いが、単純なkNN検索でここまで予測可能
- これは画像間の視覚的類似性がTarget値と強く相関していることを示す

### 3. Cloverは視覚的に識別しやすい

- 検索ベースでR² 0.65達成
- 白い花や三つ葉の形状が特徴的

### 4. Deadは視覚的に識別困難

- 検索ベースでR² 0.27と低い
- 緑の草の下に隠れて見えにくい
- Height情報などの補助入力が必要

### 5. State予測の混同パターン

```
         Predicted →
True ↓   |   NSW |   Tas |   Vic |    WA |
------------------------------------------
NSW      |    57 |    11 |     7 |     0 |
Tas      |    12 |    91 |    13 |    22 |
Vic      |     2 |     8 |   102 |     0 |
WA       |     0 |    13 |     3 |    16 |
```

- **Tas ↔ WA** で混同が多い（Tas→WA: 22件）
- **Vic** は他Stateとほとんど混同しない（特徴的な景観）

---

## 今後の活用可能性

### 1. Retrieval-Augmented Prediction
- CNNの予測に検索結果を補助情報として追加
- 類似画像のTarget値を参考にした予測の補正

### 2. Embedding特徴量としての利用
- SigLIP embeddingをモデルの追加入力として使用
- ただし1152次元は大きいため、次元削減が必要

### 3. データ拡張の指針
- State間で共通するパターンを活かした拡張
- State固有の特徴を無視した汎用的なaugmentation

### 4. 異常検知・品質管理
- 検索結果の類似度が低い画像は予測が困難な可能性
- Out-of-distribution検出への応用

---

## 実験コード

- `EXPEMBEDDING/exp_siglip_retrieval.py`: グローバルインデックス実験
- `EXPEMBEDDING/exp_siglip_retrieval_by_state.py`: State別インデックス実験
