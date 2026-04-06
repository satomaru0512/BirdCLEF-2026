# 調査計画: CV-LB Gap と State間トレードオフの解決

作成日: 2026-01-06

---

## 1. 現在の課題まとめ

### 1.1 実験結果の要約

| 実験 | OOF R² | LB | CV-LB Gap | 特徴 |
|------|--------|-----|-----------|------|
| child-exp000 (5F baseline) | 0.759 | **0.69** | -0.069 | 最良LB |
| child-exp036 (5F balanced) | 0.769 | 0.66 | -0.109 | NSW改善、Tas悪化 |
| child-exp037 (10F) | **0.785** | 0.67 | -0.115 | OOF最高、LB悪い |

### 1.2 構造的問題

```
問題1: State間トレードオフ
├── NSWを改善 → Tasが悪化
├── 理由: NSWは高Height/高Total、Tasは低Height/低Total
└── 同じモデルで両方を最適化できない

問題2: Dry_Dead_gの予測困難
├── 画像から「枯れ草」が見えにくい（緑の下に隠れる）
├── R² = 0.26 (最低)
└── Weight 10%だが全体スコアに影響

問題3: CV-LB Gap
├── CVが上振れしている（楽すぎる）
├── Testの分布がTrainと異なる可能性
└── Fold分割の工夫だけでは解決しない

問題4: 357枚の制約
├── State/季節の組み合わせが限定的
├── 汎化に必要なデータ多様性が不足
└── 過学習しやすい
```

### 1.3 テスト分布の仮説

論文の地図分析から:
- Tas: Trainに138枚（39%）だがTestには少ない可能性
- Vic/NSW: Testに多い可能性（地図上のサイト数が多い）

**しかし実験結果からは:**
- Tas/Vicの特定日付（9-11月）で悪化するとLBも下がる
- → TestにもTas/Vicの秋〜春サンプルが含まれる可能性

---

## 2. 調査の方向性

### 2.1 調査カテゴリ

```
A. Model Architecture
   ├── A1. Backbone比較
   ├── A2. Head設計
   └── A3. 補助入力（Height等）

B. Data Strategy
   ├── B1. Augmentation強化
   ├── B2. サンプリング戦略
   └── B3. 外部データ活用

C. Training Pipeline
   ├── C1. Loss function設計
   ├── C2. 2-Stage学習
   └── C3. Ensemble戦略

D. Post-processing
   ├── D1. State別補正
   └── D2. 物理的制約の適用
```

---

## 3. 優先度付き調査項目

### 【高優先度】すぐ試すべき

#### A3. Height情報の補助入力
**仮説**: 画像だけでは草丈がわからない → バイオマス予測が困難
**調査内容**:
- Height_Ave_cm をモデルの補助入力として追加
- 画像特徴 + Height → MLP → 予測
**期待効果**: Dead/Total の予測改善（特に高草丈サンプル）
**実装難易度**: 低（train.pyの修正のみ）
**注意**: 推論時にHeightが必要 → test.csvにHeight情報があるか確認

#### C1. Loss function の State/Target重み付け
**仮説**: 現在のLossはState間で均等 → 少数派Stateが学習不足
**調査内容**:
- State別サンプル重み（NSW: 1.5, Tas: 0.8 など）
- Target別重みの調整（Dead重視）
**期待効果**: State間トレードオフの緩和
**実装難易度**: 低

#### B1. Augmentation強化（State多様性）
**仮説**: 季節/地域の見た目の違いをAugmentationで吸収できる
**調査内容**:
- Color Jitter強化（季節による色味の違い）
- CutMix/MixUp（異なるStateの画像を混合）
- Random Erasing（草の一部を隠して汎化性向上）
**期待効果**: State依存性の低減
**実装難易度**: 中

---

### 【中優先度】効果が期待できる

#### A1. Backbone比較
**現状**: EVA02-CLIP-L-14-336
**調査内容**:
| Backbone | 特徴 | 期待 |
|----------|------|------|
| ConvNeXt-V2 | ImageNet-22k pretrained | 自然画像に強い |
| DINOv2 | Self-supervised | 汎化性能高い |
| SigLIP | Google製CLIP | 異なるpretrain |
| Swin-V2 | Shifted window | 高解像度に強い |

**優先**: DINOv2（self-supervisedで汎化に期待）

#### C2. 2-Stage学習
**仮説**: State/季節を考慮した2段階予測
**調査内容**:
```
Stage 1: State分類器を学習（画像→State推定）
Stage 2: State条件付きバイオマス予測
  - State embedding をモデルに追加
  - または State別のHead
```
**期待効果**: State固有のパターンを学習
**実装難易度**: 高

#### D1. 推論時State別補正
**仮説**: CVでState別のバイアスがわかっている
**調査内容**:
- OOFからState別の平均誤差を計算
- 推論時にState推定 → 補正適用
**問題**: Testでは正確なState推定が必要
**実装難易度**: 中

---

### 【低優先度/探索的】

#### A2. Head設計の見直し
- 5-head → 3-head + 制約（Dead = Total - GDM）
- Attention mechanism for target interaction

#### B2. サンプリング戦略
- Oversampling NSW
- Undersampling Tas

#### B3. 外部データ活用
- 類似のpasture datasetを探す
- pretrain on external → finetune

#### C3. Ensemble戦略
- 異なるbackboneのensemble
- State別モデルのensemble

---

## 4. 具体的な調査手順

### Phase 1: データ確認（1日）

```
□ test.csvにHeight情報があるか確認
□ train/testの画像の視覚的な違いを確認
□ 既存実験のOOF誤差パターンを可視化
```

### Phase 2: Quick Wins（2-3日）

```
□ Height補助入力の実装・実験
□ Loss重み付けの実験
□ Augmentation強化の実験
```

### Phase 3: Backbone探索（3-5日）

```
□ DINOv2で実験
□ ConvNeXt-V2で実験
□ 結果比較・最良backbone選定
```

### Phase 4: 2-Stage/Post-processing（3-5日）

```
□ State分類器の学習・精度確認
□ 2-Stage pipelineの実装
□ 推論時補正の効果検証
```

---

## 5. 357枚で「State/季節に依存しないモデル」を作る方法

### 5.1 なぜ難しいか

```
NSW: 75枚 (6日, 5 Species)
Vic: 112枚 (11日, 7 Species)
Tas: 138枚 (10日, 4 Species)
WA: 32枚 (3日, 4 Species)

→ State × 日付 × Species の組み合わせが疎
→ 見たことのない組み合わせへの汎化が困難
```

### 5.2 可能なアプローチ

#### (1) Augmentationで多様性を増やす
- ColorJitter: 季節による色味の違いを吸収
- MixUp/CutMix: 異なるStateの特徴を混合
- Style Transfer: State間のスタイル変換（実験的）

#### (2) 不変表現の学習
- DINOv2: Self-supervisedでState/季節に依存しない特徴を学習済み
- Contrastive Learning: 同じバイオマス量の画像を近づける

#### (3) ドメイン適応
- 全Stateで共通する特徴にフォーカス（緑の量、テクスチャ）
- State固有の特徴を抑制

#### (4) メタデータの活用
- Height, NDVI を補助入力として使う
- 画像だけでは判断できない情報を補完

---

## 6. 次のアクション

### 今すぐやること

1. **test.csvの確認**: Height情報があるか
2. **Height補助入力の実験設計**: child-exp038として準備
3. **Loss重み付けの実験設計**: child-exp039として準備

### 調査ノートの更新

各調査の結果は以下に記録:
- `docs/Idea_Research/20260106_HEIGHT_INPUT.md`
- `docs/Idea_Research/20260106_LOSS_WEIGHTING.md`
- `docs/Idea_Research/20260106_BACKBONE_COMPARISON.md`

---

## 7. 参考: 他のKaggle上位解法

類似コンペ（画像からバイオマス/収量予測）の上位解法を調査:
- Auxillary features（メタデータ）の活用
- Multi-task learning
- Test-Time Augmentation の工夫
- Pseudo labeling（オンライン）

---

**結論**: Fold分割の工夫は限界に達した。次はモデル/データ/パイプラインの改善にフォーカスすべき。
