# EXP_SUMMARY: BirdCLEF+ 2026

## Competition Status

| 項目 | 値 |
|------|----|
| Best OOF CV | **0.9631** (EXP000/child-exp000) |
| Best LB | - (未提出) |
| Current Best Exp | EXP000/child-exp000 |

---

## 実験一覧

### EXP000: Perch 2.0 Baseline（Stage1 headのみ）

**概要**: Google Perch 2.0をbackboneとし、embeddingを事前計算してheadのみ学習するベースライン。

**設定** (`child-exp000`):

| パラメータ | 値 |
|------------|-----|
| Backbone | Google Perch 2.0 |
| Strategy | Stage1のみ（headのみ学習） |
| EPOCHS_HEAD | 20 |
| LR_HEAD | 1e-3 (Warmup 2epoch + CosineDecay) |
| BATCH_SIZE_HEAD | 512 |
| N_FOLDS | 5 (StratifiedKFold) |
| Data | train_audio (35,549) + train_soundscapes_labels (6,244) |
| Augmentation | ガウスノイズ + 時間シフト |
| Mixed Precision | FP16 |
| GPU | 2x (MirroredStrategy) |

**Results**:

| Fold | val_AUC |
|------|---------|
| Fold 0 | 0.9600 |
| Fold 1 | 0.9592 |
| Fold 2 | 0.9621 |
| Fold 3 | 0.9744 |
| Fold 4 | 0.9597 |
| **OOF Mean ± Std** | **0.9631 ± 0.0057** |

| LB Score | LB-CV Gap |
|----------|-----------|
| - (未提出) | - |

**Kaggle Models**: `wasabi777/birdclef2026-exp000/tensorFlow2/perch-v2-baseline`

**観察・考察**:
- Perch 2.0の事前学習済み特徴量が非常に強力で、headのみ学習でOOF 0.9631を達成
- Fold 3が0.9744と他foldより高め（データ分布の偏りの可能性）
- Stage 2（全体fine-tune）は9時間制限のため無効化
- 学習時間: 約1時間（embedding事前計算: fold毎に約10分、head学習: 約2分）

**トラブル・修正履歴**:
- Perch API: `model.infer()` → `signatures['serving_default']` に変更
- バッチ非対応: per-sampleループで対処
- Mixed Precision: Perch入力を `tf.cast(float32)` でキャスト
- Lazy Building: `head.build(input_shape=(None, 1280))` / ダミー入力で対処
- Optimizer: エポックごとに再生成 → ループ外で1度作成・LRのみ更新

---

## 得られた知見

- Perch 2.0のembeddingは非常に強力。headのみ学習でも高精度
- Stage 2（全体fine-tune）はKaggle 9時間制限に対して現状の実装では困難
- Fold間のスコア差（0.9592〜0.9744）はStratifiedKFoldの分割による偏りの可能性あり

---

## 次のステップ候補

1. **推論スクリプト (`infer.py`) の作成** → LBスコアの確認
2. **データ拡張の強化** → SpecAugment、MixUp等
3. **アンサンブル** → 5foldモデルの平均
4. **Soundscapeデータの活用拡大** → ラベルなしsoundscapeのPseudo Labeling（オンライン）
