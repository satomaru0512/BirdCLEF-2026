# CSIRO Pasture Biomass Prediction - Experiment Summary

## TL;DR（要約）
- 🎉🎉 **現在のベスト: EXP060-child058**（**DinoV3 ViT-Huge+** + **Artem CV strategy**）→ **LB 0.73** ⭐⭐⭐ **CURRENT BEST** / OOF CV **0.792** / Gap **-0.062** 🎉🎉
  - CV strategyの変更が**+0.02** LB改善！
  - stratify: Clover/Dead presence (4 categories), group: day/month_State
- 前ベスト: **EXP060-child038**（DinoV3 ViT-Huge+ baseline）→ **LB 0.71** / OOF CV **0.776** / Gap **-0.066**
- **EXP060-child054**（DinoV3 512px）→ **LB 0.71** / OOF CV **0.773** / Gap **-0.063** (90th)
- **EXP060-child037**（10-fold CV）→ **LB 0.67** / OOF CV **0.785** / Gap **-0.115**（10-foldはOverfit）
- **EXP106-child000**（Persistent Optimizer）→ **LB 0.70** / CV **0.786** / Gap **-0.086**（Optimizer再初期化が重要、維持は逆効果）
- **EXP101-child000**（DinoV3 + Mamba Fusion）→ **LB 0.62** / CV **0.749** / Gap **-0.129**（Mamba Fusionは効果なし）
- **重要な発見**: 評価関数の修正でCV/LBギャップが大幅改善（-0.083 → -0.069）
  - 旧（EXP036）: `.mean()`を使用した近似計算
  - 新（EXP060）: `.sum()`を使用した正しいKaggle公式メトリック
- うまく行ったこと:
  - **評価関数の修正**（EXP060）→ CV/LBギャップが改善、CVがLBにより近い指標に。
  - **EVA-CLIP × 5-head × Consistency Loss**（EXP036/060）→ LB 0.69（最高記録）
  - 強いバックボーン（EVA-CLIP）× 50% Gradual Unfreeze（EXP030/036/060）。
  - SigLip系では5ヘッド + Consistency Loss（EXP014）でLB 0.66（SigLip最良）。
  - 低LR(0.003) + 長め学習(40ep)、過度でない正則化（EVA-CLIPはdropout=0.1最適）。
  - **デフォルトのaugmentation（RRC scale=0.85-1.0, Hue=0.15）がテスト分布に最適**。
  - **[CLS]トークンがPatch Tokenより優れている**（EXP051/052で確認）。
- うまく行かなかったこと:
  - 高LRや強すぎる正則化、MixUp過剰（EXP008-001/003）。
  - **Augmentation改善の試み（EXP030-011）はLB 0.64で大幅悪化** - RRC制限・Hue制限が逆効果。
  - Mosaic/Mixup（EXP030-010）はLB 0.64で悪化。
  - 70% Unfreeze（EVA-CLIP, EXP030-001）は悪化、3-crop（EXP031）は致命的に悪化。
  - 補助タスク（Metadata/Species, EXP021/022; **Height回帰, EXP048**; **草種分類, EXP061**）や手工特徴（ExG/HSV, EXP026）はLB低下。
  - consistency_weight増加（0.1→0.2）でLB 0.65（-0.04）、Loss weights変更でLB 0.63（-0.06）と大幅悪化（EXP036-003/004）。
  - **Patch Token Pooling（EXP051/052）は深刻な過学習** - CV 0.793でLB 0.64（Gap -0.153）。PatchDropout/Attentionでも改善せず（LB 0.66止まり）。
  - **Split label補正（EXP077）**: 左右分割学習のラベルノイズ低減を狙ったが、**CV 0.77 に対して LB 0.63 まで大幅悪化**（Gap -0.14）→ 公開LBへの汎化に失敗。
  - **外部データStage1 encoder init（EXP080/081）**: EXP080はCVリークの疑いが強く、LBも0.67止まり。リーク回避したEXP081でも**LB 0.65へ悪化**（負の転移）。
  - **Log変換（EXP082）**: 論文記載のlog(1+y)変換を適用したが、**LB 0.65（-0.04）と大幅悪化**。Kaggle評価がraw空間のため、log空間での最適化は逆効果。Cloverは改善（R² 0.40→0.60）したがTotal/GDM/Greenが悪化。
  - **DinoV3ハイパラチューニング（EXP060-child039〜055）**: LR低下、LLRD、EMA、unfreeze比率変更、loss変更など試行したが**すべてexp038より悪化またはLB低下**。exp038は局所最適に到達している。
  - **EXP060追加検証（child066〜069）**: DINOv2 Giant 518 / MSE pred loss / sum-loss / Soft-Species Conditioning を試したが **LB 0.67〜0.71** で **exp058(LB 0.73) を超えず**（いずれもGapが大きめ）。
  - **Depth後処理補正（ENSEXP_003/004）**: 線形補正(ENSEXP_003)はLB 0.65、パーセンタイルベース補正(ENSEXP_004)はLB 0.66と、両方ともEXP060単体(0.69)より悪化。後処理でのDepth補正は汎化しない。
  - **推論TTA（EXP060-child000）**: Crop-TTA（5-crop zoom）/ Photo-TTA（brightness/contrast）ともに **LB 0.69から改善せず**（むしろ僅かに悪化）。Sampling_Date差の吸収は推論TTAだけでは難しい。
  - **DinoV3 + EVA02-CLIP アンサンブル（ENSEXP_006）**: OOF最適化で+0.019改善したが、**LB 0.71（-0.02悪化）**。OOFベースの最適化はLBに転移しない。exp058単体がベスト。
  - **exp059（768px + Artem CV）**: CV 0.800（+0.008）だが**LB 0.70（-0.03悪化）**。768pxはWA性能を大幅低下させ過学習。
- 注意点: CVが高くてもLBが上がらない例が多く、分布差が顕著。**デフォルトaugmentationを変えるとLB悪化するリスク大**。**OOFベースのアンサンブル最適化も危険**。
- **KEY INSIGHT**:
  - [CLS]トークン > Patch Token（375枚では576パッチの位置パターンを記憶してしまう）
  - **正しい評価関数を使うことでCV/LBギャップを削減可能**
  - **OOFベースのアンサンブル最適化はLBに転移しない** - exp058単体 (LB 0.73) がベスト
- **EXP069 (depth fusion, this work)**: EXP060のRGB最強構成を維持しつつ、DepthAnything3のnpyマップをEVA埋め込みとLate Fusion。Height_Ave_cmを補助タスクにしてGreen/GDM/Totalのスケールを正則化し、Sampling_Dateごとの撮影高度差を吸収する狙い。CV設定・fold割り・optimizer等はEXP060から一切変えず、モデリングとlossのみ更新。
- **EXP069-child001 (height-bin strat + density loss)**: Sampling_Dateグループを保ちつつ State×高さビン(0–5/5–15/15+cm)でstratify を強化し、Fold0のように高背丈が固まるのを回避。同時に Green / Height 密度ロスで高さ依存の残差を直接抑制。

## Overview
This document summarizes experiments conducted for the CSIRO Pasture Biomass Prediction competition. The goal is to predict five biomass components from pasture images using deep learning.

**Evaluation Metric**: Globally weighted R² across all predictions
- Dry_Green_g: 10% weight
- Dry_Dead_g: 10% weight
- Dry_Clover_g: 10% weight
- GDM_g: 20% weight
- Dry_Total_g: 50% weight

## Dataset & Evaluation Context

**Training Set**: 375 images
**Test Set**: 800+ images
**Public/Private Split**: 53% / 47% (~424 public, ~376 private)

**Important Considerations:**
- **CV-LB Balance**: Both CV and LB scores must be evaluated carefully due to test set size (800 vs 375 train)
- **Distribution Shift**: Public and private test sets have different distributions (confirmed by competition host)
  - Training data: Various seasons and states throughout the year
  - Test data: Mix of overlapping and non-overlapping time periods/locations
  - Public test: Some overlap with training periods
  - Private test: Includes data from non-overlapping periods to test generalization
- **Strategic Insight**: Models must balance fitting training data and generalizing to unseen temporal/geographic patterns

---

## Experiment Timeline

### EXP007: SigLip Baseline with GroupKFold
**Model**: SigLip (google/siglip-so400m-patch14-384) with frozen embeddings + trainable MLP heads

#### EXP007-child000 (Baseline)
**Configuration:**
- Architecture: Two-stream (left/right image split), shared frozen SigLip backbone
- Predicted targets: Dry_Green_g, GDM_g, Dry_Total_g (3 heads)
- Derived targets: Dry_Dead_g, Dry_Clover_g (calculated)
- Cross-validation: 5-Fold GroupKFold (grouped by Sampling_Date + State)
- Training: 20 epochs, LR=0.005, batch_size=16, dropout=0.1, weight_decay=0.0
- Augmentation: Basic (HorizontalFlip, VerticalFlip, RandomRotate90, ColorJitter)

**Results:**
- OOF CV: 0.70
- LB: **0.61**
- LB-CV Gap: -0.09
- Mean R²: 0.70 ± 0.06

**Key Observations:**
- Baseline performance established
- Significant overfitting (LB-CV Gap = -0.09)
- Fold 3/4 consistently underperform (R² ~0.62-0.65)

---

### EXP008: SigLip with Enhanced Augmentation
**Model**: Same architecture as EXP007, with enhanced data augmentation

#### EXP008-child000: Increased Epochs + Enhanced Augmentation
**Configuration:**
- Architecture: Same as EXP007
- Training: **30 epochs** (up from 20), LR=0.005, dropout=0.1, weight_decay=0.0
- Augmentation: **Enhanced**
  - RandomResizedCrop (scale=0.85-1.0, preserves grass amount)
  - Stronger ColorJitter (brightness/contrast/saturation=0.3, hue=0.15)
  - RandomRotation (±10°)
  - GaussianBlur, RandomGamma, RandomShadow, RandomToneCurve

**Results:**
- OOF CV: 0.73
- LB: **0.63** (+0.02 from baseline)
- LB-CV Gap: -0.10 (worse than baseline)
- Mean R²: 0.71 ± 0.06

**Key Observations:**
- CV improved (+0.03) but LB improvement was modest (+0.02)
- Overfitting increased (Gap: -0.09 → -0.10)
- More epochs helped CV but not generalization

#### EXP008-child001: Higher LR + Weight Decay
**Configuration:**
- Training: 30 epochs, **LR=0.01** (increased), dropout=**0.2**, **weight_decay=0.01**
- Augmentation: Same as child-exp000

**Results:**
- OOF CV: 0.64
- LB: **0.55** (-0.08 from baseline)
- LB-CV Gap: -0.09
- Mean R²: 0.63 ± 0.04

**Key Observations:**
- ❌ **Failed experiment**: Over-regularization
- Train loss plateaued at 5.0-5.5 (convergence issues)
- Both CV and LB performance degraded significantly
- LR=0.01 + WD=0.01 was too aggressive

#### EXP008-child002: Lower LR + Longer Training ⭐ **Best Overall Model**
**Configuration:**
- Training: **40 epochs**, **LR=0.003** (reduced), dropout=0.15, weight_decay=0.001
- Augmentation: Same as child-exp000

**Results:**
- OOF CV: 0.72
- LB: **0.64** (+0.03 from baseline) ⭐
- **LB-CV Gap: -0.08** (good generalization)
- Mean R²: 0.71 ± 0.07

**Key Observations:**
- ✅ **Best overall performance** (highest LB)
- Slower, more careful training improved generalization
- LB-CV Gap improved from -0.10 to -0.08
- Train Loss: 7.03 → 3.60 (smooth convergence)
- **Sweet spot** for regularization: dropout=0.15, WD=0.001

#### EXP008-child003: Maximum Regularization + Scheduler + MixUp
**Configuration:**
- Training: **40 epochs**, LR=0.003, **dropout=0.25**, **weight_decay=0.005**
- **Early Stopping**: patience=10
- **Scheduler**: CosineAnnealingWarmRestarts (T_0=10, T_mult=2, warmup=3)
- **MixUp**: alpha=0.2 (preserves expected grass amount)
- Augmentation: Same enhanced as child-exp000

**Results:**
- OOF CV: 0.65
- LB: **0.62** (same as DINOv3 baseline)
- **LB-CV Gap: -0.03** (best gap achieved!)
- Mean R²: 0.64 ± 0.06

**Key Observations:**
- ❌ **Over-regularization**: Both CV (-0.07) and LB (-0.02) dropped significantly
- ✅ **Best LB-CV Gap**: -0.03 (vs -0.08 in child-002)
- Fold 4 severely underperformed (R² 0.54 vs 0.65+ in other experiments)
- **Trade-off**: Better generalization but lower absolute performance
- **Analysis**: dropout=0.25 + WD=0.005 + MixUp was too aggressive
- **Lesson**: Regularization sweet spot is dropout=0.15, WD=0.001 (child-002)

---

### EXP009: DINOv3 ConvNeXt Baseline
**Model**: DINOv3 ConvNeXt (timm: convnext_base.dinov3_lvd1689m) with frozen embeddings + trainable MLP heads

#### EXP009-child000: DINOv3 Baseline
**Configuration:**
- Architecture: Two-stream (left/right image split), frozen DINOv3 ConvNeXt backbone
- Predicted targets: Dry_Green_g, GDM_g, Dry_Total_g (3 heads)
- Cross-validation: 5-Fold GroupKFold (grouped by Sampling_Date + State)
- Training: 30 epochs, LR=0.005, batch_size=16, dropout=0.1, weight_decay=0.0
- Augmentation: Same enhanced augmentation as EXP008

**Results:**
- OOF CV: 0.72
- LB: **0.62** (+0.01 from EXP007 baseline)
- LB-CV Gap: -0.10
- Mean R²: 0.70 ± 0.06

**Key Observations:**
- DINOv3 performance similar to SigLip baseline
- No significant improvement over EXP007
- Similar overfitting pattern (Gap = -0.10)
- **Inference complexity**: Uses `pretrained=False` approach (no HF download needed)

---

### EXP010: SigLip with StratifiedGroupKFold ⭐ **Improved CV Strategy**
**Model**: SigLip (google/siglip-so400m-patch14-384) with frozen embeddings + trainable MLP heads

#### EXP010-child000: StratifiedGroupKFold for Better CV-LB Correlation
**Configuration:**
- Architecture: Same as EXP008-child002 (best model)
- Predicted targets: Dry_Green_g, GDM_g, Dry_Total_g (3 heads)
- **Cross-validation**: 5-Fold **StratifiedGroupKFold** ✅
  - **Stratify by**: State (ensures balanced target distribution across folds)
  - **Group by**: Sampling_Date (prevents data leakage from same date)
  - **Rationale**: Based on competition host's guidance (Discussion #615003)
- Training: 40 epochs, LR=0.003, dropout=0.15, weight_decay=0.001
- Augmentation: Enhanced (same as EXP008)
- Batch size: 16

**Results:**
- OOF CV: **0.72** (same as EXP008-002)
- LB: **0.63** (+0.02 from EXP007 baseline)
- **LB-CV Gap: -0.09** (same as EXP008-002)
- Mean R²: 0.73 ± 0.08
- Fold R² scores: [0.59, 0.75, 0.82, 0.73, 0.76]

**Key Observations:**
- ✅ **CV strategy improved**: StratifiedGroupKFold ensures balanced State distribution
- ⚠️ **No LB improvement**: Same score as EXP008-002 (LB 0.63 vs 0.64)
- **Fold imbalance**: Fold distribution uneven (52-101 images per fold) due to group constraints
- Fold 0 still underperforms (R² 0.59 vs 0.73-0.82 for other folds)
- **State distribution per fold now balanced** (stratification working as intended)
- **Analysis**: Better CV strategy didn't translate to LB improvement in this case
- **Hypothesis**: EXP008-002 was already well-regularized, or test set has different distribution

---

### EXP012: SigLip with Gradual Layer-wise Unfreezing ⭐ **Best CV Score**
**Model**: SigLip (google/siglip-so400m-patch14-384) with gradual backbone unfreezing strategy

#### EXP012-child000: Gradual Unfreezing from Top Layers (SigLip v1)
**Configuration:**
- Architecture: Same as EXP010 (SigLip with 3 heads)
- Model: google/siglip-so400m-patch14-384
- Predicted targets: Dry_Green_g, GDM_g, Dry_Total_g (3 heads)
- Cross-validation: 5-Fold StratifiedGroupKFold (stratify=State, group=Sampling_Date)
- **Training Strategy**: Gradual layer-wise unfreezing ✨
  - **Epochs 0-4 (first 5 epochs)**: Full backbone freeze, train only MLP heads (LR=0.003)
  - **Epochs 5-39**: Gradually unfreeze layers from top to bottom
    - Epoch 5: Last 2 layers (backbone_lr=1e-5)
    - Epoch 10: ~4 layers (backbone_lr=1.5e-5)
    - Epoch 20: ~8 layers (backbone_lr=2.5e-5)
    - Epoch 39: ~14 layers / 50% (backbone_lr=5e-5)
  - **Discriminative learning rates**: Heads (0.003) vs Backbone (1e-5 → 5e-5)
- Augmentation: Enhanced (same as EXP008/010)
- Batch size: 16, dropout: 0.15, weight_decay: 0.001

**Results:**
- OOF CV: **0.76** (+0.04 from EXP010) ⭐ **Highest CV achieved**
- LB: **0.64** ⭐ **NEW BEST** (higher rank than EXP008-002 in decimal precision)
- **LB-CV Gap: -0.12** (worse than frozen backbone)
- Mean R²: 0.76 ± 0.06
- Fold R² scores: [0.66, 0.76, 0.86, 0.78, 0.76]

**Key Observations:**
- ✅ **Highest CV score**: OOF 0.76 vs 0.72 (EXP010 frozen baseline, +0.04)
- ✅ **LB improved over EXP010**: 0.64 vs 0.63 (+0.01 from frozen StratifiedGroupKFold baseline)
- ⭐ **NEW BEST LB**: 0.64 (displayed), but **higher leaderboard rank** than EXP008-002
  - Both show LB 0.64, but decimal precision favors EXP012
  - **Practical best model** for submission
- ⚠️ **Increased overfitting**: LB-CV Gap increased from -0.09 (EXP010) to -0.12
- **Fold 0 improved**: R² 0.66 vs 0.59 (EXP010), suggesting unfreezing helps weaker folds
- **Fold 2 excellent**: R² 0.86 (best single-fold score across all experiments)
- **Trade-off**: Better CV fitting (+0.04) with slightly worse generalization (Gap -0.09 → -0.12)
- **Analysis**: Gradual unfreezing improves both CV and LB over frozen baseline (EXP010)
- **Comparison with EXP008-002**: Same displayed LB (0.64) but EXP012 ranks higher

**Technical Notes:**
- ⚠️ **Inference speed issue discovered**: Unfrozen layers saved with `requires_grad=True` → 40min inference (vs 32min)
- ✅ **Fixed**: Modified `train.py` to freeze all layers before saving checkpoint
- Unfreezing schedule uses exponential growth (more aggressive in later epochs)
- SigLip has 27 vision encoder layers; we unfreeze up to 14 (~52%)

#### EXP012-child001: SigLip v2 with patch14-384
**Configuration:**
- **Model change**: google/siglip2-so400m-patch14-384 (SigLip v2, same input size)
- All other settings: Same as child-exp000
- Training Strategy: Same gradual unfreezing (5 epochs frozen, then 50% unfreeze)

**Results:**
- OOF CV: **0.761** (Mean R²: 0.763 ± 0.066)
- LB: **0.65** (+0.01 from child-000)
- **LB-CV Gap: -0.111**
- Fold R² scores: [0.66, 0.73, 0.86, 0.77, 0.80]

**Key Observations:**
- ✅ **LB improved**: +0.01 from child-000 (0.64 → 0.65)
- ✅ **SigLip v2 advantage**: Newer model version shows better performance
- ⚠️ **Gap similar**: -0.111 vs -0.12 (child-000), slight improvement
- ⭐ **CV similar**: 0.761 vs 0.76 (child-000), but slightly better LB
- **Fold consistency**: Similar pattern to child-000 (Fold 2 best at 0.86)
- **Conclusion**: SigLip v2 shows modest improvement over v1 with same training strategy

#### EXP012-child002: SigLip v2 with patch16-512
**Configuration:**
- **Model change**: google/siglip2-so400m-patch16-512 (SigLip v2, larger input size)
  - Input size: 512×512 (vs 384×384 in child-001)
  - Patch size: 16×16 (vs 14×14 in child-001)
- All other settings: Same as child-exp000/001
- Training Strategy: Same gradual unfreezing (5 epochs frozen, then 50% unfreeze)

**Results:**
- OOF CV: **0.752** (Mean R²: 0.758 ± 0.072)
- LB: **0.63** (-0.03 from child-001)
- **LB-CV Gap: -0.122**
- Fold R² scores: [0.63, 0.74, 0.85, 0.78, 0.79]

**Key Observations:**
- ❌ **LB degraded**: -0.03 from child-001 (0.66 → 0.63)
- ⚠️ **Worse generalization**: Gap worsened from -0.101 to -0.122
- ⚠️ **CV also dropped**: 0.752 vs 0.761 (child-001)
- **Fold 0 worse**: R² 0.63 vs 0.66 (child-001), weak fold more affected
- **Analysis**: Larger input size (512) doesn't help with 2000×1000 images
  - Possible overfitting to higher resolution details
  - Patch16-512 may not be optimal for grass biomass prediction
- **Conclusion**: patch14-384 (child-001) is better than patch16-512 for this task

---

### EXP013: SigLip with 8-Tile Splitting (500x500) ❌ **Abandoned**
**Model**: SigLip with 8-tile splitting strategy + gradual unfreezing

#### EXP013-child000: 8-Tile Splitting Experiment (Abandoned)
**Configuration:**
- Architecture: Same as EXP012 but with 8-tile splitting
- **Tile strategy**: 2000x1000 → 8 tiles of 500x500 (4 cols x 2 rows)
  - Tile layout: [0][1][2][3] (top row), [4][5][6][7] (bottom row)
  - Labels divided by 8 (biomass is absolute quantity)
  - All tiles from same image stay in same fold
- Predicted targets: Dry_Green_g, GDM_g, Dry_Total_g (3 heads)
- Cross-validation: 5-Fold StratifiedGroupKFold (stratify=State, group=Sampling_Date)
- Training: 40 epochs planned, gradual unfreezing (same as EXP012)
- Augmentation: Enhanced (adapted for 500x500)
- Batch size: 16 (but 8x more samples per epoch vs EXP012)

**Results:**
- **Training stopped at Fold 0** (did not complete)
- Reason: Long training time with insufficient CV improvement
- No LB submission

**Key Observations:**
- ❌ **Training efficiency issue**: 8x more samples → significantly longer training time
- ❌ **No CV improvement**: Fold 0 did not show promising CV gains vs EXP012
- **Cost-benefit analysis failed**: Increased computation cost not justified by performance
- **Conclusion**: 2-tile splitting (EXP012) is more efficient
- **Hypothesis**: Smaller tiles (500x500) may lose important spatial context
- **Alternative approach**: Keep 2-tile strategy, focus on other improvements

**Technical Notes:**
- 8 tiles per image: 4x more tiles than EXP012
- Each epoch processes 8x more samples (but each sample is smaller)
- Inference would also be slower (8 tiles vs 2 tiles per image)
- Implementation complete but not fully trained

---

### EXP014: SigLip with 5-Head Multi-Task Learning + Consistency Loss ⭐ **NEW BEST LB**
**Model**: SigLip (google/siglip-so400m-patch14-384) with 5-head architecture and physical constraint consistency

#### EXP014-child000: 5-Head Multi-Task Learning + Consistency Loss ⭐
**Configuration:**
- **Architecture change**: 5 independent MLP heads (vs 3 heads in EXP012)
- **Predicted targets**: All 5 targets directly predicted
  - Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g (5 heads)
  - **vs EXP012**: Only 3 targets (Dry_Green_g, GDM_g, Dry_Total_g)
- **Consistency Loss** (NEW): Enforces physical constraints ✨
  - Constraint 1: Dry_Dead_g + GDM_g ≈ Dry_Total_g
  - Constraint 2: Dry_Clover_g + Dry_Green_g ≈ GDM_g
  - Consistency weight: 0.1
- Cross-validation: 5-Fold StratifiedGroupKFold (stratify=State, group=Sampling_Date)
- **Training Strategy**: Same as EXP012
  - Gradual unfreezing: 50% of backbone (max_unfreeze_ratio: 0.5)
  - Epochs 0-4: Backbone frozen
  - Epochs 5-39: Gradual unfreezing
  - Backbone LR: 1e-5 → 5e-5, Head LR: 0.003
- Augmentation: Enhanced (same as EXP012)
- Batch size: 16, dropout: 0.15, weight_decay: 0.001

**Results:**
- OOF CV: **0.772** (Mean R²: 0.771 ± 0.051)
- LB: **0.66** ⭐ **NEW BEST** (+0.02 from EXP012)
- **LB-CV Gap: -0.112** (similar to EXP012: -0.12)
- Fold R² scores: [0.69, 0.75, 0.85, 0.77, 0.79]

**Key Observations:**
- ✅ **NEW BEST LB**: 0.66 (+0.02 from EXP012's 0.64)
- ✅ **Strong CV**: OOF 0.772 (vs 0.76 in EXP012, +0.012)
- ✅ **5-head architecture works**: Predicting all 5 targets directly > 3-head + calculation
- ✅ **Consistency loss as regularization**: Physical constraints improve generalization
- **LB-CV Gap stable**: -0.112 (similar to EXP012's -0.12)
- **Fold 0 improved**: R² 0.69 vs 0.66 (EXP012)
- **Fold 2 excellent**: R² 0.85 (consistent with EXP012's 0.86)
- **Key insight**: Multi-task learning with explicit constraints > implicit relationships
- **Practical impact**: +0.02 LB improvement without changing training strategy

**Technical Notes:**
- 5 separate MLP heads vs 3 in EXP012
- Consistency loss weight = 0.1 (tunable hyperparameter)
- No derived calculations during training (all targets predicted)
- Physical constraints learned explicitly during training

#### EXP014-child000 (TTA Added): Test Time Augmentation Experiment
**Configuration:**
- Same as child-000 baseline
- **Inference change**: Added TTA with 3 views (original, hflip, vflip)
- All training parameters identical to child-000

**Results:**
- OOF CV: **0.772** (Mean R²: 0.771 ± 0.051)
- LB (no TTA): **0.66**
- **LB (with TTA): 0.66** (same score, but higher Public LB rank)
- **LB-CV Gap: -0.112**
- Fold R² scores: [0.69, 0.75, 0.85, 0.77, 0.79]

**Key Observations:**
- ⚠️ **No LB improvement**: TTA didn't change the displayed LB score
- ✅ **Public rank improved**: Same LB 0.66 but ranked 4th vs 6th (without TTA)
- **Inference time increased**: TTA processes 3x more images
- **Conclusion**: TTA improves stability/decimal precision but not displayed LB

#### EXP014-child004: 70% Backbone Unfreezing
**Configuration:**
- **Key change**: max_unfreeze_ratio: 0.5 → 0.7 (70% of layers unfrozen)
- Hypothesis: More layers may improve capacity without severe overfitting
- SigLip has 27 layers: 70% = ~19 layers unfrozen (vs 14 in child-000)
- All other parameters same as child-000

**Results:**
- OOF CV: **0.761** (Mean R²: 0.762 ± 0.064)
- LB: **Not yet submitted**
- Fold R² scores: [0.66, 0.73, 0.86, 0.78, 0.78]

**Key Observations:**
- ⚠️ **CV degraded**: 0.761 vs 0.772 (child-000, -0.011)
- **Fold 0 similar**: R² 0.66 vs 0.69 (child-000)
- **Awaiting LB submission** to compare with child-005

#### EXP014-child005: 70% Backbone Unfreezing ⭐ **HIGHEST PUBLIC LB RANK**
**Configuration:**
- **Key change**: max_unfreeze_ratio: 0.5 → 0.7 (70% of layers unfrozen)
- Same as child-004 (both use 70% unfreezing)
- All other parameters same as child-000

**Results:**
- OOF CV: **0.759** (Mean R²: 0.761 ± 0.059)
- LB: **0.66** ⭐ **HIGHEST PUBLIC LB RANK** (Ranked #1 among all LB 0.66 submissions)
- **LB-CV Gap: -0.099** (better than child-000: -0.112)
- Fold R² scores: [0.67, 0.74, 0.85, 0.77, 0.77]

**Key Observations:**
- ⭐ **BEST PUBLIC LB RANK**: LB 0.66 ranked #1 (vs child-000 ranked #6)
- ✅ **Gap improved**: -0.099 vs -0.112 (child-000, +0.013 improvement)
- ⚠️ **CV degraded**: 0.759 vs 0.772 (child-000, -0.013)
- **Trade-off validated**: Lower CV but better LB rank (better generalization to Public test)
- **Fold 0 similar**: R² 0.67 vs 0.69 (child-000, -0.02)
- **Fold 2 consistent**: R² 0.85 (same as child-000)
- **Key insight**: **70% unfreezing > 50% unfreezing** for Public LB
- **Practical impact**: Same displayed LB but significantly higher rank (decimal precision)

**Analysis:**
- **More unfreezing helps Public LB**: 70% provides better capacity without severe overfitting
- **Gap improvement**: -0.099 is better than -0.112 (child-000) and -0.107 (EXP021)
- **CV-LB trade-off**: Lower CV (0.759) doesn't mean worse LB - Public test favors this model
- **Unfreezing ratio sensitivity**: 50% → 70% improves Public LB rank despite lower CV
- **Conclusion**: **EXP014-child005 is the new best submission** for Public LB

---

### EXP016: SigLip with Focal Loss for Hard Samples ❌ **CV Improved but LB Degraded**
**Model**: SigLip (google/siglip-so400m-patch14-384) with Focal MSE Loss

#### EXP016-child000: Focal Loss Experiment
**Configuration:**
- **Architecture**: Same as EXP012 (SigLip with 3 heads)
- **Predicted targets**: Dry_Green_g, GDM_g, Dry_Total_g (3 heads, same as EXP012)
- **Loss Function Change** (NEW): Focal MSE Loss ✨
  - Down-weights easy samples (small errors)
  - Emphasizes hard samples (large errors)
  - Focal gamma: 2.0 (standard focal loss parameter)
  - Adaptive loss weighting based on prediction error
- Cross-validation: 5-Fold StratifiedGroupKFold (stratify=State, group=Sampling_Date)
- **Training Strategy**: Same as EXP012
  - Gradual unfreezing: 50% of backbone (max_unfreeze_ratio: 0.5)
  - Epochs 0-4: Backbone frozen
  - Epochs 5-39: Gradual unfreezing
  - Backbone LR: 1e-5 → 5e-5, Head LR: 0.003
- Augmentation: Enhanced (same as EXP012)
- Batch size: 16, dropout: 0.15, weight_decay: 0.001

**Results:**
- OOF CV: **0.781** (Mean R²: 0.776 ± 0.048)
- LB: **0.64** ❌ (-0.02 from EXP014)
- **LB-CV Gap: -0.141** (worse than EXP012/014: -0.12/-0.112)
- Fold R² scores: [0.73, 0.75, 0.87, 0.76, 0.77]

**Key Observations:**
- ❌ **LB degradation**: 0.64 vs 0.66 (EXP014) despite higher CV
- ✅ **Highest CV ever**: OOF 0.781 (+0.009 from EXP014, +0.021 from EXP012)
- ⚠️ **Increased overfitting**: LB-CV Gap worsened from -0.12 (EXP012) to -0.141
- **Fold 0 improved on CV**: R² 0.73 vs 0.69 (EXP014) vs 0.66 (EXP012)
- **Fold 2 peaked**: R² 0.87 (highest single-fold score across all experiments)
- **Best fold consistency**: std 0.048 (lowest variance across all experiments)
- **Key lesson**: Focal loss improves CV by focusing on hard samples but overfits to training distribution
- **Conclusion**: Standard MSE loss (EXP012) better balances training vs test performance
- **Trade-off**: Better CV fitting on hard samples doesn't improve LB (distribution shift)

**Analysis:**
- Focal loss successfully improves fold consistency and CV (+0.021 vs EXP012)
- But emphasis on hard training samples hurts generalization to test set
- Confirms distribution shift: Hard samples in training ≠ hard samples in test
- Standard MSE (EXP012) or consistency loss (EXP014) better for this competition
- **Failed hypothesis**: Improving weak folds (Fold 0) via focal loss doesn't help LB

---

### EXP017: DINOv3 ConvNeXt with Gradual Layer-wise Unfreezing
**Model**: DINOv3 ConvNeXt (convnext_base.dinov3_lvd1689m) with gradual backbone unfreezing strategy

#### EXP017-child000: DINOv3 + Gradual Unfreezing (EXP009 model + EXP012 strategy)
**Configuration:**
- **Architecture**: DINOv3 ConvNeXt with 3 heads (from EXP009)
- **Model**: convnext_base.dinov3_lvd1689m (timm)
- **Predicted targets**: Dry_Green_g, GDM_g, Dry_Total_g (3 heads)
- **Cross-validation**: 5-Fold StratifiedGroupKFold (stratify=State, group=Sampling_Date)
- **Training Strategy**: Gradual stage-wise unfreezing (from EXP012) ✨
  - **Epochs 0-4**: Full backbone freeze, train only MLP heads (LR=0.003)
  - **Epochs 5-39**: Gradually unfreeze stages from top to bottom
    - ConvNeXt has 4 stages (vs 27 layers in SigLip)
    - Start unfreezing last stage (epoch 5, backbone_lr=1e-5)
    - End with ~50% of stages unfrozen (~2 out of 4 stages)
    - Backbone LR gradually increases from 1e-5 to 5e-5
  - **Discriminative learning rates**: Heads (0.003) vs Backbone (1e-5 → 5e-5)
- **Augmentation**: Enhanced (same as EXP012)
- **Batch size**: 16, **dropout**: 0.1 (EXP009 setting), **weight_decay**: 0.001

**Results:**
- OOF CV: **0.745** (Mean R²: 0.752 ± 0.081)
- LB: **0.62** (-0.04 from EXP012-001)
- **LB-CV Gap: -0.125**
- Fold R² scores: [0.63, 0.72, 0.88, 0.78, 0.75]

**Key Observations:**
- ❌ **LB worse than SigLip**: 0.62 vs 0.66 (EXP012-001 with SigLip v2)
- ⚠️ **CV lower than SigLip**: 0.745 vs 0.761 (EXP012-001)
- ⚠️ **Worse generalization**: Gap -0.125 vs -0.101 (EXP012-001)
- **Fold 2 excellent**: R² 0.88 (highest single-fold score across all experiments)
- **Fold 0 weak**: R² 0.63 (similar to SigLip experiments)
- **High variance**: std 0.081 (vs 0.066 for EXP012-001)
- **Analysis**: DINOv3 + gradual unfreezing doesn't outperform SigLip v2
  - Possible reasons:
    - DINOv3 (self-supervised) may be less optimal for this task than SigLip (vision-language)
    - ConvNeXt stages (4) vs SigLip layers (27): coarser unfreezing granularity
    - Dropout 0.1 (EXP009) may be too low compared to 0.15 (EXP012)
- **Conclusion**: SigLip v2 (EXP012-001) remains superior to DINOv3 for this task
- **Key lesson**: Model architecture choice (SigLip > DINOv3) more important than training strategy

**Technical Notes:**
- ConvNeXt has 4 stages (much fewer than SigLip's 27 layers)
- Gradual unfreezing is stage-wise (not layer-wise) for ConvNeXt
- Stage unfreezing is coarser-grained than layer unfreezing
- All 4 stages: [stem → stage1 → stage2 → stage3 → stage4 → head]
- Unfreezing: stage4 first, then stage3 (50% = 2/4 stages)

---

### EXP019: SigLip v2 with 5-Head + Consistency Loss (Combining Best Approaches) ❌ **LB Degraded**
**Model**: SigLip v2 (google/siglip2-so400m-patch14-384) with 5-head architecture and consistency loss

#### EXP019-child000: SigLip v2 + 5-Head + Consistency Loss
**Configuration:**
- **Architecture**: 5-head Multi-Task Learning (from EXP014)
- **Model**: google/siglip2-so400m-patch14-384 (SigLip v2, from EXP012-001)
- **Predicted targets**: All 5 targets directly predicted (5 heads)
- **Consistency Loss**: Physical constraint regularization (from EXP014, weight=0.1)
  - Constraint 1: Dry_Dead_g + GDM_g ≈ Dry_Total_g
  - Constraint 2: Dry_Clover_g + Dry_Green_g ≈ GDM_g
- **Cross-validation**: 5-Fold StratifiedGroupKFold (stratify=State, group=Sampling_Date)
- **Training Strategy**: Gradual unfreezing (same as EXP012/014)
  - Epochs 0-4: Backbone frozen
  - Epochs 5-39: Gradual unfreezing (max 50%)
  - Backbone LR: 1e-5 → 5e-5, Head LR: 0.003
- **Augmentation**: Enhanced (same as EXP012/014)
- **Batch size**: 16, **dropout**: 0.15, **weight_decay**: 0.001

**Hypothesis:**
- Combining EXP014's best architecture (5-head + consistency) with EXP012-001's best model (SigLip v2)
- Expected: LB 0.67-0.68 (combining +0.01 from v2 and +0.02 from 5-head)

**Results:**
- OOF CV: **0.775** (Mean R²: 0.774 ± 0.053)
- LB: **0.63** ❌ (-0.03 from EXP014, -0.02 from EXP012-001)
- **LB-CV Gap: -0.145** (worst gap across all experiments)
- Fold R² scores: [0.68, 0.75, 0.83, 0.80, 0.81]

**Key Observations:**
- ❌ **Severe LB degradation**: 0.63 vs 0.66 (EXP014, -0.03) and 0.65 (EXP012-001, -0.02)
- ✅ **CV similar to best**: 0.775 vs 0.772 (EXP014, +0.003) and 0.761 (EXP012-001, +0.014)
- ❌ **Worst overfitting**: Gap -0.145 vs -0.112 (EXP014) and -0.111 (EXP012-001)
- **Failed hypothesis**: Best architecture + best model ≠ best result
- **Analysis**: SigLip v2 + 5-head creates overfitting
  - SigLip v2 works better with 3-head (EXP012-001: LB 0.65)
  - 5-head works better with SigLip v1 (EXP014: LB 0.66)
  - Combination increases model capacity too much → overfits
- **Fold 0 unchanged**: R² 0.68 vs 0.69 (EXP014), similar weak fold performance
- **Fold 2-4 high CV**: R² 0.80-0.83 (high training performance)
- **Key lesson**: **Architecture and model version must be balanced** - too much capacity overfits
- **Conclusion**: Keep EXP014 (SigLip v1 + 5-head) as best model ⭐

**Technical Notes:**
- This was a natural combination to test (best of both experiments)
- Result demonstrates **model selection complexity** - interactions matter
- Higher model capacity (v2 + 5-head) doesn't always improve generalization
- **Recommendation**: Avoid stacking improvements without validation

---

### EXP020: SigLip with 5-Head + Consistency Loss + Fixed Backbone LR
**Model**: SigLip (google/siglip-so400m-patch14-384) with fixed backbone learning rate

#### EXP020-child001: Fixed Backbone LR Strategy
**Configuration:**
- **Base architecture**: EXP014 (5-head + Consistency Loss)
- **Key change**: Fixed backbone LR (1e-5 throughout training)
  - **vs EXP014**: Linearly increasing backbone LR (1e-5 → 5e-5)
  - Gradual unfreezing still applies (number of layers increases)
  - But LR stays constant at 1e-5
- **Hypothesis**: Simpler LR strategy with more conservative backbone updates
- **Rationale**: Layer unfreezing alone provides task adaptation, constant LR adds stability
- Cross-validation: 5-Fold StratifiedGroupKFold (stratify=State, group=Sampling_Date)
- All other parameters same as EXP014-child000

**Results:**
- OOF CV: **0.751** (Mean R²: 0.760 ± 0.078)
- LB: **0.65** (-0.01 from EXP014-child000)
- **LB-CV Gap: -0.101**
- Fold R² scores: [0.63, 0.74, 0.87, 0.78, 0.77]

**Key Observations:**
- ❌ **LB degraded**: 0.65 vs 0.66 (EXP014-child000, -0.01)
- ⚠️ **CV degraded**: 0.751 vs 0.772 (EXP014-child000, -0.021)
- ✅ **Gap improved slightly**: -0.101 vs -0.112 (EXP014-child000, +0.011)
- **Fold 0 worse**: R² 0.63 vs 0.69 (EXP014-child000, -0.06)
- **Fold 2 best**: R² 0.87 (highest single fold in this experiment)
- **High variance**: std 0.078 vs 0.051 (EXP014-child000)
- **Conclusion**: Fixed backbone LR strategy doesn't improve performance
- **Analysis**: Increasing backbone LR (1e-5 → 5e-5) is beneficial - provides better optimization

**Technical Notes:**
- backbone_lr: 1e-5 (fixed) vs 1e-5 → 5e-5 (EXP014)
- More conservative updates to backbone weights
- Simpler hyperparameter configuration but worse results

---

### EXP021: SigLip with 5-Head + Consistency + Auxiliary Metadata Prediction ⚠️ **No LB Improvement**
**Model**: SigLip (google/siglip-so400m-patch14-384) with auxiliary task learning for train-only metadata

#### EXP021-child000: Auxiliary Metadata Prediction Loss
**Configuration:**
- **Base architecture**: EXP014 (5-head + Consistency Loss)
- **Model**: google/siglip-so400m-patch14-384
- **Predicted targets**: All 5 biomass targets (5 heads) + 2 auxiliary metadata heads
- **Auxiliary Tasks** (NEW): Predict train-only metadata features ✨
  - Height_Ave_cm (continuous regression)
  - Pre_GSHH_NDVI (continuous regression)
  - **Key constraint**: These features NOT available in test environment
  - Used only during training for regularization
- **Loss Components**:
  - Main prediction loss (weighted by target importance)
  - Consistency loss (weight=0.1, from EXP014)
  - **Auxiliary loss (weight=0.1, NEW)**: MSE for metadata prediction
- **Cross-validation**: 5-Fold StratifiedGroupKFold (stratify=State, group=Sampling_Date)
- **Training Strategy**: Same as EXP014
  - Epochs 0-4: Backbone frozen
  - Epochs 5-39: Gradual unfreezing (max 50%)
  - Backbone LR: 1e-5 → 5e-5, Head LR: 0.003
- **Augmentation**: Enhanced (same as EXP014)
- **Batch size**: 16, **dropout**: 0.15, **weight_decay**: 0.001

**Hypothesis:**
- Based on Kaggle competition strategies (kmat3.md article)
- Auxiliary task learning forces model to learn richer feature representations
- Train-only metadata as auxiliary tasks improves generalization (similar to Happy Whale, RSNA2024)
- **Strategy**: Use available training metadata to regularize feature learning
- Expected: Better generalization → improve LB-CV gap from -0.112 to -0.08~-0.09

**Results:**
- OOF CV: **0.767** (Mean R²: 0.772 ± 0.063)
- LB: **0.66** (same as EXP014 baseline)
- **LB-CV Gap: -0.107** (slightly better than EXP014: -0.112)
- Fold R² scores: [0.67, 0.76, 0.86, 0.78, 0.79]

**Key Observations:**
- ⚠️ **No LB improvement**: Same 0.66 as EXP014 baseline
- ⚠️ **CV degraded slightly**: 0.767 vs 0.772 (EXP014, -0.005)
- ✅ **Gap improved slightly**: -0.107 vs -0.112 (EXP014, +0.005)
- **Fold performance similar to EXP014**:
  - Fold 0: R² 0.67 vs 0.69 (EXP014, -0.02)
  - Fold 2: R² 0.86 (same as EXP014)
  - Overall pattern consistent with baseline
- **Analysis**: Auxiliary metadata prediction didn't improve performance
  - Hypothesis 1: Metadata features (Height, NDVI) may not strongly correlate with biomass in test set
  - Hypothesis 2: Test set distribution shift means train metadata patterns don't generalize
  - Hypothesis 3: Auxiliary loss (weight=0.1) may be too weak to provide meaningful regularization
  - Hypothesis 4: Model already learns sufficient features from images alone (EXP014 baseline is strong)
- **Gap analysis**: Slight improvement suggests better regularization but not enough to affect LB
- **Conclusion**: Auxiliary metadata prediction provides minimal benefit

**Technical Notes:**
- 2 additional auxiliary heads: head_height, head_ndvi
- Auxiliary heads only used during training (`return_aux=True/False` pattern)
- Inference ignores auxiliary outputs (not available in test environment)
- Dataset returns: `(pixel_values, all_targets, metadata, img_path)`
- Loss = prediction_loss + 0.1×consistency_loss + 0.1×auxiliary_loss

**Key Lessons:**
- Train-only features as auxiliary tasks don't guarantee better generalization
- Distribution shift (train vs test) limits effectiveness of metadata-based regularization
- EXP014 baseline already captures sufficient information from images alone
- Auxiliary task learning requires careful feature selection and weight tuning

#### EXP021-child001: Stronger Auxiliary Weight (0.15)
**Configuration:**
- **Key change**: auxiliary_weight: 0.1 → 0.15 (50% increase)
- Hypothesis: Gap improved in child-000 (+0.005 vs EXP014), stronger auxiliary may further help
- All other parameters same as child-000

**Results:**
- OOF CV: **0.749** (Mean R²: 0.754 ± 0.068)
- LB: **0.65** (-0.01 from child-000)
- **LB-CV Gap: -0.099** (same as EXP014-child005)
- Fold R² scores: [0.65, 0.74, 0.86, 0.75, 0.77]

**Key Observations:**
- ❌ **LB degraded**: 0.65 vs 0.66 (child-000, -0.01)
- ❌ **CV degraded significantly**: 0.749 vs 0.767 (child-000, -0.018)
- ✅ **Gap same as best**: -0.099 (same as EXP014-child005)
- **Fold 0 worse**: R² 0.65 vs 0.67 (child-000)
- **Conclusion**: Stronger auxiliary weight (0.15) hurts performance
- **Optimal auxiliary_weight**: Likely 0.1 or lower

#### EXP021-child002: Weaker Auxiliary Weight (0.05)
**Configuration:**
- **Key change**: auxiliary_weight: 0.1 → 0.05 (50% decrease)
- Hypothesis: Verify that 0.1 is not too strong
- All other parameters same as child-000

**Results:**
- OOF CV: **0.755** (Mean R²: 0.759 ± 0.064)
- LB: **0.63** (-0.03 from child-000)
- **LB-CV Gap: -0.125**
- Fold R² scores: [0.66, 0.74, 0.86, 0.76, 0.76]

**Key Observations:**
- ❌ **Severe LB degradation**: 0.63 vs 0.66 (child-000, -0.03)
- ⚠️ **CV similar**: 0.755 vs 0.767 (child-000, -0.012)
- ❌ **Gap worsened**: -0.125 vs -0.107 (child-000)
- **Conclusion**: Weaker auxiliary weight (0.05) also hurts performance
- **Auxiliary weight sensitivity**: Both 0.05 and 0.15 worse than 0.1

**Comparison: EXP021 Auxiliary Weight Ablation**

| auxiliary_weight | OOF CV | LB | Gap | Δ LB vs 0.1 |
|-----------------|--------|----|----|-------------|
| 0.05 (child-002) | 0.755 | 0.63 ❌ | -0.125 | **-0.03** |
| **0.1 (child-000)** | **0.767** | **0.66** ⭐ | **-0.107** | Baseline |
| 0.15 (child-001) | 0.749 | 0.65 ❌ | -0.099 | **-0.01** |

**Key Insights:**
- **0.1 is optimal** for auxiliary_weight in this task
- Both increasing (0.15) and decreasing (0.05) hurt LB
- **U-shaped curve**: Performance degrades in both directions
- **Conclusion**: EXP021-child000 (auxiliary_weight=0.1) is the best among auxiliary experiments
- **But still worse than EXP014**: Auxiliary metadata doesn't help overall

---

### EXP022: SigLip with 5-Head + Consistency + Auxiliary Metadata + Species Classification ❌ **LB Degraded**
**Model**: SigLip (google/siglip-so400m-patch14-384) with multi-label species classification as auxiliary task

#### EXP022-child000: Species Multi-Label Classification Auxiliary Loss
**Configuration:**
- **Base architecture**: EXP021 (5-head + Consistency + Auxiliary Metadata)
- **Model**: google/siglip-so400m-patch14-384
- **Predicted targets**: All 5 biomass targets + 2 metadata + 16 species labels
- **Species Classification Task** (NEW): Multi-label binary classification ✨
  - 16 species classes: ['BarleyGrass', 'Barleygrass', 'Bromegrass', 'Capeweed', 'Clover', 'CrumbWeed', 'Fescue', 'Lucerne', 'Mixed', 'Phalaris', 'Ryegrass', 'SilverGrass', 'SpearGrass', 'SubcloverDalkeith', 'SubcloverLosa', 'WhiteClover']
  - Species string is underscore-separated, multi-label (e.g., "Ryegrass_Clover" → [Ryegrass=1, Clover=1, others=0])
  - **CRITICAL: Clover Correction Logic**:
    - Species metadata may list "Clover" but target Dry_Clover_g = 0 (contradiction)
    - Solution: Trust target values - if Dry_Clover_g == 0, set all Clover-related species labels to 0
    - Clover-related species: ['Clover', 'SubcloverDalkeith', 'SubcloverLosa', 'WhiteClover']
  - **Rationale**: Forces model to learn species-discriminative features (domain knowledge: species correlates with biomass)
- **Loss Components**:
  - Main prediction loss (weighted by target importance)
  - Consistency loss (weight=0.1, from EXP014)
  - Auxiliary metadata loss (weight=0.1, from EXP021)
  - **Species classification loss (weight=0.05, NEW)**: BCEWithLogitsLoss for multi-label
- **Cross-validation**: 5-Fold StratifiedGroupKFold (stratify=State, group=Sampling_Date)
- **Training Strategy**: Same as EXP014/021
  - Epochs 0-4: Backbone frozen
  - Epochs 5-39: Gradual unfreezing (max 50%)
  - Backbone LR: 1e-5 → 5e-5, Head LR: 0.003
- **Augmentation**: Enhanced (same as EXP014)
- **Batch size**: 16, **dropout**: 0.15, **weight_decay**: 0.001

**Hypothesis:**
- Based on successful Kaggle strategies (e.g., Happy Whale using species as auxiliary task)
- Species strongly correlates with biomass (domain knowledge: Fescue → high biomass)
- Multi-label classification forces model to learn species-specific features
- Multi-task learning with species prediction improves generalization
- Expected: Better feature learning → improve LB from 0.66 to 0.67-0.68

**Results:**
- OOF CV: **0.752** (Mean R²: 0.753 ± 0.058)
- LB: **0.62** ❌ (-0.04 from EXP014 baseline, -0.04 from EXP021)
- **LB-CV Gap: -0.132** (worse than EXP014: -0.112, EXP021: -0.107)
- Fold R² scores: [0.66, 0.74, 0.84, 0.75, 0.77]

**Key Observations:**
- ❌ **Severe LB degradation**: 0.62 vs 0.66 (EXP014, -0.04) and 0.66 (EXP021, -0.04)
- ❌ **CV degraded significantly**: 0.752 vs 0.772 (EXP014, -0.020) and 0.767 (EXP021, -0.015)
- ❌ **Worse overfitting**: Gap -0.132 vs -0.112 (EXP014), -0.107 (EXP021)
- **Fold performance degraded across all folds**:
  - Fold 0: R² 0.66 vs 0.69 (EXP014), 0.67 (EXP021)
  - Fold 2: R² 0.84 vs 0.86 (EXP014), 0.86 (EXP021)
  - **All folds lower** than both baselines
- **Better fold consistency**: std 0.058 vs 0.063 (EXP021), 0.051 (EXP014)
- **Analysis**: Species classification auxiliary task hurt performance
  - Hypothesis 1: **Distribution shift** - Species patterns in train don't match test set
  - Hypothesis 2: **Model capacity issue** - Too many auxiliary tasks (metadata + species) overcomplicates model
  - Hypothesis 3: **Species metadata quality** - Clover contradictions suggest noisy labels
  - Hypothesis 4: **Wrong auxiliary task** - Species may not provide useful regularization signal
  - Hypothesis 5: **Auxiliary weight too weak** - species_weight=0.05 insufficient to learn meaningful features OR too strong causing harmful regularization
- **Comparison with EXP021**:
  - Adding species classification degraded both CV (-0.015) and LB (-0.04)
  - Gap worsened from -0.107 to -0.132 (-0.025)
  - Confirms species task doesn't help
- **Failed hypothesis**: Multi-task learning with species ≠ better generalization

**Technical Notes:**
- 3 auxiliary heads total: head_height, head_ndvi, head_species
- Species head outputs 16-dim vector (BCEWithLogitsLoss for multi-label)
- Clover correction implemented in `extract_species_labels()` function
- Dataset returns: `(pixel_values, all_targets, metadata, species_labels, img_path)`
- Loss = prediction_loss + 0.1×consistency_loss + 0.1×auxiliary_loss + 0.05×species_loss
- Forward returns 8 outputs when `return_aux=True`: 5 biomass + 2 metadata + 1 species

**Key Lessons:**
- ❌ **Auxiliary tasks can hurt performance** - not all auxiliary tasks provide useful regularization
- ❌ **Domain knowledge ≠ guaranteed improvement** - species-biomass correlation doesn't help generalization
- ❌ **Stacking regularization techniques backfires** - multiple auxiliary tasks (metadata + species) worse than none
- ⚠️ **Label quality matters** - Clover contradictions suggest metadata may be unreliable
- ⚠️ **Distribution shift critical** - Train species patterns don't match test set
- ⚠️ **Simpler is better** - EXP014 baseline (5-head + consistency) outperforms complex multi-task variants
- **Recommendation**: Avoid auxiliary tasks when base model (EXP014) already performs well

**Comparison: EXP014 vs EXP021 vs EXP022**

| Experiment | Architecture | Aux Tasks | OOF CV | LB | Gap | Δ LB vs EXP014 |
|------------|-------------|-----------|--------|----|----|----------------|
| **EXP014** | 5-head + Consistency | None | **0.772** | **0.66** ⭐ | **-0.112** | Baseline |
| **EXP021** | 5-head + Consistency | Metadata (2) | 0.767 | **0.66** | -0.107 | **0.00** (same) |
| **EXP022** | 5-head + Consistency | Metadata (2) + Species (16) | 0.752 ❌ | 0.62 ❌ | -0.132 ❌ | **-0.04** ❌ |

**Key Insights:**
- **EXP014 remains best**: No auxiliary tasks needed
- **EXP021 neutral**: Metadata auxiliary provides no benefit but doesn't hurt LB
- **EXP022 harmful**: Species auxiliary significantly degrades performance
- **Progressive degradation**: More auxiliary tasks → worse performance
- **Conclusion**: Keep EXP014 as best model, avoid auxiliary task complexity

---

### EXP024: SigLip with 5-Head + Feature Concatenation + TTA ⭐ **2ND BEST PUBLIC LB RANK**
**Model**: SigLip (google/siglip-so400m-patch14-384) with feature concatenation approach

#### EXP024-child000: Feature Concatenation + TTA
**Configuration:**
- **Base architecture**: EXP014 (5-head + Consistency Loss)
- **Key innovation**: Concatenate features from left/right halves BEFORE prediction
  - **vs EXP014**: Sum predictions from left/right halves AFTER independent prediction
  - Concatenated features: 2304-dim (1152 + 1152) input to each head
  - Allows model to learn interactions between left and right features
- **Predicted targets**: All 5 targets directly predicted (5 heads)
- **Consistency Loss**: Same as EXP014 (weight=0.1)
- **Inference**: TTA with 3 views (original, hflip, vflip)
- Cross-validation: 5-Fold StratifiedGroupKFold (stratify=State, group=Sampling_Date)
- **Training Strategy**: Same as EXP014
  - Gradual unfreezing: 50% of backbone (max_unfreeze_ratio: 0.5)
  - Epochs 0-4: Backbone frozen
  - Epochs 5-39: Gradual unfreezing
  - Backbone LR: 1e-5 → 5e-5, Head LR: 0.003
- Augmentation: Enhanced (same as EXP014)
- Batch size: 16, dropout: 0.15, weight_decay: 0.001

**Results:**
- OOF CV: **0.741** (Mean R²: 0.741 ± 0.066)
- LB: **0.66** ⭐ **2ND BEST PUBLIC LB RANK** (Ranked #2 among all LB 0.66 submissions)
- **LB-CV Gap: -0.094** (best gap across all experiments)
- Fold R² scores: [0.65, 0.72, 0.85, 0.72, 0.77]

**Key Observations:**
- ⭐ **2ND BEST PUBLIC LB RANK**: LB 0.66 ranked #2 (only behind EXP014-child005)
- ✅ **BEST GAP EVER**: -0.094 (better than all previous experiments)
  - vs EXP014-child000: -0.112 (+0.018 improvement)
  - vs EXP014-child005: -0.099 (+0.005 improvement)
  - vs EXP021-child000: -0.107 (+0.013 improvement)
- ❌ **CV degraded significantly**: 0.741 vs 0.772 (EXP014-child000, -0.031)
- **Trade-off validated**: Lower CV but excellent generalization
- **Fold 0 worse**: R² 0.65 vs 0.69 (EXP014-child000, -0.04)
- **Fold 2 consistent**: R² 0.85 (same as EXP014)
- **Key insight**: **Feature concatenation > prediction summing** for generalization
- **TTA effect**: Improves decimal precision (ranked #2 vs #4 for EXP014-TTA)

**Analysis:**
- **Feature interaction learning works**: Concatenation allows heads to learn from both halves jointly
- **Best generalization**: Gap -0.094 is the best across all experiments
- **Public test favors this approach**: Despite lower CV, ranks #2 on Public LB
- **Architecture innovation effective**: Changing feature fusion strategy improves LB rank
- **Comparison with EXP014-child005**:
  - EXP014-005: CV 0.759, Gap -0.099, Public rank #1
  - EXP024-000: CV 0.741, Gap -0.094, Public rank #2
  - **Both use different strategies** (70% unfreeze vs feature concat) but achieve similar results

**Technical Notes:**
- Concatenated feature dimension: 2304 (1152 left + 1152 right)
- Each MLP head input: 2304-dim (vs 1152-dim in EXP014)
- More parameters in heads due to larger input
- TTA with 3 views applied during inference
- Architecture difference visualized:
  - **EXP014**: `[Left feat] → [Head] + [Right feat] → [Head] = Final pred`
  - **EXP024**: `[Left feat + Right feat] → [Head] = Final pred`

**Conclusion:**
- **EXP024 is 2nd best submission** for Public LB (behind EXP014-child005)
- **Best generalization gap** achieved (-0.094)
- Feature concatenation is a promising architecture innovation
- Lower CV doesn't mean worse LB - gap is more predictive

---

### EXP026: SigLip with 5-Head + ExG/HSV Features ❌ **FAILED**
**Model**: SigLip (google/siglip-so400m-patch14-384) with ExG/HSV image features

#### EXP026-child000: ExG/HSV Features (10-dim)
**Configuration:**
- **Base architecture**: EXP014 (5-head + Consistency Loss)
- **Key innovation**: Add ExG + HSV features (10-dim) extracted from images
  - **Feature extraction**: Computed from raw image pixels (no metadata)
  - **ExG features** (4): Excess Green index (mean, std, p90, green_ratio)
  - **Spatial features** (4): Green ratio in 2x2 quadrants (top-left, top-right, bottom-left, bottom-right)
  - **HSV features** (2): Dead grass ratio (brown/yellow), Soil ratio (low saturation)
  - **Integration**: Concatenated with SigLip embeddings (1152 + 10 = 1162-dim) before MLP heads
- **Rationale**: Explicit pixel statistics should complement SigLip's semantic understanding
- **Predicted targets**: All 5 targets directly predicted (5 heads)
- **Consistency Loss**: Same as EXP014 (weight=0.1)
- Cross-validation: 5-Fold StratifiedGroupKFold (stratify=State, group=Sampling_Date)
- **Training Strategy**: Same as EXP014
  - Gradual unfreezing: 50% of backbone (max_unfreeze_ratio: 0.5)
  - Epochs 0-4: Backbone frozen
  - Epochs 5-39: Gradual unfreezing
  - Backbone LR: 1e-5 → 5e-5, Head LR: 0.003
- Augmentation: Enhanced (same as EXP014)
- Batch size: 16, dropout: 0.3 (increased from 0.15), hidden_dim: 512, weight_decay: 0.001

**Results:**
- OOF CV: **0.756** (Mean R²: 0.756 ± 0.069)
- LB: **0.62** ❌ **FAILED** (-0.04 from baseline)
- **LB-CV Gap: -0.136** ❌ **WORST GAP** (worse than all previous experiments)
- Fold R² scores: [0.64, 0.74, 0.86, 0.77, 0.77]

**Key Observations:**
- ❌ **SEVERE LB DEGRADATION**: LB 0.62 vs 0.66 (EXP014-child000, -0.04)
- ❌ **WORST GAP EVER**: -0.136 (vs -0.112 in EXP014-child000, -0.024 worse)
- ❌ **CV slightly degraded**: 0.756 vs 0.772 (EXP014-child000, -0.016)
- **ExG/HSV features harmful**: Adding handcrafted features degrades both CV and LB
- **Overfitting to train distribution**: Features that help CV don't generalize to test
- **Fold 2 still strong**: R² 0.86 (consistent across experiments)
- **Higher dropout didn't help**: 0.3 vs 0.15 still resulted in severe overfitting

**Analysis:**
- **Why ExG/HSV features failed**:
  1. **Distribution shift**: Pixel statistics computed from train images don't match test patterns
  2. **SigLip already captures these**: Vision-language model learns better representations than handcrafted features
  3. **Feature redundancy**: ExG green ratio likely redundant with SigLip's learned features
  4. **Increased model complexity**: More parameters in MLP heads (1162-dim vs 1152-dim input) without benefit
  5. **Overfitting signal**: Features that correlate with training labels don't generalize
- **Comparison with EXP014-child000**:
  - EXP014-000: CV 0.772, LB 0.66, Gap -0.112 ⭐
  - EXP026-000: CV 0.756, LB 0.62, Gap -0.136 ❌
  - **Adding features hurt both CV (-0.016) and LB (-0.04)**
- **Critical lesson**: **Deep learning features > handcrafted features** for this task
  - SigLip's learned representations already capture relevant pixel-level patterns
  - Explicit feature engineering doesn't help when using powerful pretrained models
- **Gap analysis**: Worst gap (-0.136) indicates severe overfitting to training distribution
  - ExG/HSV features correlate with training labels but fail on test set
  - Confirms distribution shift between train and test

**Comparison with other feature experiments**:
- **EXP021** (Auxiliary metadata): LB 0.66 (no improvement but no harm)
- **EXP022** (Auxiliary species): LB 0.62 (-0.04, same as EXP026)
- **EXP026** (ExG/HSV features): LB 0.62 (-0.04) ❌
- **Pattern**: Adding any additional features/tasks to EXP014 baseline provides no benefit

**Conclusion:**
- ❌ **Failed experiment**: ExG/HSV features severely degrade LB performance
- **Stick with EXP014 baseline**: SigLip embeddings alone are optimal
- **No feature engineering needed**: Pretrained vision models already capture relevant patterns
- **Simpler is better**: EXP014 without any additional features remains best approach
- **Distribution shift confirmed**: Features computed from images still suffer from train-test mismatch

---

### EXP027: SigLip with 5-Head + Exponential Moving Average (EMA) ❌ **CATASTROPHIC FAILURE**
**Model**: SigLip (google/siglip-so400m-patch14-384) with EMA weight averaging

#### EXP027-child001: EMA + 70% Backbone Unfreezing
**Configuration:**
- **Base architecture**: EXP014 (5-head + Consistency Loss)
- **Key innovation**: Exponential Moving Average (EMA) weight averaging ✨
  - EMA decay: 0.999
  - Shadow model tracks moving average of weights during training
  - Best model saved based on EMA shadow weights (not current weights)
- **Gradual unfreezing**: 70% of backbone (max_unfreeze_ratio: 0.7)
  - vs EXP014-child000: 50% unfreezing
  - SigLip has 27 layers: 70% = ~19 layers unfrozen
- **Predicted targets**: All 5 targets directly predicted (5 heads)
- **Consistency Loss**: Same as EXP014 (weight=0.1)
- Cross-validation: 5-Fold StratifiedGroupKFold (stratify=State, group=Sampling_Date)
- **Training Strategy**: Same as EXP014
  - Epochs 0-4: Backbone frozen
  - Epochs 5-39: Gradual unfreezing to 70%
  - Backbone LR: 1e-5 → 5e-5, Head LR: 0.003
- Augmentation: Enhanced (same as EXP014)
- Batch size: 16, dropout: 0.15, weight_decay: 0.001

**Hypothesis:**
- EMA smooths weight updates → better generalization
- Combined with 70% unfreezing → improved capacity without overfitting
- Expected: LB 0.67+ with improved gap

**Results:**
- OOF CV: **-0.170** ❌ ❌ ❌ **NEGATIVE R²** (Mean R²: 0.749 ± 0.062)
- LB: **-0.04** ❌ ❌ ❌ **NEGATIVE SCORE**
- **LB-CV Gap: N/A** (both metrics failed)
- Fold R² scores: [0.65, 0.73, 0.84, 0.75, 0.77]

**Key Observations:**
- ❌ ❌ ❌ **CATASTROPHIC FAILURE**: OOF R² = -0.170, LB = -0.04
- ❌ **Best fold R² looks normal** (0.65-0.84), but OOF aggregation failed
- ❌ **EMA implementation bug**: Shadow weights likely not properly aggregated across folds
- **Training instability observed**: Validation R² collapsed in later epochs
  - Epoch 36: Val R² = 0.545
  - Epoch 40: Val R² = 0.391 (-0.154 drop in 4 epochs)
- **Negative predictions**: OOF predictions contain negative values (e.g., Dry_Clover_g = -0.095)
  - Consistency loss doesn't enforce non-negativity
  - EMA may amplify negative predictions

**Analysis:**
- **Why EMA failed catastrophically**:
  1. **OOF aggregation bug**: Individual fold R² scores (0.65-0.84) vs OOF R² (-0.17) suggests predictions not properly saved
  2. **EMA shadow weights issue**: Best model may be using incorrect EMA shadow weights
  3. **Training collapse**: Validation R² dropped from 0.58 → 0.39 in last 4 epochs
  4. **Negative predictions**: Model outputs negative biomass values without ReLU constraint
  5. **70% unfreezing too aggressive**: More layers + EMA instability → training divergence
- **EMA decay too high**: 0.999 means EMA updates very slowly
  - With 40 epochs × ~24 batches = ~960 steps
  - Effective averaging window: 1/(1-0.999) = 1000 steps
  - EMA weights lag too far behind current weights
- **Comparison with EXP014-child005 (70% unfreezing, no EMA)**:
  - EXP014-005: CV 0.759, LB 0.66, Gap -0.099 ⭐
  - EXP027-001: CV -0.170, LB -0.04 ❌
  - **Conclusion**: 70% unfreezing works (EXP014-005), but EMA breaks training

**Technical Notes:**
- EMA implementation may have bugs:
  - Shadow model not properly synchronized during checkpoint saving
  - OOF predictions using wrong model (shadow vs current)
  - Fold-wise best model selection using EMA may be incorrect
- Negative predictions suggest missing output constraints (ReLU/Softplus)

**Critical Lessons:**
- ❌ **EMA requires careful implementation**: Shadow weight management is complex
- ❌ **Don't combine multiple risky changes**: 70% unfreezing + EMA + new codebase = recipe for failure
- ❌ **Monitor training stability**: Validation R² collapse (0.58 → 0.39) should trigger early stopping
- ❌ **Enforce output constraints**: Non-negative biomass requires ReLU or Softplus activation
- ⚠️ **EMA decay tuning critical**: 0.999 may be too high for 40-epoch training
- ⚠️ **Test new techniques on stable baseline first**: Should have tried EMA on 50% unfreezing (EXP014-000) before 70%

**Recommendation:**
- ❌ **Abandon EMA approach**: Implementation too risky, benefits unclear
- ✅ **Stick with EXP014-child005**: 70% unfreezing without EMA works well (LB 0.66, #1 Public rank)
- ⚠️ **If retrying EMA**: Fix OOF prediction bug, reduce decay to 0.99, test on stable baseline first

---

### EXP028: SigLip with 5-Head + Consistency Loss + Cosine Annealing LR Scheduler ❌
**Model**: SigLip (google/siglip-so400m-patch14-384) with Cosine Annealing with Warm Restarts

#### EXP028-child000: Cosine Annealing LR Scheduler
**Configuration:**
- **Base architecture**: EXP014 (5-head + Consistency Loss)
- **Key innovation**: Cosine Annealing LR Scheduler with Warm Restarts ✨
  - Head scheduler: CosineAnnealingWarmRestarts
  - T_0 = 10 (restart every 10 epochs)
  - T_mult = 2 (double period after each restart)
  - eta_min = 0.0001 (minimum LR)
  - Head LR: 0.003 → 0.0001 (cosine schedule)
- **Gradual unfreezing**: 50% of backbone (max_unfreeze_ratio: 0.5)
  - Freeze epochs: 5 (epochs 0-4 frozen)
  - SigLip has 27 layers: 50% = ~14 layers unfrozen
- **Predicted targets**: All 5 targets directly predicted (5 heads)
- **Consistency Loss**: Same as EXP014 (weight=0.1)
- Cross-validation: 5-Fold StratifiedGroupKFold (stratify=State, group=Sampling_Date)
- **Training Strategy**:
  - Epochs 0-4: Backbone frozen, heads trained with cosine schedule
  - Epochs 5-39: Gradual unfreezing to 50%
  - Backbone LR: 1e-5, Head LR: cosine annealing (0.003 → 0.0001)
- Augmentation: Enhanced (same as EXP014)
- Batch size: 16, dropout: 0.15, weight_decay: 0.001
- Total epochs: 40

**Hypothesis:**
- Cosine annealing with warm restarts → better optimization trajectory
- Periodic LR restarts escape local minima → improved generalization
- Expected: Better LB than constant LR (EXP014-000 baseline: LB 0.66)

**Results:**
- **OOF CV**: 0.761 (Mean R²: 0.763 ± 0.065)
- **Public LB**: **0.65**
- **LB-CV Gap**: -0.111
- Fold R² scores: [0.673, 0.734, 0.875, 0.762, 0.770]
- Best fold: Fold 2 (R² = 0.875)
- Worst fold: Fold 0 (R² = 0.673)

**Key Observations:**
- ✅ **Good CV**: 0.761 matches EXP014-000 baseline (0.772) and EXP014-005 (0.759)
- ❌ **LB degradation**: 0.65 vs baseline 0.66 (-0.01 drop)
- ⚠️ **Larger gap**: -0.111 vs baseline -0.112 (similar, slightly better)
- ⚠️ **Fold imbalance**: 0.202 spread between best (0.875) and worst (0.673)
  - vs EXP014-000: Similar imbalance (0.689 to 0.840)
- 📊 **Comparison with constant LR (EXP014-000)**:
  - Same CV (0.761 vs 0.772)
  - Worse LB (0.65 vs 0.66)
  - Similar gap (-0.111 vs -0.112)

**Analysis:**
- **Why Cosine Annealing didn't help**:
  1. **Good CV but worse LB**: Scheduler may have improved training dynamics but hurt generalization
  2. **Warm restarts may cause instability**: Periodic LR spikes could disrupt learned representations
  3. **40 epochs may be insufficient**: With T_0=10 and T_mult=2, only 2-3 restart cycles (10 + 20 = 30 epochs)
  4. **Baseline already near optimal**: EXP014-000 with constant LR achieved LB 0.66, hard to improve
  5. **Scheduler may overfit to CV folds**: Cosine schedule optimized for validation sets, not test set
- **T_0 and T_mult tuning needed**: Current schedule (10, 2) may not align with optimal learning phases
  - Freeze phase (epochs 0-4) uses cosine, but backbone is frozen → wasted schedule
  - Unfreeze phase (epochs 5-39) = 35 epochs with restarts at 10, 30 → only 1 full restart cycle
- **Comparison with EXP014-005 (70% unfreezing, constant LR)**:
  - EXP014-005: CV 0.759, LB 0.66, Gap -0.099 ⭐ #1 Public rank
  - EXP028-000: CV 0.761, LB 0.65, Gap -0.111
  - **Conclusion**: 70% unfreezing (EXP014-005) outperforms scheduler (EXP028-000)

**Technical Notes:**
- Cosine Annealing implementation:
  - PyTorch `CosineAnnealingWarmRestarts`
  - Applied only to head optimizer (not backbone during unfreeze phase)
  - eta_min = 0.0001 prevents LR from dropping to zero
- Scheduler step called after each epoch (not per batch)

---

#### EXP028-child001: Smoother Cosine Annealing (T_0=15, Higher eta_min)
**Configuration:**
- **Base architecture**: Same as EXP028-child000
- **Key changes**: Adjusted scheduler to prevent late-epoch instability
  - **T_0**: 10 → **15** (longer first cycle, fewer restarts)
  - **eta_min**: 0.0001 → **0.0005** (higher minimum LR, smaller variation)
  - **T_mult**: 2 (same as child-000)
  - Head LR: 0.003 → 0.0005 (6× variation vs 30× in child-000)
- **All other params**: Same as child-000 (50% unfreeze, dropout=0.15, etc.)

**Motivation:**
- child-000 showed training instability at epoch 36:
  - LR reset from 0.0001 → 0.003 caused R² drop from 0.559 → 0.443
  - 3rd restart cycle (epochs 35-39) was too aggressive
  - Best R² (0.673) achieved at epoch 18, couldn't improve later
- **Solution**: Reduce restart frequency and LR variation
  - Only 2 cycles (restart at epoch 20 only, not epoch 35)
  - Higher eta_min (0.0005) keeps model more stable
  - Longer cycles (15/30 epochs) for better convergence

**Hypothesis:**
- Smoother LR schedule → less disruption → better late-epoch stability
- Fewer restarts → model can converge without being reset
- Expected: Maintain or improve best R² (0.67-0.68+)

**Results:**
- **OOF CV**: 0.761 (Mean R²: 0.764 ± 0.068)
- **Public LB**: **0.65**
- **LB-CV Gap**: -0.111
- Fold R² scores: [0.659, 0.732, 0.865, 0.786, 0.777]
- Best fold: Fold 2 (R² = 0.865)
- Worst fold: Fold 0 (R² = 0.659)

**Key Observations:**
- ⚠️ **Identical results to child-000**: CV 0.761, LB 0.65, Gap -0.111 (all same)
- 📊 **No improvement from smoother scheduler**:
  - Fold scores almost identical (0.659-0.865 vs 0.673-0.875)
  - Mean R² identical (0.764 vs 0.763)
  - LB unchanged (0.65)
- ✅ **Confirmed stability improvement**: Smoother schedule prevented late-epoch drops
- ❌ **But stability didn't improve LB**: Better training dynamics ≠ better generalization

**Analysis:**
- **Why smoother schedule didn't help**:
  1. **Problem wasn't the scheduler**: The issue is scheduler vs constant LR, not T_0/eta_min tuning
  2. **Baseline already optimal**: EXP014-000 (constant LR) achieves LB 0.66, hard to beat with scheduler tricks
  3. **CV-LB mismatch**: Both child-000 and child-001 have good CV (0.761) but worse LB (0.65)
  4. **Scheduler may cause CV overfitting**: Warm restarts optimize for validation sets, not test set
- **Scheduler tuning is ineffective**:
  - Changed T_0 (10 → 15) and eta_min (0.0001 → 0.0005)
  - Results identical: CV 0.761, LB 0.65
  - **Conclusion**: The scheduler approach itself doesn't generalize well

**Comparison: child-000 vs child-001**
| Metric | child-000 (T_0=10) | child-001 (T_0=15) | Change |
|--------|-------------------|-------------------|--------|
| OOF CV | 0.761 | 0.761 | 0.000 |
| LB | 0.65 | 0.65 | 0.00 |
| Gap | -0.111 | -0.111 | 0.000 |
| Best Fold | 0.875 | 0.865 | -0.010 |
| Worst Fold | 0.673 | 0.659 | -0.014 |
| **Result** | ❌ Worse than baseline | ❌ Worse than baseline | **No difference** |

**Critical Lessons from EXP028 (both child-000 and child-001):**
- ❌ **Cosine Annealing with Warm Restarts didn't improve LB**: Good CV (0.761) but worse LB (0.65 vs 0.66)
- ❌ **Scheduler complexity vs benefit**: Added complexity without performance gain
- ❌ **Scheduler parameter tuning is ineffective**: Adjusting T_0 (10→15) and eta_min (0.0001→0.0005) made zero difference
  - child-000 (T_0=10): CV 0.761, LB 0.65
  - child-001 (T_0=15): CV 0.761, LB 0.65 (identical)
  - **Conclusion**: Problem is using scheduler at all, not the hyperparameters
- ✅ **Constant LR + proper unfreezing works better**: EXP014-005 (70% unfreeze) achieves #1 Public rank (LB 0.66)
- 💡 **Simpler is often better**: EXP014-000 baseline with constant LR remains competitive
- ⚠️ **Good training dynamics ≠ better generalization**: Smoother schedule (child-001) prevented late-epoch drops but didn't improve LB

**Recommendation:**
- ❌ **Don't use Cosine Annealing with current setup**: No improvement over constant LR, even with parameter tuning
- ✅ **Focus on unfreezing ratio**: EXP014-005 (70% unfreeze) is better than scheduler tricks
- ⚠️ **If retrying scheduler**:
  - Try OneCycleLR instead (1 cycle, simpler)
  - Align T_0 with freeze/unfreeze phases (e.g., T_0=5 for freeze phase)
  - Test on smaller experiments first before full 5-fold training
- ✅ **Stick with proven approaches**: EXP014-005 (70% unfreeze) or EXP024-000 (feature concat) for final submission

**Summary of EXP028 Results:**

| Child Exp | Scheduler Config | OOF CV | LB | Gap | Observation |
|-----------|-----------------|--------|----|----|-------------|
| child-000 | T_0=10, eta_min=0.0001 | 0.761 | 0.65 | -0.111 | Baseline scheduler |
| child-001 | T_0=15, eta_min=0.0005 | 0.761 | 0.65 | -0.111 | Smoother schedule - **identical results** |
| **vs EXP014-000** | **Constant LR** | **0.772** | **0.66** | **-0.112** | **Constant LR wins** |
| **vs EXP014-005** | **Constant LR (70% unfreeze)** | **0.759** | **0.66** | **-0.099** | **#1 Public rank** |

**Key Takeaway**: Cosine Annealing with Warm Restarts adds complexity without improving performance. Both child experiments (000 and 001) achieved identical results (CV 0.761, LB 0.65), worse than baseline constant LR (LB 0.66). Scheduler parameter tuning (T_0, eta_min) made zero difference. **Recommendation: Abandon scheduler approach, focus on unfreezing ratio instead.**

---

### EXP030: EVA-CLIP Vision Encoder with Gradual Layer-wise Unfreezing ⭐ **BREAKTHROUGH**
**Model**: EVA02-CLIP-L-14-336 (428M params) with superior vision features (80.4% ImageNet zero-shot)

#### EXP030-child000: EVA-CLIP with Gradual Unfreezing (50%)
**Configuration:**
- **Backbone**: EVA02-CLIP-L-14-336 (428M parameters)
  - State-of-the-art CLIP variant with 80.4% ImageNet zero-shot accuracy
  - 24 transformer blocks (Large model)
  - 336×336 input size (larger than SigLip's 384×384 but higher quality features)
  - 768-dim embeddings
- **Architecture**: 3-head Multi-Task Learning (same as EXP017)
  - Predicts: Dry_Green_g, GDM_g, Dry_Total_g
  - Calculates: Dry_Dead_g = max(0, Dry_Total_g - GDM_g)
  - Calculates: Dry_Clover_g = max(0, GDM_g - Dry_Green_g)
- **Training Strategy**: Gradual unfreezing from EXP017
  - Epochs 0-4: Backbone frozen, train only heads (LR=0.003)
  - Epochs 5+: Gradually unfreeze transformer blocks from top to bottom
  - max_unfreeze_ratio: 0.5 (~12 out of 24 blocks)
  - Backbone LR: starts at 1e-5, gradually increases to 5e-5
- **Cross-validation**: 5-Fold StratifiedGroupKFold (stratify=State, group=Sampling_Date)
- **Batch size**: 12 (smaller due to 336×336 input and larger model)
- **Regularization**: dropout=0.1, weight_decay=0.001 (same as EXP017)
- **Total epochs**: 40

**Hypothesis:**
- EVA-CLIP's superior vision features (80.4% zero-shot vs SigLip's ~78%) → better pasture understanding
- Strong pre-training on large-scale image-text pairs → better initialization for fine-grained visual features
- Gradual unfreezing maintains generalization while adapting to pasture domain
- Expected: LB 0.67-0.68 (vs EXP017 DINOv3: 0.62, EXP014 SigLip: 0.66)

**Results:**
- **Public LB**: **0.68** ⭐ ⭐ ⭐ **NEW BEST** (+0.02 from EXP014-005)
- **CV**: 0.762 (Mean R²: 0.762 ± 0.067)
- **LB-CV Gap**: **-0.082** ⭐ **BEST GAP EVER** (vs EXP024-000: -0.094)
- Fold R² scores: [0.672, 0.725, 0.840, 0.763, 0.812]
- Best fold: Fold 2 (R² = 0.840)
- Worst fold: Fold 0 (R² = 0.672)

**Key Observations:**
- 🎉 **BREAKTHROUGH PERFORMANCE**: LB 0.68 vs previous best 0.66 (+0.02 improvement, +3% relative)
- ⭐ **BEST GENERALIZATION GAP**: -0.082 (best across all experiments)
  - vs EXP024-000: -0.094 (+0.012 improvement)
  - vs EXP014-005: -0.099 (+0.017 improvement)
  - vs EXP014-000: -0.112 (+0.030 improvement)
- ✅ **CV competitive**: 0.762 vs EXP014-000 (0.772), EXP014-005 (0.759)
- ✅ **EVA-CLIP >>> DINOv3**: 0.68 vs 0.62 (EXP017, +0.06 improvement)
- ✅ **EVA-CLIP > SigLip**: 0.68 vs 0.66 (EXP014-005, +0.02 improvement)
- ⭐ **Fold consistency excellent**: std 0.067 (similar to best experiments)
- 📊 **Comparison with other vision encoders**:
  - EVA-CLIP (EXP030): LB 0.68, Gap -0.082 ⭐ **BEST**
  - SigLip (EXP014-005): LB 0.66, Gap -0.099
  - SigLip (EXP014-000): LB 0.66, Gap -0.112
  - DINOv3 (EXP017): LB 0.62, Gap -0.125 ❌

**Analysis:**
- **Why EVA-CLIP works so well**:
  1. **Superior vision features**: 80.4% ImageNet zero-shot vs ~78% for SigLip
     - Better fine-grained visual understanding (critical for clover/species detection)
     - Stronger pre-training on diverse image-text pairs
  2. **Better feature quality**: EVA-CLIP embeddings capture richer semantic information
     - Helps model distinguish between grass types (green/dead/clover)
     - Better spatial understanding for biomass density estimation
  3. **Optimal architecture choice**: 3-head architecture (vs 5-head) reduces overfitting
     - Fewer parameters to tune with powerful backbone
     - Physical constraints enforced through calculations
  4. **Gradual unfreezing strategy**: Maintains generalization while adapting
     - 50% unfreezing provides good capacity without overfitting
     - Layer-wise unfreezing preserves learned representations
- **Best gap ever achieved**: -0.082 indicates excellent generalization
  - Model generalizes well to Public test set
  - EVA-CLIP features transfer better than SigLip/DINOv3
- **Fold 0 improvement**: R² 0.672 vs 0.69 (EXP014-000)
  - Still the weakest fold but gap narrowed
  - EVA-CLIP helps underperforming folds

**Comparison: Vision Encoder Performance**

| Backbone | ImageNet Zero-shot | LB | CV | Gap | Δ LB vs SigLip |
|----------|-------------------|----|----|-----|----------------|
| **EVA-CLIP-L** | **80.4%** ⭐ | **0.68** ⭐ | 0.762 | **-0.082** ⭐ | **+0.02** |
| SigLip-SO400M | ~78% | 0.66 | 0.772 | -0.112 | Baseline |
| SigLip-SO400M (70% unfreeze) | ~78% | 0.66 | 0.759 | -0.099 | 0.00 |
| DINOv3-L | ~77% (self-supervised) | 0.62 ❌ | 0.745 | -0.125 | **-0.04** |

**Technical Notes:**
- EVA-CLIP setup requires GitHub clone: `https://github.com/baaivision/EVA.git`
- Model loaded from `/content/EVA/EVA-CLIP/rei` directory
- Embedding dimension: 768 (explicit mapping to avoid runtime errors)
- No consistency loss (3-head architecture doesn't need it)
- Same gradual unfreezing schedule as EXP017

**Critical Lessons:**
- ⭐ **Vision encoder matters**: EVA-CLIP (0.68) >> SigLip (0.66) >> DINOv3 (0.62)
- ⭐ **ImageNet zero-shot correlates with downstream performance**: 80.4% → LB 0.68
- ✅ **3-head + powerful backbone works**: Don't need 5-head when features are strong
- ✅ **Gradual unfreezing is effective**: Maintains generalization (gap -0.082)
- 📊 **Best gap + best LB**: First experiment to achieve both simultaneously
- 💡 **Pre-training quality > model size**: EVA-CLIP-L (428M) > SigLip-SO400M (400M)

**Next Steps to Reach LB 0.72:**
Based on this breakthrough, potential improvements:
1. **70% unfreezing** (like EXP014-005): May improve from 0.68 → 0.69-0.70
2. **5-head architecture** with EVA-CLIP: Add consistency loss for regularization
3. **Larger EVA-CLIP**: Try EVA02-CLIP-bigE-14 (4.4B params, 81.5% zero-shot)
4. **Feature concatenation** (like EXP024): Combine with EVA-CLIP backbone
5. **Ensemble**: EVA-CLIP + SigLip models (different strengths)

**Recommendation:**
- ✅ **EXP030-000 is now the best single model** (LB 0.68, Gap -0.082)
- ✅ **Try child experiments**:
  - child-001: 70% unfreezing (may reach 0.69-0.70)
  - child-002: 5-head + consistency loss
  - child-003: Larger input size (384×384 or 448×448)
- ⚠️ **EVA02-CLIP-bigE-14**: Worth trying but requires 4× more memory
- ⭐ **This is a major breakthrough**: Clear path to 0.72 now visible

---

#### EXP030-child001: EVA-CLIP with 70% Gradual Unfreezing ❌
**Configuration:**
- **Base architecture**: Same as EXP030-child000
- **Key change**: max_unfreeze_ratio: 0.5 → **0.7** (70% of transformer blocks)
  - EVA02-CLIP-L has 24 blocks: 70% = ~17 blocks unfrozen
  - vs child-000: 50% = ~12 blocks unfrozen
- **All other params**: Same as child-000 (dropout=0.1, batch_size=12, etc.)

**Motivation:**
- Based on EXP014-005 success: 70% unfreezing improved SigLip's Public LB rank (#1)
- Hypothesis: More capacity (17 blocks vs 12) → better adaptation to pasture domain
- Expected: LB 0.69-0.70 (vs child-000: 0.68)

**Results:**
- **Public LB**: **0.67** ❌ (-0.01 from child-000)
- **CV**: 0.762 (Mean R²: 0.765 ± 0.072)
- **LB-CV Gap**: **-0.092** (vs child-000: -0.082, +0.010 worse)
- Fold R² scores: [0.655, 0.725, 0.871, 0.776, 0.798]
- Best fold: Fold 2 (R² = 0.871) ⭐ **Highest single fold across all EXP030**
- Worst fold: Fold 0 (R² = 0.655)

**Key Observations:**
- ❌ **LB degradation**: 0.67 vs 0.68 (child-000, -0.01 drop)
- ❌ **Worse generalization gap**: -0.092 vs -0.082 (child-000, +0.010 worse)
- ⚠️ **CV identical**: 0.762 (both child-000 and child-001)
- ⚠️ **Higher variance**: std 0.072 vs 0.067 (child-000, +7.5% increase)
- 📊 **Fold 2 peaked**: R² 0.871 (highest in EXP030, vs child-000: 0.840)
  - But didn't translate to better LB
- 📊 **Fold 0 worse**: R² 0.655 vs 0.672 (child-000, -0.017 drop)

**Analysis:**
- **Why 70% unfreezing hurt performance**:
  1. **EVA-CLIP ≠ SigLip**: Different backbones respond differently to unfreezing
     - SigLip (EXP014-005): 70% unfreezing → LB 0.66 (#1 Public rank) ✅
     - EVA-CLIP (EXP030-001): 70% unfreezing → LB 0.67 (-0.01 drop) ❌
  2. **Overfitting to training folds**: Higher variance (std 0.072) indicates less stability
     - Fold 2 peaked at 0.871 but Public test performance dropped
     - Model overfit to strong folds, underperformed on weak folds
  3. **EVA-CLIP already powerful**: 80.4% zero-shot features may not need aggressive unfreezing
     - More unfreezing → disrupts pre-trained representations
     - 50% provides optimal balance for EVA-CLIP
  4. **Gap worsened**: -0.092 vs -0.082 (child-000)
     - Indicates worse generalization to Public test set
     - 70% unfreezing increases overfitting risk for EVA-CLIP

**Comparison: 50% vs 70% Unfreezing**

| Metric | child-000 (50%) | child-001 (70%) | Change |
|--------|----------------|----------------|--------|
| OOF CV | 0.762 | 0.762 | 0.000 |
| Public LB | **0.68** ⭐ | **0.67** ❌ | **-0.01** |
| Gap | **-0.082** ⭐ | -0.092 ❌ | +0.010 (worse) |
| Std | 0.067 | 0.072 | +0.005 (higher variance) |
| Best Fold | 0.840 | 0.871 | +0.031 |
| Worst Fold | 0.672 | 0.655 | -0.017 |
| **Result** | ⭐ **BEST** | ❌ Worse | **50% wins** |

**Comparison with SigLip (EXP014)**

| Backbone | 50% Unfreeze | 70% Unfreeze | Best |
|----------|-------------|-------------|------|
| **SigLip** (EXP014) | CV 0.772, LB 0.66 | CV 0.759, LB 0.66 (#1 rank) | **70%** wins (Public rank) ⭐ |
| **EVA-CLIP** (EXP030) | CV 0.762, LB **0.68** ⭐ | CV 0.762, LB 0.67 ❌ | **50%** wins (absolute LB) ⭐ |

**Critical Lessons:**
- ❌ **70% unfreezing doesn't universally help**: Works for SigLip, fails for EVA-CLIP
- 🔬 **Backbone-specific tuning**: Different vision encoders need different unfreezing ratios
  - SigLip: Benefits from more unfreezing (70%)
  - EVA-CLIP: Optimal at moderate unfreezing (50%)
  - Hypothesis: Stronger pre-training → less unfreezing needed
- ⚠️ **Higher fold variance = overfitting**: Std 0.072 indicates instability
  - Peaked fold (0.871) doesn't mean better generalization
  - Consistent performance across folds is more important
- 💡 **ImageNet zero-shot quality matters**: 80.4% features are already excellent
  - Aggressive unfreezing disrupts learned representations
  - Conservative unfreezing preserves transfer learning benefits
- 📊 **Gap is critical**: -0.082 (50%) vs -0.092 (70%)
  - Smaller gap indicates better generalization
  - LB improvement requires good gap, not just high CV

**Recommendation:**
- ✅ **Use EXP030-child000 (50% unfreezing)** for final submission (LB 0.68, Gap -0.082)
- ❌ **Avoid 70% unfreezing with EVA-CLIP**: Increases overfitting without LB gain
- 🔬 **Lesson learned**: Stronger backbones need conservative fine-tuning
- ⏭️ **Try child-002**: 5-head + consistency loss may provide better regularization
- 💡 **Hypothesis for 0.72**: Ensemble (EVA-CLIP 50% + SigLip 70%) may reach 0.70-0.72

---

#### EXP030-child003: EVA-CLIP with Dropout 0.15 + Batch Size 16 ⚠️
**Configuration:**
- **Base architecture**: Same as EXP030-child000
- **Key changes**:
  - dropout: 0.1 → **0.15** (match successful SigLip experiments)
  - batch_size: 12 → **16** (VRAM available)
  - batch_size_valid: 24 → **32**
- **All other params**: Same as child-000 (50% unfreezing, 3-head, etc.)

**Motivation:**
- All successful SigLip experiments (EXP014, EXP024, EXP021) used dropout=0.15
- EVA-CLIP (80.4% zero-shot) is stronger than SigLip (78%) → may need stronger regularization
- dropout=0.1 might be too conservative
- Expected: Gap improvement -0.082 → -0.07x

**Results:**
- **Public LB**: **0.68** (same as child-000)
- **CV**: 0.777 (OOF R²) ← **+0.015 improvement** vs child-000
- **Mean R²**: 0.771 ± 0.063
- **LB-CV Gap**: **-0.097** ❌ (vs child-000: -0.082, +0.015 worse)
- Fold R² scores: [0.700, 0.705, 0.868, 0.797, 0.784]
- Best fold: Fold 2 (R² = 0.868)
- Worst fold: Fold 1 (R² = 0.705)

**Key Observations:**
- ✅ **CV improved significantly**: 0.777 vs 0.762 (child-000, +0.015)
- ✅ **Better fold stability**: std 0.063 vs 0.067 (child-000, -0.004)
- ✅ **Fold 0 improved**: R² 0.700 vs 0.672 (child-000, +0.028)
- ✅ **Fold 1 improved**: R² 0.705 vs 0.725 (child-000, slightly worse but more stable)
- ❌ **LB unchanged**: 0.68 (same as child-000)
- ❌ **Gap worsened**: -0.097 vs -0.082 (child-000, +0.015 worse)
- 📊 **CV-LB trade-off**: Higher CV but worse generalization to Public test

**Analysis:**
- **Why dropout 0.15 improved CV but hurt Gap**:
  1. **Stronger regularization during training**: dropout 0.15 prevents overfitting to training folds
     - Better validation R² scores across all folds
     - More stable fold performance (lower std)
  2. **But over-regularization for test set**: EVA-CLIP features already well-regularized
     - 80.4% ImageNet zero-shot indicates strong pre-training
     - dropout 0.15 may be too aggressive, losing important features
     - Test distribution differs from training → regularization mismatch
  3. **SigLip vs EVA-CLIP difference**: SigLip (78% zero-shot) benefits from dropout 0.15, but EVA-CLIP doesn't
     - Stronger pre-trained models need **less** regularization
     - dropout 0.1 preserves more pre-trained features
  4. **Gap increased**: -0.082 → -0.097 (+0.015 worse)
     - Model fits validation better but generalizes worse to Public test
     - Suggests validation distribution ≠ Public test distribution

**Comparison: Dropout 0.1 vs 0.15**

| Metric | child-000 (dropout 0.1) | child-003 (dropout 0.15) | Change |
|--------|------------------------|-------------------------|--------|
| OOF CV | 0.762 | 0.777 | **+0.015** ✅ |
| Mean R² | 0.765 | 0.771 | +0.006 |
| Std | 0.067 | 0.063 | **-0.004** ✅ |
| Public LB | **0.68** ⭐ | **0.68** | 0.000 |
| Gap | **-0.082** ⭐ | -0.097 ❌ | **+0.015** (worse) |
| Fold 0 | 0.672 | 0.700 | +0.028 ✅ |
| Fold 1 | 0.725 | 0.705 | -0.020 |
| Fold 2 | 0.840 | 0.868 | +0.028 ✅ |
| **Result** | **BEST** ⭐ | ⚠️ CV better, Gap worse | **child-000 wins** |

**Critical Lessons:**
- ❌ **SigLip patterns don't transfer to EVA-CLIP**: dropout 0.15 works for SigLip, fails for EVA-CLIP
- 🔬 **Stronger pre-training needs less regularization**: 80.4% zero-shot → dropout 0.1 optimal
- ⚠️ **CV improvement ≠ LB improvement**: +0.015 CV but +0.015 Gap (worse)
- 📊 **Gap is the key metric**: -0.082 (dropout 0.1) better than -0.097 (dropout 0.15)
- 💡 **Pre-trained model strength matters**: More powerful features → less dropout needed
- ✅ **Validation stability improved**: Lower std, better fold balance
- ❌ **But generalization hurt**: Public test requires dropout 0.1 for EVA-CLIP

**Recommendation:**
- ✅ **Keep dropout=0.1 for EVA-CLIP** (child-000 remains best)
- ❌ **Avoid dropout=0.15 with EVA-CLIP**: Improves CV but hurts generalization
- 🔬 **Lesson learned**: Don't blindly copy SigLip hyperparameters to EVA-CLIP
- 📊 **Gap > CV**: Prioritize small gap over high CV for final model selection
- 💡 **Next**: Try other strategies (5-head, feature concat) with dropout=0.1

---

#### EXP030-child008: Strong Augmentation (RandomResizedCrop + High Hue) ⚠️
**Configuration:**
- **Base architecture**: Same as EXP030-child000
- **Key change**: Strong augmentation pipeline (`version: strong`)
  - RandomResizedCrop: scale=(0.7, 1.0) - Wider range for height variance
  - ColorJitter: brightness=0.4, contrast=0.4, saturation=0.4, **hue=0.2**
  - GaussNoise: var=(5, 30), p=0.3
  - ImageCompression: quality=70-100, p=0.3
  - MotionBlur, RGBShift(±20), CLAHE
- **Motivation**: Absorb camera height/brightness/equipment variance

**Results:**
- **Public LB**: **0.66** ❌ (-0.02 from child-000)
- **CV**: 0.732 (OOF R²)
- **Mean R²**: 0.734 ± 0.071
- **LB-CV Gap**: **-0.072** ⭐ (vs child-000: -0.082, **BEST GAP**)
- Fold R² scores: [0.621, 0.702, 0.833, 0.745, 0.768]

**Key Observations:**
- ❌ **LB degradation**: 0.66 vs 0.68 (child-000, -0.02 drop)
- ❌ **CV degradation**: 0.732 vs 0.762 (child-000, -0.030 drop)
- ⭐ **Best gap achieved**: -0.072 (vs child-000: -0.082, +0.010 improvement)
- ⚠️ **CV-LB correlation maintained**: CV dropped → LB dropped proportionally
- 📊 **Strong augmentation hurts performance**: RandomResizedCrop and high Hue likely cause issues

**Analysis:**
- **Why strong augmentation failed**:
  1. **RandomResizedCrop breaks label consistency**: Cropping unknown grass density region makes label inaccurate
     - scale=0.7 means 30% of image can be cropped
     - Label (grams) should also be adjusted but we can't know exact ratio
  2. **High Hue shift (±0.2) causes Green/Dead confusion**: Large hue change can turn green grass to brown
     - Model learns incorrect color-biomass mapping
     - Green → brown shift makes model think green grass is dead
  3. **Gap improved but absolute performance dropped**: Better CV-LB correlation but lower baseline
     - Over-augmentation acts as regularization (smaller gap)
     - But destroys useful training signal (lower CV and LB)
- **Physics-breaking augmentation is harmful**: Unlike classification, regression needs label consistency

**Critical Lessons:**
- ❌ **RandomResizedCrop is dangerous for regression**: Can't adjust label for cropped area
- ❌ **Large Hue shift is dangerous**: Confuses Green/Dead distinction (color → biomass mapping)
- ⭐ **Gap improved but at cost of performance**: Over-regularization via augmentation
- 📊 **Physics consistency matters**: Augmentation must preserve label validity
- 💡 **CV-LB correlation is good**: Gap -0.072 suggests model generalizes proportionally

---

#### EXP030-child010: Physics-Safe Augmentation + Mosaic + Mixup ❌
**Configuration:**
- **Base architecture**: Same as EXP030-child000
- **Key innovation**: Physics-consistent augmentation (`version: physics_safe`)
  - **NO RandomResizedCrop** (breaks label consistency)
  - **LIMITED Hue shift**: ±8 (minimal, avoids Green/Dead confusion)
  - Strong brightness/contrast: ±0.4 (safe, real-world variation)
  - ToGray: p=0.1 (forces texture-based learning)
- **Mosaic augmentation**: 4 images → 2x2 grid, label = mean of 4
  - mosaic_prob=0.5 (50% of samples)
  - Simulates "camera further away" scenario
- **Mixup**: alpha=0.4
  - x_new = λ·x_i + (1-λ)·x_j
  - y_new = λ·y_i + (1-λ)·y_j
  - Physics-consistent for regression

**Motivation:**
- Address camera height variance via Mosaic (scale invariance)
- Fill data manifold via Mixup (important for small dataset)
- Avoid physics-breaking augmentations (no RRC, minimal Hue)

**Results:**
- **Public LB**: **0.64** ❌❌ (-0.04 from child-000)
- **CV**: 0.729 (OOF R²)
- **Mean R²**: 0.731 ± 0.093
- **LB-CV Gap**: **-0.089** (vs child-000: -0.082, +0.007 worse)
- Fold R² scores: [0.602, 0.659, 0.872, 0.769, 0.752]

**Key Observations:**
- ❌❌ **Severe LB degradation**: 0.64 vs 0.68 (child-000, -0.04 drop)
- ❌ **CV degradation**: 0.729 vs 0.762 (child-000, -0.033 drop)
- ⚠️ **High fold variance**: std 0.093 (vs child-000: 0.067, +39% increase)
- ❌ **Fold 0 collapsed**: R² 0.602 (vs child-000: 0.672, -0.070 drop)
- ❌ **Fold 1 collapsed**: R² 0.659 (vs child-000: 0.725, -0.066 drop)
- ✅ **Fold 2 peaked**: R² 0.872 (vs child-000: 0.840, +0.032)
- 📊 **CV-LB correlation maintained**: CV dropped → LB dropped

**Analysis:**
- **Why Mosaic + Mixup failed**:
  1. **Mosaic creates artificial patterns**: 2x2 grid doesn't exist in real world
     - Model learns to recognize mosaic boundaries (overfitting to artifact)
     - Test images don't have mosaic structure → poor transfer
  2. **Mixup interpolation may not match regression manifold**:
     - Linear interpolation of grass images ≠ linear interpolation of biomass
     - Mixed images have unrealistic textures (ghosting artifacts)
  3. **Label averaging in Mosaic may be incorrect**:
     - Mean of 4 labels assumes uniform density across images
     - Reality: grass density varies spatially within each image
  4. **High variance indicates instability**: std 0.093 suggests inconsistent learning
     - Some folds benefit (Fold 2: 0.872), others collapse (Fold 0: 0.602)
     - Mosaic/Mixup helps certain data distributions but hurts others
  5. **EVA-CLIP features already strong**: 80.4% zero-shot doesn't need aggressive data augmentation
     - Simpler training (child-000) works better

**Comparison: child-000 vs child-008 vs child-010**

| Metric | child-000 (baseline) | child-008 (strong aug) | child-010 (Mosaic+Mixup) |
|--------|---------------------|----------------------|-------------------------|
| Augmentation | Default | Strong (RRC, high Hue) | Physics-safe + Mosaic + Mixup |
| OOF CV | **0.762** ⭐ | 0.732 | 0.729 |
| Public LB | **0.68** ⭐ | 0.66 ❌ | 0.64 ❌❌ |
| Gap | -0.082 | **-0.072** ⭐ | -0.089 |
| Std | **0.067** ⭐ | 0.071 | 0.093 ❌ |
| Fold 0 | 0.672 | 0.621 | 0.602 ❌ |
| **Result** | **BEST** ⭐ | ❌ Worse | ❌❌ Worst |

**Critical Lessons:**
- ❌❌ **Mosaic is harmful for this task**: Creates artificial patterns not in test set
- ❌ **Mixup doesn't help EVA-CLIP**: Strong pre-trained features don't need data augmentation tricks
- ❌ **Physics-safe ≠ performance-safe**: Avoiding RRC/Hue is good, but Mosaic/Mixup hurt more
- ⚠️ **High variance is a warning sign**: std 0.093 indicates unstable training
- 📊 **Simpler is better**: child-000 (default augmentation) remains best
- 💡 **EVA-CLIP doesn't need aggressive augmentation**: 80.4% zero-shot features are already robust
- 🔬 **Regression ≠ classification**: Tricks that work for classification (Mosaic, Mixup) may fail for regression

**Recommendation:**
- ✅ **Stick with EXP030-child000** (LB 0.68, Gap -0.082) - default augmentation is optimal
- ❌ **Avoid RandomResizedCrop**: Breaks label consistency
- ❌ **Avoid high Hue shift**: Confuses Green/Dead distinction
- ❌ **Avoid Mosaic for this task**: Creates artifacts that don't exist in test set
- ❌ **Avoid Mixup for EVA-CLIP**: Strong backbone doesn't need it
- 💡 **Lesson**: For powerful pre-trained models, simpler training is better

---

#### EXP030-child011: Improved Augmentation (Minimal RRC, Limited Hue) ❌❌
**Configuration:**
- **Base architecture**: Same as EXP030-child000
- **Key changes**: "Physics-aware" augmentation improvements
  - RandomResizedCrop: scale=(0.85, 1.0) → **(0.95, 1.0)** (minimal crop)
  - RandomResizedCrop: p=0.5 → **p=0.3** (reduced probability)
  - Hue: 0.15 → **0.05** (minimal to avoid Green/Dead confusion)
  - RandomShadow: p=0.1 → **p=0.25** (stronger, safe augmentation)
  - Added: Transpose (p=0.3)
  - Added: GaussNoise (var=3-15, p=0.15)

**Motivation:**
- Address "physics consistency" concerns from augmentation analysis
- Minimal RRC to preserve label consistency
- Limited Hue to avoid Green/Dead color confusion
- Stronger shadow for real-world lighting variation

**Results:**
- **Public LB**: **0.64** ❌❌ (-0.04 from child-000)
- **CV**: 0.769 (OOF R²) (+0.007 from child-000)
- **Mean R²**: 0.763 ± 0.058
- **LB-CV Gap**: **-0.129** ❌❌ (vs child-000: -0.082, **+0.047 worse**)
- Fold R² scores: [0.683, 0.709, 0.839, 0.792, 0.794]

**Key Observations:**
- ❌❌ **Catastrophic LB degradation**: 0.64 vs 0.68 (child-000, -0.04 drop)
- ⚠️ **CV slightly improved**: 0.769 vs 0.762 (child-000, +0.007)
- ❌❌ **Gap explosion**: -0.129 (worst gap in EXP030 series)
- 📊 **CV-LB anti-correlation**: Higher CV → Lower LB (dangerous pattern)

**Analysis - Why "Physics-Aware" Augmentation Failed:**
1. **Test set likely contains RRC-like variations**:
   - Test images may have different camera distances/zoom levels
   - Training with scale=(0.85, 1.0) simulates these variations
   - Restricting to scale=(0.95, 1.0) removes this robustness
2. **Test set likely contains Hue variations**:
   - Different cameras have different color calibrations
   - Hue=0.15 helps model learn color invariance
   - Restricting to Hue=0.05 makes model color-sensitive
3. **"Physics correct" ≠ "Test distribution correct"**:
   - Our augmentation analysis assumed ideal conditions
   - Real test images have "physics-incorrect" variations (camera artifacts, color shifts)
   - Model needs to handle these, not avoid them
4. **Default augmentation is empirically optimal**:
   - child-000's augmentation was tuned through many experiments
   - It matches the test distribution better than theory suggests

**Critical Lessons:**
- ❌❌ **Don't "fix" augmentation based on theory**: Empirical results trump physics arguments
- ❌❌ **Default augmentation (RRC scale=0.85-1.0, Hue=0.15) is optimal**: Don't change it
- ⚠️ **CV improvement can mean LB degradation**: Gap explosion is a warning sign
- 📊 **Test distribution has "physics-incorrect" variations**: Model must handle them
- 💡 **Lesson**: If augmentation works, don't "improve" it based on theoretical concerns

**Recommendation:**
- ✅ **Keep default augmentation** (child-000) for all future experiments
- ❌ **Never restrict RRC scale or Hue** based on physics arguments
- 📊 **Gap is the key metric**: Large gap (>0.1) indicates distribution mismatch

---

### EXP036: EVA-CLIP + 5-Head + Consistency Loss ⭐⭐ **NEW BEST**
**Model**: EVA02-CLIP-L-14-336 with 5-head architecture and Consistency Loss

#### EXP036-child000: EVA-CLIP + 5-Head + Consistency Loss ⭐⭐
**Configuration:**
- **Backbone**: EVA02-CLIP-L-14-336 (same as EXP030-child000)
- **Architecture**: **5-head** (vs EXP030's 3-head)
  - head_green → Dry_Green_g (10% weight)
  - head_dead → Dry_Dead_g (10% weight)
  - head_clover → Dry_Clover_g (10% weight)
  - head_gdm → GDM_g (20% weight)
  - head_total → Dry_Total_g (50% weight)
- **Loss**: Weighted SmoothL1 + **Consistency Loss** (weight=0.1)
  - Constraint 1: Dry_Dead_g + GDM_g ≈ Dry_Total_g
  - Constraint 2: Dry_Clover_g + Dry_Green_g ≈ GDM_g
- **Training**: Same as EXP030-child000 (50% gradual unfreezing, dropout=0.1)
- **Augmentation**: Default (same as EXP030-child000)

**Hypothesis:**
- EVA-CLIP's strong features (LB 0.68) + 5-head's explicit target modeling
- Consistency Loss provides physical constraint regularization
- Direct prediction of all 5 targets may improve Dead/Clover accuracy

**Results:**
- **Public LB**: **0.69** ⭐⭐ **NEW BEST** (+0.01 from EXP030-000)
- **CV**: **0.773** ⭐⭐ **NEW BEST** (OOF R², +0.011 from EXP030-000)
- **Mean R²**: 0.770 ± 0.059
- **LB-CV Gap**: **-0.083** (vs EXP030-000: -0.082, similar)
- Fold R² scores: [0.685, 0.726, 0.854, 0.784, 0.804]
- Best fold: Fold 2 (R² = 0.854)
- Worst fold: Fold 0 (R² = 0.685)

**Key Observations:**
- ⭐⭐ **NEW LB RECORD**: 0.69 (first time breaking 0.68 barrier)
- ⭐⭐ **NEW CV RECORD**: 0.773 (first time breaking 0.77 barrier)
- ✅ **Gap maintained**: -0.083 (excellent generalization preserved)
- ✅ **Fold stability**: std 0.059 (best in EXP030/036 series)
- ✅ **All folds improved**: Compared to EXP030-000
  - Fold 0: 0.685 vs 0.672 (+0.013)
  - Fold 1: 0.726 vs 0.725 (+0.001)
  - Fold 2: 0.854 vs 0.840 (+0.014)
  - Fold 3: 0.784 vs 0.763 (+0.021)
  - Fold 4: 0.804 vs 0.812 (-0.008)

**Analysis - Why 5-Head + Consistency Loss Worked:**
1. **Explicit Dead/Clover prediction improves accuracy**:
   - EXP030 (3-head) calculates Dead = Total - GDM, Clover = GDM - Green
   - EXP036 (5-head) learns each target directly
   - Direct learning captures target-specific patterns better
2. **Consistency Loss provides regularization**:
   - Physical constraints act as soft regularization
   - Prevents individual heads from overfitting independently
   - Maintains relationships between targets
3. **EVA-CLIP features + 5-head synergy**:
   - Strong features enable 5 separate heads without overfitting
   - Unlike SigLip, EVA-CLIP has enough capacity for 5 heads
4. **Why this didn't overfit (unlike some SigLip experiments)**:
   - EVA-CLIP's 80.4% zero-shot features are more generalizable
   - Consistency Loss provides implicit regularization
   - dropout=0.1 is optimal for EVA-CLIP

**Comparison: EXP030 vs EXP036**

| Metric | EXP030-000 (3-head) | EXP036-000 (5-head) | Change |
|--------|--------------------|--------------------|--------|
| Heads | 3 (Green, GDM, Total) | 5 (all targets) | +2 |
| Consistency Loss | ❌ | ✅ (weight=0.1) | Added |
| OOF CV | 0.762 | **0.773** ⭐ | **+0.011** |
| Public LB | 0.68 | **0.69** ⭐ | **+0.01** |
| Gap | -0.082 | -0.083 | -0.001 |
| Std | 0.067 | **0.059** ⭐ | **-0.008** |
| Fold 0 | 0.672 | 0.685 | +0.013 |
| **Result** | Previous best | **NEW BEST** ⭐⭐ | **+0.01 LB** |

**Critical Lessons:**
- ⭐⭐ **5-head + Consistency Loss improves EVA-CLIP**: LB 0.68 → 0.69 (+1.5% relative)
- ✅ **Direct prediction > calculated derivation**: Learning Dead/Clover explicitly is better
- ✅ **Consistency Loss is effective regularization**: Maintains physical relationships
- ✅ **Gap preserved**: 5-head doesn't hurt generalization with EVA-CLIP
- 📊 **Both CV and LB improved**: Rare case where both metrics improve together
- 💡 **Recipe for success**: EVA-CLIP + 5-head + Consistency Loss + default augmentation

**Recommendation:**
- ⭐⭐ **EXP036-child000 is the new best model** (LB 0.69, CV 0.773, Gap -0.083)
- ✅ **Use for final submission**: Best LB and CV with excellent gap
- 🔬 **Next steps to reach 0.72**:
  - Ensemble: EXP036 + EXP030 (different head structures)
  - Try consistency_weight tuning (0.05, 0.15, 0.2)
  - Feature concatenation (like EXP024) with 5-head

#### EXP036-child001: EVA-CLIP + 5-Head + 60% Unfreezing
**Configuration:**
- Same as EXP036-child000, but with **max_unfreeze_ratio=0.6** (vs 0.5)

**Results:**
- **Public LB**: **0.68**
- **OOF R²**: 0.759
- **LB-CV Gap**: **-0.079** (excellent)
- **Mean R²**: 0.762 ± 0.075
- Fold R² scores: [0.646, 0.709, 0.860, 0.789, 0.804]

**Key Observations:**
- ✅ **Stable LB 0.68**: Same as child000 with 60% unfreezing
- ✅ **Good gap**: -0.079
- ⚠️ **OOF slightly lower**: 0.759 vs 0.773 (child000)
- 📊 **CV-LB correlation maintained**: Higher unfreeze didn't help LB

#### EXP036-child002: EVA-CLIP + 5-Head + 70% Unfreezing
**Configuration:**
- Same as EXP036-child000, but with **max_unfreeze_ratio=0.7** (vs 0.5)

**Results:**
- **Public LB**: **0.68**
- **OOF R²**: 0.766
- **LB-CV Gap**: **-0.086**
- **Mean R²**: 0.775 ± 0.075
- Fold R² scores: [0.648, 0.759, 0.882, 0.797, 0.788]

**Key Observations:**
- ✅ **Stable LB 0.68**: Same as child000/001 with 70% unfreezing
- ✅ **OOF improved**: 0.766 vs 0.759 (child001)
- ⚠️ **Gap slightly worse**: -0.086 vs -0.079 (child001)
- 📊 **Best fold R²**: Fold 2 = 0.882 (highest in EXP036 series)

**EXP036 Ablation Summary:**

| Child | Change | OOF R² | LB | Gap |
|-------|--------|--------|-----|------|
| 000 | baseline (50%, cons=0.1) | 0.773 | **0.69** ⭐ | -0.083 |
| 001 | 60% unfreeze | 0.759 | 0.68 | -0.079 |
| 002 | 70% unfreeze | 0.766 | 0.68 | -0.086 |
| 003 | consistency_weight=0.2 | 0.769 | **0.65** ❌ | -0.119 |
| 004 | Dead/Clover weight↑ | 0.762 | **0.63** ❌❌ | -0.132 |

**Critical Insights:**

✅ **Unfreezing ablation (child001/002):**
- 50% unfreezing is optimal for LB (0.69)
- More unfreezing → worse LB despite similar CV
- CV-LB correlation maintained at LB 0.68 level

❌ **Hyperparameter changes FAILED (child003/004):**
- **child003**: consistency_weight 0.1→0.2 → **LB -0.04** (0.69→0.65)
  - Stronger consistency = over-regularization
  - CV barely changed (0.773→0.769) but LB collapsed
- **child004**: Loss weights shifted to Dead/Clover → **LB -0.06** (0.69→0.63)
  - Evaluation metric: Dry_Total_g=50%, but training used 40%
  - Mismatch between training objective and evaluation = severe LB drop
  - **WORST gap in EXP036 series**: -0.132

⭐ **Key Lessons:**
- **DON'T change consistency_weight from 0.1** - it's tuned optimally
- **DON'T change loss weights from evaluation metric** - must match exactly
- **child000 config is the golden standard** - any deviation hurts LB

---

### EXP037: MIL Sum-Pooling Training (3-Head)
**Model**: EVA02-CLIP-L-14-336 with MIL-style pair sum-pooling

#### EXP037-child000: MIL Sum-Pooling + Dynamic Gate
**Configuration:**
- **Backbone**: EVA02-CLIP-L-14-336 (same as EXP030/036)
- **Architecture**: **3-head** (Green, GDM, Total) - derived Dead/Clover
- **Key Innovation**: **MIL-style sum-pooling training**
  - Split image into left/right halves (1000x1000 each)
  - Predict each half separately
  - Sum predictions → compare to **full labels** (not halved)
  - This differs from EXP030 which also sums predictions but EXP037 uses different loss scaling
- **Training**: 40 epochs, LR=0.003, dropout=0.1
- **Gradual Unfreezing**: 50% max, with **Dynamic Gate**
  - patience=3 epochs, min_improve=0.002
  - Stops further unfreezing if no R² improvement
- **CV Strategy**: StratifiedGroupKFold (State, Sampling_Date)

**Hypothesis:**
- MIL-style training with full labels may learn better per-half representations
- Dynamic gate prevents over-unfreezing when R² plateaus

**Results:**
- **Public LB**: **0.68**
- **CV**: 0.758 (Mean R²)
- **LB-CV Gap**: **-0.078** ⭐ (best gap in EXP036/037 series)
- **Mean R²**: 0.758 ± 0.070
- Fold R² scores: [0.669, 0.714, 0.870, 0.740, 0.797]

**Key Observations:**
- ✅ **Stable LB 0.68**: Matches EXP036-child001/002
- ⭐ **Best gap**: -0.078 (vs -0.083 for EXP036-000, -0.095 for EXP036-002)
- ✅ **Lower CV but better generalization**: CV 0.758 < 0.773 but gap is better
- 📊 **3-head vs 5-head**: 3-head with MIL achieves same LB as 5-head + consistency

**Comparison: EXP036 vs EXP037**

| Metric | EXP036-000 (5-head) | EXP037-000 (3-head MIL) |
|--------|--------------------|-----------------------|
| Heads | 5 (all targets) | 3 (Green, GDM, Total) |
| Loss | Consistency Loss | Weighted SmoothL1 |
| CV (Mean R²) | 0.770 | 0.758 |
| Public LB | **0.69** | 0.68 |
| Gap | -0.083 | **-0.078** ⭐ |
| Dynamic Gate | ❌ | ✅ |

**Critical Insights from EXP036/037 Series:**
- ✅ **CV-LB correlation is strong**: All experiments with CV ~0.76-0.77 achieve LB 0.68
- ⭐ **Gap is the differentiator**: Lower gap → potentially better private LB
- 📊 **5-head slight edge**: EXP036-000 (LB 0.69) > EXP037-000 (LB 0.68)
- 💡 **For final submission**: Consider ensemble of EXP036-000 + EXP037-000 (different architectures)

---

### EXP048: EVA-CLIP + Height Auxiliary Task (Regression) ❌
**Model**: EVA02-CLIP-L-14-336 with Height Auxiliary Regression Task

#### EXP048-child000: Height Auxiliary Task
**Configuration:**
- **Base**: EXP036-child000 (EVA-CLIP + 5-Head + Consistency Loss)
- **Key Innovation**: **Height Auxiliary Task**
  - Adds a 6th head (head_height) to predict normalized height
  - Height normalized to [0, 1+] range (HEIGHT_NORM = 50.0cm)
  - Height prediction used only during training, not inference
  - Uses SmoothL1Loss for height regression
- **Motivation**: OOF analysis showed strong correlation between Height and prediction error
  - Green: r = -0.653 (tall grass → underestimation)
  - Teaching model to predict height forces learning of height-related visual features
- **Training**: Same as EXP036-child000 (50% gradual unfreezing)

**Results:**
- **Public LB**: **0.67** ❌ (-0.02 from EXP036-000)
- **Mean R²**: 0.768 ± 0.063
- **LB-CV Gap**: ~-0.098 (estimated, oof_r2 not available)
- Fold R² scores: [0.682, 0.734, 0.872, 0.779, 0.769]
- **Note**: oof_r2 not available (only mean_r2 from fold averages)

**Key Observations:**
- ❌ **LB degradation**: 0.67 vs 0.69 (EXP036-000, -0.02 drop)
- ⚠️ **Mean R² similar**: 0.768 vs 0.773 (EXP036-000)
- ❌ **Height auxiliary task did not help generalization**
- 📊 **Analysis**: Adding auxiliary task may have distracted model from main targets

**Why Height Auxiliary Task Failed:**
1. **Auxiliary task ≠ better features**: Model may prioritize height prediction over biomass
2. **Height not visible in test images**: Test images lack height labels, but model was trained to predict height
3. **Distribution shift**: Height patterns in training may not match test set
4. **Previous lesson confirmed**: Auxiliary tasks don't help for this competition (cf. EXP021, EXP022)

**Key Lessons:**
- ❌ **Height auxiliary regression doesn't help**: LB dropped from 0.69 → 0.67
- ❌ **Confirms auxiliary task pattern**: EXP021 (metadata), EXP022 (species), EXP048 (height) all failed
- ⭐ **Stick with EXP036-child000**: Best model remains EVA-CLIP + 5-head + Consistency Loss

---

### EXP031: EVA-CLIP with 3-Crop Training Strategy ❌❌ **SEVERE DEGRADATION**
**Model**: EVA02-CLIP-L-14-336 with 3-crop training (stride=500)

#### EXP031-child000: 3-Crop Training + Dropout 0.15
**Configuration:**
- **Base architecture**: EXP030-child000
- **Key innovation**: **3-crop training strategy** with stride=500
  - Training: 3 overlapping crops per image (0-1000, 500-1500, 1000-2000)
  - Validation/Inference: 2 crops (left/right halves, same as EXP030)
  - Training samples increased by 1.5× (2× → 3× per image)
- **Augmentation**: Enhanced probabilities (+0.1~0.15) to handle increased data
- **Model**: EVA02-CLIP-L-14-336 (same as EXP030)
- **Architecture**: 3-head (predict Green, GDM, Total)
- **Hyperparameters**:
  - dropout: **0.15** (from EXP030-003)
  - batch_size: **16**
  - Gradual unfreezing: 50%
  - Epochs: 40

**Hypothesis:**
- 3-crop training provides more diverse training samples
- Center crop (500-1500) captures transition areas between left/right
- Increased augmentation prevents overfitting from 1.5× more samples
- Expected: More robust features → better generalization (Gap improvement)

**Results:**
- **Public LB**: **0.66** ❌❌ ← **SEVERE -0.02 degradation** vs EXP030-000
- **CV**: 0.766 (OOF R²) ← +0.004 vs EXP030-000
- **Mean R²**: 0.770 ± 0.083
- **LB-CV Gap**: **-0.106** ❌❌ (vs EXP030-000: -0.082, **+0.024 worse**)
- Fold R² scores: [0.637, 0.720, 0.874, 0.796, 0.825]
- Best fold: Fold 2 (R² = 0.874) ← **Highest across all EXP030/031**
- Worst fold: Fold 0 (R² = 0.637) ← **Lowest across all EXP030/031**

**Key Observations:**
- ❌❌ **SEVERE LB degradation**: 0.66 vs 0.68 (EXP030-000, **-0.02 drop**)
- ❌❌ **Gap massively worsened**: -0.106 vs -0.082 (EXP030-000, **+0.024 worse**)
- ❌ **Extreme fold instability**: std 0.083 vs 0.067 (EXP030-000, **+0.016 increase**)
- ❌ **Fold 0 severely degraded**: R² 0.637 vs 0.672 (EXP030-000, **-0.035 drop**)
- ⚠️ **Fold 2 peaked excessively**: R² 0.874 vs 0.840 (EXP030-000, +0.034)
  - But this didn't translate to better LB → overfitting signal
- ⚠️ **CV barely changed**: 0.766 vs 0.762 (EXP030-000, +0.004)
- 📊 **Worst experiment in EXP030 series**: LB 0.66, Gap -0.106

**Analysis:**
- **Why 3-crop training failed catastrophically**:
  1. **Increased data didn't help**: 1.5× more training samples (3-crop vs 2-crop)
     - Expected: More diverse samples → better generalization
     - Reality: More samples → worse overfitting
     - Training data distribution mismatch with Public test
  2. **Center crop introduced distribution shift**:
     - Center crop (500-1500) has different characteristics than left/right
     - Model learned patterns specific to center crop
     - But inference uses only left/right → mismatch
     - Gap increased from -0.082 to -0.106 (+0.024)
  3. **Overfitting increased, not decreased**:
     - Fold 2: R² 0.874 (highest ever) but LB dropped
     - Fold variance increased: std 0.067 → 0.083
     - Model memorized training distribution instead of learning robust features
  4. **Enhanced augmentation wasn't enough**:
     - Increased aug probabilities (+0.1~0.15) couldn't compensate
     - 3-crop data diversity ≠ true data augmentation
     - Same image with different crops still shares characteristics
  5. **dropout 0.15 amplified the problem**:
     - EXP030-003 (2-crop + dropout 0.15): LB 0.68, Gap -0.097
     - EXP031-000 (3-crop + dropout 0.15): LB 0.66, Gap -0.106
     - 3-crop + dropout 0.15 = worst combination

**Comparison: 2-Crop vs 3-Crop Training**

| Metric | EXP030-000 (2-crop, dropout 0.1) | EXP031-000 (3-crop, dropout 0.15) | Change |
|--------|----------------------------------|-----------------------------------|--------|
| Training samples/fold | 714 (357 × 2) | **1071 (357 × 3)** | **+50%** |
| OOF CV | 0.762 | 0.766 | +0.004 |
| Mean R² | 0.765 | 0.770 | +0.005 |
| Std | 0.067 | **0.083** ❌ | **+0.016** (unstable) |
| Public LB | **0.68** ⭐⭐⭐ | **0.66** ❌❌ | **-0.02** |
| Gap | **-0.082** ⭐ | **-0.106** ❌❌ | **+0.024** (worse) |
| Fold 0 | 0.672 | **0.637** ❌ | **-0.035** |
| Fold 2 | 0.840 | **0.874** ⚠️ | +0.034 (overfit) |
| **Result** | ⭐ **BEST** | ❌❌ **WORST** | **2-crop wins decisively** |

**Comparison with EXP030-003**

| Experiment | Crops | Dropout | CV | LB | Gap | Result |
|------------|-------|---------|----|----|-----|--------|
| EXP030-000 | 2-crop | 0.1 | 0.762 | **0.68** ⭐ | **-0.082** ⭐ | **BEST** |
| EXP030-003 | 2-crop | 0.15 | 0.777 | 0.68 | -0.097 | ⚠️ CV↑ Gap↓ |
| **EXP031-000** | **3-crop** | **0.15** | 0.766 | **0.66** ❌ | **-0.106** ❌ | ❌❌ **WORST** |

**Critical Lessons:**
- ❌❌ **More training data ≠ better performance**: 3-crop (1.5× data) severely hurt LB
- ❌ **Data augmentation via cropping failed**: Different crops of same image don't provide true diversity
- 🔬 **Distribution mismatch is critical**: Center crop introduced patterns not present in inference
- ⚠️ **Overfitting amplified with more samples**: Higher fold variance, worse generalization
- 📊 **Worst gap ever in EVA-CLIP series**: -0.106 (vs -0.082 for 2-crop)
- 💡 **Simplicity wins**: 2-crop (left/right) is optimal for this dataset
- ❌ **Stride-based cropping doesn't work**: Overlapping crops don't help
- 🔬 **Lesson learned**: Data augmentation via multiple crops of same image backfires

**Why 3-Crop Failed (Detailed Analysis):**
1. **Training-Inference Mismatch**:
   - Training: 3 crops (left, center, right) → each label ÷ 2
   - Inference: 2 crops (left, right) → predictions summed
   - Center crop patterns learned but never used in inference
   - Model wasted capacity learning irrelevant features

2. **False Data Diversity**:
   - 3 crops from same image are highly correlated
   - Doesn't provide true data augmentation
   - Model still sees same pasture characteristics
   - Only spatial position varies, not semantic content

3. **Overfitting to Training Distribution**:
   - Fold 2 peaked at 0.874 (highest R² ever)
   - But LB dropped to 0.66 (lowest in EXP030 series)
   - Model memorized crop-specific patterns
   - Failed to generalize to Public test

4. **Augmentation Compensation Failed**:
   - Enhanced augmentation (+0.1~0.15 probabilities)
   - Couldn't compensate for bad data strategy
   - More augmentation + more crops = compounded overfitting

**Recommendation:**
- ❌❌ **NEVER use 3-crop training**: Severe LB degradation (-0.02)
- ✅ **Stick with 2-crop (left/right)**: Proven optimal for this task
- ❌ **Avoid stride-based cropping**: Doesn't improve generalization
- 📊 **Gap is critical**: -0.082 (2-crop) >> -0.106 (3-crop)
- 💡 **Data quality > data quantity**: 2 good crops >> 3 mediocre crops
- 🔬 **Lesson learned**: Training-inference distribution match is critical
- ✅ **EXP030-000 remains BEST**: 2-crop + dropout 0.1 + batch 12

---

## Comparative Analysis

### Model Comparison Table

| Experiment | Model | Key Innovation | Epochs | LR | Dropout | WD | OOF CV | LB | LB-CV Gap | Status |
|------------|-------|----------------|--------|----|---------|----|--------|----|-----------| ------ |
| EXP007-000 | SigLip | Baseline (3-head) | 20 | 0.005 | 0.1 | 0.0 | 0.70 | **0.61** | -0.09 | Baseline |
| EXP008-000 | SigLip | Enhanced Aug | 30 | 0.005 | 0.1 | 0.0 | 0.73 | **0.63** | -0.10 | ⚠️ Overfit |
| EXP008-001 | SigLip | High LR+WD | 30 | 0.01 | 0.2 | 0.01 | 0.64 | **0.55** | -0.09 | ❌ Failed |
| EXP008-002 | SigLip | Lower LR+Longer | 40 | 0.003 | 0.15 | 0.001 | 0.72 | **0.64** | **-0.08** | ✅ Best Gap |
| EXP008-003 | SigLip | Max Reg+MixUp | 40 | 0.003 | 0.25 | 0.005 | 0.65 | **0.62** | **-0.03** | ⚠️ Over-reg |
| EXP009-000 | DINOv3 | Different Backbone | 30 | 0.005 | 0.1 | 0.0 | 0.72 | **0.62** | -0.10 | Similar |
| EXP010-000 | SigLip | StratifiedGKF | 40 | 0.003 | 0.15 | 0.001 | 0.72 | **0.63** | -0.09 | Better CV |
| EXP012-000 | SigLip v1 | Gradual Unfreeze 50% | 40 | 0.003→1e-5 | 0.15 | 0.001 | 0.76 | **0.64** | -0.12 | Good |
| EXP012-001 | SigLip v2 | Gradual Unfreeze + SigLipv2 | 40 | 0.003→1e-5 | 0.15 | 0.001 | 0.761 | **0.65** | -0.111 | Good |
| EXP012-002 | SigLip v2-512 | patch16-512 (larger) | 40 | 0.003→1e-5 | 0.15 | 0.001 | 0.752 | **0.63** | -0.122 | ❌ Worse |
| EXP013-000 | SigLip | 8-Tile Split | - | - | - | - | - | - | - | ❌ Abandoned |
| **EXP014-000** | **SigLip** | **5-Head + Consistency Loss** | **40** | **0.003→1e-5** | **0.15** | **0.001** | **0.772** | **0.66** | **-0.112** | ⭐ **BEST** |
| EXP016-000 | SigLip | Focal Loss (gamma=2.0) | 40 | 0.003→1e-5 | 0.15 | 0.001 | 0.781 | **0.64** | -0.141 | ❌ Overfit |
| EXP017-000 | DINOv3 | Gradual Unfreeze + DINOv3 | 40 | 0.003→1e-5 | 0.1 | 0.001 | 0.745 | **0.62** | -0.125 | ❌ Worse |
| EXP019-000 | SigLip v2 | 5-Head + Consistency (v2) | 40 | 0.003→1e-5 | 0.15 | 0.001 | 0.775 | **0.63** | **-0.145** | ❌ **Overfit** |
| EXP021-000 | SigLip | 5-Head + Aux Metadata | 40 | 0.003→1e-5 | 0.15 | 0.001 | 0.767 | **0.66** | -0.107 | ⚠️ No benefit |
| EXP022-000 | SigLip | 5-Head + Aux Meta + Species | 40 | 0.003→1e-5 | 0.15 | 0.001 | 0.752 | **0.62** | -0.132 | ❌ Degraded |
| EXP020-001 | SigLip | 5-Head + Fixed Backbone LR | 40 | 0.003+1e-5 | 0.15 | 0.001 | 0.751 | **0.65** | -0.101 | ❌ Worse |
| EXP024-000 | SigLip | 5-Head + Feature Concat + TTA | 40 | 0.003→1e-5 | 0.15 | 0.001 | 0.741 | **0.66** | -0.094 | ⭐ **#2 Public** |
| **EXP014-004** | **SigLip** | **5-Head + 70% Unfreeze** | **40** | **0.003→1e-5** | **0.15** | **0.001** | **0.761** | **TBD** | **TBD** | ⏳ **Pending** |
| **EXP014-005** | **SigLip** | **5-Head + 70% Unfreeze** | **40** | **0.003→1e-5** | **0.15** | **0.001** | **0.759** | **0.66** | **-0.099** | ⭐ **#1 Public** |
| EXP021-001 | SigLip | 5-Head + Aux (0.15) | 40 | 0.003→1e-5 | 0.15 | 0.001 | 0.749 | **0.65** | -0.099 | ❌ Worse |
| EXP021-002 | SigLip | 5-Head + Aux (0.05) | 40 | 0.003→1e-5 | 0.15 | 0.001 | 0.755 | **0.63** | -0.125 | ❌ Worse |
| EXP026-000 | SigLip | 5-Head + ExG/HSV Features | 40 | 0.003→1e-5 | 0.3 | 0.001 | 0.756 | **0.62** | **-0.136** | ❌ **WORST** |
| EXP027-001 | SigLip | 5-Head + EMA + 70% Unfreeze | 40 | 0.003→1e-5 | 0.15 | 0.001 | **-0.170** ❌❌❌ | **-0.04** ❌❌❌ | N/A | ❌ **CATASTROPHIC** |
| EXP028-000 | SigLip | 5-Head + Consistency + Cosine Annealing | 40 | 0.003 (cosine) | 0.15 | 0.001 | 0.761 | **0.65** | -0.111 | ❌ Worse |
| EXP028-001 | SigLip | 5-Head + Consistency + Smoother Cosine (T_0=15) | 40 | 0.003 (cosine) | 0.15 | 0.001 | 0.761 | **0.65** | -0.111 | ❌ No change |
| **EXP030-000** | **EVA-CLIP-L** | **3-Head + Gradual Unfreeze (50%)** | **40** | **0.003→1e-5** | **0.1** | **0.001** | **0.762** | **0.68** ⭐⭐⭐ | **-0.082** ⭐ | ⭐ **NEW BEST** |
| EXP030-001 | EVA-CLIP-L | 3-Head + Gradual Unfreeze (70%) | 40 | 0.003→1e-5 | 0.1 | 0.001 | 0.762 | **0.67** | -0.092 | ❌ Worse |
| EXP030-003 | EVA-CLIP-L | 3-Head + Dropout 0.15 + Batch 16 | 40 | 0.003→1e-5 | **0.15** | 0.001 | **0.777** | **0.68** | -0.097 | ⚠️ CV↑ Gap↓ |
| EXP031-000 | EVA-CLIP-L | 3-Crop Training + Dropout 0.15 | 40 | 0.003→1e-5 | 0.15 | 0.001 | 0.766 | **0.66** ❌❌ | **-0.106** ❌❌ | ❌❌ **WORST** |
| ENSEXP_003 | EXP060 + depth | Linear error correction (DepthAnything3) | - | - | - | - | 0.809 | **0.65** | -0.159 | ❌ LB drop |
| ENSEXP_004 | EXP060 + depth | Percentile-based threshold correction | - | - | - | - | 0.783 | **0.66** | -0.123 | ❌ LB drop |
| **EXP060-000** | **EVA-CLIP-L** | **Correct Global Weighted R² metric** | **40** | **0.003** | **0.1** | **0.001** | **0.759** | **0.69** ⭐ | **-0.069** ⭐ | ⭐ **BEST GAP** |
| EXP060-025 | EVA-CLIP-L | Balanced date_cluster folds + stronger reg | 35 | 0.002 | 0.2 | 0.01 | 0.700 | **0.65** | -0.050 | ⚠️ Gap↓ / LB↓ |
| EXP060-027 | EVA-CLIP-L | Balanced date_cluster folds + stronger finetune | 40 | 0.003 | 0.2 | 0.001 | 0.748 | **0.68** | -0.068 | ⚠️ Gapは維持 / LBは未達 |
| EXP060-030 | EVA-CLIP-L | Sampling_Date balanced split + (early stop / clip / cap unfreeze) | 40 | 0.003 | 0.1 | 0.001 | 0.704 | **0.63** ❌ | -0.074 | ❌ LB drop |
| EXP060-034 | EVA-CLIP-L | 3-crop training (buggy: per-crop aug) | 40 | 0.003 | 0.1 | 0.001 | 0.777 | **0.65** ❌ | -0.127 ❌ | ❌ Severe overfit (aug bug) |
| EXP060-035 | EVA-CLIP-L | 3-crop training (fixed: full-image aug) | 27 | 0.003 | 0.1 | 0.001 | TBD | TBD | TBD | 🔄 Pending |
| EXP060-036 | EVA-CLIP-L | 5-fold baseline reproduction | 40 | 0.003 | 0.1 | 0.001 | 0.769 | **0.66** | -0.109 | ⚠️ Overfit |
| EXP060-037 | EVA-CLIP-L | 10-fold CV (more folds) | 40 | 0.003 | 0.1 | 0.001 | 0.785 | **0.67** | -0.115 | ⚠️ Overfit |
| **EXP060-038** | **DinoV3-Huge+** | **DinoV3 backbone via timm** | **40** | **0.003** | **0.1** | **0.001** | **0.776** | **0.71** ⭐⭐⭐ | **-0.066** ⭐ | ⭐⭐⭐ **NEW BEST** |
| EXP101-000 | DinoV3+Mamba | Two-stream + Mamba Fusion | 20 | 1e-4 | 0.1 | 0.01 | 0.749 | **0.62** | -0.129 | ❌ Mamba効果なし |

### Key Findings

#### 1. **Best Public LB Performance: EXP030-child000** ⭐⭐⭐ **BREAKTHROUGH**
- **LB 0.68** (highest score achieved, **+0.02 improvement over previous best**)
- **Best model**: EXP030-child000 (EVA-CLIP-L + 3-Head + Gradual Unfreezing)
  - CV 0.762, Gap **-0.082** ⭐ **BEST GAP EVER**
  - **dropout 0.1, batch_size 12, 2-crop training**
  - 50% backbone unfreezing with gradual layer-wise strategy
  - EVA02-CLIP-L-14-336 (428M params, 80.4% ImageNet zero-shot)
  - 3-head architecture (predicts Green, GDM, Total; calculates Dead, Clover)
- **Failed variations**:
  - EXP030-001 (70% unfreezing): LB 0.67 (-0.01), Gap -0.092 ❌
  - EXP030-003 (dropout 0.15): LB 0.68 (same), Gap -0.097 ⚠️ CV improved but Gap worsened
  - **EXP031-000 (3-crop training)**: LB **0.66** (-0.02) ❌❌ Gap **-0.106** (worst ever)
- **Previous best**: EXP014-child005 (SigLip v1 + 5-Head + 70% Unfreezing)
  - CV 0.759, Gap -0.099
  - 70% backbone unfreezing (vs 50% in child-000)
  - Lower CV but better Public LB rank than child-000
- **Second best**: EXP024-child000 (Feature Concatenation + TTA)
  - LB 0.66, **#2 Public LB rank**
  - CV 0.741, Gap -0.094 ⭐ **2nd BEST GAP**
  - Feature concatenation architecture
- **Third**: EXP014-child000 (Baseline)
  - LB 0.66, #6 Public LB rank
  - CV 0.772, Gap -0.112
- **Key comparison**:
  - **EXP030-000 is the absolute best**: LB 0.68, Gap -0.082
  - **Critical hyperparameters**: dropout 0.1, batch 12, 2-crop (not 3-crop)
  - **SigLip patterns don't transfer**: dropout 0.15 works for SigLip, fails for EVA-CLIP
  - **Data augmentation via cropping failed**: 3-crop severely degraded performance
  - **Gap is the key metric**: -0.082 >> -0.097 >> -0.106
- **Recommendation**: **Use EXP030-child000** for final submission (LB 0.68, Gap -0.082)

#### 2. **Best Generalization: EXP030-child000** ⭐⭐⭐ **NEW BEST GAP**
- **LB 0.68** ⭐ **BEST LB EVER**
- **Best LB-CV Gap**: **-0.082** ⭐⭐⭐ (best across all experiments)
  - vs EXP024-000: -0.094 (+0.012 improvement)
  - vs EXP014-005: -0.099 (+0.017 improvement)
  - vs EXP014-000: -0.112 (+0.030 improvement)
  - vs EXP008-002: -0.08 (+0.002 improvement)
- OOF CV 0.762 (competitive with best models)
- Configuration: EVA-CLIP-L, 3-head architecture, 50% gradual unfreezing
- **Key insight**: Superior vision encoder (EVA-CLIP) provides both best LB AND best gap
- **Second best gap**: EXP024-child000 with -0.094 (SigLip + feature concat)
- **Third best gap**: EXP014-child005 with -0.099 (SigLip + 70% unfreeze)
- **Breakthrough**: First model to achieve both highest LB (0.68) and best gap (-0.082) simultaneously

#### 3. **Regularization Sweet Spot: Backbone-Specific** ⭐ **UPDATED**
- **For SigLip** (78% ImageNet zero-shot):
  - **Optimal**: dropout=0.15, WD=0.001 → LB 0.66, Gap -0.099~-0.112
  - Too little (dropout=0.1): Poor generalization (Gap = -0.10)
  - Too much (dropout=0.25 + MixUp): Underfitting (LB 0.62, Gap = -0.03)
- **For EVA-CLIP** (80.4% ImageNet zero-shot) ⭐ **NEW**:
  - **Optimal**: dropout=**0.1**, WD=0.001 → LB **0.68**, Gap **-0.082** ⭐
  - Too much (dropout=0.15): Better CV but worse gap (LB 0.68, Gap -0.097)
  - **Key insight**: Stronger pre-training needs **less** regularization

#### 4. **Architecture & Loss Function Comparison** ⭐⭐⭐ **FINAL VERDICT**
- **3-Head + EVA-CLIP (EXP030-000)**: CV 0.762, LB **0.68** ⭐⭐⭐, Gap **-0.082** ⭐
  - **BEST MODEL**: Superior vision encoder + simple architecture + optimal hyperparameters
  - Predicts: Dry_Green_g, GDM_g, Dry_Total_g
  - **Critical config**: dropout 0.1, batch 12, 2-crop training
  - Calculates: Dry_Dead_g, Dry_Clover_g (derived)
  - EVA02-CLIP-L (80.4% ImageNet zero-shot)
  - **+0.02 LB** over best SigLip model
- **5-Head + Consistency Loss + SigLip (EXP014)**: CV 0.772, LB 0.66, Gap -0.112
  - **Best SigLip model**
  - Predicts: All 5 targets directly
  - Consistency loss enforces physical constraints
  - **+0.02 LB improvement** over 3-head SigLip (EXP012)
- **3-Head + SigLip (EXP012)**: CV 0.76, LB 0.64, Gap -0.12
  - Baseline 3-head architecture with SigLip
- **3-Head + Focal Loss + SigLip (EXP016)**: CV 0.781 ❌, LB 0.64, Gap -0.141
  - Same architecture as EXP012
  - Focal loss (gamma=2.0) for hard samples
  - **Best CV but overfits**: Hard samples in training ≠ test
- **Winner**: EXP030 (3-head + EVA-CLIP) ⭐⭐⭐
- **Key insights**:
  1. **Vision encoder quality matters most**: EVA-CLIP (0.68) > SigLip (0.66) > DINOv3 (0.62)
  2. With powerful backbone, simpler architecture (3-head) works best
  3. With weaker backbone, multi-task learning (5-head + consistency) helps
- **Failed approach**: Focal loss improves CV but hurts generalization (distribution shift)

#### 5. **Training Strategy Insights** ⭐⭐ **MAJOR UPDATES from EXP030/031**
- **Lower LR + Longer Training** (0.003, 40ep) > Higher LR (0.005, 30ep)
- Aggressive regularization (LR=0.01, WD=0.01) hurts convergence
- MixUp alone doesn't improve performance significantly when added to already strong augmentation
- **Gradual unfreezing** (50% backbone) is optimal for both SigLip and EVA-CLIP
  - Frozen baseline: LB 0.63 (EXP010)
  - 50% unfreeze: LB 0.64 (EXP012), **0.68 (EXP030)** ⭐
  - 70% unfreeze: LB 0.67 (EXP030-001) ❌ Worse for EVA-CLIP
- **Data augmentation strategy** ⭐ **NEW**:
  - **2-crop (left/right halves)**: LB **0.68**, Gap **-0.082** ⭐ **OPTIMAL**
  - **3-crop (stride=500)**: LB **0.66** ❌❌ Gap **-0.106** ❌ **CATASTROPHIC**
  - **Key lesson**: Training-inference distribution match > data quantity
  - **More crops ≠ better**: 3-crop severely degraded generalization
- **Dropout tuning is backbone-specific** ⭐ **NEW**:
  - SigLip (78% zero-shot): dropout **0.15** optimal
  - EVA-CLIP (80.4% zero-shot): dropout **0.1** optimal
  - **Stronger pre-training needs less regularization**
- **Architecture improvements trump training tricks**:
  - 5-head + consistency loss (EXP014): LB 0.66 ⭐ **Best SigLip**
  - 3-head + EVA-CLIP (EXP030): LB **0.68** ⭐⭐⭐ **Best overall**
  - Focal loss (EXP016): LB 0.64 ❌ **Overfits despite best CV**
- **Key lessons**:
  - Vision encoder quality > architecture complexity
  - Multi-task learning with domain constraints > complex loss functions
  - Data augmentation via cropping can backfire (EXP031)

#### 6. **Model Architecture Comparison** ⭐⭐⭐ **FINAL RANKING**
- **3-head + EVA-CLIP best overall**: LB **0.68** (EXP030-000) ⭐⭐⭐ **CHAMPION**
- **5-head + SigLip**: LB 0.66 (EXP014-000) ⭐ **Best SigLip model**
- **SigLip v1 vs v2 (with gradual unfreezing)**:
  - v1 (EXP012-000): LB 0.64, CV 0.76
  - v2 (EXP012-001): LB 0.65, CV 0.761 (+0.01 LB)
  - **Conclusion**: Modest improvement with v2
- **SigLip vs DINOv3**: SigLip significantly better
  - SigLip v2 + Gradual Unfreeze (EXP012-001): LB 0.65
  - DINOv3 + Gradual Unfreeze (EXP017-000): LB 0.62 (-0.03)
  - **Key insight**: Vision-language model (SigLip) > Self-supervised (DINOv3) for this task
- **Input size comparison (SigLip v2)**:
  - patch14-384 (EXP012-001): LB 0.65, CV 0.761 ⭐ **Best**
  - patch16-512 (EXP012-002): LB 0.63, CV 0.752 ❌ **Worse** (-0.02 LB)
  - **Conclusion**: 384 input size optimal for 2000×1000 images
- **Architecture comparison (same base strategy)**:
  - 3-head + Gradual Unfreeze (EXP012-000): LB 0.64, CV 0.76
  - 5-head + Consistency Loss (EXP014-000): LB 0.66, CV 0.772 ⭐ **Best** (+0.02 LB)
  - **Conclusion**: Multi-task learning > gradual unfreezing
- All models now support `pretrained=False` for offline inference

#### 7. **Fold Consistency Issues** ⭐ **UPDATED**
- Fold 0 often underperforms (EXP010: R² 0.59 → EXP012: 0.66 → EXP014: 0.69 → EXP016: 0.73)
- **Progressive improvement**: Full unfreezing helps weaker folds
- Fold 3/4 show variable performance (R² 0.54-0.67 in some experiments)
- Fold 1/2 show high performance (R² 0.69-0.79)
- Suggests potential data imbalance or distribution shift in GroupKFold splits
- EXP008-003 fold 4 particularly problematic (R² 0.54)
- **EXP016 best fold consistency**: std = 0.048 (lowest variance achieved)
- More aggressive unfreezing reduces fold variance but can overfit to training distribution

#### 8. **Cross-Validation Strategy Comparison**
- **StratifiedGroupKFold (EXP010)** vs **GroupKFold (EXP008-002)**:
  - LB: 0.63 vs 0.64 (StratifiedGKF scored -0.01 lower)
  - CV: 0.72 vs 0.72 (same)
  - LB-CV Gap: -0.09 vs -0.08 (StratifiedGKF slightly worse)
- **State stratification** ensures balanced distribution but doesn't improve LB
- **Fold size imbalance** (52-101 images/fold) is acceptable for stratification
- **Conclusion**: GroupKFold (group by Sampling_Date+State) was already sufficient
- **Hypothesis**: Test set may have different State/date distribution than train set

#### 9. **CV-LB Relationship and Distribution Shift** ⭐ **CRITICAL LESSON**
- **Critical pattern observed**: Higher CV doesn't guarantee better LB
  - **EXP019 (SigLip v2 + 5-head)**: CV 0.775 but LB 0.63 ❌ (worst gap: -0.145)
  - **EXP016 (Focal Loss)**: CV 0.781 (highest) but LB 0.64 ❌
  - **EXP014 (5-head)**: CV 0.772 but LB 0.66 ⭐ (best)
  - **EXP012 (3-head)**: CV 0.76 but LB 0.64
- **Distribution shift confirmed**: Public/private test sets have different distributions
  - Train: 375 images from various seasons/states
  - Test: 800+ images (53% public, 47% private)
  - Public: Some overlap with training periods
  - Private: Non-overlapping time periods for generalization testing
- **EXP019 overfitting analysis** (critical lesson):
  - Combined best architecture (5-head) + best model (SigLip v2)
  - Expected LB 0.67-0.68, but got 0.63 (-0.03 from EXP014)
  - **Worst gap ever**: -0.145 (vs -0.112 for EXP014, -0.111 for EXP012-001)
  - **Key insight**: Model capacity must be balanced - too much overfits
  - SigLip v2 works better with 3-head (EXP012-001: LB 0.65)
  - 5-head works better with SigLip v1 (EXP014: LB 0.66)
  - **Stacking improvements ≠ better results**
- **EXP016 focal loss analysis** (key lesson):
  - Focal loss emphasizes hard training samples
  - Improves CV by +0.021 (0.76 → 0.781) and fold consistency
  - **But hard samples in training ≠ hard samples in test**
  - Distribution shift means focal loss overfits to wrong samples
  - LB drops to 0.64 despite best CV ever
- **Strategic insight**: Architecture changes > loss function tricks
  - **EXP014 (consistency loss)**: Regularizes via domain knowledge → better generalization
  - **EXP016 (focal loss)**: Regularizes via sample difficulty → overfits to training
  - **EXP019 (SigLip v2 + 5-head)**: Too much capacity → severe overfitting
- **Optimal approach identified**: EXP014's multi-task learning
  - Not highest CV (0.772 vs 0.781 EXP016, 0.775 EXP019)
  - Not smallest gap (-0.112 vs -0.08)
  - **But highest LB (0.66)** - uses domain knowledge effectively
  - **Balanced model capacity** - SigLip v1 + 5-head is sweet spot

---

## Lessons Learned

### Regularization Strategy
1. **Incremental approach wins**: Small increases in regularization are better than large jumps
2. **Combination matters**: dropout=0.15 + WD=0.001 > dropout=0.25 + WD=0.005 + MixUp
3. **Diminishing returns**: Adding more regularization doesn't always help
4. **Monitor both metrics**: Optimize for LB, not just CV or LB-CV Gap

### Training Strategy
1. **Patient training**: Lower LR (0.003) with more epochs (40) beats higher LR (0.005) with fewer epochs
2. **Smooth convergence**: Gradual loss decrease correlates with better LB performance
3. **Early stopping**: Can prevent overfitting but may stop before optimal point

### Data Strategy
1. **CV Strategy**: StratifiedGroupKFold didn't improve over GroupKFold (Sampling_Date+State)
2. **Augmentation plateau**: Enhanced augmentation helps, but more isn't always better
3. **Test set mismatch**: LB-CV gap varies (-0.08 to -0.12) - frozen models generalize better
4. **Fold imbalance**: Uneven fold sizes are acceptable when grouping/stratifying constraints exist

### Architecture & Multi-Task Learning ⭐ **UPDATED**
1. **5-head multi-task learning achieves best LB**: EXP014 scores 0.66 (new best)
2. **Architecture progression validated**:
   - 3-head baseline (EXP012): LB 0.64, CV 0.76
   - **5-head + consistency loss (EXP014): LB 0.66, CV 0.772** ⭐ **Optimal**
   - 3-head + focal loss (EXP016): LB 0.64, CV 0.781 (overfits)
   - 5-head + auxiliary metadata (EXP021): LB 0.66, CV 0.767 (no improvement)
   - 5-head + auxiliary metadata + species (EXP022): LB 0.62, CV 0.752 ❌ (degraded)
   - 5-head + ExG/HSV features (EXP026): LB 0.62, CV 0.756 ❌ (degraded, worst gap -0.136)
3. **Consistency loss as regularization**: Physical constraints improve generalization
   - Constraint 1: Dry_Dead_g + GDM_g ≈ Dry_Total_g
   - Constraint 2: Dry_Clover_g + Dry_Green_g ≈ GDM_g
   - Weight = 0.1 (effective regularization)
4. **Domain knowledge > data-driven tricks**:
   - Consistency loss (domain constraints): +0.02 LB ⭐
   - Focal loss (sample difficulty): +0.021 CV but -0.02 LB
   - Auxiliary metadata prediction (Height, NDVI): 0.00 LB (no effect)
   - Auxiliary species classification: -0.04 LB ❌ (harmful)
   - ExG/HSV handcrafted features: -0.04 LB ❌ (harmful, worst gap -0.136)
5. **Auxiliary task learning lessons** (EXP021, EXP022, EXP026):
   - ❌ **Not all auxiliary tasks help**: Metadata prediction provides no benefit
   - ❌ **Species classification hurts**: LB drops from 0.66 → 0.62 (-0.04)
   - ❌ **Handcrafted features hurt**: ExG/HSV features degrade LB to 0.62 (-0.04) with worst gap (-0.136)
   - ⚠️ **Distribution shift critical**: Train metadata patterns don't match test set
   - ⚠️ **Label quality matters**: Clover contradictions suggest unreliable metadata
   - ⚠️ **Simpler is better**: EXP014 baseline outperforms complex multi-task variants
   - **Key insight**: Train-only features don't guarantee better generalization
   - **Deep learning > feature engineering**: SigLip embeddings already capture pixel-level patterns
6. **EMA (Exponential Moving Average) lessons** (EXP027): ❌❌❌ **CATASTROPHIC FAILURE**
   - ❌ **EMA implementation extremely risky**: Shadow weight management complex and error-prone
   - ❌ **Training collapse**: Validation R² dropped from 0.58 → 0.39 in 4 epochs
   - ❌ **Negative predictions**: Model outputs negative biomass without output constraints (ReLU/Softplus)
   - ❌ **OOF aggregation bug**: Individual fold R² (0.65-0.84) vs OOF R² (-0.17) indicates prediction save error
   - ❌ **Don't stack risky changes**: 70% unfreezing + EMA + new code = disaster
   - ⚠️ **EMA decay tuning critical**: 0.999 too high for 40-epoch training (lag ~1000 steps)
   - ⚠️ **Test on stable baseline first**: Should have tried EMA on 50% unfreezing before 70%
   - **Recommendation**: Abandon EMA, stick with standard training (EXP014-child005)
7. **Critical lesson**: Distribution shift means focal loss overfits to training hard samples
8. **Best generalization gap**: EXP008-002 (frozen, Gap = -0.08) but lower LB (0.64)
9. **Trade-off validated**: Better gap doesn't mean better absolute LB
10. **Recommendation**: Use EXP014 for final submission (highest LB: 0.66)
   - Avoid auxiliary tasks when base model already performs well
   - Domain constraints (consistency loss) > auxiliary prediction tasks
   - Pretrained vision models > handcrafted feature engineering
   - Never use EMA without thorough testing and validation

---

## Next Steps & Recommendations

### Immediate Actions (High Priority)

1. **Ensemble Strategy**
   - Ensemble EXP008-000 + EXP008-002 (different epochs/LR)
   - Add EXP009-000 (different architecture)
   - Expected improvement: +0.01-0.02 LB

2. **~~Cross-Validation Improvement~~** ✅ Completed
   - ✅ Tested StratifiedGroupKFold (EXP010) - no LB improvement
   - GroupKFold (Sampling_Date+State) is sufficient
   - Fold variance likely due to natural data distribution

3. **Test Time Augmentation (TTA)**
   - Current: 3 views (original, h-flip, v-flip)
   - Target: 5-7 views (add rotation, brightness variations)
   - Expected improvement: +0.005-0.01 LB

### Mid-term Actions (Medium Priority)

4. **Model Architecture Experiments**
   - Try intermediate backbone sizes (smaller than base)
   - Experiment with multi-head architecture variations
   - Fine-tune head hidden dimensions

5. **Advanced Training Techniques**
   - Pseudo-labeling with high-confidence test predictions
   - Knowledge distillation from larger models
   - Multi-stage training strategies

### Long-term Actions (Low Priority)

6. **Feature Engineering**
   - Incorporate metadata (State, Species, Height, NDVI)
   - Multi-modal learning approaches

7. **Advanced Augmentation**
   - Grid distortion (preserve grass spatial relationships)
   - Domain-specific augmentations

---

## Technical Notes

### Image Processing Pipeline
- Input: 2000×1000 pixel RGB images
- Split: Left (0-1000) and Right (1000-2000) halves
- Resize: Each half → 768×768 (SigLip) or 224×224 (DINOv3)
- Processing: Shared frozen backbone for both halves
- Output: Sum predictions from both halves

### Target Prediction Strategy
- **Direct prediction**: 3 targets (Dry_Green_g, GDM_g, Dry_Total_g)
- **Derived calculation**:
  - Dry_Clover_g = max(0, GDM_g - Dry_Green_g)
  - Dry_Dead_g = max(0, Dry_Total_g - GDM_g)

### Training Infrastructure
- **Training**: Google Colab (GPU)
- **Inference**: Kaggle Notebooks (offline, ≤9h runtime)
- **Model Checkpoints**: Saved based on best validation R² (not loss)

### Kaggle Offline Inference
- **Both EXP008 & EXP009**: Now use `pretrained=False` approach
- **SigLip**: Creates model from config, loads weights from checkpoint
- **DINOv3**: Creates model from config, loads weights from checkpoint
- **Config files**: Can be loaded from HF cache or use fallback defaults

---

## Competition Status

🎉🎉 **Current Best Score**: LB **0.73** ⭐⭐⭐ **NEW RECORD** - EXP060-child058 (DinoV3 ViT-Huge+ + Artem CV) 🎉🎉
**Best Generalization Gap**: **-0.062** - EXP060-child058 ⭐ **BEST EVER**
**Best CV Score**: 0.800 (EXP060-child059, but overfits)
**Best OOF Score**: 0.792 (EXP060-child058)
**Target Score**: LB **0.75** (current: 0.73, remaining: **+0.02**)

**Progress Summary:**
- EXP007 → **EXP060-child058**: **+0.12 LB improvement** (0.61 → 0.73) 🎉🎉🎉
- 🎉🎉 **Best model**: EXP060-child058 ⭐⭐⭐ **CURRENT BEST** (2026-01-10)
  - **LB: 0.73** (+0.02 from EXP060-child038)
  - **OOF CV: 0.792**, **Gap: -0.062** ⭐
  - **DinoV3 ViT-Huge+** + **Artem CV strategy**
  - CV: stratify by Clover/Dead presence, group by day/month_State
  - **Key insight**: CV strategyの変更だけでLB +0.02！
  - Fold scores: [0.673, 0.810, 0.866, 0.835, 0.805]
- **前ベスト**: EXP060-child038 (DinoV3 ViT-Huge+ baseline)
  - LB: 0.71, CV: 0.776, Gap: -0.066
- **EXP060-child059** (768px + Artem CV): CV 0.800, LB 0.70 ❌
  - 768pxはWA性能を大幅低下させ過学習（Gap -0.100）
- **EXP060-child060** (EVA02-CLIP + Artem CV): LB 0.67 ❌
  - EVA02-CLIPはCV strategyに敏感、Artem CVでWA悪化
- **Previous best (SigLip)**: EXP014-child005
  - LB: 0.66, CV: 0.759, Gap: -0.099
  - 5-head + 70% unfreezing

**Vision Encoder Comparison:**
- 🎉 **DinoV3 ViT-Huge+** (EXP060-038): **LB 0.71** ⭐⭐⭐ **NEW BEST** (1.3B params, self-supervised)
- **EVA-CLIP-L** (EXP060-000): **LB 0.69** (428M params, 80.4% zero-shot)
- **SigLip-SO400M** (EXP014): LB 0.66 (~78% zero-shot)
- **DINOv3-L** (EXP017): LB 0.62 (~77% zero-shot, self-supervised)
- **Conclusion**: DinoV3 Huge+ > EVA-CLIP-L > SigLip > DINOv3-L。モデルサイズと自己教師あり学習の質が重要
  - EXP020-child001 (Fixed LR): LB 0.65, CV 0.751, Gap -0.101
- **Key Model Comparisons**:
  - **70% unfreezing > 50%**: EXP014-005 (#1) vs EXP014-000 (#6), same LB but huge rank difference
  - **Feature concat > prediction sum**: EXP024-000 (#2) with best gap -0.094
  - **TTA improves rank**: EXP014-000+TTA (#4) vs without TTA (#6)
  - **5-head vs 3-head**: 5-head better (+0.02 LB, EXP014 vs EXP012-000)
  - **Auxiliary tasks**: No benefit (EXP021 auxiliary_weight ablation showed 0.1 is optimal but still worse than baseline)
  - **Public LB rank matters**: Same displayed LB can have vastly different ranks
- **Failed experiments**:
  - **EXP027-001**: EMA + 70% unfreezing ❌❌❌ **CATASTROPHIC FAILURE** (OOF R² -0.17, LB -0.04)
    - Training collapse: Val R² dropped from 0.58 → 0.39 in 4 epochs
    - Negative predictions in OOF (e.g., Dry_Clover_g = -0.095)
    - EMA shadow weight bug likely cause
    - **Never use EMA without thorough validation**
  - EXP026-000: ExG/HSV handcrafted features (LB 0.62, Gap -0.136 ❌ **WORST GAP**)
  - EXP022-000: Species auxiliary classification (LB 0.62, -0.04)
  - EXP021-002: Weaker auxiliary weight 0.05 (LB 0.63, -0.03)
  - EXP016-000: Focal loss improved CV but overfits (Gap -0.141, LB 0.64)
  - EXP019-000: SigLip v2 + 5-head overfit severely (Gap -0.145, LB 0.63)
  - EXP020-001: Fixed backbone LR didn't help (LB 0.65, -0.01)
- **Key insights**:
  - ⭐ **Public LB rank is critical**: Same LB 0.66 but ranks vary from #1 to #6
  - ⭐ **70% unfreezing wins**: Best Public rank despite lower CV than 50%
  - ⭐ **Feature concatenation**: 2nd best Public rank + best gap (-0.094)
  - ⭐ **TTA helps decimal precision**: Improves Public rank without changing displayed LB
  - ⭐ **Deep learning > feature engineering**: Handcrafted ExG/HSV features severely degrade performance (worst gap -0.136)
  - **Multi-task learning with domain constraints** (EXP014) = Best base architecture
  - **Gap predicts Public rank**: Better gap → better Public rank (EXP024: -0.094, EXP014-005: -0.099)
  - **Lower CV ≠ worse LB**: EXP024 (CV 0.741) ranks #2, EXP014-000 (CV 0.772) ranks #6
  - **Auxiliary tasks don't help**: Train-only features (metadata, species) provide no benefit
  - **Pretrained models sufficient**: SigLip embeddings already capture pixel-level patterns (ExG/HSV redundant)
  - Distribution shift confirmed (train vs public vs private test)
- **Next focus**:
  - Ensemble EXP014-child005 + EXP024-child000 + EXP014-child000-TTA
  - Submit EXP014-child004 to verify 70% unfreezing consistency
  - Explore ensemble strategies for final submission
  - Focus on models with best gaps (EXP024, EXP014-005) rather than highest CV

---

### EXP051: Patch Token GAP Pooling ❌ **SEVERE OVERFITTING**
**Model**: EVA-CLIP-L with Patch Token GAP Pooling (instead of [CLS] token)

**Motivation**: Use all 576 patch tokens instead of single [CLS] token for more granular feature extraction.

#### EXP051-child000: Patch Token GAP Pooling
**Configuration:**
- Architecture: EVA-CLIP-L + Patch Token GAP + 5-Head
- Feature extraction: Mean pooling of 576 patch tokens (24x24 grid)
- No [CLS] token used
- All other settings same as EXP036

**Results:**
- OOF CV: **0.793** ⭐ (Highest CV achieved!)
- LB: **0.64** ❌ (Worse than EXP036's 0.69)
- **LB-CV Gap: -0.153** ❌❌❌ (Severe overfitting)
- Mean R²: 0.788 ± 0.051
- Fold R²: [0.733, 0.740, 0.874, 0.793, 0.799]

**Key Observations:**
- ⚠️ **Classic overfitting pattern**: CV improved (+0.02) but LB dropped (-0.05)
- **Root cause**: 576 patch tokens with only 375 training images
  - Model learns "location patterns" instead of "grass amount"
  - Background patterns (soil, shadow, fence) in specific positions get memorized
  - Test set has different background distributions → poor generalization
- **Gap -0.153 is worst among all experiments**
- **Conclusion**: [CLS] token is better for this task - provides abstraction that prevents spatial memorization

#### EXP051-child001: Patch Token GeM Pooling
**Configuration:**
- Architecture: EVA-CLIP-L + Patch Token GeM Pooling + 5-Head
- GeM (Generalized Mean): Learnable p parameter, default p=3
- Hypothesis: GeM (p>1) emphasizes dense grass regions

**Results:**
- OOF CV: **0.781**
- LB: **0.66** (Better than GAP but still worse than EXP036)
- **LB-CV Gap: -0.121**
- Mean R²: 0.777 ± 0.045
- Fold R²: [0.714, 0.749, 0.850, 0.788, 0.786]

**Key Observations:**
- GeM provides slight regularization vs GAP
- Still worse than [CLS] token approach (EXP036)
- Patch token approach fundamentally limited for this dataset size

---

### EXP052: Anti-Overfitting Patch Token Pooling ❌ **DID NOT IMPROVE LB**
**Model**: EVA-CLIP-L with various anti-overfitting techniques for patch token pooling

**Motivation**: Address EXP051's severe overfitting (Gap -0.153) with regularization techniques.

#### EXP052-child000: Patch Dropout 0.15 + No LayerNorm
**Configuration:**
- Architecture: EVA-CLIP-L + Patch Token GAP + PatchDropout(0.15) + 5-Head
- **PatchDropout**: Randomly drop 15% of patches during training
- **No LayerNorm**: Removed to reduce overfitting risk
- Hypothesis: Destroying position information reduces spatial memorization

**Results:**
- OOF CV: **0.768** (Lower than EXP051's 0.793)
- LB: **0.66** (Better than EXP051's 0.64, but still worse than EXP036's 0.69)
- **LB-CV Gap: -0.108** (Improved from -0.153)
- Mean R²: 0.773 ± 0.060
- Fold R²: [0.669, 0.764, 0.856, 0.795, 0.778]

**Key Observations:**
- ✅ **Gap improved**: -0.153 → -0.108 (regularization working)
- ✅ **LB improved**: 0.64 → 0.66 (less overfitting)
- ❌ **Still worse than [CLS] approach**: EXP036 LB 0.69 vs EXP052-000 LB 0.66
- PatchDropout helps but doesn't solve fundamental issue

#### EXP052-child002: Lightweight Attention Pooling
**Configuration:**
- Architecture: EVA-CLIP-L + Lightweight Attention Pooling + 5-Head
- **Attention Pooling**: Single Linear layer (1024→1) learns patch weights
  - Only 1025 additional parameters (minimal overfitting risk)
  - Learns to focus on "grass patches" and ignore "background patches"
- No PatchDropout (attention already filters)

**Results:**
- OOF CV: **0.759**
- LB: **0.66** (Same as child000)
- **LB-CV Gap: -0.099**
- Mean R²: 0.758 ± 0.063
- Fold R²: [0.661, 0.719, 0.842, 0.769, 0.801]

**Key Observations:**
- ✅ **Best gap for patch token approach**: -0.099 (comparable to EXP036)
- ❌ **LB still 0.66**: Cannot match [CLS] token's LB 0.69
- Attention weights can potentially visualize which patches model focuses on

**EXP052 Conclusion:**
- **[CLS] token > Patch token pooling** for this dataset size (375 images)
- Anti-overfitting techniques (PatchDropout, Attention) reduce gap but cannot overcome fundamental limitation
- 576 patches × 375 images = too few samples per patch position
- **Recommendation**: Stay with [CLS] token approach (EXP036) for best LB

---

## Key Lessons: [CLS] Token vs Patch Token Pooling

| Approach | OOF CV | LB | Gap | Notes |
|----------|--------|-----|-----|-------|
| **[CLS] Token (EXP036)** | 0.773 | **0.69** | -0.083 | **Best LB** |
| Patch GAP (EXP051-000) | **0.793** | 0.64 | -0.153 | Severe overfitting |
| Patch GeM (EXP051-001) | 0.781 | 0.66 | -0.121 | Slight improvement |
| Patch + Dropout (EXP052-000) | 0.768 | 0.66 | -0.108 | Regularization helps |
| Patch + Attention (EXP052-002) | 0.759 | 0.66 | -0.099 | Best gap for patch |

**Why [CLS] Token is Better:**
1. **Abstraction**: [CLS] token provides semantic summary, not positional details
2. **Efficiency**: 1 token vs 576 tokens = less parameters to overfit
3. **Invariance**: [CLS] learns position-invariant features naturally
4. **Dataset size**: 375 images too few for patch-level learning

**When Patch Tokens Might Help:**
- Larger datasets (>5000 images)
- Tasks requiring localization (object detection, segmentation)
- When combined with strong spatial regularization

---

### EXP055: DINOv2 + Patch-level MLP Prediction 🧪 **NEW APPROACH**
**Model**: DINOv2-large with Patch-level MLP + 5-Head + Consistency Loss

#### EXP055-child000: DINOv2 Patch-level MLP Baseline
**Configuration:**
- **Backbone**: DINOv2-large (304M params, 1024 embed_dim)
- **Novel Architecture**:
  - Extracts dense patch features (256 patches for 768×768 input)
  - Applies shared MLP to each patch independently
  - Averages patch predictions for final output
- **Heads**: 5-head architecture (same as EXP036)
  - head_green, head_dead, head_clover, head_gdm, head_total
  - Each head processes patches with shared MLP
- **Consistency Loss**: 0.1 weight (same as EXP036)
- **Two-stream**: 2000×1000 → left/right 768×768 (same as EXP036)
- **Training**:
  - Epochs: 40
  - Batch size: 8 (smaller due to patch processing)
  - LR: 0.003
  - Dropout: 0.3
  - Gradual unfreezing: Freeze 5 epochs, then unfreeze 50% blocks
- **Augmentation**: default (same as EXP036)

**Inspiration:**
- Based on public notebook "csiro-dinov2-dense-features-lb-0-66.ipynb"
- Original notebook: 224×224 input, simple MLP → LB 0.66
- Our implementation: 768×768 input, 5-head + consistency loss

**Key Innovation:**
- **Patch-level prediction**: Instead of using global [CLS] token, uses ALL 256 patch tokens
- **Weight sharing**: Same MLP applied to each patch (captures local patterns)
- **Spatial averaging**: Final prediction = average of 256 patch predictions
- **Hypothesis**: Local biomass patterns (clover leaves, grass texture) better captured by patch-level processing

**Comparison with EXP051/052 (Failed Patch Token Approach):**
| Feature | EXP051/052 (Failed) | EXP055 (New) |
|---------|---------------------|--------------|
| Patch usage | Attention pooling / Concat | **Patch-level MLP + Average** |
| Prediction | Single prediction from pooled features | **256 predictions averaged** |
| Weight sharing | No (global pooling first) | **Yes (same MLP per patch)** |
| Inspiration | Internal idea | **Public LB 0.66 notebook** |
| Result | LB 0.64-0.66 (severe overfitting) | **TBD** |

**Why This Might Work:**
1. ✅ **Proven approach**: Public notebook achieved LB 0.66 with 224×224
2. ✅ **Higher resolution**: Our 768×768 preserves finer details (clover leaves)
3. ✅ **Better regularization**: 5-head + consistency loss vs simple MLP
4. ✅ **DINOv2 strength**: Self-supervised features excel at fine-grained patterns
5. ✅ **Local prediction**: Each patch votes independently, robust to spatial variation

**Expected Performance:**
- **Target LB**: 0.66-0.70
  - Reference notebook: 0.66 with 224×224
  - Our improvements: +768×768 resolution, +5-head regularization
- **CV**: 0.75-0.78 (similar to EXP036)
- **Gap**: -0.08 to -0.10

**Results:**
- **Public LB**: *TBD*
- **OOF CV**: *TBD*
- **LB-CV Gap**: *TBD*

---

### EXP060: Correct Global Weighted R² Evaluation Function ⭐ **BEST GAP**
**Model**: EXP036 architecture + **corrected evaluation metric**

**Key Change from EXP036:**
- **Fixed `compute_r2_score()` to match official Kaggle competition metric**
- Previous implementation (EXP036) used `.mean()` which was mathematically different
- New implementation uses `.sum()` with proper global weighted mean calculation

**Official Metric Formula:**
```python
# Flatten all predictions: (N, 5) -> (N*5,)
y_true_flat = y_true.flatten()
y_pred_flat = y_pred.flatten()
w = np.tile(weight_list, N)  # Weight for each (sample, target) pair

# Global weighted mean
y_weighted_mean = np.sum(w * y_true_flat) / np.sum(w)

# Correct calculation
SS_res = np.sum(w * (y_true_flat - y_pred_flat) ** 2)
SS_tot = np.sum(w * (y_true_flat - y_weighted_mean) ** 2)
R² = 1 - SS_res / SS_tot
```

**Old (EXP036) vs New (EXP060) Implementation:**
| Aspect | EXP036 (Old) | EXP060 (New) |
|--------|--------------|--------------|
| Aggregation | `.mean()` per target | `.sum()` over all samples |
| Weighted mean | Per-target means × weights | **Global weighted mean** |
| Match official | Approximate | **Exact** |

#### EXP060-child000: Corrected Evaluation Baseline
**Configuration:**
- Architecture: Same as EXP036 (EVA-CLIP-L + 5-head + Consistency Loss)
- Training: Same hyperparameters as EXP036
- **Only change**: Corrected `compute_r2_score()` function

**Results:**
- **OOF CV**: 0.759
- **LB**: **0.69** ⭐ (same as EXP036)
- **LB-CV Gap**: **-0.069** ⭐ (improved from -0.083!)
- Mean R²: 0.757 ± 0.063

**Fold-wise Results:**
| Fold | R² Score |
|------|----------|
| 0 | 0.684 |
| 1 | 0.708 |
| 2 | 0.863 |
| 3 | 0.746 |
| 4 | 0.783 |

**Key Observations:**
- ✅ **Same LB score (0.69)** - model performance unchanged
- ✅ **CV/LB Gap improved**: -0.083 → **-0.069** (17% reduction!)
- ✅ **More reliable CV metric**: Now CV better predicts LB performance
- ✅ **Correct evaluation** allows better experiment comparison
- 💡 **Implication**: Future experiments should use this corrected metric for more accurate CV-LB correlation

##### Inference-only TTA trials (EXP060-child000) ❌
**Goal:** Sampling_Date起因の見た目差（露出/色/画角）を推論側で吸収してLBを上げる（再学習なし）。

**Trials & Results:**
- **Crop-TTA (5-crop zoom; `--crop_tta`, `crop_scale=0.95`)**: **LB 0.69（改善なし、僅かに悪化）**
- **Photo-TTA (brightness/contrast; `--photo_tta`)**: **LB 0.69（改善なし、僅かに悪化）**

**Conclusion:**
- EXP060ベースラインでは、推論TTAだけでSampling_Date差を埋めるのは難しい（TTAはノイズ足しになりやすい）。

**Why Gap Improved:**
- Old metric over-estimated CV by using per-target means
- New metric aligns with Kaggle's official scoring
- CV now gives more realistic estimate of LB performance

**Recommendation:**
- **ALL future experiments should use EXP060's `compute_r2_score()` function**
- This enables better model selection based on CV scores

---

#### EXP060-child025: Balanced date_cluster folds + stronger regularization
**Configuration:**
- CV: **precomputed folds**（`csiro_data_split_eva02_clip_l_14_336_date_cluster_balanced.csv`）
  - Sampling_Dateを跨がない + date_cluster単位でfold固定（fold size: 75/68/75/75/64）
- Training: epochs=35, LR=0.002, WD=0.01, dropout=0.2, hidden_dim=384
- Gradual unfreezing: freeze_epochs=8, max_unfreeze_ratio=0.25
- MixUp: 0.0

**Results:**
- **OOF CV**: 0.700（TTA noCorr=0.7004 / noTTA noCorr=0.6989）
- **LB**: **0.65**
- **LB-CV Gap**: **-0.050**（EXP060-child000: -0.069 → Gapは縮小）

**Key Observations:**
- ✅ Gapは縮小したが、**LBは0.69→0.65に低下**（正則化強化 + foldの厳格化で性能が落ちた可能性）
- ❌ `Total = Green + Dead + Clover` の強制補正はOOFで大きく悪化（-0.04程度）→ デフォルトでは入れない方が安全

---

#### EXP060-child027: Balanced date_cluster folds + stronger finetune（baseline LR/WD）
**Configuration:**
- CV: **precomputed folds**（`csiro_data_split_eva02_clip_l_14_336_date_cluster_balanced.csv`）
  - Sampling_Dateを跨がない + date_cluster単位でfold固定（fold size: 75/68/75/75/64）
- Training: epochs=40, LR=0.003, WD=0.001（強いベースライン寄せ）
- Gradual unfreezing: freeze_epochs=5, max_unfreeze_ratio=0.5
- MixUp: 0.0

**Results:**
- **OOF CV**: 0.748（TTA noCorr=0.7481 / noTTA noCorr=0.7454）
- **LB**: **0.68**
- **LB-CV Gap**: **-0.068**（EXP060-child000: -0.069 と同程度）

**Key Observations:**
- ✅ child-exp025（LB 0.65）よりは回復したが、**単体LBはBest(0.69)に届かず**
- ✅ GapはEXP060-child000並みに戻り、**CVの過剰上振れは抑えられている**
- ⚠️ fold間の難易度差が大きい（厳しいfoldが存在）→ split起因の分散がまだ大きい可能性

---

#### EXP060-child030: Sampling_Date balanced split + stability tweaks ❌
**Configuration:**
- CV: **precomputed folds**（`csiro_data_split_eva02_clip_l_14_336_sampling_date_balanced_min.csv`）
  - Sampling_Dateを跨がない + Sampling_Date単位でfoldサイズを均し（74/68/73/72/70）
- Training: epochs=40, LR=0.003, WD=0.001（強いベースライン）
- Stability: grad_clip_norm=1.0, early_stopping（patience=10/start=12）, max_unfreeze_ratio=0.33, freeze_epochs=8
- MixUp: 0.0

**Results:**
- **OOF CV**: 0.704（oof_r2=0.7037）
- **LB**: **0.63** ❌
- **LB-CV Gap**: **-0.074**（0.63 - 0.7037）

**Key Observations:**
- ❌ Sampling_Date balanced split でも **単体LBが大幅に低下**（0.69 → 0.63）
- 💡 unfreeze cap / freeze長め / early stop が重なって **underfit寄り**になった可能性（要: OOF/CV確認）
- Fold R²: 0.778 / 0.688 / 0.678 / 0.637 / 0.665（mean=0.689 ± 0.048）

---

#### EXP060-child034: 3-Crop Training (Buggy Version) ❌ **SEVERE OVERFITTING**
**Configuration:**
- **3-Crop Splitting**: Left (0-1000), Center (500-1500), Right (1000-2000)
- Inference formula: `pred = (left + center + right) * (2/3)`
- epochs=40, LR=0.003, dropout=0.1, WD=0.001
- 通常のaugmentation適用

**Results:**
- **OOF CV**: 0.777 (+0.018 from EXP060-child000)
- **LB**: **0.65** ❌ (-0.04 from EXP060-child000)
- **LB-CV Gap**: **-0.127** ❌❌ (worst gap in EXP060 series!)

**Bug Discovery:**
- **問題**: Augmentationが各cropに**独立に**適用されていた
- 重複領域 (500-1000, 1000-1500) が**異なるaugmentation**を受ける
- 同じピクセルなのに異なる見た目で学習される → **ラベルノイズ**と同等
- モデルは「augmentationパターン」を記憶してCVが上がるが、テストでは通用しない

**Fold-wise Results:**
| Fold | R² |
|------|-----|
| 0 | 0.711 |
| 1 | 0.777 |
| 2 | 0.861 |
| 3 | 0.801 |
| 4 | 0.733 |
| **Mean** | **0.777 ± 0.053** |

**Key Observations:**
- ❌ EXP061 (grass_type auxiliary loss)と同じCV/LBギャップ (-0.127)
- ❌ 高CVは完全な**偽の信号** - 3-crop augmentation bugによるもの
- 💡 Augmentationはfull imageに適用してから分割すべき

---

#### EXP060-child035: 3-Crop Training (Fixed Augmentation) 🔄 **PENDING**
**Configuration:**
- **3-Crop Splitting**: Same as child-exp034
- **修正点**:
  1. Augmentation修正: **FULL imageに適用してから分割**（crop間の一貫性確保）
  2. Epochs削減: 40 → 27 (3-cropは1.5倍のサンプル数なので、実効epoch数を揃える)
  3. RandomRotate90/RandomResizedCrop除外（full image augmentationで破綻するため）
- Deterministic seeding: `img_idx + epoch * 100003` で同一画像の3cropに同じaugmentation適用

**Technical Fix:**
```python
# Apply augmentation to FULL 2000x1000 image BEFORE splitting
if self.augmentation is not None:
    aug_seed = img_idx + self.epoch * 100003  # Large prime for epoch variation
    py_random.seed(aug_seed)
    np.random.seed(aug_seed % (2**32))
    augmented = self.augmentation(image=image_np)
    image_np = augmented['image']
# Then split into 3 crops (all receive identical augmentation)
```

**3-Crop Safe Augmentation** (excludes transforms that break 2000x1000 aspect ratio):
- ✅ HorizontalFlip, VerticalFlip, Rotate(±10°), ColorJitter, RandomGamma, etc.
- ❌ RandomRotate90 (would change 2000x1000 → 1000x2000)
- ❌ RandomResizedCrop (size=1000x1000 would crop to square)

**Expected Improvement:**
- CV should decrease (no longer memorizing augmentation patterns)
- LB should improve (better generalization)
- Gap should shrink significantly

**Results:** TBD (pending training)

---

#### EXP060-child036: 5-Fold Baseline Reproduction
**Configuration:**
- CV: 5-fold StratifiedGroupKFold (standard)
- Architecture: EVA-CLIP-L (same as child-exp000)
- Training: epochs=40, LR=0.003, WD=0.001

**Results:**
- **OOF CV**: 0.769
- **LB**: **0.66**
- **LB-CV Gap**: **-0.109**

**Key Observations:**
- ❌ LBはEXP060-child000 (0.69) より悪化
- ⚠️ Gapが大きい（-0.109）

---

#### EXP060-child037: 10-Fold CV
**Configuration:**
- CV: **10-fold** StratifiedGroupKFold
- Architecture: EVA-CLIP-L (same as child-exp000)
- Training: epochs=40

**Results:**
- **OOF CV**: 0.785
- **LB**: **0.67**
- **LB-CV Gap**: **-0.115**

**Key Observations:**
- ❌ 10-foldはOOF上振れ（0.785）、LB悪化（0.67）
- ❌ Gapが最悪（-0.115）→ 10-foldはoverfitしやすい

---

#### EXP060-child038: DinoV3 ViT-Huge+ Backbone ⭐ **NEW RECORD (LB 0.71)**
**Configuration:**
- **Backbone変更**: EVA-CLIP-L → **DinoV3 ViT-Huge+** (vit_huge_plus_patch16_dinov3_qkvb via timm)
- CV: 5-fold StratifiedGroupKFold
- Training: epochs=40, LR=0.003, WD=0.001
- Gradual unfreezing: freeze_epochs=5, max_unfreeze_ratio=0.5

**Results:**
- **OOF CV**: **0.7758**
- **LB**: **0.71** ⭐⭐⭐ **NEW RECORD**
- **LB-CV Gap**: **-0.066** (Best gap in EXP060 series!)

**Fold-wise Results:**
| Fold | R² Score |
|------|----------|
| 0 | 0.649 |
| 1 | 0.830 |
| 2 | 0.879 |
| 3 | 0.821 |
| 4 | 0.788 |

**Key Observations:**
- ✅ **LB +0.02** (0.69 → 0.71) - backbone変更で大幅改善
- ✅ **Gap改善**: -0.069 → **-0.066**
- ✅ DinoV3はEVA-CLIPより牧草画像に適している
- 💡 以降のchild実験はこれをbaselineとする

---

#### EXP060-child039〜055: LR/Unfreeze/Loss チューニング実験 ❌ **すべて失敗**

**Baseline**: child-exp038 (DinoV3 ViT-Huge+, OOF 0.7758, LB 0.71)

| Exp | 変更点 | OOF R² | LB | vs 038 OOF | Gap |
|-----|--------|--------|-----|------------|-----|
| **038** | DinoV3 baseline (50% unfreeze) | **0.7758** | **0.71** | - | -0.066 |
| 039 | MSE for Total/GDM | 0.7815 | 0.69 | +0.006 | -0.092 |
| 041 | DinoV2 ViT-Giant | 0.7527 | 0.68 | -0.023 | -0.073 |
| 042 | LODO 10-fold CV | 0.7851 | 0.68 | +0.009 | -0.105 |
| 043 | low LR (head=0.001, bb=2e-5) | 0.7664 | - | -0.009 | - |
| 044 | + LLRD=0.85 | 0.7665 | - | -0.009 | - |
| 045 | + EMA | 0.7695 | - | -0.006 | - |
| 046 | LLRD=0.95 | 0.7737 | - | -0.002 | - |
| 047 | backbone LR cap 4e-5 | 0.7693 | - | -0.007 | - |
| 048 | 35ep + 4.5e-5 | 0.7579 | - | -0.018 | - |
| 049 | Tweedie loss (Clover) | 0.7709 | - | -0.005 | - |
| 051 | 80% unfreeze | 0.7715 | - | -0.004 | - |
| 052 | 30% unfreeze | 0.7734 | - | -0.002 | - |
| 053 | batch16 + LR 0.0045 | 0.7706 | 0.70 | -0.005 | -0.071 |
| **054** | 512x512 input size | 0.7725 | **0.71** | -0.003 | -0.063 |
| 055 | DinoV3 ViT-Large + 2xLR | 0.7587 | 0.67 | -0.017 | -0.089 |

**主要な発見:**

1. **LR関連 (exp043-048): すべて失敗**
   - 低LR (head=0.001, bb=2e-5): OOF -0.009
   - LLRD 0.85/0.95: ほぼ効果なし
   - EMA: わずかに改善だが不十分
   - **結論**: exp038のLR設定が最適

2. **Loss関連 (exp039, 049)**
   - MSE for Total/GDM: OOF+0.006 だが **LB -0.02 → overfitting**
   - Tweedie for Clover: OOF -0.005 → 効果なし

3. **Unfreeze比率 (exp051, 052)**
   - 80% unfreeze: OOF -0.004
   - 30% unfreeze: OOF -0.002
   - **結論**: 50%が最適

4. **Backbone変更 (exp041, 055)**
   - DinoV2 ViT-Giant: LB 0.68 (-0.03)
   - DinoV3 ViT-Large: LB 0.67 (-0.04)
   - **結論**: DinoV3 ViT-Huge+が最良

5. **Input size (exp054)**
   - 512x512: LB 0.71維持 → 学習速度向上の選択肢

**結論:**
- **exp038 (DinoV3 ViT-Huge+ + 50% unfreeze) は局所最適に到達**
- 同一アーキテクチャでのハイパラチューニングでは改善困難
- LB改善には異なるアプローチ（アーキテクチャ変更、データ拡張等）が必要

---

#### EXP060-child057: DinoV3 768px Image Size
**Configuration:**
- Backbone: DinoV3 ViT-Huge+ (same as exp054)
- **Image size**: 512px → **768px**
- CV: 5-fold StratifiedGroupKFold (standard)

**Results:**
- **OOF CV**: 0.787
- **LB**: **0.71**
- **LB-CV Gap**: **-0.077**

**Key Observations:**
- ✅ 768px維持でLB 0.71（exp054同等）
- ⚠️ Gapがexp054 (-0.063) より悪化 (-0.077)

---

#### EXP060-child058: Artem CV Strategy ⭐⭐⭐ **CURRENT BEST (LB 0.73)**
**Configuration:**
- Backbone: DinoV3 ViT-Huge+
- Image size: 512px
- **CV strategy変更**:
  - Stratify: **Clover/Dead presence** (4 categories: "0_0", "0_1", "1_0", "1_1")
  - Group: **day/month_State** (日/月と州の組み合わせ)

**Results:**
- **OOF CV**: 0.792
- **LB**: **0.73** ⭐⭐⭐ **NEW RECORD**
- **LB-CV Gap**: **-0.062**
- Fold scores: [0.673, 0.810, 0.866, 0.835, 0.805]

| State | wR² |
|-------|-----|
| NSW | 0.750 |
| Vic | 0.790 |
| Tas | 0.736 |
| WA | 0.777 |

**Key Observations:**
- 🎉 **CV strategy変更だけでLB +0.02!** (0.71 → 0.73)
- ✅ Clover/Dead presenceでのstratifyがテスト分布にマッチ
- ✅ day/month_Stateグループがtemporal leakageを適切に防止
- 💡 **モデル変更なしでCV strategyのみでLB大幅改善**

---

#### EXP060-child059: DinoV3 768px + Artem CV ❌ **Overfitting**
**Configuration:**
- Backbone: DinoV3 ViT-Huge+
- **Image size**: 512px → **768px**
- CV strategy: **Artem CV** (exp058と同じ)

**Results:**
- **OOF CV**: 0.800 (+0.008 from exp058)
- **LB**: **0.70** (-0.03 from exp058) ❌
- **LB-CV Gap**: **-0.100** (exp058: -0.062)
- Fold scores: [0.851, 0.783, 0.733, 0.833, 0.750]

| State | wR² | vs exp058 |
|-------|-----|-----------|
| NSW | 0.766 | +0.016 |
| Vic | 0.804 | +0.014 |
| Tas | 0.752 | +0.016 |
| WA | **0.688** | **-0.089** ❌ |

**Key Observations:**
- ❌ **768px + Artem CVは過学習**: CVは上がるがLBは下がる
- ❌ **WAが大幅悪化** (-0.089): 768pxで低解像度WAが苦手に
- ⚠️ NSW/Vic/Tasは改善するがWAの悪化がLBを下げる
- 💡 **Artem CVではexp058の512pxが最適**

---

#### EXP060-child060: EVA02-CLIP + Artem CV ❌ **EVA is sensitive to CV strategy**
**Configuration:**
- **Backbone変更**: DinoV3 → **EVA02-CLIP-L-14-336**
- CV strategy: **Artem CV** (exp058と同じ)

**Results:**
- **OOF CV**: 0.783
- **LB**: **0.67** (-0.02 from exp000's 0.69) ❌
- **LB-CV Gap**: **-0.113**

| State | wR² | vs exp000 |
|-------|-----|-----------|
| NSW | 0.735 | +0.063 |
| Vic | 0.802 | +0.002 |
| Tas | 0.736 | +0.012 |
| WA | **0.669** | **-0.085** ❌ |

**Key Observations:**
- ❌ EVA02-CLIPはCV strategyに敏感
- ❌ Artem CVでWAが大幅悪化 → LB低下
- 💡 DinoV3はrobust、EVA02-CLIPはsensitive to fold strategy
- 💡 EVA02-CLIPはexp000のoriginal CV strategyが最適

---

### EXP061: Grass Type Auxiliary Loss ❌ **FAILED - Severe Overfitting**
**Model**: EXP060 + grass_type auxiliary classification head

**Hypothesis:**
- Different grass species (ryegrass, phalaris, fescue, lucerne) have different biomass characteristics
- Phalaris/Fescue have 2-3x higher biomass than ryegrass despite similar appearance
- Auxiliary classification task might help backbone learn species-specific features
- This information is hard to learn from images alone

**Architecture:**
- EVA-CLIP + 5-Head regression + **1 auxiliary classification head**
- grass_type: 5 classes (ryegrass=160, phalaris=76, fescue=38, lucerne=22, other=61)
- Auxiliary head NOT used during inference (regression only)
- Auxiliary loss weight: 0.1 (small to not dominate regression task)

#### EXP061-child000: Grass Type Auxiliary Loss
**Configuration:**
- Same as EXP060 + grass_type auxiliary CrossEntropyLoss (weight=0.1)
- Auxiliary head excluded from model checkpoint (inference uses only 5 regression heads)

**Results:**
- **OOF CV**: 0.777 (+0.018 from EXP060)
- **LB**: **0.65** ❌ (-0.04 from EXP060)
- **LB-CV Gap**: **-0.127** ❌ (much worse than -0.069!)
- Mean R²: 0.771 ± 0.047

**Fold-wise Comparison:**
| Fold | EXP060 | EXP061 | Diff |
|------|--------|--------|------|
| 0 | 0.684 | 0.711 | +0.027 |
| 1 | 0.708 | 0.729 | +0.021 |
| 2 | 0.863 | 0.843 | -0.020 |
| 3 | 0.746 | 0.795 | +0.049 |
| 4 | 0.783 | 0.779 | -0.004 |

**Key Observations:**
- ❌ **Severe overfitting**: CV improved +0.018 but LB dropped -0.04
- ❌ **CV-LB Gap worsened**: -0.069 → **-0.127** (84% worse!)
- ⚠️ grass_type information exists only in training data, not in test
- ⚠️ Auxiliary task learned training-data-specific patterns
- 💡 Species distribution likely differs between train and test sets

---

### EXP071: EVA + Green-only Depth Scalar Branch (DepthAnything3) ⚖️ **CV/LBほぼ据え置き**
**Model**: EXP060 EVA-CLIP 5-head baseline + Green head 用 depth スカラー MLP 分岐  
**Depth**: `DepthAnything3` (npy マップ → 全画像共通のスケールで正規化 → 統計量 8次元）

**目的 / 仮説:**
- 画像から見える「面積」だけでなく、「高さ」や「カメラの距離」を depth で補助的に与えたい。
- 特に `Dry_Green_g` / `GDM_g` / `Dry_Total_g` のスケール感（背丈）を安定させる。
- ただし depth が効くのは Green 成分中心と予想 → **Green head にだけ depth 分岐を付与**。

**実装ポイント:**
- 既存 EVA 5-head アーキテクチャは維持しつつ、Green head の直前に `green_depth_mlp` を追加。
  - 入力: `depth_mean`, `depth_std`, `depth_min`, `depth_max`, `depth_range`, `depth_p10`, `depth_p50`, `depth_p90`（計8次元）
  - 出力: Green 埋め込みへの加算 or concat（EXP071 実装に合わせて MLP で軽く融合）
- DepthAnything3:
  - Colab 側で train 全画像の npy を生成 (`input/depth_maps_4ch_inverce`)、そこから map_feature.csv を作成。
  - Kaggle 側では **map_feature.csv だけ**を使って depth スカラーを再利用（test 用は別スクリプトで生成）。
- CV / fold 割り / optimizer / 学習スケジュールは **EXP060 のまま**（評価関数も EXP060 実装）。

**Results (child-exp000):**
- **Fold R²**: [0.642, 0.723, 0.857, 0.778, 0.807]
- **Mean R²**: 0.761 ± 0.074
- **OOF CV (global weighted R²)**: **0.758**
- **LB**: **0.69**
- **LB-CV Gap**: **-0.068**（EXP060: -0.069 とほぼ同じ）

**何ができたか:**
- ✅ CV は EXP060 と同レベル（0.759 → 0.758）を維持しつつ、fold パターンも大崩れしていない。
- ✅ depth スカラーを train / infer の両方で安全に扱うパイプライン（Colab→Kaggle）が整った。
  - Kaggle では test 画像用に DepthAnything3 で npy を生成 → 同じ統計量で features を計算。
  - test.csv の 5 行重複問題（1 image につき 5 target）も、ユニークな image_path ベースで処理し直して解決。
- ✅ 実験から「depth は主に Green に効く」「Dead/Clover には効かない」という挙動が確認できた。

**何がだめだったか / 限界:**
- ❌ **LB 改善がゼロ**: EXP060 と同じ LB 0.69。Green の安定化はしているが、全体メトリックは動かせていない。
- ❌ depth を Green のみ補助にしたため、Total/GDM に対する効果が限定的。
- ❌ depth 統計だけでは「撮影条件（State / Sampling_Date / カメラなど）のドメイン差」までは吸収しきれていない。

**学び / 次に活かすポイント:**
- Depth は「直接スコアを押し上げる main feature」というより、**Green 用のチューニング用補助特徴**として扱うのが現実的。
- EVA RGB の backbone を強くする（EXP072/073）ことが single 向上の主戦場で、depth はその上に薄く足す方向が良さそう。
- depth スカラーのパイプラインが完成したことで、今後は
  - Green 専用 loss 調整
  - State / Sampling_Date ごとの depth 統計分布の差を埋める正則化
など、**ピンポイントな使い方**に転換できる。

---

### EXP072: EVA RGB-only + 撮影条件ロバストな Augmentation バリエーション（child-exp001/002） ⚖️
**Base**: EXP060/072-child000 の EVA02-CLIP-L-14-336 + 5-head + Consistency (0.05), depth 無し  
**目的**: 「カメラ高さ・画質・日照条件が違う」テスト分布にロバストな RGB モデルを作る。

#### EXP072-child001: 強いスケール・カメラ差 Aug (`augmentation.version: "strong"`, split)
**Configuration:**
- `augmentation.version: "strong"`
  - `RandomResizedCrop(size=1000, scale=(0.7, 1.0), ratio=(0.9, 1.1))`
    - → カメラ高さ/距離の大きな違いを模倣（葉が大きく/小さく写る）。
  - 強めの ColorJitter / Gamma / Noise / JPEGCompression / Blur / RGBShift でセンサー差を再現。
- `augmentation.dataset_type: "split"`（左右 2 分割のみ、Mosaic なし）
- `mixup_alpha: 0.0`（Mixup なし）
- それ以外は EXP072-child000（= EXP060 派生）の設定と同じ。

**Results:**
- **Fold R²**: [0.666, 0.718, 0.821, 0.765, 0.790]
- **Mean R²**: 0.752 ± 0.055
- **OOF CV**: **0.755**
- **LB**: **0.65**（EXP060/071 より -0.04）
- **LB-CV Gap**: **-0.105**（EXP060: -0.069 より悪化）

**何ができたか / 評価:**
- ✅ CV 自体はそこまで悪くなく、EXP071 よりやや低い程度（0.758 → 0.755）。
- ✅ Camera 高さ・画質差に対する「過学習」を抑えようとした方向性は合っている。
- ❌ しかし **LB が 0.65 に落ち、Gap も -0.105 と悪化**。
  - 強い RandomResizedCrop + 色変換が「実際のテスト分布からズレた世界」を作り過ぎた可能性。
  - 特に葉が極端に大きく/小さく写るケースが増え過ぎ、現実の test 条件とはミスマッチになったかもしれない。

**結論（child001）:**
- 「強いスケール + カメラ差 Aug」は CV をそこそこ維持しつつ、LB を削る方向に働いた。  
  → **single の主力にはせず、CV が違うモデルとしてアンサンブル要員**に回すのが良さそう。

---

#### EXP072-child002: physics-safe + Mosaic scale + 軽 Mixup (`dataset_type: "mosaic"`)
**Configuration:**
- `augmentation.version: "physics_safe"`
  - EXP072-child000 と同じ「ラベルを壊さない」単画像 Aug（回転/flip/brightness/hue 小さめ など）。
- `augmentation.dataset_type: "mosaic"`
  - `mosaic_prob: 0.5` → 50% のサンプルで 4画像モザイク (2x2)。
  - モザイク画像のラベルは 4枚の平均 → 「カメラが遠い / 視野が広い」状況をラベル一貫性を保ったまま再現。
- `mixup_alpha: 0.2`
  - 弱い Mixup で State / Sampling_Date / カメラ差に対するスムーズさを少しだけ付与。
- それ以外の training 設定は child-exp000/001 と同一。

**Results:**
- **Fold R²**: [0.634, 0.688, 0.854, 0.702, 0.776]
- **Mean R²**: 0.731 ± 0.077
- **OOF CV**: **0.731**
- **LB**: **0.64**
- **LB-CV Gap**: **-0.091**

**何ができたか / 評価:**
- ✅ Mosaic により「カメラが遠いシーン」へのロバスト性を増やす狙いは概ね成立（Fold2 は 0.854 と高い）。
- ✅ Gap は child001(-0.105) よりややマシ（-0.091）だが、それでも EXP060(-0.069) より悪い。
- ❌ CV が明確に低下（0.758 → 0.731）し、LB も 0.64 と baseline (0.69) から大きく後退。
- ❌ Mosaic + Mixup の組み合わせは、CSIRO のデータ量（375 images）ではやや正則化過多で、  
  **Train 分布自体をぼやかし過ぎた**印象。

**結論（child002）:**
- physics-safe + Mosaic + Mixup は「物理的にはきれい」だが、**このコンペのスケールでは情報を削り過ぎ**。  
  → 単体では EXP060 系に勝てないが、child001 同様「誤差の出方が違うモデル」としてアンサンブル候補にはなる。

---

### 3 実験（EXP071, EXP072-child001/002）からの総括
- ✅ **EXP071 (Depth Green branch)**:
  - EVA RGB 主体のまま、Green 用に depth スカラーを足しても **CV/LB は維持**できる。
  - depth のパイプラインが確立したこと自体が大きな収穫（今後の Green 特化チューニングに使える）。
- ⚖️ **EXP072-child001/002 (Aug 改良)**:
  - 「撮影条件ロバスト化」を目指した強い Aug/Mosaic は、  
    **CV をそこそこ維持しつつ LB を落とす**（＝分布差の方向を外している）ことがわかった。
  - 強いスケール変換や Mosaic + Mixup は、CSIRO の小さいデータセットでは「やり過ぎ」寄り。
- 🎯 **戦略的インサイト**:
  - シングルで LB を伸ばす主役は、依然として **EVA RGB (EXP060 line)**。
  - Depth は Green スケール補助として薄く足しつつ、  
    Augmentation は **EXP060 系の「当たりパターン」に近い physics-safe/デフォルトを基軸**にするのが安全。
  - 今回の EXP072-child001/002 は、  
    「撮影条件をかなり変えた世界でも CV はそこそこ出るが、LB には直結しない」ことを確認した **探索実験**として位置づけ、  
    今後は **外部 pretrain (EXP073) + 既存 Aug を軸**にスコアアップを狙う。 

**Why It Failed:**
1. **Training data memorization**: grass_type acts as a proxy for sampling location/conditions
2. **Distribution shift**: Species distribution in test set differs from training
3. **Information leakage**: CV benefits from same species distribution, LB doesn't
4. **Auxiliary task biased backbone**: Even at weight=0.1, auxiliary task distorted learned features

**Lesson Learned:**
- ❌ **DO NOT use auxiliary tasks based on training-only metadata**
- ❌ **All auxiliary task experiments have failed**: EXP021 (metadata), EXP022 (species), EXP048 (height), EXP061 (grass_type)
- ⭐ **Stick with EXP060**: Best model remains EVA-CLIP + 5-head + Consistency Loss (LB 0.69)
- 💡 **The backbone should learn only from image content**, not metadata correlations

---

### EXP065: DINOv2 + Higher Resolution (448x448) + SAM
**Model**: DINOv2-giant with higher resolution input (448x448 vs 224x224)

**Hypothesis:**
- Higher resolution (448x448) captures more fine-grained details (clover leaves, thin grass)
- More patches (1024 vs 256) provide richer spatial information
- SAM optimizer helps generalization

**Base Experiment:** EXP056 (DINOv2-giant + 224x224 + SAM)

**Architecture:**
- DINOv2-giant backbone (frozen)
- **Input: 448x448** (vs 224x224 in EXP056)
- **Patches: 1024** (32x32 grid, vs 256 patches in EXP056)
- Simple MLP (1536 → 128 → 5)
- SAM optimizer (rho=0.05)
- Consistency Loss (weight=0.1)

#### EXP065-child000: DINOv2-giant + 448x448 + SAM
**Configuration:**
- Same as EXP056 except **img_size=448** (vs 224)
- batch_size=8 (reduced due to higher resolution)
- epochs=40, LR=0.001, dropout=0.3

**Results:**
- **OOF CV**: **0.786** (+0.030 from EXP056!)
- **LB**: **0.66** (+0.02 from EXP056)
- **LB-CV Gap**: -0.126 (worse than EXP056's -0.116)
- Mean R²: 0.807 ± 0.072

**Fold-wise Comparison:**
| Fold | EXP056 | EXP065 | Diff |
|------|--------|--------|------|
| 0 | 0.659 | 0.702 | **+0.043** |
| 1 | 0.727 | 0.735 | +0.008 |
| 2 | 0.871 | 0.880 | +0.009 |
| 3 | 0.787 | 0.809 | +0.022 |
| 4 | 0.749 | 0.779 | +0.030 |

**Key Observations:**
- ✅ **Consistent improvement across all folds**
- ✅ **LB improved +0.02** (0.64 → 0.66)
- ✅ **Fold 0 significantly improved** (+0.043, previously worst fold)
- ⚠️ **CV-LB Gap worsened**: -0.116 → -0.126
- 💡 Higher resolution helps capture fine-grained details
- 💡 More patches = better spatial information for DINOv2

**Why Gap Worsened:**
- 4x more patches (256 → 1024) = more capacity to memorize training patterns
- Higher resolution may capture training-specific details not present in test
- SAM alone cannot fully compensate for increased model capacity

**Trade-off Analysis:**
- Higher resolution improves absolute performance (LB +0.02)
- But increases overfitting risk (Gap -0.01 worse)
- Still inferior to EVA-CLIP's LB 0.69 (EXP060)

**Recommendation:**
- DINOv2 with higher resolution is a viable option for LB 0.66
- For best LB, **EXP060 (EVA-CLIP + 5-head)** remains superior (LB 0.69)
- Consider combining EVA-CLIP with higher resolution for future experiments

---

### EXP077: Left/Right Split Label Correction (Teacher OOF) ❌ **CV↑だがLB大幅悪化**
**Model**: EXP060（EVA-CLIP + 5-head + Consistency Loss）を維持、学習時の「左右半分ラベル」を補正

**目的 / 仮説:**
- train.csv のアノテーションは 2000x1000 全体に対して付いているため、左右分割学習で常に `label/2` にするのはラベルノイズになり得る。
- OOF の left/right 予測から「左右の割合」を推定し、**総量は train.csv を厳守**したまま半分ラベルだけ補正すれば汎化が上がるはず。

**実装ポイント:**
- Teacher: `EXP/EXP036/outputs/child-exp000/oof_prediction_left-right.csv`
- Green/Dead/Clover の left/right 予測から比率を計算し、`imbalance=|L-R|/(L+R)` が閾値超のときだけ 0.5 から離す（それ以外は 0.5 に縮退）。
- 比率は `[0.05, 0.95]` にクリップ、GDM/Total は成分（Green/Dead/Clover）の半分ラベルから派生して整合性を保持。

**Results:**
- **OOF CV**: **0.77**
- **LB**: **0.63** ❌（EXP060: 0.69 から -0.06）
- **LB-CV Gap**: **-0.14**（過学習/分布差が顕在化）

**Key Observations:**
- ✅ CV は上がるが、❌ LB が大きく落ちるため実運用/提出には不向き。
- 💡 左右比率の補正は train には効くが、test 側で左右の写り方（フレーミング/撮影条件）が異なると破綻しやすい。

---

### EXP080: External Warmup Init（Stage1: 外部画像蒸留 → Stage2: CSIRO fine-tune）⚠️ **CVリークの疑い / LB伸びず**
**概要:**
- Stage1で外部画像（GrassClover / Irish）に対して、CSIRO学習済みモデル（Teacher）の出力へStudentを寄せる蒸留を実施し、`eva_encoder_external.pth`（vision encoderのみ）を作成。
- Stage2で上記encoderを初期値としてCSIROを学習。

**注意（CVリーク）:**
- Stage1のTeacherに `best_model_fold0〜4` をまとめて使い、生成した `eva_encoder_external.pth` を全foldのStage2で共有すると、**fold-kのval画像が別fold teacherの学習に含まれる**ためCVが不自然に上振れしやすい（EXP060→EXP080で顕著）。

**Results (child-exp000):**
- **OOF CV**: **0.8227**
- **LB**: **0.67**（EXP060: 0.69 から **-0.02**）

**Key Observations:**
- ❌ **LBは改善せず**。CVが高いが、上記理由で信頼しづらい。

---

### EXP081: fold-safe External Distill Init ✅ **リーク回避したがLB悪化**
**概要:**
- Stage1をfoldごとに分離（Teacherは `best_model_fold{k}.pth` のみ）し、`eva_encoder_external_fold{k}.pth` を生成。
- Stage2のfold-kは対応encoderで初期化（CVリーク回避）。

**Results (child-exp000):**
- **OOF CV**: **0.7756**
- **LB**: **0.65** ❌（EXP060: 0.69 から **-0.04**）

**Key Observations:**
- ✅ CVリークは避けられる設計。
- ❌ 現状のハイパラだと外部encoder初期化は**負の転移**になっており、提出用途としては不向き。

---

### EXP082: Log Transformation for Targets ❌ **LB大幅悪化**
**Model**: EXP060ベース（EVA-CLIP + 5-head + Consistency Loss）、ターゲットをlog(1+y)変換して学習

**目的 / 仮説:**
- 論文Section 4.5に「log-stabilizing transformation: y_trans = log(1 + y)」と記載
- Log変換により右に歪んだ分布を正規化し、小さい値の相対誤差を重視した学習が可能に
- 高バイオマス値の外れ値の影響を低減

**実装:**
- 学習時: ターゲットをlog(1+y)変換してSmoothL1Loss
- 推論時: 予測値をexp(y)-1で逆変換
- Consistency Lossもlog空間で計算（LogSumExpを使用）

#### EXP082-child000: Log Transformation Baseline
**Configuration:**
- Base: EXP060-child000と同じアーキテクチャ
- **ターゲット変換**: log(1+y) for training
- **Loss**: SmoothL1 in log space
- **R²評価**: log空間で計算（results.jsonに記録）

**Results:**
- **OOF CV (log space)**: 0.7746
- **OOF CV (raw space)**: **0.7312** ← Kaggle評価と一致
- **LB**: **0.65** ❌（EXP060: 0.69 から **-0.04**）
- **LB-CV Gap**: -0.08（raw spaceで計算）

**Per-Target R² (raw space) 比較:**
| Target | EXP060 | EXP082 | 差分 |
|--------|--------|--------|------|
| Total | 0.70 | 0.66 | **-0.04** |
| GDM | 0.71 | 0.68 | -0.03 |
| Green | 0.65 | 0.59 | -0.06 |
| Dead | 0.35 | 0.36 | +0.01 |
| Clover | 0.40 | **0.60** | **+0.20** |

**Key Observations:**
- ❌ **LB大幅悪化**: 0.69 → 0.65（-0.04）
- ❌ **Total/GDM/Green が悪化**: 最も重要なターゲットの精度低下
- ✅ **Cloverは大幅改善**: R² 0.40 → 0.60（+0.20）、log変換で小さい値の学習が改善
- ⚠️ **評価空間のミスマッチ**: 論文ではlog空間R²を使用と記載があったが、Kaggle HPにはlog変換の記載なし

**失敗の原因分析:**
1. **Kaggle評価はraw空間**: log空間で最適化してもraw空間のR²は改善しない
2. **大きい値の予測精度低下**: log空間での学習は大きい値（Total, GDM）の相対誤差を軽視
3. **重み付けの効果減少**: Total(50%)の寄与がlog空間では相対的に小さくなる

**Lesson Learned:**
- ❌ **Log変換はKaggle評価と相性が悪い**（評価がraw空間のため）
- ❌ **論文の記載とKaggle評価の仕様が異なる**ことに注意
- ⭐ **EXP060（raw空間での学習）がベスト**を維持

---

### ENSEXP_003: DepthAnything3 線形補正（EXP060 OOF誤差の回帰）❌ LB悪化
**概要:**
- EXP060-child000 の予測に対して、DepthAnything3 の8統計量で誤差を線形回帰。
- 目的: depth を草丈の代理として誤差補正。

**Results:**
- **OOF CV (global weighted R²)**: **0.8088**
- **LB**: **0.65** ❌（EXP060: 0.69 から **-0.04**）
- **LB-CV Gap**: **-0.159**

**Key Observations:**
- ❌ OOFでは大幅改善だが公開LBで悪化。後段回帰の汎化が不足。
- ⚠️ 回帰は全データでfitしており、fold外評価ではないためCVが楽観的。
- ⚠️ depth正規化のglobal min/maxがtrain/testでずれると補正量が崩れるため、同一値の固定が必須。

---

### ENSEXP_004: パーセンタイルベースDepth補正（最小パラメータ）❌ LB悪化
**概要:**
- ENSEXP_003の失敗から学んだ「過学習しないDepth補正」
- **パラメータ数**: 2個のみ（percentile=60, scale=1.15）
- テストデータ自身から閾値を計算（絶対閾値ではなくパーセンタイル）

**手法:**
```python
# TEST画像のdepth_stdから60パーセンタイルを計算
threshold = np.percentile(test_depth_std, 60)

# 閾値より高いサンプルの予測を1.15倍
if depth_std > threshold:
    pred_total *= 1.15
    pred_gdm *= 1.15
    pred_green *= 1.15
```

**ENSEXP_003との違い:**
| 項目 | ENSEXP_003 | ENSEXP_004 |
|------|-----------|-----------|
| パラメータ数 | 40個（8特徴×5ターゲット） | **2個** |
| 閾値 | 絶対値（trainから学習） | **パーセンタイル（testで計算）** |
| CV改善 | +0.050 | +0.024 |
| LB | 0.65 | **0.66** |

**Results:**
- **OOF CV (global weighted R²)**: **0.783**（+0.024 from EXP060）
- **LB**: **0.66** ❌（EXP060: 0.69 から **-0.03**）
- **LB-CV Gap**: **-0.123**

**Key Observations:**
- ✅ ENSEXP_003よりLB改善（0.65 → 0.66）、Gap改善（-0.159 → -0.123）
- ❌ それでもEXP060単体（LB 0.69）より悪い
- ⚠️ Depth特徴量による後処理補正はtrain/test間の分布ズレに対応できない
- ⚠️ **EXPDEPTH/EXP001の分析結果**: depth_std と Height の相関は r=0.824 と高いが、OOF誤差との相関は最大でも r=-0.576（Green）程度

**Lesson Learned:**
- ❌ **パラメータ数を減らしても汎化しない**: 2個でも-0.03のLB悪化
- ❌ **パーセンタイルベースでも不十分**: test分布への適応は限定的
- ⭐ **EXP060単体がベスト**: 後処理補正は全て悪化
- ⚠️ **Depth補正の限界**: 画像からのEnd-to-End学習（EXP069等）が必要か

---

### EXP085: Log-Space Training with Raw-Space Consistency Loss ❌❌ **WORST EVER (LB 0.60)**
**Model**: EXP060ベース、log(1+y)空間で学習、Consistency Lossはraw空間

**目的 / 仮説:**
- EXP082の失敗を踏まえ、Consistency Lossだけはraw空間で計算することで物理的整合性を維持
- log空間での学習により小さい値の学習が改善されると期待

**実装:**
- 学習時: ターゲットをlog(1+y)変換してSmoothL1Loss
- Consistency Loss: **raw空間で計算**（log予測をexp変換してからDead, Cloverを計算）
- 推論時: exp(y)-1で逆変換

#### EXP085-child000: Log Space + Raw Consistency
**Configuration:**
- Base: EXP060-child000と同じアーキテクチャ
- **ターゲット変換**: log(1+y) for training
- **Loss**: SmoothL1 in log space + **Consistency in raw space**

**Results:**
- **OOF CV**: 0.7484
- **Mean R²**: 0.7396 ± 0.020
- Fold R² scores: [0.775, 0.720, 0.747, 0.732, 0.724]
- **LB**: **0.60** ❌❌ **← 全実験中で最悪**
- **LB-CV Gap**: **-0.15** ❌❌ **← 全実験中で最悪のGap**

**Key Observations:**
- ❌❌ **LB 0.60 = 全実験中で最悪**: EXP060 (0.69) から **-0.09**
- ❌❌ **Gap -0.15 = 全実験中で最悪**: 通常 -0.07〜-0.10 が -0.15 に悪化
- ⚠️ **物理的整合性の破綻**: log空間での学習とraw空間のConsistency Lossの**空間ミスマッチ**が致命的

**失敗の原因分析:**
1. **空間ミスマッチが最悪の組み合わせ**:
   - 5つのhead（Total, GDM, Green, Dead, Clover）はlog空間で独立に最適化
   - しかしConsistency Lossはraw空間で `Dead = Total - GDM`, `Clover = GDM - Green` を強制
   - log空間とraw空間では**加法性が異なる**: `log(a) + log(b) ≠ log(a + b)`
   - 結果: モデルは両方の制約を同時に満たせず、どちらも中途半端に学習

2. **物理制約の破綻**:
   - Raw空間: `Total = Dead + GDM` (正しい)
   - Log空間: `log(Total) ≠ log(Dead) + log(GDM)` (誤り)
   - モデルはlog空間で最適化しつつraw空間の制約を満たそうとして混乱

3. **EXP082との違い**:
   - EXP082: log空間でConsistency Lossも計算（一貫性あり、LB 0.65）
   - EXP085: log空間学習 + raw空間Consistency（不整合、**LB 0.60**）
   - **不整合は一貫して悪い方法より更に悪い**

**Lesson Learned:**
- ❌❌ **空間を混在させるな**: 学習とConsistency Lossは同じ空間で計算必須
- ❌ **log変換自体がこのコンペに不向き**: EXP082, EXP085ともに失敗
- ⭐ **EXP060（raw空間、全て一貫）がベスト**を再確認
- ⚠️ **Gap -0.15は危険信号**: 汎化性能の完全な崩壊を示す

---

### EXP095: SigLIP/EVA-CLIP + GBDT Pipeline ❌ **Neural Netに劣る**
**Model**: Vision Encoder (SigLIP/EVA-CLIP) で画像埋め込み抽出 → GBDT (GradientBoosting, LightGBM, CatBoost) で予測

**目的:**
- 公開ノートブック (LB 0.66) の SigLIP + GBDT アプローチを再現・拡張
- Neural Net (EXP060) との比較検証

**アーキテクチャ:**
- 画像埋め込み: SigLIP / SigLIP2 / EVA-CLIP
- 特徴量エンジニアリング: PCA (0.80-0.85) + PLS (8成分) + GMM (6成分)
- モデル: GradientBoosting + HistGradientBoosting + LightGBM + CatBoost アンサンブル
- Semantic features: CLIPテキストエンコーダーによるプロンプト類似度

**CV戦略の検証:**
- Visual Cluster: 画像埋め込みに基づくクラスタリング
- Sampling_Date Group: 同じ撮影日は同じFold（リーク防止）
- date_cluster: 似た撮影日を同じFoldにまとめる

#### EXP095-child000: SigLIP + Visual Cluster CV
**Results:**
- GradientBoosting: CV 0.701
- HistGradientBoosting: CV 0.689
- LGBM: CV 0.689
- CatBoost: CV 0.685
- **Ensemble: CV 0.710**
- **LB**: **0.60** ❌
- **LB-CV Gap**: **-0.110**

#### EXP095-child003: 公開ノート完全再現 ⭐ Best Gap
**Configuration:**
- SigLIP (google/siglip-so400m-patch14-384)
- 公開ノートのCSV（埋め込み + fold事前計算済み）をそのまま使用

**Results:**
- CatBoost: CV 0.717 → Processed 0.719
- LGBM: CV 0.701 → Processed 0.718
- **LB**: **0.65**
- **LB-CV Gap**: **-0.069** ⭐

**Key Observations:**
- ✅ 公開ノートのLB 0.66には届かないがGapは良好
- ⚠️ Processed CVはLGBMで+0.018と大きな改善（後処理の効果）

#### EXP095-child007: SigLIP2-giant ❌
**Configuration:**
- SigLIP2-giant-opt-patch16-384（大型モデル）

**Results:**
- CatBoost: CV 0.715 → Processed 0.715
- LGBM: CV 0.711 → Processed 0.717
- **LB**: **0.60** ❌
- **LB-CV Gap**: **-0.117**

**Key Observations:**
- ❌ 大型モデルでも改善せず、むしろGap悪化

#### EXP095-child010: SigLIP + date_cluster CV ⭐ Best Gap (overall)
**Results:**
- **Ensemble: CV 0.707 → Processed 0.708**
- **LB**: **0.65**
- **LB-CV Gap**: **-0.058** ⭐⭐ **EXP095内で最良**

#### EXP095-child011: date_cluster改良版
**Results:**
- **Ensemble: CV 0.712 → Processed 0.713**
- **LB**: **0.64**
- **LB-CV Gap**: **-0.073**

**Key Observations:**
- ⚠️ CVは上がったがLBは下がった（過学習の兆候）

---

**EXP095 総括:**

| Child | モデル/CV戦略 | Best CV | LB | Gap | 評価 |
|-------|--------------|---------|-----|------|------|
| 000 | SigLIP + Visual Cluster | 0.710 | 0.60 | -0.110 | ❌ |
| **003** | SigLIP + 公開ノート再現 | 0.719 | **0.65** | **-0.069** | △ |
| 007 | SigLIP2-giant | 0.717 | 0.60 | -0.117 | ❌ |
| **010** | SigLIP + date_cluster | 0.708 | **0.65** | **-0.058** | △ |
| 011 | date_cluster改良 | 0.713 | 0.64 | -0.073 | △ |

**教訓:**
- ❌ **GBDT << Neural Net**: Best LB 0.65 << EXP060 (LB 0.69)、**-0.04の差**
- ❌ **SigLIP2-giant効果なし**: 大型化しても改善せず (LB 0.60)
- ✅ **date_cluster CVが最良Gap**: -0.058 (child010) vs -0.110 (child000)
- ⚠️ **公開ノート再現困難**: LB 0.65 vs 公開ノート LB 0.66

**Vision Encoder + GBDT vs Neural Net 比較:**
| アプローチ | Best LB | Best Gap |
|-----------|---------|----------|
| **EVA-CLIP + Neural Net (EXP060)** | **0.69** ⭐ | -0.069 |
| SigLIP + GBDT (EXP095-003) | 0.65 | -0.069 |
| SigLIP + GBDT (EXP095-010) | 0.65 | **-0.058** ⭐ |

**結論**: GBDTアプローチはNeural Netに**LB -0.04**劣る。アンサンブル要員としての価値はあるが、単体では採用しない。

---

### EXP106: Persistent Optimizer (no momentum reset on unfreeze)
**Model**: DinoV3 ViT-Huge+ (timm) + Persistent Optimizer

**目的:**
- EXP060の学習ではunfreeze時にoptimizerを再初期化していた
- これにより、Stage1で蓄積したmomentumが失われていた
- Persistent Optimizerにより、unfreeze時もmomentum/bufferを維持することで学習効率を改善

**アーキテクチャ:**
- Backbone: DinoV3 ViT-Huge+ (timm)
- 5-head独立予測 (Total, GDM, Green, Dead, Clover)
- Consistency Loss (raw空間)
- CV: 5-fold StratifiedGroupKFold (State stratify, Sampling_Date group)

**変更点:**
- Stage1 → Stage2 (unfreeze) 時にoptimizer再初期化しない
- AdamWのmomentum (m, v) bufferを維持
- learning_rateのみ変更 (head: 1e-4 → 1e-5, backbone: 0 → 2e-5)

#### EXP106-child000: Persistent Optimizer
**Configuration:**
- img_size: 256
- batch_size: 8
- epochs: 40 (freeze 10 + unfreeze 30)
- LR: head 1e-4→1e-5, backbone 0→2e-5
- Optimizer: AdamW (momentum persistent)

**Results:**
- **OOF CV**: 0.786 (Fold R²: 0.686, 0.790, 0.856, 0.831, 0.793)
- **LB**: **0.70** △
- **LB-CV Gap**: **-0.086**

**State-level wR²:**
| State | wR² | n |
|-------|-----|---|
| Tas | 0.746 | 138 |
| NSW | 0.716 | 75 |
| WA | 0.820 | 32 |
| Vic | 0.806 | 112 |

**Key Observations:**
- △ **LB 0.70**: EXP060-child038 (LB 0.71) より**-0.01**
- ⚠️ **CV 0.786でLB 0.70**: exp038 (CV 0.776, LB 0.71) より**CVは高いがLBは低い**
- ⚠️ **Gap -0.086**: exp038 (Gap -0.066) より**-0.02悪化**
- ❌ **Persistent Optimizerは逆効果**: momentum維持がoverfittingを促進した可能性

**Lesson Learned:**
- ❌ **Optimizer再初期化が実は重要**: unfreeze時の「リセット」が汎化に寄与していた
- ⚠️ **CV高≠LB高**: 再度確認、CVの改善がLBの改善を保証しない
- 💡 **EXP060-child038/058のoptimizer再初期化はbestのまま**

---

### EXP107: DINOv2 Two-Stream Architecture ❌ **CV/LB Gap最大**
**Model**: DINOv2-Large + Two-Stream (左右分割) + Triple Pooling (CLS + avg + max)

**目的:**
- 公開ノートブック "Dinov2-2stream-LB067-single" (LB 0.67) のアプローチを実装
- Triple Pooling (CLS + avg_pool + max_pool) が+0.02改善との報告を検証
- EXP060の損失関数・評価関数との組み合わせ効果を検証

**アーキテクチャ:**
- Backbone: DINOv2-Large (facebook/dinov2-large)
- Input: 2000x1000を左右分割 → 448x448にリサイズ
- Triple Pooling: CLS token + avg_pool(patches) + max_pool(patches) → 3×1024 = 3072 dim per stream
- Combined: 3072×2 = 6144 dim
- 5 独立MLP heads (Total, GDM, Green, Dead, Clover)
- Consistency Loss (raw空間)
- CV: 5-fold StratifiedGroupKFold (clover_dead_presence stratify, day_month_state group)

**訓練設定:**
- img_size: 448
- batch_size: 4
- epochs: 40 (freeze 5 + unfreeze 35)
- LR: head 1e-4 → backbone 2e-5
- Gradual unfreeze: 各epoch 10%ずつ

#### EXP107-child000: DINOv2 Two-Stream Baseline
**Results:**
- **Fold R²**: [0.846, 0.769, 0.739, 0.789, 0.741]
- **Mean R²**: 0.777 ± 0.039
- **OOF CV**: **0.7885**
- **LB**: **0.67** ❌
- **LB-CV Gap**: **-0.1185** ❌❌ **← 全実験中でWorst Gap**

**State-level wR²:**
| State | wR² | n |
|-------|-----|---|
| NSW | 0.7581 | 75 |
| Vic | 0.7787 | 112 |
| Tas | 0.7166 | 138 |
| WA | 0.7663 | 32 |

**Key Observations:**
- ❌❌ **LB-CV Gap -0.1185 = 全実験中で最悪**: 通常 -0.07〜-0.09 が -0.12 に悪化
- ❌ **LB 0.67**: 公開ノートと同等だが、EXP060 (LB 0.73) から**-0.06**
- ⚠️ **Tas州でR²最低 (0.7166)**: DINOv2がTas州の画像に弱い
- ⚠️ **Triple Poolingの効果不明**: 他条件が異なるため単純比較不可

**失敗の原因分析:**
1. **DINOv2 vs EVA-CLIP**: DINOv2は自己教師あり学習で、EVA-CLIPより汎化性能が低い可能性
2. **過学習**: 448×448の高解像度 + 6144 dim の高次元特徴量が過学習を促進
3. **Two-Stream分離**: 左右の情報が分離されることで、全体の文脈情報が失われる可能性

**Lesson Learned:**
- ❌ **DINOv2はこのタスクに不向き**: EVA-CLIPより汎化性能が大幅に劣る
- ❌ **Triple Poolingは過学習を促進**: 高次元特徴量がtrain分布に過適合
- ⭐ **EXP060 (EVA-CLIP + 5-head) がベスト**を再確認

---

### EXP108: State Soft Label (Auxiliary Task) △ **CV高だがGap大**
**Model**: DinoV3 ViT-Huge+ + State Soft Label Auxiliary Task

**目的:**
- State情報をsoft labelとして補助タスクに追加
- State分布の違いを学習に組み込むことでCV/LB Gapを改善

**アーキテクチャ:**
- Backbone: DinoV3 ViT-Huge+ (timm)
- 5-head独立予測 (Total, GDM, Green, Dead, Clover)
- **Auxiliary Task**: State分類 (soft label)
- Consistency Loss (raw空間)

#### EXP108-child000: State Soft Label
**Results:**
- **Fold R²**: [0.870, 0.761, 0.743, 0.827, 0.765]
- **Mean R²**: 0.793 ± 0.048
- **OOF CV**: **0.8024** ← 全実験中でTop3
- **LB**: **0.72** △
- **LB-CV Gap**: **-0.0824**

**State-level wR²:**
| State | wR² | n |
|-------|-----|---|
| NSW | 0.7446 | 75 |
| Vic | 0.8070 | 112 |
| Tas | 0.7674 | 138 |
| WA | 0.8302 | 32 |

**Key Observations:**
- △ **LB 0.72**: EXP060-child058 (LB 0.73) から**-0.01**
- ✅ **OOF CV 0.8024**: 非常に高いCV (ただしGap -0.08)
- ✅ **WA州でR²最高 (0.8302)**: State補助タスクがWA州に効果的
- ⚠️ **NSW州でR²最低 (0.7446)**: NSW州の改善が課題

**EXP060-child058との比較:**
| 指標 | EXP060-058 | EXP108 | 差分 |
|------|------------|--------|------|
| OOF CV | 0.799 | 0.802 | +0.003 |
| LB | 0.73 | 0.72 | **-0.01** |
| Gap | -0.069 | -0.082 | -0.013 |

**Lesson Learned:**
- ⚠️ **State補助タスクはCV上昇に貢献するがLBは改善しない**
- ⚠️ **Gap -0.08**: State情報がtrain分布に偏っている可能性
- 💡 **EXP060-child058 (LB 0.73)** がベストのまま

---

### EXP060-child061: CutMix Augmentation △ **LB 0.72維持、Gap最小級**
**Model**: EVA-CLIP + 5-head + CutMix Augmentation

**目的:**
- CutMix augmentationによる汎化性能向上
- 異なる画像の領域を混合することで、ローカル特徴への依存を減少

**アーキテクチャ:**
- Base: EXP060-child000と同一 (EVA-CLIP + 5-head + Consistency Loss)
- **追加**: CutMix augmentation (p=0.5, alpha=1.0)

#### EXP060-child061: CutMix
**Results:**
- **Fold R²**: [0.847, 0.766, 0.743, 0.831, 0.755]
- **Mean R²**: 0.789 ± 0.042
- **OOF CV**: **0.7972**
- **LB**: **0.72** △
- **LB-CV Gap**: **-0.0772** ⭐ **← 良好なGap**

**State-level wR²:**
| State | wR² | n |
|-------|-----|---|
| NSW | 0.7272 | 75 |
| Vic | 0.8017 | 112 |
| Tas | 0.7814 | 138 |
| WA | 0.8294 | 32 |

**Key Observations:**
- △ **LB 0.72**: EXP060-child058 (LB 0.73) から**-0.01**
- ✅ **Gap -0.077**: EXP060-child058 (Gap -0.069) に次ぐ良好なGap
- ✅ **Tas州でR² 0.7814**: EXP107 (0.7166) より大幅改善
- ⚠️ **NSW州でR²最低 (0.7272)**: NSW州の改善が課題

**EXP060-child058との比較:**
| 指標 | EXP060-058 | EXP060-061 | 差分 |
|------|------------|------------|------|
| OOF CV | 0.799 | 0.797 | -0.002 |
| LB | 0.73 | 0.72 | **-0.01** |
| Gap | -0.069 | -0.077 | -0.008 |

**CutMixの効果分析:**
- ✅ **汎化に貢献**: Gap -0.077 は比較的良好
- ⚠️ **LB改善には至らず**: CutMixの正則化効果はtrain CVを下げる傾向
- 💡 **EXP060-child058 (LB 0.73)** がベストのまま

---

### 3実験（EXP107, EXP108, EXP060-061）の比較まとめ

| 実験 | モデル/手法 | OOF CV | LB | Gap | 評価 |
|------|------------|--------|-----|------|------|
| **EXP060-058** (ベスト) | DinoV3 ViT-Huge+ + Artem CV | 0.799 | **0.73** ⭐ | **-0.069** ⭐ | ⭐⭐ |
| EXP060-061 | + CutMix | 0.797 | 0.72 | -0.077 | △ |
| EXP108 | DinoV3 + State soft | 0.802 | 0.72 | -0.082 | △ |
| EXP107 | DINOv2 Two-Stream | 0.789 | **0.67** ❌ | **-0.119** ❌❌ | ❌ |

**State別R²比較:**
| State | EXP107 | EXP108 | EXP060-061 | n |
|-------|--------|--------|------------|---|
| NSW | 0.758 | 0.745 | 0.727 | 75 |
| Vic | 0.779 | **0.807** | 0.802 | 112 |
| Tas | 0.717 | 0.767 | **0.781** | 138 |
| WA | 0.766 | **0.830** | 0.829 | 32 |

**Key Insights:**
1. **DINOv2はこのタスクに不向き**: EXP107のGap -0.119は全実験中で最悪
2. **DinoV3 + Artem CVが最強（Public）**: EXP060-child058がLB/Gapともに最良
3. **CutMixは汎化に貢献**: Gapは良好だがLB改善には至らず
4. **State補助タスクはCVを上げるがLBを上げない**: 過学習リスク

**結論:**
- ⭐ **EXP060-child058 (DinoV3 ViT-Huge+ + Artem CV, LB 0.73) がCURRENT BEST**を維持
- DINOv2系、State補助タスク、CutMixはいずれもLB改善に至らず

---

### EXP060-child066〜069: 追加検証（Artem CV派生）

| Exp | 変更点 | OOF CV | LB | Gap | メモ |
|-----|--------|--------|----|-----|------|
| EXP060-066 | DINOv2 Giant 518px | 0.773 | 0.67 | -0.103 | Tas_Spring / WA_Winter が弱い |
| EXP060-067 | pred loss = MSE | 0.797 | 0.71 | -0.087 | CV↑だがGap拡大でLB伸びず |
| EXP060-068 | sum-loss | 0.801 | 0.71 | -0.091 | Tas_SpringやNSW_Summerが伸びず |
| EXP060-069 | Soft-Species Conditioning | 0.791* | 0.71 | -0.081 | *OOF未保存（mean fold） |

**メモ**
- 共通傾向: **CVが上がってもGapが大きいとLBが伸びない**（exp058はGap最小）。
- slice相関（更新済み）: `output/CV_LB/STATE_SEASON/STATE_SEASON_ANALYSIS_REPORT.md` を参照。

---

*Last Updated: 2026-01-14*
*Total Experiment dirs: 108+ (EXP000 through EXP108)*
*Key milestones: EXP030 (LB 0.68), EXP036 (LB 0.69), EXP060-child038 (LB 0.71), **EXP060-child058 (LB 0.73, CURRENT BEST)**, EXP060-child059 (768px overfits, LB 0.70), EXP060-child066 (DINOv2 Giant 518, LB 0.67), EXP106 (Persistent Optimizer, LB 0.70), EXP107 (DINOv2 Two-Stream, LB 0.67, worst gap), EXP108 (State soft label, LB 0.72)*
