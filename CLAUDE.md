# CLAUDE.md　ういー

このファイルは、リポジトリ内のコードを扱う際の Claude Code (claude.ai/code) へのガイダンスを提供します。

## プロジェクト概要

これは **BirdCLEF+ 2026** の Kaggle コンペティションプロジェクトです。連続音声データの中から、野生生物の鳴き声に基づいて種を自動的に識別することが目的です：

## 開発ワークフロー

**学習環境**: Kaggle Notebooks
- Claude Code はローカルで学習スクリプトを**実行しないこと**
- 学習は Kaggle Notebooks 上の `execute_train.ipynb` で実行
- データは `/kaggle/input/competitions/birdclef-2026` で利用可能
- スクリプト（`train.py`、設定ファイル）は Kaggle Dataset `csiro-biomass-scripts` から読み込み
- 学習済みモデルは `kagglehub` 経由で Kaggle Models にアップロード

**推論環境**: Kaggle Notebooks
- 推論スクリプトは Kaggle の提出環境で動作するよう設計
- Kaggle の制約（実行時間 ≤9時間、インターネット不可）を遵守すること

**Config の `data_root`**: 常に `/kaggle/input/csiro-biomass` を使用（`/content/...` は不可）

## 実験結果の管理

**重要: 実験後は必ず EXP/EXP_SUMMARY.md を更新すること**

### ユーザーが LB（リーダーボード）結果を報告してきたら:
1. **すぐに `EXP/EXP_SUMMARY.md` を読んで**現在の実験状況を把握する
2. **EXP_SUMMARY.md を更新する**（新しい LB 結果を追記）:
   - 実験の Results セクションに LB スコアを追加
   - LB-CV ギャップを更新
   - 結果についての考察を追加（改善/悪化）
   - モデル比較表を更新
   - 新ベストであれば Competition Status セクションを更新
3. 必要に応じて**確認質問をする**（例: フォールド固有の問題、学習途中終了など）

### ユーザーが新しいアイデアや推薦を求めてきたら:
1. **まず `EXP/EXP_SUMMARY.md` を読む**:
   - これまでに試したこと
   - うまくいったこと・うまくいかなかったこと
   - 現在の最良アプローチ
   - 得られた教訓
   - 特定された問題（フォールド不均衡、過学習パターンなど）
2. **その後、以下を踏まえたアイデアを提案する**:
   - 成功したアプローチを発展させる
   - 失敗した実験を繰り返さない
   - 過去の実験から判明した問題に対処する
   - 実験履歴に基づいた提案

### ドキュメント基準:
- **全ての実験**（成功・失敗問わず）は EXP_SUMMARY.md に記録すること
- **失敗/中断した実験**には以下を含める:
  - なぜ中断したか（例: 学習が遅すぎる、CV が改善しない）
  - 得られた知見
  - 今後の実験への提案
- **成功した実験**には以下を含める:
  - 完全な結果（CV、LB、フォールドスコア）
  - 主要な観察と洞察
  - 過去の実験との比較

## リポジトリ構成

```
csiro-biomass/
├── docs/
│   ├── OVERVIEW.md          # コンペティションの説明とルール
│   └── DATASET.md           # データセットの説明とファイル形式
├── input/
│   └── csiro-biomass/       # データセットファイル（train.csv, test.csv, 画像）
├── EXP/
│   ├── EXP000/              # ベースライン実験
│   │   ├── train.py         # 学習スクリプト（Colab で実行）
│   │   ├── infer.py         # 推論スクリプト（Kaggle で実行）
│   │   └── config/
│   │       ├── child-exp000.yaml  # 小規模実験のパラメータバリエーション
│   │       └── child-exp001.yaml  # 各 child-exp は同じ train.py を使用
│   ├── EXP001/              # 主要なアーキテクチャ変更
│   │   ├── train.py
│   │   └── infer.py
│   └── EXP{exp_no}/         # 追加実験
│       ├── train.py         # 必須: 学習スクリプト
│       └── infer.py         # 必須: 推論スクリプト
├── notebooks/               # （レガシー）旧ノートブックベースの実験
└── scripts/                 # ユーティリティスクリプト
```

## 実験管理

### 主要実験（スクリプト変更あり）

**コードの変更**（アーキテクチャ、データ処理、損失関数）を伴う場合:
1. 新しい実験ディレクトリを作成: `EXP/EXP{exp_no}/`
2. 必ず `train.py` と `infer.py` の両方を含める
3. `exp_no` は連番でインクリメント（EXP000, EXP001, EXP002, ...）

**新しい EXP{exp_no} が必要な例:**
- モデルアーキテクチャの変更（例: two-stream → single-stream）
- データ拡張パイプラインの変更
- 新しい損失関数の実装
- 学習戦略の変更（例: 異なる k-fold アプローチ）

### 小規模実験（パラメータ変更のみ）

**ハイパーパラメータの調整のみ**の場合:
1. 同じ `EXP{exp_no}/train.py` を使い続ける
2. 新しい設定ファイルを作成: `EXP/EXP{exp_no}/config/child-exp{child_no}.yaml`
3. `child_no` は連番でインクリメント（000, 001, 002, ...）

**child-exp が適切な例:**
- 学習率の変更
- バッチサイズの調整
- エポック数の変更
- データ拡張の確率
- 損失関数の重み
- モデルのハイパーパラメータ（dropout、隠れ層の次元数）

## 重要: 後方互換性

**新しい機能を既存のtrain.pyに追加する際は、必ず後方互換性を維持すること。**

### ルール:
1. **新しいconfig項目はデフォルト値を持つこと**
   - 例: `depth_enabled: false` がデフォルト
   - 古いconfigファイルでも動作するように

2. **Datasetの戻り値を変更する場合**
   - 新しい要素を追加する場合、古いconfigでも動作するようにする
   - `None`を返すとDataLoaderがエラーになる → ダミーtensorを返すか、条件分岐で要素数を変える
   - 例: depth機能追加時、`depth_enabled=false`でもエラーにならないように

3. **モデルのforward引数を変更する場合**
   - 新しい引数にはデフォルト値を設定: `def forward(self, x, depth=None)`
   - 古いコードからの呼び出しでもエラーにならないように

4. **Training loopの変更**
   - 新機能のon/offをconfigで制御できるようにする
   - `if config.get('new_feature_enabled', False):` のようにデフォルトで無効

### 悪い例（後方互換性なし）:
```python
# Dataset: 常に5要素を返す、depth=Noneの場合エラー
return image, targets, path, depth_tensor  # depth_tensor=None → DataLoaderエラー
```

### 良い例（後方互換性あり）:
```python
# Dataset: depth無効時はダミーtensorを返す
if self.depth_dir is not None:
    depth_tensor = load_depth(...)
else:
    depth_tensor = torch.zeros(1, 32, 32)  # ダミー
return image, targets, path, depth_tensor
```

## 主要なアーキテクチャ設計

### モデルアーキテクチャ: Two-Stream・マルチヘッド CNN

高解像度画像を処理するための高度なアーキテクチャを採用:

1. **Two-Stream 処理**:
   - 入力画像は 2000×1000 ピクセル
   - 各画像を左（1000×1000）と右（1000×1000）のパッチに分割
   - 各パッチを 768×768 にリサイズして細部（クローバーの葉など）を保持
   - 両ストリームは同じバックボーンの重みを共有

2. **共有バックボーン**:
   - `timm` ライブラリを使用（例: `convnext_tiny`）、ImageNet 事前学習済み重み
   - 左右両パッチが同じバックボーンを通過

3. **特徴融合**:
   - 左右ストリームの特徴量を連結

4. **マルチヘッド出力**:
   - 3つの専用 MLP ヘッドが最重要ターゲットを予測:
     - `head_total` → Dry_Total_g（重み 50%）
     - `head_gdm` → GDM_g（重み 20%）
     - `head_green` → Dry_Green_g（重み 10%）
   - 残り2つのターゲット（Dry_Dead_g、Dry_Clover_g）は推論時に線形関係から**計算**:
     - `Dry_Clover_g = max(0, GDM_g - Dry_Green_g)`
     - `Dry_Dead_g = max(0, Dry_Total_g - GDM_g)`

### 学習戦略

**二段階ファインチューニング**（フリーズ/アンフリーズ）:
- **ステージ 1（エポック 1-5）**: バックボーンを凍結し、3つの MLP ヘッドのみを LR=1e-4 で学習
- **ステージ 2（エポック 6-20）**: バックボーンを解凍し、LR=1e-5 でモデル全体をファインチューニング

**交差検証**:
- `StratifiedKFold`（`Dry_Total_g` のビンで層化）による 5 分割交差検証
- データ漏洩を防ぐため `Sampling_Date` でグループ化した `GroupKFold` を使用
- 全 5 フォールドのモデルを推論時にアンサンブル

**データ拡張**（左右パッチそれぞれに独立して適用）:
- HorizontalFlip（p=0.5）
- VerticalFlip（p=0.5）
- RandomRotate90（p=0.5）
- ColorJitter（brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1）

**損失関数**:
- 安定性のための加重 SmoothL1Loss（Huber Loss）
- Loss = 0.5×Loss_Total + 0.2×Loss_GDM + 0.1×Loss_Green

### 推論戦略

**テスト時データ拡張（TTA）**:
- 3つのビュー: オリジナル、水平反転、垂直反転
- 全ビューの予測を平均化

**アンサンブル**:
- 全 5 フォールドのモデルの予測を平均化
- 最終予測 = （5 フォールド × 3 TTA ビュー）= 15 予測値の平均

## ノートブックの使い方

### メイン学習ノートブック

**場所**: `notebooks/HIGH_SCORE_Notebook_20251108/csiro.ipynb`

**主要コンポーネント**（順番通り）:
1. **CFG クラス** - 全ハイパーパラメータの中央設定
2. **データ読み込み** - train.csv をロング形式からワイド形式（1行=1画像）にピボット
3. **BiomassDataset** - two-stream 処理用カスタム Dataset クラス
4. **BiomassModel** - モデルアーキテクチャ（two-stream、マルチヘッド）
5. **WeightedBiomassLoss** - カスタム損失関数
6. **K-Fold 設定** - 層化を用いたフォールド割り当て作成
7. **学習ループ** - フリーズ/アンフリーズによる二段階学習
8. **モデルチェックポイント** - 損失ではなく検証 R² に基づいて最良モデルを保存

**重要**: モデル保存は検証セットでの最低損失ではなく、最高 **R² スコア** を使用します。

### 推論ノートブック

**場所**: `notebooks/HIGH_SCORE_Notebook_20251108/lb-0-57-infer-model-code.ipynb`

Kaggle にスコアリング用として提出するノートブック。以下を実行します:
1. 全 5 つの学習済みモデルを読み込み（best_model_fold0.pth ～ fold4.pth）
2. 3 ビューで TTA を適用
3. フォールドと TTA ビューをまたいで予測をアンサンブル
4. 派生ターゲットを計算（Dry_Dead_g、Dry_Clover_g）
5. 予測を submission.csv にフォーマット

## データ形式のメモ

### train.csv（ロング形式）
- 1785 行（357 画像 × 5 ターゲット）
- 主要列: `image_path`、`target_name`、`target`
- 追加特徴量: `Sampling_Date`、`State`、`Species`、`Pre_GSHH_NDVI`、`Height_Ave_cm`
- 学習用にワイド形式（1行=1画像）にピボットが必要

### test.csv（ロング形式）
- テスト画像ごとに 5 行（各ターゲット 1 行）
- 列: `sample_id`、`image_path`、`target_name`
- sample_id の形式: `{image_id}__{target_name}`（例: "ID1001187975__Dry_Green_g"）

### submission.csv
- `sample_id`、`target` の 2 列が必須
- 各（画像、ターゲット）ペアにつき 1 行
- テスト画像ごとに 5 行

## 重要な設定パラメータ

コードを変更する際、学習と推論で以下のパラメータが一致していることを確認すること:

- `MODEL_NAME`: バックボーンアーキテクチャ（デフォルト: 'convnext_tiny'）
- `IMG_SIZE`: リサイズ寸法（デフォルト: 768）
- `TARGET_COLS`: モデルが予測する 3 ターゲット
- `ALL_TARGET_COLS`: 提出用の全 5 ターゲット（正しい順序で）
- モデルアーキテクチャ（ヘッド数、隠れ層の次元数、dropout 率）

## よくある落とし穴

1. **データ漏洩**: 同じ日付の類似画像が学習と検証の両方に含まれないよう、常に `Sampling_Date` で `GroupKFold` を使用する
2. **モデル読み込み**: 保存されたモデルが `nn.DataParallel` でラップされている場合がある。読み込み時にアンラップのコードを使用:
   ```python
   state_dict = torch.load(model_path)
   new_state_dict = OrderedDict()
   for k, v in state_dict.items():
       name = k.replace('module.', '')
       new_state_dict[name] = v
   model.load_state_dict(new_state_dict)
   ```
3. **負の予測値**: 派生ターゲット計算時は常に `np.maximum(0, ...)` を使用して負のバイオマス値を防ぐ
4. **変換の一貫性**: TTA 変換は左右両パッチに一貫して適用すること
5. **重要 - infer.py の変数名バグ**: 実験コードをコピーする際は、クリーンアップコードが正しい変数名を使用しているか必ず確認:
   - コードが `test_loaders_all = []` を作成している場合、クリーンアップは `del test_loaders_all` を使用
   - コードが `test_loaders = []` を作成している場合、クリーンアップは `del test_loaders` を使用
   - **よくあるエラー**: `UnboundLocalError: cannot access local variable 'test_loaders' where it is not associated with a value`
   - **解決策**: `del` 文の変数名をコード内で実際に使用している変数名と一致させる
   - このバグは EXP014、EXP027、EXP028 で発生しており、新しい実験作成時に確認必須

## 依存ライブラリ

使用する主要ライブラリ:
- `torch` - 深層学習用 PyTorch
- `timm` - 事前学習済みビジョンモデル
- `albumentations` - 画像拡張
- `cv2`（OpenCV）- 画像読み込み
- `pandas`、`numpy` - データ操作
- `scikit-learn` - 交差検証とメトリクス
- `tqdm` - プログレスバー

## コンペティションの制約

- コードは Kaggle Notebooks で実行可能であること（CPU ≤9時間 または GPU ≤9時間）
- 提出時はインターネットアクセス不可
- 事前学習済みモデルは使用可能
- 出力ファイル名は `submission.csv` であること

### 重要: テストデータの制約

**テスト画像はKaggle環境で見ることができません。**

- Kaggle推論環境では `test.csv`（メタデータのみ）と `test/` フォルダ（画像）にアクセス可能
- しかし、**テスト画像の中身を事前に確認する方法はない**
- したがって、以下のアプローチは**不可能**:
  - **Pseudo Labeling（オフライン版）**: テスト画像を事前にダウンロードして予測→再学習
  - **テスト画像の分析**: テスト画像の分布を事前に把握して学習データに反映
  - **テスト画像ベースのaugmentation調整**: テスト画像の特性に合わせた前処理

- 可能なのは:
  - Kaggle推論ノートブック内での**オンラインPseudo Labeling**（9時間制限内でself-training）
  - `test.csv`のメタデータ（`image_path`のみ）の分析

**この制約を忘れて「テスト画像を使って〜」という提案をしないこと。**

## OOF 誤差分析

**目的**: Out-of-Fold予測の誤差を分析し、モデルの弱点と改善方向を特定する

### 分析の実行方法

1. **OOF予測ファイルの準備**
   - `EXP/{exp_name}/outputs/{child_exp}/oof_predictions.csv`
   - 必要な列: `image_path`, `fold`, `{target}_pred`, `{target}_true` (5ターゲット分)

2. **基本スクリプトの実行**
   ```bash
   python scripts/analyze_oof.py --exp_name EXP036 --child_exp child-exp000 --analyze_patterns --top_n 15
   ```

3. **出力先**: `EXP/{exp_name}/outputs/{child_exp}/oof-analysis/`

### 分析の観点と目的

#### 1. 評価関数への寄与度分析（最重要）
**目的**: どのターゲットのどのサンプルがスコアを最も下げているか特定

**分析内容**:
- SS_res（残差二乗和）のターゲット別内訳
- 各サンプルの weighted contribution 計算
- Top N ワーストサンプルの特定

**なぜ重要か**:
- Total(50%)の誤差はGreen(10%)の5倍のインパクト
- 少数のワーストサンプルがスコアを大きく下げている可能性
- 「Top 20を修正すれば R² +0.10」のような改善ポテンシャルを定量化

#### 2. 過小評価 vs 過大評価の分析
**目的**: 系統的なバイアスの有無を確認

**分析内容**:
- 各ターゲットで Underestimation / Overestimation のサンプル数と SS_res 比率
- 平均的な過小/過大評価量

**なぜ重要か**:
- 系統的バイアスは後処理（スケーリング）で補正可能
- バイアスの原因究明（データ分布、損失関数の非対称性など）

#### 3. メタデータとの関連分析
**目的**: どのような画像で予測が困難か特定

**分析対象のメタデータ**:
- `Species`: 草種による予測難易度の差
- `Height_Ave_cm`: 草丈との相関（**特に重要**）
- `State`: 地域差
- `Pre_GSHH_NDVI`: 衛星由来のNDVI

**なぜ重要か**:
- 特定の草種（Fescue, Lucerne等）で予測が困難な場合、専用処理が必要
- Height との相関が強い場合、補助入力として追加すべき

#### 4. Height（草丈）と予測誤差の相関分析（特に重要）
**目的**: 「草丈が高いほど過小評価」の仮説を検証

**分析内容**:
- Pearson相関係数（Height vs Error）
- Height bin別の平均誤差
- 回帰分析: `Error ≈ a + b × Height`

**なぜ重要か**:
- 画像は上から撮影 → 草丈5cmと30cmが同じ「緑ピクセル」に見える
- しかし実際のバイオマスは草丈に比例して増加
- Height情報をモデルに追加することで改善可能

**典型的な発見パターン**:
```
Green: r = -0.65 (強い負の相関 → 草丈高で過小評価)
Clover: r = +0.54 (正の相関 → 草丈高で過大評価)
Dead: r ≈ 0 (相関なし → 緑の下に隠れて見えない)
```

#### 5. 物理的整合性の確認
**目的**: 予測が物理法則を満たしているか確認

**制約条件**:
- `Total = Dead + GDM`
- `GDM = Green + Clover`

**なぜ重要か**:
- 5ヘッドモデルでは個別予測のため整合性が崩れやすい
- ConsistencyLossの強化や後処理での調整が必要か判断

### 分析結果の活用

| 発見 | 改善提案 |
|------|----------|
| 高バイオマスで過小評価 | サンプル重み付け、オーバーサンプリング |
| 特定Species（Fescue等）で高エラー | Species情報を補助入力に追加 |
| Height と強い負の相関 | Height情報をモデル入力に追加 |
| 系統的過小評価バイアス | 推論時スケーリング補正 |
| 物理的整合性の崩れ | ConsistencyLoss強化 |

### 出力ファイル

| ファイル | 内容 |
|----------|------|
| `ERROR_ANALYSIS_REPORT.md` | 分析レポート（日本語） |
| `weighted_contribution_analysis.csv` | サンプルごとの寄与度 |
| `height_vs_error_analysis.png` | 草丈 vs 誤差の可視化 |
| `error_analysis_summary.png` | サマリ図 |
| `summary.txt` | 詳細ログ |

## CV vs LB 相関分析

**目的**: CV (OOF R²) と Public LB の相関を分析し、CVが信頼できる指標かを確認する

### ユーザーから「CV LBの分析をして」と依頼されたら

以下のコマンドを実行する:

```bash
python scripts/analyze_cv_lb.py
```

このスクリプトは自動的に:
1. EXP060/EXP113の`results.json`からLBスコアを収集
2. OOF予測ファイルからState_Season別R²を計算
3. CV vs LB の相関分析 (Pearson相関)
4. State_Season vs LB の相関分析
5. 散布図を生成 (`output/CV_LB/STATE_SEASON/`)

実行後、結果をユーザーに報告する。

### 出力ファイル

| ファイル | 内容 |
|----------|------|
| `output/CV_LB/STATE_SEASON/all_exp_cv_lb_scatter.png` | CV vs LB 散布図 |
| `output/CV_LB/STATE_SEASON/all_exp_state_season_scatter.png` | State_Season別散布図 |
| `output/CV_LB/STATE_SEASON/all_exp_state_season_correlations.csv` | 相関係数データ |
| `output/CV_LB/STATE_SEASON/STATE_SEASON_ANALYSIS_REPORT.md` | 詳細レポート |

### 主要な発見 (2026-01-26時点, n=49実験)

#### 1. CV と LB は強く相関している

```
全体相関: r = +0.82 (p < 0.001)
→ CVを上げれば基本的にLBも上がる
```

#### 2. State_Season 別相関 (全て正の相関)

| State_Season | 相関係数 | 解釈 |
|--------------|----------|------|
| NSW_Summer | +0.72 | 強い正の相関 |
| NSW_Autumn | +0.64 | 正の相関 |
| Tas_Autumn | +0.62 | 正の相関 |
| Tas_Spring | +0.61 | 正の相関 |

#### 3. EXP113単体での負の相関は無視してOK

- EXP113のみ (n=12): r = -0.32 (非有意)
- これは狭いLBレンジ (0.72-0.75) でのノイズ
- 全体で見れば正の相関

### 重要: Hand Labeling（ノイズクリーニング）は失敗

**検証結果 (child-exp022 vs child-exp012)**:

| 実験 | 内容 | OOF R² | LB |
|------|------|--------|-----|
| child-exp012 | Base (クリーニングなし) | 0.809 | **0.75** |
| child-exp022 2seed | Hand Cleaning適用 | 0.814 | **0.74** |

```
Hand Cleaning の効果:
  CV:  +0.006 (改善)
  LB:  -0.01  (悪化)
```

**教訓**: Hand Labelingは「CVを上げてLBを下げる」典型的な過学習パターン
- 今後の実験では**クリーニングなしのデータ**を使用すべき
- CV上で「外れ値」に見えるサンプルがテストには有用な可能性

### Public/Private Split について

- Host論文によると、データ収集期間は2014-2017年
- Train: 2015年, Public: 2016年(推定), Private: 2017年(推定)
- テストデータのメタデータ (State, Sampling_Date) は非公開
- **CV と LB を両方見て判断するしかない**

## 重要: アンサンブルに関する注意事項

**LB 0.73はシングルモデルで達成されている。安易なアンサンブル提案は禁止。**

### アンサンブルが効果的でない理由

1. **予測相関が極めて高い**
   - DinoV3 (LB 0.71) vs EVA-CLIP (LB 0.69): Total_g相関 r = 0.93
   - 同一アーキテクチャの異なるパラメータ: r > 0.95

2. **OOFでの改善がLBに転移しにくい**
   - OOF上: DinoV3 0.6 + EVA-CLIP 0.4 で +0.011 改善
   - しかしLBでは改善しないことが多い（過去の経験）

3. **検証結果 (2026-01-08)**
   ```
   DinoV3単体 OOF R²:    0.7758 (LB 0.71)
   EVA-CLIP単体 OOF R²:  0.7589 (LB 0.69)

   アンサンブル (OOF best):
     DinoV3 0.6 + EVA-CLIP 0.4: R² = 0.7869 (+0.011)
     → OOFでは改善するがLBでは効果なし
   ```

### アンサンブルを提案する前に確認すべきこと

- [ ] 予測相関が r < 0.85 かどうか
- [ ] 異なるアーキテクチャ/損失関数/データ拡張を使っているか
- [ ] LBで実際に検証済みかどうか

## 重要: 効果がないことが検証済みの改善策

### 1. Clover非負化 (推論時クリッピング)

```python
clover_pred = np.maximum(0, clover_pred)  # 効果なし
```

**検証結果 (2026-01-08)**:
- 負の予測数: 31個 (min=-0.597)
- 改善効果: **+0.000001** (ほぼゼロ)
- 理由: 負の予測値が小さいため影響なし

### 2. 試行済みで失敗した改善策 (exp058以降)

**Baseline**: exp058 (DinoV3 512px + Artem CV) = **LB 0.73** (現在のベスト)

| 実験 | 変更点 | LB | 結果 |
|------|--------|-----|------|
| exp059 | 768px + Artem CV | 0.70 | ❌ 過学習 |
| exp060 | EVA02-CLIP + Artem CV | - | ❌ CVは良いがLB悪化 |
| ENSEXP_006 | DinoV3 + EVA02-CLIP アンサンブル | 0.71 | ❌ OOF最適化はLBに転移しない |

**結論**: exp058 (DinoV3 512px + Artem CV, LB 0.73) が現在のベスト。
同じアーキテクチャでのハイパーパラメータチューニングやアンサンブルでは改善困難。
