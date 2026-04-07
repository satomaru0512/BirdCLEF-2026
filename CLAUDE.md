# CLAUDE.md　ういー

このファイルは、リポジトリ内のコードを扱う際の Claude Code (claude.ai/code) へのガイダンスを提供します。

## プロジェクト概要

これは **BirdCLEF+ 2026** の Kaggle コンペティションプロジェクトです。連続音声データの中から、野生生物の鳴き声に基づいて種を自動的に識別することが目的です：

## 開発ワークフロー

**学習環境**: Kaggle Notebooks
- Claude Code はローカルで学習スクリプトを**実行しないこと**
- 学習は Kaggle Notebooks 上の `execute_train.ipynb` で実行
- データは `/kaggle/input/competitions/birdclef-2026` で利用可能
- 学習済みモデルは `kagglehub` 経由で Kaggle Models にアップロード

**推論環境**: Kaggle Notebooks
- 推論スクリプトは Kaggle の提出環境で動作するよう設計
- Kaggle の制約（実行時間 ≤9時間、インターネット不可）を遵守すること

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
BirdCLEF+ 2026/
├── CLAUDE.md
├── EXECUTE_TRAIN_README.md
├── execute_train.ipynb          # Kaggle学習実行ノートブック
├── final-inferece.ipynb         # 最終推論ノートブック
├── input/
│   └── birdclef-2026/           # コンペデータ（.gitignore対象）
│       ├── train.csv
│       ├── taxonomy.csv
│       ├── sample_submission.csv
│       ├── recording_location.txt
│       ├── train_audio/         # 学習用音声ファイル
│       ├── train_soundscapes/   # 学習用サウンドスケープ
│       ├── train_soundscapes_labels.csv
│       └── test_soundscapes/    # テスト用サウンドスケープ
├── outputs/
│   └── CV_LB/                   # CV vs LB 分析結果
│       └── STATE_SEASON/
├── docs/
│   ├── OVERVIEW.md              # コンペティションの説明とルール
│   ├── DATASET.md               # データセットの説明とファイル形式
│   ├── papers/
│   │   └── HOSTPAPER.md
│   └── Idea_Research/           # 調査・アイデアメモ
└── EXP/
    ├── EXP_SUMMARY.md           # 全実験の結果サマリ
    └── EXP000/                  # ベースライン実験
        ├── train.py             # 学習スクリプト（Kaggle で実行）
        ├── infer.py             # 推論スクリプト（Kaggle で実行）
        ├── infer_ttt.py         # TTT推論スクリプト
        ├── config/              # ハイパーパラメータ設定ファイル
        └── outputs/
            └── child-exp000/    # 実験出力（OOF予測など）
```

## 実験管理

### 主要実験（スクリプト変更あり）

**コードの変更**（アーキテクチャ、データ処理、損失関数）を伴う場合:
1. 新しい実験ディレクトリを作成: `EXP/EXP{exp_no}/`
2. 必ず `train.py` と `infer.py` の両方を含める
3. `exp_no` は連番でインクリメント（EXP000, EXP001, EXP002, ...）

**新しい EXP{exp_no} が必要な例:**
- モデルアーキテクチャの変更
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
   - 古いconfigファイルでも動作するように

2. **Datasetの戻り値を変更する場合**
   - 新しい要素を追加する場合、古いconfigでも動作するようにする
   - `None`を返すとDataLoaderがエラーになる → ダミーtensorを返すか、条件分岐で要素数を変える

3. **モデルのforward引数を変更する場合**
   - 新しい引数にはデフォルト値を設定
   - 古いコードからの呼び出しでもエラーにならないように

4. **Training loopの変更**
   - 新機能のon/offをconfigで制御できるようにする
   - `if config.get('new_feature_enabled', False):` のようにデフォルトで無効

## 主要なアーキテクチャ設計

### モデルアーキテクチャ


### 学習戦略

**交差検証**:

**データ拡張**:

**損失関数**:

### 推論戦略

**アンサンブル**:

## データ形式のメモ

### train.csv

### submission.csv

## 重要な設定パラメータ

コードを変更する際、学習と推論で以下のパラメータが一致していることを確認すること:

## 依存ライブラリ

使用する主要ライブラリ:
- `torch` - 深層学習用 PyTorch
- `timm` - 事前学習済みビジョンモデル
- `albumentations` - 音声・画像拡張
- `pandas`、`numpy` - データ操作
- `scikit-learn` - 交差検証とメトリクス
- `tqdm` - プログレスバー

## 評価指標

**Macro-averaged ROC-AUC**（真のラベルが存在しないクラスを除外したバージョン）

### スコアリングロジック

```python
def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    '''
    Version of macro-averaged ROC-AUC score that ignores all classes that have no true positive labels.
    '''
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    if not pandas.api.types.is_numeric_dtype(submission.values):
        bad_dtypes = {x: submission[x].dtype for x in submission.columns if not pandas.api.types.is_numeric_dtype(submission[x])}
        raise ParticipantVisibleError(f'Invalid submission data types found: {bad_dtypes}')

    solution_sums = solution.sum(axis=0)
    scored_columns = list(solution_sums[solution_sums > 0].index.values)
    assert len(scored_columns) > 0

    return kaggle_metric_utilities.safe_call_score(sklearn.metrics.roc_auc_score, solution[scored_columns].values, submission[scored_columns].values, average='macro')
```

### 重要な仕様

- **除外クラス**: テストデータに真のラベルが1つも存在しないクラス（`solution_sums == 0`）はスコア計算から除外される
- **提出値**: 数値型である必要がある（文字列などは `ParticipantVisibleError`）
- **集計方法**: 対象クラスのROC-AUCをマクロ平均（クラスごとに等しく重み付け）

## コンペティションの制約

- コードは Kaggle Notebooks で実行可能であること（CPU ≤9時間）
- 提出時はインターネットアクセス不可
- 事前学習済みモデルは使用可能
- 出力ファイル名は `submission.csv` であること

### 重要: テストデータの制約

**テストサウンドスケープはKaggle環境でのみアクセス可能です。**

- Kaggle推論環境では `test_soundscapes/` フォルダにアクセス可能
- しかし、**テストデータの中身を事前に確認する方法はない**
- したがって、以下のアプローチは**不可能**:
  - **Pseudo Labeling（オフライン版）**: テストデータを事前にダウンロードして予測→再学習
  - **テストデータの分析**: テストデータの分布を事前に把握して学習データに反映

- 可能なのは:
  - Kaggle推論ノートブック内での**オンラインPseudo Labeling**（9時間制限内でself-training）

**この制約を忘れて「テストデータを使って〜」という提案をしないこと。**

## OOF 誤差分析

**目的**: Out-of-Fold予測の誤差を分析し、モデルの弱点と改善方向を特定する

### 分析の実行方法

### 分析の観点と目的

### 分析結果の活用

### 出力ファイル

## CV vs LB 相関分析

**目的**: CV (OOF スコア) と Public LB の相関を分析し、CVが信頼できる指標かを確認する

### ユーザーから「CV LBの分析をして」と依頼されたら

### 出力ファイル

### 主要な発見
