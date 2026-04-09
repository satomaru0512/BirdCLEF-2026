"""
EXP000: BirdCLEF+ 2026 推論スクリプト
- Backbone : Google Perch 2.0
- 推論環境  : CPU only（Kaggle制約）
- 入力     : test_soundscapes/ （1分 ogg × ~600ファイル）
- 出力     : submission.csv（5秒セグメントごとの234種確率）
- アンサンブル: 5-fold平均
"""

import glob
import librosa
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
from tqdm import tqdm


# ============================================================
# Config
# ============================================================
class CFG:
    EXP_NAME  = "EXP000"
    CHILD_EXP = "child-exp000"

    # Audio
    SAMPLE_RATE  = 32000
    DURATION     = 5
    N_SAMPLES    = SAMPLE_RATE * DURATION  # 160_000
    N_SEGMENTS   = 12                      # 1分 / 5秒 = 12セグメント

    # Data paths (Kaggle環境)
    DATA_DIR            = Path("/kaggle/input/competitions/birdclef-2026")
    TEST_SOUNDSCAPE_DIR = DATA_DIR / "test_soundscapes"

    # Classes
    N_CLASSES = 234

    # 推論バッチサイズ（CPU: 小さめ）
    BATCH_SIZE = 8

    # Kaggle Models からロードするモデル
    KAGGLE_MODEL_HANDLE = "wasabi777/birdclef2026-exp000/tensorFlow2/perch-v2-baseline"
    N_FOLDS = 1  # スコア確認用（本番は5）

    # Perch
    PERCH_HANDLE = (
        "google/bird-vocalization-classifier"
        "/tensorFlow2/bird-vocalization-classifier/2"
    )


# ============================================================
# モデル定義（train.pyと同一構造）
# ============================================================
class PerchClassifier(tf.keras.Model):
    """Perch 2.0 backbone + 分類ヘッド"""

    def __init__(self, perch_model, n_classes: int):
        super().__init__()
        self.perch = perch_model  # 外部でロード済みのPerchを受け取る
        self.head = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation="relu", dtype="float32"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(n_classes, activation="sigmoid", dtype="float32"),
        ], name="head")
        self.head.build(input_shape=(None, 1280))

    def _infer_perch_batch(self, waveform):
        infer_fn = self.perch.signatures['serving_default']
        waveform = tf.cast(waveform, tf.float32)
        all_embs = []
        for i in range(waveform.shape[0]):
            single_audio = tf.expand_dims(waveform[i], 0)
            outputs = infer_fn(inputs=single_audio)
            all_embs.append(outputs['output_1'])
        return tf.concat(all_embs, axis=0)

    def call(self, waveform, training: bool = False):
        embeddings = tf.cast(self._infer_perch_batch(waveform), tf.float32)
        return self.head(embeddings, training=training)


# ============================================================
# Utilities
# ============================================================
def load_label_cols() -> list:
    """
    [C2修正] sample_submission.csv の列順を正として label_cols を返す。
    taxonomy.csv の行順に依存しない。
    """
    sample_sub = pd.read_csv(CFG.DATA_DIR / "sample_submission.csv", nrows=1)
    label_cols = [c for c in sample_sub.columns if c != "row_id"]
    assert len(label_cols) == CFG.N_CLASSES, (
        f"sample_submission has {len(label_cols)} label cols, expected {CFG.N_CLASSES}"
    )
    return label_cols


def load_audio_full(filepath: str) -> np.ndarray:
    """音声ファイルを全て読み込む（1分）"""
    try:
        audio, sr = sf.read(filepath)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != CFG.SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=CFG.SAMPLE_RATE)
        return audio.astype(np.float32)
    except (OSError, RuntimeError, ValueError) as e:
        print(f"[WARN] Failed to load {filepath}: {e}")
        return np.zeros(CFG.SAMPLE_RATE * 60, dtype=np.float32)


def split_into_segments(audio: np.ndarray) -> list:
    """
    [C3修正] 音声をまず 12×N_SAMPLES にpadしてから分割。
    常に正確に12セグメントを返す。
    """
    target = CFG.N_SEGMENTS * CFG.N_SAMPLES
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    audio = audio[:target]  # 長すぎる場合も12セグメント分に切る
    return [audio[i * CFG.N_SAMPLES:(i + 1) * CFG.N_SAMPLES] for i in range(CFG.N_SEGMENTS)]


def parse_filename_to_row_ids(filepath: str) -> list:
    """ファイル名から row_id リスト（end_time=5,10,...,60）を生成"""
    stem = Path(filepath).stem
    return [f"{stem}_{t}" for t in range(5, 65, 5)]


# ============================================================
# 推論
# ============================================================
def predict_file(model: PerchClassifier, filepath: str) -> np.ndarray:
    """
    1ファイル（1分）を推論
    Returns: shape (12, N_CLASSES)
    """
    audio    = load_audio_full(filepath)
    segments = split_into_segments(audio)  # 常に12セグメント

    all_preds = []
    for i in range(0, len(segments), CFG.BATCH_SIZE):
        batch    = np.stack(segments[i:i + CFG.BATCH_SIZE])
        batch_tf = tf.constant(batch, dtype=tf.float32)
        preds    = model(batch_tf, training=False).numpy()  # (B, 234)
        all_preds.append(preds)

    return np.concatenate(all_preds, axis=0)  # (12, 234)


def run_inference(perch_path: str, weights_dir: Path, label_cols: list) -> pd.DataFrame:
    """5-fold アンサンブル推論 → submission DataFrame"""
    test_files = sorted(glob.glob(str(CFG.TEST_SOUNDSCAPE_DIR / "*.ogg")))
    print(f"Test files: {len(test_files)}")

    if len(test_files) == 0:
        # 提出前の編集環境ではテストファイルが存在しない（提出時に差し替えられる）
        print("No test files found. Creating empty submission for commit.")
        return pd.DataFrame(columns=["row_id"] + label_cols)

    # Perchを1回だけロード（全foldで共有）
    print("\nLoading Perch backbone (shared across folds)...")
    shared_perch = tf.saved_model.load(perch_path)

    # fold毎に累積して最後に平均
    fold_preds = {fp: np.zeros((12, CFG.N_CLASSES), dtype=np.float32) for fp in test_files}

    for fold in range(CFG.N_FOLDS):
        weights_path = weights_dir / f"best_model_fold{fold}.weights.h5"
        print(f"\n[Fold {fold}] Loading weights: {weights_path}")

        model = PerchClassifier(perch_model=shared_perch, n_classes=CFG.N_CLASSES)
        _ = model(tf.zeros((1, CFG.N_SAMPLES), dtype=tf.float32), training=False)
        model.load_weights(str(weights_path))
        model.trainable = False

        for filepath in tqdm(test_files, desc=f"Fold {fold}"):
            fold_preds[filepath] += predict_file(model, filepath)

        del model

    # 平均 → DataFrame
    all_rows = []
    for filepath in test_files:
        avg_preds = fold_preds[filepath] / CFG.N_FOLDS  # (12, 234)
        row_ids   = parse_filename_to_row_ids(filepath)
        for seg_idx, row_id in enumerate(row_ids):
            row = {"row_id": row_id}
            for j, label in enumerate(label_cols):
                row[label] = float(avg_preds[seg_idx, j])
            all_rows.append(row)

    submission = pd.DataFrame(all_rows)
    return submission[["row_id"] + label_cols]


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 55)
    print(f"EXP000 Inference: {CFG.CHILD_EXP}")
    print("=" * 55)

    # CPUのみ使用（Kaggle推論環境制約）
    tf.config.set_visible_devices([], 'GPU')
    print("Using CPU only")

    # [C2修正] label_cols は sample_submission.csv の列順を使う
    label_cols = load_label_cols()
    print(f"Classes: {len(label_cols)}")

    # [C1修正] kagglehub.model_download はローカルキャッシュ or Kaggle内プロキシ経由で動作。
    # インターネット不可でも、Notebookにモデルを入力として追加しておけばキャッシュから取得される。
    print("\nLoading Perch 2.0...")
    perch_path = kagglehub.model_download(CFG.PERCH_HANDLE)
    print(f"Perch path: {perch_path}")

    print("\nLoading trained weights...")
    weights_dir = Path(kagglehub.model_download(CFG.KAGGLE_MODEL_HANDLE))
    print(f"Weights dir: {weights_dir}")

    # 推論
    submission = run_inference(perch_path, weights_dir, label_cols)

    # 保存
    output_path = Path("/kaggle/working/submission.csv")
    submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved: {output_path}  shape={submission.shape}")
    print(submission.head(3))


if __name__ == "__main__":
    main()
