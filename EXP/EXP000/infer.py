"""
EXP000: BirdCLEF+ 2026 推論スクリプト
- Backbone : Google Perch 2.0（ファインチューニングなし）
- 推論環境  : CPU only（Kaggle制約）
- 入力     : test_soundscapes/ （1分 ogg × ~600ファイル）
- 出力     : submission.csv（5秒セグメントごとの234種確率）
- 戦略     : Perchのoutput_0（鳥類ロジット）をsigmoidしてeBirdコードでマッピング
             非鳥類72種（昆虫・両生類・哺乳類・爬虫類）は確率0を出力
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
    SAMPLE_RATE = 32000
    DURATION    = 5
    N_SAMPLES   = SAMPLE_RATE * DURATION  # 160_000
    N_SEGMENTS  = 12                      # 1分 / 5秒 = 12セグメント

    # Data paths (Kaggle環境)
    DATA_DIR            = Path("/kaggle/input/competitions/birdclef-2026")
    TEST_SOUNDSCAPE_DIR = DATA_DIR / "test_soundscapes"

    # Classes
    N_CLASSES = 234

    # バッチ処理: 何ファイルずつPerchに渡すか（16ファイル×12セグメント=192セグメント/バッチ）
    BATCH_FILES = 16

    # Perch
    PERCH_HANDLE = (
        "google/bird-vocalization-classifier"
        "/tensorFlow2/bird-vocalization-classifier/2"
    )


# ============================================================
# Utilities
# ============================================================
def load_label_cols() -> list:
    """sample_submission.csv の列順を正として label_cols を返す"""
    sample_sub = pd.read_csv(CFG.DATA_DIR / "sample_submission.csv", nrows=1)
    label_cols = [c for c in sample_sub.columns if c != "row_id"]
    assert len(label_cols) == CFG.N_CLASSES, (
        f"sample_submission has {len(label_cols)} label cols, expected {CFG.N_CLASSES}"
    )
    return label_cols


def build_mapping(perch_path: str, label_cols: list) -> np.ndarray:
    """
    コンペ種インデックス → Perch出力インデックス のマッピング配列を構築。
    非鳥類または Perch に存在しない種は -1 を返す。

    Returns:
        mapping: shape (N_CLASSES,), dtype=int
                 mapping[i] = j  → コンペ種iはPerch出力jに対応
                 mapping[i] = -1 → Perchに対応なし（確率0を出力）
    """
    # Perch のラベルファイルを読み込む（assetsディレクトリ内のlabel.csv）
    label_file = Path(perch_path) / "assets" / "label.csv"
    if not label_file.exists():
        candidates = list(Path(perch_path).rglob("label.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"Perch label.csv not found under {perch_path}. "
                "Assets directory structure may differ."
            )
        label_file = candidates[0]

    print(f"Perch label file: {label_file}")
    perch_labels_df = pd.read_csv(label_file, header=None, names=["ebird_code"])
    perch_labels = perch_labels_df["ebird_code"].tolist()
    perch_label_to_idx = {label: idx for idx, label in enumerate(perch_labels)}
    print(f"Perch species count: {len(perch_labels)}")

    # コンペの taxonomy を読み込み（eBirdコード取得のため）
    taxonomy = pd.read_csv(CFG.DATA_DIR / "taxonomy.csv")
    label_to_class = dict(zip(taxonomy["primary_label"], taxonomy["class_name"]))

    # マッピング構築
    mapping = np.full(CFG.N_CLASSES, -1, dtype=np.int32)
    matched = 0
    for comp_idx, label in enumerate(label_cols):
        class_name = label_to_class.get(label, "Unknown")
        if class_name != "Aves":
            continue  # 非鳥類はスキップ（-1のまま）
        perch_idx = perch_label_to_idx.get(label, -1)
        if perch_idx >= 0:
            mapping[comp_idx] = perch_idx
            matched += 1

    n_birds = sum(1 for lbl in label_cols if label_to_class.get(lbl) == "Aves")
    print(f"Competition bird species : {n_birds}")
    print(f"Matched to Perch         : {matched}")
    print(f"Unmatched (will be 0)    : {n_birds - matched}")
    return mapping


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
    """音声を 12×N_SAMPLES にpadしてから分割。常に12セグメントを返す。"""
    target = CFG.N_SEGMENTS * CFG.N_SAMPLES
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    audio = audio[:target]
    return [audio[i * CFG.N_SAMPLES:(i + 1) * CFG.N_SAMPLES] for i in range(CFG.N_SEGMENTS)]


def parse_filename_to_row_ids(filepath: str) -> list:
    """ファイル名から row_id リスト（end_time=5,10,...,60）を生成"""
    stem = Path(filepath).stem
    return [f"{stem}_{t}" for t in range(5, 65, 5)]


# ============================================================
# 推論
# ============================================================
def predict_batch(
    infer_fn,
    segments: list,
    mapping: np.ndarray,
) -> np.ndarray:
    """
    複数セグメントをまとめてPerchに渡して推論する（バッチ処理）。
    EXP000からの変更: 1セグメントずつ→まとめて処理（7200回→38回のPerch呼び出しに削減）

    Args:
        segments: 任意個数のセグメントリスト（各要素: shape (N_SAMPLES,)）
    Returns:
        preds: shape (len(segments), N_CLASSES)
    """
    x         = np.stack(segments).astype(np.float32)  # (N, 160000)
    x_tf      = tf.constant(x, dtype=tf.float32)
    outputs   = infer_fn(inputs=x_tf)
    logits    = outputs["output_0"].numpy()             # (N, N_perch_classes)
    probs     = 1.0 / (1.0 + np.exp(-logits))          # sigmoid

    preds     = np.zeros((len(segments), CFG.N_CLASSES), dtype=np.float32)
    bird_mask = mapping >= 0
    preds[:, bird_mask] = probs[:, mapping[bird_mask]]  # 非鳥類は0.0のまま
    return preds


def run_inference(perch_path: str, label_cols: list) -> pd.DataFrame:
    """Perch直接推論 → submission DataFrame"""
    test_files = sorted(glob.glob(str(CFG.TEST_SOUNDSCAPE_DIR / "*.ogg")))
    print(f"Test files: {len(test_files)}")

    if len(test_files) == 0:
        print("No test files found. Creating empty submission for commit.")
        return pd.DataFrame(columns=["row_id"] + label_cols)

    # マッピング構築
    print("\nBuilding species mapping...")
    mapping = build_mapping(perch_path, label_cols)

    # Perch ロード
    print("\nLoading Perch 2.0...")
    perch_model = tf.saved_model.load(perch_path)
    infer_fn    = perch_model.signatures["serving_default"]
    print("Perch loaded.")

    # 推論（バッチ処理: BATCH_FILES ファイルずつまとめてPerchに渡す）
    row_id_list = []
    preds_list  = []
    n_files     = len(test_files)

    for batch_start in tqdm(range(0, n_files, CFG.BATCH_FILES), desc="Inference"):
        batch_paths = test_files[batch_start:batch_start + CFG.BATCH_FILES]

        # バッチ内の全ファイルのセグメントを結合
        batch_segments = []
        batch_row_ids  = []
        files_n_segs   = []  # ファイルごとのセグメント数（常に12）
        for filepath in batch_paths:
            audio    = load_audio_full(filepath)
            segments = split_into_segments(audio)
            batch_segments.extend(segments)
            batch_row_ids.extend(parse_filename_to_row_ids(filepath))
            files_n_segs.append(len(segments))

        # まとめて1回のPerch呼び出し（例: 16ファイル×12セグ=192セグメント）
        batch_preds = predict_batch(infer_fn, batch_segments, mapping)  # (192, 234)

        # ファイル単位に戻して格納
        seg_cursor = 0
        for n_segs in files_n_segs:
            preds_list.append(batch_preds[seg_cursor:seg_cursor + n_segs])
            seg_cursor += n_segs
        row_id_list.extend(batch_row_ids)

    preds_array = np.vstack(preds_list)  # (N_files*12, 234)
    submission  = pd.DataFrame(preds_array, columns=label_cols)
    submission.insert(0, "row_id", row_id_list)
    return submission[["row_id"] + label_cols]


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 55)
    print(f"EXP000 Inference: {CFG.CHILD_EXP}")
    print("=" * 55)

    # CPUのみ使用（Kaggle推論環境制約）
    tf.config.set_visible_devices([], "GPU")
    print("Using CPU only")

    label_cols = load_label_cols()
    print(f"Classes: {len(label_cols)}")

    print("\nDownloading Perch 2.0...")
    perch_path = kagglehub.model_download(CFG.PERCH_HANDLE)
    print(f"Perch path: {perch_path}")

    submission = run_inference(perch_path, label_cols)

    output_path = Path("/kaggle/working/submission.csv")
    submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved: {output_path}  shape={submission.shape}")
    print(submission.head(3))


if __name__ == "__main__":
    main()
