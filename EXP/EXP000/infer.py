"""
EXP000: BirdCLEF+ 2026 推論スクリプト
- Backbone : Google Perch 2.0（ファインチューニングなし）
- 推論環境  : CPU only（Kaggle制約）
- 入力     : test_soundscapes/ （1分 ogg × ~600ファイル）
- 出力     : submission.csv（5秒セグメントごとの234種確率）
- 戦略     : Perchのoutput_0（鳥類ロジット）をsigmoidしてeBirdコードでマッピング
             非鳥類72種（昆虫・両生類・哺乳類・爬虫類）は確率0を出力

速度最適化:
- Perchへの入力を (BATCH_FILES*12, 160000) で一括推論（バッチ非対応時は逐次にフォールバック）
- 推論前にウォームアップ呼び出しでTFグラフをJITコンパイル
- ThreadPoolExecutorで音声I/Oを並列化
"""

import librosa
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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

    # バッチ処理: 何ファイルずつPerchに渡すか
    # 16ファイル × 12セグメント = 192セグメントを一括推論
    BATCH_FILES = 16

    # 音声I/O並列スレッド数
    NUM_WORKERS = 4

    # Perch: Kaggle提出環境ではインターネット不可のため固定パスで参照
    # ノートブック設定でデータセットとしてアタッチしておくこと
    PERCH_PATH = "/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1"


# ============================================================
# Utilities
# ============================================================
def load_label_cols() -> list:
    """sample_submission.csv の列順を正として label_cols を返す"""
    sample_sub = pd.read_csv(CFG.DATA_DIR / "sample_submission.csv", nrows=1)
    label_cols = [c for c in sample_sub.columns if c != "row_id"]
    if len(label_cols) != CFG.N_CLASSES:
        raise ValueError(
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

    taxonomy = pd.read_csv(CFG.DATA_DIR / "taxonomy.csv")
    label_to_class = dict(zip(taxonomy["primary_label"], taxonomy["class_name"]))

    mapping = np.full(CFG.N_CLASSES, -1, dtype=np.int32)
    matched = 0
    for comp_idx, label in enumerate(label_cols):
        class_name = label_to_class.get(label, "Unknown")
        if class_name != "Aves":
            continue
        perch_idx = perch_label_to_idx.get(label, -1)
        if perch_idx >= 0:
            mapping[comp_idx] = perch_idx
            matched += 1

    n_birds = sum(1 for lbl in label_cols if label_to_class.get(lbl) == "Aves")
    print(f"Competition bird species : {n_birds}")
    print(f"Matched to Perch         : {matched}")
    print(f"Unmatched (will be 0)    : {n_birds - matched}")
    return mapping


def load_and_split(filepath) -> tuple:
    """1ファイルを読み込み、(12セグメントのリスト, row_idのリスト) を返す"""
    try:
        audio, sr = sf.read(filepath)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != CFG.SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=CFG.SAMPLE_RATE)
        audio = audio.astype(np.float32)
    except (OSError, RuntimeError, ValueError) as e:
        print(f"[WARN] Failed to load {filepath}: {e}")
        audio = np.zeros(CFG.SAMPLE_RATE * 60, dtype=np.float32)

    target = CFG.N_SEGMENTS * CFG.N_SAMPLES
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    audio = audio[:target]

    segments = [audio[i * CFG.N_SAMPLES:(i + 1) * CFG.N_SAMPLES] for i in range(CFG.N_SEGMENTS)]
    stem = Path(filepath).stem
    row_ids = [f"{stem}_{(i + 1) * CFG.DURATION}" for i in range(CFG.N_SEGMENTS)]
    return segments, row_ids


# ============================================================
# 推論
# ============================================================
def warmup(infer_fn):
    """tf.vectorized_map のグラフコンパイルを事前に実行してレイテンシを削減"""
    dummy = tf.zeros((2, CFG.N_SAMPLES), dtype=tf.float32)
    tf.vectorized_map(
        lambda seg: infer_fn(inputs=tf.expand_dims(seg, 0))["output_0"][0],
        dummy,
    )
    print("Warm-up done.")


def predict_batch(infer_fn, segments: list, mapping: np.ndarray) -> np.ndarray:
    """
    tf.vectorized_map で疑似バッチ化して一括推論。
    Perch v2のserving_defaultは (1, 160000) 固定だが、
    tf.vectorized_map が「(1, 160000) 用の関数をN個並列実行」してくれる。

    Args:
        segments: セグメントのリスト（各要素: shape (N_SAMPLES,)）
    Returns:
        preds: shape (len(segments), N_CLASSES)
    """
    n = len(segments)
    bird_mask = mapping >= 0
    preds = np.zeros((n, CFG.N_CLASSES), dtype=np.float32)

    x = tf.constant(np.stack(segments, axis=0), dtype=tf.float32)  # (N, 160000)

    # (1, 160000) 用のシグネチャをN個まとめて並列実行
    outputs = tf.vectorized_map(
        lambda seg: infer_fn(inputs=tf.expand_dims(seg, 0))["output_0"][0],
        x,
    )  # outputs: (N, n_perch_classes)

    probs = 1.0 / (1.0 + np.exp(-outputs.numpy()))  # (N, n_perch_classes)
    preds[:, bird_mask] = probs[:, mapping[bird_mask]]

    return preds


def run_inference(perch_path: str, label_cols: list) -> pd.DataFrame:
    """Perch直接推論 → submission DataFrame"""
    test_files = sorted(CFG.TEST_SOUNDSCAPE_DIR.glob("*.ogg"))
    print(f"Test files: {len(test_files)}")

    if len(test_files) == 0:
        print("No test files found. Creating empty submission for commit.")
        return pd.DataFrame(columns=["row_id"] + label_cols)

    print("\nBuilding species mapping...")
    mapping = build_mapping(perch_path, label_cols)

    print("\nLoading Perch 2.0...")
    perch_model = tf.saved_model.load(perch_path)
    infer_fn    = perch_model.signatures["serving_default"]
    print("Perch loaded.")

    print("Warming up...")
    warmup(infer_fn)

    row_id_list = []
    preds_list  = []
    n_files     = len(test_files)

    with ThreadPoolExecutor(max_workers=CFG.NUM_WORKERS) as executor:
        for batch_start in tqdm(range(0, n_files, CFG.BATCH_FILES), desc="Inference"):
            batch_paths = test_files[batch_start:batch_start + CFG.BATCH_FILES]

            # 音声ロードをスレッド並列化
            results = list(executor.map(load_and_split, batch_paths))

            batch_segments = []
            batch_row_ids  = []
            for segments, row_ids in results:
                batch_segments.extend(segments)
                batch_row_ids.extend(row_ids)

            batch_preds = predict_batch(infer_fn, batch_segments, mapping)
            preds_list.append(batch_preds)
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

    perch_path = CFG.PERCH_PATH
    print(f"Perch path: {perch_path}")

    submission = run_inference(perch_path, label_cols)

    output_path = Path("/kaggle/working/submission.csv")
    submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved: {output_path}  shape={submission.shape}")
    print(submission.head(3))


if __name__ == "__main__":
    main()
