"""
EXP000: BirdCLEF+ 2026 推論スクリプト（Perch ベースライン）
- Backbone  : Google Perch 2.0 (perch_v2_cpu)
- 推論環境   : CPU only（Kaggle制約）
- 入力      : test_soundscapes/ （1分 ogg × ~600ファイル）
- 出力      : submission.csv（5秒セグメントごとの234種確率）
- 設計参照   : bird26-reproduce-perch-protossm-resssm-inf-train.ipynb

パイプライン:
  音声読み込み (60s → 12 × 5s)
  → Perch (batch_n*12, 160000) → logits["label"]
  → PRIMARY_LABELS へのマッピング
  → 温度スケーリング → sigmoid
  → submission.csv
"""

import re
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
    SR             = 32000
    WINDOW_SEC     = 5
    WINDOW_SAMPLES = SR * WINDOW_SEC   # 160_000
    FILE_SAMPLES   = 60 * SR           # 1_920_000
    N_WINDOWS      = 12                # 60s / 5s

    # Data paths (Kaggle環境)
    BASE               = Path("/kaggle/input/competitions/birdclef-2026")
    TEST_SOUNDSCAPE_DIR = BASE / "test_soundscapes"

    # Classes
    N_CLASSES = 234

    # Perch: ノートブック設定でデータセットとしてアタッチしておくこと
    MODEL_DIR = Path("/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1")

    # バッチ処理: 何ファイルずつPerchに渡すか（batch_n * 12セグメントを一括推論）
    BATCH_FILES    = 16
    # test_soundscapesが空の場合のドライランファイル数（notebook準拠）
    DRYRUN_N_FILES = 20

    # 温度スケーリング（notebook準拠）
    T_AVES    = 1.10   # 鳥類
    T_TEXTURE = 0.95   # 両生類・昆虫
    TEXTURE_TAXA = {"Amphibia", "Insecta"}


# ============================================================
# row_id 生成
# ============================================================
def make_row_ids(path: Path) -> list:
    """stem + _5, _10, ..., _60 のrow_idリストを生成"""
    stem = path.stem
    return [f"{stem}_{t}" for t in range(CFG.WINDOW_SEC, 60 + CFG.WINDOW_SEC, CFG.WINDOW_SEC)]


# ============================================================
# 音声読み込み
# ============================================================
def read_soundscape_60s(path: Path) -> np.ndarray:
    """60秒音声を (FILE_SAMPLES,) float32 で返す。読み込み失敗時はゼロ埋め。"""
    try:
        y, sr = sf.read(path, dtype="float32", always_2d=False)
        if y.ndim == 2:
            y = y.mean(axis=1)
        if sr != CFG.SR:
            raise ValueError(f"Unexpected sample rate {sr}; expected {CFG.SR}")
    except Exception as e:
        print(f"[WARN] {path.name}: {e}")
        return np.zeros(CFG.FILE_SAMPLES, dtype=np.float32)

    if len(y) < CFG.FILE_SAMPLES:
        y = np.pad(y, (0, CFG.FILE_SAMPLES - len(y)))
    else:
        y = y[:CFG.FILE_SAMPLES]
    return y


# ============================================================
# Species mapping: PRIMARY_LABELS → Perch bc_index
# ============================================================
def build_mapping(primary_labels: list) -> tuple:
    """
    PRIMARY_LABELS と Perch 出力インデックスのマッピングを構築。

    Returns:
        mapped_pos        : コンペ側インデックス配列
        mapped_bc_indices : 対応する Perch 側インデックス配列
    """
    taxonomy = pd.read_csv(CFG.BASE / "taxonomy.csv")

    # Perch ラベルファイルを探す（labels.csv または label.csv）
    labels_file = CFG.MODEL_DIR / "assets" / "labels.csv"
    if not labels_file.exists():
        candidates = list(CFG.MODEL_DIR.rglob("label*.csv"))
        if not candidates:
            raise FileNotFoundError(f"Perch labels file not found under {CFG.MODEL_DIR}")
        labels_file = candidates[0]
    print(f"Perch labels file: {labels_file}")

    bc_labels = pd.read_csv(labels_file)
    print(f"Perch labels columns: {bc_labels.columns.tolist()}")
    print(f"Perch species count : {len(bc_labels)}")

    # マッピング構築: scientific_name_lookup が存在する場合はそれで結合、
    # なければ eBird コード (primary_label) で直接マッチ
    if "scientific_name_lookup" in bc_labels.columns and "scientific_name_lookup" in taxonomy.columns:
        merged = taxonomy.merge(
            bc_labels[["scientific_name_lookup", "bc_index"]].dropna(),
            on="scientific_name_lookup",
            how="left",
        )
        label_to_bc = dict(zip(merged["primary_label"], merged["bc_index"]))
    elif "ebird_code" in bc_labels.columns:
        label_to_bc = dict(zip(bc_labels["ebird_code"], bc_labels.index))
    else:
        # 列名を確認してどちらかに対応
        raise ValueError(
            f"Cannot build mapping from Perch labels. Columns: {bc_labels.columns.tolist()}"
        )

    class_name_map = taxonomy.set_index("primary_label")["class_name"].to_dict()

    mapped_pos        = []
    mapped_bc_indices = []
    for i, lbl in enumerate(primary_labels):
        # 非鳥類は Perch に対応なし → スキップ（スコア0のまま）
        if class_name_map.get(lbl, "Aves") not in CFG.TEXTURE_TAXA | {"Aves"}:
            continue
        bc_idx = label_to_bc.get(lbl)
        if bc_idx is not None and not np.isnan(float(bc_idx)):
            mapped_pos.append(i)
            mapped_bc_indices.append(int(bc_idx))

    print(f"Mapped species: {len(mapped_pos)} / {len(primary_labels)}")
    return np.array(mapped_pos, dtype=np.int32), np.array(mapped_bc_indices, dtype=np.int32)


def build_class_temperatures(primary_labels: list) -> np.ndarray:
    """クラスごとの温度スケーリング配列を構築"""
    taxonomy       = pd.read_csv(CFG.BASE / "taxonomy.csv")
    class_name_map = taxonomy.set_index("primary_label")["class_name"].to_dict()

    temps = np.full(len(primary_labels), CFG.T_AVES, dtype=np.float32)
    for i, lbl in enumerate(primary_labels):
        if class_name_map.get(lbl, "Aves") in CFG.TEXTURE_TAXA:
            temps[i] = CFG.T_TEXTURE
    return temps


# ============================================================
# 推論
# ============================================================
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def run_inference(infer_fn, primary_labels: list) -> pd.DataFrame:
    """Perch推論 → submission DataFrame"""
    test_files = sorted(CFG.TEST_SOUNDSCAPE_DIR.glob("*.ogg"))
    print(f"Test files: {len(test_files)}")

    if len(test_files) == 0:
        # コミット用ドライラン: test_soundscapesが空の場合はtrain_soundscapesで代替
        fallback_dir = CFG.BASE / "train_soundscapes"
        test_files   = sorted(fallback_dir.glob("*.ogg"))[:CFG.DRYRUN_N_FILES]
        print(f"No test files found. Dry-run on {len(test_files)} train soundscapes.")

    mapped_pos, mapped_bc_indices = build_mapping(primary_labels)
    class_temps = build_class_temperatures(primary_labels)

    n_files    = len(test_files)
    n_rows     = n_files * CFG.N_WINDOWS
    all_scores = np.zeros((n_rows, len(primary_labels)), dtype=np.float32)
    all_row_ids = []
    row_ptr    = 0

    for batch_start in tqdm(range(0, n_files, CFG.BATCH_FILES), desc="Inference"):
        batch_paths = test_files[batch_start:batch_start + CFG.BATCH_FILES]
        batch_n     = len(batch_paths)

        # 音声読み込み → (batch_n * 12, 160000)
        x = np.empty((batch_n * CFG.N_WINDOWS, CFG.WINDOW_SAMPLES), dtype=np.float32)
        for i, path in enumerate(batch_paths):
            y = read_soundscape_60s(path)
            x[i * CFG.N_WINDOWS:(i + 1) * CFG.N_WINDOWS] = y.reshape(CFG.N_WINDOWS, CFG.WINDOW_SAMPLES)
            all_row_ids.extend(make_row_ids(path))

        # Perch 一括推論
        outputs = infer_fn(inputs=tf.convert_to_tensor(x))
        logits  = outputs["label"].numpy().astype(np.float32, copy=False)  # (batch_n*12, n_perch_classes)

        # PRIMARY_LABELS へのマッピング
        n_batch_rows = batch_n * CFG.N_WINDOWS
        all_scores[row_ptr:row_ptr + n_batch_rows, mapped_pos] = logits[:, mapped_bc_indices]
        row_ptr += n_batch_rows

    # 温度スケーリング → sigmoid
    probs = sigmoid(all_scores / class_temps[None, :])

    submission = pd.DataFrame(probs, columns=primary_labels)
    submission.insert(0, "row_id", all_row_ids)
    return submission[["row_id"] + primary_labels]


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 55)
    print(f"EXP000 Inference (Perch Baseline): {CFG.CHILD_EXP}")
    print("=" * 55)

    # CPUのみ使用（Kaggle推論環境制約）
    tf.config.set_visible_devices([], "GPU")
    print("Using CPU only")

    # PRIMARY_LABELS を sample_submission.csv から取得
    sample_sub     = pd.read_csv(CFG.BASE / "sample_submission.csv", nrows=1)
    primary_labels = [c for c in sample_sub.columns if c != "row_id"]
    assert len(primary_labels) == CFG.N_CLASSES, (
        f"Expected {CFG.N_CLASSES} classes, got {len(primary_labels)}"
    )
    print(f"Classes: {len(primary_labels)}")

    print(f"\nLoading Perch from {CFG.MODEL_DIR} ...")
    birdclassifier = tf.saved_model.load(str(CFG.MODEL_DIR))
    infer_fn       = birdclassifier.signatures["serving_default"]
    print("Perch loaded.")

    submission = run_inference(infer_fn, primary_labels)

    output_path = Path("/kaggle/working/submission.csv")
    submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved: {output_path}  shape={submission.shape}")
    print(submission.head(3))


if __name__ == "__main__":
    main()
