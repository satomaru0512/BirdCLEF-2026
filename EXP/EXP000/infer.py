"""
EXP000: BirdCLEF+ 2026 推論スクリプト（Perch ベースライン）
- Backbone  : Google Perch 2.0 (perch_v2_cpu)
- 推論環境   : CPU only（Kaggle制約）
- 入力      : test_soundscapes/ （1分 ogg × ~600ファイル）
- 出力      : submission.csv（5秒セグメントごとの234種確率）
- 設計参照   : bird26-reproduce-perch-protossm-resssm-inf-train.ipynb

パイプライン:
  音声読み込み (60s → 12 × 5s)
  → Perch (batch_n*12, 160000) → logits["output_0" or "label"]
  → PRIMARY_LABELS へのマッピング（ebird_code優先）
  → 温度スケーリング → sigmoid
  → sample_submission順序で結合 → submission.csv
"""

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
    BASE                = Path("/kaggle/input/competitions/birdclef-2026")
    TEST_SOUNDSCAPE_DIR = BASE / "test_soundscapes"

    # Classes
    N_CLASSES = 234

    # Perch: ノートブック設定でデータセットとしてアタッチしておくこと
    MODEL_DIR = Path("/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1")

    # バッチ処理: 何ファイルずつPerchに渡すか
    # CPU環境でのメモリ安全性を考慮して8に設定（8*12*160000*4bytes ≈ 60MB）
    BATCH_FILES    = 8
    # test_soundscapesが空の場合のドライランファイル数（notebook準拠）
    DRYRUN_N_FILES = 20

    # 温度スケーリング（notebook準拠）
    T_AVES       = 1.10   # 鳥類
    T_TEXTURE    = 0.95   # 両生類・昆虫
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
# Species mapping: PRIMARY_LABELS → Perch 出力インデックス
# ============================================================
def build_mapping(primary_labels: list) -> tuple:
    """
    PRIMARY_LABELS と Perch 出力インデックスのマッピングを構築。
    ebird_code 列を優先して使用し、なければ 0 列目で代替。

    Returns:
        mapped_pos        : コンペ側インデックス配列
        mapped_bc_indices : 対応する Perch 側インデックス配列
    """
    # Perch ラベルファイルを探す
    labels_file = CFG.MODEL_DIR / "assets" / "labels.csv"
    if not labels_file.exists():
        candidates = list(CFG.MODEL_DIR.rglob("label*.csv"))
        if not candidates:
            raise FileNotFoundError(f"Perch labels file not found under {CFG.MODEL_DIR}")
        labels_file = candidates[0]
    print(f"Perch labels file   : {labels_file}")

    bc_labels = pd.read_csv(labels_file)
    print(f"Perch labels columns: {bc_labels.columns.tolist()}")
    print(f"Perch species count : {len(bc_labels)}")

    # ebird_code 列があればそれを使用、なければ 0 列目をコードとして扱う
    if "ebird_code" in bc_labels.columns:
        label_to_bc = dict(zip(bc_labels["ebird_code"], bc_labels.index))
    else:
        label_to_bc = dict(zip(bc_labels.iloc[:, 0], bc_labels.index))

    mapped_pos        = []
    mapped_bc_indices = []
    for i, lbl in enumerate(primary_labels):
        bc_idx = label_to_bc.get(lbl)
        if bc_idx is not None:
            mapped_pos.append(i)
            mapped_bc_indices.append(int(bc_idx))

    print(f"Mapped species      : {len(mapped_pos)} / {len(primary_labels)}")
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


def run_inference(infer_fn, primary_labels: list, sample_sub: pd.DataFrame) -> pd.DataFrame:
    """Perch推論 → submission DataFrame（sample_sub順序で返す）"""
    test_files = sorted(CFG.TEST_SOUNDSCAPE_DIR.glob("*.ogg"))
    print(f"Test files: {len(test_files)}")

    if len(test_files) == 0:
        # コミット用ドライラン: test_soundscapesが空の場合はtrain_soundscapesで代替
        fallback_dir = CFG.BASE / "train_soundscapes"
        test_files   = sorted(fallback_dir.glob("*.ogg"))[:CFG.DRYRUN_N_FILES]
        print(f"No test files found. Dry-run on {len(test_files)} train soundscapes.")

    mapped_pos, mapped_bc_indices = build_mapping(primary_labels)
    class_temps = build_class_temperatures(primary_labels)

    results = []

    for batch_start in tqdm(range(0, len(test_files), CFG.BATCH_FILES), desc="Inference"):
        batch_paths = test_files[batch_start:batch_start + CFG.BATCH_FILES]
        batch_n     = len(batch_paths)

        # 音声読み込み → (batch_n * 12, 160000)
        x            = np.empty((batch_n * CFG.N_WINDOWS, CFG.WINDOW_SAMPLES), dtype=np.float32)
        batch_row_ids = []
        for i, path in enumerate(batch_paths):
            y = read_soundscape_60s(path)
            x[i * CFG.N_WINDOWS:(i + 1) * CFG.N_WINDOWS] = y.reshape(CFG.N_WINDOWS, CFG.WINDOW_SAMPLES)
            batch_row_ids.extend(make_row_ids(path))

        # Perch 一括推論（出力キーを自動判定）
        outputs   = infer_fn(inputs=tf.convert_to_tensor(x))
        logit_key = "output_0" if "output_0" in outputs else "label"
        logits    = outputs[logit_key].numpy().astype(np.float32, copy=False)

        # PRIMARY_LABELS へのマッピング
        batch_scores = np.zeros((batch_n * CFG.N_WINDOWS, len(primary_labels)), dtype=np.float32)
        batch_scores[:, mapped_pos] = logits[:, mapped_bc_indices]

        tmp_df = pd.DataFrame(batch_scores, columns=primary_labels)
        tmp_df.insert(0, "row_id", batch_row_ids)
        results.append(tmp_df)

    submission = pd.concat(results, ignore_index=True)

    # 温度スケーリング → sigmoid
    submission[primary_labels] = sigmoid(
        submission[primary_labels].values / class_temps[None, :]
    )

    # sample_submission の行順序・行数に合わせて左結合（順序保証・欠損は0埋め）
    submission = pd.merge(
        sample_sub[["row_id"]], submission, on="row_id", how="left"
    ).fillna(0.0)

    return submission


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

    # PRIMARY_LABELS と行順序の基準を sample_submission.csv から取得
    sample_sub     = pd.read_csv(CFG.BASE / "sample_submission.csv")
    primary_labels = [c for c in sample_sub.columns if c != "row_id"]
    assert len(primary_labels) == CFG.N_CLASSES, (
        f"Expected {CFG.N_CLASSES} classes, got {len(primary_labels)}"
    )
    print(f"Classes: {len(primary_labels)}")

    print(f"\nLoading Perch from {CFG.MODEL_DIR} ...")
    birdclassifier = tf.saved_model.load(str(CFG.MODEL_DIR))
    infer_fn       = birdclassifier.signatures["serving_default"]
    print("Perch loaded.")

    submission = run_inference(infer_fn, primary_labels, sample_sub)

    output_path = Path("/kaggle/working/submission.csv")
    submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved: {output_path}  shape={submission.shape}")
    print(submission.head(3))


if __name__ == "__main__":
    main()
