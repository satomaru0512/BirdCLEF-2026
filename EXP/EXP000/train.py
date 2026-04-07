"""
EXP000: BirdCLEF+ 2026 Baseline
- Backbone : Google Perch 2.0 (bird-vocalization-classifier/2)
- Strategy : Stage1 (埋め込み事前計算 + headのみ学習) → Stage2 (全体fine-tune)
- Data     : train_audio + labeled train_soundscapes
- CV       : StratifiedKFold (5 folds)
- Tracking : Weights & Biases
- GPU最適化: Mixed Precision / @tf.function / マルチGPU / LR Warmup+Cosine
"""

import gc
import json
import os
import random

import kagglehub
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
import wandb
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# ============================================================
# GPU セットアップ
# ============================================================
def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs available: {len(gpus)}")
    else:
        print("No GPU found, using CPU")

    # Mixed Precision (FP16) → 速度向上・メモリ節約
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print("Mixed precision: mixed_float16")

    # マルチGPU戦略
    strategy = tf.distribute.MirroredStrategy() if len(gpus) > 1 else tf.distribute.get_strategy()
    print(f"Strategy: {strategy.__class__.__name__}  replicas={strategy.num_replicas_in_sync}")
    return strategy


# ============================================================
# Config
# ============================================================
class CFG:
    EXP_NAME  = "EXP000"
    CHILD_EXP = "child-exp000"

    # Audio
    SAMPLE_RATE = 32000
    DURATION    = 5                       # seconds
    N_SAMPLES   = SAMPLE_RATE * DURATION  # 160_000

    # Model
    PERCH_HANDLE = (
        "google/bird-vocalization-classifier"
        "/tensorFlow2/bird-vocalization-classifier/2"
    )

    # Data paths (Kaggle環境)
    DATA_DIR              = Path("/kaggle/input/birdclef-2026")
    TRAIN_AUDIO_DIR       = DATA_DIR / "train_audio"
    TRAIN_SOUNDSCAPES_DIR = DATA_DIR / "train_soundscapes"
    OUTPUT_DIR            = Path(f"./outputs/{CHILD_EXP}")

    # Classes
    N_CLASSES = 234

    # Training - Stage 1 (head only, 埋め込みキャッシュ使用)
    BATCH_SIZE_HEAD  = 512   # 埋め込みはメモリ効率が良いので大きく
    EPOCHS_HEAD      = 20
    LR_HEAD          = 1e-3
    WARMUP_EPOCHS_HEAD = 2

    # Training - Stage 2 (full fine-tune)
    BATCH_SIZE_FINETUNE  = 64    # 音声→Perch→headの全体、メモリ多いので適度に
    EPOCHS_FINETUNE      = 10
    LR_FINETUNE          = 5e-5
    WARMUP_EPOCHS_FT     = 1
    GRAD_CLIP_NORM       = 1.0   # 勾配クリッピング

    # Augmentation
    NOISE_STD   = 0.005  # ガウスノイズ標準偏差
    TIME_SHIFT  = 0.1    # 時間シフト最大割合

    # Cross-Validation
    N_FOLDS = 5
    SEED    = 42

    # wandb
    WANDB_PROJECT = "birdclef-2026"

    # Kaggle Models アップロード先
    # 形式: "{kaggle_username}/{model-name}/tensorFlow2/{variant}"
    # 例: "satomaru0512/birdclef2026-exp000/tensorFlow2/perch-v2-baseline"
    KAGGLE_MODEL_HANDLE = "wasabi777/birdclef2026-exp000/tensorFlow2/perch-v2-baseline"


# ============================================================
# Utilities
# ============================================================
def seed_everything(seed: int = CFG.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def hms_to_seconds(hms: str) -> int:
    parts = hms.split(":")
    return sum(int(x) * (60 ** i) for i, x in enumerate(reversed(parts)))


def cosine_decay_with_warmup(epoch, total_epochs, warmup_epochs, lr_max):
    """Warmup + Cosine Decay スケジューラ"""
    if epoch < warmup_epochs:
        return lr_max * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return lr_max * 0.5 * (1.0 + np.cos(np.pi * progress))


# ============================================================
# Data Preparation
# ============================================================
def load_taxonomy():
    tax = pd.read_csv(CFG.DATA_DIR / "taxonomy.csv")
    label2idx = {label: idx for idx, label in enumerate(tax["primary_label"])}
    idx2label = {idx: label for label, idx in label2idx.items()}
    return tax, label2idx, idx2label


def prepare_train_audio(train_csv: pd.DataFrame, label2idx: dict) -> pd.DataFrame:
    df = train_csv.copy()
    df["label_idx"] = df["primary_label"].map(label2idx)
    df = df.dropna(subset=["label_idx"])
    df["label_idx"] = df["label_idx"].astype(int)
    df["filepath"]  = df["filename"].apply(lambda x: str(CFG.TRAIN_AUDIO_DIR / x))
    df["start_sec"] = 0
    return df[["filepath", "primary_label", "label_idx", "start_sec"]].reset_index(drop=True)


def prepare_train_soundscapes(labels_csv: pd.DataFrame, label2idx: dict) -> pd.DataFrame:
    records = []
    for _, row in labels_csv.iterrows():
        filepath  = str(CFG.TRAIN_SOUNDSCAPES_DIR / row["filename"])
        start_sec = hms_to_seconds(row["start"])
        labels    = [l.strip() for l in str(row["primary_label"]).split(";") if l.strip()]
        for label in labels:
            if label in label2idx:
                records.append({
                    "filepath"     : filepath,
                    "primary_label": label,
                    "label_idx"    : label2idx[label],
                    "start_sec"    : start_sec,
                })
    return pd.DataFrame(records)


# ============================================================
# Audio Loading & Augmentation
# ============================================================
def load_audio_chunk(filepath: str, start_sec: float = 0) -> np.ndarray:
    try:
        info        = sf.info(filepath)
        orig_sr     = info.samplerate
        start_frame = int(start_sec * orig_sr)
        n_frames    = int(CFG.DURATION * orig_sr)

        audio, _ = sf.read(filepath, start=start_frame, frames=n_frames)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if orig_sr != CFG.SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=CFG.SAMPLE_RATE)

        if len(audio) < CFG.N_SAMPLES:
            audio = np.pad(audio, (0, CFG.N_SAMPLES - len(audio)))
        else:
            audio = audio[: CFG.N_SAMPLES]

        return audio.astype(np.float32)
    except Exception:
        return np.zeros(CFG.N_SAMPLES, dtype=np.float32)


def augment_audio(audio: np.ndarray, training: bool = True) -> np.ndarray:
    """音声データ拡張（学習時のみ）"""
    if not training:
        return audio

    # ガウスノイズ付加
    if random.random() < 0.5:
        audio = audio + np.random.normal(0, CFG.NOISE_STD, audio.shape).astype(np.float32)

    # 時間シフト
    if random.random() < 0.5:
        shift = int(random.uniform(-CFG.TIME_SHIFT, CFG.TIME_SHIFT) * CFG.N_SAMPLES)
        audio = np.roll(audio, shift)

    return audio


# ============================================================
# モデル
# ============================================================
class PerchClassifier(tf.keras.Model):
    """Perch 2.0 backbone + 分類ヘッド"""

    def __init__(self, perch_path: str, n_classes: int):
        super().__init__()
        self.perch = tf.saved_model.load(perch_path)
        # ヘッド（出力はfloat32に戻す）
        self.head = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation="relu", dtype="float32"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(n_classes, activation="sigmoid", dtype="float32"),
        ], name="head")

    def call(self, waveform, training: bool = False):
        # NOTE: Perch 2.0 の API が異なる場合はここを修正
        outputs    = self.perch.infer(waveform)
        embeddings = tf.cast(outputs["embeddings"], tf.float32)  # FP16→FP32
        return self.head(embeddings, training=training)

    def get_embeddings(self, waveform):
        outputs = self.perch.infer(waveform)
        return tf.cast(outputs["embeddings"], tf.float32)

    def freeze_perch(self):
        self.perch.trainable = False

    def unfreeze_perch(self):
        self.perch.trainable = True


class EmbeddingClassifier(tf.keras.Model):
    """Stage1用: 事前計算した埋め込みから分類するモデル"""

    def __init__(self, embedding_dim: int, n_classes: int):
        super().__init__()
        self.head = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(n_classes, activation="sigmoid"),
        ], name="head")

    def call(self, embeddings, training: bool = False):
        return self.head(embeddings, training=training)


# ============================================================
# 埋め込み事前計算（Stage1高速化の核心）
# ============================================================
def precompute_embeddings(
    df: pd.DataFrame,
    perch_model,
    batch_size: int = 128,
    training: bool = False,
) -> np.ndarray:
    """全サンプルのPerch埋め込みを一括計算してキャッシュ"""
    rows = df.to_dict("records")
    all_embeddings = []

    for i in tqdm(range(0, len(rows), batch_size), desc="Precomputing embeddings"):
        batch_rows = rows[i: i + batch_size]
        waveforms  = np.stack([
            load_audio_chunk(r["filepath"], r["start_sec"]) for r in batch_rows
        ])
        waveforms_tf = tf.constant(waveforms)
        embeddings   = perch_model.get_embeddings(waveforms_tf).numpy()
        all_embeddings.append(embeddings)

    return np.concatenate(all_embeddings, axis=0)


# ============================================================
# tf.data.Dataset
# ============================================================
def make_embedding_dataset(
    embeddings: np.ndarray,
    labels_onehot: np.ndarray,
    is_train: bool = True,
) -> tf.data.Dataset:
    """Stage1用: 埋め込みとラベルのDataset"""
    ds = tf.data.Dataset.from_tensor_slices((embeddings, labels_onehot))
    if is_train:
        ds = ds.shuffle(len(embeddings))
    return ds.batch(CFG.BATCH_SIZE_HEAD).prefetch(tf.data.AUTOTUNE)


def make_audio_dataset(
    df: pd.DataFrame,
    n_classes: int,
    is_train: bool = True,
) -> tf.data.Dataset:
    """Stage2用: 音声波形のDataset（拡張あり）"""
    rows = df.to_dict("records")
    if is_train:
        random.shuffle(rows)

    def generator():
        for row in rows:
            audio = load_audio_chunk(row["filepath"], row["start_sec"])
            audio = augment_audio(audio, training=is_train)
            label = np.zeros(n_classes, dtype=np.float32)
            label[int(row["label_idx"])] = 1.0
            yield audio, label

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(CFG.N_SAMPLES,), dtype=tf.float32),
            tf.TensorSpec(shape=(n_classes,),     dtype=tf.float32),
        ),
    )
    if is_train:
        ds = ds.shuffle(2000)
    return (
        ds.batch(CFG.BATCH_SIZE_FINETUNE)
          .prefetch(tf.data.AUTOTUNE)
    )


# ============================================================
# 学習ステップ（@tf.function でグラフ実行）
# ============================================================
@tf.function
def train_step(model, optimizer, loss_fn, x, y):
    with tf.GradientTape() as tape:
        preds = model(x, training=True)
        loss  = loss_fn(y, preds)
    grads = tape.gradient(loss, model.trainable_variables)
    # 勾配クリッピング
    grads, _ = tf.clip_by_global_norm(grads, CFG.GRAD_CLIP_NORM)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


# ============================================================
# 評価
# ============================================================
def compute_auc(model, dataset) -> float:
    all_preds, all_labels = [], []
    for x, labels in dataset:
        preds = model(x, training=False).numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    valid = all_labels.sum(axis=0) > 0
    if valid.sum() == 0:
        return 0.0
    return roc_auc_score(all_labels[:, valid], all_preds[:, valid], average="macro")


def labels_to_onehot(df: pd.DataFrame, n_classes: int) -> np.ndarray:
    onehot = np.zeros((len(df), n_classes), dtype=np.float32)
    for i, idx in enumerate(df["label_idx"].values):
        onehot[i, int(idx)] = 1.0
    return onehot


# ============================================================
# Fold Training
# ============================================================
def train_fold(
    fold: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    perch_path: str,
    label2idx: dict,
    strategy,
) -> float:
    print(f"\n{'='*55}\nFold {fold}\n{'='*55}")
    n_classes = CFG.N_CLASSES

    run = wandb.init(
        project=CFG.WANDB_PROJECT,
        name=f"{CFG.EXP_NAME}-{CFG.CHILD_EXP}-fold{fold}",
        config={k: v for k, v in vars(CFG).items() if not k.startswith("_")},
        reinit=True,
    )

    # ---- Perch モデルをロード ----
    with strategy.scope():
        full_model = PerchClassifier(perch_path=perch_path, n_classes=n_classes)
        full_model.freeze_perch()

    # ----------------------------------------------------------
    # Stage 1: 埋め込み事前計算 → headのみ高速学習
    # ----------------------------------------------------------
    print("\n[Stage 1] Precomputing embeddings...")
    train_embs = precompute_embeddings(train_df, full_model)
    val_embs   = precompute_embeddings(val_df,   full_model)
    train_oh   = labels_to_onehot(train_df, n_classes)
    val_oh     = labels_to_onehot(val_df,   n_classes)

    print(f"  train embeddings: {train_embs.shape}")

    emb_dim = train_embs.shape[1]
    with strategy.scope():
        head_model = EmbeddingClassifier(embedding_dim=emb_dim, n_classes=n_classes)
        loss_fn    = tf.keras.losses.BinaryCrossentropy()

    best_auc_s1, best_weights_s1 = 0.0, None
    train_emb_ds = make_embedding_dataset(train_embs, train_oh, is_train=True)
    val_emb_ds   = make_embedding_dataset(val_embs,   val_oh,   is_train=False)

    for epoch in range(CFG.EPOCHS_HEAD):
        lr = cosine_decay_with_warmup(epoch, CFG.EPOCHS_HEAD, CFG.WARMUP_EPOCHS_HEAD, CFG.LR_HEAD)
        optimizer = tf.keras.optimizers.Adam(lr)
        total_loss = 0.0

        for embs, labels in tqdm(train_emb_ds, desc=f"[F{fold}] S1 Epoch {epoch+1}/{CFG.EPOCHS_HEAD}"):
            total_loss += train_step(head_model, optimizer, loss_fn, embs, labels).numpy()

        val_auc = compute_auc(head_model, val_emb_ds)
        print(f"  lr={lr:.2e}  loss={total_loss:.4f}  val_auc={val_auc:.4f}")
        wandb.log({"fold": fold, "epoch": epoch, "stage": "head",
                   "lr": lr, "train_loss": total_loss, "val_auc": val_auc})

        if val_auc > best_auc_s1:
            best_auc_s1    = val_auc
            best_weights_s1 = head_model.get_weights()

    # Stage1のheadの重みをfull_modelに転送
    head_model.set_weights(best_weights_s1)
    full_model.head.set_weights(head_model.get_weights())

    # メモリ解放
    del train_embs, val_embs, train_oh, val_oh, head_model
    gc.collect()

    # ----------------------------------------------------------
    # Stage 2: 全体 fine-tune
    # ----------------------------------------------------------
    print(f"\n[Stage 2] Full fine-tuning  (best S1 AUC={best_auc_s1:.4f})")
    full_model.unfreeze_perch()

    train_audio_ds = make_audio_dataset(train_df, n_classes, is_train=True)
    val_audio_ds   = make_audio_dataset(val_df,   n_classes, is_train=False)

    with strategy.scope():
        loss_fn = tf.keras.losses.BinaryCrossentropy()

    best_auc, best_weights = best_auc_s1, full_model.get_weights()

    for epoch in range(CFG.EPOCHS_FINETUNE):
        lr = cosine_decay_with_warmup(epoch, CFG.EPOCHS_FINETUNE, CFG.WARMUP_EPOCHS_FT, CFG.LR_FINETUNE)
        optimizer = tf.keras.optimizers.Adam(lr)
        total_loss = 0.0

        for waveforms, labels in tqdm(train_audio_ds, desc=f"[F{fold}] S2 Epoch {epoch+1}/{CFG.EPOCHS_FINETUNE}"):
            total_loss += train_step(full_model, optimizer, loss_fn, waveforms, labels).numpy()

        val_auc = compute_auc(full_model, val_audio_ds)
        print(f"  lr={lr:.2e}  loss={total_loss:.4f}  val_auc={val_auc:.4f}")
        wandb.log({"fold": fold, "epoch": epoch + CFG.EPOCHS_HEAD, "stage": "finetune",
                   "lr": lr, "train_loss": total_loss, "val_auc": val_auc})

        if val_auc > best_auc:
            best_auc, best_weights = val_auc, full_model.get_weights()

    # ベストモデルを保存
    full_model.set_weights(best_weights)
    full_model.save_weights(str(CFG.OUTPUT_DIR / f"best_model_fold{fold}.weights.h5"))
    print(f"Fold {fold} best val AUC: {best_auc:.4f}")

    wandb.log({"fold": fold, "best_val_auc": best_auc})
    wandb.finish()

    del full_model
    gc.collect()

    return best_auc


# ============================================================
# Kaggle Models アップロード
# ============================================================
def upload_to_kaggle_models(output_dir: Path, mean_auc: float) -> None:
    """学習済みモデル一式をKaggle Modelsにアップロード"""
    print(f"\nUploading to Kaggle Models: {CFG.KAGGLE_MODEL_HANDLE}")
    version_notes = (
        f"{CFG.EXP_NAME}/{CFG.CHILD_EXP} | "
        f"OOF ROC-AUC: {mean_auc:.4f} | "
        f"Perch 2.0 fine-tune | {CFG.N_FOLDS} folds"
    )
    kagglehub.model_upload(
        handle=CFG.KAGGLE_MODEL_HANDLE,
        local_model_dir=str(output_dir),
        version_notes=version_notes,
    )
    print("Upload complete!")


# ============================================================
# Main
# ============================================================
def main():
    strategy = setup_gpu()
    seed_everything(CFG.SEED)
    CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # wandb ログイン（Kaggle Secrets に WANDB_API_KEY を設定しておくこと）
    wandb.login(key=os.environ.get("WANDB_API_KEY", ""))

    # Perch 2.0 ダウンロード
    print("Downloading Perch 2.0...")
    perch_path = kagglehub.model_download(CFG.PERCH_HANDLE)
    print(f"Perch path: {perch_path}")

    # データ読み込み
    train_csv  = pd.read_csv(CFG.DATA_DIR / "train.csv")
    labels_csv = pd.read_csv(CFG.DATA_DIR / "train_soundscapes_labels.csv")
    _, label2idx, _ = load_taxonomy()

    # データ準備・結合
    audio_df      = prepare_train_audio(train_csv, label2idx)
    soundscape_df = prepare_train_soundscapes(labels_csv, label2idx)
    all_df = pd.concat([audio_df, soundscape_df], ignore_index=True)

    print(f"Total samples: {len(all_df)}  "
          f"(audio={len(audio_df)}, soundscape={len(soundscape_df)})")

    # StratifiedKFold
    skf      = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    oof_aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_df, all_df["label_idx"])):
        train_df = all_df.iloc[train_idx].reset_index(drop=True)
        val_df   = all_df.iloc[val_idx].reset_index(drop=True)

        fold_auc = train_fold(fold, train_df, val_df, perch_path, label2idx, strategy)
        oof_aucs.append(fold_auc)
        gc.collect()

    # 結果まとめ
    mean_auc = float(np.mean(oof_aucs))
    std_auc  = float(np.std(oof_aucs))
    print(f"\nOOF ROC-AUC: {mean_auc:.4f} ± {std_auc:.4f}")

    results = {"oof_aucs": oof_aucs, "mean_auc": mean_auc, "std_auc": std_auc}
    with open(CFG.OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Kaggle Models にアップロード
    upload_to_kaggle_models(CFG.OUTPUT_DIR, mean_auc)


if __name__ == "__main__":
    main()
