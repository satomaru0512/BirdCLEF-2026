"""
EXP000: BirdCLEF+ 2026 Baseline
- Backbone : Google Perch 2.0 (bird-vocalization-classifier/2)
- Strategy : Stage1 (Perch凍結, headのみ学習) → Stage2 (全体fine-tune)
- Data     : train_audio + labeled train_soundscapes
- CV       : StratifiedKFold (5 folds)
- Tracking : Weights & Biases
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
# Config
# ============================================================
class CFG:
    EXP_NAME   = "EXP000"
    CHILD_EXP  = "child-exp000"

    # Audio
    SAMPLE_RATE = 32000
    DURATION    = 5                          # seconds per chunk
    N_SAMPLES   = SAMPLE_RATE * DURATION     # 160_000

    # Model
    PERCH_HANDLE = (
        "google/bird-vocalization-classifier"
        "/tensorFlow2/bird-vocalization-classifier/2"
    )

    # Data paths (Kaggle環境)
    DATA_DIR            = Path("/kaggle/input/birdclef-2026")
    TRAIN_AUDIO_DIR     = DATA_DIR / "train_audio"
    TRAIN_SOUNDSCAPES_DIR = DATA_DIR / "train_soundscapes"
    OUTPUT_DIR          = Path(f"./outputs/{CHILD_EXP}")

    # Classes
    N_CLASSES = 234

    # Training
    BATCH_SIZE      = 32
    EPOCHS_HEAD     = 10   # Stage1: headのみ
    EPOCHS_FINETUNE = 5    # Stage2: 全体fine-tune
    LR_HEAD         = 1e-3
    LR_FINETUNE     = 1e-5

    # Cross-Validation
    N_FOLDS = 5
    SEED    = 42

    # wandb
    WANDB_PROJECT = "birdclef-2026"


# ============================================================
# Utilities
# ============================================================
def seed_everything(seed: int = CFG.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def hms_to_seconds(hms: str) -> int:
    """'HH:MM:SS' → 秒数"""
    parts = hms.split(":")
    return sum(int(x) * (60 ** i) for i, x in enumerate(reversed(parts)))


# ============================================================
# Data Preparation
# ============================================================
def load_taxonomy():
    tax = pd.read_csv(CFG.DATA_DIR / "taxonomy.csv")
    label2idx = {label: idx for idx, label in enumerate(tax["primary_label"])}
    idx2label = {idx: label for label, idx in label2idx.items()}
    return tax, label2idx, idx2label


def prepare_train_audio(train_csv: pd.DataFrame, label2idx: dict) -> pd.DataFrame:
    """train_audio のメタデータを整形"""
    df = train_csv.copy()
    df["label_idx"] = df["primary_label"].map(label2idx)
    df = df.dropna(subset=["label_idx"])
    df["label_idx"] = df["label_idx"].astype(int)
    df["filepath"]  = df["filename"].apply(lambda x: str(CFG.TRAIN_AUDIO_DIR / x))
    df["start_sec"] = 0
    return df[["filepath", "primary_label", "label_idx", "start_sec"]].reset_index(drop=True)


def prepare_train_soundscapes(labels_csv: pd.DataFrame, label2idx: dict) -> pd.DataFrame:
    """train_soundscapes_labels.csv を5秒セグメント単位に展開"""
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
# Audio Loading
# ============================================================
def load_audio_chunk(filepath: str, start_sec: float = 0) -> np.ndarray:
    """音声ファイルから5秒チャンクを読み込む"""
    try:
        info       = sf.info(filepath)
        orig_sr    = info.samplerate
        start_frame = int(start_sec * orig_sr)
        n_frames   = int(CFG.DURATION * orig_sr)

        audio, _ = sf.read(filepath, start=start_frame, frames=n_frames)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)          # stereo → mono
        if orig_sr != CFG.SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=CFG.SAMPLE_RATE)

        # パディング or トリミング
        if len(audio) < CFG.N_SAMPLES:
            audio = np.pad(audio, (0, CFG.N_SAMPLES - len(audio)))
        else:
            audio = audio[: CFG.N_SAMPLES]

        return audio.astype(np.float32)

    except Exception:
        return np.zeros(CFG.N_SAMPLES, dtype=np.float32)


# ============================================================
# Model
# ============================================================
class PerchClassifier(tf.keras.Model):
    """Perch 2.0 backbone + 分類ヘッド"""

    def __init__(self, perch_path: str, n_classes: int):
        super().__init__()
        self.perch = tf.saved_model.load(perch_path)
        self.head  = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(n_classes, activation="sigmoid"),
        ], name="head")

    def call(self, waveform, training: bool = False):
        # NOTE: Perch 2.0 の API が異なる場合はここを修正
        outputs    = self.perch.infer(waveform)
        embeddings = outputs["embeddings"]          # (batch, embedding_dim)
        return self.head(embeddings, training=training)

    def freeze_perch(self):
        self.perch.trainable = False

    def unfreeze_perch(self):
        self.perch.trainable = True


# ============================================================
# tf.data.Dataset
# ============================================================
def make_dataset(df: pd.DataFrame, label2idx: dict, is_train: bool = True):
    n_classes = len(label2idx)
    rows = df.to_dict("records")
    if is_train:
        random.shuffle(rows)

    def generator():
        for row in rows:
            audio = load_audio_chunk(row["filepath"], row["start_sec"])
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
    return ds.batch(CFG.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ============================================================
# Evaluation
# ============================================================
def compute_auc(model: PerchClassifier, dataset) -> float:
    """Macro ROC-AUC（ラベルが存在するクラスのみ）"""
    all_preds, all_labels = [], []
    for waveforms, labels in dataset:
        preds = model(waveforms, training=False).numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    valid = all_labels.sum(axis=0) > 0
    if valid.sum() == 0:
        return 0.0
    return roc_auc_score(all_labels[:, valid], all_preds[:, valid], average="macro")


# ============================================================
# Fold Training
# ============================================================
def train_fold(
    fold: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    perch_path: str,
    label2idx: dict,
) -> float:
    print(f"\n{'='*50}\nFold {fold}\n{'='*50}")

    run = wandb.init(
        project=CFG.WANDB_PROJECT,
        name=f"{CFG.EXP_NAME}-{CFG.CHILD_EXP}-fold{fold}",
        config={k: v for k, v in vars(CFG).items() if not k.startswith("_")},
        reinit=True,
    )

    train_ds = make_dataset(train_df, label2idx, is_train=True)
    val_ds   = make_dataset(val_df,   label2idx, is_train=False)

    model   = PerchClassifier(perch_path=perch_path, n_classes=CFG.N_CLASSES)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    best_auc, best_weights = 0.0, None

    # ----------------------------------------------------------
    # Stage 1: head のみ学習（Perch凍結）
    # ----------------------------------------------------------
    model.freeze_perch()
    optimizer = tf.keras.optimizers.Adam(CFG.LR_HEAD)

    for epoch in range(CFG.EPOCHS_HEAD):
        total_loss = 0.0
        for waveforms, labels in tqdm(train_ds, desc=f"[F{fold}] Head {epoch+1}/{CFG.EPOCHS_HEAD}"):
            with tf.GradientTape() as tape:
                preds = model(waveforms, training=True)
                loss  = loss_fn(labels, preds)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            total_loss += loss.numpy()

        val_auc = compute_auc(model, val_ds)
        print(f"  loss={total_loss:.4f}  val_auc={val_auc:.4f}")
        wandb.log({"fold": fold, "epoch": epoch, "stage": "head",
                   "train_loss": total_loss, "val_auc": val_auc})

        if val_auc > best_auc:
            best_auc, best_weights = val_auc, model.get_weights()

    # ----------------------------------------------------------
    # Stage 2: 全体 fine-tune
    # ----------------------------------------------------------
    model.unfreeze_perch()
    optimizer = tf.keras.optimizers.Adam(CFG.LR_FINETUNE)

    for epoch in range(CFG.EPOCHS_FINETUNE):
        total_loss = 0.0
        for waveforms, labels in tqdm(train_ds, desc=f"[F{fold}] FT {epoch+1}/{CFG.EPOCHS_FINETUNE}"):
            with tf.GradientTape() as tape:
                preds = model(waveforms, training=True)
                loss  = loss_fn(labels, preds)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            total_loss += loss.numpy()

        val_auc = compute_auc(model, val_ds)
        print(f"  loss={total_loss:.4f}  val_auc={val_auc:.4f}")
        wandb.log({"fold": fold, "epoch": epoch + CFG.EPOCHS_HEAD, "stage": "finetune",
                   "train_loss": total_loss, "val_auc": val_auc})

        if val_auc > best_auc:
            best_auc, best_weights = val_auc, model.get_weights()

    # ベストモデルを保存
    model.set_weights(best_weights)
    model.save_weights(str(CFG.OUTPUT_DIR / f"best_model_fold{fold}.weights.h5"))
    print(f"Fold {fold} best val AUC: {best_auc:.4f}")

    wandb.log({"fold": fold, "best_val_auc": best_auc})
    wandb.finish()

    return best_auc


# ============================================================
# Main
# ============================================================
def main():
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

        fold_auc = train_fold(fold, train_df, val_df, perch_path, label2idx)
        oof_aucs.append(fold_auc)
        gc.collect()

    # 結果まとめ
    mean_auc = float(np.mean(oof_aucs))
    std_auc  = float(np.std(oof_aucs))
    print(f"\nOOF ROC-AUC: {mean_auc:.4f} ± {std_auc:.4f}")

    results = {"oof_aucs": oof_aucs, "mean_auc": mean_auc, "std_auc": std_auc}
    with open(CFG.OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
