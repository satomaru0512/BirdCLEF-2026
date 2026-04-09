# idea メモ

---

## bird26-reproduce-perch-protossm-resssm-inf-train.ipynb

**参照元**: `idea/bird26-reproduce-perch-protossm-resssm-inf-train.ipynb`
**実行確認日**: 2026-04-01（papermill ログより）

---

### アーキテクチャ概要（3層構造）

```
[Perch v2]  60秒音声 → 12×5秒ウィンドウ → logits(234) + embedding(1536)
     ↓
[ProtoSSM v4]  embedding を時系列SSMで処理 + MLPプローブ
     ↓
[ResidualSSM]  第1パスの系統的誤差を補正
     ↓
[後処理]  温度スケーリング・per-class閾値・TTA・平滑化
```

---

### Perch 推論（`infer_perch_with_embeddings`）

- 60秒 ogg → `(12, WINDOW_SAMPLES)` に分割（N_WINDOWS=12）
- TensorFlow SavedModel で推論: `outputs["label"]`(logits) + `outputs["embedding"]`(1536次元)
- BirdClassifier全クラス → コンペ234クラスへ `MAPPED_BC_INDICES` でマッピング
- **属プロキシ**: BirdClassifierに対応クラスがない種は同属スコアの max/mean で代用
- バッチサイズ: `batch_files=16`（ファイル単位）

---

### ProtoSSM v4

| パラメータ | 値 |
|-----------|-----|
| d_model | 320 |
| d_state | 32 |
| n_ssm_layers | 4（双方向） |
| dropout | 0.12 |
| cross_attention_heads | 8 |
| メタデータ埋め込み | サイト(20語彙) + 時間帯 → 24次元 |

**損失関数**:
```
L = L_BCE(label_smoothing=0.03)
  + 0.15 × L_distill(Perch知識蒸留)
  + 0.15 × L_proto_margin(コントラスティブ)
  + L_family(分類族補助)
```

**学習設定**:
- epochs=80, lr=8e-4, CosineAnnealingWarmRestarts(period=20)
- SWA: 65%エポック時点から開始, lr=4e-4
- Mixup(alpha=0.4) + CutMix(時間軸), Focal Loss(gamma=2.5)
- OOF: GroupKFold(n_splits=5)

---

### MLPプローブ（クラスワイズ）

- PCA(128次元)圧縮した embedding を入力
- 各クラスに MLP(256→128) + LogReg を学習
- 特徴: embedding(128) + Perchスコア + 事前確率 + 時系列特徴(7) + インタラクション特徴(3)
- グリッドサーチ: pca_dim=[64,128,256], C=[0.5,0.75,1.0], alpha=[0.30,0.45,0.60]
- 最適: pca_dim=128, C=0.75, alpha=0.45

---

### ResidualSSM

- 軽量SSM（d_model=128, 2層）
- 入力: embedding + 第1パスlogits → 補正デルタを出力
- アンサンブルへの寄与: `final += 0.35 × correction`
- Wall time安全機構: 4分経過で自動スキップ

---

### スコア融合パイプライン

```python
# Step1: ProtoSSM (TTA: shifts=[0,±1,±2] の平均)
proto_scores_flat    # (N×12, 234)

# Step2: 事前確率融合
base_scores, prior_scores = fuse_scores_with_tables(perch_scores, site, hour)

# Step3: MLPプローブ (alpha=0.45)
mlp_scores = get_vectorized_mlp_scores(Z_TEST, ...)

# Step4: アンサンブル
final = 0.5 * proto_scores + 0.5 * mlp_scores

# Step5: ResidualSSM補正
final += 0.35 * residual_correction

# Step6: 後処理
# per-taxon 温度スケーリング → per-class閾値[0.25-0.70] → rank_aware_power=0.4 → delta_shift平滑化(alpha=0.20)
```

---

### キャッシュ設計（重要）

**キャッシュ対象**: `train_soundscapes` の Perch 出力のみ

```
学習ノートブック（GPU）
  → Perch で train_soundscapes 全件処理
  → full_perch_meta.parquet + full_perch_arrays.npz に保存
  → /kaggle/input/perch-meta/ としてKaggle Datasetにアップロード

推論ノートブック（CPU）
  → /kaggle/input/perch-meta/ からキャッシュ読み込み（train Perch再実行不要）
  → test_soundscapes は毎回 Perch 推論（約15〜20分 / 600ファイル）
  → 学習済みモデル（ProtoSSM/MLP/ResidualSSM）で推論
```

**推論時間内訳（600ファイル本番）**:

| ステップ | 時間 |
|---------|------|
| ライブラリ・モデルロード | ~150秒 |
| trainキャッシュ読込（parquet/npz） | ~217秒 |
| test Perch推論（600ファイル） | ~600秒 |
| ProtoSSM+MLP+後処理 | ~100秒 |
| **合計** | **約18〜20分** |

→ **90分制限に対して十分余裕あり**

---

