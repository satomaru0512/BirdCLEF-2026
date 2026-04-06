# シングルモデル改善アイディア

**日付**: 2025-11-15
**現在のベストスコア**: LB 0.66 (EXP014: SigLip + 5-Head + Consistency Loss)
**目標**: LB 0.67-0.68 (シングルモデルで)

---

## 📚 背景: K_mat氏の記事から得られた洞察

### 参照記事
- `docs/articles/kmat1.md`: 基本的なハイパラ調整・アンサンブル
- `docs/articles/kmat2.md`: 特徴量エンジニアリング
- `docs/articles/kmat3.md`: Loss・Augmentation工夫
- `docs/articles/kmat4.md`: タスク特化モデル・パイプライン

### 重要な原則

#### 1. **物理現象を意識させるLoss** (kmat3)
> 『奥行きの変化が起きるのは色味の変化が大きいところ』という常識をモデルに意識させるためのLoss

**我々のConsistency Loss（EXP014）はこの原則に従っている:**
- `Dry_Dead_g + GDM_g ≈ Dry_Total_g`
- `Dry_Clover_g + Dry_Green_g ≈ GDM_g`
- **既に実装済み、効果確認済み（+0.02 LB）** ✅

#### 2. **ラベルからサブタスクを作る** (kmat3)
> 位置情報を微分して速度をサブタスクとして学習 (atmaCup18)

**我々のケースでの応用:**
```python
# Biomass components間の比率を学習
- Clover_ratio = Dry_Clover_g / GDM_g
- Dead_ratio = Dry_Dead_g / Dry_Total_g
- GDM_ratio = GDM_g / Dry_Total_g

# これらをサブタスクとして予測
# メリット: 比率は画像から直接読み取りやすい（緑の割合など）
```

#### 3. **特徴量の与え方を工夫** (kmat2)
> NNは数値スケールに敏感。特徴量は親切に与える。
> 「画像における座標系を追加特徴にすることもアリ」

**我々のケースでの応用:**
```python
# 画像座標位置を明示的に追加
position_left = create_position_encoding(side='left')   # [H, W, 2]
position_right = create_position_encoding(side='right') # [H, W, 2]

# モデル入力 = RGB(3ch) + Position(2ch) = 5ch入力
```

#### 4. **NNにとってわかりやすく与える** (kmat2)
> 深層学習はGBDTとは異なり、与え方によってもパフォーマンスが変わる

**特徴量の質による適切な混ぜ方:**
1. **同質**: `Concat(features_A, features_B)` (そのまま)
2. **少し異質**: `Concat(Conv(features_A), Conv(features_B))` (一層おいて)
3. **ランクが違う**: FiLMやattentionで与える

---

## 🎯 具体的な改善提案

### **提案1: Position-Aware Feature Concatenation** ⭐⭐⭐

#### コンセプト
画像の座標位置を明示的にモデルに与えることで、spatial biasを学習可能にする。

#### 実装方針

```python
# EXP025: Position Encoding + 5-Head + Consistency Loss
class PositionAwareBiomassModel(nn.Module):
    def __init__(self, model_name, dropout=0.15, hidden_dim=512):
        super().__init__()

        # SigLip backbone (frozen or gradual unfreeze)
        self.siglip = AutoModel.from_pretrained(model_name)

        # Position encoding用のレイヤー
        # 入力: (x, y) 座標 → 出力: 16次元の位置特徴
        self.pos_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )

        # Embedding dimension (1152 for SigLip) + position encoding (16)
        self.embed_dim = 1152 + 16
        self.concat_dim = self.embed_dim * 2  # Left + Right

        # 5つのMLP heads (same as EXP014)
        self.head_total = nn.Sequential(...)
        self.head_gdm = nn.Sequential(...)
        self.head_green = nn.Sequential(...)
        self.head_dead = nn.Sequential(...)
        self.head_clover = nn.Sequential(...)

    def create_position_grid(self, batch_size, device, side='left'):
        """
        位置グリッドを生成
        side='left': x座標は0.0-0.5
        side='right': x座標は0.5-1.0
        """
        if side == 'left':
            x_offset = 0.0
        else:  # right
            x_offset = 0.5

        # 正規化された座標: x in [0, 1], y in [0, 1]
        # 左半分: x in [0, 0.5], 右半分: x in [0.5, 1.0]
        pos_encoding = torch.tensor([[x_offset + 0.25, 0.5]],
                                    device=device, dtype=torch.float32)
        pos_encoding = pos_encoding.repeat(batch_size, 1)

        return pos_encoding  # [B, 2]

    def forward(self, pixel_values_left, pixel_values_right):
        batch_size = pixel_values_left.size(0)
        device = pixel_values_left.device

        # SigLip特徴量抽出
        feat_left = self.siglip.get_image_features(pixel_values=pixel_values_left)  # [B, 1152]
        feat_right = self.siglip.get_image_features(pixel_values=pixel_values_right)  # [B, 1152]

        # 位置情報を生成
        pos_left = self.create_position_grid(batch_size, device, side='left')  # [B, 2]
        pos_right = self.create_position_grid(batch_size, device, side='right')  # [B, 2]

        # 位置情報をエンコード
        pos_feat_left = self.pos_encoder(pos_left)   # [B, 16]
        pos_feat_right = self.pos_encoder(pos_right)  # [B, 16]

        # 画像特徴量と位置特徴量を結合
        feat_left_with_pos = torch.cat([feat_left, pos_feat_left], dim=1)   # [B, 1168]
        feat_right_with_pos = torch.cat([feat_right, pos_feat_right], dim=1)  # [B, 1168]

        # 左右を結合
        feat_concat = torch.cat([feat_left_with_pos, feat_right_with_pos], dim=1)  # [B, 2336]

        # 5つのheadで予測
        pred_total = self.head_total(feat_concat)
        pred_gdm = self.head_gdm(feat_concat)
        pred_green = self.head_green(feat_concat)
        pred_dead = self.head_dead(feat_concat)
        pred_clover = self.head_clover(feat_concat)

        return pred_green, pred_dead, pred_clover, pred_gdm, pred_total
```

#### 根拠
- kmat2: 「画像における座標系を追加特徴にすることもアリ」
- **牧草地の特性**:
  - 位置によってbiomass分布が異なる可能性（日当たり、カメラ歪みなど）
  - 左側と右側で体系的な違いがあるかもしれない
  - エッジ（画像端）での推定精度が落ちる可能性

#### 期待される効果
- **+0.01-0.02 LB**: 位置情報によるspatial biasの学習
- **Gap改善**: 位置は test でも変わらない → 汎化性向上

#### リスク
- **中**: 効かない可能性もある（牧草地が本当に位置依存するか不明）
- **実装コスト**: 中程度（モデル構造の変更が必要）

---

### **提案2: Biomass Ratio as Auxiliary Task** ⭐⭐

#### コンセプト
Biomass成分間の比率をサブタスクとして学習することで、より直感的な特徴（緑の割合など）を学習。

#### 実装方針

```python
# EXP026: 5-Head + Consistency Loss + Ratio Auxiliary Task
class BiomassWithRatioModel(nn.Module):
    def __init__(self, ...):
        super().__init__()

        # メイン: 5つのbiomass値を予測するhead (same as EXP014)
        self.head_total = nn.Sequential(...)
        self.head_gdm = nn.Sequential(...)
        self.head_green = nn.Sequential(...)
        self.head_dead = nn.Sequential(...)
        self.head_clover = nn.Sequential(...)

        # サブタスク: 3つの比率を予測するhead (NEW)
        self.head_clover_ratio = nn.Sequential(  # Clover / GDM
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 0-1の範囲に制限
        )
        self.head_dead_ratio = nn.Sequential(  # Dead / Total
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.head_gdm_ratio = nn.Sequential(  # GDM / Total
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, pixel_values_left, pixel_values_right, return_aux=False):
        # 特徴量抽出 + concat (same as EXP014)
        feat_concat = ...

        # メインタスク: 5つのbiomass値
        pred_green = self.head_green(feat_concat)
        pred_dead = self.head_dead(feat_concat)
        pred_clover = self.head_clover(feat_concat)
        pred_gdm = self.head_gdm(feat_concat)
        pred_total = self.head_total(feat_concat)

        if return_aux:
            # サブタスク: 比率予測
            clover_ratio = self.head_clover_ratio(feat_concat)  # Clover/GDM
            dead_ratio = self.head_dead_ratio(feat_concat)      # Dead/Total
            gdm_ratio = self.head_gdm_ratio(feat_concat)        # GDM/Total

            return (pred_green, pred_dead, pred_clover, pred_gdm, pred_total,
                    clover_ratio, dead_ratio, gdm_ratio)
        else:
            return pred_green, pred_dead, pred_clover, pred_gdm, pred_total


class BiomassWithRatioLoss(nn.Module):
    """
    Consistency-aware + Ratio auxiliary loss for biomass prediction.

    Combines:
    1. Prediction Loss: Weighted SmoothL1Loss for each of 5 targets
    2. Consistency Loss: Enforces physical constraints (from EXP014)
    3. Ratio Loss (NEW): Learns biomass component ratios
    """
    def __init__(self, r2_weights, consistency_weight=0.1, ratio_weight=0.05):
        super().__init__()
        self.r2_weights = r2_weights
        self.consistency_weight = consistency_weight
        self.ratio_weight = ratio_weight
        self.loss_fn = nn.SmoothL1Loss()

    def forward(self, pred_green, pred_dead, pred_clover, pred_gdm, pred_total,
                target_green, target_dead, target_clover, target_gdm, target_total,
                pred_clover_ratio=None, pred_dead_ratio=None, pred_gdm_ratio=None):

        # 1. Prediction losses (same as EXP014)
        loss_green = self.loss_fn(pred_green, target_green)
        loss_dead = self.loss_fn(pred_dead, target_dead)
        loss_clover = self.loss_fn(pred_clover, target_clover)
        loss_gdm = self.loss_fn(pred_gdm, target_gdm)
        loss_total = self.loss_fn(pred_total, target_total)

        pred_loss = (self.r2_weights['Dry_Green_g'] * loss_green +
                    self.r2_weights['Dry_Dead_g'] * loss_dead +
                    self.r2_weights['Dry_Clover_g'] * loss_clover +
                    self.r2_weights['GDM_g'] * loss_gdm +
                    self.r2_weights['Dry_Total_g'] * loss_total)

        # 2. Consistency losses (same as EXP014)
        pred_sum_dead_gdm = pred_dead + pred_gdm
        consistency_total = self.loss_fn(pred_sum_dead_gdm, pred_total)

        pred_sum_clover_green = pred_clover + pred_green
        consistency_gdm = self.loss_fn(pred_sum_clover_green, pred_gdm)

        consistency_loss = (0.5 * consistency_total + 0.2 * consistency_gdm)

        # 3. Ratio losses (NEW)
        if pred_clover_ratio is not None:
            # Target ratios (with epsilon to avoid division by zero)
            target_clover_ratio = target_clover / (target_gdm + 1e-6)
            target_dead_ratio = target_dead / (target_total + 1e-6)
            target_gdm_ratio = target_gdm / (target_total + 1e-6)

            # Clamp target ratios to [0, 1]
            target_clover_ratio = torch.clamp(target_clover_ratio, 0, 1)
            target_dead_ratio = torch.clamp(target_dead_ratio, 0, 1)
            target_gdm_ratio = torch.clamp(target_gdm_ratio, 0, 1)

            # Ratio losses
            loss_clover_ratio = self.loss_fn(pred_clover_ratio, target_clover_ratio)
            loss_dead_ratio = self.loss_fn(pred_dead_ratio, target_dead_ratio)
            loss_gdm_ratio = self.loss_fn(pred_gdm_ratio, target_gdm_ratio)

            ratio_loss = (loss_clover_ratio + loss_dead_ratio + loss_gdm_ratio) / 3
        else:
            ratio_loss = torch.tensor(0.0, device=pred_green.device)

        # Total loss
        total_loss = (pred_loss +
                     self.consistency_weight * consistency_loss +
                     self.ratio_weight * ratio_loss)

        return (total_loss, pred_loss, consistency_loss, ratio_loss,
                loss_total, loss_gdm, loss_green, loss_dead, loss_clover)
```

#### 根拠
- kmat3: 「ラベルからサブタスクを作ってもいい」（速度を微分から計算）
- **比率は画像から直接読み取りやすい**:
  - 緑の割合 (Clover/GDM)
  - 枯れ草の割合 (Dead/Total)
  - 緑色物質の割合 (GDM/Total)
- **EXP021/022の失敗から学ぶ**: metadata予測は distribution shift で失敗したが、biomass由来の比率ならテストセットでも一貫性がある

#### 期待される効果
- **+0.005-0.01 LB**: 比率学習による中間特徴の改善
- モデルが「緑の割合」のような直感的な特徴を学習

#### リスク
- **中**: EXP021/022で auxiliary task は効果なし or 逆効果だった
- **対策**: Ratio weight を小さく (0.05) から始める

---

### **提案3: EXP024 Training実行** ⭐⭐⭐

#### コンセプト
EXP024推論ノートブック（LB 0.64）で使われている**特徴量concat方式**を訓練。

#### 現在との違い

**EXP014 (現在のベスト, LB 0.66):**
```python
# 左右を別々に予測 → 加算
pred_left = model(pixel_values_left)
pred_right = model(pixel_values_right)
pred_final = pred_left + pred_right
```

**EXP024 (実装済み, 未訓練):**
```python
# 左右の特徴量をconcat → 1回予測
feat_left = siglip(pixel_values_left)
feat_right = siglip(pixel_values_right)
feat_concat = torch.cat([feat_left, feat_right], dim=1)
pred_final = heads(feat_concat)
```

#### 実装状況
- ✅ `EXP/EXP024/train.py`: 実装済み
- ✅ `EXP/EXP024/infer.py`: 実装済み (TTA付き)
- ✅ `EXP/EXP024/config/child-exp000.yaml`: 実装済み
- ⏳ **訓練のみ未実施**

#### 期待される効果
- **+0.01-0.02 LB**: 左右の特徴量相互作用を学習
- EXP024推論ノートブック（EVA02, LB 0.64）と同じアプローチ
- **確実性が高い**: 既存の成功パターン

#### リスク
- **低**: 既に実装済み、訓練するだけ
- EXP014の加算方式より悪化する可能性は低い

---

## 📊 優先順位

### **第1優先: EXP024 Training実行** ⭐⭐⭐
- **確実性**: 高（既存の成功パターン）
- **実装コスト**: 低（訓練のみ）
- **期待効果**: +0.01-0.02 LB
- **次のアクション**: `EXP/EXP024/train.py` を Colab で実行

### **第2優先: Position Encoding追加 (EXP025)** ⭐⭐⭐
- **確実性**: 中（kmat2の原則に基づく）
- **実装コスト**: 中（モデル構造変更）
- **期待効果**: +0.01 LB
- **次のアクション**: EXP025として実装

### **第3優先: Ratio Auxiliary Task (EXP026)** ⭐⭐
- **確実性**: 中〜低（EXP021/022の失敗経験）
- **実装コスト**: 中（Loss関数追加）
- **期待効果**: +0.005-0.01 LB
- **次のアクション**: EXP026として実装（慎重に）

---

## 🚫 避けるべきアプローチ

### 1. **Metadata の直接入力（平均値埋め）**
- **理由**: メタデータの分散が大きい → 平均値埋めは不自然
- **代替案**: Metadata を使わない、または auxiliary task として学習のみ使用

### 2. **TTA の追加強化**
- **理由**: EXP014で既に試したが効果なし (0.66 → 0.66)
- **分析**: Consistency Loss で既に予測が安定している

### 3. **さらなる Auxiliary Tasks (Species, Height, NDVI)**
- **理由**: EXP021/022 で効果なし or 逆効果
- **教訓**: Distribution shift により train-only features は効かない

### 4. **Focal Loss など複雑な Loss 関数**
- **理由**: EXP016 で CV は向上したが LB は悪化 (Gap -0.141)
- **教訓**: 訓練の難しいサンプル ≠ テストの難しいサンプル

---

## 📝 実験ログテンプレート

各提案を実装する際は、以下のフォーマットで記録：

```markdown
### EXP0XX: [実験名]

**Configuration:**
- Base: EXP014 (or EXP024)
- Key Change: [変更点]
- Hyperparameters: [主要パラメータ]

**Hypothesis:**
- [なぜこれが効くと思うか]
- Expected improvement: +X.XX LB

**Results:**
- OOF CV: X.XXX
- LB: X.XX
- Gap: -X.XXX

**Analysis:**
- [効いたか/効かなかったか]
- [なぜそうなったか]
- [次に何をすべきか]
```

---

## 🔗 参考資料

- K_mat氏の記事: `docs/articles/kmat1-4.md`
- 実験履歴: `EXP/EXP_SUMMARY.md`
- 競技概要: `docs/OVERVIEW.md`
- データセット: `docs/DATASET.md`

---

**作成日**: 2025-11-15
**最終更新**: 2025-11-15
