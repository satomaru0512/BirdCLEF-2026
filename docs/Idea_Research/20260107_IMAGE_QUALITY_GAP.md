# Train/Test間の画質差を吸収するアプローチ

作成日: 2026-01-07

---

## 1. 背景・問題の発見

### 1.1 データ収集期間の非対称性

**HOSTPAPER (arxiv:2510.22916v1) より:**
> "Data collection was conducted across 19 locations spanning four Australian states over a **three-year period (2014–2017)**"

**しかし、train.csvの実態:**
- **全て2015年のみ** (357画像 × 5ターゲット = 1785行)
- 2014年、2016年、2017年のデータは含まれていない

**推測:**
- 通常、古いデータをtrain、新しいデータをtestにする
- **testは2016-2017年のデータである可能性が高い**
- 新しい年度 = カメラ機材の更新 → 画質向上の可能性

### 1.2 Train内の画質ばらつき

論文Table 1によると、使用カメラは多様:
- Apple iPhone 4, iPhone 5s
- Canon IXUS 125 HS
- NIKON COOLPIX AW110
- OLYMPUS SP510UZ
- Sony D5833
- HTC 0PJA10

**画像は2000×1000に正規化されているが、元の解像度・品質は大きく異なる。**

---

## 2. 定量分析

### 2.1 Laplacian Varianceによるシャープネス測定

```python
def get_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()
```

### 2.2 結果（train画像50枚サンプル）

| Metric | Value |
|--------|-------|
| Min | 806.8 |
| Max | 9231.9 |
| Mean | 4056.9 |
| Std | 2234.6 |
| **Ratio (max/min)** | **11.4x** |

**11.4倍の差は非常に大きい。**

### 2.3 具体例

**最もシャープな画像 (Top 5):**
| Path | Sharpness |
|------|-----------|
| train/ID1119761112.jpg | 9231.9 |
| train/ID1215977190.jpg | 8819.0 |
| train/ID1058383417.jpg | 8184.1 |

**最もボケた画像 (Top 5):**
| Path | Sharpness |
|------|-----------|
| train/ID1159071020.jpg | 806.8 |
| train/ID1149598723.jpg | 949.1 |
| train/ID1121692672.jpg | 990.5 |

---

## 3. 問題の整理

### 3.1 現状の限界

EXP060では既に以下のaugmentationを使用:
- `A.GaussianBlur(blur_limit=(3, 7))`
- `A.GaussNoise(var_limit=(5.0, 35.0))`
- `A.ImageCompression(quality_lower=65, quality_upper=100)`
- `A.MotionBlur(blur_limit=(3, 7))`

**しかし、これらは「一律にランダム適用」であり、画質差を意識した処理ではない。**

### 3.2 考えられる問題シナリオ

**シナリオA: Testが高画質の場合**
- Trainの低品質画像で学習 → 高品質Testでドメインシフト
- モデルが「ボケた画像」に過適合している可能性

**シナリオB: Testも画質混在の場合**
- 画質によって予測精度がばらつく
- 現状のaugmentationである程度対応できている可能性

---

## 4. 改善案

### 4.1 案1: Sharpness-aware TTA（推奨・低リスク）

**概要:** 推論時に画像の画質を測定し、適応的にTTAを適用

```python
def inference_with_quality_tta(model, img):
    sharpness = get_sharpness(img)

    preds = []
    preds.append(model(img))  # Original

    if sharpness > 5000:  # 高画質画像
        # train分布（低画質混在）に近づけるため少しぼかす
        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        preds.append(model(blurred))
    elif sharpness < 2000:  # 低画質画像
        # シャープ化
        sharpened = sharpen(img)
        preds.append(model(sharpened))

    return np.mean(preds, axis=0)
```

**メリット:**
- infer.pyのみの変更
- train.pyは変更不要
- 効果がなければ簡単にロールバック可能

### 4.2 案2: Sharpnessを補助入力に追加

**概要:** モデルに「この画像はボケている」という情報を明示的に与える

```python
class BiomassModel(nn.Module):
    def __init__(self, ...):
        self.sharpness_embed = nn.Linear(1, 64)

    def forward(self, img, sharpness):
        features = self.backbone(img)
        sharpness_feat = self.sharpness_embed(sharpness.unsqueeze(-1))
        features = features + sharpness_feat
        ...
```

**メリット:**
- モデルが画質を考慮した予測を学習
- Height, NDVIと同様の補助情報として機能

**デメリット:**
- train.py / infer.py 両方の変更が必要
- 新しい実験が必要

### 4.3 案3: 画質ベースのサンプル重み付け

**概要:** 高シャープネス画像（testに近い）のlossを重視

```python
sharpness_scores = precompute_sharpness(train_images)
sample_weights = normalize(sharpness_scores)  # 高シャープネス = 高重み

# Training loop
loss = criterion(pred, target) * sample_weights[idx]
```

**メリット:**
- test分布に近い画像を重視した学習

**デメリット:**
- 低品質画像の情報を捨てることになる
- データ量が実質的に減少

### 4.4 案4: Super Resolution前処理

**概要:** Real-ESRGAN / SwinIR で低品質画像を向上させてから学習

**デメリット:**
- Kaggle推論時間制約（9時間）に収まらない可能性
- 計算コストが非常に高い

---

## 5. 推奨アプローチ

### 第1優先: 案1 (Sharpness-aware TTA)

- リスク最小、実装最速
- 効果測定が容易
- LBで効果を確認してから他のアプローチを検討

### 第2優先: 案2 (Sharpnessを補助入力)

- 案1で効果が見られた場合、より本格的に取り組む価値あり
- Height補助入力と同様のアプローチで実装可能

---

## 6. 実験計画

1. **まずtrain全画像のシャープネス分布を確認**
   - Sampling_Dateとの相関（特定日が低品質か？）
   - Stateとの相関（特定地域が低品質か？）

2. **案1をinfer.pyに実装してLB確認**
   - ベースライン: 現在の最高LB
   - 比較: Sharpness-aware TTA適用後

3. **効果があれば案2を検討**

---

## 7. 参考：既存のaugmentation設定

EXP060での画質関連augmentation:
```python
A.GaussNoise(var_limit=(5.0, 35.0), p=0.35)
A.ImageCompression(quality_lower=65, quality_upper=100, p=0.3)
A.GaussianBlur(blur_limit=(3, 7), p=各所で0.15-0.2)
A.MotionBlur(blur_limit=(3, 7), p=0.5)  # OneOfで選択
```

これらは「一律ランダム適用」であり、画質に応じた適応的処理ではない点に注意。

---

## 8. 全画像分析結果（2026-01-07追加）

### 8.1 全357画像のシャープネス分布

| Metric | Value |
|--------|-------|
| Min | 181.4 |
| Max | 10,029.6 |
| Mean | 3,586.3 |
| Median | 3,008.8 |
| Std | 2,031.4 |
| **Ratio (max/min)** | **55.3x** |

初回サンプル（50枚）より更に大きな差（55倍）が確認された。

### 8.2 Sampling_Dateによるクラスタ

**低画質グループ (mean < 2500)** - 主に冬季（7-9月）:

| Date | Mean | Count | 特徴 |
|------|------|-------|------|
| 2015/9/29 | 1,571 | 11 | 最低品質 |
| 2015/7/8 | 1,658 | 12 | |
| 2015/9/30 | 1,732 | 10 | |
| 2015/8/21 | 1,918 | 8 | |
| 2015/9/1 | 2,049 | 12 | |
| 2015/7/1 | 2,383 | 19 | |
| 2015/8/18 | 2,386 | 10 | |
| 2015/11/9 | 2,488 | 20 | |

**高画質グループ (mean > 5000)** - 主に秋（10月）、夏（2月）:

| Date | Mean | Count | 特徴 |
|------|------|-------|------|
| 2015/10/13 | 7,142 | 11 | 最高品質 |
| 2015/2/25 | 5,989 | 9 | |
| 2015/10/14 | 5,795 | 7 | |
| 2015/5/7 | 5,717 | 13 | |
| 2015/5/18 | 5,571 | 22 | |

### 8.3 Stateによる傾向

| State | Mean | Count | 特徴 |
|-------|------|-------|------|
| WA | 1,870 | 32 | **極端に低品質** |
| Vic | 2,926 | 112 | 低〜中品質 |
| Tas | 4,109 | 138 | 中〜高品質 |
| NSW | 4,344 | 75 | 高品質 |

**WA（西オーストラリア）が極端に低品質** → 特定のカメラ/撮影者の可能性

### 8.4 考察

- 画質差は**撮影日**と強く相関（同日は同じカメラ/撮影者の可能性）
- **冬季（7-9月）**は低品質、**秋（10月）・夏（2月）**は高品質
- **State**とも相関があるが、撮影日との交絡の可能性あり
- testが2016-2017年だとすると、**同じ季節・地域の傾向が引き継がれる**可能性

---

## 9. 実装: --blur_tta オプション（2026-01-07）

### 9.1 実装内容

`EXP060/infer.py`に以下を追加：

1. **TestBiomassDatasetSplitBlurTTA クラス**
   - 3ビュー: original, light blur (r), medium blur (r×2)
   - PILのGaussianBlurを使用

2. **コマンドライン引数**
   - `--blur_tta`: Blur TTAを有効化
   - `--blur_radius`: ブラー半径（デフォルト: 0.5）

### 9.2 使用方法

```bash
python infer.py --config config.yaml --model_dir outputs/ --blur_tta --blur_radius 0.5
```

### 9.3 デフォルト値の根拠

- `blur_radius=0.5`: 軽いブラーから始める
- 3ビュー: [original, r=0.5, r=1.0]
- trainのmedian sharpness (3009) に高画質test画像を近づける想定

### 9.4 実験推奨

1. まず`--blur_radius 0.5`でLB確認
2. 効果があれば`0.3`, `0.7`, `1.0`で探索
3. 効果がなければBlur TTAは不採用

---

## 10. 撮影方向の一貫性について（2026-01-07追加）

### 10.1 発見

- **影の方向が揃っている**: 80%の画像で影が上→下に伸びている
- **南半球の撮影**: 太陽は北にあるため、南→北方向に撮影
- **論文にも記載**: 撮影方向の統一について言及あり

### 10.2 TTAへの影響

- **Flip/Rotate TTAで精度が下がる**
- モデルが方向情報を暗黙的に学習している証拠
- 方向情報は予測に有用（影の長さ ∝ 草丈など）

### 10.3 方針

- **学習時**: Flip/Rotateあり（データ量確保、357枚は少ない）
- **推論時**: Flip/Rotate TTAなし（方向情報保持）
- 極端な変更は避ける（357枚で実験リスクが高い）

### 10.4 トレードオフ

| 観点 | Flip/Rotateあり | Flip/Rotateなし |
|-----|----------------|----------------|
| データ量 | 357 × 8 = 実質2856枚 | 357枚のみ |
| 方向情報 | 壊れる | 保持される |
| 物理的正しさ | 草の量は変わらない | - |

現在の「学習でFlip/Rotate、推論でTTAなし」が妥協点。
