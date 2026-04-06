# State × Season 分析レポート

**作成日**: 2026-01-05
**目的**: Train データにおける State と Season の関係性を分析し、Test データに含まれる可能性のある組み合わせを推定する

---

## 1. 背景

### 1.1 問題提起

OOF 分析の結果、特定の State（特に NSW）で予測誤差が大きいことが判明した。この原因を調査する中で、Train データにおける State × Season の組み合わせに大きな偏りがあることが発見された。

### 1.2 論文の記述

Host Paper（HOSTPAPER.md）からの引用：

> **Abstract**: "The images were taken across multiple seasons and include a range of temperate pasture species."

> **Section 2.2**: "In terms of temporal coverage, the GrassClover dataset is concentrated between May and October, whereas **our dataset spans the entire year**."

> **Section 3.1**: "Data collection was conducted across 19 locations spanning four Australian states over a three-year period (2014–2017), representing **a wide range of soil, climate, and seasonal conditions**"

**重要**: 論文は「データセット全体として年間をカバー」と述べているが、「各 State で全季節をカバー」とは言っていない。

---

## 2. Train データの State × Season 分布

### 2.1 サンプル数

```
              Summer    Autumn    Winter    Spring    合計
    NSW         41        23         0        11       75
    Tas          0        29        38        71      138
    Vic          0         0        73        39      112
    WA           0         0        20        12       32
    ────────────────────────────────────────────────────────
    合計        41        52       131       133      357
```

### 2.2 視覚的表現

```
           Summer    Autumn    Winter    Spring
  NSW      ███ 41    ███ 23    ░░░  0    ███ 11
  Tas      ░░░  0    ███ 29    ███ 38    ███ 71
  Vic      ░░░  0    ░░░  0    ███ 73    ███ 39
  WA       ░░░  0    ░░░  0    ███ 20    ███ 12

  ███ = Train に存在
  ░░░ = Train に欠損
```

### 2.3 欠損している組み合わせ（6つ）

| 欠損組み合わせ | 備考 |
|--------------|------|
| NSW × Winter | NSW は Winter のみ欠損 |
| Tas × Summer | Tas は Summer のみ欠損 |
| Vic × Summer | Vic は Summer, Autumn 欠損 |
| Vic × Autumn | |
| WA × Summer | WA は Summer, Autumn 欠損 |
| WA × Autumn | |

---

## 3. 各 State × Season の Target 分布

### 3.1 NSW

| 季節 | N | Total | GDM | Green | Dead | Clover |
|--------|-----|-------|------|-------|------|--------|
| Summer | 41 | 71.0 | 60.7 | 60.5 | 10.2 | 0.2 |
| Autumn | 23 | 55.3 | 43.3 | 43.3 | 12.0 | 0.0 |
| Winter | 0 | - | - | - | - | - |
| Spring | 11 | 103.2 | 69.6 | 69.6 | 33.6 | 0.0 |

**NSW の特徴**:
- Clover = 0（ほぼゼロ）
- Spring が最大（Summer の 1.45 倍）
- 季節比率: Spring(1.45) > Summer(1.00) > Autumn(0.78)

### 3.2 Tas

| 季節 | N | Total | GDM | Green | Dead | Clover |
|--------|-----|-------|------|-------|------|--------|
| Summer | 0 | - | - | - | - | - |
| Autumn | 29 | 35.7 | 22.5 | 21.9 | 13.2 | 0.6 |
| Winter | 38 | 28.4 | 13.2 | 10.7 | 15.2 | 2.5 |
| Spring | 71 | 41.8 | 25.6 | 15.1 | 16.1 | 10.5 |

**Tas の特徴**:
- Dead が常に多い（13-16g）
- Clover が Spring で増加（10.5g）
- 季節比率: Spring(1.47) > Autumn(1.26) > Winter(1.00)

### 3.3 Vic

| 季節 | N | Total | GDM | Green | Dead | Clover |
|--------|-----|-------|------|-------|------|--------|
| Summer | 0 | - | - | - | - | - |
| Autumn | 0 | - | - | - | - | - |
| Winter | 73 | 34.6 | 26.1 | 22.3 | 8.5 | 3.7 |
| Spring | 39 | 57.8 | 44.7 | 31.3 | 13.1 | 13.4 |

**Vic の特徴**:
- Spring/Winter 比が最大（1.67 倍）
- Clover が多め（Spring で 13.4g）

### 3.4 WA

| 季節 | N | Total | GDM | Green | Dead | Clover |
|--------|-----|-------|------|-------|------|--------|
| Summer | 0 | - | - | - | - | - |
| Autumn | 0 | - | - | - | - | - |
| Winter | 20 | 29.8 | 29.8 | 3.9 | 0.0 | 25.9 |
| Spring | 12 | 34.0 | 34.0 | 18.3 | 0.0 | 15.7 |

**WA の特徴**:
- **Clover が主成分**（Winter で 25.9g = Total の 87%）
- Dead = 0（常にゼロ）
- GDM = Total（Dead がないため）

---

## 4. State ごとの特徴まとめ

### 4.1 Clover 比率

| State | Clover 平均 | Total に占める割合 |
|-------|------------|-------------------|
| NSW | 0.1g | 0.2% |
| Tas | 6.2g | 17.0% |
| Vic | 7.1g | 16.7% |
| **WA** | **22.1g** | **70.4%** |

### 4.2 モデルが学習する State 別平均

| State | Total 平均 | 季節カバレッジ |
|-------|-----------|---------------|
| NSW | 70.9g | Summer, Autumn, Spring |
| Tas | 36.8g | Autumn, Winter, Spring |
| Vic | 42.7g | Winter, Spring |
| WA | 31.4g | Winter, Spring |

---

## 5. 仮説: Test に含まれる可能性のある組み合わせ

### 5.1 論文の証拠

論文は「データセット全体として年間をカバー」と述べている。しかし Train には 6 つの State × Season 組み合わせが欠損している。

**可能性**:
1. 欠損は Test/Validation に含まれる
2. そもそも収集されなかった（州ごとの収集時期の偏り）
3. QC で除外された（3,187 サンプル中 2,025 が除外）

### 5.2 各欠損の影響予測

| 欠損組み合わせ | Train での学習 | Test 推定値 | 予測への影響 |
|--------------|---------------|------------|-------------|
| **Tas Summer** | Tas = 37g | 50-60g | **過小評価** |
| **Vic Summer** | Vic = 43g | 55-70g | **過小評価** |
| Vic Autumn | Vic = 43g | 45-50g | やや過小評価 |
| WA Summer | WA = 31g | 35-45g | 過小評価 |
| WA Autumn | WA = 31g | 30-35g | 影響小 |
| **NSW Winter** | NSW = 71g | 40-50g | **過大評価** |

### 5.3 推定根拠

NSW の季節比率（Summer = 1.0 基準）:
- Summer : 1.00 (= 71.0g)
- Autumn : 0.78 (= 55.3g)
- Spring : 1.45 (= 103.2g)

この比率を他の State に適用して欠損季節を推定。

---

## 6. 問題のメカニズム

### 6.1 モデルが学習すること

```
Tas の画像 → 低バイオマス (平均 37g)
Vic の画像 → 中バイオマス (平均 43g)
NSW の画像 → 高バイオマス (平均 71g)
```

### 6.2 Test で起こりうること

```
Tas Summer の画像がTestに含まれる場合:
  → モデル: "Tas っぽい画像 → 37g 程度"
  → 実際: Summer なので 50-60g
  → 結果: 過小評価（13-23g の誤差）
```

---

## 7. 対策としての学習戦略

### 7.1 戦略A: 疑似サンプル生成（Data Augmentation）


### 7.2 戦略B: State 分類器 + 推論時補正

1. **State 分類器を訓練**:
   - 画像から State (NSW/Tas/Vic/WA) を予測
   - 精度: 推定 85-95%（State は Species とほぼ 1:1 対応）

2. **推論時の補正**:
```python
# バイオマス予測
pred = model.predict(image)

# State 予測
state = state_classifier.predict(image)

# 高バイオマス + 非NSW の State → Summer の可能性
if pred > 50 and state in ['Tas', 'Vic']:
    # モデルは Summer を見たことがない
    # → 過小評価の可能性 → 上方修正
    pred = pred * 1.1  # 10% 補正
```

### 7.3 戦略C: 季節を考慮した Mixup

```python
# NSW Summer + Tas Spring を混ぜる
# → Tas Summer 的なサンプルを生成
alpha = 0.5
mixed_image = alpha * nsw_summer_img + (1-alpha) * tas_spring_img
mixed_label = alpha * nsw_summer_label + (1-alpha) * tas_spring_label * seasonal_factor
```

### 7.4 戦略D: Uncertainty-aware 予測

- MC Dropout で不確実性推定
- 見たことのない State×Season の組み合わせは不確実性が高いはず
- 高不確実性サンプルは保守的な予測（平均に近づける）

---

## 8. 関連する発見

### 8.1 State と Species の強い相関

Cramér's V = 0.792（非常に強い相関）

| State | 主な Species |
|-------|-------------|
| NSW | Lucerne, Fescue, Phalaris, Ryegrass |
| Tas | Ryegrass_Clover, Ryegrass, Clover, WhiteClover |
| Vic | Phalaris_Clover, Ryegrass_Clover, Phalaris_Ryegrass_Clover, Mixed |
| WA | SubcloverDalkeith, SubcloverLosa, Clover |

### 8.2 State と Height の相関

η² = 0.425（強い相関）

| State | Height 平均 |
|-------|------------|
| NSW | 14.9 cm |
| Tas | 7.9 cm |
| Vic | 7.7 cm |
| WA | 8.9 cm |

### 8.3 結論

**State は Species と Height を包含する「メタ変数」**

State を予測できれば、間接的に Species と Height の傾向も推定可能。

---

## 9. まとめ

### 9.1 発見事項

1. **Summer は NSW のみ**（41 サンプル = 全体の 11%）
2. **6 つの State×Season 組み合わせが欠損**
3. **State ごとに Target 分布が大きく異なる**
   - NSW: 高バイオマス、Clover なし
   - Tas: 低バイオマス、Dead 多い
   - WA: Clover が主成分

### 9.2 仮説

論文が「年間をカバー」と述べていることから、欠損組み合わせの一部は Test に含まれる可能性がある。特に Tas/Vic Summer が Test に含まれる場合、モデルは過小評価する。

### 9.3 推奨アクション

1. **State 分類器の実装**: 画像から State を推定
2. **推論時補正**: 推定 State + 予測値の組み合わせで妥当性チェック
3. **Data Augmentation**: 疑似 Summer サンプルを生成して訓練

---

## 10. 参考情報

### 10.1 オーストラリアの季節（南半球）

| 季節 | 月 |
|------|-----|
| Summer | 12月, 1月, 2月 |
| Autumn | 3月, 4月, 5月 |
| Winter | 6月, 7月, 8月 |
| Spring | 9月, 10月, 11月 |

### 10.2 QC で除外されたサンプル

論文より:
> "Of the initial collection of 3,187 samples, 1,162 passed the comprehensive quality control process."

2,025 サンプル（63.5%）が除外された。欠損している State×Season 組み合わせがこの中に含まれていた可能性もある。

---

## 11. State × Season vs Public LB 相関分析（2026-01-07 追加）

### 11.1 分析対象実験

| Exp | CV | LB | 説明 |
|-----|------|------|------|
| E014-005 | 0.759 | 0.66 | EXP014 baseline |
| E024-003 | 0.740 | 0.66 | EXP024 |
| E060-000 | 0.759 | 0.69 | EVA-CLIP baseline |
| E060-037 | 0.785 | 0.67 | EVA-CLIP 10-fold |
| E060-038 | 0.776 | 0.71 | DinoV3 ViT-Huge+ |
| E060-039 | 0.782 | 0.69 | DinoV3 + MSE |

### 11.2 State × Season vs LB 相関ランキング

| Rank | State_Season | n訓練 | 相関r | p値 | 解釈 |
|------|--------------|-------|-------|-----|------|
| **1** | **Vic_Spring** | 39 | **+0.768** | **0.075** | **境界で有意★** |
| 2 | Tas_Autumn | 29 | +0.612 | 0.196 | 強い相関 |
| 3 | Tas_Spring | 71 | +0.542 | 0.266 | 強い相関 |
| 4 | WA_Winter | 20 | +0.503 | 0.309 | 強い相関 |
| 5 | Tas_Winter | 38 | +0.356 | 0.488 | 弱い相関 |
| 6 | WA_Spring | 12 | +0.333 | 0.519 | 弱い相関 |
| 7 | NSW_Spring | 11 | +0.318 | 0.538 | 弱い相関 |
| 8 | NSW_Summer | 41 | +0.165 | 0.754 | ほぼ無相関 |
| 9 | NSW_Autumn | 23 | -0.325 | 0.530 | 弱い負相関 |
| 10 | Vic_Winter | 73 | -0.421 | 0.406 | 弱い負相関 |

### 11.3 State × Season 別 各実験の wR² 詳細

#### Vic_Spring (n=39) ★最重要★
| Exp | wR² | LB |
|-----|-----|-----|
| E060-038 | 0.853 | 0.71 |
| E060-039 | 0.851 | 0.69 |
| E060-000 | 0.839 | 0.69 |
| E014-005 | 0.826 | 0.66 |
| E024-003 | 0.791 | 0.66 |
| E060-037 | 0.784 | 0.67 |

#### Tas_Spring (n=71)
| Exp | wR² | LB |
|-----|-----|-----|
| E060-038 | 0.786 | 0.71 |
| E024-003 | 0.755 | 0.66 |
| E060-039 | 0.737 | 0.69 |
| E060-000 | 0.711 | 0.69 |
| E014-005 | 0.707 | 0.66 |
| E060-037 | 0.685 | 0.67 |

#### Tas_Autumn (n=29)
| Exp | wR² | LB |
|-----|-----|-----|
| E060-000 | 0.778 | 0.69 |
| E060-039 | 0.777 | 0.69 |
| E060-038 | 0.775 | 0.71 |
| E014-005 | 0.757 | 0.66 |
| E060-037 | 0.742 | 0.67 |
| E024-003 | 0.600 | 0.66 |

### 11.4 重要な発見

1. **Vic_Spring が最重要** (r = +0.768, p = 0.075)
   - exp038: wR² = 0.853 → LB 0.71
   - exp037: wR² = 0.784 → LB 0.67
   - **Vic_Spring を上げれば LB が上がる**

2. **Tas 全季節が正の相関**
   - Tas_Autumn: r = +0.612
   - Tas_Spring: r = +0.542
   - Tas_Winter: r = +0.356

3. **NSW は無関係または負相関**
   - NSW_Summer: r = +0.165（ほぼ無相関）
   - NSW_Autumn: r = -0.325（負相関）
   - **NSW を改善しても LB は上がらない**

4. **Vic_Winter は負相関** (r = -0.421)
   - exp037: wR² = 0.737 → LB 0.67（最低）
   - exp038: wR² = 0.643 → LB 0.71（最高）
   - **Vic_Winter を上げると LB が下がる可能性**

### 11.5 結論

**Public LB を上げるには:**
1. **Vic_Spring** を最優先で改善（相関 +0.768）
2. **Tas 系**（Spring/Autumn/Winter）を維持・改善
3. **NSW は無視しても OK**（LB に影響しない）
4. **Vic_Winter は上げすぎない**（負の相関）

**仮説の更新:**
- Public Test は **Vic_Spring** と **Tas 系** に偏っている可能性が高い
- 訓練データの欠損（Tas_Summer, Vic_Summer/Autumn）は Public Test には少ない可能性

---

## 12. 今後の調査項目

- [ ] State 分類器の精度検証
- [ ] 疑似 Summer サンプル生成の効果検証
- [ ] Test での State×Season 分布の推定（推論時に State 分類を実行）
- [x] OOF 誤差と State×Season の関係分析 → Section 11 で完了
- [ ] Vic_Spring に特化した改善策の検討
