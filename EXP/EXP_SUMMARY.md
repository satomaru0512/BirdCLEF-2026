# EXP_SUMMARY: BirdCLEF+ 2026

## Competition Status

| 項目 | 値 |
|------|----|
| Best OOF CV | - |
| Best LB | - (未提出) |
| Current Best Exp | EXP000/child-exp000 |

---

## 実験一覧

### EXP000: Perch 2.0 直接推論（ファインチューニングなし）

**概要**: Google Perch 2.0をそのまま推論に使用。ファインチューニングなし。
Perchの`output_0`（鳥類ロジット）をsigmoidし、eBirdコードでコンペ234種にマッピング。
非鳥類72種（昆虫・両生類・哺乳類・爬虫類）は確率0を出力。

**設定** (`child-exp000`):

| パラメータ | 値 |
|------------|-----|
| Backbone | Google Perch 2.0 |
| Strategy | ファインチューニングなし（直接推論） |
| 鳥類種数 | 162種 → Perch output_0 でマッピング |
| 非鳥類種数 | 72種 → 確率0 |
| 出力変換 | sigmoid(logits) |

**Results**:

| LB Score | LB-CV Gap |
|----------|-----------|
| - (未提出) | - |

**Kaggle Models**: なし（学習なし）

**観察・考察**:
- ファインチューニングなしのため学習コスト0
- 非鳥類72種は常に0のため、それらの評価スコアに影響
- Perchのラベルにない鳥類はunmatched（確率0）

---

## 得られた知見

（実験結果後に更新）

---

## 次のステップ候補

1. **LBスコア確認** → Perch直接推論のベースラインスコアを把握
2. **ヘッド学習** → Perch embeddingの上にheadを追加してファインチューニング
3. **非鳥類対応** → 非鳥類専用モデルの追加
