# Baseline-Conditioned Model

材料開発における「あるベース条件から、成分・工程条件を少し振ったら特性がどう変わるか」を予測・説明するためのプロトタイプです。

このリポジトリでは、単なる絶対値予測ではなく、以下のような意思決定支援を目指します。

- ベース条件からの変化量 `delta_y` を予測する
- どの変更がどれくらい寄与していそうかを局所線形に説明する
- 予測を支える近傍データ・変更方向のカバレッジを示す
- 同じ組織・設備・製品領域の知見と、他組織から借りた知見を分けて見せる
- 不確実性や外挿リスクを明示する

## 想定するユーザー体験

例：

> ベース条件Aに対して、Cを +0.03、Moを +0.10、焼戻し温度を -20℃ にすると、強度は +35 MPa 程度と予測されます。  
> ただし、このベース近傍で Mo と焼戻し温度を同時に振った事例が少ないため、不確実性は高めです。  
> 参考データの多くは同じ製品領域ですが、同じ組織内のデータは少ないです。

## 中心となる考え方

```text
Delta y ≈ beta(base_x, organization, product_family)^T delta_x
```

- `base_x`: ベース材料・工程条件
- `delta_x`: そこからの変更量
- `delta_y`: 目的特性の変化量
- `beta(...)`: ベース条件や組織によって変わる局所的な係数・感度

ユーザーには局所線形な説明を見せつつ、内部では以下のようなモデルを比較します。

- global absolute model
- global delta model
- local linear regression
- local linear regression with partial pooling
- Gaussian process absolute model
- Gaussian process delta model
- GPX-inspired / varying-coefficient model

## ドキュメント

- [Implementation brief](docs/implementation-brief.md): Codex向けの実装要件
- [Model comparison plan](docs/model-comparison.md): 比較対象モデルと評価指標
- [Decision UI spec](docs/ui-spec.md): 意思決定UIで見せたい情報
- [Codex task prompt](docs/codex-task.md): Codexに投げるためのタスク文

## MVP

最初のMVPでは、合成データで以下を比較します。

1. Global Delta Linear
2. Local Linear Regression
3. GP Absolute Model
4. GP Delta Model

評価は `delta_y` のRMSEだけでなく、候補ランキング、符号正解率、不確実性、適用可能領域の警告も見る方針です。

## Reference policy

技術調査では一次情報・公式ドキュメント・論文・公式実装を優先します。Qiita / Zenn は実装方針や根拠として使わない方針です。
