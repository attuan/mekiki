# PRD — unfold / 中古車価格予測 AutoML エージェント

**版**: 0.6(人間がまともに読めないものになっていたため、大幅に書き換え、修正)

（以下は英語話者のレビュアー向けの要約です。日本語の本文はその下から始まります。）

> **English summary for reviewers.** The document below is in Japanese and is the source of truth; this
> section is a condensed equivalent of it. Version 0.6.

### Why

A used-car price is not determined by the tabular columns alone (mileage, model year, accident history).
Much of the price-moving information sits in **unstructured data** — the listing title, the equipment
blurb, the photos. Classical regression could only encode that as dummy variables, which was believed to
be the accuracy ceiling. That observation is the starting point of the 2026-08-12 meeting.

`unfold` is a Python library that makes such unstructured data usable through a **scikit-learn-compatible
API** (`fit` / `transform` / `predict`). Used-car pricing is the flagship use case; the library itself is
designed to be domain-generic.

The plan changed three times — an entry-sheet-stage recommendation app, then three ways of inserting an
LLM (direct prediction / model selection / feature generation), and finally the current design document
(`dialogs/unfold-landing.html`) which folded those three into **Feature A (feature generation)** and
**Feature B (LLM Predict)**. Older material must be read with its stage in mind.

### Scope

**In scope:** `Feature`; `LLMPredictor`; confidence routing (escalate only rows below a confidence
threshold); provenance (origin, confidence, cited cases and cost per cell, reachable via `explain()`);
caching so the same row is never billed twice; and the evaluation harness for measuring all of it
(**already implemented**).

**Out of scope:** LLM-driven model selection (prior work exists, but it was dropped at the design-document
stage); scraping itself (the data is already at hand); an end-user UI or recommendation ranking (that was
the entry-sheet-stage idea, not the current one); distributed processing and large-scale workloads (tens
of thousands of rows is the working target).

### Product goal

For tabular data containing unstructured columns (free text, images), reach prediction accuracy at least
equal to hand-written preprocessing rules **without writing those rules**, in a form whose **cost and
provenance can be explained**.

### Feature A — `Feature` (feature generation)

Turn an image or free-text column into a typed column by declaration alone. The pipeline is: use existing
ground-truth labels as reference points → one vector per record, **caching mandatory** (a row is embedded
once) → nearest-neighbour search producing a confidence from similarity and label agreement → return the
label if confident, with no LLM call → otherwise escalate to the LLM and write its answer back as a new
label candidate.

Output types: `binary` / `category` / `int` / `float` / `ordinal` / `multilabel` / `embedding`. Input may
be a single column (`source="description"`) or several (`source=["image", "description"]`).

**Narrow down what the LLM is asked to do.** Measured, an embed-then-classify setup only adds **20–45
USD** worth of improvement over character TF-IDF — too little to justify running embeddings for that
alone. Where a difference should appear is in **normalising notation itself** (mapping `f-150 raptor`,
`f150 raptor` and `f 150 raptor` onto the same type), and **that is the LLM's job within Feature A**.
Implement the split explicitly: embeddings as the substrate for neighbour search, the LLM for unifying
notation. **Do not commit to a single representation** — embeddings and character TF-IDF are good at
different things, so using both must be a first-class option (measured best overall at 2,596).

### Feature B — `LLMPredictor`

Rather than handing the LLM a raw record, XGBoost, LightGBM and a semantic k-NN solve it first, and their
predictions plus similar cases are passed as *evidence* so the LLM only makes the final call.

Requirements: aggregate several statistical models' outputs into what the LLM sees; **retrieve few-shot
examples per row** rather than pasting one fixed set; and settle how a `predict_proba`-oriented
classification design applies to used-car prices, which are continuous.

### Feature C? — confidence routing (`AdaptivePredictor`)

A single threshold must move continuously between "send every row to the LLM" and "send none", and moving
it must make **accuracy, latency and cost visible at the same time**. Rows the LLM answered are queued as
label candidates; approving them widens the fast path next time (active learning). This is the answer to
the "make LLM prediction lighter by using a cache" item raised in the 2026-08-12 meeting.

### Provenance and inspection API

`model.explain(X)` returns a cell's origin (human / model / llm), confidence, cited cases and cost;
`model.confidence(X)` the per-row confidence, visible before committing to a run; `model.examples(X)` the
cases used for inference; `model.cost(X)` a pre-run cost estimate. Features are versioned together with
the labels that produced them, so that re-running three months later shows whether the numbers still match
and which labels changed.

### Non-functional requirements

scikit-learn-compatible API (`fit` / `transform` / `fit_transform` / `predict` / `predict_proba`); the
embedding model, LLM, statistical models and storage are all swappable by configuration; reproducibility
is enforced by the library (fixed seeds, fit inside the fold); **leakage prevention** — duplicate-record
detection (`unfold/leakage.py`), which must at minimum warn, because unstructured text as a feature makes
duplicates inflate the score; cost — never bill the same row twice, and estimate before running;
environment — Python ≥ 3.10, with embedding computation allowed to live in an isolated environment to
avoid dependency conflicts.

Comparison against prior work is in `docs/related-work.md`.

---

## なぜ作るのか

例えば、中古車の価格は、表になっている項目（走行距離・年式・修復歴）だけでは決まらない。
実際にはタイトル文や装備の羅列、写真といった**非構造データ**に価格を左右する情報が入っている。
従来の回帰分析ではそれを「ダミー変数」（該当すれば1、しなければ0の列）にするしかなく、
そこが精度の頭打ちの原因だと考えられていた。これが 8/12 ミーティングの出発点である。

`unfold` は、その非構造データを **scikit-learn 互換の API**（`fit` / `transform` / `predict`）で
扱えるようにする Python ライブラリである。中古車価格はその代表ユースケースであり、
ライブラリ自体は汎用に設計する。

### 構想の変遷（どの段階の資料かを見分けるために）

| 段階 | 資料 | 内容 |
|---|---|---|
| 1. エントリーシート | `dialogs/entrysheet.md` | 学生向けの中古車推薦アプリ。LLM は主役ではない |
| 2. 8/12 ミーティング | `dialogs/2026...ミーティング.md` | LLM の差し込み方3方式（①直接予測 ②モデル選択 ③特徴量生成）を全部試して比較 |
| 3. 伊藤さん仕様書【現在地】 | `dialogs/unfold-landing.html` | 3方式を **機能A（特徴量生成）** と **機能B（LLM Predict）** の2つに統合 |

会議を重ねるうちに、作ろうとしているものが変わっている。イメージの共有がされていないものも多い。

---

---

## 対象ユーザーとユースケース

| ユーザー | やりたいこと |
|---|---|
| データ分析担当（社内） | 自由記述の混じった手持ちデータを、前処理コードを書かずにモデルに載せたい |
| 中古車の価格査定 | タイトル・装備・写真から相場を出し、なぜその値かを説明したい（代表ユースケース） |
| 汎用（churn 予測など） | 仕様書に例がある。中古車に限定しない |

```python
df["vehicle_type"] = Feature(
    source="image", type="category", values=["sedan", "suv", "truck", "van"],
).fit_transform(df)
```

### スコープ

**やること**

- 機能A: `Feature` — 非構造列を型付き列に変換する
- 機能B: `LLMPredictor` — 統計モデルの予測と類似事例を証拠として LLM に渡し、最終判断させる
- 信頼度ルーティング（adaptive）: 確信度が閾値を下回った行だけ LLM に回す
- 来歴（provenance）: 各セルの由来・確信度・参照事例・費用の保持と `explain()`
- キャッシュ: 同じ行に二度課金しない
- 上記を測るための評価基盤（**実装済み**）

**やらないこと**

- 方式②（LLM によるモデル選択の自動化）— 先行研究はあるが、仕様書の段階で外れている
- スクレイピング機能そのもの（データはすでに手元にある）
- エンドユーザー向け UI・推薦ランキング（エントリーシート段階の構想であり、現在地ではない）
- 分散処理・大規模スケール（当面は数万行が対象）

---

## 目的とゴール

### プロダクトゴール

非構造データ（自由記述テキスト・画像）を含む表形式データに対して、
**人手の前処理ルールを書かずに**、それを書いた場合と同等以上の予測精度を、
**費用と来歴が説明できる形で**出せるライブラリを作る。

##  機能要件

### 機能A — `Feature`（特徴量生成）

**目的**: 画像・自由記述などの非構造列を、宣言だけで型付き列にする。

| # | 段階 | 要件 |
|---|---|---|
| 01 | 教師ラベル | 既存の正解ラベルを参照点にする。|
| 02 | 埋め込み | 1レコード1ベクトル。**キャッシュ必須**（1行の埋め込みは一度だけ計算する） |
| 03 | 近傍探索 | 類似度＋ラベル一致度から確信度を出す。|
| 04 | 確信できれば | ラベルを返す。LLM 呼び出しなし |
| 05 | 確信できなければ | LLM にエスカレーションし、答えを新しいラベル候補として書き戻す |

出力型: `binary` / `category` / `int` / `float` / `ordinal` / `multilabel` / `embedding`
入力: 単一列（`source="description"`）または複数列（`source=["image", "description"]`）

**LLM に期待する仕事を絞り込む。** 実測では「埋め込み → 近傍で分類」だけの構成が
文字 TF-IDF から得る上積みは **20〜45 USD** しかない。この程度なら埋め込みを回す価値は薄い。
差が出るとすれば **表記体系そのものの正規化**（`f-150 raptor` / `f150 raptor` / `f 150 raptor` を
同じ型に落とす）であり、**そこが機能A における LLM の担当範囲**である。
埋め込みは近傍探索の土台、LLM は表記の統合、と役割を分けて実装すること。

**表現は1つに固定しない。**埋め込みと文字 TF-IDF は得意な場面が違うので、
**併用を既定の選択肢として持つ**こと（実測で全体最良の 2,596）。


### 機能B — `LLMPredictor`

**目的**: LLM に生レコードを渡して当てさせるのではなく、
XGBoost・LightGBM・semantic k-NN に先に解かせ、その予測値と類似事例を「証拠」として渡し、
最終判断だけさせる。

```python
model = LLMPredictor(target="resale_band", models=[xgb, lgbm, semantic_knn],
                     examples="semantic", n_examples=5)
```

要件:

- 複数の統計モデルの出力を集約して LLM に渡すこと
- 少数事例（few-shot）は**行ごとに検索して差し替える**こと（一度貼って使い回さない）

### 機能C? - 信頼度ルーティング（`AdaptivePredictor`）

閾値ひとつで「全行を LLM に投げる」と「1行も投げない」の間を連続的に動かせること。
閾値を動かしたときに **精度・レイテンシ・費用の3つが同時に見える**こと。
LLM が答えた行は教師ラベル候補としてキューされ、承認すると次回は高速パスが広がる（能動学習）。

これは 8/12 ミーティングで課題に挙がった「キャッシュ活用による LLM 予測の軽量化」への回答にあたる。

### 来歴（provenance）と検査 API

| API | 返すもの |
|---|---|
| `model.explain(X)` | そのセルの由来（human / model / llm）・確信度・参照した事例・費用 |
| `model.confidence(X)` | 行ごとの確信度（実行を確定する前に見られること） |
| `model.examples(X)` | 推論に使った事例 |
| `model.cost(X)` | 実行前の費用見積もり |

特徴量はそれを作ったラベルとセットでバージョン管理し、
3ヶ月後の再実行で同じ数字が出るか、どのラベルが変わったかが分かること。

### 非機能要件

| 項目 | 要件 |
|---|---|
| API | scikit-learn 互換（`fit` / `transform` / `fit_transform` / `predict` / `predict_proba`） |
| 差し替え | 埋め込みモデル・LLM・統計モデル・ストレージは設定で差し替えられること |
| 再現性 | seed 固定・fold 内 fit の原則をライブラリ側で守らせること |
| **リーク防止** | 重複レコードの検知（`unfold/leakage.py`）。非構造テキストを特徴量にすると重複が精度を水増しするため、少なくとも警告を出せること |
| 費用 | 同じ行に二度課金しない。実行前に見積もりを出せる |
| 環境 | Python ≥ 3.10。埋め込み計算は依存衝突を避けるため隔離環境に分けてよい |


## 先行事例との対比

`docs/related-work.md` に整理されています。

