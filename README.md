# carprice

> **English summary for reviewers.** Everything below this section is in Japanese. This part is a
> condensed equivalent — enough to install the project, run it, and know where to look while reviewing.
> The Japanese text is the source of truth; if the two ever disagree, the Japanese wins.

## What this is

An internship project at INDX: automatic used-car price prediction. The deliverable is **`unfold`**, a
scikit-learn-compatible Python library. Used-car pricing is the flagship use case, not the scope — the
library itself is written to be domain-generic.

Structured columns (mileage, model year, accident history) do not fully determine a used-car price. The
remaining signal sits in unstructured data: the listing title, the equipment blurb, the photos. Classical
regression could only absorb that as dummy variables, which is where accuracy plateaued. `unfold` is the
attempt to close that gap with an LLM — but with the LLM placed *on top of* statistical models, not in
place of them.

Two features, plus a routing layer:

- **Feature A — `Feature`** turns an unstructured column into a typed column by declaration alone.
  Internally it is *embed → nearest-neighbour vote against the labels that already exist → confidence*.
  The LLM is only the fallback for rows whose confidence is low; it is not asked to write a feature every
  time.
- **Feature B — `LLMPredictor`** does not hand the LLM a raw record and ask for a price. LightGBM,
  XGBoost and a semantic k-NN solve it first; their predictions, plus similar vehicles whose actual price
  is known (retrieved per row, not pasted once), are passed to the LLM as *evidence*, and the LLM only
  makes the final call.
- **Confidence routing — `AdaptivePredictor`** decides *which rows* reach the LLM at all, using only
  signals available before any LLM call. A single threshold moves continuously between "call for every
  row" and "never call".

The project's plan changed three times (details in `CLAUDE.md`). An earlier plan — "let the LLM pick the
model" — **was dropped** at the design-document stage (`dialogs/unfold-landing.html`) and folded into the
two features above. **When reading older material in this repository, check which stage it belongs to.**

## Quick start — free, no API key, no data download

The 500-row excerpt committed to git is enough. This runs leakage checks → screening → Feature A → cost
estimation in one go, and **makes no LLM call by default**.

```bash
.venv/bin/python -m unfold.demo              # free
.venv/bin/python -m unfold.demo --run 20     # predicts 20 rows (6 go to the LLM; about $0.05)
.venv/bin/python -m pytest tests -q          # the LLM is stubbed, so tests pass without an API key
```

## Setup

Python 3.12 and a venv. Do not use the system Python.

```bash
# Ubuntu (compute node)
sudo apt update && sudo apt install -y python3.12-venv fonts-noto-cjk
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# macOS
brew install python@3.12 libomp          # libomp is required by xgboost/lightgbm
/usr/local/opt/python@3.12/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`fonts-noto-cjk` is only for Japanese labels in plots.

**There are two environments on purpose.** `.venv` (main) holds pandas / LightGBM / XGBoost / `unfold`
itself; `.venv-embed` (`bash scripts/setup_embed_env.sh`) holds torch / sentence-transformers / TabPFN.
**`unfold` runs without torch** — the default encoder is character TF-IDF — and torch is kept out of the
main environment so that an accidental dependency on it cannot pass the tests unnoticed.

Credentials go in `.env` (not tracked; template in `.env.example`). `.venv/bin/python
scripts/check_api_key.py` verifies it, `--no-call` checks it without spending anything. Reading the
contents of `.env` is blocked by policy and by hooks. The 1.4 GB raw dataset is not in the repository;
`python scripts/download_data.py` fetches it from Kaggle.

## What to look at when reviewing

| Path | What it is |
|---|---|
| `unfold/feature.py` | Feature A: embedding → neighbour classification → LLM fallback |
| `unfold/predictor.py` | Feature B: statistical models and retrieved cases as evidence for the LLM |
| `unfold/adaptive.py` | Confidence routing; `plan()` / `curve()` / `approve()` |
| `unfold/leakage.py` | Duplicate-record detection, run automatically on fit / predict |
| `unfold/llm.py` | **The only place that calls the Claude API.** Disk cache, cost accounting, parallelism |
| `unfold/demo.py` | End-to-end entry point (`python -m unfold.demo`) |
| `tests/` | `pytest tests -q` |
| `scripts/` | Data fetching and measurement scripts; `caafe.py` is a CAAFE-equivalent baseline that **executes LLM-written code**, so run it only on your own data and machine |

Three properties are deliberate and worth checking:

- **Leakage.** The same vehicle is cross-posted to several regions in the Craigslist data (42% of rows are
  duplicates; one VIN appears up to 261 times at an identical price). Keeping them across a random split
  inflates R² from 0.880 to 0.914, and **the more text you use as a feature, the larger the inflation**.
  `check_duplicates` / `check_overlap` warn rather than raise, since duplication is sometimes intentional.
- **Answer leakage in free text.** 43.7% of Craigslist `description` fields state the asking price, so
  `unfold` **masks monetary amounts by default** before passing long text to the LLM.
- **Cost and provenance.** LLM responses are cached to disk, so re-measuring after a code change is free.
  `explain()` / `confidence()` / `examples()` / `cost()` / `provenance()` trace every cell back to
  human / model / llm origin, the evidence used, and what it cost.

Measured results, design decisions and history live in `docs/` (Japanese), with `docs/README.md` as the
index — it also explains the `P3` / `S6` / `R2` notation. `docs/PRD.md` carries an English summary at the
top as well.

---

株式会社INDXのインターンにて作成する中古車価格の自動予測プログラムです。

成果物は **`unfold`** という scikit-learn 互換の Python ライブラリで、機能は2つあります。

- **機能A（`Feature`）** — 画像や自由記述などの非構造な列を、宣言するだけで型のついた列にする
- **機能B（`LLMPredictor`）** — 統計モデルに先に解かせ、その予測値と類似事例を証拠として
  LLM に渡し、最終判断だけさせる

そこに **信頼度ルーティング（`AdaptivePredictor`）** が乗り、「どの行を LLM に回すか」を決めます。

構想は3段階で変わってきました（詳細は `CLAUDE.md`）。当初は「LLM にモデル選択をさせる」案も
並べていましたが、伊藤さんの設計書（`dialogs/unfold-landing.html`）の段階で**その案は消え**、
上の2機能に統合されています。**古い資料を読むときはどの時点のものかに注意してください。**

---

## セットアップ

### 1. Python環境

Python 3.12 を使います。

Ubuntu（計算ノード）:

```bash
sudo apt update && sudo apt install -y python3.12-venv fonts-noto-cjk
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
source .venv/bin/activate
```

macOS:

```bash
brew install python@3.12 libomp                        # libomp は xgboost/lightgbm に必要
/usr/local/opt/python@3.12/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
source .venv/bin/activate
```

`fonts-noto-cjk` はグラフの日本語表示に必要です（macOS は Hiragino Sans が標準搭載）。

**環境は2つに分かれています。**

| | 中身 | いつ使うか |
|---|---|---|
| `.venv`（主環境） | pandas / LightGBM / XGBoost / `unfold` 本体 | ふだんはこちら |
| `.venv-embed` | torch / sentence-transformers / TabPFN | 埋め込みの計算と TabPFN のときだけ |

**`unfold` は torch 無しで動きます**（既定のエンコーダは文字 TF-IDF）。
主環境に torch を入れないのは、うっかり torch へ依存してもテストが通ってしまうのを
防ぐためです。`.venv-embed` が要る作業をするときだけ、次を実行します。

```bash
bash scripts/setup_embed_env.sh        # requirements-embed.txt から構築
.venv-embed/bin/python scripts/embed_text.py       # 使うときは python を切り替える
```

詳しい経緯は `docs/2026-09-01-embed-env-rebuild.md` にあります。

### 2. データの取得

このリポジトリには 1.4GB の生データを含めていません。

```bash
kaggle auth login              # 初回のみ（ブラウザで認証）
python scripts/download_data.py
```

取得せずに動作確認だけしたい場合は `sampledata/sample/` の抜粋データが使えます。

### 3. APIキー

`.env` は作成済みです。エディタで開き、`ANTHROPIC_API_KEY=` の右にキーを貼るだけで動きます。
（clone 直後で `.env` が無い場合は `cp .env.example .env` で作る。`.env` は git 管理外）

```
ANTHROPIC_API_KEY=sk-ant-api03-...
```

キーは https://console.anthropic.com/settings/keys で発行します。
貼ったら疎通確認する（実際に1回だけ Claude を呼びます。費用は $0.001 程度）:

```bash
.venv/bin/python scripts/check_api_key.py
.venv/bin/python scripts/check_api_key.py --no-call   # 課金なしで読み込みだけ確認
```

## unfold（ライブラリ本体）

`unfold/` が設計書（`dialogs/unfold-landing.html`）の実装です。
機能は2つ（`Feature` / `LLMPredictor`）で、そこに「どの行を LLM に回すか」を
決める信頼度ルーティング（`AdaptivePredictor`）が乗ります。

### まず動かす

**APIキーもデータ取得も要りません。**git に入っている 500 行の抜粋を使い、
リーク検査 → スクリーニング → 機能A → 費用の見積もり、までを一度に流します。
**既定では LLM を1回も呼ばない**ので無料です。

```bash
.venv/bin/python -m unfold.demo              # 無料
.venv/bin/python -m unfold.demo --run 20     # 20行を予測（うち6行が LLM に回る。約 $0.05）
```

### 機能A — `Feature`（特徴量生成）

非構造列を、宣言するだけで型のついた列にします。中身は「埋め込み → 近傍で分類」で、
LLM は確信度が低い行のフォールバックにしか出てきません。

```python
from unfold import Feature

df["グレード"] = Feature(
    source="タイトル",
    type="category",
    values=["G", "Z", "X", "G クエロ"],
).fit_transform(df)
```

### 機能B — `LLMPredictor`（LLM Predict）

LLM に生データを渡して当てさせるのではなく、**LightGBM・XGBoost・近傍検索に
先に解かせ、その予測値と「実際の価格が分かっている似た事例」を証拠として渡し、
最終判断だけさせます**。類似事例は行ごとに引き直します（フューショット）。

```python
from unfold import LLMPredictor

model = LLMPredictor(
    target="車両本体価格_万円", unit="万円",
    numeric=["車齢", "走行距離_km"], categorical=["グレード名"],
    text="装備テキスト",
)
pred = model.fit(train_df).predict(test_df)
```

正解ラベルを別途用意する必要はありません。訓練データの価格がそのまま
フューショットの例になります。

### 信頼度ルーティング — `AdaptivePredictor`（どの行を LLM に回すか）

機能B は**1行につき1回 LLM を呼ぶ**ので、行数がそのまま費用と時間になります
（6万行なら約 $516・約17時間）。そこで「LLM を呼ぶ前に手に入る信号」だけで
呼ぶ行を選び、残りは統計モデルに任せます。閾値ひとつで
「全行呼ぶ」と「1行も呼ばない」の間を連続に動かせます。

```python
from unfold import AdaptivePredictor

model = AdaptivePredictor(target="price", unit="USD",
                          numeric=["age", "odometer"],
                          categorical=["manufacturer", "state"], text="model",
                          escalate_rate=0.3)     # 信号の強い上位3割だけ回す
model.fit(train)
model.plan(test)      # 呼ぶ前に「何行・いくら・何秒」（LLM を呼ばないので無料）
pred = model.predict(test)
model.curve(test)     # 割合を振ったときの精度・費用・レイテンシ
model.approve()       # LLM の答えを承認 → 次回はその行を呼ばずに返る
```

既定の信号は **LightGBM と XGBoost の予測の食い違い**（統計モデル自身が
迷っている行に回す）で、実測で最良でした。`signal="unseen"` にすると
訓練データに無い値（Craigslist なら未知の `model`）を含む行を優先します。

Craigslist の 60 行で上位30%だけ回した実測は
**MAE 2,767 → 2,476（費用 $0.155）**。詳しくは `docs/2026-09-01-adaptive.md`。

### 使う前に — そのデータで効くかを確かめる

機能B は「テキストが価格を左右するデータ」でしか効きません（シエンタでは負け、
Craigslist では勝ちました）。**LLM を呼ぶ前に**、そのデータでテキストが効くかを
無料で判定できます。

```python
from unfold import screen

print(screen(df, target="price", text="model", unit="USD",
             numeric=["age", "odometer"], categorical=["manufacturer", "state"]))
# テキスト寄与率 15.4%（閾値 10%） → 判定: 試す価値あり
```

### 自由記述を渡すときの注意

出品テキストには**売り値がそのまま書いてあることが多い**です
（Craigslist の `description` は 43.7% の行が該当）。そのまま渡すと予測ではなく
答えの読み取りになるので、`unfold` は**既定で金額を伏せます**。

```python
model = LLMPredictor(..., long_text="description")  # 金額は自動で〈金額〉に伏せられる
```

詳しくは `docs/2026-09-01-description-leak.md`。

### 共通 — リーク検査（重複レコードの検知）

**同じ車が train と test の両方に入っていると、予測ではなく答えの読み取りに
なります。**Craigslist のデータは同じ車が複数の地域に出稿されており、
重複を残したまま評価すると R² が 0.880 → 0.914 に水増しされました。
しかも**テキストを特徴量にするほど水増しが大きくなる**ので、
`unfold` は fit / predict のときに自動で検知して警告します。

```python
from unfold import check_duplicates, check_overlap

print(check_duplicates(df, keys=["VIN"]))        # 1つの表の中の重複
print(check_overlap(train, test))                # train と test にまたがる重複
```

例外にはしません（重複が意図的なこともあるため）。うるさければ
`LLMPredictor(..., check_leakage=False)` で切れます。

### 共通 — 来歴（provenance）の検査 API

どちらの機能も、設計書どおり「なぜその値になったか」を辿れます。

```python
model.explain(0)      # その行の来歴（証拠・参照した事例・LLM の理由）
model.confidence()    # 行ごとの確信度
model.examples()      # 推論に使った類似事例
model.cost()          # かかった費用と1行あたりの単価
model.provenance()    # 全行の来歴を1つの表に
```

LLM の応答は `sampledata/processed/llm_cache/` にディスクキャッシュされます。
**同じ行を2度課金しない**ので、実装をいじって測り直すのは無料です。

測定結果は `docs/` の日付つきファイルにあります。テストは次のとおり
（LLM 部分は差し替えているので **APIキー無しでも全部通ります**）。

```bash
.venv/bin/python -m pytest tests -q
```

以下開発用

## ドキュメント

測定結果・設計判断・経緯はすべて `docs/` にあります（24本）。
**`docs/README.md` が入口**で、どれから読むかと、本文に出てくる
`P3` `S6` `R2` といった記号の意味をまとめてあります。
基本、バイブコーディングを行う上でのメモ置き場になってしまっているので、人間が読むには適さないものが多いです。

- **結果だけ知りたい** → `docs/progress-log.md` 
- **なぜそう作ったのか** → `docs/PRD.md`

## dialogsについて

これまでの議論によって作られた議事録や、イメージをまとめてあるフォルダです。
`entrysheet.md` はこのインターンに参加するにあたって最初のアイデアがまとめられてあります。
`20260812中古車...` はAI作成の議事録で、方針が端的にまとまってあります。
`unfold-landing.html` は伊藤さんによるアイデアをまとめた設計書の一つになります。



## sampledataについて

分析をかけるであろうデータです。「再現できるかどうか」で4つに分けています。

| ディレクトリ | git管理 | 中身 |
|---|---|---|
| `raw/` | ✕ | 外部から再取得できる巨大な生データ |
| `scraped/` | ○ | 自分たちで集めた、再取得できないデータ |
| `processed/` | ✕ | 分析途中の中間データ。コードで再生成できる |
| `sample/` | ○ | 動作確認用の小さな抜粋 |

- `raw/vehicles.csv` — Kaggleから取得したCraigslistの中古車データ（1.4GB, 英語）。
  https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data
- `scraped/usedsientaL.edit.omit.csv` — 2026年3月頃にカーセンサーから
  octoparseでデータスクレイピングしてきたものです（203KB, 日本語）。
- `sample/vehicles_sample500.csv` — `raw/vehicles.csv` の先頭500件（1.5MB）。

生データを git に入れていないのは、git が全履歴を全員に複製する仕組みで、
一度巨大ファイルを入れると削除しても永久に残るためです。
データそのものではなく、取得手順（`scripts/download_data.py`）を管理しています。
