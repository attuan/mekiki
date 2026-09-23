<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/attuan/mekiki/main/assets/logo_dark.svg">
  <img src="https://raw.githubusercontent.com/attuan/mekiki/main/assets/logo_light.svg" width="520" alt="mekiki — a trained eye for the columns your model cannot read">
</picture>
</p>

<p align="center">
  <a href="https://attuan.mintlify.site/"><img alt="Documentation" src="https://img.shields.io/badge/docs-attuan.mintlify.site-2a78d6"></a>
  <img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-2a78d6">
  <img alt="scikit-learn compatible" src="https://img.shields.io/badge/scikit--learn-compatible-2a78d6">
  <img alt="LLM: Claude by default, others through LiteLLM" src="https://img.shields.io/badge/LLM-Claude%20%C2%B7%20LiteLLM-e8692f">
  <a href="#one-key-for-everything-the-vercel-ai-gateway"><img alt="Vercel AI Gateway supported" src="https://img.shields.io/badge/Vercel%20AI%20Gateway-supported-000000"></a>
  <a href="#a-cheaper-middle-tier--jev"><img alt="Jev (TypeSafe AI) supported" src="https://img.shields.io/badge/Jev%20(TypeSafe%20AI)-supported-e8692f"></a>
  <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-59636e">
</p>

`mekiki` (目利き, "a trained eye") is a scikit-learn-compatible Python library that turns
unstructured columns (free text today; images are in the design) into typed features and puts an LLM
**on top of** statistical models rather than in place of them.

Structured columns rarely determine a target on their own. In used-car pricing, the listing
title and the seller's description carry what the mileage and model year do not. Classical
regression could only absorb that as dummy variables, which is where accuracy plateaued.
`mekiki` closes that gap with an LLM, but keeps the LLM where it is cheap, auditable and
easy to switch off:

<table>
  <tr>
    <td align="center" valign="top" width="33%">
      <img src="https://raw.githubusercontent.com/attuan/mekiki/main/assets/icon_diagnose.svg" width="40" height="40" alt=""><br>
      <b><code>diagnose()</code></b><br>
      <sub>Hand it a DataFrame and a target; get back a recommended configuration as objects you can pass straight in.</sub>
    </td>
    <td align="center" valign="top" width="33%">
      <img src="https://raw.githubusercontent.com/attuan/mekiki/main/assets/icon_typed_column.svg" width="40" height="40" alt=""><br>
      <b><code>SemanticEncoder</code></b><br>
      <sub>Declare an unstructured column as a typed one. Embed → neighbour vote → confidence; the LLM is only a fallback for unsure rows.</sub>
    </td>
    <td align="center" valign="top" width="33%">
      <img src="https://raw.githubusercontent.com/attuan/mekiki/main/assets/icon_knowledge_column.svg" width="40" height="40" alt=""><br>
      <b><code>KnowledgeEncoder</code></b><br>
      <sub>Columns the table lacks, from the LLM's world knowledge. Asked once per key value, kept only if they help.</sub>
    </td>
  </tr>
  <tr>
    <td align="center" valign="top" width="33%">
      <img src="https://raw.githubusercontent.com/attuan/mekiki/main/assets/icon_evidence_predictor.svg" width="40" height="40" alt=""><br>
      <b><code>EvidenceRegressor</code> / <code>EvidenceClassifier</code></b><br>
      <sub>LightGBM, XGBoost and a semantic k-NN solve it first; the LLM makes the final call from their evidence. Regression and classification.</sub>
    </td>
    <td align="center" valign="top" width="33%">
      <img src="https://raw.githubusercontent.com/attuan/mekiki/main/assets/icon_routing.svg" width="40" height="40" alt=""><br>
      <b>Confidence routing</b><br>
      <sub>One ratio, <code>escalate_rate</code>, decides which rows reach the LLM at all, from signals that exist before any call.</sub>
    </td>
    <td align="center" valign="top" width="33%">
      <img src="https://raw.githubusercontent.com/attuan/mekiki/main/assets/icon_provenance.svg" width="40" height="40" alt=""><br>
      <b>Provenance</b><br>
      <sub>Every cell knows its source (human / model / llm), confidence, evidence and cost. <code>explain()</code> traces it back.</sub>
    </td>
  </tr>
</table>

<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/attuan/mekiki/main/assets/concept_dark.svg">
  <img src="https://raw.githubusercontent.com/attuan/mekiki/main/assets/concept_light.svg" width="100%" alt="How mekiki works: free text becomes typed columns, statistical models solve the task first, and only the rows they are unsure about reach the LLM, which makes the final call from their predictions and similar records">
</picture>
</p>

| part | what it gives you | calls the LLM | LLM cost grows with |
|---|---|---|---|
| `diagnose()` | a recommended configuration, as objects | once per table (optional) | — |
| `screen()` | whether the text carries signal at all | never | — |
| `SemanticEncoder` | a typed column from an unstructured one | only for low-confidence rows | rows escalated |
| `KnowledgeEncoder` | columns the table does not have | once per distinct key value | distinct keys, not rows |
| `EvidenceRegressor` / `EvidenceClassifier` | predictions with evidence and a reason | once per routed row | rows × `escalate_rate` |
| `check_duplicates` / `check_overlap` | duplicate-record leakage warnings | never | — |

Every cell carries provenance (human / model / llm, confidence, evidence, cost). LLM responses
are cached on disk so no row is paid for twice. Duplicate-record leakage is checked
automatically on fit / predict. And `screen()` tells you, for free, whether text carries any
signal on your data before you spend anything.

The project started as an internship project on used-car price prediction; used cars are the
flagship use case, not the scope.

**Documentation: <https://attuan.mintlify.site/>**: the same tutorials plus an API reference, built from the `docs/` folder of this repository with Mintlify.

**New here? Open [`examples/quickstart.ipynb`](https://github.com/attuan/mekiki/blob/main/examples/quickstart.ipynb) first**: used-car prices end
to end on the bundled data, committed with the outputs of a real run so it can be read right here on
GitHub (every notebook has an *Open in Colab* badge). Running it yourself takes an API key and about
$1 of LLM calls. Then there is one notebook per part, each ending with how to plug in your own pieces:
[`diagnose`](https://github.com/attuan/mekiki/blob/main/examples/diagnose.ipynb) (free except one cell),
[`semantic_encoder`](https://github.com/attuan/mekiki/blob/main/examples/semantic_encoder.ipynb),
[`knowledge_encoder`](https://github.com/attuan/mekiki/blob/main/examples/knowledge_encoder.ipynb) and
[`evidence_predictor`](https://github.com/attuan/mekiki/blob/main/examples/evidence_predictor.ipynb).

## Installation

Python 3.12 or later. The only required dependencies are pandas, numpy and scikit-learn;
everything else is an extra.

```bash
pip install "mekiki[all]"
uv add "mekiki[all]"      # with uv
```

| extra | installs | needed for |
|---|---|---|
| `models` | lightgbm, xgboost | `EvidencePredictor`, confidence routing, `screen()`, `diagnose()` scoring |
| `llm` | anthropic | actually calling the LLM (import works without it) |
| `litellm` | litellm | calling a provider other than Claude (`openai/...`, `gemini/...`, ...) |
| `embed` | sentence-transformers (pulls in torch) | stronger embeddings than the default character TF-IDF |
| `all` | all four above | |
| `dev` | pytest | development |

For development, clone and install in editable mode. With
[uv](https://docs.astral.sh/uv/) — which fetches Python 3.12 itself, so the system Python
is never involved:

```bash
git clone git@github.com:attuan/mekiki.git && cd mekiki
uv venv --python 3.12
uv pip install -e ".[all,dev]"
.venv/bin/python -m pytest tests -q    # the LLM is stubbed; everything passes without an API key
```

Or with pip:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[all,dev]"
.venv/bin/python -m pytest tests -q
```

On macOS, xgboost / lightgbm need `brew install libomp`.

`pyproject.toml` also carries a `research` dependency group and `[tool.ruff]` settings
that belong to the development repository. Dependency groups are never part of a wheel or
an sdist, so they change nothing about what `pip install mekiki` gives you — but the
project-level commands (`uv sync`, and `uv run`, which syncs first) would install that
group here. That is why the steps above use `uv pip install` and call the interpreter in
`.venv` directly.

## API key

Only needed when the LLM is actually called. Keys are read from the environment, or from
a `.env` file found by walking up from the current directory (template: `.env.example`).

```
ANTHROPIC_API_KEY=sk-ant-api03-...
```

Keys are issued at https://console.anthropic.com/settings/keys.

Claude is the default. To use another provider, install the `litellm` extra
(`pip install "mekiki[litellm]"`), put that provider's key in the same `.env` under the
name LiteLLM expects (`OPENAI_API_KEY`, `GEMINI_API_KEY`, ... — see
https://docs.litellm.ai/docs/providers), and pass the model with its provider prefix:

```python
from mekiki import LLMClient, EvidenceRegressor

client = LLMClient(model="openai/gpt-5")          # or "gemini/gemini-2.5-pro", "ollama/..."
print(client.why_unavailable())                   # None when the key is in place
model = EvidenceRegressor(client=client, ...)
```

Claude models (`claude-...`) always go through the official Anthropic SDK, everything
else through LiteLLM. `ClaudeClient` is the former name of `LLMClient` and still works.

`TYPESAFE_API_KEY` (optional) turns on the Jev middle tier described below; leave it out and
the tier is skipped.

### One key for everything: the Vercel AI Gateway

Instead of one key per provider, a single [Vercel AI Gateway](https://vercel.com/ai-gateway)
key reaches Claude, the other providers and Jev:

```
AI_GATEWAY_API_KEY=vck_...
```

Every model whose own key is missing then goes through the gateway (`client.via_gateway`
tells which way a client goes). Claude keeps the official SDK there, with structured output
and prompt caching unchanged, and its cache key does not change, so answers cached from
Anthropic directly keep hitting. To send Claude through the gateway even with an Anthropic
key present, set `ANTHROPIC_BASE_URL=https://ai-gateway.vercel.sh` or pass
`LLMClient(base_url="https://ai-gateway.vercel.sh")`. The gateway charges each provider's
list price, and it needs a credit card on file before it serves any request.

## Quick start — no API key, no data download

The bundled 500-row excerpts are enough. `SemanticEncoder` in seven lines:

```python
import pandas as pd
from mekiki import SemanticEncoder
from mekiki.paths import sample_data

df = pd.read_csv(sample_data("news_sample500.csv"))
topic = SemanticEncoder(source="Headline", type="category",
                    values=["economy", "microsoft", "obama", "palestine"])
df["topic_typed"] = topic.fit_transform(df)       # no labels, no API key, about two seconds
print((df["topic_typed"] == df["Topic"]).mean())  # 0.92 against the true topic
print(topic.explain(1))                           # why row 1 got its value
```

That is a typed column from four words of supervision, with a confidence and a provenance
record per cell. The notebooks in [`examples/`](https://github.com/attuan/mekiki/tree/main/examples/) take it from there: which rows the
library is unsure about, whether a text column is worth an LLM at all (`screen()`), and then
`EvidencePredictor` on used-car prices with the bill shown before anything is spent.

## Start here — `diagnose()`

Give it the data and the target column and it returns a recommended setup.

```python
from mekiki import diagnose, EvidenceRegressor

rec = diagnose(df, target="price")
print(rec)              # findings and the recommended configuration
rec.spec                # ColumnSpec: which columns are numeric / categorical / text / long text
rec.domain              # Domain: role, what a record is, what the target is called
print(rec.to_code())    # a Python snippet that reproduces the recommendation

model = EvidenceRegressor(target="price", spec=rec.spec, domain=rec.domain)
```

The first stage is rule-based and free (task type, target skew and outliers, per-column
profile, duplicates, text screening, estimated cost of using the LLM on every row). The
second stage hands that diagnosis table, with column names and a few sample values but
not the data itself, to the LLM once, so it can decide what rules cannot: which columns are
identifiers, which vary between listings of the same item, what the target should be called.
It also lists candidate columns you could build next — `SemanticEncoder` candidates extracted
from the free text and `KnowledgeEncoder` candidates keyed by structured columns — as
commented-out lines in `to_code()`, since building them costs LLM calls.
Every recommendation carries `reasons` pointing back at the numbers it came from.
Without an API key (or with `llm=False`) the rule-based recommendation is returned as is.

## `SemanticEncoder` — feature generation

Turn an unstructured column into a typed one by declaration alone. Internally it is
"embed -> nearest-neighbour vote"; the LLM only appears as a fallback for low-confidence rows.

```python
from mekiki import SemanticEncoder

df["body_type"] = SemanticEncoder(
    source="model",                       # free-form string column
    type="category",
    values=["pickup truck", "sedan", "suv", "van", "coupe"],
    escalate_rate=0.15,                   # send the 15% least confident rows to the LLM
).fit_transform(df)
```

If a labelled column already exists, pass it as `labels="..."` and it becomes the reference
set. Rows the model cannot decide are queued in `review_queue()`; approving them turns them
into labels for next time, so the fast path grows.

## `EvidenceRegressor` / `EvidenceClassifier` — LLM predict

The LLM is not handed a raw record and asked for the answer. **LightGBM, XGBoost and a
nearest-neighbour index solve it first; their predictions, plus similar records whose actual
target is known, are passed to the LLM as evidence, and the LLM only makes the final call.**
Similar records are retrieved per row (few-shot, not pasted once).

```python
from mekiki import EvidenceRegressor, USED_CAR

model = EvidenceRegressor(
    target="price", unit="USD", domain=USED_CAR,
    numeric=["age", "odometer"], categorical=["manufacturer", "state"],
    text="model", long_text="description",
)
pred = model.fit(train_df).predict(test_df)
```

No separate labels are needed: the training targets are the few-shot examples.

**The column arguments can be left out.** With none of `numeric` / `boolean` / `categorical` /
`text` / `long_text` given, `fit` assigns every column except the target by the same rules
`diagnose` starts from (dtype, number of distinct values, mean length; no LLM call), leaving
out identifier, date and constant columns. The assignment it used is in `model.spec_`. Give
any column and only the columns given are used.

```python
model = EvidenceRegressor(target="price").fit(train_df)
model.spec_                             # the ColumnSpec fit chose
```

**What is being predicted is described in words through a `Domain`.** The role ("a news
editor"), what one record is ("news article"), what the target is called ("the topic") and any
domain-specific hints are assembled into the prompt. Leave it out and the wording commits
to no domain. The used-car wording is kept as the `USED_CAR` preset.

Classification is `EvidenceClassifier`, with the same arguments minus `unit` (both classes share
the base `EvidencePredictor`, which takes `task=` if you need to switch at runtime): the tree models contribute
per-class probabilities, the neighbours contribute label frequencies, and the LLM returns a
label plus per-class probabilities.

```python
from mekiki import EvidenceClassifier, Domain

model = EvidenceClassifier(
    target="Churn",
    domain=Domain(role="a churn analyst", subject="customer", target_name="churn",
                  class_names={"Yes": "churned", "No": "stayed"}),
    numeric=["tenure", "MonthlyCharges"], categorical=["Contract"],
)
model.fit(train_df)
label = model.predict(test_df)          # labels
proba = model.predict_proba(test_df)    # probabilities in the order of model.classes_
```

## Confidence routing — which rows reach the LLM

`EvidencePredictor` calls the LLM **once per row**, so the row count is the cost and the
time (60,000 rows is roughly $516 and 17 hours). Confidence routing chooses the rows to
send using only signals available before any LLM call and leaves the rest to the
statistical models. One ratio moves continuously between "call every row" and "never call".

```python
model = EvidenceRegressor(target="price", unit="USD",
                          numeric=["age", "odometer"],
                          categorical=["manufacturer", "state"], text="model",
                          escalate_rate=0.3)     # send only the 30% strongest-signal rows
model.fit(train)
model.plan(test)      # before calling: how many rows, how much, how long (free)
pred = model.predict(test)
model.route()         # per row: fast path or llm, and the signal value
model.curve(test)     # accuracy / cost / latency as the ratio is swept
model.approve()       # approve the LLM's answers -> those rows are served without a call next time
```

The default signal is the **disagreement between LightGBM and XGBoost** (rows where the
statistical models themselves are unsure), which measured best. `signal="unseen"` prefers
rows that contain values absent from the training data (an unknown `model`, for example).
Leaving `escalate_rate` out sends every row and warns with the estimated cost;
`escalate_rate=1.0` says so explicitly and silences the warning.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/attuan/mekiki/main/assets/routing_curve_dark.png">
  <img src="https://raw.githubusercontent.com/attuan/mekiki/main/assets/routing_curve_light.png" width="720"
       alt="MAE against the share of rows sent to the LLM, 600 used-car rows: 3,043 at 0% ($0.00), 2,674 at 30% ($1.57), 2,581 at 50% ($2.62), 2,265 at 100% ($5.23)">
</picture>

Measured on 600 held-out Craigslist rows: the error falls steadily as more rows are sent, so
routing is not a way to gain accuracy but a way to **choose how much accuracy to buy**. Sending
half the rows costs half as much and keeps 59% of the improvement.

| rows sent to the LLM | 0% | 10% | 20% | 30% | 50% | 75% | 100% |
|---|---:|---:|---:|---:|---:|---:|---:|
| MAE (USD) | 3,043 | 2,907 | 2,795 | 2,674 | 2,581 | 2,450 | 2,265 |
| LLM cost (USD) | 0.00 | 0.52 | 1.05 | 1.57 | 2.62 | 3.92 | 5.23 |

## A cheaper middle tier — Jev

Every part that calls an LLM can put a second, much cheaper model between the statistical
answer and the frontier LLM: **Jev**, TypeSafe AI's "System One" model. It answers typed
questions (pick one option, rate on ordered levels, yes/no) with calibrated probabilities in a
few hundred milliseconds, for about $0.00004 per row (input tokens only). It cannot write free
text or numbers, which is exactly why it fits: most of what mekiki asks a model is "which of
these", and even regression becomes "which statistical model's prediction to trust".

```python
from mekiki import SemanticEncoder, EvidenceRegressor, KnowledgeEncoder, JevFallback, LLMFallback

col = SemanticEncoder(source="description", values=["dealer", "private"],
                      fallback=[JevFallback(), LLMFallback()])   # neighbours -> Jev -> LLM
model = EvidenceRegressor(target="price", numeric=["age", "odometer"],
                          jev=True, escalate_rate=0.2)  # models -> Jev weights -> LLM
cols = KnowledgeEncoder(keys=["manufacturer", "model"], attribute="body style",
                        values=["sedan", "SUV", "pickup"], jev=True)  # Jev per key -> LLM for the rest
```

The key is `TYPESAFE_API_KEY` in the settings file, or `AI_GATEWAY_API_KEY` to reach Jev
through the Vercel AI Gateway (model `typesafe-ai/jev`). Without either the Jev tier is skipped and
everything runs as the two-tier version. Provenance records `jev` as a source, and `cost()` /
`plan()` report the Jev bill separately. The client (`JevClient`) shares the design of
`LLMClient`: disk cache, cost accounting, parallel batches, and no extra dependency.

Measured through the Vercel AI Gateway, each part with and without Jev on the same rows and the
same LLM answers:

| part | task | without Jev | with Jev |
|---|---|---|---|
| `SemanticEncoder` | model name → body type, used cars, 400 rows | accuracy 0.708, 400 LLM calls | 0.692, **190 LLM calls** |
| `SemanticEncoder` | tasting note → grape variety, wine reviews, 400 rows | 0.792, 398 LLM calls | 0.760, **133 LLM calls** |
| `EvidenceRegressor` | used-car price, 600 rows, no LLM call | MAE 3,043 (first model) | **2,883** (Jev-weighted models) |
| `EvidenceClassifier` | pet adoption speed (5 classes), 600 rows, no LLM call | accuracy 0.397 | **0.417** |
| `KnowledgeEncoder` | body type for 306 used-car models | 0.766 agreement, all by the LLM | 0.768, **229 of 306 settled by Jev** |

Jev cuts LLM calls by 52–83% for a 0.02–0.04 drop in accuracy, and its confident answers are as
accurate as the LLM's. It is a tier in front of the LLM, not a replacement: alone it leaves the
rows it is unsure about unanswered. On a binary task where the models already agree (telecom
churn) the weighting did not help. Keep the default routing signal; routing by Jev's confidence
(`signal="jev"`) chose worse rows than model disagreement. Details on the
[documentation site](https://attuan.mintlify.site/concepts/providers-and-tiers#what-the-middle-tier-buys).

## `KnowledgeEncoder` — columns the table does not have

Ask the LLM's world knowledge for an attribute of each key value, once per unique key, and
join the result to the table. With `target=` each proposed column is scored (a tree model
with and without it, on the same folds, no LLM call) and **only columns that helped are
returned**.

```python
from mekiki import KnowledgeEncoder, USED_CAR

# fully automatic: the LLM proposes what to look up, scoring keeps what helps
new_cols = KnowledgeEncoder(target="price").fit_transform(df)
df = df.join(new_cols)

# or say exactly what you want
col = KnowledgeEncoder(keys=["manufacturer", "model"], attribute="typical new price in USD",
                      type="numeric", domain=USED_CAR)
df = df.join(col.fit_transform(df))
col.cost()            # callable after fit, before any lookup: how many keys, how much
col.status()          # accepted / unknown / rejected per column, agreement with an existing column
col.review_queue()    # key values a human should look at, most frequent first
```

Cost scales with the number of distinct key values, not rows. On 60,000 Craigslist rows a
"typical new price" column cost about $4 and cut LightGBM's MAE by 13.7% without using the
`model` string at all.

## Before you start — does text help on this data?

`EvidencePredictor` only pays off when the text moves the target. `screen()` measures that
**without calling the LLM**, by comparing tree models with and without the text column.

```python
from mekiki import screen

print(screen(df, target="price", text="model", unit="USD",
             numeric=["age", "odometer"], categorical=["manufacturer", "state"]))
```

The report gives the text contribution as a percentage against a threshold and a verdict
(worth trying or not).

## A caution about free text

Listing text often **contains the asking price verbatim** (43.7% of Craigslist
`description` rows). Passed as is, the task becomes reading the answer rather than
predicting it, so `mekiki` **masks amounts by default** in `long_text` for regression.

```python
model = EvidenceRegressor(..., long_text="description")   # amounts become <AMOUNT>
```

## Leakage check — duplicate records

**When the same record is in both train and test, prediction becomes lookup.** The
Craigslist data lists the same car in several regions; evaluating with duplicates left in
inflated R² from 0.880 to 0.914, and **the more text is used as a feature, the larger the
inflation.** `mekiki` therefore checks on fit / predict and warns.

```python
from mekiki import check_duplicates, check_overlap

print(check_duplicates(df, keys=["VIN"]))        # duplicates within one table
print(check_overlap(train, test))                # duplicates across train and test
```

It warns rather than raises (duplicates can be intentional). Switch it off with
`EvidenceRegressor(..., check_leakage=False)`.

## Provenance — inspecting why

Every feature answers "why this value".

```python
model.explain(0)      # one row: evidence, retrieved examples, the LLM's reason
model.confidence()    # per-row confidence
model.examples()      # the similar records used
model.cost()          # what was spent and the cost per row
model.provenance()    # all rows in one table
model.report()        # a short text summary of a predict() run
```

LLM responses are cached on disk (inside a clone: `sampledata/processed/llm_cache/`; when
installed: the OS cache directory; override with `MEKIKI_CACHE_DIR`). **The same row is
never paid for twice**, so re-running after a code change is free.

## Bundled data

`sampledata/sample/` holds 500-row excerpts so that the examples and the tests run with no
download. Each is a random sample (seed 42) of a public dataset; column names and values are
left exactly as the source has them. **Only sources that permit redistribution are bundled**,
and each excerpt keeps the license of its source — MIT covers the code in `mekiki/`, not the
data.

- `vehicles_sample500.csv` — Craigslist used-car listings; free-form seller descriptions and
  a price. Used by most of the notebooks in `examples/`.
  *Used Cars Dataset*, Austin Reese, Kaggle. **CC0 1.0** (public domain dedication).
- `news_sample500.csv` — online news headlines, their topic, and how often each was shared.
  Both answers `screen()` can give live here: the share count barely follows from the
  headline, the topic does.
  Nuno Moniz and Luís Torgo, *Multi-Source Social Feedback of Online News Feeds* (2018),
  UCI Machine Learning Repository, [10.24432/C5H029](https://doi.org/10.24432/C5H029).
  **CC BY 4.0**.
- `bank_sample500.csv` — a Portuguese bank's term-deposit campaign. No text column at all,
  which is the baseline for what `EvidencePredictor` does without one.
  S. Moro, P. Rita and P. Cortez, *Bank Marketing* (2014), UCI Machine Learning Repository,
  [10.24432/C5K306](https://doi.org/10.24432/C5K306). **CC BY 4.0**.

## Layout

| path | contents |
|---|---|
| `mekiki/feature.py` | `SemanticEncoder`. embed -> nearest-neighbour classification -> LLM fallback |
| `mekiki/predictor.py` | `EvidencePredictor`. Statistical models and similar cases as evidence, LLM makes the final call; confidence routing lives here too |
| `mekiki/jev.py` | `JevClient`: the cheap middle tier (TypeSafe AI's Jev). Same cache / cost / parallel design as `LLMClient`; the other module that emits HTTP |
| `mekiki/adaptive.py` | Routing building blocks: signals, row selection, cost and time estimates |
| `mekiki/knowledge.py` | `KnowledgeEncoder`. Columns from world knowledge, asked once per key value |
| `mekiki/diagnose.py` | `diagnose()`. Data + target -> recommended configuration |
| `mekiki/screening.py` | `screen()`. Measures the text contribution without calling the LLM |
| `mekiki/leakage.py` | Duplicate-record detection, run automatically on fit / predict |
| `mekiki/fallback.py` | What happens to low-confidence rows: queue for review, or ask the LLM |
| `mekiki/vectorizers.py` | Embeddings: character TF-IDF (default), sentence-transformers, precomputed |
| `mekiki/llm.py` | **The frontier-LLM client** (`LLMClient`): Claude through the official SDK, other providers through LiteLLM. Disk cache, cost accounting, parallel requests |
| `mekiki/paths.py` | Where the cache, `.env` and bundled data live; differs inside a clone and after install |
| `examples/` | Tutorial notebooks, committed with their outputs |
| `assets/` | Figures used by this README |
| `tests/` | `pytest tests -q` |
