"""mekiki — a machine-learning library for unstructured data.

There are three features (`SemanticEncoder` / `EvidencePredictor` / `diagnose`), and `EvidencePredictor`
carries its own confidence routing that decides which rows go to the LLM.
`SemanticEncoder` and `EvidencePredictor` expose a scikit-learn compatible API.

**`SemanticEncoder` (feature generation).** Turn an unstructured column such as
images or free text into a typed column just by declaring it. Internally it is
"embed, then classify by the nearest existing teacher labels"; the LLM only appears
for rows whose confidence falls below the threshold.

    from mekiki import SemanticEncoder

    df["style"] = SemanticEncoder(
        source="description",        # tasting notes, listing titles, support tickets, ...
        type="category",
        values=["sparkling", "dry white", "rose", "red"],
    ).fit_transform(df)

**`EvidencePredictor` (LLM Predict).** Instead of handing the LLM a raw record
and asking it to guess, LightGBM, XGBoost and nearest-neighbour search solve the problem
first; their predictions plus "similar cases whose answer is known" are bundled as
evidence, and the LLM only makes the final call. Similar cases are retrieved from the
training data per row (few-shot). Works for both regression and classification
(`task="classification"`, with `predict_proba`). What is being predicted is described in
words through `Domain` (`USED_CAR`, a preset for used-car prices, ships as an example).

    from mekiki import EvidenceRegressor, EvidenceClassifier, Domain

    # columns omitted: fit assigns them from the data (see model.spec_)
    pred = EvidenceRegressor(target="points").fit(train_df).predict(test_df)

    model = EvidenceRegressor(target="points", unit="pts",
                              numeric=["price"], categorical=["country", "variety"],
                              long_text="description",
                              domain=Domain(role="a wine judge", subject="wine",
                                            target_name="points"))
    pred = model.fit(train_df).predict(test_df)
    model.explain(0)   # why that row got that value

    churn = EvidenceClassifier(target="Churn",
                              domain=Domain(role="a churn analyst", subject="customer",
                                            target_name="whether the customer churns"),
                              numeric=["tenure"], categorical=["Contract"])
    proba = churn.fit(train_df).predict_proba(test_df)

**Confidence routing (`escalate_rate`).** `EvidencePredictor` calls the LLM once per row,
so the row count translates directly into cost and time (60,000 rows is about $516 and
about 17 hours). It therefore picks the rows to call using only "signals available before
calling the LLM" and leaves the rest to the statistical models. A single rate moves
continuously between "call every row" and "call none". If omitted, every row is sent,
and a `MekikiWarning` is raised when the cost is large enough to matter.

    model = EvidenceRegressor(target="points", unit="pts", ...,
                              escalate_rate=0.3)   # send only the top 30% to the LLM
    model.plan(test)      # before calling: "how many rows, how much, how long" (free, no LLM)
    model.curve(test)     # accuracy / cost / latency as the rate is swept
    model.approve()       # promote LLM answers to teacher labels; those rows are skipped next time

Both carry provenance. `explain` / `confidence` / `examples` / `cost` let you trace
whether each cell came from human / model / llm, what it referenced and what it cost.

**`diagnose` (proposing the initial setup).** `diagnose` looks at a table before anything is
built: from the data and the target alone it assembles the column assignment (`ColumnSpec`), the domain
wording (`Domain`), the target transform, the columns to ignore in duplicate detection,
and whether `EvidencePredictor` is worth using. Stage 1 is rules only (including `screen()` and
duplicate detection, free); stage 2 has the LLM read only the diagnostic table to judge
what each column means, and to list candidate `SemanticEncoder` / `KnowledgeEncoder` columns
(shown commented out in `to_code()`). Every recommendation comes with its rationale.
`EvidencePredictor` already applies the stage-1 column assignment by itself when no column
is given, so `diagnose` is optional: call it to see the grounds, to catch duplicates and skew
before evaluating, or to let the LLM correct the assignment from column meanings.

    from mekiki import diagnose

    rec = diagnose(df, target="points", unit="pts")
    print(rec)             # diagnosis and recommendations
    rec.spec, rec.domain   # can be passed straight to EvidencePredictor
    print(rec.to_code())   # Python that reproduces the recommended configuration

**Knowledge columns — `KnowledgeEncoder`.** Where `SemanticEncoder` builds columns from what is
**inside** the data (text, images), this builds columns from **outside** the data (what the
LLM knows as general knowledge), keyed by the values of structured columns such as
`manufacturer` / `model` or `country` / `variety`. Each unique key value is asked once, answers
are closed over a candidate set, "unknown" becomes missing, and if the data already has a column with the
same attribute the two are cross-checked.

There is one way to use it: feed in a table and get back the columns the table lacked
(a DataFrame). What to look up may be specified or left out. Passing `target` keeps only
the columns that helped (scoring does not call the LLM).

    from mekiki import KnowledgeEncoder

    # fully automatic
    new_cols = KnowledgeEncoder(target="points", domain=Domain(subject="wine")).fit_transform(df)
    new_cols = KnowledgeEncoder(                                                     # fully specified
        keys=["manufacturer", "model"], attribute="body style",
        values=["sedan", "SUV", "pickup", ...], domain=USED_CAR,
        check_against="type",          # report agreement with an existing column
    ).fit_transform(df)
    df = df.join(new_cols)

**Only four places call the LLM**, and all of them go through `mekiki.llm.LLMClient`:
`EvidencePredictor`'s final judgement, `SemanticEncoder`'s `LLMFallback`,
`diagnose`'s stage 2, and knowledge columns. The `model` string picks the provider:
`claude-...` uses the official Anthropic SDK (`pip install "mekiki[llm]"`), anything
else such as `openai/gpt-5` or `gemini/gemini-2.5-pro` goes through LiteLLM
(`pip install "mekiki[litellm]"`) with that provider's key from `.env`.
`ClaudeClient` is the former name of the same class.
Where no API key is present, `LLMClient.available()` returns False and `SemanticEncoder` keeps
running with `QueueOnlyFallback` (no answers; rows are only queued for review).
Knowledge columns cannot be built without the LLM, so they only go as far as `fit` and
the `cost()` estimate.

**Three tiers, with Jev in the middle.** Each of those places (except `diagnose`, whose
job is free-form judgement) can put Jev, TypeSafe AI's "System One" model, between the
statistical model and the frontier LLM: it answers typed questions (pick one option, rate
against levels, yes/no) in a few hundred milliseconds with calibrated probabilities, and
only the rows it is not confident about go on to the LLM. `SemanticEncoder` takes a chain
(`fallback=[JevFallback(), LLMFallback()]`); `EvidencePredictor` and `KnowledgeEncoder`
take `jev=True` (or a `JevClient`). Jev is called only from `mekiki.jev.JevClient`, so
`mekiki/llm.py` and `mekiki/jev.py` are the only two modules that emit HTTP. The key is
`TYPESAFE_API_KEY`, or `AI_GATEWAY_API_KEY` to reach Jev through the Vercel AI Gateway.
Without either the Jev tier is skipped and everything behaves as with two tiers.
"""

from mekiki.diagnose import Diagnosis, Recommendation, diagnose
from mekiki.errors import MekikiError, MekikiWarning
from mekiki.fallback import Fallback, JevFallback, LLMFallback, QueueOnlyFallback
from mekiki.feature import SemanticEncoder
from mekiki.jev import JevClient
from mekiki.knowledge import KnowledgeEncoder
from mekiki.leakage import (
    DuplicateReport,
    OverlapReport,
    check_duplicates,
    check_overlap,
)
from mekiki.llm import ClaudeClient, LLMClient
from mekiki.predictor import (
    USED_CAR,
    ColumnSpec,
    Domain,
    EvidenceClassifier,
    EvidencePredictor,
    EvidenceRegressor,
    NeighbourIndex,
    NeighbourModel,
    TreeModel,
)
from mekiki.preprocess import drop_constant_tokens
from mekiki.screening import ScreeningReport, screen
from mekiki.vectorizers import (
    CharTfidfVectorizer,
    PrecomputedVectorizer,
    SentenceTransformerVectorizer,
    Vectorizer,
)

__all__ = [
    "SemanticEncoder",
    "KnowledgeEncoder",
    "Vectorizer",
    "CharTfidfVectorizer",
    "SentenceTransformerVectorizer",
    "PrecomputedVectorizer",
    "Fallback",
    "QueueOnlyFallback",
    "LLMFallback",
    "JevFallback",
    "LLMClient",
    "JevClient",
    "ClaudeClient",
    "EvidencePredictor",
    "EvidenceRegressor",
    "EvidenceClassifier",
    "Domain",
    "USED_CAR",
    "ColumnSpec",
    "TreeModel",
    "NeighbourModel",
    "NeighbourIndex",
    "MekikiError",
    "MekikiWarning",
    "check_duplicates",
    "check_overlap",
    "DuplicateReport",
    "OverlapReport",
    "drop_constant_tokens",
    "screen",
    "ScreeningReport",
    "diagnose",
    "Diagnosis",
    "Recommendation",
]

#: Former names, kept importable so existing code keeps working. New code should use
#: the names on the left-hand side of this table's right column.
TypedColumn = SemanticEncoder
KnowledgeColumn = KnowledgeEncoder
ClaudeFallback = LLMFallback
Encoder = Vectorizer
CharTfidfEncoder = CharTfidfVectorizer
SentenceTransformerEncoder = SentenceTransformerVectorizer
PrecomputedEncoder = PrecomputedVectorizer

__version__ = "0.2.0"
