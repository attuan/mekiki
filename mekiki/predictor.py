"""`EvidencePredictor`.

**The LLM is not handed a raw record and asked to guess the value.** The statistical
models solve the problem first; their predictions and "similar cases whose answer is
known" are bundled as evidence, and the LLM only makes the final call. For something
like used-car prices the training set has the answer but the test set does not, so for
each test record the closest training records are found and the LLM predicts "along
these lines".

In other words **no separate teacher labels are needed**. The answers in the training
data become the few-shot examples as they are.

    from mekiki import EvidencePredictor, Domain

    # Regression (default). Describe what is being predicted with a Domain
    model = EvidencePredictor(
        target="points", unit="pts",
        numeric=["price"], categorical=["country", "variety"], long_text="description",
        domain=Domain(role="a wine judge", subject="wine", target_name="points"))
    model.fit(train_df)
    pred = model.predict(test_df)
    model.explain(0)      # why that row got that value
    model.cost()          # what it cost

    # Classification. Labels are taken from the target column of the training data
    model = EvidencePredictor(
        target="Churn", task="classification",
        numeric=["tenure", "MonthlyCharges"], categorical=["Contract"],
        domain=Domain(role="a churn analyst for telecom contracts", subject="customer",
                      target_name="churn", class_names={"Yes": "churned", "No": "stayed"}))
    proba = model.fit(train_df).predict_proba(test_df)

    # Confidence routing. Only the top 30% of rows the statistical models are unsure about
    model = EvidencePredictor(target="price", unit="USD", numeric=["age", "odometer"],
                              categorical=["manufacturer"], text="model", escalate_rate=0.3)
    model.fit(train_df)
    model.plan(test_df)   # before calling: how many rows, how much, how many seconds
    pred = model.predict(test_df)
    model.route()         # which route each row took

Omitting `escalate_rate` sends every row (a `MekikiWarning` is raised when the cost is
noticeable). The signal computation and the estimates live in `mekiki.adaptive`.

**Jev as the middle tier (`jev=True`).** The rows that are not sent to the frontier LLM
no longer have to return `models[0]` unchanged. With a `JevClient`, each of them is
put to Jev (TypeSafe AI's System One model) as one typed question: the options are
the statistical models, each described by its prediction for that row, and the state
is the row plus its similar cases. Jev answers in a few hundred milliseconds with a
probability per model, and the prediction is the probability-weighted average of the
models' outputs. The rows the models disagree on still go to the LLM, so the chain is
statistical models, then Jev for the rows that stay, then the LLM. Keep the default
routing signal: Jev's confidence says which model to trust, not how hard the row is,
and in measurements `signal="jev"` chose the rows to escalate worse than the model
disagreement did. Without `TYPESAFE_API_KEY` (or `AI_GATEWAY_API_KEY`) the tier is skipped.

    model = EvidenceRegressor(target="price", unit="USD", numeric=["age", "odometer"],
                              categorical=["manufacturer"], text="model",
                              jev=True, escalate_rate=0.2)

Wording tuned for used-car prices is kept in the `USED_CAR` preset, distinct from the
generic default wording.

The evidence recipe follows our measurements. **Choosing
neighbours by semantic similarity alone does not capture closeness of the target**
(MAE 31.45), so the distance mixes in the numeric columns (18.19). This is not revisited
here; the recipe fixed by measurement is used as is.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from mekiki.adaptive import (
    COST_WARN_USD,
    SIGNALS,
    disagreement,
    latency_seconds,
    row_key,
    select,
    to_confidence,
    unit_cost,
    unseen_score,
)
from mekiki.errors import MekikiError, MekikiWarning, missing_extra
from mekiki.jev import SECONDS_PER_CALL as JEV_SECONDS_PER_CALL
from mekiki.jev import JevClient
from mekiki.leakage import check_duplicates, check_overlap, warn_if_leaky
from mekiki.llm import ClaudeClient, LLMAnswer
from mekiki.vectorizers import CharTfidfVectorizer, Vectorizer

SEED = 42

TASKS = ("regression", "classification")

#: Inside `models=[...]`, this string stands for the default models
DEFAULT_MODELS = "default"

#: Upper bound on the number of classes a classification can offer as candidates. Beyond
#: this the prompt and the answer schema balloon and "pick one of the candidates" stops
#: making sense for the LLM. Fold the labels into coarser groups with `SemanticEncoder`
#: (`SemanticEncoder`) or switch to regression instead
MAX_CLASSES = 20


# =====================================================================
# Evidence producers -- statistical models and neighbour search
# =====================================================================


@runtime_checkable
class BaseModel(Protocol):
    """One component that produces a piece of evidence for the LLM. Looser than scikit-learn.

    Column selection and missing-value handling differ per model, so the DataFrame is
    passed as is rather than a matrix.

    For classification (`task="classification"`), `y` holds **integers 0..C-1**
    (the order of `EvidencePredictor.classes_`). `predict_proba(test)` must return
    probabilities of shape (n, C). scikit-learn classifiers work unchanged.

    Every model's output is shown to the LLM as evidence. Whether it also counts towards
    the routing signal `signal="disagreement"` is opt-in: set the attribute
    `in_disagreement = True` on the model. It is left out by default because the spread
    is a max-minus-min, so a single weak model would dominate it and change which rows
    are sent. The built-in tree models take part; the neighbour model does not.
    """

    name: str

    def fit(self, train: pd.DataFrame, y: np.ndarray) -> BaseModel: ...

    def predict(self, test: pd.DataFrame) -> np.ndarray: ...


@dataclass
class ColumnSpec:
    """Which column is treated how. Swapping datasets only requires rewriting this.

    `text` and `long_text` are separate because **their length is handled differently**.

    - `text` ... short text (a product name, a list of features). It feeds the
      neighbour search and is shown in full for both the target row and the similar cases.
    - `long_text` ... free-text body (a listing's `description` in the Craigslist used-car
      data: median 1,075 chars, max 28,832). **Pasting five cases in full inflates the prompt by an order of
      magnitude**, so the target row and the similar cases get separate limits. By default
      it is not shown for similar cases (`long_text_example_chars=0`).

    This split is what lets long free text be used at all.
    """

    numeric: list[str] = field(default_factory=list)
    boolean: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    text: str | None = None
    long_text: str | None = None
    #: Character limit for the target row's free text. Anything beyond is cut and marked
    long_text_chars: int = 2000
    #: Limit for the similar cases' free text. 0 means not shown (default)
    long_text_example_chars: int = 0
    #: Whether to mask amount- or price-like numbers inside the free text.
    #: None (default) means **mask for regression, do not mask for classification**.
    #: Free text often states the answer (a listing gives its asking price); passing it
    #: turns prediction into reading the answer (measurements in the `mask_amounts` docstring).
    #: In classification the label is rarely written as a number, and masking only
    #: removes readable information, so it is off.
    mask_amounts_in_long_text: bool | None = None

    def all_columns(self) -> list[str]:
        cols = [*self.numeric, *self.boolean, *self.categorical]
        if self.text:
            cols.append(self.text)
        if self.long_text:
            cols.append(self.long_text)
        return cols

    def check(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.all_columns() if c not in df.columns]
        if missing:
            raise MekikiError(f"Columns not found in the data: {missing}")


@dataclass
class TreeModel:
    """LightGBM / XGBoost wrapped into the `BaseModel` shape.

    By default it is given the structured columns only (numeric, boolean, categorical).
    Text goes to the LLM, so it is not fed here.

    **Category levels are fixed on train at fit time.** Levels that only appear in test
    (a category value unseen in train) are treated as missing and the trees route them to
    the missing branch. Breaking this invalidates the "unseen value" evaluation.

    With `task="classification"` it becomes a classifier whose `predict_proba` returns
    (n, C). `predict` then returns the index of the most probable class.
    """

    name: str
    spec: ColumnSpec
    kind: str = "lgbm"  # "lgbm" | "xgb"
    params: dict = field(default_factory=dict)
    task: str = "regression"
    #: Counts towards `signal="disagreement"` (see `BaseModel`)
    in_disagreement: ClassVar[bool] = True

    def _frame(self, df: pd.DataFrame) -> pd.DataFrame:
        # Free text (long_text) is not vectorised, so it is not given to the trees.
        # Reading it is the LLM's job; feeding it here would muddy the comparison.
        out = (df[self.spec.numeric].astype("float64").copy()
               if self.spec.numeric else pd.DataFrame(index=df.index))
        for c in self.spec.boolean:
            out[c] = df[c].astype("float64")
        for c in self.spec.categorical:
            cats = self.categories_[c]
            out[c] = pd.Categorical(df[c].where(df[c].isin(cats)), categories=cats)
        return out

    def fit(self, train: pd.DataFrame, y: np.ndarray) -> TreeModel:
        if self.task not in TASKS:
            raise MekikiError(f"task must be one of {TASKS}: {self.task}")
        classify = self.task == "classification"
        if not (self.spec.numeric or self.spec.boolean or self.spec.categorical):
            # An empty frame makes LightGBM fail with an unreadable message
            # ("maximum feature index in dataset is -1"), so say what is missing
            raise MekikiError(
                "The tree models need at least one numeric, boolean or categorical column. "
                "`text` / `long_text` are not fed to them: reading text is the job of the "
                "neighbour model and the LLM. If the text is the only signal, build a "
                "typed column from it first (SemanticEncoder) and pass that as categorical.")
        self.categories_ = {c: pd.Index(train[c].dropna().unique())
                            for c in self.spec.categorical}
        y = (np.asarray(y, dtype=int) if classify
             else np.asarray(y, dtype=float))
        if self.kind == "lgbm":
            try:
                from lightgbm import LGBMClassifier, LGBMRegressor
            except ImportError as e:
                raise missing_extra("lightgbm", "models") from e
            p = dict(n_estimators=700, learning_rate=0.05, num_leaves=31,
                     min_child_samples=20, subsample=0.9, subsample_freq=1,
                     colsample_bytree=0.9, random_state=SEED, verbose=-1)
            cls = LGBMClassifier if classify else LGBMRegressor
            self.model_ = cls(**{**p, **self.params})
            self.model_.fit(self._frame(train), y,
                            categorical_feature=self.spec.categorical or "auto")
        elif self.kind == "xgb":
            try:
                from xgboost import XGBClassifier, XGBRegressor
            except ImportError as e:
                raise missing_extra("xgboost", "models") from e
            p = dict(n_estimators=700, learning_rate=0.05, max_depth=6,
                     subsample=0.9, colsample_bytree=0.9, random_state=SEED,
                     enable_categorical=True, tree_method="hist", verbosity=0)
            cls = XGBClassifier if classify else XGBRegressor
            self.model_ = cls(**{**p, **self.params})
            self.model_.fit(self._frame(train), y)
        else:
            raise MekikiError(f"kind must be lgbm or xgb: {self.kind}")
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        if self.task == "classification":
            return np.argmax(self.predict_proba(test), axis=1)
        return np.asarray(self.model_.predict(self._frame(test)), dtype=float)

    def predict_proba(self, test: pd.DataFrame) -> np.ndarray:
        if self.task != "classification":
            raise MekikiError("predict_proba is only available with task='classification'.")
        return np.asarray(self.model_.predict_proba(self._frame(test)), dtype=float)


@dataclass
class NeighbourIndex:
    """Index for retrieving "similar cases whose answer is known".

    similarity = cosine similarity of the text - w * mean standardised distance of the
    numeric columns. The subtraction follows our measurements: by meaning alone,
    records with very different numeric attributes (e.g. cars of a different year and
    mileage) come out as "similar".

    The default w = 0.15 was chosen by the same measurements.
    """

    spec: ColumnSpec
    k: int = 5
    w: float = 0.15
    vectorizer: Vectorizer | None = None

    def fit(self, train: pd.DataFrame, y: np.ndarray) -> NeighbourIndex:
        self.train_ = train.reset_index(drop=True)
        # For classification the labels are stored as is (strings allowed), so no dtype
        self.y_ = np.asarray(y)
        if not self.spec.text and not self.spec.numeric:
            raise MekikiError(
                "Retrieving similar cases needs at least one text or numeric column. "
                "For data without text, pass numeric columns via numeric.")
        if self.spec.text:
            if self.vectorizer is None:
                self.vectorizer = CharTfidfVectorizer()
            texts = self._texts(self.train_)
            self.vectorizer.fit(texts)
            self.V_ = self.vectorizer.transform(texts)
        else:
            # Data without text (a churn table, for instance): numeric distance only
            self.V_ = None
        if self.spec.numeric:
            N = self.train_[self.spec.numeric].astype("float64")
            self.num_med_ = N.median()
            self.N_ = N.fillna(self.num_med_).to_numpy()
            sd = self.N_.std(axis=0)
            sd[sd == 0] = 1.0
            self.sd_ = sd
        else:
            self.N_ = None
        return self

    def _texts(self, df: pd.DataFrame) -> pd.Series:
        if self.spec.text:
            return df[self.spec.text].fillna("").astype(str)
        return pd.Series([""] * len(df), index=df.index)

    def query(self, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """For each row, the indices of the closest training rows and their similarity.
        Shape (n_test, k)."""
        if self.V_ is not None:
            Vq = self.vectorizer.transform(self._texts(test))
            sim = np.asarray(Vq @ self.V_.T, dtype=float)
        else:
            sim = np.zeros((len(test), len(self.train_)), dtype=float)
        if self.N_ is not None:
            Nq = (test[self.spec.numeric].astype("float64")
                  .fillna(self.num_med_).to_numpy())
            d = np.zeros_like(sim)
            for c in range(self.N_.shape[1]):
                d += np.abs(Nq[:, [c]] - self.N_[:, c][None, :]) / self.sd_[c]
            sim = sim - self.w * d / self.N_.shape[1]
        k = min(self.k, sim.shape[1])
        # argpartition takes only the top k, then sorts within those k
        idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sim, idx, axis=1), axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        return idx, np.take_along_axis(sim, idx, axis=1)


@dataclass
class NeighbourModel:
    """Predicts the median answer of the k neighbours (class rates for classification).
    Handed to the LLM as one piece of evidence.

    On its own it does not reach the statistical models (18.19 vs 12.21 on a used-car
    dataset),
    but it carries information from another direction: "how similar cases turned out".
    """

    name: str
    index: NeighbourIndex
    task: str = "regression"
    #: Left out of `signal="disagreement"`: it answers from a few neighbours (class rates
    #: in steps of 1/k), so its distance to the trees reflects that coarseness rather
    #: than how unsure the models are. `signal="similarity"` covers the neighbour side.
    in_disagreement: ClassVar[bool] = False

    def fit(self, train: pd.DataFrame, y: np.ndarray) -> NeighbourModel:
        # The index is normally fitted by EvidencePredictor. When used standalone,
        # build it here if it is not yet built (never query an empty index silently).
        if not hasattr(self.index, "V_"):
            self.index.fit(train, y)
        if self.task == "classification":
            self.y_ = np.asarray(y, dtype=int)
            self.n_classes_ = int(self.y_.max()) + 1 if len(self.y_) else 0
        else:
            self.y_ = np.asarray(y, dtype=float)
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        if self.task == "classification":
            return np.argmax(self.predict_proba(test), axis=1)
        idx, _ = self.index.query(test)
        return np.median(self.y_[idx], axis=1)

    def predict_proba(self, test: pd.DataFrame) -> np.ndarray:
        if self.task != "classification":
            raise MekikiError("predict_proba is only available with task='classification'.")
        idx, _ = self.index.query(test)
        labels = self.y_[idx]                       # (n, k)
        out = np.zeros((len(test), self.n_classes_), dtype=float)
        for c in range(self.n_classes_):
            out[:, c] = (labels == c).mean(axis=1)
        return out


# =====================================================================
# The words that tell the LLM what is being predicted
# =====================================================================


@dataclass
class Domain:
    """The words that tell the LLM "what is predicted, about what". Prompts are built from this.

    The internals of `EvidencePredictor` (statistical models, neighbour search, provenance) do not
    depend on the dataset, but the prompt wording works better when it knows the field.
    Only that part is gathered here; the default (no arguments) is wording that does not
    commit to any field.

    role:
        The X in "You are X". E.g. "a wine judge" / "a churn analyst" / "a used-car appraiser"
    subject:
        What one record is. E.g. "wine" / "customer" / "car".
        Used as in "answer the ... of this {subject}"
    subject_heading:
        The heading that refers to the row being predicted. Default "the record to
        predict". E.g. "the customer in question" / "the car being appraised"
    target_name:
        The name of the target. E.g. "price" / "points" / "churn". None uses the column name
    hints:
        Field-specific guidance: what to pick up from the free text, what to ignore, etc.
        One bullet per string
    class_names:
        For classification, label -> its meaning. E.g. {"Yes": "churned", "No": "stayed"}.
        The prompt shows them as "Yes (churned)"
    system_prompt / answer_schema / answer_key:
        Hooks for **replacing the wording and the schema wholesale**. Used by `USED_CAR`.
        Normally left alone. Changing them changes the LLM cache key, so previously
        cached answers are no longer reused
    """

    role: str = "an analyst who estimates the target from the given evidence"
    subject: str = "record"
    subject_heading: str = "the record to predict"
    target_name: str | None = None
    hints: list[str] = field(default_factory=list)
    class_names: dict = field(default_factory=dict)
    system_prompt: str | None = None
    answer_schema: dict | None = None
    answer_key: str = "value"

    def name_of(self, target: str) -> str:
        return self.target_name or target

    def label(self, c: Any) -> str:
        """Render a label in a human-readable form, with its meaning attached."""
        s = str(c)
        meaning = self.class_names.get(c, self.class_names.get(s))
        if meaning is None:
            # Integer labels (0..4 etc.) arrive here as strings, so the keys are
            # converted to strings as well before matching
            meaning = next((v for k, v in self.class_names.items() if str(k) == s), None)
        return f"{s} ({meaning})" if meaning else s


def _heading(head: str) -> str:
    """Capitalise the subject heading for use as a section title."""
    return head[:1].upper() + head[1:]


def _regression_schema(target_name: str, unit: str) -> dict:
    u = f". Unit: {unit}" if unit else ". The unit is the one given in the prompt"
    return {
        "type": "object",
        "properties": {
            "value": {"type": "number", "description": f"Predicted {target_name}{u}"},
            "confidence": {"type": "number", "description": "Confidence from 0.0 to 1.0"},
            "reason": {"type": "string",
                       "description": "How each piece of evidence was weighted. "
                                      "One or two sentences"},
        },
        "required": ["value", "confidence", "reason"],
        "additionalProperties": False,
    }


def _classification_schema(target_name: str, classes: Sequence[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": list(classes),
                      "description": f"Predicted {target_name}. Exactly one of the candidates"},
            "probabilities": {
                "type": "object",
                "properties": {c: {"type": "number"} for c in classes},
                "required": list(classes),
                "additionalProperties": False,
                "description": "Probability of each candidate. They should sum to 1.0",
            },
            "confidence": {"type": "number",
                           "description": "How sure you are that label is correct. "
                                          "0.0 to 1.0"},
            "reason": {"type": "string",
                       "description": "How each piece of evidence was weighted. "
                                      "One or two sentences"},
        },
        "required": ["label", "probabilities", "confidence", "reason"],
        "additionalProperties": False,
    }


#: Default regression schema (its key differs from the `USED_CAR` one, which uses "price")
ANSWER_SCHEMA = _regression_schema("target", "")


_COMMON_GUIDANCE = """\
- If the free text ends with "... (N more characters omitted)", the rest could not be read.
  Judge from what was readable and lower confidence.
"""

_MASK_GUIDANCE = """\
- "<AMOUNT>" and "<NUMBER>" inside the free text are **numbers masked to prevent leaking
  the answer**. The {tn} itself may have been written there, so do not try to guess and
  fill them in.
"""


def build_system_prompt(domain: Domain, task: str, target: str, *,
                        classes: Sequence[str] = (), unit: str = "",
                        masked: bool = False) -> str:
    """Build the system prompt from a `Domain`. Returns `domain.system_prompt` when set."""
    if domain.system_prompt is not None:
        return domain.system_prompt
    tn = domain.name_of(target)
    subj, head = domain.subject, domain.subject_heading
    hints = "".join(f"- {h}\n" for h in domain.hints)
    mask = _MASK_GUIDANCE.format(tn=tn) if masked else ""

    if task == "regression":
        unit_note = f" The unit is {unit}." if unit else ""
        return f"""\
You are {domain.role}. You will receive the information for one {subj} and the "evidence"
about it; answer the {tn} of the {subj} as a single number.{unit_note}

There are three kinds of evidence.

1. Predictions of the statistical models -- outputs of tree models trained on the whole
   training set. They capture the overall trend well but have not read any of the text.
2. Similar cases -- entries from the training data (each one a {subj}) whose actual {tn}
   is known, ordered by closeness to {head}. "Closeness" is measured by a distance that
   mixes numeric attributes and text.
3. The attributes of {head} itself -- the numeric and text columns.

Guidelines:

- **Start from the statistical models' predictions.** They are the only evidence that has
  seen the whole training set. There are only a few similar cases, so they rarely justify
  moving far away from the models.
- Use the similar cases to read "differences written in the text that the statistical
  models missed": features {head} has that the similar cases lack, and vice versa.
- When the statistical models disagree with each other, weight the one closer to the
  range of {tn} among the similar cases.
- Do not answer far from the evidence. If you go outside the range spanned by the models'
  predictions and the similar cases' {tn}, state why in reason.
- If free text is attached, it is **the only information the statistical models could
  not read**. Pick up anything there that moves the {tn}. Do not rely on boilerplate
  that appears in every {subj}.
{hints}{_COMMON_GUIDANCE}{mask}
Set confidence to how sure you are that this answer is close to the actual {tn}, from
0.0 to 1.0. Lower it when the pieces of evidence disagree or the similar cases do not
resemble {head}.
"""

    options = "".join(f"- {domain.label(c)}\n" for c in classes)
    return f"""\
You are {domain.role}. You will receive the information for one {subj} and the "evidence"
about it; decide which of the following candidates the {tn} of the {subj} is.

Candidates:
{options}
There are three kinds of evidence.

1. Predictions of the statistical models -- the probability of each candidate from tree
   models trained on the whole training set. They capture the overall trend well but have
   not read any of the text.
2. Similar cases -- entries from the training data (each one a {subj}) whose actual {tn}
   is known, ordered by closeness to {head}. "Closeness" is measured by a distance that
   mixes numeric attributes and text.
3. The attributes of {head} itself -- the numeric and text columns.

Guidelines:

- **Start from the statistical models' probabilities.** They are the only evidence that
  has seen the whole training set. There are only a few similar cases, so they rarely
  justify moving far away from the models.
- Use the similar cases to read "differences written in the text that the statistical
  models missed". If the {tn} of the similar cases leans one way, that is a clue too.
- When the statistical models disagree with each other, weight the one closer to the
  distribution of {tn} among the similar cases.
- If free text is attached, it is **the only information the statistical models could
  not read**. Pick up anything there that decides the {tn}. Do not rely on boilerplate
  that appears in every {subj}.
{hints}{_COMMON_GUIDANCE}{mask}
Put exactly one of the candidates in label, and the probability of each candidate
(summing to 1.0) in probabilities.
Set confidence to how sure you are that label matches the actual {tn}, from 0.0 to 1.0.
Lower it when the pieces of evidence disagree or the similar cases do not resemble {head}.
"""


#: Wording for predicting used-car prices.
#: The fingerprint of the system prompt is pinned by a test, because changing the wording
#: invalidates the whole LLM cache and every row is charged again.
USED_CAR = Domain(
    role="a used-car appraiser",
    subject="car",
    subject_heading="the car being appraised",
    target_name="price",
    hints=[
        "Pick up anything that moves the price: accident history, modifications, defects, "
        "included accessories, an urgent sale, and so on. But sales talk (\"beautiful car\", "
        "\"great deal\") appears in every listing, so do not treat it as grounds for the price.",
    ],
    answer_key="price",
    answer_schema={
        "type": "object",
        "properties": {
            "price": {"type": "number",
                      "description": "Predicted price. The unit is the one given in the prompt"},
            "confidence": {"type": "number",
                           "description": "Confidence from 0.0 to 1.0"},
            "reason": {"type": "string",
                       "description": "How each piece of evidence was weighted. "
                                      "One or two sentences"},
        },
        "required": ["price", "confidence", "reason"],
        "additionalProperties": False,
    },
    system_prompt="""\
You are a used-car appraiser. You will receive the information for one car and the
"evidence" about it; answer the vehicle price as a single number.

There are three kinds of evidence.

1. Predictions of the statistical models -- outputs of tree models trained on the whole
   training set. They capture the overall trend well but have not read the equipment or
   condition written in the text.
2. Similar cases -- cars from the training data whose actual sale price is known, ordered
   by closeness to the car being appraised. "Closeness" is measured by a distance that
   mixes numeric attributes and text.
3. The attributes of the car being appraised itself -- the numeric and text columns.

Guidelines:

- **Start from the statistical models' predictions.** They are the only evidence that has
  seen the whole training set. There are only a few similar cases, so they rarely justify
  moving far away from the models.
- Use the similar cases to read "differences in equipment and condition that the
  statistical models missed": equipment the car being appraised has that the similar
  cases lack, and vice versa.
- When the statistical models disagree with each other, weight the one closer to the
  price range of the similar cases.
- Do not answer far from the evidence. If you go outside the range spanned by the models'
  predictions and the similar cases' prices, state why in reason.
- If free text (the seller's description) is attached, it is **the only information the
  statistical models could not read**. Pick up anything that moves the price: accident
  history, modifications, defects, included accessories, an urgent sale, and so on. But
  sales talk ("beautiful car", "great deal") appears in every listing, so do not treat it
  as grounds for the price.
- If the free text ends with "... (N more characters omitted)", the rest could not be read.
  Judge from what was readable and lower confidence.
- "<AMOUNT>" and "<NUMBER>" inside the free text are **numbers masked to prevent leaking
  the answer**. They very likely held the asking price, so do not try to guess and fill
  them in.

Set confidence to how sure you are that this answer is close to the actual price, from
0.0 to 1.0. Lower it when the pieces of evidence disagree or the similar cases do not
resemble the car being appraised.
""",
)

#: Name kept for compatibility. Its content is `USED_CAR.system_prompt`
SYSTEM_PROMPT = USED_CAR.system_prompt


# =====================================================================
# Prompt building blocks
# =====================================================================


def _format_value(v: Any) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "unknown"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, (float, np.floating)):
        return f"{v:,.6g}"
    return str(v)


def _fmt(v: Any) -> str:
    """Display of predictions and answers: thousands separators for numbers, labels as is."""
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, float, np.integer, np.floating)):
        return f"{v:,.6g}"
    return str(v)


#: Amount-like notation. Catches `$12,345` / `12345 dollars` / `USD 12,345` and yen
#: amounts written with the characters U+4E07 U+5186 ("10,000 yen") or U+5186 ("yen")
_MONEY = re.compile(
    r"(?:[$\uff04]\s?[\d][\d,.]*)"
    r"|(?:\b[\d][\d,.]*\s?(?:dollars?|usd|\u4e07\u5186|\u5186)\b)",
    re.IGNORECASE)

#: A standalone integer of 3-6 digits. Masks every number with enough digits to be a price
_BARE_NUMBER = re.compile(r"(?<![\w.$])(\d{1,3}(?:,\d{3})+|\d{3,6})(?![\w.])")

#: Lower and upper bound for masking. Numbers outside cannot be prices, so they stay
_MASK_MIN, _MASK_MAX = 300, 300_000


def mask_amounts(text: str) -> str:
    """Mask "numbers that could be the answer" in free text.

    **Why it is needed.** Free text often states the answer verbatim: a listing gives its
    asking price. Measured on the Craigslist used-car descriptions, **43% of rows contain a
    number identical to the price** and
    **51% contain a `$` amount**. Handing that to the LLM means it is not predicting but
    **reading the answer** (measured without masking, MAE appeared to improve 40%,
    2,336 -> 1,546; that is not real skill).

    This is not specific to used cars. **It always happens with listings, job ads, real
    estate and any data whose free text states prices or terms**, so the product closes
    the hole instead of relying on user discipline.

    What gets masked:

    - amount notation such as `$12,345`, `12345 dollars`, `USD 12,345`, and yen amounts
    - standalone integers from 300 to 300,000 (this also catches mileage, model year and
      stock numbers)

    **Mileage and model year disappear too, but they are passed separately as structured
    columns**, so no information is lost. Conversely, model numbers such as "F-150",
    "911" and "Model 3" are below 300 and stay. Phone numbers have too many digits and
    stay, harmlessly.

    It is not complete (spelled-out amounts like "twelve thousand" slip through).
    **Do not rely on masking alone; any measurement must still check for leaks.**
    """
    if not text:
        return text

    def _num(m: re.Match) -> str:
        raw = m.group(1).replace(",", "")
        try:
            v = int(raw)
        except ValueError:
            return m.group(0)
        return "<NUMBER>" if _MASK_MIN <= v <= _MASK_MAX else m.group(0)

    return _BARE_NUMBER.sub(_num, _MONEY.sub("<AMOUNT>", text))


def _truncate(text: str, limit: int) -> str:
    """Cut at the limit and say so explicitly.

    Cutting silently makes the LLM read "the sentence ends here". Recording the cut and
    the original length tells it that some of the evidence is missing.
    """
    if limit <= 0 or len(text) <= limit:
        return text
    return f"{text[:limit]}... ({len(text) - limit:,} more characters omitted)"


def _describe_row(row: pd.Series, spec: ColumnSpec, indent: str = "",
                  is_target: bool = True) -> str:
    """Render one row as a human-readable bullet list. This is the form given to the LLM.

    `is_target` changes how the free text is handled: shown at length for the target
    row, not shown by default for similar cases (to keep the prompt from ballooning).
    """
    lines = []
    for c in [*spec.numeric, *spec.boolean, *spec.categorical]:
        lines.append(f"{indent}- {c}: {_format_value(row.get(c))}")
    if spec.text:
        lines.append(f"{indent}- {spec.text}: {_format_value(row.get(spec.text))}")
    if spec.long_text:
        limit = (spec.long_text_chars if is_target
                 else spec.long_text_example_chars)
        if limit > 0:
            raw = row.get(spec.long_text)
            body = "" if raw is None or (isinstance(raw, float) and np.isnan(raw)) \
                else str(raw)
            # Collapse newlines. Raw newlines inside a bullet list break its structure
            body = " ".join(body.split())
            if spec.mask_amounts_in_long_text:
                body = mask_amounts(body)
            lines.append(f"{indent}- {spec.long_text}: "
                         f"{_truncate(body, limit) if body else '(none)'}")
    return "\n".join(lines)


# =====================================================================
# `EvidencePredictor` itself
# =====================================================================


@dataclass
class Prediction:
    """One row's prediction and its provenance (the inspection API).

    For regression `value` is a number. For classification `value` is a label and
    `proba` holds the probability of each candidate in `classes_` order.
    `model_predictions` likewise: numbers for regression, probability vectors for
    classification.
    """

    value: Any
    confidence: float
    reason: str
    origin: str                      # "llm" | "jev" | "fallback" | "model" | "human"
    model_predictions: dict[str, Any]
    neighbours: list[dict]
    cost: float = 0.0
    from_cache: bool = False
    error: str | None = None
    proba: np.ndarray | None = None


class EvidencePredictor:
    """Predictor that lets the LLM make the final call from statistical models and similar cases.

    Parameters
    ----------
    target:
        Name of the target column.
    unit:
        Unit for regression ("USD", "pts"). It only appears in the prompt, but it keeps
        the LLM from getting the order of magnitude wrong. Unused for classification.
    task:
        "regression" (default) or "classification". For classification the labels are
        taken from the `target` column of the training data; `predict` returns labels
        and `predict_proba` returns probabilities (in `classes_` order). Up to
        `MAX_CLASSES` classes.
    domain:
        The words that tell the LLM what is being predicted (`Domain`). Omitted, the
        wording does not commit to any field. `USED_CAR` is a ready-made example for
        used-car prices.
    numeric / boolean / categorical / text / long_text:
        Column assignment. A `ColumnSpec` can be passed directly instead. Omitted
        altogether, `fit` assigns every column except `target` by rules (the same ones
        `diagnose` starts from, no LLM call) and stores the result in `spec_`. Giving any
        of them turns this off: only the columns given are used.
    models:
        Statistical models that produce the evidence. Default: LightGBM, XGBoost and
        the neighbour model. A list **replaces** the default; write the string
        `"default"` inside it to keep the three and add your own:
        `models=["default", MyModel()]`. Anything with `name`, `fit(train, y)` and
        `predict(test)` works (`predict_proba` for classification). `models[0]` answers
        the rows that are not sent to the LLM.
    n_examples:
        Number of similar cases shown to the LLM (the number of few-shot shots).
    client:
        `ClaudeClient`. Default: claude-opus-5 / effort=low.
    jev:
        The middle tier. `True` creates a `JevClient` (key from the settings file); a
        `JevClient` is used as is. Rows that are not sent to the LLM (and not approved)
        are then put to Jev, which weighs the statistical models' predictions for that
        row; the prediction is the probability-weighted average, with `origin="jev"`.
        If Jev cannot answer a row it returns `models[0]` as without Jev. Omitted
        (None): no middle tier. Without a key the tier is skipped with a warning.
    escalate_rate:
        **What fraction of rows, strongest signal first, to send to the LLM** (0.0-1.0).
        The rest return the prediction of `models[0]` (LightGBM by default) unchanged
        (or Jev's weighted average when `jev` is set).
        It can be derived from a cost ceiling, which makes it the practical choice
        (same idea as the argument of the same name on `SemanticEncoder`).
        Omitted, every row is sent. If the estimated cost then exceeds `COST_WARN_USD`,
        a `MekikiWarning` is raised. Write `1.0` when every row is intended; no warning.
    threshold:
        Threshold on the signal value itself. Effective when `escalate_rate` is omitted.
    signal:
        "disagreement" (default, best in measurements) ... spread between the tree
        models' predictions. A model of your own joins it only if it sets
        `in_disagreement = True`.
        "similarity" ... rows whose top neighbour has low similarity (weak evidence).
        "unseen" ... rows containing category levels or words absent from the training
        data. Meant for operation where new models and grades keep arriving.
        "jev" ... 1 minus Jev's confidence in its weighting of the models (needs
        `jev`). Jev is then asked for every row, including in `plan()`. Measured
        worse than "disagreement" at low escalation rates; `jev=True` alone already
        uses Jev for the rows that are not escalated.
        For a custom signal pass `f(X, evidence) -> array`. **Larger means send to the
        LLM.**

    Usage is scikit-learn's `fit` -> `predict`. On top sit the inspection API
    (`explain` / `confidence` / `examples` / `cost` / `provenance`) and the routing
    tools: `plan` (estimate before calling), `route` (route per row),
    `review_queue` / `approve` (turn LLM answers into teacher labels), and
    `curve` (accuracy, cost and time as the rate is varied).
    """

    def __init__(self, target: str, unit: str = "", *,
                 task: str = "regression",
                 domain: Domain | None = None,
                 numeric: Sequence[str] = (), boolean: Sequence[str] = (),
                 categorical: Sequence[str] = (), text: str | None = None,
                 long_text: str | None = None,
                 spec: ColumnSpec | None = None,
                 models: Sequence[BaseModel | str] | None = None,
                 n_examples: int = 5, neighbour_weight: float = 0.15,
                 vectorizer: Vectorizer | None = None,
                 client: ClaudeClient | None = None,
                 jev: JevClient | bool | None = None,
                 fallback: str = "best_model",
                 check_leakage: bool = True,
                 escalate_rate: float | None = None,
                 threshold: float | None = None,
                 signal: str | Callable[..., np.ndarray] = "disagreement") -> None:
        if task not in TASKS:
            raise MekikiError(f"task must be one of {TASKS}: {task!r}")
        self.target = target
        self.unit = unit
        self.task = task
        self.domain = domain or Domain()
        if spec is None and (numeric or boolean or categorical or text or long_text):
            spec = ColumnSpec(
                numeric=list(numeric), boolean=list(boolean),
                categorical=list(categorical), text=text, long_text=long_text)
        #: The column assignment as given. None means "decide from the data in `fit`";
        #: the assignment actually used is `spec_`
        self.spec = self._with_mask_default(spec) if spec is not None else None
        self.n_examples = n_examples
        self.neighbour_weight = neighbour_weight
        self.vectorizer = vectorizer
        self.client = client or ClaudeClient()
        if jev is True:
            self.jev: JevClient | None = JevClient()
        elif jev is False or jev is None:
            self.jev = None
        else:
            self.jev = jev
        self._models_arg = models
        if fallback not in ("best_model", "error"):
            raise MekikiError('fallback must be "best_model" or "error"')
        self.fallback = fallback
        #: Whether to detect duplicate records and warn (leak prevention).
        #: **The more text is used as a feature, the
        #: larger the inflation from duplicates**, so it is on by default
        self.check_leakage = check_leakage

        if escalate_rate is not None and not 0.0 <= escalate_rate <= 1.0:
            raise MekikiError(
                f"escalate_rate must be between 0.0 and 1.0: {escalate_rate}")
        if isinstance(signal, str) and signal not in SIGNALS:
            raise MekikiError(
                f"signal={signal!r} is not implemented. Use one of {SIGNALS} or "
                "a function f(X, evidence) -> array.")
        if signal == "jev" and self.jev is None:
            raise MekikiError(
                "signal='jev' routes by the confidence of the Jev tier, so it needs "
                "jev=True or jev=JevClient(). Or pick another signal.")
        self.escalate_rate = escalate_rate
        self.threshold = threshold
        self.signal = signal
        #: Approved answers (row fingerprint -> value). Grows through approve()
        self.approved_: dict[str, object] = {}
        self._warned_all_rows = False
        self._warned_jev = False

    def _with_mask_default(self, spec: ColumnSpec) -> ColumnSpec:
        if spec.mask_amounts_in_long_text is None:
            # In regression the answer may be written as a number, so mask. Not in classification
            spec = replace(spec, mask_amounts_in_long_text=(self.task == "regression"))
        return spec

    @property
    def classify(self) -> bool:
        return self.task == "classification"

    @property
    def target_name(self) -> str:
        return self.domain.name_of(self.target)

    @property
    def _routing(self) -> bool:
        """Whether only some rows are sent to the LLM."""
        if self.escalate_rate is not None:
            return self.escalate_rate < 1.0
        return self.threshold is not None

    def _cut_text(self) -> str:
        if self.escalate_rate is not None:
            return f"top {self.escalate_rate:.0%} by signal"
        if self.threshold is not None:
            return f"signal >= {self.threshold}"
        return "all rows"

    def __repr__(self) -> str:
        sig = self.signal if isinstance(self.signal, str) else "custom"
        name = type(self).__name__
        task = "" if name != "EvidencePredictor" else f"task={self.task!r}, "
        return f"{name}(target={self.target!r}, {task}signal={sig!r}, {self._cut_text()})"

    # --- fit ----------------------------------------------------------

    def _encode_labels(self, y: np.ndarray) -> np.ndarray:
        """Map labels to indices 0..C-1. Labels unseen in training stop the run."""
        lookup = {c: i for i, c in enumerate(self.classes_.tolist())}
        try:
            return np.array([lookup[v] for v in y.tolist()], dtype=int)
        except KeyError as e:
            raise MekikiError(
                f"Label not present in the training data: {e.args[0]!r}. "
                f"The candidates are {self.classes_.tolist()}.") from e

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None
            ) -> EvidencePredictor:
        """Memorise the training data. **No separate teacher labels are created here.**

        Omitting `y` uses `X[target]`. The answers in the training data become the
        few-shot examples as they are.

        If no column was given, every column of `X` except `target` is assigned by the
        same rules `diagnose` starts from (dtype, distinct values, mean length; no LLM
        call). Identifier, date and constant columns are left out. The assignment used
        is in `spec_`; pass columns explicitly to override it.
        """
        if self.spec is not None:
            self.spec.check(X)
            self.spec_ = self.spec
        else:
            from mekiki.diagnose import infer_spec

            spec = self._with_mask_default(infer_spec(X, self.target))
            if not (spec.numeric or spec.text):
                raise MekikiError(
                    "Could not find a numeric or short-text column to retrieve similar "
                    f"cases with (inferred: {spec}). Pass the columns explicitly, e.g. "
                    "numeric=[...] or text=...")
            self.spec_ = spec
        if y is None:
            if self.target not in X.columns:
                raise MekikiError(
                    f"Target {self.target!r} is not in X. "
                    "Pass y, or set target to a column name.")
            y = X[self.target]
        if len(y) != len(X):
            raise MekikiError(f"X and y differ in length: {len(X)} != {len(y)}")

        self.train_ = X.reset_index(drop=True)
        if self.classify:
            y = np.asarray(pd.Series(y).to_numpy())
            if pd.isna(y).any():
                raise MekikiError("The classification labels contain missing values. "
                                  "Drop those rows.")
            self.classes_ = np.unique(y)
            if len(self.classes_) > MAX_CLASSES:
                raise MekikiError(
                    f"There are {len(self.classes_)} classes (limit {MAX_CLASSES}). "
                    "With that many candidates the LLM cannot pick one. "
                    "Fold them into coarser groups with SemanticEncoder, or use regression.")
            if len(self.classes_) < 2:
                raise MekikiError("Classification needs at least two distinct labels.")
            self.y_ = y
            self.y_idx_ = self._encode_labels(y)
            y_for_models = self.y_idx_
        else:
            y = np.asarray(y, dtype=float)
            self.y_ = y
            y_for_models = y
        if self.check_leakage:
            warn_if_leaky(
                check_duplicates(self.train_, keys=self.spec_.all_columns()),
                where="training data")
        self.index_ = NeighbourIndex(self.spec_, k=self.n_examples,
                                     w=self.neighbour_weight,
                                     vectorizer=self.vectorizer).fit(self.train_, self.y_)

        self.models_ = self._resolve_models()
        for m in self.models_:
            m.fit(self.train_, y_for_models)

        # Fixed part of the prompt. Rebuilt on every fit (the classes may change)
        self.class_labels_ = ([str(c) for c in self.classes_]
                              if self.classify else [])
        self.system_prompt_ = build_system_prompt(
            self.domain, self.task, self.target, classes=self.class_labels_,
            unit=self.unit, masked=bool(self.spec_.mask_amounts_in_long_text
                                        and self.spec_.long_text))
        if self.domain.answer_schema is not None:
            self.answer_schema_ = self.domain.answer_schema
        elif self.classify:
            self.answer_schema_ = _classification_schema(self.target_name,
                                                         self.class_labels_)
        else:
            self.answer_schema_ = _regression_schema(self.target_name, self.unit)
        self.predictions_ = None
        return self

    def _default_models(self) -> list[BaseModel]:
        return [
            TreeModel("LightGBM", self.spec_, kind="lgbm", task=self.task),
            TreeModel("XGBoost", self.spec_, kind="xgb", task=self.task),
            NeighbourModel(f"{self.n_examples}-NN "
                           f"{'class rates' if self.classify else 'median'}",
                           self.index_, task=self.task),
        ]

    def _resolve_models(self) -> list[BaseModel]:
        """Expand `models=`: omitted means the default three, a list replaces them, and
        the string "default" inside the list stands for the default three."""
        if self._models_arg is None:
            return self._default_models()
        given = ([self._models_arg] if isinstance(self._models_arg, str)
                 else list(self._models_arg))
        out: list[BaseModel] = []
        for m in given:
            if isinstance(m, str):
                if m != DEFAULT_MODELS:
                    raise MekikiError(
                        f"Unknown entry {m!r} in models=. The only string allowed is "
                        f"{DEFAULT_MODELS!r}, which stands for the default models.")
                out.extend(self._default_models())
            else:
                out.append(m)
        names = [m.name for m in out]
        repeated = sorted({n for n in names if names.count(n) > 1})
        if repeated:
            raise MekikiError(
                f"Model names must be unique, but {repeated} appear more than once in "
                "models=. The name is how each model's prediction is shown to the LLM.")
        return out

    def _check_fitted(self) -> None:
        if not hasattr(self, "train_"):
            raise MekikiError("Call fit first.")

    # --- evidence -------------------------------------------------------

    def evidence(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """Outputs of the statistical models: (n,) numbers for regression,
        (n, C) probabilities for classification."""
        out = {}
        for m in self.models_:
            if self.classify:
                if not hasattr(m, "predict_proba"):
                    raise MekikiError(
                        f"Model {m.name!r} has no predict_proba. Classification uses "
                        "the probability of each candidate as evidence, so it is required.")
                p = np.asarray(m.predict_proba(X), dtype=float)
                if p.shape != (len(X), len(self.classes_)):
                    raise MekikiError(
                        f"predict_proba of model {m.name!r} has shape "
                        f"{p.shape}; expected ({len(X)}, {len(self.classes_)}).")
                out[m.name] = p
            else:
                out[m.name] = np.asarray(m.predict(X), dtype=float)
        return out

    def _row_evidence(self, evidence: dict[str, np.ndarray], i: int) -> dict[str, Any]:
        return {name: (v[i] if self.classify else float(v[i]))
                for name, v in evidence.items()}

    def _base_values(self, evidence: dict[str, np.ndarray]) -> np.ndarray:
        """Answers for rows that do not go to the LLM: the output of `models[0]`
        (the most probable label for classification)."""
        base = evidence[self.models_[0].name]
        if self.classify:
            return self.classes_[np.argmax(base, axis=1)]
        return base.astype(float).copy()

    def _proba_text(self, p: np.ndarray) -> str:
        """Render a probability vector as "Yes 0.71 / No 0.29"."""
        return " / ".join(f"{self.domain.label(c)} {float(v):.2f}"
                          for c, v in zip(self.classes_.tolist(), p, strict=True))

    def _truth_text(self, v: Any) -> str:
        return self.domain.label(v) if self.classify else _fmt(v)

    # --- signals (only what is available before calling the LLM) ----------

    def _signal_values(self, X: pd.DataFrame, evidence: dict[str, np.ndarray],
                       jev_preds: list[Prediction | None] | None = None) -> np.ndarray:
        """"How much this row should go to the LLM". Larger means send.

        When every row is sent and the default signal cannot be computed (fewer than
        two models take part in it), return NaN instead of stopping: the signal is then not
        used for routing, only attached to the provenance. `signal="jev"` reads the
        Jev answers of every row (`jev_preds`, computed by the caller first).
        """
        if self.signal == "jev":
            if jev_preds is None or len(jev_preds) != len(X):
                raise MekikiError(
                    "signal='jev' needs Jev's answer for every row before routing. "
                    "Pass jev=True or jev=JevClient() to the predictor.")
            # A row Jev could not answer is the least certain of all, so it is sent
            return np.array([1.0 - p.confidence if p is not None else 1.0
                             for p in jev_preds], dtype=float)
        if callable(self.signal):
            s = np.asarray(self.signal(X, evidence), dtype=float)
            if s.shape != (len(X),):
                raise MekikiError(
                    f"The signal function must return an array of length {len(X)}: {s.shape}")
            return s
        if self.signal == "disagreement":
            outputs = [evidence[m.name] for m in self.models_
                       if getattr(m, "in_disagreement", False)]
            if len(outputs) < 2 and not self._routing:
                return np.full(len(X), np.nan)
            return disagreement(outputs, self.classify)
        if self.signal == "similarity":
            _, sim = self.index_.query(X)
            return -sim[:, 0]                 # rows without a similar case score higher
        return unseen_score(X, self.train_, self.spec_.categorical, self.spec_.text)

    def _row_keys(self, X: pd.DataFrame) -> list[str]:
        cols = [c for c in self.spec_.all_columns() if c in X.columns]
        return [row_key(X.iloc[i], cols) for i in range(len(X))]

    def _plan_rows(self, X: pd.DataFrame, evidence: dict[str, np.ndarray],
                   jev_preds: list[Prediction | None] | None = None,
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """(signal, rows sent to the LLM, approved rows, row fingerprints).
        Shared by plan and predict."""
        s = self._signal_values(X, evidence, jev_preds)
        sel = select(s, self.escalate_rate, self.threshold)
        keys = self._row_keys(X)
        approved = np.array([k in self.approved_ for k in keys], dtype=bool)
        return s, sel & ~approved, approved, keys

    def _warn_if_all_rows(self, n_llm: int) -> None:
        """When neither a rate nor a threshold was given and every row is sent, warn once
        if the cost is noticeable."""
        if (self._warned_all_rows or self.escalate_rate is not None
                or self.threshold is not None):
            return
        est = n_llm * unit_cost(self.client)
        if est <= COST_WARN_USD:
            return
        self._warned_all_rows = True
        warnings.warn(
            f"No escalation limit is set, so all {n_llm} rows will be sent to the LLM "
            f"(estimated ${est:.2f}, about "
            f"{latency_seconds(self.client, n_llm, with_setup=False):.0f} seconds; "
            "cached rows are free). Pass escalate_rate=0.3 or similar to limit it to "
            "the rows the statistical models are unsure about. If you mean to send every "
            "row, write escalate_rate=1.0 and this warning goes away.",
            MekikiWarning, stacklevel=3)

    # --- predict ------------------------------------------------------

    def _build_prompt(self, row: pd.Series, evidence: dict[str, Any],
                      neighbours: list[dict]) -> str:
        d, tn = self.domain, self.target_name
        u = f" (unit: {self.unit})" if self.unit and not self.classify else ""
        parts = [f"## {_heading(d.subject_heading)}\n\n{_describe_row(row, self.spec_)}"]
        if self.classify:
            parts.append("\n## Evidence 1: statistical model predictions "
                         "(probability per candidate)\n")
            for name, v in evidence.items():
                parts.append(f"- {name}: {self._proba_text(v)}")
        else:
            parts.append(f"\n## Evidence 1: statistical model predictions{u}\n")
            for name, v in evidence.items():
                parts.append(f"- {name}: {v:,.6g}")
        parts.append(f"\n## Evidence 2: similar cases (training data with known actual {tn}){u}\n")
        for j, nb in enumerate(neighbours, start=1):
            parts.append(f"### Case {j} (similarity {nb['similarity']:.3f})"
                         f" actual {tn}: {self._truth_text(nb['label'])}")
            parts.append(_describe_row(pd.Series(nb["attributes"]), self.spec_,
                                       is_target=False))
            parts.append("")
        if self.classify:
            parts.append(f"Choose the {tn} of this {d.subject} from the candidates, "
                         "and give the probability of each candidate.")
        else:
            parts.append(f"Give one value for the {tn} of this {d.subject}{u}.")
        return "\n".join(parts)

    def _parse_answer(self, ans: LLMAnswer) -> tuple[Any, np.ndarray | None] | None:
        """Extract (value, probability vector) from the LLM answer. None if unreadable."""
        if not ans.ok:
            return None
        if not self.classify:
            key = self.domain.answer_key
            if key not in ans.data:
                return None
            return float(ans.data[key]), None
        label = str(ans.data.get("label", ""))
        if label not in self.class_labels_:
            return None
        i = self.class_labels_.index(label)
        raw = ans.data.get("probabilities") or {}
        p = np.array([max(float(raw.get(c, 0.0)), 0.0) for c in self.class_labels_])
        if not np.isfinite(p).all() or p.sum() <= 0:
            p = np.zeros(len(self.classes_))
            p[i] = 1.0
        return self.classes_[i], p / p.sum()

    def _fallback_answer(self, ev: dict[str, Any]) -> tuple[Any, np.ndarray | None]:
        name = self.models_[0].name
        if self.classify:
            p = np.asarray(ev[name], dtype=float)
            return self.classes_[int(np.argmax(p))], p
        return float(ev[name]), None

    def _neighbour_rows(self, sub: pd.DataFrame) -> list[list[dict]]:
        """Retrieve the similar cases of every row of `sub`, in the shape the prompts,
        the Jev state and the provenance share (one list of dicts per row)."""
        idx, sim = self.index_.query(sub)
        out = []
        for k in range(len(sub)):
            out.append([{"train_row": int(j), "similarity": float(s),
                         "label": self.index_.y_[j].item()
                         if hasattr(self.index_.y_[j], "item") else self.index_.y_[j],
                         "attributes": self.train_.iloc[j][self.spec_.all_columns()].to_dict()}
                        for j, s in zip(idx[k], sim[k], strict=True)])
        return out

    # --- the Jev tier -------------------------------------------------------

    def _jev_ready(self, need: bool) -> bool:
        """Whether the Jev tier runs now. `need` is True for `signal="jev"`, which cannot
        work without it (stop); otherwise a missing key only warns once and the two-tier
        behaviour continues."""
        if self.jev is None:
            if need:
                raise MekikiError(
                    "signal='jev' needs the Jev tier: pass jev=True or jev=JevClient().")
            return False
        if self.jev.available():
            return True
        reason = self.jev.why_unavailable()
        if need:
            raise MekikiError(f"signal='jev' needs Jev, but it cannot be called. {reason}")
        if not self._warned_jev:
            self._warned_jev = True
            warnings.warn(f"The Jev tier is skipped: {reason}", MekikiWarning, stacklevel=3)
        return False

    def _jev_state(self, row: pd.Series, neighbours: list[dict]) -> dict[str, Any]:
        """What Jev sees: the row and its similar cases, rendered the same way as for the LLM."""
        tn = self.target_name
        return {
            _heading(self.domain.subject_heading): _describe_row(row, self.spec_),
            f"similar cases with known {tn} (closest first)": [
                {"similarity": round(float(nb["similarity"]), 3),
                 f"actual {tn}": self._truth_text(nb["label"]),
                 "attributes": _describe_row(pd.Series(nb["attributes"]), self.spec_,
                                             is_target=False)}
                for nb in neighbours],
        }

    def _jev_descriptions(self, ev: dict[str, Any]) -> dict[str, str]:
        """One option per statistical model, described by its prediction for the row."""
        if self.classify:
            return {name: f"predicts {self._proba_text(v)}" for name, v in ev.items()}
        u = f" {self.unit}" if self.unit else ""
        return {name: f"predicts {v:,.6g}{u}" for name, v in ev.items()}

    def _jev_instructions(self) -> str:
        d, tn = self.domain, self.target_name
        return (f"You are {d.role}. Several statistical models predicted the {tn} of this "
                f"{d.subject}; each option is one model, described by its prediction. Pick "
                f"the model whose prediction is most likely right, judging from the similar "
                f"cases (whose actual {tn} is known) and the attributes of {d.subject_heading}.")

    def _ask_jev(self, X: pd.DataFrame, evidence: dict[str, np.ndarray],
                 rows: np.ndarray, verbose: bool = False) -> list[Prediction | None]:
        """Put `rows` (positions in X) to Jev and return one `Prediction` per row, or None
        where Jev could not answer (the caller then keeps `models[0]`).

        The prediction is the probability-weighted average of the models' outputs
        (probability vectors for classification, then the most probable class).
        """
        if len(rows) == 0:
            return []
        sub = X.iloc[rows].reset_index(drop=True)
        neighbour_rows = self._neighbour_rows(sub)
        names = [m.name for m in self.models_]
        states, descs = [], []
        for k, gi in enumerate(rows):
            states.append(self._jev_state(sub.iloc[k], neighbour_rows[k]))
            descs.append(self._jev_descriptions(self._row_evidence(evidence, int(gi))))

        def show(done: int, total: int) -> None:
            if verbose and (done % 100 == 0 or done == total):
                print(f"    Jev {done}/{total} rows (cost ${self.jev.usage.cost:.4f}"
                      f" / cache hits {self.jev.usage.cache_hits})")

        results = self.jev.choose_many(states, names, self._jev_instructions(), descs,
                                       progress=show if verbose else None)
        preds: list[Prediction | None] = []
        for k, (gi, r) in enumerate(zip(rows, results, strict=True)):
            if r.choice is None:
                preds.append(None)
                continue
            ev = self._row_evidence(evidence, int(gi))
            w = {n: max(float(r.probabilities.get(n, 0.0)), 0.0) for n in names}
            total = sum(w.values())
            if total <= 0:           # probabilities missing: the chosen model takes all
                w = {n: 1.0 if n == r.choice else 0.0 for n in names}
                total = 1.0
            w = {n: v / total for n, v in w.items()}
            if self.classify:
                proba = np.zeros(len(self.classes_), dtype=float)
                for n in names:
                    proba += w[n] * np.asarray(ev[n], dtype=float)
                proba = proba / proba.sum() if proba.sum() > 0 else proba
                value: Any = self.classes_[int(np.argmax(proba))]
            else:
                proba = None
                value = float(sum(w[n] * float(ev[n]) for n in names))
            reason = "Jev weighted " + " / ".join(f"{n} {w[n]:.2f}" for n in names)
            preds.append(Prediction(value=value, confidence=float(r.confidence),
                                    reason=reason, origin="jev", model_predictions=ev,
                                    neighbours=neighbour_rows[k], cost=r.cost,
                                    from_cache=r.from_cache, proba=proba))
        return preds

    def _jev_unit_cost(self) -> float:
        return unit_cost(self.jev, default=self.jev.estimated_cost_per_call())

    def _jev_seconds(self, n: int) -> float:
        return latency_seconds(self.jev, n, with_setup=False,
                               seconds_per_call=JEV_SECONDS_PER_CALL)

    # --- the frontier LLM ---------------------------------------------------------

    def _ask_llm(self, X: pd.DataFrame, evidence: dict[str, np.ndarray],
                 rows: np.ndarray, verbose: bool = False) -> list[Prediction]:
        """Send only `rows` (positions in X) to the LLM and return per-row provenance.

        Prompts are built independently per row, so which rows are sent together does
        not change their content. The cache from a full-row measurement therefore hits
        unchanged when only some rows are sent.
        """
        if len(rows) == 0:
            return []
        sub = X.iloc[rows].reset_index(drop=True)
        # 1. Retrieve similar cases (the model outputs for all rows come from the caller)
        neighbour_rows = self._neighbour_rows(sub)

        # 2. Build a prompt per row
        prompts = []
        for k, gi in enumerate(rows):
            ev = {name: v[gi] for name, v in evidence.items()}
            prompts.append(self._build_prompt(sub.iloc[k], ev, neighbour_rows[k]))

        # 3. Send them in a batch
        def show(done: int, total: int) -> None:
            if verbose and (done % 25 == 0 or done == total):
                print(f"    LLM {done}/{total} rows"
                      f" (cost ${self.client.usage.cost:.3f}"
                      f" / cache hits {self.client.usage.cache_hits})")

        answers = self.client.ask_many(self.system_prompt_, prompts,
                                       self.answer_schema_,
                                       progress=show if verbose else None)

        # 4. Assemble provenance. Rows without an answer fall back to the first model
        fallback_name = self.models_[0].name
        preds: list[Prediction] = []
        for k, (gi, ans) in enumerate(zip(rows, answers, strict=True)):
            ev = self._row_evidence(evidence, int(gi))
            parsed = self._parse_answer(ans)
            if parsed is not None:
                value, proba = parsed
                p = Prediction(value=value,
                               confidence=float(ans.data.get("confidence", 0.0)),
                               reason=str(ans.data.get("reason", "")),
                               origin="llm", model_predictions=ev,
                               neighbours=neighbour_rows[k],
                               cost=ans.cost, from_cache=ans.from_cache,
                               proba=proba)
            else:
                if self.fallback == "error":
                    raise MekikiError(f"The LLM could not answer row {int(gi)}: "
                                      f"{ans.error or ans.data}")
                value, proba = self._fallback_answer(ev)
                p = Prediction(value=value, confidence=0.0,
                               reason=f"the LLM could not answer, so substituted "
                                      f"the {fallback_name} prediction",
                               origin="fallback", model_predictions=ev,
                               neighbours=neighbour_rows[k],
                               cost=ans.cost, error=ans.error or "could not parse the answer",
                               proba=proba)
            preds.append(p)
        return preds

    def predict(self, X: pd.DataFrame, verbose: bool = False) -> np.ndarray:
        """Return one prediction per row: numbers for regression, labels for classification.

        Which rows go to the LLM is decided by `escalate_rate` / `threshold` (omitted:
        every row). The other rows return the prediction of `models[0]` (LightGBM by
        default) unchanged, exactly as when the routing curve was measured.
        Approved rows (`approve`) return the approved value without calling the LLM.
        """
        self._check_fitted()
        self.spec_.check(X)
        X = X.reset_index(drop=True)
        if self.check_leakage:
            warn_if_leaky(check_overlap(self.train_, X,
                                        keys=self.spec_.all_columns()),
                          where="train and test")

        # 1. Run the statistical models and choose the rows to send from their output.
        #    With signal="jev" the routing signal is Jev's confidence, so Jev goes first
        evidence = self.evidence(X)
        jev_on = self._jev_ready(need=self.signal == "jev")
        jev_all = (self._ask_jev(X, evidence, np.arange(len(X)), verbose)
                   if self.signal == "jev" else None)
        s, sel, approved, keys = self._plan_rows(X, evidence, jev_all)
        self._warn_if_all_rows(int(sel.sum()))
        conf = to_confidence(s) if self._routing else np.zeros(len(X))

        # 2. Fill in the provenance of the rows not sent (statistical model or approved answer)
        base_name = self.models_[0].name
        base = self._base_values(evidence)
        preds: list[Prediction] = []
        for i in range(len(X)):
            ev = self._row_evidence(evidence, i)
            proba = (np.asarray(evidence[base_name][i], dtype=float)
                     if self.classify else None)
            if approved[i]:
                preds.append(Prediction(
                    value=self.approved_[keys[i]], confidence=1.0,
                    reason="reused an approved answer (LLM not called)",
                    origin="human", model_predictions=ev, neighbours=[],
                    proba=proba))
            else:
                preds.append(Prediction(
                    value=base[i], confidence=float(conf[i]),
                    reason=f"signal was weak, so left to {base_name} without calling the LLM",
                    origin="model", model_predictions=ev, neighbours=[],
                    proba=proba))

        # 2b. The middle tier: rows neither sent nor approved get Jev's weighting of the
        #     models instead of models[0]. A row Jev could not answer keeps models[0]
        if jev_on:
            fast = np.flatnonzero(~sel & ~approved)
            jp = ([jev_all[int(i)] for i in fast] if jev_all is not None
                  else self._ask_jev(X, evidence, fast, verbose))
            for i, p in zip(fast, jp, strict=True):
                if p is not None:
                    preds[int(i)] = p
                else:
                    preds[int(i)].reason = (f"Jev could not answer, so left to {base_name} "
                                            "without calling the LLM")

        # 3. Send only the selected rows to the LLM
        rows = np.flatnonzero(sel)
        for gi, p in zip(rows, self._ask_llm(X, evidence, rows, verbose), strict=True):
            preds[int(gi)] = p

        self.predictions_ = preds
        self.signal_ = s
        self.selected_ = sel
        self.keys_ = keys
        self.last_X_ = X
        return self._as_array([p.value for p in preds])

    def _as_array(self, values: Sequence[Any]) -> np.ndarray:
        """Turn the predictions into an array. For classification the label dtype
        follows `classes_`."""
        if self.classify:
            return np.asarray(values, dtype=self.classes_.dtype)
        return np.asarray(values, dtype=float)

    def predict_proba(self, X: pd.DataFrame, verbose: bool = False) -> np.ndarray:
        """Probability of each candidate per row (in `classes_` order). Classification only.

        Internally the same single inference as `predict`. Rows sent to the LLM carry
        the LLM's probabilities; rows not sent, and rows it could not answer, carry the
        probabilities of `models[0]`.
        """
        if not self.classify:
            raise MekikiError("predict_proba is only available with task='classification'.")
        self.predict(X, verbose=verbose)
        return self.proba_

    @property
    def proba_(self) -> np.ndarray:
        """Probability matrix of the last predict (classification only)."""
        preds = self._require()
        if not self.classify:
            raise MekikiError("proba_ only exists with task='classification'.")
        return np.vstack([p.proba for p in preds])

    # --- estimate before running ("visible before committing a run") ------------------

    def plan(self, X: pd.DataFrame) -> dict:
        """**Before calling**: how many rows go to the LLM, how much it costs, how long it takes.

        The statistical models run but the LLM is not called, so it is **free**
        (with `signal="jev"` Jev is asked for every row, which is cheap and cached).
        Use it to pick `escalate_rate` to fit a budget.

        Parameters
        ----------
        X:
            The rows `predict` would be called on. The target column is not needed.

        Returns
        -------
        A dict:

        - `n_rows`: number of rows in X.
        - `signal`: name of the routing signal ("custom" for a function of your own).
        - `escalation_rule`: the cut in words (a rate or a threshold).
        - `n_escalated`: rows that would go to the LLM.
        - `n_skipped_approved`: rows answered from approved examples, without the LLM.
        - `n_fast_path`: the remaining rows, which keep the statistical answer.
        - `estimated_cost_usd`, `estimated_seconds`: for the `n_escalated` rows.
        - `cost_per_row_usd`: the unit cost behind the estimate.
        - with `jev` set: `n_jev` (the fast-path rows Jev will weigh), `jev_available`,
          `jev_cost_per_row_usd`, `estimated_jev_cost_usd` and `estimated_jev_seconds`.
        """
        self._check_fitted()
        self.spec_.check(X)
        X = X.reset_index(drop=True)
        evidence = self.evidence(X)
        jev_on = self._jev_ready(need=self.signal == "jev")
        jev_all = (self._ask_jev(X, evidence, np.arange(len(X)))
                   if self.signal == "jev" else None)
        _, sel, approved, _ = self._plan_rows(X, evidence, jev_all)
        n_llm = int(sel.sum())
        n_fast = len(X) - n_llm - int(approved.sum())
        out = {
            "n_rows": len(X),
            "signal": self.signal if isinstance(self.signal, str) else "custom",
            "escalation_rule": self._cut_text(),
            "n_escalated": n_llm,
            "n_skipped_approved": int(approved.sum()),
            "n_fast_path": n_fast,
            "estimated_cost_usd": round(n_llm * unit_cost(self.client), 4),
            "estimated_seconds": round(latency_seconds(self.client, n_llm), 1),
            "cost_per_row_usd": round(unit_cost(self.client), 6),
        }
        if self.jev is not None:
            n_jev = n_fast if jev_on else 0
            out["n_jev"] = n_jev
            out["jev_available"] = jev_on
            out["jev_cost_per_row_usd"] = round(self._jev_unit_cost(), 6)
            out["estimated_jev_cost_usd"] = round(n_jev * self._jev_unit_cost(), 4)
            out["estimated_jev_seconds"] = round(self._jev_seconds(n_jev), 1)
        return out

    # --- inspection API (provenance) ------------------------------------------

    def _require(self, X: pd.DataFrame | None = None) -> list[Prediction]:
        """Return the result of the last predict. Passing X detects a mix-up.

        The inspection API can only answer about "the X most recently predicted".
        Silently returning the previous result for a different X would mean judging
        from the wrong provenance, so stop when the lengths differ.
        """
        if getattr(self, "predictions_", None) is None:
            raise MekikiError("Call predict first.")
        if X is not None and len(X) != len(self.predictions_):
            raise MekikiError(
                f"The last predict covered {len(self.predictions_)} rows but "
                f"X has {len(X)} rows. The inspection API returns the result of the "
                "most recent predict, so pass the same X.")
        return self.predictions_

    def _path(self, i: int) -> str:
        if self.selected_[i]:
            return "llm"
        return "jev" if self.predictions_[i].origin == "jev" else "fast"

    def route(self) -> pd.DataFrame:
        """Which route each row took, together with the signal value."""
        preds = self._require()
        return pd.DataFrame([
            {"row": i,
             "route": self._path(i),
             "source": p.origin,
             "signal": float(self.signal_[i]),
             "prediction": p.value,
             "confidence": p.confidence,
             "cost_usd": p.cost}
            for i, p in enumerate(preds)])

    def confidence(self, X: pd.DataFrame | None = None) -> pd.Series:
        """Confidence per row: the LLM's self-report for rows sent to it, the signal
        rescaled to 0-1 for rows not sent, and 1 for approved rows.

        Returns the result of the last `predict`. Passing X checks the row count.
        """
        return pd.Series([p.confidence for p in self._require(X)],
                         name="confidence")

    def examples(self, i: int | None = None) -> pd.DataFrame:
        """Similar cases used in the inference. With i, only that row's.
        **Only rows sent to the LLM have them.**"""
        preds = self._require()
        rows = []
        for k, p in enumerate(preds):
            if i is not None and k != i:
                continue
            for rank, nb in enumerate(p.neighbours, start=1):
                rows.append({"row": k, "rank": rank, "train_row": nb["train_row"],
                             "similarity": nb["similarity"], "label": nb["label"],
                             **nb["attributes"]})
        return pd.DataFrame(rows)

    def cost(self, X: pd.DataFrame | None = None) -> dict:
        """Cost incurred, its breakdown, and the comparison with sending every row.

        Before any predict, only the client's running totals are returned.
        """
        u = self.client.summary()
        preds = getattr(self, "predictions_", None)
        if X is not None and preds is not None and len(X) != len(preds):
            raise MekikiError(
                f"The last predict covered {len(preds)} rows but "
                f"X has {len(X)} rows.")
        if preds:
            n = len(preds)
            n_llm = int(self.selected_.sum())
            paid = [p for i, p in enumerate(preds)
                    if self.selected_[i] and not p.from_cache]
            per_row = unit_cost(self.client)
            u["n_predicted"] = n
            u["n_escalated"] = n_llm
            u["escalated_rate"] = round(n_llm / n, 4)
            u["llm_answered_rate"] = round(sum(p.origin == "llm" for p in preds) / n, 4)
            u["n_skipped_approved"] = int(sum(p.origin == "human" for p in preds))
            u["cost_per_row_usd"] = round(
                sum(p.cost for p in paid) / max(len(paid), 1), 6)
            u["estimated_all_rows_usd"] = round(n * per_row, 4)
            u["saved_usd"] = round((n - n_llm) * per_row, 4)
        if self.jev is not None:
            u["jev"] = self.jev.summary()
            if preds:
                jev_rows = [p for p in preds if p.origin == "jev"]
                u["n_jev_answered"] = len(jev_rows)
                u["jev_cost_usd"] = round(sum(p.cost for p in jev_rows if not p.from_cache), 6)
        return u

    def _evidence_lines(self, p: Prediction) -> list[str]:
        """For explain: one line per statistical model output."""
        lines = []
        for name, v in p.model_predictions.items():
            shown = self._proba_text(v) if self.classify else _fmt(v)
            lines.append(f"  - {name}: {shown}")
        return lines

    def _value_text(self, p: Prediction) -> str:
        if self.classify:
            s = self.domain.label(p.value)
            return f"{s} ({self._proba_text(p.proba)})" if p.proba is not None else s
        return f"{_fmt(p.value)} {self.unit}".rstrip()

    def explain(self, i: int) -> str:
        """Why that row got that value, as prose with the evidence.

        When only some rows are sent, it starts with which route the row took.
        """
        p = self._require()[i]
        lines = [f"[row {i}] prediction {self._value_text(p)}"
                 f" (source {p.origin} / confidence {p.confidence:.2f})"]
        if self._routing:
            if self.selected_[i]:
                path = "LLM"
            elif p.origin == "jev":
                path = "Jev (LLM not called)"
            else:
                path = "fast (LLM not called)"
            lines.append(f"route {path} / signal {float(self.signal_[i]):,.6g}"
                         f" (escalation rule: {self._cut_text()})")
        lines += ["", "Statistical model predictions:"]
        lines += self._evidence_lines(p)
        if p.neighbours:
            lines.append("")
            lines.append("Similar cases consulted:")
            for rank, nb in enumerate(p.neighbours, start=1):
                desc = _describe_row(pd.Series(nb["attributes"]), self.spec_,
                                     indent="      ", is_target=False)
                lines.append(f"  {rank}. {self.target_name} {self._truth_text(nb['label'])}"
                             f" (similarity {nb['similarity']:.3f})")
                lines.append(desc)
        lines.append("")
        if p.origin in ("llm", "fallback"):
            who = "LLM reason"
        elif p.origin == "jev":
            who = "Jev reason"
        else:
            who = "reason"
        lines.append(f"{who}: {p.reason}")
        if p.error:
            lines.append(f"error: {p.error}")
        return "\n".join(lines)

    def _evidence_columns(self, p: Prediction) -> dict[str, Any]:
        """For provenance: raw numbers for regression, the most probable label and its
        probability for classification."""
        out: dict[str, Any] = {}
        for k, v in p.model_predictions.items():
            if self.classify:
                j = int(np.argmax(v))
                out[f"evidence_{k}"] = self.classes_[j]
                out[f"evidence_{k}_proba"] = float(v[j])
            else:
                out[f"evidence_{k}"] = v
        return out

    def provenance(self) -> pd.DataFrame:
        """Provenance of every row as a table: `explain` in list form, plus route and signal."""
        return pd.DataFrame([
            {"row": i, "prediction": p.value, "source": p.origin,
             "route": self._path(i), "signal": float(self.signal_[i]),
             "confidence": p.confidence, "cost_usd": p.cost,
             "from_cache": p.from_cache, "reason": p.reason,
             **self._evidence_columns(p)}
            for i, p in enumerate(self._require())
        ])

    def report(self) -> str:
        """Summary of what one predict did."""
        preds = self._require()
        n = len(preds)
        n_llm = int(self.selected_.sum())
        n_human = sum(p.origin == "human" for p in preds)
        n_jev = sum(p.origin == "jev" for p in preds)
        paid = sum(p.cost for p in preds if not p.from_cache)
        per_row = unit_cost(self.client)
        lines = [
            f"{n} rows predicted (signal: "
            f"{self.signal if isinstance(self.signal, str) else 'custom'} / "
            f"{self._cut_text()})",
            f"  sent to LLM:          {n_llm} rows ({n_llm / max(n, 1):.1%})",
            f"  skipped (approved):   {n_human} rows",
        ]
        if self.jev is not None:
            lines.append(f"  answered by Jev:      {n_jev} rows")
        lines += [
            f"  statistical models:   {n - n_llm - n_human - n_jev} rows",
            f"  cost:                 ${paid:.4f}"
            f" (estimated ${n * per_row:.4f} if every row were sent)",
            f"  estimated time:       {latency_seconds(self.client, n_llm):.1f} s",
        ]
        if self.jev is not None:
            lines.append(f"  Jev time:             {self._jev_seconds(n_jev):.1f} s")
        return "\n".join(lines)

    # --- active learning (LLM answers are queued as label candidates) ------------------

    def review_queue(self) -> pd.DataFrame:
        """Rows the LLM answered = teacher-label candidates. Approving them widens the
        fast path next time."""
        preds = self._require()
        rows = []
        for i, p in enumerate(preds):
            if p.origin != "llm":
                continue
            rec = {"row": i, "fingerprint": self.keys_[i], "llm_answer": p.value,
                   "confidence": p.confidence, "signal": float(self.signal_[i]),
                   "reason": p.reason}
            for c in self.spec_.all_columns():
                if c in self.last_X_.columns:
                    rec[c] = self.last_X_[c].iloc[i]
            rows.append(rec)
        return pd.DataFrame(rows)

    def approve(self, rows: Sequence[int] | None = None) -> int:
        """Approve answers. **From then on those rows return without calling the LLM.**

        Omitting `rows` approves the whole latest queue.
        Matching is by row content (fingerprint), not row number, so it still works
        when the next predict has a different order or count.
        Returns the number of rows approved.
        """
        preds = self._require()
        targets = (range(len(preds)) if rows is None
                   else [int(r) for r in rows])
        n = 0
        for i in targets:
            if preds[i].origin != "llm":
                continue
            v = preds[i].value
            self.approved_[self.keys_[i]] = v if self.classify else float(v)
            n += 1
        return n

    # --- sweep the rate (accuracy, cost and time visible at once) -----------------

    def curve(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None,
              rates: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0),
              verbose: bool = False) -> pd.DataFrame:
        """**Accuracy, cost and latency** in one table as the rate is varied.

        **This function sends every row to the LLM.** Drawing the curve needs "what
        would it have answered" for every row; from the second run on the disk cache
        makes it free (the same row is never charged twice). The cost and latency
        columns are the values "if only that fraction had been sent".

        The accuracy column is `MAE` for regression and `accuracy` for classification.
        Omitting `y` uses `X[target]`.

        Parameters
        ----------
        X:
            Test rows with known answers. Keep it small: every row is billed once.
        y:
            The true values. Leave out to read them from `X[target]`.
        rates:
            The shares of rows to evaluate, each between 0 and 1.

        Returns
        -------
        A `pd.DataFrame` with one row per rate and the columns:

        - `rate`: the share of rows sent, as given in `rates`.
        - `n_escalated`: the number of rows that share amounts to (largest signal first).
        - `MAE` or `accuracy`: the score when those rows take the LLM's answer and the rest
          keep the statistical answer.
        - `cost_usd`, `estimated_seconds`: estimates for that many rows, from the unit cost
          per row (not the amount actually billed by this call).
        - with `jev` set, the rows not sent take Jev's weighted average instead of the
          statistical answer, and `jev_cost_usd` estimates that tier's cost.

        The result of the last `predict` is discarded, so call `predict` again before `explain`.
        """
        self._check_fitted()
        self.spec_.check(X)
        X = X.reset_index(drop=True)
        if y is None:
            if self.target not in X.columns:
                raise MekikiError(
                    f"Label column {self.target!r} is not in X. Pass y.")
            y = X[self.target]
        y = (np.asarray(pd.Series(y).to_numpy()) if self.classify
             else np.asarray(y, dtype=float))
        if self.check_leakage:
            warn_if_leaky(check_overlap(self.train_, X,
                                        keys=self.spec_.all_columns()),
                          where="train and test")

        # Every row is sent, so the provenance of the last predict becomes invalid.
        # Keeping it silently would make explain() return another row's evidence
        self.predictions_ = None

        evidence = self.evidence(X)
        jev_on = self._jev_ready(need=self.signal == "jev")
        jev_all = self._ask_jev(X, evidence, np.arange(len(X)), verbose) if jev_on else None
        s = self._signal_values(X, evidence, jev_all)
        if np.isnan(s).any():
            raise MekikiError(
                "Drawing the curve needs a signal. Keep the default tree models "
                "(models=['default', ...]), set in_disagreement = True on at least two "
                "models, or pass signal='similarity' or similar.")
        base = self._base_values(evidence)
        if jev_all is not None:
            # What predict would return for the rows not sent: Jev's weighting where it answered
            for i, p in enumerate(jev_all):
                if p is not None:
                    base[i] = p.value
        llm = self._as_array([p.value for p in
                              self._ask_llm(X, evidence, np.arange(len(X)), verbose)])

        order = np.argsort(-s, kind="stable")
        per_row = unit_cost(self.client)
        rows = []
        for r in rates:
            k = int(round(len(X) * float(r)))
            pred = base.copy()
            if k:
                pick = order[:k]
                pred[pick] = llm[pick]
            score = ({"accuracy": float(np.mean(pred == y))} if self.classify
                     else {"MAE": float(np.mean(np.abs(pred - y)))})
            rec = {
                "rate": r,
                "n_escalated": k,
                **score,
                "cost_usd": round(k * per_row, 4),
                "estimated_seconds": round(latency_seconds(self.client, k), 1),
            }
            if self.jev is not None:
                n_jev = (len(X) - k) if jev_on else 0
                rec["jev_cost_usd"] = round(n_jev * self._jev_unit_cost(), 4)
                rec["jev_seconds"] = round(self._jev_seconds(n_jev), 1)
            rows.append(rec)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# The two task-specific entry points (scikit-learn style)
# ---------------------------------------------------------------------

class EvidenceRegressor(EvidencePredictor):
    """`EvidencePredictor` fixed to regression: `predict` returns numbers.

    Takes every argument of `EvidencePredictor` except `task`. `unit` ("USD", "pts")
    appears in the prompt so that the LLM does not get the order of magnitude wrong.
    """

    _estimator_type = "regressor"

    def __init__(self, target: str, unit: str = "", *,
                 domain: Domain | None = None,
                 numeric: Sequence[str] = (), boolean: Sequence[str] = (),
                 categorical: Sequence[str] = (), text: str | None = None,
                 long_text: str | None = None,
                 spec: ColumnSpec | None = None,
                 models: Sequence[BaseModel | str] | None = None,
                 n_examples: int = 5, neighbour_weight: float = 0.15,
                 vectorizer: Vectorizer | None = None,
                 client: ClaudeClient | None = None,
                 jev: JevClient | bool | None = None,
                 fallback: str = "best_model",
                 check_leakage: bool = True,
                 escalate_rate: float | None = None,
                 threshold: float | None = None,
                 signal: str | Callable[..., np.ndarray] = "disagreement") -> None:
        super().__init__(target, unit, task="regression", domain=domain,
                         numeric=numeric, boolean=boolean, categorical=categorical,
                         text=text, long_text=long_text, spec=spec, models=models,
                         n_examples=n_examples, neighbour_weight=neighbour_weight,
                         vectorizer=vectorizer, client=client, jev=jev, fallback=fallback,
                         check_leakage=check_leakage, escalate_rate=escalate_rate,
                         threshold=threshold, signal=signal)


class EvidenceClassifier(EvidencePredictor):
    """`EvidencePredictor` fixed to classification: `predict` returns labels and
    `predict_proba` returns probabilities in `classes_` order.

    Takes every argument of `EvidencePredictor` except `task` and `unit`. The labels are
    taken from the `target` column of the training data (up to `MAX_CLASSES` classes).
    """

    _estimator_type = "classifier"

    def __init__(self, target: str, *,
                 domain: Domain | None = None,
                 numeric: Sequence[str] = (), boolean: Sequence[str] = (),
                 categorical: Sequence[str] = (), text: str | None = None,
                 long_text: str | None = None,
                 spec: ColumnSpec | None = None,
                 models: Sequence[BaseModel | str] | None = None,
                 n_examples: int = 5, neighbour_weight: float = 0.15,
                 vectorizer: Vectorizer | None = None,
                 client: ClaudeClient | None = None,
                 jev: JevClient | bool | None = None,
                 fallback: str = "best_model",
                 check_leakage: bool = True,
                 escalate_rate: float | None = None,
                 threshold: float | None = None,
                 signal: str | Callable[..., np.ndarray] = "disagreement") -> None:
        super().__init__(target, "", task="classification", domain=domain,
                         numeric=numeric, boolean=boolean, categorical=categorical,
                         text=text, long_text=long_text, spec=spec, models=models,
                         n_examples=n_examples, neighbour_weight=neighbour_weight,
                         vectorizer=vectorizer, client=client, jev=jev, fallback=fallback,
                         check_leakage=check_leakage, escalate_rate=escalate_rate,
                         threshold=threshold, signal=signal)
