"""`diagnose` — given data and a target, return an initial mekiki configuration.

## Why this is needed

`SemanticEncoder` and `EvidencePredictor` both run only **after a human
has decided which column to treat how**: the numeric / categorical / text assignment, the
target transform, the columns to exclude from duplicate matching, and the words that tell the
LLM about the domain (`Domain`). All of it is written by hand for every new dataset, and the
places where people get it wrong are always the same. (`EvidencePredictor` applies the
stage-1 column assignment below by itself when no column is given; `diagnose` is where the
grounds, the warnings and the LLM's reading of column meanings come from.)

- The target is skewed but not log-transformed (Craigslist price: mean is 5x the median)
- The same item appears in several rows, yet the split is random (42% duplicates there)
- Identifier- or date-like columns are fed in as features
- LLM cost is spent on a text column that does not explain the target (seen on a used-car dataset)
- Classification without looking at class imbalance (PetFinder class 0 is 2.7%)

`diagnose()` fills this gap in **two stages**.

1. **Rule-based diagnosis (no LLM call, free).** Column kinds, target distribution,
   duplicates, explanatory power of text (`screen()`), and a cost estimate for `EvidencePredictor`.
   The result is a `Diagnosis`
2. **The LLM reads the diagnosis table and writes the configuration (one call).** The LLM
   receives only the table plus each column's name and a few sample values, never the data
   itself. It is asked only for what rules cannot tell: **the meaning of columns** (a region
   column differs between duplicate rows of the same item, a serial number is an identifier,
   a free-text column tends to state the answer, a maker / model column could be looked up
   for attributes the table lacks). The result is a `Recommendation`

Where no LLM is available, the `Recommendation` is built from stage 1 alone. It is coarser
because it lacks the interpretation of column meanings, but `spec` and `to_code()` still work.
`diagnose` has no Jev tier: reading a diagnosis table and writing a configuration is free-form
judgement, not a typed question, so it goes straight to the frontier LLM.

    from mekiki import diagnose

    rec = diagnose(df, target="price", unit="USD")
    print(rec)                 # diagnosis and recommendation, human-readable
    rec.spec                   # ColumnSpec. Can be passed straight to EvidencePredictor
    rec.domain                 # Domain. Draft of role / subject / target name
    print(rec.to_code())       # Python that reproduces the recommended configuration

**The LLM's recommendations must be grounded in the numbers of the diagnosis table**
(`reasons`). Just as `EvidencePredictor` "decides on evidence", ungrounded advice is rejected here.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from mekiki.errors import MekikiError
from mekiki.leakage import DuplicateReport, check_duplicates
from mekiki.llm import ClaudeClient
from mekiki.predictor import ColumnSpec, Domain
from mekiki.screening import ScreeningReport, screen

TASKS = ("regression", "classification")

#: Column kinds. Those that go into `ColumnSpec` are numeric / boolean / categorical / text /
#: long_text. The rest (id / datetime / constant) are not used as features.
KINDS = ("numeric", "boolean", "categorical", "text", "long_text",
         "id", "datetime", "constant")

#: String columns whose mean length is at least this are treated as free text (`long_text`).
#: Craigslist descriptions average 1,075 characters, wine tasting notes 243, PetFinder
#: profiles 339. Short strings such as product names or lists of features fall within
#: 10-60 characters.
LONG_TEXT_CHARS = 100

#: String columns with at most this many distinct values are treated as categorical.
CATEGORY_MAX_UNIQUE = 200

#: If the ratio of distinct values to rows is at least this, suspect an identifier.
ID_UNIQUE_RATE = 0.95

#: A numeric target with at most this many distinct (integer) values is treated as
#: classification. PetFinder's AdoptionSpeed (0-4) becomes classification, wine points
#: (80-100) regression.
CLASSIFICATION_MAX_CLASSES = 10

#: Mean / median ratio above which the target counts as "strongly skewed" and a log
#: transform is recommended.
SKEW_RATIO = 1.5

#: Warn about "strong imbalance" when the minority class rate is below this.
IMBALANCE_RATE = 0.10

#: Warn about columns whose missing rate is at least this.
MISSING_WARN_RATE = 0.5

#: Columns recognisable as identifiers / dates by name alone. Case-insensitive, partial match.
_ID_NAMES = re.compile(r"(^|_)(id|uuid|guid|vin|url|hash|key|index|no|number)($|_)",
                       re.IGNORECASE)
#: The same in camelCase / PascalCase ("RescuerID", "PetId", "IDLink"). Case-sensitive, so
#: words that merely contain the letters ("Paid", "Idle") do not match.
_ID_CAMEL = re.compile(r"(?<=[a-z0-9])(ID|Id)(?=$|[A-Z0-9_])|^(ID|Id)(?=[A-Z0-9_])")
_DATE_NAMES = re.compile(r"(date|time|posted|created|updated|_at$|timestamp)",
                         re.IGNORECASE)
_URL = re.compile(r"^\s*https?://", re.IGNORECASE)
#: The name pandas gives an unnamed CSV column. It is the original row number, so an id.
_INDEX_NAMES = re.compile(r"^unnamed: ?\d+$", re.IGNORECASE)

#: The stage-2 answer is long (column assignment, reasons, `SemanticEncoder` candidates). With the
#: default 1,024 it gets cut off and the JSON breaks, so `diagnose()` uses this value when it
#: creates its own client. When passing your own `ClaudeClient`, set `max_tokens` to at least
#: this.
DIAGNOSE_MAX_TOKENS = 4096
_BOOL_VALUES = {"0", "1", "true", "false", "yes", "no", "y", "n", "t", "f"}


# =====================================================================
# Diagnosis (stage 1)
# =====================================================================


@dataclass
class ColumnProfile:
    """Diagnosis of one column."""

    name: str
    kind: str                      # one of KINDS
    dtype: str
    n_unique: int
    unique_rate: float
    missing_rate: float
    mean_chars: float = 0.0        # string columns only
    samples: list[str] = field(default_factory=list)
    note: str = ""                 # grounds for the decision, or doubts. Also shown to the LLM

    def as_dict(self) -> dict[str, Any]:
        d = {"column": self.name, "kind": self.kind, "dtype": self.dtype,
             "n_unique": self.n_unique, "unique_rate": round(self.unique_rate, 3),
             "missing_rate": round(self.missing_rate, 3)}
        if self.mean_chars:
            d["mean_chars"] = round(self.mean_chars, 1)
        if self.samples:
            d["samples"] = self.samples
        if self.note:
            d["note"] = self.note
        return d


@dataclass
class TargetProfile:
    """Diagnosis of the target. Regression and classification fill different fields."""

    name: str
    task: str
    n: int
    n_missing: int
    # regression
    median: float = float("nan")
    mean: float = float("nan")
    minimum: float = float("nan")
    maximum: float = float("nan")
    p995: float = float("nan")
    skew_ratio: float = float("nan")      # mean / median
    n_nonpositive: int = 0
    suggest_log: bool = False
    suggest_clip: bool = False
    # classification
    classes: dict[str, int] = field(default_factory=dict)
    minority_rate: float = float("nan")
    imbalanced: bool = False
    #: stringified label -> actual value (to tell 0 from "0"). Not included in the JSON
    native_classes: dict[str, Any] = field(default_factory=dict, repr=False)

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"column": self.name, "task": self.task,
                             "n_rows": self.n, "missing": self.n_missing}
        if self.task == "regression":
            d.update({"median": _r(self.median), "mean": _r(self.mean),
                      "min": _r(self.minimum), "max": _r(self.maximum),
                      "p995": _r(self.p995),
                      "mean_over_median": _r(self.skew_ratio),
                      "n_nonpositive": self.n_nonpositive,
                      "recommend_log_target": self.suggest_log,
                      "recommend_clip_outliers": self.suggest_clip})
        else:
            d.update({"classes": self.classes,
                      "minority_rate": _r(self.minority_rate),
                      "imbalanced": self.imbalanced})
        return d


@dataclass
class Diagnosis:
    """Stage-1 result. Printing it gives a readable report. The LLM receives `as_dict()`."""

    n_rows: int
    n_cols: int
    target: TargetProfile
    columns: list[ColumnProfile]
    duplicates: DuplicateReport | None = None
    duplicates_by_text: DuplicateReport | None = None
    screening: dict[str, ScreeningReport] = field(default_factory=dict)
    #: Description of the preprocessing applied to the target before measuring the
    #: explanatory power of text (empty if none)
    screening_note: str = ""
    cost_estimate: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def by_kind(self, kind: str) -> list[str]:
        return [c.name for c in self.columns if c.kind == kind]

    def column(self, name: str) -> ColumnProfile | None:
        return next((c for c in self.columns if c.name == name), None)

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "n_rows": self.n_rows, "n_columns": self.n_cols,
            "target": self.target.as_dict(),
            "columns": [c.as_dict() for c in self.columns],
        }
        if self.duplicates is not None:
            d["duplicates_all_columns"] = {
                "n_duplicates": self.duplicates.n_duplicate_rows,
                "rate": _r(self.duplicates.rate),
                "compared_columns": self.duplicates.columns}
        if self.duplicates_by_text is not None:
            d["duplicates_text_only"] = {
                "n_duplicates": self.duplicates_by_text.n_duplicate_rows,
                "rate": _r(self.duplicates_by_text.rate),
                "compared_columns": self.duplicates_by_text.columns}
        if self.screening:
            d["text_explanatory_power"] = {
                col: {"contribution": _r(r.text_contribution), "verdict": r.verdict,
                      "metric": r.metric, "score_without_text": _r(r.score_without_text),
                      "score_with_text": _r(r.score_with_text)}
                for col, r in self.screening.items()}
            if self.screening_note:
                d["text_explanatory_power"]["how_measured"] = self.screening_note
        if self.cost_estimate:
            d["feature_b_on_all_rows"] = self.cost_estimate
        if self.warnings:
            d["warnings"] = self.warnings
        return d

    def __str__(self) -> str:
        t = self.target
        lines = [f"Diagnosis ({self.n_rows:,} rows x {self.n_cols} columns)", ""]
        if t.task == "regression":
            lines.append(
                f"  Target {t.name}: regression. median {_fmt(t.median)} / mean {_fmt(t.mean)}"
                f" / max {_fmt(t.maximum)} (mean/median {t.skew_ratio:.2f})")
            if t.suggest_log:
                lines.append("    -> strongly skewed. A log transform (log1p) is recommended")
            if t.suggest_clip:
                lines.append(
                    f"    -> the max is an order of magnitude above the 99.5th percentile"
                    f" ({_fmt(t.p995)}). Clipping outliers is recommended")
        else:
            dist = " / ".join(f"{k}: {v:,} ({100 * v / max(t.n, 1):.1f}%)"
                              for k, v in t.classes.items())
            lines.append(f"  Target {t.name}: classification. {dist}")
            if t.imbalanced:
                lines.append(
                    f"    -> the minority class is only {t.minority_rate:.1%}."
                    " Use a stratified split and evaluate with log loss")
        if t.n_missing:
            lines.append(f"    missing in {t.n_missing:,} rows (excluded from the diagnosis)")
        lines.append("")
        lines.append("  Column kinds:")
        for kind, label in (("numeric", "numeric"), ("boolean", "boolean"),
                            ("categorical", "categorical"), ("text", "short text"),
                            ("long_text", "free text"), ("id", "id-like"),
                            ("datetime", "date-like"), ("constant", "constant/empty")):
            names = self.by_kind(kind)
            if names:
                lines.append(f"    {label:<14} {', '.join(names)}")
        notes = [c for c in self.columns if c.note]
        if notes:
            lines.append("  Columns with doubts:")
            for c in notes:
                lines.append(f"    {c.name}: {c.note}")
        lines.append("")
        if self.duplicates is not None:
            d = self.duplicates
            lines.append(f"  Duplicates (compared on {len(d.columns)} columns): "
                         + (f"{d.n_duplicate_rows:,} rows ({d.rate:.1%})" if not d.ok else "none"))
        if self.duplicates_by_text is not None:
            d = self.duplicates_by_text
            lines.append(f"  Duplicates (compared on {', '.join(d.columns)} only): "
                         + (f"{d.n_duplicate_rows:,} rows ({d.rate:.1%})" if not d.ok else "none"))
        for col, r in self.screening.items():
            lines.append(f"  Text {col}: contribution {r.text_contribution * 100:+.1f}%"
                         f" ({r.metric} {r.score_without_text:,.4g} -> {r.score_with_text:,.4g})"
                         f" -> {r.verdict}")
        if self.screening and self.screening_note:
            lines.append(f"    Note: {self.screening_note}")
        if self.cost_estimate:
            c = self.cost_estimate
            lines.append(f"  EvidencePredictor on all rows: about ${c['estimated_cost_usd']:,.2f}"
                         f" / about {c['estimated_minutes']:,.0f} min"
                         f" (${c['cost_per_row_usd']} per row)")
        if self.warnings:
            lines.append("")
            lines.append("  Warnings:")
            lines.extend(f"    - {w}" for w in self.warnings)
        return "\n".join(lines)


def _r(x: float) -> float | None:
    """A rounded number that fits in JSON. NaN becomes None."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    return round(float(x), 4)


def _fmt(x: float) -> str:
    if not math.isfinite(x):
        return "-"
    return f"{x:,.4g}" if abs(x) < 1e6 else f"{x:,.3e}"


def _id_like_name(name: str) -> bool:
    return bool(_ID_NAMES.search(name) or _ID_CAMEL.search(name))


def _looks_like_date(s: pd.Series) -> bool:
    """Whether a string column looks like dates. The first 20 non-null values must parse."""
    v = s.dropna().astype(str).head(20)
    if v.empty or not v.str.contains(r"[-/:]").mean() > 0.9:
        return False
    parsed = pd.to_datetime(v, errors="coerce", format="mixed")
    return bool(parsed.notna().mean() > 0.9)


def _token_novelty(text: pd.Series, n: int = 500) -> float:
    """How little vocabulary is shared across rows (0-1).

    Short phrases such as "model sport edition" share words, so the value is low; codes such
    as "WBA3A5C51CF256789" are close to 1. Used to tell identifiers from short phrases.
    """
    tokens = text.head(n).str.split().explode().dropna()
    if tokens.empty:
        return 1.0
    return float(tokens.nunique() / len(tokens))


def profile_column(s: pd.Series, n_rows: int, n_samples: int = 3) -> ColumnProfile:
    """Decide the kind of one column. Order matters (constant -> date -> boolean -> numeric
    -> string)."""
    name = str(s.name)
    non_null = s.dropna()
    n_unique = int(non_null.nunique())
    missing_rate = 1.0 - len(non_null) / n_rows if n_rows else 0.0
    unique_rate = n_unique / len(non_null) if len(non_null) else 0.0
    dtype = str(s.dtype)
    samples = [str(v)[:60] for v in non_null.drop_duplicates().head(n_samples)]
    p = ColumnProfile(name=name, kind="numeric", dtype=dtype, n_unique=n_unique,
                      unique_rate=unique_rate, missing_rate=missing_rate,
                      samples=samples)

    if n_unique <= 1:
        p.kind = "constant"
        p.note = "At most one distinct value. Cannot be a feature"
        return p
    if _INDEX_NAMES.match(name):
        p.kind = "id"
        p.note = "Unnamed CSV column (original row number). Not used as a feature"
        return p
    if pd.api.types.is_datetime64_any_dtype(s):
        p.kind = "datetime"
        return p
    if pd.api.types.is_bool_dtype(s):
        p.kind = "boolean"
        return p
    if n_unique == 2 and set(non_null.astype(str).str.lower().unique()) <= _BOOL_VALUES:
        # Tree models cast boolean columns straight to float, so only numeric 0/1 becomes
        # boolean. Strings such as "Yes"/"No" are passed as categorical (same meaning, no
        # breakage)
        if pd.api.types.is_numeric_dtype(s):
            p.kind = "boolean"
        else:
            p.kind = "categorical"
            p.note = ("Two-valued strings. Passed as categorical"
                      " (only numeric 0/1 becomes boolean)")
        return p

    if pd.api.types.is_numeric_dtype(s):
        is_int = pd.api.types.is_integer_dtype(s) or bool(
            np.all(np.mod(non_null.to_numpy(dtype=float), 1) == 0))
        if is_int and unique_rate >= ID_UNIQUE_RATE and _id_like_name(name):
            p.kind = "id"
            p.note = "Integer, distinct in nearly every row, and the name looks like an id"
            return p
        p.kind = "numeric"
        if is_int and unique_rate >= ID_UNIQUE_RATE and len(non_null) >= 100:
            p.note = "Integer, distinct in nearly every row. Drop from features if it is an id"
        elif is_int and n_unique <= CLASSIFICATION_MAX_CLASSES:
            p.note = f"Integer with only {n_unique} distinct values. May be a code (category)"
        return p

    # strings from here on
    text = non_null.astype(str)
    p.mean_chars = float(text.str.len().mean())
    if text.head(20).str.match(_URL).mean() > 0.9:
        p.kind = "id"
        p.note = "URL. Not a feature as is"
        return p
    if _id_like_name(name) and unique_rate >= ID_UNIQUE_RATE:
        p.kind = "id"
        p.note = "Distinct in nearly every row, and the name looks like an id"
        return p
    if _id_like_name(name) and p.mean_chars < 40 and unique_rate >= 0.5:
        p.kind = "id"
        p.note = (f"The name looks like an id but the same value appears in several rows"
                  f" (unique rate {unique_rate:.0%}). The same item may be listed multiple times")
        return p
    if _looks_like_date(text):
        p.kind = "datetime"
        if not _DATE_NAMES.search(name):
            p.note = "Values parse as dates, but the name does not say so"
        return p
    if p.mean_chars >= LONG_TEXT_CHARS:
        p.kind = "long_text"
        return p
    if (unique_rate >= ID_UNIQUE_RATE and len(non_null) >= 100 and p.mean_chars < 40
            and _token_novelty(text) > 0.8):
        # distinct in nearly every row and no vocabulary shared across rows -> code-like
        p.kind = "id"
        p.note = ("Short strings, distinct in nearly every row, sharing no words."
                  " Drop from features if it is an id")
        return p
    if n_unique <= CATEGORY_MAX_UNIQUE or unique_rate <= 0.02:
        p.kind = "categorical"
        return p
    p.kind = "text"
    p.note = f"Short strings with many ({n_unique:,}) distinct values. Could also be categorical"
    return p


def profile_target(y: pd.Series, task: str | None = None) -> TargetProfile:
    """Diagnose the target distribution and decide the task type."""
    name = str(y.name)
    non_null = y.dropna()
    n_missing = int(len(y) - len(non_null))
    if task is None:
        numeric = pd.api.types.is_numeric_dtype(non_null) and not pd.api.types.is_bool_dtype(non_null)
        if not numeric:
            task = "classification"
        else:
            vals = non_null.to_numpy(dtype=float)
            few = non_null.nunique() <= CLASSIFICATION_MAX_CLASSES
            task = "classification" if few and np.all(np.mod(vals, 1) == 0) else "regression"
    if task not in TASKS:
        raise MekikiError(f"task must be one of {TASKS}: {task!r}")

    p = TargetProfile(name=name, task=task, n=int(len(non_null)), n_missing=n_missing)
    if task == "classification":
        counts = non_null.value_counts()
        p.classes = {str(k): int(v) for k, v in counts.sort_index().items()}
        p.native_classes = {str(k): k for k in counts.sort_index().index}
        p.minority_rate = float(counts.min() / counts.sum()) if len(counts) else float("nan")
        p.imbalanced = bool(p.minority_rate < IMBALANCE_RATE)
        return p

    v = non_null.to_numpy(dtype=float)
    if len(v) == 0:
        return p
    p.median, p.mean = float(np.median(v)), float(np.mean(v))
    p.minimum, p.maximum = float(v.min()), float(v.max())
    p.p995 = float(np.quantile(v, 0.995))
    p.skew_ratio = p.mean / p.median if p.median > 0 else float("nan")
    p.n_nonpositive = int((v <= 0).sum())
    p.suggest_log = bool(math.isfinite(p.skew_ratio) and p.skew_ratio >= SKEW_RATIO
                         and p.minimum >= 0)
    p.suggest_clip = bool(p.p995 > 0 and p.maximum > 10 * p.p995)
    return p


def _pick_text_columns(columns: list[ColumnProfile]) -> tuple[str | None, str | None]:
    """`ColumnSpec` has one slot each for text / long_text. With several candidates, pick the
    one that looks most informative."""
    shorts = [c for c in columns if c.kind == "text"]
    longs = [c for c in columns if c.kind == "long_text"]
    text = max(shorts, key=lambda c: c.mean_chars * c.n_unique).name if shorts else None
    long_text = max(longs, key=lambda c: c.mean_chars).name if longs else None
    return text, long_text


def spec_from_profiles(columns: list[ColumnProfile]) -> ColumnSpec:
    """The column assignment decided by rules alone. Identifier, date and constant columns
    are left out; short-text columns that lose the single `text` slot become categorical."""
    text, long_text = _pick_text_columns(columns)

    def kind(k: str) -> list[str]:
        return [c.name for c in columns if c.kind == k]

    return ColumnSpec(numeric=kind("numeric"), boolean=kind("boolean"),
                      categorical=kind("categorical") + [
                          c.name for c in columns if c.kind == "text" and c.name != text],
                      text=text, long_text=long_text)


def infer_spec(df: pd.DataFrame, target: str | None = None) -> ColumnSpec:
    """Assign every column except `target` by rules alone (no LLM call, no model training).

    The same assignment `diagnose` starts from. `EvidencePredictor.fit` uses it when no
    column was given.
    """
    n_rows = len(df)
    return spec_from_profiles(
        [profile_column(df[c], n_rows) for c in df.columns if c != target])


def run_diagnosis(df: pd.DataFrame, target: str, *, task: str | None = None,
                  unit: str = "", target_name: str | None = None,
                  sample: int | None = 20_000, n_splits: int = 5,
                  screen_text: bool = True,
                  cost_per_row: float | None = None,
                  max_workers: int = 8) -> Diagnosis:
    """Stage 1. Build the diagnosis table without calling the LLM."""
    from mekiki.adaptive import COST_PER_ROW_USD, SECONDS_PER_CALL, SETUP_SECONDS

    if target not in df.columns:
        raise MekikiError(f"Target column not found: {target!r}")
    n_rows = len(df)
    if n_rows < 10:
        raise MekikiError(f"Too few rows ({n_rows}).")

    tp = profile_target(df[target], task)
    cols = [profile_column(df[c], n_rows) for c in df.columns if c != target]
    warnings_: list[str] = []

    if tp.task == "regression" and tp.suggest_log:
        warnings_.append(
            f"The mean of target {target} is {tp.skew_ratio:.1f}x its median. Training on MAE"
            " as is will be pulled towards the high end. Consider a log transform")
    if tp.task == "regression" and tp.suggest_clip:
        warnings_.append(
            f"The max of target {target}, {_fmt(tp.maximum)}, exceeds 10x the 99.5th percentile"
            f" {_fmt(tp.p995)}. Possible input error. Clip outliers before training")
    if tp.task == "regression" and tp.n_nonpositive:
        warnings_.append(f"Target {target} is <= 0 in {tp.n_nonpositive:,} rows."
                         " If it is a price, suspect input errors")
    if tp.task == "classification" and tp.imbalanced:
        warnings_.append(f"The minority class is only {tp.minority_rate:.1%}."
                         " Evaluate with a stratified split and look at log loss, not accuracy")
    for c in cols:
        if c.missing_rate >= MISSING_WARN_RATE and c.kind != "constant":
            warnings_.append(f"Missing rate of {c.name} is {c.missing_rate:.0%}")
    ids = [c.name for c in cols if c.kind == "id"]
    if ids:
        warnings_.append(f"Id-like columns {ids} are not used as features")

    # Duplicates. Ids, dates and constants "differ even for the same item", so they are
    # excluded from the comparison
    ignore = [c.name for c in cols if c.kind in ("id", "datetime", "constant")]
    dup = check_duplicates(df, ignore=ignore) if len(df.columns) > len(ignore) else None
    if dup is not None and not dup.ok:
        warnings_.append(f"{dup.n_duplicate_rows:,} rows ({dup.rate:.1%}) have identical content."
                         " Collapse to one row per item before evaluating")
    text_col, long_col = _pick_text_columns(cols)
    dup_text = None
    key = long_col or text_col
    if key is not None:
        non_empty = df[df[key].notna() & (df[key].astype(str).str.strip() != "")]
        if len(non_empty) >= 2:
            dup_text = check_duplicates(non_empty, keys=[key])
            if dup is not None and dup_text.n_duplicate_rows > dup.n_duplicate_rows:
                warnings_.append(
                    f"{dup_text.n_duplicate_rows:,} rows ({dup_text.rate:.1%}) share the same"
                    f" {key}. The same item may be listed multiple times with a different"
                    " region etc. Consider ignoring that column in duplicate detection")

    # Explanatory power of text (screen). Only when there is a text column
    screening: dict[str, ScreeningReport] = {}
    screening_note = ""
    numeric = [c.name for c in cols if c.kind == "numeric"]
    boolean = [c.name for c in cols if c.kind == "boolean"]
    categorical = [c.name for c in cols if c.kind == "categorical"]
    if screen_text and (text_col or long_col) and not (numeric or boolean or categorical):
        warnings_.append("No structured column at all, so the explanatory power of text"
                         " (which needs a baseline to compare against) cannot be measured")
    elif screen_text:
        # Outliers in the target dominate the MAE and hide the contribution of text
        # (the Craigslist sample has a max of $990M and showed a contribution of -100%).
        # Apply the recommended preprocessing (clip outliers, log transform) first.
        df_s, note = _preprocessed_for_screening(df, target, tp)
        screening_note = note
        for col in [c for c in (text_col, long_col) if c]:
            try:
                screening[col] = screen(
                    df_s, target=target, text=col, task=tp.task,
                    target_name=target_name, numeric=numeric, boolean=boolean,
                    categorical=categorical, unit=unit, n_splits=n_splits, sample=sample)
            except Exception as e:  # too few rows (MekikiError), column-name errors in the tree model, etc.
                warnings_.append(f"Skipped screening of {col} ({type(e).__name__}: {e})")

    per_row = cost_per_row if cost_per_row is not None else COST_PER_ROW_USD
    batches = math.ceil(tp.n / max(max_workers, 1))
    cost = {"n_rows": tp.n, "estimated_cost_usd": round(tp.n * per_row, 2),
            "estimated_minutes": round((SETUP_SECONDS + batches * SECONDS_PER_CALL) / 60, 1),
            "cost_per_row_usd": per_row}

    return Diagnosis(n_rows=n_rows, n_cols=len(df.columns), target=tp, columns=cols,
                     duplicates=dup, duplicates_by_text=dup_text, screening=screening,
                     screening_note=screening_note, cost_estimate=cost, warnings=warnings_)


def _preprocessed_for_screening(df: pd.DataFrame, target: str,
                                tp: TargetProfile) -> tuple[pd.DataFrame, str]:
    """Return the table with the recommended preprocessing applied to the target, plus a
    description of it. Regression only."""
    if tp.task != "regression" or not (tp.suggest_clip or tp.suggest_log):
        return df, ""
    out = df[df[target].notna()]
    steps = []
    if tp.suggest_clip:
        out = out[out[target] <= tp.p995]
        steps.append("dropped the top 0.5%")
    if tp.suggest_log:
        out = out[out[target] >= 0]
        out = out.assign(**{target: np.log1p(out[target].to_numpy(dtype=float))})
        steps.append("applied log1p (MAE is in log units)")
    return (out.reset_index(drop=True),
            "measured after preprocessing the target: " + "; ".join(steps))


# =====================================================================
# Recommendation (stage 2)
# =====================================================================


#: Column kinds that can serve as the key of a knowledge column. The same set that
#: `KnowledgeEncoder` marks as usable when it proposes columns on its own.
KNOWLEDGE_KEY_KINDS = ("numeric", "boolean", "categorical", "text")


def _knowledge_args(kc: dict[str, Any], target: str) -> str:
    """The argument list of a `KnowledgeEncoder(...)` line in `to_code()`."""
    args = [f"keys={list(kc.get('keys', []))!r}", f"attribute={kc.get('attribute')!r}",
            f"type={kc.get('type', 'category')!r}"]
    if kc.get("type") == "numeric":
        if kc.get("unit"):
            args.append(f"unit={kc['unit']!r}")
    else:
        args.append(f"values={list(kc.get('values', []))!r}")
    if kc.get("name"):
        args.append(f"name={kc['name']!r}")
    args.append(f"target={target!r}")
    return ", ".join(args)


@dataclass
class Recommendation:
    """The answer of `diagnose`. `spec` and `domain` can be passed straight to
    `EvidencePredictor`.

    source:
        "rule" if built from the diagnosis table by rules alone. "llm" if the LLM built it
        after reading the column meanings. When the LLM fails it falls back to "rule" and
        `warnings` says so
    reasons:
        Grounds for each recommendation: which numbers of the diagnosis table it came from.
        LLM recommendations always carry this
    typed_columns, knowledge_columns:
        Candidate columns, not yet built. `typed_columns` are `SemanticEncoder` candidates
        extracted from the free text (name / source / type / values / why).
        `knowledge_columns` are `KnowledgeEncoder` candidates supplied from general knowledge,
        keyed by structured columns (name / keys / attribute / type / values / unit / why).
        Both appear in `to_code()` as commented-out lines, because building them costs LLM
        calls and whether they help is only known after scoring
    """

    task: str
    spec: ColumnSpec
    domain: Domain
    unit: str = ""
    target: str = ""
    target_transform: str | None = None          # e.g. "log1p". None means no transform
    clip_outliers: bool = False
    dedup_ignore: list[str] = field(default_factory=list)
    typed_columns: list[dict[str, Any]] = field(default_factory=list)
    knowledge_columns: list[dict[str, Any]] = field(default_factory=list)
    use_evidence: bool = True
    escalate_rate: float | None = 0.3
    warnings: list[str] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)
    source: str = "rule"
    diagnosis: Diagnosis | None = None
    cost: dict[str, Any] = field(default_factory=dict)

    @property
    def classify(self) -> bool:
        return self.task == "classification"

    def to_code(self) -> str:
        """Python that reproduces the recommended configuration. Runs as is when pasted."""
        s = self.spec
        d = self.domain
        cls = "EvidenceClassifier" if self.classify else "EvidenceRegressor"
        imports = [cls, "Domain", "check_duplicates"]
        if self.typed_columns:
            imports.append("SemanticEncoder")
        if self.knowledge_columns:
            imports.append("KnowledgeEncoder")
        lines = ["import numpy as np", "from mekiki import " + ", ".join(imports), ""]
        if self.dedup_ignore:
            lines += ["# Compare duplicates without the columns that differ even for the"
                      " same item, then collapse to one row per item",
                      f"ignore = {self.dedup_ignore!r}",
                      "print(check_duplicates(df, ignore=ignore))",
                      "df = df.loc[~df.drop(columns=ignore).duplicated()].reset_index(drop=True)", ""]
        else:
            lines += ["print(check_duplicates(df))",
                      "df = df.drop_duplicates().reset_index(drop=True)", ""]
        if self.clip_outliers and not self.classify:
            lines += ["# The top 0.5% may be input errors, so drop them",
                      f"df = df[df[{self.target!r}] <= df[{self.target!r}].quantile(0.995)]", ""]
        if self.target_transform == "log1p" and not self.classify:
            lines += ["# Strongly skewed: train on the log and map predictions back with expm1",
                      f"df[{self.target + '_log'!r}] = np.log1p(df[{self.target!r}])", ""]
        for tc in self.typed_columns:
            lines += [f"# SemanticEncoder candidate: {tc.get('why', '')}".rstrip(),
                      f"# df[{tc.get('name')!r}] = SemanticEncoder(source={tc.get('source')!r}, "
                      f"type={tc.get('type', 'category')!r}, values={tc.get('values', [])!r})"
                      ".fit_transform(df)", ""]
        for kc in self.knowledge_columns:
            lines += [f"# Knowledge column candidate: {kc.get('why', '')}".rstrip(),
                      "# df = df.join(KnowledgeEncoder(" + _knowledge_args(kc, self.target)
                      + ").fit_transform(df))", ""]
        dom = [f"role={d.role!r}", f"subject={d.subject!r}"]
        if d.target_name:
            dom.append(f"target_name={d.target_name!r}")
        if d.class_names:
            dom.append(f"class_names={d.class_names!r}")
        if d.hints:
            dom.append("hints=" + json.dumps(d.hints, ensure_ascii=False))
        lines += ["domain = Domain(" + ", ".join(dom) + ")", ""]
        log_target = self.target_transform == "log1p" and not self.classify
        target = self.target + "_log" if log_target else self.target
        args = [f"target={target!r}"]
        if self.unit and not self.classify:
            args.append(f"unit={self.unit!r}")
        args.append("domain=domain")
        for k in ("numeric", "boolean", "categorical"):
            v = getattr(s, k)
            if v:
                args.append(f"{k}={v!r}")
        if s.text:
            args.append(f"text={s.text!r}")
        if s.long_text:
            args.append(f"long_text={s.long_text!r}")
        rate = self.escalate_rate if self.use_evidence else 0.0
        args.append(f"escalate_rate={rate}")
        lines += [f"model = {cls}(", "    " + ",\n    ".join(args) + ",", ")"]
        if not self.use_evidence:
            lines += ["# Little to gain from the LLM, so escalate_rate=0.0"
                      " (statistical models alone answer)"]
        lines += ["model.fit(train)",
                  "print(model.plan(test))   # how many rows, how much, how long (free)",
                  "pred = model.predict(test)"]
        if self.target_transform == "log1p" and not self.classify:
            lines += ["pred = np.expm1(pred)"]
        return "\n".join(lines)

    def __str__(self) -> str:
        s = self.spec
        lines = []
        if self.diagnosis is not None:
            lines += [str(self.diagnosis), "", ""]
        src = ("built by the LLM from the column meanings" if self.source == "llm"
               else "built by rules only (LLM not used)")
        lines += [f"Recommendation ({src})", ""]
        task = "regression" if not self.classify else "classification"
        tail = ""
        if self.target_transform and not self.classify:
            tail += f", target transformed with {self.target_transform}"
        if self.clip_outliers and not self.classify:
            tail += ", top 0.5% dropped as outliers"
        lines.append(f"  Task: {task}{tail}")
        lines.append(f"  spec: numeric={s.numeric}")
        if s.boolean:
            lines.append(f"        boolean={s.boolean}")
        lines.append(f"        categorical={s.categorical}")
        if s.text:
            lines.append(f"        text={s.text!r}")
        if s.long_text:
            mask = " (amounts masked)" if (s.mask_amounts_in_long_text
                                          or (s.mask_amounts_in_long_text is None
                                              and not self.classify)) else ""
            lines.append(f"        long_text={s.long_text!r}{mask}")
        d = self.domain
        dom = (f'  domain: role "{d.role}", subject "{d.subject}",'
               f' target "{d.name_of(self.target)}"')
        if self.unit and not self.classify:
            dom += f" ({self.unit})"
        lines.append(dom)
        if d.class_names:
            lines.append("          classes: " + " / ".join(f"{k}={v}" for k, v in d.class_names.items()))
        for h in d.hints:
            lines.append(f"          hint: {h}")
        if self.dedup_ignore:
            lines.append(f"  Ignored in duplicate detection: {self.dedup_ignore}")
        if self.typed_columns:
            lines.append("  SemanticEncoder candidates:")
            for tc in self.typed_columns:
                lines.append(f"    \"{tc.get('name')}\" ({tc.get('type', 'category')})"
                             f" <- {tc.get('source')}: {tc.get('values', [])}"
                             f"  {tc.get('why', '')}".rstrip())
        if self.knowledge_columns:
            lines.append("  Knowledge column candidates:")
            for kc in self.knowledge_columns:
                what = kc.get("values") or (kc.get("unit") or "number")
                lines.append(f"    \"{kc.get('name')}\" ({kc.get('type', 'category')})"
                             f" <- {kc.get('keys', [])}: {kc.get('attribute')} -> {what}"
                             f"  {kc.get('why', '')}".rstrip())
        ev = ("worth trying" if self.use_evidence
              else "little to gain (statistical models may be enough)")
        lines.append(f"  EvidencePredictor: {ev}")
        if self.use_evidence and self.escalate_rate is not None:
            lines.append(f"  adaptive: start with the top {self.escalate_rate:.0%}")
        if self.reasons:
            lines.append("")
            lines.append("  Reasons:")
            for k, v in self.reasons.items():
                lines.append(f"    {k}: {v}")
        if self.warnings:
            lines.append("")
            lines.append("  Warnings:")
            lines.extend(f"    - {w}" for w in self.warnings)
        if self.cost:
            lines.append("")
            lines.append(f"  LLM cost of this diagnosis: ${self.cost.get('cost_usd', 0):.4f}"
                         f" ({self.cost.get('n_calls', 0)} calls / "
                         f"{self.cost.get('cache_hits', 0)} cache hits)")
        return "\n".join(lines)


def _rule_recommendation(dg: Diagnosis, target: str, unit: str,
                         target_name: str | None) -> Recommendation:
    """Build the recommendation from the diagnosis table by rules alone. This is the default
    without an LLM, and also the draft the LLM starts from."""
    spec = spec_from_profiles(dg.columns)
    tp = dg.target
    reasons: dict[str, str] = {}
    domain = Domain(target_name=target_name)
    reasons["column_assignment"] = ("Decided mechanically from dtype, number of distinct values"
                                    " and mean length. Column meanings were not considered")

    verdicts = [r.verdict for r in dg.screening.values()]
    if not dg.screening:
        use_evidence = False
        rate = 0.0
        reasons["feature_b"] = ("No text column (or not measured), so there is no evidence that"
                                " letting the LLM read it adds anything. Start with the"
                                " statistical models")
    elif "worth_trying" in verdicts:
        use_evidence = True
        best = max(dg.screening.values(), key=lambda r: r.text_contribution)
        reasons["feature_b"] = (f"The text contribution of {best.text_column},"
                                f" {best.text_contribution:.1%}, exceeds the threshold"
                                f" {best.threshold:.0%}")
        rate = 1.0 if dg.cost_estimate.get("estimated_cost_usd", 0) <= 2 else 0.2
        reasons["adaptive"] = (f"About ${dg.cost_estimate.get('estimated_cost_usd', 0):,.2f}"
                               " for all rows. "
                               + ("Cheap, so send every row" if rate == 1.0
                                  else "Start with 20% to see the effect"))
    elif "inconclusive" in verdicts:
        use_evidence = True
        rate = 0.2
        reasons["feature_b"] = ("The difference in contribution is too small to decide."
                                " Measure on a small scale")
    else:
        use_evidence = False
        rate = 0.0
        reasons["feature_b"] = ("The text barely explains the target. Statistical models are"
                                " very likely enough")

    return Recommendation(
        task=tp.task, spec=spec, domain=domain, unit=unit, target=target,
        target_transform="log1p" if (tp.task == "regression" and tp.suggest_log) else None,
        clip_outliers=bool(tp.task == "regression" and tp.suggest_clip),
        dedup_ignore=dg.by_kind("id") + dg.by_kind("datetime") + dg.by_kind("constant"),
        typed_columns=[], knowledge_columns=[], use_evidence=use_evidence, escalate_rate=rate,
        warnings=list(dg.warnings), reasons=reasons, source="rule", diagnosis=dg)


# --- LLM ---------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an analyst who helps design machine learning pipelines.
The user is trying to predict a target from tabular data with the Python library mekiki.
What you receive is **not the data itself but a diagnosis table computed by rules** (per
column: dtype, number of distinct values, missing rate, a few sample values; the target
distribution; duplicates; the explanatory power of text; a cost estimate) together with a
draft configuration built from that table by rules alone.

Your job is to read the **meaning of the columns**, which rules cannot tell, and fix the draft.

Building blocks of mekiki:
- numeric / boolean / categorical: columns fed to the statistical models (LightGBM, XGBoost)
- text: a short text column (product name, list of features, ...). One column only. Also used
  to retrieve similar cases
- long_text: the free-text body. One column only. The LLM reads it. In regression, amounts
  inside it are masked automatically
- identifier, URL, date and constant columns are not used as features
- dedup_ignore: "columns that differ even for the same item" (region, posting date,
  identifiers, ...). Ignoring them in duplicate detection reveals items listed several times
- Domain: the words that tell the LLM what it is predicting. role ("You are ..."), subject
  (what one record is), target_name (what the target is called), class_names (meaning of the
  labels in classification), hints (guidance on where to look in the free text; 1 to 3 items)
- SemanticEncoder: candidate typed columns that could be extracted from the free
  text. List the possible values
- KnowledgeEncoder: candidate columns the table lacks but general knowledge could supply,
  keyed by the values of structured columns (e.g. maker and model -> body style, country and
  grape -> typical price range). keys must be numeric / boolean / categorical / text columns
  with a moderate number of distinct values, never the target or the free-text body. type is
  category / binary (list the values) or numeric (give the unit). Propose only attributes a
  well-read person could answer from the key values alone
- use_evidence: whether the LLM is worth using for the final decision. Ground it in the
  "text_explanatory_power" of the diagnosis table
- escalate_rate: the fraction of rows (0 to 1) sent to the LLM. Ground it in the cost estimate

Rules:
- Use only column names that appear in the diagnosis table. Never invent one
- **Give a reason for every recommendation. Reasons must be numbers from the diagnosis table
  or meanings readable from column names and sample values**
- Do not write general advice absent from the table (such as "XGBoost is good")
- Do not guess what you do not know; write "needs checking" in warnings instead
- Write in English
"""


def _answer_schema() -> dict:
    str_list = {"type": "array", "items": {"type": "string"}}
    nullable_str = {"type": ["string", "null"]}
    return {
        "type": "object",
        "properties": {
            "task": {"type": "string", "enum": list(TASKS)},
            "target_name": {"type": "string",
                            "description": "What the target is called (e.g. price, churn)"},
            "unit": {"type": "string",
                     "description": "Unit for regression (e.g. USD, JPY). Empty if"
                                    " classification or unknown"},
            "role": {"type": "string", "description": 'The "..." in "You are ..."'},
            "subject": {"type": "string",
                        "description": "What one record is (e.g. a customer, a wine, a car)"},
            "class_names": {"type": "array",
                            "description": "For classification, each label and its meaning."
                                           " Empty for regression",
                            "items": {"type": "object",
                                      "properties": {"label": {"type": "string"},
                                                     "meaning": {"type": "string"}},
                                      "required": ["label", "meaning"],
                                      "additionalProperties": False}},
            "hints": str_list,
            "numeric": str_list, "boolean": str_list, "categorical": str_list,
            "text": nullable_str, "long_text": nullable_str,
            "target_transform": {"type": "string", "enum": ["none", "log1p"]},
            "clip_outliers": {"type": "boolean"},
            "dedup_ignore": str_list,
            "typed_columns": {"type": "array",
                              "items": {"type": "object",
                                        "properties": {"name": {"type": "string"},
                                                       "source": {"type": "string"},
                                                       "type": {"type": "string",
                                                                "enum": ["category", "binary"]},
                                                       "values": str_list,
                                                       "why": {"type": "string"}},
                                        "required": ["name", "source", "type", "values", "why"],
                                        "additionalProperties": False}},
            "knowledge_columns": {"type": "array",
                                  "items": {"type": "object",
                                            "properties": {"name": {"type": "string"},
                                                           "keys": str_list,
                                                           "attribute": {"type": "string"},
                                                           "type": {"type": "string",
                                                                    "enum": ["category", "binary",
                                                                             "numeric"]},
                                                           "values": str_list,
                                                           "unit": {"type": "string"},
                                                           "why": {"type": "string"}},
                                            "required": ["name", "keys", "attribute", "type",
                                                         "values", "unit", "why"],
                                            "additionalProperties": False}},
            "use_evidence": {"type": "boolean"},
            "escalate_rate": {"type": "number"},
            "warnings": str_list,
            "reasons": {"type": "array",
                        "items": {"type": "object",
                                  "properties": {"topic": {"type": "string"},
                                                 "reason": {"type": "string"}},
                                  "required": ["topic", "reason"],
                                  "additionalProperties": False}},
        },
        "required": ["task", "target_name", "unit", "role", "subject", "class_names", "hints",
                     "numeric", "boolean", "categorical", "text", "long_text",
                     "target_transform", "clip_outliers", "dedup_ignore", "typed_columns",
                     "knowledge_columns", "use_evidence", "escalate_rate", "warnings", "reasons"],
        "additionalProperties": False,
    }


def build_user_prompt(dg: Diagnosis, draft: Recommendation, target: str,
                      unit: str, target_name: str | None) -> str:
    """The body shown to the LLM: diagnosis table and draft. The data itself is not included."""
    s = draft.spec
    draft_d = {
        "task": draft.task, "numeric": s.numeric, "boolean": s.boolean,
        "categorical": s.categorical, "text": s.text, "long_text": s.long_text,
        "target_transform": draft.target_transform or "none",
        "clip_outliers": draft.clip_outliers, "dedup_ignore": draft.dedup_ignore,
        "use_evidence": draft.use_evidence, "escalate_rate": draft.escalate_rate,
    }
    given = {"target_column": target}
    if unit:
        given["unit"] = unit
    if target_name:
        given["target_name"] = target_name
    return "\n".join([
        "## Specified by the user",
        json.dumps(given, ensure_ascii=False),
        "",
        "## Diagnosis table (computed by rules)",
        json.dumps(dg.as_dict(), ensure_ascii=False, indent=1),
        "",
        "## Draft (configuration built by rules alone)",
        json.dumps(draft_d, ensure_ascii=False, indent=1),
        "",
        "Fix the draft and write the reason for each recommendation in reasons.",
    ])


def _apply_llm_answer(data: dict[str, Any], dg: Diagnosis, draft: Recommendation,
                      unit: str) -> Recommendation:
    """Validate the LLM's answer and turn it into a `Recommendation`. Unknown column names
    are dropped and recorded in the warnings."""
    known = {c.name for c in dg.columns}
    warnings_ = list(dg.warnings) + [str(w) for w in data.get("warnings", [])]
    dropped: list[str] = []

    def cols(key: str) -> list[str]:
        out = []
        for c in data.get(key, []) or []:
            if c in known and c not in out:
                out.append(c)
            elif c not in known:
                dropped.append(f"{key}: {c}")
        return out

    def col(key: str) -> str | None:
        c = data.get(key)
        if c in (None, ""):
            return None
        if c not in known:
            dropped.append(f"{key}: {c}")
            return None
        return str(c)

    numeric, boolean, categorical = cols("numeric"), cols("boolean"), cols("categorical")
    text, long_text = col("text"), col("long_text")
    # Tree models cast boolean columns to float, so non-numeric columns placed there are
    # moved to categorical
    moved = [c for c in boolean if (dg.column(c) or ColumnProfile(c, "", "", 0, 0, 0)).kind != "boolean"]
    if moved:
        boolean = [c for c in boolean if c not in moved]
        categorical = categorical + [c for c in moved if c not in categorical]
        warnings_.append(f"The LLM marked {moved} as boolean, but they are not numeric,"
                         " so they were moved to categorical")
    # Never place the same column in two slots. The first occurrence wins
    seen: set[str] = set()
    for lst in (numeric, boolean, categorical):
        kept = []
        for c in lst:
            if c not in seen:
                kept.append(c)
                seen.add(c)
        lst[:] = kept
    if text in seen:
        text = None
    if long_text in seen or (long_text is not None and long_text == text):
        long_text = None
    if dropped:
        warnings_.append("The LLM used column names absent from the diagnosis table;"
                         " dropped: " + ", ".join(dropped))

    task = data.get("task") if data.get("task") in TASKS else draft.task
    classify = task == "classification"
    class_names = {}
    if classify:
        # Map labels back to the target's actual values (0 vs "0"). Unknown labels are dropped
        native = dg.target.native_classes
        for item in data.get("class_names", []) or []:
            lab = str(item.get("label", ""))
            if lab in native:
                class_names[native[lab]] = str(item.get("meaning", ""))
    domain = Domain(role=_bare_role(str(data.get("role") or Domain.role)),
                    subject=str(data.get("subject") or Domain.subject),
                    target_name=str(data.get("target_name") or "") or None,
                    hints=[str(h) for h in (data.get("hints") or [])][:3],
                    class_names=class_names)
    spec = ColumnSpec(numeric=numeric, boolean=boolean, categorical=categorical,
                      text=text, long_text=long_text)
    typed_columns = []
    for tc in data.get("typed_columns", []) or []:
        if tc.get("source") in known:
            typed_columns.append({"name": tc.get("name"), "source": tc.get("source"),
                                  "type": tc.get("type", "category"),
                                  "values": list(tc.get("values", [])), "why": tc.get("why", "")})
    knowledge_columns = _validate_knowledge_columns(data.get("knowledge_columns", []) or [],
                                                    dg, draft.target, warnings_)
    rate = data.get("escalate_rate", draft.escalate_rate)
    try:
        rate = min(1.0, max(0.0, float(rate)))
    except (TypeError, ValueError):
        rate = draft.escalate_rate
    reasons = {str(r.get("topic", "")): str(r.get("reason", ""))
               for r in (data.get("reasons") or []) if r.get("topic")}
    if not reasons:
        warnings_.append("The LLM returned no reasons. Check the recommendation against the"
                         " diagnosis table before trusting it")
    transform = data.get("target_transform")
    return Recommendation(
        task=task, spec=spec, domain=domain,
        unit=str(data.get("unit") or unit or ""), target=draft.target,
        target_transform=None if transform in (None, "none") or classify else str(transform),
        clip_outliers=bool(data.get("clip_outliers", draft.clip_outliers)) and not classify,
        dedup_ignore=cols("dedup_ignore"), typed_columns=typed_columns,
        knowledge_columns=knowledge_columns,
        use_evidence=bool(data.get("use_evidence", draft.use_evidence)),
        escalate_rate=rate, warnings=warnings_, reasons=reasons, source="llm", diagnosis=dg)


def _validate_knowledge_columns(items: list[dict[str, Any]], dg: Diagnosis, target: str,
                                warnings_: list[str]) -> list[dict[str, Any]]:
    """Keep only knowledge column candidates that `KnowledgeEncoder` could actually run.

    Keys must be columns of the diagnosis table of a kind usable as a key (not the free-text
    body, an identifier or the target). category / binary need values (binary exactly two;
    a binary with another count is demoted to category). numeric carries no values.
    Dropped candidates are recorded in the warnings so the user can see what the LLM proposed.
    """
    usable = {c.name for c in dg.columns if c.kind in KNOWLEDGE_KEY_KINDS} - {target}
    out: list[dict[str, Any]] = []
    dropped: list[str] = []
    for kc in items:
        keys = [str(k) for k in (kc.get("keys") or [])]
        attribute = str(kc.get("attribute") or "").strip()
        label = kc.get("name") or attribute or "?"
        bad = [k for k in keys if k not in usable]
        if not keys or bad or not attribute:
            dropped.append(f"{label} (keys {keys})")
            continue
        type_ = kc.get("type") if kc.get("type") in ("category", "binary", "numeric") else "category"
        values = [str(v) for v in (kc.get("values") or [])]
        if type_ == "numeric":
            values = []
        elif not values:
            dropped.append(f"{label} (no values)")
            continue
        elif type_ == "binary" and len(values) != 2:
            type_ = "category"
        out.append({"name": str(kc.get("name") or ""), "keys": keys, "attribute": attribute,
                    "type": type_, "values": values, "unit": str(kc.get("unit") or ""),
                    "why": str(kc.get("why") or "")})
    if dropped:
        warnings_.append("Knowledge column candidates whose keys are not usable columns"
                         " (or that lack values) were dropped: " + ", ".join(dropped))
    return out


def _bare_role(role: str) -> str:
    """Reduce a role written as the sentence "You are ..." to the "..." part.

    The prompt fills `Domain.role` into the form "You are {role}", so when the LLM answers
    with a full sentence the result reads "You are You are ...". In a run on the 500-row
    samples, 3 of 4 datasets came back in that form, so it is stripped here.
    """
    r = role.strip()
    r = re.sub(r"^(you\s+are|you're)\s*", "", r, flags=re.IGNORECASE)
    r = re.sub(r"[.!]+$", "", r)
    return r.strip() or Domain.role


def diagnose(df: pd.DataFrame, target: str, *, task: str | None = None,
             unit: str = "", target_name: str | None = None,
             llm: bool | str = "auto", client: ClaudeClient | None = None,
             sample: int | None = 20_000, n_splits: int = 5,
             screen_text: bool = True) -> Recommendation:
    """`diagnose`. Build an initial mekiki configuration from data and a target.

    Parameters
    ----------
    df, target:
        The data and the name of the target column. Every other column is diagnosed
    task:
        "regression" / "classification". If omitted, inferred from the target values
        (strings / booleans -> classification, integers with at most 10 distinct values ->
        classification, otherwise regression)
    unit, target_name:
        Pass them if known. Used in the report text and in the input to the LLM
    llm:
        "auto" (default) lets the LLM read the column meanings only when an API key exists.
        True makes it required (error if missing); False runs stage 1 only
    client:
        A `ClaudeClient`. Defaults to the standard one (with cache)
    sample, n_splits:
        Passed to `screen()`. Subsampling and fold count so large data does not take long
    screen_text:
        False skips measuring the explanatory power of text (saves tens of seconds to minutes)

    Returns
    -------
    A `Recommendation`. Printing it gives the diagnosis and the recommendation as text.
    `.spec` and `.domain` can be passed straight to `EvidencePredictor`.
    """
    if llm not in (True, False, "auto"):
        raise MekikiError('llm must be True, False or "auto"')
    dg = run_diagnosis(df, target, task=task, unit=unit, target_name=target_name,
                       sample=sample, n_splits=n_splits, screen_text=screen_text)
    draft = _rule_recommendation(dg, target, unit, target_name)
    if llm is False:
        return draft

    client = client or ClaudeClient(max_tokens=DIAGNOSE_MAX_TOKENS)
    if not client.available():
        if llm is True:
            raise MekikiError(
                f"llm=True but the LLM cannot be called. {client.why_unavailable()} "
                "Or use llm=False for stage 1 only.")
        draft.warnings.append("No API key, so the LLM was not used."
                              " Column meanings were not considered")
        return draft

    user = build_user_prompt(dg, draft, target, unit, target_name)
    ans = client.ask(SYSTEM_PROMPT, user, _answer_schema())
    if not ans.ok:
        draft.warnings.append(f"The LLM call failed, so the rule-based recommendation"
                              f" is returned: {ans.error}")
        draft.cost = client.summary()
        return draft
    rec = _apply_llm_answer(ans.data, dg, draft, unit)
    rec.cost = client.summary()
    return rec
