"""Estimate, before calling the LLM, whether a dataset is worth `EvidencePredictor`.

## Why this is needed

With the same implementation and the same LLM, `EvidencePredictor` lost on one
used-car dataset (Sienta, a single model) and won significantly on another (Craigslist,
many models). The difference lay in the nature of the data.

| | Sienta | Craigslist |
|---|---|---|
| Effect of adding text to the tree | -5.2% | -16.5% |
| Effect of `EvidencePredictor` (vs. the fair baseline) | +1.1% (loss) | -8.8% (win) |

**Where the text does not explain the target in the first place, having the LLM read it
does not help.** So "just send every row to the LLM first" is expensive in both time
and money.

`screen()` measures whether text helps on that data **without a single LLM call**.
The verdict is only a comparison of two tree models, so it finishes in tens of seconds
and costs nothing.

    from mekiki import screen

    report = screen(df, target="points", text="description",
                    numeric=["price"],
                    categorical=["country", "variety"])
    print(report)

    # For classification, pass task. The metric becomes log loss
    report = screen(df, target="Churn", text="note", task="classification",
                    numeric=["tenure"], categorical=["Contract"])

## What is measured

Two identical LightGBM models are cross-validated, differing only in **whether the text
column is included**. The gap (`text_contribution`) is the amount of information the
unstructured text carries on that data. The metric is MAE for regression and log loss
for classification (lower is better for both).

**This metric is not the effect of `EvidencePredictor` itself.** `EvidencePredictor` adds value only by
"the LLM reading the text better than character TF-IDF", so what is checked here is the
premise that **the gain can only appear where TF-IDF already helps**.

Correspondence with measurements (two used-car datasets only, both regression):

| text contribution | `EvidencePredictor` result |
|---|---|
| 5.2% (Sienta) | loss (not significant) |
| 16.5% (Craigslist) | win (p = 0.009) |

**With only two points the threshold is provisional.** The default 10% merely sits
between the two points and has no principled basis. The relationship for
classification is unmeasured and the same threshold is used tentatively.
Redraw it once more data is available.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mekiki.errors import MekikiError

SEED = 42

#: Provisional line: a text contribution at or above this makes `EvidencePredictor` worth trying.
#: **Merely the midpoint of two measured datasets, not a principled value.**
DEFAULT_THRESHOLD = 0.10

TASKS = ("regression", "classification")

#: Verdict values of `ScreeningReport.verdict`
VERDICT_WORTH_TRYING = "worth_trying"
VERDICT_UNLIKELY = "unlikely_to_help"
VERDICT_INCONCLUSIVE = "inconclusive"


@dataclass
class ScreeningReport:
    """The result of `screen()`. Printing it gives a readable summary."""

    n_rows: int
    unit: str
    score_without_text: float
    score_with_text: float
    text_contribution: float          # (without - with) / without
    threshold: float
    text_column: str
    n_unique_text: int
    #: Verdict. "worth_trying" / "unlikely_to_help" / "inconclusive"
    verdict: str
    #: Maximum number of text characters used for the verdict (0 = not truncated)
    max_text_chars: int = 0
    #: Mean number of characters of the original text
    mean_text_chars: float = 0.0
    #: Name of the metric. "MAE" for regression, "log loss" for classification
    metric: str = "MAE"
    #: How the target is referred to (appears in the text). Defaults to the column name
    target_name: str = "target"
    task: str = "regression"

    # Names from when this was regression-only. Kept so callers do not break
    @property
    def mae_without_text(self) -> float:
        return self.score_without_text

    @property
    def mae_with_text(self) -> float:
        return self.score_with_text

    def __str__(self) -> str:
        pct = self.text_contribution * 100
        unit = f" {self.unit}" if self.unit and self.task == "regression" else ""
        label = f"{self.metric:<8}"
        lines = [
            f"Measured how much the text column \"{self.text_column}\" explains "
            f"\"{self.target_name}\""
            f" ({self.n_rows:,} rows / {self.n_unique_text:,} distinct values"
            f" / mean {self.mean_text_chars:,.0f} characters)",
            "",
            f"  LightGBM without text   {label} {self.score_without_text:>10,.4g}{unit}",
            f"  LightGBM with text      {label} {self.score_with_text:>10,.4g}{unit}",
            f"  text contribution       {'':<8} {pct:>10.1f} %"
            f" (threshold {self.threshold * 100:.0f}%)",
            "",
            f"  Verdict: {self.verdict}",
        ]
        if self.verdict == VERDICT_WORTH_TRYING:
            lines.append(
                f"  -> The text explains {self.target_name}. There is room for the LLM\n"
                "     to read it better than character TF-IDF, so EvidencePredictor is worth trying.")
        elif self.verdict == VERDICT_UNLIKELY:
            lines.append(
                f"  -> The text barely explains {self.target_name}. Little gain can be\n"
                "     expected from having the LLM read it; the statistical models are\n"
                "     probably enough. (Consider building another column with SemanticEncoder,\n"
                "     or collecting different columns altogether, first.)")
        else:
            lines.append(
                "  -> The gap is small and the row count is low, so no verdict can be made.\n"
                "     Add rows, or try EvidencePredictor on a small scale and measure.")
        lines.append("")
        if self.max_text_chars and self.mean_text_chars > self.max_text_chars:
            lines.append(
                f"  Note: only the first {self.max_text_chars} characters were used"
                f" (mean {self.mean_text_chars:,.0f} characters).\n"
                "     If the rest carries information too, the real contribution may be larger.")
        lines.append("  Note: this verdict is a provisional line drawn from two measured"
                     " datasets (both regression).")
        lines.append("     The final call should come from a small real run of EvidencePredictor"
                     " (about 60 rows).")
        return "\n".join(lines)


def screen(df: pd.DataFrame, target: str, text: str, *,
           task: str = "regression", target_name: str | None = None,
           numeric: Sequence[str] = (), boolean: Sequence[str] = (),
           categorical: Sequence[str] = (), unit: str = "",
           n_splits: int = 5, sample: int | None = 20_000,
           max_text_chars: int = 500,
           threshold: float = DEFAULT_THRESHOLD) -> ScreeningReport:
    """Measure how much a text column explains the target. **Does not call the LLM.**

    Parameters
    ----------
    df, target, text:
        The data, the target column name, and the text column to evaluate.
    task:
        "regression" (default, metric MAE) or "classification" (metric log loss).
    target_name:
        How the target is referred to in the report text. Defaults to the column name.
    numeric / boolean / categorical:
        Structured columns forming the baseline. A weak baseline overstates the
        text contribution.
    sample:
        Upper bound on rows. Thinned to 20,000 rows by default so large data does
        not keep you waiting. `None` uses every row.
    max_text_chars:
        How many leading characters of the text to look at. **Character n-gram
        TF-IDF gets very slow on long text** (the Craigslist used-car descriptions average 2,320
        characters; 20,000 rows take minutes). The verdict only needs the order of
        magnitude of "does text help", so the default cuts at 500 characters.
        `0` means no cut.
    threshold:
        Lower bound of the contribution for a "worth_trying" verdict.

    Returns
    -------
    `ScreeningReport`. Printing it gives a readable summary.
    """
    from sklearn.model_selection import KFold, StratifiedKFold

    from mekiki.predictor import ColumnSpec, TreeModel

    if task not in TASKS:
        raise MekikiError(f"task must be one of {TASKS}: {task!r}")
    for col in [target, text, *numeric, *boolean, *categorical]:
        if col not in df.columns:
            raise MekikiError(f"Column not in the data: {col}")

    df = df[df[target].notna()].reset_index(drop=True)
    if sample is not None and sample < len(df):
        df = df.sample(n=sample, random_state=SEED).reset_index(drop=True)
    if len(df) < n_splits * 2:
        raise MekikiError(f"Too few rows ({len(df)} rows).")

    spec = ColumnSpec(numeric=list(numeric), boolean=list(boolean),
                      categorical=list(categorical))
    classify = task == "classification"
    if classify:
        classes = np.unique(df[target].to_numpy())
        if len(classes) < 2:
            raise MekikiError("Classification needs at least 2 distinct labels.")
        lookup = {c: i for i, c in enumerate(classes.tolist())}
        y = np.array([lookup[v] for v in df[target].tolist()], dtype=int)
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        splits = kf.split(df, y)
    else:
        y = df[target].to_numpy(dtype=float)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        splits = kf.split(df)

    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import log_loss

    def _clip(col: pd.Series) -> pd.Series:
        v = col.fillna("").astype(str)
        return v.str.slice(0, max_text_chars) if max_text_chars > 0 else v

    def score(model: TreeModel, test: pd.DataFrame, y_te: np.ndarray) -> float:
        if classify:
            p = model.predict_proba(test)
            return float(log_loss(y_te, p, labels=list(range(len(classes)))))
        return float(np.mean(np.abs(model.predict(test) - y_te)))

    def with_text(train: pd.DataFrame, test: pd.DataFrame):
        """Compress the text with character n-gram TF-IDF -> SVD and add it as columns.

        Fitting is always done on train only (leaking test information overstates
        the contribution).
        """
        # min_df is relaxed with the row count. Leaving it at 5 on few rows empties
        # the vocabulary and sklearn raises a confusing exception.
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                              min_df=min(5, max(1, len(train) // 50)),
                              max_features=2000)
        A = vec.fit_transform(_clip(train[text]))
        B = vec.transform(_clip(test[text]))
        n_comp = min(64, A.shape[1] - 1)
        if n_comp < 2:
            return None, None
        svd = TruncatedSVD(n_components=n_comp, random_state=SEED)
        return svd.fit_transform(A), svd.transform(B)

    scores_without, scores_with = [], []
    for tr_idx, te_idx in splits:
        train = df.iloc[tr_idx].reset_index(drop=True)
        test = df.iloc[te_idx].reset_index(drop=True)
        y_te = y[te_idx]

        m0 = TreeModel("without_text", spec, task=task).fit(train, y[tr_idx])
        scores_without.append(score(m0, test, y_te))

        Etr, Ete = with_text(train, test)
        if Etr is None:                     # text too short to build features
            scores_with.append(scores_without[-1])
            continue
        cols = [f"_txt{i}" for i in range(Etr.shape[1])]
        spec2 = ColumnSpec(numeric=[*spec.numeric, *cols], boolean=spec.boolean,
                           categorical=spec.categorical)
        tr2 = pd.concat([train, pd.DataFrame(Etr, columns=cols)], axis=1)
        te2 = pd.concat([test, pd.DataFrame(Ete, columns=cols)], axis=1)
        m1 = TreeModel("with_text", spec2, task=task).fit(tr2, y[tr_idx])
        scores_with.append(score(m1, te2, y_te))

    s0, s1 = float(np.mean(scores_without)), float(np.mean(scores_with))
    contribution = (s0 - s1) / s0 if s0 > 0 else 0.0

    # If the sign is not consistent across folds, the verdict is "inconclusive".
    # Deciding on the mean alone lets one outlier fold swing the verdict.
    wins = sum(a > b for a, b in zip(scores_without, scores_with, strict=True))
    if contribution >= threshold and wins >= n_splits - 1:
        verdict = VERDICT_WORTH_TRYING
    elif contribution < threshold / 2:
        verdict = VERDICT_UNLIKELY
    else:
        verdict = VERDICT_INCONCLUSIVE

    return ScreeningReport(
        n_rows=len(df), unit=unit, score_without_text=s0, score_with_text=s1,
        text_contribution=contribution, threshold=threshold, text_column=text,
        n_unique_text=int(df[text].nunique()), verdict=verdict,
        max_text_chars=max_text_chars,
        mean_text_chars=float(df[text].fillna("").astype(str).str.len().mean()),
        metric="log loss" if classify else "MAE",
        target_name=target_name or target, task=task)
