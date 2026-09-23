"""Building blocks of confidence routing.

**Not every row has to go to the LLM.** `EvidencePredictor` calls the LLM
once per row, so the row count translates directly into cost and time. Applying the
measured figures to a 60,000-row dataset (the Craigslist used-car listings) gives

    60,000 rows x $0.0086 = about $516 , 60,000 rows x 1.03 s = about 17 hours (8 workers)

which cannot be run as is. So **only "signals available before calling the LLM"
choose which rows to send**, and the rest are left to the statistical models. A single
rate moves continuously between "call every row" and "call none".

Routing is a feature of `EvidencePredictor` itself, enabled by passing `escalate_rate`.
This module holds what it calls: signal computation, row selection, and the cost and
time estimates. It has no class (so as not to add another entry point).

    from mekiki import EvidencePredictor

    model = EvidenceRegressor(target="points", unit="pts",
                              numeric=["price"], categorical=["country", "variety"],
                              long_text="description",
                              escalate_rate=0.3)       # send only the top 30% to the LLM
    model.plan(test)        # before calling: how many rows, how much, how many seconds
    pred = model.predict(test)
    model.route()           # which route each row took
    model.review_queue()    # rows the LLM answered = teacher-label candidates
    model.approve()         # approved rows are not called next time (the fast path widens)

## Which signal to select by (fixed by measurement)

Six candidates were measured; **the best was "disagreement between the
two trees"**. Sending the rows where LightGBM and XGBoost disagree most -- that is, the
rows the statistical models themselves are unsure about -- to the LLM is what works.
Measured on 600 rows of the Craigslist used-car data:

| rate sent | 10% | 20% | 30% | 50% | 75% | 100% |
|---|---|---|---|---|---|---|
| MAE (USD) | 2,907 | 2,795 | 2,674 | 2,581 | 2,450 | **2,265** |
| cost | $0.52 | $1.05 | $1.57 | $2.62 | $3.92 | $5.23 |
| estimated time (s) | 61 | 93 | 130 | 199 | 286 | 369 |

(0% = 3,043.)

**On this data, sending every row (2,265) is the most accurate.** Routing is not a
"device for raising accuracy" but **a device for choosing how much accuracy to give up
for how much saving**. Sending half costs $2.62 for MAE 2,581, which reads as taking
59% of the gap from 0% at half the price.

Note that **on a second used-car dataset (a single model, Sienta) no signal worked**.
Whether a signal works depends on the data, so always check with `curve()` on your own.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from mekiki.errors import MekikiError

#: Measured response time per call (median, seconds). **More workers do not make a
#: single call faster**, so the total is estimated as "rows / workers x this"
#: (measured on the Craigslist used-car data)
SECONDS_PER_CALL = 4.6

#: Measured setup time (fit + building the neighbour index), seconds. Paid once,
#: independent of the row count
SETUP_SECONDS = 23.85

#: Measured cost per row (USD). `EvidencePredictor` measurements ranged $0.0086-0.0121, so the
#: lower end is used. Once calls have been made, the actual figure replaces it
COST_PER_ROW_USD = 0.0086

#: When neither `escalate_rate` nor `threshold` is given and every row goes to the LLM,
#: a `MekikiWarning` is raised if the estimated cost exceeds this. It is a money cutoff
#: rather than a row count so that a few-row smoke test does not warn every time
#: (and it follows a change in unit cost)
COST_WARN_USD = 0.10

#: Available signals. All of them are **available before calling the LLM**.
#: "How much the LLM moved the value" is the best signal, but it is only known after
#: the call, so it cannot save cost. "jev" is 1 minus the confidence of the Jev tier
#: (`EvidencePredictor(jev=...)`): rows Jev is unsure about go to the LLM
SIGNALS = ("disagreement", "similarity", "unseen", "jev")


def row_key(row: pd.Series, columns: Sequence[str]) -> str:
    """Fingerprint built from the row's content. Used to recognise an approved row next time.

    Matching by content rather than row number keeps it working when the next predict
    has a different order or count of rows.
    """
    text = "\x1f".join(f"{c}={row[c]!r}" for c in columns)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


# --- signals (only what is available before calling the LLM) -----------------

def disagreement(tree_outputs: Sequence[np.ndarray], classify: bool) -> np.ndarray:
    """Spread between the predictions of the models that take part (the default tree
    models, plus any model that sets `in_disagreement = True`). Larger means the models
    themselves are unsure.

    Regression: max minus min of the predictions. Classification: total variation
    distance between the probability distributions (max over pairs).
    Fewer than two outputs raises `MekikiError`.
    """
    if len(tree_outputs) < 2:
        raise MekikiError(
            "signal='disagreement' needs at least two models that take part in it"
            f" (currently {len(tree_outputs)}). The default tree models do: keep them "
            "with models=['default', MyModel()], or set in_disagreement = True on your "
            "own models, or use signal='similarity'.")
    if classify:
        n = len(tree_outputs[0])
        out = np.zeros(n, dtype=float)
        for a in range(len(tree_outputs)):
            for b in range(a + 1, len(tree_outputs)):
                out = np.maximum(
                    out, 0.5 * np.abs(tree_outputs[a] - tree_outputs[b]).sum(axis=1))
        return out
    stack = np.vstack(tree_outputs)
    return stack.max(axis=0) - stack.min(axis=0)


def unseen_score(X: pd.DataFrame, train: pd.DataFrame, categorical: Sequence[str],
                 text: str | None) -> np.ndarray:
    """How many levels and words are absent from the training data. Larger means
    "a row never seen before"."""
    score = np.zeros(len(X), dtype=float)
    for c in categorical:
        known = set(train[c].dropna().astype(str))
        score += (~X[c].astype(str).isin(known)).to_numpy(dtype=float)
    if text:
        known_tokens: set[str] = set()
        for t in train[text].fillna("").astype(str):
            known_tokens.update(t.lower().split())
        for i, t in enumerate(X[text].fillna("").astype(str)):
            toks = t.lower().split()
            if toks:
                score[i] += sum(w not in known_tokens for w in toks) / len(toks)
    return score


# --- selection ---------------------------------------------------------------

def select(s: np.ndarray, escalate_rate: float | None,
           threshold: float | None) -> np.ndarray:
    """Turn the signal into a boolean mask of "rows to send to the LLM".

    With `escalate_rate`, the top fraction by signal strength; otherwise rows at or
    above `threshold`. With neither, every row.
    """
    n = len(s)
    if escalate_rate is not None:
        k = int(round(n * escalate_rate))
        sel = np.zeros(n, dtype=bool)
        if k > 0:
            # Ties are broken by input order (stable), for reproducible measurements
            order = np.argsort(-s, kind="stable")
            sel[order[:k]] = True
        return sel
    if threshold is not None:
        return s >= float(threshold)
    return np.ones(n, dtype=bool)


def to_confidence(s: np.ndarray) -> np.ndarray:
    """Rescale the signal to a 0-1 confidence (weaker signal = higher confidence)."""
    if len(s) == 0:
        return np.ones(0, dtype=float)
    lo, hi = float(np.nanmin(s)), float(np.nanmax(s))
    if hi - lo < 1e-12:
        return np.ones(len(s), dtype=float)
    return 1.0 - (s - lo) / (hi - lo)


# --- estimate before running ("visible before committing a run") --------------------

def unit_cost(client, default: float = COST_PER_ROW_USD) -> float:
    """Cost per row: the actual figure if there is one, else `default` (the measured
    figure for the frontier LLM; a Jev client passes its own list-price estimate)."""
    u = client.usage
    paid = u.calls - u.cache_hits
    if paid > 0 and u.cost > 0:
        return u.cost / paid
    return default


def latency_seconds(client, n_calls: int, with_setup: bool = True,
                    seconds_per_call: float = SECONDS_PER_CALL) -> float:
    """Estimated time (seconds) to run `n_calls` rows. `seconds_per_call` is the
    frontier LLM's measured figure by default; the Jev tier passes its own."""
    workers = max(int(getattr(client, "max_workers", 1) or 1), 1)
    batches = math.ceil(n_calls / workers) if n_calls else 0
    return (SETUP_SECONDS if with_setup else 0.0) + batches * seconds_per_call
