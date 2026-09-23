"""Tests for pre-screening (mekiki.screen).

Folds are reduced to 2. The goal is to verify the decision logic (does the verdict
change with the contribution), not absolute accuracy. With the default 5 folds the
whole test suite takes minutes and stops being run.

**No LLM is called, so no API key is needed** and the tests are fast.
Synthetic data is built for "text helps" / "text does not help" and the verdict is checked.

Run: .venv/bin/python -m pytest tests -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mekiki import MekikiError, screen


def make(n: int = 300, text_matters: bool = True, seed: int = 0) -> pd.DataFrame:
    """Build data where the price is / is not determined by the text."""
    rng = np.random.default_rng(seed)
    age = rng.integers(1, 12, n)
    grade = rng.choice(["standard", "sport", "luxury"], n)
    bonus = {"standard": 0, "sport": 60, "luxury": 160}
    base = 300 - age * 15 + rng.normal(0, 5, n)
    price = base + (np.array([bonus[g] for g in grade]) if text_matters else 0)
    return pd.DataFrame({
        "age": age,
        # The grade is only in the text, not in any structured column
        "description": [f"model {g} edition" for g in grade],
        "region": rng.choice(["A", "B", "C"], n),
        "price": price,
    })


def test_data_where_text_helps_is_worth_trying():
    r = screen(make(text_matters=True), target="price", text="description",
               numeric=["age"], categorical=["region"], sample=None, n_splits=2)
    assert r.verdict == "worth_trying"
    assert r.text_contribution > 0.10
    assert r.mae_with_text < r.mae_without_text


def test_data_where_text_does_not_help_is_unlikely_to_help():
    r = screen(make(text_matters=False), target="price", text="description",
               numeric=["age"], categorical=["region"], sample=None, n_splits=2)
    assert r.verdict == "unlikely_to_help"
    assert r.text_contribution < 0.05


def test_report_is_readable():
    r = screen(make(), target="price", text="description", numeric=["age"],
               unit="10k JPY", sample=None, n_splits=2)
    s = str(r)
    assert "text contribution" in s and "Verdict:" in s and "10k JPY" in s
    # Always state that the line is provisional (only two data points behind it)
    assert "provisional" in s


def test_nonexistent_column_raises():
    with pytest.raises(MekikiError):
        screen(make(), target="price", text="no_such_column", numeric=["age"], n_splits=2)


def test_too_few_rows_raises():
    with pytest.raises(MekikiError):
        screen(make(n=6), target="price", text="description", numeric=["age"],
               sample=None)


def test_long_text_is_truncated_for_the_verdict():
    """Character n-gram TF-IDF gets very slow on long text, so it is cut by default."""
    df = make()
    df["description"] = df["description"] + " " + "x" * 2000
    r = screen(df, target="price", text="description", numeric=["age"],
               sample=None, max_text_chars=100)
    assert r.max_text_chars == 100
    assert r.mean_text_chars > 500
    assert "first 100 characters" in str(r)


def make_churn(n: int = 300, text_matters: bool = True, seed: int = 0) -> pd.DataFrame:
    """Data where churn is / is not determined by the text (classification)."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 72, n)
    plan = rng.choice(["basic", "plus", "premium"], n)
    risk = {"basic": 2.5, "plus": 0.0, "premium": -2.5}
    logit = -0.03 * tenure + 0.8 + (np.array([risk[p] for p in plan]) if text_matters else 0)
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))
    return pd.DataFrame({
        "tenure": tenure,
        "note": [f"customer on {p} plan" for p in plan],   # the plan is only in the text
        "region": rng.choice(["A", "B", "C"], n),
        "Churn": np.where(churn, "Yes", "No"),
    })


def test_classification_data_where_text_helps_is_worth_trying():
    r = screen(make_churn(text_matters=True), target="Churn", text="note",
               task="classification", target_name="whether the customer churns",
               numeric=["tenure"], categorical=["region"], sample=None, n_splits=2)
    assert r.metric == "log loss" and r.verdict == "worth_trying"
    assert r.score_with_text < r.score_without_text
    s = str(r)
    assert "log loss" in s and "whether the customer churns" in s and "price" not in s


def test_classification_data_where_text_does_not_help_is_unlikely_to_help():
    r = screen(make_churn(text_matters=False), target="Churn", text="note",
               task="classification", numeric=["tenure"], categorical=["region"],
               sample=None, n_splits=2)
    assert r.verdict == "unlikely_to_help"


def test_report_text_uses_the_target_name():
    r = screen(make(), target="price", text="description", numeric=["age"],
               target_name="vehicle price", unit="10k JPY", sample=None, n_splits=2)
    s = str(r)
    assert '"vehicle price"' in s and "10k JPY" in s
    # The old property names still work
    assert r.mae_without_text == r.score_without_text


def test_invalid_task_stops():
    with pytest.raises(MekikiError):
        screen(make(), target="price", text="description", task="ranking", numeric=["age"])
