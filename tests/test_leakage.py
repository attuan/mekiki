"""Tests for duplicate-record detection (mekiki/leakage.py).

Three properties to protect.

1. **Duplicates are counted correctly.** "Rows beyond the first of each group" are
   counted, so a group of 3 counts as 2 duplicate rows
2. **Duplicates straddling train and test are found.** This is the actual harm
3. **Warn without stopping.** Duplicates are sometimes intentional, so no exception

Run: .venv/bin/python -m pytest tests -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mekiki import (
    EvidencePredictor,
    EvidenceRegressor,
    MekikiWarning,
    check_duplicates,
    check_overlap,
)
from mekiki.llm import ClaudeClient, LLMAnswer


class FakeClient(ClaudeClient):
    def __init__(self, **kw):
        kw.setdefault("cache_dir", None)
        kw.setdefault("api_key", "test-key")
        super().__init__(**kw)

    def ask(self, system, user, schema) -> LLMAnswer:
        self.usage.calls += 1
        return LLMAnswer(data={"price": 200.0, "confidence": 0.8,
                               "reason": "test"}, cost=0.0)


@pytest.fixture
def data() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 40
    age = rng.integers(1, 10, n)
    km = rng.integers(5_000, 120_000, n)
    return pd.DataFrame({
        "age": age,
        "odometer_km": km,
        "grade_name": rng.choice(["G", "X", "Z"], n),
        "equipment_text": [f"navigation ETC equipment{i}" for i in range(n)],
        "price": 250 - age * 12 - km / 8000 + rng.normal(0, 3, n),
    })


# --- Duplicates within one table --------------------------------------

def test_ok_when_no_duplicates(data):
    r = check_duplicates(data)
    assert r.ok and r.n_duplicate_rows == 0
    assert "No duplicates" in str(r)


def test_counts_rows_beyond_the_first(data):
    # Add row 0 twice -> 3 identical rows = 2 duplicate rows
    df = pd.concat([data, data.iloc[[0]], data.iloc[[0]]], ignore_index=True)
    r = check_duplicates(df)
    assert r.n_duplicate_rows == 2
    assert r.n_groups == 1
    assert r.largest_group == 3
    assert not r.ok
    assert "2 duplicate rows found" in str(r)


def test_columns_to_compare_can_be_narrowed(data):
    """With a single identifier column such as a VIN, comparing on it alone is most accurate."""
    df = data.copy()
    df["vin"] = [f"VIN{i:03d}" for i in range(len(df))]
    df.loc[1, "vin"] = "VIN000"          # different contents but the same car
    assert check_duplicates(df).ok             # distinct when compared on all columns
    assert check_duplicates(df, keys=["vin"]).n_duplicate_rows == 1


def test_columns_to_ignore_can_be_specified(data):
    """The same car differing only in region is counted as the same."""
    df = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    df["region"] = [f"region{i}" for i in range(len(df))]
    assert check_duplicates(df).ok
    assert check_duplicates(df, ignore=["region"]).n_duplicate_rows == 1


def test_stops_on_a_nonexistent_column(data):
    with pytest.raises(KeyError):
        check_duplicates(data, keys=["does_not_exist"])


# --- Duplicates straddling train and test ------------------------------

def test_ok_when_no_overlap(data):
    r = check_overlap(data.iloc[:30], data.iloc[30:])
    assert r.ok
    assert "No overlap" in str(r)


def test_finds_train_rows_present_in_test(data):
    train = data.iloc[:30]
    test = pd.concat([data.iloc[30:], train.iloc[[0, 1]]], ignore_index=True)
    r = check_overlap(train, test)
    assert r.n_leaked_rows == 2
    assert r.examples == [10, 11]
    assert not r.ok
    assert "answer already known" in str(r)


def test_compares_even_when_test_lacks_the_target(data):
    train = data.iloc[:30]
    test = pd.concat([data.iloc[30:], train.iloc[[0]]], ignore_index=True)
    r = check_overlap(train, test.drop(columns=["price"]))
    assert r.n_leaked_rows == 1
    assert "price" not in r.columns


def test_stops_when_no_common_columns(data):
    with pytest.raises(KeyError):
        check_overlap(data, pd.DataFrame({"other_column": [1, 2]}))


# --- Integrated into EvidencePredictor -----------------------------------------

def make(data, **kw) -> EvidencePredictor:
    return EvidenceRegressor(target="price", unit="10k JPY",
                             numeric=["age", "odometer_km"],
                             categorical=["grade_name"], text="equipment_text",
                             n_examples=3, client=FakeClient(),
                             **{"escalate_rate": 1.0, **kw})


def test_warns_on_duplicates_in_training_data(data):
    dup = pd.concat([data, data.iloc[[0, 1]]], ignore_index=True)
    with pytest.warns(MekikiWarning, match="2 duplicate rows"):
        make(dup).fit(dup)


def test_warns_on_overlap_between_train_and_test(data):
    train = data.iloc[:30].reset_index(drop=True)
    test = pd.concat([data.iloc[30:], train.iloc[[0]]], ignore_index=True)
    m = make(data).fit(train)
    with pytest.warns(MekikiWarning, match="same contents as train"):
        m.predict(test)


def test_warns_but_does_not_stop(data):
    """Duplicates are sometimes intentional, so no exception."""
    train = data.iloc[:30].reset_index(drop=True)
    test = pd.concat([data.iloc[30:], train.iloc[[0]]], ignore_index=True)
    m = make(data).fit(train)
    with pytest.warns(MekikiWarning):
        pred = m.predict(test)
    assert len(pred) == len(test)


def test_check_leakage_False_disables_it(data, recwarn):
    dup = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    make(dup, check_leakage=False).fit(dup)
    assert not [w for w in recwarn if issubclass(w.category, MekikiWarning)]


def test_no_warning_on_clean_data(data, recwarn):
    train, test = data.iloc[:30].reset_index(drop=True), data.iloc[30:]
    make(data).fit(train).predict(test)
    assert not [w for w in recwarn if issubclass(w.category, MekikiWarning)]


# --- The routing side checks every row exactly once --------------------

def test_routing_also_detects_overlap(data):
    train = data.iloc[:30].reset_index(drop=True)
    test = pd.concat([data.iloc[30:], train.iloc[[0]]], ignore_index=True)
    m = make(data, escalate_rate=0.5).fit(train)
    with pytest.warns(MekikiWarning, match="same contents as train") as rec:
        m.predict(test)
    # The overlap is checked once over all rows. The rows sent to the LLM must not
    # be checked again and produce a second warning
    assert len([w for w in rec if issubclass(w.category, MekikiWarning)]) == 1
