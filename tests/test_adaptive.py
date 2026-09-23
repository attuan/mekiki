"""Tests for confidence routing (`escalate_rate` / `threshold` of `EvidencePredictor`).

Like `test_predictor.py`, **everything passes without an API key**. The only part that
calls the LLM is `ClaudeClient.ask`, so a fake client that overrides it is used.

Five properties are protected here.

1. **The number of rows called matches the setting.** Otherwise the cost estimate
   falls apart
2. **Rows not called return the statistical model's prediction unchanged.** If they
   were silently swapped for something else, the routing accuracy curve could not be
   measured
3. **Approved rows are not called next time** (active learning)
4. **Provenance is kept with the route** (the inspection API)
5. **Sending every row without a limit warns when the cost is noticeable.**
   Omitting `escalate_rate` sends every row, so this prevents unnoticed charges

Run: .venv/bin/python -m pytest tests -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mekiki import EvidencePredictor, EvidenceRegressor, MekikiError, MekikiWarning
from mekiki.llm import ClaudeClient, LLMAnswer


class FakeClient(ClaudeClient):
    """Returns a canned answer without any HTTP. Its main purpose is counting calls."""

    def __init__(self, price: float = 200.0, **kw):
        kw.setdefault("cache_dir", None)
        kw.setdefault("api_key", "test-key")
        super().__init__(**kw)
        self.price = price
        self.prompts: list[str] = []

    def ask(self, system, user, schema) -> LLMAnswer:
        self.prompts.append(user)
        self.usage.calls += 1
        self.usage.cost += 0.001
        key = next(k for k in schema["properties"] if k not in ("confidence", "reason"))
        return LLMAnswer(data={key: self.price, "confidence": 0.8,
                               "reason": "test"},
                         input_tokens=100, output_tokens=20, cost=0.001)


@pytest.fixture
def data() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 80
    age = rng.integers(1, 10, n)
    km = rng.integers(5_000, 120_000, n)
    grade = rng.choice(["G", "X", "Z"], n)
    return pd.DataFrame({
        "age": age,
        "mileage_km": km,
        "has_repair_history": rng.random(n) < 0.2,
        "grade": grade,
        "equipment_text": [f"navi ETC {g} grade equipment{i % 5}" for i, g in enumerate(grade)],
        "price": 250 - age * 12 - km / 8000 + rng.normal(0, 3, n),
    })


def make(data, **kw) -> EvidencePredictor:
    return EvidenceRegressor(
        target="price", unit="10k JPY",
        numeric=["age", "mileage_km"], boolean=["has_repair_history"],
        categorical=["grade"], text="equipment_text",
        n_examples=3, **kw)


# --- number of rows called -------------------------------------------

@pytest.mark.parametrize("rate,expected", [(0.0, 0), (0.25, 5), (0.5, 10), (1.0, 20)])
def test_calls_the_LLM_for_exactly_the_given_fraction(data, rate, expected):
    client = FakeClient()
    m = make(data, client=client, escalate_rate=rate).fit(data.iloc[:60])
    m.predict(data.iloc[60:])
    assert len(client.prompts) == expected
    assert int(m.selected_.sum()) == expected


def test_rows_not_called_return_the_statistical_model_prediction_unchanged(data):
    train, test = data.iloc[:60], data.iloc[60:]
    m = make(data, client=FakeClient(), escalate_rate=0.25).fit(train)
    pred = m.predict(test)

    base = m.models_[0]
    base_pred = np.asarray(base.predict(test.reset_index(drop=True)), dtype=float)
    fast = ~m.selected_
    assert np.allclose(pred[fast], base_pred[fast])
    assert np.allclose(pred[m.selected_], 200.0)


def test_rate_zero_never_calls(data):
    client = FakeClient()
    m = make(data, client=client, escalate_rate=0.0).fit(data.iloc[:60])
    pred = m.predict(data.iloc[60:])
    assert client.prompts == []
    assert (m.provenance()["source"] == "model").all()
    assert len(pred) == 20


# --- signals ----------------------------------------------------------

def test_rows_with_the_largest_disagreement_are_selected(data):
    m = make(data, client=FakeClient(), escalate_rate=0.25).fit(data.iloc[:60])
    m.predict(data.iloc[60:])
    s = m.signal_
    assert s.min() >= 0                      # disagreement is an absolute value, so non-negative
    assert s[m.selected_].min() >= s[~m.selected_].max()


def test_unseen_signal_selects_rows_with_words_absent_from_training(data):
    train = data.iloc[:60].copy()
    test = data.iloc[60:].copy().reset_index(drop=True)
    test.loc[0, "equipment_text"] = "sunroof leather seats unknown equipment"
    test.loc[1, "grade"] = "GR SPORT"

    m = make(data, client=FakeClient(), escalate_rate=0.1,
             signal="unseen").fit(train)
    m.predict(test)
    assert m.selected_[0] and m.selected_[1]


def test_a_custom_signal_can_be_passed(data):
    def longest_mileage_first(X, evidence):
        return X["mileage_km"].to_numpy(dtype=float)

    m = make(data, client=FakeClient(), escalate_rate=0.25,
             signal=longest_mileage_first).fit(data.iloc[:60])
    test = data.iloc[60:].reset_index(drop=True)
    m.predict(test)
    km = test["mileage_km"].to_numpy()
    assert km[m.selected_].min() >= km[~m.selected_].max()


def test_unknown_signal_name_stops(data):
    with pytest.raises(MekikiError, match="not implemented"):
        make(data, signal="nonsense")


def test_signal_function_returning_the_wrong_length_stops(data):
    m = make(data, client=FakeClient(), escalate_rate=0.5,
             signal=lambda X, ev: np.zeros(3)).fit(data.iloc[:60])
    with pytest.raises(MekikiError, match="length"):
        m.predict(data.iloc[60:])


def test_disagreement_is_unavailable_with_a_single_tree_model(data):
    from mekiki.predictor import TreeModel

    m = EvidenceRegressor(
        target="price", unit="10k JPY", numeric=["age", "mileage_km"],
        text="equipment_text", n_examples=3, client=FakeClient(), escalate_rate=0.5)
    m._models_arg = [TreeModel("LightGBM", m.spec, kind="lgbm")]
    m.fit(data.iloc[:60])
    with pytest.raises(MekikiError, match="at least two models"):
        m.predict(data.iloc[60:])


def test_sending_every_row_works_with_a_single_tree_model(data):
    """The signal is not used for routing, so being unable to compute it does not stop
    the run (the provenance gets NaN)."""
    from mekiki.predictor import TreeModel

    m = EvidenceRegressor(
        target="price", unit="10k JPY", numeric=["age", "mileage_km"],
        text="equipment_text", n_examples=3, client=FakeClient(), escalate_rate=1.0)
    m._models_arg = [TreeModel("LightGBM", m.spec, kind="lgbm")]
    m.fit(data.iloc[:60])
    pred = m.predict(data.iloc[60:])
    assert len(pred) == 20 and np.isnan(m.signal_).all()
    with pytest.raises(MekikiError, match="needs a signal"):
        m.curve(data.iloc[60:], rates=(0.0, 1.0))


# --- threshold --------------------------------------------------------

def test_threshold_also_cuts(data):
    m = make(data, client=FakeClient(), escalate_rate=None,
             threshold=0.0).fit(data.iloc[:60])
    m.predict(data.iloc[60:])
    assert m.selected_.all()                 # disagreement is always >= 0


def test_no_rule_at_all_sends_every_row(data):
    client = FakeClient()
    m = make(data, client=client).fit(data.iloc[:60])
    with pytest.warns(MekikiWarning):
        m.predict(data.iloc[60:])
    assert len(client.prompts) == 20 and m.selected_.all()
    assert (m.provenance()["route"] == "llm").all()


# --- warning when every row is sent without a limit -------------------

def test_sending_every_row_without_a_limit_warns_about_cost_once(data, recwarn):
    m = make(data, client=FakeClient()).fit(data.iloc[:60])
    with pytest.warns(MekikiWarning, match="No escalation limit") as rec:
        m.predict(data.iloc[60:])            # 20 rows x $0.0086 > $0.10
    msg = str(rec[0].message)
    assert "20 rows" in msg and "escalate_rate=0.3" in msg and "escalate_rate=1.0" in msg
    recwarn.clear()
    m.predict(data.iloc[60:])                # the same instance does not warn a second time
    assert not [w for w in recwarn if issubclass(w.category, MekikiWarning)]


def test_escalate_rate_1_sends_every_row_without_a_warning(data, recwarn):
    m = make(data, client=FakeClient(), escalate_rate=1.0).fit(data.iloc[:60])
    m.predict(data.iloc[60:])
    assert not [w for w in recwarn if issubclass(w.category, MekikiWarning)]


def test_small_amounts_do_not_warn(data, recwarn):
    m = make(data, client=FakeClient()).fit(data.iloc[:60])
    m.predict(data.iloc[60:65])              # 5 rows x $0.0086 < $0.10
    assert not [w for w in recwarn if issubclass(w.category, MekikiWarning)]


def test_rate_out_of_range_stops(data):
    with pytest.raises(MekikiError, match="between 0.0 and 1.0"):
        make(data, escalate_rate=1.5)


# --- estimate before running ------------------------------------------

def test_plan_estimates_without_calling_the_LLM(data):
    client = FakeClient()
    m = make(data, client=client, escalate_rate=0.5).fit(data.iloc[:60])
    plan = m.plan(data.iloc[60:])
    assert client.prompts == []              # estimating is not charged
    assert plan["n_rows"] == 20
    assert plan["n_escalated"] == 10
    assert plan["n_fast_path"] == 10
    assert plan["estimated_cost_usd"] > 0
    assert plan["estimated_seconds"] > 0


def test_plan_row_count_matches_the_actual_number_of_calls(data):
    client = FakeClient()
    m = make(data, client=client, escalate_rate=0.3).fit(data.iloc[:60])
    plan = m.plan(data.iloc[60:])
    m.predict(data.iloc[60:])
    assert plan["n_escalated"] == len(client.prompts)


# --- active learning (approval widens the fast path) ------------------

def test_approved_rows_are_not_called_next_time(data):
    client = FakeClient()
    m = make(data, client=client, escalate_rate=0.5).fit(data.iloc[:60])
    test = data.iloc[60:]

    m.predict(test)
    assert len(client.prompts) == 10
    n = m.approve()
    assert n == 10

    m.predict(test)                          # the same X once more
    assert len(client.prompts) == 10         # no increase
    prov = m.provenance()
    assert (prov["source"] == "human").sum() == 10


def test_approval_matches_by_content_not_row_number(data):
    client = FakeClient()
    m = make(data, client=client, escalate_rate=0.5).fit(data.iloc[:60])
    test = data.iloc[60:].reset_index(drop=True)
    m.predict(test)
    m.approve()

    shuffled = test.sample(frac=1.0, random_state=1).reset_index(drop=True)
    m.predict(shuffled)
    assert len(client.prompts) == 10         # reordering does not trigger new calls


def test_review_queue_holds_only_rows_the_LLM_answered(data):
    m = make(data, client=FakeClient(), escalate_rate=0.25).fit(data.iloc[:60])
    m.predict(data.iloc[60:])
    q = m.review_queue()
    assert len(q) == 5
    assert set(q["row"]) == set(np.flatnonzero(m.selected_))
    assert "equipment_text" in q.columns     # content a person can judge from is included


def test_partial_approval(data):
    client = FakeClient()
    m = make(data, client=client, escalate_rate=0.5).fit(data.iloc[:60])
    test = data.iloc[60:]
    m.predict(test)
    rows = list(np.flatnonzero(m.selected_))[:3]
    assert m.approve(rows) == 3
    m.predict(test)
    assert len(client.prompts) == 10 + 7      # only the 3 approved rows are skipped


# --- provenance (the inspection API) ----------------------------------

def test_provenance_has_route_and_signal(data):
    m = make(data, client=FakeClient(), escalate_rate=0.25).fit(data.iloc[:60])
    m.predict(data.iloc[60:])
    prov = m.provenance()
    assert len(prov) == 20
    assert set(prov["route"]) == {"llm", "fast"}
    assert prov["source"].isin(["llm", "model", "human", "fallback"]).all()
    assert "signal" in prov.columns


def test_explain_reads_on_either_route(data):
    m = make(data, client=FakeClient(), escalate_rate=0.25).fit(data.iloc[:60])
    m.predict(data.iloc[60:])
    llm_row = int(np.flatnonzero(m.selected_)[0])
    fast_row = int(np.flatnonzero(~m.selected_)[0])

    s1 = m.explain(llm_row)
    assert "route LLM" in s1 and "Similar cases consulted" in s1
    s2 = m.explain(fast_row)
    assert "fast" in s2 and "Statistical model predictions" in s2


def test_examples_exist_only_for_rows_sent_to_the_LLM(data):
    m = make(data, client=FakeClient(), escalate_rate=0.25).fit(data.iloc[:60])
    m.predict(data.iloc[60:])
    llm_row = int(np.flatnonzero(m.selected_)[0])
    fast_row = int(np.flatnonzero(~m.selected_)[0])
    assert len(m.examples(llm_row)) == 3
    assert len(m.examples(fast_row)) == 0
    assert len(m.examples()) == 5 * 3


def test_route_is_a_per_row_table(data):
    m = make(data, client=FakeClient(), escalate_rate=0.25).fit(data.iloc[:60])
    m.predict(data.iloc[60:])
    r = m.route()
    assert len(r) == 20
    assert (r.loc[r["route"] == "llm", "cost_usd"] > 0).all()
    assert (r.loc[r["route"] == "fast", "cost_usd"] == 0).all()


def test_cost_reports_the_saving(data):
    m = make(data, client=FakeClient(), escalate_rate=0.25).fit(data.iloc[:60])
    m.predict(data.iloc[60:])
    c = m.cost()
    assert c["n_escalated"] == 5
    assert c["escalated_rate"] == 0.25
    assert c["saved_usd"] > 0


def test_inspection_api_before_predict_stops(data):
    m = make(data, client=FakeClient()).fit(data.iloc[:60])
    with pytest.raises(MekikiError, match="Call predict first"):
        m.provenance()


def test_an_X_with_a_different_row_count_is_detected(data):
    m = make(data, client=FakeClient(), escalate_rate=0.25).fit(data.iloc[:60])
    m.predict(data.iloc[60:])
    with pytest.raises(MekikiError, match="rows"):
        m.confidence(data.iloc[:10])


# --- curve (accuracy, cost, latency) ----------------------------------

def test_curve_returns_all_three(data):
    m = make(data, client=FakeClient(), escalate_rate=0.25).fit(data.iloc[:60])
    c = m.curve(data.iloc[60:], rates=(0.0, 0.5, 1.0))
    assert list(c["rate"]) == [0.0, 0.5, 1.0]
    assert list(c["n_escalated"]) == [0, 10, 20]
    assert set(["MAE", "cost_usd", "estimated_seconds"]) <= set(c.columns)
    assert c["cost_usd"].is_monotonic_increasing
    assert c["estimated_seconds"].is_monotonic_increasing


def test_inspection_api_after_curve_stops(data):
    """Drawing the curve runs every row internally, so the last provenance is dropped
    (to prevent a mix-up)."""
    m = make(data, client=FakeClient(), escalate_rate=0.25).fit(data.iloc[:60])
    m.predict(data.iloc[60:])
    m.curve(data.iloc[60:], rates=(0.0, 1.0))
    with pytest.raises(MekikiError, match="Call predict first"):
        m.explain(0)


def test_report_returns_a_summary(data):
    m = make(data, client=FakeClient(), escalate_rate=0.25).fit(data.iloc[:60])
    m.predict(data.iloc[60:])
    s = m.report()
    assert "sent to LLM:" in s and "cost:" in s


def test_explain_row_numbers_match_the_caller(data):
    """Rows sent to the LLM go as a subset, but the row numbers in the provenance must
    be those of the original X."""
    m = make(data, client=FakeClient(), escalate_rate=0.25).fit(data.iloc[:60])
    m.predict(data.iloc[60:])
    llm_row = int(np.flatnonzero(m.selected_)[1])   # the second row that was sent
    s = m.explain(llm_row)
    assert s.startswith(f"[row {llm_row}]")
    assert "[row " not in s[s.index("\n"):]         # no inner row number survives
