"""Tests for `EvidencePredictor`.

**Everything passes without an API key.** The only part that calls the LLM is
`ClaudeClient.ask`, so a fake client that overrides it is used. This way breakage in
the pipeline is caught in CI, and even when the key has expired.

Run: .venv/bin/python -m pytest tests -q
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from mekiki import ColumnSpec, EvidencePredictor, EvidenceRegressor, MekikiError
from mekiki.llm import ClaudeClient, LLMAnswer


class FakeClient(ClaudeClient):
    """Records the prompts it receives and returns a canned answer, without any HTTP."""

    def __init__(self, price: float = 200.0, fail_rows: set[int] | None = None,
                 **kw):
        kw.setdefault("cache_dir", None)   # do not touch the disk
        kw.setdefault("api_key", "test-key")
        super().__init__(**kw)
        self.price = price
        self.fail_rows = fail_rows or set()
        self.prompts: list[str] = []

    def ask(self, system, user, schema) -> LLMAnswer:
        i = len(self.prompts)
        self.prompts.append(user)
        self.systems = getattr(self, "systems", []) + [system]
        if i in self.fail_rows:
            self.usage.calls += 1
            self.usage.errors += 1
            return LLMAnswer(data={}, error="deliberate failure")
        self.usage.calls += 1
        self.usage.cost += 0.001
        # The answer key follows the schema (default "value", USED_CAR uses "price")
        key = next(k for k in schema["properties"] if k not in ("confidence", "reason"))
        return LLMAnswer(data={key: self.price, "confidence": 0.8,
                               "reason": "test"},
                         input_tokens=100, output_tokens=20, cost=0.001)


@pytest.fixture
def data() -> pd.DataFrame:
    """Small synthetic data where the price is almost determined by age and mileage."""
    rng = np.random.default_rng(0)
    n = 60
    age = rng.integers(1, 10, n)
    km = rng.integers(5_000, 120_000, n)
    grade = rng.choice(["G", "X", "Z"], n)
    return pd.DataFrame({
        "age": age,
        "mileage_km": km,
        "has_repair_history": rng.random(n) < 0.2,
        "grade": grade,
        "equipment_text": [f"navi ETC {g} grade equipment{i%5}" for i, g in enumerate(grade)],
        "price": 250 - age * 12 - km / 8000 + rng.normal(0, 3, n),
    })


def make(data, **kw) -> EvidencePredictor:
    return EvidenceRegressor(
        target="price", unit="10k JPY",
        numeric=["age", "mileage_km"], boolean=["has_repair_history"],
        categorical=["grade"], text="equipment_text",
        n_examples=3, **kw)


# --- basic shape ------------------------------------------------------

def test_fit_predict_shape(data):
    client = FakeClient()
    m = make(data, client=client).fit(data.iloc[:40])
    pred = m.predict(data.iloc[40:])
    assert pred.shape == (20,)
    assert np.allclose(pred, 200.0)
    assert len(client.prompts) == 20


def test_omitting_y_uses_the_target_column(data):
    m = make(data, client=FakeClient()).fit(data.iloc[:40])
    assert len(m.y_) == 40
    assert np.allclose(m.y_, data["price"].to_numpy()[:40])


def test_predict_before_fit_raises(data):
    with pytest.raises(MekikiError):
        make(data, client=FakeClient()).predict(data)


def test_missing_column_raises(data):
    m = EvidenceRegressor(target="price", numeric=["no_such_column"], client=FakeClient())
    with pytest.raises(MekikiError):
        m.fit(data)


def test_text_only_spec_raises_a_readable_error(data):
    """Text is not fed to the tree models, so a spec with nothing else must stop with
    a MekikiError, not with LightGBM's "maximum feature index in dataset is -1"."""
    m = EvidenceRegressor(target="price", text="equipment_text", client=FakeClient())
    with pytest.raises(MekikiError, match="numeric, boolean or categorical"):
        m.fit(data)


# --- is the evidence really in the prompt -----------------------------

def test_prompt_contains_statistical_model_predictions(data):
    client = FakeClient()
    m = make(data, client=client).fit(data.iloc[:40])
    m.predict(data.iloc[40:42])
    p = client.prompts[0]
    assert "LightGBM" in p and "XGBoost" in p
    assert "Evidence 1" in p and "Evidence 2" in p


def test_prompt_contains_n_examples_similar_cases(data):
    client = FakeClient()
    m = make(data, client=client).fit(data.iloc[:40])
    m.predict(data.iloc[40:41])
    p = client.prompts[0]
    assert "Case 1" in p and "Case 3" in p and "Case 4" not in p


def test_similar_cases_come_only_from_training_data(data):
    """Mixing test answers into the evidence would invalidate the evaluation."""
    m = make(data, client=FakeClient()).fit(data.iloc[:40])
    m.predict(data.iloc[40:50])
    ex = m.examples()
    assert ex["train_row"].max() < 40


# --- provenance (the inspection API) ---------------------------------

def test_explain_traces_the_evidence(data):
    m = make(data, client=FakeClient()).fit(data.iloc[:40])
    m.predict(data.iloc[40:42])
    s = m.explain(0)
    assert "LightGBM" in s and "Similar cases" in s and "confidence" in s


def test_confidence_has_one_entry_per_row(data):
    m = make(data, client=FakeClient()).fit(data.iloc[:40])
    m.predict(data.iloc[40:50])
    c = m.confidence()
    assert len(c) == 10 and (c == 0.8).all()


def test_cost_contains_total_and_unit_cost(data):
    m = make(data, client=FakeClient()).fit(data.iloc[:40])
    m.predict(data.iloc[40:50])
    c = m.cost()
    assert c["n_predicted"] == 10
    assert c["cost_usd"] == pytest.approx(0.01, abs=1e-6)
    assert c["cost_per_row_usd"] == pytest.approx(0.001, abs=1e-6)


def test_provenance_has_one_row_per_row(data):
    m = make(data, client=FakeClient()).fit(data.iloc[:40])
    m.predict(data.iloc[40:50])
    prov = m.provenance()
    assert len(prov) == 10
    assert set(prov["source"]) == {"llm"}
    assert "evidence_LightGBM" in prov.columns


# --- behaviour on failure ---------------------------------------------

def test_unanswered_rows_are_filled_by_the_statistical_model(data):
    client = FakeClient(fail_rows={0, 3})
    m = make(data, client=client).fit(data.iloc[:40])
    pred = m.predict(data.iloc[40:50])
    prov = m.provenance()
    assert list(prov["source"]).count("fallback") == 2
    # The substitute is the prediction of the first model (LightGBM)
    for i in (0, 3):
        assert pred[i] == pytest.approx(prov.loc[i, "evidence_LightGBM"])
    assert not np.isnan(pred).any()


def test_fallback_error_raises(data):
    client = FakeClient(fail_rows={0})
    m = make(data, client=client, fallback="error").fit(data.iloc[:40])
    with pytest.raises(MekikiError):
        m.predict(data.iloc[40:50])


# --- cache ------------------------------------------------------------

def test_same_prompt_is_not_charged_twice(tmp_path, data):
    """Cross-validation and re-measurement hit the same rows again and again;
    without this we go bankrupt."""
    calls = {"n": 0}

    class Counting(FakeClient):
        def ask(self, system, user, schema):
            # Bypass the parent's ask and exercise only the real cache mechanism
            key = self._key(system, user, schema)
            cached = self._read_cache(key)
            if cached is not None:
                self.usage.calls += 1
                self.usage.cache_hits += 1
                return LLMAnswer(data=cached["data"], from_cache=True)
            calls["n"] += 1
            data_ = {"value": 200.0, "confidence": 0.8, "reason": "test"}
            self._write_cache(key, {"data": data_, "input_tokens": 100,
                                    "output_tokens": 20, "cache_read_tokens": 0})
            self.usage.calls += 1
            return LLMAnswer(data=data_, cost=0.001)

    client = Counting(cache_dir=tmp_path)
    m = make(data, client=client).fit(data.iloc[:40])
    m.predict(data.iloc[40:50])
    assert calls["n"] == 10

    client2 = Counting(cache_dir=tmp_path)
    m2 = make(data, client=client2).fit(data.iloc[:40])
    m2.predict(data.iloc[40:50])
    assert calls["n"] == 10                      # not a single extra call
    assert client2.usage.cache_hits == 10
    assert client2.usage.cost == 0.0


# --- neighbour search recipe --------------------------------------------

def test_mixing_numeric_distance_brings_neighbour_age_closer(data):
    """By semantic similarity alone, cars with a different year and mileage come out
    as "similar"."""
    train, test = data.iloc[:40], data.iloc[40:]
    spec = ColumnSpec(numeric=["age", "mileage_km"], text="equipment_text")

    from mekiki.predictor import NeighbourIndex
    gaps = {}
    for w in (0.0, 1.0):
        idx = NeighbourIndex(spec, k=3, w=w).fit(train, train["price"].to_numpy())
        nb, _ = idx.query(test)
        age_tr = train["age"].to_numpy()
        age_te = test["age"].to_numpy()
        gaps[w] = float(np.mean(np.abs(age_tr[nb] - age_te[:, None])))
    assert gaps[1.0] < gaps[0.0]


def test_neighbours_are_sorted_by_descending_similarity(data):
    m = make(data, client=FakeClient()).fit(data.iloc[:40])
    m.predict(data.iloc[40:45])
    ex = m.examples(i=0)
    assert list(ex["rank"]) == [1, 2, 3]
    assert list(ex["similarity"]) == sorted(ex["similarity"], reverse=True)


# --- schema -----------------------------------------------------------

def test_answer_schema_is_valid_json(data):
    from mekiki.predictor import ANSWER_SCHEMA, USED_CAR
    json.dumps(ANSWER_SCHEMA)
    assert ANSWER_SCHEMA["additionalProperties"] is False
    assert set(ANSWER_SCHEMA["required"]) == {"value", "confidence", "reason"}
    # The used-car preset keeps its own answer key
    json.dumps(USED_CAR.answer_schema)
    assert set(USED_CAR.answer_schema["required"]) == {"price", "confidence", "reason"}


# --- LLM fallback of SemanticEncoder (step 05) --------------------------------

class FakeClassifyClient(FakeClient):
    """For the classification fallback. The returned value can be swapped."""

    def __init__(self, value="G", **kw):
        super().__init__(**kw)
        self.value = value

    def ask(self, system, user, schema) -> LLMAnswer:
        self.prompts.append(user)
        self.usage.calls += 1
        self.usage.cost += 0.002
        return LLMAnswer(data={"value": self.value, "confidence": 0.9,
                               "reason": "test"}, cost=0.002)


def test_fallback_does_not_accept_values_outside_the_candidates():
    """An arbitrary value mixed into the features breaks the downstream categorical column."""
    from mekiki.fallback import LLMFallback
    fb = LLMFallback(client=FakeClassifyClient(value="no such grade"))
    out = fb.answer(["with navi"], ["G", "Z"], [{}])
    assert out[0].value is None
    assert out[0].confidence == 0.0


def test_fallback_accepts_a_candidate_value_and_queues_it():
    from mekiki.fallback import LLMFallback
    fb = LLMFallback(client=FakeClassifyClient(value="Z"))
    out = fb.answer(["with navi"], ["G", "Z"], [{}])
    assert out[0].value == "Z" and out[0].origin == "llm"
    # Every answer is queued as a teacher-label candidate (one side of active learning)
    assert len(fb.queued) == 1 and fb.queued[0]["llm_answer"] == "Z"


def test_fallback_prompt_contains_neighbour_examples():
    """Can it read the shape SemanticEncoder._escalate passes (key "examples")?"""
    from mekiki.fallback import LLMFallback
    client = FakeClassifyClient(value="G")
    fb = LLMFallback(client=client)
    ctx = [{"examples": [{"text": "a G grade car", "value": "G",
                          "similarity": 0.82, "source": "human"}]}]
    fb.answer(["with navi"], ["G", "Z"], ctx)
    assert "a G grade car" in client.prompts[0]
    assert "0.820" in client.prompts[0]


def test_fallback_without_a_key_cannot_be_called():
    from mekiki.fallback import LLMFallback
    from mekiki.llm import ClaudeClient
    fb = LLMFallback(client=ClaudeClient(api_key="", cache_dir=None))
    assert fb.can_answer() is False
    with pytest.raises(NotImplementedError):
        fb.answer(["x"], ["G"], [{}])


def test_corrupt_cache_is_ignored_and_the_call_is_retried(tmp_path, data):
    """One corrupt file must not bring down a measurement of hundreds of rows."""
    client = FakeClient(cache_dir=tmp_path)
    key = client._key("s", "u", {"type": "object"})
    path = client._cache_path(key)
    for broken in ("{broken JSON", '{"data": "not a dict"}', '"a string"'):
        path.write_text(broken, encoding="utf-8")
        assert client._read_cache(key) is None


def test_passing_an_X_with_a_different_row_count_to_the_inspection_api_stops(data):
    """Silently returning the provenance of another X means judging from the wrong grounds."""
    m = make(data, client=FakeClient()).fit(data.iloc[:40])
    m.predict(data.iloc[40:50])
    m.confidence(data.iloc[40:50])            # the same row count passes
    with pytest.raises(MekikiError):
        m.confidence(data.iloc[40:45])
    with pytest.raises(MekikiError):
        m.cost(data.iloc[40:45])


def test_NeighbourModel_used_standalone_builds_the_index(data):
    from mekiki.predictor import NeighbourIndex, NeighbourModel
    spec = ColumnSpec(numeric=["age"], text="equipment_text")
    m = NeighbourModel("kNN", NeighbourIndex(spec, k=3))
    train = data.iloc[:40].reset_index(drop=True)
    m.fit(train, train["price"].to_numpy())
    pred = m.predict(data.iloc[40:45].reset_index(drop=True))
    assert pred.shape == (5,) and np.isfinite(pred).all()


# --- free text and answer-leak protection ------------------------------

def test_amount_masking():
    """If the listing text states the asking price, prediction becomes reading."""
    from mekiki.predictor import mask_amounts
    assert "$5,900" not in mask_amounts("Lowered! $5,900 plus fees")
    assert "12500" not in mask_amounts("asking 12500 dollars")
    assert "<AMOUNT>" in mask_amounts("price: $6,999")
    # Mileage and model year are caught too, but they are passed separately as
    # structured columns, so nothing is lost
    assert "222,617" not in mask_amounts("222,617 miles")
    assert "2008" not in mask_amounts("2008 Toyota Sienna")


def test_masking_keeps_model_numbers():
    """Losing F-150 or Model 3 would lose the essential model information."""
    from mekiki.predictor import mask_amounts
    out = mask_amounts("Ford F-150 Raptor 5.7L V8, Tesla Model 3")
    assert "F-150" in out and "5.7L" in out and "Model 3" in out


def test_free_text_is_shown_for_the_target_row_only_not_for_cases(data):
    """Pasting five cases in full inflates the prompt by an order of magnitude
    (confidence routing)."""
    d = data.assign(description=[f"this car is in good shape, {i}00 miles driven"
                                 for i in range(len(data))])
    client = FakeClient()
    m = EvidenceRegressor(target="price", unit="10k JPY", numeric=["age"],
                          text="equipment_text", long_text="description",
                          n_examples=3, client=client).fit(d.iloc[:40])
    m.predict(d.iloc[40:41])
    p = client.prompts[0]
    # The target row's description is shown
    assert "this car is in good shape" in p
    # Not shown for the cases (default long_text_example_chars=0)
    assert p.count("this car is in good shape") == 1


def test_free_text_is_cut_at_the_limit_and_the_cut_is_marked(data):
    d = data.assign(description=["@" * 5000] * len(data))
    client = FakeClient()
    spec = ColumnSpec(numeric=["age"], text="equipment_text",
                      long_text="description", long_text_chars=100)
    m = EvidenceRegressor(target="price", spec=spec, n_examples=2,
                          client=client).fit(d.iloc[:40])
    m.predict(d.iloc[40:41])
    p = client.prompts[0]
    assert "more characters omitted" in p
    assert p.count("@") <= 200      # the 5000 characters are not in there verbatim


def test_masking_can_be_switched_off(data):
    d = data.assign(description=["asking $9,999"] * len(data))
    client = FakeClient()
    spec = ColumnSpec(numeric=["age"], text="equipment_text", long_text="description",
                      mask_amounts_in_long_text=False)
    m = EvidenceRegressor(target="price", spec=spec, n_examples=2,
                          client=client).fit(d.iloc[:40])
    m.predict(d.iloc[40:41])
    assert "$9,999" in client.prompts[0]


# --- adding your own statistical model ---------------------------------

class LinearModel:
    """A user's own model: only `name`, `fit` and `predict` (least squares on two columns)."""

    name = "Linear"
    cols = ["age", "mileage_km"]

    def _design(self, df):
        return np.column_stack([np.ones(len(df)), df[self.cols].to_numpy(dtype=float)])

    def fit(self, train, y):
        self.coef_, *_ = np.linalg.lstsq(self._design(train), np.asarray(y, dtype=float),
                                         rcond=None)
        return self

    def predict(self, test):
        return self._design(test) @ self.coef_


class VotingLinearModel(LinearModel):
    name = "Linear (voting)"
    in_disagreement = True


class OffsetModel(LinearModel):
    """Always 10 above the linear fit, so the spread against it is known exactly."""

    name = "Offset"
    in_disagreement = True

    def predict(self, test):
        return super().predict(test) + 10.0


def test_default_keyword_adds_a_model_to_the_default_ones(data):
    client = FakeClient()
    m = make(data, client=client, escalate_rate=0.25,
             models=["default", LinearModel()]).fit(data.iloc[:40])
    assert [x.name for x in m.models_] == ["LightGBM", "XGBoost", "3-NN median", "Linear"]
    assert m.plan(data.iloc[40:])["n_escalated"] == 5
    m.predict(data.iloc[40:])
    assert len(client.prompts) == 5
    assert all("Linear" in p and "LightGBM" in p for p in client.prompts)


def test_an_added_model_does_not_move_the_disagreement_signal(data):
    """Evidence only, unless it opts in: the rows sent to the LLM stay the same."""
    plain = make(data, client=FakeClient(), escalate_rate=0.25).fit(data.iloc[:40])
    added = make(data, client=FakeClient(), escalate_rate=0.25,
                 models=["default", LinearModel()]).fit(data.iloc[:40])
    plain.predict(data.iloc[40:])
    added.predict(data.iloc[40:])
    np.testing.assert_array_equal(plain.signal_, added.signal_)
    np.testing.assert_array_equal(plain.selected_, added.selected_)


def test_a_model_that_opts_in_joins_the_disagreement_signal(data):
    plain = make(data, client=FakeClient(), escalate_rate=0.25).fit(data.iloc[:40])
    voting = make(data, client=FakeClient(), escalate_rate=0.25,
                  models=["default", VotingLinearModel()]).fit(data.iloc[:40])
    plain.predict(data.iloc[40:])
    voting.predict(data.iloc[40:])
    assert (voting.signal_ >= plain.signal_).all()      # max - min over a superset
    assert (voting.signal_ > plain.signal_).any()


def test_routing_works_without_tree_models_when_two_models_opt_in(data):
    m = make(data, client=FakeClient(), escalate_rate=0.25,
             models=[VotingLinearModel(), OffsetModel()]).fit(data.iloc[:40])
    m.predict(data.iloc[40:])
    np.testing.assert_allclose(m.signal_, 10.0)


def test_routing_without_opted_in_models_says_how_to_fix_it(data):
    m = make(data, client=FakeClient(), escalate_rate=0.25,
             models=[LinearModel()]).fit(data.iloc[:40])
    with pytest.raises(MekikiError, match="in_disagreement = True"):
        m.plan(data.iloc[40:])


def test_a_plain_list_still_replaces_the_default_models(data):
    m = make(data, client=FakeClient(), escalate_rate=1.0,
             models=[LinearModel()]).fit(data.iloc[:40])
    assert [x.name for x in m.models_] == ["Linear"]


def test_unknown_string_and_repeated_names_in_models_raise(data):
    with pytest.raises(MekikiError, match="Unknown entry 'defaults'"):
        make(data, client=FakeClient(), models=["defaults"]).fit(data)
    with pytest.raises(MekikiError, match="must be unique"):
        make(data, client=FakeClient(),
             models=["default", LinearModel(), LinearModel()]).fit(data)


class PriorRateModel:
    """A user's own classifier: the training class rates for every row."""

    name = "Prior"
    in_disagreement = True

    def fit(self, train, y):
        self.p_ = np.bincount(y) / len(y)
        return self

    def predict(self, test):
        return np.full(len(test), int(np.argmax(self.p_)))

    def predict_proba(self, test):
        return np.tile(self.p_, (len(test), 1))


def test_adding_a_model_works_for_classification(data):
    df = data.assign(expensive=np.where(data["price"] > data["price"].median(), "yes", "no"))
    kw = dict(target="expensive", task="classification", numeric=["age", "mileage_km"],
              categorical=["grade"], n_examples=3, escalate_rate=0.0)
    plain = EvidencePredictor(client=FakeClient(), **kw).fit(df.iloc[:40])
    added = EvidencePredictor(client=FakeClient(), models=["default", PriorRateModel()],
                              **kw).fit(df.iloc[:40])
    plain.predict(df.iloc[40:])
    pred = added.predict(df.iloc[40:])
    assert set(pred) <= {"yes", "no"}
    assert added.evidence(df.iloc[40:])["Prior"].shape == (20, 2)
    # total variation distance, max over pairs: within [0, 1] and never below the trees'
    assert ((added.signal_ >= plain.signal_) & (added.signal_ <= 1.0)).all()
