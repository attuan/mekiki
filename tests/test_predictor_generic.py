"""Tests that `EvidencePredictor` works beyond used cars (`Domain` and `task="classification"`).

Like `test_predictor.py`, **everything passes without an API key**.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mekiki import USED_CAR, Domain, EvidenceClassifier, EvidencePredictor, EvidenceRegressor, MekikiError
from mekiki.llm import ClaudeClient, LLMAnswer, _sha8
from mekiki.predictor import build_system_prompt


class FakeClient(ClaudeClient):
    """Returns a canned answer matching the schema, without any HTTP. Works for both
    regression and classification."""

    def __init__(self, value=200.0, label: str | None = None,
                 probabilities: dict | None = None, fail_rows: set[int] | None = None,
                 **kw):
        kw.setdefault("cache_dir", None)
        kw.setdefault("api_key", "test-key")
        super().__init__(**kw)
        self.value, self.label, self.probabilities = value, label, probabilities
        self.fail_rows = fail_rows or set()
        self.prompts: list[str] = []
        self.systems: list[str] = []
        self.schemas: list[dict] = []

    def ask(self, system, user, schema) -> LLMAnswer:
        i = len(self.prompts)
        self.prompts.append(user)
        self.systems.append(system)
        self.schemas.append(schema)
        self.usage.calls += 1
        if i in self.fail_rows:
            self.usage.errors += 1
            return LLMAnswer(data={}, error="deliberate failure")
        self.usage.cost += 0.001
        props = schema["properties"]
        if "label" in props:
            data = {"label": self.label, "probabilities": self.probabilities,
                    "confidence": 0.8, "reason": "test"}
        else:
            key = next(k for k in props if k not in ("confidence", "reason"))
            data = {key: self.value, "confidence": 0.8, "reason": "test"}
        return LLMAnswer(data=data, input_tokens=100, output_tokens=20, cost=0.001)


@pytest.fixture
def wine() -> pd.DataFrame:
    """Wine-like synthetic data where the points are almost determined by text and
    price (regression)."""
    rng = np.random.default_rng(1)
    n = 60
    price = rng.integers(8, 120, n)
    country = rng.choice(["US", "France", "Italy"], n)
    words = ["oak", "cherry", "tannin", "citrus", "mineral"]
    return pd.DataFrame({
        "price": price,
        "country": country,
        "description": [f"{words[i % 5]} notes, {c} style, batch {i}"
                        for i, c in enumerate(country)],
        "points": np.clip(80 + price / 6 + rng.normal(0, 1.5, n), 80, 100).round(),
    })


@pytest.fixture
def churn() -> pd.DataFrame:
    """Telecom-like synthetic data where churn is almost determined by tenure and
    monthly charges (binary classification)."""
    rng = np.random.default_rng(2)
    n = 80
    tenure = rng.integers(1, 72, n)
    monthly = rng.uniform(20, 110, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n)
    logit = -0.05 * tenure + 0.03 * monthly + (contract == "Month-to-month") * 1.5 - 1
    churned = rng.random(n) < 1 / (1 + np.exp(-logit))
    return pd.DataFrame({
        "tenure": tenure, "MonthlyCharges": monthly.round(2), "Contract": contract,
        "note": [f"customer {i} on {c}" for i, c in enumerate(contract)],
        "Churn": np.where(churned, "Yes", "No"),
    })


@pytest.fixture
def pets() -> pd.DataFrame:
    """Three-class synthetic data (multiclass classification with integer labels)."""
    rng = np.random.default_rng(3)
    n = 90
    age = rng.integers(1, 120, n)
    photos = rng.integers(0, 8, n)
    speed = np.clip((age // 40) + (photos < 2), 0, 2)
    return pd.DataFrame({
        "Age": age, "PhotoAmt": photos,
        "Description": [f"pet {i} age {a} months, friendly" for i, a in enumerate(age)],
        "AdoptionSpeed": speed.astype(int),
    })


# --- wording (Domain) -----------------------------------------------------

def test_used_car_preset_wording_is_pinned():
    """If the fingerprint of the system prompt changes, the whole LLM cache is lost and
    every row is charged again, so the wording is pinned here."""
    assert _sha8(USED_CAR.system_prompt) == "32ceaf1e"
    assert USED_CAR.answer_key == "price"


def test_used_car_preset_user_prompt_keeps_its_headings(wine):
    """The headings are part of the user prompt (the cache key), so they are pinned too."""
    client = FakeClient()
    m = EvidenceRegressor(target="points", unit="pts", domain=USED_CAR,
                          numeric=["price"], text="description", n_examples=2,
                          client=client).fit(wine.iloc[:40])
    m.predict(wine.iloc[40:41])
    p = client.prompts[0]
    assert p.startswith("## The car being appraised")
    assert "similar cases (training data with known actual price)" in p
    assert "actual price:" in p
    assert p.rstrip().endswith("Give one value for the price of this car (unit: pts).")
    assert client.systems[0] == USED_CAR.system_prompt


def test_default_wording_has_no_used_car_vocabulary(wine):
    client = FakeClient()
    m = EvidenceRegressor(target="points", unit="pts",
                          numeric=["price"], categorical=["country"],
                          text="description", n_examples=2,
                          client=client).fit(wine.iloc[:40])
    m.predict(wine.iloc[40:41])
    for text in (client.systems[0], client.prompts[0]):
        for word in ("used car", "apprais", "vehicle", "equipment", "listing", "asking price"):
            assert word not in text, word
    # The target name defaults to the column name
    assert "Give one value for the points of this record (unit: pts)." in client.prompts[0]
    assert client.schemas[0]["required"] == ["value", "confidence", "reason"]


def test_Domain_wording_is_reflected_in_the_prompt(wine):
    client = FakeClient()
    d = Domain(role="a wine judge", subject="wine", target_name="points",
               hints=["Mentions of acidity, tannin and finish move the points."])
    m = EvidenceRegressor(target="points", unit="pts", domain=d,
                          numeric=["price"], text="description", n_examples=2,
                          client=client).fit(wine.iloc[:40])
    m.predict(wine.iloc[40:41])
    s, p = client.systems[0], client.prompts[0]
    assert s.startswith("You are a wine judge.")
    assert "answer the points of the wine as a single number" in s
    assert "acidity, tannin and finish" in s
    assert "## The record to predict" in p
    assert "actual points:" in p
    assert p.rstrip().endswith("Give one value for the points of this wine (unit: pts).")


def test_masking_note_appears_only_when_masking():
    d = Domain(subject="property", target_name="rent")
    with_mask = build_system_prompt(d, "regression", "rent", masked=True)
    without = build_system_prompt(d, "regression", "rent", masked=False)
    assert "<AMOUNT>" in with_mask and "The rent itself" in with_mask
    assert "<AMOUNT>" not in without


def test_invalid_task_stops():
    with pytest.raises(MekikiError):
        EvidencePredictor(target="y", task="ranking", client=FakeClient())


# --- classification (binary) ----------------------------------------------

def make_churn(churn, **kw) -> EvidencePredictor:
    return EvidenceClassifier(
        target="Churn",
        domain=Domain(role="a churn analyst", subject="customer", target_name="churn",
                      class_names={"Yes": "churned", "No": "stayed"}),
        numeric=["tenure", "MonthlyCharges"], categorical=["Contract"], text="note",
        n_examples=3, **kw)


def test_classification_fit_predict_predict_proba(churn):
    client = FakeClient(label="Yes", probabilities={"Yes": 0.7, "No": 0.3})
    m = make_churn(churn, client=client).fit(churn.iloc[:60])
    assert list(m.classes_) == ["No", "Yes"]
    pred = m.predict(churn.iloc[60:])
    assert pred.shape == (20,) and set(pred) == {"Yes"}
    proba = m.predict_proba(churn.iloc[60:])
    assert proba.shape == (20, 2)
    assert np.allclose(proba[:, 1], 0.7) and np.allclose(proba.sum(axis=1), 1.0)


def test_classification_prompt_has_candidates_probabilities_and_case_labels(churn):
    client = FakeClient(label="No", probabilities={"Yes": 0.2, "No": 0.8})
    m = make_churn(churn, client=client).fit(churn.iloc[:60])
    m.predict(churn.iloc[60:61])
    s, p, schema = client.systems[0], client.prompts[0], client.schemas[0]
    assert "Candidates:" in s and "Yes (churned)" in s and "No (stayed)" in s
    assert "which of the following candidates the churn of the customer is" in s
    assert "statistical model predictions (probability per candidate)" in p
    assert "LightGBM: No (stayed)" in p and "XGBoost:" in p and "3-NN class rates" in p
    assert "actual churn: " in p
    assert p.rstrip().endswith("Choose the churn of this customer from the candidates, "
                               "and give the probability of each candidate.")
    assert schema["properties"]["label"]["enum"] == ["No", "Yes"]
    assert set(schema["properties"]["probabilities"]["required"]) == {"No", "Yes"}


def test_classification_does_not_mask_numbers_in_free_text(churn):
    d = churn.assign(memo=["paid 12500 dollars last year"] * len(churn))
    client = FakeClient(label="No", probabilities={"Yes": 0.2, "No": 0.8})
    m = EvidenceClassifier(target="Churn",
                          numeric=["tenure"], long_text="memo", n_examples=2,
                          client=client).fit(d.iloc[:60])
    assert m.spec.mask_amounts_in_long_text is False
    m.predict(d.iloc[60:61])
    assert "12500 dollars" in client.prompts[0]
    assert "<AMOUNT>" not in client.systems[0]


def test_regression_masks_by_default(wine):
    d = wine.assign(memo=["asking $9,999"] * len(wine))
    client = FakeClient()
    m = EvidenceRegressor(target="points", numeric=["price"], long_text="memo",
                          n_examples=2, client=client).fit(d.iloc[:40])
    assert m.spec.mask_amounts_in_long_text is True
    m.predict(d.iloc[40:41])
    assert "<AMOUNT>" in client.prompts[0] and "$9,999" not in client.prompts[0]


def test_label_outside_the_candidates_falls_back_to_the_statistical_model(churn):
    client = FakeClient(label="Maybe", probabilities={"Yes": 0.5, "No": 0.5})
    m = make_churn(churn, client=client).fit(churn.iloc[:60])
    pred = m.predict(churn.iloc[60:65])
    prov = m.provenance()
    assert (prov["source"] == "fallback").all()
    # The substitute matches the most probable label of the first model (LightGBM)
    assert list(pred) == list(prov["evidence_LightGBM"])
    assert "evidence_LightGBM_proba" in prov.columns
    assert np.allclose(m.proba_.sum(axis=1), 1.0)


def test_rows_where_the_LLM_failed_still_get_probabilities(churn):
    client = FakeClient(label="Yes", probabilities={"Yes": 0.9, "No": 0.1},
                        fail_rows={1})
    m = make_churn(churn, client=client).fit(churn.iloc[:60])
    proba = m.predict_proba(churn.iloc[60:64])
    assert proba.shape == (4, 2) and np.isfinite(proba).all()
    assert m.provenance()["source"].tolist() == ["llm", "fallback", "llm", "llm"]


def test_broken_probabilities_are_normalised(churn):
    client = FakeClient(label="Yes", probabilities={"Yes": 3.0, "No": -1.0})
    m = make_churn(churn, client=client).fit(churn.iloc[:60])
    proba = m.predict_proba(churn.iloc[60:62])
    assert np.allclose(proba, [[0.0, 1.0], [0.0, 1.0]])


def test_classification_explain_and_examples(churn):
    client = FakeClient(label="Yes", probabilities={"Yes": 0.7, "No": 0.3})
    m = make_churn(churn, client=client).fit(churn.iloc[:60])
    m.predict(churn.iloc[60:62])
    s = m.explain(0)
    assert "Yes (churned)" in s and "LightGBM" in s and "churn" in s
    ex = m.examples(i=0)
    assert set(ex["label"]) <= {"Yes", "No"} and len(ex) == 3


def test_missing_labels_or_a_single_label_stop(churn):
    with pytest.raises(MekikiError):
        make_churn(churn, client=FakeClient()).fit(churn.assign(Churn="No"))
    bad = churn.copy()
    bad.loc[0, "Churn"] = None
    with pytest.raises(MekikiError):
        make_churn(churn, client=FakeClient()).fit(bad)


def test_too_many_classes_stop():
    n = 60
    df = pd.DataFrame({"x": np.arange(n), "label": [f"c{i % 30}" for i in range(n)]})
    m = EvidenceClassifier(target="label", numeric=["x"],
                          client=FakeClient())
    with pytest.raises(MekikiError, match="limit"):
        m.fit(df)


def test_predict_proba_in_regression_stops(wine):
    m = EvidenceRegressor(target="points", numeric=["price"], n_examples=2,
                          client=FakeClient()).fit(wine.iloc[:40])
    with pytest.raises(MekikiError):
        m.predict_proba(wine.iloc[40:])


# --- classification (multiclass, integer labels) -----------------------------

def test_multiclass_classification_with_integer_labels(pets):
    client = FakeClient(label="2", probabilities={"0": 0.1, "1": 0.2, "2": 0.7})
    m = EvidenceClassifier(target="AdoptionSpeed",
                          numeric=["Age", "PhotoAmt"], long_text="Description",
                          n_examples=3, client=client).fit(pets.iloc[:70])
    assert list(m.classes_) == [0, 1, 2]
    pred = m.predict(pets.iloc[70:])
    assert pred.dtype.kind in "iu" and set(pred) == {2}
    assert m.proba_.shape == (20, 3)
    assert client.schemas[0]["properties"]["label"]["enum"] == ["0", "1", "2"]


# --- classification under confidence routing ---------------------------------

def make_adaptive(churn, **kw) -> EvidencePredictor:
    return EvidenceClassifier(
        target="Churn",
        domain=Domain(subject="customer", target_name="churn"),
        numeric=["tenure", "MonthlyCharges"], categorical=["Contract"], text="note",
        n_examples=3, **kw)


def test_adaptive_classification_sends_only_selected_rows_to_the_LLM(churn):
    client = FakeClient(label="Yes", probabilities={"Yes": 0.9, "No": 0.1})
    m = make_adaptive(churn, client=client, escalate_rate=0.25).fit(churn.iloc[:60])
    test = churn.iloc[60:]
    pred = m.predict(test)
    assert len(client.prompts) == 5 and int(m.selected_.sum()) == 5
    assert set(pred) <= {"Yes", "No"}
    assert (pred[m.selected_] == "Yes").all()
    # Rows not sent carry LightGBM's most probable label
    base = m.models_[0].predict_proba(test.reset_index(drop=True))
    fast = ~m.selected_
    assert (pred[fast] == m.classes_[np.argmax(base, axis=1)][fast]).all()
    proba = m.predict_proba(test)
    assert proba.shape == (20, 2) and np.allclose(proba.sum(axis=1), 1.0)


def test_adaptive_disagreement_signal_is_a_distance_between_probabilities(churn):
    m = make_adaptive(churn, client=FakeClient(label="Yes",
                                               probabilities={"Yes": 0.9, "No": 0.1}),
                      escalate_rate=0.5).fit(churn.iloc[:60])
    test = churn.iloc[60:].reset_index(drop=True)
    ev = m.evidence(test)
    s = m._signal_values(test, ev)
    assert s.shape == (20,) and (s >= 0).all() and (s <= 1).all()
    tv = 0.5 * np.abs(ev["LightGBM"] - ev["XGBoost"]).sum(axis=1)
    assert np.allclose(s, tv)


def test_adaptive_curve_returns_accuracy(churn):
    client = FakeClient(label="Yes", probabilities={"Yes": 0.9, "No": 0.1})
    m = make_adaptive(churn, client=client, escalate_rate=0.5).fit(churn.iloc[:60])
    c = m.curve(churn.iloc[60:], rates=(0.0, 1.0))
    assert "accuracy" in c.columns and "MAE" not in c.columns
    truth = churn.iloc[60:]["Churn"].to_numpy()
    assert c.loc[1, "accuracy"] == pytest.approx(float(np.mean(truth == "Yes")))
    assert 0.0 <= c.loc[0, "accuracy"] <= 1.0


def test_adaptive_approval_keeps_the_label_as_is(churn):
    client = FakeClient(label="Yes", probabilities={"Yes": 0.9, "No": 0.1})
    m = make_adaptive(churn, client=client, escalate_rate=0.25).fit(churn.iloc[:60])
    test = churn.iloc[60:]
    m.predict(test)
    assert m.approve() == 5
    assert set(m.approved_.values()) == {"Yes"}
    pred = m.predict(test)
    assert len(client.prompts) == 5          # approved rows are not called again
    prov = m.provenance()
    assert (prov["source"] == "human").sum() == 5
    assert (pred[prov["source"] == "human"] == "Yes").all()
    assert "Yes" in m.explain(int(np.flatnonzero(prov["source"] == "human")[0]))


def test_integer_class_names_match_stringified_labels():
    """Even when `classes_` are integers, the candidates in the prompt carry their meaning."""
    d = Domain(class_names={0: "same day", 1: "within a week"})
    assert d.label("0") == "0 (same day)"
    assert d.label(np.int64(1)) == "1 (within a week)"
    assert d.label("9") == "9"
    prompt = build_system_prompt(d, "classification", "y", classes=["0", "1"])
    assert "0 (same day)" in prompt and "1 (within a week)" in prompt


# --- columns omitted: assigned from the data in fit ----------------------------


def test_regression_without_columns_assigns_them_in_fit(wine):
    client = FakeClient()
    m = EvidenceRegressor(target="points", n_examples=2, client=client)
    assert m.spec is None
    m.fit(wine.iloc[:40])
    assert m.spec_.numeric == ["price"]
    assert "points" not in m.spec_.all_columns()
    assert m.spec_.mask_amounts_in_long_text is True
    pred = m.predict(wine.iloc[40:42])
    assert len(pred) == 2 and len(client.prompts) == 2
    assert "price" in client.prompts[0]


def test_classification_without_columns_leaves_out_identifiers(churn):
    d = churn.assign(customerID=[f"{i:04d}-XKQZ" for i in range(len(churn))])
    client = FakeClient(label="No", probabilities={"Yes": 0.2, "No": 0.8})
    m = EvidenceClassifier(target="Churn", n_examples=2, client=client).fit(d.iloc[:60])
    assert set(m.spec_.numeric) == {"tenure", "MonthlyCharges"}
    assert "Contract" in m.spec_.categorical
    assert "customerID" not in m.spec_.all_columns()
    assert m.spec_.mask_amounts_in_long_text is False
    assert m.predict_proba(d.iloc[60:62]).shape == (2, 2)


def test_given_columns_are_used_as_they_are(wine):
    m = EvidenceRegressor(target="points", numeric=["price"], n_examples=2,
                          client=FakeClient()).fit(wine.iloc[:40])
    assert m.spec_ is m.spec
    assert m.spec_.all_columns() == ["price"]


def test_fit_with_y_outside_X_assigns_every_column_of_X(wine):
    X, y = wine.drop(columns="points"), wine["points"]
    m = EvidenceRegressor(target="points", n_examples=2,
                          client=FakeClient()).fit(X.iloc[:40], y.iloc[:40])
    assert set(m.spec_.all_columns()) == set(X.columns)


def test_without_columns_and_nothing_to_retrieve_by_it_stops():
    d = pd.DataFrame({"colour": ["red", "blue", "green"] * 10, "y": np.arange(30.0)})
    with pytest.raises(MekikiError, match="Pass the columns explicitly"):
        EvidenceRegressor(target="y", client=FakeClient()).fit(d)
