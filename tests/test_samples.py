"""Tests that `SemanticEncoder`, `EvidencePredictor`, routing and screen run on the bundled samples
(500 real rows).

`test_predictor_generic.py` checks the API contract on synthetic data. This file checks
that the same paths work with **real column names, real missing values and real text**.
vehicles (regression, long free-form listings), news (regression, short headlines),
bank (binary classification, no text at all).

Everything passes without an API key (the LLM is a fake client). The three datasets are
exactly the ones the distribution bundles, so nothing here depends on data that cannot
be redistributed. Column choices are kept minimal here.

Run: .venv/bin/python -m pytest tests -q
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

from mekiki import (
    Domain,
    EvidencePredictor,
    MekikiError,
    SemanticEncoder,
    check_duplicates,
    paths,
    screen,
)
from mekiki.llm import ClaudeClient, LLMAnswer


class FakeClient(ClaudeClient):
    """Emits no HTTP; returns a canned answer that echoes the first statistical model."""

    def __init__(self, **kw):
        kw.setdefault("cache_dir", None)
        kw.setdefault("api_key", "test-key")
        super().__init__(**kw)
        self.calls = 0

    def ask(self, system, user, schema) -> LLMAnswer:
        self.calls += 1
        self.usage.calls += 1
        self.usage.cost += 0.001
        props = schema["properties"]
        if "label" in props:
            labels = props["label"]["enum"]
            probs = {c: 1.0 / len(labels) for c in labels}
            data = {"label": labels[0], "probabilities": probs,
                    "confidence": 0.6, "reason": "test"}
        else:
            key = next(k for k in props if k not in ("confidence", "reason"))
            data = {key: 1.0, "confidence": 0.6, "reason": "test"}
        return LLMAnswer(data=data, input_tokens=100, output_tokens=20, cost=0.001)


@dataclass(frozen=True)
class Spec:
    file: str
    target: str
    task: str
    numeric: list[str]
    categorical: list[str]
    text: str | None
    domain: Domain
    values: list[str] = field(default_factory=list)   # declared values for `SemanticEncoder`
    truth: str | None = None                          # ground-truth column for `SemanticEncoder`
    prep: object = None                               # post-load preparation


def _prep_vehicles(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the rows that can be trained on: a usable price and no missing basics.

    The raw excerpt has a price of up to $3,736,928,711 and empty `year` /
    `manufacturer` on the first row, so this runs before every test. 422 of the
    500 rows survive.
    """
    df = df[df["price"].between(1_000, 100_000)]
    df = df[df["year"].notna() & df["odometer"].notna()
            & df["manufacturer"].notna() & df["description"].notna()]
    return df.reset_index(drop=True)


def _prep_news(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the rows whose share count was never measured (`-1`, not zero). 451 of 500 survive."""
    return df[(df["Facebook"] >= 0) & df["Headline"].notna()].reset_index(drop=True)


SPECS = {
    "vehicles": Spec(
        file="vehicles_sample500.csv", target="price", task="regression",
        numeric=["year", "odometer"],
        categorical=["manufacturer", "state", "fuel"],
        text="description",
        domain=Domain(role="a used-car appraiser", subject="used car",
                      target_name="the price"),
        values=["sedan", "SUV", "pickup", "truck", "coupe"],
        truth="type",
        prep=_prep_vehicles,
    ),
    "news": Spec(
        file="news_sample500.csv", target="Facebook", task="regression",
        numeric=["SentimentTitle", "SentimentHeadline"], categorical=["Topic"],
        text="Headline",
        domain=Domain(role="a news editor", subject="news article",
                      target_name="the number of Facebook shares"),
        values=["economy", "microsoft", "obama", "palestine"],
        truth="Topic",
        prep=_prep_news,
    ),
    "bank": Spec(
        file="bank_sample500.csv", target="y", task="classification",
        numeric=["age", "balance", "duration"],
        categorical=["job", "marital", "poutcome"],
        text=None,
        domain=Domain(role="a bank marketing analyst", subject="customer",
                      target_name="whether the customer subscribes to a term deposit",
                      class_names={"yes": "subscribed", "no": "did not subscribe"}),
    ),
}
NAMES = sorted(SPECS)


@pytest.fixture(params=NAMES)
def sample(request) -> tuple[str, Spec, pd.DataFrame]:
    name = request.param
    spec = SPECS[name]
    path = paths.sample_data(spec.file)
    assert path.is_file(), f"sample data is missing: {path}"
    df = pd.read_csv(path)
    if spec.prep:
        df = spec.prep(df)
    if spec.text:
        df[spec.text] = df[spec.text].fillna("")
    return name, spec, df.reset_index(drop=True)


def _split(df: pd.DataFrame, n_test: int = 100):
    return (df.iloc[n_test:].reset_index(drop=True),
            df.iloc[:n_test].reset_index(drop=True))


# --- screen ------------------------------------------------------------

def test_screen_uses_the_metric_for_the_task_on_samples(sample):
    name, spec, df = sample
    if spec.text is None:
        with pytest.raises(MekikiError):
            screen(df, target=spec.target, text="description", task=spec.task,
                   numeric=spec.numeric, categorical=spec.categorical)
        return
    r = screen(df, target=spec.target, text=spec.text, task=spec.task,
               numeric=spec.numeric, categorical=spec.categorical, n_splits=3)
    assert r.task == spec.task
    assert r.metric == ("log loss" if spec.task == "classification" else "MAE")
    assert r.verdict in ("worth_trying", "unlikely_to_help", "inconclusive")
    assert r.n_rows == len(df)


# --- SemanticEncoder ---------------------------------------------------------

def test_feature_A_returns_only_declared_values_with_provenance(sample):
    name, spec, df = sample
    if spec.text is None:
        pytest.skip("no text column")
    col = SemanticEncoder(source=spec.text, type="category", values=spec.values,
                      escalate_rate=0.1, name=f"{spec.truth}_typed")
    out = col.fit_transform(df)
    assert len(out) == len(df)
    assert set(out.astype(str).unique()) <= set(spec.values)
    prov = col.provenance_
    assert set(prov["source"].unique()) <= {"model", "llm", "needs_review"}
    # No LLM is plugged in, so escalated rows are only queued for review (cost 0)
    assert int((prov["source"] == "needs_review").sum()) == round(len(df) * 0.1)
    assert col.cost()["actual_cost_usd"] == 0
    assert 0.0 <= col.confidence().min() <= col.confidence().max() <= 1.0


def test_feature_A_embedding_type_builds_columns_without_extra_dependencies(sample):
    name, spec, df = sample
    if spec.text is None:
        pytest.skip("no text column")
    train, test = _split(df)
    col = SemanticEncoder(source=spec.text, type="embedding", name="emb")
    Etr = col.fit_transform(train)
    Ete = col.transform(test)
    assert Etr.shape[0] == len(train) and Ete.shape[1] == Etr.shape[1]
    assert np.isfinite(Ete.to_numpy()).all()


# --- EvidencePredictor ---------------------------------------------------------

def _predictor(spec: Spec, client: FakeClient, **kw) -> EvidencePredictor:
    return EvidencePredictor(target=spec.target, task=spec.task, domain=spec.domain,
                             numeric=spec.numeric, categorical=spec.categorical,
                             text=spec.text, n_examples=3, client=client,
                             check_leakage=False, **{"escalate_rate": 1.0, **kw})


def test_feature_B_can_fit_and_predict_on_samples(sample):
    name, spec, df = sample
    train, test = _split(df)
    client = FakeClient()
    m = _predictor(spec, client).fit(train)
    pred = m.predict(test)
    assert pred.shape == (len(test),)
    assert client.calls == len(test)
    if spec.task == "classification":
        classes = np.unique(train[spec.target])
        assert list(m.classes_) == list(classes)
        proba = m.proba_
        assert proba.shape == (len(test), len(classes))
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert set(pred.tolist()) <= set(classes.tolist())
    else:
        assert np.isfinite(pred).all()
        with pytest.raises(MekikiError):
            m.predict_proba(test)
    # Provenance and inspection API
    assert len(m.provenance()) == len(test)
    assert "llm" in m.explain(0)
    assert m.cost()["n_predicted"] == len(test)


def test_feature_B_prompt_contains_Domain_wording_and_real_column_names(sample):
    name, spec, df = sample
    train, test = _split(df, n_test=3)
    client = FakeClient()
    m = _predictor(spec, client).fit(train)
    m.predict(test)
    body = m.explain(0)
    assert spec.domain.target_name in m.system_prompt_
    assert spec.domain.role in m.system_prompt_
    for c in spec.numeric[:1] + spec.categorical[:1]:
        assert c in body


# --- Routing -----------------------------------------------------------

def test_routing_plan_and_curve_work_on_samples(sample):
    name, spec, df = sample
    train, test = _split(df)
    client = FakeClient()
    m = _predictor(spec, client, escalate_rate=0.2).fit(train)
    plan = m.plan(test)
    assert plan["n_escalated"] == round(len(test) * 0.2)
    assert client.calls == 0, "plan() does not call the LLM"
    pred = m.predict(test)
    assert client.calls == round(len(test) * 0.2)
    assert len(pred) == len(test)
    curve = m.curve(test, rates=(0.0, 0.5, 1.0))
    key = "accuracy" if spec.task == "classification" else "MAE"
    assert key in curve.columns
    assert list(curve["n_escalated"]) == [0, len(test) // 2, len(test)]


# --- Leak check --------------------------------------------------------

def test_duplicate_free_text_is_detected(sample):
    name, spec, df = sample
    if spec.text is None:
        pytest.skip("no text column")
    # If the 500-row sample has no duplicates, make one by copying a row
    # (7% of the full news data repeats a headline, 42% of vehicles repeats a car)
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    r = check_duplicates(dup, keys=[spec.text])
    assert r.n_duplicate_rows >= 1
    assert r.n_groups >= 1
