"""Tests for `diagnose`.

Stage 1 (rule-based diagnosis) is checked on synthetic data: column kinds, target
distribution and duplicate detection. Stage 2 (LLM) is replaced by `FakeClient`, so
**everything passes without an API key**. Tests that call `screen()` use 2 folds (the point
is the wiring, not accuracy).

Run: .venv/bin/python -m pytest tests/test_diagnose.py -q
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from mekiki import ColumnSpec, Domain, MekikiError, diagnose
from mekiki.diagnose import (
    Diagnosis,
    Recommendation,
    build_user_prompt,
    profile_column,
    profile_target,
    run_diagnosis,
)
from mekiki.llm import ClaudeClient, LLMAnswer


def make_listings(n: int = 240, seed: int = 0) -> pd.DataFrame:
    """Synthetic listing-like data. Skewed price; has an id, URL, date, constant and free text."""
    rng = np.random.default_rng(seed)
    age = rng.integers(1, 15, n)
    km = rng.integers(5_000, 200_000, n)
    grade = rng.choice(["standard", "sport", "luxury"], n)
    bonus = {"standard": 0, "sport": 60, "luxury": 160}
    price = np.exp(rng.normal(5.5, 0.6, n)) - age * 5 + np.array([bonus[g] for g in grade])
    price = np.clip(price, 10, None)
    price[0] = price.max() * 50          # outlier from an input error
    return pd.DataFrame({
        "listing_id": np.arange(1000, 1000 + n),
        "url": [f"https://example.com/l/{i}" for i in range(n)],
        "posted_at": pd.date_range("2026-01-01", periods=n, freq="h").strftime("%Y-%m-%d %H:%M"),
        "age": age,
        "km": km,
        "region": rng.choice(["Tokyo", "Osaka", "Aichi", "Fukuoka"], n),
        "has_warranty": rng.choice(["Yes", "No"], n),
        "model": [f"model {g} edition" for g in grade],
        "description": ["This is a long free-text description of the listing. " * 4
                        + f"grade {g}" for g in grade],
        "county": [None] * n,
        "price": price,
    })


class FakeClient(ClaudeClient):
    """Sends no HTTP; records the prompts it receives and returns a prepared answer."""

    def __init__(self, answer: dict | None = None, fail: bool = False, **kw):
        kw.setdefault("cache_dir", None)
        kw.setdefault("api_key", "test-key")
        super().__init__(**kw)
        self.answer = answer or {}
        self.fail = fail
        self.prompts: list[str] = []
        self.schemas: list[dict] = []

    def ask(self, system, user, schema) -> LLMAnswer:
        self.prompts.append(user)
        self.schemas.append(schema)
        self.usage.calls += 1
        if self.fail:
            self.usage.errors += 1
            return LLMAnswer(data={}, error="deliberate failure")
        self.usage.cost += 0.01
        return LLMAnswer(data=self.answer, input_tokens=500, output_tokens=200, cost=0.01)


GOOD_ANSWER = {
    "task": "regression", "target_name": "price", "unit": "10k JPY",
    "role": "a used-car appraiser", "subject": "car", "class_names": [],
    "hints": ["Weigh accident history and repairs mentioned in the free text heavily"],
    "numeric": ["age", "km"], "boolean": ["has_warranty"], "categorical": ["region"],
    "text": "model", "long_text": "description",
    "target_transform": "log1p", "clip_outliers": True,
    "dedup_ignore": ["listing_id", "url", "posted_at", "region"],
    "typed_columns": [{"name": "grade", "source": "description", "type": "category",
                       "values": ["standard", "sport", "luxury"],
                       "why": "The grade is written at the end of description"}],
    "knowledge_columns": [{"name": "body_style", "keys": ["model"], "attribute": "body style",
                           "type": "category", "values": ["sedan", "SUV", "minivan"], "unit": "",
                           "why": "model names a car whose body style is common knowledge"},
                          {"name": "new_price", "keys": ["model"],
                           "attribute": "approximate price when new", "type": "numeric",
                           "values": [], "unit": "10k JPY",
                           "why": "the price when new anchors the used price"}],
    "use_evidence": True, "escalate_rate": 0.25,
    "warnings": ["Needs checking whether region can differ for the same car"],
    "reasons": [{"topic": "dedup", "reason": "region can differ for the same car, so ignore it"},
                {"topic": "log_transform", "reason": "mean_over_median exceeds the threshold"}],
}


# --- Stage 1: column kinds --------------------------------------------------

def test_column_kind_is_decided_from_dtype_and_distinct_values():
    df = make_listings()
    kinds = {c: profile_column(df[c], len(df)).kind for c in df.columns if c != "price"}
    assert kinds["listing_id"] == "id"
    assert kinds["url"] == "id"
    assert kinds["posted_at"] == "datetime"
    assert kinds["age"] == "numeric" and kinds["km"] == "numeric"
    assert kinds["region"] == "categorical"
    assert kinds["has_warranty"] == "categorical"   # string Yes/No is not boolean (trees cast to float)
    assert kinds["model"] == "categorical"          # short text with only 3 values is categorical
    assert kinds["description"] == "long_text"
    assert kinds["county"] == "constant"


def test_id_like_name_with_repeated_values_is_treated_as_id_suspecting_relisting():
    s = pd.Series([f"VIN{i % 120:05d}" for i in range(200)], name="VIN")
    p = profile_column(s, 200)
    assert p.kind == "id" and "multiple times" in p.note


def test_short_strings_with_many_distinct_values_become_text():
    s = pd.Series([f"model {i} {'sport' if i % 2 else 'base'} edition" for i in range(400)],
                  name="title")
    assert profile_column(s, 400).kind == "text"        # all distinct, but words are shared


def test_short_strings_all_distinct_sharing_no_words_become_id():
    s = pd.Series([f"WBA3A5C51CF{i:06d}" for i in range(400)], name="serial")
    assert profile_column(s, 400).kind == "id"


def test_unnamed_csv_column_is_treated_as_row_number_id():
    p = profile_column(pd.Series(np.arange(300), name="Unnamed: 0"), 300)
    assert p.kind == "id" and "row number" in p.note


def test_integer_column_with_few_values_gets_a_code_note():
    p = profile_column(pd.Series([1, 2, 3] * 50, name="Health"), 150)
    assert p.kind == "numeric" and "code" in p.note


# --- Stage 1: target --------------------------------------------------------

def test_skewed_price_is_regression_with_log_and_clip_recommended():
    t = profile_target(make_listings()["price"])
    assert t.task == "regression"
    assert t.suggest_log and t.suggest_clip
    assert t.skew_ratio > 1.5


def test_integers_with_few_distinct_values_are_classification():
    t = profile_target(pd.Series([0, 1, 2, 3, 4] * 40, name="AdoptionSpeed"))
    assert t.task == "classification" and set(t.classes) == {"0", "1", "2", "3", "4"}
    assert t.native_classes["0"] == 0


def test_integers_with_many_distinct_values_are_regression():
    t = profile_target(pd.Series(np.arange(80, 101).repeat(5), name="points"))
    assert t.task == "regression"


def test_class_imbalance_is_detected():
    y = pd.Series(["No"] * 190 + ["Yes"] * 10, name="Churn")
    t = profile_target(y)
    assert t.task == "classification" and t.imbalanced
    assert t.minority_rate == pytest.approx(0.05)


def test_explicit_task_overrides_inference():
    assert profile_target(pd.Series([0, 1, 2] * 30), task="regression").task == "regression"
    with pytest.raises(MekikiError):
        profile_target(pd.Series([0, 1]), task="ranking")


# --- Stage 1: diagnosis table -----------------------------------------------

def test_diagnosis_is_readable_and_json_serialisable():
    df = make_listings()
    dg = run_diagnosis(df, "price", unit="10k JPY", screen_text=False)
    assert isinstance(dg, Diagnosis)
    s = str(dg)
    assert "log transform" in s and "id-like" in s and "EvidencePredictor on all rows" in s
    payload = json.dumps(dg.as_dict(), ensure_ascii=False)   # fails if NaN leaks in
    assert "listing_id" in payload
    assert dg.cost_estimate["n_rows"] == len(df)


def test_duplicates_are_counted_without_id_and_date_columns():
    df = make_listings(n=120)
    dup = pd.concat([df, df.iloc[:30].assign(listing_id=lambda d: d.listing_id + 10_000,
                                             url=lambda d: d.url + "?x")],
                    ignore_index=True)
    dg = run_diagnosis(dup, "price", screen_text=False)
    assert dg.duplicates is not None and dg.duplicates.n_duplicate_rows == 30
    assert "listing_id" not in dg.duplicates.columns
    assert any("identical content" in w for w in dg.warnings)


def test_rows_identical_only_in_text_are_counted_separately_as_suspected_relisting():
    df = make_listings(n=120)
    dg = run_diagnosis(df, "price", screen_text=False)
    # description has only 3 distinct values, so comparing on text alone makes most rows duplicates
    assert dg.duplicates_by_text is not None
    assert dg.duplicates_by_text.n_duplicate_rows > dg.duplicates.n_duplicate_rows
    assert any("multiple times" in w for w in dg.warnings)


def test_missing_target_column_raises():
    with pytest.raises(MekikiError):
        run_diagnosis(make_listings(), "no_such_column", screen_text=False)


def test_screening_is_skipped_without_structured_columns():
    df = make_listings()[["description", "price"]]
    dg = run_diagnosis(df, "price", n_splits=2)
    assert not dg.screening
    assert any("cannot be measured" in w for w in dg.warnings)


def test_screening_applies_the_recommended_target_preprocessing_first():
    """Outliers dominate the MAE and hide the contribution, so measure after clipping and log."""
    df = make_listings()
    dg = run_diagnosis(df, "price", n_splits=2, sample=None)
    assert dg.target.suggest_log and dg.target.suggest_clip
    assert "log1p" in dg.screening_note and "top 0.5%" in dg.screening_note
    assert "how_measured" in dg.as_dict()["text_explanatory_power"]
    assert "Note:" in str(dg)


def test_screening_is_skipped_and_diagnosis_continues_on_column_names_the_tree_rejects():
    df = make_listings().rename(columns={"age": "age:years"})   # a character LightGBM rejects
    dg = run_diagnosis(df, "price", n_splits=2, sample=None)
    assert not dg.screening
    assert any("Skipped screening" in w for w in dg.warnings)


def test_screening_measures_only_the_chosen_text_column():
    df = make_listings()
    dg = run_diagnosis(df, "price", n_splits=2, sample=None)
    assert set(dg.screening) == {"description"}       # model is categorical, so not measured
    assert dg.screening["description"].metric == "MAE"


# --- Rule-only recommendation ----------------------------------------------

def test_spec_and_to_code_are_returned_without_llm():
    df = make_listings()
    rec = diagnose(df, "price", unit="10k JPY", llm=False, screen_text=False)
    assert isinstance(rec, Recommendation) and rec.source == "rule"
    assert isinstance(rec.spec, ColumnSpec) and isinstance(rec.domain, Domain)
    assert set(rec.spec.numeric) == {"age", "km"}
    assert rec.spec.boolean == []
    assert "has_warranty" in rec.spec.categorical
    assert "region" in rec.spec.categorical and "model" in rec.spec.categorical
    assert rec.spec.long_text == "description"
    assert rec.target_transform == "log1p" and rec.clip_outliers
    assert set(rec.dedup_ignore) >= {"listing_id", "url", "posted_at", "county"}
    code = rec.to_code()
    assert "EvidenceRegressor(" in code and "np.log1p" in code and "quantile(0.995)" in code
    compile(code, "<to_code>", "exec")                   # syntactically valid
    s = str(rec)
    assert "Recommendation" in s and "rules only" in s and "Reasons" in s


def test_feature_b_is_not_recommended_without_text():
    df = make_listings().drop(columns=["model", "description"])
    rec = diagnose(df, "price", llm=False, screen_text=False)
    assert rec.use_evidence is False and rec.escalate_rate == 0.0
    assert "escalate_rate=0.0" in rec.to_code()


def test_classification_has_no_log_transform_or_amount_masking():
    df = make_listings()
    df["Churn"] = np.where(df["age"] > 7, "Yes", "No")
    rec = diagnose(df.drop(columns=["price"]), "Churn", llm=False, screen_text=False)
    assert rec.task == "classification" and rec.target_transform is None
    assert "log1p" not in rec.to_code() and "EvidenceClassifier(" in rec.to_code()


def test_invalid_llm_argument_raises():
    with pytest.raises(MekikiError):
        diagnose(make_listings(), "price", llm="yes")


def test_llm_required_without_key_raises():
    with pytest.raises(MekikiError):
        diagnose(make_listings(), "price", llm=True, screen_text=False,
                 client=ClaudeClient(api_key="", cache_dir=None))


def test_without_key_falls_back_to_rules_with_a_warning():
    rec = diagnose(make_listings(), "price", screen_text=False,
                   client=ClaudeClient(api_key="", cache_dir=None))
    assert rec.source == "rule"
    assert any("No API key" in w for w in rec.warnings)


# --- Stage 2: LLM -------------------------------------------------------------

def test_llm_receives_only_the_diagnosis_table_and_draft_not_the_data():
    df = make_listings()
    client = FakeClient(answer=GOOD_ANSWER)
    diagnose(df, "price", unit="10k JPY", client=client, screen_text=False)
    assert len(client.prompts) == 1
    user = client.prompts[0]
    assert "Diagnosis table" in user and "Draft" in user and "listing_id" in user
    # rows themselves are absent (url appears as a column name, but per-row values only as 3 samples)
    assert user.count("https://example.com/l/") <= 3
    schema = client.schemas[0]
    assert "reasons" in schema["required"] and "dedup_ignore" in schema["properties"]


def test_llm_answer_becomes_spec_domain_and_reasons():
    df = make_listings()
    rec = diagnose(df, "price", unit="10k JPY", client=FakeClient(answer=GOOD_ANSWER),
                   screen_text=False)
    assert rec.source == "llm"
    assert rec.spec.numeric == ["age", "km"] and rec.spec.text == "model"
    # a Yes/No string column marked boolean by the LLM is moved to categorical with a warning
    assert rec.spec.boolean == [] and "has_warranty" in rec.spec.categorical
    assert any("moved to categorical" in w for w in rec.warnings)
    assert rec.domain.role == "a used-car appraiser" and rec.domain.name_of("price") == "price"
    assert rec.domain.hints == ["Weigh accident history and repairs mentioned in the free text heavily"]
    assert rec.dedup_ignore == ["listing_id", "url", "posted_at", "region"]
    assert rec.typed_columns[0]["values"] == ["standard", "sport", "luxury"]
    assert [k["name"] for k in rec.knowledge_columns] == ["body_style", "new_price"]
    assert rec.knowledge_columns[0]["keys"] == ["model"]
    assert rec.escalate_rate == 0.25 and rec.use_evidence
    assert rec.reasons["dedup"].startswith("region")
    assert rec.cost["n_calls"] == 1
    code = rec.to_code()
    assert "SemanticEncoder(source='description'" in code and "'region'" in code
    assert "KnowledgeEncoder(keys=['model'], attribute='body style', type='category'" in code
    assert "values=['sedan', 'SUV', 'minivan']" in code and "name='body_style'" in code
    assert "type='numeric', unit='10k JPY'" in code and "target='price'" in code
    assert ("from mekiki import EvidenceRegressor, Domain, check_duplicates,"
            " SemanticEncoder, KnowledgeEncoder") in code
    compile(code, "<to_code>", "exec")
    s = str(rec)
    assert "built by the LLM" in s and "a used-car appraiser" in s and "grade" in s
    assert "Knowledge column candidates" in s and "body style -> ['sedan', 'SUV', 'minivan']" in s
    assert "approximate price when new -> 10k JPY" in s


def test_knowledge_column_candidates_are_validated_against_the_table():
    bad = dict(GOOD_ANSWER, knowledge_columns=[
        # keyed by the free-text body: not usable as a key
        {"name": "x1", "keys": ["description"], "attribute": "a", "type": "category",
         "values": ["p", "q"], "unit": "", "why": ""},
        # keyed by a column absent from the table
        {"name": "x2", "keys": ["maker"], "attribute": "a", "type": "category",
         "values": ["p", "q"], "unit": "", "why": ""},
        # keyed by the target
        {"name": "x3", "keys": ["price"], "attribute": "a", "type": "category",
         "values": ["p", "q"], "unit": "", "why": ""},
        # category without values
        {"name": "x4", "keys": ["model"], "attribute": "a", "type": "category",
         "values": [], "unit": "", "why": ""},
        # binary with three values becomes category; numeric drops its values
        {"name": "ok1", "keys": ["model", "region"], "attribute": "b", "type": "binary",
         "values": ["p", "q", "r"], "unit": "", "why": ""},
        {"name": "ok2", "keys": ["model"], "attribute": "c", "type": "numeric",
         "values": ["1", "2"], "unit": "kg", "why": ""},
    ])
    rec = diagnose(make_listings(), "price", client=FakeClient(answer=bad), screen_text=False)
    assert [k["name"] for k in rec.knowledge_columns] == ["ok1", "ok2"]
    assert rec.knowledge_columns[0]["type"] == "category"
    assert rec.knowledge_columns[0]["keys"] == ["model", "region"]
    assert rec.knowledge_columns[1]["type"] == "numeric" and rec.knowledge_columns[1]["values"] == []
    w = [w for w in rec.warnings if "Knowledge column candidates" in w]
    assert len(w) == 1 and all(x in w[0] for x in ("x1", "x2", "x3", "x4"))
    code = rec.to_code()
    assert "values=" not in code.split("attribute='c'")[1].split("\n")[0]
    compile(code, "<to_code>", "exec")


def test_rule_only_recommendation_proposes_no_knowledge_columns():
    rec = diagnose(make_listings(), "price", llm=False, screen_text=False)
    assert rec.knowledge_columns == []
    assert "KnowledgeEncoder" not in rec.to_code() and "Knowledge column" not in str(rec)


def test_role_returned_as_a_sentence_is_reduced_to_the_noun_phrase():
    """The prompt fills "You are {role}", so a full sentence would double up."""
    ans = dict(GOOD_ANSWER, role="You are a used-car appraiser.")
    rec = diagnose(make_listings(), "price", client=FakeClient(answer=ans), screen_text=False)
    assert rec.domain.role == "a used-car appraiser"
    rec = diagnose(make_listings(), "price", client=FakeClient(answer=dict(GOOD_ANSWER, role="  ")),
                   screen_text=False)
    assert rec.domain.role == Domain.role                  # empty falls back to the default


def test_unknown_column_names_from_llm_are_dropped_with_a_warning():
    bad = dict(GOOD_ANSWER, numeric=["age", "no_such_column"], text="other_column", escalate_rate=7)
    rec = diagnose(make_listings(), "price", client=FakeClient(answer=bad), screen_text=False)
    assert rec.spec.numeric == ["age"] and rec.spec.text is None
    assert rec.escalate_rate == 1.0                       # clamped to 0-1
    assert any("absent from the diagnosis table" in w for w in rec.warnings)


def test_column_placed_in_two_slots_keeps_the_first():
    dup = dict(GOOD_ANSWER, categorical=["region", "age"], long_text="model")
    rec = diagnose(make_listings(), "price", client=FakeClient(answer=dup), screen_text=False)
    # age stays numeric; has_warranty moves from boolean
    assert rec.spec.categorical == ["region", "has_warranty"]
    assert rec.spec.long_text is None                     # model stays as text


def test_missing_reasons_from_llm_triggers_a_warning():
    rec = diagnose(make_listings(), "price", client=FakeClient(answer=dict(GOOD_ANSWER, reasons=[])),
                   screen_text=False)
    assert any("no reasons" in w for w in rec.warnings)


def test_llm_failure_falls_back_to_rule_recommendation():
    rec = diagnose(make_listings(), "price", client=FakeClient(fail=True), screen_text=False)
    assert rec.source == "rule"
    assert any("failed" in w for w in rec.warnings)
    assert rec.cost["errors"] == 1


def test_classification_maps_class_names_back_to_native_values():
    df = make_listings()
    df["speed"] = np.tile([0, 1, 2, 3, 4], len(df) // 5)
    ans = dict(GOOD_ANSWER, task="classification", target_transform="log1p", clip_outliers=True,
               class_names=[{"label": "0", "meaning": "same day"}, {"label": "9", "meaning": "never"}])
    rec = diagnose(df.drop(columns=["price"]), "speed", client=FakeClient(answer=ans),
                   screen_text=False)
    assert rec.classify
    assert rec.domain.class_names == {0: "same day"}       # "9" is dropped; 0 is back to int
    assert rec.target_transform is None and not rec.clip_outliers


def test_prompt_contains_the_diagnosis_numbers():
    df = make_listings()
    dg = run_diagnosis(df, "price", unit="10k JPY", screen_text=False)
    from mekiki.diagnose import _rule_recommendation
    draft = _rule_recommendation(dg, "price", "10k JPY", None)
    user = build_user_prompt(dg, draft, "price", "10k JPY", None)
    assert "mean_over_median" in user and "10k JPY" in user and '"task": "regression"' in user


def test_camel_case_identifier_names():
    from mekiki.diagnose import _id_like_name
    for name in ["RescuerID", "PetId", "IDLink", "customerID", "user_id", "Region2ID"]:
        assert _id_like_name(name), name
    for name in ["Paid", "Idle", "Identity", "Video", "Width"]:
        assert not _id_like_name(name), name
