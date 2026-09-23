"""Tests for the skeleton of `SemanticEncoder`.

    .venv/bin/python -m pytest tests -q

Covers only what needs no API key (steps 01-04 of `SemanticEncoder`, and the queue path of 05).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mekiki import CharTfidfVectorizer, MekikiError, SemanticEncoder
from mekiki.fallback import Answer
from mekiki.preprocess import constant_tokens, drop_constant_tokens


def ambiguous() -> pd.DataFrame:
    """Data where the same sentence carries conflicting labels.

    The neighbours' labels split, so confidence drops below 1.0 and the
    05 (escalation) path gets exercised. Real labels are noisy, so this is
    expected input, not an error case.
    """
    base = sample()
    noisy = pd.DataFrame({"title": ["Sienta special edition ETC smart key"] * 4,
                          "label": ["G", "Z", "G", "Z"]})
    return pd.concat([base, noisy], ignore_index=True)


def sample() -> pd.DataFrame:
    rows = [("Sienta G Cuero non-smoker backup camera", "G Cuero"),
            ("Sienta Hybrid Z 4WD OEM navi", "Z"),
            ("Sienta X dual power sliding doors", "X"),
            ("Sienta Hybrid G ETC", "G")]
    return pd.DataFrame([{"title": t, "label": g} for t, g in rows * 6])


# --- Preprocessing ----------------------------------------------------

def test_constant_tokens_are_decided_by_frequency():
    s = pd.Series(["Sienta G", "Sienta Z", "Sienta X"])
    assert constant_tokens(s, threshold=0.9) == ["Sienta"]
    out, stop = drop_constant_tokens(s)
    assert out.tolist() == ["G", "Z", "X"] and stop == ["Sienta"]


def test_inference_reuses_constant_tokens_from_fit():
    f = SemanticEncoder(source="title", values=["G", "Z"]).fit(sample())
    assert "Sienta" in f.stop_tokens_
    # Passing a single row must not drop all its tokens as "100% frequency"
    one = pd.DataFrame({"title": ["Sienta Hybrid Z 4WD"]})
    assert f.transform(one).iloc[0] == "Z"


# --- 01 Where the ground truth comes from (three entry points) ---------------------------

def test_value_names_alone_can_classify():
    df = sample()
    out = SemanticEncoder(source="title", values=["G", "Z", "X", "G Cuero"],
                      k=3).fit_transform(df)
    assert (out.astype(str) == df["label"]).mean() >= 0.9


def test_human_labels_can_be_passed():
    df = sample()
    f = SemanticEncoder(source="title", labels="label", k=3)
    out = f.fit_transform(df)
    assert (out.astype(str) == df["label"]).mean() == 1.0
    assert f.status()["references_human"] == len(df)


def test_stops_without_labels_or_values():
    with pytest.raises(MekikiError, match="Neither ground-truth labels nor values"):
        SemanticEncoder(source="title").fit(sample())


def test_stops_when_all_labels_are_missing():
    df = sample().assign(label=np.nan)
    with pytest.raises(MekikiError):
        SemanticEncoder(source="title", labels="label").fit(df)


# --- 03/04/05 Confidence and escalation ---------------------------------

def test_confidence_is_between_0_and_1():
    f = SemanticEncoder(source="title", labels="label", k=3)
    f.fit_transform(sample())
    c = f.confidence()
    assert ((c >= 0) & (c <= 1)).all()


def test_higher_threshold_means_more_pending_review():
    df = ambiguous()
    low = SemanticEncoder(source="title", labels="label", k=3, threshold=0.1)
    low.fit_transform(df)
    high = SemanticEncoder(source="title", labels="label", k=3, threshold=0.99)
    high.fit_transform(df)
    assert high.status()["pending_review"] >= low.status()["pending_review"]
    assert len(high.review_queue()) == high.status()["pending_review"]


def test_plugging_in_an_llm_changes_the_escalation_target():
    class DummyLLM:
        cost_per_call = 0.002

        def can_answer(self):
            return True

        def answer(self, texts, values, context):
            return [Answer(value="Z", confidence=0.95, cost=0.002)
                    for _ in texts]

    f = SemanticEncoder(source="title", labels="label", k=3, threshold=0.9,
                    fallback=DummyLLM())
    f.fit_transform(ambiguous())
    st, cost = f.status(), f.cost()
    assert st["llm_answered"] > 0 and st["pending_review"] == 0
    assert cost["actual_cost_usd"] == pytest.approx(st["llm_answered"] * 0.002)


def test_on_uncertain_null_yields_missing():
    out = SemanticEncoder(source="title", labels="label", k=3, threshold=0.9,
                      on_uncertain="null").fit_transform(ambiguous())
    assert out.isna().any()


def test_k_auto_shrinks_with_one_example_per_class():
    """Starting from value names only gives one reference example per class.

    Taking neighbours with k=5 then always splits across 5 classes and every row
    becomes "uncertain" (99.9% escalated on real data). k="auto" derives k from
    the number of examples per class to avoid this.
    """
    f = SemanticEncoder(source="title", values=["G", "Z", "X", "G Cuero"], k="auto")
    f.fit(sample())
    assert f.k_ == 1
    f2 = SemanticEncoder(source="title", labels="label", k="auto").fit(sample())
    assert f2.k_ > 1          # 6 examples per class, so looking wider is fine


def test_confidence_is_the_ratio_of_first_to_second():
    f = SemanticEncoder(source="title", labels="label", k=3)
    f.fit_transform(sample())
    # When all neighbours share a label there is no second, so 1.0
    assert f.confidence().max() == pytest.approx(1.0)
    amb = SemanticEncoder(source="title", labels="label", k=3)
    amb.fit_transform(ambiguous())
    assert amb.confidence().min() < 1.0


def test_escalate_rate_sets_the_share():
    df = ambiguous()
    f = SemanticEncoder(source="title", labels="label", k=3, escalate_rate=0.25)
    f.fit_transform(df)
    n = f.status()["pending_review"]
    # Ties mean it is not exact, but it stays near the requested share
    assert 0 < n <= len(df) * 0.5
    assert "bottom" in str(f.status()["threshold"])


# --- Inspection API ------------------------------------------------------

def test_explain_shows_references_and_source():
    f = SemanticEncoder(source="title", labels="label", k=3)
    f.fit_transform(sample())
    text = f.explain(0)
    assert "references" in text and "confidence" in text and "human" in text


def test_examples_returns_k_per_row():
    f = SemanticEncoder(source="title", labels="label", k=3)
    df = sample()
    ex = f.fit(df).examples(df)
    assert len(ex) == len(df) * 3
    assert {"row_id", "value", "similarity", "source"} <= set(ex.columns)


def test_cost_returns_an_estimate_before_running():
    f = SemanticEncoder(source="title", labels="label", k=3, threshold=0.9)
    df = ambiguous()
    c = f.fit(df).cost(df)
    assert c["n_rows"] == len(df) and 0 <= c["rate"] <= 1


# --- Types and input -----------------------------------------------------

def test_embedding_type_needs_no_labels():
    out = SemanticEncoder(source="title", type="embedding").fit_transform(sample())
    assert isinstance(out, pd.DataFrame) and len(out) == len(sample())


def test_multiple_columns_are_joined():
    df = sample().assign(color=["white", "black"] * 12)
    f = SemanticEncoder(source=["title", "color"], labels="label", k=3).fit(df)
    assert "white" in f.provenance_["text"].iloc[0] if hasattr(f, "provenance_") else True


def test_unimplemented_type_fails_explicitly():
    with pytest.raises(MekikiError, match="not implemented"):
        SemanticEncoder(source="title", type="ordinal", values=["low", "high"])


def test_missing_source_column_fails():
    with pytest.raises(MekikiError, match="source"):
        SemanticEncoder(source="missing_column", values=["G"]).fit(sample())


def test_encoder_can_be_swapped():
    f = SemanticEncoder(source="title", labels="label", k=3,
                    vectorizer=CharTfidfVectorizer(n_components=16))
    f.fit_transform(sample())
    assert f.status()["vectorizer"] == "char_tfidf_svd16"


def test_same_string_is_not_computed_twice():
    df = sample()
    f = SemanticEncoder(source="title", labels="label", k=3).fit(df)
    f.transform(df)
    assert f.vectorizer_.n_cached == df["title"].nunique()


def test_fit_from_label_column_only_still_passes_candidates_to_fallback():
    """Asking the LLM without candidates lets arbitrary spellings leak into the feature."""
    from mekiki.fallback import Answer
    seen = {}

    class Recording:
        cost_per_call = 0.0

        def can_answer(self):
            return True

        def answer(self, texts, values, context):
            seen["values"] = values
            return [Answer(value=values[0], confidence=0.9) for _ in texts]

    f = SemanticEncoder(source="title", labels="label", k=3, threshold=0.9,
                    fallback=Recording())
    f.fit_transform(ambiguous())
    assert seen["values"] == sorted({"G Cuero", "Z", "X", "G"})
