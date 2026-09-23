"""Tests for knowledge columns (`mekiki.knowledge.KnowledgeEncoder`).

The LLM is replaced by `FakeClient`, so **everything passes without an API key**. FakeClient
reads the key values out of the user prompt it receives and answers from a prepared table.
It also returns prepared candidates for the proposal query (automatic mode). What is tested is
the wiring: "the same key value is asked only once", "outside the candidates / unknown / out of
range become missing", batching, the shape of the provenance, cross-checking and scoring --
not the quality of the LLM's knowledge.

Run: .venv/bin/python -m pytest tests/test_knowledge.py -q
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from mekiki import Domain, KnowledgeEncoder, MekikiError
from mekiki.llm import ClaudeClient, LLMAnswer

BODY = ["sedan", "SUV", "pickup", "hatchback"]


class FakeClient(ClaudeClient):
    """Makes no HTTP calls; looks up the key values in the prompt in a table and answers.

    table: key value -> answer dict (everything but key). Key values not in the table are
        answered with known=false.
    tables: attribute (a string contained in the prompt's "Attribute to answer") -> table.
        For the multi-column case.
    proposal: the list of candidates returned to the proposal query (automatic mode).
    `echo=False` deliberately corrupts the copied key.
    """

    def __init__(self, table: dict | None = None, proposal: list | None = None,
                 fail: bool = False, echo: bool = True, tables: dict | None = None, **kw):
        kw.setdefault("cache_dir", None)
        kw.setdefault("api_key", "test-key")
        super().__init__(**kw)
        self.table = table or {}
        self.tables = tables or {}
        self.proposal = proposal or []
        self.fail = fail
        self.echo = echo
        self.prompts: list[str] = []
        self.batches: list[list[str]] = []
        self.proposal_prompts: list[str] = []

    def ask(self, system, user, schema) -> LLMAnswer:
        with self._lock:
            self.usage.calls += 1
        if "candidates" in schema.get("properties", {}):
            self.proposal_prompts.append(user)
            self.usage.cost += 0.05
            return LLMAnswer(data={"candidates": self.proposal}, cost=0.05)
        self.prompts.append(user)
        keys = re.findall(r"^\d+\. (.+)$", user, flags=re.M)
        self.batches.append(keys)
        if self.fail:
            self.usage.errors += 1
            return LLMAnswer(data={}, error="deliberate failure")
        table = self.table
        for attr, t in self.tables.items():
            if attr in user:
                table = t
        answers = []
        for k in keys:
            a = table.get(k)
            if a is None:
                answers.append({"key": k, "known": False, "value": "", "confidence": 0.0,
                                "reason": "unknown"})
            else:
                answers.append({"key": k if self.echo else k + "?", **a})
        self.usage.cost += 0.01
        return LLMAnswer(data={"answers": answers}, input_tokens=300, output_tokens=200, cost=0.01)


def good(value, conf=0.9, reason="well-known model"):
    return {"known": True, "value": value, "confidence": conf, "reason": reason}


def cars(n_each: int = 10) -> pd.DataFrame:
    rows = []
    for m, mdl, body in [("ford", "f-150", "pickup"), ("toyota", "camry", "sedan"),
                         ("honda", "cr-v", "SUV")]:
        rows += [{"manufacturer": m, "model": mdl, "type": body}] * n_each
    return pd.DataFrame(rows)


TABLE = {"ford | f-150": good("pickup"), "toyota | camry": good("sedan"),
         "honda | cr-v": good("SUV")}


def col(client, **kw):
    """A fully specified (manual) knowledge column. The column name is body_type."""
    kw.setdefault("keys", ["manufacturer", "model"])
    kw.setdefault("attribute", "body style")
    kw.setdefault("type", "category")
    kw.setdefault("values", BODY)
    kw.setdefault("cache_dir", None)
    kw.setdefault("name", "body_type")
    return KnowledgeEncoder(client=client, **kw)


def first(out: pd.DataFrame) -> pd.Series:
    assert isinstance(out, pd.DataFrame)
    return out.iloc[:, 0]


# --- A table in, a DataFrame out ----------------------------------------------

def test_returns_a_dataframe_even_with_one_column():
    out = col(FakeClient(TABLE)).fit_transform(cars(n_each=2))
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["body_type"]
    assert len(out) == 6


# --- The unit is the key value -------------------------------------------------

def test_same_key_value_is_asked_only_once():
    c = FakeClient(TABLE)
    df = cars(n_each=40)                       # 120 rows, 3 distinct key values
    out = first(col(c).fit_transform(df))
    assert c.usage.calls == 1
    assert c.batches == [["ford | f-150", "honda | cr-v", "toyota | camry"]]   # sorted
    assert list(out.unique()) == ["pickup", "sedan", "SUV"]
    assert str(out.dtype) == "category"
    assert list(out.cat.categories) == BODY


def test_key_values_are_normalized_before_batching():
    c = FakeClient(TABLE)
    df = pd.DataFrame({"manufacturer": ["Ford", " FORD", "ford"],
                       "model": ["F-150", "f-150 ", "F-150"]})
    out = first(col(c).fit_transform(df))
    assert c.batches == [["ford | f-150"]]
    assert out.tolist() == ["pickup"] * 3


def test_batch_size_batches_keys_and_answers_map_back_to_rows():
    table = {f"m{i:02d} | x": good("sedan" if i % 2 else "SUV") for i in range(45)}
    c = FakeClient(table)
    df = pd.DataFrame({"manufacturer": [f"m{i:02d}" for i in range(45)] * 2, "model": "x"})
    out = first(col(c, batch_size=20).fit_transform(df))
    assert c.usage.calls == 3
    assert [len(b) for b in c.batches] == [20, 20, 5]
    assert out.iloc[1] == "sedan" and out.iloc[2] == "SUV" and out.iloc[46] == "sedan"


def test_key_values_below_min_count_are_not_asked_and_become_skipped():
    c = FakeClient(TABLE)
    df = pd.concat([cars(n_each=5), pd.DataFrame([{"manufacturer": "rare", "model": "one", "type": None}])],
                   ignore_index=True)
    k = col(c, min_count=2)
    out = first(k.fit_transform(df))
    assert "rare | one" not in sum(c.batches, [])
    assert pd.isna(out.iloc[-1])
    assert k.provenance_["source"].iloc[-1] == "skipped"
    assert k.status()["per_column"][0]["skipped"] == 1


def test_rows_with_missing_key_are_not_asked_and_become_missing():
    c = FakeClient(TABLE)
    df = cars(n_each=2)
    df.loc[0, "model"] = None
    df.loc[1, "model"] = "  "
    k = col(c)
    out = first(k.fit_transform(df))
    assert pd.isna(out.iloc[0]) and pd.isna(out.iloc[1])
    assert k.provenance_["source"].iloc[0] == "skipped"
    assert "missing" in k.explain(0)


# --- Guards against hallucination ----------------------------------------------

def test_answer_outside_candidates_is_rejected_and_missing():
    c = FakeClient({**TABLE, "ford | f-150": good("lorry")})
    k = col(c)
    out = first(k.fit_transform(cars(n_each=2)))
    assert out.isna().sum() == 2
    row = k.answers_["body_type"].set_index("key").loc["ford | f-150"]
    assert row["source"] == "rejected" and "not among the candidates" in row["reason"]
    assert k.review_queue()["key"].tolist() == ["ford | f-150"]


def test_candidate_spelling_variants_are_normalized():
    c = FakeClient({**TABLE, "honda | cr-v": good(" suv ")})
    out = first(col(c).fit_transform(cars(n_each=1)))
    assert out.iloc[2] == "SUV"


def test_unknown_key_values_become_missing_and_land_in_review_queue():
    c = FakeClient({k: v for k, v in TABLE.items() if k != "toyota | camry"})
    k = col(c)
    out = first(k.fit_transform(cars(n_each=3)))
    assert out.isna().sum() == 3
    assert (k.provenance_["source"] == "unknown").sum() == 3
    assert k.status()["per_column"][0]["unknown"] == 1
    assert k.review_queue()["source"].tolist() == ["unknown"]


def test_confidence_below_threshold_is_rejected():
    c = FakeClient({**TABLE, "honda | cr-v": good("SUV", conf=0.4)})
    k = col(c, threshold=0.7)
    out = first(k.fit_transform(cars(n_each=1)))
    assert pd.isna(out.iloc[2])
    assert k.provenance_["source"].iloc[2] == "rejected"
    assert k.provenance_["confidence"].iloc[2] == pytest.approx(0.4)


def test_default_threshold_depends_on_type_and_changing_it_does_not_reask(tmp_path):
    assert col(FakeClient(TABLE)).fit(cars(n_each=1)).lookups_[0].threshold == 0.7
    num = col(FakeClient({}), type="numeric", values=None, range=(0, 1)).fit(cars(n_each=1))
    assert num.lookups_[0].threshold == 0.4

    c1 = FakeClient({**TABLE, "honda | cr-v": good("SUV", conf=0.5)})
    k1 = col(c1, threshold=0.7, cache_dir=tmp_path)
    assert pd.isna(first(k1.fit_transform(cars(n_each=1))).iloc[2])
    c2 = FakeClient({})                                       # knows nothing, whatever is asked
    k2 = col(c2, threshold=0.4, cache_dir=tmp_path)
    out = first(k2.fit_transform(cars(n_each=1)))
    assert c2.usage.calls == 0                                # served from the parquet
    assert out.iloc[2] == "SUV"                               # accepted now that the threshold is lower


def test_numeric_rejects_out_of_range_and_returns_float():
    table = {"ford | f-150": good(35000), "toyota | camry": good(2_000_000),
             "honda | cr-v": good("28,500")}
    c = FakeClient(table)
    k = col(c, attribute="approximate price when new", type="numeric", values=None, unit="USD",
            range=(3000, 500000))
    out = first(k.fit_transform(cars(n_each=1)))
    assert str(out.dtype) == "float64"
    assert out.iloc[0] == 35000.0
    assert np.isnan(out.iloc[1]) and "out of range" in k.provenance_["reason"].iloc[1]
    assert out.iloc[2] == 28500.0            # numeric-looking strings are parsed


def test_answers_with_corrupted_key_are_rejected():
    c = FakeClient(TABLE, echo=False)
    k = col(c)
    out = first(k.fit_transform(cars(n_each=1)))
    assert out.isna().all()
    ans = k.answers_["body_type"]
    assert (ans["source"] == "rejected").all()
    assert "key missing" in ans["reason"].iloc[0]


def test_failed_calls_are_not_saved_and_are_asked_again(tmp_path):
    c = FakeClient(TABLE, fail=True)
    k = col(c, cache_dir=tmp_path)
    out = first(k.fit_transform(cars(n_each=1)))
    assert out.isna().all()
    assert k.status()["per_column"][0]["errors"] == 3
    assert not list(tmp_path.glob("*.parquet"))


def test_check_against_cross_checks_an_existing_column():
    c = FakeClient({**TABLE, "honda | cr-v": good("sedan")})    # one deliberately wrong
    k = col(c, check_against="type")
    df = cars(n_each=4)
    df.loc[0, "type"] = None                                    # missing rows are not in the denominator
    k.fit_transform(df)
    agree = k.status()["per_column"][0]["agreement"]
    assert agree["column"] == "type" and agree["n_compared"] == 11
    assert agree["agreement"] == pytest.approx(7 / 11, abs=1e-4)


# --- Persistence ---------------------------------------------------------------

def test_second_run_asks_only_new_key_values(tmp_path):
    c1 = FakeClient(TABLE)
    col(c1, cache_dir=tmp_path).fit_transform(cars(n_each=1))
    assert c1.usage.calls == 1
    assert len(list(tmp_path.glob("mekiki_knowledge_*.parquet"))) == 1

    table2 = {**TABLE, "mazda | cx-5": good("SUV")}
    c2 = FakeClient(table2)
    extra = pd.DataFrame([{"manufacturer": "mazda", "model": "cx-5", "type": "SUV"}])
    df = pd.concat([cars(n_each=1), extra], ignore_index=True)
    out = first(col(c2, cache_dir=tmp_path).fit_transform(df))
    assert c2.batches == [["mazda | cx-5"]]
    assert out.tolist() == ["pickup", "sedan", "SUV", "SUV"]


def test_different_attribute_uses_a_different_cache(tmp_path):
    col(FakeClient(TABLE), cache_dir=tmp_path).fit_transform(cars(n_each=1))
    c = FakeClient(TABLE)
    col(c, attribute="typical drivetrain", values=["fwd", "rwd", "4wd"],
        cache_dir=tmp_path).fit_transform(cars(n_each=1))
    assert c.usage.calls == 1
    assert len(list(tmp_path.glob("mekiki_knowledge_*.parquet"))) == 2


# --- Inspection API ------------------------------------------------------------

def test_shape_of_provenance_and_explain():
    c = FakeClient(TABLE)
    k = col(c)
    k.fit_transform(cars(n_each=2))
    prov = k.provenance_
    assert list(prov.columns) == ["column", "value", "confidence", "source", "cost", "key", "reason"]
    assert len(prov) == 6
    # prorated over rows, sums back to the actual cost
    assert prov["cost"].sum() == pytest.approx(0.01)
    s = k.explain(0)
    for word in ("body_type", "pickup", "0.900", "llm", "ford | f-150", "well-known model", "cost"):
        assert word in s
    st = k.status()
    assert st["n_columns"] == 1 and st["returned_columns"] == ["body_type"]
    assert st["per_column"][0]["accepted_llm"] == 3 and st["per_column"][0]["n_rows_filled"] == 6
    assert st["actual_cost_usd"] == pytest.approx(0.01)
    assert k.columns()[0]["specified"] == "human"


def test_cost_can_be_estimated_after_fit_alone():
    c = FakeClient(TABLE)
    k = col(c, batch_size=2, min_count=2)
    df = pd.concat([cars(n_each=3), pd.DataFrame([{"manufacturer": "rare", "model": "one", "type": None}])],
                   ignore_index=True)
    est = k.fit(df).cost()
    assert c.usage.calls == 0
    assert est["n_columns"] == 1 and est["n_to_ask"] == 3 and est["n_requests"] == 2
    per = est["per_column"][0]
    assert per["n_rows"] == 10 and per["n_key_values"] == 4 and per["skipped_below_min_count"] == 1
    assert est["estimated_total"] == pytest.approx(3 * est["cost_per_key_usd"], abs=1e-5)
    assert est["cost_per_key_usd"] > 0


def test_system_prompt_uses_domain_role_and_does_not_hardcode_a_domain():
    c = FakeClient(TABLE)
    k = col(c, domain=Domain(role="a wine judge", subject="wine")).fit(cars(n_each=1))
    assert "You are a wine judge" in k.lookups_[0].system()
    assert "about wine" in k.lookups_[0].system()
    generic = col(c).fit(cars(n_each=1)).lookups_[0].system()
    assert not re.search(r"\b(car|cars|used)\b", generic)


# --- Automatic mode (omitted arguments are decided by the LLM) -----------------

PROPOSAL = [
    {"name": "body_type", "keys": ["manufacturer", "model"], "attribute": "body style",
     "type": "category", "values": BODY, "unit": "", "range_low": 0, "range_high": 0,
     "why": "body style separates price bands"},
    {"name": "msrp", "keys": ["manufacturer", "model"], "attribute": "approximate price when new",
     "type": "numeric", "values": [], "unit": "USD", "range_low": 3000, "range_high": 500000,
     "why": "the new-car price caps the used price"},
    {"name": "bad", "keys": ["url"], "attribute": "something", "type": "category",
     "values": ["a", "b"], "unit": "", "range_low": 0, "range_high": 0,
     "why": "uses a column that does not exist as the key"},
]

MSRP = {"ford | f-150": good(35000, conf=0.6), "toyota | camry": good(25000, conf=0.6),
        "honda | cr-v": good(30000, conf=0.6)}


def test_with_nothing_specified_llm_proposes_and_unusable_candidates_are_dropped():
    c = FakeClient(tables={"body style": TABLE, "price when new": MSRP}, proposal=PROPOSAL)
    k = KnowledgeEncoder(client=c, cache_dir=None)
    out = k.fit_transform(cars(n_each=2))
    assert list(out.columns) == ["body_type", "msrp"]
    assert out["body_type"].tolist()[:3] == ["pickup", "pickup", "sedan"]
    assert out["msrp"].iloc[0] == 35000.0
    assert c.proposal_prompts and "Columns" in c.proposal_prompts[0]
    assert any("bad" in w for w in k.warnings_)          # url is not a column in the data
    assert k.columns()[0]["specified"] == "llm" and "price band" in k.columns()[0]["reason"]
    assert k.status()["proposal_cost_usd"] == pytest.approx(0.05)
    assert "proposed because" in k.explain(0, "body_type")


def test_with_only_attribute_specified_exactly_one_proposal_is_requested():
    c = FakeClient(TABLE, proposal=PROPOSAL[:1])
    k = KnowledgeEncoder(attribute="body style", client=c, cache_dir=None)
    out = k.fit_transform(cars(n_each=1))
    assert list(out.columns) == ["body_type"]
    assert "fixed" in c.proposal_prompts[0] and "body style" in c.proposal_prompts[0]
    assert k.lookups_[0].attribute == "body style"


def test_automatic_mode_renames_on_clash_with_existing_column():
    prop = [dict(PROPOSAL[0], name="type")]
    c = FakeClient(TABLE, proposal=prop)
    out = KnowledgeEncoder(client=c, cache_dir=None).fit_transform(cars(n_each=1))
    assert list(out.columns) == ["type_2"]


def test_with_target_only_columns_that_helped_are_returned():
    rng = np.random.default_rng(0)
    # 60 models x 10 rows. Price is determined by body style and unrelated to color (random).
    # A constant manufacturer cannot be a key (the column kind becomes "constant"), so use 2 makers.
    # LightGBM needs 100+ rows per group to split on a category, so take just enough rows
    models = [f"m{i:02d}" for i in range(60)]
    maker = {m: "x" if i % 2 else "y" for i, m in enumerate(models)}
    body = {m: BODY[i % 4] for i, m in enumerate(models)}
    base = {"sedan": 100, "SUV": 200, "pickup": 300, "hatchback": 80}
    rows = []
    for m in models:
        for _ in range(10):
            rows.append({"manufacturer": maker[m], "model": m,
                         "price": base[body[m]] + rng.normal(0, 5)})
    df = pd.DataFrame(rows)
    table = {f"{maker[m]} | {m}": good(body[m]) for m in models}
    colors = ["red", "blue", "green"]
    table_color = {f"{maker[m]} | {m}": good(colors[i % 3]) for i, m in enumerate(models)}
    prop = [PROPOSAL[0],
            {"name": "color", "keys": ["manufacturer", "model"], "attribute": "typical color",
             "type": "category", "values": colors, "unit": "", "range_low": 0, "range_high": 0,
             "why": "color"}]
    c = FakeClient(tables={"body style": table, "typical color": table_color}, proposal=prop)
    k = KnowledgeEncoder(target="price", client=c, cache_dir=None, n_splits=3, sample=None)
    out = k.fit_transform(df)
    assert list(out.columns) == ["body_type"]              # color is dropped
    sc = k.scores_
    assert bool(sc.loc["body_type", "accepted"]) and not bool(sc.loc["color", "accepted"])
    assert sc.loc["body_type", "contribution"] > 0.5
    assert k.status()["returned_columns"] == ["body_type"]
    assert "Target variable: price" in c.proposal_prompts[0]


def test_without_target_no_scoring_and_all_columns_are_returned():
    c = FakeClient(tables={"body style": TABLE, "price when new": MSRP}, proposal=PROPOSAL[:2])
    k = KnowledgeEncoder(client=c, cache_dir=None)
    out = k.fit_transform(cars(n_each=2))
    assert k.scores_ is None and list(out.columns) == ["body_type", "msrp"]


def test_manual_mode_also_scores_when_target_is_given():
    df = cars(n_each=20)
    df["price"] = df["type"].map({"pickup": 300, "sedan": 100, "SUV": 200}) + np.arange(len(df)) % 3
    df = df.drop(columns="type")
    k = col(FakeClient(TABLE), target="price", n_splits=2, sample=None)
    out = k.fit_transform(df)
    assert k.scores_ is not None and "body_type" in k.scores_.index
    assert list(out.columns) in (["body_type"], [])          # scoring runs (acceptance depends on the data)


# --- Input validation ----------------------------------------------------------

def test_missing_api_key_raises_mekiki_error():
    k = col(ClaudeClient(api_key="", cache_dir=None))
    with pytest.raises(MekikiError, match="ANTHROPIC_API_KEY"):
        k.fit_transform(cars(n_each=1))
    assert k.fit(cars(n_each=1)).cost()["n_to_ask"] == 3      # the estimate alone still works
    with pytest.raises(MekikiError, match="ANTHROPIC_API_KEY"):
        KnowledgeEncoder(client=ClaudeClient(api_key="", cache_dir=None), cache_dir=None).fit(cars(n_each=1))


def test_configuration_mistakes_raise_mekiki_error():
    with pytest.raises(MekikiError, match="not supported"):
        KnowledgeEncoder(keys="model", attribute="x", type="text", cache_dir=None)
    with pytest.raises(MekikiError, match="binary"):
        KnowledgeEncoder(keys="model", attribute="x", type="binary", values=["a"], cache_dir=None)
    with pytest.raises(MekikiError, match="range"):
        KnowledgeEncoder(keys="model", attribute="x", type="numeric", values=["a"], cache_dir=None)
    with pytest.raises(MekikiError, match="Columns given in keys"):
        col(FakeClient(TABLE), keys=["nope"]).fit(cars(n_each=1))
    with pytest.raises(MekikiError, match="check_against"):
        col(FakeClient(TABLE), check_against="nope").fit(cars(n_each=1))
    with pytest.raises(MekikiError, match="target"):
        col(FakeClient(TABLE), target="nope").fit(cars(n_each=1))
    with pytest.raises(MekikiError, match="fit first"):
        col(FakeClient(TABLE)).transform(cars(n_each=1))
    with pytest.raises(MekikiError, match="No such column"):
        k = col(FakeClient(TABLE))
        k.fit_transform(cars(n_each=1))
        k.explain(0, "nope")
