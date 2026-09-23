"""Tests for the Jev tier (`mekiki/jev.py`) and the three-stage fallback it enables.

**Everything passes without a network or a key.** `JevClient` takes an injected
`transport` (request body in, decoded response body out), so these tests stub the
wire format of the API exactly and exercise the cache, the cost accounting, the
error handling, and the wiring into `SemanticEncoder`, `EvidencePredictor` and
`KnowledgeEncoder`. The frontier LLM is replaced by the same fake clients the other
test files use.

Run: .venv/bin/python -m pytest tests/test_jev.py -q
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from mekiki import (
    Domain,
    EvidenceClassifier,
    EvidenceRegressor,
    JevClient,
    JevFallback,
    KnowledgeEncoder,
    LLMFallback,
    MekikiError,
    MekikiWarning,
    SemanticEncoder,
)
from mekiki import jev as jev_module
from mekiki.fallback import Answer
from mekiki.jev import JevHTTPError, choice_question
from mekiki.llm import ClaudeClient, LLMAnswer


@pytest.fixture(autouse=True)
def no_settings(monkeypatch):
    """Hide the real settings file and environment from `JevClient`, so a key the
    developer has configured neither changes the defaults under test nor sends HTTP.
    Tests that need a setting put it in the returned dict."""
    settings: dict[str, str] = {}
    monkeypatch.setattr(jev_module, "load_api_key", settings.get)
    return settings

# ---------------------------------------------------------------------------
# A transport that speaks the API's wire format without any HTTP
# ---------------------------------------------------------------------------


def choice_response(choice: str, probabilities: dict[str, float], confidence: float | None = None,
                    input_tokens: int = 500, name: str = "choice") -> dict:
    """The response body of one `choice` question, as the API returns it."""
    conf = confidence if confidence is not None else probabilities.get(choice, 0.0)
    return {"model": "jev-1.13",
            "usage": {"input_tokens": input_tokens, "output_tokens": 3},
            "answers": {name: {"type": "choice", "choice": choice,
                               "confidence": conf, "probabilities": probabilities}}}


class Transport:
    """Records every request body and answers with `rule(body)`."""

    def __init__(self, rule: Callable[[dict], dict]) -> None:
        self.rule = rule
        self.bodies: list[dict] = []

    def __call__(self, body: dict) -> dict:
        self.bodies.append(json.loads(json.dumps(body)))    # must be JSON-serialisable
        return self.rule(body)


def first_option_rule(body: dict) -> dict:
    """Pick the first option of the (single) choice question with probability 0.9."""
    name, q = next(iter(body["questions"].items()))
    opts = list(q["criteria"])
    probs = {o: 0.1 / max(len(opts) - 1, 1) for o in opts}
    probs[opts[0]] = 0.9
    return choice_response(opts[0], probs, name=name)


def client(rule=first_option_rule, **kw) -> tuple[JevClient, Transport]:
    t = Transport(rule)
    kw.setdefault("cache_dir", None)
    return JevClient(transport=t, **kw), t


# --- the client ------------------------------------------------------------


def test_without_a_key_it_is_unavailable_and_says_which_key():
    c = JevClient(api_key="", cache_dir=None)
    assert not c.available()
    assert "TYPESAFE_API_KEY" in c.why_unavailable()
    assert "AI_GATEWAY_API_KEY" in c.why_unavailable()
    a = c.ask("state", {"q": choice_question(["a", "b"], "pick")})
    assert not a.ok and c.usage.errors == 1


def test_an_injected_transport_needs_no_key_and_sends_the_wire_format():
    c, t = client(api_key="")
    assert c.available()
    a = c.ask({"text": "hello"}, {"tone": choice_question(["calm", "angry"], "What tone?",
                                                           {"angry": "hostile"})})
    assert a.ok and a.answers["tone"]["choice"] == "calm"
    body = t.bodies[0]
    assert body["model"] == "jev-latest" and body["state"] == {"text": "hello"}
    assert body["questions"] == {"tone": {"type": "choice", "instructions": "What tone?",
                                          "criteria": {"calm": None, "angry": "hostile"}}}


def test_cost_counts_input_tokens_only(monkeypatch):
    c, _ = client()
    a = c.ask("s", {"q": choice_question(["a", "b"], "pick")})
    assert a.input_tokens == 500 and a.output_tokens == 3
    assert a.cost == pytest.approx(500 * 0.042 / 1_000_000)
    assert c.usage.cost == pytest.approx(a.cost)
    assert c.unit_prices() == (0.042, 0.0)
    assert c.summary()["n_calls"] == 1
    # The estimate uses the list price before any call, the measured average after
    fresh, _ = client()
    assert fresh.estimated_cost_per_call(1000) == pytest.approx(0.042 / 1000)
    assert c.estimated_cost_per_call() == pytest.approx(a.cost)


def test_a_vercel_gateway_key_switches_endpoint_and_model(no_settings):
    no_settings["AI_GATEWAY_API_KEY"] = "vck_test"
    c = JevClient(cache_dir=None)
    assert c.available() and c.via_gateway and c.api_key == "vck_test"
    assert c.base_url == "https://ai-gateway.vercel.sh/typesafe"
    assert c.model == "typesafe-ai/jev"
    assert c.unit_prices() == (0.042, 0.0)
    # A TypeSafe key wins over the gateway key
    no_settings["TYPESAFE_API_KEY"] = "ts_test"
    direct = JevClient(cache_dir=None)
    assert not direct.via_gateway and direct.api_key == "ts_test"
    assert direct.base_url == "https://api.typesafe.ai" and direct.model == "jev-latest"


def test_the_cost_reported_by_the_gateway_is_recorded():
    def rule(body):
        r = first_option_rule(body)
        r["provider_metadata"] = {"gateway": {"cost": "0.00001155"}}
        return r
    c, _ = client(rule)
    a = c.ask("s", {"q": choice_question(["a", "b"], "pick")})
    assert a.cost == pytest.approx(0.00001155) and c.usage.cost == pytest.approx(0.00001155)
    # A malformed figure falls back to the list price
    c2, _ = client(lambda b: {**first_option_rule(b),
                              "provider_metadata": {"gateway": {"cost": "n/a"}}})
    assert c2.ask("s", {"q": choice_question(["a", "b"], "pick")}).cost == \
        pytest.approx(500 * 0.042 / 1_000_000)


def test_same_request_is_served_from_disk_the_second_time(tmp_path):
    c, t = client(cache_dir=tmp_path)
    q = {"q": choice_question(["a", "b"], "pick")}
    a = c.ask("s", q)
    b = c.ask("s", q)
    assert len(t.bodies) == 1
    assert not a.from_cache and b.from_cache and b.cost == 0.0
    assert b.answers == a.answers and b.input_tokens == 500
    assert c.usage.calls == 2 and c.usage.cache_hits == 1
    # A different client sharing the directory hits the same entry
    c2, t2 = client(cache_dir=tmp_path)
    assert c2.ask("s", q).from_cache and t2.bodies == []
    # A different state or question misses
    c2.ask("other", q)
    assert len(t2.bodies) == 1


def test_a_corrupt_cache_entry_is_ignored_and_the_call_is_retried(tmp_path):
    c, t = client(cache_dir=tmp_path)
    q = {"q": choice_question(["a", "b"], "pick")}
    path = c._cache_path(c._key("s", q))
    path.write_text("{not json", encoding="utf-8")
    assert c.ask("s", q).ok and len(t.bodies) == 1
    path.write_text(json.dumps({"answers": "not a dict"}), encoding="utf-8")
    assert c.ask("s", q).ok and len(t.bodies) == 2


def test_http_errors_become_answers_with_error_and_are_not_cached(tmp_path):
    def unauthorised(body):
        raise JevHTTPError(401, "invalid key")
    c, _ = client(unauthorised, cache_dir=tmp_path)
    a = c.ask("s", {"q": choice_question(["a"], "pick")})
    assert not a.ok and "401" in a.error and c.usage.errors == 1
    assert not any(tmp_path.rglob("*.json")), "a failed call must not be cached"
    assert c.choose("s", ["a", "b"], "pick").choice is None


def test_rate_limits_and_server_errors_are_retried_with_a_bound(monkeypatch):
    from mekiki import jev as jev_module
    sleeps: list[float] = []
    monkeypatch.setattr(jev_module.time, "sleep", sleeps.append)
    calls = {"n": 0}

    def flaky(body):
        calls["n"] += 1
        if calls["n"] == 1:
            raise JevHTTPError(429, "slow down", retry_after=0.2)
        if calls["n"] == 2:
            raise JevHTTPError(503, "unavailable")
        return first_option_rule(body)

    c, _ = client(flaky, max_retries=2)
    assert c.ask("s", {"q": choice_question(["a"], "pick")}).ok
    assert calls["n"] == 3 and sleeps == [0.2, 1.0]      # retry-after, then backoff

    calls["n"] = 0
    c, _ = client(flaky, max_retries=1)
    assert not c.ask("s", {"q": choice_question(["a"], "pick")}).ok     # bound respected

    def bad_request(body):
        calls["n"] += 1
        raise JevHTTPError(422, "criteria: Field required")
    calls["n"] = 0
    c, _ = client(bad_request, max_retries=2)
    assert not c.ask("s", {"q": choice_question(["a"], "pick")}).ok
    assert calls["n"] == 1                                # 4xx other than 429: no retry


def test_choose_rejects_an_answer_outside_the_options():
    c, _ = client(lambda body: choice_response("zzz", {"zzz": 1.0}))
    r = c.choose("s", ["a", "b"], "pick")
    assert r.choice is None and "not one of the options" in r.error
    choice, conf, probs, cost = r                       # unpacks as the documented tuple
    assert choice is None and conf == 0.0 and probs == {} and cost > 0


def test_choose_many_keeps_order_and_takes_options_per_row():
    def echo(body):
        # Answer with the option whose description mentions the state
        name, q = next(iter(body["questions"].items()))
        pick = next(o for o, d in q["criteria"].items() if d and body["state"] in d)
        return choice_response(pick, {pick: 0.8}, name=name)
    c, t = client(echo, max_workers=4)
    states = [f"row{i}" for i in range(10)]
    options = [[f"m{i}", f"n{i}"] for i in range(10)]
    descs = [{f"m{i}": f"about row{i}"} for i in range(10)]
    out = c.choose_many(states, options, "pick", descs)
    assert [r.choice for r in out] == [f"m{i}" for i in range(10)]
    assert len(t.bodies) == 10
    with pytest.raises(MekikiError):
        c.choose_many(states, options[:3], "pick")


def test_score_returns_the_expected_level():
    def rate(body):
        return {"model": "jev-1.13", "usage": {"input_tokens": 100, "output_tokens": 2},
                "answers": {"score": {"type": "score", "score": 1.7, "confidence": 0.9,
                                      "probabilities": {"0": 0.1, "1": 0.1, "2": 0.8},
                                      "legend": {"0": "low", "1": "mid", "2": "high"}}}}
    c, _ = client(rate)
    s = c.score("s", ["low", "mid", "high"], "how much?")
    assert s.score == 1.7 and s.confidence == 0.9 and s.legend["2"] == "high"
    assert [r.score for r in c.score_many(["a", "b"], ["low", "high"], "how much?")] == [1.7, 1.7]


def test_choice_question_validates_its_options():
    with pytest.raises(MekikiError, match="at most 255"):
        choice_question([str(i) for i in range(256)], "pick")
    with pytest.raises(MekikiError, match="distinct"):
        choice_question(["a", "a"], "pick")
    with pytest.raises(MekikiError):
        JevClient(transport=lambda b: {}, cache_dir=None).ask("s", {})


# ---------------------------------------------------------------------------
# SemanticEncoder: classifier -> Jev -> LLM
# ---------------------------------------------------------------------------


def titles() -> pd.DataFrame:
    """Clear rows, plus two pairs of identical texts with conflicting labels ("alpha"
    and "beta") so that those rows are uncertain and reach the fallback chain."""
    rows = [("Sienta G Cuero non-smoker backup camera", "G Cuero"),
            ("Sienta Hybrid Z 4WD OEM navi", "Z"),
            ("Sienta X dual power sliding doors", "X"),
            ("Sienta Hybrid G ETC", "G")] * 6
    rows += [("Sienta special edition smart key alpha", "G"),
             ("Sienta special edition smart key alpha", "Z"),
             ("Sienta special edition smart key beta", "G"),
             ("Sienta special edition smart key beta", "Z")]
    return pd.DataFrame([{"title": t, "label": g} for t, g in rows])


def jev_by_text(body: dict) -> dict:
    """Confident "Z" when the text mentions alpha, an unsure "G" otherwise."""
    text = body["state"]["text"]
    if "alpha" in text:
        return choice_response("Z", {"Z": 0.95, "G": 0.05})
    return choice_response("G", {"G": 0.4, "Z": 0.35, "X": 0.25})


class DummyLLM:
    """A fallback stage without a threshold: it settles everything it is given."""
    cost_per_call = 0.002

    def __init__(self) -> None:
        self.texts: list[str] = []
        self.queued: list[dict] = []

    def can_answer(self):
        return True

    def answer(self, texts, values, context):
        self.texts += list(texts)
        return [Answer(value="X", confidence=0.9, cost=0.002, origin="llm") for _ in texts]


def test_jev_fallback_sends_the_text_and_the_neighbours_as_the_state():
    c, t = client(jev_by_text)
    fb = JevFallback(client=c, threshold=0.7)
    ctx = [{"examples": [{"text": "a G grade car", "value": "G", "similarity": 0.82}]}]
    out = fb.answer(["with navi alpha"], ["G", "Z"], ctx)
    assert out[0].value == "Z" and out[0].origin == "jev" and out[0].confidence == 0.95
    state = t.bodies[0]["state"]
    assert state["text"] == "with navi alpha"
    assert state["similar_labelled_examples"][0] == {"text": "a G grade car", "value": "G",
                                                     "similarity": 0.82}
    assert list(t.bodies[0]["questions"]["choice"]["criteria"]) == ["G", "Z"]
    assert fb.queued[0]["jev_answer"] == "Z"           # confident answers are queued for review
    assert fb.cost_per_call > 0


def test_chain_settles_confident_rows_at_jev_and_passes_the_rest_to_the_llm():
    c, t = client(jev_by_text)
    llm = DummyLLM()
    f = SemanticEncoder(source="title", labels="label", k=3, threshold=0.9,
                        fallback=[JevFallback(client=c, threshold=0.7), llm])
    out = f.fit_transform(titles())
    prov = f.provenance_
    alpha = prov["text"].str.contains("alpha")
    beta = prov["text"].str.contains("beta")
    assert (prov.loc[alpha, "source"] == "jev").all()
    assert (prov.loc[alpha, "value"] == "Z").all()
    assert (prov.loc[beta, "source"] == "llm").all()
    assert (prov.loc[beta, "value"] == "X").all()
    assert all("beta" in x for x in llm.texts) and not any("alpha" in x for x in llm.texts)
    assert (out.astype(str)[alpha.to_numpy()] == "Z").all()
    st = f.status()
    assert st["jev_answered"] >= 2 and st["llm_answered"] >= 2 and st["pending_review"] == 0
    # The cost of every stage a row passed through adds up
    jev_cost = 500 * 0.042 / 1_000_000
    assert prov.loc[alpha, "cost"].iloc[0] == pytest.approx(jev_cost)
    assert prov.loc[beta, "cost"].iloc[0] == pytest.approx(jev_cost + 0.002)
    assert f.cost()["actual_cost_usd"] == pytest.approx(prov["cost"].sum())
    assert f.cost()["n_escalated"] == st["jev_answered"] + st["llm_answered"]
    text = f.explain(int(np.flatnonzero(alpha)[0]))
    assert "source            jev" in text and "Jev calls         1" in text
    assert len(f.review_queue()) == st["jev_answered"]


def test_rows_jev_is_unsure_about_are_queued_when_there_is_no_next_stage():
    c, _ = client(jev_by_text)
    f = SemanticEncoder(source="title", labels="label", k=3, threshold=0.9,
                        fallback=JevFallback(client=c, threshold=0.7))
    f.fit_transform(titles())
    prov = f.provenance_
    beta = prov["text"].str.contains("beta")
    assert (prov.loc[beta, "source"] == "needs_review").all()
    assert f.status()["pending_review"] >= 2
    queue = f.review_queue()
    assert "classifier_guess" in queue.columns and "jev_answer" in queue.columns


def test_chain_without_any_key_degrades_to_the_queue():
    jev_fb = JevFallback(client=JevClient(api_key="", cache_dir=None))
    llm_fb = LLMFallback(client=ClaudeClient(api_key="", cache_dir=None))
    assert not jev_fb.can_answer() and not llm_fb.can_answer()
    df = titles()
    chained = SemanticEncoder(source="title", labels="label", k=3, threshold=0.9,
                              fallback=[jev_fb, llm_fb]).fit_transform(df)
    plain = SemanticEncoder(source="title", labels="label", k=3, threshold=0.9)
    expected = plain.fit_transform(df)
    assert chained.astype(str).tolist() == expected.astype(str).tolist()
    st = plain.status()
    assert st["pending_review"] > 0 and st["jev_answered"] == 0
    assert len(llm_fb.queued) == st["pending_review"]      # queued on the last stage


def test_an_empty_chain_is_refused():
    with pytest.raises(MekikiError):
        SemanticEncoder(source="title", values=["a"], fallback=[])


# ---------------------------------------------------------------------------
# EvidencePredictor: statistical models -> Jev -> LLM
# ---------------------------------------------------------------------------


class FakeLLM(ClaudeClient):
    """The frontier LLM, answering a constant without any HTTP."""

    def __init__(self, value=200.0, label=None, probabilities=None, **kw):
        kw.setdefault("cache_dir", None)
        kw.setdefault("api_key", "test-key")
        super().__init__(**kw)
        self.value, self.label, self.probabilities = value, label, probabilities
        self.prompts: list[str] = []

    def ask(self, system, user, schema) -> LLMAnswer:
        self.prompts.append(user)
        self.usage.calls += 1
        self.usage.cost += 0.001
        props = schema["properties"]
        if "label" in props:
            data = {"label": self.label, "probabilities": self.probabilities,
                    "confidence": 0.8, "reason": "test"}
        else:
            key = next(k for k in props if k not in ("confidence", "reason"))
            data = {key: self.value, "confidence": 0.8, "reason": "test"}
        return LLMAnswer(data=data, input_tokens=100, output_tokens=20, cost=0.001)


WEIGHTS = {"LightGBM": 0.6, "XGBoost": 0.3}      # the neighbour model gets the rest


def weigh_models(body: dict, confidence: float | None = None) -> dict:
    """Weigh the statistical models (the options of the question) with fixed probabilities."""
    q = body["questions"]["choice"]
    names = list(q["criteria"])
    probs = {n: WEIGHTS.get(n, 0.0) for n in names}
    rest = [n for n in names if n not in WEIGHTS]
    for n in rest:
        probs[n] = (1.0 - sum(WEIGHTS.values())) / len(rest)
    best = max(probs, key=probs.get)
    return choice_response(best, probs, confidence=confidence)


@pytest.fixture
def cars() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 80
    age = rng.integers(1, 10, n)
    km = rng.integers(5_000, 120_000, n)
    grade = rng.choice(["G", "X", "Z"], n)
    return pd.DataFrame({
        "age": age,
        "mileage_km": km,
        "grade": grade,
        "equipment_text": [f"navi ETC {g} grade equipment{i % 5}" for i, g in enumerate(grade)],
        "price": 250 - age * 12 - km / 8000 + rng.normal(0, 3, n),
    })


def regressor(**kw) -> EvidenceRegressor:
    return EvidenceRegressor(target="price", unit="10k JPY", numeric=["age", "mileage_km"],
                             categorical=["grade"], text="equipment_text", n_examples=3, **kw)


def test_fast_path_rows_take_jev_weighted_average_of_the_models(cars):
    jev, t = client(weigh_models)
    llm = FakeLLM()
    m = regressor(client=llm, jev=jev, escalate_rate=0.25).fit(cars.iloc[:60])
    pred = m.predict(cars.iloc[60:])
    prov = m.provenance()
    fast = ~m.selected_
    assert set(prov.loc[fast, "source"]) == {"jev"}
    assert set(prov.loc[m.selected_, "source"]) == {"llm"}
    assert len(llm.prompts) == 5 and len(t.bodies) == 15
    nn = next(c for c in prov.columns if c.startswith("evidence_") and "NN" in c)
    expected = (0.6 * prov["evidence_LightGBM"] + 0.3 * prov["evidence_XGBoost"] + 0.1 * prov[nn])
    assert np.allclose(pred[fast], expected[fast])
    assert np.allclose(pred[m.selected_], 200.0)
    p = m.predictions_[int(np.flatnonzero(fast)[0])]
    assert p.reason.startswith("Jev weighted LightGBM 0.60 / XGBoost 0.30 /")
    assert p.confidence == pytest.approx(0.6) and p.cost > 0
    assert p.neighbours and len(p.neighbours) == 3

    # The question Jev saw: one option per model, described by its prediction, and the
    # row plus its similar cases as the state
    body = t.bodies[0]
    crit = body["questions"]["choice"]["criteria"]
    assert set(crit) == {mm.name for mm in m.models_}
    assert all(d.startswith("predicts ") and "10k JPY" in d for d in crit.values())
    assert "The record to predict" in body["state"]
    assert len(body["state"]["similar cases with known price (closest first)"]) == 3

    # The inspection API shows the tier
    r = m.route()
    assert set(r.loc[fast, "route"]) == {"jev"} and set(r.loc[m.selected_, "route"]) == {"llm"}
    c = m.cost()
    assert c["n_jev_answered"] == 15 and c["jev"]["n_calls"] == 15 and c["jev_cost_usd"] > 0
    rep = m.report()
    assert "answered by Jev:      15 rows" in rep and "statistical models:   0 rows" in rep
    text = m.explain(int(np.flatnonzero(fast)[0]))
    assert "Jev (LLM not called)" in text and "Jev reason: Jev weighted" in text
    assert m.review_queue()["row"].tolist() == list(np.flatnonzero(m.selected_))


def test_plan_shows_the_jev_tier_without_calling_it(cars):
    jev, t = client(weigh_models)
    m = regressor(client=FakeLLM(), jev=jev, escalate_rate=0.25).fit(cars.iloc[:60])
    plan = m.plan(cars.iloc[60:])
    assert t.bodies == []
    assert plan["n_escalated"] == 5 and plan["n_fast_path"] == 15
    assert plan["n_jev"] == 15 and plan["jev_available"] is True
    assert plan["jev_cost_per_row_usd"] == pytest.approx(600 * 0.042 / 1_000_000, abs=1e-6)
    assert plan["estimated_jev_cost_usd"] == pytest.approx(15 * 600 * 0.042 / 1_000_000, abs=1e-4)
    assert plan["estimated_jev_seconds"] > 0


def test_signal_jev_sends_the_rows_jev_is_least_sure_about(cars, tmp_path):
    def by_age(body):
        # Confidence that depends on the row, so that routing has something to rank
        age = int(re.search(r"- age: (\d+)", body["state"]["The record to predict"]).group(1))
        return weigh_models(body, confidence=0.5 + (age % 5) / 10)
    jev, t = client(by_age, cache_dir=tmp_path)
    llm = FakeLLM()
    m = regressor(client=llm, jev=jev, signal="jev", escalate_rate=0.25).fit(cars.iloc[:60])
    test = cars.iloc[60:].reset_index(drop=True)
    m.predict(test)
    conf = 0.5 + (test["age"].to_numpy() % 5) / 10
    assert np.allclose(m.signal_, 1 - conf)
    assert m.signal_[m.selected_].min() >= m.signal_[~m.selected_].max()
    assert len(llm.prompts) == 5 and len(t.bodies) == 20        # Jev saw every row once
    assert m.plan(test)["n_escalated"] == 5 and len(t.bodies) == 20   # cached: no new call
    assert m.plan(test)["signal"] == "jev"


def test_signal_jev_needs_the_jev_tier(cars):
    with pytest.raises(MekikiError, match="jev=True"):
        regressor(client=FakeLLM(), signal="jev")
    m = regressor(client=FakeLLM(), jev=JevClient(api_key="", cache_dir=None), signal="jev")
    m.fit(cars.iloc[:60])
    with pytest.raises(MekikiError, match="TYPESAFE_API_KEY"):
        m.plan(cars.iloc[60:])
    with pytest.raises(MekikiError, match="TYPESAFE_API_KEY"):
        m.predict(cars.iloc[60:])


def test_without_jev_nothing_changes(cars):
    m = regressor(client=FakeLLM(), escalate_rate=0.25).fit(cars.iloc[:60])
    m.predict(cars.iloc[60:])
    assert set(m.provenance()["source"]) == {"model", "llm"}
    assert set(m.route()["route"]) == {"fast", "llm"}
    for key in ("n_jev", "jev_available"):
        assert key not in m.plan(cars.iloc[60:])
    assert "jev" not in m.cost() and "Jev" not in m.report()


def test_jev_without_a_key_warns_once_and_keeps_the_statistical_answer(cars):
    m = regressor(client=FakeLLM(), jev=JevClient(api_key="", cache_dir=None),
                  escalate_rate=0.25).fit(cars.iloc[:60])
    with pytest.warns(MekikiWarning, match="Jev tier is skipped"):
        m.predict(cars.iloc[60:])
    assert set(m.provenance()["source"]) == {"model", "llm"}
    assert m.plan(cars.iloc[60:])["n_jev"] == 0


def test_a_row_jev_cannot_answer_keeps_the_first_model(cars):
    def flaky(body):
        if "- age: 3" in body["state"]["The record to predict"]:
            raise JevHTTPError(500, "boom")
        return weigh_models(body)
    jev, _ = client(flaky, max_retries=0)
    m = regressor(client=FakeLLM(), jev=jev, escalate_rate=0.0).fit(cars.iloc[:60])
    test = cars.iloc[60:].reset_index(drop=True)
    pred = m.predict(test)
    prov = m.provenance()
    failed = (test["age"] == 3).to_numpy()
    assert failed.any()
    assert set(prov.loc[failed, "source"]) == {"model"}
    assert np.allclose(pred[failed], prov.loc[failed, "evidence_LightGBM"])
    assert all("Jev could not answer" in r for r in prov.loc[failed, "reason"])
    assert set(prov.loc[~failed, "source"]) == {"jev"}


def test_curve_uses_jev_for_the_rows_not_sent(cars):
    jev, _ = client(weigh_models)
    m = regressor(client=FakeLLM(), jev=jev, escalate_rate=0.25).fit(cars.iloc[:60])
    cv = m.curve(cars.iloc[60:], rates=(0.0, 1.0))
    assert "jev_cost_usd" in cv.columns and cv.loc[0, "jev_cost_usd"] > cv.loc[1, "jev_cost_usd"]
    assert cv.loc[1, "MAE"] == pytest.approx(
        np.mean(np.abs(200.0 - cars.iloc[60:]["price"].to_numpy())))


@pytest.fixture
def churn() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    n = 80
    tenure = rng.integers(1, 72, n)
    monthly = rng.uniform(20, 110, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n)
    logit = -0.05 * tenure + 0.03 * monthly + (contract == "Month-to-month") * 1.5 - 1
    churned = rng.random(n) < 1 / (1 + np.exp(-logit))
    return pd.DataFrame({"tenure": tenure, "monthly": monthly.round(2), "contract": contract,
                         "Churn": np.where(churned, "Yes", "No")})


def test_classification_averages_the_probability_vectors(churn):
    jev, t = client(weigh_models)
    m = EvidenceClassifier(target="Churn", numeric=["tenure", "monthly"], categorical=["contract"],
                           domain=Domain(role="a churn analyst", subject="customer",
                                         class_names={"Yes": "churned", "No": "stayed"}),
                           n_examples=3, client=FakeLLM(label="Yes", probabilities={"Yes": 0.7, "No": 0.3}),
                           jev=jev, escalate_rate=0.0).fit(churn.iloc[:60])
    proba = m.predict_proba(churn.iloc[60:])
    preds = m.predictions_
    assert all(p.origin == "jev" for p in preds)
    names = [mm.name for mm in m.models_]
    for p, row in zip(preds, proba, strict=True):
        w = [0.6, 0.3, 0.1]
        expected = sum(wi * np.asarray(p.model_predictions[n]) for wi, n in zip(w, names, strict=True))
        assert np.allclose(row, expected / expected.sum())
        assert p.value == m.classes_[int(np.argmax(row))]
    crit = t.bodies[0]["questions"]["choice"]["criteria"]
    assert all("Yes (churned)" in d for d in crit.values())


# ---------------------------------------------------------------------------
# KnowledgeEncoder: Jev before the LLM
# ---------------------------------------------------------------------------

BODY = ["sedan", "SUV", "pickup", "hatchback"]


class FakeKnowledgeLLM(ClaudeClient):
    """Answers the key values of a lookup prompt from a table, without any HTTP."""

    def __init__(self, table: dict, **kw):
        kw.setdefault("cache_dir", None)
        kw.setdefault("api_key", "test-key")
        super().__init__(**kw)
        self.table = table
        self.batches: list[list[str]] = []

    def ask(self, system, user, schema) -> LLMAnswer:
        with self._lock:
            self.usage.calls += 1
        keys = re.findall(r"^\d+\. (.+)$", user, flags=re.M)
        self.batches.append(keys)
        answers = []
        for k in keys:
            a = self.table.get(k)
            if a is None:
                answers.append({"key": k, "known": False, "value": "", "confidence": 0.0,
                                "reason": "unknown"})
            else:
                answers.append({"key": k, "known": True, "confidence": 0.9,
                                "reason": "well-known", **a})
        self.usage.cost += 0.01
        return LLMAnswer(data={"answers": answers}, input_tokens=300, output_tokens=200, cost=0.01)


TABLE = {"ford | f-150": {"value": "pickup"}, "toyota | camry": {"value": "sedan"},
         "honda | cr-v": {"value": "SUV"}, "kia | soul": {"value": 50.0}}


def jev_knowledge(body: dict) -> dict:
    """Confident on Ford, unsure on Toyota, declines on Honda."""
    state = body["state"]
    opts = list(body["questions"]["choice"]["criteria"])
    assert opts[-1] == "unknown" and set(opts[:-1]) == set(BODY)
    if state == {"manufacturer": "ford", "model": "f-150"}:
        return choice_response("pickup", {"pickup": 0.9, "SUV": 0.1})
    if state["manufacturer"] == "toyota":
        return choice_response("sedan", {"sedan": 0.5, "hatchback": 0.5})
    return choice_response("unknown", {"unknown": 0.8})


def cars_table() -> pd.DataFrame:
    rows = []
    for m, mdl in [("ford", "f-150"), ("toyota", "camry"), ("honda", "cr-v")]:
        rows += [{"manufacturer": m, "model": mdl}] * 3
    return pd.DataFrame(rows)


def test_category_lookups_go_to_jev_first_and_the_rest_to_the_llm(tmp_path):
    jev, t = client(jev_knowledge)
    llm = FakeKnowledgeLLM(TABLE)
    kc = KnowledgeEncoder(keys=["manufacturer", "model"], attribute="body style",
                          type="category", values=BODY, client=llm, jev=jev,
                          cache_dir=tmp_path, name="body_type")
    out = kc.fit_transform(cars_table())
    assert len(t.bodies) == 3                                   # every unknown key, once
    assert t.bodies[0]["questions"]["choice"]["instructions"].startswith("You are ")
    assert llm.batches == [["honda | cr-v", "toyota | camry"]]  # unsure + unknown go to the LLM
    assert out["body_type"].tolist() == ["pickup"] * 3 + ["sedan"] * 3 + ["SUV"] * 3
    prov = kc.provenance_
    assert prov.loc[prov["key"] == "ford | f-150", "source"].unique().tolist() == ["jev"]
    assert prov.loc[prov["key"] != "ford | f-150", "source"].unique().tolist() == ["llm"]
    st = kc.status()["per_column"][0]
    assert st["accepted_jev"] == 1 and st["accepted_llm"] == 2 and st["n_rows_filled"] == 9
    assert "Jev chose 'pickup'" in kc.explain(0)
    # The Jev cost of keys it could not settle is carried over to their LLM answer
    ans = kc.answers_["body_type"].set_index("key")
    jev_cost = 500 * 0.042 / 1_000_000
    assert ans.loc["ford | f-150", "cost"] == pytest.approx(jev_cost)
    assert ans.loc["toyota | camry", "cost"] == pytest.approx(0.005 + jev_cost)

    # Persisted like LLM answers: a second encoder asks neither Jev nor the LLM
    jev2, t2 = client(jev_knowledge)
    llm2 = FakeKnowledgeLLM(TABLE)
    kc2 = KnowledgeEncoder(keys=["manufacturer", "model"], attribute="body style",
                           type="category", values=BODY, client=llm2, jev=jev2,
                           cache_dir=tmp_path, name="body_type")
    out2 = kc2.fit_transform(cars_table())
    assert t2.bodies == [] and llm2.batches == []
    assert out2["body_type"].tolist() == out["body_type"].tolist()


def test_cost_shows_the_jev_split():
    jev, _ = client(jev_knowledge)
    kc = KnowledgeEncoder(keys=["manufacturer", "model"], attribute="body style",
                          type="category", values=BODY, client=FakeKnowledgeLLM(TABLE),
                          jev=jev, cache_dir=None).fit(cars_table())
    c = kc.cost()
    assert c["n_to_ask"] == 3 and c["n_to_ask_jev_first"] == 3 and c["jev_available"]
    assert c["per_column"][0]["jev_first"] is True
    assert 0 < c["jev_cost_per_key_usd"] < c["cost_per_key_usd"]


def test_numeric_lookups_go_straight_to_the_llm():
    jev, t = client(jev_knowledge)
    llm = FakeKnowledgeLLM(TABLE)
    kc = KnowledgeEncoder(keys=["manufacturer", "model"], attribute="price when new",
                          type="numeric", unit="kUSD", range=(1, 500), client=llm, jev=jev,
                          cache_dir=None, name="msrp")
    df = pd.DataFrame({"manufacturer": ["kia"] * 2, "model": ["soul"] * 2})
    out = kc.fit_transform(df)
    assert t.bodies == [] and llm.batches == [["kia | soul"]]
    assert out["msrp"].tolist() == [50.0, 50.0]
    assert kc.cost()["per_column"][0]["jev_first"] is False


def test_knowledge_without_a_jev_key_uses_the_llm_for_everything():
    llm = FakeKnowledgeLLM(TABLE)
    kc = KnowledgeEncoder(keys=["manufacturer", "model"], attribute="body style",
                          type="category", values=BODY, client=llm,
                          jev=JevClient(api_key="", cache_dir=None), cache_dir=None)
    kc.fit_transform(cars_table())
    assert llm.batches == [["ford | f-150", "honda | cr-v", "toyota | camry"]]
    assert kc.status()["per_column"][0]["accepted_jev"] == 0
