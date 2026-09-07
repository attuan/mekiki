"""信頼度ルーティング（AdaptivePredictor）のテスト。

`test_predictor.py` と同じく **APIキー無しで全部通る**。LLM を呼ぶのは
`ClaudeClient.ask` だけなので、そこを差し替えた偽クライアントで回す。

ここで守りたい性質は4つ。

1. **呼ぶ行数が指定どおりであること。** 費用の見積もりが成り立たなくなる
2. **呼ばなかった行は統計モデルの予測がそのまま出ること。** 黙って別の値に
   すり替わると、ルーティングの精度曲線が測れない
3. **承認した行は次回呼ばれないこと**（設計書 05 の能動学習）
4. **来歴が経路つきで残ること**（PRD「来歴（provenance）と検査 API」）

実行: .venv/bin/python -m pytest tests -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from unfold import AdaptivePredictor, LLMPredictor, UnfoldError
from unfold.llm import ClaudeClient, LLMAnswer


class FakeClient(ClaudeClient):
    """HTTP を出さずに定型の答えを返す。呼ばれた回数を数えるのが主目的。"""

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
        return LLMAnswer(data={"price": self.price, "confidence": 0.8,
                               "reason": "テスト"},
                         input_tokens=100, output_tokens=20, cost=0.001)


@pytest.fixture
def data() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 80
    age = rng.integers(1, 10, n)
    km = rng.integers(5_000, 120_000, n)
    grade = rng.choice(["G", "X", "Z"], n)
    return pd.DataFrame({
        "車齢": age,
        "走行距離_km": km,
        "修復歴あり": rng.random(n) < 0.2,
        "グレード名": grade,
        "装備テキスト": [f"ナビ ETC {g}グレード 装備{i % 5}" for i, g in enumerate(grade)],
        "価格": 250 - age * 12 - km / 8000 + rng.normal(0, 3, n),
    })


def make(data, **kw) -> AdaptivePredictor:
    return AdaptivePredictor(
        target="価格", unit="万円",
        numeric=["車齢", "走行距離_km"], boolean=["修復歴あり"],
        categorical=["グレード名"], text="装備テキスト",
        n_examples=3, **kw)


# --- 呼ぶ行数 ---------------------------------------------------------

@pytest.mark.parametrize("rate,expected", [(0.0, 0), (0.25, 5), (0.5, 10), (1.0, 20)])
def test_指定した割合ぶんだけ_LLM_を呼ぶ(data, rate, expected):
    client = FakeClient()
    m = make(data, client=client, escalate_rate=rate).fit(data.iloc[:60])
    m.predict(data.iloc[60:])
    assert len(client.prompts) == expected
    assert int(m.selected_.sum()) == expected


def test_呼ばなかった行は統計モデルの予測がそのまま出る(data):
    train, test = data.iloc[:60], data.iloc[60:]
    m = make(data, client=FakeClient(), escalate_rate=0.25).fit(train)
    pred = m.predict(test)

    base = m.predictor.models_[0]
    base_pred = np.asarray(base.predict(test.reset_index(drop=True)), dtype=float)
    fast = ~m.selected_
    assert np.allclose(pred[fast], base_pred[fast])
    assert np.allclose(pred[m.selected_], 200.0)


def test_割合0なら一度も呼ばない(data):
    client = FakeClient()
    m = make(data, client=client, escalate_rate=0.0).fit(data.iloc[:60])
    pred = m.predict(data.iloc[60:])
    assert client.prompts == []
    assert (m.provenance()["由来"] == "model").all()
    assert len(pred) == 20


# --- 信号 -------------------------------------------------------------

def test_食い違いの大きい行が選ばれる(data):
    m = make(data, client=FakeClient(), escalate_rate=0.25).fit(data.iloc[:60])
    m.predict(data.iloc[60:])
    s = m.signal_
    assert s.min() >= 0                      # 食い違いは絶対値なので非負
    assert s[m.selected_].min() >= s[~m.selected_].max()


def test_未知語の信号は訓練に無い語を含む行を選ぶ(data):
    train = data.iloc[:60].copy()
    test = data.iloc[60:].copy().reset_index(drop=True)
    test.loc[0, "装備テキスト"] = "サンルーフ 本革シート 未知の装備"
    test.loc[1, "グレード名"] = "GR SPORT"

    m = make(data, client=FakeClient(), escalate_rate=0.1,
             signal="unseen").fit(train)
    m.predict(test)
    assert m.selected_[0] and m.selected_[1]


def test_自作の信号を渡せる(data):
    def 走行距離が長い順(X, evidence):
        return X["走行距離_km"].to_numpy(dtype=float)

    m = make(data, client=FakeClient(), escalate_rate=0.25,
             signal=走行距離が長い順).fit(data.iloc[:60])
    test = data.iloc[60:].reset_index(drop=True)
    m.predict(test)
    km = test["走行距離_km"].to_numpy()
    assert km[m.selected_].min() >= km[~m.selected_].max()


def test_知らない信号名は止める(data):
    with pytest.raises(UnfoldError, match="未実装"):
        make(data, signal="でたらめ")


def test_信号関数が変な長さを返したら止める(data):
    m = make(data, client=FakeClient(), escalate_rate=0.5,
             signal=lambda X, ev: np.zeros(3)).fit(data.iloc[:60])
    with pytest.raises(UnfoldError, match="長さ"):
        m.predict(data.iloc[60:])


def test_木モデルが1つしかなければ食い違いは使えない(data):
    from unfold.predictor import TreeModel

    inner = LLMPredictor(
        target="価格", unit="万円", numeric=["車齢", "走行距離_km"],
        text="装備テキスト", n_examples=3, client=FakeClient())
    inner._models_arg = [TreeModel("LightGBM", inner.spec, kind="lgbm")]
    m = AdaptivePredictor(predictor=inner, escalate_rate=0.5).fit(data.iloc[:60])
    with pytest.raises(UnfoldError, match="木モデルが2つ以上"):
        m.predict(data.iloc[60:])


# --- 閾値の指定 -------------------------------------------------------

def test_threshold_でも切れる(data):
    m = make(data, client=FakeClient(), escalate_rate=None,
             threshold=0.0).fit(data.iloc[:60])
    m.predict(data.iloc[60:])
    assert m.selected_.all()                 # 食い違いは常に 0 以上


def test_基準を何も渡さないと止める(data):
    with pytest.raises(UnfoldError, match="escalate_rate か threshold"):
        make(data, escalate_rate=None)


def test_割合が範囲外なら止める(data):
    with pytest.raises(UnfoldError, match="0.0〜1.0"):
        make(data, escalate_rate=1.5)


# --- 実行前の見積もり -------------------------------------------------

def test_plan_は_LLM_を呼ばずに見積もる(data):
    client = FakeClient()
    m = make(data, client=client, escalate_rate=0.5).fit(data.iloc[:60])
    plan = m.plan(data.iloc[60:])
    assert client.prompts == []              # 見積もりで課金しない
    assert plan["行数"] == 20
    assert plan["LLM に回す行数"] == 10
    assert plan["統計モデルで済ませる行数"] == 10
    assert plan["推定費用_usd"] > 0
    assert plan["推定時間_秒"] > 0


def test_plan_の行数は実際に呼ぶ行数と一致する(data):
    client = FakeClient()
    m = make(data, client=client, escalate_rate=0.3).fit(data.iloc[:60])
    plan = m.plan(data.iloc[60:])
    m.predict(data.iloc[60:])
    assert plan["LLM に回す行数"] == len(client.prompts)


# --- 能動学習（承認すると高速パスが広がる）----------------------------

def test_承認した行は次回呼ばれない(data):
    client = FakeClient()
    m = make(data, client=client, escalate_rate=0.5).fit(data.iloc[:60])
    test = data.iloc[60:]

    m.predict(test)
    assert len(client.prompts) == 10
    n = m.approve()
    assert n == 10

    m.predict(test)                          # 同じ X をもう一度
    assert len(client.prompts) == 10         # 増えていない
    prov = m.provenance()
    assert (prov["由来"] == "human").sum() == 10


def test_承認は行番号ではなく中身で照合する(data):
    client = FakeClient()
    m = make(data, client=client, escalate_rate=0.5).fit(data.iloc[:60])
    test = data.iloc[60:].reset_index(drop=True)
    m.predict(test)
    m.approve()

    shuffled = test.sample(frac=1.0, random_state=1).reset_index(drop=True)
    m.predict(shuffled)
    assert len(client.prompts) == 10         # 並べ替えても呼び直さない


def test_review_queue_は_LLM_が答えた行だけ(data):
    m = make(data, client=FakeClient(), escalate_rate=0.25).fit(data.iloc[:60])
    m.predict(data.iloc[60:])
    q = m.review_queue()
    assert len(q) == 5
    assert set(q["行"]) == set(np.flatnonzero(m.selected_))
    assert "装備テキスト" in q.columns        # 人が見て判断できる中身が入る


def test_一部だけ承認できる(data):
    client = FakeClient()
    m = make(data, client=client, escalate_rate=0.5).fit(data.iloc[:60])
    test = data.iloc[60:]
    m.predict(test)
    rows = list(np.flatnonzero(m.selected_))[:3]
    assert m.approve(rows) == 3
    m.predict(test)
    assert len(client.prompts) == 10 + 7      # 承認した3行だけ減る


# --- 来歴（PRD「来歴（provenance）と検査 API」）------------------------

def test_provenance_に経路と信号が入る(data):
    m = make(data, client=FakeClient(), escalate_rate=0.25).fit(data.iloc[:60])
    m.predict(data.iloc[60:])
    prov = m.provenance()
    assert len(prov) == 20
    assert set(prov["経路"]) == {"llm", "高速"}
    assert prov["由来"].isin(["llm", "model", "human", "fallback"]).all()
    assert "信号" in prov.columns


def test_explain_はどちらの経路でも読める(data):
    m = make(data, client=FakeClient(), escalate_rate=0.25).fit(data.iloc[:60])
    m.predict(data.iloc[60:])
    llm_row = int(np.flatnonzero(m.selected_)[0])
    fast_row = int(np.flatnonzero(~m.selected_)[0])

    s1 = m.explain(llm_row)
    assert "経路 LLM" in s1 and "参照した類似事例" in s1
    s2 = m.explain(fast_row)
    assert "高速" in s2 and "統計モデルの予測" in s2


def test_examples_は_LLM_に回した行にしかない(data):
    m = make(data, client=FakeClient(), escalate_rate=0.25).fit(data.iloc[:60])
    m.predict(data.iloc[60:])
    llm_row = int(np.flatnonzero(m.selected_)[0])
    fast_row = int(np.flatnonzero(~m.selected_)[0])
    assert len(m.examples(llm_row)) == 3
    assert len(m.examples(fast_row)) == 0
    assert len(m.examples()) == 5 * 3


def test_route_は行ごとの経路表(data):
    m = make(data, client=FakeClient(), escalate_rate=0.25).fit(data.iloc[:60])
    m.predict(data.iloc[60:])
    r = m.route()
    assert len(r) == 20
    assert (r.loc[r["経路"] == "llm", "費用_usd"] > 0).all()
    assert (r.loc[r["経路"] == "高速", "費用_usd"] == 0).all()


def test_cost_に節約額が出る(data):
    m = make(data, client=FakeClient(), escalate_rate=0.25).fit(data.iloc[:60])
    m.predict(data.iloc[60:])
    c = m.cost()
    assert c["LLM に回した行数"] == 5
    assert c["LLM に回した割合"] == 0.25
    assert c["節約できた額_usd"] > 0


def test_predict_前に検査_API_を呼ぶと止める(data):
    m = make(data, client=FakeClient()).fit(data.iloc[:60])
    with pytest.raises(UnfoldError, match="predict を先に"):
        m.provenance()


def test_行数の違う_X_を渡すと取り違えを検知する(data):
    m = make(data, client=FakeClient(), escalate_rate=0.25).fit(data.iloc[:60])
    m.predict(data.iloc[60:])
    with pytest.raises(UnfoldError, match="行数"):
        m.confidence(data.iloc[:10])


# --- 曲線（精度・費用・レイテンシ）------------------------------------

def test_curve_は3つを同時に返す(data):
    m = make(data, client=FakeClient(), escalate_rate=0.25).fit(data.iloc[:60])
    c = m.curve(data.iloc[60:], rates=(0.0, 0.5, 1.0))
    assert list(c["割合"]) == [0.0, 0.5, 1.0]
    assert list(c["LLM に回す行数"]) == [0, 10, 20]
    assert set(["MAE", "費用_usd", "推定時間_秒"]) <= set(c.columns)
    assert c["費用_usd"].is_monotonic_increasing
    assert c["推定時間_秒"].is_monotonic_increasing


def test_curve_のあとに検査_API_を呼ぶと止める(data):
    """曲線を引くと内側が全行で動くので、直前の来歴は捨てる（取り違え防止）。"""
    m = make(data, client=FakeClient(), escalate_rate=0.25).fit(data.iloc[:60])
    m.predict(data.iloc[60:])
    m.curve(data.iloc[60:], rates=(0.0, 1.0))
    with pytest.raises(UnfoldError, match="predict を先に"):
        m.explain(0)


def test_report_が要約を返す(data):
    m = make(data, client=FakeClient(), escalate_rate=0.25).fit(data.iloc[:60])
    m.predict(data.iloc[60:])
    s = m.report()
    assert "LLM に回した:" in s and "費用:" in s


def test_explain_の行番号は呼び出し側に揃う(data):
    """内側の機能B は回した行だけを 0 から数え直すので、そのままでは食い違う。"""
    m = make(data, client=FakeClient(), escalate_rate=0.25).fit(data.iloc[:60])
    m.predict(data.iloc[60:])
    llm_row = int(np.flatnonzero(m.selected_)[1])   # 2番目に回った行
    s = m.explain(llm_row)
    assert s.startswith(f"[{llm_row}行目]")
    assert "行目]" not in s[s.index("\n"):]         # 内側の行番号が残っていない
