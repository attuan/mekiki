"""機能B（LLMPredictor）のテスト。

**APIキー無しで全部通る**ようにしてある。LLM を呼ぶ部分は `ClaudeClient.ask`
だけなので、そこを差し替えた偽クライアントで回す。こうしておくと
CI でも、キーが切れているときでも、パイプライン側の壊れを検出できる。

実行: .venv/bin/python -m pytest tests -q
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from unfold import ColumnSpec, LLMPredictor, UnfoldError
from unfold.llm import ClaudeClient, LLMAnswer


class FakeClient(ClaudeClient):
    """HTTP を出さずに、受け取ったプロンプトを記録して定型の答えを返す。"""

    def __init__(self, price: float = 200.0, fail_rows: set[int] | None = None,
                 **kw):
        kw.setdefault("cache_dir", None)   # ディスクを汚さない
        kw.setdefault("api_key", "test-key")
        super().__init__(**kw)
        self.price = price
        self.fail_rows = fail_rows or set()
        self.prompts: list[str] = []

    def ask(self, system, user, schema) -> LLMAnswer:
        i = len(self.prompts)
        self.prompts.append(user)
        if i in self.fail_rows:
            self.usage.calls += 1
            self.usage.errors += 1
            return LLMAnswer(data={}, error="わざと失敗させた")
        self.usage.calls += 1
        self.usage.cost += 0.001
        return LLMAnswer(data={"price": self.price, "confidence": 0.8,
                               "reason": "テスト"},
                         input_tokens=100, output_tokens=20, cost=0.001)


@pytest.fixture
def data() -> pd.DataFrame:
    """価格が車齢と走行距離でほぼ決まる、小さな人工データ。"""
    rng = np.random.default_rng(0)
    n = 60
    age = rng.integers(1, 10, n)
    km = rng.integers(5_000, 120_000, n)
    grade = rng.choice(["G", "X", "Z"], n)
    return pd.DataFrame({
        "車齢": age,
        "走行距離_km": km,
        "修復歴あり": rng.random(n) < 0.2,
        "グレード名": grade,
        "装備テキスト": [f"ナビ ETC {g}グレード 装備{i%5}" for i, g in enumerate(grade)],
        "価格": 250 - age * 12 - km / 8000 + rng.normal(0, 3, n),
    })


def make(data, **kw) -> LLMPredictor:
    return LLMPredictor(
        target="価格", unit="万円",
        numeric=["車齢", "走行距離_km"], boolean=["修復歴あり"],
        categorical=["グレード名"], text="装備テキスト",
        n_examples=3, **kw)


# --- 基本の形 ---------------------------------------------------------

def test_fit_predict_の形(data):
    client = FakeClient()
    m = make(data, client=client).fit(data.iloc[:40])
    pred = m.predict(data.iloc[40:])
    assert pred.shape == (20,)
    assert np.allclose(pred, 200.0)
    assert len(client.prompts) == 20


def test_y_を省略すると_target_列を使う(data):
    m = make(data, client=FakeClient()).fit(data.iloc[:40])
    assert len(m.y_) == 40
    assert np.allclose(m.y_, data["価格"].to_numpy()[:40])


def test_fit_前に_predict_すると_エラー(data):
    with pytest.raises(UnfoldError):
        make(data, client=FakeClient()).predict(data)


def test_無い列を指定すると_エラー(data):
    m = LLMPredictor(target="価格", numeric=["存在しない列"], client=FakeClient())
    with pytest.raises(UnfoldError):
        m.fit(data)


# --- 証拠が本当にプロンプトに入っているか -----------------------------

def test_プロンプトに統計モデルの予測が入る(data):
    client = FakeClient()
    m = make(data, client=client).fit(data.iloc[:40])
    m.predict(data.iloc[40:42])
    p = client.prompts[0]
    assert "LightGBM" in p and "XGBoost" in p
    assert "証拠1" in p and "証拠2" in p


def test_プロンプトに類似事例が_n_examples_件入る(data):
    client = FakeClient()
    m = make(data, client=client).fit(data.iloc[:40])
    m.predict(data.iloc[40:41])
    p = client.prompts[0]
    assert "事例1" in p and "事例3" in p and "事例4" not in p


def test_類似事例は訓練データからだけ引かれる(data):
    """テストデータの正解を証拠に混ぜたら評価が成り立たない。"""
    m = make(data, client=FakeClient()).fit(data.iloc[:40])
    m.predict(data.iloc[40:50])
    ex = m.examples()
    assert ex["訓練行"].max() < 40


# --- 来歴（PRD「来歴（provenance）と検査 API」）------------------------

def test_explain_が証拠を辿れる(data):
    m = make(data, client=FakeClient()).fit(data.iloc[:40])
    m.predict(data.iloc[40:42])
    s = m.explain(0)
    assert "LightGBM" in s and "類似事例" in s and "confidence" in s


def test_confidence_が行数ぶん返る(data):
    m = make(data, client=FakeClient()).fit(data.iloc[:40])
    m.predict(data.iloc[40:50])
    c = m.confidence()
    assert len(c) == 10 and (c == 0.8).all()


def test_cost_に費用と単価が入る(data):
    m = make(data, client=FakeClient()).fit(data.iloc[:40])
    m.predict(data.iloc[40:50])
    c = m.cost()
    assert c["予測した行数"] == 10
    assert c["費用_usd"] == pytest.approx(0.01, abs=1e-6)
    assert c["1行あたりの費用_usd"] == pytest.approx(0.001, abs=1e-6)


def test_provenance_が全行ぶん返る(data):
    m = make(data, client=FakeClient()).fit(data.iloc[:40])
    m.predict(data.iloc[40:50])
    prov = m.provenance()
    assert len(prov) == 10
    assert set(prov["由来"]) == {"llm"}
    assert "証拠_LightGBM" in prov.columns


# --- 失敗したときの振る舞い -------------------------------------------

def test_答えられなかった行は統計モデルで埋まる(data):
    client = FakeClient(fail_rows={0, 3})
    m = make(data, client=client).fit(data.iloc[:40])
    pred = m.predict(data.iloc[40:50])
    prov = m.provenance()
    assert list(prov["由来"]).count("fallback") == 2
    # 代替に使ったのは1つ目のモデル（LightGBM）の予測値
    for i in (0, 3):
        assert pred[i] == pytest.approx(prov.loc[i, "証拠_LightGBM"])
    assert not np.isnan(pred).any()


def test_fallback_error_なら例外を投げる(data):
    client = FakeClient(fail_rows={0})
    m = make(data, client=client, fallback="error").fit(data.iloc[:40])
    with pytest.raises(UnfoldError):
        m.predict(data.iloc[40:50])


# --- キャッシュ -------------------------------------------------------

def test_同じプロンプトは2度課金されない(tmp_path, data):
    """交差検証や測り直しで同じ行を何度も引くので、ここが効かないと破産する。"""
    calls = {"n": 0}

    class Counting(FakeClient):
        def ask(self, system, user, schema):
            # 親の ask を通さず、実際のキャッシュ機構だけを試す
            key = self._key(system, user, schema)
            cached = self._read_cache(key)
            if cached is not None:
                self.usage.calls += 1
                self.usage.cache_hits += 1
                return LLMAnswer(data=cached["data"], from_cache=True)
            calls["n"] += 1
            data_ = {"price": 200.0, "confidence": 0.8, "reason": "テスト"}
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
    assert calls["n"] == 10                      # 1回も呼び足していない
    assert client2.usage.cache_hits == 10
    assert client2.usage.cost == 0.0


# --- 近傍検索の recipe（docs/2026-08-29-denoise.md）--------------------

def test_数値距離を混ぜると近傍の車齢が近づく(data):
    """意味的類似度だけだと年式も走行距離も違う車が「似ている」と出てくる。"""
    train, test = data.iloc[:40], data.iloc[40:]
    spec = ColumnSpec(numeric=["車齢", "走行距離_km"], text="装備テキスト")

    from unfold.predictor import NeighbourIndex
    gaps = {}
    for w in (0.0, 1.0):
        idx = NeighbourIndex(spec, k=3, w=w).fit(train, train["価格"].to_numpy())
        nb, _ = idx.query(test)
        age_tr = train["車齢"].to_numpy()
        age_te = test["車齢"].to_numpy()
        gaps[w] = float(np.mean(np.abs(age_tr[nb] - age_te[:, None])))
    assert gaps[1.0] < gaps[0.0]


def test_近傍は類似度の降順で並ぶ(data):
    m = make(data, client=FakeClient()).fit(data.iloc[:40])
    m.predict(data.iloc[40:45])
    ex = m.examples(i=0)
    assert list(ex["順位"]) == [1, 2, 3]
    assert list(ex["類似度"]) == sorted(ex["類似度"], reverse=True)


# --- スキーマ ---------------------------------------------------------

def test_回答スキーマがJSONとして妥当(data):
    from unfold.predictor import ANSWER_SCHEMA
    json.dumps(ANSWER_SCHEMA)
    assert ANSWER_SCHEMA["additionalProperties"] is False
    assert set(ANSWER_SCHEMA["required"]) == {"price", "confidence", "reason"}


# --- 機能A の LLM フォールバック（設計書05）--------------------------

class FakeClassifyClient(FakeClient):
    """分類フォールバック用。返す値を差し替えられるようにしたもの。"""

    def __init__(self, value="G", **kw):
        super().__init__(**kw)
        self.value = value

    def ask(self, system, user, schema) -> LLMAnswer:
        self.prompts.append(user)
        self.usage.calls += 1
        self.usage.cost += 0.002
        return LLMAnswer(data={"value": self.value, "confidence": 0.9,
                               "reason": "テスト"}, cost=0.002)


def test_フォールバックは候補外の値を採用しない():
    """勝手な値が特徴量に混ざると、下流のカテゴリ列が壊れる。"""
    from unfold.fallback import ClaudeFallback
    fb = ClaudeFallback(client=FakeClassifyClient(value="存在しないグレード"))
    out = fb.answer(["ナビ付き"], ["G", "Z"], [{}])
    assert out[0].value is None
    assert out[0].confidence == 0.0


def test_フォールバックは候補内の値なら採用してキューにも積む():
    from unfold.fallback import ClaudeFallback
    fb = ClaudeFallback(client=FakeClassifyClient(value="Z"))
    out = fb.answer(["ナビ付き"], ["G", "Z"], [{}])
    assert out[0].value == "Z" and out[0].origin == "llm"
    # 設計書: 答えたものは教師ラベル候補としてキューに積む（能動学習の片側）
    assert len(fb.queued) == 1 and fb.queued[0]["LLMの答え"] == "Z"


def test_フォールバックのプロンプトに近傍事例が入る():
    """Feature._escalate が渡す形（キー "examples"）を読めているか。"""
    from unfold.fallback import ClaudeFallback
    client = FakeClassifyClient(value="G")
    fb = ClaudeFallback(client=client)
    ctx = [{"examples": [{"テキスト": "G グレードの車", "値": "G",
                          "類似度": 0.82, "由来": "human"}]}]
    fb.answer(["ナビ付き"], ["G", "Z"], ctx)
    assert "G グレードの車" in client.prompts[0]
    assert "0.820" in client.prompts[0]


def test_キー無しのフォールバックは呼べない():
    from unfold.fallback import ClaudeFallback
    from unfold.llm import ClaudeClient
    fb = ClaudeFallback(client=ClaudeClient(api_key="", cache_dir=None))
    assert fb.can_answer() is False
    with pytest.raises(NotImplementedError):
        fb.answer(["x"], ["G"], [{}])


def test_壊れたキャッシュは無視して呼び直す(tmp_path, data):
    """1ファイルの破損で数百行の測定が落ちないこと。"""
    client = FakeClient(cache_dir=tmp_path)
    key = client._key("s", "u", {"type": "object"})
    path = client._cache_path(key)
    for broken in ("{壊れたJSON", '{"data": "辞書ではない"}', '"文字列"'):
        path.write_text(broken, encoding="utf-8")
        assert client._read_cache(key) is None


def test_違う行数のXを検査APIに渡すと止まる(data):
    """別の X の来歴を黙って返すと、間違った根拠で判断することになる。"""
    m = make(data, client=FakeClient()).fit(data.iloc[:40])
    m.predict(data.iloc[40:50])
    m.confidence(data.iloc[40:50])            # 同じ行数なら通る
    with pytest.raises(UnfoldError):
        m.confidence(data.iloc[40:45])
    with pytest.raises(UnfoldError):
        m.cost(data.iloc[40:45])


def test_NeighbourModel_を単体で使っても索引が張られる(data):
    from unfold.predictor import NeighbourIndex, NeighbourModel
    spec = ColumnSpec(numeric=["車齢"], text="装備テキスト")
    m = NeighbourModel("近傍", NeighbourIndex(spec, k=3))
    train = data.iloc[:40].reset_index(drop=True)
    m.fit(train, train["価格"].to_numpy())
    pred = m.predict(data.iloc[40:45].reset_index(drop=True))
    assert pred.shape == (5,) and np.isfinite(pred).all()


# --- 自由記述と、正解の漏れ対策 ---------------------------------------

def test_金額の伏字():
    """出品テキストに売り値が書いてあると、予測ではなく読み取りになる。"""
    from unfold.predictor import mask_amounts
    assert "$5,900" not in mask_amounts("Lowered! $5,900 plus fees")
    assert "12500" not in mask_amounts("asking 12500 dollars")
    assert "〈金額〉" in mask_amounts("price: $6,999")
    # 走行距離・年式も巻き込むが、構造化列として別に渡しているので損はない
    assert "222,617" not in mask_amounts("222,617 miles")
    assert "2008" not in mask_amounts("2008 Toyota Sienna")


def test_伏字は車種の数字を残す():
    """F-150 や Model 3 が消えると、肝心の車種情報が失われる。"""
    from unfold.predictor import mask_amounts
    out = mask_amounts("Ford F-150 Raptor 5.7L V8, Tesla Model 3")
    assert "F-150" in out and "5.7L" in out and "Model 3" in out


def test_自由記述は査定対象にだけ載り事例には載らない(data):
    """5事例ぶん貼るとプロンプトが桁で膨らむ（PRD「信頼度ルーティング」）。"""
    d = data.assign(説明文=[f"この車は良好です 走行{i}00 マイル" for i in range(len(data))])
    client = FakeClient()
    m = LLMPredictor(target="価格", unit="万円", numeric=["車齢"],
                     text="装備テキスト", long_text="説明文",
                     n_examples=3, client=client).fit(d.iloc[:40])
    m.predict(d.iloc[40:41])
    p = client.prompts[0]
    # 査定対象の説明文は載る
    assert "この車は良好です" in p
    # 事例側には載らない（既定 long_text_example_chars=0）
    assert p.count("この車は良好です") == 1


def test_自由記述は上限で切られ切ったことが明示される(data):
    d = data.assign(説明文=["あ" * 5000] * len(data))
    client = FakeClient()
    spec = ColumnSpec(numeric=["車齢"], text="装備テキスト",
                      long_text="説明文", long_text_chars=100)
    m = LLMPredictor(target="価格", spec=spec, n_examples=2,
                     client=client).fit(d.iloc[:40])
    m.predict(d.iloc[40:41])
    p = client.prompts[0]
    assert "文字を省略" in p
    assert p.count("あ") <= 200      # 5000 文字そのままは入っていない


def test_伏字は切ることができる(data):
    d = data.assign(説明文=["asking $9,999"] * len(data))
    client = FakeClient()
    spec = ColumnSpec(numeric=["車齢"], text="装備テキスト", long_text="説明文",
                      mask_amounts_in_long_text=False)
    m = LLMPredictor(target="価格", spec=spec, n_examples=2,
                     client=client).fit(d.iloc[:40])
    m.predict(d.iloc[40:41])
    assert "$9,999" in client.prompts[0]
