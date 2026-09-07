"""重複レコードの検知（unfold/leakage.py）のテスト。

守りたい性質は3つ。

1. **重複を数え間違えないこと。** 「2件目以降の行数」を数えるので、
   3件ある組は 2 行が重複という勘定になる
2. **train と test にまたがる重複を見つけること。** これが実害そのもの
3. **止めずに警告すること。** 重複が意図的なこともあるので例外にはしない

実行: .venv/bin/python -m pytest tests -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from unfold import (
    LLMPredictor,
    UnfoldWarning,
    check_duplicates,
    check_overlap,
)
from unfold.llm import ClaudeClient, LLMAnswer


class FakeClient(ClaudeClient):
    def __init__(self, **kw):
        kw.setdefault("cache_dir", None)
        kw.setdefault("api_key", "test-key")
        super().__init__(**kw)

    def ask(self, system, user, schema) -> LLMAnswer:
        self.usage.calls += 1
        return LLMAnswer(data={"price": 200.0, "confidence": 0.8,
                               "reason": "テスト"}, cost=0.0)


@pytest.fixture
def data() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 40
    age = rng.integers(1, 10, n)
    km = rng.integers(5_000, 120_000, n)
    return pd.DataFrame({
        "車齢": age,
        "走行距離_km": km,
        "グレード名": rng.choice(["G", "X", "Z"], n),
        "装備テキスト": [f"ナビ ETC 装備{i}" for i in range(n)],
        "価格": 250 - age * 12 - km / 8000 + rng.normal(0, 3, n),
    })


# --- 1つの表の中の重複 ------------------------------------------------

def test_重複が無ければ_ok(data):
    r = check_duplicates(data)
    assert r.ok and r.n_duplicate_rows == 0
    assert "重複なし" in str(r)


def test_2件目以降を数える(data):
    # 0行目を2回足す → 同じ内容が3件 = 重複は2行
    df = pd.concat([data, data.iloc[[0]], data.iloc[[0]]], ignore_index=True)
    r = check_duplicates(df)
    assert r.n_duplicate_rows == 2
    assert r.n_groups == 1
    assert r.largest_group == 3
    assert not r.ok
    assert "重複が 2 行あります" in str(r)


def test_照合する列を絞れる(data):
    """VIN のような識別子が1列あるなら、それだけで照合するのが正確。"""
    df = data.copy()
    df["車台番号"] = [f"VIN{i:03d}" for i in range(len(df))]
    df.loc[1, "車台番号"] = "VIN000"          # 中身は違うが同じ車
    assert check_duplicates(df).ok             # 全列で見れば別物
    assert check_duplicates(df, keys=["車台番号"]).n_duplicate_rows == 1


def test_無視する列を指定できる(data):
    """地域だけ違う同じ車を、同じものとして数える。"""
    df = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    df["地域"] = [f"地域{i}" for i in range(len(df))]
    assert check_duplicates(df).ok
    assert check_duplicates(df, ignore=["地域"]).n_duplicate_rows == 1


def test_存在しない列を指定したら止める(data):
    with pytest.raises(KeyError):
        check_duplicates(data, keys=["ありません"])


# --- train と test にまたがる重複 --------------------------------------

def test_重なりが無ければ_ok(data):
    r = check_overlap(data.iloc[:30], data.iloc[30:])
    assert r.ok
    assert "重なりなし" in str(r)


def test_train_と同じ行が_test_にあると見つかる(data):
    train = data.iloc[:30]
    test = pd.concat([data.iloc[30:], train.iloc[[0, 1]]], ignore_index=True)
    r = check_overlap(train, test)
    assert r.n_leaked_rows == 2
    assert r.examples == [10, 11]
    assert not r.ok
    assert "答えを知っている" in str(r)


def test_目的変数が_test_に無くても照合できる(data):
    train = data.iloc[:30]
    test = pd.concat([data.iloc[30:], train.iloc[[0]]], ignore_index=True)
    r = check_overlap(train, test.drop(columns=["価格"]))
    assert r.n_leaked_rows == 1
    assert "価格" not in r.columns


def test_共通の列が無ければ止める(data):
    with pytest.raises(KeyError):
        check_overlap(data, pd.DataFrame({"別の列": [1, 2]}))


# --- 機能B に組み込まれているか ---------------------------------------

def make(data, **kw) -> LLMPredictor:
    return LLMPredictor(target="価格", unit="万円",
                        numeric=["車齢", "走行距離_km"],
                        categorical=["グレード名"], text="装備テキスト",
                        n_examples=3, client=FakeClient(), **kw)


def test_訓練データの重複で警告が出る(data):
    dup = pd.concat([data, data.iloc[[0, 1]]], ignore_index=True)
    with pytest.warns(UnfoldWarning, match="重複が 2 行"):
        make(dup).fit(dup)


def test_train_と_test_の重なりで警告が出る(data):
    train = data.iloc[:30].reset_index(drop=True)
    test = pd.concat([data.iloc[30:], train.iloc[[0]]], ignore_index=True)
    m = make(data).fit(train)
    with pytest.warns(UnfoldWarning, match="train と test"):
        m.predict(test)


def test_警告は出るが止まらない(data):
    """重複が意図的なこともあるので、例外にはしない。"""
    train = data.iloc[:30].reset_index(drop=True)
    test = pd.concat([data.iloc[30:], train.iloc[[0]]], ignore_index=True)
    m = make(data).fit(train)
    with pytest.warns(UnfoldWarning):
        pred = m.predict(test)
    assert len(pred) == len(test)


def test_check_leakage_False_で切れる(data, recwarn):
    dup = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    make(dup, check_leakage=False).fit(dup)
    assert not [w for w in recwarn if issubclass(w.category, UnfoldWarning)]


def test_きれいなデータでは警告が出ない(data, recwarn):
    train, test = data.iloc[:30].reset_index(drop=True), data.iloc[30:]
    make(data).fit(train).predict(test)
    assert not [w for w in recwarn if issubclass(w.category, UnfoldWarning)]


# --- ルーティング側は全行を1回だけ見る --------------------------------

def test_ルーティングでも重なりを検知する(data):
    from unfold import AdaptivePredictor

    train = data.iloc[:30].reset_index(drop=True)
    test = pd.concat([data.iloc[30:], train.iloc[[0]]], ignore_index=True)
    m = AdaptivePredictor(predictor=make(data), escalate_rate=0.5).fit(train)
    with pytest.warns(UnfoldWarning, match="train と test") as rec:
        m.predict(test)
    # 内側の機能B が回した行だけをもう一度見て二重に警告しないこと
    assert len([w for w in rec if issubclass(w.category, UnfoldWarning)]) == 1
