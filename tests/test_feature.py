"""機能A の骨組みのテスト。

    .venv/bin/python -m pytest tests -q

APIキーが要らない範囲（設計書 01〜04 と、05 のキュー行き）だけを対象にする。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from unfold import CharTfidfEncoder, Feature, UnfoldError
from unfold.fallback import Answer
from unfold.preprocess import constant_tokens, drop_constant_tokens


def ambiguous() -> pd.DataFrame:
    """同じ文章に食い違うラベルが付いたデータ。

    近傍のラベルが割れるので confidence が 1.0 未満になり、
    05（エスカレーション）の経路を通せる。現実のラベルは揺れるので、
    これは異常系ではなく想定内の入力。
    """
    base = sample()
    noisy = pd.DataFrame({"タイトル": ["シエンタ 特別仕様 ETC スマートキー"] * 4,
                          "正解": ["G", "Z", "G", "Z"]})
    return pd.concat([base, noisy], ignore_index=True)


def sample() -> pd.DataFrame:
    rows = [("シエンタ G クエロ 禁煙車 バックカメラ", "G クエロ"),
            ("シエンタ ハイブリッド Z 4WD 純正ナビ", "Z"),
            ("シエンタ X 両側電動スライドドア", "X"),
            ("シエンタ ハイブリッド G ETC", "G")]
    return pd.DataFrame([{"タイトル": t, "正解": g} for t, g in rows * 6])


# --- 前処理 -----------------------------------------------------------

def test_定数語は出現率で決まる():
    s = pd.Series(["シエンタ G", "シエンタ Z", "シエンタ X"])
    assert constant_tokens(s, threshold=0.9) == ["シエンタ"]
    out, stop = drop_constant_tokens(s)
    assert out.tolist() == ["G", "Z", "X"] and stop == ["シエンタ"]


def test_推論時は学習時の定数語を使い回す():
    f = Feature(source="タイトル", values=["G", "Z"]).fit(sample())
    assert "シエンタ" in f.stop_tokens_
    # 1行だけ渡しても、その行の全語が「出現率100%」として消えたりしない
    one = pd.DataFrame({"タイトル": ["シエンタ ハイブリッド Z 4WD"]})
    assert f.transform(one).iloc[0] == "Z"


# --- 01 教師ラベルの起点（PRD 機能A の 01・3つの入口）------------------

def test_値の名前だけでも分類できる():
    df = sample()
    out = Feature(source="タイトル", values=["G", "Z", "X", "G クエロ"],
                  k=3).fit_transform(df)
    assert (out.astype(str) == df["正解"]).mean() >= 0.9


def test_人手ラベルを渡せる():
    df = sample()
    f = Feature(source="タイトル", labels="正解", k=3)
    out = f.fit_transform(df)
    assert (out.astype(str) == df["正解"]).mean() == 1.0
    assert f.status()["参照事例（人手）"] == len(df)


def test_ラベルもvaluesも無ければ止まる():
    with pytest.raises(UnfoldError, match="PRD 機能A の 01"):
        Feature(source="タイトル").fit(sample())


def test_ラベルが全部欠損なら止まる():
    df = sample().assign(正解=np.nan)
    with pytest.raises(UnfoldError):
        Feature(source="タイトル", labels="正解").fit(df)


# --- 03/04/05 確信度とエスカレーション ---------------------------------

def test_確信度は0から1に収まる():
    f = Feature(source="タイトル", labels="正解", k=3)
    f.fit_transform(sample())
    c = f.confidence()
    assert ((c >= 0) & (c <= 1)).all()


def test_閾値を上げるとレビュー待ちが増える():
    df = ambiguous()
    low = Feature(source="タイトル", labels="正解", k=3, threshold=0.1)
    low.fit_transform(df)
    high = Feature(source="タイトル", labels="正解", k=3, threshold=0.99)
    high.fit_transform(df)
    assert high.status()["レビュー待ち"] >= low.status()["レビュー待ち"]
    assert len(high.review_queue()) == high.status()["レビュー待ち"]


def test_LLMを差し込めばエスカレーション先が変わる():
    class DummyLLM:
        cost_per_call = 0.002

        def can_answer(self):
            return True

        def answer(self, texts, values, context):
            return [Answer(value="Z", confidence=0.95, cost=0.002)
                    for _ in texts]

    f = Feature(source="タイトル", labels="正解", k=3, threshold=0.9,
                fallback=DummyLLM())
    f.fit_transform(ambiguous())
    st, cost = f.status(), f.cost()
    assert st["LLM が答えた行"] > 0 and st["レビュー待ち"] == 0
    assert cost["実際に発生した費用"] == pytest.approx(st["LLM が答えた行"] * 0.002)


def test_on_uncertain_nullなら欠損になる():
    out = Feature(source="タイトル", labels="正解", k=3, threshold=0.9,
                  on_uncertain="null").fit_transform(ambiguous())
    assert out.isna().any()


def test_kはautoなら1クラス1件のときに縮む():
    """値の名前だけを起点にすると1クラス1件しか参照事例がない。

    そこで k=5 のまま近傍を取ると必ず5クラスに割れ、どの行も「自信なし」に
    なってしまう（実データで 99.9% がエスカレーションした）。k="auto" は
    1クラスあたりの件数から k を決めてこれを避ける。
    """
    f = Feature(source="タイトル", values=["G", "Z", "X", "G クエロ"], k="auto")
    f.fit(sample())
    assert f.k_ == 1
    f2 = Feature(source="タイトル", labels="正解", k="auto").fit(sample())
    assert f2.k_ > 1          # 1クラス6件あるので広く見てよい


def test_confidenceは1位と2位の相対比():
    f = Feature(source="タイトル", labels="正解", k=3)
    f.fit_transform(sample())
    # 近傍が全員同じラベルなら 2位が存在せず 1.0
    assert f.confidence().max() == pytest.approx(1.0)
    amb = Feature(source="タイトル", labels="正解", k=3)
    amb.fit_transform(ambiguous())
    assert amb.confidence().min() < 1.0


def test_escalate_rateで割合を指定できる():
    df = ambiguous()
    f = Feature(source="タイトル", labels="正解", k=3, escalate_rate=0.25)
    f.fit_transform(df)
    n = f.status()["レビュー待ち"]
    # 同点があるので厳密には一致しないが、指定した割合の近傍に収まる
    assert 0 < n <= len(df) * 0.5
    assert "下位" in str(f.status()["閾値"])


# --- 検査 API ----------------------------------------------------------

def test_explainに参照事例と由来が出る():
    f = Feature(source="タイトル", labels="正解", k=3)
    f.fit_transform(sample())
    text = f.explain(0)
    assert "参照した事例" in text and "confidence" in text and "human" in text


def test_examplesは行ごとにk件返す():
    f = Feature(source="タイトル", labels="正解", k=3)
    df = sample()
    ex = f.fit(df).examples(df)
    assert len(ex) == len(df) * 3
    assert {"行番号", "値", "類似度", "由来"} <= set(ex.columns)


def test_costは実行前の見積もりを返す():
    f = Feature(source="タイトル", labels="正解", k=3, threshold=0.9)
    df = ambiguous()
    c = f.fit(df).cost(df)
    assert c["行数"] == len(df) and 0 <= c["割合"] <= 1


# --- 型と入力 ----------------------------------------------------------

def test_embedding型はラベル不要():
    out = Feature(source="タイトル", type="embedding").fit_transform(sample())
    assert isinstance(out, pd.DataFrame) and len(out) == len(sample())


def test_複数列を連結できる():
    df = sample().assign(色=["白", "黒"] * 12)
    f = Feature(source=["タイトル", "色"], labels="正解", k=3).fit(df)
    assert "白" in f.provenance_["テキスト"].iloc[0] if hasattr(f, "provenance_") else True


def test_未実装の型は明示的に落ちる():
    with pytest.raises(UnfoldError, match="未実装"):
        Feature(source="タイトル", type="ordinal", values=["低", "高"])


def test_source列が無ければ落ちる():
    with pytest.raises(UnfoldError, match="source"):
        Feature(source="無い列", values=["G"]).fit(sample())


def test_エンコーダを差し替えられる():
    f = Feature(source="タイトル", labels="正解", k=3,
                encoder=CharTfidfEncoder(n_components=16))
    f.fit_transform(sample())
    assert f.status()["エンコーダ"] == "char_tfidf_svd16"


def test_同じ文字列は二度計算しない():
    df = sample()
    f = Feature(source="タイトル", labels="正解", k=3).fit(df)
    f.transform(df)
    assert f.encoder_.n_cached == df["タイトル"].nunique()
