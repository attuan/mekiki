"""事前スクリーニング（unfold.screen）のテスト。

fold を 2 に絞っている。判定ロジック（寄与率で verdict が変わるか）の
検証が目的で、精度の絶対値は問わないため。既定の 5 fold で回すと
テスト全体が数分かかり、実行しなくなる。

**LLM を呼ばないので APIキー不要**で、実行も速い。
人工データで「テキストが効く場合／効かない場合」を作り、判定が変わるかを見る。

実行: .venv/bin/python -m pytest tests -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from unfold import UnfoldError, screen


def make(n: int = 300, text_matters: bool = True, seed: int = 0) -> pd.DataFrame:
    """価格がテキストで決まる／決まらないデータを作り分ける。"""
    rng = np.random.default_rng(seed)
    age = rng.integers(1, 12, n)
    grade = rng.choice(["standard", "sport", "luxury"], n)
    bonus = {"standard": 0, "sport": 60, "luxury": 160}
    base = 300 - age * 15 + rng.normal(0, 5, n)
    price = base + (np.array([bonus[g] for g in grade]) if text_matters else 0)
    return pd.DataFrame({
        "車齢": age,
        # テキストにだけグレードが入っていて、構造化列には入っていない
        "説明": [f"model {g} edition" for g in grade],
        "地域": rng.choice(["A", "B", "C"], n),
        "価格": price,
    })


def test_テキストが効くデータは試す価値ありと判定される():
    r = screen(make(text_matters=True), target="価格", text="説明",
               numeric=["車齢"], categorical=["地域"], sample=None, n_splits=2)
    assert r.verdict == "試す価値あり"
    assert r.text_contribution > 0.10
    assert r.mae_with_text < r.mae_without_text


def test_テキストが効かないデータは効きにくいと判定される():
    r = screen(make(text_matters=False), target="価格", text="説明",
               numeric=["車齢"], categorical=["地域"], sample=None, n_splits=2)
    assert r.verdict == "効きにくい"
    assert r.text_contribution < 0.05


def test_レポートは日本語で読める():
    r = screen(make(), target="価格", text="説明", numeric=["車齢"],
               unit="万円", sample=None, n_splits=2)
    s = str(r)
    assert "テキスト寄与率" in s and "判定:" in s and "万円" in s
    # 暫定であることを必ず添える（2点しか根拠がないため）
    assert "暫定" in s


def test_無い列を指定するとエラー():
    with pytest.raises(UnfoldError):
        screen(make(), target="価格", text="存在しない列", numeric=["車齢"], n_splits=2)


def test_行数が少なすぎるとエラー():
    with pytest.raises(UnfoldError):
        screen(make(n=6), target="価格", text="説明", numeric=["車齢"],
               sample=None)


def test_長いテキストは上限で切って判定する():
    """文字 n-gram TF-IDF は長文で急激に重くなるので既定で切る。"""
    df = make()
    df["説明"] = df["説明"] + " " + "x" * 2000
    r = screen(df, target="価格", text="説明", numeric=["車齢"],
               sample=None, max_text_chars=100)
    assert r.max_text_chars == 100
    assert r.mean_text_chars > 500
    assert "先頭 100 文字" in str(r)
