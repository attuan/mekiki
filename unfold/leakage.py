"""重複レコードの検知（PRD「非機能要件」のリーク防止）。

**同じ車が train と test の両方に入っていると、予測ではなく答えの読み取りになる。**
Craigslist のデータは同じ車が複数の地域に出稿されており、同一 VIN が最大 261 件、
価格まで同一だった。フィルタ後 346,371 行のうち 145,997 行（42%）が重複で、
これを残したままランダム分割の交差検証をすると **MAE が 4.2%・R² が 0.03 だけ
良く見える**（`scripts/check_duplicate_leak.py` の実測。R² 0.880 → 0.914）。

しかも **非構造テキストを特徴量にするほど水増しが大きくなる。** unfold は
まさにテキストを特徴量にする道具なので、利用者の心得に任せず**製品側で検知する**
ことにした（`docs/2026-09-01-description-leak.md`）。同じ理屈で自由記述の金額はすでに既定で伏せている
（`mask_amounts`）。こちらは伏せると学習そのものが壊れるので、**警告にとどめる。**

    from unfold import check_duplicates, check_overlap

    print(check_duplicates(df))              # 1つの表の中の重複
    print(check_overlap(train, test))        # train と test にまたがる重複

`Feature` と `LLMPredictor` は fit / predict のときにこれを自動で呼び、
見つかれば `UnfoldWarning` を出す（`check_leakage=False` で止められる）。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd

from unfold.errors import UnfoldWarning

#: 重複が何割を超えたら警告するか。0 にすると1件でも警告する
DEFAULT_WARN_RATE = 0.0


def _keys(df: pd.DataFrame, keys: Sequence[str] | None,
          ignore: Sequence[str] | None) -> list[str]:
    """照合に使う列を決める。既定は「全列一致」。

    id や地域のように**同じ車でも値が違う列**は `ignore` で外す。
    Craigslist なら `ignore=["id", "region", "state"]` が実態に合う。
    """
    cols = list(keys) if keys else list(df.columns)
    if ignore:
        cols = [c for c in cols if c not in set(ignore)]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"照合に使う列がありません: {missing}")
    return cols


def _fingerprint(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    """行の内容から作る指紋。数万行でも一瞬で終わる。"""
    return pd.util.hash_pandas_object(df[list(cols)].astype(str), index=False)


@dataclass
class DuplicateReport:
    """1つの表の中の重複。そのまま print すると日本語で読める。"""

    n_rows: int
    n_duplicate_rows: int             # 2件目以降の行数（＝落とせる行数）
    n_groups: int                     # 2件以上ある組の数
    largest_group: int
    columns: list[str] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.n_duplicate_rows / self.n_rows if self.n_rows else 0.0

    @property
    def ok(self) -> bool:
        return self.n_duplicate_rows == 0

    def __str__(self) -> str:
        if self.ok:
            return (f"重複なし（{self.n_rows:,} 行 / "
                    f"{len(self.columns)} 列で照合）")
        return "\n".join([
            f"**重複が {self.n_duplicate_rows:,} 行あります**"
            f"（{self.n_rows:,} 行中 {self.rate:.1%} / "
            f"{self.n_groups:,} 組 / 最大の組 {self.largest_group:,} 件）",
            "  同じ内容の行が複数あると、ランダム分割の交差検証で"
            "同じレコードが train と test の両方に入ります。",
            "  そうなると予測ではなく答えの読み取りになり、"
            "スコアが実力より良く出ます（実測で R² が 0.880 → 0.914）。",
            "  → 1件1行に潰してから評価してください。",
        ])


@dataclass
class OverlapReport:
    """train と test にまたがる重複。**こちらが実害そのもの。**"""

    n_train: int
    n_test: int
    n_leaked_rows: int                # train にも同じ内容がある test の行数
    columns: list[str] = field(default_factory=list)
    examples: list[int] = field(default_factory=list)   # test 側の行番号

    @property
    def rate(self) -> float:
        return self.n_leaked_rows / self.n_test if self.n_test else 0.0

    @property
    def ok(self) -> bool:
        return self.n_leaked_rows == 0

    def __str__(self) -> str:
        if self.ok:
            return (f"train と test の重なりなし"
                    f"（train {self.n_train:,} / test {self.n_test:,} 行）")
        head = ", ".join(str(i) for i in self.examples[:5])
        return "\n".join([
            f"**test の {self.n_leaked_rows:,} 行（{self.rate:.1%}）が "
            f"train と同じ内容です**（{len(self.columns)} 列で照合 / "
            f"例: {head} 行目）",
            "  その行は答えを知っている状態で解いていることになり、"
            "評価が実力より良く出ます。",
            "  → 分割の前に1件1行へ潰すか、同一グループが分割をまたがない"
            "ようにしてください（GroupKFold など）。",
        ])


def check_duplicates(df: pd.DataFrame, keys: Sequence[str] | None = None,
                     ignore: Sequence[str] | None = None) -> DuplicateReport:
    """1つの表の中に、内容が同じ行がいくつあるかを数える。

    Parameters
    ----------
    keys:
        照合に使う列。省略すると全列。VIN のような識別子が1列あるなら
        `keys=["VIN"]` が最も正確（実測では VIN の重複が最大 261 件）。
    ignore:
        照合から外す列。id・地域など**同じ車でも値が違う列**を入れる。
    """
    cols = _keys(df, keys, ignore)
    fp = _fingerprint(df, cols)
    counts = fp.value_counts()
    dup_groups = counts[counts > 1]
    return DuplicateReport(
        n_rows=len(df),
        n_duplicate_rows=int(dup_groups.sum() - len(dup_groups)),
        n_groups=int(len(dup_groups)),
        largest_group=int(dup_groups.max()) if len(dup_groups) else 1,
        columns=cols)


def check_overlap(train: pd.DataFrame, test: pd.DataFrame,
                  keys: Sequence[str] | None = None,
                  ignore: Sequence[str] | None = None) -> OverlapReport:
    """test の行のうち、train にも同じ内容があるものを数える。

    照合に使う列は train と test の**両方にある列**に限る
    （test には目的変数が無いことがあるため）。
    """
    cols = [c for c in _keys(train, keys, ignore) if c in test.columns]
    if not cols:
        raise KeyError("train と test に共通の列がありません。")
    known = set(_fingerprint(train, cols).tolist())
    fp = _fingerprint(test, cols)
    hit = [i for i, v in enumerate(fp.tolist()) if v in known]
    return OverlapReport(n_train=len(train), n_test=len(test),
                         n_leaked_rows=len(hit), columns=cols, examples=hit)


def warn_if_leaky(report: DuplicateReport | OverlapReport,
                  where: str = "", rate: float = DEFAULT_WARN_RATE) -> bool:
    """報告が問題ありなら警告を出す。出したら True。

    **例外にはしない。** 重複が意図的なこともあるし、止めてしまうと
    「とりあえず動かす」ができなくなる。気づけることが目的である。
    """
    if report.ok or report.rate <= rate:
        return False
    prefix = f"[{where}] " if where else ""
    warnings.warn(prefix + str(report), UnfoldWarning, stacklevel=3)
    return True
