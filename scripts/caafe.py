"""CAAFE 相当の実装 — 「LLM に特徴量生成コードを書かせる」側の比較対象。

R2 の比較対象（`docs/2026-09-01-caafe.md`）。CAAFE（Context-Aware Automated Feature Engineering,
NeurIPS 2023, Hollmann et al. https://arxiv.org/abs/2305.03403）は、
**データセットの説明と列の様子を LLM に見せ、pandas のコードを書かせて
特徴量を足す**手法である。14 データセット中 11 で改善したと報告されている。

機能A（`unfold.Feature`）とは設計が正面から対立する。

| | CAAFE | 機能A |
|---|---|---|
| LLM の仕事 | 前処理コードを書く | 表記のゆれを吸収する（既定では埋め込み＋近傍） |
| 出力 | pandas のコード | 型のついた列 / 埋め込み列 |
| 実行時の LLM 呼び出し | 生成時だけ | 確信度の低い行だけ |
| 生成物の検査 | コードを読める | 来歴（`explain`）を読める |

**どちらが強いかは測らないと分からない**ので、同じデータ・同じ分割・
同じ下流モデル（LightGBM）で並べる。これが受け入れ基準 S4 にあたる。

## 本家との違い（意図的なもの）

- **下流モデルは LightGBM、指標は MAE。** 本家は TabPFN + ROC AUC だが、
  ここでは既存の測定（`docs/2026-08-29-feature-vehicles.md`）と土俵を
  揃えることを優先した。比較したいのは特徴量の良し悪しであって器ではない
- **LLM は Claude（`unfold.llm.ClaudeClient` 経由）。** 本家は OpenAI。
  ディスクキャッシュ・費用計上が乗るので、測り直しが無料になる
- 反復回数・採否の判定（検証 MAE が下がったコードだけ残す）は本家と同じ骨格

## 生成コードを実行することについて

**CAAFE は LLM が書いたコードをその場で実行する手法であり、それは避けられない。**
ここでは次の3つで囲っている。

1. 実行時の名前空間は `pd` / `np` / `df` だけ。組み込みは最小限に絞る
2. `import`・ファイル操作・ネットワークを含むコードは実行前に弾く
3. 例外が出たコードは「採用しない」で済ませる（測定を止めない）

それでも任意コード実行であることに変わりはないので、**このスクリプトは
自分のデータ・自分の環境でだけ動かすこと。**
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from unfold.llm import ClaudeClient

#: 生成コードに現れたら実行しない語。CAAFE の想定は「列を足す pandas 式」だけ
FORBIDDEN = re.compile(
    r"\b(import|__import__|open|exec|eval|compile|globals|locals|getattr|"
    r"setattr|delattr|input|breakpoint|os|sys|subprocess|shutil|socket|"
    r"requests|urllib|pickle|joblib)\b")

CODE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string",
                 "description": "追加する特徴量の短い名前（英数字と _ のみ）"},
        "reason": {"type": "string",
                   "description": "なぜこの特徴量が価格を説明すると考えるか。日本語で1〜2文"},
        "code": {"type": "string",
                 "description": "df に列を足す Python コード。import は書かない。"
                                "使えるのは df / pd / np のみ。df を返さず、その場で列を代入する"},
    },
    "required": ["name", "reason", "code"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
あなたは表形式データの特徴量エンジニアです。中古車の価格（回帰）を当てるための
特徴量を1つずつ提案してください。

守ること:

- 出力は **df に新しい列を足す Python コード**。`import` は書かない。
  使えるのは `df`（pandas.DataFrame）・`pd`・`np` だけ
- **1回の提案で足す列は1〜3列まで。** 少しずつ試して、効いたものだけ残す
- 目的変数の列は入力に含まれない。**価格そのものを使うコードは書けない**
- 欠損に強いコードにする（`fillna` や `.str` アクセサの `na=` を使う）
- 既に採用済みの特徴量と同じものを再提案しない
- 文字列列は自由記述で表記がゆれている。正規表現や部分一致で意味を取り出すのが有効

直前の提案が不採用だったときは、**同じ方向を繰り返さず別の切り口**を試すこと。
"""


def _describe(df: pd.DataFrame, target: str, n_rows: int = 8) -> str:
    """列の様子を LLM に見せる。**目的変数は渡さない。**"""
    cols = [c for c in df.columns if c != target]
    lines = [f"行数: {len(df):,}", "", "列:"]
    for c in cols:
        s = df[c]
        na = s.isna().mean()
        if pd.api.types.is_numeric_dtype(s):
            lines.append(f"- {c}（数値 / 欠損 {na:.0%}）"
                         f" 中央値 {s.median():,.1f} / 範囲 {s.min():,.0f}〜{s.max():,.0f}")
        else:
            top = s.value_counts().head(5)
            ex = " / ".join(f"{k}({v})" for k, v in top.items())
            lines.append(f"- {c}（文字列 / 欠損 {na:.0%} / 種類 {s.nunique():,}）"
                         f" よくある値: {ex}")
    # **長い文字列はそのまま貼らない。** 自由記述をそのまま載せると
    # プロンプトが桁で膨らむ（1行 3,000 字の列がある）
    head = df[cols].head(n_rows).copy()
    for c in head.columns:
        if not pd.api.types.is_numeric_dtype(head[c]):
            head[c] = head[c].astype(str).str.slice(0, 60)
    lines += ["", f"先頭 {n_rows} 行:", head.to_string()]
    return "\n".join(lines)


def is_safe(code: str) -> tuple[bool, str]:
    """実行してよいコードか。危ないものは理由つきで弾く。"""
    hit = FORBIDDEN.search(code)
    if hit:
        return False, f"禁止した語が含まれています: {hit.group(0)}"
    return True, ""


def run_code(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """生成コードを df のコピーに適用して返す。例外はそのまま投げる。"""
    out = df.copy()
    env = {"df": out, "pd": pd, "np": np,
           "__builtins__": {"len": len, "range": range, "min": min, "max": max,
                            "abs": abs, "round": round, "sum": sum, "str": str,
                            "int": int, "float": float, "bool": bool,
                            "list": list, "dict": dict, "set": set,
                            "sorted": sorted, "zip": zip, "enumerate": enumerate,
                            "True": True, "False": False, "None": None}}
    exec(compile(code, "<caafe>", "exec"), env)          # noqa: S102
    return env["df"]


@dataclass
class Step:
    """1回の提案とその結末。**採用しなかったものも残す**（過程が資産なので）。"""

    iteration: int
    name: str
    reason: str
    code: str
    score: float | None            # 検証 MAE。実行できなければ None
    accepted: bool
    note: str = ""
    cost: float = 0.0


@dataclass
class CaafeResult:
    """採用されたコードと、そこに至る過程。"""

    code: str                                    # 採用したコードを連結したもの
    steps: list[Step] = field(default_factory=list)
    base_score: float = float("nan")
    best_score: float = float("nan")

    @property
    def n_accepted(self) -> int:
        return sum(s.accepted for s in self.steps)

    def summary(self) -> str:
        lines = [f"検証 MAE {self.base_score:,.2f} → {self.best_score:,.2f}"
                 f"（{self.n_accepted}/{len(self.steps)} 件を採用）"]
        for s in self.steps:
            mark = "採用" if s.accepted else "却下"
            sc = f"{s.score:,.2f}" if s.score is not None else "実行不可"
            lines.append(f"  [{s.iteration}] {mark} {s.name}: {sc}"
                         + (f" — {s.note}" if s.note else ""))
        return "\n".join(lines)


def generate(train: pd.DataFrame, target: str,
             score_fn: Callable[[pd.DataFrame], float],
             client: ClaudeClient,
             description: str = "",
             n_iterations: int = 8,
             min_gain: float = 0.0,
             verbose: bool = True) -> CaafeResult:
    """CAAFE 本体。**検証スコアが下がったコードだけ残す。**

    Parameters
    ----------
    train:
        特徴量を作る対象（目的変数の列を含む）。
    target:
        目的変数の列名。LLM には見せない。
    score_fn:
        特徴量を足した DataFrame を受け取り、**小さいほど良いスコア**
        （ここでは検証 MAE）を返す関数。学習と検証はこの中で完結させる。
    description:
        データセットの説明。CAAFE の肝は「意味情報を使うこと」なので、
        列名だけでなく素性を書いて渡す。
    n_iterations:
        提案を何回もらうか。本家は 10 回程度。
    min_gain:
        これより大きく改善したときだけ採用する。0 なら「少しでも下がれば採用」。
    """
    accepted_code: list[str] = []
    base = score_fn(train)
    best = base
    steps: list[Step] = []
    if verbose:
        print(f"  初期スコア（検証 MAE）: {base:,.2f}")

    for it in range(1, n_iterations + 1):
        history = []
        for s in steps:
            mark = "採用" if s.accepted else "却下"
            sc = f"{s.score:,.2f}" if s.score is not None else "実行できず"
            history.append(f"- [{mark}] {s.name}（検証 MAE {sc}）\n```python\n"
                           f"{s.code}\n```")
        user = "\n".join([
            f"## データセット\n\n{description}\n",
            f"## 列の様子\n\n{_describe(train, target)}\n",
            f"## 現在の検証 MAE\n\n{best:,.2f}（小さいほど良い。"
            f"改善しない提案は採用されません）\n",
            "## これまでの提案\n\n"
            + ("\n".join(history) if history else "（まだありません）"),
            "\n次の特徴量を1つ提案してください。",
        ])
        ans = client.ask(SYSTEM_PROMPT, user, CODE_SCHEMA)
        if not ans.ok:
            steps.append(Step(it, "(応答なし)", "", "", None, False,
                              note=str(ans.error), cost=ans.cost))
            if verbose:
                print(f"  [{it}] LLM が答えられませんでした: {ans.error}")
            continue

        name = str(ans.data.get("name", f"f{it}"))
        code = str(ans.data.get("code", ""))
        reason = str(ans.data.get("reason", ""))

        ok, why = is_safe(code)
        if ok and target in code:
            # **目的変数を参照するコードは弾く。** 「価格から作った特徴量で
            # 価格を当てる」と検証スコアだけが良くなり、比較が壊れる
            ok, why = False, f"目的変数 {target!r} を参照しています"
        if not ok:
            steps.append(Step(it, name, reason, code, None, False, why, ans.cost))
            if verbose:
                print(f"  [{it}] 却下 {name}: {why}")
            continue

        trial = "\n".join(accepted_code + [code])
        try:
            new_train = run_code(train, trial)
            score = score_fn(new_train)
        except Exception as exc:                       # 生成コードは壊れうる
            steps.append(Step(it, name, reason, code, None, False,
                              f"{type(exc).__name__}: {exc}", ans.cost))
            if verbose:
                print(f"  [{it}] 却下 {name}: 実行できず（{type(exc).__name__}）")
            continue

        gain = best - score
        accept = gain > min_gain
        if accept:
            accepted_code.append(code)
            best = score
        steps.append(Step(it, name, reason, code, score, accept,
                          f"{gain:+,.2f}", ans.cost))
        if verbose:
            print(f"  [{it}] {'採用' if accept else '却下'} {name}: "
                  f"{score:,.2f}（{gain:+,.2f}）")

    return CaafeResult(code="\n".join(accepted_code), steps=steps,
                       base_score=base, best_score=best)
