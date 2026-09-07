"""埋め込みに入れる前のテキストを作り分ける（主環境・隔離環境の両方から import する）。

狙い: 埋め込みの「似ている」が価格の近さを意味しない問題（`docs/2026-08-29-denoise.md`）が、
**入力テキストからノイズを落とすだけで改善するか**を測るための素材を作る。

作る版は4つ。V1〜V3 はどれも**車種に依存しない汎用ルール**であることが重要で、
「シエンタのグレードは G/Z/X」のような車種固有の知識は一切使わない。
使ってしまうと unfold 機能A（人手のルールを書かずに済ませる）の趣旨に反する。

  V0 生            … 現状。比較の基準
  V1 既出語の除去  … 店舗名・都道府県・車名など、**別の構造化列にすでに入っている
                     値**をテキストから消す。埋め込みが同じ情報を二重に持つ必要はなく、
                     しかも実測ではその書式の一致に反応していた（ガリバー系の例）
  V2 定数語の除去  … コーパス内の出現率が閾値以上のトークンを消す。
                     全行に出る語は行を区別しないので、ベクトルの向きを揃えるだけの重り。
                     人がリストを書くのではなく**データから決める**のが要点
  V3 先頭区切りまで … 「モデル名 … 装備の羅列」という構造を仮定し、最初の区切り記号までで
                     切る。カーセンサーのタイトル形式には依存するが車種には依存しない
"""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

# 全角スペース・スラッシュ・中黒などを区切りとみなす。装備の羅列はこれで並んでいる
SEP = r"[\s/・,、|｜]+"
IDEO_SPACE = "　"

# V3 で「ここから先は装備・宣伝文の羅列」と判断する区切り。半角スペースは
# 「シエンタ ハイブリッド 1.5 G」の中にも出るので区切りに含めない。
# Craigslist の model は "* " や " / " で宣伝文をつなぐので、そちらは呼び出し側で足す
BREAK_CHARS = IDEO_SPACE + "/・|｜"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace(IDEO_SPACE, " ")).strip()


def tokens(s: str) -> list[str]:
    return [t for t in re.split(SEP, s.replace(IDEO_SPACE, " ")) if t]


def v1_drop_known(text: pd.Series, others: list[pd.Series]) -> pd.Series:
    """別の構造化列にすでにある値をテキストから除く。

    others には店舗・都道府県・車名などの列を渡す。列の値そのものと、
    その値を分割したトークン（「ガリバー Ｒ１豊橋店」→「ガリバー」「Ｒ１豊橋店」）の
    両方を消す。部分一致で消すので、店舗名が装備語と重なると過剰に消える可能性はある。
    """
    out = []
    for i, s in enumerate(text.fillna("")):
        for col in others:
            val = str(col.iloc[i]) if pd.notna(col.iloc[i]) else ""
            if not val:
                continue
            for piece in sorted({val, *tokens(val)}, key=len, reverse=True):
                if len(piece) >= 2:
                    s = s.replace(piece, " ")
        out.append(_norm(s))
    return pd.Series(out, index=text.index)


def constant_tokens(text: pd.Series, threshold: float = 0.9) -> list[str]:
    """出現率が threshold 以上のトークン（＝行を区別しない定数語）を返す。"""
    n = len(text)
    df = Counter()
    for s in text.fillna(""):
        df.update(set(tokens(s)))
    return [w for w, k in df.items() if k / n >= threshold]


def v2_drop_constant(text: pd.Series, threshold: float = 0.9) -> tuple[pd.Series, list[str]]:
    stop = set(constant_tokens(text, threshold))
    out = text.fillna("").map(
        lambda s: _norm(" ".join(t for t in tokens(s) if t not in stop))
    )
    return out, sorted(stop)


def v3_head_segment(text: pd.Series, break_chars: str = BREAK_CHARS) -> pd.Series:
    """最初の区切り記号までを取る。無ければ全体をそのまま返す。

    break_chars はデータの書式に合わせて渡す（カーセンサーは全角スペース、
    Craigslist は "*" や "/"）。**車種には依存しない**ので機能A の前処理として通る。
    """
    pat = re.compile("[" + re.escape(break_chars) + "]")

    def head(s: str) -> str:
        m = pat.search(s)
        return _norm(s[: m.start()] if m else s)
    return text.fillna("").map(head)
