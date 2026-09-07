"""機能B を使う価値があるデータかを、LLM を呼ぶ前に見積もる。

## なぜ要るのか

同じ実装・同じ日・同じ LLM で、機能B はシエンタ（単一車種）では負け、
Craigslist（複数車種）では有意に勝った。差はデータの性質にあった。

| | シエンタ | Craigslist |
|---|---|---|
| テキストを木に入れた効果 | −5.2% | −16.5% |
| 機能B の効果（対 公平線） | +1.1%（負け） | −8.8%（勝ち） |

**テキストがそもそも価格を説明しない場所では、LLM に読ませても効かない。**
だから「まず全行に LLM を投げてみる」のは、時間と費用の両方で高くつく。

`screen()` は **LLM を1回も呼ばずに**、そのデータでテキストが効くかを測る。
判定は木モデル2本の比較だけなので、数十秒で終わり費用は 0 である。

    from unfold import screen

    report = screen(df, target="price", text="model",
                    numeric=["age", "odometer"],
                    categorical=["manufacturer", "state"])
    print(report)

## 何を測っているのか

同じ LightGBM を2本、**テキスト列を入れる/入れない**だけ変えて交差検証する。
その差（`テキスト寄与率`）が、そのデータで非構造テキストが持っている情報量にあたる。

**この指標は機能B の効果そのものではない。** 機能B は「文字 TF-IDF より LLM の
ほうがうまくテキストを読める」ぶんだけ上積みする仕組みなので、
**TF-IDF ですでに効いている場所でしか上積みは起きない**、という前提を確かめている。

実測との対応（2026-09-01 時点、2データセットのみ）:

| テキスト寄与率 | 機能B の結果 |
|---|---|
| 5.2%（シエンタ） | 負け（有意でない） |
| 16.5%（Craigslist） | 勝ち（p = 0.009） |

**2点しかないので閾値は暫定である。** 既定の 10% は2点の間を取っただけで、
根拠のある値ではない。データが増えたら引き直すこと。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from unfold.errors import UnfoldError

SEED = 42

#: テキスト寄与率がこれ以上なら機能B を試す価値がある、という暫定の線。
#: **2データセットの実測の間を取っただけで、根拠のある値ではない。**
DEFAULT_THRESHOLD = 0.10


@dataclass
class ScreeningReport:
    """`screen()` の結果。そのまま print すると日本語で読める。"""

    n_rows: int
    unit: str
    mae_without_text: float
    mae_with_text: float
    text_contribution: float          # (無し − 有り) / 無し
    threshold: float
    text_column: str
    n_unique_text: int
    #: 判定。"試す価値あり" / "効きにくい" / "判断できない"
    verdict: str
    #: 判定に使ったテキストの上限文字数（0 なら切っていない）
    max_text_chars: int = 0
    #: 元のテキストの平均文字数
    mean_text_chars: float = 0.0

    def __str__(self) -> str:
        pct = self.text_contribution * 100
        lines = [
            f"テキスト列「{self.text_column}」の寄与を測りました"
            f"（{self.n_rows:,} 行 / 値の種類 {self.n_unique_text:,}"
            f" / 平均 {self.mean_text_chars:,.0f} 文字）",
            "",
            f"  テキスト無し LightGBM   MAE {self.mae_without_text:>10,.2f} {self.unit}",
            f"  テキスト有り LightGBM   MAE {self.mae_with_text:>10,.2f} {self.unit}",
            f"  テキスト寄与率          {pct:>10.1f} %"
            f"（閾値 {self.threshold * 100:.0f}%）",
            "",
            f"  判定: {self.verdict}",
        ]
        if self.verdict == "試す価値あり":
            lines.append(
                "  → テキストが価格を説明しています。文字 TF-IDF より LLM の\n"
                "     ほうがうまく読める余地があるので、機能B を試す価値があります。")
        elif self.verdict == "効きにくい":
            lines.append(
                "  → テキストがほとんど価格を説明していません。LLM に読ませても\n"
                "     上積みは期待しにくいので、統計モデルで十分な可能性が高いです。\n"
                "     （機能A で別の列を作る、そもそも別の列を集める、を先に検討）")
        else:
            lines.append(
                "  → 差が小さく、行数も少ないので判断できません。\n"
                "     行数を増やすか、小さく機能B を試して実測してください。")
        lines.append("")
        if self.max_text_chars and self.mean_text_chars > self.max_text_chars:
            lines.append(
                f"  ※ 判定には先頭 {self.max_text_chars} 文字だけを使いました"
                f"（平均 {self.mean_text_chars:,.0f} 文字）。\n"
                "     続きにも情報があるなら、実際の寄与はこれより大きい可能性があります。")
        lines.append("  ※ この判定は2データセットの実測から引いた暫定の線です。")
        lines.append("     最終的な判断は、小さく実測すること（--n-eval 60 程度）。")
        return "\n".join(lines)


def screen(df: pd.DataFrame, target: str, text: str, *,
           numeric: Sequence[str] = (), boolean: Sequence[str] = (),
           categorical: Sequence[str] = (), unit: str = "",
           n_splits: int = 5, sample: int | None = 20_000,
           max_text_chars: int = 500,
           threshold: float = DEFAULT_THRESHOLD) -> ScreeningReport:
    """テキスト列が価格をどれだけ説明しているかを測る。**LLM は呼ばない。**

    パラメータ
    ----------
    df, target, text:
        データ、目的変数の列名、判定したいテキスト列。
    numeric / boolean / categorical:
        比較の土台にする構造化列。ここが弱いとテキストの寄与が過大に出る。
    sample:
        行数の上限。大きいデータで待たされないよう既定で 20,000 行に間引く。
        `None` なら全行。
    max_text_chars:
        テキストの先頭何文字を見るか。**文字 n-gram TF-IDF は文章が長いと
        急激に重くなる**（Craigslist の description は平均 2,320 字あり、20,000 行で
        数分待たされる）。判定に必要なのは「テキストが効くか」の桁だけなので、
        既定で先頭 500 字に切る。`0` なら切らない。
    threshold:
        「試す価値あり」と判定する寄与率の下限。

    返り値
    ------
    `ScreeningReport`。print するとそのまま日本語で読める。
    """
    from sklearn.model_selection import KFold

    from unfold.predictor import ColumnSpec, TreeModel

    for col in [target, text, *numeric, *boolean, *categorical]:
        if col not in df.columns:
            raise UnfoldError(f"データに無い列を指定しています: {col}")

    df = df[df[target].notna()].reset_index(drop=True)
    if sample is not None and sample < len(df):
        df = df.sample(n=sample, random_state=SEED).reset_index(drop=True)
    if len(df) < n_splits * 2:
        raise UnfoldError(f"行数が少なすぎます（{len(df)} 行）。")

    spec = ColumnSpec(numeric=list(numeric), boolean=list(boolean),
                      categorical=list(categorical))
    y = df[target].to_numpy(dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    def _clip(col: pd.Series) -> pd.Series:
        v = col.fillna("").astype(str)
        return v.str.slice(0, max_text_chars) if max_text_chars > 0 else v

    def with_text(train: pd.DataFrame, test: pd.DataFrame):
        """テキストを文字 n-gram TF-IDF → SVD で圧縮して列として足す。

        fit は必ず train だけで行う（test の情報が漏れると寄与が過大に出る）。
        """
        # min_df は行数に応じて緩める。少ない行数で 5 のままだと
        # 語彙が空になり、sklearn が分かりにくい例外を投げる。
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                              min_df=min(5, max(1, len(train) // 50)),
                              max_features=2000)
        A = vec.fit_transform(_clip(train[text]))
        B = vec.transform(_clip(test[text]))
        n_comp = min(64, A.shape[1] - 1)
        if n_comp < 2:
            return None, None
        svd = TruncatedSVD(n_components=n_comp, random_state=SEED)
        return svd.fit_transform(A), svd.transform(B)

    maes_without, maes_with = [], []
    for tr_idx, te_idx in kf.split(df):
        train = df.iloc[tr_idx].reset_index(drop=True)
        test = df.iloc[te_idx].reset_index(drop=True)
        y_te = y[te_idx]

        m0 = TreeModel("テキスト無し", spec).fit(train, y[tr_idx])
        maes_without.append(float(np.mean(np.abs(m0.predict(test) - y_te))))

        Etr, Ete = with_text(train, test)
        if Etr is None:                     # テキストが短すぎて特徴が作れない
            maes_with.append(maes_without[-1])
            continue
        cols = [f"_txt{i}" for i in range(Etr.shape[1])]
        spec2 = ColumnSpec(numeric=[*spec.numeric, *cols], boolean=spec.boolean,
                           categorical=spec.categorical)
        tr2 = pd.concat([train, pd.DataFrame(Etr, columns=cols)], axis=1)
        te2 = pd.concat([test, pd.DataFrame(Ete, columns=cols)], axis=1)
        m1 = TreeModel("テキスト有り", spec2).fit(tr2, y[tr_idx])
        maes_with.append(float(np.mean(np.abs(m1.predict(te2) - y_te))))

    mae0, mae1 = float(np.mean(maes_without)), float(np.mean(maes_with))
    contribution = (mae0 - mae1) / mae0 if mae0 > 0 else 0.0

    # fold ごとに符号が揃っていなければ「判断できない」にする。
    # 平均だけ見て決めると、1 fold の外れ値で判定が動いてしまう。
    wins = sum(a > b for a, b in zip(maes_without, maes_with))
    if contribution >= threshold and wins >= n_splits - 1:
        verdict = "試す価値あり"
    elif contribution < threshold / 2:
        verdict = "効きにくい"
    else:
        verdict = "判断できない"

    return ScreeningReport(
        n_rows=len(df), unit=unit, mae_without_text=mae0, mae_with_text=mae1,
        text_contribution=contribution, threshold=threshold, text_column=text,
        n_unique_text=int(df[text].nunique()), verdict=verdict,
        max_text_chars=max_text_chars,
        mean_text_chars=float(df[text].fillna("").astype(str).str.len().mean()))
