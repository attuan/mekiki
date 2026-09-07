"""機能B — `LLMPredictor`（設計書 dialogs/unfold-landing.html の Capability B）。

**LLM に生レコードを渡して値段を当てさせるのではない。** 先に統計モデルに
解かせ、その予測値と「実際の価格が分かっている似た車」を証拠としてまとめ、
LLM には最終判断だけさせる。8/31 のミーティングで伊藤さんが説明した形そのもの:

> トレイン・テスト・バリッドがあると思うので、トレインでやる話かな。中古車価格
> みたいなのはトレーニングデータセットについてはあると。けどテストデータセット
> についてはないので、テストデータの中から一番近いトレインデータを見つけてきて
> 「こんな感じで」予測する。

つまり **教師ラベルを別途作る必要はない**。訓練データの正解価格が
そのままフューショット（few-shot）の例になる。これで「教師ラベルをどう作るか」は解決し、
「分類か回帰か」も回帰に決まった（どちらも 8/31 ミーティング）。

    from unfold import LLMPredictor

    model = LLMPredictor(target="車両本体価格_万円", unit="万円",
                         numeric=["車齢", "走行距離_km"], text="装備テキスト")
    model.fit(train_df)
    pred = model.predict(test_df)
    model.explain(0)      # その1行がなぜその値になったか
    model.cost()          # かかった費用

証拠の作り方は実測に従っている（`docs/2026-08-29-denoise.md`）。**近傍を意味的類似度だけで
選ぶと価格の近さを拾わない**（MAE 31.45）ため、距離に数値列の差を混ぜる
（18.19）。ここは再検討せず、測って決まった recipe をそのまま使う。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd

from unfold.encoders import CharTfidfEncoder, Encoder
from unfold.errors import UnfoldError
from unfold.leakage import check_duplicates, check_overlap, warn_if_leaky
from unfold.llm import ClaudeClient, LLMAnswer

SEED = 42


# =====================================================================
# 証拠を作る側 — 統計モデルと近傍検索
# =====================================================================


@runtime_checkable
class BaseModel(Protocol):
    """LLM に渡す証拠を1つ作る部品。scikit-learn より緩い形にしてある。

    列の選び方や欠損の扱いがモデルごとに違うので、行列ではなく
    DataFrame をそのまま渡す。
    """

    name: str

    def fit(self, train: pd.DataFrame, y: np.ndarray) -> "BaseModel": ...

    def predict(self, test: pd.DataFrame) -> np.ndarray: ...


@dataclass
class ColumnSpec:
    """どの列をどう扱うか。データセットを差し替えるときはここだけ書き換える。

    `text` と `long_text` を分けているのは**長さの扱いが違う**ため。

    - `text` … 短いテキスト（車種の記述・装備の羅列）。近傍検索の入力にもなり、
      査定対象にも類似事例にも丸ごと載る。
    - `long_text` … 自由記述の本文（Craigslist の `description` は中央値 1,075 字・
      最大 28,832 字）。**5事例ぶん丸ごと貼るとプロンプトが桁で膨らむ**ので、
      査定対象と類似事例で別々に上限を切る。既定では類似事例には載せない
      （`long_text_example_chars=0`）。

    この分離が PRD 信頼度ルーティングの「長い自由記述はそのままでは扱えない」への回答にあたる。
    """

    numeric: list[str] = field(default_factory=list)
    boolean: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    text: str | None = None
    long_text: str | None = None
    #: 査定対象の自由記述に載せる上限文字数。超えた分は切って印を付ける
    long_text_chars: int = 2000
    #: 類似事例の自由記述に載せる上限。0 なら載せない（既定）
    long_text_example_chars: int = 0
    #: 自由記述の中の金額・価格らしい数字を伏せるか。**既定で伏せる。**
    #: 出品テキストには売り値が書いてあることが多く、渡すと予測ではなく
    #: 答えの読み取りになる（`mask_amounts` の docstring に実測値）。
    mask_amounts_in_long_text: bool = True

    def all_columns(self) -> list[str]:
        cols = [*self.numeric, *self.boolean, *self.categorical]
        if self.text:
            cols.append(self.text)
        if self.long_text:
            cols.append(self.long_text)
        return cols

    def check(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.all_columns() if c not in df.columns]
        if missing:
            raise UnfoldError(f"データに無い列を指定しています: {missing}")


@dataclass
class TreeModel:
    """LightGBM / XGBoost を `BaseModel` の形に揃えたもの。

    `run_baselines.py` の B 相当（構造化列だけ）を既定にしている。
    テキストは LLM 側に渡すので、ここでは入れない。

    **カテゴリの水準は fit のとき train だけで決める。** test にしか無い
    水準（train に無かったグレード名）は欠損として扱い、木は欠損の枝に流す。
    これを破ると「未知の値」の評価（`docs/2026-08-29-feature-unseen.md`）が成り立たなくなる。
    """

    name: str
    spec: ColumnSpec
    kind: str = "lgbm"  # "lgbm" | "xgb"
    params: dict = field(default_factory=dict)

    def _frame(self, df: pd.DataFrame) -> pd.DataFrame:
        # 自由記述（long_text）は数値化していないので木には渡さない。
        # そこを読ませるのが LLM の役目なので、ここで入れると比較が濁る。
        out = (df[self.spec.numeric].astype("float64").copy()
               if self.spec.numeric else pd.DataFrame(index=df.index))
        for c in self.spec.boolean:
            out[c] = df[c].astype("float64")
        for c in self.spec.categorical:
            cats = self.categories_[c]
            out[c] = pd.Categorical(df[c].where(df[c].isin(cats)), categories=cats)
        return out

    def fit(self, train: pd.DataFrame, y: np.ndarray) -> "TreeModel":
        self.categories_ = {c: pd.Index(train[c].dropna().unique())
                            for c in self.spec.categorical}
        if self.kind == "lgbm":
            from lightgbm import LGBMRegressor
            p = dict(n_estimators=700, learning_rate=0.05, num_leaves=31,
                     min_child_samples=20, subsample=0.9, subsample_freq=1,
                     colsample_bytree=0.9, random_state=SEED, verbose=-1)
            self.model_ = LGBMRegressor(**{**p, **self.params})
            self.model_.fit(self._frame(train), np.asarray(y, dtype=float),
                            categorical_feature=self.spec.categorical or "auto")
        elif self.kind == "xgb":
            from xgboost import XGBRegressor
            p = dict(n_estimators=700, learning_rate=0.05, max_depth=6,
                     subsample=0.9, colsample_bytree=0.9, random_state=SEED,
                     enable_categorical=True, tree_method="hist", verbosity=0)
            self.model_ = XGBRegressor(**{**p, **self.params})
            self.model_.fit(self._frame(train), np.asarray(y, dtype=float))
        else:
            raise UnfoldError(f"kind は lgbm か xgb です: {self.kind}")
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model_.predict(self._frame(test)), dtype=float)


@dataclass
class NeighbourIndex:
    """「実際の価格が分かっている似た車」を引く索引。

    類似度 = テキストのコサイン類似度 − w × 数値列の標準化距離の平均。
    引き算しているのは `docs/2026-08-29-denoise.md` の実測結果に従ったもので、意味だけで
    選ぶと年式も走行距離も違う車が「似ている」と出てくるため。

    w の既定 0.15 は `scripts/run_denoise.py` で測った値。
    """

    spec: ColumnSpec
    k: int = 5
    w: float = 0.15
    encoder: Encoder | None = None

    def fit(self, train: pd.DataFrame, y: np.ndarray) -> "NeighbourIndex":
        self.train_ = train.reset_index(drop=True)
        self.y_ = np.asarray(y, dtype=float)
        if self.encoder is None:
            self.encoder = CharTfidfEncoder()
        texts = self._texts(self.train_)
        self.encoder.fit(texts)
        self.V_ = self.encoder.transform(texts)
        if self.spec.numeric:
            N = self.train_[self.spec.numeric].astype("float64")
            self.num_med_ = N.median()
            self.N_ = N.fillna(self.num_med_).to_numpy()
            sd = self.N_.std(axis=0)
            sd[sd == 0] = 1.0
            self.sd_ = sd
        else:
            self.N_ = None
        return self

    def _texts(self, df: pd.DataFrame) -> pd.Series:
        if self.spec.text:
            return df[self.spec.text].fillna("").astype(str)
        return pd.Series([""] * len(df), index=df.index)

    def query(self, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """各行について近い訓練行の添字と類似度を返す。形は (n_test, k)。"""
        Vq = self.encoder.transform(self._texts(test))
        sim = Vq @ self.V_.T
        if self.N_ is not None:
            Nq = (test[self.spec.numeric].astype("float64")
                  .fillna(self.num_med_).to_numpy())
            d = np.zeros_like(sim)
            for c in range(self.N_.shape[1]):
                d += np.abs(Nq[:, [c]] - self.N_[:, c][None, :]) / self.sd_[c]
            sim = sim - self.w * d / self.N_.shape[1]
        k = min(self.k, sim.shape[1])
        # argpartition で上位 k 件だけ取り、そのあと k 件の中だけ並べ替える
        idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sim, idx, axis=1), axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        return idx, np.take_along_axis(sim, idx, axis=1)


@dataclass
class NeighbourModel:
    """近傍 k 件の価格中央値を予測値にする。証拠のひとつとして LLM に渡す。

    これ単体では統計モデルに届かない（シエンタで 18.19 対 12.21）が、
    「似た車がいくらだったか」という別方向の情報を持っている。
    """

    name: str
    index: NeighbourIndex

    def fit(self, train: pd.DataFrame, y: np.ndarray) -> "NeighbourModel":
        # 索引は通常 LLMPredictor 側で fit 済み。単体で使われたときのために、
        # まだならここで張る（黙って空の索引を引かせないため）。
        if not hasattr(self.index, "V_"):
            self.index.fit(train, y)
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        idx, _ = self.index.query(test)
        return np.median(self.index.y_[idx], axis=1)


# =====================================================================
# プロンプト
# =====================================================================

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "price": {"type": "number",
                  "description": "予測価格。単位はプロンプトで指定されたもの"},
        "confidence": {"type": "number",
                       "description": "0.0〜1.0 の確信度"},
        "reason": {"type": "string",
                   "description": "どの証拠をどう重み付けたか。日本語で1〜2文"},
    },
    "required": ["price", "confidence", "reason"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
あなたは中古車の査定士です。1台ぶんの情報と、それについての「証拠」を受け取り、
車両本体価格を1つの数値で答えてください。

証拠は3種類あります。

1. 統計モデルの予測値 — 訓練データ全体から学習した木モデルの出力。
   全体傾向をよく捉えていますが、テキストに書かれた装備や状態は見ていません。
2. 類似事例 — 実際の成約価格が分かっている訓練データの車。査定対象に近い順です。
   「近い」は数値属性とテキストの両方を混ぜた距離で測っています。
3. 査定対象の属性そのもの — 数値列とテキスト列。

判断の指針:

- **統計モデルの予測値を出発点にしてください。** 訓練データ全体を見ているのは
  これだけです。類似事例は数件しかないので、そこから大きく離れる根拠には
  なりにくいです。
- 類似事例は「統計モデルが見落としている装備・状態の差」を読むために使います。
  査定対象にあって類似事例に無い装備、その逆、を見てください。
- 統計モデル同士が食い違うときは、類似事例の価格帯に近いほうを重く見てください。
- 証拠から離れた値を出さないでください。統計モデルの予測と類似事例の価格が
  すべて収まる範囲の外に出るなら、その理由を reason に書いてください。
- 自由記述（出品者が書いた説明）が付いている場合、そこは**統計モデルが
  読めていない唯一の情報**です。事故歴・改造・欠陥・付属品・急ぎの売却など、
  価格を動かす記述があれば拾ってください。ただし宣伝文句（「美車」「お買い得」）は
  どの出品にも書いてあるので、価格の根拠にしないでください。
- 自由記述が「…（以下 N 文字を省略）」で終わっている場合、続きは読めていません。
  読めた範囲だけで判断し、confidence を下げてください。
- 自由記述の中の「〈金額〉」「〈数値〉」は、**答えの漏れを防ぐために伏せた数字**です。
  そこに売り値が書かれていた可能性が高いので、推測して埋めようとしないでください。

confidence は「この答えが実際の価格に近い自信」を 0.0〜1.0 で入れてください。
証拠どうしが食い違うとき、類似事例が査定対象と似ていないときは低くしてください。
"""


def _format_value(v: Any) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "不明"
    if isinstance(v, (bool, np.bool_)):
        return "あり" if v else "なし"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, (float, np.floating)):
        return f"{v:,.6g}"
    return str(v)


#: 金額らしい表記。`$12,345` / `12345 dollars` / `USD 12,345` を拾う
_MONEY = re.compile(
    r"(?:[$＄]\s?[\d][\d,.]*)"
    r"|(?:\b[\d][\d,.]*\s?(?:dollars?|usd|万円|円)\b)",
    re.IGNORECASE)

#: 単独で現れる 3〜6 桁の整数。価格になりうる桁の数字をまとめて伏せる
_BARE_NUMBER = re.compile(r"(?<![\w.$])(\d{1,3}(?:,\d{3})+|\d{3,6})(?![\w.])")

#: 伏せない下限・上限。これを外れる数字は価格ではありえないので残す
_MASK_MIN, _MASK_MAX = 300, 300_000


def mask_amounts(text: str) -> str:
    """自由記述から「正解になりうる数字」を伏せる。

    **なぜ要るか。** 出品テキストには売り値がそのまま書いてある。
    Craigslist の description を実測すると **43% の行に価格と完全一致する数字**があり、
    **51% に `$` 付きの金額**があった。これを LLM に渡すと、予測しているのではなく
    **答えを読んでいる**だけになる（実際、伏せずに測ると MAE が 2,336 → 1,546 と
    40% 改善して見えた。これは実力ではない）。

    これは中古車に限らない。**出品・求人・不動産など、自由記述に価格や条件が
    書かれるデータでは常に起きる**ので、利用者の心得ではなく製品側で塞ぐ。

    伏せる対象:

    - `$12,345` `12345 dollars` `USD 12,345` `123万円` などの金額表記
    - 単独で現れる 300〜300,000 の整数（走行距離・年式・在庫番号も巻き込む）

    **走行距離や年式も消えるが、それらは構造化列として別に渡している**ので
    情報は失われない。逆に「F-150」「911」「Model 3」のような車種の数字は
    300 未満なので残る。電話番号は桁が大きいので残るが害はない。

    完全ではない（「twelve thousand」のような綴りは抜ける）。
    **伏字だけに頼らず、`docs/` の測定では必ずリークの有無を確認すること。**
    """
    if not text:
        return text

    def _num(m: re.Match) -> str:
        raw = m.group(1).replace(",", "")
        try:
            v = int(raw)
        except ValueError:
            return m.group(0)
        return "〈数値〉" if _MASK_MIN <= v <= _MASK_MAX else m.group(0)

    return _BARE_NUMBER.sub(_num, _MONEY.sub("〈金額〉", text))


def _truncate(text: str, limit: int) -> str:
    """上限で切り、切ったことを明示する。

    黙って切ると LLM は「そこで文が終わっている」と読んでしまう。
    切った事実と元の長さを書いておけば、判断材料が欠けていることが伝わる。
    """
    if limit <= 0 or len(text) <= limit:
        return text
    return f"{text[:limit]}…（以下 {len(text) - limit:,} 文字を省略）"


def _describe_row(row: pd.Series, spec: ColumnSpec, indent: str = "",
                  is_target: bool = True) -> str:
    """1行を人が読める箇条書きにする。LLM に渡すのはこの形。

    `is_target` で自由記述の扱いが変わる。査定対象には長く載せ、
    類似事例には既定で載せない（プロンプトの膨張を抑えるため）。
    """
    lines = []
    for c in [*spec.numeric, *spec.boolean, *spec.categorical]:
        lines.append(f"{indent}- {c}: {_format_value(row.get(c))}")
    if spec.text:
        lines.append(f"{indent}- {spec.text}: {_format_value(row.get(spec.text))}")
    if spec.long_text:
        limit = (spec.long_text_chars if is_target
                 else spec.long_text_example_chars)
        if limit > 0:
            raw = row.get(spec.long_text)
            body = "" if raw is None or (isinstance(raw, float) and np.isnan(raw)) \
                else str(raw)
            # 改行を潰す。箇条書きの中に生の改行が混ざると構造が読めなくなる
            body = " ".join(body.split())
            if spec.mask_amounts_in_long_text:
                body = mask_amounts(body)
            lines.append(f"{indent}- {spec.long_text}: "
                         f"{_truncate(body, limit) if body else '記載なし'}")
    return "\n".join(lines)


# =====================================================================
# 機能B 本体
# =====================================================================


@dataclass
class Prediction:
    """1行ぶんの予測と、その来歴（PRD「来歴（provenance）と検査 API」）。"""

    value: float
    confidence: float
    reason: str
    origin: str                      # "llm" | "fallback"
    model_predictions: dict[str, float]
    neighbours: list[dict]
    cost: float = 0.0
    from_cache: bool = False
    error: str | None = None


class LLMPredictor:
    """統計モデルと類似事例を証拠に、LLM に最終判断をさせる回帰器。

    パラメータ
    ----------
    target:
        目的変数の列名。
    unit:
        価格の単位（"万円" / "USD"）。プロンプトに出るだけだが、
        LLM が桁を間違えないために効く。
    numeric / boolean / categorical / text:
        列の割り当て。`ColumnSpec` を直接渡してもよい。
    models:
        証拠を作る統計モデル。省略すると LightGBM・XGBoost・近傍の3つ。
    n_examples:
        LLM に見せる類似事例の件数（few-shot の shot 数）。
    client:
        `ClaudeClient`。省略すると既定（claude-opus-5 / effort=low）。

    使い方は scikit-learn と同じ `fit` → `predict`。
    そこに `explain` / `confidence` / `examples` / `cost` が乗る。
    """

    def __init__(self, target: str, unit: str = "", *,
                 numeric: Sequence[str] = (), boolean: Sequence[str] = (),
                 categorical: Sequence[str] = (), text: str | None = None,
                 long_text: str | None = None,
                 spec: ColumnSpec | None = None,
                 models: Sequence[BaseModel] | None = None,
                 n_examples: int = 5, neighbour_weight: float = 0.15,
                 encoder: Encoder | None = None,
                 client: ClaudeClient | None = None,
                 fallback: str = "best_model",
                 check_leakage: bool = True) -> None:
        self.target = target
        self.unit = unit
        self.spec = spec or ColumnSpec(
            numeric=list(numeric), boolean=list(boolean),
            categorical=list(categorical), text=text, long_text=long_text)
        self.n_examples = n_examples
        self.neighbour_weight = neighbour_weight
        self.encoder = encoder
        self.client = client or ClaudeClient()
        self._models_arg = models
        if fallback not in ("best_model", "error"):
            raise UnfoldError('fallback は "best_model" か "error" です')
        self.fallback = fallback
        #: 重複レコードを検知して警告するか（PRD「非機能要件」のリーク防止）。
        #: **テキストを特徴量にするほど重複の水増しが大きくなる**ので既定で入れる
        self.check_leakage = check_leakage

    # --- fit ----------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None
            ) -> "LLMPredictor":
        """訓練データを覚える。**ここで正解ラベルを別途作る必要はない。**

        `y` を省略すると `X[target]` を使う。訓練データの正解価格が
        そのままフューショットの例になる（8/31 ミーティングでの決着）。
        """
        self.spec.check(X)
        if y is None:
            if self.target not in X.columns:
                raise UnfoldError(
                    f"目的変数 {self.target!r} が X にありません。"
                    "y を渡すか、target を列名に合わせてください。")
            y = X[self.target]
        y = np.asarray(y, dtype=float)
        if len(y) != len(X):
            raise UnfoldError(f"X と y の長さが違います: {len(X)} != {len(y)}")

        self.train_ = X.reset_index(drop=True)
        self.y_ = y
        if self.check_leakage:
            warn_if_leaky(
                check_duplicates(self.train_, keys=self.spec.all_columns()),
                where="訓練データ")
        self.index_ = NeighbourIndex(self.spec, k=self.n_examples,
                                     w=self.neighbour_weight,
                                     encoder=self.encoder).fit(self.train_, y)

        if self._models_arg is None:
            self.models_ = [
                TreeModel("LightGBM", self.spec, kind="lgbm"),
                TreeModel("XGBoost", self.spec, kind="xgb"),
                NeighbourModel(f"近傍{self.n_examples}件の中央値", self.index_),
            ]
        else:
            self.models_ = list(self._models_arg)
        for m in self.models_:
            m.fit(self.train_, y)
        self.predictions_ = None
        return self

    # --- predict ------------------------------------------------------

    def _build_prompt(self, row: pd.Series, evidence: dict[str, float],
                      neighbours: list[dict]) -> str:
        u = f"（単位: {self.unit}）" if self.unit else ""
        parts = [f"## 査定対象\n\n{_describe_row(row, self.spec)}",
                 f"\n## 証拠1: 統計モデルの予測値{u}\n"]
        for name, v in evidence.items():
            parts.append(f"- {name}: {v:,.6g}")
        parts.append(f"\n## 証拠2: 類似事例（実際の価格が分かっている訓練データ）{u}\n")
        for j, nb in enumerate(neighbours, start=1):
            parts.append(f"### 事例{j}（類似度 {nb['類似度']:.3f}）"
                         f" 実際の価格: {nb['価格']:,.6g}")
            parts.append(_describe_row(pd.Series(nb["属性"]), self.spec,
                                       is_target=False))
            parts.append("")
        parts.append(f"この車の価格を{u}で1つ答えてください。")
        return "\n".join(parts)

    def predict(self, X: pd.DataFrame, verbose: bool = False) -> np.ndarray:
        """各行の価格を返す。LLM を1行1回呼ぶ（キャッシュが効けば無料）。"""
        if not hasattr(self, "train_"):
            raise UnfoldError("fit を先に呼んでください。")
        self.spec.check(X)
        X = X.reset_index(drop=True)
        if self.check_leakage:
            warn_if_leaky(check_overlap(self.train_, X,
                                        keys=self.spec.all_columns()),
                          where="train と test")

        # 1. 統計モデルに解かせる
        evidence = {m.name: np.asarray(m.predict(X), dtype=float)
                    for m in self.models_}
        # 2. 類似事例を引く
        idx, sim = self.index_.query(X)

        # 3. 行ごとにプロンプトを組む
        prompts, neighbour_rows = [], []
        for i in range(len(X)):
            nbs = [{"訓練行": int(j), "類似度": float(s),
                    "価格": float(self.index_.y_[j]),
                    "属性": self.train_.iloc[j][self.spec.all_columns()].to_dict()}
                   for j, s in zip(idx[i], sim[i])]
            neighbour_rows.append(nbs)
            ev = {name: float(v[i]) for name, v in evidence.items()}
            prompts.append(self._build_prompt(X.iloc[i], ev, nbs))

        # 4. まとめて投げる
        def show(done: int, total: int) -> None:
            if verbose and (done % 25 == 0 or done == total):
                print(f"    LLM {done}/{total} 行"
                      f"（費用 ${self.client.usage.cost:.3f}"
                      f" / キャッシュ命中 {self.client.usage.cache_hits}）")

        answers = self.client.ask_many(SYSTEM_PROMPT, prompts, ANSWER_SCHEMA,
                                       progress=show if verbose else None)

        # 5. 来歴を組み立てる。答えられなかった行は統計モデルの1つ目で埋める
        fallback_name = self.models_[0].name
        out = np.empty(len(X), dtype=float)
        preds: list[Prediction] = []
        for i, ans in enumerate(answers):
            ev = {name: float(v[i]) for name, v in evidence.items()}
            if ans.ok and "price" in ans.data:
                p = Prediction(value=float(ans.data["price"]),
                               confidence=float(ans.data.get("confidence", 0.0)),
                               reason=str(ans.data.get("reason", "")),
                               origin="llm", model_predictions=ev,
                               neighbours=neighbour_rows[i],
                               cost=ans.cost, from_cache=ans.from_cache)
            else:
                if self.fallback == "error":
                    raise UnfoldError(f"{i}行目で LLM が答えられませんでした: "
                                      f"{ans.error}")
                p = Prediction(value=ev[fallback_name], confidence=0.0,
                               reason=f"LLM が答えられなかったので "
                                      f"{fallback_name} の予測で代替",
                               origin="fallback", model_predictions=ev,
                               neighbours=neighbour_rows[i],
                               cost=ans.cost, error=ans.error)
            preds.append(p)
            out[i] = p.value

        self.predictions_ = preds
        self.last_X_ = X
        return out

    # --- 検査 API（PRD「来歴（provenance）と検査 API」）----------------

    def _require(self, X: pd.DataFrame | None = None) -> list[Prediction]:
        """直近の predict の結果を返す。X を渡したら取り違えを検知する。

        検査 API は「直前に predict した X」についてしか答えられない。
        別の X を渡されたまま黙って前回の結果を返すと、間違った来歴を
        読んで判断することになるので、長さが違えば止める。
        """
        if getattr(self, "predictions_", None) is None:
            raise UnfoldError("predict を先に呼んでください。")
        if X is not None and len(X) != len(self.predictions_):
            raise UnfoldError(
                f"直近に predict した行数（{len(self.predictions_)}）と "
                f"X の行数（{len(X)}）が違います。検査 API は直前の "
                "predict の結果を返すので、同じ X を渡してください。")
        return self.predictions_

    def confidence(self, X: pd.DataFrame | None = None) -> pd.Series:
        """行ごとの確信度。信頼度ルーティング（機能C）の入力になる。

        直前の `predict` の結果を返す。X を渡すと行数の一致を検査する。
        """
        return pd.Series([p.confidence for p in self._require(X)],
                         name="confidence")

    def examples(self, i: int | None = None) -> pd.DataFrame:
        """推論に使った類似事例。i を渡すとその行のぶんだけ。"""
        preds = self._require()
        rows = []
        for k, p in enumerate(preds):
            if i is not None and k != i:
                continue
            for rank, nb in enumerate(p.neighbours, start=1):
                rows.append({"行": k, "順位": rank, "訓練行": nb["訓練行"],
                             "類似度": nb["類似度"], "価格": nb["価格"],
                             **nb["属性"]})
        return pd.DataFrame(rows)

    def cost(self, X: pd.DataFrame | None = None) -> dict:
        """かかった費用と内訳。

        まだ predict していなければ、クライアントの累計だけを返す。
        """
        u = self.client.summary()
        preds = getattr(self, "predictions_", None)
        if X is not None and preds is not None and len(X) != len(preds):
            raise UnfoldError(
                f"直近に predict した行数（{len(preds)}）と "
                f"X の行数（{len(X)}）が違います。")
        if preds:
            n = len(preds)
            paid = [p for p in preds if not p.from_cache]
            u["予測した行数"] = n
            u["1行あたりの費用_usd"] = round(
                sum(p.cost for p in paid) / max(len(paid), 1), 6)
            u["LLM が答えた割合"] = round(
                sum(p.origin == "llm" for p in preds) / n, 4)
        return u

    def explain(self, i: int) -> str:
        """その1行がなぜその値になったかを、証拠ごと文章で返す。"""
        p = self._require()[i]
        lines = [f"[{i}行目] 予測 {p.value:,.6g} {self.unit}"
                 f"（由来 {p.origin} / confidence {p.confidence:.2f}）",
                 "",
                 "統計モデルの予測:"]
        for name, v in p.model_predictions.items():
            lines.append(f"  - {name}: {v:,.6g}")
        lines.append("")
        lines.append("参照した類似事例:")
        for rank, nb in enumerate(p.neighbours, start=1):
            desc = _describe_row(pd.Series(nb["属性"]), self.spec,
                                 indent="      ", is_target=False)
            lines.append(f"  {rank}. 価格 {nb['価格']:,.6g}"
                         f"（類似度 {nb['類似度']:.3f}）")
            lines.append(desc)
        lines.append("")
        lines.append(f"LLM の理由: {p.reason}")
        if p.error:
            lines.append(f"エラー: {p.error}")
        return "\n".join(lines)

    def provenance(self) -> pd.DataFrame:
        """全行の来歴を表にする。`explain` の一覧版。"""
        return pd.DataFrame([
            {"行": i, "予測": p.value, "由来": p.origin,
             "confidence": p.confidence, "費用_usd": p.cost,
             "キャッシュ": p.from_cache, "理由": p.reason,
             **{f"証拠_{k}": v for k, v in p.model_predictions.items()}}
            for i, p in enumerate(self._require())
        ])
