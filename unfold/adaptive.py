"""信頼度ルーティング — `AdaptivePredictor`（設計書の Capability C / PRD「信頼度ルーティング」）。

**全行を LLM に投げる必要はない。** 機能B（`LLMPredictor`）は1行ごとに LLM を
呼ぶので、行数がそのまま費用と時間になる。実測値を Craigslist 6万行に当てると

    60,000 行 × $0.0086 = 約 $516 、 60,000 行 × 1.03 秒 = 約 17 時間（並列8）

となり、そのままでは回せない。そこで **「LLM を呼ぶ前に手に入る信号」だけで
呼ぶ行を選び**、残りは統計モデルに任せる。閾値ひとつで「全行呼ぶ」と
「1行も呼ばない」の間を連続に動かせるようにしたのがこのクラスである。

    from unfold import AdaptivePredictor

    model = AdaptivePredictor(target="price", unit="USD",
                              numeric=["age", "odometer"],
                              categorical=["manufacturer", "state"], text="model",
                              escalate_rate=0.3)       # 上位3割だけ LLM に回す
    model.plan(test)        # 呼ぶ前に「何行・いくら・何秒」を見る
    pred = model.predict(test)
    model.route()           # 行ごとにどちらの経路を通ったか
    model.review_queue()    # LLM が答えた行 = 教師ラベル候補
    model.approve()         # 承認すると次回はその行を呼ばずに済む（高速パスが広がる）

## どの信号で選ぶか（実測で決まっている）

`scripts/find_routing_signal.py` で候補を6通り測った結果
（`docs/2026-09-01-llm-predictor-vehicles.md`）、**最良は「木2つの食い違い」**
だった。LightGBM と XGBoost の予測が大きく食い違う行 —— つまり統計モデル自身が
迷っている行 —— に LLM を回すのが効く。600行での実測は次のとおり。

| 呼ぶ割合 | 10% | 20% | 30% | 50% | 75% | 100% |
|---|---|---|---|---|---|---|
| MAE（USD） | 2,907 | 2,795 | 2,674 | 2,581 | 2,450 | **2,265** |
| 費用 | $0.52 | $1.05 | $1.57 | $2.62 | $3.92 | $5.23 |
| 推定時間（秒） | 61 | 93 | 130 | 199 | 286 | 369 |

（0% = 3,043。9/1 に `scripts/run_adaptive.py` でこのクラス自身が引いた曲線）

**このデータでは全行呼ぶ 2,265 が最も精度が良い。**ルーティングは
「精度を上げる仕掛け」ではなく **「精度をいくら落とせば費用がいくら浮くか」を
選べる仕掛け**である。半分だけ呼べば $2.62 で MAE 2,581 となり、
0% との差の 59% を半額で取れる、という読み方をする。

なお **シエンタ（単一車種）ではどの信号も効かなかった**（`results/routing_signals.csv`）。
信号が効くかどうかはデータ次第なので、`curve()` で必ず自分のデータで確かめること。
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from unfold.encoders import Encoder
from unfold.errors import UnfoldError
from unfold.leakage import check_overlap, warn_if_leaky
from unfold.llm import ClaudeClient
from unfold.predictor import (
    BaseModel,
    ColumnSpec,
    LLMPredictor,
    Prediction,
    TreeModel,
)

#: 1件の応答時間の実測（中央値・秒）。**並列数を上げても1件の速さは変わらない**
#: ので、全体の時間は「行数 ÷ 並列数 × これ」で見積もる（results/latency_vehicles.csv）
SECONDS_PER_CALL = 4.6

#: 準備処理（fit + 近傍索引の構築）の実測（秒）。行数によらず1回だけかかる
SETUP_SECONDS = 23.85

#: 1行あたり費用の実測（USD）。機能B の測定は $0.0086〜0.0121 だったので下限側を使う。
#: 実際に呼んだあとは、その実績値で置き換わる
COST_PER_ROW_USD = 0.0086

#: 使える信号。いずれも **LLM を呼ぶ前に手に入るもの**だけである。
#: 「LLM が動かした量」は最良の信号だが、呼んでからでないと分からないので費用の節約には使えない
SIGNALS = ("disagreement", "similarity", "unseen")


def _row_key(row: pd.Series, columns: Sequence[str]) -> str:
    """行の中身から作る指紋。承認済みの行を次回も同じ行だと見分けるために使う。

    行番号ではなく中身で照合するのは、次回の predict で行の順番や件数が
    変わっていても効くようにするため。
    """
    text = "\x1f".join(f"{c}={row[c]!r}" for c in columns)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


class AdaptivePredictor:
    """機能B に「どの行を LLM に回すか」の判断を付けた回帰器。

    Parameters
    ----------
    target, unit, numeric, boolean, categorical, text, long_text, spec,
    models, n_examples, neighbour_weight, encoder, client:
        `LLMPredictor` にそのまま渡る。既にある `LLMPredictor` を包みたいときは
        `predictor=` に渡せば、こちらは無視される。
    escalate_rate : float | None
        **信号が強い順に何割を LLM に回すか**（0.0〜1.0）。既定 0.3。
        `threshold` より優先される。費用の上限から逆算できるので、
        実務ではこちらのほうが使いやすい（`Feature` の同名引数と同じ考え方）。
    threshold : float | None
        信号の値そのもので切りたいときの閾値。`escalate_rate=None` にすると効く。
    signal : str | Callable
        "disagreement"（既定・実測で最良）… 木モデル同士の予測の開き。
        "similarity" … 近傍1位の類似度が低い行（証拠そのものが弱い行）。
        "unseen" … 訓練データに無いカテゴリ水準・未知語を含む行。
        新車種・新グレードが絶えず入る運用を想定した信号（PRD「信頼度ルーティング」）。
        自作するときは `f(X, evidence) -> 配列` を渡す。**大きいほど LLM に回す**向きで返すこと。
    predictor : LLMPredictor | None
        包む対象。省略すると上の引数から組み立てる。
    """

    def __init__(self, target: str = "", unit: str = "", *,
                 numeric: Sequence[str] = (), boolean: Sequence[str] = (),
                 categorical: Sequence[str] = (), text: str | None = None,
                 long_text: str | None = None,
                 spec: ColumnSpec | None = None,
                 models: Sequence[BaseModel] | None = None,
                 n_examples: int = 5, neighbour_weight: float = 0.15,
                 encoder: Encoder | None = None,
                 client: ClaudeClient | None = None,
                 escalate_rate: float | None = 0.3,
                 threshold: float | None = None,
                 signal: str | Callable[..., np.ndarray] = "disagreement",
                 predictor: LLMPredictor | None = None) -> None:
        if predictor is None:
            if not target:
                raise UnfoldError("target か predictor のどちらかが要ります。")
            predictor = LLMPredictor(
                target, unit, numeric=numeric, boolean=boolean,
                categorical=categorical, text=text, long_text=long_text,
                spec=spec, models=models, n_examples=n_examples,
                neighbour_weight=neighbour_weight, encoder=encoder,
                client=client)
        self.predictor = predictor
        if escalate_rate is None and threshold is None:
            raise UnfoldError(
                "escalate_rate か threshold のどちらかを指定してください。"
                "全行に呼ぶなら escalate_rate=1.0、1行も呼ばないなら 0.0 です。")
        if escalate_rate is not None and not 0.0 <= escalate_rate <= 1.0:
            raise UnfoldError(f"escalate_rate は 0.0〜1.0 です: {escalate_rate}")
        if isinstance(signal, str) and signal not in SIGNALS:
            raise UnfoldError(
                f"signal={signal!r} は未実装です。使えるのは {SIGNALS} か、"
                "f(X, evidence) -> 配列 の関数です。")
        self.escalate_rate = escalate_rate
        self.threshold = threshold
        self.signal = signal
        #: 承認済みの答え（行の指紋 → 値）。approve() で増える
        self.approved_: dict[str, float] = {}

    # --- 委譲 ----------------------------------------------------------

    @property
    def target(self) -> str:
        return self.predictor.target

    @property
    def unit(self) -> str:
        return self.predictor.unit

    @property
    def spec(self) -> ColumnSpec:
        return self.predictor.spec

    @property
    def client(self) -> ClaudeClient:
        return self.predictor.client

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None
            ) -> "AdaptivePredictor":
        """機能B と同じ。訓練データの正解価格がそのまま証拠になる。"""
        self.predictor.fit(X, y)
        return self

    def _check_fitted(self) -> LLMPredictor:
        if not hasattr(self.predictor, "train_"):
            raise UnfoldError("fit を先に呼んでください。")
        return self.predictor

    # --- 信号（LLM を呼ぶ前に手に入るものだけ）--------------------------

    def _unseen_score(self, X: pd.DataFrame) -> np.ndarray:
        """訓練データに無かった水準・未知語の多さ。大きいほど「初めて見る行」。"""
        m = self.predictor
        score = np.zeros(len(X), dtype=float)
        for c in m.spec.categorical:
            known = set(m.train_[c].dropna().astype(str))
            score += (~X[c].astype(str).isin(known)).to_numpy(dtype=float)
        if m.spec.text:
            known_tokens = set()
            for t in m.train_[m.spec.text].fillna("").astype(str):
                known_tokens.update(t.lower().split())
            for i, t in enumerate(X[m.spec.text].fillna("").astype(str)):
                toks = t.lower().split()
                if toks:
                    score[i] += sum(w not in known_tokens for w in toks) / len(toks)
        return score

    def _evidence(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        m = self.predictor
        return {mo.name: np.asarray(mo.predict(X), dtype=float)
                for mo in m.models_}

    def _signal_values(self, X: pd.DataFrame,
                       evidence: dict[str, np.ndarray]) -> np.ndarray:
        """「LLM に回すべき度合い」。大きいほど回す。"""
        m = self.predictor
        if callable(self.signal):
            s = np.asarray(self.signal(X, evidence), dtype=float)
            if s.shape != (len(X),):
                raise UnfoldError(
                    f"signal 関数は長さ {len(X)} の配列を返してください: {s.shape}")
            return s
        if self.signal == "disagreement":
            tree = [evidence[mo.name] for mo in m.models_
                    if isinstance(mo, TreeModel)]
            if len(tree) < 2:
                raise UnfoldError(
                    "signal='disagreement' は木モデルが2つ以上要ります"
                    f"（今は {len(tree)} つ）。models= を見直すか、"
                    "signal='similarity' を使ってください。")
            stack = np.vstack(tree)
            return stack.max(axis=0) - stack.min(axis=0)
        if self.signal == "similarity":
            _, sim = m.index_.query(X)
            return -sim[:, 0]                 # 似た事例が無い行ほど大きく
        return self._unseen_score(X)

    def _select(self, s: np.ndarray) -> np.ndarray:
        """信号から「LLM に回す行」の真偽値を作る。"""
        n = len(s)
        if self.escalate_rate is not None:
            k = int(round(n * self.escalate_rate))
            sel = np.zeros(n, dtype=bool)
            if k > 0:
                # 同点は入力順で決める（stable）。測定の再現性のため
                order = np.argsort(-s, kind="stable")
                sel[order[:k]] = True
            return sel
        return s >= float(self.threshold)

    @staticmethod
    def _to_confidence(s: np.ndarray) -> np.ndarray:
        """信号を 0〜1 の confidence に直す（信号が弱い行ほど確信が高い）。"""
        lo, hi = float(np.min(s)), float(np.max(s))
        if hi - lo < 1e-12:
            return np.ones(len(s), dtype=float)
        return 1.0 - (s - lo) / (hi - lo)

    def _row_keys(self, X: pd.DataFrame) -> list[str]:
        cols = [c for c in self.spec.all_columns() if c in X.columns]
        return [_row_key(X.iloc[i], cols) for i in range(len(X))]

    # --- 実行前の見積もり（PRD 来歴「実行を確定する前に見られること」）----

    def _unit_cost(self) -> float:
        """1行あたりの費用。実績があればそれを、無ければ実測の既定値を使う。"""
        u = self.client.usage
        paid = u.calls - u.cache_hits
        if paid > 0 and u.cost > 0:
            return u.cost / paid
        return COST_PER_ROW_USD

    def _latency_seconds(self, n_calls: int, with_setup: bool = True) -> float:
        workers = max(int(getattr(self.client, "max_workers", 1) or 1), 1)
        batches = math.ceil(n_calls / workers) if n_calls else 0
        return (SETUP_SECONDS if with_setup else 0.0) + batches * SECONDS_PER_CALL

    def plan(self, X: pd.DataFrame) -> dict:
        """**呼ぶ前に**「何行を LLM に回し、いくら・何秒かかるか」を返す。

        統計モデルは動かすが LLM は呼ばないので**無料**。
        予算に合わせて `escalate_rate` を決めるときはこれを見る。
        """
        m = self._check_fitted()
        m.spec.check(X)
        X = X.reset_index(drop=True)
        ev = self._evidence(X)
        s = self._signal_values(X, ev)
        sel = self._select(s)
        keys = self._row_keys(X)
        approved = np.array([k in self.approved_ for k in keys])
        n_llm = int((sel & ~approved).sum())
        return {
            "行数": len(X),
            "信号": self.signal if isinstance(self.signal, str) else "自作",
            "回す基準": (f"信号の強い上位 {self.escalate_rate:.0%}"
                       if self.escalate_rate is not None
                       else f"信号 >= {self.threshold}"),
            "LLM に回す行数": n_llm,
            "承認済みで省ける行数": int((sel & approved).sum()),
            "統計モデルで済ませる行数": len(X) - n_llm,
            "推定費用_usd": round(n_llm * self._unit_cost(), 4),
            "推定時間_秒": round(self._latency_seconds(n_llm), 1),
            "1行あたり_usd": round(self._unit_cost(), 6),
        }

    # --- predict --------------------------------------------------------

    def predict(self, X: pd.DataFrame, verbose: bool = False) -> np.ndarray:
        """選ばれた行だけ LLM に回し、残りは統計モデルの予測を返す。

        呼ばなかった行は `models[0]`（既定では LightGBM）に任せる。
        これは測定でルーティング曲線を引いたときと同じ扱いである。
        """
        m = self._check_fitted()
        m.spec.check(X)
        X = X.reset_index(drop=True)

        # 重複の検知は**全行に対して1回だけ**行う。内側の機能B に任せると
        # LLM に回した行だけを見ることになり、見落とす
        if self.predictor.check_leakage:
            warn_if_leaky(check_overlap(self.predictor.train_, X,
                                        keys=self.spec.all_columns()),
                          where="train と test")

        ev = self._evidence(X)
        s = self._signal_values(X, ev)
        sel = self._select(s)
        conf = self._to_confidence(s)

        keys = self._row_keys(X)
        approved = np.array([k in self.approved_ for k in keys])
        sel = sel & ~approved             # 承認済みは呼ばずに済む（高速パス）

        base_name = m.models_[0].name
        out = ev[base_name].astype(float).copy()
        preds: list[Prediction] = []
        for i in range(len(X)):
            row_ev = {name: float(v[i]) for name, v in ev.items()}
            if approved[i]:
                out[i] = self.approved_[keys[i]]
                preds.append(Prediction(
                    value=out[i], confidence=1.0,
                    reason="承認済みの答えを再利用した（LLM は呼んでいない）",
                    origin="human", model_predictions=row_ev, neighbours=[]))
            else:
                preds.append(Prediction(
                    value=float(out[i]), confidence=float(conf[i]),
                    reason=f"信号が弱いので LLM に回さず {base_name} に任せた",
                    origin="model", model_predictions=row_ev, neighbours=[]))

        # LLM に回す行だけ、機能B をそのまま使う（プロンプトも同一なのでキャッシュが効く）
        self._llm_pos: dict[int, int] = {}
        rows = np.flatnonzero(sel)
        if len(rows):
            sub = X.iloc[rows].reset_index(drop=True)
            was = m.check_leakage
            m.check_leakage = False          # 上で全行ぶん見たので二重に出さない
            try:
                values = m.predict(sub, verbose=verbose)
            finally:
                m.check_leakage = was
            for pos, gi in enumerate(rows):
                out[gi] = float(values[pos])
                preds[int(gi)] = m.predictions_[pos]
                self._llm_pos[int(gi)] = pos

        self.predictions_ = preds
        self.signal_ = s
        self.selected_ = sel
        self.keys_ = keys
        self.last_X_ = X
        return out

    # --- 検査 API（PRD「来歴（provenance）と検査 API」）-----------------

    def _require(self, X: pd.DataFrame | None = None) -> list[Prediction]:
        if getattr(self, "predictions_", None) is None:
            raise UnfoldError("predict を先に呼んでください。")
        if X is not None and len(X) != len(self.predictions_):
            raise UnfoldError(
                f"直近に predict した行数（{len(self.predictions_)}）と "
                f"X の行数（{len(X)}）が違います。検査 API は直前の "
                "predict の結果を返すので、同じ X を渡してください。")
        return self.predictions_

    def route(self) -> pd.DataFrame:
        """行ごとにどちらの経路を通ったか。信号の値も一緒に見られる。"""
        preds = self._require()
        return pd.DataFrame([
            {"行": i,
             "経路": "llm" if self.selected_[i] else "高速",
             "由来": p.origin,
             "信号": float(self.signal_[i]),
             "予測": p.value,
             "confidence": p.confidence,
             "費用_usd": p.cost}
            for i, p in enumerate(preds)])

    def confidence(self, X: pd.DataFrame | None = None) -> pd.Series:
        """行ごとの確信度。LLM に回した行は LLM の自己申告、
        回さなかった行は信号を 0〜1 に直したもの。"""
        return pd.Series([p.confidence for p in self._require(X)],
                         name="confidence")

    def examples(self, i: int | None = None) -> pd.DataFrame:
        """推論に使った類似事例。**LLM に回した行にしか無い。**"""
        self._require()
        if not self._llm_pos:
            return pd.DataFrame()
        if i is not None:
            if i not in self._llm_pos:
                return pd.DataFrame()
            df = self.predictor.examples(self._llm_pos[i])
            return df.assign(行=i)
        frames = []
        for gi, pos in self._llm_pos.items():
            df = self.predictor.examples(pos)
            if len(df):
                frames.append(df.assign(行=gi))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def cost(self, X: pd.DataFrame | None = None) -> dict:
        """実際にかかった費用と、全行に呼んだ場合との比較。"""
        preds = getattr(self, "predictions_", None)
        if X is not None and preds is not None and len(X) != len(preds):
            raise UnfoldError(
                f"直近に predict した行数（{len(preds)}）と "
                f"X の行数（{len(X)}）が違います。")
        u = self.client.summary()
        if preds:
            n = len(preds)
            n_llm = int(self.selected_.sum())
            u["予測した行数"] = n
            u["LLM に回した行数"] = n_llm
            u["LLM に回した割合"] = round(n_llm / n, 4)
            u["承認済みで省いた行数"] = int(sum(p.origin == "human" for p in preds))
            u["全行に呼んだ場合の推定_usd"] = round(n * self._unit_cost(), 4)
            u["節約できた額_usd"] = round((n - n_llm) * self._unit_cost(), 4)
        return u

    def explain(self, i: int) -> str:
        """その1行がなぜその値になったか。経路の説明から始める。"""
        p = self._require()[i]
        head = (f"[{i}行目] 経路 {'LLM' if self.selected_[i] else '高速（LLM を呼んでいない）'}"
                f" / 信号 {float(self.signal_[i]):,.6g}"
                f"（{'上位 %.0f%%' % (self.escalate_rate * 100) if self.escalate_rate is not None else '閾値 %s' % self.threshold} が対象）")
        if i in self._llm_pos:
            # 内側の機能B は「LLM に回した行だけ」を 0 から数え直しているので、
            # その行番号をそのまま出すと呼び出し側の行番号と食い違う。頭を落とす
            body = self.predictor.explain(self._llm_pos[i])
            return head + "\n" + re.sub(r"^\[\d+行目\]\s*", "", body)
        lines = [head, "",
                 f"予測 {p.value:,.6g} {self.unit}"
                 f"（由来 {p.origin} / confidence {p.confidence:.2f}）",
                 "",
                 "統計モデルの予測:"]
        for name, v in p.model_predictions.items():
            lines.append(f"  - {name}: {v:,.6g}")
        lines += ["", f"理由: {p.reason}"]
        return "\n".join(lines)

    def provenance(self) -> pd.DataFrame:
        """全行の来歴。`explain` の一覧版で、経路と信号の列が付く。"""
        preds = self._require()
        return pd.DataFrame([
            {"行": i, "予測": p.value, "由来": p.origin,
             "経路": "llm" if self.selected_[i] else "高速",
             "信号": float(self.signal_[i]),
             "confidence": p.confidence, "費用_usd": p.cost,
             "キャッシュ": p.from_cache, "理由": p.reason,
             **{f"証拠_{k}": v for k, v in p.model_predictions.items()}}
            for i, p in enumerate(preds)])

    # --- 能動学習（設計書 05「答えは教師ラベル候補としてキューされる」）----

    def review_queue(self) -> pd.DataFrame:
        """LLM が答えた行 = 教師ラベル候補。承認すると次回の高速パスに載る。"""
        preds = self._require()
        rows = []
        for i, p in enumerate(preds):
            if p.origin != "llm":
                continue
            rec = {"行": i, "指紋": self.keys_[i], "LLMの答え": p.value,
                   "confidence": p.confidence, "信号": float(self.signal_[i]),
                   "理由": p.reason}
            for c in self.spec.all_columns():
                if c in self.last_X_.columns:
                    rec[c] = self.last_X_[c].iloc[i]
            rows.append(rec)
        return pd.DataFrame(rows)

    def approve(self, rows: Sequence[int] | None = None) -> int:
        """答えを承認する。**次回以降その行は LLM を呼ばずに返る。**

        `rows` を省略すると、直近のキュー全部を承認する。
        照合は行番号ではなく行の中身（指紋）で行うので、
        次の predict で順番や件数が変わっていても効く。
        戻り値は承認した件数。
        """
        preds = self._require()
        targets = (range(len(preds)) if rows is None
                   else [int(r) for r in rows])
        n = 0
        for i in targets:
            if preds[i].origin != "llm":
                continue
            self.approved_[self.keys_[i]] = float(preds[i].value)
            n += 1
        return n

    # --- 閾値を振る（PRD 信頼度ルーティング「3つが同時に見える」）--------

    def curve(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None,
              rates: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0),
              verbose: bool = False) -> pd.DataFrame:
        """割合を振ったときの **精度・費用・レイテンシ**を1つの表にする。

        **この関数は全行に LLM を呼ぶ。**曲線を描くには「もし呼んでいたら
        いくつと答えたか」が全行ぶん要るためで、2回目からは
        ディスクキャッシュが効いて無料になる（同じ行に二度課金しない）。
        費用とレイテンシの列は「その割合だけ呼んだ場合」の値である。

        `y` を省略すると `X[target]` を使う。
        """
        m = self._check_fitted()
        m.spec.check(X)
        X = X.reset_index(drop=True)
        if y is None:
            if self.target not in X.columns:
                raise UnfoldError(
                    f"正解 {self.target!r} が X にありません。y を渡してください。")
            y = X[self.target]
        y = np.asarray(y, dtype=float)

        # 内側の機能B を全行で回すので、直前の predict の来歴は無効になる。
        # 黙って残すと explain() が別の行の証拠を返すため、ここで捨てる
        self.predictions_ = None
        self._llm_pos = {}

        ev = self._evidence(X)
        s = self._signal_values(X, ev)
        base = ev[m.models_[0].name].astype(float)
        llm = np.asarray(m.predict(X, verbose=verbose), dtype=float)

        order = np.argsort(-s, kind="stable")
        unit = self._unit_cost()
        rows = []
        for r in rates:
            k = int(round(len(X) * float(r)))
            pred = base.copy()
            if k:
                pick = order[:k]
                pred[pick] = llm[pick]
            rows.append({
                "割合": r,
                "LLM に回す行数": k,
                "MAE": float(np.mean(np.abs(pred - y))),
                "費用_usd": round(k * unit, 4),
                "推定時間_秒": round(self._latency_seconds(k), 1),
            })
        return pd.DataFrame(rows)

    # --- まとめ ---------------------------------------------------------

    def report(self) -> str:
        """1回の predict が何をしたかの要約。"""
        preds = self._require()
        n = len(preds)
        n_llm = int(self.selected_.sum())
        n_human = sum(p.origin == "human" for p in preds)
        paid = sum(p.cost for p in preds if not p.from_cache)
        return "\n".join([
            f"{n} 行を予測（信号: "
            f"{self.signal if isinstance(self.signal, str) else '自作'} / "
            f"{'上位 %.0f%%' % (self.escalate_rate * 100) if self.escalate_rate is not None else '閾値 %s' % self.threshold}）",
            f"  LLM に回した:        {n_llm} 行（{n_llm / max(n, 1):.1%}）",
            f"  承認済みで省いた:    {n_human} 行",
            f"  統計モデルで返した:  {n - n_llm - n_human} 行",
            f"  費用:                ${paid:.4f}"
            f"（全行に呼んだ場合の推定 ${n * self._unit_cost():.4f}）",
            f"  推定時間:            {self._latency_seconds(n_llm):.1f} 秒",
        ])

    def __repr__(self) -> str:
        sig = self.signal if isinstance(self.signal, str) else "自作"
        cut = (f"escalate_rate={self.escalate_rate}"
               if self.escalate_rate is not None else f"threshold={self.threshold}")
        return f"AdaptivePredictor(target={self.target!r}, signal={sig!r}, {cut})"
