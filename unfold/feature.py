"""機能A — `Feature`。非構造列を型のついた列に変える。

設計書の「What runs per row」（01〜05）をそのまま実装したもの。

  01 Ground truth      … 手元の正解ラベル。無い場合の扱いは下記（PRD 機能A の 01）
  02 Embedding         … 1行1ベクトル。同じ文字列は二度計算しない（キャッシュ）
  03 Nearest examples  … 近傍の類似度とラベルの一致度から confidence を出す
  04 Confident?        … 閾値以上ならその場で返す。LLM は呼ばない
  05 Uncertain?        … 閾値未満は fallback へ。既定はレビュー待ちに積むだけ

**01 の正解ラベルをどうするか**は未決（PRD 機能A の 01「教師ラベル」）なので、
このクラスは3つの入口を用意して**どれを選んだかが呼び出し側に見える**ようにした。

  (a) 利用者がラベルを渡す        … `fit(X, y)` または `Feature(labels="列名")`
  (b) 値の名前そのものを起点にする … `values=[...]` だけ渡す（ゼロショット）
  (c) 何も無い                    … `UnfoldError` で止める。黙って推測しない

(b) は仕様書の `df["x"] = Feature(source=..., values=[...]).fit_transform(df)` という
書き方（y を渡していない）を成り立たせるために要る。ラベル名を1件の参照事例と
みなして近傍分類にかけるだけなので、**LLM もラベル付け作業も要らない**。
起点として使えるかは実測できる（`scripts/demo_feature.py`）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from unfold.encoders import CachedEncoder, CharTfidfEncoder, Encoder
from unfold.errors import UnfoldError
from unfold.fallback import LLMFallback, QueueOnlyFallback
from unfold.preprocess import drop_constant_tokens

SUPPORTED_TYPES = ("category", "binary", "embedding")


class Feature:
    """非構造列 → 型のついた列。

    Parameters
    ----------
    source : str | list[str]
        入力にする列。複数渡すと空白で連結する（設計書の `source=["image", "description"]`）。
    type : str
        "category" / "binary" / "embedding"。int・float・ordinal・multilabel は未実装。
    values : list[str] | None
        取りうる値。type="category" では必須。ラベルが1件も無いときは
        この名前自体を参照点にする（上記 (b)）。
    labels : str | None
        正解ラベルが入っている列の名前。`fit(X, y)` の代わりに使える。
    k : int
        近傍として見る件数。
    threshold : float
        この confidence を下回った行を 05（フォールバック）に回す。
    escalate_rate : float | None
        「confidence が低い順に何割を回すか」で指定する。threshold より優先される。
        **費用の上限が決まっていない段階ではこちらの方が実務的**で、
        「予算内に収まる割合」から逆算できる。設計書のスライダーに相当する。
    on_uncertain : str
        "keep" なら分類器の推測を残す（来歴には needs_review と記録）。
        "null" なら欠損にする。
    preprocess : bool
        定数語の除去を行うか（既定 True。根拠は unfold/preprocess.py）。
    encoder : Encoder | None
        差し替え可能。既定は追加依存なしで動く CharTfidfEncoder。
    cache_dir : str | Path | None
        埋め込みキャッシュの置き場所。
    fallback : LLMFallback | None
        05 の逃がし先。既定はキューに積むだけの QueueOnlyFallback。
    """

    def __init__(self, source: str | list[str], type: str = "category",
                 values: list[str] | None = None, labels: str | None = None,
                 k: int | str = "auto", threshold: float = 0.9,
                 escalate_rate: float | None = None, on_uncertain: str = "keep",
                 preprocess: bool = True, encoder: Encoder | None = None,
                 cache_dir: str | Path | None = None,
                 fallback: LLMFallback | None = None, name: str | None = None):
        if type not in SUPPORTED_TYPES:
            raise UnfoldError(
                f"type={type!r} は未実装です。使えるのは {SUPPORTED_TYPES}。"
                "int / float / ordinal / multilabel は設計書にはあるが未着手。")
        if on_uncertain not in ("keep", "null"):
            raise UnfoldError("on_uncertain は 'keep' か 'null' です。")
        self.source = source
        self.type = type
        self.values = list(values) if values else None
        self.labels = labels
        self.k = k
        self.threshold = threshold
        self.escalate_rate = escalate_rate
        self.on_uncertain = on_uncertain
        self.preprocess = preprocess
        self.encoder = encoder
        self.cache_dir = cache_dir
        self.fallback = fallback if fallback is not None else QueueOnlyFallback()
        self.name = name

    # --- 入力の取り出し ------------------------------------------------

    def _source_cols(self) -> list[str]:
        return [self.source] if isinstance(self.source, str) else list(self.source)

    def _texts(self, X: pd.DataFrame) -> pd.Series:
        cols = self._source_cols()
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise UnfoldError(f"source に指定した列がありません: {missing}")
        s = X[cols[0]].fillna("").astype(str)
        for c in cols[1:]:
            s = s + " " + X[c].fillna("").astype(str)
        return s.reset_index(drop=True)

    def _prepared(self, X: pd.DataFrame, fitting: bool) -> pd.Series:
        s = self._texts(X)
        if not self.preprocess:
            return s
        if fitting:
            out, stop = drop_constant_tokens(s)
            self.stop_tokens_ = stop
            return out
        # 推論時は学習時に決めた語を使い回す（数え直すと表現がずれる）
        out, _ = drop_constant_tokens(s, stop=getattr(self, "stop_tokens_", []))
        return out

    # --- 01 教師ラベルの起点 -------------------------------------------

    def _ground_truth(self, X: pd.DataFrame, y=None) -> tuple[pd.Series, str]:
        """(ラベル, 起点の種類) を返す。起点は human / labelname。"""
        if y is not None:
            lab = pd.Series(np.asarray(y), index=range(len(X))).astype("object")
            return lab, "human"
        if self.labels is not None:
            if self.labels not in X.columns:
                raise UnfoldError(f"labels に指定した列がありません: {self.labels!r}")
            return X[self.labels].reset_index(drop=True).astype("object"), "human"
        if self.values:
            # (b) 値の名前そのものを参照点にする
            return pd.Series(dtype="object"), "labelname"
        raise UnfoldError(
            "教師ラベルも values も渡されていません。機能A は\n"
            "  (a) fit(X, y) か Feature(labels='列名') でラベルを渡す\n"
            "  (b) values=[...] を渡して値の名前を起点にする（ゼロショット）\n"
            "  (c) type='embedding' にしてラベルなしで埋め込み列を得る\n"
            "のいずれかが要ります。どれを既定にするかは未決（PRD 機能A の 01「教師ラベル」）。")

    # --- fit / transform -----------------------------------------------

    def fit(self, X: pd.DataFrame, y=None) -> "Feature":
        texts = self._prepared(X, fitting=True)
        enc = self.encoder if self.encoder is not None else CharTfidfEncoder()
        self.encoder_ = CachedEncoder(enc, cache_dir=self.cache_dir)

        if self.type == "embedding":
            self.encoder_.fit(texts)
            self.reference_ = None
            return self

        lab, origin = self._ground_truth(X, y)
        if origin == "human":
            mask = lab.notna() & (lab.astype(str) != "")
            if not mask.any():
                raise UnfoldError(
                    "渡されたラベルが全て欠損です。機能A は正解を1件も持たずには"
                    "近傍分類できません（PRD 機能A の 01「教師ラベル」）。")
            ref_texts = texts[mask.to_numpy()].reset_index(drop=True)
            ref_labels = lab[mask].astype(str).reset_index(drop=True)
            ref_origin = np.array(["human"] * len(ref_labels))
        else:
            # 値の名前そのものを1件ずつの参照事例にする
            ref_texts = pd.Series(self.values, dtype="object")
            ref_labels = pd.Series(self.values, dtype="object")
            ref_origin = np.array(["labelname"] * len(ref_labels))

        # エンコーダは**行全体**で学習する（参照事例だけだと語彙が偏る）。
        # 目的変数を見ないので、これは交差検証のリークにはあたらない
        self.encoder_.fit(texts)
        self.reference_ = {
            "texts": ref_texts,
            "labels": ref_labels.to_numpy(),
            "origin": ref_origin,
            "V": self.encoder_.transform(ref_texts),
        }
        self.classes_ = sorted(set(ref_labels) | set(self.values or []))
        # k="auto": 1クラスあたりの参照事例数を超える k は意味がない。
        # 値の名前だけを起点にした場合（1クラス1件）に k=5 とすると、
        # 近傍が必ず5クラスに割れて「全行が自信なし」になる（実測で確認）
        per_class = len(ref_labels) / max(len(set(ref_labels)), 1)
        self.k_ = max(1, min(5, int(per_class))) if self.k == "auto" else int(self.k)
        return self

    def transform(self, X: pd.DataFrame):
        if not hasattr(self, "encoder_"):
            raise UnfoldError("先に fit してください。")
        texts = self._prepared(X, fitting=False)
        V = self.encoder_.transform(texts)

        if self.type == "embedding":
            cols = [f"{self._name()}_{i}" for i in range(V.shape[1])]
            self.provenance_ = pd.DataFrame({
                "値": ["<埋め込み>"] * len(texts), "confidence": 1.0,
                "由来": "model", "コスト": 0.0})
            return pd.DataFrame(V, columns=cols, index=X.index)

        ref = self.reference_
        sim = V @ ref["V"].T                       # 正規化済み → 内積＝コサイン類似度
        # 近傍は k 件見るが、2位のクラスまで比べたいので最低2件は取る
        k = min(max(getattr(self, "k_", 5), 2), sim.shape[1])
        top = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
        # argpartition は順不同なので、選んだ k 件だけを並べ替える
        rows = np.arange(len(V))[:, None]
        order = np.argsort(-sim[rows, top], axis=1)
        top = top[rows, order]

        values, confs, origins, examples = [], [], [], []
        for r in range(len(V)):
            idx = top[r]
            sims = np.clip(sim[r, idx], 0, None)
            labs = ref["labels"][idx]
            # 03 類似度で重みづけした多数決。ラベルの一致度がそのまま confidence
            score: dict[str, float] = {}
            for l, s in zip(labs, sims):
                score[l] = score.get(l, 0.0) + float(s)
            ranked = sorted(score.values(), reverse=True)
            best = max(score, key=score.get)
            second = ranked[1] if len(ranked) > 1 else 0.0
            # 03 confidence = 1位のクラスが2位に対してどれだけ強いか。
            # 「近傍の何割が同じラベルか」を素直に使うと、1クラス1件の参照事例
            # （値の名前を起点にした場合）で必ず割れてしまうため、相対比にする。
            # **この値はまだ較正されていない**（絶対値に意味を持たせない）。
            # 順序づけとしては使えることを demo_feature.py で確認している
            denom = ranked[0] + second
            conf = float(ranked[0] / denom) if denom > 0 else 0.0
            values.append(best)
            confs.append(conf)
            origins.append("model")
            examples.append([
                {"参照": int(i), "値": str(ref["labels"][i]),
                 "類似度": float(sim[r, i]), "由来": str(ref["origin"][i]),
                 "テキスト": str(ref["texts"].iloc[i])} for i in idx])

        prov = pd.DataFrame({"値": values, "confidence": confs, "由来": origins,
                             "コスト": 0.0})
        prov["参照事例"] = examples
        prov["テキスト"] = texts.to_numpy()

        # 04 / 05 — 閾値、または「低い順に何割」で振り分ける
        if self.escalate_rate is not None:
            # 分位点で切ると同点が多いときに割合がずれる（実測で 25% 指定が
            # 100% になった）ので、確信度の低い順に必要な件数だけ選ぶ
            n_esc = int(round(len(prov) * self.escalate_rate))
            order = np.argsort(prov["confidence"].to_numpy(), kind="stable")
            flag = np.zeros(len(prov), dtype=bool)
            flag[order[:n_esc]] = True
            uncertain = pd.Series(flag, index=prov.index)
        else:
            uncertain = prov["confidence"] < self.threshold
        n_escalate = int(uncertain.sum())
        if n_escalate:
            self._escalate(prov, uncertain)

        self.provenance_ = prov
        out = pd.Series(prov["値"].to_numpy(), index=X.index, name=self._name())
        if self.on_uncertain == "null":
            out = out.where(~uncertain.to_numpy())
        return out.astype("category") if self.type == "category" else out

    def fit_transform(self, X: pd.DataFrame, y=None):
        return self.fit(X, y).transform(X)

    def _escalate(self, prov: pd.DataFrame, uncertain: pd.Series) -> None:
        """05 — confidence が閾値未満の行を fallback に回す。"""
        rows = np.flatnonzero(uncertain.to_numpy())
        fb = self.fallback
        if fb is not None and getattr(fb, "can_answer", lambda: False)():
            ans = fb.answer([prov["テキスト"].iloc[r] for r in rows], self.values,
                            [{"examples": prov["参照事例"].iloc[r]} for r in rows])
            for r, a in zip(rows, ans):
                if a.value is not None:
                    prov.loc[r, "値"] = a.value
                prov.loc[r, "confidence"] = a.confidence
                prov.loc[r, "由来"] = a.origin
                prov.loc[r, "コスト"] = a.cost
            return
        # 呼べないときは「レビュー待ち」として積む。値は推測のまま残す
        for r in rows:
            prov.loc[r, "由来"] = "needs_review"
            if hasattr(fb, "enqueue"):
                fb.enqueue(int(r), prov["テキスト"].iloc[r], prov["値"].iloc[r],
                           float(prov["confidence"].iloc[r]))

    # --- 検査 API（設計書 "the part that is new"）------------------------

    def _name(self) -> str:
        if self.name:
            return self.name
        cols = self._source_cols()
        return f"{'_'.join(cols)}_feature"

    def _prov(self) -> pd.DataFrame:
        if not hasattr(self, "provenance_"):
            raise UnfoldError("先に transform してください。")
        return self.provenance_

    def confidence(self, X: pd.DataFrame | None = None) -> pd.Series:
        """行ごとの確信度。X を渡すと計算し直す。"""
        if X is not None:
            self.transform(X)
        return self._prov()["confidence"]

    def examples(self, X: pd.DataFrame | None = None) -> pd.DataFrame:
        """各行が参照した事例を平らな表にして返す。"""
        if X is not None:
            self.transform(X)
        rows = []
        for i, ex in enumerate(self._prov()["参照事例"]):
            for e in ex:
                rows.append({"行番号": i, **e})
        return pd.DataFrame(rows)

    def cost(self, X: pd.DataFrame | None = None) -> dict:
        """費用の見積もり。**実行前に**エスカレーション率を知るためのもの。"""
        if X is not None:
            self.transform(X)
        prov = self._prov()
        n = len(prov)
        n_esc = int((prov["由来"].isin(["llm", "needs_review"])).sum())
        per = float(getattr(self.fallback, "cost_per_call", 0.0))
        return {"行数": n, "LLM に回る行": n_esc,
                "割合": round(n_esc / n, 4) if n else 0.0,
                "1行あたり": per, "合計見積もり": round(n_esc * per, 6),
                "実際に発生した費用": float(prov["コスト"].sum())}

    def explain(self, i: int) -> str:
        """1行ぶんの来歴を人が読める形にする（設計書の model.explain 相当）。"""
        prov = self._prov()
        r = prov.iloc[i]
        lines = [
            f"予測値            {r['値']}",
            f"confidence        {r['confidence']:.3f}",
            f"由来              {r['由来']}",
            f"戦略              embedding_classifier"
            f"（{'下位 %.0f%%' % (self.escalate_rate * 100) if self.escalate_rate is not None else '閾値 %s' % self.threshold}）",
            f"エンコーダ        {self.encoder_.name}",
            f"入力テキスト      {r['テキスト'][:60]}",
            "参照した事例:",
        ]
        for e in r["参照事例"]:
            lines.append(f"  #{e['参照']:<6} {e['値']:<12} 類似 {e['類似度']:.3f}"
                         f"  {e['由来']:<10} {e['テキスト'][:36]}")
        lines.append(f"LLM 呼び出し      {1 if r['由来'] == 'llm' else 0}")
        lines.append(f"費用              ${r['コスト']:.4f}")
        return "\n".join(lines)

    def review_queue(self) -> pd.DataFrame:
        """レビュー待ちの候補（設計書の `unfold review` 相当）。"""
        q = getattr(self.fallback, "queued", [])
        return pd.DataFrame(q)

    def status(self) -> dict:
        """設計書の `unfold status <feature>` 相当。"""
        prov = self._prov()
        ref = self.reference_ or {"origin": np.array([])}
        counts = pd.Series(ref["origin"]).value_counts().to_dict()
        return {
            "特徴量": self._name(),
            "レコード数": len(prov),
            "参照事例（人手）": int(counts.get("human", 0)),
            "参照事例（値の名前）": int(counts.get("labelname", 0)),
            "モデルが答えた行": int((prov["由来"] == "model").sum()),
            "LLM が答えた行": int((prov["由来"] == "llm").sum()),
            "レビュー待ち": int((prov["由来"] == "needs_review").sum()),
            "エンコーダ": self.encoder_.name,
            "近傍数 k": getattr(self, "k_", self.k),
            "閾値": (f"下位 {self.escalate_rate:.0%}"
                     if self.escalate_rate is not None else self.threshold),
        }
