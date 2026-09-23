"""`SemanticEncoder`. Turns an unstructured column into a typed column.

What runs per row, in five steps:

  01 Ground truth      ... labels at hand. What to do without them is below
  02 Embedding         ... one vector per row. The same string is never computed twice (cache)
  03 Nearest examples  ... confidence comes from neighbour similarity and label agreement
  04 Confident?        ... at or above the threshold, return on the spot. The LLM is not called
  05 Uncertain?        ... below the threshold goes to the fallback. By default it is
                           only queued for review. The fallback can be a chain
                           (`[JevFallback(), LLMFallback()]`): Jev answers what it is
                           confident about, the rest goes on to the frontier LLM

**What to use as the ground truth in 01** depends on the data, so this class offers three
entry points and **makes the chosen one visible to the caller**.

  (a) the user passes labels              ... `fit(X, y)` or `SemanticEncoder(labels="column")`
  (b) start from the value names themselves ... pass only `values=[...]` (zero-shot)
  (c) nothing                             ... stop with `MekikiError`. Never guess silently

(b) is what makes `df["x"] = SemanticEncoder(source=..., values=[...]).fit_transform(df)`
(no y passed) work. Each label name is treated as a single reference
example for nearest-neighbour classification, so **neither an LLM nor labelling work is
needed**. Whether it is good enough as a starting point is worth measuring on your own data.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from mekiki.errors import MekikiError
from mekiki.fallback import Fallback, JevFallback, QueueOnlyFallback
from mekiki.preprocess import drop_constant_tokens
from mekiki.vectorizers import CachedVectorizer, CharTfidfVectorizer, Vectorizer

SUPPORTED_TYPES = ("category", "binary", "embedding")


class SemanticEncoder:
    """Unstructured column -> typed column.

    Parameters
    ----------
    source : str | list[str]
        Input column(s). Several are joined with spaces
        (e.g. `source=["title", "description"]`).
    type : str
        "category" / "binary" / "embedding". int, float, ordinal and multilabel are not implemented.
    values : list[str] | None
        Possible values. Required for type="category". When there is not a single label,
        the names themselves become the reference points ((b) above).
    labels : str | None
        Name of the column holding the ground-truth labels. Can be used instead of `fit(X, y)`.
    k : int
        Number of neighbours to look at.
    threshold : float
        Rows whose confidence falls below this go to 05 (the fallback).
    escalate_rate : float | None
        Specify instead "what share of rows, lowest confidence first, to escalate".
        Takes precedence over threshold. **More practical when you think in budgets**,
        since it can be derived from "the share that fits the budget".
    on_uncertain : str
        "keep" keeps the classifier's guess (provenance records needs_review).
        "null" turns it into a missing value.
    preprocess : bool
        Whether to drop constant tokens (default True; rationale in mekiki/preprocess.py).
    vectorizer : Vectorizer | None
        Swappable. The default is CharTfidfVectorizer, which needs no extra dependencies.
    cache_dir : str | Path | None
        Where the embedding cache lives.
    fallback : Fallback | Sequence[Fallback] | None
        Where 05 escapes to. The default is QueueOnlyFallback, which only queues.
        A list is a chain of stages tried in order (`[JevFallback(), LLMFallback()]`):
        each stage answers the rows still open, a row whose answer falls below that
        stage's `threshold` (or has no value) moves on to the next stage, and whatever
        is left at the end is queued for review. A stage that cannot be called (no
        key) is skipped, so the chain degrades to a single fallback, and to the queue.
    """

    def __init__(self, source: str | list[str], type: str = "category",
                 values: list[str] | None = None, labels: str | None = None,
                 k: int | str = "auto", threshold: float = 0.9,
                 escalate_rate: float | None = None, on_uncertain: str = "keep",
                 preprocess: bool = True, vectorizer: Vectorizer | None = None,
                 cache_dir: str | Path | None = None,
                 fallback: Fallback | Sequence[Fallback] | None = None,
                 name: str | None = None):
        if type not in SUPPORTED_TYPES:
            raise MekikiError(
                f"type={type!r} is not implemented. Available: {SUPPORTED_TYPES}. "
                "int / float / ordinal / multilabel are not supported yet.")
        if on_uncertain not in ("keep", "null"):
            raise MekikiError("on_uncertain must be 'keep' or 'null'.")
        self.source = source
        self.type = type
        self.values = list(values) if values else None
        self.labels = labels
        self.k = k
        self.threshold = threshold
        self.escalate_rate = escalate_rate
        self.on_uncertain = on_uncertain
        self.preprocess = preprocess
        self.vectorizer = vectorizer
        self.cache_dir = cache_dir
        if fallback is None:
            fallback = QueueOnlyFallback()
        elif isinstance(fallback, (list, tuple)):
            fallback = list(fallback)
            if not fallback:
                raise MekikiError("fallback=[] is empty. Pass at least one stage, or omit it.")
        self.fallback = fallback
        self.name = name

    def _stages(self) -> list:
        """The fallback chain as a list (a single fallback is a chain of one)."""
        fb = self.fallback
        return list(fb) if isinstance(fb, list) else [fb]

    # --- Extracting the input ------------------------------------------

    def _source_cols(self) -> list[str]:
        return [self.source] if isinstance(self.source, str) else list(self.source)

    def _texts(self, X: pd.DataFrame) -> pd.Series:
        cols = self._source_cols()
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise MekikiError(f"Columns given in source do not exist: {missing}")
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
        # At inference, reuse the tokens decided at fit time (recounting shifts the representation)
        out, _ = drop_constant_tokens(s, stop=getattr(self, "stop_tokens_", []))
        return out

    # --- 01 Where the ground truth comes from ---------------------------

    def _ground_truth(self, X: pd.DataFrame, y=None) -> tuple[pd.Series, str]:
        """Return (labels, kind of origin). The origin is human / labelname."""
        if y is not None:
            lab = pd.Series(np.asarray(y), index=range(len(X))).astype("object")
            return lab, "human"
        if self.labels is not None:
            if self.labels not in X.columns:
                raise MekikiError(f"Column given in labels does not exist: {self.labels!r}")
            return X[self.labels].reset_index(drop=True).astype("object"), "human"
        if self.values:
            # (b) the value names themselves become the reference points
            return pd.Series(dtype="object"), "labelname"
        raise MekikiError(
            "Neither ground-truth labels nor values were given. SemanticEncoder needs one of\n"
            "  (a) labels via fit(X, y) or SemanticEncoder(labels='column')\n"
            "  (b) values=[...] so the value names become the starting point (zero-shot)\n"
            "  (c) type='embedding' to get embedding columns without labels\n"
            "There is no default; pick the one that fits your data.")

    # --- fit / transform -----------------------------------------------

    def fit(self, X: pd.DataFrame, y=None) -> SemanticEncoder:
        """Fit the vectorizer and store the reference points. **Does not call the LLM.**

        Parameters
        ----------
        X:
            A table containing the `source` column(s), and the `labels` column if one was declared.
        y:
            Ground-truth labels, one per row of X. Missing entries are allowed; only the labelled
            rows become reference points. Leave out to use `labels`, or the value names themselves.

        Returns
        -------
        `self`.
        """
        texts = self._prepared(X, fitting=True)
        enc = self.vectorizer if self.vectorizer is not None else CharTfidfVectorizer()
        self.vectorizer_ = CachedVectorizer(enc, cache_dir=self.cache_dir)

        if self.type == "embedding":
            self.vectorizer_.fit(texts)
            self.reference_ = None
            return self

        lab, origin = self._ground_truth(X, y)
        if origin == "human":
            mask = lab.notna() & (lab.astype(str) != "")
            if not mask.any():
                raise MekikiError(
                    "All given labels are missing. SemanticEncoder cannot run nearest-neighbour "
                    "classification without a single label.")
            ref_texts = texts[mask.to_numpy()].reset_index(drop=True)
            ref_labels = lab[mask].astype(str).reset_index(drop=True)
            ref_origin = np.array(["human"] * len(ref_labels))
        else:
            # Each value name becomes one reference example
            ref_texts = pd.Series(self.values, dtype="object")
            ref_labels = pd.Series(self.values, dtype="object")
            ref_origin = np.array(["labelname"] * len(ref_labels))

        # The vectorizer is fitted on **all rows** (reference examples alone skew the vocabulary).
        # It never sees the target, so this is not leakage in cross-validation
        self.vectorizer_.fit(texts)
        self.reference_ = {
            "texts": ref_texts,
            "labels": ref_labels.to_numpy(),
            "origin": ref_origin,
            "V": self.vectorizer_.transform(ref_texts),
        }
        self.classes_ = sorted(set(ref_labels) | set(self.values or []))
        # k="auto": a k larger than the number of reference examples per class is meaningless.
        # Starting from value names only (one example per class) with k=5 always splits the
        # neighbours across 5 classes and makes "every row uncertain" (confirmed empirically)
        per_class = len(ref_labels) / max(len(set(ref_labels)), 1)
        self.k_ = max(1, min(5, int(per_class))) if self.k == "auto" else int(self.k)
        return self

    def transform(self, X: pd.DataFrame) -> pd.Series | pd.DataFrame:
        """Build the new column for the rows of X. **X itself is not modified.**

        Rows whose confidence is too low go to the `fallback`. With an `Fallback` this is
        the step that calls the LLM and costs money; the default fallback only queues them.

        Parameters
        ----------
        X:
            A table containing the `source` column(s).

        Returns
        -------
        Only the new column, carrying the index of X, so it can be assigned straight back
        (`df["name"] = col.transform(df)`).

        - type="category" / "binary": a `pd.Series` named after the column (category dtype for
          "category"). With on_uncertain="null" the uncertain rows are missing.
        - type="embedding": a `pd.DataFrame` with one column per dimension (`<name>_0`, `<name>_1`, ...).

        The per-row record (value, confidence, source, cost, references) is kept in `provenance_`
        and replaced on every call.
        """
        if not hasattr(self, "vectorizer_"):
            raise MekikiError("Call fit first.")
        texts = self._prepared(X, fitting=False)
        V = self.vectorizer_.transform(texts)

        if self.type == "embedding":
            cols = [f"{self._name()}_{i}" for i in range(V.shape[1])]
            self.provenance_ = pd.DataFrame({
                "value": ["<EMBEDDING>"] * len(texts), "confidence": 1.0,
                "source": "model", "cost": 0.0})
            return pd.DataFrame(V, columns=cols, index=X.index)

        ref = self.reference_
        sim = V @ ref["V"].T                       # normalised -> dot product = cosine similarity
        # Look at k neighbours, but take at least 2 so the second-best class can be compared
        k = min(max(getattr(self, "k_", 5), 2), sim.shape[1])
        top = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
        # argpartition is unordered, so sort only the k chosen entries
        rows = np.arange(len(V))[:, None]
        order = np.argsort(-sim[rows, top], axis=1)
        top = top[rows, order]

        values, confs, origins, examples = [], [], [], []
        for r in range(len(V)):
            idx = top[r]
            sims = np.clip(sim[r, idx], 0, None)
            labs = ref["labels"][idx]
            # 03 similarity-weighted majority vote. Label agreement is the confidence
            score: dict[str, float] = {}
            for label, s in zip(labs, sims, strict=True):
                score[label] = score.get(label, 0.0) + float(s)
            ranked = sorted(score.values(), reverse=True)
            best = max(score, key=score.get)
            second = ranked[1] if len(ranked) > 1 else 0.0
            # 03 confidence = how strongly the top class beats the second.
            # Using "what share of neighbours carry the same label" naively always splits
            # with one reference example per class (value names as the starting point),
            # so a relative ratio is used instead.
            # **This value is not calibrated yet** (do not read meaning into the absolute value).
            # It is usable for ordering rows (least confident first), which is how it is used
            denom = ranked[0] + second
            conf = float(ranked[0] / denom) if denom > 0 else 0.0
            values.append(best)
            confs.append(conf)
            origins.append("model")
            examples.append([
                {"ref": int(i), "value": str(ref["labels"][i]),
                 "similarity": float(sim[r, i]), "source": str(ref["origin"][i]),
                 "text": str(ref["texts"].iloc[i])} for i in idx])

        prov = pd.DataFrame({"value": values, "confidence": confs, "source": origins,
                             "cost": 0.0})
        prov["references"] = examples
        prov["text"] = texts.to_numpy()

        # 04 / 05 — split by threshold, or by "what share, lowest first"
        if self.escalate_rate is not None:
            # Cutting at a quantile drifts when there are many ties (a 25% request
            # became 100% empirically), so pick exactly the needed count, lowest confidence first
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
        out = pd.Series(prov["value"].to_numpy(), index=X.index, name=self._name())
        if self.on_uncertain == "null":
            out = out.where(~uncertain.to_numpy())
        return out.astype("category") if self.type == "category" else out

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.Series | pd.DataFrame:
        """`fit(X, y)` followed by `transform(X)`. Returns what `transform` returns."""
        return self.fit(X, y).transform(X)

    def _escalate(self, prov: pd.DataFrame, uncertain: pd.Series) -> None:
        """05 — send rows whose confidence is below the threshold down the fallback chain.

        Stage k answers the rows still open. A row is settled by the first stage whose
        answer has a value and a confidence at or above that stage's `threshold` (a
        stage without one, such as `LLMFallback`, settles everything it can answer).
        The cost of every stage a row passed through is added up. Rows nobody settled
        are queued as "needs_review" with the classifier's guess left in place.
        """
        pending = [int(r) for r in np.flatnonzero(uncertain.to_numpy())]
        # Candidates are the declared values. When fitted from a label column only,
        # the values that appeared there (classes_) become the candidates. Asking a
        # model without candidates lets arbitrary spellings leak into the feature,
        # so always pass something
        candidates = self.values or [str(c) for c in getattr(self, "classes_", [])] or None
        stages = self._stages()
        for stage in stages:
            if not pending:
                break
            if stage is None or not getattr(stage, "can_answer", lambda: False)():
                continue          # no key for this stage: fall through to the next one
            ans = stage.answer([prov["text"].iloc[r] for r in pending], candidates,
                               [{"examples": prov["references"].iloc[r]} for r in pending])
            cut = getattr(stage, "threshold", None)
            still_open = []
            for r, a in zip(pending, ans, strict=True):
                prov.loc[r, "cost"] = float(prov["cost"].iloc[r]) + a.cost
                if a.value is None or (cut is not None and a.confidence < cut):
                    still_open.append(r)
                    continue
                prov.loc[r, "value"] = a.value
                prov.loc[r, "confidence"] = a.confidence
                prov.loc[r, "source"] = a.origin
            pending = still_open
        # Whatever no stage settled is queued as "pending review". The guess stays as the
        # value. The queue lives on the last stage that has one
        queue = next((st for st in reversed(stages) if hasattr(st, "enqueue")), None)
        for r in pending:
            prov.loc[r, "source"] = "needs_review"
            if queue is not None:
                queue.enqueue(r, prov["text"].iloc[r], prov["value"].iloc[r],
                              float(prov["confidence"].iloc[r]))

    # --- Inspection API (provenance) ---

    def _name(self) -> str:
        if self.name:
            return self.name
        cols = self._source_cols()
        return f"{'_'.join(cols)}_feature"

    def _prov(self) -> pd.DataFrame:
        if not hasattr(self, "provenance_"):
            raise MekikiError("Call transform first.")
        return self.provenance_

    def confidence(self, X: pd.DataFrame | None = None) -> pd.Series:
        """Per-row confidence. Passing X recomputes it."""
        if X is not None:
            self.transform(X)
        return self._prov()["confidence"]

    def examples(self, X: pd.DataFrame | None = None) -> pd.DataFrame:
        """Return the examples each row referenced as a flat table."""
        if X is not None:
            self.transform(X)
        rows = []
        for i, ex in enumerate(self._prov()["references"]):
            for e in ex:
                rows.append({"row_id": i, **e})
        return pd.DataFrame(rows)

    def cost(self, X: pd.DataFrame | None = None) -> dict:
        """Cost estimate. Meant to learn the escalation rate **before** running."""
        if X is not None:
            self.transform(X)
        prov = self._prov()
        n = len(prov)
        n_esc = int((prov["source"].isin(["jev", "llm", "needs_review"])).sum())
        # With a chain, a row costs at most the sum of every stage it can pass through
        per = float(sum(float(getattr(st, "cost_per_call", 0.0)) for st in self._stages()))
        return {"n_rows": n, "n_escalated": n_esc,
                "rate": round(n_esc / n, 4) if n else 0.0,
                "cost_per_row_usd": per, "estimated_total": round(n_esc * per, 6),
                "actual_cost_usd": float(prov["cost"].sum())}

    def explain(self, i: int) -> str:
        """Render one row's provenance in human-readable form."""
        prov = self._prov()
        r = prov.iloc[i]
        how = (f"bottom {self.escalate_rate * 100:.0f}%" if self.escalate_rate is not None
               else f"threshold {self.threshold}")
        lines = [
            f"prediction        {r['value']}",
            f"confidence        {r['confidence']:.3f}",
            f"source            {r['source']}",
            f"strategy          embedding_classifier ({how})",
            f"vectorizer           {self.vectorizer_.name}",
            f"input text        {r['text'][:60]}",
            "references:",
        ]
        for e in r["references"]:
            lines.append(f"  #{e['ref']:<6} {e['value']:<12} sim {e['similarity']:.3f}"
                         f"  {e['source']:<10} {e['text'][:36]}")
        if any(isinstance(st, JevFallback) for st in self._stages()):
            # A row that Jev could not settle also passed through Jev before the LLM
            lines.append(f"Jev calls         {1 if r['source'] in ('jev', 'llm', 'needs_review') else 0}")
        lines.append(f"LLM calls         {1 if r['source'] == 'llm' else 0}")
        lines.append(f"cost              ${r['cost']:.4f}")
        return "\n".join(lines)

    def review_queue(self) -> pd.DataFrame:
        """Candidates pending review (over every stage of the chain)."""
        q: list[dict] = []
        for st in self._stages():
            q.extend(getattr(st, "queued", []) or [])
        return pd.DataFrame(q)

    def status(self) -> dict:
        """Summary of where the ground truth came from and how rows were routed."""
        prov = self._prov()
        ref = self.reference_ or {"origin": np.array([])}
        counts = pd.Series(ref["origin"]).value_counts().to_dict()
        return {
            "feature": self._name(),
            "n_records": len(prov),
            "references_human": int(counts.get("human", 0)),
            "references_value_names": int(counts.get("labelname", 0)),
            "model_answered": int((prov["source"] == "model").sum()),
            "jev_answered": int((prov["source"] == "jev").sum()),
            "llm_answered": int((prov["source"] == "llm").sum()),
            "pending_review": int((prov["source"] == "needs_review").sum()),
            "vectorizer": self.vectorizer_.name,
            "k": getattr(self, "k_", self.k),
            "threshold": (f"bottom {self.escalate_rate:.0%}"
                          if self.escalate_rate is not None else self.threshold),
        }
