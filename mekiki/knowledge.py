"""Knowledge columns — `KnowledgeEncoder`. Adds columns the table lacks from the LLM's knowledge.

`SemanticEncoder` builds columns from information **inside the data** (text, images).
This module builds columns from information **outside the data** (what the LLM knows as general
knowledge), keyed by the values of structured columns such as `manufacturer` / `model`
or `country` / `variety`. Because the information source differs, it is expected to stack on top of
`SemanticEncoder`.

There is one way to use it: **put a table in, get back the columns it lacked (a DataFrame).**
What to look up may be specified, or left unspecified.

    KnowledgeEncoder(target="price").fit_transform(df)
        ... the LLM also decides what to look up. Only columns that helped are kept
            (scoring does not call the LLM)
    KnowledgeEncoder(attribute="approximate price when new").fit_transform(df)
        ... only what to look up is decided. The LLM decides the keys and the type
    KnowledgeEncoder(keys=["manufacturer", "model"], attribute="body style",
                    values=[...]).fit_transform(df)
        ... everything is decided

In a measurement on the Craigslist used-car data, predicting
"`model` -> body style" with `SemanticEncoder` reached accuracy 0.14-0.23 from value names alone, and
0.64-0.82 even on rows sent to the LLM. Guessing an attribute that is not written in the text
by neighbor classification is a poor fit; asking the LLM directly per key value is more natural.
That is the starting point of this module.

Three design points.

1. **The unit is a "key value", not a row.** Each unique combination of key values is asked
   once and merged back onto the rows. The same key value is never asked twice (the same idea as
   `CachedVectorizer`). Answers are persisted to parquet, so the same key value is reused across
   datasets.
2. **Four guards against hallucination.** (1) closed candidate sets (answers outside the
   candidates or the range are rejected), (2) letting the LLM say "I don't know"
   (`known=false` becomes missing), (3) cross-checking against a column with the same attribute
   when the data has one, reporting the agreement rate (`check_against`), (4) requiring a reason
   and keeping it in the provenance.
3. **Automatically added columns are only those that pass scoring.** With `target`, a tree model
   with the column and one without are compared on the same folds, and only columns that helped
   are returned. Adding whatever the LLM came up with would mix in useless or noisy columns.
   Scoring does not call the LLM.

Provenance has the same shape as in `SemanticEncoder`. `source` takes one of four values.

  llm       ... the LLM answered and the answer passed validation
  jev       ... Jev answered with enough confidence (category / binary only, `jev=True`)
  unknown   ... the LLM answered that it does not know
  rejected  ... an answer came back but was rejected: outside the candidates / range,
                insufficient confidence, or a key mismatch
  skipped   ... not asked (key missing, or seen fewer than min_count times)

Rows other than llm become missing (NaN). Tree models handle missing values natively, so the
column can be added to `ColumnSpec` numeric / categorical as is.

LLM calls go through `ClaudeClient` (cache, cost accounting and parallelism are shared).
**No domain-specific wording is written in this module.** The role and subject come from `Domain`.

**Jev before the LLM (`jev=True`).** For closed candidate sets (`type="category"` /
`"binary"`) each unknown key value is first put to Jev as one `choice` question: the state
is the key columns and their values, the options are the candidates plus "unknown". A key
Jev answers with confidence at or above the lookup's threshold is stored with
`source="jev"` (and persisted like an LLM answer); the rest, and every "unknown", go to the
LLM batch as before. `type="numeric"` lookups go straight to the LLM, since Jev has no
numeric output. Without `TYPESAFE_API_KEY` (or `AI_GATEWAY_API_KEY`) the Jev tier is
skipped.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mekiki.errors import MekikiError
from mekiki.jev import JevClient
from mekiki.llm import DEFAULT_MODEL, PRICING, ClaudeClient, LLMAnswer
from mekiki.paths import cache_dir as _cache_dir
from mekiki.predictor import Domain

SUPPORTED_TYPES = ("category", "binary", "numeric")

#: Separator used when listing key values. The answer's `key` must be copied in this form
KEY_SEP = " | "

#: Approximate token counts for cost estimation. Calibrated on a measurement with the
#: Craigslist 500-row excerpt (306 key values, batch_size=20, 32 requests): input 358 / request,
#: cache read 592 / request, output 69 / key value. Estimate $1.14 against a measured $1.126.
#: The system prompt is shared by all requests, so it lands in the prompt cache and costs 1/10
#: from the second request on.
EST_SYSTEM_TOKENS = 600
EST_CALL_OVERHEAD_TOKENS = 60      # fixed part of the user prompt (columns, candidates, instructions)
EST_INPUT_TOKENS_PER_KEY = 15
EST_OUTPUT_TOKENS_PER_KEY = 70

#: Approximate cost of one proposal call (automatic mode). The column list is sent and a few
#: candidates come back
EST_PROPOSAL_USD = 0.05
PROPOSAL_MAX_TOKENS = 4096

#: Default confidence thresholds. For numeric "approximate" values the LLM tends to give
#: 0.45-0.65 (measured)
DEFAULT_THRESHOLD = {"category": 0.7, "binary": 0.7, "numeric": 0.4}

#: Minimum contribution (improvement rate from without -> with) for a column to count as "helped"
SCORE_THRESHOLD = 0.0

#: The extra option offered to Jev so that it can decline instead of guessing
JEV_UNKNOWN = "unknown"

#: Sources that carry an accepted value (the others become missing)
ANSWERED = ("llm", "jev")


def _max_tokens_for(batch_size: int) -> int:
    """max_tokens so that one request's answer is not cut off. The default 1024 cannot hold 20 items."""
    return max(1024, batch_size * 80 + 200)


LOOKUP_SYSTEM = """\
You are {role}. You answer generally known facts about {subject}.

For each given key (a combination of column names and values), answer "{attribute}".

- Answer only from reliable general knowledge; do not fill gaps by guessing. For a key you do
  not know, set known to false and put an empty value in value
- {value_rule}
- confidence is 0.0 to 1.0. Give a low value to answers you are unsure of
- reason is one sentence of grounds: say what you know that led to the answer
- Return answers in the same order and count as the keys, and copy each key string verbatim
  into key
"""

PROPOSAL_SYSTEM = """\
You are {role}. Looking at the list of columns in a {subject} dataset, you propose
**attributes that are not in the table but can be looked up from general knowledge using the
values of some column as the key**. Each proposed attribute will later be asked of you per key
value, turned into a column, added to a prediction model, and scored on whether it helped.

Rules:

- keys must be column names from the list. Do not use identifier, free-text, or date columns
  as keys. Choose columns whose value identifies the subject (product name, place name,
  variety, and so on)
- attribute must be a property of the subject itself. **Never propose the target variable
  itself, an estimate of it, or anything with the same meaning (market value, rating, outcome)**
- Choose things that are widely known as general knowledge and can be answered reliably from
  the key value alone. Do not propose anything that depends on dataset-specific circumstances
  (this seller, this day's stock)
- type is one of category / binary / numeric. For category and binary, write the possible
  values in values as a closed list (2 to 15 short alphanumeric strings). For numeric, write
  unit and a plausible range (range_low, range_high)
- name is a short alphanumeric string to use as the column name (not clashing with existing
  column names)
- why is one sentence on why it is likely to help predict the target variable
- At most {max_columns} items, ordered from most to least likely to help. An empty list is fine
"""


def _norm(s: Any) -> str:
    """Normalize a key value: strip surrounding whitespace, lowercase, collapse whitespace."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    return re.sub(r"\s+", " ", str(s).strip()).lower()


def _as_float(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v) if np.isfinite(v) else None
    try:
        return float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _role_subject(domain: Domain | None) -> tuple[str, str]:
    if domain is not None:
        return domain.role, domain.subject
    return "an investigator who answers generally known facts about the subject", "the subject"


# =====================================================================
# Lookup of one attribute (internal)
# =====================================================================


class _Lookup:
    """Key column values -> one attribute. `KnowledgeEncoder` holds one per column."""

    def __init__(self, keys: list[str], attribute: str, type: str, values: list[str] | None,
                 unit: str, range: tuple[float, float] | None, name: str,
                 threshold: float, min_count: int, batch_size: int,
                 check_against: str | None, domain: Domain | None,
                 cache_dir: Path | None, why: str = "", source: str = "human"):
        if type not in SUPPORTED_TYPES:
            raise MekikiError(f"type={type!r} is not supported. Supported: {SUPPORTED_TYPES}.")
        if type in ("category", "binary") and not values:
            raise MekikiError(
                f"type={type!r} requires values (the possible values). "
                "Asking the LLM without fixed candidates lets arbitrary spellings in, "
                "so the set must be closed.")
        if type == "binary" and len(values or []) != 2:
            raise MekikiError("type='binary' requires exactly 2 values.")
        if type == "numeric" and values:
            raise MekikiError("type='numeric' cannot take values. Use range to bound the answer.")
        if not keys:
            raise MekikiError("Pass at least one column in keys.")
        if not attribute or not str(attribute).strip():
            raise MekikiError("Write attribute (what you want to know).")
        self.keys = list(keys)
        self.attribute = str(attribute).strip()
        self.type = type
        self.values = [str(v) for v in values] if values else None
        self.unit = unit
        self.range = tuple(range) if range is not None else None
        self.name = name
        self.threshold = threshold
        self.min_count = min_count
        self.batch_size = batch_size
        self.check_against = check_against
        self.domain = domain
        self.cache_dir = cache_dir
        self.why = why
        self.source = source            # human (specified by a person) / llm (proposed by the LLM)
        #: key value -> answer. {"value", "confidence", "source", "reason", "cost"}
        self.answers: dict[str, dict[str, Any]] = {}
        #: Jev cost of keys Jev could not settle, added to the LLM answer's cost afterwards
        self._carried_cost: dict[str, float] = {}
        self._loaded = False

    # --- keys ----------------------------------------------------------

    def key_series(self, X: pd.DataFrame) -> pd.Series:
        """Key value per row. Empty string (not asked) if any key column is missing."""
        missing = [c for c in self.keys if c not in X.columns]
        if missing:
            raise MekikiError(f"Columns given in keys do not exist: {missing}")
        parts = [X[c].map(_norm) for c in self.keys]
        empty = parts[0] == ""
        for p in parts[1:]:
            empty |= p == ""
        joined = parts[0].astype(str)
        for p in parts[1:]:
            joined = joined + KEY_SEP + p.astype(str)
        return joined.where(~empty, "").reset_index(drop=True)

    # --- persistence ---------------------------------------------------

    def fingerprint(self) -> str:
        """Fingerprint of "what is asked, and how". A new attribute or candidates means a new cache."""
        role, subject = _role_subject(self.domain)
        payload = {"attribute": self.attribute, "type": self.type, "values": self.values,
                   "unit": self.unit, "range": list(self.range) if self.range else None,
                   "role": role, "subject": subject, "keys": self.keys}
        return hashlib.sha1(json.dumps(payload, sort_keys=True,
                                       ensure_ascii=False).encode("utf-8")).hexdigest()[:12]

    def _cache_path(self) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"mekiki_knowledge_{self.fingerprint()}.parquet"

    def load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        path = self._cache_path()
        if path is None or not path.exists():
            return
        try:
            df = pd.read_parquet(path)
        except Exception:      # a broken cache must not abort a measurement (same policy as llm.py)
            return
        for r in df.to_dict("records"):
            self.answers.setdefault(r["key"], {
                "value": None if pd.isna(r["value"]) else r["value"],
                "confidence": float(r["confidence"]),
                "source": r["source"], "reason": r["reason"], "cost": 0.0})

    def save(self) -> Path | None:
        path = self._cache_path()
        if path is None:
            return None
        rows = [{"key": k, "value": None if v["value"] is None else str(v["value"]),
                 "confidence": v["confidence"], "source": v["source"], "reason": v["reason"]}
                for k, v in self.answers.items() if v["source"] != "error"]
        if not rows:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(path, index=False)
        return path

    # --- prompts -------------------------------------------------------

    def system(self) -> str:
        role, subject = _role_subject(self.domain)
        if self.type == "numeric":
            unit = f"The unit is {self.unit}. " if self.unit else ""
            rng = (f"The plausible range is {self.range[0]:g} to {self.range[1]:g}."
                   if self.range else "")
            rule = f"Answer value as a number. {unit}{rng}".strip()
        else:
            rule = ("value must be a string that exactly matches one of the candidates. "
                    "Do not invent values outside the candidates")
        return LOOKUP_SYSTEM.format(role=role, subject=subject,
                                    attribute=self.attribute, value_rule=rule)

    def user(self, batch: list[str]) -> str:
        lines = [f"Key columns: {', '.join(self.keys)}"]
        if self.values:
            lines.append("Candidates: " + json.dumps(self.values, ensure_ascii=False))
        lines.append(f"Attribute to answer: {self.attribute}")
        lines.append("")
        lines += [f"{i}. {k}" for i, k in enumerate(batch, start=1)]
        return "\n".join(lines)

    def schema(self) -> dict:
        value = {"type": "number"} if self.type == "numeric" else {"type": "string"}
        return {
            "type": "object",
            "properties": {
                "answers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "key": {"type": "string"},
                            "known": {"type": "boolean"},
                            "value": value,
                            "confidence": {"type": "number"},
                            "reason": {"type": "string"},
                        },
                        "required": ["key", "known", "value", "confidence", "reason"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["answers"],
            "additionalProperties": False,
        }

    # --- answer validation ---------------------------------------------

    def validate(self, a: dict[str, Any]) -> dict[str, Any]:
        """Turn one LLM answer into provenance form. Out-of-candidates / out-of-range answers are dropped.

        The confidence threshold is not applied here. It is applied at transform time, so
        changing it does not trigger re-asking (answers stay in the cache).
        """
        conf = _as_float(a.get("confidence"))
        conf = min(max(conf, 0.0), 1.0) if conf is not None else 0.0
        reason = str(a.get("reason", "")).strip()
        if not a.get("known", False):
            return {"value": None, "confidence": conf, "source": "unknown",
                    "reason": reason or "LLM answered that it does not know"}
        raw = a.get("value")
        if self.type == "numeric":
            v = _as_float(raw)
            if v is None:
                return {"value": None, "confidence": conf, "source": "rejected",
                        "reason": f"non-numeric answer: {raw!r}"}
            if self.range and not (self.range[0] <= v <= self.range[1]):
                return {"value": None, "confidence": conf, "source": "rejected",
                        "reason": (f"answer {v:g} is out of range "
                                   f"({self.range[0]:g} to {self.range[1]:g}): {reason}")}
            value: Any = v
        else:
            s = str(raw)
            assert self.values is not None
            if s in self.values:
                value = s
            else:
                by_norm = {_norm(v): v for v in self.values}
                if _norm(s) in by_norm:
                    value = by_norm[_norm(s)]
                else:
                    return {"value": None, "confidence": conf, "source": "rejected",
                            "reason": f"answer {s!r} is not among the candidates: {reason}"}
        return {"value": value, "confidence": conf, "source": "llm", "reason": reason}

    def ask(self, client: ClaudeClient, unknown: list[str]) -> None:
        """Batch the unknown key values, ask the LLM, and store the answers in `self.answers`."""
        batches = [unknown[i:i + self.batch_size]
                   for i in range(0, len(unknown), self.batch_size)]
        users = [self.user(b) for b in batches]
        answers = client.ask_many(self.system(), users, self.schema())
        for batch, ans in zip(batches, answers, strict=True):
            per_key_cost = ans.cost / len(batch)
            if not ans.ok:
                # A failed call itself is not saved (it will be asked again next time)
                for k in batch:
                    self.answers[k] = {"value": None, "confidence": 0.0, "source": "error",
                                       "reason": f"LLM call failed: {ans.error}",
                                       "cost": per_key_cost}
                continue
            got = {}
            for a in ans.data.get("answers", []) or []:
                if isinstance(a, dict):
                    got.setdefault(_norm(a.get("key", "")), a)
            for k in batch:
                a = got.get(_norm(k))
                if a is None:
                    rec = {"value": None, "confidence": 0.0, "source": "rejected",
                           "reason": "key missing from the answer (key not copied verbatim)"}
                else:
                    rec = self.validate(a)
                rec["cost"] = per_key_cost + self._carried_cost.pop(k, 0.0)
                self.answers[k] = rec

    # --- the Jev tier -----------------------------------------------------

    @property
    def jev_eligible(self) -> bool:
        """Jev only picks from a closed set, so numeric lookups skip it."""
        return self.type != "numeric" and bool(self.values)

    def jev_state(self, key: str) -> dict[str, str]:
        """The key as a JSON object of column -> value (what Jev sees)."""
        parts = key.split(KEY_SEP, maxsplit=len(self.keys) - 1)
        return dict(zip(self.keys, parts, strict=False))

    def jev_instructions(self) -> str:
        role, subject = _role_subject(self.domain)
        return (f"You are {role}. For the {subject} identified by this key, what is its "
                f"{self.attribute}? Pick '{self._jev_unknown()}' when it is not generally known "
                "or cannot be determined from the key alone.")

    def _jev_unknown(self) -> str:
        assert self.values is not None
        return JEV_UNKNOWN if JEV_UNKNOWN not in self.values else f"({JEV_UNKNOWN})"

    def ask_jev(self, jev: JevClient, unknown: list[str]) -> list[str]:
        """Put the unknown key values to Jev. Keys it settles (a candidate, with confidence
        at or above the threshold) are stored with source "jev"; the rest are returned for
        the LLM batch. The Jev cost of those is carried over to their LLM answer."""
        assert self.values is not None
        unk = self._jev_unknown()
        options = [*self.values, unk]
        desc = {unk: "not generally known, or not determined by the key alone"}
        results = jev.choose_many([self.jev_state(k) for k in unknown], options,
                                  self.jev_instructions(), desc)
        remaining = []
        for k, r in zip(unknown, results, strict=True):
            if r.choice is None or r.choice == unk or r.confidence < self.threshold:
                remaining.append(k)
                if r.cost:
                    self._carried_cost[k] = self._carried_cost.get(k, 0.0) + r.cost
                continue
            p = r.probabilities.get(r.choice, r.confidence)
            self.answers[k] = {"value": r.choice, "confidence": r.confidence, "source": "jev",
                               "reason": f"Jev chose {r.choice!r} with probability {p:.2f}",
                               "cost": r.cost}
        return remaining

    # --- planning and resolution ---------------------------------------

    def plan(self, counts: pd.Series) -> dict[str, Any]:
        """For this set of key values, how many will be newly asked."""
        eligible = counts[counts >= self.min_count]
        # Key values whose call failed (source error) are counted as unknown again and re-asked
        unknown = sorted(k for k in eligible.index
                         if k not in self.answers or self.answers[k]["source"] == "error")
        skipped = int((counts < self.min_count).sum())
        known_jev = sum(1 for k in eligible.index
                        if k in self.answers and self.answers[k]["source"] == "jev")
        return {"n_key_values": int(len(counts)), "skipped_below_min_count": skipped,
                "known": int(len(eligible) - len(unknown)), "known_jev": int(known_jev),
                "n_to_ask": len(unknown), "unknown_keys": unknown}

    def resolve(self, X: pd.DataFrame, client: ClaudeClient | None,
                jev: JevClient | None = None) -> dict[str, Any]:
        """Attach a value to each row of X, asking Jev and then the LLM if needed.

        Returns the column, the provenance, the per-key-value table and the cross-check.
        """
        k = self.key_series(X)
        counts = k[k != ""].value_counts()
        self.load()
        plan = self.plan(counts)
        if plan["unknown_keys"]:
            unknown = plan["unknown_keys"]
            if jev is not None and self.jev_eligible and jev.available():
                unknown = self.ask_jev(jev, unknown)
            if unknown:
                if client is None or not client.available():
                    reason = (client.why_unavailable() if client is not None
                              else "No client was given.")
                    raise MekikiError(
                        f"Knowledge columns need an LLM. {reason} "
                        "For a cost estimate only, call fit and then cost().")
                self.ask(client, unknown)
            self.save()

        # Per-key-value table (only key values that appear in this data)
        rows = []
        for key, n in counts.items():
            if n < self.min_count:
                rows.append({"key": key, "value": None, "confidence": 0.0, "source": "skipped",
                             "reason": f"seen {n} time(s), below min_count={self.min_count}",
                             "cost": 0.0, "count": int(n)})
                continue
            a = dict(self.answers[key])
            if a["source"] in ANSWERED and a["confidence"] < self.threshold:
                a["value"] = None
                a["source"] = "rejected"
                a["reason"] = (f"confidence {a['confidence']:.2f} is below the threshold "
                               f"{self.threshold}: {a['reason']}")
            rows.append({"key": key,
                         **{c: a[c] for c in ("value", "confidence", "source", "reason", "cost")},
                         "count": int(n)})
        answers_ = pd.DataFrame(rows, columns=["key", "value", "confidence", "source", "reason",
                                               "cost", "count"])
        by_key = answers_.set_index("key") if len(answers_) else None

        # Per-row provenance
        n_rows = len(k)
        values: list[Any] = [None] * n_rows
        confs = np.zeros(n_rows)
        origins = ["skipped"] * n_rows
        reasons = ["key is missing"] * n_rows
        costs = np.zeros(n_rows)
        for i, key in enumerate(k.to_numpy()):
            if key == "" or by_key is None:
                continue
            r = by_key.loc[key]
            values[i] = r["value"]
            confs[i] = float(r["confidence"])
            origins[i] = str(r["source"])
            reasons[i] = str(r["reason"])
            # Prorated over rows so that the sum adds back up to the actual cost
            costs[i] = float(r["cost"]) / int(r["count"])
        prov = pd.DataFrame({"value": values, "confidence": confs, "source": origins,
                             "cost": costs, "key": k.to_numpy(), "reason": reasons})
        prov.loc[~prov["source"].isin(ANSWERED), "value"] = None

        out = pd.Series(prov["value"].to_numpy(), index=X.index, name=self.name, dtype="object")
        if self.type == "numeric":
            out = pd.to_numeric(out, errors="coerce").astype("float64")
        elif self.type == "category":
            out = out.astype(pd.CategoricalDtype(categories=self.values))
        return {"column": out, "provenance": prov, "answers": answers_,
                "agreement": self._agreement(X, prov)}

    def _agreement(self, X: pd.DataFrame, prov: pd.DataFrame) -> dict[str, Any] | None:
        """Cross-check against the `check_against` column on the rows the LLM answered."""
        if self.check_against is None:
            return None
        truth = X[self.check_against].reset_index(drop=True)
        mask = truth.notna() & prov["source"].isin(ANSWERED).to_numpy()
        if self.type != "numeric":
            mask &= truth.map(_norm) != ""
        n = int(mask.sum())
        if n == 0:
            return {"column": self.check_against, "n_compared": 0, "agreement": None}
        got = prov.loc[mask, "value"]
        want = truth[mask]
        if self.type == "numeric":
            g = pd.to_numeric(got, errors="coerce").to_numpy(dtype=float)
            w = pd.to_numeric(want, errors="coerce").to_numpy(dtype=float)
            ok = np.abs(g - w) <= 0.2 * np.abs(w)
            note = "relative error within 20%"
        else:
            ok = got.map(_norm).to_numpy() == want.map(_norm).to_numpy()
            note = "normalized string match"
        return {"column": self.check_against, "n_compared": n,
                "agreement": round(float(np.mean(ok)), 4), "criterion": note}

    def describe(self) -> dict[str, Any]:
        d = {"column": self.name, "key": self.keys, "attribute": self.attribute, "type": self.type,
             "specified": "human" if self.source == "human" else "llm"}
        if self.values:
            d["values"] = self.values
        if self.unit:
            d["unit"] = self.unit
        if self.range:
            d["range"] = list(self.range)
        if self.why:
            d["reason"] = self.why
        return d


# =====================================================================
# Public class
# =====================================================================


class KnowledgeEncoder:
    """Adds columns the table lacks, from the LLM's world knowledge.

    Put a table (DataFrame) in and get back the columns it lacked (a DataFrame with zero to a
    few columns). Add them to the original table with `df.join(...)`. What to look up may be
    specified, or left unspecified.

    Parameters
    ----------
    keys : str | list[str] | None
        Which columns to use as the key. If omitted, the LLM picks from the column list.
    attribute : str | None
        What to find out (natural language). If omitted, the LLM proposes up to
        `max_columns` attributes.
    type, values, unit, range :
        Answer type ("category" / "binary" / "numeric"), possible values, unit, plausible range.
        category / binary require values. If omitted, the LLM decides.
    target : str | None
        Name of the target column. When given, a tree model with each column and one without
        are compared on the same folds, and **only columns that helped are returned**
        (scoring does not call the LLM). Without it, every column is returned.
        In automatic mode it also tells the LLM "do not propose this".
    task : str | None
        "regression" / "classification". If omitted, inferred from the target values.
    domain : Domain | None
        Role and subject. Decides the "You are ..." of the prompt. None gives wording that does
        not commit to any domain.
    client : ClaudeClient | None
        If omitted, one is created with max_tokens matched to `batch_size`. When passing your
        own, set max_tokens to at least `batch_size * 80` (the default 1024 cuts answers off).
    jev : JevClient | bool | None
        The middle tier. `True` creates a `JevClient` (key from the settings file); a
        `JevClient` is used as is. For category / binary lookups each unknown key value is
        put to Jev first, and only the keys it cannot settle (below `threshold`, or
        "unknown") go to the LLM. Numeric lookups always go to the LLM (Jev has no numeric
        output). Omitted, or without a key: no Jev tier.
    threshold : float | None
        Answers with confidence below this are rejected. Defaults: category / binary 0.7,
        numeric 0.4 (for numeric "approximate" values the LLM tends to give 0.45-0.65
        in our measurements). The check runs on every transform, so changing it does not re-ask
        the LLM.
    min_count : int
        Key values seen fewer times than this are not asked. Avoids spending on the long tail of
        free-form keys (e.g. 60% appearing only once).
    batch_size : int
        Number of key values packed into one request. Larger is cheaper but answers break more
        easily.
    check_against : str | None
        If the data has a column with the same attribute, its name. The agreement rate is
        computed on non-missing rows (only when a single column is specified).
    max_columns : int
        Upper limit on the number of columns proposed in automatic mode.
    sample, n_splits :
        Row cap and number of folds for scoring (with `target`).
    cache_dir : str | Path | None
        Where key value -> answer is persisted. "default" means
        `sampledata/processed/knowledge_cache/` inside the repository; None disables
        persistence (the `ClaudeClient` cache still applies separately).
    name : str | None
        Output column name when a single column is specified. Default `<keys>_knowledge`.
    """

    def __init__(self, keys: str | list[str] | None = None, attribute: str | None = None, *,
                 type: str | None = None, values: list[str] | None = None,
                 unit: str = "", range: tuple[float, float] | None = None,
                 target: str | None = None, task: str | None = None,
                 domain: Domain | None = None, client: ClaudeClient | None = None,
                 jev: JevClient | bool | None = None,
                 threshold: float | None = None, min_count: int = 1, batch_size: int = 20,
                 check_against: str | None = None, max_columns: int = 5,
                 sample: int | None = 5_000, n_splits: int = 3,
                 cache_dir: str | Path | None = "default", name: str | None = None):
        if type is not None and type not in SUPPORTED_TYPES:
            raise MekikiError(f"type={type!r} is not supported. Supported: {SUPPORTED_TYPES}.")
        if type == "numeric" and values:
            raise MekikiError("type='numeric' cannot take values. Use range to bound the answer.")
        if type == "binary" and values is not None and len(values) != 2:
            raise MekikiError("type='binary' requires exactly 2 values.")
        if batch_size < 1:
            raise MekikiError("batch_size must be at least 1.")
        if min_count < 1:
            raise MekikiError("min_count must be at least 1.")
        if max_columns < 1:
            raise MekikiError("max_columns must be at least 1.")
        self.keys = [keys] if isinstance(keys, str) else (list(keys) if keys else None)
        self.attribute = str(attribute).strip() if attribute else None
        self.type = type
        self.values = [str(v) for v in values] if values else None
        self.unit = unit
        self.range = tuple(range) if range is not None else None
        self.target = target
        self.task = task
        self.domain = domain
        self.client = client
        if jev is True:
            self.jev: JevClient | None = JevClient()
        elif jev is False or jev is None:
            self.jev = None
        else:
            self.jev = jev
        self.threshold = threshold
        self.min_count = min_count
        self.batch_size = batch_size
        self.check_against = check_against
        self.max_columns = max_columns
        self.sample = sample
        self.n_splits = n_splits
        if cache_dir == "default":
            cache_dir = _cache_dir("knowledge_cache")
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.name = name

    # --- mode ------------------------------------------------------------

    @property
    def manual(self) -> bool:
        """Whether a person has decided everything: what, keyed by what, and of which type."""
        if not (self.keys and self.attribute and self.type):
            return False
        return self.type == "numeric" or bool(self.values)

    def _client(self) -> ClaudeClient:
        if self.client is None:
            self.client = ClaudeClient(max_tokens=max(_max_tokens_for(self.batch_size),
                                                      PROPOSAL_MAX_TOKENS))
        return self.client

    def _threshold_for(self, type: str) -> float:
        return self.threshold if self.threshold is not None else DEFAULT_THRESHOLD[type]

    def _jev_for(self, lk: _Lookup) -> JevClient | None:
        """The Jev client to use for one lookup: only for closed candidate sets, and only
        when it can be called (no key means the LLM answers everything, as before)."""
        if self.jev is None or not lk.jev_eligible or not self.jev.available():
            return None
        return self.jev

    # --- proposal (automatic mode) ---------------------------------------

    def _profiles(self, X: pd.DataFrame) -> list:
        from mekiki.diagnose import profile_column
        return [profile_column(X[c], len(X)) for c in X.columns if c != self.target]

    def _proposal_user(self, profiles: list) -> str:
        lines = []
        if self.target:
            lines.append(f"Target variable: {self.target} (do not propose it or an estimate of it)")
        fixed = []
        if self.keys:
            fixed.append(f"keys fixed to {self.keys}")
        if self.attribute:
            fixed.append(f'attribute fixed to "{self.attribute}" (propose only this one)')
        if self.type:
            fixed.append(f"type fixed to {self.type}")
        if self.values:
            fixed.append(f"values fixed to {self.values}")
        if self.unit:
            fixed.append(f"unit is {self.unit}")
        if self.range:
            fixed.append(f"range is {list(self.range)}")
        if fixed:
            lines.append("Fixed conditions: " + "; ".join(fixed))
        lines.append("")
        lines.append("Columns (name / kind / number of distinct values / examples):")
        usable = {"numeric", "boolean", "categorical", "text"}
        for p in profiles:
            mark = "" if p.kind in usable else " (not usable as a key)"
            ex = ", ".join(p.samples[:3])
            lines.append(f"- {p.name} / {p.kind}{mark} / {p.n_unique} / {ex}")
        return "\n".join(lines)

    @staticmethod
    def _proposal_schema() -> dict:
        str_list = {"type": "array", "items": {"type": "string"}}
        return {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "keys": str_list,
                            "attribute": {"type": "string"},
                            "type": {"type": "string", "enum": list(SUPPORTED_TYPES)},
                            "values": str_list,
                            "unit": {"type": "string"},
                            "range_low": {"type": "number"},
                            "range_high": {"type": "number"},
                            "why": {"type": "string"},
                        },
                        "required": ["name", "keys", "attribute", "type", "values", "unit",
                                     "range_low", "range_high", "why"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["candidates"],
            "additionalProperties": False,
        }

    def _propose(self, X: pd.DataFrame, profiles: list) -> list[_Lookup]:
        """Have the LLM propose candidates, validate them and turn them into `_Lookup`.

        Dropped candidates are recorded in `warnings_`.
        """
        client = self._client()
        if not client.available():
            raise MekikiError(
                "When keys / attribute / values are omitted the LLM decides them, "
                f"but it cannot be called. {client.why_unavailable()} "
                "Or specify keys / attribute / values yourself.")
        role, subject = _role_subject(self.domain)
        max_columns = 1 if self.attribute else self.max_columns
        system = PROPOSAL_SYSTEM.format(role=role, subject=subject, max_columns=max_columns)
        ans: LLMAnswer = client.ask(system, self._proposal_user(profiles), self._proposal_schema())
        self.proposal_cost_ = ans.cost
        if not ans.ok:
            raise MekikiError(f"The proposal LLM call failed: {ans.error}")
        self.proposal_raw_ = ans.data

        usable = {p.name for p in profiles if p.kind in ("numeric", "boolean", "categorical", "text")}
        existing = set(map(str, X.columns))
        lookups: list[_Lookup] = []
        for c in ans.data.get("candidates", []) or []:
            if not isinstance(c, dict):
                continue
            try:
                keys = self.keys or [str(k) for k in c.get("keys", [])]
                bad = [k for k in keys if k not in usable]
                if bad:
                    raise MekikiError(f"columns that cannot be keys: {bad}")
                type = self.type or str(c.get("type", "category"))
                values = self.values
                if values is None and type != "numeric":
                    values = [str(v) for v in c.get("values", []) if str(v).strip()]
                    if len(values) < 2:
                        raise MekikiError("fewer than 2 candidate values")
                    if type == "binary" and len(values) != 2:
                        type = "category"
                rng = self.range
                if rng is None and type == "numeric":
                    lo, hi = _as_float(c.get("range_low")), _as_float(c.get("range_high"))
                    rng = (lo, hi) if lo is not None and hi is not None and lo < hi else None
                name = self.name or re.sub(r"\W+", "_", str(c.get("name", "")).strip()).strip("_")
                if not name:
                    name = f"{'_'.join(keys)}_knowledge"
                base, n = name, 2
                while name in existing:
                    name = f"{base}_{n}"
                    n += 1
                existing.add(name)
                lookups.append(_Lookup(
                    keys=keys, attribute=self.attribute or str(c.get("attribute", "")),
                    type=type, values=values, unit=self.unit or str(c.get("unit", "") or ""),
                    range=rng, name=name, threshold=self._threshold_for(type),
                    min_count=self.min_count, batch_size=self.batch_size,
                    check_against=self.check_against if self.attribute else None,
                    domain=self.domain, cache_dir=self.cache_dir,
                    why=str(c.get("why", "")), source="llm"))
            except MekikiError as e:
                self.warnings_.append(
                    f"Dropped candidate {c.get('name')!r} ({c.get('attribute')!r}): {e}")
            if len(lookups) >= max_columns:
                break
        if not lookups:
            self.warnings_.append("The LLM returned no usable candidates")
        return lookups

    # --- fit / transform -------------------------------------------------

    def fit(self, X: pd.DataFrame, y=None) -> KnowledgeEncoder:
        """Decide what to look up and count the occurrences of key values.

        If everything is specified, the LLM is not called. If something is omitted, the LLM is
        asked once for candidates (`proposal_cost_`). The LLM calls that fill the values happen
        in transform.

        Parameters
        ----------
        X:
            A table containing the key columns (and `target` / `check_against` if declared).
        y:
            Target values, as an alternative to a `target` column in X.

        Returns
        -------
        `self`. `cost(X)` then gives the estimate before anything is spent.
        """
        if self.check_against is not None and self.check_against not in X.columns:
            raise MekikiError(
                f"Column given in check_against does not exist: {self.check_against!r}")
        if self.target is not None and self.target not in X.columns and y is None:
            raise MekikiError(f"Column given in target does not exist: {self.target!r}")
        self.warnings_: list[str] = []
        self.proposal_cost_ = 0.0
        if self.manual:
            assert self.keys and self.attribute and self.type
            self.lookups_ = [_Lookup(
                keys=self.keys, attribute=self.attribute, type=self.type, values=self.values,
                unit=self.unit, range=self.range,
                name=self.name or f"{'_'.join(self.keys)}_knowledge",
                threshold=self._threshold_for(self.type), min_count=self.min_count,
                batch_size=self.batch_size, check_against=self.check_against,
                domain=self.domain, cache_dir=self.cache_dir, source="human")]
        else:
            self.lookups_ = self._propose(X, self._profiles(X))
        self.counts_ = {}
        for lk in self.lookups_:
            k = lk.key_series(X)
            self.counts_[lk.name] = k[k != ""].value_counts()
            lk.load()
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Look up the key values of X and return the new columns. **X itself is not modified.**

        **This is the step that calls the LLM**, once per distinct key value not yet in the
        disk cache. With everything cached it is free and needs no API key.

        Parameters
        ----------
        X:
            A table containing the key columns.
        y:
            Target values, as an alternative to a `target` column in X. Only used for scoring.

        Returns
        -------
        A `pd.DataFrame` holding only the new columns, carrying the index of X, to join onto
        the table (`df.join(kc.transform(df))`). It is a DataFrame even for a single column.
        Rows where the LLM declined, or whose answer was rejected, are missing. When a target
        is given, each column is scored and only the ones that helped are returned (`scores_`
        has them all).

        The per-row record is kept in `provenance_` and replaced on every call.
        """
        if not hasattr(self, "lookups_"):
            raise MekikiError("Call fit first.")
        cols, provs, answers, agreements = {}, [], {}, {}
        for lk in self.lookups_:
            lk.load()
            need = lk.plan(self._counts_for(lk, X))["n_to_ask"] > 0
            # No client is created unless something must be asked (works from the cache alone
            # without an API key)
            r = lk.resolve(X, self._client() if need else None, self._jev_for(lk))
            cols[lk.name] = r["column"]
            prov = r["provenance"].copy()
            prov.insert(0, "column", lk.name)
            provs.append(prov)
            answers[lk.name] = r["answers"]
            if r["agreement"] is not None:
                agreements[lk.name] = r["agreement"]
        self.provenance_ = (pd.concat(provs, ignore_index=True) if provs
                            else pd.DataFrame(columns=["column", "value", "confidence", "source",
                                                       "cost", "key", "reason"]))
        self.answers_ = answers
        self.agreement_ = agreements
        out = pd.DataFrame(cols, index=X.index)

        target = self._target_values(X, y)
        if target is not None and len(out.columns):
            self.scores_ = self._score(X, out, target)
            keep = [c for c in out.columns if bool(self.scores_.loc[c, "accepted"])]
            out = out[keep]
        else:
            self.scores_ = None
        return out

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """`fit(X, y)` followed by `transform(X, y)`. Returns what `transform` returns."""
        return self.fit(X, y).transform(X, y)

    def _counts_for(self, lk: _Lookup, X: pd.DataFrame) -> pd.Series:
        k = lk.key_series(X)
        return k[k != ""].value_counts()

    def _target_values(self, X: pd.DataFrame, y) -> pd.Series | None:
        if y is not None:
            return pd.Series(np.asarray(y), index=X.index)
        if self.target is not None and self.target in X.columns:
            return X[self.target]
        return None

    # --- scoring (does not call the LLM) ---------------------------------

    def _score(self, X: pd.DataFrame, new: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """Compare each column as "tree with it vs tree without it" on the same folds."""
        from sklearn.metrics import log_loss
        from sklearn.model_selection import KFold, StratifiedKFold

        from mekiki.diagnose import profile_column, profile_target
        from mekiki.predictor import SEED, ColumnSpec, TreeModel

        df = X.copy()
        df["_target"] = target.to_numpy()
        for c in new.columns:
            df[c] = new[c].to_numpy()
        df = df[df["_target"].notna()].reset_index(drop=True)
        if self.sample is not None and self.sample < len(df):
            df = df.sample(n=self.sample, random_state=SEED).reset_index(drop=True)
        n_splits = min(self.n_splits, max(2, len(df) // 2))
        if len(df) < 2 * n_splits:
            raise MekikiError(f"Too few rows to score ({len(df)} rows).")

        task = self.task or profile_target(df["_target"]).task
        classify = task == "classification"
        # Base columns: numeric / boolean / categorical (and short text with few distinct
        # values), excluding the target and the new columns
        base_num, base_bool, base_cat = [], [], []
        for c in X.columns:
            if c == self.target or c in new.columns:
                continue
            p = profile_column(X[c], len(X))
            if p.kind == "numeric":
                base_num.append(c)
            elif p.kind == "boolean":
                base_bool.append(c)
            elif p.kind == "categorical" or (p.kind == "text" and p.n_unique <= 200):
                base_cat.append(c)
        base = ColumnSpec(numeric=base_num, boolean=base_bool, categorical=base_cat)
        for c in base_bool:
            df[c] = df[c].map(lambda v: 1.0 if str(v).strip().lower() in ("1", "true", "yes", "y", "t")
                              else (0.0 if str(v).strip().lower() in ("0", "false", "no", "n", "f")
                                    else np.nan))

        if classify:
            classes = np.unique(df["_target"].to_numpy())
            lookup = {c: i for i, c in enumerate(classes.tolist())}
            yv = np.array([lookup[v] for v in df["_target"].tolist()], dtype=int)
            splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED).split(df, yv))
        else:
            yv = df["_target"].to_numpy(dtype=float)
            splits = list(KFold(n_splits=n_splits, shuffle=True, random_state=SEED).split(df))

        def score(model, test, y_te) -> float:
            if classify:
                return float(log_loss(y_te, model.predict_proba(test), labels=list(range(len(classes)))))
            return float(np.mean(np.abs(model.predict(test) - y_te)))

        def cv(spec: ColumnSpec) -> list[float]:
            out = []
            for tr, te in splits:
                train, test = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
                m = TreeModel("scoring", spec, task=task).fit(train, yv[tr])
                out.append(score(m, test, yv[te]))
            return out

        without = cv(base)
        rows = []
        for lk in self.lookups_:
            c = lk.name
            if lk.type == "numeric":
                spec = ColumnSpec(numeric=[*base.numeric, c], boolean=base.boolean,
                                  categorical=base.categorical)
            else:
                df[c] = df[c].astype("object")
                spec = ColumnSpec(numeric=base.numeric, boolean=base.boolean,
                                  categorical=[*base.categorical, c])
            with_ = cv(spec)
            s0, s1 = float(np.mean(without)), float(np.mean(with_))
            contribution = (s0 - s1) / s0 if s0 > 0 else 0.0
            wins = sum(a > b for a, b in zip(without, with_, strict=True))
            adopt = contribution > SCORE_THRESHOLD and wins >= n_splits - 1
            rows.append({"column": c, "metric": "log loss" if classify else "MAE",
                         "without": round(s0, 4), "with": round(s1, 4),
                         "contribution": round(contribution, 4),
                         "fold_wins": f"{wins}/{n_splits}", "accepted": bool(adopt),
                         "n_rows": len(df)})
        return pd.DataFrame(rows).set_index("column")

    # --- inspection API (same names as SemanticEncoder) ----------------------

    def _prov(self) -> pd.DataFrame:
        if not hasattr(self, "provenance_"):
            raise MekikiError("Call transform first.")
        return self.provenance_

    def _column(self, column: str | None) -> str:
        if not hasattr(self, "lookups_") or not self.lookups_:
            raise MekikiError("Call fit first.")
        if column is None:
            return self.lookups_[0].name
        if column not in {lk.name for lk in self.lookups_}:
            raise MekikiError(f"No such column was created: {column!r}. See columns() for a list.")
        return column

    def columns(self) -> list[dict[str, Any]]:
        """The columns to create (or created): what is asked, keyed by what, and why the LLM proposed it."""
        if not hasattr(self, "lookups_"):
            raise MekikiError("Call fit first.")
        return [lk.describe() for lk in self.lookups_]

    def cost(self, X: pd.DataFrame | None = None) -> dict:
        """Cost estimate. Can be called after fit alone, **before filling with the LLM**.

        With X, the key values of that data are counted afresh (no transform).
        The unit cost is derived from approximate token counts and is meant to be replaced
        by measurements.
        """
        if not hasattr(self, "lookups_"):
            raise MekikiError("Call fit first (or specify everything, then call fit).")
        per_key = self.cost_per_key()
        cols = []
        total_ask, total_req, total_jev = 0, 0, 0
        for lk in self.lookups_:
            counts = self._counts_for(lk, X) if X is not None else self.counts_[lk.name]
            lk.load()
            plan = lk.plan(counts)
            n = plan["n_to_ask"]
            total_ask += n
            total_req += int(np.ceil(n / self.batch_size))
            col = {"column": lk.name, "n_rows": int(counts.sum()),
                   "n_key_values": plan["n_key_values"],
                   "skipped_below_min_count": plan["skipped_below_min_count"],
                   "known": plan["known"], "n_to_ask": n,
                   "estimated_total": round(n * per_key, 6)}
            if self.jev is not None:
                via_jev = self._jev_for(lk) is not None
                col["jev_first"] = via_jev
                col["known_jev"] = plan["known_jev"]
                total_jev += n if via_jev else 0
            cols.append(col)
        actual = float(self._prov()["cost"].sum()) if hasattr(self, "provenance_") else 0.0
        out = {"n_columns": len(self.lookups_), "n_to_ask": total_ask, "n_requests": total_req,
               "cost_per_key_usd": round(per_key, 6),
               "estimated_total": round(total_ask * per_key, 6),
               "proposal_cost_usd": round(float(getattr(self, "proposal_cost_", 0.0)), 6),
               "actual_cost_usd": round(actual, 6), "per_column": cols}
        if self.jev is not None:
            # The LLM estimate above is the ceiling; every key Jev settles costs this instead
            per_jev = self.jev.estimated_cost_per_call()
            out["jev_available"] = self.jev.available()
            out["n_to_ask_jev_first"] = total_jev
            out["jev_cost_per_key_usd"] = round(per_jev, 6)
            out["estimated_jev_total"] = round(total_jev * per_jev, 6)
        return out

    def cost_per_key(self) -> float:
        """Approximate USD per key value (from batch_size and the price table)."""
        if self.client is not None:
            pin, pout = self.client.unit_prices()
        else:
            pin, pout = PRICING[DEFAULT_MODEL]
        b = self.batch_size
        per_call = ((EST_CALL_OVERHEAD_TOKENS + EST_INPUT_TOKENS_PER_KEY * b) * pin
                    + EST_SYSTEM_TOKENS * pin * 0.1
                    + EST_OUTPUT_TOKENS_PER_KEY * b * pout) / 1_000_000
        return per_call / b

    def explain(self, i: int, column: str | None = None) -> str:
        """Human-readable provenance of one row. With several columns, `column` picks one (default: first)."""
        name = self._column(column)
        lk = next(lk for lk in self.lookups_ if lk.name == name)
        prov = self._prov()
        r = prov[prov["column"] == name].iloc[i]
        lines = [
            f"column            {name}",
            f"value             {r['value']}",
            f"confidence        {r['confidence']:.3f}",
            f"source            {r['source']}",
            f"strategy          knowledge_lookup ({', '.join(lk.keys)} -> {lk.attribute})",
            f"key               {r['key'] or '(missing)'}",
            f"reason            {r['reason']}",
            f"cost              ${r['cost']:.5f}",
        ]
        if lk.source == "llm":
            lines.append(f"proposed because  {lk.why}")
        return "\n".join(lines)

    def review_queue(self, column: str | None = None) -> pd.DataFrame:
        """Key values a person should look at: answered "unknown" or rejected. Most frequent first."""
        if not hasattr(self, "answers_"):
            raise MekikiError("Call transform first.")
        names = [self._column(column)] if column is not None else list(self.answers_)
        parts = []
        for n in names:
            q = self.answers_[n]
            q = q[q["source"].isin(["unknown", "rejected", "error"])].copy()
            q.insert(0, "column", n)
            parts.append(q)
        if not parts:
            return pd.DataFrame(columns=["column", "key", "value", "confidence", "source", "reason",
                                         "cost", "count"])
        out = pd.concat(parts, ignore_index=True).sort_values("count", ascending=False)
        return out.reset_index(drop=True)

    def status(self) -> dict:
        """Summary of the last `transform`: what was asked, what was accepted, what it cost.

        Returns
        -------
        A dict with `n_records`, `n_columns`, `returned_columns`, `proposal_cost_usd`,
        `actual_cost_usd`, `per_column` and, if there were any, `warnings`. `per_column` has one
        dict per column: its declaration, the number of distinct key values (`n_key_values`),
        how their answers ended up (`accepted_llm`, `accepted_jev`, `unknown`, `rejected`,
        `errors`, `skipped`),
        `n_rows_filled`, `threshold`, `actual_cost_usd`, and `agreement` / `score` when
        `check_against` / a target was given.
        """
        prov = self._prov()
        cols = []
        for lk in self.lookups_:
            ans = self.answers_[lk.name]
            p = prov[prov["column"] == lk.name]
            d = {
                **lk.describe(),
                "n_key_values": int(len(ans)),
                "accepted_llm": int((ans["source"] == "llm").sum()),
                "accepted_jev": int((ans["source"] == "jev").sum()),
                "unknown": int((ans["source"] == "unknown").sum()),
                "rejected": int((ans["source"] == "rejected").sum()),
                "errors": int((ans["source"] == "error").sum()),
                "skipped": int((ans["source"] == "skipped").sum()),
                "n_rows_filled": int(p["source"].isin(ANSWERED).sum()),
                "threshold": lk.threshold,
                "actual_cost_usd": round(float(p["cost"].sum()), 6),
            }
            if lk.name in self.agreement_:
                d["agreement"] = self.agreement_[lk.name]
            if self.scores_ is not None and lk.name in self.scores_.index:
                d["score"] = self.scores_.loc[lk.name].to_dict()
            cols.append(d)
        out = {"n_records": int(len(prov) / max(len(self.lookups_), 1)),
               "n_columns": len(self.lookups_),
               "returned_columns": ([c for c in self.scores_.index if bool(self.scores_.loc[c, "accepted"])]
                                    if self.scores_ is not None else [lk.name for lk in self.lookups_]),
               "proposal_cost_usd": round(float(getattr(self, "proposal_cost_", 0.0)), 6),
               "actual_cost_usd": round(float(prov["cost"].sum()), 6),
               "per_column": cols}
        if self.warnings_:
            out["warnings"] = list(self.warnings_)
        return out
