"""Client for Jev, the "System One" model of TypeSafe AI: the middle tier between the
statistical models and the frontier LLM.

Every LLM-using part of mekiki has three stages: the statistical or embedding model
answers first, Jev answers what it can, and only what is still unsure reaches the
frontier LLM (`mekiki.llm.LLMClient`). Jev sits in the middle because it is fast
(a few hundred milliseconds), cheap (input tokens only, no charge for output) and
**returns calibrated probabilities** rather than prose: its confidence can be compared
against a threshold to decide whether the frontier LLM is still needed.

Jev cannot write free text or numbers. It answers *typed questions* about a "state"
(a string, a JSON object or a list):

- `choice` picks one of a fixed set of options and returns `choice`, `confidence`
  and `probabilities` (one per option).
- `score` rates the state against ordered levels and returns `score` (the
  probability-weighted level index), `confidence`, `probabilities` and `legend`.
- `noul` is a yes/no question and returns `noul`, the probability of "yes".

Several questions can go in one request and are evaluated together.

**This module and `mekiki/llm.py` are the only places that emit HTTP.** The layers
above (`JevFallback`, `EvidencePredictor`, `KnowledgeEncoder`) only know "send a
state and questions, get answers back". The same three design points as the LLM
client apply: every call goes through the disk cache (keyed by model, state and
questions), every call is counted, and batches are sent in parallel.

The API key is `TYPESAFE_API_KEY` in `.env` or the environment (`TYPESAFE_BASE_URL`
and `TYPESAFE_DEFAULT_MODEL` are honoured too). Jev is also served by the Vercel AI
Gateway, which speaks the same request and response format: without a TypeSafe key,
`AI_GATEWAY_API_KEY` is used instead, and the client then talks to the gateway
(`https://ai-gateway.vercel.sh/typesafe`, model `typesafe-ai/jev`) and records the
cost the gateway reports for each call. Importing works without a key;
`available()` merely returns False and every caller degrades to the two-stage
behaviour (statistical model, then frontier LLM). The HTTP layer uses only the
standard library, so no dependency is added, and the transport is injectable
(`transport=`) so that the callers can be tested without a network.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from mekiki.errors import MekikiError
from mekiki.llm import load_api_key
from mekiki.paths import cache_dir

#: Cache location. Inside a clone of the repository this is `sampledata/processed/jev_cache/`;
#: when used as an installed package it becomes the per-user cache location
#: (`mekiki/paths.py`). Can be moved with `MEKIKI_CACHE_DIR`.
DEFAULT_CACHE_DIR = cache_dir("jev_cache")

#: Input / output dollar prices per million tokens. Output tokens are free of charge.
#: The prices are only used for cost estimates, so if they change fix them here only.
JEV_PRICING: dict[str, tuple[float, float]] = {
    "jev-latest":      (0.042, 0.0),
    "typesafe-ai/jev": (0.042, 0.0),    # the same model through the Vercel AI Gateway
}
DEFAULT_MODEL = "jev-latest"
DEFAULT_BASE_URL = "https://api.typesafe.ai"
SYSTEM_ONE_PATH = "/v1/systemone"

#: The Vercel AI Gateway's TypeSafe-compatible endpoint and its name for Jev.
GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/typesafe"
GATEWAY_MODEL = "typesafe-ai/jev"
GATEWAY_HOST = "ai-gateway.vercel.sh"

#: Names of the settings read from `.env` or the environment.
TYPESAFE_KEY = "TYPESAFE_API_KEY"
TYPESAFE_BASE_URL = "TYPESAFE_BASE_URL"
TYPESAFE_DEFAULT_MODEL = "TYPESAFE_DEFAULT_MODEL"
AI_GATEWAY_KEY = "AI_GATEWAY_API_KEY"

#: Upper bound on the options of one `choice` question (the API rejects more).
MAX_OPTIONS = 255

#: Response time of one call (seconds), used for time estimates. Calls run in
#: parallel, so the total is "calls / workers x this".
SECONDS_PER_CALL = 0.3

#: Input tokens of a typical request (one record, its evidence and a handful of
#: similar cases as a JSON state), used for cost estimates before any call is made.
EST_INPUT_TOKENS_PER_CALL = 600

#: HTTP statuses that are worth retrying (rate limit and server-side failures).
RETRY_STATUSES = frozenset({408, 429, *range(500, 600)})


@dataclass
class JevAnswer:
    """The answers to one request, and what it cost.

    `answers` is keyed by the question names of the request; each value is the raw
    answer object (`{"type": "choice", "choice": ..., "confidence": ..., "probabilities": ...}`
    and so on). A request with `from_cache` True was not billed this time (`cost` is 0.0).
    """

    answers: dict[str, dict[str, Any]]
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    from_cache: bool = False
    error: str | None = None
    model: str = ""

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class JevChoice:
    """One `choice` answer, flattened. Unpacks as `(choice, confidence, probabilities, cost)`.

    `choice` is None when the call failed (`error` says why) or the answer was not one
    of the options; `probabilities` is then empty.
    """

    choice: str | None
    confidence: float = 0.0
    probabilities: dict[str, float] = field(default_factory=dict)
    cost: float = 0.0
    from_cache: bool = False
    error: str | None = None

    def __iter__(self):
        yield self.choice
        yield self.confidence
        yield self.probabilities
        yield self.cost


@dataclass
class JevScore:
    """One `score` answer, flattened. Unpacks as `(score, confidence, probabilities, cost)`.

    `score` is the probability-weighted level index (a float between 0 and the number
    of levels minus one), None when the call failed.
    """

    score: float | None
    confidence: float = 0.0
    probabilities: dict[str, float] = field(default_factory=dict)
    legend: dict[str, Any] = field(default_factory=dict)
    cost: float = 0.0
    from_cache: bool = False
    error: str | None = None

    def __iter__(self):
        yield self.score
        yield self.confidence
        yield self.probabilities
        yield self.cost


@dataclass
class JevUsage:
    """Running totals over calls (the same shape as `mekiki.llm.Usage`, minus the
    prompt-cache column, which Jev does not have)."""

    calls: int = 0
    cache_hits: int = 0
    errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_calls": self.calls,
            "cache_hits": self.cache_hits,
            "errors": self.errors,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost, 6),
        }


class JevHTTPError(Exception):
    """A non-2xx response. `status` decides whether it is retried; `retry_after` is the
    server's wait in seconds when it sent one."""

    def __init__(self, status: int, message: str, retry_after: float | None = None) -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.retry_after = retry_after


# --- Question builders ----------------------------------------------------------


def choice_question(options: Sequence[str], instructions: str,
                    descriptions: dict[str, str] | None = None) -> dict[str, Any]:
    """Build one `choice` question. An option without a description is interpreted
    by its name alone."""
    opts = [str(o) for o in options]
    if not opts:
        raise MekikiError("A choice question needs at least one option.")
    if len(opts) > MAX_OPTIONS:
        raise MekikiError(
            f"A choice question takes at most {MAX_OPTIONS} options, got {len(opts)}. "
            "Fold the values into coarser groups first.")
    if len(set(opts)) != len(opts):
        raise MekikiError("The options of a choice question must be distinct.")
    descriptions = descriptions or {}
    return {"type": "choice", "instructions": instructions,
            "criteria": {o: descriptions.get(o) for o in opts}}


def score_question(levels: Sequence[str], instructions: str) -> dict[str, Any]:
    """Build one `score` question. `levels` are ordered from the lowest (index 0) up."""
    lv = [str(x) for x in levels]
    if not lv:
        raise MekikiError("A score question needs at least one level.")
    return {"type": "score", "instructions": instructions, "criteria": lv}


def noul_question(instructions: str, when_true: str | None = None,
                  when_false: str | None = None) -> dict[str, Any]:
    """Build one `noul` (yes/no) question."""
    q: dict[str, Any] = {"type": "noul", "instructions": instructions}
    if when_true is not None or when_false is not None:
        q["criteria"] = {"true": when_true, "false": when_false}
    return q


def _json_default(o: Any) -> Any:
    """Make numpy scalars and other odd values serialisable, so a state built from a
    DataFrame row does not break the cache key."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, (np.ndarray, set, frozenset)):
        return list(o)
    return str(o)


def _reported_cost(resp: dict) -> float | None:
    """The dollars the Vercel AI Gateway billed for this call
    (`provider_metadata.gateway.cost`, a decimal string). None when the response
    does not carry it, as TypeSafe's own API does not."""
    meta = resp.get("provider_metadata") or resp.get("providerMetadata")
    gateway = meta.get("gateway") if isinstance(meta, dict) else None
    if not isinstance(gateway, dict) or gateway.get("cost") is None:
        return None
    try:
        return max(float(gateway["cost"]), 0.0)
    except (TypeError, ValueError):
        return None


def _dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=_json_default)


class JevClient:
    """A thin client that asks Jev typed questions and returns calibrated probabilities.

    Parameters
    ----------
    model:
        Model name or alias. Default: `TYPESAFE_DEFAULT_MODEL` from the settings,
        else `"jev-latest"` (`"typesafe-ai/jev"` when going through the Vercel AI
        Gateway).
    cache_dir:
        Where answers are stored. `"default"` is the shared cache location; None
        disables the disk cache.
    api_key:
        Overrides `TYPESAFE_API_KEY`. Leave unset to read `.env` or the environment:
        `TYPESAFE_API_KEY` first, then `AI_GATEWAY_API_KEY` (the Vercel AI Gateway).
    base_url:
        Overrides `TYPESAFE_BASE_URL`. Default `https://api.typesafe.ai`, or
        `https://ai-gateway.vercel.sh/typesafe` when the key is `AI_GATEWAY_API_KEY`.
    max_workers:
        Parallelism of `ask_many`.
    timeout:
        Seconds per HTTP request.
    max_retries:
        Retries after the first attempt on 429 / 5xx / connection errors, with a
        short bounded backoff (the server's `retry-after` is honoured).
    transport:
        A function that takes the request body (a dict) and returns the decoded
        response body (a dict). When given, no HTTP is sent and no key is needed;
        this is how the callers are tested without a network.
    """

    def __init__(self, model: str | None = None,
                 cache_dir: str | Path | None = "default",
                 api_key: str | None = None, base_url: str | None = None,
                 max_workers: int = 8, timeout: float = 10.0, max_retries: int = 2,
                 transport: Callable[[dict], dict] | None = None) -> None:
        # A TypeSafe key wins; without one, a Vercel AI Gateway key switches the
        # default endpoint and model name to the gateway's.
        gateway_key = None
        if api_key is None:
            api_key = load_api_key(TYPESAFE_KEY)
            if not api_key:
                gateway_key = load_api_key(AI_GATEWAY_KEY)
        self.api_key = api_key or gateway_key
        default_url = GATEWAY_BASE_URL if gateway_key else DEFAULT_BASE_URL
        self.base_url = (base_url or load_api_key(TYPESAFE_BASE_URL)
                         or default_url).rstrip("/")
        default_model = GATEWAY_MODEL if self.via_gateway else DEFAULT_MODEL
        self.model = model or load_api_key(TYPESAFE_DEFAULT_MODEL) or default_model
        self.max_workers = max_workers
        self.timeout = timeout
        self.max_retries = max(int(max_retries), 0)
        self.transport = transport
        if cache_dir == "default":
            cache_dir = DEFAULT_CACHE_DIR
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.usage = JevUsage()
        self._lock = threading.Lock()

    # --- State --------------------------------------------------------

    @property
    def via_gateway(self) -> bool:
        """Whether requests go through the Vercel AI Gateway rather than TypeSafe."""
        return GATEWAY_HOST in self.base_url

    def available(self) -> bool:
        """Whether a call can actually be made right now (False without a key)."""
        return self.why_unavailable() is None

    def why_unavailable(self) -> str | None:
        """The reason a call would fail, as a sentence telling the user what to do.
        None when everything is in place (a key, or an injected transport)."""
        if self.transport is not None or self.api_key:
            return None
        return (f"Neither {TYPESAFE_KEY} nor {AI_GATEWAY_KEY} is set. Run "
                "`cp .env.example .env` and write a TypeSafe AI key, or a Vercel AI "
                "Gateway key, there (or export it as an environment variable). "
                "Without either the Jev tier is skipped.")

    # --- Cache --------------------------------------------------------

    def _key(self, state: Any, questions: dict[str, dict]) -> str:
        payload = _dumps({"model": self.model, "state": state, "questions": questions})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path | None:
        if self.cache_dir is None:
            return None
        # One level of subdirectories, to avoid tens of thousands of files in one directory
        d = self.cache_dir / key[:2]
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{key}.json"

    def _read_cache(self, key: str) -> dict | None:
        """Read the cache. Returns None on a corrupt entry so the call is retried
        (a broken file must not take down a long run; same policy as the LLM cache)."""
        path = self._cache_path(key)
        if path is None or not path.exists():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return None
        if not isinstance(record, dict) or not isinstance(record.get("answers"), dict):
            return None
        return record

    def _write_cache(self, key: str, record: dict) -> None:
        path = self._cache_path(key)
        if path is None:
            return
        tmp = path.with_suffix(".tmp")
        tmp.write_text(_dumps(record), encoding="utf-8")
        tmp.replace(path)  # never leaves a half-written JSON behind on a crash

    # --- Prices -------------------------------------------------------

    def unit_prices(self) -> tuple[float, float]:
        """Input / output dollars per million tokens for this model. Unknown model
        names fall back to the default model's prices (an estimate beats none)."""
        return JEV_PRICING.get(self.model, JEV_PRICING[DEFAULT_MODEL])

    def _price(self, input_tokens: int, output_tokens: int) -> float:
        pin, pout = self.unit_prices()
        return (input_tokens * pin + output_tokens * pout) / 1_000_000

    def estimated_cost_per_call(self, input_tokens: int = EST_INPUT_TOKENS_PER_CALL) -> float:
        """Dollars for one call: the measured average when calls have been billed,
        otherwise `input_tokens` at the list price."""
        u = self.usage
        paid = u.calls - u.cache_hits - u.errors
        if paid > 0 and u.cost > 0:
            return u.cost / paid
        return self._price(input_tokens, 0)

    # --- Transport ----------------------------------------------------

    def _post(self, body: dict) -> dict:
        """One HTTP request with the standard library. Raises `JevHTTPError` on a
        non-2xx status and `OSError` when the server cannot be reached."""
        reason = self.why_unavailable()
        if reason is not None:
            raise RuntimeError(reason)
        data = _dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + SYSTEM_ONE_PATH, data=data, method="POST",
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json",
                     "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = raw
            retry_after = None
            header = e.headers.get("retry-after") if e.headers is not None else None
            if header:
                try:
                    retry_after = max(float(header.strip()), 0.0)
                except ValueError:
                    retry_after = None
            raise JevHTTPError(e.code, _error_message(parsed), retry_after) from None

    def _send(self, body: dict) -> dict:
        """Send with bounded retries. 429 / 5xx and connection errors are retried;
        4xx other than 429 (bad key, bad request) are not, since repeating them
        cannot help."""
        send = self.transport if self.transport is not None else self._post
        attempt = 0
        while True:
            try:
                return send(body)
            except JevHTTPError as e:
                if e.status not in RETRY_STATUSES or attempt >= self.max_retries:
                    raise
                delay = e.retry_after if e.retry_after is not None else 0.5 * (2 ** attempt)
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt >= self.max_retries:
                    raise
                delay = 0.5 * (2 ** attempt)
            attempt += 1
            time.sleep(min(delay, 5.0))

    # --- Core ---------------------------------------------------------

    def ask(self, state: Any, questions: dict[str, dict[str, Any]]) -> JevAnswer:
        """Ask one or more typed questions about one state. Returns the answers keyed
        by question name, or an error (never raises for a failed call)."""
        if not isinstance(questions, dict) or not questions:
            raise MekikiError("questions must be a non-empty dict of name -> question.")
        key = self._key(state, questions)
        cached = self._read_cache(key)
        if cached is not None:
            with self._lock:
                self.usage.calls += 1
                self.usage.cache_hits += 1
            return JevAnswer(answers=cached["answers"],
                             input_tokens=int(cached.get("input_tokens", 0)),
                             output_tokens=int(cached.get("output_tokens", 0)),
                             cost=0.0, from_cache=True, model=str(cached.get("model", "")))

        body = {"state": state, "model": self.model, "questions": questions}
        try:
            resp = self._send(body)
        except Exception as exc:          # HTTP status, connection, missing key - all of them
            with self._lock:
                self.usage.calls += 1
                self.usage.errors += 1
            return JevAnswer(answers={}, error=f"{type(exc).__name__}: {exc}")

        answers = resp.get("answers") if isinstance(resp, dict) else None
        if not isinstance(answers, dict):
            with self._lock:
                self.usage.calls += 1
                self.usage.errors += 1
            return JevAnswer(answers={}, error="The response carries no answers object")
        usage = resp.get("usage") or {}
        in_tok = int(usage.get("input_tokens", 0) or 0)
        out_tok = int(usage.get("output_tokens", 0) or 0)
        cost = _reported_cost(resp)
        if cost is None:
            cost = self._price(in_tok, out_tok)
        model = str(resp.get("model", ""))

        # The request is stored with the answers: the key is a sha256, so without the
        # body there is no way to trace afterwards what question produced this answer.
        self._write_cache(key, {"answers": answers, "input_tokens": in_tok,
                                "output_tokens": out_tok, "model": model,
                                "state": state, "questions": questions})
        with self._lock:
            self.usage.calls += 1
            self.usage.input_tokens += in_tok
            self.usage.output_tokens += out_tok
            self.usage.cost += cost
        return JevAnswer(answers=answers, input_tokens=in_tok, output_tokens=out_tok,
                         cost=cost, model=model)

    def ask_many(self, states: Sequence[Any],
                 questions: dict[str, dict[str, Any]] | Sequence[dict[str, dict[str, Any]]],
                 progress: Callable[[int, int], None] | None = None) -> list[JevAnswer]:
        """Ask about several states at once. Parallel, but the results keep the input
        order. `questions` is one dict shared by every state, or one dict per state."""
        n = len(states)
        if n == 0:
            return []
        if isinstance(questions, dict):
            qs: list[dict[str, dict[str, Any]]] = [questions] * n
        else:
            qs = list(questions)
            if len(qs) != n:
                raise MekikiError(
                    f"questions has {len(qs)} entries but there are {n} states.")
        out: list[JevAnswer | None] = [None] * n
        done = 0
        with ThreadPoolExecutor(max_workers=max(self.max_workers, 1)) as pool:
            futures = {pool.submit(self.ask, states[i], qs[i]): i for i in range(n)}
            for fut in as_completed(futures):
                out[futures[fut]] = fut.result()
                done += 1
                if progress:
                    progress(done, n)
        return [a for a in out if a is not None]

    # --- Convenience: one question per state ----------------------------------

    @staticmethod
    def _to_choice(ans: JevAnswer, options: Sequence[str], name: str) -> JevChoice:
        if not ans.ok:
            return JevChoice(choice=None, cost=ans.cost, from_cache=ans.from_cache,
                             error=ans.error)
        a = ans.answers.get(name) or {}
        choice = a.get("choice")
        probs = a.get("probabilities") or {}
        if choice is None or str(choice) not in [str(o) for o in options]:
            return JevChoice(choice=None, cost=ans.cost, from_cache=ans.from_cache,
                             error=f"answer {choice!r} is not one of the options")
        probs = {str(k): float(v) for k, v in probs.items()}
        return JevChoice(choice=str(choice), confidence=float(a.get("confidence", 0.0)),
                         probabilities=probs, cost=ans.cost, from_cache=ans.from_cache)

    def choose(self, state: Any, options: Sequence[str], instructions: str,
               descriptions: dict[str, str] | None = None) -> JevChoice:
        """Pick one of `options` for one state. Returns `(choice, confidence,
        probabilities, cost)` (a `JevChoice`)."""
        q = choice_question(options, instructions, descriptions)
        return self._to_choice(self.ask(state, {"choice": q}), options, "choice")

    def choose_many(self, states: Sequence[Any], options: Sequence[str] | Sequence[Sequence[str]],
                    instructions: str,
                    descriptions: dict[str, str] | Sequence[dict[str, str] | None] | None = None,
                    progress: Callable[[int, int], None] | None = None) -> list[JevChoice]:
        """`choose` for several states in parallel, keeping the order.

        `options` is one list shared by every state, or one list per state (the
        options may differ per row, e.g. when each row's evidence has its own
        description); `descriptions` likewise.
        """
        n = len(states)
        if n == 0:
            return []
        # One shared option list, or one list per state (the first entry tells which)
        per_row = bool(options) and not isinstance(options[0], str)
        opts: list[Sequence[str]] = [list(o) for o in options] if per_row else [list(options)] * n
        if isinstance(descriptions, dict) or descriptions is None:
            descs: list[dict[str, str] | None] = [descriptions] * n
        else:
            descs = list(descriptions)
        if len(opts) != n or len(descs) != n:
            raise MekikiError("options / descriptions must have one entry per state.")
        qs = [{"choice": choice_question(o, instructions, d)}
              for o, d in zip(opts, descs, strict=True)]
        answers = self.ask_many(states, qs, progress=progress)
        return [self._to_choice(a, o, "choice") for a, o in zip(answers, opts, strict=True)]

    @staticmethod
    def _to_score(ans: JevAnswer, name: str) -> JevScore:
        if not ans.ok:
            return JevScore(score=None, cost=ans.cost, from_cache=ans.from_cache,
                            error=ans.error)
        a = ans.answers.get(name) or {}
        if a.get("score") is None:
            return JevScore(score=None, cost=ans.cost, from_cache=ans.from_cache,
                            error="the answer carries no score")
        probs = {str(k): float(v) for k, v in (a.get("probabilities") or {}).items()}
        return JevScore(score=float(a["score"]), confidence=float(a.get("confidence", 0.0)),
                        probabilities=probs, legend=dict(a.get("legend") or {}),
                        cost=ans.cost, from_cache=ans.from_cache)

    def score(self, state: Any, levels: Sequence[str], instructions: str) -> JevScore:
        """Rate one state against ordered `levels`. Returns `(score, confidence,
        probabilities, cost)` (a `JevScore`); `score` is the expected level index."""
        q = score_question(levels, instructions)
        return self._to_score(self.ask(state, {"score": q}), "score")

    def score_many(self, states: Sequence[Any], levels: Sequence[str], instructions: str,
                   progress: Callable[[int, int], None] | None = None) -> list[JevScore]:
        """`score` for several states in parallel, keeping the order."""
        q = {"score": score_question(levels, instructions)}
        return [self._to_score(a, "score") for a in self.ask_many(states, q, progress=progress)]

    def summary(self) -> dict[str, Any]:
        return {"model": self.model, "provider": "typesafe", **self.usage.as_dict()}


def _error_message(body: Any) -> str:
    """Pull the human-readable part out of an error body (the API returns
    `{"error": ...}`, `{"message": ...}` or a validation `detail` list)."""
    if isinstance(body, str):
        return body[:300] or "(no body)"
    if not isinstance(body, dict):
        return "(no body)"
    err = body.get("error")
    if isinstance(err, str):
        return err
    if isinstance(err, dict) and isinstance(err.get("message"), str):
        return err["message"]
    if isinstance(body.get("message"), str):
        return body["message"]
    detail = body.get("detail")
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        parts = []
        for d in detail:
            if isinstance(d, dict) and isinstance(d.get("msg"), str):
                loc = d.get("loc")
                path = ".".join(str(x) for x in loc if x != "body") if isinstance(loc, list) else ""
                parts.append(f"{path}: {d['msg']}" if path else d["msg"])
        if parts:
            return "; ".join(parts)
    return _dumps(body)[:300]
