"""Foundation for calling an LLM (shared by `EvidencePredictor`, `SemanticEncoder`'s fallback, `diagnose` and
knowledge columns).

**This is the only place that emits HTTP.** The layers above (`EvidencePredictor` /
`LLMFallback` / `diagnose` / `KnowledgeEncoder`) only know "send JSON, get JSON back".

Two routes sit behind one class, chosen by the `model` string (`provider_of`).

- **Claude** (`claude-...`, or `anthropic/claude-...`) goes through the official
  `anthropic` SDK: structured output via `output_config.format`, depth via
  `output_config.effort`, the system prompt on the prompt cache.
- **Everything else** (`openai/gpt-5`, `gemini/gemini-2.5-pro`, `ollama/...`, ...) goes
  through LiteLLM (`pip install "mekiki[litellm]"`). LiteLLM finds the provider's key in
  the environment (`OPENAI_API_KEY`, `GEMINI_API_KEY`, ...), translates the JSON schema
  into that provider's structured-output feature, and knows the price of each model.
  Claude is *not* routed through LiteLLM on purpose: LiteLLM implements structured
  output there with a forced tool call and maps `effort` onto thinking budgets, and both
  break on current Claude models.

Three design points are non-negotiable, on both routes.

1. **Always go through the disk cache.** 5-fold cross-validation hits the same rows
   again and again, and every re-run after a code change would re-bill every row.
   Results are stored in `sampledata/processed/llm_cache/` keyed by the prompt hash,
   and the second time onwards is returned free of charge.
   **The user prompt is stored, not only the response.** The key is a sha256 and
   cannot be reversed, so without the body there is no way to trace afterwards
   "what question produced this answer". The system prompt is identical for every
   row, so only its fingerprint (`system_sha`) is kept.
2. **Always count the cost.** The unit cost per row cannot be reported without
   measuring it. Tokens and dollars are accumulated on every call.
3. **Send in parallel.** One request per row means that a serial run takes tens of
   minutes for 300 rows.

**Both routes can go through the Vercel AI Gateway** instead of each provider,
with one key (`AI_GATEWAY_API_KEY`) and one bill. The gateway is used when the
provider's own key is missing and the gateway key is present, or when asked for
with `base_url` (or `ANTHROPIC_BASE_URL`) pointing at `ai-gateway.vercel.sh`.
Claude keeps the official SDK and its native fields there (the gateway passes
`output_config` and `cache_control` through unchanged); other providers go through
LiteLLM's `vercel_ai_gateway/` provider. **The model name sent on the wire gains the
gateway's prefix (`anthropic/claude-...`), but the cache key does not**, so answers
cached from the direct route keep hitting and vice versa.

API keys live in `.env` (or the environment). Importing works without a key;
`available()` merely returns False (so the tests can run) and `why_unavailable()`
says which key or package is missing.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import warnings
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mekiki.errors import MekikiWarning, missing_extra
from mekiki.paths import cache_dir, find_dotenv

#: Cache location. Inside a clone of the repository this is `sampledata/processed/llm_cache/`;
#: when used as an installed package it becomes the per-user cache location
#: (`mekiki/paths.py`). Can be moved with `MEKIKI_CACHE_DIR`.
DEFAULT_CACHE_DIR = cache_dir("llm_cache")

#: Claude models, with input/output dollar prices per million tokens. Only the Claude
#: route uses this table (LiteLLM carries its own price list for every other provider).
#: The prices are only used for cost estimates, so if they change fix them here only.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
DEFAULT_MODEL = "claude-opus-5"

#: Name of the key the Claude route reads.
ANTHROPIC_KEY = "ANTHROPIC_API_KEY"
ANTHROPIC_BASE_URL = "ANTHROPIC_BASE_URL"

#: The Vercel AI Gateway: its key, its host, and its Anthropic-compatible base URL.
AI_GATEWAY_KEY = "AI_GATEWAY_API_KEY"
GATEWAY_HOST = "ai-gateway.vercel.sh"
GATEWAY_ANTHROPIC_URL = "https://ai-gateway.vercel.sh"
#: LiteLLM's prefix for the gateway (it speaks the gateway's OpenAI-compatible API).
GATEWAY_LITELLM_PREFIX = "vercel_ai_gateway/"
#: LiteLLM provider prefix -> (the gateway's name for the same maker, the key LiteLLM
#: would otherwise need). Only these are sent to the gateway automatically; a local
#: model (`ollama/...`) never is.
GATEWAY_VENDORS: dict[str, tuple[str, str]] = {
    "openai":   ("openai", "OPENAI_API_KEY"),
    "gemini":   ("google", "GEMINI_API_KEY"),
    "xai":      ("xai", "XAI_API_KEY"),
    "mistral":  ("mistral", "MISTRAL_API_KEY"),
    "deepseek": ("deepseek", "DEEPSEEK_API_KEY"),
}


@dataclass
class LLMAnswer:
    """The answer to one request, and what it cost.

    A row with `from_cache` True was not billed this time (`cost` is 0.0).
    `input_tokens` and friends still hold the values from the first call, so
    "what would this have cost without the cache" can be computed afterwards.
    """

    data: dict[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost: float = 0.0
    from_cache: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class Usage:
    """Running totals over calls. `EvidencePredictor.cost()` returns this as is."""

    calls: int = 0
    cache_hits: int = 0
    errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_calls": self.calls,
            "cache_hits": self.cache_hits,
            "errors": self.errors,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cost_usd": round(self.cost, 4),
        }


# --- Settings file ------------------------------------------------------------


def read_dotenv() -> dict[str, str]:
    """Read every `NAME=value` line of the settings file found by `find_dotenv`.

    Parsed by hand so this works even where python-dotenv is not installed.
    The search walks **upward from the current directory** (`mekiki/paths.py`):
    when used as an installed library, the settings live in the user's project,
    not next to site-packages. Returns an empty dict when there is no file.
    """
    env_path = find_dotenv()
    if env_path is None:
        return {}
    values: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        if value:
            values[name.strip()] = value
    return values


def load_api_key(name: str = ANTHROPIC_KEY) -> str | None:
    """Look for one key in the environment, then in `.env`. None if absent.

    The environment wins so that a key exported for a one-off run overrides the
    file without editing it.
    """
    key = os.environ.get(name)
    if key:
        return key.strip() or None
    return read_dotenv().get(name)


def export_dotenv() -> list[str]:
    """Copy `.env` into `os.environ` without overwriting anything already set.

    The LiteLLM route needs this: LiteLLM looks up each provider's key
    (`OPENAI_API_KEY`, `GEMINI_API_KEY`, ...) in the environment itself, and
    there are too many names to pass one by one. Returns the names that were set.
    """
    added = []
    for name, value in read_dotenv().items():
        if not os.environ.get(name):
            os.environ[name] = value
            added.append(name)
    return added


# --- Route selection ----------------------------------------------------------


def provider_of(model: str) -> tuple[str, str]:
    """Decide the route for a model string. Returns `(provider, model)`.

    `provider` is `"anthropic"` (official SDK) or `"litellm"`. For the Claude route
    the returned model is the **bare** name (`anthropic/claude-opus-5` becomes
    `claude-opus-5`), so that the cache key does not depend on how the model was
    spelled: the cache was accumulated under bare names.
    """
    prefix, _, rest = model.partition("/")
    if prefix == "anthropic" and rest:
        return "anthropic", rest
    if "/" not in model and model.startswith("claude"):
        return "anthropic", model
    return "litellm", model


def _is_gateway(url: str | None) -> bool:
    return bool(url) and GATEWAY_HOST in url


def _gateway_cost(resp: Any) -> float | None:
    """The dollars the Vercel AI Gateway reports for one call, when the response
    carries them (`provider_metadata.gateway.cost`); None otherwise."""
    extra = getattr(resp, "model_extra", None)
    if not isinstance(extra, dict):
        extra = getattr(resp, "_hidden_params", None)
    if not isinstance(extra, dict):
        return None
    meta = extra.get("provider_metadata") or extra.get("providerMetadata")
    gateway = meta.get("gateway") if isinstance(meta, dict) else None
    if not isinstance(gateway, dict) or gateway.get("cost") is None:
        return None
    try:
        return max(float(gateway["cost"]), 0.0)
    except (TypeError, ValueError):
        return None


def _sha8(text: str) -> str:
    """Fingerprint of a prompt body (first 8 hex digits), to tell which version answered."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


class LLMClient:
    """A thin client that only asks an LLM to write JSON.

    A JSON Schema is passed with every request and the provider guarantees the
    return value matches it (Claude: `output_config.format`; others: LiteLLM's
    `response_format`, which needs `additionalProperties: false` and a full
    `required` list on every object, as mekiki's own schemas have). There is no
    need to write regular expressions here to pick numbers out of text.

    `effort` limits the depth of reasoning. On Claude it is `output_config.effort`
    with adaptive thinking left on ("low" is enough for a judgement made on top
    of supplied evidence; switching thinking off invites tool calls or tags leaking
    into the output). On other providers it is sent as `reasoning_effort` and
    silently dropped where the model has no such setting.

    `api_key` overrides the environment for the chosen route. Leave it unset to
    read `.env` (Claude: `ANTHROPIC_API_KEY`; others: whatever LiteLLM names,
    see `why_unavailable()`). When that key is missing and `AI_GATEWAY_API_KEY` is
    set, the call goes through the Vercel AI Gateway instead (`via_gateway`).

    `base_url` sends the Claude route elsewhere (default: `ANTHROPIC_BASE_URL`, else
    Anthropic). Pointing it at `https://ai-gateway.vercel.sh` uses the gateway even
    when an Anthropic key is present; the key is then `AI_GATEWAY_API_KEY`.
    """

    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "low",
                 max_tokens: int = 1024, cache_dir: str | Path | None = "default",
                 api_key: str | None = None, max_workers: int = 8,
                 timeout: float = 120.0, base_url: str | None = None) -> None:
        self.provider, self.model = provider_of(model)
        self.effort = effort
        self.max_tokens = max_tokens
        self.max_workers = max_workers
        self.timeout = timeout
        self.api_key: str | None = api_key
        self.base_url: str | None = None
        #: The model name sent on the wire. It differs from `self.model` (which the
        #: cache key uses) only by the gateway's prefix.
        self.wire_model = self.model
        if self.provider == "anthropic":
            self._resolve_anthropic(api_key, base_url)
        else:
            self._resolve_litellm(api_key)
        if cache_dir == "default":
            cache_dir = DEFAULT_CACHE_DIR
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.usage = Usage()
        self._client = None
        self._lock = threading.Lock()
        self._warned_price = False

    def _resolve_anthropic(self, api_key: str | None, base_url: str | None) -> None:
        """Key and endpoint of the Claude route: Anthropic's key first, then the gateway's."""
        self.base_url = base_url or load_api_key(ANTHROPIC_BASE_URL)
        if api_key is None:
            if _is_gateway(self.base_url):
                self.api_key = load_api_key(AI_GATEWAY_KEY)
            else:
                self.api_key = load_api_key(ANTHROPIC_KEY)
                if not self.api_key and not self.base_url:
                    self.api_key = load_api_key(AI_GATEWAY_KEY)
                    if self.api_key:
                        self.base_url = GATEWAY_ANTHROPIC_URL
        if self.via_gateway:
            self.wire_model = f"anthropic/{self.model}"

    def _resolve_litellm(self, api_key: str | None) -> None:
        """LiteLLM reads the provider's key itself; the gateway is used for a known maker
        whose key is missing when the gateway key is present, or when named explicitly."""
        if self.model.startswith(GATEWAY_LITELLM_PREFIX):
            if api_key is None:
                self.api_key = load_api_key(AI_GATEWAY_KEY)
            return
        vendor, _, rest = self.model.partition("/")
        if api_key is not None or vendor not in GATEWAY_VENDORS or not rest:
            return
        gateway_vendor, provider_key = GATEWAY_VENDORS[vendor]
        if load_api_key(provider_key):
            return
        gateway_key = load_api_key(AI_GATEWAY_KEY)
        if gateway_key:
            self.api_key = gateway_key
            self.wire_model = f"{GATEWAY_LITELLM_PREFIX}{gateway_vendor}/{rest}"

    # --- State --------------------------------------------------------

    @property
    def via_gateway(self) -> bool:
        """Whether calls go through the Vercel AI Gateway rather than the provider."""
        if self.provider == "anthropic":
            return _is_gateway(self.base_url)
        return self.wire_model.startswith(GATEWAY_LITELLM_PREFIX)

    def available(self) -> bool:
        """Whether a call can actually be made right now (False without a key)."""
        return self.why_unavailable() is None

    def why_unavailable(self) -> str | None:
        """The reason a call would fail, as a sentence telling the user what to do.

        None when everything is in place. The Claude route only checks the key
        (the `anthropic` package is reported at call time). The LiteLLM
        route needs the package to know which key names the provider expects, so
        a missing package is reported first.
        """
        if self.provider == "anthropic":
            if self.api_key:
                return None
            if self.via_gateway:
                return (f"{AI_GATEWAY_KEY} is not set, but base_url points at the Vercel "
                        "AI Gateway. Write the gateway key in `.env` (or export it).")
            return (f"Neither {ANTHROPIC_KEY} nor {AI_GATEWAY_KEY} is set. Run "
                    "`cp .env.example .env` and write your Anthropic key, or a Vercel "
                    "AI Gateway key, there (or export it as an environment variable).")
        if self.api_key:
            return None
        try:
            import litellm
        except ImportError:
            return (f"{self.model} is called through LiteLLM, which is not installed. "
                    "Install it with `pip install \"mekiki[litellm]\"`.")
        export_dotenv()
        try:
            check = litellm.validate_environment(self.model)
        except Exception as exc:                     # unknown provider prefix, etc.
            return f"LiteLLM cannot resolve the model {self.model!r}: {exc}"
        missing = list(check.get("missing_keys") or [])
        if check.get("keys_in_environment", False) or not missing:
            return None
        names = " or ".join(missing)
        return (f"{names} is not set (needed for {self.model}). Run `cp .env.example .env` "
                "and write the key there (or export it as an environment variable).")

    def _ensure_client(self):
        """The Claude route's SDK client, created on first use."""
        if self._client is None:
            reason = self.why_unavailable()
            if reason is not None:
                raise RuntimeError(reason)
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover - depends on the environment
                raise missing_extra("anthropic", "llm") from e
            self._client = anthropic.Anthropic(api_key=self.api_key,
                                               base_url=self.base_url,
                                               timeout=self.timeout)
        return self._client

    def _ensure_litellm(self):
        """The LiteLLM module, checked on first use."""
        if self._client is None:
            try:
                import litellm
            except ImportError as e:  # pragma: no cover - depends on the environment
                raise missing_extra("litellm", "litellm") from e
            reason = self.why_unavailable()
            if reason is not None:
                raise RuntimeError(reason)
            self._client = litellm
        return self._client

    # --- Cache --------------------------------------------------------

    def _key(self, system: str, user: str, schema: dict) -> str:
        payload = json.dumps(
            {"model": self.model, "effort": self.effort, "system": system,
             "user": user, "schema": schema},
            sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path | None:
        if self.cache_dir is None:
            return None
        # One level of subdirectories, to avoid tens of thousands of files in one directory
        d = self.cache_dir / key[:2]
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{key}.json"

    def _read_cache(self, key: str) -> dict | None:
        """Read the cache. Returns None on a corrupt entry so the call is retried.

        Corruption does happen in practice (power loss mid-write, a clash with
        another process, a format written by an older version). **Raising here
        would take down a several-hundred-row measurement over one broken file.**
        """
        path = self._cache_path(key)
        if path is None or not path.exists():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return None
        if not isinstance(record, dict) or not isinstance(record.get("data"), dict):
            return None
        return record

    def _write_cache(self, key: str, record: dict) -> None:
        path = self._cache_path(key)
        if path is None:
            return
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)  # never leaves a half-written JSON behind on a crash

    # --- Prices -------------------------------------------------------

    def unit_prices(self) -> tuple[float, float]:
        """Input / output dollars per million tokens for this model, for estimates.

        Claude comes from `PRICING`; other models from LiteLLM's price list.
        Unknown models fall back to the default Claude model's prices rather
        than raising, because an estimate is better than no estimate.
        """
        if self.provider == "anthropic":
            return PRICING.get(self.model, PRICING[DEFAULT_MODEL])
        try:
            import litellm
            info = litellm.get_model_info(self._price_model())
            pin = float(info.get("input_cost_per_token") or 0.0) * 1_000_000
            pout = float(info.get("output_cost_per_token") or 0.0) * 1_000_000
            return pin, pout
        except Exception:
            return PRICING[DEFAULT_MODEL]

    def _price_model(self) -> str:
        """The name LiteLLM's price list knows this model by (the gateway's prefix
        removed; the gateway charges the provider's list price)."""
        return self.model.removeprefix(GATEWAY_LITELLM_PREFIX)

    def _price(self, input_tokens: int, output_tokens: int,
               cache_read: int) -> float:
        pin, pout = PRICING.get(self.model, PRICING[DEFAULT_MODEL])
        # Cache reads cost 1/10 of the normal rate; that is the cached system prompt.
        return ((input_tokens * pin + cache_read * pin * 0.1
                 + output_tokens * pout) / 1_000_000)

    # --- Core ---------------------------------------------------------

    def _call_anthropic(self, system: str, user: str,
                        schema: dict) -> tuple[str, int, int, int, float]:
        """One request on the official SDK. Returns (text, in, out, cache_read, cost)."""
        client = self._ensure_client()
        resp = client.messages.create(
            model=self.wire_model,
            max_tokens=self.max_tokens,
            # Put the system prompt on the prompt cache. It is shared by every
            # row, so from the second row on its input is billed at 1/10.
            system=[{"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort,
                           "format": {"type": "json_schema", "schema": schema}},
        )
        text = next((b.text for b in resp.content if b.type == "text"), "")
        u = resp.usage
        in_tok = int(getattr(u, "input_tokens", 0) or 0)
        out_tok = int(getattr(u, "output_tokens", 0) or 0)
        cr_tok = int(getattr(u, "cache_read_input_tokens", 0) or 0)
        cw_tok = int(getattr(u, "cache_creation_input_tokens", 0) or 0)
        cost = _gateway_cost(resp)
        if cost is None:
            cost = self._price(in_tok + int(cw_tok * 1.25), out_tok, cr_tok)
        return text, in_tok, out_tok, cr_tok, cost

    def _call_litellm(self, system: str, user: str,
                      schema: dict) -> tuple[str, int, int, int, float]:
        """One request through LiteLLM. Returns (text, in, out, cache_read, cost).

        The schema goes in `response_format` (OpenAI's shape, which LiteLLM
        translates per provider). `reasoning_effort` is dropped by `drop_params`
        where the model does not accept it. The cost comes from LiteLLM's price
        list; a model it does not know is billed as 0 with one warning.
        """
        litellm = self._ensure_litellm()
        kwargs: dict[str, Any] = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        resp = litellm.completion(
            model=self.wire_model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            response_format={"type": "json_schema",
                             "json_schema": {"name": "answer", "schema": schema,
                                             "strict": True}},
            reasoning_effort=self.effort,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            num_retries=2,
            drop_params=True,
            **kwargs,
        )
        text = resp.choices[0].message.content or ""
        u = resp.usage
        in_tok = int(getattr(u, "prompt_tokens", 0) or 0)
        out_tok = int(getattr(u, "completion_tokens", 0) or 0)
        details = getattr(u, "prompt_tokens_details", None)
        cr_tok = int(getattr(details, "cached_tokens", 0) or 0)
        cost = _gateway_cost(resp)
        try:
            if cost is None:
                cost = float(litellm.completion_cost(completion_response=resp,
                                                     model=self._price_model()))
        except Exception as exc:
            cost = 0.0
            if not self._warned_price:
                self._warned_price = True
                warnings.warn(f"LiteLLM has no price for {self.model}; its cost is "
                              f"counted as $0 ({type(exc).__name__}).",
                              MekikiWarning, stacklevel=3)
        return text, in_tok, out_tok, cr_tok, cost

    def ask(self, system: str, user: str, schema: dict) -> LLMAnswer:
        """Send system + user and receive one JSON object conforming to schema."""
        key = self._key(system, user, schema)
        cached = self._read_cache(key)
        if cached is not None:
            ans = LLMAnswer(data=cached["data"],
                            input_tokens=cached.get("input_tokens", 0),
                            output_tokens=cached.get("output_tokens", 0),
                            cache_read_tokens=cached.get("cache_read_tokens", 0),
                            cost=0.0, from_cache=True)
            with self._lock:
                self.usage.calls += 1
                self.usage.cache_hits += 1
            return ans

        call = self._call_anthropic if self.provider == "anthropic" else self._call_litellm
        try:
            text, in_tok, out_tok, cr_tok, cost = call(system, user, schema)
        except Exception as exc:  # network, rate limit, schema violation - all of them
            with self._lock:
                self.usage.calls += 1
                self.usage.errors += 1
            return LLMAnswer(data={}, error=f"{type(exc).__name__}: {exc}")

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            with self._lock:
                self.usage.calls += 1
                self.usage.errors += 1
            return LLMAnswer(data={}, error=f"Could not parse JSON: {exc}")
        if not isinstance(data, dict):
            with self._lock:
                self.usage.calls += 1
                self.usage.errors += 1
            return LLMAnswer(data={}, error=f"Expected a JSON object, got {type(data).__name__}")

        # **Store the user prompt together with the answer.** The key is a sha256,
        # so without this there is no way to trace afterwards "what question
        # produced this answer" (explaining behaviour, chasing a suspicious answer).
        # The system prompt is identical for every row, so only its fingerprint
        # is kept, not the body. The body lives in source, e.g. SYSTEM_PROMPT in
        # mekiki/predictor.py.
        record = {"data": data, "input_tokens": in_tok, "output_tokens": out_tok,
                  "cache_read_tokens": cr_tok,
                  "user": user, "system_sha": _sha8(system)}
        self._write_cache(key, record)

        with self._lock:
            self.usage.calls += 1
            self.usage.input_tokens += in_tok
            self.usage.output_tokens += out_tok
            self.usage.cache_read_tokens += cr_tok
            self.usage.cost += cost

        return LLMAnswer(data=data, input_tokens=in_tok, output_tokens=out_tok,
                         cache_read_tokens=cr_tok, cost=cost)

    def ask_many(self, system: str, users: Sequence[str], schema: dict,
                 progress: Callable[[int, int], None] | None = None,
                 ) -> list[LLMAnswer]:
        """Send several rows at once. Parallel, but the results keep the input order.

        Only the first row is sent serially, ahead of the rest, **for the sake of
        the prompt cache**. Firing eight requests at once would run them all in
        the "cache not yet created" state, and the shared system prompt would be
        billed at full price eight times. (Providers without an explicit cache
        lose nothing from this.)
        """
        if not users:
            return []

        out: list[LLMAnswer | None] = [None] * len(users)
        out[0] = self.ask(system, users[0], schema)
        if progress:
            progress(1, len(users))

        rest = list(range(1, len(users)))
        if rest:
            done = 1
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = {pool.submit(self.ask, system, users[i], schema): i
                           for i in rest}
                for fut in as_completed(futures):
                    out[futures[fut]] = fut.result()
                    done += 1
                    if progress:
                        progress(done, len(users))
        return [a for a in out if a is not None]

    def summary(self) -> dict[str, Any]:
        return {"model": self.model, "provider": self.provider, "effort": self.effort,
                "via_gateway": self.via_gateway, **self.usage.as_dict()}


#: Former name, kept so existing code and the tests' fake clients keep working.
ClaudeClient = LLMClient
