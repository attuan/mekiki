"""Tests for the LLM client (mekiki/llm.py) that need no network.

What to protect:

1. **The Claude route is stable.** Cache keys for `claude-...` models are pinned,
   and `anthropic/claude-...` must map onto the same key. The cache was accumulated with real charges.
2. **Other providers go through LiteLLM**, read their keys from `.env`, and share
   the cache, the cost accounting and the error handling with the Claude route.
   These tests use LiteLLM's `mock_response` so nothing leaves the machine.
3. **Without a key nothing crashes**: `available()` is False and
   `why_unavailable()` names the missing key or package.

Run: .venv/bin/python -m pytest tests/test_llm.py -q
"""

from __future__ import annotations

import json
import os

import pytest

from mekiki import llm
from mekiki.llm import ClaudeClient, LLMClient, provider_of

SCHEMA = {"type": "object", "properties": {"value": {"type": "number"}},
          "required": ["value"], "additionalProperties": False}

litellm = pytest.importorskip("litellm", reason="the LiteLLM route needs mekiki[litellm]")


# --- Route selection -----------------------------------------------------------


def test_claude_names_take_the_official_route():
    assert provider_of("claude-opus-5") == ("anthropic", "claude-opus-5")
    assert provider_of("anthropic/claude-opus-5") == ("anthropic", "claude-opus-5")


def test_other_names_take_litellm():
    assert provider_of("openai/gpt-5") == ("litellm", "openai/gpt-5")
    assert provider_of("gemini/gemini-2.5-pro") == ("litellm", "gemini/gemini-2.5-pro")
    assert provider_of("gpt-5") == ("litellm", "gpt-5")


def test_claude_cache_key_is_unchanged_and_prefix_insensitive():
    """The exact bytes of the pre-LiteLLM key, so existing cache files keep hitting."""
    bare = LLMClient(model="claude-opus-5", api_key="k", cache_dir=None)
    prefixed = LLMClient(model="anthropic/claude-opus-5", api_key="k", cache_dir=None)
    expected = llm.hashlib.sha256(json.dumps(
        {"model": "claude-opus-5", "effort": "low", "system": "s", "user": "u",
         "schema": SCHEMA}, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    assert bare._key("s", "u", SCHEMA) == expected
    assert prefixed._key("s", "u", SCHEMA) == expected


def test_old_name_is_the_same_class():
    assert ClaudeClient is LLMClient


# --- Keys and the settings file ------------------------------------------------


@pytest.fixture
def dotenv(tmp_path, monkeypatch):
    """Point the settings-file lookup at tmp_path only.

    `find_dotenv` also falls back to the repository's own `.env`, and the
    developer's real keys must never decide whether these tests pass. Importing
    LiteLLM copies the working directory's `.env` into the environment, so the
    gateway settings are removed from there too.
    """
    path = tmp_path / ".env"
    monkeypatch.setattr(llm, "find_dotenv", lambda: path if path.is_file() else None)
    for name in ("AI_GATEWAY_API_KEY", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    return path


def test_dotenv_is_read_for_any_key_and_environment_wins(dotenv, monkeypatch):
    dotenv.write_text(
        "# comment\nANTHROPIC_API_KEY='from-file'\nOPENAI_API_KEY=\"sk-file\"\nEMPTY=\n",
        encoding="utf-8")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    assert llm.load_api_key("ANTHROPIC_API_KEY") == "from-file"
    assert llm.load_api_key("OPENAI_API_KEY") == "sk-env"
    assert llm.load_api_key("EMPTY") is None
    assert llm.load_api_key("ABSENT") is None


def test_export_dotenv_does_not_overwrite(dotenv, monkeypatch):
    dotenv.write_text("GEMINI_API_KEY=file\nOPENAI_API_KEY=file\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "env")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert llm.export_dotenv() == ["GEMINI_API_KEY"]
    assert os.environ["OPENAI_API_KEY"] == "env"
    assert os.environ["GEMINI_API_KEY"] == "file"


def test_missing_key_is_reported_by_name(dotenv, monkeypatch):
    assert not dotenv.exists()           # no settings file at all
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    c = LLMClient(model="openai/gpt-5", cache_dir=None)
    assert not c.available()
    assert "OPENAI_API_KEY" in c.why_unavailable()

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    a = LLMClient(model="claude-opus-5", cache_dir=None)
    assert not a.available()
    assert "ANTHROPIC_API_KEY" in a.why_unavailable()


def test_key_from_dotenv_makes_litellm_route_available(dotenv, monkeypatch):
    dotenv.write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    c = LLMClient(model="openai/gpt-5", cache_dir=None)
    assert c.available()


# --- The LiteLLM route, offline --------------------------------------------------


def _mocked(monkeypatch, text: str):
    """Make litellm.completion answer `text` without a network call, recording kwargs."""
    seen: dict = {}
    real = litellm.completion

    def fake(**kw):
        seen.update(kw)
        return real(model=kw["model"], messages=kw["messages"], mock_response=text)

    monkeypatch.setattr(litellm, "completion", fake)
    return seen


def test_litellm_route_parses_json_bills_and_caches(tmp_path, monkeypatch):
    seen = _mocked(monkeypatch, '{"value": 42}')
    c = LLMClient(model="openai/gpt-5", api_key="sk-test", cache_dir=tmp_path)

    a = c.ask("sys", "usr", SCHEMA)
    assert a.ok and a.data == {"value": 42} and not a.from_cache
    assert a.input_tokens > 0 and a.cost > 0          # priced from LiteLLM's table
    assert seen["response_format"]["json_schema"]["schema"] == SCHEMA
    assert seen["messages"][0] == {"role": "system", "content": "sys"}
    assert seen["reasoning_effort"] == "low" and seen["api_key"] == "sk-test"

    b = c.ask("sys", "usr", SCHEMA)                   # second time: from disk
    assert b.from_cache and b.cost == 0.0 and b.data == {"value": 42}
    assert c.usage.calls == 2 and c.usage.cache_hits == 1
    assert c.summary()["provider"] == "litellm"


def test_litellm_route_reports_bad_json_as_an_error_not_an_exception(tmp_path, monkeypatch):
    _mocked(monkeypatch, "not json")
    c = LLMClient(model="openai/gpt-5", api_key="sk-test", cache_dir=tmp_path)
    a = c.ask("sys", "usr", SCHEMA)
    assert not a.ok and "JSON" in a.error
    assert c.usage.errors == 1
    assert not any(tmp_path.rglob("*.json")), "a failed answer must not be cached"


def test_unit_prices_come_from_litellm_for_other_providers():
    pin, pout = LLMClient(model="openai/gpt-5", api_key="k", cache_dir=None).unit_prices()
    assert 0 < pin < pout
    assert LLMClient(model="claude-opus-5", api_key="k", cache_dir=None).unit_prices() \
        == llm.PRICING["claude-opus-5"]


# --- The Vercel AI Gateway -------------------------------------------------------


def test_claude_falls_back_to_the_gateway_and_keeps_its_cache_key(dotenv, monkeypatch):
    dotenv.write_text("AI_GATEWAY_API_KEY=vck_test\n", encoding="utf-8")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    c = LLMClient(model="claude-opus-5", cache_dir=None)
    assert c.available() and c.via_gateway and c.api_key == "vck_test"
    assert c.base_url == "https://ai-gateway.vercel.sh"
    assert c.wire_model == "anthropic/claude-opus-5"
    direct = LLMClient(model="claude-opus-5", api_key="k", cache_dir=None)
    assert not direct.via_gateway and direct.wire_model == "claude-opus-5"
    assert c._key("s", "u", SCHEMA) == direct._key("s", "u", SCHEMA)
    assert c.summary()["via_gateway"] is True


def test_an_anthropic_key_wins_unless_base_url_names_the_gateway(dotenv, monkeypatch):
    dotenv.write_text("ANTHROPIC_API_KEY=sk-ant-test\nAI_GATEWAY_API_KEY=vck_test\n",
                      encoding="utf-8")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert not LLMClient(cache_dir=None).via_gateway
    forced = LLMClient(cache_dir=None, base_url="https://ai-gateway.vercel.sh")
    assert forced.via_gateway and forced.api_key == "vck_test"


def test_claude_on_the_gateway_sends_the_prefixed_name_and_native_fields(monkeypatch):
    sent: dict = {}

    class Usage:
        input_tokens, output_tokens = 100, 10
        cache_read_input_tokens = cache_creation_input_tokens = 0

    class Block:
        type, text = "text", '{"value": 1}'

    class Resp:
        content, usage = [Block()], Usage()
        model_extra = {"provider_metadata": {"gateway": {"cost": "0.0012"}}}

    class Messages:
        def create(self, **kw):
            sent.update(kw)
            return Resp()

    class Fake:
        messages = Messages()

    c = LLMClient(model="claude-opus-5", api_key="vck_test", cache_dir=None,
                  base_url="https://ai-gateway.vercel.sh")
    c._client = Fake()
    a = c.ask("sys", "usr", SCHEMA)
    assert a.ok and a.data == {"value": 1}
    assert sent["model"] == "anthropic/claude-opus-5"
    assert sent["output_config"]["format"]["schema"] == SCHEMA
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert a.cost == pytest.approx(0.0012)             # the gateway's own figure


def test_other_providers_use_the_gateway_only_without_their_own_key(dotenv, monkeypatch):
    dotenv.write_text("AI_GATEWAY_API_KEY=vck_test\n", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    gpt = LLMClient(model="openai/gpt-5", cache_dir=None)
    assert gpt.via_gateway and gpt.available()
    assert gpt.wire_model == "vercel_ai_gateway/openai/gpt-5" and gpt.model == "openai/gpt-5"
    gem = LLMClient(model="gemini/gemini-2.5-pro", cache_dir=None)
    assert gem.wire_model == "vercel_ai_gateway/google/gemini-2.5-pro"
    assert not LLMClient(model="ollama/llama3", cache_dir=None).via_gateway
    monkeypatch.setenv("OPENAI_API_KEY", "sk-own")
    assert not LLMClient(model="openai/gpt-5", cache_dir=None).via_gateway


def test_litellm_on_the_gateway_is_priced_from_the_provider_list(dotenv, monkeypatch):
    dotenv.write_text("AI_GATEWAY_API_KEY=vck_test\n", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    seen = _mocked(monkeypatch, '{"value": 7}')
    c = LLMClient(model="openai/gpt-5", cache_dir=None)
    a = c.ask("sys", "usr", SCHEMA)
    assert a.ok and a.data == {"value": 7} and a.cost > 0
    assert seen["model"] == "vercel_ai_gateway/openai/gpt-5" and seen["api_key"] == "vck_test"
    assert c.unit_prices() == LLMClient(model="openai/gpt-5", api_key="k",
                                        cache_dir=None).unit_prices()
