"""OpenAI-compatible client parsing, retries and fallback, using a fake SDK client (no network)."""
from types import SimpleNamespace as NS

import httpx
import openai
import pytest

from errand.agent.llm.base import LLMError
from errand.agent.llm.openai_compat import FallbackClient, OpenAICompatClient
from errand.config import Limits, LLMSettings

SETTINGS = LLMSettings(provider="groq", base_url="http://fake", model="m", api_key="k", fallback=None)
REQ = httpx.Request("POST", "http://fake/chat/completions")


def message(content=None, calls=()):
    tool_calls = [NS(id=f"c{i}", function=NS(name=n, arguments=a)) for i, (n, a) in enumerate(calls)]
    return NS(choices=[NS(message=NS(content=content, tool_calls=tool_calls or None))],
              usage=NS(prompt_tokens=120, completion_tokens=8))


class FakeSDK:
    def __init__(self, *outcomes):
        self.outcomes, self.calls = list(outcomes), []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        o = self.outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return o


def client(*outcomes, retries=2):
    sdk = FakeSDK(*outcomes)
    return OpenAICompatClient(SETTINGS, Limits(llm_retries=retries), client=sdk, sleep=lambda s: None), sdk


def test_parses_tool_calls_usage_and_strips_think_blocks():
    c, sdk = client(message("<think>plan…</think>Sure.", [("search_providers", '{"service": "plumber"}')]))
    r = c.complete([{"role": "user", "content": "hi"}], [{"type": "function"}])
    assert r.text == "Sure." and r.tool_calls[0].name == "search_providers"
    assert r.tool_calls[0].args == {"service": "plumber"} and r.input_tokens == 120
    assert sdk.calls[0]["tool_choice"] == "auto" and sdk.calls[0]["temperature"] == 0


def test_invalid_tool_arguments_are_reported_not_raised():
    c, _ = client(message(None, [("hold_slot", "{not json"), ("book", "[1, 2]")]))
    r = c.complete([], [])
    assert r.tool_calls[0].parse_error.startswith("invalid JSON") and r.tool_calls[0].args == {}
    assert r.tool_calls[1].parse_error == "arguments must be a JSON object"


def test_no_tools_means_no_tool_choice():
    c, sdk = client(message("hello"))
    c.complete([], [])
    assert "tools" not in sdk.calls[0] and "tool_choice" not in sdk.calls[0]


def test_rate_limit_is_retried_then_succeeds():
    rate = openai.RateLimitError("slow down", response=httpx.Response(429, request=REQ), body=None)
    c, sdk = client(rate, message("ok"))
    assert c.complete([], []).text == "ok" and len(sdk.calls) == 2


def test_retries_exhausted_raise_retryable_llm_error():
    conn = openai.APIConnectionError(request=REQ)
    c, _ = client(conn, conn, conn, retries=2)
    with pytest.raises(LLMError) as e:
        c.complete([], [])
    assert e.value.retryable


def test_auth_error_is_not_retried():
    auth = openai.AuthenticationError("bad key", response=httpx.Response(401, request=REQ), body=None)
    c, sdk = client(auth, message("never"))
    with pytest.raises(LLMError) as e:
        c.complete([], [])
    assert not e.value.retryable and len(sdk.calls) == 1


def test_fallback_switches_once_and_reports():
    auth = openai.AuthenticationError("bad key", response=httpx.Response(401, request=REQ), body=None)
    primary, _ = client(auth)
    secondary, _ = client(message("from fallback"), message("again"))
    secondary.model = "qwen3:4b"
    notes = []
    fb = FallbackClient(primary, secondary, on_switch=notes.append)
    assert fb.complete([], []).text == "from fallback" and fb.model == "qwen3:4b"
    assert fb.complete([], []).text == "again" and len(notes) == 1
