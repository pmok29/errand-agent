"""OpenAI-compatible client parsing, retries and fallback, using a fake SDK client (no network)."""
from types import SimpleNamespace as NS

import httpx
import openai
import pytest

from errand.agent.llm.base import LLMError
from errand.agent.llm.openai_compat import FallbackClient, OpenAICompatClient, rate_limit_wait
from errand.config import Limits, LLMSettings, llm_settings

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


def client(*outcomes, retries=2, slept=None):
    sdk = FakeSDK(*outcomes)
    sleep = slept.append if slept is not None else (lambda s: None)
    return OpenAICompatClient(SETTINGS, Limits(llm_retries=retries), client=sdk, sleep=sleep), sdk


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


def test_rate_limit_waits_for_the_providers_reset_hint():
    headers = {"x-ratelimit-reset-tokens": "4.5s"}
    rate = openai.RateLimitError("slow down", response=httpx.Response(429, request=REQ, headers=headers), body=None)
    slept = []
    c, _ = client(rate, message("ok"), slept=slept)
    assert c.complete([], []).text == "ok" and slept == [4.5]
    assert rate_limit_wait({"retry-after": "2"}, 0) == 2.0
    assert rate_limit_wait({"x-ratelimit-reset-tokens": "1m2s"}, 0) == 15.0   # capped
    assert rate_limit_wait({}, 1) == 2.0                                     # no hint: exponential
    # milliseconds are read, not skipped in favour of the (minutes-long) request reset
    assert rate_limit_wait({"x-ratelimit-reset-tokens": "860ms", "x-ratelimit-reset-requests": "2m59.5s"}, 0) == 0.86


def test_transient_failure_uses_fallback_for_that_call_only():
    conn = openai.APIConnectionError(request=REQ)
    primary, psdk = client(conn, message("primary again"), retries=0)
    secondary, _ = client(message("from fallback"))
    notes = []
    fb = FallbackClient(primary, secondary, on_switch=notes.append)
    assert fb.complete([], []).text == "from fallback" and "this call only" in notes[0]
    assert fb.complete([], []).text == "primary again" and len(psdk.calls) == 2


def test_both_failing_reports_both_errors():
    conn = openai.APIConnectionError(request=REQ)
    primary, _ = client(conn, retries=0)
    secondary, _ = client(conn, retries=0)
    secondary.model = "qwen3:4b"
    with pytest.raises(LLMError) as e:
        FallbackClient(primary, secondary).complete([], [])
    assert "fallback qwen3:4b" in str(e.value) and e.value.retryable


def test_model_overrides_do_not_leak_into_the_fallback(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "qwen/qwen3.8-27b")
    assert llm_settings("groq").model == "qwen/qwen3.8-27b"
    assert llm_settings("ollama", overrides=False).model == "qwen3:4b"
    monkeypatch.delenv("LLM_MODEL")
    assert llm_settings("groq").options == {"reasoning_effort": "low"}


def test_server_side_tool_validation_error_is_fed_back_once():
    bad = openai.BadRequestError("Tool call validation failed: /service must be one of ...",
                                 response=httpx.Response(400, request=REQ), body=None)
    c, sdk = client(bad, message(None, [("search_providers", '{"service": "electrician"}')]))
    r = c.complete([{"role": "user", "content": "hi"}], [{"type": "function"}])
    assert r.tool_calls[0].args == {"service": "electrician"}
    assert "previous tool call was rejected" in sdk.calls[1]["messages"][-1]["content"]


def test_only_auth_or_unknown_model_switch_the_fallback_for_good():
    bad = openai.BadRequestError("context too long", response=httpx.Response(400, request=REQ), body=None)
    primary, _ = client(bad, message("primary ok"))
    secondary, _ = client(message("fallback"))
    fb = FallbackClient(primary, secondary)
    assert fb.complete([], []).text == "fallback" and fb.active is primary


def test_fallback_can_be_another_model_on_the_same_provider(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    s = llm_settings("groq:qwen/qwen3.8-27b", overrides=False)
    assert (s.provider, s.model, s.options) == ("groq", "qwen/qwen3.8-27b", {})
