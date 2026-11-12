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
    assert "previous reply was rejected" in sdk.calls[1]["messages"][-1]["content"]


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


def test_daily_limit_gives_up_at_once_instead_of_waiting():
    msg = "Rate limit reached ... on tokens per day (TPD): Limit 200000. Please try again in 6m13.248s."
    rate = openai.RateLimitError(msg, response=httpx.Response(429, request=REQ), body=None)
    slept = []
    c, sdk = client(rate, message("never"), slept=slept)
    with pytest.raises(LLMError) as e:
        c.complete([], [])
    assert e.value.retryable and slept == [] and len(sdk.calls) == 1


def test_waits_never_run_past_the_runs_time_limit():
    from errand.agent.llm.base import LLMDeadline
    headers = {"retry-after": "10"}
    rate = openai.RateLimitError("slow down", response=httpx.Response(429, request=REQ, headers=headers), body=None)
    slept = []
    c, sdk = client(rate, message("ok"), slept=slept)
    with pytest.raises(LLMDeadline):
        c.complete([], [], time_left_s=5)
    assert slept == [] and len(sdk.calls) == 1 and sdk.calls[0]["timeout"] <= 5
    fb = FallbackClient(client(rate)[0], client(message("never"))[0])
    with pytest.raises(LLMDeadline):                       # no fallback attempt once time is up
        fb.complete([], [], time_left_s=5)


def test_per_minute_limits_are_waited_out_beyond_40_seconds():
    """Free tier: 7-8k input tokens/min at ~3k per call, so a turn can need several 15 s waits in a row."""
    headers = {"retry-after": "15"}
    rate = openai.RateLimitError("slow down", response=httpx.Response(429, request=REQ, headers=headers), body=None)
    slept = []
    c, _ = client(rate, rate, rate, rate, message("ok"), slept=slept)
    assert c.complete([], []).text == "ok" and sum(slept) == 60


def test_unparseable_model_output_is_retried_once():
    """Demo run: the fallback gpt-oss-20b got a Groq 400 "Parsing failed" and the turn ended as an outage;
    replaying the same context succeeded, so one corrective retry is enough."""
    bad = openai.BadRequestError("Error code: 400 - Parsing failed. The model generated output that could not be "
                                 "parsed. See 'failed_generation' for more details.",
                                 response=httpx.Response(400, request=REQ), body=None)
    c, sdk = client(bad, message("Here are three options."))
    assert c.complete([{"role": "user", "content": "plumber"}], []).text == "Here are three options."
    assert len(sdk.calls) == 2 and "previous reply was rejected" in sdk.calls[1]["messages"][-1]["content"]
    c, _ = client(bad, bad)
    with pytest.raises(LLMError):                          # only once: a second failure is reported
        c.complete([], [])


def test_daily_limit_skips_the_primary_until_it_resets():
    msg = "Rate limit reached ... on tokens per day (TPD). Please try again in 11m48.48s."
    daily = openai.RateLimitError(msg, response=httpx.Response(429, request=REQ), body=None)
    primary, psdk = client(daily, message("primary is back"))
    secondary, _ = client(message("fb 1"), message("fb 2"))
    clock, notes = [1000.0], []
    fb = FallbackClient(primary, secondary, on_switch=notes.append, now=lambda: clock[0])
    assert fb.complete([], []).text == "fb 1" and fb.complete([], []).text == "fb 2"
    assert len(psdk.calls) == 1 and len(notes) == 1 and "next 12 min" in notes[0]   # one switch, traced once
    clock[0] += 709
    assert fb.complete([], []).text == "primary is back"


def test_rate_limit_waits_are_reported_so_the_trace_explains_the_pause():
    """Web demo: a 15 s per-minute-limit wait left a silent gap in the trace that looked like a hang."""
    headers = {"retry-after": "15"}
    rate = openai.RateLimitError("slow down", response=httpx.Response(429, request=REQ, headers=headers), body=None)
    waits = []
    c = OpenAICompatClient(SETTINGS, Limits(), client=FakeSDK(rate, message("ok")), sleep=lambda s: None,
                           on_wait=lambda model, s, why: waits.append((model, s, why)))
    assert c.complete([], []).text == "ok" and waits == [("m", 15.0, "rate limit (tokens per minute)")]


def test_with_a_fallback_the_primary_waits_one_round_then_hands_over(monkeypatch):
    """Review: with a 150 s wait cap the primary could spend the run on per-minute 429s while the fallback idled."""
    from errand.agent.llm.openai_compat import MAX_RATE_WAIT_S, make_client
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("LLM_FALLBACK", "groq:openai/gpt-oss-20b")
    fb = make_client("groq")
    assert fb.primary.max_rate_wait_s == MAX_RATE_WAIT_S and fb.fallback.max_rate_wait_s > MAX_RATE_WAIT_S
    headers = {"retry-after": "15"}
    rate = openai.RateLimitError("slow down", response=httpx.Response(429, request=REQ, headers=headers), body=None)
    slept = []
    primary, psdk = client(rate, rate, rate, slept=slept)
    primary.max_rate_wait_s = MAX_RATE_WAIT_S
    secondary, _ = client(message("fallback answered"))
    assert FallbackClient(primary, secondary).complete([], []).text == "fallback answered"
    assert slept == [15.0] and len(psdk.calls) == 2


def test_while_the_primary_is_skipped_model_and_errors_name_the_fallback():
    msg = "Rate limit reached ... on tokens per day (TPD). Please try again in 11m48.48s."
    daily = openai.RateLimitError(msg, response=httpx.Response(429, request=REQ), body=None)
    conn = openai.APIConnectionError(request=REQ)
    primary, _ = client(daily)
    secondary, _ = client(message("fb"), conn, conn, conn, retries=2)
    secondary.model = "openai/gpt-oss-20b"
    fb = FallbackClient(primary, secondary, now=lambda: 1000.0)
    assert fb.complete([], []).text == "fb" and fb.model == "openai/gpt-oss-20b"
    with pytest.raises(LLMError) as e:
        fb.complete([], [])
    assert "skipped until its daily limit resets" in str(e.value)
