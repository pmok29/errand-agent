"""OpenAI-compatible chat client (Groq, Ollama, OpenRouter) with an optional fallback provider."""
from __future__ import annotations

import json
import re
import time
from typing import Any, Callable

from errand.agent.llm.base import LLMError, LLMResponse, ToolCall
from errand.config import Limits, LLMSettings, llm_settings

THINK = re.compile(r"<think>.*?</think>", re.S)  # qwen3 reasoning blocks
DURATION = re.compile(r"(?:(\d+)m)?(?:([\d.]+)s)?$")
MAX_RATE_WAIT_S = 15.0        # per wait
MAX_RATE_WAIT_TOTAL_S = 40.0  # per call; rate-limit waits don't use up the error retries
PERMANENT_STATUS = {401, 403, 404}
# Groq validates tool arguments server-side and rejects the whole response with a 400.
TOOL_VALIDATION = re.compile(r"tool call validation failed|tool_use_failed", re.I)


def rate_limit_wait(headers: Any, attempt: int) -> float:
    """How long to wait after a 429: the provider's retry-after / token-reset hint, capped."""
    for name in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        raw = (headers or {}).get(name)
        if not raw:
            continue
        try:
            return min(float(raw), MAX_RATE_WAIT_S)
        except ValueError:
            if (m := DURATION.match(raw)) and (m.group(1) or m.group(2)):
                return min(int(m.group(1) or 0) * 60 + float(m.group(2) or 0), MAX_RATE_WAIT_S)
    return float(2 ** attempt)


def parse_message(message: Any, model: str, usage: Any, latency_ms: float) -> LLMResponse:
    calls = []
    for tc in message.tool_calls or []:
        try:
            args = json.loads(tc.function.arguments or "{}")
            err = None if isinstance(args, dict) else "arguments must be a JSON object"
        except json.JSONDecodeError as e:
            args, err = {}, f"invalid JSON arguments: {e.msg}"
        calls.append(ToolCall(id=tc.id, name=tc.function.name, args=args if isinstance(args, dict) else {},
                              parse_error=err))
    text = THINK.sub("", message.content or "").strip() or None
    return LLMResponse(text=text, tool_calls=calls, model=model, latency_ms=latency_ms,
                       input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                       output_tokens=getattr(usage, "completion_tokens", 0) or 0)


class OpenAICompatClient:
    def __init__(self, settings: LLMSettings, limits: Limits | None = None, client: Any = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.settings, self.model, self.limits, self.sleep = settings, settings.model, limits or Limits(), sleep
        if client is None:
            from openai import OpenAI
            client = OpenAI(base_url=settings.base_url, api_key=settings.api_key or "missing",
                            timeout=self.limits.llm_timeout_s, max_retries=0)
        self.client = client

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        import openai
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0,
                                  **self.settings.options}
        if tools:
            kwargs |= {"tools": tools, "tool_choice": "auto"}
        failures, waited, waits, corrected = 0, 0.0, 0, False
        while True:
            t0 = time.perf_counter()
            try:
                resp = self.client.chat.completions.create(**kwargs)
            except openai.RateLimitError as e:
                wait = max(rate_limit_wait(e.response.headers, waits), 1.0)
                if waited + wait > MAX_RATE_WAIT_TOTAL_S:
                    raise LLMError(f"{type(e).__name__}: {e}", retryable=True) from e
                waited, waits = waited + wait, waits + 1
                self.sleep(wait)
                continue
            except (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError) as e:
                if failures == self.limits.llm_retries:
                    raise LLMError(f"{type(e).__name__}: {e}", retryable=True) from e
                self.sleep(2 ** failures)
                failures += 1
                continue
            except openai.APIStatusError as e:  # 4xx other than 429: auth, bad request, unknown model
                if e.status_code == 400 and TOOL_VALIDATION.search(str(e)) and not corrected:
                    # the model's mistake, not the provider's: tell it what was wrong and let it try once more
                    corrected = True
                    kwargs["messages"] = [*messages, {"role": "system", "content": (
                        f"Your previous tool call was rejected: {e.message}. Call the tool again with valid "
                        "arguments, using the exact enum values from its schema.")}]
                    continue
                raise LLMError(f"{type(e).__name__} {e.status_code}: {e}", retryable=False,
                               permanent=e.status_code in PERMANENT_STATUS) from e
            latency = round((time.perf_counter() - t0) * 1000, 1)
            return parse_message(resp.choices[0].message, self.model, resp.usage, latency)


class FallbackClient:
    """Primary provider first. A transient failure (rate limit, outage) uses the fallback for that call
    only; a permanent one (bad key, unknown model) switches to the fallback for good."""

    def __init__(self, primary: Any, fallback: Any, on_switch: Callable[[str], None] | None = None):
        self.primary, self.fallback, self.on_switch = primary, fallback, on_switch
        self.active = primary

    @property
    def model(self) -> str:
        return self.active.model

    def complete(self, messages, tools) -> LLMResponse:
        try:
            return self.active.complete(messages, tools)
        except LLMError as e:
            if self.active is self.fallback:
                raise
            if e.permanent:
                self.active = self.fallback
            if self.on_switch:
                scope = "switched" if self.active is self.fallback else "this call only"
                self.on_switch(f"{self.primary.model} failed ({e}); using {self.fallback.model} ({scope})")
            try:
                return self.fallback.complete(messages, tools)
            except LLMError as fb:
                raise LLMError(f"{e}; fallback {self.fallback.model}: {fb}", retryable=e.retryable) from fb


def make_client(provider: str | None = None, limits: Limits | None = None,
                on_switch: Callable[[str], None] | None = None):
    primary_settings = llm_settings(provider)
    primary = OpenAICompatClient(primary_settings, limits)
    fb = primary_settings.fallback  # "ollama", or e.g. "groq:qwen/qwen3.8-27b" (its own rate-limit bucket)
    if not fb or llm_settings(fb, overrides=False).model == primary_settings.model:
        return primary
    return FallbackClient(primary, OpenAICompatClient(llm_settings(fb, overrides=False), limits), on_switch)
