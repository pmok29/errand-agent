"""OpenAI-compatible chat client (Groq, Ollama, OpenRouter) with an optional fallback provider."""
from __future__ import annotations

import json
import re
import time
from typing import Any, Callable

from errand.agent.llm.base import LLMError, LLMResponse, ToolCall
from errand.config import Limits, LLMSettings, llm_settings

THINK = re.compile(r"<think>.*?</think>", re.S)  # qwen3 reasoning blocks


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
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0}
        if tools:
            kwargs |= {"tools": tools, "tool_choice": "auto"}
        for attempt in range(self.limits.llm_retries + 1):
            t0 = time.perf_counter()
            try:
                resp = self.client.chat.completions.create(**kwargs)
            except (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError,
                    openai.InternalServerError) as e:
                if attempt == self.limits.llm_retries:
                    raise LLMError(f"{type(e).__name__}: {e}", retryable=True) from e
                self.sleep(2 ** attempt)
                continue
            except openai.APIStatusError as e:  # 4xx other than 429: auth, bad request, unknown model
                raise LLMError(f"{type(e).__name__} {e.status_code}: {e}", retryable=False) from e
            latency = round((time.perf_counter() - t0) * 1000, 1)
            return parse_message(resp.choices[0].message, self.model, resp.usage, latency)
        raise AssertionError("unreachable")


class FallbackClient:
    """Uses the primary provider; on an LLMError switches permanently to the fallback."""

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
            self.active = self.fallback
            if self.on_switch:
                self.on_switch(f"{self.primary.model} failed ({e}); switched to {self.fallback.model}")
            return self.active.complete(messages, tools)


def make_client(provider: str | None = None, limits: Limits | None = None):
    primary_settings = llm_settings(provider)
    primary = OpenAICompatClient(primary_settings, limits)
    fb = primary_settings.fallback
    if not fb or fb == primary_settings.provider:
        return primary
    return FallbackClient(primary, OpenAICompatClient(llm_settings(fb), limits))
