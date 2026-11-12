"""Provider-neutral LLM interface. Messages and tool schemas use the OpenAI chat format."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]
    parse_error: str | None = None   # set when the model emitted arguments that aren't valid JSON


@dataclass
class LLMResponse:
    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""


class LLMError(Exception):
    """retryable: transient (rate limit, outage). permanent: this provider can't work (bad key, unknown model)."""

    def __init__(self, message: str, retryable: bool = False, permanent: bool = False):
        super().__init__(message)
        self.retryable, self.permanent = retryable, permanent


class LLMClient(Protocol):
    model: str

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse: ...


def estimate_tokens(text: str, chars_per_token: float = 3.5) -> int:
    return max(1, round(len(text) / chars_per_token)) if text else 0
