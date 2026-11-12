"""Central configuration: limits, context budget, LLM provider presets, price table, paths."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Limits:
    max_llm_turns: int = 12          # per user message / proactive run
    max_tool_calls: int = 25
    max_wall_seconds: float = 180.0   # includes provider rate-limit waits (Groq free tier: 8k tokens/min)
    max_tool_retries: int = 3
    status_poll_backoff_s: tuple[float, ...] = (1.0, 3.0, 6.0)
    llm_timeout_s: float = 30.0
    llm_retries: int = 2


@dataclass(frozen=True)
class ContextConfig:
    budget_tokens: int = 6000
    max_memories: int = 8
    chars_per_token: float = 3.5     # deterministic offline estimate


@dataclass(frozen=True)
class ProviderPreset:
    base_url: str
    model: str
    key_env: str | None


PROVIDER_PRESETS: dict[str, ProviderPreset] = {
    # Groq alternative: LLM_MODEL=qwen/qwen3.8-27b
    "groq": ProviderPreset("https://api.groq.com/openai/v1", "openai/gpt-oss-120b", "GROQ_API_KEY"),
    # 4b, not 8b: dev machine is an M1 with 8 GB RAM
    "ollama": ProviderPreset("http://localhost:11434/v1", "qwen3:4b", None),
    "openrouter": ProviderPreset("https://openrouter.ai/api/v1", "", "OPENROUTER_API_KEY"),
}

# Estimated list prices, USD per 1M tokens (input, output). Estimates only; verify before quoting.
# Models missing from the table are traced with cost=null.
PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "openai/gpt-oss-120b": (0.15, 0.60),   # Groq list price, checked Nov 2026
    "qwen3:4b": (0.0, 0.0),
}
USD_TO_INR = 84.0
# Extra request options per model. gpt-oss reasons at "medium" by default; "low" cuts output tokens
# (the Groq free tier allows 8k tokens/minute) without hurting this tool-calling workload.
MODEL_OPTIONS: dict[str, dict] = {
    "openai/gpt-oss-120b": {"reasoning_effort": "low"},
    "openai/gpt-oss-20b": {"reasoning_effort": "low"},
}


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    base_url: str
    model: str
    api_key: str | None
    fallback: str | None
    options: dict = field(default_factory=dict)


def llm_settings(provider: str | None = None, overrides: bool = True) -> LLMSettings:
    """Settings for a provider preset, given as "groq" or "groq:<model>". LLM_BASE_URL/LLM_MODEL/LLM_API_KEY
    apply only when `overrides` (the primary provider), so a fallback keeps its own preset."""
    provider, _, pinned = (provider or os.getenv("LLM_PROVIDER", "groq")).partition(":")
    preset = PROVIDER_PRESETS[provider]
    env = (lambda name: os.getenv(name)) if overrides else (lambda name: None)
    api_key = env("LLM_API_KEY") or (os.getenv(preset.key_env) if preset.key_env else None)
    model = pinned or env("LLM_MODEL") or preset.model
    return LLMSettings(
        provider=provider,
        base_url=env("LLM_BASE_URL") or preset.base_url,
        model=model,
        api_key=api_key or ("ollama" if provider == "ollama" else None),
        fallback=os.getenv("LLM_FALLBACK", "ollama") or None,
        options=MODEL_OPTIONS.get(model, {}),
    )


@dataclass(frozen=True)
class Paths:
    agent_db: Path = ROOT / "agent.db"
    mock_db: Path = ROOT / "mock.db"
    traces: Path = ROOT / "traces"
    evals: Path = ROOT / "evals"


@dataclass(frozen=True)
class Settings:
    limits: Limits = field(default_factory=Limits)
    context: ContextConfig = field(default_factory=ContextConfig)
    paths: Paths = field(default_factory=Paths)
    approval_ttl_s: int = 600
