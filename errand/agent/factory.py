"""Wires an Agent to its databases, the mock provider and an LLM client."""
from __future__ import annotations

from pathlib import Path

from errand.agent.llm.base import LLMClient
from errand.agent.orchestrator import Agent
from errand.agent.store import Store
from errand.clock import Clock
from errand.config import Settings
from errand.db import DB
from errand.mocks import tools
from errand.mocks.service import MockProviderService
from errand.tracing.tracer import Tracer


def build_agent(agent_db: Path, mock: MockProviderService, llm: LLMClient, clock: Clock, tracer: Tracer,
                settings: Settings | None = None, faults: set[str] | None = None) -> Agent:
    store = Store(DB(agent_db), clock)
    return Agent(store, lambda name, args: tools.call(mock, name, args), llm, tracer, clock, settings, faults)
