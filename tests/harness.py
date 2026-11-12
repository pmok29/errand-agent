"""Scenario harness: one simulated deployment (agent + mock provider + clock + tracer) per eval.

`Sim` drives the real orchestrator the way the web app does: user messages, confirmation codes,
watcher ticks and process restarts. Everything it observes comes from the trace, the agent DB and
the mock provider's ledger, which are also what the invariants check.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from errand.agent.factory import build_agent
from errand.agent.llm.base import LLMClient
from errand.agent.llm.scripted import ASK, ReactivePolicy
from errand.agent.watcher import Watcher
from errand.clock import FrozenClock
from errand.mocks.scenario import ScenarioConfig, preset
from errand.mocks.service import MockProviderService
from errand.seed.seed import seed_user
from errand.tracing.tracer import Tracer

PLUMBER = "Find a plumber for my kitchen tap this week, mornings only, under ₹1,500."


@dataclass
class Sim:
    dir: Path
    scenario: ScenarioConfig | str = "normal"
    llm: LLMClient | None = None
    seed: bool = False
    faults: set[str] = field(default_factory=set)
    replies: list[dict[str, Any]] = field(default_factory=list)
    notifications: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        if isinstance(self.scenario, str):
            self.scenario = preset(self.scenario)
        self.clock = FrozenClock()
        self.tracer = Tracer(None, self.clock)
        self.llm = self.llm or ReactivePolicy()
        self._boot(self.faults)
        if self.seed:
            seed_user(self.agent.store)
        self.sid = self.agent.new_session()

    def _boot(self, faults: set[str] | None) -> None:
        # a fresh process: new objects over the same agent.db and mock.db; the trace keeps going
        self.mock = MockProviderService(self.dir / "mock.db", self.clock, self.scenario)
        self.agent = build_agent(self.dir / "agent.db", self.mock, self.llm, self.clock, self.tracer,
                                 faults=set(faults or ()))
        self.watcher = Watcher(self.agent)

    # ---------- driving ----------
    def say(self, text: str, session_id: str | None = None) -> list[dict[str, Any]]:
        out = self.agent.handle_user_message(session_id or self.sid, text)
        self.replies += out
        return out

    def last_card(self) -> dict[str, Any]:
        return next(r for r in reversed(self.replies) if r["kind"] == "card")

    def confirm(self, card: dict | None = None) -> list[dict[str, Any]]:
        return self.say(f"CONFIRM {(card or self.last_card())['card']['code']}")

    def new_session(self) -> str:
        self.sid = self.agent.new_session()
        return self.sid

    def restart(self, faults: set[str] | None = None) -> list:
        self.mock.close()
        self._boot(faults)
        return self.agent.recover()

    def tick(self) -> list[dict[str, Any]]:
        notes = self.watcher.tick()
        self.notifications += notes
        return notes

    def book(self, request: str = PLUMBER, choice: str = "option 1") -> list[dict[str, Any]]:
        """Request, pick an option, confirm the card: the common path several scenarios start with."""
        out = self.say(request)
        if out[-1]["kind"] != "card":
            out = self.say(choice)
        assert out[-1]["kind"] == "card", out
        return self.confirm(out[-1])

    # ---------- observing ----------
    def events(self, type: str) -> list[dict[str, Any]]:
        return self.tracer.of_type(type)

    def tool_calls(self, tool: str | None = None, actor: str | None = None) -> list[dict[str, Any]]:
        return [e for e in self.events("tool_call")
                if (tool is None or e["tool"] == tool) and (actor is None or e["actor"] == actor)]

    def new_bookings(self) -> list[dict]:
        return self.mock.agent_bookings()

    def booking(self, booking_id: str) -> dict:
        return next(b for b in self.mock.ledger() if b["booking_id"] == booking_id)

    def errand(self, session_id: str | None = None):
        return self.agent.store.latest_errand(session_id or self.sid)

    def contexts(self) -> list[dict[str, Any]]:
        return self.events("context_sent")

    def texts(self, kind: str | None = None) -> list[str]:
        return [r["text"] for r in self.replies if kind is None or r["kind"] == kind]


ASKED_FIELD = {f: re.escape(q[:25]) for f, q in ASK.items()}


@dataclass
class SimUser:
    """Answers the agent's questions from a fixed profile, and counts what it was asked."""
    facts: dict[str, str]
    asked: Counter = field(default_factory=Counter)

    def answer(self, agent_text: str) -> str | None:
        wanted = [f for f, pattern in ASKED_FIELD.items() if re.search(pattern, agent_text)]
        self.asked.update(wanted)
        parts = [self.facts[f] for f in wanted if f in self.facts]
        return ", ".join(parts) if parts else None

    def converse(self, sim: Sim, opening: str, max_turns: int = 6) -> list[dict[str, Any]]:
        out = sim.say(opening)
        for _ in range(max_turns):
            reply = self.answer(out[-1]["text"])
            if reply is None:
                return out
            out = sim.say(reply)
        return out


def finish(sim: Sim, ev, claims_blocked: int = 0, crashes: int = 0) -> None:
    """Check I1–I7 and record the result in the eval row."""
    from tests import invariants
    broken = invariants.check(sim, claims_blocked=claims_blocked, crashes=crashes)
    note = "I1–I7 hold" if not broken else "invariants broken: " + "; ".join(f"{k}: {v[0]}" for k, v in broken.items())
    if note not in ev.notes:  # scenarios with several sub-runs report once
        ev.note(note)
    assert not broken, broken
