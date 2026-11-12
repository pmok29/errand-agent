"""One running deployment for the web demo: agent, mock provider, watcher, tracer, and demo controls.

Everything the browser does goes through here. Chat replies return from `say`; anything the system
says on its own (watcher notifications, recovery after a restart) and every trace event go out
through `publish` to the live stream.
"""
from __future__ import annotations

import functools
import os
import shutil
import threading
from pathlib import Path
from typing import Any, Callable

from errand.agent.executor import SimulatedCrash
from errand.agent.factory import build_agent
from errand.agent.llm.scripted import ReactivePolicy
from errand.agent.watcher import Watcher
from errand.clock import DemoClock
from errand.config import llm_settings
from errand.mocks.scenario import PRESETS, preset
from errand.mocks.service import MockProviderService
from errand.models import MemoryStatus
from errand.seed.seed import seed_user
from errand.tracing.tracer import Tracer

TRACE_BACKLOG = 400


def serialized(method):
    """Run under the runtime's lock: reset/restart swap the agent and its databases, so nothing else may be
    using them meanwhile (request threads and the watcher loop call in concurrently)."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)
    return wrapper
CRASH_POINT = "book_returned"


def for_browser(e: dict) -> dict:
    """Trace events as the UI gets them: context_sent without the full prompt (it stays in the JSONL)."""
    return {k: v for k, v in e.items() if k != "messages"} if e["type"] == "context_sent" else e


class TracePublisher:
    """Tracer subscriber that holds only the publish callable, not the whole Runtime."""

    def __init__(self, publish: Callable[[dict], None]):
        self.publish = publish

    def __call__(self, e: dict) -> None:
        self.publish({"channel": "trace", "event": for_browser(e)})


def ui_message(kind: str, text: str, card: dict | None = None, role: str = "assistant") -> dict:
    return {"role": role, "kind": kind, "text": text, **({"card": card} if card else {})}


def make_llm(on_switch: Callable[[str], None] | None = None):
    """LLM_PROVIDER=scripted, or a provider whose key is missing, runs the offline policy."""
    provider = os.getenv("LLM_PROVIDER", "groq")
    if provider != "scripted":
        try:
            settings = llm_settings(provider)
            if settings.api_key:
                from errand.agent.llm.openai_compat import make_client
                return make_client(provider, on_switch=on_switch), f"{settings.provider}: {settings.model}"
        except KeyError:
            pass
    return ReactivePolicy(), "scripted (offline)"


class Runtime:
    def __init__(self, data_dir: Path, traces_dir: Path | None, publish: Callable[[dict], None] = lambda m: None,
                 llm_factory=make_llm):
        self.data_dir, self.traces_dir, self.publish, self.llm_factory = data_dir, traces_dir, publish, llm_factory
        self.lock = threading.RLock()
        self.preset, self.seeded = "normal", True
        self.faults: set[str] = set()   # shared with every executor we boot; firing a fault removes it
        self.reset()

    @property
    def crash_next(self) -> bool:
        return CRASH_POINT in self.faults

    @serialized
    def arm_crash(self, armed: bool) -> None:
        (self.faults.add if armed else self.faults.discard)(CRASH_POINT)

    # ---------- lifecycle ----------
    @serialized
    def reset(self, preset_name: str | None = None, seed: bool | None = None) -> None:
        if preset_name is not None:
            if preset_name not in PRESETS:
                raise ValueError(f"unknown preset {preset_name}")
            self.preset = preset_name
        if seed is not None:
            self.seeded = seed
        shutil.rmtree(self.data_dir, ignore_errors=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.clock = DemoClock()
        self.tracer = Tracer(self.traces_dir, self.clock)
        self.tracer.subscribe(TracePublisher(self.publish))
        self.llm, self.llm_label = self.llm_factory(self._llm_fallback)
        self._boot()
        if self.seeded:
            seed_user(self.agent.store)
        self.session_id = self.agent.new_session()
        self.watcher.tick()                      # baseline snapshot of the user's bookings
        self.tracer.emit("demo", action="reset", preset=self.preset, seeded=self.seeded, llm=self.llm_label)

    def _llm_fallback(self, message: str) -> None:
        self.tracer.emit("llm_fallback", message=message)

    def _boot(self) -> None:
        self.mock = MockProviderService(self.data_dir / "mock.db", self.clock, preset(self.preset))
        self.agent = build_agent(self.data_dir / "agent.db", self.mock, self.llm, self.clock, self.tracer,
                                 faults=self.faults)
        self.watcher = Watcher(self.agent)

    @serialized
    def restart(self) -> list[dict]:
        """Simulate a process restart: new objects over the same databases, then recovery."""
        self.mock.close()
        self._boot()
        self.tracer.emit("demo", action="restart")
        self.agent.recover()
        return self.deliver()

    # ---------- chat ----------
    @serialized
    def say(self, text: str) -> dict[str, Any]:
        try:
            replies = self.agent.handle_user_message(self.session_id, text)
        except SimulatedCrash:
            self.tracer.emit("demo", action="crashed", point=CRASH_POINT)
            return {"replies": [], "crashed": True}
        return {"replies": replies, "crashed": False}

    @serialized
    def history(self) -> list[dict[str, Any]]:
        return [ui_message(m["kind"], m["content"], m["card"], m["role"])
                for m in self.agent.store.session_messages(self.session_id)
                if not (m["role"] == "user" and m["kind"] == "event")]  # notices sent to the model, not chat

    # ---------- background work ----------
    @serialized
    def tick(self) -> list[dict]:
        self.watcher.tick()
        return self.deliver()

    def deliver(self) -> list[dict]:
        """Push system-initiated messages (watcher, recovery) to the browser; they are already in history."""
        sent = [ui_message(m["kind"], m["text"], m.get("card"))
                for n in self.agent.store.take_notifications(self.session_id) for m in n["messages"]]
        for msg in sent:
            self.publish({"channel": "chat", "message": msg})
        return sent

    # ---------- demo controls ----------
    @serialized
    def provider_cancel(self, booking_id: str | None = None) -> dict:
        with self.agent.lock:
            self.tick()
            now = self.clock.now().isoformat()[:16]
            upcoming = [b for b in self.mock.ledger()
                        if b["status"] == "confirmed" and b["slot"] > now and b["user_id"] == "u1"]
            if booking_id:
                upcoming = [b for b in upcoming if b["booking_id"] == booking_id]
            if not upcoming:
                raise ValueError("no upcoming confirmed booking to cancel")
            b = max(upcoming, key=lambda b: (b["hold_id"] is not None, b["slot"]))  # prefer the agent's own
            self.mock.trigger_provider_cancel(b["booking_id"])
            self.tracer.emit("demo", action="provider_cancel", booking_id=b["booking_id"])
            self.tick()
        return b

    @serialized
    def advance(self, seconds: int) -> list[dict]:
        self.clock.advance(seconds)
        self.tracer.emit("demo", action="advance_clock", seconds=seconds, now=self.clock.now().isoformat())
        return self.tick()

    @serialized
    def memories(self) -> list[dict]:
        rows = self.agent.memory.all("u1")
        shown = [m for m in rows if m.status != MemoryStatus.SUPERSEDED]
        return [{"key": m.key, "value": m.value if m.status != MemoryStatus.FORGOTTEN else "[forgotten]",
                 "source": m.source, "status": m.status, "scope": m.scope} for m in shown]

    @serialized
    def forget(self, key: str) -> int:
        with self.agent.lock:
            gone = self.agent.memory.forget("u1", key)
            self.tracer.emit("memory_forgotten", key=key, count=len(gone), by="ui")
        return len(gone)

    @serialized
    def bookings(self) -> list[dict]:
        return self.mock.ledger()

    @serialized
    def state(self) -> dict[str, Any]:
        e = self.agent.store.latest_errand(self.session_id)
        return {"session_id": self.session_id, "preset": self.preset, "presets": list(PRESETS),
                "seeded": self.seeded, "llm": self.llm_label, "crash_next": self.crash_next,
                "now": self.clock.now().strftime("%a %d %b %H:%M"),
                "errand": {"kind": e.kind, "state": e.state} if e else None}

    @serialized
    def backlog(self) -> list[dict]:
        return [for_browser(e) for e in list(self.tracer.events)[-TRACE_BACKLOG:]]
