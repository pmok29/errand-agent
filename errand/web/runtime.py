"""One running deployment for the web demo: agent, mock provider, watcher, tracer, and demo controls.

Everything the browser does goes through here. Chat replies return from `say`; anything the system
says on its own (watcher notifications, recovery after a restart) and every trace event go out
through `publish` to the live stream.
"""
from __future__ import annotations

import os
import re
import shutil
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

CODE_IN_TEXT = re.compile(r"CONFIRM ([A-Z2-7]{4})")
TRACE_BACKLOG = 400


def make_llm():
    """LLM_PROVIDER=scripted, or a provider whose key is missing, runs the offline policy."""
    provider = os.getenv("LLM_PROVIDER", "groq")
    if provider != "scripted":
        try:
            settings = llm_settings(provider)
            if settings.api_key:
                from errand.agent.llm.openai_compat import make_client
                return make_client(provider), f"{settings.provider}: {settings.model}"
        except KeyError:
            pass
    return ReactivePolicy(), "scripted (offline)"


class Runtime:
    def __init__(self, data_dir: Path, traces_dir: Path | None, publish: Callable[[dict], None] = lambda m: None,
                 llm_factory=make_llm):
        self.data_dir, self.traces_dir, self.publish, self.llm_factory = data_dir, traces_dir, publish, llm_factory
        self.preset, self.seeded, self.crash_next = "normal", True, False
        self.reset()

    # ---------- lifecycle ----------
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
        self.tracer.subscribe(lambda e: self.publish({"channel": "trace", "event": e}))
        self.llm, self.llm_label = self.llm_factory()
        self._boot(set())
        if self.seeded:
            seed_user(self.agent.store)
        self.session_id = self.agent.new_session()
        self.watcher.tick()                      # baseline snapshot of the user's bookings
        self.tracer.emit("demo", action="reset", preset=self.preset, seeded=self.seeded, llm=self.llm_label)

    def _boot(self, faults: set[str]) -> None:
        self.mock = MockProviderService(self.data_dir / "mock.db", self.clock, preset(self.preset))
        self.agent = build_agent(self.data_dir / "agent.db", self.mock, self.llm, self.clock, self.tracer,
                                 faults=faults)
        self.watcher = Watcher(self.agent)

    def restart(self) -> list[dict]:
        """Simulate a process restart: new objects over the same databases, then recovery."""
        self.mock.close()
        self._boot({"book_returned"} if self.crash_next else set())
        self.tracer.emit("demo", action="restart")
        self.agent.recover()
        return self.deliver()

    # ---------- chat ----------
    def say(self, text: str) -> dict[str, Any]:
        faults = self.agent.executor.faults
        if self.crash_next:
            faults.add("book_returned")
        try:
            replies = self.agent.handle_user_message(self.session_id, text)
        except SimulatedCrash:
            self.crash_next = False
            self.tracer.emit("demo", action="crashed", point="book_returned")
            return {"replies": [], "crashed": True}
        finally:
            if not self.crash_next:
                faults.discard("book_returned")
        return {"replies": replies, "crashed": False}

    def history(self) -> list[dict[str, Any]]:
        out = []
        for m in self.agent.store.session_messages(self.session_id):
            if m["role"] == "user" and m["kind"] == "event":
                continue  # system-authored notices sent to the model, not shown as chat
            msg = {"role": m["role"], "kind": m["kind"], "text": m["content"]}
            if m["kind"] == "card" and (code := CODE_IN_TEXT.search(m["content"])):
                msg["code"] = code.group(1)
            out.append(msg)
        return out

    # ---------- background work ----------
    def tick(self) -> list[dict]:
        self.watcher.tick()
        return self.deliver()

    def deliver(self) -> list[dict]:
        """Push pending notifications to the browser. Recovery outcomes are also kept in the chat history."""
        sent = []
        for n in self.agent.store.take_notifications(self.session_id):
            messages = n.get("messages") or [{"kind": n.get("kind", "template"), "text": n["text"]}]
            if "messages" not in n:
                for m in messages:
                    self.agent.store.add_message(self.session_id, "assistant", m["text"], m["kind"])
            for m in messages:
                msg = {"role": "assistant", "kind": m["kind"], "text": m["text"]}
                if m["kind"] == "card" and (code := CODE_IN_TEXT.search(m["text"])):
                    msg["code"] = code.group(1)
                self.publish({"channel": "chat", "message": msg})
                sent.append(msg)
        return sent

    # ---------- demo controls ----------
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

    def advance(self, seconds: int) -> list[dict]:
        self.clock.advance(seconds)
        self.tracer.emit("demo", action="advance_clock", seconds=seconds, now=self.clock.now().isoformat())
        return self.tick()

    def memories(self) -> list[dict]:
        rows = self.agent.memory.all("u1")
        shown = [m for m in rows if m.status != MemoryStatus.SUPERSEDED]
        return [{"key": m.key, "value": m.value if m.status != MemoryStatus.FORGOTTEN else "[forgotten]",
                 "source": m.source, "status": m.status, "scope": m.scope} for m in shown]

    def forget(self, key: str) -> int:
        with self.agent.lock:
            gone = self.agent.memory.forget("u1", key)
            self.tracer.emit("memory_forgotten", key=key, count=len(gone), by="ui")
        return len(gone)

    def bookings(self) -> list[dict]:
        return self.mock.ledger()

    def state(self) -> dict[str, Any]:
        e = self.agent.store.latest_errand(self.session_id)
        return {"session_id": self.session_id, "preset": self.preset, "presets": list(PRESETS),
                "seeded": self.seeded, "llm": self.llm_label, "crash_next": self.crash_next,
                "now": self.clock.now().strftime("%a %d %b %H:%M"),
                "errand": {"kind": e.kind, "state": e.state} if e else None}

    def backlog(self) -> list[dict]:
        return list(self.tracer.events)[-TRACE_BACKLOG:]
