"""Run tracer. Every component emits structured events through here.

Each event gets a sequence number, timestamp, run id and ms offset from run start; strings are
sanitized (phones/emails masked). Events go to memory, to traces/<session>/<run_id>.jsonl when a
traces dir is configured, and to any subscribers (the web UI's SSE stream).
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from errand.clock import Clock, SystemClock
from errand.tracing.cost import estimate

# \w lookarounds so digit runs inside IDs and hex idempotency keys are never masked
PHONE = re.compile(r"(?<![\w+-])(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?![\w-])")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


def sanitize(value: Any) -> Any:
    if isinstance(value, str):
        return EMAIL.sub("[email]", PHONE.sub("[phone]", value))
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    return value


class Tracer:
    def __init__(self, traces_dir: Path | None = None, clock: Clock | None = None):
        self.traces_dir, self.clock = traces_dir, clock or SystemClock()
        self.events: list[dict[str, Any]] = []
        self._subscribers: list[Callable[[dict], None]] = []
        self._lock = threading.Lock()
        self._seq = 0
        self.run_id: str | None = None
        self.session_id: str | None = None
        self.path: Path | None = None
        self._t0 = 0.0
        self._totals: dict[str, Any] = {}

    # ---------- runs ----------
    def start_run(self, session_id: str, kind: str = "user_message", **data: Any) -> str:
        self.run_id = f"r-{uuid.uuid4().hex[:10]}"
        self.session_id = session_id
        self._t0 = time.perf_counter()
        self._totals = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "tool_calls": 0,
                        "cost_usd": 0.0, "cost_known": True}
        self.path = None
        if self.traces_dir is not None:
            d = self.traces_dir / session_id
            d.mkdir(parents=True, exist_ok=True)
            self.path = d / f"{self.run_id}.jsonl"
        self.emit("run_start", session_id=session_id, kind=kind, **data)
        return self.run_id

    def end_run(self, outcome: str, **data: Any) -> dict[str, Any]:
        totals = dict(self._totals, wall_ms=round((time.perf_counter() - self._t0) * 1000, 1))
        self.emit("run_end", outcome=outcome, totals=totals, **data)
        self.run_id = None
        return totals

    # ---------- events ----------
    def emit(self, type: str, **data: Any) -> dict[str, Any]:
        with self._lock:
            self._seq += 1
            event = {"seq": self._seq, "type": type, "ts": self.clock.now().isoformat(),
                     "run_id": self.run_id,
                     "t_ms": round((time.perf_counter() - self._t0) * 1000, 1) if self.run_id else None,
                     **sanitize(data)}
            if type == "tool_call" and self.run_id:
                self._totals["tool_calls"] += 1
            self.events.append(event)
            if self.path is not None:
                with self.path.open("a") as f:
                    f.write(json.dumps(event, default=str, ensure_ascii=False) + "\n")
        for cb in list(self._subscribers):
            try:
                cb(event)
            except Exception:  # a broken subscriber must never break the agent
                pass
        return event

    def llm_call(self, model: str, latency_ms: float, input_tokens: int, output_tokens: int,
                 **data: Any) -> dict[str, Any]:
        cost = estimate(model, input_tokens, output_tokens)
        if self.run_id:
            t = self._totals
            t["llm_calls"] += 1
            t["input_tokens"] += input_tokens
            t["output_tokens"] += output_tokens
            if cost is None:
                t["cost_known"] = False
            else:
                t["cost_usd"] = round(t["cost_usd"] + cost["usd"], 6)
        return self.emit("llm_call", model=model, latency_ms=latency_ms, input_tokens=input_tokens,
                         output_tokens=output_tokens, cost=cost, **data)

    # ---------- consumers ----------
    def subscribe(self, callback: Callable[[dict], None]) -> Callable[[], None]:
        self._subscribers.append(callback)
        return lambda: self._subscribers.remove(callback)

    def of_type(self, type: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["type"] == type]

    def run_events(self, run_id: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["run_id"] == run_id]
