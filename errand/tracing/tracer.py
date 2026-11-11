"""Run tracer. Every component emits structured events through here.

S3 keeps events in memory; S4 adds the per-run JSONL file and SSE fan-out.
"""
from __future__ import annotations

from typing import Any


class Tracer:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, type: str, **data: Any) -> None:
        self.events.append({"type": type, **data})

    def of_type(self, type: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["type"] == type]
