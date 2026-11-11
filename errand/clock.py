"""All time flows through a Clock so tests can freeze and advance it without sleeping."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Protocol

IST = timezone(timedelta(hours=5, minutes=30))


def is_expired(expires_at: str, now: datetime) -> bool:
    """Single expiry rule for holds and approvals: expired at the exact expiry instant."""
    return datetime.fromisoformat(expires_at) <= now


class Clock(Protocol):
    def now(self) -> datetime: ...
    def sleep(self, seconds: float) -> None: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(IST)

    def sleep(self, seconds: float) -> None:
        import time
        time.sleep(seconds)


class FrozenClock:
    def __init__(self, start: datetime | None = None):
        self._now = start or datetime(2026, 11, 10, 9, 0, tzinfo=IST)

    def now(self) -> datetime:
        return self._now

    def sleep(self, seconds: float) -> None:
        self.advance(seconds)

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)
