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


class DemoClock:
    """Real time, but in the fixed demo week (starts Tue 10 Nov 09:00 IST, like the tests).

    Waits (backoff between status polls) jump the clock instead of blocking, and `advance` lets
    the demo skip ahead to show expiry and follow-ups."""

    def __init__(self, start: datetime | None = None):
        import time
        self._mono, self._start, self._offset = time.monotonic, start or FrozenClock().now(), 0.0
        self._t0 = self._mono()

    def now(self) -> datetime:
        return self._start + timedelta(seconds=self._mono() - self._t0 + self._offset)

    def sleep(self, seconds: float) -> None:
        self.advance(seconds)

    def advance(self, seconds: float) -> None:
        self._offset += seconds
