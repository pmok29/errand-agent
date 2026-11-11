"""MockProviderService: the simulated remote provider, with its own mock.db.

It plays the "other side" of the network: state here survives agent restarts, idempotency keys
are honoured server-side, and faults are driven by a seeded ScenarioConfig.
"""
from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

from errand.clock import Clock
from errand.mocks.catalog import AREAS, CATALOG, SEED_BOOKINGS, SERVICES, availability, days
from errand.mocks.scenario import BOOK_OUTCOMES, BOOK_WEIGHTS, ScenarioConfig
from errand.models import ToolResult, ToolTimeout

HOLD_TTL_S = 600
MAX_SEARCH_RESULTS = 8

SCHEMA = """
CREATE TABLE IF NOT EXISTS holds (
    hold_id TEXT PRIMARY KEY, provider_id TEXT NOT NULL, slot TEXT NOT NULL,
    quoted_price INTEGER NOT NULL, expires_at TEXT NOT NULL, status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS bookings (
    booking_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, provider_id TEXT NOT NULL,
    service TEXT NOT NULL, slot TEXT NOT NULL, price INTEGER NOT NULL, status TEXT NOT NULL,
    hold_id TEXT
);
CREATE TABLE IF NOT EXISTS idem (
    key TEXT PRIMARY KEY, action TEXT NOT NULL, payload_hash TEXT NOT NULL, result_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fault_counters (name TEXT PRIMARY KEY, count INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS provider_events (
    booking_id TEXT NOT NULL, kind TEXT NOT NULL, at TEXT NOT NULL
);
"""

ACTIVE_BOOKING = ("confirmed",)


def _payload_hash(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


class MockProviderService:
    def __init__(self, path: Path | str, clock: Clock, scenario: ScenarioConfig | None = None,
                 user_id: str = "u1"):
        self.clock = clock
        self.scenario = scenario or ScenarioConfig()
        self.conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        if self.conn.execute("SELECT COUNT(*) FROM bookings").fetchone()[0] == 0:
            for b in SEED_BOOKINGS:
                if b["user_id"] == user_id:
                    self._insert_booking(**b, hold_id=None)

    # ---------- helpers ----------
    def _counter(self, name: str, bump: bool = True) -> int:
        row = self.conn.execute("SELECT count FROM fault_counters WHERE name=?", (name,)).fetchone()
        n = row["count"] if row else 0
        if bump:
            self.conn.execute("INSERT INTO fault_counters VALUES (?, ?) ON CONFLICT(name) "
                              "DO UPDATE SET count=excluded.count", (name, n + 1))
        return n

    def _rng(self, tool: str) -> random.Random:
        # keyed by call index (persisted) so results don't depend on what other tools were called
        return random.Random(f"{self.scenario.seed}:{tool}:{self._counter(f'calls:{tool}')}")

    def _latency(self, tool: str) -> None:
        ms = self.scenario.latency_ms.get(tool)
        if ms:
            self.clock.sleep(ms / 1000)

    def _now_slot(self) -> str:
        return self.clock.now().strftime("%Y-%m-%dT%H:%M")

    def _insert_booking(self, booking_id, user_id, provider_id, service, slot, price, status, hold_id):
        self.conn.execute("INSERT INTO bookings VALUES (?,?,?,?,?,?,?,?)",
                          (booking_id, user_id, provider_id, service, slot, price, status, hold_id))

    def _idem_lookup(self, key: str, action: str, payload: str) -> ToolResult | None:
        row = self.conn.execute("SELECT * FROM idem WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        if self.scenario.strict_idempotency and (row["action"] != action or row["payload_hash"] != payload):
            return ToolResult.fail("idempotency_payload_mismatch",
                                   message=f"key already used for a different {row['action']} request")
        return ToolResult.model_validate_json(row["result_json"])

    def _idem_store(self, key: str, action: str, payload: str, result: ToolResult) -> ToolResult:
        # only called for requests the provider actually processed
        self.conn.execute("INSERT INTO idem VALUES (?,?,?,?)", (key, action, payload, result.model_dump_json()))
        return result

    def _taken_slots(self, provider_id: str) -> set[str]:
        now = self.clock.now().isoformat()
        booked = {r["slot"] for r in self.conn.execute(
            "SELECT slot FROM bookings WHERE provider_id=? AND status='confirmed'", (provider_id,))}
        held = {r["slot"] for r in self.conn.execute(
            "SELECT slot FROM holds WHERE provider_id=? AND status='active' AND expires_at>?",
            (provider_id, now))}
        return booked | held

    def free_slots(self, provider_id: str, start: date, end: date) -> list[str]:
        taken, now = self._taken_slots(provider_id), self._now_slot()
        return [s for d in days(start, end) for s in availability(provider_id, d)
                if s not in taken and s > now]

    def _malformed(self, tool: str) -> ToolResult | None:
        if self.scenario.malformed_tool == tool:
            return ToolResult(ok=True, data={"unexpected": "\x00garbled", "providers": "not-a-list"})
        return None

    @staticmethod
    def _parse_range(date_range) -> tuple[date, date]:
        if isinstance(date_range, dict):
            date_range = (date_range.get("start"), date_range.get("end"))
        if isinstance(date_range, str):
            date_range = date_range.split("..")
        start, end = (date.fromisoformat(str(x)) for x in date_range)
        if end < start:
            raise ValueError("date_range end before start")
        return start, end

    # ---------- contract tools ----------
    def search_providers(self, service: str, area: str, date_range) -> ToolResult:
        self._latency("search_providers")
        if bad := self._malformed("search_providers"):
            return bad
        if service not in SERVICES:
            return ToolResult.fail("invalid_service", message=f"known services: {', '.join(SERVICES)}")
        if area not in AREAS:
            return ToolResult.fail("invalid_area", message=f"known areas: {', '.join(AREAS)}")
        try:
            start, end = self._parse_range(date_range)
        except (ValueError, TypeError) as e:
            return ToolResult.fail("invalid_date_range", message=str(e))
        matches = sorted((p for p in CATALOG.values() if p.service == service and area in p.areas),
                         key=lambda p: (-p.rating, p.provider_id))[:MAX_SEARCH_RESULTS]
        return ToolResult.success(providers=[
            {"provider_id": p.provider_id, "name": p.name, "price": p.base_price, "rating": p.rating,
             "jobs_done": p.jobs_done, "slots": self.free_slots(p.provider_id, start, end)[:12],
             "attributes": p.attributes}
            for p in matches])

    def get_provider(self, provider_id: str) -> ToolResult:
        self._latency("get_provider")
        if bad := self._malformed("get_provider"):
            return bad
        p = CATALOG.get(provider_id)
        if p is None:
            return ToolResult.fail("not_found", message=f"no provider {provider_id}")
        return ToolResult.success(provider_id=p.provider_id, name=p.name, service=p.service,
                                  areas=p.areas, price=p.base_price, rating=p.rating,
                                  jobs_done=p.jobs_done, attributes=p.attributes, reviews=p.reviews)

    def hold_slot(self, provider_id: str, slot: str) -> ToolResult:
        self._latency("hold_slot")
        p = CATALOG.get(provider_id)
        if p is None:
            return ToolResult.fail("not_found", message=f"no provider {provider_id}")
        day = date.fromisoformat(slot[:10])
        if slot not in self.free_slots(provider_id, day, day):
            return ToolResult.fail("slot_unavailable", message=f"{slot} is not free for {provider_id}")
        hold_id = f"h-{uuid.uuid4().hex[:8]}"
        expires = (self.clock.now() + timedelta(seconds=HOLD_TTL_S)).isoformat()
        price = p.base_price + self.scenario.price_drift.get(provider_id, 0)
        self.conn.execute("INSERT INTO holds VALUES (?,?,?,?,?,'active')",
                          (hold_id, provider_id, slot, price, expires))
        return ToolResult.success(hold_id=hold_id, expires_at=expires, quoted_price=price)

    def _next_book_outcome(self) -> str:
        forced = self.scenario.book_outcomes or []
        idx = self._counter("book_queue_idx")
        if idx < len(forced):
            return forced[idx]
        return self._rng("book").choices(BOOK_OUTCOMES, weights=BOOK_WEIGHTS)[0]

    def book(self, hold_id: str, idempotency_key: str, user_id: str = "u1") -> ToolResult:
        self._latency("book")
        payload = _payload_hash(hold_id)
        if (replay := self._idem_lookup(idempotency_key, "book", payload)) is not None:
            return replay
        hold = self.conn.execute("SELECT * FROM holds WHERE hold_id=?", (hold_id,)).fetchone()
        if hold is None:
            return ToolResult.fail("hold_not_found")
        if hold["status"] != "active" or datetime.fromisoformat(hold["expires_at"]) <= self.clock.now():
            return self._idem_store(idempotency_key, "book", payload,
                                    ToolResult.fail("hold_expired", message="hold is no longer valid"))
        outcome = self._next_book_outcome()
        if outcome == "slot_taken":
            self.conn.execute("UPDATE holds SET status='released' WHERE hold_id=?", (hold_id,))
            return self._idem_store(idempotency_key, "book", payload,
                                    ToolResult.fail("slot_taken", message="slot was taken by another customer"))
        if outcome == "timeout_not_booked":
            raise ToolTimeout("book timed out")  # never processed: nothing stored
        booking_id = f"b-{uuid.uuid4().hex[:8]}"
        p = CATALOG[hold["provider_id"]]
        self._insert_booking(booking_id, user_id, p.provider_id, p.service, hold["slot"],
                             hold["quoted_price"], "confirmed", hold_id)
        self.conn.execute("UPDATE holds SET status='consumed' WHERE hold_id=?", (hold_id,))
        result = self._idem_store(idempotency_key, "book", payload,
                                  ToolResult.success(status="confirmed", booking_id=booking_id))
        if outcome == "timeout_booked":
            raise ToolTimeout("book timed out")  # processed, but the response was lost
        return result

    def get_booking_status(self, hold_id: str | None = None, idempotency_key: str | None = None) -> ToolResult:
        self._latency("get_booking_status")
        if (hold_id is None) == (idempotency_key is None):
            return ToolResult.fail("invalid_arguments", message="pass exactly one of hold_id or idempotency_key")
        if self._counter("status_pending") < self.scenario.status_pending_checks:
            return ToolResult.success(status="pending")
        if idempotency_key is not None:
            row = self.conn.execute("SELECT * FROM idem WHERE key=? AND action='book'",
                                    (idempotency_key,)).fetchone()
            if row is not None:
                stored = ToolResult.model_validate_json(row["result_json"])
                if stored.ok:
                    return ToolResult.success(status="booked", booking_id=stored.data["booking_id"])
                return ToolResult.success(status="not_booked", reason=stored.error.code)
            return ToolResult.success(status="not_booked")
        b = self.conn.execute("SELECT booking_id FROM bookings WHERE hold_id=?", (hold_id,)).fetchone()
        return ToolResult.success(status="booked", booking_id=b["booking_id"]) if b \
            else ToolResult.success(status="not_booked")

    def cancel(self, booking_id: str, idempotency_key: str) -> ToolResult:
        self._latency("cancel")
        payload = _payload_hash(booking_id)
        if (replay := self._idem_lookup(idempotency_key, "cancel", payload)) is not None:
            return replay
        b = self.conn.execute("SELECT * FROM bookings WHERE booking_id=?", (booking_id,)).fetchone()
        if b is None:
            return ToolResult.fail("not_found", message=f"no booking {booking_id}")
        attempts = self._counter(f"cancel:{booking_id}")
        if self.scenario.cancel_always_fail or (self.scenario.cancel_fail_first and attempts == 0):
            return ToolResult.fail("service_unavailable", retryable=True, message="503 Service Unavailable")
        if b["status"] not in ACTIVE_BOOKING:
            return self._idem_store(idempotency_key, "cancel", payload,
                                    ToolResult.fail("not_cancellable", message=f"booking is {b['status']}"))
        self.conn.execute("UPDATE bookings SET status='cancelled' WHERE booking_id=?", (booking_id,))
        return self._idem_store(idempotency_key, "cancel", payload,
                                ToolResult.success(status="cancelled", booking_id=booking_id))

    def reschedule(self, booking_id: str, slot: str, idempotency_key: str) -> ToolResult:
        self._latency("reschedule")
        payload = _payload_hash(booking_id, slot)
        if (replay := self._idem_lookup(idempotency_key, "reschedule", payload)) is not None:
            return replay
        b = self.conn.execute("SELECT * FROM bookings WHERE booking_id=?", (booking_id,)).fetchone()
        if b is None:
            return ToolResult.fail("not_found", message=f"no booking {booking_id}")
        if b["status"] not in ACTIVE_BOOKING:
            return self._idem_store(idempotency_key, "reschedule", payload,
                                    ToolResult.fail("not_reschedulable", message=f"booking is {b['status']}"))
        day = date.fromisoformat(slot[:10])
        conflict = {(booking_id, slot), (b["provider_id"], slot)} & self.scenario.conflict_slots
        if conflict or slot not in self.free_slots(b["provider_id"], day, day):
            return self._idem_store(idempotency_key, "reschedule", payload,
                                    ToolResult.fail("slot_conflict", message=f"{slot} conflicts for this provider"))
        self.conn.execute("UPDATE bookings SET slot=? WHERE booking_id=?", (slot, booking_id))
        return self._idem_store(idempotency_key, "reschedule", payload,
                                ToolResult.success(status="rescheduled", booking_id=booking_id, slot=slot))

    def list_bookings(self, user_id: str) -> ToolResult:
        self._latency("list_bookings")
        rows = self.conn.execute("SELECT * FROM bookings WHERE user_id=? ORDER BY slot", (user_id,))
        return ToolResult.success(bookings=[
            {"booking_id": r["booking_id"],
             "provider": {"provider_id": r["provider_id"], "name": CATALOG[r["provider_id"]].name},
             "service": r["service"], "slot": r["slot"], "price": r["price"], "status": r["status"]}
            for r in rows])

    # ---------- demo / test hooks (never exposed to the LLM) ----------
    def trigger_provider_cancel(self, booking_id: str) -> None:
        self.conn.execute("UPDATE bookings SET status='cancelled_by_provider' WHERE booking_id=?", (booking_id,))
        self.conn.execute("INSERT INTO provider_events VALUES (?, 'provider_cancelled', ?)",
                          (booking_id, self.clock.now().isoformat()))

    def ledger(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM bookings ORDER BY booking_id")]

    def idem_keys(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT key, action, payload_hash FROM idem")]

    def close(self) -> None:
        self.conn.close()
