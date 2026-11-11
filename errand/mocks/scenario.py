"""Configurable, seeded scenarios so every failure branch can be forced reproducibly."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

BOOK_OUTCOMES = ("success", "slot_taken", "timeout_booked", "timeout_not_booked")
# 70 / 15 / 15 split, with half of the timeouts actually booked
BOOK_WEIGHTS = (0.70, 0.15, 0.075, 0.075)


@dataclass
class ScenarioConfig:
    seed: int = 42
    # Forced book outcomes consumed in order; when exhausted (or None), draw from BOOK_WEIGHTS.
    book_outcomes: list[str] | None = None
    # provider_id -> rupees added to the quoted price at hold time
    price_drift: dict[str, int] = field(default_factory=dict)
    # (booking_id or provider_id, slot) pairs that return slot_conflict on reschedule
    conflict_slots: set[tuple[str, str]] = field(default_factory=set)
    cancel_fail_first: bool = True
    cancel_always_fail: bool = False
    # get_booking_status answers "pending" this many times before telling the truth
    status_pending_checks: int = 0
    # Reusing an idempotency key with a different payload returns idempotency_payload_mismatch.
    # Test-harness addition beyond the fixed contract; on by default.
    strict_idempotency: bool = True
    malformed_tool: str | None = None
    latency_ms: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for o in self.book_outcomes or []:
            if o not in BOOK_OUTCOMES:
                raise ValueError(f"unknown book outcome {o!r}")


def preset(name: str) -> ScenarioConfig:
    """Fresh copy of a preset, so callers never mutate the shared sets/dicts inside PRESETS."""
    return copy.deepcopy(PRESETS[name])


PRESETS: dict[str, ScenarioConfig] = {
    "normal": ScenarioConfig(),
    "price_drift": ScenarioConfig(price_drift={"p-plb-03": 300, "p-plb-01": 200, "p-plb-02": 250}),
    "timeout_booked": ScenarioConfig(book_outcomes=["timeout_booked"]),
    "injection_first": ScenarioConfig(book_outcomes=["success"]),
    # p-acs-02 is free Sat 14:00/15:00/18:00; 14:00 conflicts so the retry lands on another slot
    "slot_conflict": ScenarioConfig(conflict_slots={("b-ac-wed", "2026-11-14T14:00")}),
}
