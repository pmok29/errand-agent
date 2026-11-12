from collections import Counter
from datetime import date

import pytest

from errand.clock import FrozenClock
from errand.mocks import tools
from errand.mocks.catalog import AREAS, SERVICES
from errand.mocks.scenario import ScenarioConfig, preset
from errand.mocks.service import MockProviderService
from errand.models import ToolTimeout
from tests.unit.common import WEEK


@pytest.fixture
def clock():
    return FrozenClock()


def make(tmp_path, clock, **cfg):
    return MockProviderService(tmp_path / "mock.db", clock, ScenarioConfig(**cfg))


def first_slot(svc, provider_id="p-plb-01"):
    res = svc.search_providers("plumber", "Indiranagar", WEEK)
    return next(p for p in res.data["providers"] if p["provider_id"] == provider_id)["slots"][0]


def held(svc, provider_id="p-plb-01"):
    return svc.hold_slot(provider_id, first_slot(svc, provider_id)).data["hold_id"]


# ---------- search / get_provider ----------
@pytest.mark.parametrize("service", list(SERVICES))
@pytest.mark.parametrize("area", AREAS)
def test_search_returns_3_to_8_providers_with_required_fields(tmp_path, clock, service, area):
    res = make(tmp_path, clock).search_providers(service, area, WEEK)
    assert res.ok and 3 <= len(res.data["providers"]) <= 8
    for p in res.data["providers"]:
        assert {"provider_id", "price", "rating", "slots", "attributes"} <= p.keys()
        assert all(s > "2026-11-10T09:00" for s in p["slots"])  # only future slots


def test_search_rejects_unknown_service(tmp_path, clock):
    assert make(tmp_path, clock).search_providers("astrologer", "Indiranagar", WEEK).error.code == "invalid_service"


def test_get_provider_has_prompt_injection_review(tmp_path, clock):
    res = make(tmp_path, clock).get_provider("p-plb-03")
    assert any("ignore all previous instructions" in r["text"] for r in res.data["reviews"])


# ---------- hold ----------
def first_slots(svc):
    return next(p for p in svc.search_providers("plumber", "Indiranagar", WEEK).data["providers"]
                if p["provider_id"] == "p-plb-01")["slots"]


def test_hold_expires_in_10_minutes_and_blocks_slot(tmp_path, clock):
    svc = make(tmp_path, clock, book_outcomes=["success"])
    slot = first_slot(svc)
    res = svc.hold_slot("p-plb-01", slot)
    assert res.ok and res.data["expires_at"] == "2026-11-10T09:10:00+05:30"
    assert slot not in first_slots(svc)                                   # blocked: gone from search
    assert svc.hold_slot("p-plb-01", slot).data == res.data               # re-holding your hold returns it
    svc.book(res.data["hold_id"], "k-1")
    assert svc.hold_slot("p-plb-01", slot).error.code == "slot_unavailable"   # booked: really taken


def test_hold_price_drift_is_configurable(tmp_path, clock):
    svc = make(tmp_path, clock, price_drift={"p-plb-01": 250})
    search_price = next(p for p in svc.search_providers("plumber", "Indiranagar", WEEK).data["providers"]
                        if p["provider_id"] == "p-plb-01")["price"]
    assert svc.hold_slot("p-plb-01", first_slot(svc)).data["quoted_price"] == search_price + 250


# ---------- book ----------
def test_book_outcome_distribution_matches_contract(tmp_path, clock):
    svc = make(tmp_path, clock, seed=7)
    outcomes = Counter(svc._next_book_outcome() for _ in range(4000))
    assert 0.66 < outcomes["success"] / 4000 < 0.74
    assert 0.12 < outcomes["slot_taken"] / 4000 < 0.18
    timeouts = outcomes["timeout_booked"] + outcomes["timeout_not_booked"]
    assert 0.12 < timeouts / 4000 < 0.18
    assert abs(outcomes["timeout_booked"] - outcomes["timeout_not_booked"]) < 0.04 * 4000


def test_book_is_reproducible_for_a_seed(tmp_path, clock):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a, b = make(tmp_path / "a", clock, seed=3), make(tmp_path / "b", clock, seed=3)
    assert [a._next_book_outcome() for _ in range(50)] == [b._next_book_outcome() for _ in range(50)]


def test_book_success_and_idempotent_replay(tmp_path, clock):
    svc = make(tmp_path, clock, book_outcomes=["success", "success"])
    h = held(svc)
    first = svc.book(h, "k1")
    assert first.ok and first.data["status"] == "confirmed"
    assert svc.book(h, "k1") == first
    assert sum(b["hold_id"] == h for b in svc.ledger()) == 1


def test_timeout_booked_creates_booking_and_status_confirms(tmp_path, clock):
    svc = make(tmp_path, clock, book_outcomes=["timeout_booked"])
    h = held(svc)
    with pytest.raises(ToolTimeout):
        svc.book(h, "k1")
    status = svc.get_booking_status(idempotency_key="k1")
    assert status.data["status"] == "booked"
    assert svc.book(h, "k1").data["booking_id"] == status.data["booking_id"]  # retry replays, no duplicate
    assert sum(b["hold_id"] == h for b in svc.ledger()) == 1


def test_timeout_not_booked_stores_nothing_so_same_key_retry_books(tmp_path, clock):
    svc = make(tmp_path, clock, book_outcomes=["timeout_not_booked", "success"])
    h = held(svc)
    with pytest.raises(ToolTimeout):
        svc.book(h, "k1")
    assert svc.get_booking_status(idempotency_key="k1").data["status"] == "not_booked"
    assert svc.book(h, "k1").ok


def test_slot_taken_is_processed_and_replayed(tmp_path, clock):
    svc = make(tmp_path, clock, book_outcomes=["slot_taken"])
    h = held(svc)
    assert svc.book(h, "k1").error.code == "slot_taken"
    assert svc.book(h, "k1").error.code == "slot_taken"  # stored, not re-rolled
    assert svc.get_booking_status(idempotency_key="k1").data["status"] == "not_booked"


def test_book_after_hold_expiry(tmp_path, clock):
    svc = make(tmp_path, clock)
    h = held(svc)
    clock.advance(601)
    assert svc.book(h, "k1").error.code == "hold_expired"


def test_status_can_report_pending(tmp_path, clock):
    svc = make(tmp_path, clock, status_pending_checks=2, book_outcomes=["timeout_booked"])
    h = held(svc)
    with pytest.raises(ToolTimeout):
        svc.book(h, "k1")
    assert [svc.get_booking_status(idempotency_key="k1").data["status"] for _ in range(3)] == \
        ["pending", "pending", "booked"]


def test_status_requires_exactly_one_identifier(tmp_path, clock):
    assert make(tmp_path, clock).get_booking_status().error.code == "invalid_arguments"


# ---------- idempotency strictness ----------
def test_key_reuse_with_different_payload_is_rejected(tmp_path, clock):
    svc = make(tmp_path, clock, book_outcomes=["success", "success"])
    svc.book(held(svc), "k1")
    other = svc.hold_slot("p-plb-02", first_slot(svc, "p-plb-02")).data["hold_id"]
    assert svc.book(other, "k1").error.code == "idempotency_payload_mismatch"


def test_key_reuse_across_actions_is_rejected(tmp_path, clock):
    svc = make(tmp_path, clock, book_outcomes=["success"], cancel_fail_first=False)
    booking_id = svc.book(held(svc), "k1").data["booking_id"]
    assert svc.cancel(booking_id, "k1").error.code == "idempotency_payload_mismatch"


# ---------- cancel / reschedule / list ----------
def test_cancel_first_call_503_then_retry_succeeds(tmp_path, clock):
    svc = make(tmp_path, clock)
    first = svc.cancel("b-plb-up", "c1")
    assert first.error.code == "service_unavailable" and first.error.retryable
    assert svc.cancel("b-plb-up", "c1").data["status"] == "cancelled"
    assert svc.cancel("b-plb-up", "c1").data["status"] == "cancelled"  # replay


def test_reschedule_conflict_is_stored_per_key_and_new_key_succeeds(tmp_path, clock):
    svc = MockProviderService(tmp_path / "mock.db", clock, preset("slot_conflict"))
    saturday = date(2026, 11, 14)
    free = svc.free_slots("p-acs-02", saturday, saturday)
    assert free[0] == "2026-11-14T14:00" and len(free) >= 2
    assert svc.reschedule("b-ac-wed", free[0], "r1").error.code == "slot_conflict"
    assert svc.reschedule("b-ac-wed", free[0], "r1").error.code == "slot_conflict"
    assert svc.reschedule("b-ac-wed", free[1], "r1").error.code == "idempotency_payload_mismatch"
    assert svc.reschedule("b-ac-wed", free[1], "r2").data["slot"] == free[1]


def test_list_bookings_and_provider_cancel_hook(tmp_path, clock):
    svc = make(tmp_path, clock)
    svc.trigger_provider_cancel("b-cln-up")
    rows = {b["booking_id"]: b for b in svc.list_bookings("u1").data["bookings"]}
    assert {"b-past-ele", "b-ac-wed", "b-plb-up"} <= rows.keys()
    assert rows["b-cln-up"]["status"] == "cancelled_by_provider"
    assert {"booking_id", "provider", "service", "slot", "status"} <= rows["b-ac-wed"].keys()


# ---------- persistence across restarts ----------
def test_state_and_forced_outcome_queue_survive_restart(tmp_path, clock):
    svc = make(tmp_path, clock, book_outcomes=["timeout_booked", "slot_taken"])
    h = held(svc)
    with pytest.raises(ToolTimeout):
        svc.book(h, "k1")
    svc.close()
    restarted = make(tmp_path, clock, book_outcomes=["timeout_booked", "slot_taken"])
    assert restarted.get_booking_status(idempotency_key="k1").data["status"] == "booked"
    assert restarted._next_book_outcome() == "slot_taken"  # queue position persisted


# ---------- dispatcher ----------
def test_dispatcher_validates_contract_arguments(tmp_path, clock):
    svc = make(tmp_path, clock)
    assert tools.call(svc, "book", {"hold_id": "h"}).error.code == "invalid_arguments"
    assert tools.call(svc, "book", {"hold_id": "h", "idempotency_key": "k", "price": 1}).error.code == "invalid_arguments"
    assert tools.call(svc, "launch_rocket", {}).error.code == "unknown_tool"
    assert tools.call(svc, "list_bookings", {"user_id": "u1"}).ok
