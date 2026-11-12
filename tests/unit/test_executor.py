"""Executor branches and restart recovery against the real mock service."""

import pytest

from errand.agent import approvals
from errand.agent.executor import SimulatedCrash
from errand.agent.state import transition
from errand.agent.templates import render
from errand.models import Action, ErrandKind, ErrandState as S, ExecState, ToolResult


def status_before_success(w):
    """Success must be verified: get_booking_status precedes the DONE transition."""
    names = [n for n, _ in w.calls]
    return "get_booking_status" in names and names.index("get_booking_status") > names.index("book")


def test_success_is_verified_before_reporting(world_factory):
    w = world_factory(book_outcomes=["success"])
    e, exe = w.booked_and_approved()
    out = w.executor.run(exe)
    assert out.kind == "booked" and status_before_success(w)
    assert w.store.get_errand(e.id).state == S.DONE
    assert w.bookings_for_hold(exe.payload["hold_id"]) == 1


def test_timeout_after_success_checks_status_and_never_rebooks(world_factory):
    w = world_factory(book_outcomes=["timeout_booked"])
    e, exe = w.booked_and_approved()
    out = w.executor.run(exe)
    assert out.kind == "booked"
    assert w.count("book") == 1 and w.count("get_booking_status") >= 1
    assert len(w.mock.agent_bookings()) == 1


def test_timeout_without_booking_retries_with_same_key(world_factory):
    w = world_factory(book_outcomes=["timeout_not_booked", "success"])
    e, exe = w.booked_and_approved()
    out = w.executor.run(exe)
    keys = [a["idempotency_key"] for n, a in w.calls if n == "book"]
    assert out.kind == "booked" and keys == [exe.idem_key, exe.idem_key]
    assert w.bookings_for_hold(exe.payload["hold_id"]) == 1


def test_slot_taken_closes_attempt_and_returns_to_comparing(world_factory):
    w = world_factory(book_outcomes=["slot_taken"])
    e, exe = w.booked_and_approved()
    out = w.executor.run(exe)
    assert out.kind == "slot_taken"
    assert w.store.get_errand(e.id).state == S.COMPARING
    assert w.store.get_execution(exe.id).state == ExecState.FAILED
    assert w.store.next_attempt_no(e.id, Action.BOOK) == 2


def test_hold_expired_same_terms_rebooks_under_same_approval_with_new_key(world_factory):
    w = world_factory(book_outcomes=["success"])
    e, exe = w.booked_and_approved()
    w.clock.advance(601)
    out = w.executor.run(exe)
    execs = w.store.executions_for(e.id, Action.BOOK)
    assert out.kind == "booked" and [x.attempt_no for x in execs] == [1, 2]
    assert execs[0].idem_key != execs[1].idem_key and execs[1].parent_execution_id == execs[0].id
    assert len(w.mock.agent_bookings()) == 1


def test_hold_expired_with_new_price_requires_fresh_approval(world_factory):
    w = world_factory(book_outcomes=["success"])
    e, exe = w.booked_and_approved()
    w.clock.advance(601)
    w.mock.scenario.price_drift = {"p-plb-01": 200}
    out = w.executor.run(exe)
    assert out.kind == "needs_reapproval" and out.card and "CONFIRM" in out.card["text"]
    assert out.data["new_price"] == out.data["old_price"] + 200
    assert w.store.get_errand(e.id).state == S.AWAITING_APPROVAL
    assert w.count("book") == 1 and not w.mock.agent_bookings()  # only the expired attempt, nothing booked


def test_status_stuck_pending_ends_unverified_without_retry(world_factory):
    w = world_factory(book_outcomes=["timeout_booked"], status_pending_checks=10)
    e, exe = w.booked_and_approved()
    out = w.executor.run(exe)
    assert out.kind == "unverified" and w.count("book") == 1
    assert w.store.get_errand(e.id).state == S.UNVERIFIED


def test_errand_guard_blocks_second_booking(world_factory):
    w = world_factory(book_outcomes=["success", "success"])
    e, exe = w.booked_and_approved()
    w.executor.run(exe)
    d = w.gate.check_side_effect(exe, "book", {"hold_id": exe.payload["hold_id"], "idempotency_key": exe.idem_key})
    assert not d.allow


# ---------- cancel / reschedule ----------
def cancel_execution(w, booking_id="b-plb-up"):
    e = w.errand(ErrandKind.CANCEL, booking_id=booking_id)
    r = w.call("list_bookings", {"user_id": "u1"})
    w.store.record_tool_refs(e.id, "list_bookings", r)
    transition(w.store, w.tracer, e, S.COMPARING, "test")
    a, _ = w.approve(e, Action.CANCEL)
    return e, approvals.consume(w.store, w.tracer, e, a.code).execution


def test_cancel_503_then_retry_same_key(world_factory):
    w = world_factory()
    e, exe = cancel_execution(w)
    out = w.executor.run(exe)
    keys = [a["idempotency_key"] for n, a in w.calls if n == "cancel"]
    assert out.kind == "cancelled" and keys == [exe.idem_key] * 2
    assert w.store.get_errand(e.id).state == S.DONE


def test_cancel_retries_exhausted_fails_clearly(world_factory):
    w = world_factory(cancel_always_fail=True)
    e, exe = cancel_execution(w)
    out = w.executor.run(exe)
    assert out.kind == "cancel_failed" and w.count("cancel") == 4
    assert w.store.get_errand(e.id).state == S.FAILED


def test_reschedule_conflict_then_new_slot_uses_new_key(world_factory):
    w = world_factory(conflict_slots={("b-ac-wed", "2026-11-14T14:00")})
    e = w.errand(ErrandKind.RESCHEDULE, booking_id="b-ac-wed", date_range=("2026-11-14", "2026-11-14"))
    w.store.record_tool_refs(e.id, "list_bookings", w.call("list_bookings", {"user_id": "u1"}))
    w.store.record_tool_refs(e.id, "search_providers", w.call(
        "search_providers", {"service": "ac_servicing", "area": "Indiranagar", "date_range": ["2026-11-14", "2026-11-14"]}))
    transition(w.store, w.tracer, e, S.COMPARING, "test")

    a, card = w.approve(e, Action.RESCHEDULE, slot="2026-11-14T14:00")
    assert "Wed 11 Nov, 11:00" in card["text"] and "Sat 14 Nov, 14:00" in card["text"]
    first = approvals.consume(w.store, w.tracer, e, a.code).execution
    assert w.executor.run(first).kind == "slot_conflict"

    e = w.store.get_errand(e.id)
    a2, _ = w.approve(e, Action.RESCHEDULE, slot="2026-11-14T15:00")
    second = approvals.consume(w.store, w.tracer, e, a2.code).execution
    out = w.executor.run(second)
    assert out.kind == "rescheduled" and second.attempt_no == 2 and second.idem_key != first.idem_key
    assert not any(c["key"] == second.idem_key and c["action"] != "reschedule" for c in w.mock.idem_keys())
    assert w.booking("b-ac-wed")["slot"] == "2026-11-14T15:00"


# ---------- restart recovery ----------
@pytest.mark.parametrize("outcome", ["timeout_booked", "success", "timeout_not_booked"])
def test_crash_after_book_then_restart_gives_exactly_one_booking(world_factory, outcome):
    w = world_factory(book_outcomes=[outcome, "success"])
    e, exe = w.booked_and_approved()
    w.rebuild(faults={"book_returned"})
    with pytest.raises(SimulatedCrash):
        w.executor.run(w.store.get_execution(exe.id))
    assert w.store.get_execution(exe.id).state == ExecState.EXECUTING  # write-ahead survived the crash

    w.rebuild()  # new process: fresh agent objects, same agent.db and mock.db
    outcomes = w.executor.recover_all()
    assert [o.kind for o in outcomes] == ["booked"]
    assert w.bookings_for_hold(exe.payload["hold_id"]) == 1
    names = [n for n, _ in w.calls]
    first_after_crash = names[names.index("book") + 1]
    assert first_after_crash == "get_booking_status"  # recovery consults the source of truth first


# ---------- unexpected errors never claim "nothing changed" after the side effect ----------
def test_error_after_book_was_sent_is_unverified_not_failed(world_factory, monkeypatch):
    w = world_factory(book_outcomes=["success"])
    e, exe = w.booked_and_approved()
    monkeypatch.setattr(w.executor, "_verify_book", lambda exe: 1 / 0)
    out = w.executor.run(exe)
    assert out.kind == "unverified" and w.store.get_errand(e.id).state == S.UNVERIFIED
    assert "booking went through" in render(out) and w.bookings_for_hold(exe.payload["hold_id"]) == 1
    assert w.executor.recheck_unverified(w.store.get_execution(exe.id)).kind == "booked"   # the watcher resolves it


def test_error_before_any_side_effect_fails_closed(world_factory, monkeypatch):
    w = world_factory()
    e, exe = w.booked_and_approved()
    monkeypatch.setattr(w.executor, "_side_effect", lambda exe: 1 / 0)
    assert w.executor.run(exe).kind == "failed" and w.store.get_errand(e.id).state == S.FAILED


def test_transient_book_error_checks_status_then_retries_same_key(world_factory, monkeypatch):
    w = world_factory(book_outcomes=["success"])
    e, exe = w.booked_and_approved()
    real, calls = w.mock.book, []

    def flaky(hold_id, idempotency_key):
        calls.append(idempotency_key)
        if len(calls) == 1:
            return ToolResult.fail("service_unavailable", retryable=True, message="503")
        return real(hold_id, idempotency_key)
    monkeypatch.setattr(w.mock, "book", flaky)
    assert w.executor.run(exe).kind == "booked" and calls == [exe.idem_key] * 2
    names = [n for n, _ in w.calls]
    assert names.index("get_booking_status") < len(names) - 1 - names[::-1].index("book")


def test_unverified_cancel_is_resolved_from_the_booking_list(world_factory, monkeypatch):
    w = world_factory(cancel_fail_first=False)
    e, exe = cancel_execution(w)
    monkeypatch.setattr(w.executor, "_applied", lambda exe, reflected: 1 / 0)   # breaks after cancel() is sent
    out = w.executor.run(exe)
    assert out.kind == "unverified" and "cancellation" in render(out)
    monkeypatch.undo()
    assert w.executor.recheck_unverified(w.store.get_execution(exe.id)).kind == "cancelled"
