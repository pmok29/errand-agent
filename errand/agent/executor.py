"""Runs book / cancel / reschedule under a write-ahead execution record, verifies, and retries safely.

Rules:
- every side-effect call passes gate.check_side_effect against the persisted record;
- an unprocessed failure (timeout, 503, status not_booked) retries with the SAME key;
- a processed failure (slot_taken, slot_conflict, hold_expired) closes the record; any further
  attempt gets attempt_no+1 and therefore a new key;
- success is only reported after get_booking_status / list_bookings confirms it.
"""
from __future__ import annotations

import time
from typing import Any, Callable

from errand.agent import approvals
from errand.agent.gate import Gate
from errand.agent.keys import side_effect_args
from errand.agent.state import transition
from errand.agent.store import Store
from errand.agent.templates import Outcome
from errand.clock import Clock, is_expired
from errand.config import Limits, Settings
from errand.models import (Action, ApprovalDraft, ErrandState as S, Execution, ExecState, ToolResult,
                           ToolTimeout)
from errand.tracing.tracer import Tracer

ToolCall = Callable[[str, dict], ToolResult]
MAX_ATTEMPTS = 3          # re-hold chain length for one approval


class SimulatedCrash(BaseException):
    """Raised at a configured fault point to simulate the agent process dying.

    A BaseException (like SystemExit) so that no `except Exception` safety net can swallow it."""


class Executor:
    def __init__(self, store: Store, gate: Gate, tracer: Tracer, clock: Clock, call: ToolCall,
                 settings: Settings | None = None, faults: set[str] | None = None):
        self.store, self.gate, self.tracer, self.clock, self.call = store, gate, tracer, clock, call
        self.settings = settings or Settings()
        self.limits: Limits = self.settings.limits
        self.faults = faults if faults is not None else set()  # may be shared with the caller
        self._handlers = {Action.BOOK: self._book, Action.CANCEL: self._cancel,
                          Action.RESCHEDULE: self._reschedule}

    # ---------- entry points ----------
    def run(self, exe: Execution) -> Outcome:
        try:
            return self._handlers[exe.action](exe)
        except Exception as err:  # unexpected tool/agent error: fail closed with a clear outcome
            self.tracer.emit("error", execution_id=exe.id, error=repr(err))
            return self._fail(exe, f"unexpected error: {type(err).__name__}")

    def resume(self, exe: Execution) -> Outcome:
        """Recovery after a restart: always consult the source of truth before calling again."""
        return self._handlers[exe.action](exe, verify_first=True)

    def recheck_unverified(self, exe: Execution) -> Outcome | None:
        """Watcher follow-up for an UNVERIFIED booking. None while the provider still says pending."""
        s, status = self._status(exe)
        if status == "booked":
            return self._booked(exe, s.data["booking_id"])
        if status == "not_booked":
            return self._fail(exe, "the provider confirmed the booking did not go through")
        return None

    # ---------- plumbing ----------
    def _fault(self, point: str) -> None:
        if point in self.faults:
            self.faults.discard(point)
            raise SimulatedCrash(point)

    def _invoke(self, exe: Execution, tool: str, args: dict) -> ToolResult | None:
        """Call a tool with tracing. None means the call timed out."""
        t0 = time.perf_counter()
        try:
            result = self.call(tool, args)
        except ToolTimeout:
            result = None
        self.tracer.emit("tool_call", actor="executor", errand_id=exe.errand_id, execution_id=exe.id,
                         tool=tool, args=args, attempt_no=exe.attempt_no, retry=exe.retries,
                         latency_ms=round((time.perf_counter() - t0) * 1000, 2), timed_out=result is None,
                         result=result.model_dump() if result else None)
        return result

    def _side_effect(self, exe: Execution) -> tuple[ToolResult | None, str | None]:
        """Run the execution's side effect through the gate. Returns (result or None on timeout, denial)."""
        args = side_effect_args(exe.action, exe.payload, exe.idem_key)
        d = self.gate.check_side_effect(exe, exe.action, args)
        if not d.allow:
            return None, d.reason
        return self._invoke(exe, exe.action, args), None

    def _move(self, exe: Execution, exec_state: ExecState, errand_state: S | None, reason: str,
              booking_id: str | None = None, **changes: Any) -> None:
        """Update the execution and its errand together, atomically."""
        with self.store.db.tx():
            exe.state = exec_state
            for k, v in changes.items():
                setattr(exe, k, v)
            self.store.update_execution(exe)
            if errand_state is not None:
                e = self.store.get_errand(exe.errand_id)
                if booking_id:
                    e.booking_id = booking_id
                    if e.state == errand_state:
                        self.store.save_errand(e)
                transition(self.store, self.tracer, e, errand_state, reason)
        self.tracer.emit("execution_update", execution_id=exe.id, errand_id=exe.errand_id, state=exec_state,
                         attempt_no=exe.attempt_no, retries=exe.retries, last_error=exe.last_error)

    def _draft(self, exe: Execution) -> ApprovalDraft:
        return self.store.approval_by_hash(exe.errand_id, exe.approval_hash).draft

    def _fail(self, exe: Execution, reason: str, kind: str = "failed") -> Outcome:
        self._move(exe, ExecState.FAILED, S.FAILED, reason, last_error=reason)
        return Outcome(kind, exe.errand_id, {"reason": reason})

    def _back_to_compare(self, exe: Execution, code: str, data: dict) -> Outcome:
        """A processed failure: close this attempt and let the user pick something else."""
        self._move(exe, ExecState.FAILED, S.COMPARING, code, last_error=code)
        return Outcome(code, exe.errand_id, data)

    def _done(self, exe: Execution, kind: str, data: dict, booking_id: str) -> Outcome:
        """Verified success. The outcome is saved so later status messages re-render it verbatim."""
        self._move(exe, ExecState.DONE, S.DONE, f"{kind}_verified", booking_id=booking_id,
                   result={"booking_id": booking_id, "outcome": {"kind": kind, "data": data}})
        return Outcome(kind, exe.errand_id, data)

    def _backoff(self, i: int) -> float:
        b = self.limits.status_poll_backoff_s
        return b[min(i, len(b) - 1)]

    # ---------- book ----------
    def _book(self, exe: Execution, verify_first: bool = False) -> Outcome:
        while True:
            if not verify_first:
                r, denied = self._side_effect(exe)
                if denied:
                    return self._fail(exe, f"denied: {denied}")
                self._fault("book_returned")
                if r is not None and not r.ok:
                    return self._book_error(exe, r)
            verify_first = False
            self._move(exe, ExecState.VERIFYING, S.VERIFYING, "verify_booking")
            out = self._verify_book(exe)
            if out is not None:
                return out
            self._move(exe, ExecState.EXECUTING, S.EXECUTING, "retry_same_key", retries=exe.retries + 1)

    def _book_error(self, exe: Execution, r: ToolResult) -> Outcome:
        code = r.error.code
        if code == "slot_taken":
            return self._back_to_compare(exe, code, {"slot": exe.payload["slot"]})
        if code == "hold_expired":
            return self._rehold(exe)
        return self._fail(exe, code)

    def _status(self, exe: Execution) -> tuple[ToolResult | None, str | None]:
        """The provider's view of this execution's key: (result, booked | not_booked | pending | None)."""
        s = self._invoke(exe, "get_booking_status", {"idempotency_key": exe.idem_key})
        return s, s.data.get("status") if (s and s.ok) else None

    def _booked(self, exe: Execution, booking_id: str) -> Outcome:
        d = self._draft(exe)
        return self._done(exe, "booked", {"provider_name": d.provider_name, "slot": d.slot,
                                          "price": exe.payload["price"], "booking_id": booking_id}, booking_id)

    def _verify_book(self, exe: Execution) -> Outcome | None:
        """Ask the source of truth. Returns an Outcome, or None to retry with the same key."""
        polls = len(self.limits.status_poll_backoff_s) + 1
        for i in range(polls):
            s, status = self._status(exe)
            if status == "booked":
                return self._booked(exe, s.data["booking_id"])
            if status == "not_booked":
                if s.data.get("reason"):  # the provider processed and rejected it
                    return self._book_error(exe, ToolResult.fail(s.data["reason"]))
                if is_expired(self.store.get_hold(exe.payload["hold_id"]).expires_at, self.clock.now()):
                    return self._rehold(exe)
                if exe.retries >= self.limits.max_tool_retries:
                    return self._fail(exe, "booking retries exhausted")
                return None
            if i < polls - 1:  # pending, or the status call itself failed: back off and ask again
                self.clock.sleep(self._backoff(i))
        self._move(exe, ExecState.UNVERIFIED, S.UNVERIFIED, "status_pending", last_error="status still pending")
        return Outcome("unverified", exe.errand_id)

    def _rehold(self, exe: Execution) -> Outcome:
        self._move(exe, ExecState.FAILED, None, "hold_expired", last_error="hold_expired")
        if exe.attempt_no >= MAX_ATTEMPTS:
            return self._fail(exe, "hold kept expiring")
        draft = self._draft(exe)
        r = self._invoke(exe, "hold_slot", {"provider_id": draft.provider_id, "slot": draft.slot})
        if r is None or not r.ok:
            return self._back_to_compare(exe, "slot_unavailable", {"slot": draft.slot})
        prev = self.store.get_hold(exe.payload["hold_id"])
        hold = self.store.record_hold(exe.errand_id, draft.provider_id, draft.slot, r,
                                      prev.search_price if prev else None)
        if hold.quoted_price != draft.price:
            # material change: the old approval cannot cover this; ask again with a fresh card
            self._move(exe, ExecState.FAILED, S.AWAITING_APPROVAL, "price_changed_on_rehold")
            e = self.store.get_errand(exe.errand_id)
            new_draft = approvals.build_draft(self.store, e, Action.BOOK)
            _, card = approvals.create(self.store, self.tracer, e, new_draft, self.settings.approval_ttl_s)
            return Outcome("needs_reapproval", exe.errand_id,
                           {"old_price": draft.price, "new_price": new_draft.price}, card=card)
        # material details unchanged: continue under the same approval with a new attempt and key
        child = self.store.open_execution(exe.errand_id, Action.BOOK, approvals.book_payload(hold),
                                          exe.approval_hash, parent_execution_id=exe.id)
        self.tracer.emit("execution_rehold", parent=exe.id, child=child.id, attempt_no=child.attempt_no)
        self._move(child, ExecState.EXECUTING, S.EXECUTING, "rehold_same_terms")
        return self._book(child)

    # ---------- cancel / reschedule ----------
    def _booking_row(self, exe: Execution) -> dict | None:
        user = self.store.get_errand(exe.errand_id).user_id
        r = self._invoke(exe, "list_bookings", {"user_id": user})
        if r is None or not r.ok:
            return None
        return next((b for b in r.data["bookings"] if b["booking_id"] == exe.payload["booking_id"]), None)

    def _applied(self, exe: Execution, reflected: Callable[[dict], bool]) -> bool:
        row = self._booking_row(exe)
        return row is not None and reflected(row)

    def _cancel(self, exe: Execution, verify_first: bool = False) -> Outcome:
        d = self._draft(exe)
        return self._mutate_booking(exe, verify_first, lambda row: row["status"] == "cancelled",
                                    "cancelled", {"provider_name": d.provider_name, "slot": d.slot},
                                    exhausted_kind="cancel_failed")

    def _reschedule(self, exe: Execution, verify_first: bool = False) -> Outcome:
        d, new_slot = self._draft(exe), exe.payload["slot"]
        return self._mutate_booking(exe, verify_first, lambda row: row["slot"] == new_slot,
                                    "rescheduled", {"provider_name": d.provider_name, "slot": new_slot,
                                                    "old_slot": d.old_slot},
                                    conflict_code="slot_conflict")

    def _mutate_booking(self, exe: Execution, verify_first: bool, reflected: Callable[[dict], bool],
                        kind: str, data: dict, conflict_code: str | None = None,
                        exhausted_kind: str = "failed") -> Outcome:
        """Shared cancel/reschedule loop: side effect, verify via list_bookings, same-key retries."""
        if verify_first:
            if self._applied(exe, reflected):
                self._move(exe, ExecState.VERIFYING, S.VERIFYING, f"verify_{exe.action}")
                return self._done(exe, kind, data, exe.payload["booking_id"])
            self._move(exe, ExecState.EXECUTING, S.EXECUTING, f"resume_{exe.action}")
        while True:
            r, denied = self._side_effect(exe)
            if denied:
                return self._fail(exe, f"denied: {denied}")
            if r is None or r.ok:  # success or timeout: the booking list is the source of truth
                self._move(exe, ExecState.VERIFYING, S.VERIFYING, f"verify_{exe.action}")
                if self._applied(exe, reflected):
                    return self._done(exe, kind, data, exe.payload["booking_id"])
                if r is not None:
                    return self._fail(exe, f"{exe.action} not reflected")
            elif r.error.code == conflict_code:
                return self._back_to_compare(exe, conflict_code, {"slot": exe.payload.get("slot")})
            elif not r.error.retryable:
                return self._fail(exe, r.error.code)
            if exe.retries >= self.limits.max_tool_retries:
                return self._fail(exe, f"{exe.action} retries exhausted", kind=exhausted_kind)
            self._move(exe, ExecState.EXECUTING, S.EXECUTING, "retry_same_key",
                       retries=exe.retries + 1, last_error="transient")
            self.clock.sleep(self._backoff(exe.retries - 1))
