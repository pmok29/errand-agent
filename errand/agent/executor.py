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
from datetime import datetime
from typing import Callable

from errand.agent import approvals
from errand.agent.state import transition
from errand.agent.store import Store
from errand.agent.gate import Gate
from errand.agent.keys import idem_key, payload_hash
from errand.agent.templates import Outcome
from errand.clock import Clock
from errand.config import Limits, Settings
from errand.db import new_id
from errand.models import (Action, ApprovalDraft, ErrandState as S, Execution, ExecState, Hold,
                           ToolResult, ToolTimeout)
from errand.tracing.tracer import Tracer

ToolCall = Callable[[str, dict], ToolResult]
MAX_ATTEMPTS = 3          # re-hold chain length for one approval
_RETRY = object()         # sentinel from verify: retry the same key


class SimulatedCrash(Exception):
    """Raised at a configured fault point to simulate the agent process dying."""


class Executor:
    def __init__(self, store: Store, gate: Gate, tracer: Tracer, clock: Clock, call: ToolCall,
                 settings: Settings | None = None, faults: set[str] | None = None):
        self.store, self.gate, self.tracer, self.clock, self.call = store, gate, tracer, clock, call
        self.settings = settings or Settings()
        self.limits: Limits = self.settings.limits
        self.faults = faults or set()

    # ---------- plumbing ----------
    def _fault(self, point: str) -> None:
        if point in self.faults:
            self.faults.discard(point)
            raise SimulatedCrash(point)

    def _invoke(self, exe: Execution, tool: str, args: dict) -> tuple[ToolResult | None, bool]:
        """Call a tool with tracing. Returns (result, timed_out)."""
        t0 = time.perf_counter()
        try:
            result, timed_out = self.call(tool, args), False
        except ToolTimeout:
            result, timed_out = None, True
        self.tracer.emit("tool_call", actor="executor", errand_id=exe.errand_id, execution_id=exe.id,
                         tool=tool, args=args, attempt_no=exe.attempt_no, retry=exe.retries,
                         latency_ms=round((time.perf_counter() - t0) * 1000, 2), timed_out=timed_out,
                         result=result.model_dump() if result else None)
        return result, timed_out

    def _side_effect(self, exe: Execution, tool: str, args: dict) -> tuple[ToolResult | None, bool, str | None]:
        d = self.gate.check_side_effect(exe, tool, args)
        if not d.allow:
            return None, False, d.reason
        r, timed_out = self._invoke(exe, tool, args)
        return r, timed_out, None

    def _set(self, exe: Execution, state: ExecState, **changes) -> None:
        exe.state = state
        for k, v in changes.items():
            setattr(exe, k, v)
        self.store.update_execution(exe)
        self.tracer.emit("execution_update", execution_id=exe.id, errand_id=exe.errand_id, state=state,
                         attempt_no=exe.attempt_no, retries=exe.retries, last_error=exe.last_error)

    def _errand_to(self, exe: Execution, to: S, reason: str):
        e = self.store.get_errand(exe.errand_id)
        return transition(self.store, self.tracer, e, to, reason)

    def _draft(self, exe: Execution) -> ApprovalDraft:
        return self.store.approval_by_hash(exe.errand_id, exe.approval_hash).draft

    def _fail(self, exe: Execution, reason: str, kind: str = "failed") -> Outcome:
        self._set(exe, ExecState.FAILED, last_error=reason)
        self._errand_to(exe, S.FAILED, reason)
        return Outcome(kind, exe.errand_id, {"reason": reason})

    def _backoff(self, i: int) -> float:
        b = self.limits.status_poll_backoff_s
        return b[min(i, len(b) - 1)]

    # ---------- entry points ----------
    def run(self, exe: Execution) -> Outcome:
        try:
            if exe.action == Action.BOOK:
                return self._book(exe)
            if exe.action == Action.CANCEL:
                return self._cancel(exe)
            return self._reschedule(exe)
        except SimulatedCrash:
            raise
        except Exception as err:  # unexpected tool/agent error: fail closed with a clear outcome
            self.tracer.emit("error", execution_id=exe.id, error=repr(err))
            return self._fail(exe, f"unexpected error: {type(err).__name__}")

    def resume(self, exe: Execution) -> Outcome:
        """Recovery after a restart: always consult the source of truth before calling again."""
        if exe.action == Action.BOOK:
            return self._book(exe, verify_first=True)
        if exe.action == Action.CANCEL:
            return self._cancel(exe, verify_first=True)
        return self._reschedule(exe, verify_first=True)

    # ---------- book ----------
    def _book(self, exe: Execution, verify_first: bool = False) -> Outcome:
        while True:
            if not verify_first:
                r, timed_out, denied = self._side_effect(
                    exe, "book", {"hold_id": exe.payload["hold_id"], "idempotency_key": exe.idem_key})
                if denied:
                    return self._fail(exe, f"denied: {denied}")
                self._fault("book_returned")
                if not timed_out and not r.ok:
                    return self._book_error(exe, r)
            verify_first = False
            if exe.state != ExecState.VERIFYING:
                self._set(exe, ExecState.VERIFYING)
            self._errand_to(exe, S.VERIFYING, "verify_booking")
            out = self._verify_book(exe)
            if out is not _RETRY:
                return out
            self._set(exe, ExecState.EXECUTING, retries=exe.retries + 1)
            self._errand_to(exe, S.EXECUTING, "retry_same_key")

    def _book_error(self, exe: Execution, r: ToolResult) -> Outcome:
        code = r.error.code
        if code == "slot_taken":
            self._set(exe, ExecState.FAILED, last_error=code)
            self._errand_to(exe, S.COMPARING, code)
            return Outcome("slot_taken", exe.errand_id, {"slot": exe.payload["slot"]})
        if code == "hold_expired":
            return self._rehold(exe)
        return self._fail(exe, code)

    def _verify_book(self, exe: Execution):
        polls = len(self.limits.status_poll_backoff_s) + 1
        for i in range(polls):
            s, timed_out = self._invoke(exe, "get_booking_status", {"idempotency_key": exe.idem_key})
            status = s.data.get("status") if (s and s.ok) else None
            if status == "booked":
                booking_id = s.data["booking_id"]
                self._set(exe, ExecState.DONE, result={"booking_id": booking_id})
                e = self.store.get_errand(exe.errand_id)
                e.booking_id = booking_id
                self.store.save_errand(e)
                self._errand_to(exe, S.DONE, "booking_verified")
                d = self._draft(exe)
                return Outcome("booked", exe.errand_id, {"provider_name": d.provider_name, "slot": d.slot,
                                                         "price": exe.payload["price"], "booking_id": booking_id})
            if status == "not_booked":
                if s.data.get("reason"):  # the provider processed and rejected it
                    return self._book_error(exe, ToolResult.fail(s.data["reason"]))
                hold = self.store.get_hold(exe.payload["hold_id"])
                if datetime.fromisoformat(hold.expires_at) <= self.clock.now():
                    return self._rehold(exe)
                if exe.retries >= self.limits.max_tool_retries:
                    return self._fail(exe, "booking retries exhausted")
                return _RETRY
            if i < polls - 1:  # pending, or the status call itself failed: back off and ask again
                self.clock.sleep(self._backoff(i))
        self._set(exe, ExecState.UNVERIFIED, last_error="status still pending")
        self._errand_to(exe, S.UNVERIFIED, "status_pending")
        return Outcome("unverified", exe.errand_id)

    def _rehold(self, exe: Execution) -> Outcome:
        self._set(exe, ExecState.FAILED, last_error="hold_expired")
        if exe.attempt_no >= MAX_ATTEMPTS:
            return self._fail(exe, "hold kept expiring")
        draft = self._draft(exe)
        r, _ = self._invoke(exe, "hold_slot", {"provider_id": draft.provider_id, "slot": draft.slot})
        if r is None or not r.ok:
            self._errand_to(exe, S.COMPARING, "rehold_failed")
            return Outcome("slot_unavailable", exe.errand_id, {"slot": draft.slot})
        prev = self.store.get_hold(exe.payload["hold_id"])
        self.store.add_hold(Hold(hold_id=r.data["hold_id"], errand_id=exe.errand_id, provider_id=draft.provider_id,
                                 slot=draft.slot, quoted_price=r.data["quoted_price"],
                                 search_price=prev.search_price if prev else None, expires_at=r.data["expires_at"]))
        if r.data["quoted_price"] != draft.price:
            # material change: the old approval cannot cover this; ask again with a fresh card
            self._errand_to(exe, S.AWAITING_APPROVAL, "price_changed_on_rehold")
            e = self.store.get_errand(exe.errand_id)
            new_draft = approvals.build_draft(self.store, e, Action.BOOK)
            _, card = approvals.create(self.store, self.tracer, e, new_draft, self.settings.approval_ttl_s)
            return Outcome("needs_reapproval", exe.errand_id,
                           {"old_price": draft.price, "new_price": new_draft.price}, card=card)
        # material details unchanged: continue under the same approval with a new attempt and key
        attempt = self.store.next_attempt_no(exe.errand_id, Action.BOOK)
        payload = {"hold_id": r.data["hold_id"], "slot": draft.slot, "price": r.data["quoted_price"]}
        child = Execution(id=new_id("x"), errand_id=exe.errand_id, action=Action.BOOK, attempt_no=attempt,
                          idem_key=idem_key(exe.errand_id, Action.BOOK, attempt), payload_hash=payload_hash(payload),
                          approval_hash=exe.approval_hash, parent_execution_id=exe.id, payload=payload,
                          state=ExecState.EXECUTING)
        self.store.insert_execution(child)
        self.tracer.emit("execution_rehold", parent=exe.id, child=child.id, attempt_no=attempt)
        self._errand_to(child, S.EXECUTING, "rehold_same_terms")
        return self._book(child)

    # ---------- cancel ----------
    def _booking_row(self, exe: Execution) -> dict | None:
        user = self.store.get_errand(exe.errand_id).user_id
        r, _ = self._invoke(exe, "list_bookings", {"user_id": user})
        if r is None or not r.ok:
            return None
        return next((b for b in r.data["bookings"] if b["booking_id"] == exe.payload["booking_id"]), None)

    def _cancel(self, exe: Execution, verify_first: bool = False) -> Outcome:
        draft = self._draft(exe)
        done = lambda: self._finish(exe, "cancelled", {"provider_name": draft.provider_name, "slot": draft.slot})  # noqa: E731
        if verify_first:
            row = self._booking_row(exe)
            if row and row["status"] == "cancelled":
                return done()
            self._set(exe, ExecState.EXECUTING)
            self._errand_to(exe, S.EXECUTING, "resume_cancel")
        while True:
            r, timed_out, denied = self._side_effect(
                exe, "cancel", {"booking_id": exe.payload["booking_id"], "idempotency_key": exe.idem_key})
            if denied:
                return self._fail(exe, f"denied: {denied}")
            if r is not None and r.ok:
                row = self._booking_row(exe)
                return done() if row and row["status"] == "cancelled" else self._fail(exe, "cancellation not reflected")
            if timed_out:
                row = self._booking_row(exe)
                if row and row["status"] == "cancelled":
                    return done()
            elif not r.error.retryable:
                return self._fail(exe, r.error.code)
            if exe.retries >= self.limits.max_tool_retries:
                return self._fail(exe, "cancel retries exhausted", kind="cancel_failed")
            self._set(exe, ExecState.EXECUTING, retries=exe.retries + 1, last_error="transient")
            self.clock.sleep(self._backoff(exe.retries - 1))

    # ---------- reschedule ----------
    def _reschedule(self, exe: Execution, verify_first: bool = False) -> Outcome:
        draft = self._draft(exe)
        new_slot = exe.payload["slot"]
        done = lambda: self._finish(exe, "rescheduled", {"provider_name": draft.provider_name,  # noqa: E731
                                                         "slot": new_slot, "old_slot": draft.old_slot})
        if verify_first:
            row = self._booking_row(exe)
            if row and row["slot"] == new_slot:
                return done()
            self._set(exe, ExecState.EXECUTING)
            self._errand_to(exe, S.EXECUTING, "resume_reschedule")
        while True:
            r, timed_out, denied = self._side_effect(
                exe, "reschedule", {"booking_id": exe.payload["booking_id"], "slot": new_slot,
                                    "idempotency_key": exe.idem_key})
            if denied:
                return self._fail(exe, f"denied: {denied}")
            if r is not None and r.ok:
                row = self._booking_row(exe)
                return done() if row and row["slot"] == new_slot else self._fail(exe, "reschedule not reflected")
            if timed_out:
                row = self._booking_row(exe)
                if row and row["slot"] == new_slot:
                    return done()
            elif r.error.code == "slot_conflict":
                self._set(exe, ExecState.FAILED, last_error="slot_conflict")
                self._errand_to(exe, S.COMPARING, "slot_conflict")
                return Outcome("slot_conflict", exe.errand_id, {"slot": new_slot})
            elif not r.error.retryable:
                return self._fail(exe, r.error.code)
            if exe.retries >= self.limits.max_tool_retries:
                return self._fail(exe, "reschedule retries exhausted")
            self._set(exe, ExecState.EXECUTING, retries=exe.retries + 1, last_error="transient")
            self.clock.sleep(self._backoff(exe.retries - 1))

    def _finish(self, exe: Execution, kind: str, data: dict) -> Outcome:
        self._set(exe, ExecState.DONE, result={"booking_id": exe.payload["booking_id"]})
        e = self.store.get_errand(exe.errand_id)
        if e.state != S.VERIFYING:
            e = transition(self.store, self.tracer, e, S.VERIFYING, "verify")
        transition(self.store, self.tracer, e, S.DONE, f"{kind}_verified")
        return Outcome(kind, exe.errand_id, data)
