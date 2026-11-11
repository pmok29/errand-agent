"""Proactive work with no user prompt: provider cancellations, UNVERIFIED follow-ups, approval expiry.

tick() is idempotent and cheap; the web app calls it every few seconds and tests call it directly
with a frozen clock.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from errand.agent.approvals import fmt_slot
from errand.agent.orchestrator import Agent
from errand.agent.state import transition
from errand.agent.templates import Outcome, render
from errand.clock import is_expired
from errand.models import (ApprovalStatus, ErrandKind, ErrandState as S, ExecState, Request, ToolResult,
                           ToolTimeout)

UNVERIFIED_GAPS_S = (30, 90, 480)        # waits before each re-check: at +30s, +2m, +10m
REPLACEMENT_WINDOW_DAYS = 2              # look this many days past the cancelled slot's date
FALLBACK_BUDGET_MARGIN = 1.2             # no stored budget: allow up to 20% above the original price


class Watcher:
    def __init__(self, agent: Agent):
        self.agent = agent
        self.store, self.tracer, self.clock = agent.store, agent.tracer, agent.clock

    def tick(self) -> list[dict[str, Any]]:
        """Run all checks once. Returns the notifications produced: [{session_id, messages}]."""
        with self.agent.lock:
            self._expire_approvals()
            return self._recheck_unverified() + self._provider_cancellations()

    # ---------- helpers ----------
    def _notify(self, session_id: str, messages: list[dict]) -> dict[str, Any]:
        self.store.add_notification(session_id, {"messages": messages})
        return {"session_id": session_id, "messages": messages}

    def _say(self, session_id: str, text: str, kind: str = "notification") -> dict:
        return self.agent._reply(session_id, kind, text)

    def _list_bookings(self, user_id: str) -> ToolResult | None:
        try:
            r = self.agent.call("list_bookings", {"user_id": user_id})
        except ToolTimeout:
            r = None
        self.tracer.emit("tool_call", actor="watcher", tool="list_bookings", args={"user_id": user_id},
                         ok=bool(r and r.ok))
        return r if r and r.ok else None

    # ---------- approvals ----------
    def _expire_approvals(self) -> None:
        now = self.clock.now()
        for a in self.store.pending_approvals():
            if is_expired(a.expires_at, now):
                self.store.set_approval_status(a.id, ApprovalStatus.EXPIRED)
                self.tracer.emit("approval_expired", errand_id=a.errand_id, approval_id=a.id, by="watcher")
                e = self.store.get_errand(a.errand_id)
                if e.state == S.AWAITING_APPROVAL:
                    transition(self.store, self.tracer, e, S.COMPARING, "approval_expired")

    # ---------- UNVERIFIED follow-up ----------
    def _recheck_unverified(self) -> list[dict[str, Any]]:
        now, out = self.clock.now(), []
        for exe in self.store.executions_in(ExecState.UNVERIFIED):
            job = self.store.job("unverified", exe.errand_id) or self.store.add_job(
                "unverified", exe.errand_id, (now + timedelta(seconds=UNVERIFIED_GAPS_S[0])).isoformat(),
                len(UNVERIFIED_GAPS_S))
            if job["status"] != "open" or now < datetime.fromisoformat(job["next_run_at"]):
                continue
            e = self.store.get_errand(exe.errand_id)
            self.tracer.start_run(e.session_id, kind="watcher", job="unverified_recheck", attempt=job["attempts"] + 1)
            outcome = self.agent.executor.recheck_unverified(exe)
            attempts = job["attempts"] + 1
            if outcome is not None:
                self.store.update_job(job["id"], attempts=attempts, status="done")
                messages = [self._say(e.session_id, "Update: " + render(outcome))]
            elif attempts >= job["max_attempts"]:
                self.store.update_job(job["id"], attempts=attempts, status="done")
                messages = [self._say(e.session_id, render(Outcome("unverified_final", e.id)))]
            else:
                nxt = datetime.fromisoformat(job["next_run_at"]) + timedelta(seconds=UNVERIFIED_GAPS_S[attempts])
                self.store.update_job(job["id"], attempts=attempts, next_run_at=nxt.isoformat())
                messages = []
            self.tracer.end_run(outcome=outcome.kind if outcome else "still_pending")
            if messages:
                out.append(self._notify(e.session_id, messages))
        return out

    # ---------- provider cancellations ----------
    def _provider_cancellations(self) -> list[dict[str, Any]]:
        baseline = not self.store.has_snapshots()  # first ever tick: record, don't alert on history
        out = []
        for user in self.store.users():
            r = self._list_bookings(user)
            if r is None:
                continue
            for b in r.data["bookings"]:
                prev = self.store.snapshot(b["booking_id"])
                self.store.set_snapshot(b["booking_id"], b["status"])
                if baseline or b["status"] != "cancelled_by_provider" or prev == "cancelled_by_provider":
                    continue
                if note := self._handle_provider_cancel(user, b):
                    out.append(note)
        return out

    def _decide(self, user: str, b: dict) -> str | None:
        """Whether to act on a provider cancellation. Returns the reason to stay quiet, or None to act."""
        if b["slot"] <= self.clock.now().strftime("%Y-%m-%dT%H:%M"):
            return "booking_in_past"
        if self.store.rebook_exists(user, b["booking_id"]):
            return "already_handled"
        if self.store.latest_session(user) is None:
            return "no_session_to_notify"
        return None

    def _replacement_request(self, user: str, b: dict) -> Request:
        """Replacement constraints from the cancelled booking plus usable (never candidate) memories."""
        mem, service = self.agent.memory, b["service"]
        start = max(date.fromisoformat(b["slot"][:10]), self.clock.now().date())
        earliest = mem.value(user, "earliest_time")
        budget = mem.value(user, f"budget_{service}")
        return Request(service=service, area=mem.value(user, "preferred_area"),
                       date_range=(start.isoformat(), (start + timedelta(days=REPLACEMENT_WINDOW_DAYS)).isoformat()),
                       time_window=(earliest, "21:00") if earliest else None,
                       budget_max=int(budget) if budget and budget.isdigit()
                       else round((b.get("price") or 0) * FALLBACK_BUDGET_MARGIN) or None,
                       booking_id=b["booking_id"])

    def _handle_provider_cancel(self, user: str, b: dict) -> dict[str, Any] | None:
        if reason := self._decide(user, b):
            self.tracer.emit("provider_cancel_ignored", booking_id=b["booking_id"], reason=reason)
            return None
        session = self.store.latest_session(user)
        req = self._replacement_request(user, b)
        e = self.store.create_errand(user, session, ErrandKind.REBOOK, req)
        self.tracer.start_run(session, kind="proactive", booking_id=b["booking_id"])
        self.tracer.emit("provider_cancel_detected", booking_id=b["booking_id"], errand_id=e.id,
                         decision="notify_and_find_replacement", request=req.model_dump(mode="json"))
        if not req.missing(e.kind):
            transition(self.store, self.tracer, e, S.COMPARING, "proactive_replacement")
        provider = b["provider"]["name"]
        messages = [self._say(session, render(Outcome("provider_cancelled", e.id, {
            "provider_name": provider, "service": b["service"], "slot": b["slot"]})))]
        notice = (f"Booking {b['booking_id']} ({b['service']} with {provider} on {fmt_slot(b['slot'])}) was cancelled "
                  f"by the provider. Find a replacement within the recorded request and request approval. "
                  f"Do not book anything.")
        messages += self.agent.run_proactive(e, notice)
        e = self.store.get_errand(e.id)
        if e.state != S.AWAITING_APPROVAL and not e.request.missing(e.kind):
            messages.append(self._say(session, render(Outcome("no_replacement", e.id))))
        self.tracer.end_run(outcome=e.state)
        return self._notify(session, messages)
