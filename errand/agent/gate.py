"""The policy gate: every tool call, from the LLM or the executor, is checked here first."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from errand.agent.approvals import action_for
from errand.agent.keys import side_effect_args
from errand.agent.state import TOOLS_BY_STATE
from errand.agent.store import Store, slot_ref
from errand.config import Limits
from errand.contract import SIDE_EFFECT_TOOLS
from errand.models import (Action, ApprovalStatus, Errand, ErrandState, Execution, ExecState, Request,
                           split_date_range)
from errand.tracing.tracer import Tracer


@dataclass
class Decision:
    allow: bool
    reason: str = "ok"


def slot_outside_request(req: Request, slot: str) -> str | None:
    """Why a slot violates the recorded date range / time window, or None if it fits."""
    if req.date_range and not (req.date_range[0] <= slot[:10] <= req.date_range[1]):
        return f"slot {slot} is outside the requested dates {req.date_range[0]}..{req.date_range[1]}"
    if req.time_window and not (req.time_window[0] <= slot[11:16] < req.time_window[1]):
        return f"slot {slot} is outside the requested time window {req.time_window[0]}-{req.time_window[1]}"
    return None


# IDs a tool's arguments must reference: (ref kind, how to build the ref from args, denial message)
REQUIRED_REFS: dict[str, tuple[tuple[str, Callable[[dict], str], str], ...]] = {
    "get_provider": (("provider", lambda a: a.get("provider_id", ""),
                      "unknown provider_id; use one returned by search_providers"),),
    "hold_slot": (("provider", lambda a: a.get("provider_id", ""),
                   "unknown provider_id; use one returned by search_providers"),
                  ("slot", lambda a: slot_ref(a.get("provider_id", ""), a.get("slot", "")),
                   "that slot was not offered by this provider")),
}


class Gate:
    def __init__(self, store: Store, tracer: Tracer, limits: Limits | None = None):
        self.store, self.tracer, self.limits = store, tracer, limits or Limits()

    def _decide(self, actor: str, errand_id: str, tool: str, args: dict, d: Decision) -> Decision:
        self.tracer.emit("gate_decision", actor=actor, errand_id=errand_id, tool=tool, args=args,
                         allow=d.allow, reason=d.reason)
        return d

    # ---------- LLM-originated calls ----------
    def check_llm(self, errand: Errand, tool: str, args: dict[str, Any]) -> Decision:
        return self._decide("llm", errand.id, tool, args, self._llm_rules(errand, tool, args))

    def _llm_rules(self, e: Errand, tool: str, args: dict[str, Any]) -> Decision:
        if tool in SIDE_EFFECT_TOOLS:
            return Decision(False, f"{tool} is run by the system only after the user confirms; "
                                   "call request_approval instead")
        if tool not in TOOLS_BY_STATE[e.state]:
            return Decision(False, f"{tool} is not available in state {e.state}")
        if tool == "hold_slot" and action_for(e.kind) != Action.BOOK:
            return Decision(False, f"hold_slot is not used for a {e.kind}")
        for kind, make_ref, message in REQUIRED_REFS.get(tool, ()):
            if self.store.ref(e.id, kind, make_ref(args)) is None:
                return Decision(False, message)

        if tool == "search_providers":
            if missing := e.request.missing(e.kind):
                return Decision(False, f"missing required details: {', '.join(missing)}; ask the user")
            req = e.request
            for field in ("service", "area"):
                if getattr(req, field) is not None and args.get(field) != getattr(req, field):
                    return Decision(False, f"{field} differs from the recorded request; call update_request first")
            try:
                given = split_date_range(args.get("date_range"))
            except ValueError:
                given = None
            if req.date_range is not None and given != tuple(req.date_range):
                return Decision(False, "date_range differs from the recorded request; call update_request first")
        elif tool == "hold_slot" and (why := slot_outside_request(e.request, args.get("slot", ""))):
            return Decision(False, why)
        elif tool == "hold_slot" and (other := self.store.unconfirmed_booking(e.user_id, e.request.service)):
            return Decision(False, f"an earlier {e.request.service} booking ({other.id}) may already have gone "
                                   "through and is still being checked with the provider; tell the user to wait "
                                   "for that update instead of booking again")
        elif tool == "list_bookings" and args.get("user_id") != e.user_id:
            return Decision(False, "list_bookings is only allowed for the current user")
        elif tool == "request_approval":
            try:
                action = Action(args.get("action"))
            except ValueError:
                return Decision(False, "action must be book, cancel or reschedule")
            if action != action_for(e.kind):
                return Decision(False, f"this errand is a {e.kind}; it cannot request {action}")
            if action == Action.BOOK and e.state != ErrandState.HOLDING:
                return Decision(False, "hold the chosen slot first, then request approval")
            if action == Action.RESCHEDULE and (why := slot_outside_request(e.request, args.get("slot") or "")):
                return Decision(False, why)
        return Decision(True)

    # ---------- executor side effects ----------
    def check_side_effect(self, exe: Execution, tool: str, args: dict[str, Any]) -> Decision:
        return self._decide("executor", exe.errand_id, tool, args, self._side_effect_rules(exe, tool, args))

    def _side_effect_rules(self, exe: Execution, tool: str, args: dict[str, Any]) -> Decision:
        if tool != exe.action:
            return Decision(False, f"execution is for {exe.action}, not {tool}")
        current = self.store.get_execution(exe.id)  # the persisted write-ahead record is authoritative
        if current.state != ExecState.EXECUTING:
            return Decision(False, "no open write-ahead record for this call")
        approval = self.store.approval_by_hash(exe.errand_id, current.approval_hash)
        if approval is None or approval.status != ApprovalStatus.CONSUMED:
            return Decision(False, "no consumed approval backs this execution")
        if args != side_effect_args(current.action, current.payload, current.idem_key):
            return Decision(False, "arguments or idempotency key differ from the write-ahead record")
        if current.retries > self.limits.max_tool_retries:
            return Decision(False, "retry budget exhausted")
        if tool == "book" and self.store.other_guarded_booking(exe.errand_id, exe.id):
            return Decision(False, "this errand already has a booking in flight or confirmed")
        return Decision(True)
