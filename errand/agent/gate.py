"""The policy gate: every tool call, from the LLM or the executor, is checked here first."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from errand.agent.approvals import action_for
from errand.agent.state import TOOLS_BY_STATE
from errand.agent.store import GUARD_STATES, Store
from errand.config import Limits
from errand.mocks.tools import SIDE_EFFECT_TOOLS
from errand.models import Action, ApprovalStatus, Errand, ErrandKind, Execution, ExecState
from errand.tracing.tracer import Tracer


@dataclass
class Decision:
    allow: bool
    reason: str = "ok"


def _norm_range(value: Any) -> tuple[str, str] | None:
    if isinstance(value, dict):
        value = (value.get("start"), value.get("end"))
    if isinstance(value, str):
        value = value.split("..")
    try:
        start, end = value
        return str(start), str(end)
    except (TypeError, ValueError):
        return None


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
        ref = lambda kind, r: self.store.ref(e.id, kind, r) is not None  # noqa: E731

        if tool == "search_providers":
            if missing := e.request.missing(e.kind):
                return Decision(False, f"missing required details: {', '.join(missing)}; ask the user")
            req = e.request
            for field, given in (("service", args.get("service")), ("area", args.get("area"))):
                if getattr(req, field) is not None and given != getattr(req, field):
                    return Decision(False, f"{field} differs from the recorded request; call update_request first")
            if req.date_range is not None and _norm_range(args.get("date_range")) != tuple(req.date_range):
                return Decision(False, "date_range differs from the recorded request; call update_request first")
        elif tool == "get_provider":
            if not ref("provider", args.get("provider_id", "")):
                return Decision(False, "unknown provider_id; use one returned by search_providers")
        elif tool == "hold_slot":
            if e.kind not in (ErrandKind.BOOK, ErrandKind.REBOOK):
                return Decision(False, f"hold_slot is not used for a {e.kind}")
            pid = args.get("provider_id", "")
            if not ref("provider", pid):
                return Decision(False, "unknown provider_id; use one returned by search_providers")
            if not ref("slot", f"{pid}|{args.get('slot')}"):
                return Decision(False, "that slot was not offered by this provider")
        elif tool == "list_bookings":
            if args.get("user_id") != e.user_id:
                return Decision(False, "list_bookings is only allowed for the current user")
        elif tool == "request_approval":
            try:
                action = Action(args.get("action"))
            except ValueError:
                return Decision(False, "action must be book, cancel or reschedule")
            if action != action_for(e.kind):
                return Decision(False, f"this errand is a {e.kind}; it cannot request {action}")
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
        if args.get("idempotency_key") != current.idem_key:
            return Decision(False, "idempotency key does not match the write-ahead record")
        bound = {k: current.payload[k] for k in ("hold_id", "booking_id", "slot")
                 if k in current.payload and k in args}
        if any(args[k] != v for k, v in bound.items()):
            return Decision(False, "arguments differ from the approved payload")
        if current.retries > self.limits.max_tool_retries:
            return Decision(False, "retry budget exhausted")
        if tool == "book":
            others = [x for x in self.store.executions_for(exe.errand_id, Action.BOOK)
                      if x.id != exe.id and x.state in GUARD_STATES]
            if others:
                return Decision(False, "this errand already has a booking in flight or confirmed")
        return Decision(True)
