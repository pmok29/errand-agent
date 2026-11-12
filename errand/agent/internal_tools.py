"""Internal tools the LLM calls: record the request, ask for approval, and manage memory."""
from __future__ import annotations

import re
from datetime import date
from typing import Any

from errand.agent import approvals
from errand.agent.memory import MemoryStore
from errand.agent.state import transition
from errand.agent.store import Store
from errand.models import Errand, ErrandKind, ErrandState as S, MemorySource, MemoryStatus, ToolResult
from errand.tracing.tracer import Tracer

MATERIAL = ("service", "area", "date_range", "time_window", "budget_max", "provider_pref", "booking_id")


def _pair(args: dict, a: str, b: str, current: tuple | None, label: str,
          open_ends: tuple[str, str] | None = None) -> tuple | None:
    """Merge a (from, to) pair. With only one end given: `open_ends` fills the other ("not before 10:00"
    is 10:00 to end of day); without it, the range is that single value (one date)."""
    lo, hi = args.get(a), args.get(b)
    if lo is None and hi is None:
        return current
    if current:
        lo, hi = lo or current[0], hi or current[1]
    lo, hi = (lo or open_ends[0], hi or open_ends[1]) if open_ends else (lo or hi, hi or lo)
    if hi < lo:
        raise ValueError(f"{label}: {b} is before {a}")
    return (lo, hi)


def update_request(store: Store, tracer: Tracer, e: Errand, args: dict[str, Any]) -> ToolResult:
    req = e.request.model_copy()
    try:
        if "kind" in args:
            kind = ErrandKind(args["kind"])
            if kind != e.kind and e.state != S.GATHERING:
                raise ValueError("the kind of errand can only change before searching; start a new request")
            e.kind = kind
        for f in ("service", "area", "budget_max", "provider_pref", "booking_id"):
            if f in args:
                setattr(req, f, args[f])
        req.date_range = _pair(args, "date_from", "date_to", req.date_range, "dates")
        if req.date_range:
            for d in req.date_range:
                date.fromisoformat(d)
            if req.date_range[1] < store.clock.now().date().isoformat():
                raise ValueError("those dates are in the past")
        req.time_window = _pair(args, "time_from", "time_to", req.time_window, "time window", ("00:00", "23:59"))
    except ValueError as err:
        return ToolResult.fail("invalid_request", message=str(err))

    changed = [f for f in MATERIAL if getattr(req, f) != getattr(e.request, f)]
    e.request = req
    store.save_errand(e)
    tracer.emit("request_updated", errand_id=e.id, changed=changed, request=req.model_dump(mode="json"))

    missing = req.missing(e.kind)
    if missing:
        if e.state != S.GATHERING:
            transition(store, tracer, e, S.GATHERING, "details_missing")
    elif e.state == S.GATHERING:
        transition(store, tracer, e, S.COMPARING, "request_complete")
    elif changed and e.state == S.HOLDING:
        transition(store, tracer, e, S.COMPARING, "request_changed")
    if changed:
        e.selected.pop("options", None)
        store.save_errand(e)
    hint = {"next_step": "call list_bookings and pick the matching booking; don't ask the user for an ID"} \
        if "booking_id" in missing else {}
    return ToolResult.success(request=req.model_dump(mode="json"), missing=missing, already_asked=e.asked,
                              state=e.state, changed=changed, **hint)


def request_approval(store: Store, tracer: Tracer, e: Errand, args: dict[str, Any], ttl_s: int) -> tuple[ToolResult, dict | None]:
    try:
        draft = approvals.build_draft(store, e, approvals.action_for(e.kind), slot=args.get("slot"),
                                      booking_id=args.get("booking_id"))
    except approvals.DraftError as err:
        return ToolResult.fail("cannot_request_approval", message=str(err)), None
    _, card = approvals.create(store, tracer, e, draft, ttl_s)
    transition(store, tracer, e, S.AWAITING_APPROVAL, "approval_requested")
    return ToolResult.success(status="shown_to_user", code=card["code"]), card


def _key(raw: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")


def propose_memory(mem: MemoryStore, tracer: Tracer, e: Errand, args: dict[str, Any]) -> ToolResult:
    key, value, explicit = _key(args.get("key", "")), str(args.get("value", "")).strip(), bool(args.get("explicit"))
    if not key or not value:
        return ToolResult.fail("invalid_arguments", message="key and value are required")
    evidence = mem.stated_by_user(e.session_id, value)
    if evidence is None:  # provenance: only what the user said may become a memory
        tracer.emit("memory_rejected", errand_id=e.id, key=key, reason="not_stated_by_user")
        return ToolResult.fail("not_stated_by_user", message="only facts the user stated in this conversation can be remembered")
    scope = args.get("scope") or "global"
    if explicit:
        m = mem.correct(e.user_id, key, value, scope, evidence, e.session_id)
    elif any(c.value == value for c in mem.current(e.user_id, key)):
        return ToolResult.success(status="already_known", key=key)
    else:
        m = mem.add(e.user_id, key, value, MemorySource.INFERRED, MemoryStatus.CANDIDATE, scope, evidence, e.session_id)
    tracer.emit("memory_write", errand_id=e.id, key=key, value=value, source=m.source, status=m.status)
    return ToolResult.success(status=m.status, key=key, source=m.source)


def memory_op(mem: MemoryStore, tracer: Tracer, e: Errand, args: dict[str, Any]) -> ToolResult:
    op = args.get("op")
    if op == "list":
        return ToolResult.success(memories=[{"key": m.key, "value": m.value, "source": m.source, "status": m.status}
                                            for m in mem.visible(e.user_id)])
    key = _key(args.get("key", ""))
    if op == "forget":
        gone = mem.forget(e.user_id, key)
        if not gone:
            return ToolResult.fail("not_found", message=f"no memory '{key}'; known keys: "
                                   + ", ".join(sorted({m.key for m in mem.visible(e.user_id)})))
        tracer.emit("memory_forgotten", errand_id=e.id, key=key, count=len(gone))
        return ToolResult.success(status="forgotten", key=key)
    if op == "correct":
        return propose_memory(mem, tracer, e, {"key": key, "value": args.get("value", ""), "explicit": True,
                                               "scope": args.get("scope")})
    return ToolResult.fail("invalid_arguments", message="op must be list, correct or forget")
