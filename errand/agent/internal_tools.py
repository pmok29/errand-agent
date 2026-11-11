"""Internal tools the LLM calls to record the request and to ask for approval."""
from __future__ import annotations

from datetime import date
from typing import Any

from errand.agent import approvals
from errand.agent.state import transition
from errand.agent.store import Store
from errand.models import Errand, ErrandKind, ErrandState as S, ToolResult
from errand.tracing.tracer import Tracer

MATERIAL = ("service", "area", "date_range", "time_window", "budget_max", "provider_pref", "booking_id")


def _pair(args: dict, a: str, b: str, current: tuple | None, label: str) -> tuple | None:
    lo, hi = args.get(a), args.get(b)
    if lo is None and hi is None:
        return current
    lo, hi = lo or (current[0] if current else None), hi or (current[1] if current else None)
    if lo is None or hi is None:
        lo, hi = lo or hi, hi or lo
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
        tw = _pair(args, "time_from", "time_to", req.time_window, "time window")
        req.time_window = (tw[0], tw[1]) if tw else None
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
    return ToolResult.success(request=req.model_dump(mode="json"), missing=missing, already_asked=e.asked,
                              state=e.state, changed=changed)


def request_approval(store: Store, tracer: Tracer, e: Errand, args: dict[str, Any], ttl_s: int) -> tuple[ToolResult, dict | None]:
    try:
        draft = approvals.build_draft(store, e, approvals.action_for(e.kind), slot=args.get("slot"),
                                      booking_id=args.get("booking_id"))
    except approvals.DraftError as err:
        return ToolResult.fail("cannot_request_approval", message=str(err)), None
    _, card = approvals.create(store, tracer, e, draft, ttl_s, budget=e.request.budget_max)
    transition(store, tracer, e, S.AWAITING_APPROVAL, "approval_requested")
    return ToolResult.success(status="shown_to_user", code=card["code"]), card
