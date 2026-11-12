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
STANDING = re.compile(r"\b(remember|always|usually|never|from now on|going forward|in future|by default|"
                      r"i prefer|preferred|my \w+(?: \w+)? is)\b", re.I)


def _pair(args: dict, a: str, b: str, current: tuple | None, label: str,
          open_ends: tuple[str, str] | None = None) -> tuple | None:
    """Merge a (from, to) pair. With only one end given it keeps the other current end when that still
    makes a valid range; otherwise `open_ends` fills it ("not before 10:00" = 10:00 to end of day), or,
    without open ends, the range is that single value ("make it the 20th" = that one day)."""
    lo, hi = args.get(a), args.get(b)
    if lo is None and hi is None:
        return current
    if lo is not None and hi is not None:
        if hi < lo:
            raise ValueError(f"{label}: {b} is before {a}")
        return (lo, hi)
    if current and (merged := (lo or current[0], hi or current[1]))[0] <= merged[1]:
        return merged
    if open_ends:
        return (lo or open_ends[0], hi or open_ends[1])
    return (lo or hi, lo or hi)


def update_request(store: Store, tracer: Tracer, e: Errand, args: dict[str, Any]) -> ToolResult:
    if args.get("kind") in set(ErrandKind) and args["kind"] != e.kind and e.state != S.GATHERING:
        return _start_over(store, tracer, e, args)
    req = e.request.model_copy()
    try:
        if "kind" in args:
            e.kind = ErrandKind(args["kind"])
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


def _start_over(store: Store, tracer: Tracer, e: Errand, args: dict[str, Any]) -> ToolResult:
    """A different kind of errand mid-flow ("actually, cancel my cleaner instead"): close this one and record
    the new request on a fresh errand. Nothing has run yet: no tools exist from AWAITING_APPROVAL on."""
    approvals.invalidate_pending(store, tracer, e, "replaced_by_new_request")
    new = store.create_errand(e.user_id, e.session_id, ErrandKind(args["kind"]))
    e.selected["superseded_by"] = new.id
    store.save_errand(e)
    transition(store, tracer, e, S.FAILED, f"replaced_by_new_request:{new.id}")
    return update_request(store, tracer, new, args)


def request_approval(store: Store, tracer: Tracer, e: Errand, args: dict[str, Any], ttl_s: int) -> tuple[ToolResult, dict | None]:
    try:
        draft = approvals.build_draft(store, e, approvals.action_for(e.kind), slot=args.get("slot"),
                                      booking_id=args.get("booking_id"))
    except approvals.DraftError as err:
        return ToolResult.fail("cannot_request_approval", message=str(err)), None
    _, card = approvals.create(store, tracer, e, draft, ttl_s)
    transition(store, tracer, e, S.AWAITING_APPROVAL, "approval_requested")
    return ToolResult.success(status="shown_to_user", next_step="the user confirms with the card's button"), card


def _key(raw: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")


def propose_memory(mem: MemoryStore, tracer: Tracer, e: Errand, args: dict[str, Any],
                   correcting: bool = False) -> ToolResult:
    key, value, explicit = _key(args.get("key", "")), str(args.get("value", "")).strip(), bool(args.get("explicit"))
    if not key or not value:
        return ToolResult.fail("invalid_arguments", message="key and value are required")
    evidence = mem.stated_by_user(e.session_id, value)
    if evidence is None:  # provenance: only what the user said may become a memory
        tracer.emit("memory_rejected", errand_id=e.id, key=key, reason="not_stated_by_user")
        return ToolResult.fail("not_stated_by_user", message="only facts the user stated in this conversation can be remembered")
    scope = args.get("scope") or "global"
    # The model's explicit=true is only a claim: code checks the user framed it as a standing preference
    # ("always", "from now on", "my area is …"); otherwise the stated value is an inference to confirm later.
    downgraded = explicit and not correcting and not STANDING.search(mem.message_text(evidence))
    explicit = explicit and not downgraded
    if explicit:
        m = mem.correct(e.user_id, key, value, scope, evidence, e.session_id)
    elif any(c.value == value for c in mem.current(e.user_id, key)):
        return ToolResult.success(status="already_known", key=key)
    else:
        m = mem.add(e.user_id, key, value, MemorySource.INFERRED, MemoryStatus.CANDIDATE, scope, evidence, e.session_id)
    tracer.emit("memory_write", errand_id=e.id, key=key, value=value, source=m.source, status=m.status,
                **({"downgraded": "not_stated_as_a_standing_preference"} if downgraded else {}))
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
                                               "scope": args.get("scope")}, correcting=True)
    return ToolResult.fail("invalid_arguments", message="op must be list, correct or forget")
