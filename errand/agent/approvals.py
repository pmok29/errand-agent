"""Approval drafts, hashes, confirmation codes, code-rendered cards, and consumption.

The card the user sees and the hash the gate checks are both derived from one ApprovalDraft,
which is built only from persisted records (hold quote, seen booking), never from LLM arguments.
"""
from __future__ import annotations

import base64
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from errand.agent.keys import payload_hash
from errand.agent.state import transition
from errand.agent.store import Store, slot_ref
from errand.clock import is_expired
from errand.db import new_id
from errand.models import (Action, Approval, ApprovalDraft, ApprovalStatus, Errand, ErrandKind,
                           ErrandState, Execution, Hold)
from errand.tracing.tracer import Tracer

CODE_RE = re.compile(r"\s*confirm\s+([a-z2-7]{4})\s*[.!]?\s*", re.I)


class DraftError(Exception):
    pass


def action_for(kind: ErrandKind) -> Action:
    return Action.BOOK if kind in (ErrandKind.BOOK, ErrandKind.REBOOK) else Action(kind)


def fmt_slot(slot: str) -> str:
    return datetime.fromisoformat(slot).strftime("%a %d %b, %H:%M")


def fmt_money(v: int | None) -> str:
    return f"₹{v:,}" if v is not None else ""


def book_payload(hold: Hold) -> dict:
    return {"hold_id": hold.hold_id, "slot": hold.slot, "price": hold.quoted_price}


def build_draft(store: Store, errand: Errand, action: Action, slot: str | None = None,
                booking_id: str | None = None) -> ApprovalDraft:
    if action != action_for(errand.kind):
        raise DraftError(f"this errand is a {errand.kind}, not a {action}")
    if action == Action.BOOK:
        hold = store.latest_hold(errand.id)
        if hold is None:
            raise DraftError("hold a slot before requesting approval")
        if is_expired(hold.expires_at, store.clock.now()):
            raise DraftError("the hold has expired; hold the slot again")
        prov = store.ref(errand.id, "provider", hold.provider_id) or {}
        return ApprovalDraft(action=action, provider_id=hold.provider_id,
                             provider_name=prov.get("name", hold.provider_id), slot=hold.slot,
                             price=hold.quoted_price)
    booking_id = booking_id or errand.request.booking_id
    b = store.ref(errand.id, "booking", booking_id) if booking_id else None
    if b is None:
        raise DraftError("unknown booking; list the user's bookings first")
    if action == Action.CANCEL:
        return ApprovalDraft(action=action, provider_id=b["provider_id"], provider_name=b["provider_name"],
                             slot=b["slot"], price=b.get("price"), booking_id=booking_id)
    if slot is None or store.ref(errand.id, "slot", slot_ref(b["provider_id"], slot)) is None:
        raise DraftError("pick a slot that this provider actually offered")
    return ApprovalDraft(action=action, provider_id=b["provider_id"], provider_name=b["provider_name"],
                         slot=slot, old_slot=b["slot"], price=b.get("price"), booking_id=booking_id)


def draft_hash(draft: ApprovalDraft) -> str:
    return payload_hash(draft.model_dump(mode="json"))


def code_for(hash: str, approval_id: str) -> str:
    """Short code for one card. Salted with the approval id, so a re-shown card never revives an old code."""
    return base64.b32encode(hashlib.sha256(f"{hash}|{approval_id}".encode()).digest())[:4].decode()


def render_card(draft: ApprovalDraft, code: str, budget: int | None = None) -> dict:
    over_budget = draft.price is not None and budget is not None and draft.price > budget
    if draft.action == Action.BOOK:
        line = f"Book {draft.provider_name} on {fmt_slot(draft.slot)} for {fmt_money(draft.price)}"
        if over_budget:
            line += f" (above your {fmt_money(budget)} budget)"
    elif draft.action == Action.CANCEL:
        line = f"Cancel your {draft.provider_name} booking on {fmt_slot(draft.slot)}"
    else:
        line = (f"Move your {draft.provider_name} booking from {fmt_slot(draft.old_slot)} "
                f"to {fmt_slot(draft.slot)}")
    text = f"{line}. Reply CONFIRM {code} to go ahead, or tell me what to change."
    return {"kind": "card", "text": text, "code": code, "draft": draft.model_dump(mode="json"),
            "over_budget": over_budget}


def payload_for(store: Store, errand: Errand, draft: ApprovalDraft) -> dict:
    if draft.action == Action.BOOK:
        return book_payload(store.latest_hold(errand.id))
    if draft.action == Action.CANCEL:
        return {"booking_id": draft.booking_id}
    return {"booking_id": draft.booking_id, "slot": draft.slot}


def create(store: Store, tracer: Tracer, errand: Errand, draft: ApprovalDraft, ttl_s: int,
           budget: int | None = None) -> tuple[Approval, dict]:
    invalidate_pending(store, tracer, errand, "superseded")
    expires = store.clock.now() + timedelta(seconds=ttl_s)
    if draft.action == Action.BOOK:
        expires = min(expires, datetime.fromisoformat(store.latest_hold(errand.id).expires_at))
    h = draft_hash(draft)
    approval_id = new_id("a")
    a = Approval(id=approval_id, errand_id=errand.id, hash=h, code=code_for(h, approval_id), draft=draft,
                 status=ApprovalStatus.PENDING, expires_at=expires.isoformat())
    store.insert_approval(a)
    tracer.emit("approval_created", errand_id=errand.id, approval_id=a.id, hash=h, code=a.code,
                draft=draft.model_dump(mode="json"), expires_at=a.expires_at)
    return a, render_card(draft, a.code, budget)


def invalidate_pending(store: Store, tracer: Tracer, errand: Errand, reason: str) -> None:
    for a in store.approvals_for(errand.id, ApprovalStatus.PENDING):
        store.set_approval_status(a.id, ApprovalStatus.INVALIDATED)
        tracer.emit("approval_invalidated", errand_id=errand.id, approval_id=a.id, reason=reason)


def match_code(text: str) -> str | None:
    m = CODE_RE.fullmatch(text)
    return m.group(1).upper() if m else None


@dataclass
class ConsumeResult:
    status: str        # ok | no_match | already_consumed | invalidated | expired | changed
    execution: Execution | None = None
    approval: Approval | None = None


def consume(store: Store, tracer: Tracer, errand: Errand, code: str) -> ConsumeResult:
    """Atomically: approval -> consumed, write-ahead execution row inserted, errand -> EXECUTING."""
    with store.db.tx():
        a = store.approval_by_code(errand.id, code)
        if a is None:
            return ConsumeResult("no_match")
        if a.status != ApprovalStatus.PENDING:
            return ConsumeResult("already_consumed" if a.status == ApprovalStatus.CONSUMED else str(a.status), approval=a)
        if is_expired(a.expires_at, store.clock.now()):
            store.set_approval_status(a.id, ApprovalStatus.EXPIRED)
            tracer.emit("approval_expired", errand_id=errand.id, approval_id=a.id)
            return ConsumeResult("expired", approval=a)
        try:
            current = build_draft(store, errand, a.draft.action, slot=a.draft.slot, booking_id=a.draft.booking_id)
        except DraftError:
            current = None
        if current is None or draft_hash(current) != a.hash:
            store.set_approval_status(a.id, ApprovalStatus.INVALIDATED)
            tracer.emit("approval_invalidated", errand_id=errand.id, approval_id=a.id, reason="material_change")
            return ConsumeResult("changed", approval=a)
        store.set_approval_status(a.id, ApprovalStatus.CONSUMED)
        exe = store.open_execution(errand.id, a.draft.action, payload_for(store, errand, a.draft), a.hash)
        transition(store, tracer, errand, ErrandState.EXECUTING, "approval_consumed")
    tracer.emit("approval_consumed", errand_id=errand.id, approval_id=a.id, execution_id=exe.id,
                idem_key=exe.idem_key, attempt_no=exe.attempt_no)
    return ConsumeResult("ok", execution=exe, approval=a)
