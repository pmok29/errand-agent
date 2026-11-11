"""Code-owned outcome messages. Side-effect outcomes are never worded by the LLM."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from errand.agent.approvals import fmt_slot
from errand.agent.store import Store
from errand.models import ApprovalStatus, Errand, ErrandState, ExecState


@dataclass
class Outcome:
    kind: str
    errand_id: str
    data: dict[str, Any] = field(default_factory=dict)
    card: dict | None = None


def _money(v) -> str:
    return f"₹{v:,}" if v is not None else ""


def render(o: Outcome) -> str:
    d = o.data
    slot = fmt_slot(d["slot"]) if d.get("slot") else ""
    match o.kind:
        case "booked":
            return (f"Booked: {d['provider_name']} on {slot} for {_money(d.get('price'))}. "
                    f"Confirmed with the provider (booking {d['booking_id']}).")
        case "cancelled":
            return f"Cancelled: your {d['provider_name']} booking on {slot}. Confirmed with the provider."
        case "rescheduled":
            return (f"Rescheduled: {d['provider_name']} moved from {fmt_slot(d['old_slot'])} to {slot}. "
                    "Confirmed with the provider.")
        case "slot_taken":
            return "That slot was taken before I could book it. Nothing was booked. Want me to find another slot?"
        case "slot_conflict":
            return f"{slot} isn't available for this provider, so nothing changed. Pick another slot and I'll ask you to confirm it."
        case "slot_unavailable":
            return "The hold expired and that slot is no longer free. Nothing was booked. Let's pick another slot."
        case "needs_reapproval":
            return (f"The hold expired and the provider now quotes {_money(d.get('new_price'))} "
                    f"instead of {_money(d.get('old_price'))}. Nothing was booked; please confirm the new price.")
        case "unverified":
            return ("The provider didn't confirm in time, so I can't tell yet whether the booking went through. "
                    "I have not made a second booking, and I'll check again and let you know.")
        case "failed":
            return f"That didn't go through ({d.get('reason', 'unexpected error')}). Nothing was changed on your side."
        case "cancel_failed":
            return "The provider's system kept failing, so the cancellation did not go through. Your booking is unchanged."
        case "already_done":
            return "That's already done. I didn't repeat it. " + d.get("summary", "")
        case "already_processing":
            return "I'm already working on that confirmation. I won't submit it twice."
        case "approval_expired":
            return "That confirmation code has expired. Nothing was booked. I'll show you the current details again."
        case "approval_changed":
            return "The details changed since that code was issued (provider, slot or price), so it no longer applies. Nothing was booked."
        case "awaiting":
            return f"Waiting for your confirmation. Reply CONFIRM {d['code']} to go ahead, or tell me what to change."
        case "limit":
            return ("I had to stop here because this request hit my step limit. "
                    f"Current status: {d.get('status', 'nothing has been booked')}. You can rephrase or try again.")
        case "nothing_yet":
            return "Nothing has been booked or changed yet."
    return d.get("text", "")


def _done_outcome(store: Store, e: Errand) -> Outcome:
    exe = [x for x in store.executions_for(e.id) if x.state == ExecState.DONE][-1]
    draft = store.approval_by_hash(e.id, exe.approval_hash).draft
    kind = {"book": "booked", "cancel": "cancelled", "reschedule": "rescheduled"}[exe.action]
    return Outcome(kind, e.id, {"provider_name": draft.provider_name, "slot": draft.slot, "old_slot": draft.old_slot,
                                "price": draft.price, "booking_id": exe.result.get("booking_id", draft.booking_id)})


def for_state(store: Store, e: Errand) -> Outcome:
    """The truthful status message for an errand, derived from persisted state only."""
    if e.state == ErrandState.DONE:
        return _done_outcome(store, e)
    if e.state == ErrandState.UNVERIFIED:
        return Outcome("unverified", e.id)
    if e.state in (ErrandState.EXECUTING, ErrandState.VERIFYING):
        return Outcome("already_processing", e.id)
    if e.state == ErrandState.AWAITING_APPROVAL:
        pending = store.approvals_for(e.id, ApprovalStatus.PENDING)
        if pending:
            return Outcome("awaiting", e.id, {"code": pending[-1].code})
    if e.state == ErrandState.FAILED:
        last = store.executions_for(e.id)
        return Outcome("failed", e.id, {"reason": last[-1].last_error if last else "unexpected error"})
    return Outcome("nothing_yet", e.id)


def repeat_confirmation(store: Store, e: Errand) -> Outcome:
    o = for_state(store, e)
    if o.kind in ("booked", "cancelled", "rescheduled"):
        return Outcome("already_done", e.id, {"summary": render(o)})
    return o
