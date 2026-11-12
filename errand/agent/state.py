"""Errand state machine: legal transitions and the tools the LLM may see in each state."""
from __future__ import annotations

from errand.agent.store import Store
from errand.models import Errand, ErrandState as S
from errand.tracing.tracer import Tracer

INTERNAL_TOOLS = frozenset({"update_request", "request_approval", "propose_memory", "memory_op"})
MEMORY_TOOLS = {"propose_memory", "memory_op"}

# LLM-visible tools per state. book/cancel/reschedule never appear: only the executor runs them.
TOOLS_BY_STATE: dict[S, frozenset[str]] = {
    S.GATHERING: frozenset({"list_bookings", "update_request", *MEMORY_TOOLS}),
    S.COMPARING: frozenset({"search_providers", "get_provider", "list_bookings", "hold_slot",
                            "request_approval", "update_request", *MEMORY_TOOLS}),
    S.HOLDING: frozenset({"hold_slot", "search_providers", "get_provider", "request_approval",
                          "update_request", *MEMORY_TOOLS}),
    S.AWAITING_APPROVAL: frozenset(),
    S.EXECUTING: frozenset(),
    S.VERIFYING: frozenset(),
    S.DONE: frozenset({"list_bookings", *MEMORY_TOOLS}),
    S.FAILED: frozenset({"list_bookings", *MEMORY_TOOLS}),
    S.UNVERIFIED: frozenset({"list_bookings", *MEMORY_TOOLS}),
    S.ABORTED_LIMIT: frozenset({"list_bookings", *MEMORY_TOOLS}),
}

# States where the LLM is not called at all: code drives the side effect and its verification.
NO_LLM_STATES = frozenset({S.EXECUTING, S.VERIFYING})

TRANSITIONS: dict[S, frozenset[S]] = {
    S.GATHERING: frozenset({S.COMPARING, S.ABORTED_LIMIT, S.FAILED}),
    S.COMPARING: frozenset({S.GATHERING, S.HOLDING, S.AWAITING_APPROVAL, S.ABORTED_LIMIT, S.FAILED}),
    S.HOLDING: frozenset({S.GATHERING, S.COMPARING, S.AWAITING_APPROVAL, S.ABORTED_LIMIT, S.FAILED}),
    S.AWAITING_APPROVAL: frozenset({S.EXECUTING, S.HOLDING, S.COMPARING, S.GATHERING, S.FAILED}),
    S.EXECUTING: frozenset({S.VERIFYING, S.COMPARING, S.AWAITING_APPROVAL, S.FAILED, S.UNVERIFIED}),
    S.VERIFYING: frozenset({S.DONE, S.FAILED, S.UNVERIFIED, S.EXECUTING, S.COMPARING, S.AWAITING_APPROVAL}),
    S.UNVERIFIED: frozenset({S.DONE, S.FAILED}),
    S.DONE: frozenset(),
    S.FAILED: frozenset(),
    S.ABORTED_LIMIT: frozenset(),
}


class IllegalTransition(Exception):
    pass


def transition(store: Store, tracer: Tracer, errand: Errand, to: S, reason: str) -> Errand:
    if errand.state == to:
        return errand
    if to not in TRANSITIONS[errand.state]:
        raise IllegalTransition(f"{errand.state} -> {to} ({reason})")
    frm, errand.state = errand.state, to
    store.save_errand(errand)
    tracer.emit("state_transition", errand_id=errand.id, frm=frm, to=to, reason=reason)
    return errand
