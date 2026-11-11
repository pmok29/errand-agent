"""State machine, keys, approvals, gate and claim guard."""
import pytest

from errand.agent import approvals, claim_guard
from errand.agent.keys import idem_key
from errand.agent.state import NO_LLM_STATES, TOOLS_BY_STATE, IllegalTransition, transition
from errand.contract import SIDE_EFFECT_TOOLS
from errand.models import Action, ApprovalStatus, ErrandKind, ErrandState as S, ExecState


# ---------- state machine ----------
def test_llm_never_sees_side_effect_tools():
    for state, allowed in TOOLS_BY_STATE.items():
        assert not (allowed & SIDE_EFFECT_TOOLS), state
    for state in NO_LLM_STATES | {S.AWAITING_APPROVAL}:
        assert TOOLS_BY_STATE[state] == frozenset()


def test_illegal_transition_raises(world_factory):
    w = world_factory()
    e = w.errand()
    with pytest.raises(IllegalTransition):
        transition(w.store, w.tracer, e, S.EXECUTING, "skip approval")


# ---------- keys ----------
def test_key_includes_action_and_attempt():
    assert idem_key("e1", "book", 1) != idem_key("e1", "cancel", 1)
    assert idem_key("e1", "book", 1) != idem_key("e1", "book", 2)
    assert idem_key("e1", "book", 1) == idem_key("e1", "book", 1)


# ---------- approvals ----------
def test_card_and_hash_come_from_the_hold_quote(world_factory):
    w = world_factory(price_drift={"p-plb-01": 300})
    e = w.errand()
    hold = w.hold(e)
    a, card = w.approve(e)
    assert a.draft.price == hold.data["quoted_price"]
    assert f"₹{hold.data['quoted_price']:,}" in card["text"] and a.code in card["text"]
    assert a.hash == approvals.draft_hash(a.draft)


def test_only_exact_confirm_code_matches():
    assert approvals.match_code("CONFIRM ABCD") == "ABCD"
    assert approvals.match_code("  confirm abcd. ") == "ABCD"
    for text in ("yes", "yeah ok", "confirm", "sure, CONFIRM ABCD but at 11", "CONFIRM ABCDE"):
        assert approvals.match_code(text) is None


def test_consume_creates_write_ahead_record_atomically(world_factory):
    w = world_factory()
    e, exe = w.booked_and_approved()
    assert e.state == S.EXECUTING
    assert exe.state == ExecState.EXECUTING and exe.attempt_no == 1
    assert exe.idem_key == idem_key(e.id, "book", 1)
    assert w.store.approvals_for(e.id)[-1].status == ApprovalStatus.CONSUMED


def test_repeated_code_does_not_create_second_execution(world_factory):
    w = world_factory()
    e = w.errand()
    w.hold(e)
    a, _ = w.approve(e)
    assert approvals.consume(w.store, w.tracer, e, a.code).status == "ok"
    again = approvals.consume(w.store, w.tracer, w.store.get_errand(e.id), a.code)
    assert again.status == "already_consumed"
    assert len(w.store.executions_for(e.id)) == 1


def test_material_change_invalidates_approval(world_factory):
    w = world_factory()
    e = w.errand()
    w.hold(e, "p-plb-01")
    a, _ = w.approve(e)
    transition(w.store, w.tracer, e, S.COMPARING, "user changed slot")
    w.hold(e, "p-plb-01", slot_index=1)  # new hold -> different slot in the recomputed draft
    assert approvals.consume(w.store, w.tracer, w.store.get_errand(e.id), a.code).status == "changed"
    assert w.store.executions_for(e.id) == []


def test_expired_approval_is_rejected(world_factory):
    w = world_factory()
    e = w.errand()
    w.hold(e)
    a, _ = w.approve(e)
    w.clock.advance(601)
    assert approvals.consume(w.store, w.tracer, e, a.code).status == "expired"


# ---------- gate: LLM calls ----------
def test_gate_denies_side_effects_from_llm(world_factory):
    w = world_factory()
    e = w.errand()
    for tool in SIDE_EFFECT_TOOLS:
        assert not w.gate.check_llm(e, tool, {}).allow


def test_gate_denies_tools_outside_state(world_factory):
    w = world_factory()
    e = w.errand()  # GATHERING
    assert not w.gate.check_llm(e, "hold_slot", {"provider_id": "p-plb-01", "slot": "x"}).allow


def test_gate_requires_details_before_search(world_factory):
    w = world_factory()
    e = w.errand(area=None)
    transition(w.store, w.tracer, e, S.COMPARING, "test")
    d = w.gate.check_llm(e, "search_providers", {"service": "plumber", "area": "HSR Layout",
                                                 "date_range": ["2026-11-10", "2026-11-16"]})
    assert not d.allow and "area" in d.reason


def test_gate_rejects_search_args_that_drift_from_request(world_factory):
    w = world_factory()
    e = w.errand()
    transition(w.store, w.tracer, e, S.COMPARING, "test")
    args = {"service": "plumber", "area": "Koramangala", "date_range": "2026-11-10..2026-11-16"}
    assert not w.gate.check_llm(e, "search_providers", args).allow
    args["area"] = "Indiranagar"
    assert w.gate.check_llm(e, "search_providers", args).allow


def test_gate_rejects_hallucinated_ids(world_factory):
    w = world_factory()
    e = w.errand()
    transition(w.store, w.tracer, e, S.COMPARING, "test")
    w.search(e)
    assert not w.gate.check_llm(e, "get_provider", {"provider_id": "p-plb-99"}).allow
    assert not w.gate.check_llm(e, "hold_slot", {"provider_id": "p-plb-01", "slot": "2026-11-11T03:00"}).allow
    assert w.gate.check_llm(e, "get_provider", {"provider_id": "p-plb-01"}).allow


def test_gate_scopes_list_bookings_to_user(world_factory):
    w = world_factory()
    e = w.errand()
    assert not w.gate.check_llm(e, "list_bookings", {"user_id": "u2"}).allow


# ---------- gate: executor calls ----------
def test_side_effect_without_write_ahead_record_is_denied(world_factory):
    w = world_factory()
    e, exe = w.booked_and_approved()
    exe.state = ExecState.FAILED
    w.store.update_execution(exe)
    d = w.gate.check_side_effect(exe, "book", {"hold_id": exe.payload["hold_id"], "idempotency_key": exe.idem_key})
    assert not d.allow and "write-ahead" in d.reason


def test_side_effect_with_wrong_key_or_payload_is_denied(world_factory):
    w = world_factory()
    _, exe = w.booked_and_approved()
    assert not w.gate.check_side_effect(exe, "book", {"hold_id": exe.payload["hold_id"], "idempotency_key": "x"}).allow
    assert not w.gate.check_side_effect(exe, "book", {"hold_id": "h-other", "idempotency_key": exe.idem_key}).allow
    assert w.gate.check_side_effect(exe, "book", {"hold_id": exe.payload["hold_id"], "idempotency_key": exe.idem_key}).allow


# ---------- claim guard ----------
@pytest.mark.parametrize("text", [
    "Booked ✅", "Booked! See you Thursday.", "I've booked QuickFix for Thursday 9am.",
    "Your plumber is booked for Thursday.", "Your booking is confirmed.", "I have cancelled your booking.",
    "Your AC service has been rescheduled to Saturday.", "You're all set for Thursday.",
    "Done! I successfully booked it.", "The appointment was moved to Saturday.",
])
def test_claims_are_detected(text):
    assert claim_guard.claims_outcome(text)
    assert not claim_guard.allowed(text, S.AWAITING_APPROVAL)
    assert claim_guard.allowed(text, S.DONE)


@pytest.mark.parametrize("text", [
    "Shall I book QuickFix for Thursday 9am?", "Nothing has been booked yet.",
    "I haven't booked anything.", "Once you confirm, I'll book it.",
    "It will be confirmed after you reply with the code.", "You booked Suresh Electricals last time.",
    "Here are three plumbers available this week.", "Do you want me to cancel the Thursday booking?",
    "I can reschedule it to Saturday if you like.", "The booking is not confirmed until the provider replies.",
])
def test_non_claims_pass(text):
    assert not claim_guard.claims_outcome(text)
