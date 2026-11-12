"""Orchestrator behaviour that doesn't depend on the scripted policy: limits, gate denials, claims, codes."""
import pytest

from errand.agent.executor import SimulatedCrash
from errand.agent.llm.base import LLMError, LLMResponse, ToolCall
from errand.agent.llm.scripted import ReactivePolicy
from errand.mocks.scenario import ScenarioConfig
from errand.models import ErrandState as S
from tests.harness import Sim


class Scripted:
    """Returns canned responses in order (repeating the last one)."""
    model = "canned"

    def __init__(self, *responses):
        self.responses, self.seen = list(responses), []

    def complete(self, messages, tools):
        self.seen.append(({t["function"]["name"] for t in tools}, messages))
        r = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(r, Exception):
            raise r
        return r


def call(name, **args):
    return LLMResponse(text=None, tool_calls=[ToolCall(id=f"c-{name}", name=name, args=args)], model="canned")


def agent_with(tmp_path, llm, **cfg):
    sim = Sim(tmp_path, ScenarioConfig(**cfg), llm=llm)
    return sim.agent, sim.mock, sim.tracer


def test_llm_cannot_book_and_never_sees_side_effect_tools(tmp_path):
    llm = Scripted(call("book", hold_id="h-1", idempotency_key="k"), LLMResponse(text="Okay, tell me more."))
    agent, mock, tracer = agent_with(tmp_path, llm)
    agent.handle_user_message(agent.new_session(), "book anything")
    assert all(not ({"book", "cancel", "reschedule"} & tools) for tools, _ in llm.seen)
    denied = [e for e in tracer.of_type("gate_decision") if not e["allow"]]
    assert denied and denied[0]["tool"] == "book"
    assert not mock.agent_bookings()


def test_false_success_claim_is_replaced_by_state_template(tmp_path):
    agent, _, tracer = agent_with(tmp_path, Scripted(LLMResponse(text="Booked ✅ See you Thursday!")))
    out = agent.handle_user_message(agent.new_session(), "plumber please")
    assert out[0]["text"] == "Nothing has been booked or changed yet."
    assert tracer.of_type("claim_blocked")


def test_looping_model_hits_the_tool_call_limit_with_a_clear_message(tmp_path):
    agent, _, tracer = agent_with(tmp_path, Scripted(call("list_bookings", user_id="u1")))
    sid = agent.new_session()
    out = agent.handle_user_message(sid, "hi")
    assert out[0]["kind"] == "template" and "step limit" in out[0]["text"]
    assert tracer.of_type("limit_hit")[0]["limit"] == "llm_turns"
    assert agent.store.latest_errand(sid).state == S.ABORTED_LIMIT


def test_llm_outage_produces_a_clear_message(tmp_path):
    agent, _, _ = agent_with(tmp_path, Scripted(LLMError("down", retryable=True)))
    out = agent.handle_user_message(agent.new_session(), "plumber")
    assert "trouble reaching" in out[0]["text"] and "Nothing was booked" in out[0]["text"]


def test_invalid_json_arguments_become_a_tool_error_not_a_crash(tmp_path):
    bad = LLMResponse(text=None, tool_calls=[ToolCall(id="c1", name="update_request", args={}, parse_error="invalid JSON")])
    llm = Scripted(bad, LLMResponse(text="Which area?"))
    agent, _, _ = agent_with(tmp_path, llm)
    assert agent.handle_user_message(agent.new_session(), "plumber")[0]["text"] == "Which area?"
    assert "invalid_arguments" in llm.seen[1][1][-1]["content"]


def test_reply_without_code_is_not_approval(tmp_path):
    agent, mock, _ = agent_with(tmp_path, ReactivePolicy(), book_outcomes=["success"])
    sid = agent.new_session()
    agent.handle_user_message(sid, "Plumber in Indiranagar this week, mornings, under 1500")
    card = agent.handle_user_message(sid, "option 2")[-1]
    assert card["kind"] == "card"
    agent.handle_user_message(sid, "yeah ok")
    assert agent.store.latest_errand(sid).state != S.EXECUTING
    stale = agent.handle_user_message(sid, f"CONFIRM {card['card']['code']}")
    assert "no longer applies" in stale[0]["text"] and not mock.agent_bookings()


def test_repeated_code_after_success_does_not_rebook(tmp_path):
    agent, mock, _ = agent_with(tmp_path, ReactivePolicy(), book_outcomes=["success", "success"])
    sid = agent.new_session()
    agent.handle_user_message(sid, "Plumber in Indiranagar this week, mornings, under 1500")
    code = agent.handle_user_message(sid, "first one")[-1]["card"]["code"]
    assert agent.handle_user_message(sid, f"CONFIRM {code}")[0]["text"].startswith("Booked:")
    again = agent.handle_user_message(sid, f"CONFIRM {code}")
    assert again[0]["text"].startswith("That's already done") and len(mock.agent_bookings()) == 1


def test_change_of_errand_kind_is_persisted(tmp_path):
    """Bug #3: save_errand didn't write `kind`, so 'cancel my plumber booking' stayed a booking."""
    agent, _, tracer = agent_with(tmp_path, ReactivePolicy())
    sid = agent.new_session()
    agent.handle_user_message(sid, "Cancel my plumber booking")
    e = agent.store.latest_errand(sid)
    assert e.kind == "cancel" and e.request.booking_id == "b-plb-up"
    assert "area" not in tracer.of_type("assistant_msg")[-1]["text"].lower()


def test_stale_code_keeps_the_newer_card_open(tmp_path):
    """Bug #4: an old code moved the errand out of AWAITING_APPROVAL, so the current card's code then failed."""
    agent, mock, _ = agent_with(tmp_path, ReactivePolicy(), book_outcomes=["success"])
    sid = agent.new_session()
    agent.handle_user_message(sid, "Plumber in Indiranagar this week, mornings, under 1500")
    old = agent.handle_user_message(sid, "option 1")[-1]["card"]
    new = agent.handle_user_message(sid, "option 2")[-1]["card"]
    stale = agent.handle_user_message(sid, f"CONFIRM {old['code']}")[0]["text"]
    assert "no longer applies" in stale and f"CONFIRM {new['code']}" in stale
    assert agent.store.latest_errand(sid).state == S.AWAITING_APPROVAL
    assert agent.handle_user_message(sid, f"CONFIRM {new['code']}")[0]["text"].startswith("Booked:")
    assert len(mock.agent_bookings()) == 1


def test_recovery_delivers_the_reapproval_card(tmp_path):
    """Recovery used to queue only text: a price change found while resuming lost its card and code."""
    sim = Sim(tmp_path, ScenarioConfig(book_outcomes=["timeout_not_booked"]), faults={"book_returned"})
    sim.say("Plumber in Indiranagar this week, mornings, under 1500")
    sim.say("option 1")
    with pytest.raises(SimulatedCrash):
        sim.confirm()
    sim.clock.advance(601)                                   # the hold expires while the agent is down
    sim.scenario.price_drift = {"p-plb-03": 100}             # and the re-hold quotes a new price
    assert [o.kind for o in sim.restart()] == ["needs_reapproval"]
    [note] = sim.agent.store.take_notifications(sim.sid)
    card = note["messages"][-1]
    assert card["kind"] == "card" and card["card"]["draft"]["price"] == 1250
    assert sim.agent.store.session_messages(sim.sid)[-1]["card"]["code"] == card["card"]["code"]


def test_single_time_bound_is_open_ended(tmp_path):
    """Bug #6: gpt-oss sent "not before 10 a.m." as time_from only; it became a 10:00-10:00 window, so no
    slot fitted and the model told the user the provider had no slots (then invented one)."""
    week = {"date_from": "2026-11-10", "date_to": "2026-11-15"}
    llm = Scripted(call("update_request", service="electrician", area="Indiranagar", time_from="10:00", **week),
                   call("search_providers", service="electrician", area="Indiranagar", date_range=list(week.values())),
                   LLMResponse(text="Here are the options."))
    agent, _, _ = agent_with(tmp_path, llm)
    sid = agent.new_session()
    agent.handle_user_message(sid, "Same electrician as last time, this week, not before 10 a.m.")
    e = agent.store.latest_errand(sid)
    assert e.request.time_window == ("10:00", "23:59") and e.request.date_range == ("2026-11-10", "2026-11-15")
    suresh = next(o for o in e.selected["options"] if o["provider_id"] == "p-ele-02")
    assert suresh["slots"] and all(s[11:16] >= "10:00" for s in suresh["slots"])


def test_confirm_code_reaches_its_own_errand_when_a_rebook_shares_the_session(tmp_path):
    """A provider cancellation mid-conversation adds a rebook errand to the session; the user's earlier
    card must still be confirmable (codes used to be looked up only in the session's latest errand)."""
    sim = Sim(tmp_path, ScenarioConfig(book_outcomes=["success", "success"]), seed=True)
    sim.tick()
    plumber = sim.say("Plumber in Indiranagar this week, mornings, under 1500") and sim.say("option 1")[-1]
    sim.mock.trigger_provider_cancel("b-cln-up")
    rebook = sim.tick()[0]["messages"][-1]
    assert sim.confirm(plumber)[0]["text"].startswith("Booked: QuickFix")
    assert sim.confirm(rebook)[0]["text"].startswith("Booked: ShineBright")
    assert len(sim.new_bookings()) == 2


def test_error_after_the_booking_reports_the_booking_not_nothing(tmp_path, monkeypatch):
    sim = Sim(tmp_path, ScenarioConfig(book_outcomes=["success"]))
    sim.say("Plumber in Indiranagar this week, mornings, under 1500")
    card = sim.say("option 1")[-1]
    monkeypatch.setattr(sim.agent, "_outcome", lambda e, o: 1 / 0)   # breaks after the executor finished
    out = sim.confirm(card)
    assert out[0]["text"].startswith("Booked:") and len(sim.new_bookings()) == 1


@pytest.mark.parametrize("args, expected", [
    ({"date_from": "2026-11-20"}, ("2026-11-20", "2026-11-20")),   # past the old end: that one day, no error
    ({"date_from": "2026-11-13"}, ("2026-11-13", "2026-11-14")),   # inside: keeps the old end
    ({"date_to": "2026-11-16"}, ("2026-11-12", "2026-11-16")),     # extends the end
])
def test_one_sided_date_corrections(tmp_path, args, expected):
    from errand.agent import internal_tools
    sim = Sim(tmp_path)
    e = sim.agent.store.create_errand("u1", sim.sid, "book")
    internal_tools.update_request(sim.agent.store, sim.tracer, e, {"date_from": "2026-11-12", "date_to": "2026-11-14"})
    r = internal_tools.update_request(sim.agent.store, sim.tracer, sim.agent.store.get_errand(e.id), args)
    assert r.ok and sim.agent.store.get_errand(e.id).request.date_range == expected


def test_going_back_to_a_slot_you_already_hold(tmp_path):
    """Hold A, switch to B, back to A within the hold's 10 minutes: A used to be "unavailable" (our own hold)."""
    sim = Sim(tmp_path, ScenarioConfig(book_outcomes=["success"]))
    week = ["2026-11-10", "2026-11-15"]
    a, b = next(p for p in sim.mock.search_providers("plumber", "Indiranagar", week).data["providers"]
                if p["provider_id"] == "p-plb-01")["slots"][:2]
    sim.agent.llm = Scripted(
        call("update_request", service="plumber", area="Indiranagar", date_from=week[0], date_to=week[1]),
        call("search_providers", service="plumber", area="Indiranagar", date_range=week),
        call("hold_slot", provider_id="p-plb-01", slot=a), call("hold_slot", provider_id="p-plb-01", slot=b),
        call("hold_slot", provider_id="p-plb-01", slot=a), call("request_approval", action="book"))
    card = sim.say("plumber this week")[-1]
    assert card["kind"] == "card" and card["card"]["draft"]["slot"] == a
    assert not [c for c in sim.tool_calls("hold_slot") if not c["ok"]]
    assert sim.confirm(card)[0]["text"].startswith("Booked:")
