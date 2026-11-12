"""Baseline conversation: happy path (#1), missing details (#2), correction (#3), approval (#27), repeated code (#13)."""
import pytest

from tests.harness import PLUMBER, Sim, SimUser, finish


@pytest.mark.eval(id="1", title="Happy path: plumber, this week, mornings, under ₹1,500",
                  expected="asks only for area; 1 booking ≤ ₹1,500 in a morning slot; status verified before 'Booked'")
def test_01_happy_path(tmp_path, ev):
    sim = Sim(tmp_path, "normal")
    sim.scenario.book_outcomes = ["success"]
    out = sim.say(PLUMBER)
    assert "area" in out[0]["text"].lower() and "date" not in out[0]["text"].lower()
    sim.say("Indiranagar")
    assert "Which one" in sim.texts()[-1]
    card = sim.say("Book option 1")[-1]
    draft = card["card"]["draft"]
    out = sim.confirm(card)
    assert out[0]["text"].startswith("Booked:")
    [b] = sim.new_bookings()
    assert b["price"] == draft["price"] <= 1500 and "06:00" <= b["slot"][11:16] < "12:00"
    tools = [c["tool"] for c in sim.tool_calls()]
    assert tools.index("get_booking_status") > tools.index("book")
    ev.actual = f"1 booking: {draft['provider_name']} {b['slot'].replace('T', ' ')} ₹{b['price']:,}; status checked after book"
    finish(sim, ev)


@pytest.mark.eval(id="2", title="Missing details: 'I need a plumber'",
                  expected="asks for area and dates only, never twice; no search until both are known")
def test_02_missing_details(tmp_path, ev):
    sim = Sim(tmp_path)
    user = SimUser({"area": "Koramangala", "date_range": "tomorrow"})
    out = user.converse(sim, "I need a plumber")
    assert "Which one" in out[-1]["text"]
    assert set(user.asked) == {"area", "date_range"} and max(user.asked.values()) == 1
    assert "budget" not in " ".join(sim.texts()).lower()
    first_search = sim.tool_calls("search_providers")[0]["seq"]
    last_question = max(m["seq"] for m in sim.events("assistant_msg") if "?" in m["text"] and "Which one" not in m["text"])
    assert first_search > last_question
    assert not [d for d in sim.events("gate_decision") if d["tool"] == "search_providers" and not d["allow"]]
    ev.actual = f"asked {dict(user.asked)}; first search after the last answer"
    finish(sim, ev)


@pytest.mark.eval(id="3", title="Correction keeps other details",
                  expected="date change keeps service/area/budget/time; new search uses the new dates")
def test_03_correction_keeps_details(tmp_path, ev):
    sim = Sim(tmp_path)
    sim.say("Plumber in Indiranagar this week, mornings, under 1500")
    before = sim.contexts()[-1]
    sim.say("Actually, make it next week")
    req = sim.errand().request
    assert req.service == "plumber" and req.area == "Indiranagar" and req.budget_max == 1500
    assert req.time_window == ("06:00", "12:00") and req.date_range == ("2026-11-16", "2026-11-22")
    searches = sim.tool_calls("search_providers")
    assert len(searches) == 2 and searches[-1]["args"]["date_range"] == ["2026-11-16", "2026-11-22"]
    assert sim.events("request_updated")[-1]["changed"] == ["date_range"]
    assert "Which one" in sim.texts()[-1] and before["seq"] < searches[-1]["seq"]
    ev.actual = "only date_range changed; second search used 16–22 Nov"
    finish(sim, ev)


@pytest.mark.eval(id="27", title="'yeah ok' is not approval",
                  expected="0 executions; card dismissed, then a fresh card with a new code")
def test_27_reply_without_code_is_not_approval(tmp_path, ev):
    sim = Sim(tmp_path)
    sim.say("Plumber in Indiranagar this week, mornings, under 1500")
    first = sim.say("option 1")[-1]
    out = sim.say("yeah ok")
    assert out[-1]["kind"] == "card" and out[-1]["card"]["code"] != first["card"]["code"]
    assert not sim.agent.store.executions_for(sim.errand().id) and not sim.tool_calls("book")
    states = [(t["frm"], t["to"]) for t in sim.events("state_transition")]
    assert ("AWAITING_APPROVAL", "HOLDING") in states and states[-1] == ("HOLDING", "AWAITING_APPROVAL")
    assert "no longer applies" in sim.confirm(first)[0]["text"] and not sim.new_bookings()
    ev.actual = "no execution; card re-shown with new code; old code rejected"
    finish(sim, ev)


@pytest.mark.eval(id="13", title="Repeated confirmation code",
                  expected="1 execution, 1 booking; second CONFIRM gets 'already done'")
def test_13_repeated_confirmation(tmp_path, ev):
    sim = Sim(tmp_path)
    sim.scenario.book_outcomes = ["success", "success"]
    sim.book("Plumber in Indiranagar this week, mornings, under 1500")
    again = sim.confirm()
    assert again[0]["text"].startswith("That's already done")
    assert len(sim.agent.store.executions_for(sim.errand().id)) == 1 and len(sim.new_bookings()) == 1
    assert len(sim.tool_calls("book")) == 1
    ev.actual = "1 execution, 1 book call; repeat answered from saved outcome"
    finish(sim, ev)
