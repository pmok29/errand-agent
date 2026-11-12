"""Changed terms (#7, #8), timeouts (#9, #10), tool failures (#11, #17), slot conflict (#15), restart (#14a)."""
import pytest

from errand.agent.executor import SimulatedCrash
from errand.mocks.scenario import ScenarioConfig
from errand.models import ApprovalStatus, ErrandState as S
from tests.harness import Sim, finish

REQUEST = "Plumber in Indiranagar this week, mornings, under 1500"


def keys(sim, tool):
    return [c["args"]["idempotency_key"] for c in sim.tool_calls(tool, actor="executor")]


@pytest.mark.eval(id="7", title="Price drift at hold time",
                  expected="agent asks before a card above budget; card shows the held price; no approval at the search price")
def test_07_price_drift(tmp_path, ev):
    sim = Sim(tmp_path, ScenarioConfig(price_drift={"p-plb-03": 500}, book_outcomes=["success"]))
    sim.say(REQUEST)
    out = sim.say("option 1")                                         # QuickFix: searched at ₹1,150
    assert out[-1]["kind"] == "chat" and "₹1,650" in out[-1]["text"] and "budget" in out[-1]["text"]
    assert not sim.agent.store.approvals_for(sim.errand().id)        # nothing to confirm yet
    card = sim.say("yes, go ahead")[-1]
    assert card["card"]["draft"]["price"] == 1650 and "above your ₹1,500 budget" in card["text"]
    sim.confirm(card)
    [b] = sim.new_bookings()
    prices = {a.draft.price for a in sim.agent.store.approvals_for(sim.errand().id)}
    assert b["price"] == 1650 and prices == {1650}
    ev.actual = "asked at ₹1,650 (search ₹1,150) → card ₹1,650 marked over budget → booked at ₹1,650"
    finish(sim, ev)


@pytest.mark.eval(id="8", title="Slot change after the card, before the code",
                  expected="old approval invalidated; new card with a different hash and code; old code rejected")
def test_08_slot_change_after_card(tmp_path, ev):
    sim = Sim(tmp_path)
    sim.scenario.book_outcomes = ["success"]
    sim.say(REQUEST)
    first = sim.say("option 1")[-1]
    held = sim.agent.store.latest_hold(sim.errand().id)
    option = next(o for o in sim.errand().selected["options"] if o["provider_id"] == held.provider_id)
    other = next(s for s in option["slots"] if s[11:16] != held.slot[11:16])
    second = sim.say(f"Actually, can we do {other[11:16]} instead?")[-1]
    assert second["kind"] == "card" and second["card"]["draft"]["slot"][11:16] == other[11:16]
    old, new = sim.agent.store.approvals_for(sim.errand().id)
    assert old.status == ApprovalStatus.INVALIDATED and old.hash != new.hash and old.code != new.code
    assert "no longer applies" in sim.confirm(first)[0]["text"] and not sim.new_bookings()
    sim.confirm(second)
    assert [b["slot"] for b in sim.new_bookings()] == [second["card"]["draft"]["slot"]]
    ev.actual = f"{held.slot[11:16]} → {other[11:16]}: old code rejected, booked the new slot only"
    finish(sim, ev)


@pytest.mark.eval(id="9", title="Book times out but the provider booked it",
                  expected="status checked; 0 book retries; 1 booking; 'Booked' template")
def test_09_timeout_but_booked(tmp_path, ev):
    sim = Sim(tmp_path, "timeout_booked")
    out = sim.book(REQUEST)
    book_calls = sim.tool_calls("book", actor="executor")
    assert book_calls[0]["timed_out"] and len(book_calls) == 1
    assert sim.tool_calls("get_booking_status")[0]["result"]["data"]["status"] == "booked"
    assert out[0]["text"].startswith("Booked:") and len(sim.new_bookings()) == 1
    ev.actual = "book timed out → get_booking_status=booked → 'Booked'; 1 book call, 1 booking"
    finish(sim, ev)


@pytest.mark.eval(id="10", title="Book times out and was not booked",
                  expected="retried with the same key; 1 booking; one key in the provider's ledger")
def test_10_timeout_not_booked(tmp_path, ev):
    sim = Sim(tmp_path, ScenarioConfig(book_outcomes=["timeout_not_booked", "success"]))
    out = sim.book(REQUEST)
    ks = keys(sim, "book")
    assert len(ks) == 2 and len(set(ks)) == 1
    assert [k["key"] for k in sim.mock.idem_keys() if k["action"] == "book"] == ks[:1]
    assert out[0]["text"].startswith("Booked:") and len(sim.new_bookings()) == 1
    ev.actual = "timeout → status not_booked → retry with the same key → booked once"
    finish(sim, ev)


@pytest.mark.eval(id="11", title="slot_taken at book time",
                  expected="nothing booked; user picks again; new approval; attempt_no 2 with a new key")
def test_11_slot_taken(tmp_path, ev):
    sim = Sim(tmp_path, ScenarioConfig(book_outcomes=["slot_taken", "success"]))
    out = sim.book(REQUEST)
    assert "taken" in out[0]["text"] and not sim.new_bookings() and sim.errand().state == S.COMPARING
    card = sim.say("option 2")[-1]
    assert card["kind"] == "card"
    sim.confirm(card)
    exes = sim.agent.store.executions_for(sim.errand().id)
    assert [x.attempt_no for x in exes] == [1, 2] and exes[0].idem_key != exes[1].idem_key
    assert len(sim.new_bookings()) == 1 and len(sim.agent.store.approvals_for(sim.errand().id)) == 2
    ev.actual = "slot_taken → re-pick → second approval → attempt 2, new key → 1 booking"
    finish(sim, ev)


@pytest.mark.eval(id="14a", title="Agent crashes after book was sent, before the reply",
                  expected="restart → recovery asks the provider with the persisted key; 1 booking; user notified")
def test_14a_restart_mid_booking(tmp_path, ev):
    sim = Sim(tmp_path, "timeout_booked", faults={"book_returned"})
    sim.say(REQUEST)
    sim.say("option 1")
    with pytest.raises(SimulatedCrash):
        sim.confirm()
    exe = sim.agent.store.executions_for(sim.errand().id)[0]
    assert exe.state == "EXECUTING"                                   # the write-ahead record survived
    outcomes = sim.restart()
    assert [o.kind for o in outcomes] == ["booked"]
    after = [c["tool"] for c in sim.tool_calls(actor="executor")]
    assert after[after.index("book") + 1] == "get_booking_status" and after.count("book") == 1
    [note] = sim.agent.store.take_notifications(sim.sid)
    assert note["messages"][0]["text"].startswith("Booked:") and len(sim.new_bookings()) == 1
    ev.actual = "crash after book → restart → status by persisted key → booked; 1 booking; notification queued"
    finish(sim, ev, crashes=1)


@pytest.mark.eval(id="15", title="Reschedule hits a slot conflict, then another slot",
                  expected="conflict reported; second attempt has attempt_no 2 and a new key; no payload mismatch")
def test_15_reschedule_conflict(tmp_path, ev):
    sim = Sim(tmp_path, "slot_conflict")
    out = sim.say("Move my AC service to Saturday")
    assert "14:00" in out[-1]["text"] and "Which time" in out[-1]["text"]
    first = sim.say("14:00")[-1]
    assert first["kind"] == "card" and "Wed 11 Nov, 11:00" in first["text"]
    out = sim.confirm(first)
    assert "isn't available" in out[0]["text"] and sim.errand().state == S.COMPARING
    second = sim.say("15:00 then")[-1]
    out = sim.confirm(second)
    assert out[0]["text"].startswith("Rescheduled:")
    exes = sim.agent.store.executions_for(sim.errand().id)
    assert [x.attempt_no for x in exes] == [1, 2] and len(set(keys(sim, "reschedule"))) == 2
    assert sim.booking("b-ac-wed")["slot"] == "2026-11-14T15:00"
    ev.actual = "14:00 → slot_conflict; 15:00 → attempt 2, new key → rescheduled once"
    finish(sim, ev)


@pytest.mark.eval(id="17", title="Cancel gets a 503, then succeeds",
                  expected="retried with the same key; cancelled; verified with list_bookings")
def test_17_cancel_503_then_ok(tmp_path, ev):
    sim = Sim(tmp_path)                                               # cancel_fail_first is the default
    card = sim.say("Cancel my plumber booking")[-1]
    assert card["kind"] == "card" and card["card"]["draft"]["booking_id"] == "b-plb-up"
    out = sim.confirm(card)
    ks = keys(sim, "cancel")
    assert len(ks) == 2 and len(set(ks)) == 1 and out[0]["text"].startswith("Cancelled:")
    calls = [c["tool"] for c in sim.tool_calls(actor="executor")]
    assert calls[-1] == "list_bookings"
    assert sim.booking("b-plb-up")["status"] == "cancelled"
    ev.actual = "503 → same-key retry → cancelled; confirmed via list_bookings"
    finish(sim, ev)


@pytest.mark.eval(id="32", title="Asking again while the last booking is unconfirmed", category="stretch",
                  expected="no second hold or booking while the provider is being re-checked; watcher reports the first")
def test_32_no_second_booking_while_the_first_is_unverified(tmp_path, ev):
    sim = Sim(tmp_path, ScenarioConfig(book_outcomes=["timeout_booked", "success"], status_pending_checks=4))
    sim.say(REQUEST)
    assert "can't confirm yet" in sim.confirm(sim.say("option 1")[-1])[0]["text"]
    sim.say(REQUEST)
    out = sim.say("option 1")
    assert out[-1]["kind"] != "card" and "still being checked" in out[-1]["text"]
    assert len([c for c in sim.tool_calls("hold_slot") if c["ok"]]) == 1 and len(sim.new_bookings()) == 1
    assert sim.tick() == []                                     # schedules the re-check at +30 s
    sim.clock.advance(31)
    [note] = sim.tick()
    assert note["messages"][0]["text"].startswith("Update: Booked:") and len(sim.new_bookings()) == 1
    ev.actual = "second hold denied while UNVERIFIED; watcher then confirmed the first; 1 booking"
    finish(sim, ev)
