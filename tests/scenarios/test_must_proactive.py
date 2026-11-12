"""Proactive: provider cancels (#23); stretch: UNVERIFIED follow-up (#28), no replacement fits (#29)."""
import pytest

from errand.mocks.scenario import ScenarioConfig
from errand.models import ErrandState as S, MemorySource as Src, MemoryStatus as St
from tests.harness import Sim, finish

REQUEST = "Plumber in Indiranagar this week, mornings, under 1500"


@pytest.mark.eval(id="23", title="Provider cancels an upcoming booking",
                  expected="user notified; replacement fits stored prefs; 0 book calls before CONFIRM; booked after")
def test_23_provider_cancel(tmp_path, ev):
    sim = Sim(tmp_path, ScenarioConfig(book_outcomes=["success"]), seed=True)
    assert sim.tick() == []                                     # baseline: history is not news
    sim.mock.trigger_provider_cancel("b-cln-up")                # cleaner, Sun 15 Nov 10:00, ₹1,500
    [note] = sim.tick()
    heads_up, card = note["messages"][0], note["messages"][-1]
    assert note["session_id"] == sim.sid and heads_up["text"].startswith("Heads up:")
    draft = card["card"]["draft"]
    assert "2026-11-15" <= draft["slot"][:10] <= "2026-11-17" and draft["slot"][11:16] >= "08:00"
    assert draft["price"] <= 1800                               # no usable budget: original + 20%
    assert sim.events("provider_cancel_detected")[0]["request"]["area"] == "Indiranagar"
    assert not sim.tool_calls("book", actor="executor") and sim.tick() == []
    out = sim.confirm(card)
    assert out[0]["text"].startswith("Booked:") and len(sim.new_bookings()) == 1
    ev.actual = f"notified; card {draft['provider_name']} {draft['slot'].replace('T', ' ')} ₹{draft['price']:,}; booked only after CONFIRM"
    finish(sim, ev)


@pytest.mark.eval(id="28", title="UNVERIFIED booking resolved by the watcher", category="stretch",
                  expected="status rechecked at +30s; resolves to DONE; user told; still 1 book call")
def test_28_unverified_is_rechecked(tmp_path, ev):
    sim = Sim(tmp_path, ScenarioConfig(book_outcomes=["timeout_booked"], status_pending_checks=4))
    assert sim.book(REQUEST)[0]["text"].startswith("I can't confirm yet whether the booking")
    assert sim.tick() == []                                     # schedules the first recheck
    sim.clock.advance(29)
    assert sim.tick() == []
    sim.clock.advance(1)
    [note] = sim.tick()
    assert note["messages"][0]["text"].startswith("Update: Booked:") and sim.errand().state == S.DONE
    assert len(sim.tool_calls("book", actor="executor")) == 1 and len(sim.new_bookings()) == 1
    assert sim.tick() == []
    ev.actual = "UNVERIFIED → recheck at +30s → DONE, 'Update: Booked'; 1 book call"
    finish(sim, ev)


@pytest.mark.eval(id="29", title="Provider cancels and nothing fits the preferences", category="stretch",
                  expected="user told no replacement fits; nothing held or booked")
def test_29_no_replacement(tmp_path, ev):
    sim = Sim(tmp_path)
    sim.agent.memory.add("u1", "preferred_area", "Indiranagar", Src.EXPLICIT, St.ACTIVE)
    sim.agent.memory.add("u1", "budget_cleaner", "100", Src.EXPLICIT, St.ACTIVE, "service:cleaner")
    sim.tick()
    sim.mock.trigger_provider_cancel("b-cln-up")
    [note] = sim.tick()
    assert "couldn't find a replacement" in note["messages"][-1]["text"]
    assert not sim.tool_calls("hold_slot") and not sim.tool_calls("book")
    ev.actual = "no option within ₹100 → told the user; 0 holds, 0 bookings"
    finish(sim, ev)


def test_cancellation_of_a_past_booking_is_logged_not_notified(tmp_path):
    sim = Sim(tmp_path)
    sim.tick()
    sim.mock.trigger_provider_cancel("b-past-cln")
    assert sim.tick() == []
    assert sim.events("provider_cancel_ignored")[0]["reason"] == "booking_in_past"


def test_unverified_gives_up_after_the_capped_schedule(tmp_path):
    sim = Sim(tmp_path, ScenarioConfig(book_outcomes=["timeout_booked"], status_pending_checks=100))
    sim.book(REQUEST)
    sim.tick()
    notes = []
    for wait in (30, 90, 480):
        sim.clock.advance(wait)
        notes += sim.tick()
    assert len(notes) == 1 and "won't rebook automatically" in notes[0]["messages"][0]["text"]
    assert sim.errand().state == S.UNVERIFIED and len(sim.tool_calls("book", actor="executor")) == 1
    sim.clock.advance(3600)
    assert sim.tick() == []


def test_expired_approval_is_swept_and_its_code_rejected(tmp_path):
    sim = Sim(tmp_path)
    sim.say(REQUEST)
    card = sim.say("option 1")[-1]
    sim.clock.advance(601)
    sim.tick()
    assert sim.errand().state == S.COMPARING
    assert "expired" in sim.confirm(card)[0]["text"] and not sim.new_bookings()


def test_a_failed_cancellation_handler_is_retried_on_the_next_tick(tmp_path, monkeypatch):
    sim = Sim(tmp_path, seed=True)
    sim.tick()
    sim.mock.trigger_provider_cancel("b-cln-up")
    monkeypatch.setattr(sim.watcher, "_handle_provider_cancel", lambda user, b: 1 / 0)
    assert sim.tick() == [] and sim.events("error")[-1]["where"] == "provider_cancel"
    monkeypatch.undo()
    [note] = sim.tick()                               # the snapshot wasn't advanced, so it's picked up now
    assert note["messages"][0]["text"].startswith("Heads up:")


def test_a_new_users_first_booking_cancellation_is_not_swallowed(tmp_path):
    sim = Sim(tmp_path)
    sim.tick()                                        # baseline for u1, who has seed bookings
    sim.agent.new_session("u2")
    sim.tick()                                        # u2's first look: no bookings yet
    sim.mock.conn.execute("INSERT INTO bookings VALUES ('b-u2', 'u2', 'p-cln-03', 'cleaner', '2026-11-15T10:00', "
                          "1500, 'confirmed', NULL)")
    sim.mock.trigger_provider_cancel("b-u2")
    assert [n["session_id"] for n in sim.tick()] == [sim.agent.store.latest_session("u2")]
