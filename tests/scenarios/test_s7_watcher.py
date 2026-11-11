"""S7 checkpoint: proactive provider-cancel handling (#23, #29), UNVERIFIED follow-up (#28), approval expiry.

S8 folds these into the scenario harness with the global invariants.
"""
from errand.agent.factory import build_agent
from errand.agent.llm.scripted import ReactivePolicy
from errand.agent.watcher import Watcher
from errand.clock import FrozenClock
from errand.mocks.scenario import ScenarioConfig
from errand.mocks.service import MockProviderService
from errand.models import ErrandState as S, MemorySource as Src, MemoryStatus as St
from errand.seed.seed import seed_user
from errand.tracing.tracer import Tracer

REQUEST = "Plumber in Indiranagar this week, mornings, under 1500"


def setup(tmp_path, seed=True, **cfg):
    clock = FrozenClock()
    mock = MockProviderService(tmp_path / "mock.db", clock, ScenarioConfig(**cfg))
    tracer = Tracer(None, clock)
    agent = build_agent(tmp_path / "agent.db", mock, ReactivePolicy(), clock, tracer)
    if seed:
        seed_user(agent.store)
    return agent, Watcher(agent), mock, tracer, clock


def tools_called(tracer):
    return [e["tool"] for e in tracer.of_type("tool_call")]


def test_23_provider_cancel_is_detected_replaced_within_prefs_and_needs_fresh_approval(tmp_path):
    agent, watcher, mock, tracer, _ = setup(tmp_path, book_outcomes=["success"])
    sid = agent.new_session()
    assert watcher.tick() == []                       # baseline: history is not news
    mock.trigger_provider_cancel("b-cln-up")          # cleaner, Sun 15 Nov 10:00
    notes = watcher.tick()
    assert len(notes) == 1 and notes[0]["session_id"] == sid
    msgs = notes[0]["messages"]
    assert msgs[0]["text"].startswith("Heads up:") and "cleaner" in msgs[0]["text"]
    card = msgs[-1]
    assert card["kind"] == "card"
    draft = card["card"]["draft"]
    assert "2026-11-15" <= draft["slot"][:10] <= "2026-11-17"     # near the original date
    assert draft["slot"][11:16] >= "08:00"                         # stored earliest_time
    assert draft["price"] <= 1800                                  # no usable budget: original ₹1,500 + 20%
    detected = tracer.of_type("provider_cancel_detected")[0]
    assert detected["request"]["area"] == "Indiranagar"            # stored preferred_area
    assert "book" not in tools_called(tracer)                      # nothing booked without the user

    assert watcher.tick() == []                                    # no duplicate alert
    out = agent.handle_user_message(sid, f"CONFIRM {card['card']['code']}")
    assert out[0]["text"].startswith("Booked:")
    assert sum(1 for b in mock.ledger() if b["hold_id"]) == 1


def test_29_no_replacement_within_prefs_holds_nothing(tmp_path):
    agent, watcher, mock, tracer, _ = setup(tmp_path, seed=False)
    agent.memory.add("u1", "preferred_area", "Indiranagar", Src.EXPLICIT, St.ACTIVE)
    agent.memory.add("u1", "budget_cleaner", "100", Src.EXPLICIT, St.ACTIVE, "service:cleaner")
    agent.new_session()
    watcher.tick()
    mock.trigger_provider_cancel("b-cln-up")
    msgs = watcher.tick()[0]["messages"]
    assert "couldn't find a replacement" in msgs[-1]["text"]
    assert "hold_slot" not in tools_called(tracer) and "book" not in tools_called(tracer)


def test_cancellation_of_a_past_booking_is_logged_not_notified(tmp_path):
    agent, watcher, mock, tracer, _ = setup(tmp_path)
    agent.new_session()
    watcher.tick()
    mock.trigger_provider_cancel("b-past-cln")
    assert watcher.tick() == []
    assert tracer.of_type("provider_cancel_ignored")[0]["reason"] == "booking_in_past"


def confirm_booking(agent, sid):
    agent.handle_user_message(sid, REQUEST)
    card = agent.handle_user_message(sid, "option 1")[-1]
    return agent.handle_user_message(sid, f"CONFIRM {card['card']['code']}")


def test_28_unverified_is_rechecked_and_resolved(tmp_path):
    agent, watcher, mock, tracer, clock = setup(tmp_path, seed=False, book_outcomes=["timeout_booked"],
                                                status_pending_checks=4)
    sid = agent.new_session()
    assert confirm_booking(agent, sid)[0]["text"].startswith("The provider didn't confirm in time")
    assert watcher.tick() == []                      # schedules the first re-check at +30s
    clock.advance(29)
    assert watcher.tick() == []
    clock.advance(1)
    note = watcher.tick()[0]
    assert note["messages"][0]["text"].startswith("Update: Booked:")
    assert agent.store.latest_errand(sid).state == S.DONE
    assert tools_called(tracer).count("book") == 1 and sum(1 for b in mock.ledger() if b["hold_id"]) == 1
    assert watcher.tick() == []                      # job closed


def test_unverified_gives_up_after_the_capped_schedule(tmp_path):
    agent, watcher, mock, tracer, clock = setup(tmp_path, seed=False, book_outcomes=["timeout_booked"],
                                                status_pending_checks=100)
    sid = agent.new_session()
    confirm_booking(agent, sid)
    watcher.tick()
    notes = []
    for wait in (30, 90, 480):
        clock.advance(wait)
        notes += watcher.tick()
    assert len(notes) == 1 and "won't rebook automatically" in notes[0]["messages"][0]["text"]
    assert agent.store.latest_errand(sid).state == S.UNVERIFIED
    assert tools_called(tracer).count("book") == 1
    clock.advance(3600)
    assert watcher.tick() == []


def test_expired_approval_is_swept_and_its_code_rejected(tmp_path):
    agent, watcher, mock, _, clock = setup(tmp_path, seed=False)
    sid = agent.new_session()
    agent.handle_user_message(sid, REQUEST)
    card = agent.handle_user_message(sid, "option 1")[-1]
    clock.advance(601)
    watcher.tick()
    assert agent.store.latest_errand(sid).state == S.COMPARING
    out = agent.handle_user_message(sid, f"CONFIRM {card['card']['code']}")
    assert "expired" in out[0]["text"] and len(mock.ledger()) == 6
