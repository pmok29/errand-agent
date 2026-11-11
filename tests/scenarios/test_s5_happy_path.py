"""S5 checkpoint: eval #1 (happy path) end to end, offline, through the real orchestrator.

S8 folds this into the scenario harness and adds the invariant checks.
"""
from errand.agent.factory import build_agent
from errand.agent.llm.scripted import ReactivePolicy
from errand.clock import FrozenClock
from errand.mocks.scenario import ScenarioConfig
from errand.mocks.service import MockProviderService
from errand.tracing.tracer import Tracer


def test_happy_path_booking(tmp_path):
    clock = FrozenClock()
    mock = MockProviderService(tmp_path / "mock.db", clock, ScenarioConfig(book_outcomes=["success"]))
    tracer = Tracer(tmp_path / "traces", clock)
    agent = build_agent(tmp_path / "agent.db", mock, ReactivePolicy(), clock, tracer)
    sid = agent.new_session()

    out = agent.handle_user_message(sid, "Find a plumber for my kitchen tap this week, mornings only, under ₹1,500.")
    assert len(out) == 1 and "area" in out[0]["text"].lower()
    assert "budget" not in out[0]["text"].lower() and "date" not in out[0]["text"].lower()  # only the missing detail

    out = agent.handle_user_message(sid, "Indiranagar")
    assert "1." in out[0]["text"] and "Which one" in out[0]["text"]

    out = agent.handle_user_message(sid, "Book option 1")
    card = out[-1]
    assert card["kind"] == "card" and "CONFIRM" in card["text"]
    draft = card["card"]["draft"]
    assert draft["price"] <= 1500 and "06:00" <= draft["slot"][11:16] < "12:00"

    out = agent.handle_user_message(sid, f"CONFIRM {card['card']['code']}")
    assert out[0]["kind"] == "template" and out[0]["text"].startswith("Booked:")

    new = [b for b in mock.ledger() if b["hold_id"]]
    assert len(new) == 1 and new[0]["slot"] == draft["slot"] and new[0]["price"] == draft["price"]

    tools = [e["tool"] for e in tracer.of_type("tool_call")]
    assert tools.index("get_booking_status") > tools.index("book")  # verified before success was reported
    assert tracer.of_type("injection_flagged")  # QuickFix's review was withheld from the model
    assert not tracer.of_type("claim_blocked")
    assert len(list((tmp_path / "traces" / sid).glob("*.jsonl"))) == 4  # one trace file per run
