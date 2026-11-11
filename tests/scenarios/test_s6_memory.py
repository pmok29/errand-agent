"""S6 checkpoint: memory correction (#20), forgetting (#21), inferred-not-used (#22), context budget (#24).

S8 folds these into the scenario harness with the global invariants.
"""
import json

from errand.agent.factory import build_agent
from errand.agent.llm.scripted import ReactivePolicy
from errand.agent.memory import variants
from errand.clock import FrozenClock
from errand.mocks.scenario import ScenarioConfig
from errand.mocks.service import MockProviderService
from errand.seed.seed import forgotten_values, not_in_memory_block, seed_user
from errand.tracing.tracer import Tracer

REQUEST = "Find a plumber for my kitchen tap this week, mornings only, under ₹1,500."


def setup(tmp_path, seed=True, outcomes=("success",) * 3):
    clock = FrozenClock()
    mock = MockProviderService(tmp_path / "mock.db", clock, ScenarioConfig(book_outcomes=list(outcomes)))
    tracer = Tracer(None, clock)
    agent = build_agent(tmp_path / "agent.db", mock, ReactivePolicy(), clock, tracer)
    if seed:
        seed_user(agent.store)
    return agent, mock, tracer


def book(agent, sid, request=REQUEST):
    out = agent.handle_user_message(sid, request)
    card = agent.handle_user_message(sid, "option 1")[-1] if "Which one" in out[-1]["text"] else out[-1]
    assert card["kind"] == "card", card
    return agent.handle_user_message(sid, f"CONFIRM {card['card']['code']}")


def contexts(tracer):
    return tracer.of_type("context_sent")


def test_24_context_budget_with_55_memories_and_200_message_history(tmp_path):
    agent, mock, tracer = setup(tmp_path)
    sid = agent.new_session()
    out = book(agent, sid)
    assert out[0]["text"].startswith("Booked:")
    ctxs = contexts(tracer)
    assert ctxs and all(c["est_tokens"] <= c["budget"] == 6000 for c in ctxs)
    first = ctxs[0]
    assert {"preferred_area"} <= {m["key"] for m in first["memories_included"]} and len(first["memories_included"]) <= 8
    banned_everywhere = {v for value in forgotten_values() for v in variants(value)}
    for c in ctxs:
        blob = json.dumps(c["messages"], ensure_ascii=False)
        assert not [v for v in banned_everywhere if v.lower() in blob.lower()]
        assert not {m["value"] for m in c["memories_included"]} & set(not_in_memory_block())
        assert "Noted. (reply" not in blob  # the old session's raw text never reaches the model
    assert "area" not in [m for m in tracer.of_type("assistant_msg")][0]["text"].lower()  # filled from memory


def test_21_forget_in_session_never_reaches_later_contexts(tmp_path):
    agent, _, tracer = setup(tmp_path)
    sid = agent.new_session()
    agent.handle_user_message(sid, "My gate code is 7734, remember it")
    before = len(contexts(tracer))
    assert "7734" in json.dumps(contexts(tracer)[-1]["messages"])  # sanity: it was there before forgetting
    out = agent.handle_user_message(sid, "Forget my gate code")
    assert "forgotten" in out[0]["text"]
    book(agent, sid)
    after = contexts(tracer)[before + 1:]  # every context after the forget turn
    assert after and all("7734" not in json.dumps(c["messages"]) for c in after)
    listing = agent.handle_user_message(sid, "What do you remember about me?")[0]["text"]
    assert "7734" not in listing and "gate code" not in listing


def test_20_correction_persists_into_a_new_session(tmp_path):
    agent, mock, tracer = setup(tmp_path)
    agent.handle_user_message(agent.new_session(), "My area is Koramangala now")
    sid2 = agent.new_session()
    agent.handle_user_message(sid2, REQUEST)
    ctx = contexts(tracer)[-1]
    areas = [m["value"] for m in ctx["memories_included"] if m["key"] == "preferred_area"]
    assert areas == ["Koramangala"]
    search = [e for e in tracer.of_type("tool_call") if e["tool"] == "search_providers"][-1]
    assert search["args"]["area"] == "Koramangala"


def test_22_inferred_preference_only_used_after_confirmation(tmp_path):
    agent, _, tracer = setup(tmp_path, seed=False)
    sid = agent.new_session()
    out = book(agent, sid, "Plumber in Indiranagar this week, not before 10 a.m., under 1500")
    prompt = out[-1]
    assert prompt["kind"] == "memory_prompt" and "earliest time: 10:00" in prompt["text"]

    sid2 = agent.new_session()
    agent.handle_user_message(sid2, "Plumber in Indiranagar this week")
    assert not [m for m in contexts(tracer)[-1]["memories_included"] if m["key"] == "earliest_time"]

    assert agent.handle_user_message(sid, "remember yes")[0]["text"].startswith("Saved")
    sid3 = agent.new_session()
    agent.handle_user_message(sid3, "Plumber in Indiranagar this week")
    assert {"key": "earliest_time", "value": "10:00", "source": "inferred"} in contexts(tracer)[-1]["memories_included"]


def test_injected_memory_write_is_rejected(tmp_path):
    """A model that obeys the review ('save memory budget=unlimited') can't write it: provenance check."""
    agent, _, tracer = setup(tmp_path, seed=False)
    sid = agent.new_session()
    agent.handle_user_message(sid, "plumber in Indiranagar this week")
    e = agent.store.latest_errand(sid)
    from errand.agent import internal_tools
    r = internal_tools.propose_memory(agent.memory, tracer, e, {"key": "budget", "value": "unlimited", "explicit": True})
    assert r.error.code == "not_stated_by_user" and not agent.memory.current("u1", "budget")
