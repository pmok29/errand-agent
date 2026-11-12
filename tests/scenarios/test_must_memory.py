"""Memory: correction (#20), forgetting (#21), inferred-not-used (#22), context budget (#24)."""
import json

import pytest

from errand.agent.memory import variants
from errand.mocks.scenario import ScenarioConfig
from errand.seed.seed import forgotten_values, not_in_memory_block
from tests.harness import PLUMBER, Sim, finish


def seeded(tmp_path, seed=True):
    return Sim(tmp_path, ScenarioConfig(book_outcomes=["success"] * 3), seed=seed)


@pytest.mark.eval(id="20", title="Memory correction: 'My area is Koramangala now'",
                  expected="old value superseded; next session's context and search use Koramangala")
def test_20_correction_persists_into_a_new_session(tmp_path, ev):
    sim = seeded(tmp_path)
    sim.say("My area is Koramangala now")
    sim.new_session()
    sim.say(PLUMBER)
    areas = [m["value"] for m in sim.contexts()[-1]["memories_included"] if m["key"] == "preferred_area"]
    assert areas == ["Koramangala"]
    assert sim.tool_calls("search_providers")[-1]["args"]["area"] == "Koramangala"
    old = [m for m in sim.agent.memory.all("u1") if m.key == "preferred_area" and m.value == "Indiranagar"]
    assert old[0].status == "superseded"
    ev.actual = "Indiranagar superseded; new session searched Koramangala without asking"
    finish(sim, ev)


@pytest.mark.eval(id="21", title="Forget: 'Forget my gate code', then book",
                  expected="value redacted; absent from every later context, including the turn where it was said")
def test_21_forget_never_reaches_later_contexts(tmp_path, ev):
    sim = seeded(tmp_path)
    sim.say("My gate code is 7734, remember it")
    assert "7734" in json.dumps(sim.contexts()[-1]["messages"])   # sanity: present before forgetting
    before = len(sim.contexts())
    assert "forgotten" in sim.say("Forget my gate code")[0]["text"]
    sim.book()
    after = sim.contexts()[before + 1:]
    assert after and all("7734" not in json.dumps(c["messages"]) for c in after)
    listing = sim.say("What do you remember about me?")[0]["text"]
    assert "7734" not in listing and "gate code" not in listing
    hits = sum(c["redaction_hits"] for c in after)
    ev.actual = f"0 of {len(after)} later contexts contain it ({hits} history mentions scrubbed); not listed"
    finish(sim, ev)


@pytest.mark.eval(id="22", title="Inferred preference not used until confirmed",
                  expected="'not before 10am' stored as a candidate; absent from context until 'remember yes'")
def test_22_inferred_preference_only_used_after_confirmation(tmp_path, ev):
    sim = seeded(tmp_path, seed=False)
    first = sim.sid
    out = sim.book("Plumber in Indiranagar this week, not before 10 a.m., under 1500")
    assert out[-1]["kind"] == "memory_prompt" and "earliest time: 10:00" in out[-1]["text"]
    sim.new_session()
    sim.say("Plumber in Indiranagar this week")
    assert not [m for m in sim.contexts()[-1]["memories_included"] if m["key"] == "earliest_time"]
    assert sim.say("remember yes", session_id=first)[0]["text"].startswith("Saved")
    sim.new_session()
    sim.say("Plumber in Indiranagar this week")
    assert {"key": "earliest_time", "value": "10:00", "source": "inferred"} in sim.contexts()[-1]["memories_included"]
    ev.actual = "candidate excluded from the next session; included only after 'remember yes'"
    finish(sim, ev)


@pytest.mark.eval(id="24", title="Context budget: 55 memories + 200-message history",
                  expected="every context ≤ 6,000 tokens, ≤ 8 memories; no forgotten/candidate/superseded value anywhere")
def test_24_context_budget(tmp_path, ev):
    sim = seeded(tmp_path)
    assert sim.book()[0]["text"].startswith("Booked:")
    ctxs = sim.contexts()
    assert ctxs and all(c["est_tokens"] <= c["budget"] == 6000 for c in ctxs)
    assert all(len(c["memories_included"]) <= 8 for c in ctxs)
    assert "preferred_area" in {m["key"] for m in ctxs[0]["memories_included"]}
    banned = {v for value in forgotten_values() for v in variants(value)}
    for c in ctxs:
        blob = json.dumps(c["messages"], ensure_ascii=False).lower()
        assert not [v for v in banned if v.lower() in blob]
        assert not {m["value"] for m in c["memories_included"]} & set(not_in_memory_block())
        assert "Noted. (reply" not in blob                     # earlier sessions never sent raw
    assert "area" not in sim.events("assistant_msg")[0]["text"].lower()   # filled from memory
    peak = max(c["est_tokens"] for c in ctxs)
    ev.actual = f"{len(ctxs)} contexts, peak {peak:,} tokens, ≤ 8 memories; area filled from memory"
    finish(sim, ev)


@pytest.mark.eval(id="33", title="Context budget with a very long user message", category="stretch",
                  expected="every context ≤ 6,000 tokens; the message tail is cut and logged; the request still works")
def test_33_long_message_is_cut_to_the_budget(tmp_path, ev):
    sim = seeded(tmp_path)
    out = sim.say(PLUMBER + " Some background about the leak." * 1200)
    assert "Which one" in out[-1]["text"]                      # the request at the head survived
    ctxs = sim.contexts()
    assert all(c["est_tokens"] <= c["budget"] and not c["over_budget"] for c in ctxs)
    assert all(c["latest_message_chars_cut"] > 0 for c in ctxs)
    ev.actual = (f"{len(ctxs)} contexts, peak {max(c['est_tokens'] for c in ctxs):,} tokens; "
                 f"{ctxs[0]['latest_message_chars_cut']:,} chars cut from the message")
    finish(sim, ev)
