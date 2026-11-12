import pytest

from errand.agent import context
from errand.agent.memory import MemoryStore, variants
from errand.agent.store import Store
from errand.clock import FrozenClock
from errand.config import ContextConfig
from errand.db import DB
from errand.models import ErrandKind, MemorySource as Src, MemoryStatus as St, Request
from errand.seed.seed import forgotten_values, not_in_memory_block, seed_user


@pytest.fixture
def store(tmp_path):
    return Store(DB(tmp_path / "agent.db"), FrozenClock())


def keys(selection):
    return [m.key for m in selection.included]


def test_request_filling_memories_survive_the_cap(store):
    """Regression (BUGS.md #1): the cap used to keep the newest globals and drop preferred_area."""
    seed_user(store)
    for service in ("plumber", "electrician", "ac_servicing", "cleaner"):
        sel = MemoryStore(store).select("u1", service, 8)
        assert {"preferred_area", "earliest_time"} <= set(keys(sel)), service


def test_selection_excludes_every_unusable_status_with_a_reason(store):
    seed_user(store)
    sel = MemoryStore(store).select("u1", "plumber", 8)
    reasons = {r for _, r in sel.excluded}
    assert reasons == {"forgotten", "superseded", "unconfirmed_inference", "irrelevant_scope", "over_limit"}
    shown = {m.value for m in sel.included}
    assert not shown & set(not_in_memory_block()) and not shown & set(forgotten_values())


def test_correct_supersedes_and_confirm_replaces_older_value(store):
    mem = MemoryStore(store)
    old = mem.add("u1", "preferred_area", "HSR Layout", Src.EXPLICIT, St.ACTIVE)
    new = mem.correct("u1", "preferred_area", "Koramangala")
    assert mem.get(old.id).status == St.SUPERSEDED and mem.get(old.id).superseded_by == new.id
    cand = mem.add("u1", "preferred_area", "Indiranagar", Src.INFERRED, St.CANDIDATE)
    assert "preferred_area" not in [m.key for m in mem.select("u1", None, 8).included if m.value == "Indiranagar"]
    mem.confirm(cand.id, accept=True)
    assert [(m.value, m.source) for m in mem.select("u1", None, 8).included] == [("Indiranagar", Src.INFERRED)]


def test_forget_records_all_spellings():
    assert {"+91 98450 12345", "919845012345", "98450 12345", "9845012345"} <= set(variants("+91 98450 12345"))
    assert {"₹1,500", "1500", "1,500", "Rs 1500"} <= set(variants("₹1,500"))


def test_provenance_requires_the_user_to_have_said_it(store):
    mem = MemoryStore(store)
    sid = store.create_session("u1")
    store.add_message(sid, "user", "Plumber please, not before 10 a.m.")
    store.add_message(sid, "assistant", "Budget unlimited, noted!")  # assistant text never counts
    assert mem.stated_by_user(sid, "10:00") is not None
    assert mem.stated_by_user(sid, "unlimited") is None


def test_forgotten_value_is_scrubbed_from_current_session_history(store):
    mem = MemoryStore(store)
    sid = store.create_session("u1")
    store.add_message(sid, "user", "my gate code is 7734")
    mem.add("u1", "gate_code", "7734", Src.EXPLICIT, St.ACTIVE, evidence_session_id=sid)
    e = store.create_errand("u1", sid, ErrandKind.BOOK)
    assert "7734" in str(context.build(store, mem, e, []).messages)
    mem.forget("u1", "gate_code")
    store.add_message(sid, "user", "find a plumber")
    ctx = context.build(store, mem, e, [])
    assert "7734" not in str(ctx.messages) and ctx.meta["redaction_hits"] == 1


def test_budget_is_enforced_with_seed_and_long_history(store):
    seed_user(store)
    mem = MemoryStore(store)
    sid = store.create_session("u1")
    for i in range(150):
        store.add_message(sid, "user" if i % 2 == 0 else "assistant", f"message {i} " + "lorem ipsum " * 30)
    e = store.create_errand("u1", sid, ErrandKind.BOOK, Request(service="plumber"))
    big_tool_output = [{"role": "tool", "tool_call_id": f"c{i}", "content": "x" * 4000} for i in range(4)]
    ctx = context.build(store, mem, e, big_tool_output, ContextConfig(budget_tokens=6000))
    assert ctx.meta["est_tokens"] <= 6000
    assert ctx.meta["history_dropped"] > 0 and ctx.meta["history_kept"] >= 1
    assert ctx.messages[0]["role"] == "system" and "message 149" in ctx.messages[-5]["content"]


def test_rejected_inference_is_kept_out_but_not_scrubbed(store):
    """'remember no' used to redact the value like a forgotten fact, erasing e.g. every "10:00" slot from history."""
    mem = MemoryStore(store)
    sid = store.create_session("u1")
    store.add_message(sid, "assistant", "3. Ampere Home Care: ₹790, earliest 2026-11-11 10:00")
    m = mem.add("u1", "earliest_time", "10:00", Src.INFERRED, St.CANDIDATE, evidence_session_id=sid)
    assert mem.confirm(m.id, accept=False).status == St.REJECTED
    e = store.create_errand("u1", sid, ErrandKind.BOOK)
    ctx = context.build(store, mem, e, [])
    assert "10:00" in ctx.messages[1]["content"] and ctx.meta["redaction_hits"] == 0
    assert {"key": "earliest_time", "status": "rejected", "reason": "rejected_inference"} in ctx.meta["memories_excluded"]
    assert not mem.visible("u1") and mem.value("u1", "earliest_time") is None


def test_a_long_message_that_fits_is_not_cut(store):
    """Review: the up-front cut to a quarter of the budget used to apply even when everything fit."""
    sid = store.create_session("u1")
    text = "Plumber please. " + "The leak started last week under the sink. " * 150 + "Area: Koramangala, budget 900."
    store.add_message(sid, "user", text)
    e = store.create_errand("u1", sid, ErrandKind.BOOK)
    ctx = context.build(store, MemoryStore(store), e, [])
    assert ctx.meta["est_tokens"] <= 6000 and ctx.meta["latest_message_chars_cut"] == 0
    assert "Area: Koramangala, budget 900." in ctx.messages[-1]["content"]


def test_a_declined_inference_is_not_proposed_again(store):
    from errand.agent.internal_tools import propose_memory
    from errand.tracing.tracer import Tracer
    mem = MemoryStore(store)
    sid = store.create_session("u1")
    store.add_message(sid, "user", "Electrician tomorrow, not before 10:00")
    e = store.create_errand("u1", sid, ErrandKind.BOOK)
    args = {"key": "earliest_time", "value": "10:00", "explicit": False}
    assert propose_memory(mem, Tracer(), e, args).data["status"] == "candidate"
    mem.confirm(mem.current("u1", "earliest_time")[0].id, accept=False)
    assert propose_memory(mem, Tracer(), e, args).data["status"] == "declined_by_user"
    assert mem.next_prompt("u1", sid) is None and not mem.current("u1", "earliest_time")


def test_standing_preference_counts_from_any_message_in_the_session(store):
    from errand.agent.internal_tools import propose_memory
    from errand.tracing.tracer import Tracer
    mem = MemoryStore(store)
    sid = store.create_session("u1")
    store.add_message(sid, "user", "From now on my area is Indiranagar")
    store.add_message(sid, "user", "any Indiranagar plumbers tomorrow?")
    e = store.create_errand("u1", sid, ErrandKind.BOOK)
    r = propose_memory(mem, Tracer(), e, {"key": "preferred_area", "value": "Indiranagar", "explicit": True})
    assert r.data["status"] == "active" and r.data["source"] == "explicit"


def test_routing_prefers_a_recent_user_errand_but_not_an_abandoned_one(store):
    sid = store.create_session("u1")
    mine = store.create_errand("u1", sid, ErrandKind.BOOK)
    rebook = store.create_errand("u1", sid, ErrandKind.REBOOK)
    assert store.active_errand(sid).id == mine.id               # in the middle of their own errand
    store.clock.advance(11 * 60)
    store.save_errand(store.get_errand(rebook.id))              # the rebook is what the user is looking at now
    assert store.active_errand(sid).id == rebook.id             # the old errand was abandoned
