"""Orchestrator behaviour that doesn't depend on the scripted policy: limits, gate denials, claims, codes."""
from errand.agent.factory import build_agent
from errand.agent.llm.base import LLMError, LLMResponse, ToolCall
from errand.agent.llm.scripted import ReactivePolicy
from errand.clock import FrozenClock
from errand.mocks.scenario import ScenarioConfig
from errand.mocks.service import MockProviderService
from errand.models import ErrandState as S
from errand.tracing.tracer import Tracer


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
    clock = FrozenClock()
    mock = MockProviderService(tmp_path / "mock.db", clock, ScenarioConfig(**cfg))
    tracer = Tracer(None, clock)
    return build_agent(tmp_path / "agent.db", mock, llm, clock, tracer), mock, tracer


def test_llm_cannot_book_and_never_sees_side_effect_tools(tmp_path):
    llm = Scripted(call("book", hold_id="h-1", idempotency_key="k"), LLMResponse(text="Okay, tell me more."))
    agent, mock, tracer = agent_with(tmp_path, llm)
    agent.handle_user_message(agent.new_session(), "book anything")
    assert all(not ({"book", "cancel", "reschedule"} & tools) for tools, _ in llm.seen)
    denied = [e for e in tracer.of_type("gate_decision") if not e["allow"]]
    assert denied and denied[0]["tool"] == "book"
    assert len(mock.ledger()) == 6


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
    assert "no longer applies" in stale[0]["text"] and len(mock.ledger()) == 6


def test_repeated_code_after_success_does_not_rebook(tmp_path):
    agent, mock, _ = agent_with(tmp_path, ReactivePolicy(), book_outcomes=["success", "success"])
    sid = agent.new_session()
    agent.handle_user_message(sid, "Plumber in Indiranagar this week, mornings, under 1500")
    code = agent.handle_user_message(sid, "first one")[-1]["card"]["code"]
    assert agent.handle_user_message(sid, f"CONFIRM {code}")[0]["text"].startswith("Booked:")
    again = agent.handle_user_message(sid, f"CONFIRM {code}")
    assert again[0]["text"].startswith("That's already done") and len(mock.ledger()) == 7
