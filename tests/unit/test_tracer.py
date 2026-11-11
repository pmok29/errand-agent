import json

from errand.clock import FrozenClock
from errand.tracing.cost import estimate
from errand.tracing.tracer import Tracer, sanitize


def test_run_writes_jsonl_with_ordered_events_and_totals(tmp_path):
    t = Tracer(tmp_path, FrozenClock())
    run = t.start_run("s1")
    t.emit("tool_call", tool="search_providers", args={"service": "plumber"})
    t.llm_call("openai/gpt-oss-120b", 120.0, input_tokens=2000, output_tokens=300)
    totals = t.end_run("done")

    lines = [json.loads(l) for l in (tmp_path / "s1" / f"{run}.jsonl").read_text().splitlines()]
    assert [e["type"] for e in lines] == ["run_start", "tool_call", "llm_call", "run_end"]
    assert [e["seq"] for e in lines] == [1, 2, 3, 4] and all(e["run_id"] == run for e in lines)
    assert totals["tool_calls"] == 1 and totals["input_tokens"] == 2000 and totals["cost_known"]
    assert totals["cost_usd"] == estimate("openai/gpt-oss-120b", 2000, 300)["usd"]


def test_unknown_model_cost_is_null_and_marks_totals(tmp_path):
    t = Tracer(None, FrozenClock())
    t.start_run("s1")
    e = t.llm_call("qwen/qwen3.8-27b", 50.0, 100, 10)
    assert e["cost"] is None and t.end_run("done")["cost_known"] is False


def test_local_model_costs_zero():
    assert estimate("qwen3:4b", 10_000, 1_000)["usd"] == 0.0


def test_pii_is_masked_everywhere():
    data = {"note": "call me on +91 98765 43210 or 9876543210", "who": ["me@example.com"],
            "slot": "2026-11-12T09:00", "price": 1350, "booking": "b-1a2b3c4d"}
    out = sanitize(data)
    assert out["note"] == "call me on [phone] or [phone]" and out["who"] == ["[email]"]
    assert out["slot"] == data["slot"] and out["price"] == 1350 and out["booking"] == data["booking"]


def test_ids_containing_digit_runs_are_not_masked():
    data = {"idempotency_key": "a9876543210f4c2e9b1d0e7a6c5b4a39", "hold": "h-9876543210",
            "booking": "9876543210-x"}
    assert sanitize(data) == data


def test_subscribers_get_events_and_failures_are_isolated():
    t = Tracer()
    seen = []
    t.subscribe(lambda e: 1 / 0)
    unsubscribe = t.subscribe(seen.append)
    t.emit("state_transition", frm="A", to="B")
    unsubscribe()
    t.emit("state_transition", frm="B", to="C")
    assert [e["to"] for e in seen] == ["B"]


def test_events_outside_a_run_stay_in_memory(tmp_path):
    t = Tracer(tmp_path)
    t.emit("recovery_resume", execution_id="x1")
    assert t.of_type("recovery_resume")[0]["run_id"] is None
    assert list(tmp_path.iterdir()) == []
