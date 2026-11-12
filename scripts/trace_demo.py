"""Records the end-to-end trace for the design note: a real-LLM booking where `book` times out after
the provider actually booked it (preset timeout_booked).

    .venv/bin/python scripts/trace_demo.py [--provider groq]

Writes traces/timeout_after_success.jsonl (every run of the conversation, in order) and prints the
key lines for the annotated excerpt.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smoke_llm  # noqa: E402  (also loads .env and puts the repo on sys.path)

OUT = smoke_llm.ROOT / "traces" / "timeout_after_success.jsonl"
KEY_TYPES = {"user_msg", "llm_call", "tool_call", "state_transition", "approval_created", "approval_consumed",
             "gate_decision", "assistant_msg", "llm_fallback", "claim_blocked", "injection_flagged"}


def summary(e: dict) -> str:
    t = e["type"]
    if t == "llm_call":
        calls = ", ".join(c["name"] for c in e["tool_calls"]) or f"text: {(e.get('text') or '')[:60]!r}"
        return f"{e['model']} in={e['input_tokens']} out={e['output_tokens']} {round(e['latency_ms'])}ms → {calls}"
    if t == "tool_call":
        res = e.get("result") or {}
        out = "TIMEOUT" if e.get("timed_out") else ("ok" if e.get("ok", res.get("ok")) else "error")
        data = res.get("data") or {}
        extra = f" {data.get('status', '')} {data.get('booking_id', '')}".rstrip() if e["tool"] == "get_booking_status" else ""
        return f"{e['actor']}: {e['tool']}({json.dumps(e['args'], ensure_ascii=False)[:70]}) → {out}{extra}"
    if t == "state_transition":
        return f"{e['frm']} → {e['to']} ({e['reason']})"
    if t == "gate_decision":
        return f"{e['actor']} {e['tool']} {'allow' if e['allow'] else 'DENY: ' + e['reason']}"
    if t in ("approval_created", "approval_consumed"):
        return " ".join(f"{k}={e[k]}" for k in ("code", "attempt_no", "idem_key") if k in e)
    return (e.get("text") or e.get("message") or e.get("original") or "")[:110].replace("\n", " ")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=None)
    a = ap.parse_args()
    work = smoke_llm.ROOT / "traces" / "demo_work"
    result = smoke_llm.run("1", a.provider, "timeout_booked", work)
    files = sorted((work / "traces" / result["session"]).glob("*.jsonl"))
    events = sorted((json.loads(line) for f in files for line in f.open()), key=lambda e: e["seq"])
    OUT.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events))
    print(f"\nwrote {OUT.relative_to(smoke_llm.ROOT)}: {len(events)} events, {len(files)} runs\n")
    timeout = any(e["type"] == "tool_call" and e["tool"] == "book" and e.get("timed_out") for e in events)
    for e in events:
        if e["type"] in KEY_TYPES and not (e["type"] == "gate_decision" and e["actor"] == "llm" and e["allow"]):
            print(f"{e['seq']:>4} {e['type']:<17} {summary(e)}")
    ok = bool(result["outcome"]) and timeout and result["bookings"] == 1
    print("\nOK: timeout after success, verified, one booking" if ok else "\nNOT OK", result)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
