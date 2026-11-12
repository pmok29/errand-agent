"""Real-LLM smoke run: the brief's three example requests, end to end, against the mock provider.

    .venv/bin/python scripts/smoke_llm.py [--provider groq] [--only 1,2,3] [--preset normal]

A tiny simulated user answers questions and confirms cards. Traces land in traces/s10/<session>/.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from errand.agent.factory import build_agent  # noqa: E402
from errand.agent.llm.openai_compat import make_client  # noqa: E402
from errand.clock import DemoClock  # noqa: E402
from errand.config import Limits, llm_settings  # noqa: E402
from errand.mocks.scenario import preset  # noqa: E402
from errand.mocks.service import MockProviderService  # noqa: E402
from errand.seed.seed import seed_user  # noqa: E402
from errand.tracing.tracer import Tracer  # noqa: E402

# (opening request, how the simulated user answers a non-card reply)
# (opening request, the simulated user's replies to non-card messages, in order; the last one repeats)
CONVERSATIONS = {
    "1": ("Find a plumber for my kitchen tap this week, mornings only, under ₹1,500.", ["The first option, please."]),
    "2": ("Move my AC servicing to Saturday.", ["15:00 please."]),
    "3": ("Book the same electrician as last time, but not before 10 a.m.",
          ["This week, please.", "The earliest slot, please."]),
}
AREA_QUESTION = re.compile(r"which area|what area|area should", re.I)


def answer(replies: list[str], turn: int, agent_text: str) -> str:
    """The simulated user: names an area if asked, otherwise gives its next scripted reply."""
    return "Indiranagar" if AREA_QUESTION.search(agent_text) else replies[min(turn, len(replies) - 1)]


MAX_USER_TURNS = 5


def run(key: str, provider: str, preset_name: str, out: Path) -> dict:
    opening, replies_script = CONVERSATIONS[key]
    work = out / f"conv{key}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    clock = DemoClock()
    tracer = Tracer(out / "traces", clock)
    scenario = preset(preset_name)
    scenario.book_outcomes = scenario.book_outcomes or ["success"]
    mock = MockProviderService(work / "mock.db", clock, scenario)
    llm = make_client(provider, Limits(), on_switch=lambda m: tracer.emit("llm_fallback", message=m))
    agent = build_agent(work / "agent.db", mock, llm, clock, tracer)
    seed_user(agent.store)
    sid = agent.new_session()
    print(f"\n=== [{key}] {opening}")
    text, outcome, said = opening, None, 0
    for _ in range(MAX_USER_TURNS):
        print(f"  user > {text}")
        replies = agent.handle_user_message(sid, text)
        for r in replies:
            print(f"  {r['kind']:>8} < {r['text']}")
        last = replies[-1] if replies else {"kind": "none", "text": ""}
        if last["kind"] == "card":
            text = f"CONFIRM {last['card']['code']}"
        elif last["kind"] in ("template", "memory_prompt") and re.match(r"(Booked|Rescheduled|Cancelled):", replies[0]["text"]):
            outcome = replies[0]["text"]
            break
        else:
            text = answer(replies_script, said, last["text"])
            said += 1
    llm = tracer.of_type("llm_call")
    denied = [(d["tool"], d["reason"]) for d in tracer.of_type("gate_decision") if not d["allow"]]
    usd = sum((c["cost"] or {}).get("usd", 0) for c in llm)
    summary = {"conv": key, "session": sid, "outcome": outcome, "llm_calls": len(llm),
               "tokens_in": sum(c["input_tokens"] for c in llm), "tokens_out": sum(c["output_tokens"] for c in llm),
               "cost_usd": round(usd, 5), "denied": denied, "claims_blocked": len(tracer.of_type("claim_blocked")),
               "errors": [e.get("error") for e in tracer.of_type("error") + tracer.of_type("llm_error")],
               "fallbacks": [e["message"] for e in tracer.of_type("llm_fallback")],
               "bookings": len(mock.agent_bookings())}
    print("  ->", summary)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=None)
    ap.add_argument("--only", default="1,2,3")
    ap.add_argument("--preset", default="normal")
    ap.add_argument("--out", default=str(ROOT / "traces" / "s10"))
    a = ap.parse_args()
    s = llm_settings(a.provider)
    print(f"provider={s.provider} model={s.model} base_url={s.base_url}")
    results = [run(k, a.provider, a.preset, Path(a.out)) for k in a.only.split(",")]
    ok = all(r["outcome"] for r in results)
    print("\nALL OK" if ok else "\nSOME CONVERSATIONS DID NOT COMPLETE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
