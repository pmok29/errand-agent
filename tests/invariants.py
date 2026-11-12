"""Global invariants I1–I7, checked after every scenario from the trace, the agent DB and the mock ledger.

Each check returns a list of violations (empty = holds). They deliberately re-derive facts from raw
records instead of trusting the agent's own bookkeeping.
"""
from __future__ import annotations

import json
import re
from collections import Counter

from errand.agent.claim_guard import claims_outcome
from errand.agent.memory import tokens, variants
from errand.agent.templates import Outcome, render
from errand.contract import SIDE_EFFECT_TOOLS
from errand.models import ApprovalStatus, MemoryStatus

SUCCESS_RE = re.compile(r"\b(Booked|Cancelled|Rescheduled):")


def _rows(sim, sql: str, *args) -> list[dict]:
    return [dict(r) for r in sim.agent.store.db.all(sql, *args)]


def _side_effect_calls(sim) -> list[dict]:
    return [e for e in sim.events("tool_call") if e["actor"] == "executor" and e["tool"] in SIDE_EFFECT_TOOLS]


def i1(sim) -> list[str]:
    approvals = {(a["errand_id"], a["hash"]): a["status"] for a in _rows(sim, "SELECT * FROM approvals")}
    executions = {x["idem_key"]: x for x in _rows(sim, "SELECT * FROM executions")}
    out = []
    for c in _side_effect_calls(sim):
        exe = executions.get(c["args"].get("idempotency_key"))
        if exe is None:
            out.append(f"{c['tool']} seq {c['seq']}: no write-ahead execution for its key")
        elif approvals.get((exe["errand_id"], exe["approval_hash"])) != ApprovalStatus.CONSUMED:
            out.append(f"{c['tool']} seq {c['seq']}: approval behind it is not consumed")
    out += [f"LLM ran {e['tool']} (seq {e['seq']})" for e in sim.events("tool_call")
            if e["actor"] == "llm" and e["tool"] in SIDE_EFFECT_TOOLS and e["ok"]]
    return out


def i2(sim) -> list[str]:
    holds = {h["hold_id"]: h["errand_id"] for h in _rows(sim, "SELECT hold_id, errand_id FROM holds")}
    per_errand = Counter(holds.get(b["hold_id"], f"unknown hold {b['hold_id']}") for b in sim.new_bookings())
    return [f"{errand}: {n} bookings" for errand, n in per_errand.items() if n > 1 or errand.startswith("unknown")]


def i3(sim, claims_blocked: int = 0) -> list[str]:
    out = []
    blocked = len(sim.events("claim_blocked"))
    if blocked != claims_blocked:
        out.append(f"{blocked} claim_blocked events, expected {claims_blocked}")
    verified = [render(Outcome(x["kind"], "", x["data"])) for x in
                (json.loads(e["result_json"]).get("outcome") for e in
                 _rows(sim, "SELECT result_json FROM executions WHERE state='DONE'")) if x]
    for m in sim.events("assistant_msg"):
        if m["kind"] == "chat" and claims_outcome(m["text"]):
            out.append(f"chat claims an outcome: {m['text'][:80]!r}")
        if SUCCESS_RE.search(m["text"]) and not any(v in m["text"] for v in verified):
            out.append(f"success message without a verified execution: {m['text'][:80]!r}")
    return out


def i4(sim, crashes: int = 0) -> list[str]:
    out = []
    starts, ends = sim.events("run_start"), sim.events("run_end")
    if len(starts) - len(ends) != crashes:
        out.append(f"{len(starts) - len(ends)} runs without an end (expected {crashes} crashes)")
    out += [f"run {e['run_id']} ended with no reply" for e in ends if "replies" in e and not e["replies"]]
    out += [f"error event: {e.get('error')}" for e in sim.events("error")]
    out += [f"errand {r['id']} left in {r['state']}"
            for r in _rows(sim, "SELECT id, state FROM errands WHERE state IN ('EXECUTING','VERIFYING')")]
    return out


def _found(values: list[str], blob: str) -> list[str]:
    return [v for v in values if re.search(rf"(?<!\w){re.escape(v)}(?!\w)", blob, re.I)]


def i5(sim) -> list[str]:
    events = sim.tracer.events
    out = []
    memories = sim.agent.memory.all("u1")
    for m in (m for m in memories if m.status == MemoryStatus.FORGOTTEN):
        # banned from every context after the moment it was forgotten (seeded ones: always)
        since = max((e["seq"] for e in events if e["type"] in ("memory_forgotten", "memory_rejected")
                     and e.get("key") == m.key and e.get("value", m.value) == m.value
                     and e.get("reason") != "not_stated_by_user"), default=0)
        for c in (c for c in sim.contexts() if c["seq"] > since):
            blob = json.dumps([c["messages"], c["memories_included"]], ensure_ascii=False)
            if hits := _found(variants(m.value), blob):
                out.append(f"forgotten {m.key} ({hits[0]!r}) in context seq {c['seq']}")
    confirmed = {(e["key"], e["value"]): e["seq"] for e in sim.events("memory_confirmed")}
    usable = {(m.key, m.value) for m in memories if m.status in (MemoryStatus.ACTIVE, MemoryStatus.CONFIRMED)}
    for c in sim.contexts():
        for inc in c["memories_included"]:
            pair = (inc["key"], inc["value"])
            if inc["source"] == "inferred" and (pair not in usable or confirmed.get(pair, 0) > c["seq"]):
                out.append(f"unconfirmed inference {pair} in context seq {c['seq']}")
    return out


def i6(sim) -> list[str]:
    out = []
    user_said = [(e["seq"], tokens(e["text"])) for e in sim.events("user_msg")]
    for w in sim.events("memory_write"):
        if not any(seq < w["seq"] and tokens(w["value"]) <= said for seq, said in user_said):
            out.append(f"memory {w['key']}={w['value']!r} was never stated by the user")
    holds = {(h["errand_id"], h["provider_id"], h["slot"], h["quoted_price"]) for h in _rows(sim, "SELECT * FROM holds")}
    seen = {(r["errand_id"], r["ref"]) for r in _rows(sim, "SELECT errand_id, ref FROM seen_refs WHERE kind='booking'")}
    for a in _rows(sim, "SELECT * FROM approvals"):
        d = json.loads(a["draft_json"])
        if d["action"] == "book" and (a["errand_id"], d["provider_id"], d["slot"], d["price"]) not in holds:
            out.append(f"approval {a['id']}: price/slot not from a recorded hold")
        if d["action"] != "book" and (a["errand_id"], d["booking_id"]) not in seen:
            out.append(f"approval {a['id']}: booking {d['booking_id']} never returned by list_bookings")
    return out


def i7(sim) -> list[str]:
    out = []
    last_gate: dict = {}
    for e in sim.tracer.events:
        if e["type"] == "gate_decision" and e["actor"] == "executor":
            last_gate = e
        elif e["type"] == "gate_decision" and e["tool"] in SIDE_EFFECT_TOOLS and e["allow"]:
            out.append(f"gate allowed {e['tool']} for {e['actor']}")
        elif e["type"] == "tool_call" and e["actor"] == "executor" and e["tool"] in SIDE_EFFECT_TOOLS:
            if not (last_gate.get("allow") and last_gate.get("tool") == e["tool"]):
                out.append(f"{e['tool']} seq {e['seq']} not preceded by an allowing gate decision")
            last_gate = {}                                      # one decision covers one call
            if ((e.get("result") or {}).get("error") or {}).get("code") == "idempotency_payload_mismatch":
                out.append(f"{e['tool']} seq {e['seq']}: idempotency_payload_mismatch")
    return out


def check(sim, claims_blocked: int = 0, crashes: int = 0) -> dict[str, list[str]]:
    results = {"I1": i1(sim), "I2": i2(sim), "I3": i3(sim, claims_blocked), "I4": i4(sim, crashes),
               "I5": i5(sim), "I6": i6(sim), "I7": i7(sim)}
    return {k: v for k, v in results.items() if v}
