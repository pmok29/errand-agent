"""Builds the messages sent to the model under a fixed token budget.

Layout: [system prompt + STATE block (incl. relevant memories)] + current-session history + this turn.
Earlier sessions never reach the model as raw text: only filtered memories do. Forgotten values are
scrubbed from history; unconfirmed inferences are kept out of the memory block. Every build reports
what it included, excluded and why, so tests can prove it.
"""
from __future__ import annotations

import json
import math
import re
from datetime import date, timedelta
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from errand.agent.llm.base import estimate_tokens
from errand.agent.memory import MemoryStore
from errand.agent.state import TOOLS_BY_STATE
from errand.agent.store import Store
from errand.clock import is_expired
from errand.config import ContextConfig
from errand.models import ApprovalStatus, Errand, Memory

SYSTEM_PROMPT = (Path(__file__).parent / "prompts" / "system.md").read_text()
HISTORY_MAX = 40          # candidate messages from the current session, before budgeting
CURRENT_SHARE = 0.25      # most of the budget the latest message may take before it is cut
REMOVED = "[removed]"
CUT = " …[the rest of this message was cut to fit the context budget]"
TRIMMED = '<tool_output untrusted="true">{"trimmed": "older tool output dropped to fit the context budget"}</tool_output>'


@dataclass
class Context:
    messages: list[dict[str, Any]]
    meta: dict[str, Any] = field(default_factory=dict)


def calendar(today: date) -> dict[str, Any]:
    """Resolved relative dates, so the model never has to do calendar arithmetic."""
    sunday = today + timedelta(days=6 - today.weekday())
    nxt = {(today + timedelta(days=i)).strftime("%a"): (today + timedelta(days=i)).isoformat() for i in range(1, 8)}
    return {"today": f"{today.isoformat()} {today.strftime('%a')}",
            "tomorrow": (today + timedelta(days=1)).isoformat(),
            "this_week": [today.isoformat(), sunday.isoformat()],
            "next_week": [(sunday + timedelta(days=1)).isoformat(), (sunday + timedelta(days=7)).isoformat()],
            "next": nxt}


def memory_view(m: Memory) -> dict[str, str]:
    """A memory as the model (and the context_sent trace) sees it."""
    return {"key": m.key, "value": m.value, "source": m.source}


def state_block(store: Store, e: Errand, memories: list[Memory]) -> dict[str, Any]:
    now = store.clock.now()
    hold = store.latest_hold(e.id)
    return {
        "now": now.strftime("%Y-%m-%d %a %H:%M"),
        "dates": calendar(now.date()),
        "user_id": e.user_id,
        "errand": {"kind": e.kind, "state": e.state},
        "request": e.request.model_dump(mode="json", exclude_none=True),
        "missing": e.request.missing(e.kind),
        "already_asked": e.asked,
        "memories": [memory_view(m) for m in memories],
        "options": e.selected.get("options", []),
        "hold": None if hold is None or is_expired(hold.expires_at, now) else
        {"provider_id": hold.provider_id, "slot": hold.slot, "quoted_price": hold.quoted_price,
         "search_price": hold.search_price},
        "bookings": store.refs(e.id, "booking"),
        "awaiting_confirmation": bool(store.approvals_for(e.id, ApprovalStatus.PENDING)),
        "allowed_tools": sorted(TOOLS_BY_STATE[e.state]),
    }


def _scrubber(values: list[str]):
    if not values:
        return lambda text: (text, 0)
    pattern = re.compile("|".join(rf"(?<!\w){re.escape(v)}(?!\w)" for v in values), re.I)
    return lambda text: pattern.subn(REMOVED, text)


def _tokens(messages: list[dict[str, Any]]) -> int:
    return sum(estimate_tokens(json.dumps(m, ensure_ascii=False)) for m in messages)


def build(store: Store, mem: MemoryStore, e: Errand, turn: list[dict[str, Any]],
          cfg: ContextConfig | None = None, tools: list[dict[str, Any]] | None = None) -> Context:
    """`tools` are the schemas sent with the request: they count against the same budget."""
    cfg = cfg or ContextConfig()
    sel = mem.select(e.user_id, e.request.service, cfg.max_memories)
    scrub = _scrubber(mem.redaction_values(e.user_id))
    hits = 0

    memories = list(sel.included)
    state = state_block(store, e, memories)

    def system_message() -> dict:
        state["memories"] = [memory_view(m) for m in memories]
        return {"role": "system", "content": f"{SYSTEM_PROMPT}\n<state>{json.dumps(state, ensure_ascii=False)}</state>"}

    history = []
    for m in store.session_messages(e.session_id, limit=HISTORY_MAX):
        if m["kind"] == "error":  # outage notices tell the model nothing, and models copy them
            continue
        text, n = scrub(m["content"])
        hits += n
        if m["role"] == "assistant" and m["kind"] != "chat":  # code-written: label it so models don't imitate it
            text = f"[{m['kind']} from the system] {text}"
        history.append({"role": "assistant" if m["role"] == "assistant" else "user", "content": text})
    current, older = history[-1:], history[:-1]   # the latest user message is always kept
    turn = [dict(t) for t in turn]

    # Must-keep parts first. An oversized latest message is cut to its share up front (so it can't push out
    # the memories that fill the request), then memories, old tool outputs and, last, more of that message
    # go until it fits. Token counts are taken once per message and updated only for what changes.
    system = system_message()
    sizes = {"tools": _tokens(tools or []), "system": _tokens([system]), "current": _tokens(current)}
    turn_sizes = [_tokens([t]) for t in turn]
    trimmed, dropped_memories, chars_cut = 0, [], 0

    def cut_current(tokens: int) -> bool:
        nonlocal current, chars_cut
        if not (current and (text := current[0]["content"].removesuffix(CUT))):
            return False
        keep = max(0, len(text) - math.ceil(tokens * cfg.chars_per_token))
        current, chars_cut = [{**current[0], "content": text[:keep] + CUT}], chars_cut + len(text) - keep
        sizes["current"] = _tokens(current)
        return True

    over_budget = sum(sizes.values()) + sum(turn_sizes) > cfg.budget_tokens
    if over_budget and (excess := sizes["current"] - cfg.budget_tokens * CURRENT_SHARE) > 0:
        cut_current(math.ceil(excess))
    while (over := sum(sizes.values()) + sum(turn_sizes) - cfg.budget_tokens) > 0:
        if memories:
            dropped_memories.append(memories.pop())
            system = system_message()
            sizes["system"] = _tokens([system])
        elif (i := next((i for i, t in enumerate(turn) if t["role"] == "tool" and t["content"] != TRIMMED), None)) is not None:
            turn[i]["content"], trimmed = TRIMMED, trimmed + 1
            turn_sizes[i] = _tokens([turn[i]])
        elif not cut_current(over):
            break
    used = sum(sizes.values()) + sum(turn_sizes)
    kept: list[dict] = []
    for m in reversed(older):  # newest history first, while it fits
        t = _tokens([m])
        if used + t > cfg.budget_tokens:
            break
        kept.insert(0, m)
        used += t

    messages = [system, *kept, *current, *turn]
    excluded = [(m, r) for m, r in sel.excluded] + [(m, "budget") for m in dropped_memories]
    meta = {"est_tokens": used, "budget": cfg.budget_tokens, "tool_schema_tokens": sizes["tools"],
            "memories_included": [memory_view(m) for m in memories],
            "memories_excluded": [{"key": m.key, "status": m.status, "reason": r} for m, r in excluded],
            "history_kept": len(kept) + len(current), "history_dropped": len(older) - len(kept),
            "redaction_hits": hits, "tool_outputs_trimmed": trimmed, "latest_message_chars_cut": chars_cut,
            "over_budget": used > cfg.budget_tokens}
    return Context(messages=messages, meta=meta)
