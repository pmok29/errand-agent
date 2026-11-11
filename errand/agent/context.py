"""Builds the messages sent to the model under a fixed token budget.

Layout: [system prompt + STATE block (incl. relevant memories)] + current-session history + this turn.
Earlier sessions never reach the model as raw text: only filtered memories do. Forgotten values (and
inferred candidates from other sessions) are scrubbed from history. Every build reports what it
included, excluded and why, so tests can prove it.
"""
from __future__ import annotations

import json
import re
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
REMOVED = "[removed]"
TRIMMED = '<tool_output untrusted="true">{"trimmed": "older tool output dropped to fit the context budget"}</tool_output>'


@dataclass
class Context:
    messages: list[dict[str, Any]]
    state: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)


def state_block(store: Store, e: Errand, memories: list[Memory]) -> dict[str, Any]:
    now = store.clock.now()
    hold = store.latest_hold(e.id)
    return {
        "now": now.strftime("%Y-%m-%d %a %H:%M"),
        "user_id": e.user_id,
        "errand": {"kind": e.kind, "state": e.state},
        "request": e.request.model_dump(mode="json", exclude_none=True),
        "missing": e.request.missing(e.kind),
        "already_asked": e.asked,
        "memories": [{"key": m.key, "value": m.value, "source": m.source} for m in memories],
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
          cfg: ContextConfig | None = None) -> Context:
    cfg = cfg or ContextConfig()
    sel = mem.select(e.user_id, e.request.service, cfg.max_memories)
    scrub = _scrubber(mem.redaction_values(e.user_id))
    hits = 0

    def system_for(memories: list[Memory]) -> tuple[dict, dict]:
        st = state_block(store, e, memories)
        return {"role": "system", "content": f"{SYSTEM_PROMPT}\n<state>{json.dumps(st, ensure_ascii=False)}</state>"}, st

    memories = list(sel.included)
    system, state = system_for(memories)
    history = []
    for m in store.session_messages(e.session_id, limit=HISTORY_MAX):
        text, n = scrub(m["content"])
        hits += n
        history.append({"role": "assistant" if m["role"] == "assistant" else "user", "content": text})
    current, older = history[-1:], history[:-1]   # the latest user message is always kept
    turn = [dict(t) for t in turn]

    # Must-keep parts first; shrink memories, then old tool outputs, if even those don't fit.
    trimmed, dropped_memories = 0, []
    while _tokens([system, *current, *turn]) > cfg.budget_tokens:
        if memories:
            dropped_memories.append(memories.pop())
            system, state = system_for(memories)
        elif (i := next((i for i, t in enumerate(turn) if t["role"] == "tool" and t["content"] != TRIMMED), None)) is not None:
            turn[i]["content"], trimmed = TRIMMED, trimmed + 1
        else:
            break
    used = _tokens([system, *current, *turn])
    kept: list[dict] = []
    for m in reversed(older):  # newest history first, while it fits
        t = _tokens([m])
        if used + t > cfg.budget_tokens:
            break
        kept.insert(0, m)
        used += t

    messages = [system, *kept, *current, *turn]
    excluded = [(m, r) for m, r in sel.excluded] + [(m, "budget") for m in dropped_memories]
    meta = {"est_tokens": _tokens(messages), "budget": cfg.budget_tokens,
            "memories_included": [{"key": m.key, "value": m.value, "source": m.source} for m in memories],
            "memories_excluded": [{"key": m.key, "status": m.status, "reason": r} for m, r in excluded],
            "history_kept": len(kept) + len(current), "history_dropped": len(older) - len(kept),
            "redaction_hits": hits, "tool_outputs_trimmed": trimmed}
    return Context(messages=messages, state=state, meta=meta)
