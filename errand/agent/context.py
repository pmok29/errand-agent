"""Builds the messages sent to the model: system prompt + code-built STATE block + chat history + this turn.

S5 scope: state block and recent current-session history. S6 adds memories, the token budget and scrubbing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from errand.agent.state import TOOLS_BY_STATE
from errand.agent.store import Store
from errand.clock import is_expired
from errand.models import ApprovalStatus, Errand

SYSTEM_PROMPT = (Path(__file__).parent / "prompts" / "system.md").read_text()
HISTORY_TURNS = 12


@dataclass
class Context:
    messages: list[dict[str, Any]]
    state: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)


def state_block(store: Store, e: Errand) -> dict[str, Any]:
    now = store.clock.now()
    hold = store.latest_hold(e.id)
    pending = store.approvals_for(e.id, ApprovalStatus.PENDING)
    return {
        "now": now.strftime("%Y-%m-%d %a %H:%M"),
        "user_id": e.user_id,
        "errand": {"kind": e.kind, "state": e.state},
        "request": e.request.model_dump(mode="json", exclude_none=True),
        "missing": e.request.missing(e.kind),
        "already_asked": e.asked,
        "options": e.selected.get("options", []),
        "hold": None if hold is None or is_expired(hold.expires_at, now) else
        {"provider_id": hold.provider_id, "slot": hold.slot, "quoted_price": hold.quoted_price,
         "search_price": hold.search_price},
        "bookings": store.refs(e.id, "booking"),
        "awaiting_confirmation": bool(pending),
        "allowed_tools": sorted(TOOLS_BY_STATE[e.state]),
    }


def build(store: Store, e: Errand, turn_messages: list[dict[str, Any]]) -> Context:
    state = state_block(store, e)
    system = f"{SYSTEM_PROMPT}\n<state>{json.dumps(state, ensure_ascii=False)}</state>"
    history = [{"role": "assistant" if m["role"] == "assistant" else "user", "content": m["content"]}
               for m in store.session_messages(e.session_id, limit=HISTORY_TURNS)]
    return Context(messages=[{"role": "system", "content": system}, *history, *turn_messages], state=state)
