"""Offline models for evals.

ReactivePolicy reads exactly what a real model would get (the system prompt's STATE block, the chat,
and this turn's tool results) and picks the obvious next step. A tiny rule-based parser stands in for
language understanding. Because it only sees the context, it also exercises the context builder.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from errand.agent.llm.base import LLMResponse, ToolCall, estimate_tokens
from errand.contract import AREAS

STATE_RE = re.compile(r"<state>(.*)</state>", re.S)
TOOL_RE = re.compile(r"<tool_output[^>]*>(.*)</tool_output>", re.S)
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
SERVICE_WORDS = [(r"plumb|\btap\b|leak|pipe|drain", "plumber"), (r"electric|wiring|switch|\bfan\b", "electrician"),
                 (r"\bac\b|air ?con", "ac_servicing"), (r"clean", "cleaner")]
TIME_WORDS = {"morning": ("06:00", "12:00"), "afternoon": ("12:00", "17:00"), "evening": ("17:00", "21:00")}
ORDINALS = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2}
ASK = {"service": "What do you need done (plumber, electrician, AC servicing or cleaning)?",
       "area": "Which area should I search in: Indiranagar, Koramangala or HSR Layout?",
       "date_range": "Which day or dates work for you?",
       "booking_id": "Which booking do you mean?"}


def _hour(h: str, ampm: str | None) -> str:
    n = int(h) % 12 + (12 if ampm and ampm.lower().startswith("p") else 0)
    return f"{n:02d}:00"


def parse_request(text: str, today: date) -> dict[str, Any]:
    """Fields the user stated, in update_request's argument format."""
    t, f = text.lower(), {}
    if re.search(r"\bcancel\b", t):
        f["kind"] = "cancel"
    elif re.search(r"\b(move|reschedule|shift)\b", t):
        f["kind"] = "reschedule"
    for pattern, service in SERVICE_WORDS:
        if re.search(pattern, t):
            f["service"] = service
            break
    for area in AREAS:
        if area.lower() in t or (area == "HSR Layout" and "hsr" in t):
            f["area"] = area
    if "this week" in t:
        f["date_from"], f["date_to"] = today.isoformat(), (today + timedelta(days=6 - today.weekday())).isoformat()
    elif "next week" in t:
        start = today + timedelta(days=7 - today.weekday())
        f["date_from"], f["date_to"] = start.isoformat(), (start + timedelta(days=6)).isoformat()
    elif "tomorrow" in t:
        f["date_from"] = f["date_to"] = (today + timedelta(days=1)).isoformat()
    else:
        for i, name in enumerate(WEEKDAYS):
            if re.search(rf"\b{name}\b", t):
                d = today + timedelta(days=(i - today.weekday()) % 7 or 7)
                f["date_from"] = f["date_to"] = d.isoformat()
    for word, (lo, hi) in TIME_WORDS.items():
        if word in t:
            f["time_from"], f["time_to"] = lo, hi
    if m := re.search(r"(?:not before|after|from)\s+(\d{1,2})\s*([ap]\.?m\.?)?", t):
        f["time_from"], f["time_to"] = _hour(m.group(1), m.group(2)), "23:59"
    if m := re.search(r"(?:under|below|less than|within|max(?:imum)?|up to)\s*(?:₹|rs\.?|inr)?\s*([\d,]+)", t):
        f["budget_max"] = int(m.group(1).replace(",", ""))
    return f


MEMORY_KEYS = {"area": "preferred_area", "earliest time": "earliest_time"}


def _memory_key(phrase: str, known: list[str]) -> str:
    phrase = phrase.strip().lower()
    if phrase in MEMORY_KEYS:
        return MEMORY_KEYS[phrase]
    words = set(re.findall(r"[a-z]+", phrase))
    best = max(known, key=lambda k: len(words & set(k.split("_"))), default=None)
    if best and words & set(best.split("_")):
        return best
    return re.sub(r"[^a-z0-9]+", "_", phrase).strip("_")


def parse_memory_command(text: str, known: list[str]) -> tuple[str, dict] | None:
    t = text.lower().strip()
    if re.search(r"what do you (know|remember)|show (me )?(my |your )?memor", t):
        return ("memory_op", {"op": "list"})
    if m := re.search(r"\bforget (?:about )?(?:my |the )?(.+?)[.!]?$", t):
        return ("memory_op", {"op": "forget", "key": _memory_key(m.group(1), known)})
    if m := re.search(r"\bmy (.+?) is (?:now )?(.+?)(?: now)?(?:,? (?:please )?remember (?:it|that))?[.!]?$", text, re.I):
        key = _memory_key(m.group(1), known)
        # "my kitchen tap is leaking" is a request, not a preference
        if " now" in t or "remember" in t or key in known:
            return ("memory_op", {"op": "correct", "key": key, "value": m.group(2).strip()})
    return None


def parse_choice(text: str) -> int | None:
    t = text.lower()
    if m := re.search(r"\b(?:option|number|no\.?|#)\s*(\d)\b", t):
        return int(m.group(1)) - 1
    for word, i in ORDINALS.items():
        if re.search(rf"\b{word}\b", t):
            return i
    return None


@dataclass
class Observation:
    state: dict[str, Any]
    user: str
    done: list[tuple[str, dict]] = field(default_factory=list)        # (tool, args) called this turn
    last_result: dict[str, Any] | None = None


def observe(messages: list[dict[str, Any]]) -> Observation:
    state = json.loads(STATE_RE.search(messages[0]["content"]).group(1))
    last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
    obs = Observation(state=state, user=messages[last_user]["content"])
    for m in messages[last_user + 1:]:
        if m["role"] == "assistant":
            obs.done += [(tc["function"]["name"], json.loads(tc["function"]["arguments"])) for tc in m.get("tool_calls", [])]
        elif m["role"] == "tool":
            body = TOOL_RE.search(m["content"])
            obs.last_result = json.loads(body.group(1)) if body else None
    return obs


class ReactivePolicy:
    model = "scripted"

    def __init__(self) -> None:
        self._n = 0

    # ---------- response helpers ----------
    def _respond(self, messages, text: str | None = None, call: tuple[str, dict] | None = None) -> LLMResponse:
        self._n += 1
        calls = [ToolCall(id=f"call_{self._n}", name=call[0], args=call[1])] if call else []
        out = text or json.dumps(call)
        return LLMResponse(text=text, tool_calls=calls, model=self.model, latency_ms=0.0,
                           input_tokens=estimate_tokens(json.dumps(messages, ensure_ascii=False)),
                           output_tokens=estimate_tokens(out))

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        obs = observe(messages)
        decision = self.decide(obs, {t["function"]["name"] for t in tools})
        if isinstance(decision, str):
            return self._respond(messages, text=decision)
        return self._respond(messages, call=decision)

    # ---------- policy ----------
    def decide(self, obs: Observation, tools: set[str]) -> str | tuple[str, dict]:
        s, done_names = obs.state, [n for n, _ in obs.done]
        if obs.last_result is not None and not obs.last_result.get("ok", True):
            err = obs.last_result.get("error") or {}
            return f"Sorry, that didn't work ({err.get('message') or err.get('code', 'error')})."

        known = [m["key"] for m in s.get("memories", [])]
        if (cmd := parse_memory_command(obs.user, known)) and "memory_op" in tools:
            if "memory_op" not in done_names:
                return cmd
            return self._memory_reply(cmd[1], obs.last_result or {})

        today = datetime.strptime(s["now"][:10], "%Y-%m-%d").date()
        fields = parse_request(obs.user, today)
        remembered = {m["key"]: m["value"] for m in s.get("memories", [])}
        if "area" in s["missing"] and "area" not in fields and remembered.get("preferred_area"):
            fields["area"] = remembered["preferred_area"]  # fill from memory rather than asking
        if fields and "update_request" in tools and "update_request" not in done_names:
            return ("update_request", fields)
        if (m := re.search(r"not before (\d{1,2})\s*([ap]\.?m\.?)?", obs.user.lower())) \
                and "propose_memory" in tools and "propose_memory" not in done_names:
            # a time limit stated for this request might be a standing preference: an inference to confirm later
            return ("propose_memory", {"key": "earliest_time", "value": _hour(m.group(1), m.group(2)), "explicit": False})
        if s["missing"]:
            return " ".join(ASK.get(f, f"What is the {f}?") for f in s["missing"])
        if s["errand"]["kind"] in ("book", "rebook"):
            return self._book(obs, tools, done_names)
        return "Rescheduling and cancelling are handled in a later build of this policy."

    @staticmethod
    def _memory_reply(cmd: dict, result: dict) -> str:
        data = result.get("data", {})
        if cmd["op"] == "list":
            items = data.get("memories", [])
            return "Here's what I remember: " + "; ".join(
                f"{m['key'].replace('_', ' ')}: {m['value']} ({m['source']})" for m in items) if items else "I don't remember anything about you yet."
        if cmd["op"] == "forget":
            return f"I've forgotten your {cmd['key'].replace('_', ' ')}."
        return f"Got it, I'll use {cmd['value']} from now on."

    def _book(self, obs: Observation, tools: set[str], done_names: list[str]) -> str | tuple[str, dict]:
        s = obs.state
        options = s["options"]
        viable = [o for o in options if o["within_budget"] and o["slots"]]
        if s["errand"]["state"] == "HOLDING" and s["hold"] and "hold_slot" in done_names:
            return ("request_approval", {"action": "book"})
        choice = parse_choice(obs.user)
        if choice is not None and viable and "hold_slot" in tools and "hold_slot" not in done_names:
            pick = viable[min(choice, len(viable) - 1)]
            return ("hold_slot", {"provider_id": pick["provider_id"], "slot": pick["slots"][0]})
        if not options and "search_providers" in tools and "search_providers" not in done_names:
            r = s["request"]
            return ("search_providers", {"service": r["service"], "area": r["area"], "date_range": r["date_range"]})
        if "search_providers" in done_names:
            detailed = {a.get("provider_id") for n, a in obs.done if n == "get_provider"}
            for o in viable[:3]:
                if o["provider_id"] not in detailed:
                    return ("get_provider", {"provider_id": o["provider_id"]})
        if viable:
            lines = [f"{i}. {o['name']}: ₹{o['price']:,}, rating {o['rating']}, earliest {o['slots'][0].replace('T', ' ')}"
                     for i, o in enumerate(viable[:3], start=1)]
            return "Here are the best options:\n" + "\n".join(lines) + "\nWhich one would you like?"
        return "I couldn't find a provider that fits your budget and times. Want me to widen the search?"
