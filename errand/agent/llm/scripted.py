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
NOTICE = "[Automated notice"   # system-started run: no user to ask, pick the best fit
TOOL_RE = re.compile(r"<tool_output[^>]*>(.*)</tool_output>", re.S)
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
SERVICE_WORDS = [(r"plumb|\btap\b|leak|pipe|drain", "plumber"), (r"electric|wiring|switch|\bfan\b", "electrician"),
                 (r"\bac\b|air ?con", "ac_servicing"), (r"clean", "cleaner")]
TIME_WORDS = {"morning": ("06:00", "12:00"), "afternoon": ("12:00", "17:00"), "evening": ("17:00", "21:00")}
ORDINALS = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2}
AFFIRM = re.compile(r"\s*(yes|yeah|yep|ok|okay|sure|go ahead|fine|sounds good)\b", re.I)
CLOCK_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap]\.?m\.?)|\b(\d{1,2}):(\d{2})\b", re.I)
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
    return phrase  # memory_op normalises unknown keys itself


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


def ask(missing: list[str]) -> str:
    return " ".join(ASK.get(f, f"What is the {f}?") for f in missing)


def search_args(request: dict) -> dict:
    """search_providers arguments for a recorded request (the gate requires them to match exactly)."""
    return {"service": request["service"], "area": request["area"], "date_range": request["date_range"]}


def parse_time(text: str) -> str | None:
    """A clock time the user named ("11am", "2:30 pm", "14:00") as HH:MM."""
    if not (m := CLOCK_RE.search(text)):
        return None
    if m.group(3):
        return _hour(m.group(1), m.group(3))[:3] + (m.group(2) or "00")
    return f"{int(m.group(4)):02d}:{m.group(5)}"


def pick_slot(text: str, slots: list[str]) -> str | None:
    """The slot the user chose by number or by time."""
    if (i := parse_choice(text)) is not None and slots:
        return slots[min(i, len(slots) - 1)]
    if t := parse_time(text):
        return next((s for s in slots if s[11:16] == t), None)
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
    def _respond(self, messages, decision: str | tuple[str, dict], text: str | None = None) -> LLMResponse:
        """A reply for `decision` (text, or a tool call optionally with `text` alongside)."""
        call = None if isinstance(decision, str) else decision
        text = decision if call is None else text
        self._n += 1
        calls = [ToolCall(id=f"call_{self._n}", name=call[0], args=call[1])] if call else []
        out = text or json.dumps(call)
        return LLMResponse(text=text, tool_calls=calls, model=self.model, latency_ms=0.0,
                           input_tokens=estimate_tokens(json.dumps(messages, ensure_ascii=False)),
                           output_tokens=estimate_tokens(out))

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                 time_left_s: float | None = None) -> LLMResponse:
        return self._respond(messages, self.decide(observe(messages), {t["function"]["name"] for t in tools}))

    # ---------- policy ----------
    def decide(self, obs: Observation, tools: set[str]) -> str | tuple[str, dict]:
        s, done_names = obs.state, [n for n, _ in obs.done]
        if obs.last_result is not None and not obs.last_result.get("ok", True):
            err = obs.last_result.get("error") or {}
            return f"Sorry, that didn't work ({err.get('message') or err.get('code', 'error')})."

        if obs.user.startswith(NOTICE):
            if s["missing"]:
                return ask(s["missing"])
            return self._book(obs, tools, done_names, auto_pick=True)
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
        if "not before" in obs.user.lower() and "time_from" in fields \
                and "propose_memory" in tools and "propose_memory" not in done_names:
            # a time limit stated for this request might be a standing preference: an inference to confirm later
            return ("propose_memory", {"key": "earliest_time", "value": fields["time_from"], "explicit": False})
        kind = s["errand"]["kind"]
        if kind in ("reschedule", "cancel") and "booking_id" in s["missing"]:
            return self._find_booking(obs, tools, done_names)
        if s["missing"]:
            return ask(s["missing"])
        if kind in ("book", "rebook"):
            return self._book(obs, tools, done_names)
        return self._change_booking(obs, tools, done_names, remembered)

    # ---------- cancel / reschedule ----------
    @staticmethod
    def _find_booking(obs: Observation, tools: set[str], done_names: list[str]) -> str | tuple[str, dict]:
        s = obs.state
        if not s["bookings"] and "list_bookings" in tools and "list_bookings" not in done_names:
            return ("list_bookings", {"user_id": s["user_id"]})
        service = s["request"].get("service")
        upcoming = [b for b in s["bookings"] if b["status"] == "confirmed" and b["slot"][:10] >= s["now"][:10]
                    and (service is None or b["service"] == service)]
        if len(upcoming) == 1 and "update_request" in tools:
            return ("update_request", {"booking_id": upcoming[0]["id"]})
        if not upcoming:
            return "I couldn't find an upcoming booking that matches. Which booking do you mean?"
        return "Which booking do you mean? " + "; ".join(
            f"{b['provider_name']} ({b['service'].replace('_', ' ')}) on {b['slot'].replace('T', ' ')}" for b in upcoming)

    def _change_booking(self, obs: Observation, tools: set[str], done_names: list[str],
                        remembered: dict[str, str]) -> str | tuple[str, dict]:
        s, r = obs.state, obs.state["request"]
        if s["errand"]["kind"] == "cancel":
            return ("request_approval", {"action": "cancel", "booking_id": r["booking_id"]})
        booking = next(b for b in s["bookings"] if b["id"] == r["booking_id"])
        mine = next((o for o in s["options"] if o["provider_id"] == booking["provider_id"]), None)
        if mine is None:
            if "search_providers" in done_names:
                return f"{booking['provider_name']} has no free slots on those dates. Want to try other days?"
            area = r.get("area") or remembered.get("preferred_area")
            if area is None:  # the provider's own service areas
                if "get_provider" not in done_names:
                    return ("get_provider", {"provider_id": booking["provider_id"]})
                area = (obs.last_result or {}).get("data", {}).get("areas", [None])[0]
            return ("search_providers", search_args({**r, "service": booking["service"], "area": area}))
        if (slot := pick_slot(obs.user, mine["slots"])) and "request_approval" not in done_names:
            return ("request_approval", {"action": "reschedule", "booking_id": booking["id"], "slot": slot})
        if not mine["slots"]:
            return f"{booking['provider_name']} has no free slots that fit. Want to try other days or times?"
        lines = [f"{i}. {slot.replace('T', ' ')}" for i, slot in enumerate(mine["slots"], start=1)]
        return f"{booking['provider_name']} can come at:\n" + "\n".join(lines) + "\nWhich time works?"

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

    def _book(self, obs: Observation, tools: set[str], done_names: list[str],
              auto_pick: bool = False) -> str | tuple[str, dict]:
        s = obs.state
        options = s["options"]
        viable = [o for o in options if o["within_budget"] and o["slots"]]
        hold = s["hold"]
        if s["errand"]["state"] == "HOLDING" and hold:
            if "hold_slot" in done_names:
                if (obs.last_result or {}).get("data", {}).get("over_budget") and not auto_pick:
                    # ask before showing a card above the budget
                    return (f"The provider now quotes ₹{hold['quoted_price']:,} (search showed ₹{hold['search_price']:,}), "
                            f"above your ₹{s['request']['budget_max']:,} budget. Should I go ahead at that price?")
                return ("request_approval", {"action": "book"})
            if AFFIRM.match(obs.user):
                return ("request_approval", {"action": "book"})
        if hold and (t := parse_time(obs.user)) and "hold_slot" in tools and "hold_slot" not in done_names:
            # a different time with the provider already held
            held = next((o for o in options if o["provider_id"] == hold["provider_id"]), {"slots": []})
            slot = next((x for x in held["slots"] if x[11:16] == t and x != hold["slot"]), None)
            if slot is None:
                return f"{t} isn't available with that provider. Pick another time?"
            return ("hold_slot", {"provider_id": hold["provider_id"], "slot": slot})
        choice = 0 if auto_pick and "search_providers" in done_names else parse_choice(obs.user)
        if choice is not None and viable and "hold_slot" in tools and "hold_slot" not in done_names:
            pick = viable[min(choice, len(viable) - 1)]
            return ("hold_slot", {"provider_id": pick["provider_id"], "slot": pick["slots"][0]})
        if not options and "search_providers" in tools and "search_providers" not in done_names:
            r = s["request"]
            return ("search_providers", search_args(r))
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
