"""The agent loop. The LLM proposes tool calls; code checks, executes, and owns every outcome message."""
from __future__ import annotations

import json
import re
import threading
import time
from datetime import timedelta
from typing import Any, Callable

from errand.agent import approvals, claim_guard, context, internal_tools, sanitize
from errand.agent.executor import Executor
from errand.agent.gate import Gate, slot_outside_request
from errand.agent.llm.base import LLMClient, LLMError, ToolCall
from errand.agent.memory import MemoryStore
from errand.agent.recovery import recover
from errand.agent.state import transition
from errand.agent.store import Store
from errand.agent.templates import Outcome, for_state, render, repeat_confirmation
from errand.agent.tool_schemas import schemas_for
from errand.clock import Clock
from errand.config import Settings
from errand.models import Errand, ErrandKind, ErrandState as S, Request, ToolResult, ToolTimeout
from errand.tracing.tracer import Tracer

ToolCallFn = Callable[[str, dict], ToolResult]
LIMIT_STATES = {S.GATHERING, S.COMPARING, S.HOLDING}
SUCCESS = {"booked", "cancelled", "rescheduled"}
REMEMBER_RE = re.compile(r"\s*remember\s+(yes|no)\s*[.!]?\s*", re.I)
MAX_SLOTS_SHOWN = 6

LLM_UNAVAILABLE = ("I'm having trouble reaching my language model right now. Nothing was booked or changed; "
                   "please try again in a moment.")
INTERNAL_ERROR = "Something went wrong on my side. Nothing was booked or changed."


def compact_options(providers: list[dict], req: Request) -> list[dict]:
    """What the model (and the state block) sees of a search: only slots that fit the request."""
    out = []
    for p in providers:
        attrs = p.get("attributes") or {}
        out.append({"provider_id": p["provider_id"], "name": p["name"], "price": p["price"], "rating": p["rating"],
                    "jobs_done": p.get("jobs_done"), "verified": attrs.get("verified"),
                    "warranty_days": attrs.get("warranty_days"),
                    "within_budget": req.budget_max is None or p["price"] <= req.budget_max,
                    "slots": [s for s in p["slots"] if not slot_outside_request(req, s)][:MAX_SLOTS_SHOWN]})
    return out


class Agent:
    def __init__(self, store: Store, call: ToolCallFn, llm: LLMClient, tracer: Tracer, clock: Clock,
                 settings: Settings | None = None, faults: set[str] | None = None):
        self.store, self.call, self.llm, self.tracer, self.clock = store, call, llm, tracer, clock
        self.settings = settings or Settings()
        self.limits = self.settings.limits
        self.memory = MemoryStore(store)
        self.gate = Gate(store, tracer, self.limits)
        self.executor = Executor(store, self.gate, tracer, clock, call, self.settings, faults)
        # one conversation step at a time: the web app's request threads and the watcher share the agent
        self.lock = threading.RLock()

    # ---------- public ----------
    def new_session(self, user_id: str = "u1") -> str:
        return self.store.create_session(user_id)

    def recover(self) -> list[Outcome]:
        return recover(self.executor, self.tracer)

    def handle_user_message(self, session_id: str, text: str) -> list[dict[str, Any]]:
        with self.lock:
            return self._handle_user_message(session_id, text)

    def run_proactive(self, e: Errand, notice: str) -> list[dict[str, Any]]:
        """An LLM run started by the system (no user message). The notice is code-authored and marked as such."""
        self.store.add_message(e.session_id, "user", f"[Automated notice, not from the user] {notice}", kind="event")
        try:
            return self._llm_loop(e)
        except Exception as err:
            self.tracer.emit("error", error=repr(err))
            return [self._reply(e.session_id, "template", INTERNAL_ERROR)]

    def _handle_user_message(self, session_id: str, text: str) -> list[dict[str, Any]]:
        self.tracer.start_run(session_id, kind="user_message")
        self.tracer.emit("user_msg", text=text)
        self.store.add_message(session_id, "user", text)
        try:
            out = self._handle(session_id, text)
        except Exception as err:  # every exit path produces a clear user-facing outcome
            self.tracer.emit("error", error=repr(err))
            out = [self._reply(session_id, "template", INTERNAL_ERROR)]
        e = self.store.latest_errand(session_id)
        self.tracer.end_run(outcome=e.state if e else "no_errand", replies=[m["kind"] for m in out])
        return out

    # ---------- routing ----------
    def _handle(self, session_id: str, text: str) -> list[dict[str, Any]]:
        if m := REMEMBER_RE.fullmatch(text):
            return self._answer_memory_prompt(session_id, m.group(1).lower() == "yes")
        if (code := approvals.match_code(text)) is not None:
            e = self.store.latest_errand(session_id)
            if e is None:
                return [self._reply(session_id, "template", "There's nothing waiting for confirmation.")]
            return self._confirm(e, code)
        e = self.store.active_errand(session_id) or self.store.create_errand(
            self.store.session_user(session_id), session_id, ErrandKind.BOOK)
        if e.state == S.AWAITING_APPROVAL:  # anything other than the code is not approval
            approvals.invalidate_pending(self.store, self.tracer, e, "user_replied_without_code")
            transition(self.store, self.tracer, e, S.COMPARING, "user_replied_without_code")
        return self._llm_loop(e)

    def _confirm(self, e: Errand, code: str) -> list[dict[str, Any]]:
        res = approvals.consume(self.store, self.tracer, e, code)
        e = self.store.get_errand(e.id)
        if res.status == "ok":
            return self._outcome(e, self.executor.run(res.execution))
        if res.status == "already_consumed":
            return self._outcome(e, repeat_confirmation(self.store, e))
        if e.state == S.AWAITING_APPROVAL and res.status in ("expired", "changed", "invalidated"):
            transition(self.store, self.tracer, e, S.COMPARING, f"approval_{res.status}")
        kind = {"expired": "approval_expired", "changed": "approval_changed",
                "invalidated": "approval_changed"}.get(res.status)
        if kind:
            return self._outcome(e, Outcome(kind, e.id))
        return [self._reply(e.session_id, "template", "That code doesn't match anything waiting for confirmation. "
                            + render(for_state(self.store, e)))]

    # ---------- LLM loop ----------
    def _llm_loop(self, e: Errand) -> list[dict[str, Any]]:
        turn: list[dict[str, Any]] = []
        llm_turns = tool_calls = 0
        deadline = self.clock.now() + timedelta(seconds=self.limits.max_wall_seconds)
        while True:
            e = self.store.get_errand(e.id)
            if llm_turns >= self.limits.max_llm_turns:
                return self._limit(e, "llm_turns")
            if tool_calls >= self.limits.max_tool_calls:
                return self._limit(e, "tool_calls")
            if self.clock.now() >= deadline:
                return self._limit(e, "wall_clock")

            ctx = context.build(self.store, self.memory, e, turn, self.settings.context)
            self.tracer.emit("context_sent", errand_id=e.id, messages=ctx.messages, **ctx.meta)
            try:
                resp = self.llm.complete(ctx.messages, schemas_for(e.state))
            except LLMError as err:
                self.tracer.emit("llm_error", error=str(err), retryable=err.retryable)
                return [self._reply(e.session_id, "template", LLM_UNAVAILABLE)]
            llm_turns += 1
            self.tracer.llm_call(resp.model, resp.latency_ms, resp.input_tokens, resp.output_tokens,
                                 text=resp.text, tool_calls=[{"name": c.name, "args": c.args} for c in resp.tool_calls])
            if not resp.tool_calls:
                return [self._chat(e, resp.text)]

            turn.append({"role": "assistant", "content": resp.text, "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.args)}}
                for c in resp.tool_calls]})
            card = None
            for c in resp.tool_calls:  # every tool call gets a tool message, even when skipped
                tool_calls += 1
                if card is not None or tool_calls > self.limits.max_tool_calls:
                    content = sanitize.wrap(c.name, {"ok": False, "error": "skipped"})[0]
                else:
                    content, card = self._run_tool(self.store.get_errand(e.id), c)
                turn.append({"role": "tool", "tool_call_id": c.id, "content": content})
            if card is not None:
                return self._card(self.store.get_errand(e.id), resp.text, card)

    def _run_tool(self, e: Errand, c: ToolCall) -> tuple[str, dict | None]:
        t0, card = time.perf_counter(), None
        if c.parse_error:
            result = ToolResult.fail("invalid_arguments", message=c.parse_error)
        elif not (d := self.gate.check_llm(e, c.name, c.args)).allow:
            result = ToolResult.fail("denied", message=d.reason)
        else:
            try:
                result, card = self._dispatch(e, c.name, c.args)
            except Exception as err:  # malformed tool output or unexpected error: report, don't crash
                self.tracer.emit("error", tool=c.name, error=repr(err))
                result = ToolResult.fail("tool_error", message=f"{c.name} returned something unusable")
        content, flags = sanitize.wrap(c.name, result.model_dump(exclude_none=True))
        self.tracer.emit("tool_call", actor="llm", errand_id=e.id, tool=c.name, args=c.args, ok=result.ok,
                         error=result.error.model_dump() if result.error else None, sent_to_model=content,
                         latency_ms=round((time.perf_counter() - t0) * 1000, 2))
        if flags:
            self.tracer.emit("injection_flagged", errand_id=e.id, tool=c.name, fields=flags)
        return content, card

    def _dispatch(self, e: Errand, name: str, args: dict) -> tuple[ToolResult, dict | None]:
        if name == "update_request":
            return internal_tools.update_request(self.store, self.tracer, e, args), None
        if name == "request_approval":
            return internal_tools.request_approval(self.store, self.tracer, e, args, self.settings.approval_ttl_s)
        if name == "propose_memory":
            return internal_tools.propose_memory(self.memory, self.tracer, e, args), None
        if name == "memory_op":
            return internal_tools.memory_op(self.memory, self.tracer, e, args), None
        try:
            result = self.call(name, args)
        except ToolTimeout:
            return ToolResult.fail("timeout", retryable=True, message=f"{name} timed out"), None
        self.store.record_tool_refs(e.id, name, result)
        if not result.ok:
            return result, None
        if name == "search_providers":
            options = compact_options(result.data["providers"], e.request)
            e.selected["options"] = options
            self.store.save_errand(e)
            return ToolResult.success(providers=options), None
        if name == "hold_slot":
            prov = self.store.ref(e.id, "provider", args["provider_id"]) or {}
            hold = self.store.record_hold(e.id, args["provider_id"], args["slot"], result, prov.get("price"))
            transition(self.store, self.tracer, e, S.HOLDING, "slot_held")
            budget = e.request.budget_max
            return ToolResult.success(**result.data, search_price=hold.search_price,
                                      price_changed=hold.search_price is not None and hold.quoted_price != hold.search_price,
                                      over_budget=budget is not None and hold.quoted_price > budget), None
        return result, None

    # ---------- replies ----------
    def _reply(self, session_id: str, kind: str, text: str, card: dict | None = None) -> dict[str, Any]:
        self.store.add_message(session_id, "assistant", text, kind)
        self.tracer.emit("assistant_msg", kind=kind, text=text, card=card)
        return {"kind": kind, "text": text, **({"card": card} if card else {})}

    def _chat(self, e: Errand, text: str | None) -> dict[str, Any]:
        e = self.store.get_errand(e.id)
        if missing := [f for f in e.request.missing(e.kind) if f not in e.asked]:
            e.asked += missing  # the model just asked for these; don't let it ask again
            self.store.save_errand(e)
        text = text or "Sorry, could you rephrase that?"
        if not claim_guard.allowed(text, e.state):
            replacement = render(for_state(self.store, e))
            self.tracer.emit("claim_blocked", errand_id=e.id, state=e.state, original=text, replacement=replacement)
            text = replacement
        return self._reply(e.session_id, "chat", text)

    def _card(self, e: Errand, preface: str | None, card: dict) -> list[dict[str, Any]]:
        out = []
        if preface and claim_guard.allowed(preface, e.state):
            out.append(self._reply(e.session_id, "chat", preface))
        out.append(self._reply(e.session_id, "card", card["text"], card))
        return out

    def _outcome(self, e: Errand, o: Outcome) -> list[dict[str, Any]]:
        out = [self._reply(e.session_id, "template", render(o))]
        if o.card:
            out.append(self._reply(e.session_id, "card", o.card["text"], o.card))
        if o.kind in SUCCESS and (m := self.memory.next_prompt(e.user_id, e.session_id)):
            self.memory.mark_prompted(m.id)  # ask once, after the errand, about one inferred preference
            out.append(self._reply(e.session_id, "memory_prompt",
                                   f'Should I remember "{m.key.replace("_", " ")}: {m.value}" for future requests? '
                                   'Reply "remember yes" or "remember no".', {"memory_id": m.id}))
        return out

    def _answer_memory_prompt(self, session_id: str, accept: bool) -> list[dict[str, Any]]:
        user = self.store.session_user(session_id)
        m = self.memory.last_prompted(user, session_id)
        if m is None:
            return [self._reply(session_id, "template", "There's nothing waiting for a yes or no.")]
        self.memory.confirm(m.id, accept)
        self.tracer.emit("memory_confirmed" if accept else "memory_rejected", key=m.key, value=m.value)
        text = (f'Saved: "{m.key.replace("_", " ")}: {m.value}".' if accept
                else f'Okay, I won\'t remember "{m.key.replace("_", " ")}".')
        return [self._reply(session_id, "template", text)]

    def _limit(self, e: Errand, which: str) -> list[dict[str, Any]]:
        self.tracer.emit("limit_hit", errand_id=e.id, limit=which, state=e.state)
        status = render(for_state(self.store, e))
        if e.state in LIMIT_STATES:
            transition(self.store, self.tracer, e, S.ABORTED_LIMIT, f"limit:{which}")
        return self._outcome(e, Outcome("limit", e.id, {"status": status}))
