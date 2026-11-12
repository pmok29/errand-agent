# Errand Agent: design note

**Shape.** One LLM agent in a deterministic shell: *the LLM proposes, code disposes.* The model reads
requests and calls read-only tools; code owns the state machine, gate, approvals, side effects,
verification and outcome text. The hard parts belong in code, not in more agents.

**State model.** Each errand (`book | reschedule | cancel | rebook`) is a persisted state machine with
optimistic-lock writes. The model sees only the tools its state allows; the gate re-checks every call.

| State | LLM may call |
|---|---|
| GATHERING | `update_request`, `list_bookings`, memory tools |
| COMPARING / HOLDING | + `search_providers`, `get_provider`, `hold_slot`, `request_approval` |
| AWAITING_APPROVAL, EXECUTING, VERIFYING | nothing: code runs these states |
| DONE / FAILED / UNVERIFIED / ABORTED_LIMIT | `list_bookings`, memory tools |

`book`, `cancel`, `reschedule` are never shown to the model; IDs and slots must come from earlier tool
results. Limits: 12 LLM turns, 25 tool calls, 180 s per run (LLM retries and waits included) → ABORTED_LIMIT
plus a status. Chat goes to the errand the user started, not a proactive rebook; a new kind of errand
mid-flow closes the old one (FAILED, `replaced_by_new_request`).

**Approval enforcement.** `request_approval` builds an `ApprovalDraft` from persisted records (the hold's
quote, or a booking seen in `list_bookings`), never from model arguments; card and hash both derive from
it. Only an exact `CONFIRM <code>` (regex, in code) approves, and each card has its own code. One
transaction recomputes the draft (changed provider, slot or price → rejected), consumes the approval,
writes the write-ahead execution and enters EXECUTING. Approvals expire with the hold; any other reply
dismisses the card.

**Idempotency and recovery.** Key = `sha256(errand_id | action | attempt_no)`. Unprocessed failures
(timeout, 503, `not_booked`) retry with the **same** key; processed ones (`slot_taken`, `slot_conflict`,
`hold_expired`) close the attempt, and the next uses `attempt_no + 1`. The mock replays per key and
rejects a reused key with a new payload. A book timeout always asks `get_booking_status` first; success
is reported only once the provider confirms. Still pending → UNVERIFIED, re-checked by the watcher at
+30 s, +2 min, +10 min, with no new hold for that service meanwhile. On restart, open executions resume by
asking the provider, never by repeating the call.

**Memory and context.** Memories are *explicit* (stated as a standing preference) or *inferred* (a
candidate until the user confirms; a "no" keeps it out). Code checks both against this session's
messages. Users can list, correct (supersede) and forget (every spelling redacted). Each turn sends the
system prompt, a STATE block with ≤ 8 relevant usable memories, this session's history scrubbed of
forgotten values, and this turn's tool outputs, under a hard **6,000-token** budget including tool
schemas (an oversized message is cut). Earlier sessions never go raw; `context_sent` logs what was in and
out, and why. Tool output is wrapped as untrusted and instruction-like text withheld, but safety doesn't
rely on that: an obeyed injection still hits the gate (eval #4).

**Observability.** One JSONL trace per run: context sent, LLM calls (tokens, latency, est. cost), gate
decisions, tool args and sanitised results, transitions, approvals, executions, outcomes; PII masked.
The UI streams it beside the chat; evals assert invariants I1–I7 on it.

**Trade-offs.** *Hold before approval*: the card shows the real quote, at the cost of unused holds.
*Code-owned outcome text* can't overclaim but is less fluent. *Offline evals* use a scripted model that
reads the same context a real LLM gets: deterministic, but language understanding is only exercised in
real-LLM runs.

**Trace excerpt (timeout after success):** `traces/timeout_after_success.jsonl`, real Groq run (`make trace-demo`).

```
 18 tool_call         get_provider(p-plb-03) + injection_flagged   ← planted review never reaches the model
 53 approval_created  code FRPX (QuickFix, Tue 10 Nov 10:00, ₹1,150: from the hold record)
 59 user_msg          "CONFIRM FRPX"                    ← matched in code; no LLM call from here on
 60 state_transition  AWAITING_APPROVAL → EXECUTING     ← approval consumed + write-ahead row, one tx
 63 tool_call         book(h-2b8ec3bd, key 1a4b3c6b…) → TIMEOUT
 64 state_transition  EXECUTING → VERIFYING             ← never retry blind: ask the source of truth
 66 tool_call         get_booking_status(key 1a4b3c6b…) → booked b-2c0313ce
 69 assistant_msg     template "Booked: QuickFix Plumbing … (booking b-2c0313ce)"   ← 1 book call, 1 booking
```
