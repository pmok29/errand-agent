# Errand Agent: design note

**Shape.** One LLM agent in a deterministic shell: *the LLM proposes, code disposes.* The model
understands requests and picks read-only tools. Code owns the state machine, gate, approvals, side
effects, verification and outcome messages. The hard parts belong in code, not in more agents.

**State model.** Each errand (`book | reschedule | cancel | rebook`) is a persisted state machine
(optimistic-lock writes). The model sees only the tools its state allows; the gate re-checks each call.

| State | LLM may call |
|---|---|
| GATHERING | `update_request`, `list_bookings`, memory tools |
| COMPARING / HOLDING | + `search_providers`, `get_provider`, `hold_slot`, `request_approval` |
| AWAITING_APPROVAL, EXECUTING, VERIFYING | nothing: code runs these states |
| DONE / FAILED / UNVERIFIED / ABORTED_LIMIT | `list_bookings`, memory tools |

`book`, `cancel` and `reschedule` are never shown to the model. IDs and slots must come from earlier
tool results. Limits: 12 LLM turns, 25 tool calls, 180 s per run (incl. rate-limit waits); hitting one gives ABORTED_LIMIT plus a status.

**Approval enforcement.** `request_approval` builds an `ApprovalDraft` from persisted records (the
hold's quote, or a booking from `list_bookings`), never from model arguments. The card and the hash
both come from that draft. Only an exact `CONFIRM <code>` (regex, in code) approves; each card has its
own code. One transaction recomputes the draft (changed provider, slot or price → rejected), consumes
the approval, writes the write-ahead execution and enters EXECUTING. Approvals expire with the hold;
any other reply dismisses the card.

**Idempotency and recovery.** Key = `sha256(errand_id | action | attempt_no)`. An unprocessed failure
(timeout, 503, status `not_booked`) retries with the **same** key. A processed one (`slot_taken`,
`slot_conflict`, `hold_expired`) closes the attempt, and the next one uses `attempt_no + 1`. The mock
replays results per key and rejects a reused key with a new payload. A book timeout always queries
`get_booking_status` first; success is reported only once it (or `list_bookings`) confirms. Still
pending after 4 polls → UNVERIFIED, re-checked by the watcher at +30 s, +2 min, +10 min. On restart,
open executions resume by asking the provider, never by repeating the side effect.

**Memory and context.** Memories are *explicit* (the user said it; checked against this session's
messages) or *inferred* (a candidate until confirmed). Users can list, correct (supersede) and forget
(redact every spelling: digits, +91, ₹/Rs). Each turn sends: system prompt, a STATE block with ≤ 8
relevant usable memories, this session's history scrubbed of forgotten values, and this turn's tool
outputs, under a fixed **6,000-token** budget that includes the tool schemas. Earlier sessions never go raw. `context_sent` logs
what was included and excluded, and why. Tool results are wrapped as untrusted; instruction-like free
text is withheld. Safety doesn't rely on that: an obeyed injection still hits the gate (eval #4).

**Observability.** One JSONL trace per run: context sent, LLM calls (tokens, latency, est. cost), gate
decisions, tool args and sanitised results, transitions, approvals, executions, outcomes; phones and
emails masked. The UI streams it beside the chat; evals assert invariants I1–I7 on it.

**Trade-offs.** *Hold before approval*: the card shows the real quote, at the cost of unused holds.
*Code-owned outcome text* can't overclaim but is less fluent (a claim guard also blocks "booked ✅").
*Offline evals* use a scripted model reading the same context a real LLM gets, plus bad variants:
deterministic, but language understanding is only exercised in real-LLM runs.

**Trace excerpt (timeout after success)**: `traces/timeout_after_success.jsonl`, a real Groq run
recorded with `make trace-demo` (preset `timeout_booked`). Seq numbers are from the file.

```
 11 llm_call          qwen3.8-27b → search_providers       ← gpt-oss at its daily cap: per-call fallback, traced
 18 tool_call         get_provider(p-plb-03) + injection_flagged   ← the planted review never reaches the model
 40 assistant_msg     "I've held the slot … Please confirm to book it."   ← prose: nothing can act on this
 53 approval_created  code FRPX (QuickFix, Tue 10 Nov 10:00, ₹1,150: from the hold record)
 59 user_msg          "CONFIRM FRPX"                         ← matched in code; no LLM call from here on
 60 state_transition  AWAITING_APPROVAL → EXECUTING          ← approval consumed + write-ahead row, one tx
 62 gate_decision     executor book allow                    ← args must equal the write-ahead record
 63 tool_call         book(h-2b8ec3bd, key 1a4b3c6b…) → TIMEOUT
 64 state_transition  EXECUTING → VERIFYING                  ← never retry blind: ask the source of truth
 66 tool_call         get_booking_status(key 1a4b3c6b…) → booked b-2c0313ce
 67 state_transition  VERIFYING → DONE (booked_verified)
 69 assistant_msg     template "Booked: QuickFix Plumbing … (booking b-2c0313ce)"   ← 1 book call, 1 booking
```
