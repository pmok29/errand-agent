# Bugs hit while building

Real bugs found by running the code, each with the regression test that now guards it.

## 1. The memory cap dropped the user's preferred area

**What broke.** With the 55-memory seed profile, a plumber request was built with 8 memories that did
not include `preferred_area: Indiranagar`. The agent would have asked for the area again, which breaks
"don't re-ask facts already supplied" for anyone with a stored area.

**Why.** `MemoryStore.select` ranked memories as "service-specific first, then global, newest first",
then cut at 8. The 4 plumber memories took half the budget, and the 4 *newest* global memories (pets,
payment, floor, verified) beat `preferred_area`, which was the oldest global row. Recency is a poor
proxy for usefulness: the memories that fill a request field matter most whatever their age.

**Fix.** A `relevance()` tier ranks memories that fill a request field (`preferred_area`,
`earliest_time`, `budget_<service>`, `preferred_provider_<service>`, `<service>_time`) above
service-specific ones, which rank above global ones. Recency only breaks ties within a tier.
(`errand/agent/memory.py`)

**Regression test.** `tests/unit/test_memory.py::test_request_filling_memories_survive_the_cap`
seeds the full profile and checks that `preferred_area` and `earliest_time` are selected for every
service. With the old ranking restored it fails, along with
`tests/scenarios/test_s6_memory.py::test_24_*` and `::test_21_*`.

## 2. A dismissed approval card's code came back to life

**What broke.** The user got a card, replied "yeah ok" instead of the code, and the agent showed the
card again. That part is correct: anything other than the exact code is not approval. But the
re-shown card had the *same* code as the dismissed one, so the old code still worked. It surfaced
when the dismissed card started going back to the still-valid hold (HOLDING) rather than COMPARING:
`test_reply_without_code_is_not_approval` expected the stale code to be rejected and got
`Booked: PipeDoctor …` instead.

**Why.** `code_for(hash)` derived the 4-character code from the draft hash alone. Same provider,
slot and price meant the same hash, so every re-issue of identical terms produced the same code. An
invalidated code then matched the new pending approval. "This code belongs to this card" was never
actually enforced. Nothing was booked on terms the user hadn't seen, but a code typed from an old,
dismissed message must not count as approval.

**Fix.** The code is now `base32(sha256(hash | approval_id))[:4]`, so each card has its own code.
The hash is unchanged and still drives the material-change check. (`errand/agent/approvals.py`)

**Regression tests.** `tests/unit/test_core.py::test_reissued_card_with_same_terms_gets_a_new_code`
re-issues identical terms and checks that the old code is rejected as `invalidated`.
`tests/unit/test_orchestrator.py::test_reply_without_code_is_not_approval` covers the same path
through the full agent. Both fail with the unsalted code restored.

## 3. "Cancel my plumber booking" stayed a booking

**What broke.** Writing the cancel and reschedule scenarios (#15, #17), "Cancel my plumber booking"
made the agent ask *"Which area should I search in? Which day works?"*: the questions for a new
booking. The model had called `update_request(kind="cancel", service="plumber")` correctly, and the
call returned ok.

**Why.** `Store.save_errand` updated state, request, asked fields and selections, but not `kind`.
`update_request` changed `e.kind` in memory and saved it, the save silently dropped it, and the next
turn reloaded the errand as a `book`. So the missing-details list, the allowed actions
(`request_approval` must match the kind) and the card type were all wrong. Every errand starts as a
`book` until the first message is understood, so no cancel or reschedule could ever work through chat.
Earlier tests built cancel and reschedule errands directly with `create_errand(kind=…)` and never
exercised the change.

**Fix.** `save_errand` now writes `kind`. (`errand/agent/store.py`)

**Regression test.** `tests/unit/test_orchestrator.py::test_change_of_errand_kind_is_persisted`.
Scenarios #15 and #17 also fail without the fix.

## 4. A stale confirmation code broke the card that replaced it

**What broke.** In scenario #8, the user got a card, changed the time, and got a new card. They then
typed the *old* code: it was correctly rejected ("no longer applies"). But typing the *new* code
afterwards returned "Something went wrong on my side". Nothing was booked: it failed closed, but
the user could not complete a valid approval.

**Why.** `_confirm` moved any errand in AWAITING_APPROVAL to COMPARING whenever the code it was
given was invalidated, expired or changed. That is right when the code belongs to the card on
screen, but here the code was from an earlier card while a newer one was still pending. The errand
left AWAITING_APPROVAL with a live approval. Consuming that approval then tried COMPARING → EXECUTING,
an illegal transition, and the exception surfaced as the generic error.

**Fix.** If a newer approval is still pending, a stale code leaves the state alone. The reply says
the old code no longer applies and repeats the current one. The state only moves back when no pending
approval remains. (`errand/agent/orchestrator.py`)

**Regression test.** `tests/unit/test_orchestrator.py::test_stale_code_keeps_the_newer_card_open`.
Scenario #8 also fails without the fix.

## 5. A ✅ in a comparison table was treated as a false success claim

**What broke.** In the first real-LLM run (Groq, `openai/gpt-oss-120b`), the model answered the
plumber request with a correct comparison table. The user got *"Nothing has been booked or changed
yet."* instead, twice in a row. The trace showed two `claim_blocked` events in COMPARING, nothing
else wrong.

**Why.** The claim guard's pattern started with a bare `✅`, on the theory that a checkmark from the
model means "done". gpt-oss puts ✅ in a "Verified" column, so every comparison table counted as a
success claim and was replaced by the truthful status template. Safety held, but the agent became
useless at exactly the step where the user chooses a provider. The scripted model never used ✅
that way, so the offline evals couldn't catch it.

**Fix.** A ✅ counts as a claim only next to a done-word ("✅ Booked", "booked ✅", "done ✅"). The
other claim patterns ("Booked …", "you're all set", "is confirmed", first-person "I've booked")
are unchanged. (`errand/agent/claim_guard.py`)

**Regression test.** `tests/unit/test_core.py::test_non_claims_pass` now includes a table row with
a ✅ in the verified column and "Verified ✅, 7-day warranty …".
`tests/unit/test_core.py::test_claims_are_detected` adds "✅ Booked for Thursday 9am" and "done ✅".

## 6. "Not before 10 a.m." became a 10:00–10:00 window, and the agent said there were no slots

**What broke.** Real-LLM run of *"Book the same electrician as last time, but not before 10 a.m."*:
the model correctly called `list_bookings`, found Suresh Electricals and recorded `provider_pref`. It
then told the user Suresh had *no slots at 10 a.m. or later this week*, which was false. Next it
tried to hold an invented 10:30 slot, which the gate correctly denied.

**Why.** The model sent `update_request(time_from="10:00")` with no `time_to`. `_pair` treated a
single end as a single value (`lo, hi = lo or hi, hi or lo`), which is right for dates ("Saturday"
= that day) but wrong for times. The window became 10:00–10:00, `slot_outside_request` rejected
every slot, and the search options for Suresh had an empty slot list. The scripted model always
sends both ends, so no offline scenario took this path.

**Fix.** `_pair` takes `open_ends`: a single time bound is open-ended (`10:00`–`23:59`, or
`00:00`–`HH:MM`). A single date still means that one day. (`errand/agent/internal_tools.py`)

**Regression test.** `tests/unit/test_orchestrator.py::test_single_time_bound_is_open_ended` replays
the real model's call shape with a canned LLM. It checks the window and that Suresh's options have
slots at or after 10:00. It fails with the old `_pair`.

## 7. The claim guard blocked the agent's own "nothing was booked" message

**What broke.** In a real-LLM run that hit Groq's rate limit, the agent replied with its templated
*"I'm having trouble reaching my language model right now. Nothing was booked or changed…"*. On the
next turn the fallback model (qwen3.8-27b) repeated that sentence from the history, and the claim
guard blocked it as a false success claim.

**Why.** Two things combined. The claim pattern matches passive completions ("… was booked"), and
the negation list held "not / n't / never / no longer" but not "nothing". So "Nothing was booked"
read as a claim, and several code-owned status templates would have tripped the guard if a model
ever echoed them. Also, templates sat in the history as ordinary assistant turns, which invited the
model to imitate them.

**Fix.** "nothing" is now a negation in `claim_guard.HEDGE`. History messages that code wrote
(templates, cards, notifications) reach the model labelled `[template from the system] …`, so models
don't present them as their own words. (`errand/agent/claim_guard.py`, `errand/agent/context.py`)

**Regression test.** `tests/unit/test_core.py::test_no_status_template_reads_as_a_success_claim`
renders every non-success template, plus the LLM-unavailable and internal-error messages, and checks
that none reads as a claim. `test_non_claims_pass` gains the exact sentence. Both fail without the
fix.
