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
