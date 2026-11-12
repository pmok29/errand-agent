"""Blocks LLM replies that claim an outcome the persisted state doesn't support."""
from __future__ import annotations

import re

from errand.models import ErrandState

_DONE = r"(booked|cancel+ed|rescheduled|confirmed|moved)"
# First-person or passive claims of a completed side effect. Plain history ("you booked Suresh
# last time") is deliberately not matched.
CLAIM = re.compile(
    r"✅|^\s*" + _DONE + r"\b|"
    r"\b(i'?ve|i have|i)\s+(just\s+|successfully\s+)?" + _DONE + r"\b|"
    r"(\b(is|are|has been|have been|was|been|now)|'s)\s+(successfully\s+)?" + _DONE + r"\b|"
    r"\bbooking (is )?confirmed\b|\bsuccessfully " + _DONE + r"\b|"
    r"\b(you'?re|you are) (all )?set\b|\ball (set|done)\b", re.I)
# Questions, negations, conditionals and future tense are not claims of a completed outcome.
# "after/before/when/once/until" hedge only when a clause follows ("after you confirm"), not a time
# ("confirmed for after 10am" is still a claim).
HEDGE = re.compile(
    r"\b(not|n't|never|no longer|yet|will|would|shall|should|if|can|could|"
    r"want|like me|to be|pending|waiting|haven'?t|hasn'?t)\b|"
    r"\b(once|after|before|when|until)\b(?!\s+(\d|noon|midday|midnight|lunch))|\?", re.I)
SENTENCE = re.compile(r"[^.!?\n]+[.!?]?")


def claims_outcome(text: str) -> bool:
    for s in SENTENCE.findall(text):
        if CLAIM.search(s) and not HEDGE.search(s):
            return True
    return False


def allowed(text: str, state: ErrandState) -> bool:
    return state == ErrandState.DONE or not claims_outcome(text)
