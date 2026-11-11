"""Seed profile for user u1: 55 memories across every status, and a 200-message earlier session.

The earlier session deliberately repeats forgotten values and inferred candidates verbatim, so the
context-budget eval can prove none of them reach the model.
"""
from __future__ import annotations

from errand.agent.memory import MemoryStore
from errand.agent.store import Store
from errand.models import MemorySource as Src, MemoryStatus as St

OLD_SESSION_MESSAGES = 200

# (key, value, scope) — explicit, active (30)
EXPLICIT = [
    ("preferred_area", "Indiranagar", "global"), ("earliest_time", "08:00", "global"),
    ("language", "English", "global"), ("contact_preference", "WhatsApp", "global"),
    ("home_type", "2BHK apartment", "global"), ("parking", "visitor parking near gate 2", "global"),
    ("pets", "has a friendly dog", "global"), ("prefers_verified", "yes", "global"),
    ("payment", "UPI", "global"), ("floor", "4th floor, lift available", "global"),
    ("budget_plumber", "1500", "service:plumber"), ("plumber_notes", "kitchen sink is under-mount", "service:plumber"),
    ("plumber_brand_fittings", "Jaquar", "service:plumber"), ("plumber_time", "mornings", "service:plumber"),
    ("preferred_provider_electrician", "p-ele-02 Suresh Electricals", "service:electrician"),
    ("electrician_time", "not before 10am", "service:electrician"),
    ("electrician_notes", "main board is in the utility area", "service:electrician"),
    ("budget_electrician", "1200", "service:electrician"),
    ("ac_brand", "Daikin", "service:ac_servicing"), ("ac_units", "2", "service:ac_servicing"),
    ("ac_preferred_day", "Saturday", "service:ac_servicing"), ("ac_notes", "bedroom unit drips", "service:ac_servicing"),
    ("budget_ac_servicing", "1800", "service:ac_servicing"),
    ("cleaner_supplies", "bring own supplies", "service:cleaner"), ("cleaning_type", "deep clean", "service:cleaner"),
    ("cleaner_rooms", "kitchen and two bathrooms", "service:cleaner"), ("cleaner_day", "Saturday", "service:cleaner"),
    ("cleaner_avoid", "no bleach on marble", "service:cleaner"), ("budget_cleaner_max", "2200", "service:cleaner"),
    ("cleaner_time", "after 10am", "service:cleaner"),
]
# superseded by the explicit row with the same key (8)
SUPERSEDED = [
    ("preferred_area", "HSR Layout", "global"), ("earliest_time", "07:00", "global"),
    ("budget_plumber", "1200", "service:plumber"), ("contact_preference", "phone call", "global"),
    ("ac_units", "1", "service:ac_servicing"), ("cleaner_day", "Sunday", "service:cleaner"),
    ("language", "Hindi", "global"), ("payment", "cash", "global"),
]
# forgotten by the user (6); every one appears verbatim in the old session
FORGOTTEN = [
    ("phone", "+91 98450 12345"), ("gate_code", "4821"), ("old_address", "12th Main, HAL 2nd Stage"),
    ("wifi_password", "sunflower42"), ("spare_key_location", "under the doormat"), ("office_shift", "Infosys night shift"),
]
# inferred from the old session, never confirmed (6)
CANDIDATES = [
    ("prefers_evenings", "evenings", "global"), ("prefers_cheapest", "cheapest option", "global"),
    ("weekend_only", "weekends only", "global"), ("likes_provider", "QuickFix Plumbing", "service:plumber"),
    ("budget_cleaner", "2000 rupees", "service:cleaner"), ("ac_time", "late afternoons", "service:ac_servicing"),
]
# unrelated to errands (5)
IRRELEVANT = [
    ("likes", "cricket"), ("favourite_food", "masala dosa"), ("birthday_month", "March"),
    ("car", "Maruti Swift"), ("kids_school", "Greenwood High"),
]


def _old_session_lines() -> list[tuple[str, str]]:
    facts = [
        "my gate code is 4821, tell the plumber", "call me on +91 98450 12345 when you reach",
        "I used to live at 12th Main, HAL 2nd Stage", "wifi password is sunflower42 if they need it",
        "the spare key is under the doormat", "I'm on the Infosys night shift so mornings are tough",
        "honestly I prefer evenings", "just get me the cheapest option", "weekends only for me",
        "QuickFix Plumbing was great", "keep the cleaner under 2000 rupees", "AC folks in late afternoons please",
    ]
    filler = ["can you check the electrician slots", "what did the AC service cost last time",
              "ok thanks", "move it if needed", "any plumber with good reviews?", "never mind"]
    lines, i = [], 0
    while len(lines) < OLD_SESSION_MESSAGES:
        text = facts[i % len(facts)] if i % 3 == 0 else filler[i % len(filler)]
        lines.append(("user", text))
        lines.append(("assistant", f"Noted. (reply {i})"))
        i += 1
    return lines[:OLD_SESSION_MESSAGES]


def seed_user(store: Store, user_id: str = "u1") -> str:
    """Seeds memories and an old session; returns the old session id."""
    mem = MemoryStore(store)
    old = store.create_session(user_id)
    for role, text in _old_session_lines():
        store.add_message(old, role, text)
    with store.db.tx():
        for key, value, scope in SUPERSEDED:
            m = mem.add(user_id, key, value, Src.EXPLICIT, St.ACTIVE, scope, evidence_session_id=old)
            mem._set(m.id, status=St.SUPERSEDED)
        for key, value, scope in EXPLICIT:
            m = mem.add(user_id, key, value, Src.EXPLICIT, St.ACTIVE, scope, evidence_session_id=old)
            store.db.conn.execute("UPDATE memories SET superseded_by=? WHERE user_id=? AND key=? AND status=?",
                                  (m.id, user_id, key, St.SUPERSEDED))
        for key, value in FORGOTTEN:
            mem.add(user_id, key, value, Src.EXPLICIT, St.ACTIVE, evidence_session_id=old)
            mem.forget(user_id, key)
        for key, value, scope in CANDIDATES:
            mem.add(user_id, key, value, Src.INFERRED, St.CANDIDATE, scope, evidence_session_id=old)
        for key, value in IRRELEVANT:
            mem.add(user_id, key, value, Src.EXPLICIT, St.ACTIVE, "misc", evidence_session_id=old)
    return old


def forgotten_values() -> list[str]:
    """Must never appear anywhere in any context sent to the model."""
    return [v for _, v in FORGOTTEN]


def not_in_memory_block() -> list[str]:
    """Must never be presented to the model as a remembered fact (they may still occur as live data,
    e.g. "QuickFix Plumbing" in search results)."""
    return [v for _, v, _ in CANDIDATES] + [v for _, v, _ in SUPERSEDED] + [v for _, v in IRRELEVANT]
