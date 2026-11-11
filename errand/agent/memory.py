"""Persistent user memory: explicit vs inferred facts, correction, forgetting, and what may reach the model."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from errand.agent.store import Store
from errand.db import new_id
from errand.models import Memory, MemorySource as Src, MemoryStatus as St

TOKEN = re.compile(r"[a-z]+|\d+")
IGNORED_TOKENS = {"00"}          # "10:00" should match "10 a.m."
USABLE = (St.ACTIVE, St.CONFIRMED)
# Memories that can fill a request field rank above everything else, whatever their age.
REQUEST_KEYS = ("preferred_area", "earliest_time", "latest_time")
SERVICE_REQUEST_KEYS = ("budget_{s}", "preferred_provider_{s}", "{s}_time")


def relevance(m: Memory, service: str | None) -> int:
    """Lower is more relevant: 0 fills a request field, 1 service-specific, 2 global."""
    if m.key in REQUEST_KEYS or (service and m.key in {k.format(s=service) for k in SERVICE_REQUEST_KEYS}):
        return 0
    return 1 if m.scope != "global" else 2


def tokens(text: str) -> set[str]:
    return set(TOKEN.findall(text.lower())) - IGNORED_TOKENS


def variants(value: str) -> list[str]:
    """Spellings of a value to scrub: as written, digits only, and money with or without a unit."""
    out = {value}
    digits = re.sub(r"\D", "", value)
    if len(digits) >= 4:
        out.add(digits)
    if re.match(r"\s*\+?91[\s-]", value):  # Indian phone written with a country code
        local = re.sub(r"^\s*\+?91[\s-]*", "", value)
        out |= {local, re.sub(r"\D", "", local)}
    if m := re.fullmatch(r"(?:₹|rs\.?\s*)?([\d,]+)", value.strip(), re.I):
        for n in {m.group(1), m.group(1).replace(",", "")}:
            out |= {n, f"₹{n}", f"Rs {n}", f"Rs.{n}"}
    return sorted((v for v in out if len(v) >= 3), key=len, reverse=True)


@dataclass
class Selection:
    included: list[Memory]
    excluded: list[tuple[Memory, str]]      # (memory, reason)


class MemoryStore:
    def __init__(self, store: Store):
        self.store, self.db = store, store.db

    def _row(self, r) -> Memory:
        return Memory(**{k: r[k] for k in r.keys() if k not in ("created_at", "updated_at")})

    def all(self, user_id: str) -> list[Memory]:
        return [self._row(r) for r in self.db.all("SELECT * FROM memories WHERE user_id=? ORDER BY rowid", user_id)]

    def get(self, memory_id: str) -> Memory | None:
        r = self.db.one("SELECT * FROM memories WHERE id=?", memory_id)
        return self._row(r) if r else None

    def add(self, user_id: str, key: str, value: str, source: Src, status: St, scope: str = "global",
            evidence_msg_id: str | None = None, evidence_session_id: str | None = None) -> Memory:
        now = self.store._now()
        m = Memory(id=new_id("mem"), user_id=user_id, key=key, value=value, scope=scope, source=source,
                   status=status, evidence_msg_id=evidence_msg_id, evidence_session_id=evidence_session_id)
        self.db.conn.execute(
            "INSERT INTO memories (id,user_id,key,value,scope,source,status,evidence_msg_id,evidence_session_id,"
            "superseded_by,prompted,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,NULL,0,?,?)",
            (m.id, user_id, key, value, scope, source, status, evidence_msg_id, evidence_session_id, now, now))
        return m

    def _set(self, memory_id: str, **fields) -> None:
        sets = ", ".join(f"{k}=?" for k in fields)
        self.db.conn.execute(f"UPDATE memories SET {sets}, updated_at=? WHERE id=?",
                             (*fields.values(), self.store._now(), memory_id))

    def current(self, user_id: str, key: str) -> list[Memory]:
        """Rows that currently stand for `key` (usable or awaiting confirmation)."""
        return [m for m in self.all(user_id) if m.key == key and m.status in (*USABLE, St.CANDIDATE)]

    def value(self, user_id: str, key: str) -> str | None:
        """The usable (explicit or confirmed) value for `key`, if any. Candidates never count."""
        rows = [m for m in self.current(user_id, key) if m.status in USABLE]
        return rows[-1].value if rows else None

    # ---------- user operations ----------
    def visible(self, user_id: str) -> list[Memory]:
        """What the user sees on inspect: everything not forgotten or superseded."""
        return [m for m in self.all(user_id) if m.status in (*USABLE, St.CANDIDATE)]

    def correct(self, user_id: str, key: str, value: str, scope: str = "global",
                evidence_msg_id: str | None = None, session_id: str | None = None) -> Memory:
        with self.db.tx():
            new = self.add(user_id, key, value, Src.EXPLICIT, St.ACTIVE, scope, evidence_msg_id, session_id)
            for old in self.current(user_id, key):
                if old.id != new.id:
                    self._set(old.id, status=St.SUPERSEDED, superseded_by=new.id)
        return new

    def forget(self, user_id: str, key: str) -> list[Memory]:
        forgotten = self.current(user_id, key)
        with self.db.tx():
            for m in forgotten:
                self._set(m.id, status=St.FORGOTTEN)
                self._redact(user_id, m.value, "forgotten")
        return forgotten

    def confirm(self, memory_id: str, accept: bool) -> Memory:
        m = self.get(memory_id)
        if accept:
            with self.db.tx():
                for old in self.current(m.user_id, m.key):
                    if old.id != m.id:
                        self._set(old.id, status=St.SUPERSEDED, superseded_by=m.id)
                self._set(m.id, status=St.CONFIRMED)
        else:
            self._set(m.id, status=St.FORGOTTEN)
            self._redact(m.user_id, m.value, "rejected_inference")
        return self.get(memory_id)

    def _redact(self, user_id: str, value: str, reason: str) -> None:
        self.db.conn.execute("INSERT INTO redactions VALUES (?,?,?,?,?)",
                             (user_id, value, json.dumps(variants(value)), reason, self.store._now()))

    # ---------- proposals from the model ----------
    def stated_by_user(self, session_id: str, value: str) -> str | None:
        """Provenance: the id of a user message in this session that states `value`, if any."""
        want = tokens(value)
        if not want:
            return None
        for r in self.db.all("SELECT id, content FROM messages WHERE session_id=? AND role='user' ORDER BY rowid DESC",
                             session_id):
            if want <= tokens(r["content"]):
                return r["id"]
        return None

    def next_prompt(self, user_id: str, session_id: str) -> Memory | None:
        rows = [m for m in self.all(user_id) if m.status == St.CANDIDATE and not m.prompted
                and m.evidence_session_id == session_id]
        return rows[0] if rows else None

    def mark_prompted(self, memory_id: str) -> None:
        self._set(memory_id, prompted=1)

    def last_prompted(self, user_id: str, session_id: str) -> Memory | None:
        rows = [m for m in self.all(user_id) if m.status == St.CANDIDATE and m.prompted
                and m.evidence_session_id == session_id]
        return rows[-1] if rows else None

    # ---------- what the model may see ----------
    def select(self, user_id: str, service: str | None, limit: int) -> Selection:
        included, excluded = [], []
        ranked: list[tuple[int, Memory]] = []
        for age, m in enumerate(self.all(user_id)):  # all() is oldest first
            if m.status == St.FORGOTTEN:
                excluded.append((m, "forgotten"))
            elif m.status == St.SUPERSEDED:
                excluded.append((m, "superseded"))
            elif m.status == St.CANDIDATE:
                excluded.append((m, "unconfirmed_inference"))
            elif m.scope == "global" or (service and m.scope == f"service:{service}"):
                ranked.append((age, m))
            else:
                excluded.append((m, "irrelevant_scope"))
        ranked.sort(key=lambda am: (relevance(am[1], service), -am[0]))  # then newest first
        for i, (_, m) in enumerate(ranked):
            (included.append(m) if i < limit else excluded.append((m, "over_limit")))
        return Selection(included, excluded)

    def redaction_values(self, user_id: str) -> list[str]:
        """Forgotten values (all spellings): scrubbed from any history sent to the model.

        Inferred candidates are kept out by select() instead. Scrubbing their values from history
        would also erase live data, e.g. a provider name the user is choosing right now."""
        values = {v for r in self.db.all("SELECT variants_json FROM redactions WHERE user_id=?", user_id)
                  for v in json.loads(r["variants_json"])}
        return sorted(values, key=len, reverse=True)
