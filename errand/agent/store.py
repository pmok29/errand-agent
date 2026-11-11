"""Persistence for errands, holds, approvals, executions, seen references and notifications."""
from __future__ import annotations

import json
from typing import Any

from errand.clock import Clock
from errand.db import DB, new_id
from errand.models import (Action, Approval, ApprovalDraft, ApprovalStatus, Errand, ErrandKind,
                           ErrandState, Execution, ExecState, Hold, Request, ToolResult)

# Errand states in which the errand is still "in the conversation"
OPEN_STATES = {ErrandState.GATHERING, ErrandState.COMPARING, ErrandState.HOLDING,
               ErrandState.AWAITING_APPROVAL, ErrandState.EXECUTING, ErrandState.VERIFYING}
# Execution states that block another book on the same errand (the errand guard)
GUARD_STATES = {ExecState.EXECUTING, ExecState.VERIFYING, ExecState.DONE, ExecState.UNVERIFIED}


class StaleErrand(Exception):
    """Optimistic-lock failure: the errand changed since it was loaded."""


class Store:
    def __init__(self, db: DB, clock: Clock):
        self.db, self.clock = db, clock

    def _now(self) -> str:
        return self.clock.now().isoformat()

    # ---------- sessions / errands ----------
    def create_session(self, user_id: str) -> str:
        sid = new_id("s")
        self.db.conn.execute("INSERT INTO sessions VALUES (?,?,?)", (sid, user_id, self._now()))
        return sid

    def session_user(self, session_id: str) -> str:
        return self.db.one("SELECT user_id FROM sessions WHERE id=?", session_id)["user_id"]

    def create_errand(self, user_id: str, session_id: str, kind: ErrandKind,
                      request: Request | None = None, parent_errand_id: str | None = None) -> Errand:
        e = Errand(id=new_id("e"), user_id=user_id, session_id=session_id, kind=kind,
                   state=ErrandState.GATHERING, request=request or Request(),
                   parent_errand_id=parent_errand_id)
        now = self._now()
        self.db.conn.execute(
            "INSERT INTO errands (id,user_id,session_id,kind,state,request_json,asked_json,selected_json,"
            "booking_id,parent_errand_id,created_at,updated_at,version) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0)",
            (e.id, user_id, session_id, kind, e.state, e.request.model_dump_json(), "[]", "{}", None,
             parent_errand_id, now, now))
        return e

    def _errand(self, row) -> Errand:
        return Errand(id=row["id"], user_id=row["user_id"], session_id=row["session_id"], kind=row["kind"],
                      state=row["state"], request=Request.model_validate_json(row["request_json"]),
                      asked=json.loads(row["asked_json"]), selected=json.loads(row["selected_json"]),
                      booking_id=row["booking_id"], parent_errand_id=row["parent_errand_id"],
                      version=row["version"])

    def get_errand(self, errand_id: str) -> Errand:
        return self._errand(self.db.one("SELECT * FROM errands WHERE id=?", errand_id))

    def active_errand(self, session_id: str) -> Errand | None:
        rows = self.db.all("SELECT * FROM errands WHERE session_id=? ORDER BY created_at DESC, rowid DESC",
                           session_id)
        for r in rows:
            if r["state"] in OPEN_STATES:
                return self._errand(r)
        return None

    def save_errand(self, e: Errand) -> Errand:
        cur = self.db.conn.execute(
            "UPDATE errands SET state=?, request_json=?, asked_json=?, selected_json=?, booking_id=?, "
            "updated_at=?, version=version+1 WHERE id=? AND version=?",
            (e.state, e.request.model_dump_json(), json.dumps(e.asked), json.dumps(e.selected),
             e.booking_id, self._now(), e.id, e.version))
        if cur.rowcount != 1:
            raise StaleErrand(e.id)
        e.version += 1
        return e

    # ---------- seen references (provenance) ----------
    def add_ref(self, errand_id: str, kind: str, ref: str, meta: dict[str, Any]) -> None:
        self.db.conn.execute("INSERT INTO seen_refs VALUES (?,?,?,?) ON CONFLICT DO UPDATE SET "
                             "meta_json=excluded.meta_json", (errand_id, kind, ref, json.dumps(meta)))

    def ref(self, errand_id: str, kind: str, ref: str) -> dict[str, Any] | None:
        row = self.db.one("SELECT meta_json FROM seen_refs WHERE errand_id=? AND kind=? AND ref=?",
                          errand_id, kind, ref)
        return json.loads(row["meta_json"]) if row else None

    def record_tool_refs(self, errand_id: str, tool: str, result: ToolResult) -> None:
        """Remember every ID a tool result showed us; the gate rejects IDs that never appeared."""
        if not result.ok:
            return
        d = result.data
        if tool == "search_providers" and isinstance(d.get("providers"), list):
            for p in d["providers"]:
                self.add_ref(errand_id, "provider", p["provider_id"], {"name": p["name"], "price": p["price"]})
                for s in p.get("slots", []):
                    self.add_ref(errand_id, "slot", f"{p['provider_id']}|{s}", {})
        elif tool == "get_provider" and "provider_id" in d:
            self.add_ref(errand_id, "provider", d["provider_id"], {"name": d["name"], "price": d["price"]})
        elif tool == "list_bookings" and isinstance(d.get("bookings"), list):
            for b in d["bookings"]:
                prov = b["provider"]
                self.add_ref(errand_id, "booking", b["booking_id"],
                             {"provider_id": prov["provider_id"], "provider_name": prov["name"],
                              "service": b["service"], "slot": b["slot"], "price": b.get("price"),
                              "status": b["status"]})
                self.add_ref(errand_id, "provider", prov["provider_id"],
                             {"name": prov["name"], "price": b.get("price")})

    # ---------- holds ----------
    def add_hold(self, h: Hold) -> None:
        self.db.conn.execute("INSERT INTO holds VALUES (?,?,?,?,?,?,?)",
                             (h.hold_id, h.errand_id, h.provider_id, h.slot, h.quoted_price,
                              h.search_price, h.expires_at))

    def latest_hold(self, errand_id: str) -> Hold | None:
        row = self.db.one("SELECT * FROM holds WHERE errand_id=? ORDER BY rowid DESC LIMIT 1", errand_id)
        return Hold(**dict(row)) if row else None

    def get_hold(self, hold_id: str) -> Hold | None:
        row = self.db.one("SELECT * FROM holds WHERE hold_id=?", hold_id)
        return Hold(**dict(row)) if row else None

    # ---------- approvals ----------
    def _approval(self, row) -> Approval:
        return Approval(id=row["id"], errand_id=row["errand_id"], hash=row["hash"], code=row["code"],
                        draft=ApprovalDraft.model_validate_json(row["draft_json"]),
                        status=row["status"], expires_at=row["expires_at"])

    def insert_approval(self, a: Approval) -> None:
        self.db.conn.execute("INSERT INTO approvals VALUES (?,?,?,?,?,?,?,?)",
                             (a.id, a.errand_id, a.hash, a.code, a.draft.model_dump_json(), a.status,
                              a.expires_at, self._now()))

    def approvals_for(self, errand_id: str, status: ApprovalStatus | None = None) -> list[Approval]:
        sql, args = "SELECT * FROM approvals WHERE errand_id=?", [errand_id]
        if status:
            sql, args = sql + " AND status=?", args + [status]
        return [self._approval(r) for r in self.db.all(sql + " ORDER BY rowid", *args)]

    def approval_by_hash(self, errand_id: str, hash: str) -> Approval | None:
        row = self.db.one("SELECT * FROM approvals WHERE errand_id=? AND hash=? ORDER BY rowid DESC LIMIT 1",
                          errand_id, hash)
        return self._approval(row) if row else None

    def set_approval_status(self, approval_id: str, status: ApprovalStatus) -> None:
        self.db.conn.execute("UPDATE approvals SET status=? WHERE id=?", (status, approval_id))

    # ---------- executions ----------
    def _execution(self, row) -> Execution:
        return Execution(id=row["id"], errand_id=row["errand_id"], action=row["action"],
                         attempt_no=row["attempt_no"], idem_key=row["idem_key"],
                         payload_hash=row["payload_hash"], approval_hash=row["approval_hash"],
                         parent_execution_id=row["parent_execution_id"],
                         payload=json.loads(row["payload_json"]), state=row["state"],
                         retries=row["retries"], last_error=row["last_error"],
                         result=json.loads(row["result_json"]))

    def insert_execution(self, x: Execution) -> None:
        now = self._now()
        self.db.conn.execute(
            "INSERT INTO executions (id,errand_id,action,attempt_no,idem_key,payload_hash,approval_hash,"
            "parent_execution_id,state,payload_json,retries,last_error,result_json,created_at,updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (x.id, x.errand_id, x.action, x.attempt_no, x.idem_key, x.payload_hash, x.approval_hash,
             x.parent_execution_id, x.state, json.dumps(x.payload), x.retries, x.last_error,
             json.dumps(x.result), now, now))

    def get_execution(self, execution_id: str) -> Execution:
        return self._execution(self.db.one("SELECT * FROM executions WHERE id=?", execution_id))

    def update_execution(self, x: Execution) -> None:
        self.db.conn.execute(
            "UPDATE executions SET state=?, retries=?, last_error=?, result_json=?, updated_at=? WHERE id=?",
            (x.state, x.retries, x.last_error, json.dumps(x.result), self._now(), x.id))

    def executions_for(self, errand_id: str, action: Action | None = None) -> list[Execution]:
        sql, args = "SELECT * FROM executions WHERE errand_id=?", [errand_id]
        if action:
            sql, args = sql + " AND action=?", args + [action]
        return [self._execution(r) for r in self.db.all(sql + " ORDER BY attempt_no", *args)]

    def next_attempt_no(self, errand_id: str, action: Action) -> int:
        row = self.db.one("SELECT MAX(attempt_no) AS n FROM executions WHERE errand_id=? AND action=?",
                          errand_id, action)
        return (row["n"] or 0) + 1

    def open_executions(self) -> list[Execution]:
        return [self._execution(r) for r in self.db.all(
            "SELECT * FROM executions WHERE state IN (?,?) ORDER BY rowid",
            ExecState.EXECUTING, ExecState.VERIFYING)]

    # ---------- notifications ----------
    def add_notification(self, session_id: str, payload: dict[str, Any]) -> None:
        self.db.conn.execute("INSERT INTO notifications VALUES (?,?,?,0,?)",
                             (new_id("n"), session_id, json.dumps(payload), self._now()))

    def pending_notifications(self, session_id: str) -> list[dict[str, Any]]:
        rows = self.db.all("SELECT * FROM notifications WHERE session_id=? AND delivered=0 ORDER BY rowid",
                           session_id)
        return [json.loads(r["payload_json"]) for r in rows]
