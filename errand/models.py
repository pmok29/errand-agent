"""Shared domain models and enums."""
from __future__ import annotations

from enum import StrEnum
from typing import Any, ClassVar

from pydantic import BaseModel, Field


class ErrandState(StrEnum):
    GATHERING = "GATHERING"
    COMPARING = "COMPARING"          # search + compare share one state
    HOLDING = "HOLDING"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    EXECUTING = "EXECUTING"
    VERIFYING = "VERIFYING"
    DONE = "DONE"
    FAILED = "FAILED"
    UNVERIFIED = "UNVERIFIED"
    ABORTED_LIMIT = "ABORTED_LIMIT"


class ErrandKind(StrEnum):
    BOOK = "book"
    RESCHEDULE = "reschedule"
    CANCEL = "cancel"
    REBOOK = "rebook"                # proactive replacement after provider cancel


class Action(StrEnum):
    BOOK = "book"
    CANCEL = "cancel"
    RESCHEDULE = "reschedule"


class ExecState(StrEnum):
    EXECUTING = "EXECUTING"
    VERIFYING = "VERIFYING"
    DONE = "DONE"
    FAILED = "FAILED"
    UNVERIFIED = "UNVERIFIED"


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    CONSUMED = "consumed"
    INVALIDATED = "invalidated"
    EXPIRED = "expired"


class MemorySource(StrEnum):
    EXPLICIT = "explicit"
    INFERRED = "inferred"


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    SUPERSEDED = "superseded"
    FORGOTTEN = "forgotten"


class ToolError(BaseModel):
    code: str
    retryable: bool = False
    message: str = ""


class ToolResult(BaseModel):
    ok: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: ToolError | None = None

    @classmethod
    def success(cls, **data: Any) -> "ToolResult":
        return cls(ok=True, data=data)

    @classmethod
    def fail(cls, code: str, retryable: bool = False, message: str = "") -> "ToolResult":
        return cls(ok=False, error=ToolError(code=code, retryable=retryable, message=message))


class ToolTimeout(Exception):
    """Raised by a mock tool when the call times out (the side effect may still have happened)."""


def split_date_range(value: Any) -> tuple[str, str]:
    """Accepts {"start","end"}, "a..b", or a 2-sequence. Raises ValueError on anything else."""
    if isinstance(value, dict):
        value = (value.get("start"), value.get("end"))
    if isinstance(value, str):
        value = value.split("..")
    try:
        start, end = value
    except (TypeError, ValueError):
        raise ValueError(f"bad date_range {value!r}") from None
    return str(start).strip(), str(end).strip()


class Request(BaseModel):
    service: str | None = None
    area: str | None = None
    date_range: tuple[str, str] | None = None   # ISO dates, inclusive
    time_window: tuple[str, str] | None = None  # "HH:MM", "HH:MM"
    budget_max: int | None = None
    provider_pref: str | None = None
    booking_id: str | None = None               # target of a reschedule / cancel

    REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {
        "book": ("service", "area", "date_range"),
        "rebook": ("service", "area", "date_range"),
        "reschedule": ("booking_id", "date_range"),
        "cancel": ("booking_id",),
    }

    def missing(self, kind: str = "book") -> list[str]:
        return [f for f in self.REQUIRED[kind] if getattr(self, f) is None]


class Errand(BaseModel):
    id: str
    user_id: str
    session_id: str
    kind: ErrandKind
    state: ErrandState
    request: Request = Field(default_factory=Request)
    asked: list[str] = Field(default_factory=list)
    selected: dict[str, Any] = Field(default_factory=dict)
    booking_id: str | None = None
    parent_errand_id: str | None = None
    version: int = 0


class Hold(BaseModel):
    hold_id: str
    errand_id: str
    provider_id: str
    slot: str
    quoted_price: int
    search_price: int | None
    expires_at: str


class ApprovalDraft(BaseModel):
    """Material details of a consequential action. The card and the hash are both derived from this."""
    action: Action
    provider_id: str
    provider_name: str
    slot: str
    price: int | None = None
    booking_id: str | None = None
    old_slot: str | None = None


class Approval(BaseModel):
    id: str
    errand_id: str
    hash: str
    code: str
    draft: ApprovalDraft
    status: ApprovalStatus
    expires_at: str


class Execution(BaseModel):
    id: str
    errand_id: str
    action: Action
    attempt_no: int
    idem_key: str
    payload_hash: str
    approval_hash: str
    parent_execution_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    state: ExecState
    retries: int = 0
    last_error: str | None = None
    result: dict[str, Any] = Field(default_factory=dict)


class Memory(BaseModel):
    id: str
    user_id: str
    key: str
    value: str
    scope: str = "global"            # "global" or "service:<name>"
    source: MemorySource
    status: MemoryStatus
    evidence_msg_id: str | None = None
    superseded_by: str | None = None
