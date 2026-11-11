"""Idempotency keys and payload hashes for side-effect executions."""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def idem_key(errand_id: str, action: str, attempt_no: int) -> str:
    # action is part of the key so a later cancel/reschedule never collides with the booking's key
    return hashlib.sha256(f"{errand_id}|{action}|{attempt_no}".encode()).hexdigest()[:32]


def payload_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(payload).encode()).hexdigest()[:32]
