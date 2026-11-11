"""Untrusted tool output: free text is flagged/withheld, and every result is wrapped as data."""
from __future__ import annotations

import json
import re
from typing import Any

FREE_TEXT_KEYS = {"text", "notes", "review", "description", "message"}
MAX_FREE_TEXT = 300
INJECTION = re.compile(
    r"ignore (all |any )?(previous|prior|above) (instructions|messages)|\bsystem( note| prompt)?\s*:|"
    r"\bassistant\s*:|\bcall (book|cancel|reschedule|hold_slot)\b|\b(pre-?approved|already approved)\b|"
    r"\bskip (the )?confirmation\b|\bsave memory\b|\bdeveloper mode\b", re.I)
WITHHELD = "[withheld: this text contained instructions addressed to the assistant]"
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(text: str) -> tuple[str, bool]:
    text = CONTROL.sub("", text)
    if INJECTION.search(text):
        return WITHHELD, True
    return (text[:MAX_FREE_TEXT] + "…") if len(text) > MAX_FREE_TEXT else text, False


def scrub(value: Any, flags: list[str], path: str = "") -> Any:
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if k in FREE_TEXT_KEYS and isinstance(v, str):
                out[k], hit = clean_text(v)
                if hit:
                    flags.append(f"{path}.{k}".lstrip("."))
            else:
                out[k] = scrub(v, flags, f"{path}.{k}")
        return out
    if isinstance(value, list):
        return [scrub(v, flags, f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, str):
        return CONTROL.sub("", value)
    return value


def wrap(tool: str, payload: dict[str, Any]) -> tuple[str, list[str]]:
    """Returns (content for the model, paths of free-text fields withheld as injection attempts)."""
    flags: list[str] = []
    data = scrub(payload, flags)
    if flags:
        data = {"injection_suspected": True, **data}
    body = json.dumps(data, ensure_ascii=False, default=str)
    return f'<tool_output source="{tool}" untrusted="true">{body}</tool_output>', flags
