"""JSON schemas (OpenAI function format) for the tools the LLM may see. Filtered per state."""
from __future__ import annotations

from errand.agent.state import TOOLS_BY_STATE
from errand.contract import AREAS, SERVICES
from errand.models import ErrandState

DATE = {"type": "string", "description": "ISO date YYYY-MM-DD"}
TIME = {"type": "string", "description": "24h time HH:MM"}


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required, "additionalProperties": False}}}


SCHEMAS: dict[str, dict] = {
    "search_providers": _fn(
        "search_providers", "Find providers for a service in an area over a date range. Arguments must match "
        "the recorded request.",
        {"service": {"type": "string", "enum": list(SERVICES)}, "area": {"type": "string", "enum": list(AREAS)},
         "date_range": {"type": "array", "items": DATE, "minItems": 2, "maxItems": 2,
                        "description": "[start, end] inclusive"}},
        ["service", "area", "date_range"]),
    "get_provider": _fn("get_provider", "Details and reviews for one provider returned by search_providers.",
                        {"provider_id": {"type": "string"}}, ["provider_id"]),
    "hold_slot": _fn("hold_slot", "Temporarily hold a provider's slot (10 minutes) to get the exact quoted price. "
                     "Holding is not booking.",
                     {"provider_id": {"type": "string"}, "slot": {"type": "string", "description": "YYYY-MM-DDTHH:MM"}},
                     ["provider_id", "slot"]),
    "list_bookings": _fn("list_bookings", "The user's past and upcoming bookings.",
                         {"user_id": {"type": "string"}}, ["user_id"]),
    "update_request": _fn(
        "update_request", "Record details the USER stated (only those). Corrections overwrite just the named "
        "fields. Returns what is still missing.",
        {"kind": {"type": "string", "enum": ["book", "reschedule", "cancel"],
                  "description": "every errand starts as book; set reschedule or cancel first when the user "
                                 "wants to change or cancel an existing booking"},
         "service": {"type": "string", "enum": list(SERVICES)}, "area": {"type": "string", "enum": list(AREAS)},
         "date_from": DATE, "date_to": DATE, "time_from": TIME, "time_to": TIME,
         "budget_max": {"type": "integer", "description": "rupees"},
         "provider_pref": {"type": "string", "description": "provider_id the user asked for"},
         "booking_id": {"type": "string", "description": "existing booking to reschedule or cancel"}},
        []),
    "request_approval": _fn(
        "request_approval", "Ask the user to confirm a booking, cancellation or reschedule. The system shows "
        "the exact details and a confirmation code; you never perform the action yourself.",
        {"action": {"type": "string", "enum": ["book", "cancel", "reschedule"]},
         "slot": {"type": "string", "description": "new slot, for reschedule"},
         "booking_id": {"type": "string", "description": "for cancel / reschedule"}},
        ["action"]),
    "propose_memory": _fn(
        "propose_memory", "Remember a stable preference the USER stated (e.g. preferred area, earliest time, "
        "a technician they like). explicit=true only if the user asked you to remember it or stated it as a "
        "standing preference; otherwise it is an inference the user must confirm later.",
        {"key": {"type": "string", "description": "snake_case, e.g. preferred_area"},
         "value": {"type": "string", "description": "in the user's words"},
         "explicit": {"type": "boolean"},
         "scope": {"type": "string", "description": "global, or service:<service>"}},
        ["key", "value", "explicit"]),
    "memory_op": _fn(
        "memory_op", "Show the user what you remember (list), correct a stored fact (correct), or forget one (forget).",
        {"op": {"type": "string", "enum": ["list", "correct", "forget"]}, "key": {"type": "string"},
         "value": {"type": "string"}},
        ["op"]),
}


def schemas_for(state: ErrandState) -> list[dict]:
    return [SCHEMAS[name] for name in sorted(TOOLS_BY_STATE[state]) if name in SCHEMAS]
