"""The eight fixed-contract tools: names, argument validation, and dispatch to the mock service."""
from __future__ import annotations

from typing import Any

from errand.mocks.service import MockProviderService
from errand.models import ToolResult

# name -> required argument names (get_booking_status takes exactly one of its two)
CONTRACT: dict[str, tuple[str, ...]] = {
    "search_providers": ("service", "area", "date_range"),
    "get_provider": ("provider_id",),
    "hold_slot": ("provider_id", "slot"),
    "book": ("hold_id", "idempotency_key"),
    "get_booking_status": (),
    "cancel": ("booking_id", "idempotency_key"),
    "reschedule": ("booking_id", "slot", "idempotency_key"),
    "list_bookings": ("user_id",),
}
OPTIONAL: dict[str, tuple[str, ...]] = {"get_booking_status": ("hold_id", "idempotency_key")}
SIDE_EFFECT_TOOLS = frozenset({"book", "cancel", "reschedule"})


def call(service: MockProviderService, name: str, args: dict[str, Any]) -> ToolResult:
    """Validate arguments against the contract and invoke the tool. May raise ToolTimeout."""
    if name not in CONTRACT:
        return ToolResult.fail("unknown_tool", message=name)
    allowed = set(CONTRACT[name]) | set(OPTIONAL.get(name, ()))
    missing = [a for a in CONTRACT[name] if a not in args]
    extra = [a for a in args if a not in allowed]
    if missing or extra:
        return ToolResult.fail("invalid_arguments", message=f"missing={missing} unexpected={extra}")
    return getattr(service, name)(**args)
