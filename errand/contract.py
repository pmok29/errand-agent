"""The fixed tool contract shared by the agent and the mock provider."""
from __future__ import annotations

# name -> required argument names (get_booking_status takes exactly one of its optional two)
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
