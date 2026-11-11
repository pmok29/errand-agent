"""Estimated LLM cost from the configured list-price table."""
from __future__ import annotations

from errand.config import PRICES_USD_PER_MTOK, USD_TO_INR


def estimate(model: str, input_tokens: int, output_tokens: int) -> dict | None:
    """Returns {usd, inr, estimated: True}, or None when the model has no listed price."""
    price = PRICES_USD_PER_MTOK.get(model)
    if price is None:
        return None
    usd = (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000
    return {"usd": round(usd, 6), "inr": round(usd * USD_TO_INR, 4), "estimated": True}
