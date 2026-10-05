"""Small shared receipt accounting; no network calls or ledger/history writes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# Operational estimate snapshot for the project's two configured models.
# Gemini: research/openrouter-runtime-decision-2026-08-13.md; GPT-4.1 mini:
# current BYOK receipts (2026-09-07). Exact receipts always take precedence.
# No fuzzy model-name matching: a new/dated/custom model needs explicit pricing.
MODEL_PRICES_PER_MILLION: dict[str, tuple[float, float]] = {
    "google/gemini-2.5-flash-lite": (0.10, 0.40),
    "openai/gpt-4.1-mini": (0.40, 1.60),
    # Coordinating task's public OpenRouter model-list read, 2026-09-07.
    "openai/gpt-5.6-luna": (0.20, 1.20),
}


def nonnegative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (ValueError, TypeError, OverflowError):
        return 0


def finite_money(value: Any) -> float | None:
    return (
        float(value)
        if isinstance(value, (float, int))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
        else None
    )


@dataclass(frozen=True)
class UsageCost:
    cost_usd: float
    estimated: bool
    unavailable: bool
    billing_basis: str
    router_cost_usd: float | None
    upstream_cost_usd: float | None
    is_byok: bool | None

    def metadata(self) -> dict[str, Any]:
        return {
            "cost_estimated": self.estimated,
            "cost_unavailable": self.unavailable,
            "billing_basis": self.billing_basis,
            "reported_cost_usd": self.router_cost_usd,
            "upstream_inference_cost": self.upstream_cost_usd,
            "is_byok": self.is_byok,
        }


def resolve_usage_cost(
    usage: dict[str, Any] | None,
    *,
    model: str | None = None,
    price_ceiling_per_million: tuple[float, float] | None = None,
    reservation_usd: float | None = None,
) -> UsageCost:
    """Return economic cost, preserving routing versus direct BYOK components.

    `reservation_usd` is usable only with known/explicit pricing. Missing price
    or usage is marked unavailable, never silently presented as a free request.
    Token estimates intentionally ignore cache discounts, so they are upper
    estimates at the supplied pricing snapshot rather than guessed exact bills.
    """
    usage = usage if isinstance(usage, dict) else {}
    router = finite_money(usage.get("cost"))
    details = usage.get("cost_details")
    upstream = finite_money(
        details.get("upstream_inference_cost") if isinstance(details, dict) else None
    )
    byok = usage.get("is_byok") if isinstance(usage.get("is_byok"), bool) else None
    prompt = nonnegative_int(usage.get("prompt_tokens"))
    completion = nonnegative_int(usage.get("completion_tokens"))
    if byok is True and router is not None and upstream is not None:
        return UsageCost(
            router + upstream, False, False, "openrouter_plus_byok_upstream", router, upstream, byok
        )
    # Explicit non-BYOK zero is legitimate. Missing routing flag plus zero
    # with consumed tokens remains ambiguous until reconciled.
    if (
        byok is not True
        and router is not None
        and (router > 0 or byok is False or not (prompt or completion))
    ):
        return UsageCost(router, False, False, "openrouter_reported", router, upstream, byok)
    prices = price_ceiling_per_million or MODEL_PRICES_PER_MILLION.get(model or "")
    known_partial = (router or 0) + ((upstream or 0) if byok is True else 0)
    if prices and len(prices) == 2 and all(finite_money(value) is not None for value in prices):
        estimate = (prompt * prices[0] + completion * prices[1]) / 1_000_000
        if prompt or completion:
            return UsageCost(
                max(estimate, known_partial),
                True,
                False,
                "token_ceiling_estimate",
                router,
                upstream,
                byok,
            )
        reservation = finite_money(reservation_usd)
        if reservation is not None:
            return UsageCost(
                max(reservation, known_partial),
                True,
                False,
                "request_reservation_estimate",
                router,
                upstream,
                byok,
            )
    # Retain any known partial routing charge; unavailable=true means the total
    # may be higher. Do not invent a price for an unknown model.
    return UsageCost(known_partial, False, True, "cost_unavailable", router, upstream, byok)
