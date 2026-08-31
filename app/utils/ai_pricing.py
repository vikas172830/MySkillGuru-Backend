# ============================================================
# Per-model $/1M-token price table for the aiUsageEvents ledger
# (app/services/ai_usage.py). Internal cost *estimation*, not billing-grade
# precision — providers' actual list prices should be re-checked
# periodically and this table updated; effective_date records when each
# entry was last verified so a stale row is easy to spot.
#
# Unknown models fall back to cost_usd=0.0 (token counts still get recorded)
# rather than raising — a new model showing up in a call site must never
# break the generation call it's costing.
# ============================================================
import logging
from dataclasses import dataclass
from datetime import date

logger = logging.getLogger("ai_pricing")


@dataclass(frozen=True)
class ModelPrice:
    input_per_million: float
    output_per_million: float
    effective_date: date


# List prices in USD, per 1M tokens. Verify against
# https://www.anthropic.com/pricing and https://ai.google.dev/pricing
# before trusting this for anything beyond an internal cost estimate.
_PRICES: dict[str, ModelPrice] = {
    # Claude — Sonnet family
    "claude-sonnet-4-6": ModelPrice(3.00, 15.00, date(2026, 1, 1)),
    "claude-sonnet-4-5": ModelPrice(3.00, 15.00, date(2026, 1, 1)),
    "claude-sonnet-4-20250514": ModelPrice(3.00, 15.00, date(2026, 1, 1)),
    # Claude — Haiku family (Test Engine question generation)
    "claude-haiku-4-5-20251001": ModelPrice(1.00, 5.00, date(2026, 1, 1)),
    # Gemini
    "gemini-2.5-flash": ModelPrice(0.30, 2.50, date(2026, 1, 1)),
    # Embeddings — billed on input tokens only; output side is unused (0).
    "gemini-embedding-001": ModelPrice(0.15, 0.0, date(2026, 1, 1)),
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Best-effort cost estimate for one call. Returns 0.0 for an unrecognized
    model rather than raising — token counts are still worth recording even
    when a $ estimate isn't available yet for a brand-new model string."""
    price = _PRICES.get(model)
    if price is None:
        logger.warning("ai_pricing: no price entry for model=%r — recording cost_usd=0.0", model)
        return 0.0

    cost = (input_tokens / 1_000_000) * price.input_per_million + (output_tokens / 1_000_000) * price.output_per_million
    return round(cost, 6)
