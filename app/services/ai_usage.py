# ============================================================
# Single write path for AI usage tracking across the MySkillGuru surface.
# Dual-writes on every call:
#   1. one row in aiUsageEvents (the full ledger — per-feature dashboards,
#      RAG-savings measurement, cost drill-down)
#   2. an increment on users.token_usage.<provider> (the fast rollup —
#      quota checks read this single document instead of aggregating a
#      growing event collection on every generation request)
#
# Best-effort / non-fatal, matching every other token-tracking helper in
# this codebase (app/utils/token_usage.py, the old
# roadmap_ai.increment_student_*_tokens this replaces) — a tracking failure
# must never break the AI feature that earned the tokens.
#
# Accepts the raw usage object/dict exactly as returned by whichever SDK
# helper produced it (four different shapes are in live use across this
# codebase — see _extract_tokens) so call sites don't need their own
# extraction logic, only to say which provider/model/feature they are.
# ============================================================
import logging
from typing import Any, Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.models.ai_usage_event import create_ai_usage_event_document
from app.utils.ai_pricing import estimate_cost_usd

logger = logging.getLogger("ai_usage")


def _extract_tokens(usage: Any) -> tuple[int, int]:
    """Normalizes the four usage shapes already in use across this codebase:
    - dict from app.services.claude (generate_text/generate_html): input_tokens/output_tokens
    - dict from app.services.gemini (_parse_token_usage): prompt_tokens/candidate_tokens
    - anthropic SDK Usage object (roadmap_ai.py's generate_claude_json/generate_curriculum/generate_claude_text)
    - google-genai SDK usage_metadata object (roadmap_ai.py's generate_gemini_json)
    """
    if usage is None:
        return 0, 0

    if isinstance(usage, dict):
        if "input_tokens" in usage:
            return int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)
        return int(usage.get("prompt_tokens") or 0), int(usage.get("candidate_tokens") or 0)

    input_tokens = getattr(usage, "input_tokens", None)
    if input_tokens is not None:
        return int(input_tokens or 0), int(getattr(usage, "output_tokens", 0) or 0)

    return (
        int(getattr(usage, "prompt_token_count", 0) or 0),
        int(getattr(usage, "candidates_token_count", 0) or 0),
    )


async def record_ai_usage(
    db: AsyncIOMotorDatabase,
    *,
    user_id: str,
    provider: str,
    model: str,
    feature: str,
    usage: Any,
    grounded: Optional[bool] = None,
    context_id: Optional[str] = None,
    job_id: Optional[str] = None,
    institute_id: Optional[str] = None,
    school_id: Optional[str] = None,
    programme_id: Optional[str] = None,
) -> None:
    try:
        input_tokens, output_tokens = _extract_tokens(usage)
        if input_tokens == 0 and output_tokens == 0:
            # No real API spend to record (e.g. a local-extraction path that
            # never called the model) — skip rather than pollute the ledger
            # with zero-cost rows.
            return

        cost_usd = estimate_cost_usd(model, input_tokens, output_tokens)

        event = create_ai_usage_event_document(
            user_id=user_id,
            provider=provider,
            model=model,
            feature=feature,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            grounded=grounded,
            context_id=context_id,
            job_id=job_id,
            institute_id=institute_id,
            school_id=school_id,
            programme_id=programme_id,
        )
        await db["aiUsageEvents"].insert_one(event)

        await db["users"].update_one(
            {"_id": event["user_id"]},
            {"$inc": {
                f"token_usage.{provider}.input_tokens": input_tokens,
                f"token_usage.{provider}.output_tokens": output_tokens,
            }},
        )
    except Exception as e:
        logger.warning(
            "record_ai_usage failed (non-fatal): user_id=%s provider=%s feature=%s error=%s",
            user_id, provider, feature, e,
        )


async def ensure_ai_usage_indexes(db: AsyncIOMotorDatabase) -> None:
    """Idempotent — safe to call on every app startup (see app/main.py's
    lifespan). No index-creation mechanism exists elsewhere in this
    codebase, so this is the only place aiUsageEvents' indexes are defined."""
    await db["aiUsageEvents"].create_index([("user_id", 1), ("created_at", -1)])
    await db["aiUsageEvents"].create_index([("institute_id", 1), ("school_id", 1), ("created_at", -1)])
    await db["aiUsageEvents"].create_index([("feature", 1), ("created_at", -1)])
