import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

# An institute is warned once it has less than this fraction of a pool's
# limit left, so it gets some runway to request a top-up before it's
# actually blocked. Percentage (not a flat token count) so it scales
# sensibly whether the limit is a 5M demo allocation or a much larger paid
# plan.
WARNING_THRESHOLD_RATIO = 0.05


async def increment_institute_gemini_tokens(
    db: AsyncIOMotorDatabase, faculty_id: str, prompt_tokens: int, candidate_tokens: int
) -> None:
    """Increment Gemini token counters on the institute document for a given faculty."""
    faculty = await db["facultyDetails"].find_one({"_id": ObjectId(faculty_id)})
    if not faculty:
        logging.warning("institute not found for faculty %s", faculty_id)
        return

    institute_id = faculty.get("institute_id")
    if not institute_id:
        return
    if isinstance(institute_id, str) and ObjectId.is_valid(institute_id):
        institute_id = ObjectId(institute_id)

    total = prompt_tokens + candidate_tokens
    await db["instituteDetails"].update_one(
        {"_id": institute_id},
        {
            "$inc": {
                "token_usage.gemini.total_prompt_tokens": prompt_tokens,
                "token_usage.gemini.total_candidate_tokens": candidate_tokens,
                "token_usage.gemini.total_tokens": total,
                "token_usage.gemini.call_count": 1,
                "token_usage.grand_total_tokens": total,
            }
        },
    )
    logging.info(
        "[tokens/gemini] institute=%s +%d prompt +%d candidate +%d total",
        institute_id, prompt_tokens, candidate_tokens, total,
    )


async def increment_institute_claude_tokens(
    db: AsyncIOMotorDatabase, faculty_id: str, input_tokens: int, output_tokens: int
) -> None:
    """Increment Claude token counters on the institute document for a given faculty."""
    faculty = await db["facultyDetails"].find_one({"_id": ObjectId(faculty_id)})
    if not faculty:
        logging.warning("institute not found for faculty %s", faculty_id)
        return

    institute_id = faculty.get("institute_id")
    if not institute_id:
        return
    if isinstance(institute_id, str) and ObjectId.is_valid(institute_id):
        institute_id = ObjectId(institute_id)

    total = input_tokens + output_tokens
    await db["instituteDetails"].update_one(
        {"_id": institute_id},
        {
            "$inc": {
                "token_usage.claude.total_input_tokens": input_tokens,
                "token_usage.claude.total_output_tokens": output_tokens,
                "token_usage.claude.total_tokens": total,
                "token_usage.claude.call_count": 1,
                "token_usage.grand_total_tokens": total,
            }
        },
    )
    logging.info(
        "[tokens/claude] institute=%s +%d input +%d output +%d total",
        institute_id, input_tokens, output_tokens, total,
    )


def aggregate_grading_tokens(
    gemini_calls: List[Dict[str, Any]], claude_calls: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Combines the per-call token dicts collected across a grading job (OCR +
    grading + optional transcript) into one nested usage document. Mirrors
    server.py's _aggregate_tokens.
    """
    gemini_prompt = sum(c.get("prompt_tokens", 0) for c in gemini_calls)
    gemini_candidate = sum(c.get("candidate_tokens", 0) for c in gemini_calls)
    gemini_total = sum(c.get("total_tokens", 0) for c in gemini_calls)

    claude_input = sum(c.get("input_tokens", 0) for c in claude_calls)
    claude_output = sum(c.get("output_tokens", 0) for c in claude_calls)
    claude_total = sum(c.get("total_tokens", 0) for c in claude_calls)

    return {
        "gemini": {
            "calls": gemini_calls,
            "total_prompt_tokens": gemini_prompt,
            "total_candidate_tokens": gemini_candidate,
            "total_tokens": gemini_total,
        },
        "claude": {
            "calls": claude_calls,
            "total_input_tokens": claude_input,
            "total_output_tokens": claude_output,
            "total_tokens": claude_total,
        },
        "grand_total_tokens": gemini_total + claude_total,
        "recorded_at": datetime.now(timezone.utc),
    }


async def save_grading_tokens_to_institute(
    db: AsyncIOMotorDatabase, institute_id: Optional[Any], token_usage: Dict[str, Any]
) -> None:
    """
    Best-effort — a failure here must never fail the grading job itself.
    Mirrors server.py's _save_tokens_to_institute (evaluation_count += 1,
    unlike the per-call incrementers above).
    """
    if not institute_id:
        return
    try:
        if isinstance(institute_id, str) and ObjectId.is_valid(institute_id):
            institute_id = ObjectId(institute_id)

        gemini = token_usage.get("gemini", {})
        claude = token_usage.get("claude", {})

        await db["instituteDetails"].update_one(
            {"_id": institute_id},
            {
                "$inc": {
                    "token_usage.gemini.total_prompt_tokens": gemini.get("total_prompt_tokens", 0),
                    "token_usage.gemini.total_candidate_tokens": gemini.get("total_candidate_tokens", 0),
                    "token_usage.gemini.total_tokens": gemini.get("total_tokens", 0),
                    "token_usage.gemini.call_count": len(gemini.get("calls", [])),
                    "token_usage.claude.total_input_tokens": claude.get("total_input_tokens", 0),
                    "token_usage.claude.total_output_tokens": claude.get("total_output_tokens", 0),
                    "token_usage.claude.total_tokens": claude.get("total_tokens", 0),
                    "token_usage.claude.call_count": len(claude.get("calls", [])),
                    "token_usage.grand_total_tokens": token_usage.get("grand_total_tokens", 0),
                    "token_usage.evaluation_count": 1,
                }
            },
        )
    except Exception as e:
        logging.error("save_grading_tokens_to_institute error: %s", e)


async def _resolve_institute_id_for_faculty(db: AsyncIOMotorDatabase, faculty_id: str) -> Optional[ObjectId]:
    faculty = await db["facultyDetails"].find_one({"_id": ObjectId(faculty_id)})
    if not faculty:
        return None
    institute_id = faculty.get("institute_id")
    if not institute_id:
        return None
    if isinstance(institute_id, str) and ObjectId.is_valid(institute_id):
        institute_id = ObjectId(institute_id)
    return institute_id


_PROVIDER_LABELS = {"gemini": "Gemini", "claude": "Claude"}


async def check_institute_token_budget(
    db: AsyncIOMotorDatabase, faculty_id: str, providers: List[str]
) -> Dict[str, Any]:
    """
    Pre-flight check run before kicking off an AI job, so a caller gets an
    immediate error instead of a job that starts and fails partway through.

    An institute with no token_limit document (pre-existing institutes,
    before this feature) is treated as unlimited and always allowed.

    Returns:
      {
        "allowed": bool,
        "message": str | None,       # set when allowed is False
        "warnings": [                # non-blocking, still allowed
          {"provider": "gemini", "remaining": 123456, "limit": 5000000}
        ],
      }
    """
    institute_id = await _resolve_institute_id_for_faculty(db, faculty_id)
    if not institute_id:
        return {"allowed": True, "message": None, "warnings": []}

    institute = await db["instituteDetails"].find_one(
        {"_id": institute_id}, {"token_limit": 1, "token_usage": 1}
    )
    token_limit = (institute or {}).get("token_limit")
    if not token_limit:
        return {"allowed": True, "message": None, "warnings": []}

    token_usage = (institute or {}).get("token_usage") or {}
    warnings: List[Dict[str, Any]] = []

    for provider in providers:
        limit = token_limit.get(provider)
        if not limit:
            continue
        used = token_usage.get(provider, {}).get("total_tokens", 0)
        remaining = limit - used
        if remaining <= 0:
            label = _PROVIDER_LABELS.get(provider, provider)
            return {
                "allowed": False,
                "message": (
                    f"{label} token limit reached for this institute. "
                    "Contact your administrator to increase the limit."
                ),
                "warnings": [],
            }
        if remaining <= limit * WARNING_THRESHOLD_RATIO:
            warnings.append({"provider": provider, "remaining": remaining, "limit": limit})

    return {"allowed": True, "message": None, "warnings": warnings}
