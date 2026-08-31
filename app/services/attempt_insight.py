# ============================================================
# "Get Detailed Feedback" — one batched Claude call per attempt (not per
# question) that returns reasoning/feedback/improvement for every question
# in a completed mock-test or roadmap weekly-quiz attempt.
#
# Ported from the Flask prototype's rag/generation/prompts.py::
# attempt_insight_prompt + generator.py::Generator.generate_attempt_insight,
# but — same pattern as app/services/rag/tree_index.py — reuses this
# backend's existing app.services.claude.generate_text instead of porting
# Flask's full Generator/GenerationError orchestration class, since this is
# the only generation call this feature needs.
# ============================================================
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List

from app.models.ai_usage_event import Feature, Provider
from app.services.ai_usage import record_ai_usage
from app.services.claude import generate_text

logger = logging.getLogger(__name__)

ATTEMPT_INSIGHT_SCHEMA = """
Return ONLY a valid JSON object with this exact shape. No markdown, no extra text.
{
  "insights": [
    {
      "reasoning": string,
      "feedback": string,
      "improvement": string
    }
  ]
}
"insights" must have exactly one entry per question below, in the same order.
Every field is REQUIRED for every question, including ones the student answered correctly —
minimum 2-3 complete sentences each, written directly to the student:
- "reasoning": what the question was really testing, and why the student's answer was
  right/wrong/partially right — be specific about which part of their answer mattered.
- "feedback": a direct, encouraging-but-honest assessment of their answer as given.
- "improvement": a concrete, actionable next step to strengthen this specific topic
  (what to review, practice, or think about differently next time).
Avoid generic filler like "needs improvement" or "good job" without specifics.
"""


def attempt_insight_prompt(items: List[Dict[str, Any]]) -> str:
    def _format_item(i: int, item: Dict[str, Any]) -> str:
        lines = [f"QUESTION {i + 1}: {item['question']}"]
        if item.get("options"):
            lines.append(f"OPTIONS: {', '.join(item['options'])}")
        lines.append(f"STUDENT'S ANSWER: {item.get('studentAnswer') or '(no answer given)'}")
        lines.append(f"CORRECT/MODEL ANSWER: {item.get('correctAnswer', '')}")
        lines.append(f"RESULT: {'Correct' if item.get('isCorrect') else 'Incorrect'}")
        return "\n".join(lines)

    items_block = "\n\n".join(_format_item(i, item) for i, item in enumerate(items))
    return (
        "You are a supportive but rigorous tutor giving a student detailed, personalized feedback "
        "on a test/quiz they just completed. For EACH question below, explain your reasoning, give "
        "honest feedback on their specific answer, and suggest a concrete improvement step.\n\n"
        f"{items_block}\n\n{ATTEMPT_INSIGHT_SCHEMA}"
    )


def _strip_code_fence(text: str) -> str:
    return text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()


async def generate_attempt_insight(items: List[Dict[str, Any]], db=None, user_id=None) -> Dict[str, Any]:
    """
    Returns {"insights": [{reasoning, feedback, improvement}, ...]} — one
    entry per item, same order. db/user_id optional: when provided, token
    usage is tracked under the requesting student.
    """
    prompt = attempt_insight_prompt(items)
    # Scales with question count — Practice Tests allow up to 200 questions,
    # and a flat 4000 wasn't enough room for 3 written fields per question
    # once a test had more than ~15-20 of them.
    max_tokens = max(2000, min(16000, len(items) * 500))
    text, token_usage = await asyncio.to_thread(generate_text, prompt, max_tokens=max_tokens)

    if db is not None and user_id is not None:
        await record_ai_usage(
            db, user_id=user_id, provider=Provider.CLAUDE, model="claude-sonnet-4-6",
            feature=Feature.DETAILED_FEEDBACK, usage=token_usage,
        )

    try:
        return json.loads(_strip_code_fence(text))
    except json.JSONDecodeError:
        logger.warning("generate_attempt_insight: failed to parse insight JSON: %r", text[:200])
        return {"insights": []}
