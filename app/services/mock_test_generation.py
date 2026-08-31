# ============================================================
# AI mock-test question generation — ported from
# controllers/institute/mock_test_controller.py's _build_prompt /
# _generate_questions_background.
# ============================================================

import json
import re
from typing import Any, Dict, List, Tuple

from bson import ObjectId

from app.services.claude import generate_text

MOCK_TEST_MODEL = "claude-haiku-4-5-20251001"


def build_mock_test_prompt(
    subject_name: str, topic: str, difficulty: str, question_count: int,
    question_types: List[str], marks_per_question: float,
) -> str:
    types_str = ", ".join(question_types)
    topic_line = f"Topic focus: {topic}\n" if topic else ""

    return f"""Generate {question_count} exam-style practice questions for the subject "{subject_name}".
{topic_line}Difficulty: {difficulty}
Allowed question types: {types_str}
Marks per question: {marks_per_question}

Return ONLY a JSON array (no markdown fences, no explanations, no text outside the array). Each \
element must have exactly these fields:
{{
  "type": "<one of: {types_str}>",
  "questionText": "<string>",
  "options": [<4 strings for mcq, ["True","False"] for true_false, [] for other types>],
  "correct_answer": "<string, must match one of the options exactly for mcq/true_false>",
  "marks": {marks_per_question},
  "difficulty": "<easy|medium|hard>",
  "explanation": "<string explaining why the correct answer is correct>"
}}

Generate exactly {question_count} questions."""


def generate_mock_test_questions(prompt: str) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Blocking — run via asyncio.to_thread(). Raises on any failure (invalid JSON,
    wrong shape, or the underlying Claude call failing) — the caller is
    responsible for catching this and setting `generationError` on the mock
    test document, matching Flask's behavior.

    Returns (questions, usage) — the caller is responsible for tracking
    usage (see app.services.ai_usage.record_ai_usage), matching every other
    generate_* helper in this codebase.
    """
    text, usage = generate_text(prompt, model=MOCK_TEST_MODEL, max_tokens=8192)

    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    questions = json.loads(cleaned)
    if not isinstance(questions, list):
        raise ValueError("Expected a JSON array of questions")

    for q in questions:
        q["_id"] = str(ObjectId())

    return questions, usage
