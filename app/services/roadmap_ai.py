# ============================================================
# ROADMAP AI HELPERS
# Ported from controllers/self_learner/roadmap_controller.py.
#
# Uses its own lazy Anthropic/Gemini clients — mirroring the Flask
# original's own _get_anthropic()/_get_gemini() — rather than the shared
# app/services/claude.py / gemini.py helpers, because this flow needs
# access to stop_reason / finish_reason for truncation detection and
# Gemini's JSON response_mime_type config, which the simpler shared
# generate_text()/generate_content_from_file() helpers don't expose.
#
# All functions here are blocking SDK calls — run via asyncio.to_thread()
# from the router. Token-usage objects are always returned even when the
# response was truncated (Gemini/Claude bill for it regardless), so the
# router can increment usage before branching on the truncated flag —
# matching Flask's own ordering.
# ============================================================

import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import anthropic
from google import genai as google_genai
from google.genai import types as google_genai_types
# pyrefly: ignore [missing-import]
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.config import settings
from app.models.ai_usage_event import Feature, Provider
from app.schemas.interactive_lesson import InteractiveLesson, lesson_block_adapter
from app.services.ai_usage import record_ai_usage

logger = logging.getLogger(__name__)

_anthropic_client: Optional[anthropic.Anthropic] = None
_gemini_client: Optional[google_genai.Client] = None


def _get_anthropic() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    return _anthropic_client


def _get_gemini() -> google_genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = google_genai.Client(api_key=settings.GEMINI_API_KEY)
    return _gemini_client


def extract_json(text: str) -> Any:
    """Robustly extract the first JSON object/array from a Claude/Gemini response.
    Handles nested markdown code blocks (e.g. in programming questions) and ignores
    brackets inside string literals by leveraging raw_decode.
    """
    # 1. Try loading directly if it already starts and ends with brackets/braces
    text_clean = text.strip()
    if (text_clean.startswith("{") and text_clean.endswith("}")) or (text_clean.startswith("[") and text_clean.endswith("]")):
        try:
            return json.loads(text_clean)
        except json.JSONDecodeError:
            pass

    # 2. Find the earliest occurrence of '{' or '['
    indices = [text.find("{"), text.find("[")]
    valid_indices = [idx for idx in indices if idx != -1]
    if not valid_indices:
        raise ValueError("No valid JSON found in model output")

    # Start looking from the earliest character
    curr_idx = min(valid_indices)
    while curr_idx != -1:
        try:
            # raw_decode parses exactly one JSON entity starting at curr_idx
            obj, _ = json.JSONDecoder().raw_decode(text[curr_idx:])
            return obj
        except json.JSONDecodeError:
            # Fallback: search for the next '{' or '[' in the remaining text
            next_brace = text.find("{", curr_idx + 1)
            next_bracket = text.find("[", curr_idx + 1)
            next_indices = [idx for idx in (next_brace, next_bracket) if idx != -1]
            if not next_indices:
                break
            curr_idx = min(next_indices)

    raise ValueError("No valid JSON found in model output")


# ============================================================
# Token tracking for every self-learner/MySkillGuru AI call now lives in
# app.services.ai_usage.record_ai_usage (dual-writes the aiUsageEvents
# ledger + the users.token_usage rollup this module used to maintain on its
# own) — distinct from the shared, institute-scoped helpers in
# app/utils/token_usage.py, which use a different document shape entirely.
# ============================================================
# CURRICULUM GENERATION (Claude — create_roadmap background job)
# ============================================================

def build_curriculum_prompt(
    subject, goal, skill_level, daily_study_time, revision_frequency, assessment_score,
    grounding_context: Optional[str] = None,
    custom_instruction: Optional[str] = None,
) -> str:
    score_context = (
        f"The student scored {assessment_score}% on the pre-assessment quiz, "
        f"so they already have some familiarity with the subject. "
        f"Adjust difficulty accordingly — avoid trivially basic content and start from where the assessment indicates."
        if assessment_score is not None
        else ""
    )
    # Optional RAG grounding: real content retrieved from a course material
    # the institute uploaded for this subject (see app/services/rag/). When
    # present, the curriculum should follow this document's actual structure
    # and weighting instead of the model's generic knowledge of the subject.
    grounding_block = (
        f"""
## Course Material (use this as the authoritative source for structure, topics, and emphasis)
The following was retrieved from a real course document uploaded for this subject. Ground the
roadmap's weeks/subtopics in what is actually here — do not invent topics that
contradict it, and prioritize what it emphasizes.

{grounding_context}
"""
        if grounding_context
        else ""
    )
    # Independent of grounding: applies whether or not source material was
    # found, so "no doc + custom instruction" still reaches this block.
    instruction_block = (
        f"""
## Additional Student Instructions (follow these closely when shaping weeks, topic emphasis, and pacing)
{custom_instruction}
"""
        if custom_instruction
        else ""
    )
    return f"""You are an expert curriculum designer and senior educator.
Your task is to create a **highly detailed, production-quality self-learning roadmap**, broken into
weekly units, for the following student profile.

## Student Profile
- Subject: {subject}
- Goal: {goal}
- Skill Level: {skill_level}
- Daily Study Time: {daily_study_time}
- Revision Frequency: {revision_frequency}
{score_context}
{grounding_block}
{instruction_block}

## Roadmap Requirements
Decide how many weeks this roadmap needs on your own — base it on the subject's real scope, the
student's goal, and how much they can realistically do at {daily_study_time}/day. Use as many weeks
as the subject genuinely requires to go from {skill_level} to the stated goal; do not pad or compress
artificially. As a sanity range, most roadmaps land between **4 and 16 weeks**, but go outside that
range if the subject truly calls for it.

Each week must have exactly **one main topic** plus **3 to 5 subtopics** under it. Every subtopic
title must be specific, actionable, and unique — never a generic name like "Introduction" alone.
Weeks must be in a logical learning order (foundations first, building toward the stated goal).

### Curriculum Stats (generate realistic estimates)
- estimatedWeeks: the week count you chose (should equal len(weeks))
- totalTopics: total subtopic count across all weeks
- difficultyScore: integer 1–10 for the roadmap as a whole

### Practice Questions (per week)
Generate **5 MCQ practice questions** per week that target conceptual understanding of that week's
main topic. Each question: {{ "question": "...", "options": ["A", "B", "C", "D"], "answer": <0-indexed int>, "explanation": "..." }}

## Output Format
Return ONLY a valid JSON object matching this exact schema. No prose, no markdown, only JSON.

{{
  "subject_display_name": "Full display name of the subject",
  "stats": {{
    "estimatedWeeks": <week count>,
    "totalTopics": <total subtopic count>,
    "difficultyScore": <1-10>
  }},
  "weeks": [
    {{
      "week": 1,
      "title": "<This week's main topic — tailored to {subject}>",
      "introDescription": "<2-3 sentence intro: what this week covers and why it matters now>",
      "subtopics": [
        {{
          "title": "<Specific subtopic — must be unique and actionable>",
          "summary": "<2-3 sentence overview of what the student will learn in this subtopic>",
          "keyPoints": ["<point 1>", "<point 2>", "<point 3>"],
          "difficulty": "Beginner | Intermediate | Advanced"
        }}
      ],
      "practiceQuestions": [
        {{
          "question": "<Question text>",
          "options": ["<A>", "<B>", "<C>", "<D>"],
          "answer": <0-indexed correct option int>,
          "explanation": "<Brief explanation of why the answer is correct>"
        }}
      ]
    }},
    {{
      "week": 2,
      "title": "...",
      "introDescription": "...",
      "subtopics": [ ... ],
      "practiceQuestions": [ ... ]
    }}
  ]
}}

Include one object per week in the "weeks" array — as many as the week count you chose."""


def generate_claude_json(prompt: str, max_tokens: int = 4000) -> Tuple[Optional[Any], Any, bool]:
    """Returns (parsed_json_or_None, usage, truncated). May raise anthropic.APIError."""
    client = _get_anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-5", max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}]
    )
    if message.stop_reason == "max_tokens":
        return None, message.usage, True
    return extract_json(message.content[0].text), message.usage, False


def generate_curriculum(prompt: str) -> Tuple[Optional[Dict[str, Any]], Any, bool]:
    """Returns (curriculum_dict_or_None, usage, truncated). May raise anthropic.APIError."""
    client = _get_anthropic()
    start_time = time.time()
    message = client.messages.create(
        # AI-sized week count (up to ~16 weeks x 3-5 subtopics x 5 practice
        # questions each) can produce more output than the old fixed
        # 4-level curriculum — bumped from 25000 to give it room.
        model="claude-sonnet-4-5", max_tokens=32000, messages=[{"role": "user", "content": prompt}]
    )
    logger.info("Claude curriculum call took %.2fs", time.time() - start_time)

    if message.stop_reason == "max_tokens":
        return None, message.usage, True

    return extract_json(message.content[0].text), message.usage, False


# ============================================================
# GEMINI JSON GENERATION (subtopic notes / stage quiz / pre-assessment)
# ============================================================

def is_gemini_quota_error(exc: Exception) -> bool:
    """
    True only for a Gemini quota/rate-limit failure (HTTP 429 /
    RESOURCE_EXHAUSTED) — the one Gemini failure mode worth failing over to
    Claude for. Other Gemini errors (malformed prompt, transient 5xx,
    network blip) aren't retried on a different provider; they surface as a
    normal error instead, same as before.
    """
    code = getattr(exc, "code", None)
    status = (getattr(exc, "status", "") or "").upper()
    return code == 429 or "RESOURCE_EXHAUSTED" in status or "QUOTA" in status


def is_claude_overloaded_error(exc: Exception) -> bool:
    """
    True only for a Claude rate-limit/overload failure (HTTP 429 RateLimitError,
    or 529 "overloaded_error") — the one Claude failure mode worth failing over
    to Gemini for, mirroring is_gemini_quota_error's treatment of the reverse
    direction. Other Claude errors (malformed prompt, auth, transient 5xx)
    aren't retried on a different provider; they surface as a normal error
    instead.
    """
    status_code = getattr(exc, "status_code", None)
    return status_code in (429, 529)


def generate_gemini_json(prompt: str) -> Tuple[Optional[Any], Any, bool]:
    """Returns (parsed_json_or_None, usage_metadata, truncated)."""
    client = _get_gemini()
    start_time = time.time()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=google_genai_types.GenerateContentConfig(response_mime_type="application/json"),
    )
    logger.info("Gemini call took %.2fs", time.time() - start_time)

    usage = response.usage_metadata
    truncated = response.candidates[0].finish_reason.name == "MAX_TOKENS"
    if truncated:
        return None, usage, True

    return extract_json(response.text), usage, False


# ============================================================
# PROMPT BUILDERS (notes / stage quiz / pre-assessment)
# ============================================================

# ============================================================
# VARK BLEND + DIFFICULTY HELPERS
# ============================================================

VARK_STYLES = ("visual", "auditory", "reading", "kinesthetic")
DIFFICULTIES = ("Easy", "Moderate", "Difficult")

_VARK_STYLE_GUIDANCE = {
    "Visual": (
        "This student learns best visually. Describe structure, relationships, and flow "
        "explicitly in words (e.g. \"picture this as three connected stages...\"), use "
        "spatial/structural language throughout, and organize the explanation so a reader "
        "could sketch it as a diagram from your words alone."
    ),
    "Auditory": (
        "This student learns best by hearing/discussing ideas. Write in a conversational, "
        "narrated tone — as if explaining out loud to the student — with rhetorical "
        "questions, verbal analogies, and a natural spoken rhythm."
    ),
    "Reading": (
        "This student learns best through reading dense text. Write a thorough, "
        "well-organized reference-style explanation with precise terminology and complete "
        "sentences — the kind of notes someone reads closely and annotates, not skims."
    ),
    "Kinesthetic": (
        "This student learns best by doing. Emphasize concrete, actionable steps and real "
        "hands-on practice — frame explanations around \"try this\" / \"do this\" rather "
        "than passive description, and connect every concept to something the student can "
        "physically or practically perform."
    ),
}

_DIFFICULTY_GUIDANCE = {
    "Easy": "Keep language simple, add extra scaffolding and analogies, avoid unexplained jargon, assume minimal prior context.",
    "Moderate": "Balance clarity with technical depth appropriate for someone actively learning this topic for the first time.",
    "Difficult": "Assume strong prior context, use precise technical language, and go into advanced nuance and edge cases.",
}


def _normalize_vark(visual: Optional[int], auditory: Optional[int], reading: Optional[int], kinesthetic: Optional[int]) -> Dict[str, int]:
    """Clamp each style to >= 0 (missing/negative -> 0). Does not require the
    blend to sum to 100 — the frontend picker enforces that; the backend just
    needs relative weights to find the dominant style."""
    raw = {"visual": visual, "auditory": auditory, "reading": reading, "kinesthetic": kinesthetic}
    return {k: max(0, v) if v is not None else 25 for k, v in raw.items()}


def _dominant_vark_style(vark: Dict[str, int]) -> str:
    """Highest % wins; ties broken by VARK_STYLES order (stable — max() keeps the first max seen)."""
    return max(VARK_STYLES, key=lambda k: vark.get(k, 0))


def _normalize_difficulty(difficulty: Optional[str]) -> str:
    d = (difficulty or "Moderate").strip().title()
    return d if d in DIFFICULTIES else "Moderate"


# ============================================================
# MERMAID CONCEPT DIAGRAM — generation rules + structural validation/repair
#
# The generation prompt only constrains the model loosely (models drift from
# instructions), so every diagram is structurally re-validated server-side
# BEFORE it's ever cached or served — not just prompt-tightened and trusted.
# This is a regex-based structural check against the strict subset defined
# by MERMAID_SYNTAX_RULES below, not a real Mermaid grammar parser (none
# available server-side) — good enough to catch the common failure mode
# (unquoted node labels, wrong diagram type, multi-statement lines) without
# needing a JS parser in the request path. Extracted to one shared constant
# so the generation prompt and the repair prompt can't drift out of sync.
# ============================================================

MERMAID_SYNTAX_RULES = """STRICT Mermaid diagram syntax rules (violating any of these breaks rendering):
1. First line must be exactly "graph TD" or "graph LR" — no other diagram types (no mindmap, flowchart, sequenceDiagram, etc.).
2. Every other line is a node definition and/or a chain of nodes connected by edges.
3. EVERY node must be written as NodeId["Label text here"] — the label MUST be wrapped in double quotes inside square brackets. Never use round brackets like NodeId(Label) or unquoted square brackets like NodeId[Label].
4. Node IDs contain only letters, numbers, and underscores — no spaces or punctuation.
5. Edges use only: --> or --- or -.-> or ==>. An edge may have a quoted label: A -->|"Yes"| B.
6. One statement per line. No semicolons, no subgraphs, no styling/class directives.
7. Labels may contain any characters except double quotes (parentheses, commas, colons, etc. are fine as long as the whole label stays inside the quotes).

Example of VALID syntax:
graph TD
    A["Start: Define the Problem"] --> B["Gather Requirements"]
    B --> C["Design Solution"]
    C -->|"Approved"| D["Implement"]
    C -->|"Rejected"| B"""

_MERMAID_HEADER_RE = re.compile(r"^graph\s+(TD|LR)\s*$")
_MERMAID_NODE_DEF = r'\["[^"]*"\]'
_MERMAID_LINE_RE = re.compile(
    rf'^[A-Za-z0-9_]+({_MERMAID_NODE_DEF})?'
    rf'(\s*(-->|---|-\.->|==>)\s*(\|"[^"]*"\|\s*)?[A-Za-z0-9_]+({_MERMAID_NODE_DEF})?)*$'
)


def _is_valid_mermaid_diagram(diagram: Any) -> bool:
    if not diagram or not isinstance(diagram, str):
        return False
    lines = [ln.strip() for ln in diagram.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    if not _MERMAID_HEADER_RE.match(lines[0]):
        return False
    return all(_MERMAID_LINE_RE.match(ln) for ln in lines[1:])


def build_mermaid_repair_prompt(broken_diagram: str) -> str:
    return f"""The following Mermaid diagram has invalid syntax and failed to render. Fix it so it
strictly follows these rules, keeping the same content and meaning:

{MERMAID_SYNTAX_RULES}

BROKEN DIAGRAM:
{broken_diagram}

Return ONLY the corrected Mermaid diagram source — no markdown fences, no explanation, no JSON.
Start directly with "graph TD" or "graph LR"."""


def generate_claude_text(prompt: str, max_tokens: int = 1500) -> Tuple[str, Any]:
    """Returns (text, usage). May raise anthropic.APIError. Blocking — run via asyncio.to_thread()."""
    client = _get_anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-5", max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text.strip(), message.usage


async def validate_and_repair_diagram(
    diagram: Any, db: AsyncIOMotorDatabase, user_id: str,
) -> str:
    """Valid diagrams (or an empty/absent one) pass through untouched. Invalid
    ones get ONE cheap, targeted repair call. If the repair also fails
    validation, the field is dropped to "" rather than ever caching/serving
    broken syntax — the frontend already renders nothing for an empty
    diagram."""
    if not diagram:
        return ""
    if _is_valid_mermaid_diagram(diagram):
        return diagram

    try:
        prompt = build_mermaid_repair_prompt(diagram)
        repaired, usage = await asyncio.to_thread(generate_claude_text, prompt, 1000)
        await record_ai_usage(
            db, user_id=user_id, provider=Provider.CLAUDE, model="claude-sonnet-4-5",
            feature=Feature.ROADMAP_DIAGRAM_REPAIR, usage=usage,
        )
    except Exception as e:
        logger.warning("mermaid diagram repair call failed (dropping diagram): %s", e)
        return ""

    return repaired if _is_valid_mermaid_diagram(repaired) else ""


def log_style_requirement_gaps(notes: Dict[str, Any], dominant_style: str, week: int, subtopic_idx: int) -> None:
    """Purely for server-log visibility into how often generation under-delivers
    against its own schema's stated per-style requirements. Logs only — never
    modifies or retries (semantic/pedagogical quality of the content isn't
    something this can verify, only whether the required fields are present
    and non-trivial)."""
    if dominant_style == "Visual" and not notes.get("conceptDiagram"):
        logger.warning("VARK gap: Visual-dominant notes (week=%s subtopic=%s) returned with no conceptDiagram", week, subtopic_idx)

    if dominant_style == "Kinesthetic":
        task = notes.get("handsOnTask") or {}
        steps = task.get("steps") or []
        if len(steps) < 3 or not task.get("expectedOutcome"):
            logger.warning(
                "VARK gap: Kinesthetic-dominant notes (week=%s subtopic=%s) handsOnTask under-delivered (%d steps, has_outcome=%s)",
                week, subtopic_idx, len(steps), bool(task.get("expectedOutcome")),
            )


def _repair_grid_visual_aid(raw_lesson: Dict[str, Any]) -> None:
    """Best-effort in-place repair for the most common visualAid failure: a
    grid/table row whose `values` length doesn't match `columnHeaders`. Pads
    short rows with "" and truncates long ones, rather than losing the
    entire (mandatory, non-block) visualAid — and therefore the whole
    interactive lesson — over a cosmetic row/column mismatch."""
    visual_aid = raw_lesson.get("visualAid")
    if not isinstance(visual_aid, dict) or visual_aid.get("kind") not in ("grid", "table"):
        return
    headers = visual_aid.get("columnHeaders")
    rows = visual_aid.get("rows")
    if not isinstance(headers, list) or not isinstance(rows, list):
        return
    column_count = len(headers)
    for row in rows:
        if not isinstance(row, dict):
            continue
        values = row.get("values")
        if not isinstance(values, list) or len(values) == column_count:
            continue
        row["values"] = values[:column_count] + [""] * (column_count - len(values))


def validate_interactive_lesson(notes: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize the AI lesson DSL without rejecting useful notes.

    A completely valid lesson is retained. If only some generated blocks are
    malformed, valid blocks are salvaged. If none validate, the interactive
    section is omitted and the existing notes UI remains the safe fallback.
    """
    if not isinstance(notes, dict):
        return notes

    sanitized = dict(notes)
    raw_lesson = sanitized.get("interactiveLesson")
    if not isinstance(raw_lesson, dict):
        sanitized.pop("interactiveLesson", None)
        return sanitized

    _repair_grid_visual_aid(raw_lesson)

    try:
        lesson = InteractiveLesson.model_validate(raw_lesson)
        sanitized["interactiveLesson"] = lesson.model_dump()
        sanitized["notesSchemaVersion"] = 4
        return sanitized
    except Exception as exc:
        logger.warning("interactive lesson validation failed; attempting block salvage: %s", exc)

    valid_blocks = []
    for raw_block in raw_lesson.get("blocks", []):
        try:
            valid_blocks.append(lesson_block_adapter.validate_python(raw_block).model_dump())
        except Exception as exc:
            block_type = raw_block.get("type", "unknown") if isinstance(raw_block, dict) else "unknown"
            logger.warning("dropping invalid interactive block type=%s: %s", block_type, exc)

    # Preserve only pedagogically complete lessons. Salvaging individual
    # blocks is useful, but missing mission/context/mastery metadata should
    # fall back to the existing notes rather than produce another confusing
    # collection of disconnected cards.
    if valid_blocks:
        candidate = dict(raw_lesson)
        candidate["blocks"] = valid_blocks[:12]
        try:
            lesson = InteractiveLesson.model_validate(candidate)
            sanitized["interactiveLesson"] = lesson.model_dump()
            sanitized["notesSchemaVersion"] = 4
            return sanitized
        except Exception as exc:
            logger.warning("salvaged lesson still failed pedagogical validation: %s", exc)

    sanitized.pop("interactiveLesson", None)
    return sanitized


_DOMAIN_GUIDANCE = {
    "technology": "Use architecture flows, code walkthroughs, dry runs, debugging, and system scenarios only when they genuinely help.",
    "business": "Use customer/business scenarios, case studies, decisions, comparisons, and meaningful metrics. Do not add code unless the topic requires it.",
    "quantitative": "Use formula derivation, worked calculations, visual state changes, estimation, and practice problems.",
    "science": "Use observable processes, cause and effect, classification, experiments, and evidence-based conclusions.",
    "humanities": "Use timelines, competing perspectives, source/context analysis, arguments, and real-world interpretation.",
    "general": "Use concrete scenarios, comparisons, guided decisions, reflection, and practical application.",
}


def infer_lesson_domain(subject: str, sub_title: str = "") -> str:
    """Small deterministic hint for the generator; the lesson remains generic.

    This is intentionally broad rather than a course catalogue. It selects a
    teaching strategy, not a hard-coded subject component.
    """
    text = f"{subject} {sub_title}".lower()
    keyword_groups = (
        ("technology", ("computer", "software", "programming", "python", "java", "api", "fastapi", "data structure", "algorithm", "database", "cloud", "cyber", "machine learning", "nlp", "neural")),
        ("business", ("marketing", "management", "business", "sales", "branding", "customer", "commerce", "entrepreneur", "human resource", "operations")),
        ("quantitative", ("mathematics", "math", "statistics", "calculus", "algebra", "accounting", "finance", "economics", "probability")),
        ("science", ("biology", "chemistry", "physics", "medicine", "environment", "anatomy", "ecology")),
        ("humanities", ("history", "geography", "literature", "language", "political", "sociology", "psychology", "philosophy", "law")),
    )
    for domain, keywords in keyword_groups:
        if any(keyword in text for keyword in keywords):
            return domain
    return "general"


_GUIDED_LESSON_PROMPT = """  "interactiveLesson": {
    "schemaVersion": 4,
    "language": "English",
    "domain": "__DOMAIN__",
    "title": "<clear lesson title>",
    "mission": {
      "goal": "<what the learner will accomplish>",
      "whyItMatters": "<practical relevance>",
      "estimatedMinutes": 25,
      "successCriteria": ["<observable criterion 1>", "<criterion 2>"]
    },
    "prerequisites": ["<what to know first, or 'No prior knowledge required'>"],
    "learningOutcomes": ["<I can... outcome 1>", "<outcome 2>"],
    "keyTerms": [
      {"term":"<term>","meaning":"<plain-English meaning>","example":"<short example>"},
      {"term":"<term 2>","meaning":"<meaning>","example":"<example>"}
    ],
    "anchorExample": {
      "title": "<meaningful example name>",
      "context": "<what the example represents>",
      "whyChosen": "<why this example helps this learner understand the concept>"
    },
    "visualAid": {
      "kind": "grid",
      "title": "<what the learner is looking at>",
      "purpose": "<how this visual makes the core relationship concrete>",
      "columnHeaders": ["<column 1>", "<column 2>"],
      "rows": [
        {"label":"<row 1>","values":["<value>","<value>"]},
        {"label":"<row 2>","values":["<value>","<value>"]}
      ],
      "interactionPrompt": "<what the learner should click and notice>",
      "caption": "<the important pattern revealed by the visual>"
    },
    "blocks": [],
    "summary": {
      "keyTakeaways": ["<takeaway 1>", "<takeaway 2>", "<takeaway 3>"],
      "masteryChecklist": ["<I can... check 1>", "<check 2>"],
      "nextStep": "<what to study or practise next>"
    }
  }
}

Populate blocks with exactly 5 items in teaching order. Use ONLY these exact block schemas:
- {"type":"concept","title":"...","simpleExplanation":"...","whyItMatters":"...","realWorldConnection":"..."}
- {"type":"mental_model","title":"...","analogy":"...","explanation":"...","remember":"..."}
- {"type":"worked_example","title":"...","scenario":"...","exampleReason":"...","steps":[{"label":"...","action":"...","explanation":"...","result":"..."}],"outcome":"..."}
- {"type":"guided_walkthrough","title":"...","purpose":"...","example":"...","exampleReason":"...","steps":[{"focus":"...","action":"...","state":"...","why":"..."}],"conclusion":"..."}
- {"type":"formula_walkthrough","title":"...","formula":"...","explanation":"...","steps":[{"label":"...","expression":"...","result":"...","explanation":"..."}]}
- {"type":"comparison","title":"...","columns":[{"heading":"...","points":["..."]},{"heading":"...","points":["..."]}],"conclusion":"..."}
- {"type":"code_walkthrough","title":"...","language":"...","code":"...","purpose":"...","steps":[{"line":1,"focus":"...","state":"...","explanation":"..."}],"outcome":"..."}
- {"type":"common_mistakes","title":"...","items":[{"mistake":"...","whyItHappens":"...","correction":"..."}]}
- {"type":"practical_activity","title":"...","instructions":"...","steps":["...","..."],"expectedOutcome":"...","reflectionQuestion":"..."}
- {"type":"case_study","title":"...","scenario":"...","facts":["...","..."],"decision":"...","recommendedApproach":"...","reasoning":"..."}
- {"type":"debugging_lab","title":"...","scenario":"...","brokenExample":"...","hints":["..."],"solution":"...","explanation":"..."}
- {"type":"parameter_explorer","title":"...","prompt":"...","parameterLabel":"...","options":[{"value":"...","label":"...","effect":"..."}]}
- {"type":"prediction","title":"...","question":"...","options":["...","..."],"correctAnswerIndex":0,"explanation":"..."}
- {"type":"quick_check","title":"...","question":"...","options":["...","..."],"correctAnswerIndex":0,"explanation":"..."}

GUIDED-LESSON RULES:
- Keep the entire response below 4,000 words. Each prose field must be at most 70 words, each list at most 5 items, and each walkthrough/worked example at most 5 steps.
- Do not repeat content across summary, detailedExplanation, and interactiveLesson. Put teaching depth in interactiveLesson and keep legacy reference fields brief.
- Write every field in clear English. Never use Hinglish or unexplained jargon.
- Assume no prior knowledge at Beginner difficulty. Define a technical term before any block uses it.
- Build one coherent narrative around anchorExample. Never introduce a random input, person, string, number, or scenario without explaining what it represents and why it was selected.
- visualAid is required and must make the lesson's core relationship visible with meaningful, domain-specific data. It is a learning model, not decoration.
- Select exactly one approved visual kind: grid, table, sequence, flow, timeline, or hierarchy. Use grid for arrays/matrices/spatial coordinates; table for comparisons or metrics; sequence for ordered transformations; flow for processes/funnels/system movement; timeline for dated change; hierarchy for levels, categories, or parent-child relationships.
- For grid/table, include 1-6 columnHeaders and 1-6 rows; every row's values array must exactly match the number of headers. For sequence/flow/timeline/hierarchy, replace columnHeaders and rows with "items":[{"label":"...","value":"...","description":"...","level":0}] and include 2-10 ordered items. A hierarchy must start at level 0 and may increase only one level at a time.
- Keep visual labels and values short enough to scan. interactionPrompt must tell the learner what to click or compare; caption must explain the pattern they should discover.
- Use this five-stage sequence: (1) concept or mental_model, (2) worked_example or guided_walkthrough, (3) one domain-appropriate application, (4) common_mistakes or comparison, (5) prediction or quick_check.
- A walkthrough step must state the current focus, what happens, the resulting state when applicable, and why it happens. Its conclusion must show the final result explicitly.
- Use code_walkthrough/debugging_lab only when code is inherently relevant. Use case_study for business decisions, formula_walkthrough for calculations, and practical_activity for hands-on application.
- Include common_mistakes where useful. Keep each block focused enough for one screen.
- Do not output HTML, JavaScript, generated UI code, unsupported block types, or executable expressions. The React client owns all behavior.
- Domain teaching guidance: __DOMAIN_GUIDANCE__"""


def build_guided_lesson_schema_prompt(domain: str) -> str:
    return (
        _GUIDED_LESSON_PROMPT
        .replace("__DOMAIN__", domain)
        .replace("__DOMAIN_GUIDANCE__", _DOMAIN_GUIDANCE[domain])
    )


def build_notes_prompt(
    subject: str, week_title: str, sub_title: str, sub_summary: str, key_points: List[str],
    dominant_style: str, difficulty: str,
    grounding_context: Optional[str] = None,
    goal: Optional[str] = None,
) -> str:
    lesson_domain = infer_lesson_domain(subject, sub_title)
    guided_lesson_prompt = build_guided_lesson_schema_prompt(lesson_domain)
    grounding_block = (
        f"""
## Course Material (ground the notes in this real content where relevant)
{grounding_context}
"""
        if grounding_context
        else ""
    )
    style_note = _VARK_STYLE_GUIDANCE.get(dominant_style, _VARK_STYLE_GUIDANCE["Reading"])
    difficulty_note = _DIFFICULTY_GUIDANCE.get(difficulty, _DIFFICULTY_GUIDANCE["Moderate"])
    # Interview tips only make sense for an interview-prep goal — an Exam
    # Preparation / Academic Learning student (who may well be a school-age
    # class 10/12 student) has no use for "high-frequency interview
    # question" content, so the field is omitted from the schema entirely
    # rather than generated and then hidden client-side.
    include_interview_tips = _is_interview_goal(goal)
    interview_tips_field = (
        """  "interviewTips": [
    "<High-frequency interview question or tip related to this subtopic>",
    "<tip 2>"
  ],
"""
        if include_interview_tips
        else ""
    )
    return f"""You are an expert educator writing a coherent, premium self-study lesson personalized
to this student's learning style, level, goal, and course domain.

## Context
Subject: {subject}
Week: {week_title}
Subtopic: {sub_title}
Overview: {sub_summary}
Key Points to Cover: {json.dumps(key_points)}
Student Goal: {goal or "General learning and practical understanding"}
Suggested Domain Family: {lesson_domain}

## Personalization (apply throughout every section below, not just the summary)
Learning style — {dominant_style}: {style_note}
Difficulty — {difficulty}: {difficulty_note}
{grounding_block}

## Task
Write a **complete but concise, student-friendly guided lesson** for the subtopic "{sub_title}".
The interactiveLesson is the primary teaching experience. Legacy fields before it are compact reference
material only; do not repeat the same explanation in both places.

Return ONLY a valid JSON object (no prose, no markdown wrapper) with this exact schema:

{{
  "summary": "<2-3 sentence engaging overview of what this subtopic covers and why it matters>",
  "detailedExplanation": [
    {{ "heading": "<What it is>", "content": "<one concise paragraph, maximum 100 words>" }},
    {{ "heading": "<How and when it is used>", "content": "<one concise paragraph, maximum 100 words>" }}
  ],
  "keyPoints": [
    "<Concise, memorable bullet — start with a verb>",
    "<point 2>",
    "<point 3>",
    "<point 4>"
  ],
  "formulasOrRules": [
    {{
      "name": "<Formula/Rule name>",
      "formula": "<The actual formula, rule, or pattern>",
      "explanation": "<When and how to use it>"
    }}
  ],
  "codeExample": {{
      "language": "<programming language or 'N/A'>",
      "code": "<relevant code snippet of at most 25 lines, or 'N/A' when code is not inherent to this topic>",
      "explanation": "<Line-by-line walkthrough, or why code is not applicable>"
  }},
  "commonMistakes": [
    "<Mistake students commonly make and how to avoid it>",
    "<mistake 2>"
  ],
{interview_tips_field}  "revisionChecklist": [
    "<I can explain ... >",
    "<I can apply ... >",
    "<I can evaluate or identify ... >"
  ],
  "conceptDiagram": {(
      '"<A Mermaid graph TD/LR diagram source string illustrating this subtopic'
      ' — REQUIRED since Visual is the dominant style. Follow the STRICT syntax'
      ' rules below exactly.>"'
  ) if dominant_style == "Visual" else '""'},
  "handsOnTask": {(
      '{ "title": "<short hands-on task title>", '
      '"steps": ["<concrete step 1>", "<step 2>", "<step 3>", "... at least 3 steps>"], '
      '"expectedOutcome": "<what the student should observe or produce when done>" }'
  ) if dominant_style == "Kinesthetic" else '{ "title": "", "steps": [], "expectedOutcome": "" }'},
{guided_lesson_prompt}

{MERMAID_SYNTAX_RULES if dominant_style == "Visual" else ""}"""


# ============================================================
# AUTO TEST — configurable MCQ/Subjective/Practical week evaluation.
# Replaces the earlier flat, fixed-10-MCQ week quiz. Regenerated fresh on
# every "Generate Test" (student reconfigures percentages/count/prompt each
# attempt), not long-term cached like notes. Generation and grading both
# stay on Gemini, matching every other quiz/practice-question call site in
# this module — Claude is reserved for curriculum + VARK notes.
# ============================================================

def _split_question_counts(mcq_percent: float, subjective_percent: float, practical_percent: float, total: int) -> Dict[str, int]:
    """Largest-remainder rounding so the 3 percentages always split into
    integer counts summing exactly to `total` — plain truncation (int(pct/100*total))
    can under-count by 1-2 questions whenever the percentages don't divide evenly."""
    raw = {
        "mcq": mcq_percent / 100 * total,
        "subjective": subjective_percent / 100 * total,
        "practical": practical_percent / 100 * total,
    }
    counts = {k: int(v) for k, v in raw.items()}
    remainder = total - sum(counts.values())
    by_largest_fraction = sorted(raw.keys(), key=lambda k: raw[k] - counts[k], reverse=True)
    for i in range(remainder):
        counts[by_largest_fraction[i % len(by_largest_fraction)]] += 1
    return counts


def build_auto_test_prompt(
    subject: str, week_title: str, subtopic_names: List[str],
    counts: Dict[str, int], custom_prompt: Optional[str] = None,
    grounding_context: Optional[str] = None,
) -> str:
    custom_block = f"\n\nADDITIONAL INSTRUCTIONS FROM THE STUDENT:\n{custom_prompt}" if custom_prompt else ""
    grounding_block = (
        f"""
## Course Material (ground questions in this real content where relevant)
{grounding_context}
"""
        if grounding_context
        else ""
    )
    total = sum(counts.values())
    return f"""You are an expert technical examiner designing a rigorous week-completion test.

## Context
Subject: {subject}
Week: {week_title}
Subtopics Covered: {json.dumps(subtopic_names)}
{custom_block}
{grounding_block}

## Task
Create exactly **{total} questions**, split as:
- {counts['mcq']} Multiple Choice Questions (MCQ)
- {counts['subjective']} Subjective (short/long-form written answer) questions
- {counts['practical']} Practical (applied, hands-on problem-solving) questions

Rules:
- Cover concepts from all subtopics evenly.
- MCQ options must be plausible (no obviously wrong distractors); the correct answer index is 0-based.
- Every Subjective/Practical question needs a detailed model answer a grader can compare a
  student's response against — this is never shown to the student before they answer.
- Vary difficulty roughly evenly across Easy/Medium/Hard.

Return ONLY a valid JSON array (no prose, no markdown) with exactly {total} entries, MCQ questions
first, then Subjective, then Practical:

[
  {{
    "type": "mcq",
    "question": "<Question text>",
    "options": ["<A>", "<B>", "<C>", "<D>"],
    "answer": <0-indexed correct int>,
    "explanation": "<Why this is the correct answer>",
    "difficulty": "Easy | Medium | Hard",
    "topic": "<Which subtopic this question tests>"
  }},
  {{
    "type": "subjective",
    "question": "<Question text>",
    "modelAnswer": "<Detailed reference answer covering every point a full-credit response needs>",
    "explanation": "<Grading guidance — what to look for>",
    "difficulty": "Easy | Medium | Hard",
    "topic": "<Which subtopic this question tests>"
  }}
]

("practical" entries use the exact same shape as "subjective" — just type: "practical".)"""


def build_open_ended_grading_prompt(items: List[Dict[str, Any]]) -> str:
    """items: [{type, question, modelAnswer, explanation, studentAnswer}, ...]"""
    def _format_item(i: int, item: Dict[str, Any]) -> str:
        return (
            f"QUESTION {i + 1} ({item.get('type', 'subjective')}): {item.get('question', '')}\n"
            f"MODEL ANSWER: {item.get('modelAnswer', '')}\n"
            f"GRADING GUIDANCE: {item.get('explanation', '')}\n"
            f"STUDENT'S ANSWER: {item.get('studentAnswer') or '(no answer given)'}"
        )

    items_block = "\n\n".join(_format_item(i, item) for i, item in enumerate(items))
    return f"""You are grading a student's written answers against model answers. For EACH question
below, award partial credit from 0 to 100 based on the correctness, completeness, and understanding
actually demonstrated — not keyword matching — and give brief, specific feedback.

{items_block}

Return ONLY a valid JSON array (no prose, no markdown), exactly one entry per question in the same
order:

[
  {{ "score": <0-100 integer>, "feedback": "<1-2 sentence specific feedback on this answer>" }}
]"""


def build_pre_assessment_prompt(subject: str) -> str:
    return f"""You are an expert educator designing a beginner-level knowledge assessment quiz.

## Context
Subject: {subject}
Level: Beginner (no prior knowledge assumed)
Purpose: Pre-assessment to gauge the student's starting knowledge before creating their learning roadmap.

## Task
Create exactly **10 MCQ questions** that assess fundamental beginner-level understanding of {subject}.

Rules:
- All questions must be at beginner level — suitable for someone just starting to learn {subject}
- Cover a broad range of fundamental concepts (not one narrow topic)
- Each question must have exactly 4 options (A, B, C, D style)
- Options must be plausible — avoid obviously wrong distractors
- The correct answer index is 0-based (0 = first option, 1 = second, etc.)
- No trick questions; test understanding, not memorization of obscure facts

Return ONLY a valid JSON array (no prose, no markdown fences):

[
  {{
    "question": "<Clear, beginner-level question text>",
    "options": ["<Option A>", "<Option B>", "<Option C>", "<Option D>"],
    "answer": <0-indexed correct option integer>
  }}
]"""


# ============================================================
# PRACTICE QUESTIONS  (per-week self-check, "think then reveal")
# ============================================================
# Distinct from the Auto Test: never scored, never gates week unlock — just
# a self-check the student reveals the answer to. That means it doesn't need
# an auto-gradable answer shape, which is what makes it safe to be genuinely
# open-ended (theoretical) for goals where the real exam is written, not MCQ.

def _is_interview_goal(goal: Optional[str]) -> bool:
    return "interview" in (goal or "").lower()


PRACTICE_MCQ_SCHEMA = """
Return strict JSON with this shape:
{
  "questions": [
    {"type": "MCQ", "question": "...", "options": ["...", "...", "...", "..."],
     "answer": 0, "explanation": "..."}
  ]
}
"answer" is the 0-based index into "options" of the correct choice.
"""

PRACTICE_THEORETICAL_SCHEMA = """
Return strict JSON with this shape:
{
  "questions": [
    {"type": "Theoretical", "question": "...", "modelAnswer": "...", "explanation": "..."}
  ]
}
These are open-ended, written-exam-style questions — NOT multiple choice, no
"options" field at all. "modelAnswer" is a complete, well-structured reference
answer (several sentences, exam-quality) the student compares their own
written answer against after thinking it through themselves.
"""


def build_practice_questions_prompt(
    week_title: str, num_questions: int, grounding_context: Optional[str] = None, goal: Optional[str] = None,
) -> str:
    grounding_note = (
        f"SOURCE MATERIAL:\n{grounding_context}\n\nGround questions in this material — use its "
        "actual examples/data where possible.\n\n"
        if grounding_context else
        "No source material found — generate from general subject knowledge.\n\n"
    )
    if _is_interview_goal(goal):
        schema = PRACTICE_MCQ_SCHEMA
        format_note = "multiple-choice"
    else:
        schema = PRACTICE_THEORETICAL_SCHEMA
        format_note = "open-ended, written-exam-style"
    return (
        f"Create {num_questions} {format_note} self-check practice questions covering: {week_title}\n\n"
        f"{grounding_note}{schema}"
    )


# ============================================================
# PRACTICE ANSWER EVALUATION  (theoretical questions only)
# ============================================================
# The student writes their own answer instead of just revealing the model
# answer. This is a lightweight, encouraging check, not a formal grade —
# there's no scoring number, no gate, just a verdict and feedback the way a
# study partner would give it.

ANSWER_EVALUATION_SCHEMA = """
Return strict JSON with this shape:
{
  "verdict": "correct" | "partially_correct" | "incorrect",
  "feedback": "..."
}
"feedback" is 2-4 sentences, addressed directly to the student:
- If "correct": confirm what they got right, in an encouraging tone.
- If "partially_correct": say what they got right, then what's missing or
  imprecise, referencing the model answer's content.
- If "incorrect": be kind but clear about what's wrong, and explain the
  actual correct idea briefly.
Do not repeat the full model answer verbatim in "feedback" — the student can
already see it separately. Judge on substance and understanding, not exact
wording or completeness of prose style.
"""


def build_practice_answer_evaluation_prompt(question: str, model_answer: str, student_answer: str) -> str:
    return (
        f"A student is self-checking their answer to a practice question.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"MODEL ANSWER (reference — the student has not seen your evaluation of it):\n{model_answer}\n\n"
        f"STUDENT'S WRITTEN ANSWER:\n{student_answer}\n\n"
        "Evaluate whether the student's answer captures the substance of the model answer — "
        "minor wording differences or omitted secondary details don't make it wrong; missing "
        "the core concept does.\n\n"
        f"{ANSWER_EVALUATION_SCHEMA}"
    )


# ============================================================
# LEARNING RESOURCES  (index-selection over real search results)
# ============================================================
# Hallucination-proofing: the model only ever sees title+description per
# candidate (never a URL) and can only respond with an index into that
# list. The caller maps the returned index back to the original candidate
# dict to get the real URL — the model's own output is never trusted for
# the link itself, only for which one to pick and why.

LEARNING_RESOURCES_SCHEMA = """
Return ONLY a valid JSON object with this exact shape. No markdown, no extra text.
{
  "video":   [{"index": 0, "blurb": "one sentence on why this helps"}],
  "reading": [{"index": 0, "blurb": "..."}],
  "paper":   [{"index": 0, "blurb": "..."}]
}
Rules:
- "index" MUST be one of the indices actually listed for that category below — never invent one.
- Pick at most 3 per category. If none of a category's candidates are a good fit for the
  topic, return an empty list for that category — do not force a weak match.
- Never include a "url" field and never write out a URL in the blurb — you are choosing
  WHICH listed item is best, not producing a link yourself.
- "blurb" is one specific sentence: what the student will get from THIS item for THIS topic,
  not a generic description.
- Omit any category key entirely if it wasn't provided below.
"""


def build_learning_resources_prompt(topic: str, candidates_by_category: Dict[str, List[Dict[str, Any]]]) -> str:
    def _format_category(name: str, items: List[Dict[str, Any]]) -> str:
        if not items:
            return ""
        lines = [f"{name.upper()} CANDIDATES:"]
        for i, item in enumerate(items):
            desc = (item.get("description") or "").strip()
            lines.append(f"[{i}] {item.get('title', '').strip()}" + (f" — {desc}" if desc else ""))
        return "\n".join(lines)

    blocks = [b for b in (
        _format_category(name, items) for name, items in candidates_by_category.items()
    ) if b]
    candidates_block = "\n\n".join(blocks)

    return (
        f"A student is studying: {topic}\n\n"
        "Below are REAL search results already fetched from YouTube, Wikipedia, and arXiv. "
        "Your only job is to pick the most genuinely useful ones for a student learning this "
        "specific topic, by index, and explain why each pick helps.\n\n"
        f"{candidates_block}\n\n{LEARNING_RESOURCES_SCHEMA}"
    )
