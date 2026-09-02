# ============================================================
# ROADMAP ROUTER
#
# weeks[] schema (schemaVersion 2) — restructured from the earlier fixed
# 4-"levels" -> topics -> subtopics shape. A week holds one main topic +
# introDescription + 3-5 subtopics directly (no topic grouping layer).
# Renamed throughout: level -> week, stageQuiz -> weekQuiz,
# unlockedLevels -> unlockedWeeks, quiz-submit response's
# nextLevelUnlocked -> nextWeekUnlocked. Subtopic keys are now
# "<week>-<subtopicIdx>-<subtopicTitle>" (no topic_idx segment).
#
# Two behavior changes over the prior port, both closing real gaps:
# 1. GET .../quiz now strips `answer`/`explanation` from every question
#    before it reaches the client (previously sent the cached quiz object,
#    including answers, as-is — inspectable via dev tools before
#    submitting). Only the post-submission response includes them.
# 2. `progress.streakDays` is now a real date-boundary daily streak
#    (`_record_daily_activity`) instead of incrementing once per passed
#    quiz — same day = no-op, +1 day = increment, gap = reset to 1.
#
# VARK-personalized notes (style/difficulty-keyed, Claude-generated, with
# Mermaid concept diagrams + hands-on tasks) and a fully configurable Auto
# Test (MCQ/Subjective/Practical mix, replacing the earlier flat 10-MCQ
# weekQuiz) are both built on top of this file's schema foundation — see
# `week.autoTest = {config, questions[], generatedAt}`, regenerated fresh
# per attempt rather than long-term cached like notes.
#
# Async curriculum generation still uses the existing Redis-backed
# job_store.py + BackgroundTasks pattern (matching ai_tutor.py/pomodoro.py).
# Job-status endpoint (GET /status/{job_id}) stays scoped to the requesting
# user (embedded user_id at creation, checked on lookup).
# ============================================================

import asyncio
import html as html_module
import io
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import anthropic
from bson import ObjectId
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ReturnDocument

from app.api.deps import get_current_identity, require_myskillguru_access
from app.core.rate_limit import ai_rate_limit
from app.db.mongodb import get_database
from app.models.ai_usage_event import Feature, Provider
from app.models.roadmap import create_roadmap_document, serialize_roadmap
from app.schemas.roadmap import (
    CreateRoadmapRequest,
    EvaluatePracticeAnswerRequest,
    GenerateAutoTestRequest,
    PreAssessmentRequest,
    SubmitQuizRequest,
    UpdateSubtopicRequest,
)
from app.services.ai_usage import record_ai_usage
from app.services.job_store import get_job, set_job, update_job
from app.services.pdf_render import render_html_to_pdf
from app.services.rag import mongo_store as rag_mongo_store
from app.services.rag import singletons as rag_singletons
from app.services.rag.retrieval import router as rag_router
from app.services.rag.search_clients import search_arxiv, search_wikipedia, search_youtube
from app.services.roadmap_ai import (
    build_auto_test_prompt,
    build_curriculum_prompt,
    build_learning_resources_prompt,
    build_notes_prompt,
    build_open_ended_grading_prompt,
    build_practice_answer_evaluation_prompt,
    build_practice_questions_prompt,
    build_pre_assessment_prompt,
    generate_claude_json,
    generate_curriculum,
    generate_gemini_json,
    is_gemini_quota_error,
    log_style_requirement_gaps,
    validate_interactive_lesson,
    validate_and_repair_diagram,
    _dominant_vark_style,
    _normalize_difficulty,
    _normalize_vark,
    _split_question_counts,
)

router = APIRouter(
    prefix="/api/self-learner/roadmap",
    dependencies=[Depends(get_current_identity), Depends(require_myskillguru_access)],
    tags=["roadmap"],
)

ROADMAP_JOB_PREFIX = "roadmap_job:"
QUIZ_PASS_THRESHOLD = 50
MAX_ACTIVITY_DATES = 30
NOTES_CACHE_VERSION = 4
NOTES_LANGUAGE = "English"


# ============================================================
# PRIVATE HELPERS
# ============================================================

def _is_week_unlocked(doc: Dict[str, Any], week: int) -> bool:
    return week in doc.get("unlockedWeeks", [1])


def _find_week(doc: Dict[str, Any], week: int) -> Optional[Dict[str, Any]]:
    return next((w for w in doc.get("weeks", []) if w.get("week") == week), None)


def _is_subtopic_key_valid(doc: Dict[str, Any], subtopic_key: str) -> bool:
    for wk in doc.get("weeks", []):
        week = wk.get("week")
        for sub_idx, subtopic in enumerate(wk.get("subtopics", [])):
            if subtopic_key == f"{week}-{sub_idx}-{subtopic.get('title')}":
                return True
    return False


def _strip_quiz_answers(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Never send answer/modelAnswer/explanation to the client before submission."""
    return [
        {k: v for k, v in q.items() if k not in ("answer", "modelAnswer", "explanation")}
        for q in questions
    ]


def _sanitize_quiz_for_history(questions: List[Dict[str, Any]], results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_idx = {item.get("questionIdx"): item for item in results}
    safe_questions = []
    for idx, question in enumerate(questions):
        result = by_idx.get(idx, {})
        safe_questions.append({
            "questionIdx": idx,
            "type": question.get("type", "mcq"),
            "question": question.get("question", ""),
            "options": question.get("options", []),
            "yourAnswer": result.get("yourAnswer"),
            "correctAnswer": result.get("correctAnswer"),
            "isCorrect": result.get("isCorrect", False),
            "score": result.get("score", 0),
            "feedback": result.get("feedback", ""),
            "difficulty": question.get("difficulty", ""),
            "topic": question.get("topic", ""),
        })
    return safe_questions


def _recalculate_progress(doc: Dict[str, Any]) -> int:
    weeks = doc.get("weeks", [])
    progress = doc.get("progress", {})
    completed_sub = progress.get("completedSubtopics", [])
    passed_quizzes = progress.get("passedQuizzes", {})

    total_sub = sum(len(wk.get("subtopics", [])) for wk in weeks)
    total_actions = total_sub + len(weeks)  # subtopics + one passed week-quiz per week
    completed_actions = len(completed_sub) + len(passed_quizzes)
    return min(100, round((completed_actions / total_actions * 100))) if total_actions > 0 else 0


_PDF_STATUS_COLORS = {"Completed": "#43C6AC", "In Progress": "#6C63FF", "Locked": "#9CA3AF"}


def _week_status_for_pdf(doc: Dict[str, Any], week: int) -> str:
    if not _is_week_unlocked(doc, week):
        return "Locked"
    if str(week) in (doc.get("progress", {}).get("passedQuizzes") or {}):
        return "Completed"
    return "In Progress"


def _build_roadmap_pdf_html(doc: Dict[str, Any]) -> str:
    """All user-supplied strings (subject, goal, week/subtopic titles, intro
    descriptions) are HTML-escaped — this content ultimately comes from an
    AI generation call seeded by the student's own free-text subject/goal
    input, so it must never be interpolated into the PDF's HTML unescaped."""
    esc = html_module.escape
    subject = esc(doc.get("subject", "Untitled Roadmap"))
    goal = esc(doc.get("goal", ""))
    skill_level = esc(doc.get("skill_level", ""))
    daily_time = esc(doc.get("daily_study_time", ""))
    progress = doc.get("progress", {}) or {}
    overall_progress = progress.get("overallProgress", 0)
    completed_subtopics = set(progress.get("completedSubtopics", []))
    weeks = doc.get("weeks", [])

    weeks_html = []
    for wk in weeks:
        week_num = wk.get("week")
        status = _week_status_for_pdf(doc, week_num)
        color = _PDF_STATUS_COLORS.get(status, "#9CA3AF")
        title = esc(wk.get("title", ""))
        intro = esc(wk.get("introDescription", ""))

        subtopics_html = []
        for idx, sub in enumerate(wk.get("subtopics", [])):
            sub_key = f"{week_num}-{idx}-{sub.get('title')}"
            checkbox = "&#9745;" if sub_key in completed_subtopics else "&#9744;"
            subtopics_html.append(f'<li style="margin-bottom:4px;">{checkbox} {esc(sub.get("title", ""))}</li>')

        weeks_html.append(f"""
        <div style="page-break-inside:avoid; margin-bottom:16px; border:1px solid #E5E7EB; border-radius:10px; padding:16px;">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-size:11px; font-weight:700; color:#6C63FF; text-transform:uppercase;">Week {week_num}</span>
            <span style="font-size:10px; font-weight:700; color:#fff; background:{color}; padding:3px 10px; border-radius:12px;">{status}</span>
          </div>
          <h3 style="margin:6px 0 4px; font-size:15px; color:#1E1B4B;">{title}</h3>
          <p style="margin:0 0 10px; font-size:11px; color:#6B7280; line-height:1.5;">{intro}</p>
          <ul style="margin:0; padding-left:18px; font-size:11px; color:#374151; line-height:1.6;">
            {''.join(subtopics_html)}
          </ul>
        </div>
        """)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ font-family: Arial, Helvetica, sans-serif; color: #1E1B4B; margin: 0; padding: 0; }}
  .header {{ background: linear-gradient(135deg, #1E1B4B, #6C63FF); color: #fff; padding: 24px; border-radius: 12px; margin-bottom: 20px; }}
  .header h1 {{ margin: 0 0 4px; font-size: 22px; }}
  .header p {{ margin: 2px 0; font-size: 12px; opacity: 0.9; }}
  .stats {{ display: flex; gap: 12px; margin-bottom: 20px; }}
  .stat {{ flex: 1; border: 1px solid #E5E7EB; border-radius: 10px; padding: 10px 14px; }}
  .stat .label {{ display:block; font-size: 9px; font-weight: 700; color: #9CA3AF; text-transform: uppercase; }}
  .stat .value {{ display:block; font-size: 16px; font-weight: 800; color: #1E1B4B; margin-top: 2px; }}
</style>
</head>
<body>
  <div class="header">
    <h1>{subject}</h1>
    <p>Goal: {goal}</p>
    <p>Skill Level: {skill_level} &nbsp;&#8226;&nbsp; Daily Study Time: {daily_time}</p>
  </div>

  <div class="stats">
    <div class="stat"><span class="label">Overall Progress</span><span class="value">{overall_progress}%</span></div>
    <div class="stat"><span class="label">Total Weeks</span><span class="value">{len(weeks)}</span></div>
    <div class="stat"><span class="label">Streak</span><span class="value">{progress.get("streakDays", 0)} days</span></div>
  </div>

  {''.join(weeks_html)}
</body>
</html>"""


def _record_daily_activity(progress: Dict[str, Any]) -> Dict[str, Any]:
    """Date-boundary daily streak: same day = no-op, +1 day = increment, any
    other gap = reset to 1. Counts as activity: completing a subtopic, or
    submitting a week quiz (pass or fail)."""
    today = datetime.now(timezone.utc).date()
    today_iso = today.isoformat()
    last_iso = progress.get("lastActivityDate")

    if last_iso != today_iso:
        gap_days = None
        if last_iso:
            try:
                gap_days = (today - datetime.fromisoformat(last_iso).date()).days
            except ValueError:
                gap_days = None
        progress["streakDays"] = (progress.get("streakDays", 0) + 1) if gap_days == 1 else 1
        progress["lastActivityDate"] = today_iso

    activity_dates = progress.get("activityDates", [])
    if today_iso not in activity_dates:
        activity_dates.append(today_iso)
    progress["activityDates"] = activity_dates[-MAX_ACTIVITY_DATES:]

    return progress


async def _retrieve_grounding_context(db: AsyncIOMotorDatabase, record, query: str, user_id: str) -> Optional[str]:
    """Shared retrieval step once a course-material record has already been
    resolved (or not — `record=None` short-circuits to ungrounded)."""
    if record is None:
        return None

    vector_store = None
    if record.doc_type.value == "unstructured":
        vector_store = await asyncio.to_thread(rag_singletons.get_vector_store)
        if vector_store is None:
            return None

    result = await rag_router.retrieve(
        query, record.id, record.doc_type, db, user_id=user_id, vector_store=vector_store,
    )
    if not rag_router.should_use_rag(result):
        return None
    return result.context_text


async def _resolve_grounding(
    db: AsyncIOMotorDatabase, doc: Dict[str, Any], query: str, user_id: str,
) -> Optional[str]:
    """
    Best-effort RAG grounding lookup for any call site AFTER roadmap
    creation (notes, Auto Test, ...). Trusts `doc["grounded_doc_id"]`
    directly via one precise `_id` lookup — which also doubles as a
    staleness check (returns None if the material was deleted since).
    Deliberately does NOT fall back to a subject-text match: a roadmap that
    was never grounded in a document at creation time must stay ungrounded,
    never silently pick up an unrelated (possibly another student's)
    upload that happens to share a subject substring. Returns None (never
    raises) on any failure or when nothing usable resolves — generation
    must always be able to fall back to ungrounded.
    """
    grounded_doc_id = doc.get("grounded_doc_id")
    if not grounded_doc_id:
        return None
    try:
        # owner-scoped: grounded_doc_id was set from record.id at roadmap
        # creation, but re-checking ownership here too means a doc whose
        # access was later revoked (or that was never this user's) can never
        # leak back in through an already-created roadmap.
        record = await rag_mongo_store.find_document_by_id(db, grounded_doc_id, owner_user_id=user_id)
        return await _retrieve_grounding_context(db, record, query, user_id)
    except Exception as e:
        logging.warning("roadmap: RAG grounding lookup failed (falling back to ungrounded): %s", e)
        return None


# ============================================================
# BACKGROUND JOB — ROADMAP CREATION
# ============================================================

async def _run_create_roadmap_job(
    job_id: str, user_id: str, subject: str, goal: str, skill_level: str,
    daily_study_time: str, revision_frequency: str, assessment_score: Optional[float],
    doc_id: Optional[str] = None, custom_instruction: Optional[str] = None,
) -> None:
    db = get_database()
    try:
        await update_job(ROADMAP_JOB_PREFIX, job_id, {"step": "Checking for course material to ground the roadmap in…"})

        query = f"Curriculum structure, topics, and assessment weighting for: {subject} — {goal}"
        if doc_id:
            # Explicit doc_id from an upload the student just did this
            # session — trust it directly. No document was attached ->
            # no grounding lookup at all (see _resolve_grounding's docstring
            # for why there's deliberately no subject-text fallback).
            # owner-scoped: doc_id arrives straight off the request body, so
            # without this a caller could name another user's document and
            # read its contents back out through the notes, practice
            # questions and auto-tests generated from it. An id they don't
            # own resolves to None, indistinguishable from one that doesn't
            # exist, and generation continues ungrounded.
            record = await rag_mongo_store.find_document_by_id(db, doc_id, owner_user_id=user_id)
            if record is None:
                logging.warning(
                    "roadmap job %s: explicit doc_id=%s not found in courseMaterials for this user — "
                    "treating as ungrounded", job_id, doc_id,
                )
                grounding_context, grounded_doc_id = None, None
            else:
                grounding_context = await _retrieve_grounding_context(db, record, query, user_id)
                grounded_doc_id = record.id
                logging.info("roadmap job %s: using explicit doc_id=%s (doc_type=%s)", job_id, doc_id, record.doc_type.value)
        else:
            grounding_context, grounded_doc_id = None, None

        await update_job(ROADMAP_JOB_PREFIX, job_id, {"step": "Generating curriculum with AI…"})

        prompt = build_curriculum_prompt(
            subject, goal, skill_level, daily_study_time, revision_frequency, assessment_score,
            grounding_context=grounding_context,
            custom_instruction=custom_instruction,
        )

        try:
            curriculum, usage, truncated = await asyncio.to_thread(generate_curriculum, prompt)
        except anthropic.APIError as e:
            logging.error("Anthropic API error in roadmap job %s: %s", job_id, e)
            await update_job(ROADMAP_JOB_PREFIX, job_id, {"status": "error", "error": f"AI generation failed: {e}"})
            return

        await record_ai_usage(
            db, user_id=user_id, provider=Provider.CLAUDE, model="claude-sonnet-4-5",
            feature=Feature.ROADMAP_CURRICULUM, usage=usage,
            grounded=grounding_context is not None, job_id=job_id,
        )

        if truncated:
            logging.error("Claude curriculum response truncated — max_tokens limit hit (job %s)", job_id)
            await update_job(ROADMAP_JOB_PREFIX, job_id, {
                "status": "error", "error": "AI response was too long. Try a more specific subject or goal.",
            })
            return

        weeks = curriculum.get("weeks", [])
        if not weeks:
            await update_job(ROADMAP_JOB_PREFIX, job_id, {
                "status": "error", "error": "AI did not return any weeks. Please try again.",
            })
            return

        unlocked = [1]
        if assessment_score is not None and assessment_score >= 80 and len(weeks) > 1:
            unlocked.append(2)

        user_object_id = ObjectId(user_id)

        await db["selfLearnerRoadmaps"].update_many({"user_id": user_object_id}, {"$set": {"active": False}})

        doc_data = {
            "subject": curriculum.get("subject_display_name", subject),
            "goal": goal,
            "skill_level": skill_level,
            "daily_study_time": daily_study_time,
            "revision_frequency": revision_frequency,
            "assessment_score": assessment_score,
            "stats": curriculum.get("stats", {}),
            "weeks": weeks,
            "unlockedWeeks": unlocked,
            "grounded_doc_id": grounded_doc_id,
        }
        doc = create_roadmap_document(user_id, doc_data)
        result = await db["selfLearnerRoadmaps"].insert_one(doc)

        logging.info("Roadmap created for user %s, subject=%s, weeks=%d", user_id, subject, len(weeks))
        await update_job(ROADMAP_JOB_PREFIX, job_id, {
            "status": "done", "roadmap_id": str(result.inserted_id), "step": "Done",
        })

    except Exception as e:
        logging.error("roadmap creation job %s failed: %s", job_id, e, exc_info=True)
        await update_job(ROADMAP_JOB_PREFIX, job_id, {
            "status": "error", "error": "Internal server error during roadmap generation.",
        })


# ── Roadmap creation job status (must be before /{roadmap_id} routes) ──────

@router.get("/status/{job_id}")
async def get_creation_status(job_id: str, identity: dict = Depends(get_current_identity)):
    job = await get_job(ROADMAP_JOB_PREFIX, job_id)
    if job is None or job.get("user_id") != identity["user_id"]:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


# ── Pre-Assessment Quiz (must be before /{roadmap_id} routes) ──────────────

@router.post("/assess", dependencies=[Depends(ai_rate_limit)])
async def generate_pre_assessment(
    payload: PreAssessmentRequest,
    identity: dict = Depends(get_current_identity),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    subject = payload.subject
    prompt = build_pre_assessment_prompt(subject)

    try:
        questions, usage, truncated = await asyncio.to_thread(generate_gemini_json, prompt)
    except Exception as e:
        logging.error("generate_pre_assessment_quiz_controller: %s", e, exc_info=True)
        raise HTTPException(status_code=502, detail=f"AI quiz generation failed: {e}")

    await record_ai_usage(
        db, user_id=identity["user_id"], provider=Provider.GEMINI, model="gemini-2.5-flash",
        feature=Feature.ROADMAP_PRE_ASSESSMENT, usage=usage,
    )

    if truncated:
        logging.error("Gemini pre-assessment response truncated — MAX_TOKENS limit hit")
        raise HTTPException(status_code=502, detail="AI response was too long. Please try again.")

    if not isinstance(questions, list):
        raise HTTPException(status_code=502, detail="AI returned unexpected format. Please try again.")

    logging.info("Pre-assessment quiz generated: subject=%s questions=%d", subject, len(questions))
    return {"questions": questions}


# ── List & Create ───────────────────────────────────────────────────────────

@router.post("", dependencies=[Depends(ai_rate_limit)])
async def create_roadmap(
    background_tasks: BackgroundTasks,
    payload: CreateRoadmapRequest,
    identity: dict = Depends(get_current_identity),
):
    subject = payload.subject
    goal = payload.goal
    skill_level = (payload.skill_level or "Beginner").strip()
    daily_study_time = (payload.daily_study_time or "1 Hour").strip()
    revision_frequency = (payload.revision_frequency or "Every Week").strip()
    assessment_score = payload.assessment_score

    job_id = str(uuid.uuid4())
    await set_job(ROADMAP_JOB_PREFIX, job_id, {
        "status": "processing", "step": "Starting…", "user_id": identity["user_id"],
    })

    background_tasks.add_task(
        _run_create_roadmap_job, job_id, identity["user_id"], subject, goal,
        skill_level, daily_study_time, revision_frequency, assessment_score,
        payload.doc_id, payload.custom_instruction,
    )

    return JSONResponse(status_code=202, content={"job_id": job_id, "status": "processing"})


@router.get("")
async def get_roadmaps(
    identity: dict = Depends(get_current_identity),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    docs = [
        d async for d in
        db["selfLearnerRoadmaps"].find({"user_id": ObjectId(identity["user_id"])}).sort("created_at", -1)
    ]
    return [serialize_roadmap(d) for d in docs]


# ── Single Roadmap ──────────────────────────────────────────────────────────

@router.get("/{roadmap_id}")
async def get_roadmap(
    roadmap_id: str,
    identity: dict = Depends(get_current_identity),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    if not ObjectId.is_valid(roadmap_id):
        raise HTTPException(status_code=400, detail="Invalid roadmap id")

    doc = await db["selfLearnerRoadmaps"].find_one(
        {"_id": ObjectId(roadmap_id), "user_id": ObjectId(identity["user_id"])}
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Roadmap not found")

    return serialize_roadmap(doc)


@router.delete("/{roadmap_id}")
async def delete_roadmap(
    roadmap_id: str,
    identity: dict = Depends(get_current_identity),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    if not ObjectId.is_valid(roadmap_id):
        raise HTTPException(status_code=400, detail="Invalid roadmap id")

    result = await db["selfLearnerRoadmaps"].delete_one(
        {"_id": ObjectId(roadmap_id), "user_id": ObjectId(identity["user_id"])}
    )
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Roadmap not found")

    return {"success": True}


# ── Subtopic Progress ───────────────────────────────────────────────────────

@router.patch("/{roadmap_id}/subtopic")
async def update_subtopic(
    roadmap_id: str,
    payload: UpdateSubtopicRequest,
    identity: dict = Depends(get_current_identity),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    if not ObjectId.is_valid(roadmap_id):
        raise HTTPException(status_code=400, detail="Invalid roadmap id")

    subtopic_key = payload.subtopic_key
    completed = payload.completed

    user_object_id = ObjectId(identity["user_id"])
    roadmap_object_id = ObjectId(roadmap_id)

    doc = await db["selfLearnerRoadmaps"].find_one({"_id": roadmap_object_id, "user_id": user_object_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Roadmap not found")
    if not _is_subtopic_key_valid(doc, subtopic_key):
        raise HTTPException(status_code=400, detail="Invalid subtopic key")

    try:
        week = int(str(subtopic_key).split("-", 1)[0])
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid subtopic key")
    if not _is_week_unlocked(doc, week):
        raise HTTPException(status_code=403, detail=f"Week {week} is locked. Pass previous week quizzes to unlock it.")

    sub_list = doc["progress"]["completedSubtopics"]
    if completed:
        if subtopic_key not in sub_list:
            sub_list.append(subtopic_key)
    else:
        if subtopic_key in sub_list:
            sub_list.remove(subtopic_key)

    progress = doc["progress"]
    progress["completedSubtopics"] = sub_list
    if completed:
        progress = _record_daily_activity(progress)

    doc["progress"] = progress
    progress["overallProgress"] = _recalculate_progress(doc)

    updated_doc = await db["selfLearnerRoadmaps"].find_one_and_update(
        {"_id": roadmap_object_id, "user_id": user_object_id},
        {"$set": {"progress": progress, "updated_at": datetime.now(timezone.utc)}},
        return_document=ReturnDocument.AFTER,
    )

    return serialize_roadmap(updated_doc)


# ── AI Study Notes (on-demand, cached) ──────────────────────────────────────

@router.get("/{roadmap_id}/notes")
async def get_subtopic_notes(
    roadmap_id: str,
    week: int = Query(1),
    subtopic_idx: int = Query(0),
    visual: int = Query(25, ge=0, le=100),
    auditory: int = Query(25, ge=0, le=100),
    reading: int = Query(25, ge=0, le=100),
    kinesthetic: int = Query(25, ge=0, le=100),
    difficulty: str = Query("Moderate"),
    regenerate: bool = Query(False),
    identity: dict = Depends(get_current_identity),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    if not ObjectId.is_valid(roadmap_id):
        raise HTTPException(status_code=400, detail="Invalid roadmap id")

    user_object_id = ObjectId(identity["user_id"])
    roadmap_object_id = ObjectId(roadmap_id)

    doc = await db["selfLearnerRoadmaps"].find_one({"_id": roadmap_object_id, "user_id": user_object_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Roadmap not found")
    if not _is_week_unlocked(doc, week):
        raise HTTPException(status_code=403, detail=f"Week {week} is locked. Pass previous week quizzes to unlock it.")

    week_data = _find_week(doc, week)
    if not week_data:
        raise HTTPException(status_code=404, detail=f"Week {week} not found in roadmap")

    subtopics = week_data.get("subtopics", [])
    if subtopic_idx >= len(subtopics):
        raise HTTPException(status_code=400, detail="subtopic_idx out of range")

    subtopic = subtopics[subtopic_idx]

    vark = _normalize_vark(visual, auditory, reading, kinesthetic)
    dominant_style = _dominant_vark_style(vark).capitalize()
    difficulty_norm = _normalize_difficulty(difficulty)
    # Version the cache whenever the notes JSON contract changes. Existing
    # v1 entries remain in MongoDB but are never served as v2 lessons.
    cache_key = f"v{NOTES_CACHE_VERSION}-{NOTES_LANGUAGE}-{dominant_style}-{difficulty_norm}"

    notes_map = subtopic.get("notes")
    notes_map = notes_map if isinstance(notes_map, dict) else {}

    if not regenerate and cache_key in notes_map:
        return {"notes": notes_map[cache_key], "cached": True, "style": dominant_style, "difficulty": difficulty_norm}

    subject = doc.get("subject", "")
    week_title = week_data.get("title", "")
    sub_title = subtopic.get("title", "")
    sub_summary = subtopic.get("summary", "")
    key_points = subtopic.get("keyPoints", [])

    grounding_context = await _resolve_grounding(
        db, doc, f"{week_title} — {sub_title}: {sub_summary}", identity["user_id"],
    )
    prompt = build_notes_prompt(
        subject, week_title, sub_title, sub_summary, key_points,
        dominant_style, difficulty_norm,
        grounding_context=grounding_context,
        goal=doc.get("goal"),
    )

    try:
        notes, usage, truncated = await asyncio.to_thread(generate_gemini_json, prompt)
        await record_ai_usage(
            db, user_id=identity["user_id"], provider=Provider.GEMINI, model="gemini-2.5-flash",
            feature=Feature.ROADMAP_NOTES, usage=usage, grounded=grounding_context is not None,
        )
    except Exception as e:
        if not is_gemini_quota_error(e):
            logging.error("generate_subtopic_notes: Gemini error: %s", e, exc_info=True)
            raise HTTPException(status_code=502, detail=f"AI notes generation failed: {e}")

        # Gemini quota exhausted — fail over to Claude rather than block
        # notes generation entirely. Same prompt works unmodified: both
        # generators return the same (data, usage, truncated) shape, and
        # extract_json handles a top-level JSON object from either provider
        # identically.
        logging.warning("generate_subtopic_notes: Gemini quota exceeded, falling back to Claude: %s", e)
        try:
            # 10000, not a smaller flat budget — content-heavy notes (Visual-
            # dominant style's concept diagram, Kinesthetic's hands-on task,
            # interview-goal tips, etc.) routinely need more than a few
            # thousand tokens; a too-small flat budget was silently
            # truncating responses.
            notes, usage, truncated = await asyncio.to_thread(generate_claude_json, prompt, 10000)
        except Exception as e2:
            logging.error("generate_subtopic_notes: Claude fallback also failed: %s", e2, exc_info=True)
            raise HTTPException(status_code=502, detail=f"AI notes generation failed: {e2}")
        await record_ai_usage(
            db, user_id=identity["user_id"], provider=Provider.CLAUDE, model="claude-sonnet-4-5",
            feature=Feature.ROADMAP_NOTES, usage=usage, grounded=grounding_context is not None,
        )

    if truncated:
        logging.error("AI notes response truncated — max_tokens limit hit")
        raise HTTPException(status_code=502, detail="AI response was too long. Try a more specific subtopic.")

    if not isinstance(notes, dict):
        logging.error("AI notes response was not a JSON object")
        raise HTTPException(status_code=502, detail="AI returned an invalid notes format. Please regenerate.")

    notes = validate_interactive_lesson(notes)
    notes["notesSchemaVersion"] = NOTES_CACHE_VERSION
    notes["conceptDiagram"] = await validate_and_repair_diagram(
        notes.get("conceptDiagram"), db, identity["user_id"],
    )
    log_style_requirement_gaps(notes, dominant_style, week, subtopic_idx)

    await db["selfLearnerRoadmaps"].update_one(
        {"_id": roadmap_object_id, "user_id": user_object_id},
        {"$set": {
            f"weeks.$[wk].subtopics.{subtopic_idx}.notes.{cache_key}": notes,
            "updated_at": datetime.now(timezone.utc),
        }},
        array_filters=[{"wk.week": week}],
    )

    logging.info(
        "Notes generated: roadmap=%s week=%s subtopic=%s style=%s difficulty=%s",
        roadmap_id, week, subtopic_idx, dominant_style, difficulty_norm,
    )
    return {"notes": notes, "cached": False, "style": dominant_style, "difficulty": difficulty_norm}


# ── AI Learning Resources (on-demand, cached — real external links) ────────

@router.get("/{roadmap_id}/resources")
async def get_subtopic_resources(
    roadmap_id: str,
    week: int = Query(1),
    subtopic_idx: int = Query(0),
    regenerate: bool = Query(False),
    identity: dict = Depends(get_current_identity),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """
    Real external learning resources (YouTube videos, Wikipedia background
    reading, arXiv papers) for a subtopic. Unlike notes, this is NOT
    personalized by VARK/difficulty — resource links are objective facts, so
    there's exactly one cached slot per subtopic.

    Hallucination-proofing: raw candidates always come from real search APIs
    (app/services/rag/search_clients.py) before the model ever sees them,
    and the model only ever picks by index — it never writes a URL, so a
    broken/invented link can never reach a student.
    """
    if not ObjectId.is_valid(roadmap_id):
        raise HTTPException(status_code=400, detail="Invalid roadmap id")

    user_object_id = ObjectId(identity["user_id"])
    roadmap_object_id = ObjectId(roadmap_id)

    doc = await db["selfLearnerRoadmaps"].find_one({"_id": roadmap_object_id, "user_id": user_object_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Roadmap not found")
    if not _is_week_unlocked(doc, week):
        raise HTTPException(status_code=403, detail=f"Week {week} is locked. Pass previous week quizzes to unlock it.")

    week_data = _find_week(doc, week)
    if not week_data:
        raise HTTPException(status_code=404, detail=f"Week {week} not found in roadmap")

    subtopics = week_data.get("subtopics", [])
    if subtopic_idx >= len(subtopics):
        raise HTTPException(status_code=400, detail="subtopic_idx out of range")

    subtopic = subtopics[subtopic_idx]
    cached = subtopic.get("resources")

    if cached and not regenerate:
        return {"resources": cached, "cached": True}

    # ── Fetch REAL candidates from official search APIs (no LLM involved yet) ──
    # Run concurrently — each has its own ~6s network timeout, and chaining
    # three of them back-to-back on top of the Gemini call below risks
    # pushing a single request past the dev proxy's timeout.
    topic = subtopic.get("title", "")
    query = f"{topic} {doc.get('subject', '')}".strip()
    video, reading, paper = await asyncio.gather(
        asyncio.to_thread(search_youtube, query, 5),
        asyncio.to_thread(search_wikipedia, query, 3),
        asyncio.to_thread(search_arxiv, query, 3),
    )
    candidates_by_category = {"video": video, "reading": reading, "paper": paper}

    if not any(candidates_by_category.values()):
        # All three search APIs came back empty (e.g. network hiccup, or
        # YOUTUBE_API_KEY unset) — don't cache a permanent empty result, so
        # a later request can retry once the underlying issue clears.
        logging.warning("No resource candidates found: roadmap=%s W%s S%s", roadmap_id, week, subtopic_idx)
        return {"resources": {"video": [], "reading": [], "paper": []}, "cached": False}

    # ── Model only picks an index into the real candidates above ──────────
    prompt = build_learning_resources_prompt(topic, candidates_by_category)
    try:
        picks, usage, truncated = await asyncio.to_thread(generate_gemini_json, prompt)
        await record_ai_usage(
            db, user_id=identity["user_id"], provider=Provider.GEMINI, model="gemini-2.5-flash",
            feature=Feature.ROADMAP_RESOURCES, usage=usage,
        )
    except Exception as e:
        if not is_gemini_quota_error(e):
            logging.error("get_subtopic_resources: %s", e, exc_info=True)
            raise HTTPException(status_code=502, detail=f"Learning resources fetch failed: {e}")

        # Gemini quota exhausted — fail over to Claude rather than block
        # resource selection entirely. Same prompt works unmodified: both
        # generators return the same (data, usage, truncated) shape.
        logging.warning("get_subtopic_resources: Gemini quota exceeded, falling back to Claude: %s", e)
        try:
            picks, usage, truncated = await asyncio.to_thread(generate_claude_json, prompt, 2000)
        except anthropic.APIError as e2:
            logging.error("get_subtopic_resources: Claude fallback also failed: %s", e2)
            raise HTTPException(status_code=502, detail=f"Learning resources fetch failed: {e2}")
        await record_ai_usage(
            db, user_id=identity["user_id"], provider=Provider.CLAUDE, model="claude-sonnet-4-5",
            feature=Feature.ROADMAP_RESOURCES, usage=usage,
        )

    if truncated:
        logging.error("Learning resources response truncated — MAX_TOKENS limit hit")
        raise HTTPException(status_code=502, detail="Learning resources selection was too long. Please try again.")

    resolved: Dict[str, List[Dict[str, Any]]] = {}
    for category, candidates in candidates_by_category.items():
        chosen = []
        for pick in (picks.get(category) if isinstance(picks, dict) else None) or []:
            idx = pick.get("index") if isinstance(pick, dict) else None
            if not isinstance(idx, int) or idx < 0 or idx >= len(candidates):
                continue  # drop any out-of-range/malformed index rather than guessing
            candidate = candidates[idx]
            chosen.append({
                "title": candidate.get("title", ""),
                "url": candidate.get("url", ""),
                "source": candidate.get("source", ""),
                "blurb": (pick.get("blurb") or "").strip() if isinstance(pick, dict) else "",
            })
        resolved[category] = chosen

    resolved["generatedAt"] = datetime.now(timezone.utc).isoformat()

    await db["selfLearnerRoadmaps"].update_one(
        {"_id": roadmap_object_id, "user_id": user_object_id},
        {"$set": {
            f"weeks.$[wk].subtopics.{subtopic_idx}.resources": resolved,
            "updated_at": datetime.now(timezone.utc),
        }},
        array_filters=[{"wk.week": week}],
    )

    logging.info("Learning resources generated: roadmap=%s week=%s subtopic=%s", roadmap_id, week, subtopic_idx)
    return {"resources": resolved, "cached": False}


# ── Auto Test — configure + generate ────────────────────────────────────────

@router.post("/{roadmap_id}/quiz/generate", dependencies=[Depends(ai_rate_limit)])
async def generate_auto_test(
    roadmap_id: str,
    payload: GenerateAutoTestRequest,
    identity: dict = Depends(get_current_identity),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    if not ObjectId.is_valid(roadmap_id):
        raise HTTPException(status_code=400, detail="Invalid roadmap id")

    week = payload.week
    user_object_id = ObjectId(identity["user_id"])
    roadmap_object_id = ObjectId(roadmap_id)

    doc = await db["selfLearnerRoadmaps"].find_one({"_id": roadmap_object_id, "user_id": user_object_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Roadmap not found")
    if not _is_week_unlocked(doc, week):
        raise HTTPException(status_code=403, detail=f"Week {week} is locked. Pass previous week quizzes to unlock it.")

    week_data = _find_week(doc, week)
    if not week_data:
        raise HTTPException(status_code=404, detail=f"Week {week} not found")

    counts = _split_question_counts(
        payload.mcq_percent, payload.subjective_percent, payload.practical_percent, payload.question_count,
    )

    subject = doc.get("subject", "")
    week_title = week_data.get("title", f"Week {week}")
    subtopic_names = [s.get("title", "") for s in week_data.get("subtopics", [])]

    grounding_context = await _resolve_grounding(
        db, doc, f"Assessment questions for {week_title}: {', '.join(subtopic_names)}", identity["user_id"],
    )
    prompt = build_auto_test_prompt(
        subject, week_title, subtopic_names, counts, payload.custom_prompt, grounding_context=grounding_context,
    )

    try:
        questions, usage, truncated = await asyncio.to_thread(generate_gemini_json, prompt)
        await record_ai_usage(
            db, user_id=identity["user_id"], provider=Provider.GEMINI, model="gemini-2.5-flash",
            feature=Feature.ROADMAP_QUIZ_GENERATE, usage=usage, grounded=grounding_context is not None,
        )
    except Exception as e:
        if not is_gemini_quota_error(e):
            logging.error("generate_auto_test: %s", e, exc_info=True)
            raise HTTPException(status_code=502, detail=f"AI test generation failed: {e}")

        # Gemini quota exhausted — fail over to Claude rather than block quiz
        # generation entirely. Same prompt works unmodified: both generators
        # return the same (data, usage, truncated) shape, and extract_json
        # handles a top-level JSON array from either provider identically.
        logging.warning("generate_auto_test: Gemini quota exceeded, falling back to Claude: %s", e)
        try:
            questions, usage, truncated = await asyncio.to_thread(generate_claude_json, prompt, 8000)
        except anthropic.APIError as e2:
            logging.error("generate_auto_test: Claude fallback also failed: %s", e2)
            raise HTTPException(status_code=502, detail=f"AI test generation failed: {e2}")
        await record_ai_usage(
            db, user_id=identity["user_id"], provider=Provider.CLAUDE, model="claude-sonnet-4-5",
            feature=Feature.ROADMAP_QUIZ_GENERATE, usage=usage, grounded=grounding_context is not None,
        )

    if truncated:
        logging.error("Auto Test response truncated — MAX_TOKENS limit hit")
        raise HTTPException(status_code=502, detail="AI response was too long. Try fewer questions.")

    if not isinstance(questions, list) or not questions:
        raise HTTPException(status_code=502, detail="AI returned no questions. Please try again.")

    config = {
        "mcqPercent": payload.mcq_percent,
        "subjectivePercent": payload.subjective_percent,
        "practicalPercent": payload.practical_percent,
        "questionCount": payload.question_count,
        "customPrompt": payload.custom_prompt,
    }
    auto_test = {"config": config, "questions": questions, "generatedAt": datetime.now(timezone.utc)}

    await db["selfLearnerRoadmaps"].update_one(
        {"_id": roadmap_object_id, "user_id": user_object_id},
        {"$set": {"weeks.$[wk].autoTest": auto_test, "updated_at": datetime.now(timezone.utc)}},
        array_filters=[{"wk.week": week}],
    )

    logging.info("Auto Test generated: roadmap=%s week=%s counts=%s", roadmap_id, week, counts)
    return {"questions": _strip_quiz_answers(questions), "config": config}


# ── Auto Test — resume only (never generates on its own) ───────────────────

@router.get("/{roadmap_id}/quiz")
async def get_active_auto_test(
    roadmap_id: str,
    week: int = Query(1),
    identity: dict = Depends(get_current_identity),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    if not ObjectId.is_valid(roadmap_id):
        raise HTTPException(status_code=400, detail="Invalid roadmap id")

    doc = await db["selfLearnerRoadmaps"].find_one(
        {"_id": ObjectId(roadmap_id), "user_id": ObjectId(identity["user_id"])}
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Roadmap not found")
    if not _is_week_unlocked(doc, week):
        raise HTTPException(status_code=403, detail=f"Week {week} is locked. Pass previous week quizzes to unlock it.")

    week_data = _find_week(doc, week)
    if not week_data:
        raise HTTPException(status_code=404, detail=f"Week {week} not found")

    auto_test = week_data.get("autoTest")
    if not auto_test or not auto_test.get("questions"):
        return {"questions": None, "config": None}

    return {"questions": _strip_quiz_answers(auto_test["questions"]), "config": auto_test.get("config")}


# ── AI Practice Questions (on-demand, cached, self-check only — not scored) ─

@router.get("/{roadmap_id}/practice")
async def get_practice_questions(
    roadmap_id: str,
    week: int = Query(1),
    subtopic_idx: int = Query(0),
    identity: dict = Depends(get_current_identity),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """
    Self-check questions the student reveals the answer to — never scored,
    never gates week unlock (unlike the week quiz). Scoped to ONE subtopic
    (not the whole week) so the questions are specifically about the topic
    the student just finished — the frontend calls this the moment a
    subtopic is marked complete. Format follows the roadmap's goal:
    multiple-choice for interview-prep, open-ended written-exam-style
    questions otherwise.
    """
    if not ObjectId.is_valid(roadmap_id):
        raise HTTPException(status_code=400, detail="Invalid roadmap id")

    user_object_id = ObjectId(identity["user_id"])
    roadmap_object_id = ObjectId(roadmap_id)

    doc = await db["selfLearnerRoadmaps"].find_one({"_id": roadmap_object_id, "user_id": user_object_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Roadmap not found")
    if not _is_week_unlocked(doc, week):
        raise HTTPException(status_code=403, detail=f"Week {week} is locked. Pass previous week quizzes to unlock it.")

    week_data = _find_week(doc, week)
    if not week_data:
        raise HTTPException(status_code=404, detail=f"Week {week} not found")

    subtopics = week_data.get("subtopics", [])
    if subtopic_idx >= len(subtopics):
        raise HTTPException(status_code=400, detail="subtopic_idx out of range")
    subtopic = subtopics[subtopic_idx]

    cached = subtopic.get("practiceQuestions")
    if cached:
        return {"questions": cached, "cached": True}

    sub_title = subtopic.get("title", "")
    sub_summary = subtopic.get("summary", "")
    key_points = subtopic.get("keyPoints", [])
    topic = f"{sub_title} — {sub_summary}" if sub_summary else sub_title
    if key_points:
        topic += f" (covers: {', '.join(key_points)})"

    grounding_context = await _resolve_grounding(
        db, doc, f"Practice questions for {sub_title}", identity["user_id"],
    )
    prompt = build_practice_questions_prompt(topic, 10, grounding_context, doc.get("goal"))

    try:
        data, usage, truncated = await asyncio.to_thread(generate_gemini_json, prompt)
    except Exception as e:
        logging.error("get_practice_questions: %s", e, exc_info=True)
        raise HTTPException(status_code=502, detail=f"AI practice question generation failed: {e}")

    await record_ai_usage(
        db, user_id=identity["user_id"], provider=Provider.GEMINI, model="gemini-2.5-flash",
        feature=Feature.ROADMAP_PRACTICE_QUESTIONS, usage=usage, grounded=bool(grounding_context),
    )

    if truncated:
        logging.error("Gemini practice questions response truncated — MAX_TOKENS limit hit")
        raise HTTPException(status_code=502, detail="AI response was too long. Please try again.")

    questions = (data or {}).get("questions", []) if isinstance(data, dict) else []

    await db["selfLearnerRoadmaps"].update_one(
        {"_id": roadmap_object_id, "user_id": user_object_id},
        {"$set": {
            f"weeks.$[wk].subtopics.{subtopic_idx}.practiceQuestions": questions,
            "updated_at": datetime.now(timezone.utc),
        }},
        array_filters=[{"wk.week": week}],
    )

    logging.info(
        "Practice questions generated: roadmap=%s week=%s subtopic=%s count=%s",
        roadmap_id, week, subtopic_idx, len(questions),
    )
    return {"questions": questions, "cached": False}


@router.post("/{roadmap_id}/practice/evaluate", dependencies=[Depends(ai_rate_limit)])
async def evaluate_practice_answer(
    roadmap_id: str,
    payload: EvaluatePracticeAnswerRequest,
    identity: dict = Depends(get_current_identity),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """
    Judges a student's written answer to a theoretical practice question
    against its stored model answer. Not scored, not saved — a live check,
    the way a study partner would react to reading it.
    """
    if not ObjectId.is_valid(roadmap_id):
        raise HTTPException(status_code=400, detail="Invalid roadmap id")

    user_object_id = ObjectId(identity["user_id"])
    roadmap_object_id = ObjectId(roadmap_id)
    week = payload.week

    doc = await db["selfLearnerRoadmaps"].find_one({"_id": roadmap_object_id, "user_id": user_object_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Roadmap not found")
    if not _is_week_unlocked(doc, week):
        raise HTTPException(status_code=403, detail=f"Week {week} is locked. Pass previous week quizzes to unlock it.")

    week_data = _find_week(doc, week)
    if not week_data:
        raise HTTPException(status_code=404, detail=f"Week {week} not found")

    subtopics = week_data.get("subtopics", [])
    if payload.subtopic_idx < 0 or payload.subtopic_idx >= len(subtopics):
        raise HTTPException(status_code=400, detail="Invalid subtopic_idx")

    questions = subtopics[payload.subtopic_idx].get("practiceQuestions") or []
    if payload.question_idx < 0 or payload.question_idx >= len(questions):
        raise HTTPException(status_code=400, detail="Invalid question_idx")

    question = questions[payload.question_idx]
    if question.get("type") != "Theoretical" or not question.get("modelAnswer"):
        raise HTTPException(status_code=400, detail="This question doesn't support written-answer evaluation")

    prompt = build_practice_answer_evaluation_prompt(
        question["question"], question["modelAnswer"], payload.student_answer,
    )
    try:
        result, usage, truncated = await asyncio.to_thread(generate_gemini_json, prompt)
    except Exception as e:
        logging.error("evaluate_practice_answer: %s", e, exc_info=True)
        raise HTTPException(status_code=502, detail=f"Answer evaluation failed: {e}")

    await record_ai_usage(
        db, user_id=identity["user_id"], provider=Provider.GEMINI, model="gemini-2.5-flash",
        feature=Feature.ROADMAP_PRACTICE_EVALUATE, usage=usage,
    )

    if truncated or not isinstance(result, dict):
        raise HTTPException(status_code=502, detail="AI evaluation response was incomplete. Please try again.")

    return {
        "verdict": result.get("verdict", "incorrect"),
        "feedback": result.get("feedback", ""),
        "modelAnswer": question["modelAnswer"],
    }


# ── AI Quiz Submission & Grading ────────────────────────────────────────────

@router.post("/{roadmap_id}/quiz/submit")
async def submit_quiz(
    roadmap_id: str,
    payload: SubmitQuizRequest,
    identity: dict = Depends(get_current_identity),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    if not ObjectId.is_valid(roadmap_id):
        raise HTTPException(status_code=400, detail="Invalid roadmap id")

    user_object_id = ObjectId(identity["user_id"])
    roadmap_object_id = ObjectId(roadmap_id)

    week = payload.week
    answers = payload.answers

    doc = await db["selfLearnerRoadmaps"].find_one({"_id": roadmap_object_id, "user_id": user_object_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Roadmap not found")
    if not _is_week_unlocked(doc, week):
        raise HTTPException(status_code=403, detail=f"Week {week} is locked. Pass previous week quizzes to unlock it.")

    week_data = _find_week(doc, week)
    if not week_data:
        raise HTTPException(status_code=404, detail=f"Week {week} not found")

    auto_test = week_data.get("autoTest")
    questions = auto_test.get("questions") if auto_test else None
    if not questions:
        raise HTTPException(status_code=400, detail="Auto Test not generated yet. Call POST .../quiz/generate first.")

    # MCQ graded immediately (exact match). Subjective/Practical are batched
    # into ONE AI call (not N calls) for partial-credit scoring + feedback —
    # unanswered open-ended questions are scored 0 without spending a call.
    per_question_score: Dict[int, float] = {}
    per_question_feedback: Dict[int, str] = {}
    open_ended_indices: List[int] = []
    open_ended_items: List[Dict[str, Any]] = []

    for idx, q in enumerate(questions):
        student_answer = answers.get(str(idx))
        if q.get("type", "mcq") == "mcq":
            is_correct = student_answer == q.get("answer")
            per_question_score[idx] = 100.0 if is_correct else 0.0
            per_question_feedback[idx] = q.get("explanation", "")
        elif student_answer is None or not str(student_answer).strip():
            per_question_score[idx] = 0.0
            per_question_feedback[idx] = "No answer provided."
        else:
            open_ended_indices.append(idx)
            open_ended_items.append({
                "type": q.get("type"),
                "question": q.get("question", ""),
                "modelAnswer": q.get("modelAnswer", ""),
                "explanation": q.get("explanation", ""),
                "studentAnswer": student_answer,
            })

    if open_ended_items:
        prompt = build_open_ended_grading_prompt(open_ended_items)
        try:
            grading, usage, truncated = await asyncio.to_thread(generate_gemini_json, prompt)
        except Exception as e:
            logging.error("submit_quiz: open-ended grading failed: %s", e, exc_info=True)
            raise HTTPException(status_code=502, detail=f"AI grading failed: {e}")

        await record_ai_usage(
            db, user_id=identity["user_id"], provider=Provider.GEMINI, model="gemini-2.5-flash",
            feature=Feature.ROADMAP_QUIZ_GRADING, usage=usage,
        )

        if truncated or not isinstance(grading, list) or len(grading) != len(open_ended_items):
            logging.error(
                "submit_quiz: grading response malformed/truncated (expected %d entries, got %s)",
                len(open_ended_items), grading if not truncated else "TRUNCATED",
            )
            raise HTTPException(status_code=502, detail="AI grading response was incomplete. Please try submitting again.")

        for i, idx in enumerate(open_ended_indices):
            entry = grading[i] if isinstance(grading[i], dict) else {}
            try:
                score = float(entry.get("score", 0))
            except (TypeError, ValueError):
                score = 0.0
            per_question_score[idx] = max(0.0, min(100.0, score))
            per_question_feedback[idx] = str(entry.get("feedback", "") or "")

    correct_count = 0
    wrong_topics: List[str] = []
    results_detail = []

    for idx, q in enumerate(questions):
        score = per_question_score[idx]
        is_correct = score >= 50
        if is_correct:
            correct_count += 1
        else:
            topic_label = q.get("topic", "")
            if topic_label and topic_label not in wrong_topics:
                wrong_topics.append(topic_label)

        results_detail.append({
            "questionIdx": idx,
            "type": q.get("type", "mcq"),
            "yourAnswer": answers.get(str(idx)),
            "correctAnswer": q.get("answer") if q.get("type", "mcq") == "mcq" else q.get("modelAnswer"),
            "isCorrect": is_correct,
            "score": round(score),
            "feedback": per_question_feedback.get(idx, ""),
            "topic": q.get("topic", ""),
        })

    final_score = round(sum(per_question_score.values()) / len(questions)) if questions else 0
    passed = final_score >= QUIZ_PASS_THRESHOLD

    progress = doc.get("progress", {})
    total_weeks = len(doc.get("weeks", []))
    unlocked = list(doc.get("unlockedWeeks", [1]))
    previously_unlocked = set(unlocked)
    next_week = week + 1
    next_week_unlocked = False

    if passed:
        progress.setdefault("passedQuizzes", {})[str(week)] = final_score
        if next_week <= total_weeks and next_week not in unlocked:
            unlocked.append(next_week)
            next_week_unlocked = next_week not in previously_unlocked

    # Counts as activity whether the student passed or not.
    progress = _record_daily_activity(progress)

    existing_weak = progress.get("weakTopics", [])
    for wt in wrong_topics:
        if wt not in existing_weak:
            existing_weak.append(wt)
    progress["weakTopics"] = existing_weak[:10]
    progress.setdefault("quizHistory", []).append({
        "week": week,
        "score": final_score,
        "passed": passed,
        "correctCount": correct_count,
        "totalQuestions": len(questions),
        "weakTopics": wrong_topics,
        "submittedAt": datetime.now(timezone.utc),
        "questions": _sanitize_quiz_for_history(questions, results_detail),
    })
    progress["quizHistory"] = progress["quizHistory"][-25:]

    doc["progress"] = progress
    doc["unlockedWeeks"] = unlocked
    progress["overallProgress"] = _recalculate_progress(doc)

    updated_doc = await db["selfLearnerRoadmaps"].find_one_and_update(
        {"_id": roadmap_object_id, "user_id": user_object_id},
        {"$set": {"progress": progress, "unlockedWeeks": unlocked, "updated_at": datetime.now(timezone.utc)}},
        return_document=ReturnDocument.AFTER,
    )

    logging.info("Quiz graded: roadmap=%s week=%s score=%s%% passed=%s", roadmap_id, week, final_score, passed)
    return {
        "score": final_score,
        "passed": passed,
        "correctCount": correct_count,
        "totalQuestions": len(questions),
        "nextWeekUnlocked": next_week_unlocked,
        "weakTopics": wrong_topics,
        "results": results_detail,
        "roadmap": serialize_roadmap(updated_doc),
    }


# ── Quiz History ─────────────────────────────────────────────────────────────

@router.get("/{roadmap_id}/quiz/history")
async def get_quiz_history(
    roadmap_id: str,
    identity: dict = Depends(get_current_identity),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    if not ObjectId.is_valid(roadmap_id):
        raise HTTPException(status_code=400, detail="Invalid roadmap id")

    doc = await db["selfLearnerRoadmaps"].find_one(
        {"_id": ObjectId(roadmap_id), "user_id": ObjectId(identity["user_id"])},
        {"progress.quizHistory": 1},
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Roadmap not found")

    history = doc.get("progress", {}).get("quizHistory", [])
    return {"history": [serialize_roadmap(item) for item in reversed(history)]}


# ── PDF Export ───────────────────────────────────────────────────────────────

@router.get("/{roadmap_id}/pdf")
async def download_roadmap_pdf(
    roadmap_id: str,
    identity: dict = Depends(get_current_identity),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    if not ObjectId.is_valid(roadmap_id):
        raise HTTPException(status_code=400, detail="Invalid roadmap id")

    doc = await db["selfLearnerRoadmaps"].find_one(
        {"_id": ObjectId(roadmap_id), "user_id": ObjectId(identity["user_id"])}
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Roadmap not found")

    html_content = _build_roadmap_pdf_html(doc)

    try:
        pdf_bytes = await asyncio.to_thread(render_html_to_pdf, html_content)
    except Exception as e:
        logging.error("download_roadmap_pdf: PDF rendering failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate PDF. Please try again.")

    safe_subject = re.sub(r"[^A-Za-z0-9_-]+", "_", doc.get("subject", "roadmap")).strip("_") or "roadmap"
    filename = f"roadmap_{safe_subject}.pdf"

    logging.info("Roadmap PDF generated: roadmap=%s bytes=%d", roadmap_id, len(pdf_bytes))
    return StreamingResponse(
        io.BytesIO(pdf_bytes), media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
