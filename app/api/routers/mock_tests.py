# ============================================================
# MOCK TESTS ROUTER (Test Engine)
# Ported from routes/institute/mock_test_routes.py +
# controllers/institute/mock_test_controller.py +
# controllers/institute/test_attempt_controller.py
#
# Two creation modes: "subject" (practice tests, original flow, this file's
# only concern originally) and "roadmap" (week-range tests reusing the
# roadmap module's Auto Test machinery — see _create_roadmap_test /
# _submit_roadmap_test / _review_roadmap_test).
#
# Analytics (GET .../analytics, POST .../attempts/.../insight) used to live
# here too but has moved to self_learner_analytics.py, mounted at
# /self-learner/analytics/* — see that file for the merged-across-sources
# dashboard and "Get Detailed Feedback" logic.
#
# Deviation from Flask: GET /mock-tests/{id} (used to fetch a test for the
# student to take) redacts correct_answer/explanation from each question —
# Flask returned them as-is, inspectable via dev tools before submitting.
# Only the post-submission /review response includes them.
# ============================================================

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from bson import ObjectId
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.api.deps import get_current_identity, require_myskillguru_access
from app.api.routers.roadmap import _resolve_grounding
from app.core.rate_limit import ai_rate_limit
from app.db.mongodb import get_database
from app.models.ai_usage_event import Feature, Provider
from app.models.mock_test import (
    build_create_document,
    build_roadmap_test_document,
    serialize_mock_test,
    serialize_question,
    serialize_roadmap_test_question,
)
from app.schemas.mock_test import MockTestCreateRequest, MockTestSubmitRequest
from app.services.ai_usage import record_ai_usage
from app.services.mock_test_generation import MOCK_TEST_MODEL, build_mock_test_prompt, generate_mock_test_questions
from app.services.roadmap_ai import (
    build_auto_test_prompt,
    build_open_ended_grading_prompt,
    generate_gemini_json,
    _split_question_counts,
)

router = APIRouter(
    dependencies=[Depends(get_current_identity), Depends(require_myskillguru_access)],
    tags=["mock-tests"],
)


def _serialize_attempt(doc: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(doc)
    out["_id"] = str(out["_id"])
    out["student_id"] = str(out["student_id"])
    out["test_id"] = str(out["test_id"])
    if out.get("subject_id"):
        out["subject_id"] = str(out["subject_id"])
    for field in ("submitted_at", "created_at"):
        if out.get(field):
            out[field] = out[field].isoformat()
    return out


def _evaluate_answers(
    questions: List[Dict[str, Any]], answers: Dict[str, Any], marks_per_question: float,
    negative_marking: bool, negative_marks: float,
) -> Tuple[int, int, int, float, List[Dict[str, Any]]]:
    correct = wrong = skipped = 0
    scored = 0.0
    qwise = []

    for q in questions:
        qid = str(q.get("_id") or q.get("id") or "")
        student_answer = str(answers.get(qid, "")).strip().lower()
        correct_answer = str(q.get("correct_answer", "")).strip().lower()
        q_marks = q.get("marks", marks_per_question)

        if not student_answer:
            skipped += 1
            status, marks_awarded = "skipped", 0
        elif student_answer == correct_answer:
            correct += 1
            status, marks_awarded = "correct", q_marks
            scored += q_marks
        else:
            wrong += 1
            marks_awarded = -negative_marks if negative_marking else 0
            status = "wrong"
            scored += marks_awarded

        qwise.append({
            "question_id": qid,
            "student_answer": answers.get(qid, ""),
            "correct_answer": q.get("correct_answer"),
            "status": status,
            "marks_awarded": marks_awarded,
        })

    return correct, wrong, skipped, max(0, round(scored, 2)), qwise


# ============================================================
# BACKGROUND GENERATION
# ============================================================

async def _run_generation(test_id: ObjectId, prompt: str, user_id: str) -> None:
    db = get_database()
    now = datetime.now(timezone.utc)
    try:
        questions, usage = await asyncio.to_thread(generate_mock_test_questions, prompt)
        await record_ai_usage(
            db, user_id=user_id, provider=Provider.CLAUDE, model=MOCK_TEST_MODEL,
            feature=Feature.TEST_ENGINE_GENERATE, usage=usage, context_id=str(test_id),
        )
        await db["mockTests"].update_one(
            {"_id": test_id},
            {"$set": {"questions": questions, "questionCount": len(questions), "updated_at": now}},
        )
        logging.info("[mock-test:%s] generation completed — %d questions.", test_id, len(questions))
    except Exception as e:
        logging.error("[mock-test:%s] generation failed: %s", test_id, e, exc_info=True)
        await db["mockTests"].update_one({"_id": test_id}, {"$set": {"generationError": str(e), "updated_at": now}})


async def _run_roadmap_generation(test_id: ObjectId, prompt: str, user_id: str, grounded: bool = False) -> None:
    """Roadmap-mode counterpart of _run_generation — reuses Auto Test's
    question generation (Gemini, same as every other roadmap quiz/practice
    call site) instead of subject-mode's own prompt/schema."""
    db = get_database()
    now = datetime.now(timezone.utc)
    try:
        questions, usage, truncated = await asyncio.to_thread(generate_gemini_json, prompt)
        await record_ai_usage(
            db, user_id=user_id, provider=Provider.GEMINI, model="gemini-2.5-flash",
            feature=Feature.TEST_ENGINE_GENERATE, usage=usage, grounded=grounded, context_id=str(test_id),
        )

        if truncated or not isinstance(questions, list) or not questions:
            raise ValueError("AI did not return valid questions (truncated or empty response)")

        await db["mockTests"].update_one(
            {"_id": test_id},
            {"$set": {"questions": questions, "questionCount": len(questions), "updated_at": now}},
        )
        logging.info("[roadmap-test:%s] generation completed — %d questions.", test_id, len(questions))
    except Exception as e:
        logging.error("[roadmap-test:%s] generation failed: %s", test_id, e, exc_info=True)
        await db["mockTests"].update_one({"_id": test_id}, {"$set": {"generationError": str(e), "updated_at": now}})


# ============================================================
# ROUTES
# ============================================================

@router.post("/mock-tests", dependencies=[Depends(ai_rate_limit)])
async def create_mock_test(
    background_tasks: BackgroundTasks,
    payload: MockTestCreateRequest,
    identity: dict = Depends(get_current_identity),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    student_id = ObjectId(identity["user_id"])

    if payload.mode == "roadmap":
        return await _create_roadmap_test(background_tasks, payload, student_id, identity["user_id"], db)

    try:
        doc = build_create_document(payload.model_dump(exclude_unset=True), student_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result = await db["mockTests"].insert_one(doc)
    test_id = result.inserted_id

    prompt = build_mock_test_prompt(
        doc.get("subjectName") or "General", doc.get("topic"), doc["difficulty"],
        doc["questionCount"], doc["questionTypes"], doc["marksPerQuestion"],
    )
    background_tasks.add_task(_run_generation, test_id, prompt, identity["user_id"])

    return {
        "success": True,
        "message": "Mock test created. Questions are being generated…",
        "testId": str(test_id),
        "mockTest": {"_id": str(test_id)},
        "test": {"_id": str(test_id)},
    }


async def _create_roadmap_test(
    background_tasks: BackgroundTasks, payload: MockTestCreateRequest,
    student_id: ObjectId, user_id: str, db: AsyncIOMotorDatabase,
) -> Dict[str, Any]:
    if not payload.roadmap_id or not ObjectId.is_valid(payload.roadmap_id):
        raise HTTPException(status_code=400, detail="A valid roadmap_id is required for roadmap-mode tests")
    if not payload.week_start or not payload.week_end:
        raise HTTPException(status_code=400, detail="week_start and week_end are required for roadmap-mode tests")
    if payload.week_start > payload.week_end:
        raise HTTPException(status_code=400, detail="week_start must be <= week_end")

    roadmap_object_id = ObjectId(payload.roadmap_id)
    roadmap_doc = await db["selfLearnerRoadmaps"].find_one({"_id": roadmap_object_id, "user_id": student_id})
    if not roadmap_doc:
        raise HTTPException(status_code=404, detail="Roadmap not found")

    weeks_in_range = [w for w in roadmap_doc.get("weeks", []) if payload.week_start <= w.get("week", 0) <= payload.week_end]
    if len(weeks_in_range) != (payload.week_end - payload.week_start + 1):
        raise HTTPException(status_code=400, detail="One or more weeks in that range don't exist on this roadmap")

    unlocked = set(roadmap_doc.get("unlockedWeeks", [1]))
    if not all(w.get("week") in unlocked for w in weeks_in_range):
        raise HTTPException(status_code=403, detail="One or more weeks in that range are locked")

    mcq_pct = payload.mcq_percent if payload.mcq_percent is not None else 100
    subj_pct = payload.subjective_percent if payload.subjective_percent is not None else 0
    prac_pct = payload.practical_percent if payload.practical_percent is not None else 0
    if round(mcq_pct + subj_pct + prac_pct) != 100:
        raise HTTPException(status_code=400, detail="mcq_percent + subjective_percent + practical_percent must sum to 100")

    question_count = max(1, min(50, payload.questionCount or 10))
    config = {
        "mcqPercent": mcq_pct, "subjectivePercent": subj_pct, "practicalPercent": prac_pct,
        "questionCount": question_count, "customPrompt": payload.custom_prompt,
    }

    subject = roadmap_doc.get("subject", "")
    doc = build_roadmap_test_document(
        student_id, roadmap_object_id, subject, payload.week_start, payload.week_end, config,
    )
    result = await db["mockTests"].insert_one(doc)
    test_id = result.inserted_id

    counts = _split_question_counts(mcq_pct, subj_pct, prac_pct, question_count)
    week_title = (
        f"Weeks {payload.week_start}-{payload.week_end}"
        if payload.week_end != payload.week_start else f"Week {payload.week_start}"
    )
    subtopic_names = [s.get("title", "") for w in weeks_in_range for s in w.get("subtopics", [])]

    grounding_context = await _resolve_grounding(
        db, roadmap_doc, subject, f"Assessment questions for {week_title}: {', '.join(subtopic_names)}", user_id,
    )
    prompt = build_auto_test_prompt(
        subject, week_title, subtopic_names, counts, payload.custom_prompt, grounding_context=grounding_context,
    )
    background_tasks.add_task(_run_roadmap_generation, test_id, prompt, user_id, grounding_context is not None)

    return {
        "success": True,
        "message": "Roadmap test created. Questions are being generated…",
        "testId": str(test_id),
        "mockTest": {"_id": str(test_id)},
        "test": {"_id": str(test_id)},
    }


@router.get("/mock-tests")
async def list_mock_tests(identity: dict = Depends(get_current_identity), db: AsyncIOMotorDatabase = Depends(get_database)):
    student_id = ObjectId(identity["user_id"])
    cursor = db["mockTests"].find({"student_id": student_id}, {"questions": 0}).sort("created_at", -1)
    tests = [serialize_mock_test(doc) async for doc in cursor]
    return {"success": True, "mockTests": tests, "total": len(tests)}


@router.get("/mock-tests/{test_id}")
async def get_mock_test(
    test_id: str, identity: dict = Depends(get_current_identity), db: AsyncIOMotorDatabase = Depends(get_database)
):
    if not ObjectId.is_valid(test_id):
        raise HTTPException(status_code=400, detail="Invalid test_id")

    doc = await db["mockTests"].find_one({"_id": ObjectId(test_id), "student_id": ObjectId(identity["user_id"])})
    if not doc:
        raise HTTPException(status_code=404, detail="Mock test not found")

    return {"success": True, "test": serialize_mock_test(doc, include_answers=False)}


@router.delete("/mock-tests/{test_id}")
async def delete_mock_test(
    test_id: str, identity: dict = Depends(get_current_identity), db: AsyncIOMotorDatabase = Depends(get_database)
):
    if not ObjectId.is_valid(test_id):
        raise HTTPException(status_code=400, detail="Invalid test_id")

    result = await db["mockTests"].delete_one({"_id": ObjectId(test_id), "student_id": ObjectId(identity["user_id"])})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Mock test not found")

    return {"success": True, "message": "Mock test deleted"}


@router.post("/mock-tests/{test_id}/submit")
async def submit_test(
    test_id: str,
    payload: MockTestSubmitRequest,
    identity: dict = Depends(get_current_identity),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    if not ObjectId.is_valid(test_id):
        raise HTTPException(status_code=400, detail="Invalid test_id")

    student_id = ObjectId(identity["user_id"])
    doc = await db["mockTests"].find_one({"_id": ObjectId(test_id), "student_id": student_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Mock test not found")

    if doc.get("mode") == "roadmap":
        return await _submit_roadmap_test(
            ObjectId(test_id), doc, payload.answers, student_id, identity["user_id"], db,
        )

    answers = payload.answers

    questions = doc.get("questions", [])
    correct, wrong, skipped, scored, qwise = _evaluate_answers(
        questions, answers, doc.get("marksPerQuestion", 1),
        doc.get("negativeMarking", False), doc.get("negativeMarks", 0),
    )

    total_marks = doc.get("totalMarks", 0) or 0
    accuracy = round(correct / (correct + wrong) * 100, 1) if (correct + wrong) else 0.0
    percentage = round(scored / total_marks * 100, 1) if total_marks else 0.0

    now = datetime.now(timezone.utc)
    attempt_doc = {
        "student_id": student_id,
        "test_id": ObjectId(test_id),
        "testTitle": doc.get("subjectName") or doc.get("topic") or "Mock Test",
        "subjectName": doc.get("subjectName"),
        "subject_id": doc.get("subject_id"),
        "answers": answers,
        "questionwise": qwise,
        "correct": correct, "wrong": wrong, "skipped": skipped,
        "scored": scored, "totalMarks": total_marks,
        "accuracy": accuracy, "percentage": percentage,
        "submitted_at": now, "created_at": now,
    }
    result = await db["testAttempts"].insert_one(attempt_doc)
    attempt_id = result.inserted_id

    await db["mockTests"].update_one(
        {"_id": ObjectId(test_id)},
        {"$set": {"status": "submitted", "last_attempt": attempt_id, "updated_at": now}, "$inc": {"attempts_count": 1}},
    )

    return {
        "success": True,
        "attemptId": str(attempt_id),
        "result": {
            "scored": scored, "totalMarks": total_marks,
            "correct": correct, "wrong": wrong, "skipped": skipped,
            "accuracy": accuracy, "percentage": percentage,
            "submittedAt": now.isoformat(),
        },
    }


async def _submit_roadmap_test(
    test_id: ObjectId, doc: Dict[str, Any], answers: Dict[str, Any],
    student_id: ObjectId, user_id: str, db: AsyncIOMotorDatabase,
) -> Dict[str, Any]:
    """Mirrors roadmap.py's submit_quiz grading exactly (MCQ exact-index
    match; Subjective/Practical batched into one AI call) — questions here
    are Auto-Test-shaped and identified by list index, not the _id-keyed
    scheme subject-mode's _evaluate_answers uses, so this is a fully
    separate path rather than a branch inside that function."""
    questions = doc.get("questions", [])
    if not questions:
        raise HTTPException(status_code=400, detail="This test hasn't finished generating yet.")

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
                "type": q.get("type"), "question": q.get("question", ""),
                "modelAnswer": q.get("modelAnswer", ""), "explanation": q.get("explanation", ""),
                "studentAnswer": student_answer,
            })

    if open_ended_items:
        prompt = build_open_ended_grading_prompt(open_ended_items)
        try:
            grading, usage, truncated = await asyncio.to_thread(generate_gemini_json, prompt)
        except Exception as e:
            logging.error("submit_roadmap_test: open-ended grading failed: %s", e, exc_info=True)
            raise HTTPException(status_code=502, detail=f"AI grading failed: {e}")

        await record_ai_usage(
            db, user_id=user_id, provider=Provider.GEMINI, model="gemini-2.5-flash",
            feature=Feature.TEST_ENGINE_GRADING, usage=usage, context_id=str(test_id),
        )

        if truncated or not isinstance(grading, list) or len(grading) != len(open_ended_items):
            logging.error(
                "submit_roadmap_test: grading response malformed/truncated (expected %d entries)",
                len(open_ended_items),
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

    correct = wrong = 0
    qwise = []
    for idx, q in enumerate(questions):
        score = per_question_score[idx]
        is_correct = score >= 50
        if is_correct:
            correct += 1
        else:
            wrong += 1
        qwise.append({
            "question_id": idx,
            "student_answer": answers.get(str(idx)),
            "correct_answer": q.get("answer") if q.get("type", "mcq") == "mcq" else q.get("modelAnswer"),
            "status": "correct" if is_correct else "wrong",
            "score": round(score),
            "feedback": per_question_feedback.get(idx, ""),
            "type": q.get("type", "mcq"),
            "topic": q.get("topic", ""),
        })

    total_marks = len(questions)
    scored = round(sum(per_question_score.values()) / 100, 2)
    percentage = round(sum(per_question_score.values()) / total_marks, 1)

    now = datetime.now(timezone.utc)
    roadmap_object_id = doc.get("roadmapId")
    week_range = doc.get("weekRange") or {}
    attempt_doc = {
        "student_id": student_id,
        "test_id": test_id,
        "mode": "roadmap",
        "roadmapId": str(roadmap_object_id) if roadmap_object_id else None,
        "weekRange": week_range,
        "testTitle": doc.get("testTitle") or doc.get("subjectName") or "Roadmap Test",
        "subjectName": doc.get("subjectName"),
        "answers": answers,
        "questionwise": qwise,
        "correct": correct, "wrong": wrong, "skipped": 0,
        "scored": scored, "totalMarks": total_marks,
        "accuracy": percentage, "percentage": percentage,
        "submitted_at": now, "created_at": now,
    }
    result = await db["testAttempts"].insert_one(attempt_doc)
    attempt_id = result.inserted_id

    await db["mockTests"].update_one(
        {"_id": test_id},
        {"$set": {"status": "submitted", "last_attempt": attempt_id, "updated_at": now}, "$inc": {"attempts_count": 1}},
    )

    # A passing attempt covering the FULL roadmap (week 1 -> the last week)
    # marks the roadmap complete.
    is_final_test = False
    if percentage >= 50 and roadmap_object_id:
        roadmap_doc = await db["selfLearnerRoadmaps"].find_one({"_id": roadmap_object_id})
        if roadmap_doc:
            last_week = max((w.get("week", 0) for w in roadmap_doc.get("weeks", [])), default=0)
            if week_range.get("start") == 1 and week_range.get("end") == last_week and last_week > 0:
                is_final_test = True
                await db["selfLearnerRoadmaps"].update_one(
                    {"_id": roadmap_object_id},
                    {"$set": {"progress.roadmapCompleted": True, "updated_at": now}},
                )

    return {
        "success": True,
        "attemptId": str(attempt_id),
        "result": {
            "scored": scored, "totalMarks": total_marks,
            "correct": correct, "wrong": wrong, "skipped": 0,
            "accuracy": percentage, "percentage": percentage,
            "submittedAt": now.isoformat(),
            "isFinalTest": is_final_test,
        },
    }


@router.get("/mock-tests/{test_id}/review")
async def review_test(
    test_id: str, identity: dict = Depends(get_current_identity), db: AsyncIOMotorDatabase = Depends(get_database)
):
    if not ObjectId.is_valid(test_id):
        raise HTTPException(status_code=400, detail="Invalid test_id")

    student_id = ObjectId(identity["user_id"])

    attempt = await db["testAttempts"].find_one(
        {"test_id": ObjectId(test_id), "student_id": student_id}, sort=[("submitted_at", -1)]
    )
    if not attempt:
        raise HTTPException(status_code=404, detail="No attempt found for this test")

    doc = await db["mockTests"].find_one({"_id": ObjectId(test_id), "student_id": student_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Mock test not found")

    if doc.get("mode") == "roadmap":
        return _review_roadmap_test(doc, attempt)

    questions = doc.get("questions", [])
    q_map = {str(q.get("_id")): q for q in questions}

    enriched_qwise = []
    for qw in attempt.get("questionwise", []):
        q = q_map.get(qw.get("question_id"), {})
        enriched_qwise.append({
            **qw,
            "questionText": q.get("questionText"),
            "options": q.get("options"),
            "explanation": q.get("explanation"),
            "type": q.get("type"),
            "marks": q.get("marks"),
            "difficulty": q.get("difficulty"),
        })

    attempt_out = _serialize_attempt(attempt)
    attempt_out["questionwise"] = enriched_qwise

    return {
        "success": True,
        "attempt": attempt_out,
        "questions": [serialize_question(q, include_answers=True) for q in questions],
        "testInfo": {
            "mode": "subject",
            "testTitle": doc.get("subjectName") or doc.get("topic") or "Mock Test",
            "subjectName": doc.get("subjectName"),
            "totalMarks": doc.get("totalMarks"),
        },
    }


def _review_roadmap_test(doc: Dict[str, Any], attempt: Dict[str, Any]) -> Dict[str, Any]:
    """Roadmap-mode counterpart of review_test — questions are identified by
    list index (question_id holds the int index, set in _submit_roadmap_test),
    not the _id-keyed scheme subject-mode uses."""
    questions = doc.get("questions", [])

    enriched_qwise = []
    for qw in attempt.get("questionwise", []):
        idx = qw.get("question_id")
        q = questions[idx] if isinstance(idx, int) and 0 <= idx < len(questions) else {}
        enriched_qwise.append({
            **qw,
            "question": q.get("question"),
            "options": q.get("options"),
        })

    attempt_out = _serialize_attempt(attempt)
    attempt_out["questionwise"] = enriched_qwise

    return {
        "success": True,
        "attempt": attempt_out,
        "questions": [serialize_roadmap_test_question(q, include_answers=True) for q in questions],
        "testInfo": {
            "mode": "roadmap",
            "testTitle": doc.get("testTitle") or doc.get("subjectName") or "Roadmap Test",
            "subjectName": doc.get("subjectName"),
            "weekRange": doc.get("weekRange"),
        },
    }
