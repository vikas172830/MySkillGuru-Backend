# ============================================================
# SELF-LEARNER ANALYTICS ROUTER
# Extracted from mock_tests.py (which originally carried this logic as
# GET /mock-tests/analytics + POST /mock-tests/attempts/.../insight) into
# its own module mounted at /self-learner/analytics/*, matching this
# project's convention of giving each self-learner feature area its own
# router rather than bundling unrelated concerns into Test Engine's router.
#
# Merges three score sources into one attempts[] list:
#   - testAttempts, sourceType "practice_test" (subject-mode Test Engine)
#   - testAttempts, sourceType "roadmap_test" (roadmap-mode Test Engine,
#     tagged via mockTests.mode / the attempt doc's own "mode" field)
#   - selfLearnerRoadmaps.progress.quizHistory[], sourceType "weekly_quiz"
#     (embedded per-roadmap, not its own collection — given a synthetic
#     base64url attempt id since it has no real _id of its own)
#
# "Get Detailed Feedback" (the /insight endpoint) generates or returns
# cached per-question reasoning/feedback/improvement for one attempt via
# one batched Claude call — same caching philosophy as roadmap notes.
# ============================================================

import base64
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.api.deps import get_current_identity, require_myskillguru_access
from app.core.rate_limit import ai_rate_limit
from app.db.mongodb import get_database
from app.services.attempt_insight import generate_attempt_insight

router = APIRouter(
    prefix="/self-learner/analytics",
    dependencies=[Depends(get_current_identity), Depends(require_myskillguru_access)],
    tags=["self-learner-analytics"],
)


def _quiz_attempt_id(roadmap_id: Any, week: Any, submitted_at: datetime) -> str:
    # base64url instead of a raw "quiz:<id>:<week>:<iso timestamp>" string —
    # the timestamp itself contains colons, and those get mangled somewhere
    # in the browser -> Next.js rewrite -> backend round trip (colons are
    # exactly the kind of "special but not always re-escaped" character that
    # breaks across that many encode/decode layers). base64url's alphabet
    # (A-Za-z0-9-_) needs zero percent-encoding anywhere in that chain.
    raw = f"{roadmap_id}:{week}:{submitted_at.isoformat()}"
    token = base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")
    return f"q_{token}"


def _parse_quiz_attempt_id(attempt_id: str) -> Optional[Tuple[str, int, datetime]]:
    if not attempt_id.startswith("q_"):
        return None
    token = attempt_id[2:]
    padded = token + "=" * (-len(token) % 4)
    try:
        raw = base64.urlsafe_b64decode(padded.encode()).decode()
    except Exception:
        return None
    parts = raw.split(":", 2)
    if len(parts) != 3:
        return None
    roadmap_id, week_str, ts = parts
    try:
        week = int(week_str)
        submitted_at = datetime.fromisoformat(ts)
    except ValueError:
        return None
    if not ObjectId.is_valid(roadmap_id):
        return None
    return roadmap_id, week, submitted_at


def _date_filter(time_range: str) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    if time_range == "week":
        return {"submitted_at": {"$gte": now - timedelta(days=7)}}
    if time_range == "month":
        return {"submitted_at": {"$gte": now - timedelta(days=30)}}
    return {}


def _safe_pct(num: float, den: float) -> float:
    if not den:
        return 0
    return round((num / den) * 100, 1)


@router.get("/overview")
async def get_analytics_overview(
    range: str = "all",
    identity: dict = Depends(get_current_identity),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    student_id = ObjectId(identity["user_id"])
    date_filter = _date_filter(range)
    base_filter: Dict[str, Any] = {"student_id": student_id, **date_filter}

    # testAttempts (Practice Test + Roadmap Test) — kept as raw DB docs
    # since topic/difficulty/question-type breakdowns below need the real
    # test_id join; weekly-quiz entries have no equivalent per-question
    # tagging to offer there.
    attempts = await db["testAttempts"].find(base_filter).sort("submitted_at", -1).to_list(length=None)

    # Weekly Quiz history — lives embedded per-roadmap
    # (selfLearnerRoadmaps.progress.quizHistory[]), not in its own collection.
    cutoff = date_filter.get("submitted_at", {}).get("$gte")
    roadmap_docs = await db["selfLearnerRoadmaps"].find(
        {"user_id": student_id}, {"subject": 1, "progress.quizHistory": 1},
    ).to_list(length=None)

    quiz_normalized: List[Dict[str, Any]] = []
    for rd in roadmap_docs:
        subject = rd.get("subject", "Roadmap")
        for entry in rd.get("progress", {}).get("quizHistory", []):
            submitted_at = entry.get("submittedAt")
            if not submitted_at or (cutoff and submitted_at < cutoff):
                continue
            week = entry.get("week")
            total_q = entry.get("totalQuestions", 0)
            correct_q = entry.get("correctCount", 0)
            quiz_normalized.append({
                "id": _quiz_attempt_id(rd["_id"], week, submitted_at),
                "test_id": "",
                "sourceType": "weekly_quiz",
                "testTitle": f"Week {week}: {subject}",
                "subjectName": subject,
                "scored": correct_q,
                "totalMarks": total_q,
                "percentage": entry.get("score", 0),
                "correct": correct_q,
                "wrong": max(0, total_q - correct_q),
                "skipped": 0,
                "accuracy": _safe_pct(correct_q, total_q),
                "submitted_at": submitted_at,
                "hasInsight": bool(entry.get("aiInsight")),
            })

    test_normalized: List[Dict[str, Any]] = [
        {
            "id": str(a["_id"]),
            "test_id": str(a.get("test_id", "")),
            # testAttempts holds both Practice Test (subject-mode) and
            # Roadmap Test (roadmap-mode) submissions — _submit_roadmap_test
            # tags its own attempts with mode: "roadmap" directly, so this
            # is a same-collection distinction, not a separate query.
            "sourceType": "roadmap_test" if a.get("mode") == "roadmap" else "practice_test",
            "testTitle": a.get("testTitle", ""),
            "subjectName": a.get("subjectName", ""),
            "scored": a.get("scored", 0),
            "totalMarks": a.get("totalMarks", 0),
            "percentage": a.get("percentage", 0),
            "correct": a.get("correct", 0),
            "wrong": a.get("wrong", 0),
            "skipped": a.get("skipped", 0),
            "accuracy": a.get("accuracy", 0),
            "submitted_at": a.get("submitted_at"),
            "hasInsight": bool(a.get("aiInsight")),
        }
        for a in attempts
    ]

    all_attempts = sorted(
        test_normalized + quiz_normalized,
        key=lambda a: a["submitted_at"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True,
    )

    if not all_attempts:
        return {
            "success": True,
            "summary": {
                "testsAttempted": 0, "totalQuestions": 0, "avgScore": 0,
                "bestScore": 0, "avgAccuracy": 0, "totalTimeMins": 0,
            },
            "attempts": [],
            "subjectPerformance": [],
            "topicPerformance": [],
            "difficultyBreakdown": {
                "easy": {"correct": 0, "total": 0},
                "medium": {"correct": 0, "total": 0},
                "hard": {"correct": 0, "total": 0},
            },
            "scoreTrend": [],
            "questionTypeBreakdown": [],
            "strengths": [],
            "improvements": [],
            "improvementPct": None,
        }

    # ── Summary ──────────────────────────────────────────
    tests_attempted = len(all_attempts)
    total_questions = sum(a["correct"] + a["wrong"] + a["skipped"] for a in all_attempts)
    avg_score = round(sum(a["percentage"] for a in all_attempts) / tests_attempted, 1)
    best_score = round(max(a["percentage"] for a in all_attempts), 1)
    avg_accuracy = round(sum(a["accuracy"] for a in all_attempts) / tests_attempted, 1)

    summary = {
        "testsAttempted": tests_attempted,
        "totalQuestions": total_questions,
        "avgScore": avg_score,
        "bestScore": best_score,
        "avgAccuracy": avg_accuracy,
        "totalTimeMins": 0,
    }

    # ── Improvement ──────────────────────────────────────
    improvement_pct = None
    if tests_attempted >= 4:
        mid = tests_attempted // 2
        first = [a["percentage"] for a in all_attempts[mid:]]
        second = [a["percentage"] for a in all_attempts[:mid]]
        avg_first = sum(first) / len(first)
        avg_second = sum(second) / len(second)
        improvement_pct = round(avg_second - avg_first, 1)

    # ── Attempt list ─────────────────────────────────────
    # Includes correct/wrong/skipped/accuracy (not just scored/totalMarks/
    # percentage) so the frontend's source/subject filter row can re-derive
    # summary stats, score trend, and subject performance client-side from
    # this same filtered list — otherwise switching the filter changes the
    # attempts table but leaves the trend graph and stat tiles showing the
    # unfiltered (and after enough activity, effectively stale-looking)
    # all-sources numbers, which don't match what's selected above them.
    attempt_list = [
        {
            "_id": a["id"],
            "sourceType": a["sourceType"],
            "test_id": a["test_id"],
            "testTitle": a["testTitle"],
            "subjectName": a["subjectName"],
            "scored": a["scored"],
            "totalMarks": a["totalMarks"],
            "percentage": a["percentage"],
            "correct": a["correct"],
            "wrong": a["wrong"],
            "skipped": a["skipped"],
            "accuracy": a["accuracy"],
            "date": a["submitted_at"].isoformat() if a["submitted_at"] else None,
            "hasInsight": a["hasInsight"],
        }
        for a in all_attempts
    ]

    # ── Subject performance ───────────────────────────────
    subject_map: Dict[str, Dict[str, float]] = {}
    for a in all_attempts:
        subj = a["subjectName"] or "Unknown"
        entry = subject_map.setdefault(subj, {"scored": 0.0, "total": 0.0})
        entry["scored"] += a["scored"]
        entry["total"] += a["totalMarks"]

    subject_perf = [{"subject": k, "scored": v["scored"], "total": v["total"]} for k, v in subject_map.items()]

    # ── Topic + difficulty + question type ────────────────
    test_ids = [a["test_id"] for a in attempts if a.get("test_id")]
    test_docs: Dict[str, Dict[str, Any]] = {}
    if test_ids:
        cursor = db["mockTests"].find({"_id": {"$in": test_ids}}, {"topic": 1, "difficulty": 1})
        async for t in cursor:
            test_docs[str(t["_id"])] = t

    topic_map: Dict[str, Dict[str, int]] = {}
    diff_map = {
        "easy": {"correct": 0, "total": 0},
        "medium": {"correct": 0, "total": 0},
        "hard": {"correct": 0, "total": 0},
    }
    qtype_map: Dict[str, Dict[str, int]] = {}

    for a in attempts:
        tid = str(a.get("test_id", ""))
        tdoc = test_docs.get(tid, {})
        topic = (tdoc.get("topic") or "").strip() or "General"
        diff = tdoc.get("difficulty", "mixed")

        topic_entry = topic_map.setdefault(topic, {"correct": 0, "total": 0})
        topic_entry["correct"] += a.get("correct", 0)
        topic_entry["total"] += a.get("correct", 0) + a.get("wrong", 0)

        if diff in diff_map:
            diff_map[diff]["correct"] += a.get("correct", 0)
            diff_map[diff]["total"] += a.get("correct", 0) + a.get("wrong", 0)

        for qw in a.get("questionwise", []):
            qtype = qw.get("type", "mcq")
            qtype_entry = qtype_map.setdefault(qtype, {"correct": 0, "total": 0})
            qtype_entry["total"] += 1
            if qw.get("status") == "correct":
                qtype_entry["correct"] += 1

    topic_perf = [
        {"topic": k, "correct": v["correct"], "total": v["total"]}
        for k, v in topic_map.items()
        if v["total"] > 0
    ]
    qtype_breakdown = [{"type": k, "correct": v["correct"], "total": v["total"]} for k, v in qtype_map.items()]

    # ── Score trend ───────────────────────────────────────
    trend_data = [
        {
            "label": a["submitted_at"].strftime("%d %b") if a["submitted_at"] else f"T{i + 1}",
            "score": a["percentage"],
        }
        for i, a in enumerate(reversed(all_attempts[:10]))
    ]

    # ── Strengths & Improvements ─────────────────────────
    strengths: List[Dict[str, str]] = []
    improvements: List[Dict[str, str]] = []

    for subj, v in subject_map.items():
        p = _safe_pct(v["scored"], v["total"])
        entry = {"label": subj, "detail": f"{p}% avg score"}
        if p >= 75:
            strengths.append(entry)
        elif p < 50:
            improvements.append(entry)

    for topic, v in topic_map.items():
        if not v["total"]:
            continue
        p = _safe_pct(v["correct"], v["total"])
        entry = {"label": topic, "detail": f"{v['correct']}/{v['total']} correct"}
        if p >= 75:
            strengths.append(entry)
        elif p < 50 and topic != "General":
            improvements.append(entry)

    return {
        "success": True,
        "summary": summary,
        "attempts": attempt_list,
        "subjectPerformance": subject_perf,
        "topicPerformance": topic_perf,
        "difficultyBreakdown": diff_map,
        "scoreTrend": trend_data,
        "questionTypeBreakdown": qtype_breakdown,
        "strengths": strengths[:5],
        "improvements": improvements[:5],
        "improvementPct": improvement_pct,
    }


# ============================================================
# GET DETAILED FEEDBACK — per-question reasoning/feedback/improvement
# ============================================================

def _items_from_test_attempt(attempt: Dict[str, Any], test_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """questionwise only has question_id + answers; question text/options
    live on the mockTests doc — same join /mock-tests/{id}/review already does."""
    q_map = {str(q.get("_id")): q for q in test_doc.get("questions", []) if q.get("_id") is not None}
    items = []
    for qw in attempt.get("questionwise", []):
        q = q_map.get(str(qw.get("question_id", "")), {})
        items.append({
            "question": q.get("questionText") or q.get("text", ""),
            "options": q.get("options", []),
            "studentAnswer": qw.get("student_answer"),
            "correctAnswer": qw.get("correct_answer", ""),
            "isCorrect": qw.get("status") == "correct",
        })
    return items


def _items_from_roadmap_test_attempt(attempt: Dict[str, Any], test_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Roadmap-mode counterpart of _items_from_test_attempt — questions are
    identified by list index (question_id holds the int index set in
    _submit_roadmap_test), not an _id, and use Auto Test's field names
    (question/options) rather than subject-mode's (questionText). MCQ
    answers are stored as option indices — resolved to their text here so
    the report page always renders a readable answer, not a bare number."""
    questions = test_doc.get("questions", [])
    items = []
    for qw in attempt.get("questionwise", []):
        idx = qw.get("question_id")
        q = questions[idx] if isinstance(idx, int) and 0 <= idx < len(questions) else {}
        options = q.get("options") or []

        student_answer = qw.get("student_answer")
        if isinstance(student_answer, int) and 0 <= student_answer < len(options):
            student_answer = options[student_answer]

        correct_answer = qw.get("correct_answer", "")
        if isinstance(correct_answer, int) and 0 <= correct_answer < len(options):
            correct_answer = options[correct_answer]

        items.append({
            "question": q.get("question", ""),
            "options": options,
            "studentAnswer": student_answer,
            "correctAnswer": correct_answer,
            "isCorrect": qw.get("status") == "correct",
        })
    return items


def _items_from_quiz_history(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """entry["questions"] shape comes from roadmap.py's _sanitize_quiz_for_history — already self-contained."""
    return [
        {
            "question": q.get("question", ""),
            "options": q.get("options", []),
            "studentAnswer": q.get("yourAnswer"),
            "correctAnswer": q.get("correctAnswer", ""),
            "isCorrect": q.get("isCorrect", False),
        }
        for q in entry.get("questions", [])
    ]


@router.post("/attempts/{source_type}/{attempt_id}/insight", dependencies=[Depends(ai_rate_limit)])
async def get_attempt_insight(
    source_type: str,
    attempt_id: str,
    identity: dict = Depends(get_current_identity),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """Generates (or returns the cached) per-question reasoning/feedback/
    improvement for one attempt. source_type is "practice_test",
    "roadmap_test", or "weekly_quiz" (see _quiz_attempt_id for that id's
    synthetic format)."""
    student_id = ObjectId(identity["user_id"])

    if source_type == "practice_test":
        if not ObjectId.is_valid(attempt_id):
            raise HTTPException(status_code=400, detail="Invalid attempt id")
        attempt = await db["testAttempts"].find_one({"_id": ObjectId(attempt_id), "student_id": student_id})
        if not attempt:
            raise HTTPException(status_code=404, detail="Attempt not found")

        test_doc = await db["mockTests"].find_one({"_id": attempt.get("test_id")}) or {}
        items = _items_from_test_attempt(attempt, test_doc)
        if not items:
            raise HTTPException(status_code=400, detail="No questions found for this attempt")

        cached = attempt.get("aiInsight")
        if not cached:
            result = await generate_attempt_insight(items, db=db, user_id=identity["user_id"])
            cached = result.get("insights", [])
            await db["testAttempts"].update_one({"_id": attempt["_id"]}, {"$set": {"aiInsight": cached}})

        questions = [{**item, **(cached[i] if i < len(cached) else {})} for i, item in enumerate(items)]
        return {
            "success": True,
            "attempt": {
                "testTitle": attempt.get("testTitle", ""),
                "subjectName": attempt.get("subjectName", ""),
                "percentage": attempt.get("percentage", 0),
                "scored": attempt.get("scored", 0),
                "totalMarks": attempt.get("totalMarks", 0),
                "date": attempt["submitted_at"].isoformat() if attempt.get("submitted_at") else None,
            },
            "questions": questions,
        }

    if source_type == "roadmap_test":
        if not ObjectId.is_valid(attempt_id):
            raise HTTPException(status_code=400, detail="Invalid attempt id")
        attempt = await db["testAttempts"].find_one(
            {"_id": ObjectId(attempt_id), "student_id": student_id, "mode": "roadmap"}
        )
        if not attempt:
            raise HTTPException(status_code=404, detail="Attempt not found")

        test_doc = await db["mockTests"].find_one({"_id": attempt.get("test_id")}) or {}
        items = _items_from_roadmap_test_attempt(attempt, test_doc)
        if not items:
            raise HTTPException(status_code=400, detail="No questions found for this attempt")

        cached = attempt.get("aiInsight")
        if not cached:
            result = await generate_attempt_insight(items, db=db, user_id=identity["user_id"])
            cached = result.get("insights", [])
            await db["testAttempts"].update_one({"_id": attempt["_id"]}, {"$set": {"aiInsight": cached}})

        questions = [{**item, **(cached[i] if i < len(cached) else {})} for i, item in enumerate(items)]
        return {
            "success": True,
            "attempt": {
                "testTitle": attempt.get("testTitle", ""),
                "subjectName": attempt.get("subjectName", ""),
                "percentage": attempt.get("percentage", 0),
                "scored": attempt.get("scored", 0),
                "totalMarks": attempt.get("totalMarks", 0),
                "date": attempt["submitted_at"].isoformat() if attempt.get("submitted_at") else None,
            },
            "questions": questions,
        }

    if source_type == "weekly_quiz":
        parsed = _parse_quiz_attempt_id(attempt_id)
        if not parsed:
            raise HTTPException(status_code=400, detail="Invalid attempt id")
        roadmap_id, week, submitted_at = parsed

        roadmap_doc = await db["selfLearnerRoadmaps"].find_one({"_id": ObjectId(roadmap_id), "user_id": student_id})
        if not roadmap_doc:
            raise HTTPException(status_code=404, detail="Roadmap not found")

        history = roadmap_doc.get("progress", {}).get("quizHistory", [])
        entry = next((h for h in history if h.get("week") == week and h.get("submittedAt") == submitted_at), None)
        if not entry:
            raise HTTPException(status_code=404, detail="Quiz attempt not found")

        items = _items_from_quiz_history(entry)
        if not items:
            raise HTTPException(status_code=400, detail="No questions found for this attempt")

        cached = entry.get("aiInsight")
        if not cached:
            result = await generate_attempt_insight(items, db=db, user_id=identity["user_id"])
            cached = result.get("insights", [])
            await db["selfLearnerRoadmaps"].update_one(
                {"_id": roadmap_doc["_id"], "user_id": student_id},
                {"$set": {"progress.quizHistory.$[entry].aiInsight": cached}},
                array_filters=[{"entry.week": week, "entry.submittedAt": submitted_at}],
            )

        questions = [{**item, **(cached[i] if i < len(cached) else {})} for i, item in enumerate(items)]
        return {
            "success": True,
            "attempt": {
                "testTitle": f"Week {week}: {roadmap_doc.get('subject', 'Roadmap')}",
                "subjectName": roadmap_doc.get("subject", ""),
                "percentage": entry.get("score", 0),
                "scored": entry.get("correctCount", 0),
                "totalMarks": entry.get("totalQuestions", 0),
                "date": submitted_at.isoformat(),
            },
            "questions": questions,
        }

    raise HTTPException(status_code=400, detail="Invalid sourceType")


# ============================================================
# AI USAGE — the student's own spend across every AI feature they've used
# (roadmap generation, self-review, Test Engine, RAG-grounded material),
# sourced from the aiUsageEvents ledger app.services.ai_usage.record_ai_usage
# writes on every tracked call. See app/models/ai_usage_event.py for the
# feature-tag vocabulary this groups by.
# ============================================================

@router.get("/ai-usage")
async def get_my_ai_usage(
    identity: dict = Depends(get_current_identity),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    user_id = ObjectId(identity["user_id"])

    pipeline = [
        {"$match": {"user_id": user_id}},
        {"$group": {
            "_id": "$feature",
            "input_tokens": {"$sum": "$input_tokens"},
            "output_tokens": {"$sum": "$output_tokens"},
            "total_tokens": {"$sum": "$total_tokens"},
            "cost_usd": {"$sum": "$cost_usd"},
            "call_count": {"$sum": 1},
        }},
        {"$sort": {"total_tokens": -1}},
    ]

    by_feature = []
    async for row in db["aiUsageEvents"].aggregate(pipeline):
        by_feature.append({
            "feature": row["_id"],
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
            "total_tokens": row["total_tokens"],
            "cost_usd": round(row["cost_usd"], 4),
            "call_count": row["call_count"],
        })

    totals = {
        "input_tokens": sum(r["input_tokens"] for r in by_feature),
        "output_tokens": sum(r["output_tokens"] for r in by_feature),
        "total_tokens": sum(r["total_tokens"] for r in by_feature),
        "cost_usd": round(sum(r["cost_usd"] for r in by_feature), 4),
        "call_count": sum(r["call_count"] for r in by_feature),
    }

    return {"success": True, "byFeature": by_feature, "totals": totals}
