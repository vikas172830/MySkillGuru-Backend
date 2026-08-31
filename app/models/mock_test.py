from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from bson import ObjectId

ALLOWED_QUESTION_TYPES = {"mcq", "true_false", "short_answer", "descriptive", "fill_blanks", "match_following"}
ALLOWED_DIFFICULTIES = {"easy", "medium", "hard", "mixed"}

# Fields returned to a student *taking* the test — correct_answer/explanation
# are deliberately excluded (see Phase 5a plan: Flask's GET /mock-tests/{id}
# leaked answers to any caller who inspected the response before submitting).
_TAKE_TEST_QUESTION_FIELDS = ("_id", "type", "questionText", "options", "marks", "difficulty")


def _validate_question_types(raw: Any) -> List[str]:
    if not raw:
        return ["mcq"]
    if isinstance(raw, str):
        raw = [raw]
    cleaned = [t for t in raw if t in ALLOWED_QUESTION_TYPES]
    return cleaned or ["mcq"]


def _parse_schedule_date(value: Optional[str]):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def build_create_document(data: Dict[str, Any], student_id: ObjectId) -> Dict[str, Any]:
    subject_id = data.get("subject_id") or data.get("subjectId")
    subject_name = data.get("subjectName")
    if not subject_id and not subject_name:
        raise ValueError("Either subject_id or subjectName is required")

    question_count = max(1, int(data.get("questionCount", 10) or 10))
    marks_per_question = max(0.5, float(data.get("marksPerQuestion", 1) or 1))
    duration = max(5, int(data.get("duration", 30) or 30))

    difficulty = data.get("difficulty", "mixed")
    if difficulty not in ALLOWED_DIFFICULTIES:
        difficulty = "mixed"

    negative_marking = bool(data.get("negativeMarking", False))
    negative_marks = float(data.get("negativeMarks", 0.25 if negative_marking else 0) or 0) if negative_marking else 0

    now = datetime.now(timezone.utc)

    return {
        "student_id": student_id,
        "subject_id": ObjectId(subject_id) if subject_id and ObjectId.is_valid(str(subject_id)) else None,
        "subjectName": subject_name,
        "topic": data.get("topic"),
        "instructions": data.get("instructions"),
        "questionCount": question_count,
        "marksPerQuestion": marks_per_question,
        "totalMarks": round(question_count * marks_per_question, 2),
        "duration": duration,
        "difficulty": difficulty,
        "negativeMarking": negative_marking,
        "negativeMarks": negative_marks,
        "questionTypes": _validate_question_types(data.get("questionTypes")),
        "scheduleDate": _parse_schedule_date(data.get("scheduleDate")),
        "status": "pending",
        "questions": [],
        "attempts_count": 0,
        "last_attempt": None,
        "created_at": now,
        "updated_at": now,
    }


def serialize_question(q: Dict[str, Any], include_answers: bool) -> Dict[str, Any]:
    q = dict(q)
    if "_id" in q and not isinstance(q["_id"], str):
        q["_id"] = str(q["_id"])
    if include_answers:
        return q
    return {k: q.get(k) for k in _TAKE_TEST_QUESTION_FIELDS if k in q}


def serialize_mock_test(doc: Dict[str, Any], include_answers: bool = False) -> Optional[Dict[str, Any]]:
    if not doc:
        return None

    if doc.get("mode") == "roadmap":
        return serialize_roadmap_test(doc, include_answers)

    out = {
        "_id": str(doc["_id"]),
        "mode": "subject",
        "student_id": str(doc.get("student_id")) if doc.get("student_id") else None,
        "subject_id": str(doc.get("subject_id")) if doc.get("subject_id") else None,
        "subjectName": doc.get("subjectName"),
        "topic": doc.get("topic"),
        "instructions": doc.get("instructions"),
        "questionCount": doc.get("questionCount"),
        "marksPerQuestion": doc.get("marksPerQuestion"),
        "totalMarks": doc.get("totalMarks"),
        "duration": doc.get("duration"),
        "difficulty": doc.get("difficulty"),
        "negativeMarking": doc.get("negativeMarking", False),
        "negativeMarks": doc.get("negativeMarks", 0),
        "questionTypes": doc.get("questionTypes", []),
        "status": doc.get("status"),
        "generationError": doc.get("generationError"),
        "attempts_count": doc.get("attempts_count", 0),
        "created_at": doc.get("created_at").isoformat() if doc.get("created_at") else None,
        "updated_at": doc.get("updated_at").isoformat() if doc.get("updated_at") else None,
    }

    if "questions" in doc:
        out["questions"] = [serialize_question(q, include_answers) for q in (doc.get("questions") or [])]

    return out


# ============================================================
# ROADMAP MODE — week-range testing, reusing the Auto Test config/question
# shape (question/options/answer-index/modelAnswer/type/topic) directly
# rather than subject-mode's shape (questionText/correct_answer/marks) —
# the two are genuinely different schemas, not just a naming difference, so
# every function here has its own roadmap-mode counterpart rather than
# trying to force one function to branch internally on field names.
# ============================================================

# Never send answer/modelAnswer/explanation to a student before submission
# — same exclude-list as roadmap.py's _strip_quiz_answers, kept in sync
# deliberately since both serve the exact same question shape.
_ROADMAP_TEST_ANSWER_KEYS = ("answer", "modelAnswer", "explanation")


def build_roadmap_test_document(
    student_id: ObjectId, roadmap_id: ObjectId, subject_name: str,
    week_start: int, week_end: int, config: Dict[str, Any],
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    question_count = max(1, int(config.get("questionCount", 10) or 10))

    return {
        "student_id": student_id,
        "mode": "roadmap",
        "roadmapId": roadmap_id,
        "weekRange": {"start": week_start, "end": week_end},
        "subjectName": subject_name,
        "testTitle": f"{subject_name}: Week {week_start}" + (f"-{week_end}" if week_end != week_start else ""),
        "config": {
            "mcqPercent": config.get("mcqPercent", 100),
            "subjectivePercent": config.get("subjectivePercent", 0),
            "practicalPercent": config.get("practicalPercent", 0),
            "questionCount": question_count,
            "customPrompt": config.get("customPrompt"),
        },
        "questionCount": question_count,
        "status": "pending",
        "questions": [],
        "attempts_count": 0,
        "last_attempt": None,
        "created_at": now,
        "updated_at": now,
    }


def serialize_roadmap_test_question(q: Dict[str, Any], include_answers: bool) -> Dict[str, Any]:
    if include_answers:
        return dict(q)
    return {k: v for k, v in q.items() if k not in _ROADMAP_TEST_ANSWER_KEYS}


def serialize_roadmap_test(doc: Dict[str, Any], include_answers: bool = False) -> Optional[Dict[str, Any]]:
    if not doc:
        return None

    out = {
        "_id": str(doc["_id"]),
        "mode": "roadmap",
        "student_id": str(doc.get("student_id")) if doc.get("student_id") else None,
        "roadmapId": str(doc.get("roadmapId")) if doc.get("roadmapId") else None,
        "weekRange": doc.get("weekRange"),
        "subjectName": doc.get("subjectName"),
        "testTitle": doc.get("testTitle"),
        "config": doc.get("config", {}),
        "questionCount": doc.get("questionCount"),
        "status": doc.get("status"),
        "generationError": doc.get("generationError"),
        "attempts_count": doc.get("attempts_count", 0),
        "created_at": doc.get("created_at").isoformat() if doc.get("created_at") else None,
        "updated_at": doc.get("updated_at").isoformat() if doc.get("updated_at") else None,
    }

    if "questions" in doc:
        out["questions"] = [serialize_roadmap_test_question(q, include_answers) for q in (doc.get("questions") or [])]

    return out
