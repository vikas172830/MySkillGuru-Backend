from datetime import datetime, timezone
from unittest.mock import patch

from bson import ObjectId

from app.models.roadmap import create_roadmap_document
from tests.test_security_fixes import _seed_and_login_user

_ATTEMPT_INSIGHT_PATCH = "app.api.routers.self_learner_analytics.generate_attempt_insight"


async def _learner_client(client_factory, test_db):
    return await _seed_and_login_user(test_db, client_factory, role=7, name="Analytics Learner")


async def _seed_roadmap(test_db, user_id: str, num_weeks: int = 1) -> str:
    weeks = [
        {"week": i, "title": f"Week {i}", "introDescription": "Intro.", "subtopics": [], "practiceQuestions": []}
        for i in range(1, num_weeks + 1)
    ]
    doc = create_roadmap_document(user_id, {
        "subject": "Python", "goal": "Interview Prep", "weeks": weeks,
        "unlockedWeeks": list(range(1, num_weeks + 1)),
    })
    result = await test_db["selfLearnerRoadmaps"].insert_one(doc)
    return str(result.inserted_id)


# ============================================================
# OVERVIEW — merges Practice Test / Roadmap Test / Weekly Quiz
# ============================================================

async def test_overview_requires_auth(client):
    resp = await client.get("/self-learner/analytics/overview")
    assert resp.status_code == 401


async def test_overview_empty_for_new_learner(client_factory, test_db):
    learner = await _learner_client(client_factory, test_db)
    resp = await learner.get("/self-learner/analytics/overview")
    assert resp.status_code == 200
    body = resp.json()
    assert body["attempts"] == []
    assert body["summary"]["testsAttempted"] == 0


async def test_overview_tags_practice_and_roadmap_sourcetype(client_factory, test_db):
    """testAttempts holds both subject-mode and roadmap-mode submissions,
    distinguished only by the "mode" field the submit endpoints write —
    confirms the overview endpoint still derives sourceType correctly after
    moving out of mock_tests.py."""
    learner = await _learner_client(client_factory, test_db)
    user_id = str((await test_db["users"].find_one({"role": 7}))["_id"])
    now = datetime.now(timezone.utc)

    await test_db["testAttempts"].insert_many([
        {
            "student_id": ObjectId(user_id), "test_id": ObjectId(), "mode": "roadmap",
            "testTitle": "Roadmap Test", "subjectName": "Python",
            "scored": 8, "totalMarks": 10, "percentage": 80, "correct": 4, "wrong": 1, "skipped": 0,
            "accuracy": 80, "submitted_at": now, "questionwise": [],
        },
        {
            "student_id": ObjectId(user_id), "test_id": ObjectId(),
            "testTitle": "Practice Test", "subjectName": "Math",
            "scored": 5, "totalMarks": 10, "percentage": 50, "correct": 5, "wrong": 5, "skipped": 0,
            "accuracy": 50, "submitted_at": now, "questionwise": [],
        },
    ])

    resp = await learner.get("/self-learner/analytics/overview")
    assert resp.status_code == 200
    source_types = {a["sourceType"] for a in resp.json()["attempts"]}
    assert source_types == {"roadmap_test", "practice_test"}


async def test_overview_includes_weekly_quiz_from_roadmap_history(client_factory, test_db):
    learner = await _learner_client(client_factory, test_db)
    user_id = str((await test_db["users"].find_one({"role": 7}))["_id"])
    roadmap_id = await _seed_roadmap(test_db, user_id)

    await test_db["selfLearnerRoadmaps"].update_one(
        {"_id": ObjectId(roadmap_id)},
        {"$set": {"progress.quizHistory": [{
            "week": 1, "score": 90, "passed": True, "correctCount": 9, "totalQuestions": 10,
            "submittedAt": datetime.now(timezone.utc), "questions": [],
        }]}},
    )

    resp = await learner.get("/self-learner/analytics/overview")
    assert resp.status_code == 200
    source_types = {a["sourceType"] for a in resp.json()["attempts"]}
    assert "weekly_quiz" in source_types


# ============================================================
# GET DETAILED FEEDBACK
# ============================================================

async def test_insight_requires_auth(client):
    resp = await client.post(f"/self-learner/analytics/attempts/practice_test/{ObjectId()}/insight")
    assert resp.status_code == 401


async def test_insight_rejects_invalid_source_type(client_factory, test_db):
    learner = await _learner_client(client_factory, test_db)
    resp = await learner.post(f"/self-learner/analytics/attempts/bogus_type/{ObjectId()}/insight")
    assert resp.status_code == 400


async def test_insight_practice_test_not_found(client_factory, test_db):
    learner = await _learner_client(client_factory, test_db)
    resp = await learner.post(f"/self-learner/analytics/attempts/practice_test/{ObjectId()}/insight")
    assert resp.status_code == 404


async def test_insight_roadmap_test_not_found(client_factory, test_db):
    learner = await _learner_client(client_factory, test_db)
    resp = await learner.post(f"/self-learner/analytics/attempts/roadmap_test/{ObjectId()}/insight")
    assert resp.status_code == 404


async def test_insight_generates_and_caches_for_roadmap_test(client_factory, test_db):
    """The main new capability this phase adds — roadmap_test previously
    fell through to the generic "Invalid sourceType" 400. Also verifies the
    MCQ answer-index -> option-text resolution in
    _items_from_roadmap_test_attempt (student picked index 1 = "4")."""
    learner = await _learner_client(client_factory, test_db)
    user_id = str((await test_db["users"].find_one({"role": 7}))["_id"])

    test_id = ObjectId()
    await test_db["mockTests"].insert_one({
        "_id": test_id, "student_id": ObjectId(user_id), "mode": "roadmap",
        "questions": [{"type": "mcq", "question": "2+2?", "options": ["3", "4"], "answer": 1}],
    })
    attempt_result = await test_db["testAttempts"].insert_one({
        "student_id": ObjectId(user_id), "test_id": test_id, "mode": "roadmap",
        "testTitle": "Roadmap Test", "subjectName": "Python",
        "scored": 1, "totalMarks": 1, "percentage": 100,
        "questionwise": [{"question_id": 0, "student_answer": 1, "correct_answer": 1, "status": "correct"}],
        "submitted_at": datetime.now(timezone.utc),
    })
    attempt_id = str(attempt_result.inserted_id)

    fake_insight = {"insights": [{"reasoning": "Correct.", "feedback": "Well done.", "improvement": "None needed."}]}
    with patch(_ATTEMPT_INSIGHT_PATCH, return_value=fake_insight) as mock_insight:
        resp = await learner.post(f"/self-learner/analytics/attempts/roadmap_test/{attempt_id}/insight")
    assert resp.status_code == 200
    body = resp.json()
    q0 = body["questions"][0]
    assert q0["question"] == "2+2?"
    assert q0["studentAnswer"] == "4"
    assert q0["correctAnswer"] == "4"
    assert q0["feedback"] == "Well done."
    assert mock_insight.call_count == 1

    # Second call served from cache — no re-generation.
    resp2 = await learner.post(f"/self-learner/analytics/attempts/roadmap_test/{attempt_id}/insight")
    assert resp2.status_code == 200
    assert mock_insight.call_count == 1


async def test_insight_generates_for_practice_test(client_factory, test_db):
    """Smoke test that the practice_test branch still works after the move
    out of mock_tests.py (unchanged logic, moved verbatim)."""
    learner = await _learner_client(client_factory, test_db)
    user_id = str((await test_db["users"].find_one({"role": 7}))["_id"])

    test_id = ObjectId()
    await test_db["mockTests"].insert_one({
        "_id": test_id, "student_id": ObjectId(user_id),
        "questions": [{"_id": "q1", "questionText": "What is 2+2?", "options": ["3", "4"], "correct_answer": "4"}],
    })
    attempt_result = await test_db["testAttempts"].insert_one({
        "student_id": ObjectId(user_id), "test_id": test_id,
        "testTitle": "Practice Test", "subjectName": "Math",
        "scored": 1, "totalMarks": 1, "percentage": 100,
        "questionwise": [{"question_id": "q1", "student_answer": "4", "correct_answer": "4", "status": "correct"}],
        "submitted_at": datetime.now(timezone.utc),
    })
    attempt_id = str(attempt_result.inserted_id)

    fake_insight = {"insights": [{"reasoning": "Correct.", "feedback": "Nice.", "improvement": "—"}]}
    with patch(_ATTEMPT_INSIGHT_PATCH, return_value=fake_insight):
        resp = await learner.post(f"/self-learner/analytics/attempts/practice_test/{attempt_id}/insight")
    assert resp.status_code == 200
    assert resp.json()["questions"][0]["question"] == "What is 2+2?"


# ============================================================
# AI USAGE
# ============================================================

async def test_ai_usage_requires_auth(client):
    resp = await client.get("/self-learner/analytics/ai-usage")
    assert resp.status_code == 401


async def test_ai_usage_empty_for_new_learner(client_factory, test_db):
    learner = await _learner_client(client_factory, test_db)
    resp = await learner.get("/self-learner/analytics/ai-usage")
    assert resp.status_code == 200
    body = resp.json()
    assert body["byFeature"] == []
    assert body["totals"] == {
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_usd": 0, "call_count": 0,
    }


async def test_ai_usage_aggregates_by_feature_for_requesting_user_only(client_factory, test_db):
    learner = await _learner_client(client_factory, test_db)
    user_id = str((await test_db["users"].find_one({"role": 7}))["_id"])
    other_user_id = ObjectId()

    await test_db["aiUsageEvents"].insert_many([
        {
            "user_id": ObjectId(user_id), "tenant_type": "individual", "institute_id": None,
            "school_id": None, "programme_id": None, "provider": "claude", "model": "claude-sonnet-4-5",
            "feature": "roadmap_curriculum", "input_tokens": 100, "output_tokens": 50, "total_tokens": 150,
            "cost_usd": 0.001, "grounded": None, "context_id": None, "job_id": None,
            "created_at": datetime.now(timezone.utc),
        },
        {
            "user_id": ObjectId(user_id), "tenant_type": "individual", "institute_id": None,
            "school_id": None, "programme_id": None, "provider": "claude", "model": "claude-sonnet-4-5",
            "feature": "roadmap_curriculum", "input_tokens": 40, "output_tokens": 10, "total_tokens": 50,
            "cost_usd": 0.0005, "grounded": None, "context_id": None, "job_id": None,
            "created_at": datetime.now(timezone.utc),
        },
        {
            "user_id": ObjectId(user_id), "tenant_type": "individual", "institute_id": None,
            "school_id": None, "programme_id": None, "provider": "gemini", "model": "gemini-2.5-flash",
            "feature": "roadmap_notes", "input_tokens": 20, "output_tokens": 5, "total_tokens": 25,
            "cost_usd": 0.0001, "grounded": None, "context_id": None, "job_id": None,
            "created_at": datetime.now(timezone.utc),
        },
        # Belongs to a different user entirely — must never leak into this learner's totals.
        {
            "user_id": other_user_id, "tenant_type": "individual", "institute_id": None,
            "school_id": None, "programme_id": None, "provider": "claude", "model": "claude-sonnet-4-5",
            "feature": "roadmap_curriculum", "input_tokens": 999, "output_tokens": 999, "total_tokens": 1998,
            "cost_usd": 99.0, "grounded": None, "context_id": None, "job_id": None,
            "created_at": datetime.now(timezone.utc),
        },
    ])

    resp = await learner.get("/self-learner/analytics/ai-usage")
    assert resp.status_code == 200
    body = resp.json()

    by_feature = {row["feature"]: row for row in body["byFeature"]}
    assert by_feature["roadmap_curriculum"]["total_tokens"] == 200
    assert by_feature["roadmap_curriculum"]["call_count"] == 2
    assert by_feature["roadmap_notes"]["total_tokens"] == 25

    assert body["totals"]["total_tokens"] == 225
    assert body["totals"]["call_count"] == 3
