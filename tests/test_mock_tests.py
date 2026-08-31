from unittest.mock import patch

from bson import ObjectId

from app.models.roadmap import create_roadmap_document
from tests.test_security_fixes import _seed_and_login_user

# The router's create-test endpoint fires a real AI-generation background
# task synchronously within the request/response cycle under httpx's
# ASGITransport (no ASGI lifespan means nothing schedules it separately) —
# tests must not depend on a live Claude/Gemini call, so it's patched out
# wherever a test creates a mock test.
_PATCH_TARGET = "app.api.routers.mock_tests.generate_mock_test_questions"
_GEMINI_JSON_PATCH = "app.api.routers.mock_tests.generate_gemini_json"


async def _learner_client(client_factory, test_db):
    return await _seed_and_login_user(test_db, client_factory, role=7, name="Mock Test Learner")


async def _create_test(learner, **payload):
    with patch(_PATCH_TARGET, return_value=([], {"input_tokens": 0, "output_tokens": 0})):
        return await learner.post("/mock-tests", json=payload)


async def _seed_roadmap(test_db, user_id: str, num_weeks: int = 2) -> str:
    """Two unlocked weeks by default — enough to test both a partial-range
    and a full-range (1 -> last week) roadmap test."""
    weeks = [
        {
            "week": i, "title": f"Week {i} Title", "introDescription": "Intro.",
            "subtopics": [{"title": f"W{i} Subtopic A"}, {"title": f"W{i} Subtopic B"}],
            "practiceQuestions": [],
        }
        for i in range(1, num_weeks + 1)
    ]
    doc = create_roadmap_document(user_id, {
        "subject": "Python", "goal": "Interview Prep", "weeks": weeks,
        "unlockedWeeks": list(range(1, num_weeks + 1)),
    })
    result = await test_db["selfLearnerRoadmaps"].insert_one(doc)
    return str(result.inserted_id)


async def test_create_mock_test_requires_auth(client):
    resp = await client.post("/mock-tests", json={"subjectName": "Math"})
    assert resp.status_code == 401


async def test_create_mock_test_requires_subject(client_factory, test_db):
    learner = await _learner_client(client_factory, test_db)
    resp = await learner.post("/mock-tests", json={})
    assert resp.status_code == 400


async def test_create_mock_test_rejects_non_int_question_count(client_factory, test_db):
    learner = await _learner_client(client_factory, test_db)
    resp = await learner.post("/mock-tests", json={"subjectName": "Math", "questionCount": "lots"})
    assert resp.status_code == 422


async def test_create_and_list_mock_test(client_factory, test_db):
    learner = await _learner_client(client_factory, test_db)
    created = await _create_test(learner, subjectName="Math", questionCount=5)
    assert created.status_code == 200
    test_id = created.json()["testId"]

    listed = await learner.get("/mock-tests")
    assert listed.status_code == 200
    assert len(listed.json()["tests"]) == 1
    assert listed.json()["tests"][0]["_id"] == test_id


async def test_get_mock_test_not_found(client_factory, test_db):
    learner = await _learner_client(client_factory, test_db)
    resp = await learner.get(f"/mock-tests/{ObjectId()}")
    assert resp.status_code == 404


async def test_submit_test_rejects_non_dict_answers(client_factory, test_db):
    learner = await _learner_client(client_factory, test_db)
    created = await _create_test(learner, subjectName="Math")
    test_id = created.json()["testId"]

    resp = await learner.post(f"/mock-tests/{test_id}/submit", json={"answers": ["not", "a", "dict"]})
    assert resp.status_code == 422


async def test_submit_test_scores_answers(client_factory, test_db):
    learner = await _learner_client(client_factory, test_db)
    created = await _create_test(learner, subjectName="Math", questionCount=2)
    test_id = created.json()["testId"]

    # Overwrite with known questions (the create call was patched to return
    # none, so the questions we score against are set here directly).
    await test_db["mockTests"].update_one(
        {"_id": ObjectId(test_id)},
        {"$set": {"questions": [
            {"_id": "q1", "correct_answer": "A", "marks": 1},
            {"_id": "q2", "correct_answer": "B", "marks": 1},
        ]}},
    )

    resp = await learner.post(f"/mock-tests/{test_id}/submit", json={"answers": {"q1": "A", "q2": "C"}})
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert result["correct"] == 1
    assert result["wrong"] == 1
    assert result["skipped"] == 0
    assert result["scored"] == 1

    review = await learner.get(f"/mock-tests/{test_id}/review")
    assert review.status_code == 200


# ============================================================
# ROADMAP MODE — week-range Test Engine
# ============================================================

async def test_create_roadmap_test_requires_roadmap_id(client_factory, test_db):
    learner = await _learner_client(client_factory, test_db)
    resp = await learner.post("/mock-tests", json={"mode": "roadmap", "week_start": 1, "week_end": 1})
    assert resp.status_code == 400


async def test_create_roadmap_test_requires_week_range(client_factory, test_db):
    learner = await _learner_client(client_factory, test_db)
    resp = await learner.post("/mock-tests", json={"mode": "roadmap", "roadmap_id": str(ObjectId())})
    assert resp.status_code == 400


async def test_create_roadmap_test_rejects_bad_percentages(client_factory, test_db):
    learner = await _learner_client(client_factory, test_db)
    user_id = str((await test_db["users"].find_one({"role": 7}))["_id"])
    roadmap_id = await _seed_roadmap(test_db, user_id)

    resp = await learner.post("/mock-tests", json={
        "mode": "roadmap", "roadmap_id": roadmap_id, "week_start": 1, "week_end": 1,
        "mcq_percent": 50, "subjective_percent": 20, "practical_percent": 20,
    })
    assert resp.status_code == 400


async def test_create_roadmap_test_rejects_locked_week(client_factory, test_db):
    learner = await _learner_client(client_factory, test_db)
    user_id = str((await test_db["users"].find_one({"role": 7}))["_id"])
    # Only week 1 unlocked, but seed 3 weeks total.
    roadmap_id = await _seed_roadmap(test_db, user_id, num_weeks=3)
    await test_db["selfLearnerRoadmaps"].update_one(
        {"_id": ObjectId(roadmap_id)}, {"$set": {"unlockedWeeks": [1]}},
    )

    resp = await learner.post("/mock-tests", json={
        "mode": "roadmap", "roadmap_id": roadmap_id, "week_start": 1, "week_end": 2,
        "mcq_percent": 100, "subjective_percent": 0, "practical_percent": 0,
    })
    assert resp.status_code == 403


async def test_create_roadmap_test_rejects_nonexistent_week_range(client_factory, test_db):
    learner = await _learner_client(client_factory, test_db)
    user_id = str((await test_db["users"].find_one({"role": 7}))["_id"])
    roadmap_id = await _seed_roadmap(test_db, user_id, num_weeks=2)

    resp = await learner.post("/mock-tests", json={
        "mode": "roadmap", "roadmap_id": roadmap_id, "week_start": 1, "week_end": 5,
        "mcq_percent": 100, "subjective_percent": 0, "practical_percent": 0,
    })
    assert resp.status_code == 400


async def test_create_roadmap_test_success(client_factory, test_db):
    learner = await _learner_client(client_factory, test_db)
    user_id = str((await test_db["users"].find_one({"role": 7}))["_id"])
    roadmap_id = await _seed_roadmap(test_db, user_id, num_weeks=2)

    mcq_q = {"type": "mcq", "question": "2+2?", "options": ["3", "4"], "answer": 1, "topic": "Arithmetic"}
    fake_usage = type("Usage", (), {"prompt_token_count": 10, "candidates_token_count": 20})()

    with patch(_GEMINI_JSON_PATCH, return_value=([mcq_q], fake_usage, False)):
        resp = await learner.post("/mock-tests", json={
            "mode": "roadmap", "roadmap_id": roadmap_id, "week_start": 1, "week_end": 2,
            "mcq_percent": 100, "subjective_percent": 0, "practical_percent": 0, "questionCount": 1,
        })
    assert resp.status_code == 200
    test_id = resp.json()["testId"]

    fetched = await learner.get(f"/mock-tests/{test_id}")
    assert fetched.status_code == 200
    test_body = fetched.json()["test"]
    assert test_body["mode"] == "roadmap"
    assert len(test_body["questions"]) == 1
    assert "answer" not in test_body["questions"][0]  # answer key stripped before submission


async def _seed_roadmap_test(test_db, roadmap_id: str, user_id: str, week_start: int, week_end: int, questions):
    doc = {
        "student_id": ObjectId(user_id),
        "mode": "roadmap",
        "roadmapId": ObjectId(roadmap_id),
        "weekRange": {"start": week_start, "end": week_end},
        "subjectName": "Python",
        "testTitle": f"Python: Week {week_start}-{week_end}",
        "config": {"mcqPercent": 50, "subjectivePercent": 50, "practicalPercent": 0, "questionCount": 2, "customPrompt": None},
        "questionCount": len(questions),
        "status": "pending",
        "questions": questions,
        "attempts_count": 0,
        "last_attempt": None,
    }
    result = await test_db["mockTests"].insert_one(doc)
    return str(result.inserted_id)


_ROADMAP_MCQ_Q = {"type": "mcq", "question": "2+2?", "options": ["3", "4"], "answer": 1, "explanation": "Basic math.", "topic": "Arithmetic"}
_ROADMAP_SUBJ_Q = {"type": "subjective", "question": "Explain recursion.", "modelAnswer": "A function calling itself.", "explanation": "Look for base case.", "topic": "Recursion"}


async def test_submit_roadmap_test_mixed_grading(client_factory, test_db):
    learner = await _learner_client(client_factory, test_db)
    user_id = str((await test_db["users"].find_one({"role": 7}))["_id"])
    roadmap_id = await _seed_roadmap(test_db, user_id, num_weeks=2)
    test_id = await _seed_roadmap_test(test_db, roadmap_id, user_id, 1, 1, [_ROADMAP_MCQ_Q, _ROADMAP_SUBJ_Q])

    fake_usage = type("Usage", (), {"prompt_token_count": 10, "candidates_token_count": 20})()
    grading_response = [{"score": 80, "feedback": "Good but incomplete."}]

    with patch(_GEMINI_JSON_PATCH, return_value=(grading_response, fake_usage, False)):
        resp = await learner.post(f"/mock-tests/{test_id}/submit", json={
            "answers": {"0": 1, "1": "A function that calls itself."},
        })
    assert resp.status_code == 200
    result = resp.json()["result"]
    # MCQ correct (100) + subjective 80 -> mean = 90
    assert result["percentage"] == 90.0
    assert result["correct"] == 2
    assert result["isFinalTest"] is False  # only week 1 of 2, not the full roadmap

    review = await learner.get(f"/mock-tests/{test_id}/review")
    assert review.status_code == 200
    review_body = review.json()
    assert review_body["testInfo"]["weekRange"] == {"start": 1, "end": 1}
    qwise = review_body["attempt"]["questionwise"]
    assert qwise[0]["question"] == "2+2?"
    assert qwise[1]["score"] == 80


async def test_submit_roadmap_test_marks_roadmap_completed_on_full_range_pass(client_factory, test_db):
    learner = await _learner_client(client_factory, test_db)
    user_id = str((await test_db["users"].find_one({"role": 7}))["_id"])
    roadmap_id = await _seed_roadmap(test_db, user_id, num_weeks=2)
    # Full range: week 1 -> 2 (the last week)
    test_id = await _seed_roadmap_test(test_db, roadmap_id, user_id, 1, 2, [_ROADMAP_MCQ_Q])

    resp = await learner.post(f"/mock-tests/{test_id}/submit", json={"answers": {"0": 1}})
    assert resp.status_code == 200
    assert resp.json()["result"]["isFinalTest"] is True

    roadmap_doc = await test_db["selfLearnerRoadmaps"].find_one({"_id": ObjectId(roadmap_id)})
    assert roadmap_doc["progress"]["roadmapCompleted"] is True


async def test_submit_roadmap_test_partial_range_does_not_mark_completed(client_factory, test_db):
    learner = await _learner_client(client_factory, test_db)
    user_id = str((await test_db["users"].find_one({"role": 7}))["_id"])
    roadmap_id = await _seed_roadmap(test_db, user_id, num_weeks=2)
    test_id = await _seed_roadmap_test(test_db, roadmap_id, user_id, 1, 1, [_ROADMAP_MCQ_Q])

    resp = await learner.post(f"/mock-tests/{test_id}/submit", json={"answers": {"0": 1}})
    assert resp.status_code == 200
    assert resp.json()["result"]["isFinalTest"] is False

    roadmap_doc = await test_db["selfLearnerRoadmaps"].find_one({"_id": ObjectId(roadmap_id)})
    assert roadmap_doc.get("progress", {}).get("roadmapCompleted") is not True
