import json
from unittest.mock import ANY, patch

from bson import ObjectId

from app.api.routers.roadmap import (
    _build_roadmap_pdf_html,
    _resolve_grounding,
    _week_status_for_pdf,
)
from app.models.roadmap import create_roadmap_document
from app.services.rag.mongo_store import find_document_by_id
from app.services.rag.schemas import DocType, DocumentRecord, SourceFormat
from app.services.roadmap_ai import (
    build_notes_prompt,
    _dominant_vark_style,
    _is_valid_mermaid_diagram,
    _normalize_difficulty,
    _normalize_vark,
    _split_question_counts,
    infer_lesson_domain,
    validate_interactive_lesson,
)
from tests.test_security_fixes import _seed_and_login_user

_GEMINI_JSON_PATCH = "app.api.routers.roadmap.generate_gemini_json"
_CURRICULUM_PATCH = "app.api.routers.roadmap.generate_curriculum"
_CLAUDE_JSON_PATCH = "app.api.routers.roadmap.generate_claude_json"
_CLAUDE_TEXT_PATCH = "app.services.roadmap_ai.generate_claude_text"
_PDF_RENDER_PATCH = "app.api.routers.roadmap.render_html_to_pdf"
_FIND_BY_ID_PATCH = "app.services.rag.mongo_store.find_document_by_id"
_RAG_RETRIEVE_PATCH = "app.services.rag.retrieval.router.retrieve"
_RAG_SHOULD_USE_PATCH = "app.services.rag.retrieval.router.should_use_rag"


async def _learner(client_factory, test_db):
    return await _seed_and_login_user(test_db, client_factory, role=7, name="Roadmap Learner")


async def _seed_roadmap(test_db, user_id: str) -> str:
    """Inserts a minimal one-week roadmap doc directly (bypassing the AI
    generation job) so notes/quiz endpoints have real data to operate on."""
    doc = create_roadmap_document(user_id, {
        "subject": "Python",
        "goal": "Interview Prep",
        "weeks": [{
            "week": 1,
            "title": "Basics",
            "introDescription": "Getting started.",
            "subtopics": [{
                "title": "Variables",
                "summary": "What variables are.",
                "keyPoints": ["Assignment", "Types"],
                "difficulty": "Beginner",
            }],
            "practiceQuestions": [],
        }],
        "unlockedWeeks": [1],
    })
    result = await test_db["selfLearnerRoadmaps"].insert_one(doc)
    return str(result.inserted_id)


async def test_assess_requires_auth(client):
    resp = await client.post("/api/self-learner/roadmap/assess", json={"subject": "Math"})
    assert resp.status_code == 401


async def test_assess_rejects_blank_subject(client_factory, test_db):
    learner = await _learner(client_factory, test_db)
    resp = await learner.post("/api/self-learner/roadmap/assess", json={"subject": "  "})
    assert resp.status_code == 422


async def test_assess_rejects_too_long_subject(client_factory, test_db):
    learner = await _learner(client_factory, test_db)
    resp = await learner.post("/api/self-learner/roadmap/assess", json={"subject": "x" * 201})
    assert resp.status_code == 422


async def test_assess_success(client_factory, test_db):
    learner = await _learner(client_factory, test_db)
    with patch(_GEMINI_JSON_PATCH, return_value=([{"question": "2+2?"}], {}, False)):
        resp = await learner.post("/api/self-learner/roadmap/assess", json={"subject": "Math"})
    assert resp.status_code == 200
    assert resp.json()["questions"] == [{"question": "2+2?"}]


async def test_create_roadmap_requires_subject(client_factory, test_db):
    learner = await _learner(client_factory, test_db)
    resp = await learner.post("/api/self-learner/roadmap", json={})
    assert resp.status_code == 422


async def test_create_roadmap_rejects_too_long_goal(client_factory, test_db):
    learner = await _learner(client_factory, test_db)
    resp = await learner.post(
        "/api/self-learner/roadmap", json={"subject": "Math", "goal": "x" * 501}
    )
    assert resp.status_code == 422


async def test_create_roadmap_queues_job(client_factory, test_db):
    learner = await _learner(client_factory, test_db)
    with patch(_CURRICULUM_PATCH, return_value=({"weeks": []}, {}, False)):
        resp = await learner.post("/api/self-learner/roadmap", json={"subject": "Math"})
    assert resp.status_code == 202
    assert resp.json()["status"] == "processing"


async def test_create_roadmap_end_to_end_without_document_skips_grounding(client_factory, test_db):
    """Full HTTP flow: POST create -> background job runs -> GET status returns
    done with a real, persisted roadmap. No doc_id is attached (the common
    case — most students never upload material), so this guards against the
    bug just fixed: grounding lookups must be skipped entirely in that case,
    never fall back to a cross-user subject-text scan."""
    learner = await _learner(client_factory, test_db)

    fake_curriculum = {
        "subject_display_name": "Python",
        "weeks": [{
            "week": 1, "title": "Basics", "introDescription": "Getting started.",
            "subtopics": [{"title": "Variables", "summary": "...", "keyPoints": [], "difficulty": "Beginner"}],
        }],
        "stats": {},
    }
    fake_usage = type("Usage", (), {"input_tokens": 10, "output_tokens": 20})()

    with patch(_CURRICULUM_PATCH, return_value=(fake_curriculum, fake_usage, False)), \
         patch(_FIND_BY_ID_PATCH) as mock_find_by_id:
        create_resp = await learner.post(
            "/api/self-learner/roadmap", json={"subject": "Python", "goal": "Interview Prep"},
        )
        assert create_resp.status_code == 202
        job_id = create_resp.json()["job_id"]

        status_resp = await learner.get(f"/api/self-learner/roadmap/status/{job_id}")

    assert status_resp.status_code == 200
    body = status_resp.json()
    assert body["status"] == "done"
    roadmap_id = body["roadmap_id"]

    # No doc_id was attached -> grounding must be skipped entirely, never
    # fall back to a cross-user document lookup.
    mock_find_by_id.assert_not_called()

    get_resp = await learner.get(f"/api/self-learner/roadmap/{roadmap_id}")
    assert get_resp.status_code == 200
    roadmap = get_resp.json()
    assert roadmap["subject"] == "Python"
    assert len(roadmap["weeks"]) == 1
    assert roadmap["grounded_doc_id"] is None


async def test_create_roadmap_end_to_end_with_document_grounds_via_explicit_doc_id(client_factory, test_db):
    """Same full HTTP flow, but WITH a doc_id from a course-material upload the
    student just did — the explicit-id path must still ground normally and
    persist grounded_doc_id, unaffected by the no-fallback fix above."""
    learner = await _learner(client_factory, test_db)

    fake_curriculum = {
        "subject_display_name": "Python",
        "weeks": [{
            "week": 1, "title": "Basics", "introDescription": "Getting started.",
            "subtopics": [{"title": "Variables", "summary": "...", "keyPoints": [], "difficulty": "Beginner"}],
        }],
        "stats": {},
    }
    fake_usage = type("Usage", (), {"input_tokens": 10, "output_tokens": 20})()

    with patch(_CURRICULUM_PATCH, return_value=(fake_curriculum, fake_usage, False)), \
         patch(_FIND_BY_ID_PATCH, return_value=_FAKE_RECORD) as mock_find_by_id, \
         patch(_RAG_RETRIEVE_PATCH, return_value=_fake_retrieval_result("grounded curriculum context")), \
         patch(_RAG_SHOULD_USE_PATCH, return_value=True):
        create_resp = await learner.post(
            "/api/self-learner/roadmap",
            json={"subject": "Python", "goal": "Interview Prep", "doc_id": _FAKE_RECORD.id},
        )
        assert create_resp.status_code == 202
        job_id = create_resp.json()["job_id"]

        status_resp = await learner.get(f"/api/self-learner/roadmap/status/{job_id}")

    assert status_resp.status_code == 200
    body = status_resp.json()
    assert body["status"] == "done"

    mock_find_by_id.assert_called_once_with(ANY, _FAKE_RECORD.id)

    get_resp = await learner.get(f"/api/self-learner/roadmap/{body['roadmap_id']}")
    assert get_resp.status_code == 200
    assert get_resp.json()["grounded_doc_id"] == _FAKE_RECORD.id


async def test_update_subtopic_requires_key(client_factory, test_db):
    learner = await _learner(client_factory, test_db)
    resp = await learner.patch(f"/api/self-learner/roadmap/{ObjectId()}/subtopic", json={})
    assert resp.status_code == 422


async def test_update_subtopic_invalid_roadmap_id(client_factory, test_db):
    learner = await _learner(client_factory, test_db)
    resp = await learner.patch(
        "/api/self-learner/roadmap/not-valid/subtopic", json={"subtopic_key": "1-0"}
    )
    assert resp.status_code == 400


async def test_update_subtopic_not_found(client_factory, test_db):
    learner = await _learner(client_factory, test_db)
    resp = await learner.patch(
        f"/api/self-learner/roadmap/{ObjectId()}/subtopic", json={"subtopic_key": "1-0"}
    )
    assert resp.status_code == 404


async def test_submit_quiz_invalid_roadmap_id(client_factory, test_db):
    learner = await _learner(client_factory, test_db)
    resp = await learner.post(
        "/api/self-learner/roadmap/not-valid/quiz/submit", json={"week": 1, "answers": {}}
    )
    assert resp.status_code == 400


async def test_submit_quiz_rejects_non_int_week(client_factory, test_db):
    learner = await _learner(client_factory, test_db)
    resp = await learner.post(
        f"/api/self-learner/roadmap/{ObjectId()}/quiz/submit", json={"week": "not-a-number", "answers": {}}
    )
    assert resp.status_code == 422


async def test_submit_quiz_not_found(client_factory, test_db):
    learner = await _learner(client_factory, test_db)
    resp = await learner.post(
        f"/api/self-learner/roadmap/{ObjectId()}/quiz/submit", json={"week": 1, "answers": {}}
    )
    assert resp.status_code == 404


async def test_get_roadmaps_empty(client_factory, test_db):
    learner = await _learner(client_factory, test_db)
    resp = await learner.get("/api/self-learner/roadmap")
    assert resp.status_code == 200
    assert resp.json() == []


# ============================================================
# VARK HELPERS (pure functions — no DB/AI needed)
# ============================================================

def test_normalize_vark_defaults_missing_to_25():
    assert _normalize_vark(None, None, None, None) == {
        "visual": 25, "auditory": 25, "reading": 25, "kinesthetic": 25,
    }


def test_normalize_vark_clamps_negative_to_zero():
    assert _normalize_vark(-10, 50, 0, 60)["visual"] == 0


def test_dominant_vark_style_picks_highest():
    assert _dominant_vark_style({"visual": 10, "auditory": 70, "reading": 10, "kinesthetic": 10}) == "auditory"


def test_dominant_vark_style_breaks_ties_by_style_order():
    # visual comes first in VARK_STYLES, so a tie should resolve to it.
    assert _dominant_vark_style({"visual": 25, "auditory": 25, "reading": 25, "kinesthetic": 25}) == "visual"


def test_normalize_difficulty_defaults_and_rejects_unknown():
    assert _normalize_difficulty(None) == "Moderate"
    assert _normalize_difficulty("easy") == "Easy"
    assert _normalize_difficulty("Nonsense") == "Moderate"


# ============================================================
# VARK NOTES ENDPOINT
# ============================================================

async def test_notes_requires_auth(client):
    resp = await client.get(f"/api/self-learner/roadmap/{ObjectId()}/notes")
    assert resp.status_code == 401


async def test_notes_generates_and_caches_per_style_and_difficulty(client_factory, test_db):
    learner = await _learner(client_factory, test_db)
    roadmap_id = await _seed_roadmap(test_db, str((await test_db["users"].find_one({"role": 7}))["_id"]))

    fake_notes = {"summary": "Variables hold values.", "detailedExplanation": [], "keyPoints": []}
    fake_usage = type("Usage", (), {"input_tokens": 10, "output_tokens": 20})()

    with patch(_CLAUDE_JSON_PATCH, return_value=(fake_notes, fake_usage, False)) as mock_gen:
        # Auditory-dominant, Difficult
        resp = await learner.get(
            f"/api/self-learner/roadmap/{roadmap_id}/notes",
            params={"week": 1, "subtopic_idx": 0, "visual": 10, "auditory": 70, "reading": 10, "kinesthetic": 10, "difficulty": "difficult"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["cached"] is False
        assert body["style"] == "Auditory"
        assert body["difficulty"] == "Difficult"
        assert body["notes"]["summary"] == fake_notes["summary"]
        assert body["notes"]["notesSchemaVersion"] == 4
        assert body["notes"]["conceptDiagram"] == ""
        assert mock_gen.call_count == 1

        stored = await test_db["selfLearnerRoadmaps"].find_one({"_id": ObjectId(roadmap_id)})
        stored_notes = stored["weeks"][0]["subtopics"][0]["notes"]
        assert "v4-English-Auditory-Difficult" in stored_notes

        # Same blend/difficulty again -> served from cache, AI not called again
        resp2 = await learner.get(
            f"/api/self-learner/roadmap/{roadmap_id}/notes",
            params={"week": 1, "subtopic_idx": 0, "visual": 10, "auditory": 70, "reading": 10, "kinesthetic": 10, "difficulty": "difficult"},
        )
        assert resp2.status_code == 200
        assert resp2.json()["cached"] is True
        assert mock_gen.call_count == 1

        # Different dominant style -> a fresh generation, separate cache slot
        resp3 = await learner.get(
            f"/api/self-learner/roadmap/{roadmap_id}/notes",
            params={"week": 1, "subtopic_idx": 0, "visual": 80, "auditory": 10, "reading": 5, "kinesthetic": 5, "difficulty": "difficult"},
        )
        assert resp3.status_code == 200
        assert resp3.json()["cached"] is False
        assert resp3.json()["style"] == "Visual"
        assert mock_gen.call_count == 2


def test_notes_prompt_exposes_only_supported_interactive_blocks():
    prompt = build_notes_prompt(
        "Machine Learning", "Neural Networks", "Backpropagation", "Update weights.",
        ["Gradients"], "Visual", "Beginner",
    )
    assert '"schemaVersion": 4' in prompt
    assert '"visualAid"' in prompt
    assert "Use grid for arrays/matrices/spatial coordinates" in prompt
    assert '"language": "English"' in prompt
    assert '"type":"guided_walkthrough"' in prompt
    assert '"type":"parameter_explorer"' in prompt
    assert "exactly 5 items in teaching order" in prompt
    assert "entire response below 4,000 words" in prompt
    assert "Do not output HTML, JavaScript" in prompt


def test_lesson_domain_inference_supports_technical_and_non_technical_courses():
    assert infer_lesson_domain("Computer Science", "FastAPI") == "technology"
    assert infer_lesson_domain("Digital Marketing", "Customer Segmentation") == "business"
    assert infer_lesson_domain("Finance", "Compound Interest") == "quantitative"
    assert infer_lesson_domain("History", "Industrial Revolution") == "humanities"


def test_marketing_prompt_selects_business_teaching_without_forcing_code():
    prompt = build_notes_prompt(
        "Digital Marketing", "Audience Strategy", "Customer Segmentation",
        "Group customers using meaningful characteristics.", ["Segments", "Campaign fit"],
        "Reading", "Beginner",
    )
    assert "Suggested Domain Family: business" in prompt
    assert '"domain": "business"' in prompt
    assert "Do not add code unless the topic requires it" in prompt


def test_notes_prompt_json_schema_stays_valid_for_every_learning_style():
    for style in ("Visual", "Auditory", "Reading", "Kinesthetic"):
        prompt = build_notes_prompt(
            "General Science", "Foundations", "Observation", "Observe a process.",
            ["Evidence"], style, "Beginner",
        )
        start = prompt.index("{", prompt.index("with this exact schema:"))
        end = prompt.index("\n\nPopulate blocks", start)
        schema = json.loads(prompt[start:end])
        assert schema["interactiveLesson"]["schemaVersion"] == 4
        assert schema["interactiveLesson"]["visualAid"]["kind"] == "grid"


def _valid_guided_lesson(domain="technology"):
    return {
        "schemaVersion": 4,
        "language": "English",
        "domain": domain,
        "title": "Understand one complete example",
        "mission": {
            "goal": "Understand and apply the concept.",
            "whyItMatters": "It supports practical decisions.",
            "estimatedMinutes": 25,
            "successCriteria": ["Explain the idea", "Apply it to an example"],
        },
        "prerequisites": ["No prior knowledge required"],
        "learningOutcomes": ["Explain the concept", "Apply the process"],
        "keyTerms": [
            {"term": "Input", "meaning": "Information used by a process."},
            {"term": "Result", "meaning": "The outcome after the process."},
        ],
        "anchorExample": {
            "title": "A small practical example",
            "context": "A learner follows one input through the complete process.",
            "whyChosen": "The values are small enough to inspect at every step.",
        },
        "visualAid": {
            "kind": "sequence",
            "title": "Follow one input",
            "purpose": "Make each stage and its result visible.",
            "items": [
                {"label": "Input", "value": "Start", "description": "The information entering the process.", "level": 0},
                {"label": "Result", "value": "Finish", "description": "The outcome after the process.", "level": 0},
            ],
            "interactionPrompt": "Select each stage in order.",
            "caption": "The result depends on passing the input through every required stage.",
        },
        "blocks": [
            {
                "type": "concept", "title": "The core idea",
                "simpleExplanation": "Start with a plain-language definition.",
                "whyItMatters": "It explains the rest of the process.",
            },
            {
                "type": "mental_model", "title": "A useful model",
                "analogy": "Think of a labelled path.",
                "explanation": "Each stage has one purpose.",
                "remember": "Understand the reason before memorising a step.",
            },
            {
                "type": "worked_example", "title": "Follow the example",
                "scenario": "Process one small input.",
                "exampleReason": "It keeps every change visible.",
                "steps": [
                    {"label": "Begin", "action": "Read the input.", "explanation": "The process needs a starting value."},
                    {"label": "Finish", "action": "Produce the result.", "explanation": "All required steps are complete."},
                ],
                "outcome": "The learner can explain the complete path.",
            },
            {
                "type": "common_mistakes", "title": "Avoid this",
                "items": [{"mistake": "Memorising without context", "whyItHappens": "The example was not connected to a goal.", "correction": "State the purpose first."}],
            },
            {
                "type": "quick_check", "title": "Check your understanding",
                "question": "What should come before memorisation?",
                "options": ["Understanding the purpose", "Skipping the example"],
                "correctAnswerIndex": 0,
                "explanation": "Purpose gives each step meaning.",
            },
        ],
        "summary": {
            "keyTakeaways": ["Start with purpose", "Follow one example", "Check understanding"],
            "masteryChecklist": ["I can explain the idea", "I can apply the process"],
            "nextStep": "Practise with a different example.",
        },
    }


def test_interactive_lesson_validation_keeps_valid_blocks_and_drops_invalid_ones():
    lesson = _valid_guided_lesson()
    lesson["blocks"].append({"type": "invented_widget", "title": "Unsafe"})
    notes = {"summary": "Learn by exploring.", "interactiveLesson": lesson}

    validated = validate_interactive_lesson(notes)

    assert validated["notesSchemaVersion"] == 4
    assert len(validated["interactiveLesson"]["blocks"]) == 5
    assert all(block["type"] != "invented_widget" for block in validated["interactiveLesson"]["blocks"])


def test_business_lesson_accepts_a_domain_specific_case_study():
    lesson = _valid_guided_lesson("business")
    lesson["anchorExample"] = {
        "title": "Online clothing store",
        "context": "The store needs a campaign for repeat customers.",
        "whyChosen": "It connects segmentation to a familiar business decision.",
    }
    lesson["blocks"][3] = {
        "type": "case_study",
        "title": "Choose a customer campaign",
        "scenario": "A store has new, repeat, and inactive customers.",
        "facts": ["Repeat customers convert more often", "Inactive customers need a reason to return"],
        "decision": "Which segment should receive a loyalty offer?",
        "recommendedApproach": "Target repeat customers with the loyalty offer.",
        "reasoning": "The offer reinforces existing purchase behaviour and supports retention.",
    }

    validated = validate_interactive_lesson({"interactiveLesson": lesson})

    assert validated["interactiveLesson"]["domain"] == "business"
    assert validated["interactiveLesson"]["blocks"][3]["type"] == "case_study"


def test_visual_aid_accepts_an_interactive_array_grid():
    lesson = _valid_guided_lesson()
    lesson["visualAid"] = {
        "kind": "grid",
        "title": "A 2 by 3 array",
        "purpose": "Show how row and column coordinates meet at one value.",
        "columnHeaders": ["Column 0", "Column 1", "Column 2"],
        "rows": [
            {"label": "Row 0", "values": ["4", "8", "2"]},
            {"label": "Row 1", "values": ["7", "1", "9"]},
        ],
        "interactionPrompt": "Select a cell and identify its row and column.",
        "caption": "Each value has two coordinates: its row first and its column second.",
    }

    validated = validate_interactive_lesson({"interactiveLesson": lesson})

    visual = validated["interactiveLesson"]["visualAid"]
    assert visual["kind"] == "grid"
    assert visual["rows"][1]["values"][2] == "9"


def test_visual_aid_accepts_a_marketing_metrics_table():
    lesson = _valid_guided_lesson("business")
    lesson["visualAid"] = {
        "kind": "table",
        "title": "Campaign performance",
        "purpose": "Compare the same meaningful metrics across two campaigns.",
        "columnHeaders": ["Click rate", "Conversion rate"],
        "rows": [
            {"label": "Campaign A", "values": ["4%", "2%"]},
            {"label": "Campaign B", "values": ["3%", "5%"]},
        ],
        "interactionPrompt": "Select values and compare which campaign converts better.",
        "caption": "A higher click rate does not automatically mean a higher conversion rate.",
    }

    validated = validate_interactive_lesson({"interactiveLesson": lesson})

    assert validated["interactiveLesson"]["visualAid"]["kind"] == "table"


def test_visual_aid_rejects_grid_rows_with_wrong_dimensions():
    lesson = _valid_guided_lesson()
    lesson["visualAid"] = {
        "kind": "grid",
        "title": "Broken grid",
        "purpose": "This grid has mismatched dimensions.",
        "columnHeaders": ["Column 0", "Column 1"],
        "rows": [{"label": "Row 0", "values": ["Only one value"]}],
        "interactionPrompt": "Select a cell.",
        "caption": "This should not be shown.",
    }

    validated = validate_interactive_lesson({"interactiveLesson": lesson})

    assert "interactiveLesson" not in validated


def test_interactive_lesson_validation_omits_section_when_no_block_is_valid():
    notes = {
        "summary": "Still useful as ordinary notes.",
        "interactiveLesson": {
            "schemaVersion": 4,
            "blocks": [{"type": "quick_check", "title": "Broken"}],
        },
    }

    validated = validate_interactive_lesson(notes)

    assert "interactiveLesson" not in validated


async def test_notes_week_locked(client_factory, test_db):
    learner = await _learner(client_factory, test_db)
    roadmap_id = await _seed_roadmap(test_db, str((await test_db["users"].find_one({"role": 7}))["_id"]))

    resp = await learner.get(f"/api/self-learner/roadmap/{roadmap_id}/notes", params={"week": 2})
    assert resp.status_code == 403


# ============================================================
# MERMAID CONCEPT DIAGRAM VALIDATION
# ============================================================

_VALID_DIAGRAM = (
    'graph TD\n'
    '    A["Start: Define the Problem"] --> B["Gather Requirements"]\n'
    '    B --> C["Design Solution"]\n'
    '    C -->|"Approved"| D["Implement"]'
)

# The actual documented production failure case (see CLAUDE.md's roadmap_ai_todo.md
# §21 Post-Pivot notes): an unquoted round-bracket node whose label contains its
# own parentheses. Confirming the validator correctly rejects it is the whole
# point of this test — a looser regex could easily let this back through.
_BROKEN_DIAGRAM = 'graph TD\n    A(International Political Economy (IPE)) --> B[Trade]'


def test_mermaid_validator_accepts_valid_diagram():
    assert _is_valid_mermaid_diagram(_VALID_DIAGRAM) is True


def test_mermaid_validator_rejects_unquoted_parens_label():
    assert _is_valid_mermaid_diagram(_BROKEN_DIAGRAM) is False


def test_mermaid_validator_rejects_wrong_diagram_type():
    assert _is_valid_mermaid_diagram('mindmap\n  root((Topic))') is False


def test_mermaid_validator_rejects_empty_or_none():
    assert _is_valid_mermaid_diagram("") is False
    assert _is_valid_mermaid_diagram(None) is False


async def test_notes_visual_dominant_repairs_invalid_diagram(client_factory, test_db):
    learner = await _learner(client_factory, test_db)
    roadmap_id = await _seed_roadmap(test_db, str((await test_db["users"].find_one({"role": 7}))["_id"]))

    fake_notes = {"summary": "...", "conceptDiagram": _BROKEN_DIAGRAM}
    fake_usage = type("Usage", (), {"input_tokens": 10, "output_tokens": 20})()

    with patch(_CLAUDE_JSON_PATCH, return_value=(fake_notes, fake_usage, False)), \
         patch(_CLAUDE_TEXT_PATCH, return_value=(_VALID_DIAGRAM, fake_usage)) as mock_repair:
        resp = await learner.get(
            f"/api/self-learner/roadmap/{roadmap_id}/notes",
            params={"week": 1, "subtopic_idx": 0, "visual": 80, "auditory": 10, "reading": 5, "kinesthetic": 5},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["style"] == "Visual"
        assert body["notes"]["conceptDiagram"] == _VALID_DIAGRAM
        assert mock_repair.call_count == 1


async def test_notes_visual_dominant_drops_diagram_when_repair_also_fails(client_factory, test_db):
    learner = await _learner(client_factory, test_db)
    roadmap_id = await _seed_roadmap(test_db, str((await test_db["users"].find_one({"role": 7}))["_id"]))

    fake_notes = {"summary": "...", "conceptDiagram": _BROKEN_DIAGRAM}
    fake_usage = type("Usage", (), {"input_tokens": 10, "output_tokens": 20})()

    with patch(_CLAUDE_JSON_PATCH, return_value=(fake_notes, fake_usage, False)), \
         patch(_CLAUDE_TEXT_PATCH, return_value=(_BROKEN_DIAGRAM, fake_usage)):
        resp = await learner.get(
            f"/api/self-learner/roadmap/{roadmap_id}/notes",
            params={"week": 1, "subtopic_idx": 0, "visual": 80, "auditory": 10, "reading": 5, "kinesthetic": 5},
        )
        assert resp.status_code == 200
        assert resp.json()["notes"]["conceptDiagram"] == ""


# ============================================================
# AUTO TEST — question-count splitting (pure function)
# ============================================================

def test_split_question_counts_sums_to_total_even_split():
    counts = _split_question_counts(34, 33, 33, 10)
    assert sum(counts.values()) == 10


def test_split_question_counts_sums_to_total_various_percentages():
    # A representative sweep, not exhaustive — the invariant (sums to total)
    # is what actually matters, checked across several awkward splits that
    # plain truncation would under-count.
    for mcq, subj, prac, total in [
        (100, 0, 0, 7), (0, 100, 0, 13), (33, 33, 34, 1), (60, 20, 20, 9),
        (70, 15, 15, 20), (1, 1, 98, 3), (25, 25, 50, 6),
    ]:
        counts = _split_question_counts(mcq, subj, prac, total)
        assert sum(counts.values()) == total, f"mcq={mcq} subj={subj} prac={prac} total={total} -> {counts}"


def test_split_question_counts_100_percent_mcq_puts_everything_in_mcq():
    counts = _split_question_counts(100, 0, 0, 10)
    assert counts == {"mcq": 10, "subjective": 0, "practical": 0}


# ============================================================
# AUTO TEST — generate / resume / submit
# ============================================================

_MCQ_Q = {
    "type": "mcq", "question": "2+2?", "options": ["3", "4", "5", "6"], "answer": 1,
    "explanation": "Basic arithmetic.", "difficulty": "Easy", "topic": "Arithmetic",
}
_SUBJ_Q = {
    "type": "subjective", "question": "Explain variables.",
    "modelAnswer": "A variable is a named storage location.",
    "explanation": "Look for: naming, storage, mutability.", "difficulty": "Easy", "topic": "Variables",
}


async def test_generate_auto_test_rejects_percentages_not_summing_to_100(client_factory, test_db):
    learner = await _learner(client_factory, test_db)
    roadmap_id = await _seed_roadmap(test_db, str((await test_db["users"].find_one({"role": 7}))["_id"]))

    resp = await learner.post(
        f"/api/self-learner/roadmap/{roadmap_id}/quiz/generate",
        json={"week": 1, "mcq_percent": 50, "subjective_percent": 20, "practical_percent": 20, "question_count": 10},
    )
    assert resp.status_code == 422


async def test_generate_auto_test_strips_answer_keys_and_caches_on_week(client_factory, test_db):
    learner = await _learner(client_factory, test_db)
    user_id = str((await test_db["users"].find_one({"role": 7}))["_id"])
    roadmap_id = await _seed_roadmap(test_db, user_id)

    questions = [_MCQ_Q, _SUBJ_Q]
    fake_usage = type("Usage", (), {"prompt_token_count": 10, "candidates_token_count": 20})()

    with patch(_GEMINI_JSON_PATCH, return_value=(questions, fake_usage, False)):
        resp = await learner.post(
            f"/api/self-learner/roadmap/{roadmap_id}/quiz/generate",
            json={"week": 1, "mcq_percent": 50, "subjective_percent": 50, "practical_percent": 0, "question_count": 2},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["questions"]) == 2
    for q in body["questions"]:
        assert "answer" not in q
        assert "modelAnswer" not in q
        assert "explanation" not in q
    assert body["config"]["mcqPercent"] == 50

    # It's now stored on the week doc — GET /quiz (resume) should return it.
    resume = await learner.get(f"/api/self-learner/roadmap/{roadmap_id}/quiz", params={"week": 1})
    assert resume.status_code == 200
    assert len(resume.json()["questions"]) == 2


async def test_resume_auto_test_returns_null_when_none_generated(client_factory, test_db):
    learner = await _learner(client_factory, test_db)
    roadmap_id = await _seed_roadmap(test_db, str((await test_db["users"].find_one({"role": 7}))["_id"]))

    resp = await learner.get(f"/api/self-learner/roadmap/{roadmap_id}/quiz", params={"week": 1})
    assert resp.status_code == 200
    assert resp.json() == {"questions": None, "config": None}


async def test_resume_auto_test_never_generates_on_its_own(client_factory, test_db):
    """GET /quiz must be resume-only — it should never call the AI itself."""
    learner = await _learner(client_factory, test_db)
    roadmap_id = await _seed_roadmap(test_db, str((await test_db["users"].find_one({"role": 7}))["_id"]))

    with patch(_GEMINI_JSON_PATCH) as mock_gen:
        resp = await learner.get(f"/api/self-learner/roadmap/{roadmap_id}/quiz", params={"week": 1})
    assert resp.status_code == 200
    mock_gen.assert_not_called()


async def _seed_roadmap_with_auto_test(test_db, user_id: str, questions) -> str:
    roadmap_id = await _seed_roadmap(test_db, user_id)
    await test_db["selfLearnerRoadmaps"].update_one(
        {"_id": ObjectId(roadmap_id)},
        {"$set": {"weeks.0.autoTest": {
            "config": {"mcqPercent": 50, "subjectivePercent": 50, "practicalPercent": 0, "questionCount": 2, "customPrompt": None},
            "questions": questions,
        }}},
    )
    return roadmap_id


async def test_submit_auto_test_mixed_mcq_and_subjective_grading(client_factory, test_db):
    learner = await _learner(client_factory, test_db)
    user_id = str((await test_db["users"].find_one({"role": 7}))["_id"])
    roadmap_id = await _seed_roadmap_with_auto_test(test_db, user_id, [_MCQ_Q, _SUBJ_Q])

    fake_usage = type("Usage", (), {"prompt_token_count": 10, "candidates_token_count": 20})()
    grading_response = [{"score": 80, "feedback": "Good, but missed mutability."}]

    with patch(_GEMINI_JSON_PATCH, return_value=(grading_response, fake_usage, False)) as mock_grade:
        resp = await learner.post(
            f"/api/self-learner/roadmap/{roadmap_id}/quiz/submit",
            json={"week": 1, "answers": {"0": 1, "1": "A variable stores a value under a name."}},
        )
    assert resp.status_code == 200
    body = resp.json()
    # MCQ scored 100 (correct), subjective scored 80 -> mean = 90
    assert body["score"] == 90
    assert body["passed"] is True
    assert mock_grade.call_count == 1
    results = {r["questionIdx"]: r for r in body["results"]}
    assert results[0]["score"] == 100
    assert results[1]["score"] == 80
    assert results[1]["feedback"] == "Good, but missed mutability."


async def test_submit_auto_test_unanswered_open_ended_scored_zero_without_ai_call(client_factory, test_db):
    learner = await _learner(client_factory, test_db)
    user_id = str((await test_db["users"].find_one({"role": 7}))["_id"])
    roadmap_id = await _seed_roadmap_with_auto_test(test_db, user_id, [_MCQ_Q, _SUBJ_Q])

    with patch(_GEMINI_JSON_PATCH) as mock_grade:
        resp = await learner.post(
            f"/api/self-learner/roadmap/{roadmap_id}/quiz/submit",
            # Q0 (MCQ) wrong, Q1 (subjective) left blank
            json={"week": 1, "answers": {"0": 0}},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["score"] == 0  # both questions score 0
    mock_grade.assert_not_called()  # no open-ended answers -> no AI call needed
    results = {r["questionIdx"]: r for r in body["results"]}
    assert results[1]["score"] == 0
    assert results[1]["feedback"] == "No answer provided."


async def test_submit_auto_test_not_generated_yet(client_factory, test_db):
    learner = await _learner(client_factory, test_db)
    roadmap_id = await _seed_roadmap(test_db, str((await test_db["users"].find_one({"role": 7}))["_id"]))

    resp = await learner.post(
        f"/api/self-learner/roadmap/{roadmap_id}/quiz/submit",
        json={"week": 1, "answers": {}},
    )
    assert resp.status_code == 400


# ============================================================
# PDF EXPORT
# ============================================================

def test_week_status_for_pdf_locked_completed_in_progress():
    doc = {
        "unlockedWeeks": [1, 2],
        "progress": {"passedQuizzes": {"1": 90}},
    }
    assert _week_status_for_pdf(doc, 1) == "Completed"
    assert _week_status_for_pdf(doc, 2) == "In Progress"
    assert _week_status_for_pdf(doc, 3) == "Locked"


def test_build_roadmap_pdf_html_escapes_xss_subject():
    doc = create_roadmap_document("000000000000000000000000", {
        "subject": '<script>alert("xss")</script>',
        "goal": "Test",
        "weeks": [{
            "week": 1, "title": "Intro", "introDescription": "<img src=x onerror=alert(1)>",
            "subtopics": [{"title": "A"}],
        }],
        "unlockedWeeks": [1],
    })
    html_out = _build_roadmap_pdf_html(doc)
    assert "<script>alert" not in html_out
    assert "<img src=x onerror" not in html_out
    assert "&lt;script&gt;" in html_out


async def test_download_pdf_requires_auth(client):
    resp = await client.get(f"/api/self-learner/roadmap/{ObjectId()}/pdf")
    assert resp.status_code == 401


async def test_download_pdf_invalid_roadmap_id(client_factory, test_db):
    learner = await _learner(client_factory, test_db)
    resp = await learner.get("/api/self-learner/roadmap/not-valid/pdf")
    assert resp.status_code == 400


async def test_download_pdf_not_found(client_factory, test_db):
    learner = await _learner(client_factory, test_db)
    resp = await learner.get(f"/api/self-learner/roadmap/{ObjectId()}/pdf")
    assert resp.status_code == 404


async def test_download_pdf_success(client_factory, test_db):
    learner = await _learner(client_factory, test_db)
    roadmap_id = await _seed_roadmap(test_db, str((await test_db["users"].find_one({"role": 7}))["_id"]))

    fake_pdf_bytes = b"%PDF-1.4 fake pdf content"
    with patch(_PDF_RENDER_PATCH, return_value=fake_pdf_bytes) as mock_render:
        resp = await learner.get(f"/api/self-learner/roadmap/{roadmap_id}/pdf")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert "attachment" in resp.headers["content-disposition"]
    assert resp.content == fake_pdf_bytes
    assert mock_render.call_count == 1


# ============================================================
# RAG GROUNDING RESOLUTION
# ============================================================

_FAKE_RECORD = DocumentRecord(
    id="doc-abc-123", filename="syllabus.pdf", source_format=SourceFormat.PDF,
    doc_type=DocType.STRUCTURED, course_title="Python Fundamentals",
)


def _fake_retrieval_result(context_text="grounded content"):
    result = type("RetrievalResult", (), {
        "context_text": context_text, "source_nodes": [], "confidence": 0.9, "doc_id": _FAKE_RECORD.id,
    })()
    return result


async def test_find_document_by_id_found_and_not_found(test_db):
    # Real stored docs always have both "_id" and "id" (save_document_record
    # does `doc = asdict(record)` — which already includes "id" as a
    # dataclass field — then separately sets doc["_id"] = record.id).
    await test_db.courseMaterials.insert_one({
        "_id": "doc-xyz", "id": "doc-xyz", "filename": "book.pdf", "source_format": "pdf",
        "doc_type": "structured", "course_code": None, "course_title": "Algebra",
        "content_hash": None, "created_at": None,
    })
    found = await find_document_by_id(test_db, "doc-xyz")
    assert found is not None
    assert found.id == "doc-xyz"
    assert found.course_title == "Algebra"

    missing = await find_document_by_id(test_db, "does-not-exist")
    assert missing is None

    empty = await find_document_by_id(test_db, "")
    assert empty is None


async def test_resolve_grounding_trusts_grounded_doc_id_when_present_and_valid(test_db):
    doc = {"grounded_doc_id": "doc-abc-123"}
    with patch(_FIND_BY_ID_PATCH, return_value=_FAKE_RECORD) as mock_by_id, \
         patch(_RAG_RETRIEVE_PATCH, return_value=_fake_retrieval_result("trusted grounding")), \
         patch(_RAG_SHOULD_USE_PATCH, return_value=True):
        result = await _resolve_grounding(test_db, doc, "some query", "user-1")

    assert result == "trusted grounding"
    mock_by_id.assert_called_once_with(test_db, "doc-abc-123")


async def test_resolve_grounding_returns_none_when_no_grounded_doc_id(test_db):
    """A roadmap that was never grounded in a document at creation time must
    stay ungrounded — no subject-text fallback search across every student's
    uploads. find_document_by_id should never even be called."""
    doc = {}  # no grounded_doc_id at all
    with patch(_FIND_BY_ID_PATCH) as mock_by_id:
        result = await _resolve_grounding(test_db, doc, "some query", "user-1")

    assert result is None
    mock_by_id.assert_not_called()


async def test_resolve_grounding_returns_none_when_grounded_doc_id_is_stale(test_db):
    """The material behind grounded_doc_id was deleted since the roadmap was
    created — find_document_by_id correctly returns None, and that must
    resolve to ungrounded rather than falling back to a subject-text match."""
    doc = {"grounded_doc_id": "deleted-doc-id"}
    with patch(_FIND_BY_ID_PATCH, return_value=None) as mock_by_id, \
         patch(_RAG_RETRIEVE_PATCH) as mock_retrieve:
        result = await _resolve_grounding(test_db, doc, "some query", "user-1")

    assert result is None
    mock_by_id.assert_called_once()
    mock_retrieve.assert_not_called()
