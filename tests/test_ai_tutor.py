from unittest.mock import patch

from app.api.routers.ai_tutor import _extract_homework_content, _extract_study_content
from tests.test_security_fixes import _seed_and_login_user

_CLAUDE_PATCH = "app.api.routers.ai_tutor.generate_html"
_PDF_PATCH = "app.api.routers.ai_tutor.render_html_to_pdf"
_UPLOAD_PATCH = "app.api.routers.ai_tutor.upload_file_to_imagekit"

# Extraction boundaries: extract_pdf_text is the local-first (pdfplumber +
# per-page Gemini OCR) path; generate_content_from_file_checked is the old
# whole-file-to-Gemini path, now reached only for images and as a fallback
# for PDFs pdfplumber cannot open at all.
_LOCAL_PDF_PATCH = "app.api.routers.ai_tutor.extract_pdf_text"
_GEMINI_FILE_PATCH = "app.api.routers.ai_tutor.generate_content_from_file_checked"
_GEMINI_HTML_PATCH = "app.api.routers.ai_tutor.generate_html_from_prompt"

_NO_USAGE = {"prompt_tokens": 0, "candidate_tokens": 0, "total_tokens": 0}


async def _learner(client_factory, test_db):
    return await _seed_and_login_user(test_db, client_factory, role=7, name="AI Tutor Learner")


async def test_homework_help_requires_auth(client):
    resp = await client.post("/api/ai-tutor/homework-help", data={"prompt": "help me"})
    assert resp.status_code == 401


async def test_homework_help_requires_prompt_or_file(client_factory, test_db):
    learner = await _learner(client_factory, test_db)
    resp = await learner.post("/api/ai-tutor/homework-help", data={"prompt": ""})
    assert resp.status_code == 400


async def test_homework_help_rejects_unsupported_file_type(client_factory, test_db):
    learner = await _learner(client_factory, test_db)
    resp = await learner.post(
        "/api/ai-tutor/homework-help",
        data={"prompt": ""},
        files={"file": ("virus.exe", b"binary", "application/octet-stream")},
    )
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.json()["error"]


async def test_homework_help_queues_job_and_completes(client_factory, test_db):
    learner = await _learner(client_factory, test_db)
    with patch(_CLAUDE_PATCH, return_value=("<html>solution</html>", {"total_tokens": 100})), \
         patch(_PDF_PATCH, return_value=b"%PDF-fake"), \
         patch(_UPLOAD_PATCH, return_value={"url": "https://cdn.test/sol.pdf", "file_id": "fid1"}):
        created = await learner.post("/api/ai-tutor/homework-help", data={"prompt": "solve 2+2"})
    assert created.status_code == 202
    job_id = created.json()["jobId"]

    status = await learner.get(f"/api/ai-tutor/homework-help/status/{job_id}")
    assert status.status_code == 200
    body = status.json()
    assert body["status"] == "completed"
    assert body["solution_url"] == "https://cdn.test/sol.pdf"


async def test_homework_help_status_not_found(client_factory, test_db):
    learner = await _learner(client_factory, test_db)
    resp = await learner.get("/api/ai-tutor/homework-help/status/does-not-exist")
    assert resp.status_code == 404


async def test_generate_notes_requires_prompt_or_file(client_factory, test_db):
    learner = await _learner(client_factory, test_db)
    resp = await learner.post("/api/ai-tutor/generate-notes", data={"prompt": ""})
    assert resp.status_code == 400


async def test_generate_notes_queues_job_and_completes(client_factory, test_db):
    learner = await _learner(client_factory, test_db)
    # Prompt-only, so this is the Gemini path — see _generate_notes_html.
    with patch(_GEMINI_HTML_PATCH, return_value=("<html>notes</html>", _NO_USAGE)), \
         patch(_PDF_PATCH, return_value=b"%PDF-fake"), \
         patch(_UPLOAD_PATCH, return_value={"url": "https://cdn.test/notes.pdf", "file_id": "fid2"}):
        created = await learner.post("/api/ai-tutor/generate-notes", data={"prompt": "photosynthesis"})
    assert created.status_code == 202
    job_id = created.json()["jobId"]

    status = await learner.get(f"/api/ai-tutor/generate-notes/status/{job_id}")
    assert status.status_code == 200
    assert status.json()["status"] == "completed"


# ============================================================
# LOCAL-FIRST PDF EXTRACTION
#
# PDFs used to be shipped whole to Gemini with an "extract all text" prompt,
# which both spent an LLM call on documents carrying a perfectly good text
# layer and silently truncated anything past the model's output-token cap.
# These cover the routing between the two extraction paths and the fact that
# a short read is now reported instead of passing as a complete result.
# ============================================================

def test_study_pdf_extracted_locally_without_gemini():
    with patch(_LOCAL_PDF_PATCH, return_value=("chapter one text", _NO_USAGE, False)) as local, \
         patch(_GEMINI_FILE_PATCH) as gemini:
        text, usage, truncated = _extract_study_content(b"%PDF-1.7 fake", "textbook.pdf")

    assert text == "chapter one text"
    assert truncated is False
    local.assert_called_once()
    gemini.assert_not_called()


def test_homework_pdf_extracted_locally_without_gemini():
    with patch(_LOCAL_PDF_PATCH, return_value=("Q1. Solve for x", _NO_USAGE, False)) as local, \
         patch(_GEMINI_FILE_PATCH) as gemini:
        text, _, truncated = _extract_homework_content(b"%PDF-1.7 fake", "homework.pdf")

    assert text == "Q1. Solve for x"
    assert truncated is False
    local.assert_called_once()
    gemini.assert_not_called()


def test_homework_image_still_uses_gemini():
    """Images carry no text layer to read locally, so they must keep going to
    Gemini vision — the local-first change applies to PDFs only."""
    with patch(_LOCAL_PDF_PATCH) as local, \
         patch(_GEMINI_FILE_PATCH, return_value=("Q1 from photo", _NO_USAGE, False)) as gemini:
        text, _, _ = _extract_homework_content(b"\x89PNG fake", "homework.png")

    assert text == "Q1 from photo"
    local.assert_not_called()
    gemini.assert_called_once()


def test_unreadable_pdf_falls_back_to_gemini():
    """Encrypted/malformed PDFs that pdfplumber rejects must still reach the
    old whole-file path rather than failing the job outright."""
    with patch(_LOCAL_PDF_PATCH, side_effect=Exception("file has not been decrypted")), \
         patch(_GEMINI_FILE_PATCH, return_value=("recovered text", _NO_USAGE, False)) as gemini:
        text, _, _ = _extract_study_content(b"%PDF-encrypted", "locked.pdf")

    assert text == "recovered text"
    gemini.assert_called_once()


def test_docx_never_touches_either_gemini_path():
    with patch(_LOCAL_PDF_PATCH) as local, \
         patch(_GEMINI_FILE_PATCH) as gemini, \
         patch("app.api.routers.ai_tutor.extract_text_from_file", return_value="docx body"):
        text, usage, truncated = _extract_study_content(b"PK fake", "notes.docx")

    assert (text, truncated) == ("docx body", False)
    assert usage["total_tokens"] == 0
    local.assert_not_called()
    gemini.assert_not_called()


async def test_generate_notes_reports_truncated_extraction(client_factory, test_db):
    learner = await _learner(client_factory, test_db)
    with patch(_LOCAL_PDF_PATCH, return_value=("only the first few pages", _NO_USAGE, True)), \
         patch(_CLAUDE_PATCH, return_value=("<html>notes</html>", {"total_tokens": 200})), \
         patch(_PDF_PATCH, return_value=b"%PDF-fake"), \
         patch(_UPLOAD_PATCH, return_value={"url": "https://cdn.test/n.pdf", "file_id": "fid3"}):
        created = await learner.post(
            "/api/ai-tutor/generate-notes",
            data={"prompt": "chapter 1"},
            files={"file": ("huge-textbook.pdf", b"%PDF-fake", "application/pdf")},
        )
    assert created.status_code == 202

    body = (await learner.get(f"/api/ai-tutor/generate-notes/status/{created.json()['jobId']}")).json()
    assert body["status"] == "completed"
    assert body["warning"], "a short read must be surfaced, not passed off as a complete document"


# ============================================================
# PROVIDER ROUTING
#
# Prompt-only notes are written from the model's own knowledge and go to
# Gemini Flash; document-backed notes stay on Claude, where faithfulness to
# the student's own source material is worth the cost.
# ============================================================

async def test_prompt_only_notes_use_gemini_not_claude(client_factory, test_db):
    learner = await _learner(client_factory, test_db)
    with patch(_GEMINI_HTML_PATCH, return_value=("<html>gemini notes</html>", _NO_USAGE)) as gemini, \
         patch(_CLAUDE_PATCH) as claude, \
         patch(_PDF_PATCH, return_value=b"%PDF-fake"), \
         patch(_UPLOAD_PATCH, return_value={"url": "https://cdn.test/n.pdf", "file_id": "fid5"}):
        created = await learner.post("/api/ai-tutor/generate-notes", data={"prompt": "photosynthesis"})

    body = (await learner.get(f"/api/ai-tutor/generate-notes/status/{created.json()['jobId']}")).json()
    assert body["status"] == "completed"
    gemini.assert_called_once()
    claude.assert_not_called()


async def test_document_backed_notes_use_claude_not_gemini(client_factory, test_db):
    learner = await _learner(client_factory, test_db)
    with patch(_LOCAL_PDF_PATCH, return_value=("the source textbook", _NO_USAGE, False)), \
         patch(_GEMINI_HTML_PATCH) as gemini, \
         patch(_CLAUDE_PATCH, return_value=("<html>claude notes</html>", {"total_tokens": 200})) as claude, \
         patch(_PDF_PATCH, return_value=b"%PDF-fake"), \
         patch(_UPLOAD_PATCH, return_value={"url": "https://cdn.test/n.pdf", "file_id": "fid6"}):
        created = await learner.post(
            "/api/ai-tutor/generate-notes",
            data={"prompt": "chapter 1"},
            files={"file": ("book.pdf", b"%PDF-fake", "application/pdf")},
        )

    body = (await learner.get(f"/api/ai-tutor/generate-notes/status/{created.json()['jobId']}")).json()
    assert body["status"] == "completed"
    claude.assert_called_once()
    gemini.assert_not_called()


async def test_gemini_quota_exhaustion_fails_over_to_claude(client_factory, test_db):
    """A quota wall must not fail the job — roadmap.py treats Gemini
    exhaustion the same way."""
    quota_error = Exception("RESOURCE_EXHAUSTED")
    quota_error.code = 429

    learner = await _learner(client_factory, test_db)
    with patch(_GEMINI_HTML_PATCH, side_effect=quota_error) as gemini, \
         patch(_CLAUDE_PATCH, return_value=("<html>failover</html>", {"total_tokens": 200})) as claude, \
         patch(_PDF_PATCH, return_value=b"%PDF-fake"), \
         patch(_UPLOAD_PATCH, return_value={"url": "https://cdn.test/n.pdf", "file_id": "fid7"}):
        created = await learner.post("/api/ai-tutor/generate-notes", data={"prompt": "photosynthesis"})

    body = (await learner.get(f"/api/ai-tutor/generate-notes/status/{created.json()['jobId']}")).json()
    assert body["status"] == "completed"
    gemini.assert_called_once()
    claude.assert_called_once()


async def test_non_quota_gemini_error_does_not_fail_over(client_factory, test_db):
    """Only quota errors change provider. Anything else surfaces as a failed
    job rather than quietly costing a second, more expensive call."""
    learner = await _learner(client_factory, test_db)
    with patch(_GEMINI_HTML_PATCH, side_effect=RuntimeError("malformed prompt")), \
         patch(_CLAUDE_PATCH) as claude, \
         patch(_PDF_PATCH, return_value=b"%PDF-fake"), \
         patch(_UPLOAD_PATCH, return_value={"url": "https://cdn.test/n.pdf", "file_id": "fid8"}):
        created = await learner.post("/api/ai-tutor/generate-notes", data={"prompt": "photosynthesis"})

    body = (await learner.get(f"/api/ai-tutor/generate-notes/status/{created.json()['jobId']}")).json()
    assert body["status"] == "failed"
    claude.assert_not_called()


async def test_generate_notes_has_no_warning_on_clean_extraction(client_factory, test_db):
    learner = await _learner(client_factory, test_db)
    with patch(_LOCAL_PDF_PATCH, return_value=("the whole document", _NO_USAGE, False)), \
         patch(_CLAUDE_PATCH, return_value=("<html>notes</html>", {"total_tokens": 200})), \
         patch(_PDF_PATCH, return_value=b"%PDF-fake"), \
         patch(_UPLOAD_PATCH, return_value={"url": "https://cdn.test/n.pdf", "file_id": "fid4"}):
        created = await learner.post(
            "/api/ai-tutor/generate-notes",
            data={"prompt": "chapter 1"},
            files={"file": ("small.pdf", b"%PDF-fake", "application/pdf")},
        )

    body = (await learner.get(f"/api/ai-tutor/generate-notes/status/{created.json()['jobId']}")).json()
    assert body["status"] == "completed"
    assert body["warning"] is None


# ============================================================
# NOTES HISTORY — persistence, list, preview, delete
#
# Generated notes previously lived only in the Redis job record, which
# expires after JOB_TTL (1 hour) with no way to revisit a past generation.
# selfLearnerNotes is the durable record; these cover that it's actually
# written on completion, that history/preview/delete are scoped to the
# owning user, and that ImageKit cleanup failing on delete doesn't block it.
# ============================================================

_DELETE_IMAGEKIT_PATCH = "app.api.routers.ai_tutor.delete_imagekit_file"


async def _generate_one_note(learner, prompt="photosynthesis", upload_return=None):
    """Runs a full prompt-only generation (Gemini path) and returns the
    completed status body, which carries note_id."""
    with patch(_GEMINI_HTML_PATCH, return_value=("<html>notes</html>", _NO_USAGE)), \
         patch(_PDF_PATCH, return_value=b"%PDF-fake"), \
         patch(_UPLOAD_PATCH, return_value=upload_return or {"url": "https://cdn.test/n.pdf", "file_id": "fid-x"}):
        created = await learner.post("/api/ai-tutor/generate-notes", data={"prompt": prompt})
    return (await learner.get(f"/api/ai-tutor/generate-notes/status/{created.json()['jobId']}")).json()


async def test_completed_generation_persists_a_note(client_factory, test_db):
    learner = await _learner(client_factory, test_db)
    body = await _generate_one_note(learner, prompt="photosynthesis")

    assert body["note_id"] is not None
    stored = await test_db.selfLearnerNotes.find_one({"_id": __import__("bson").ObjectId(body["note_id"])})
    assert stored is not None
    assert stored["title"] == "photosynthesis"
    assert stored["html_content"] == "<html>notes</html>"
    assert stored["solution_url"] == "https://cdn.test/n.pdf"
    assert stored["provider"] == "gemini"
    assert stored["grounded"] is False


async def test_list_notes_returns_history_without_html_content(client_factory, test_db):
    learner = await _learner(client_factory, test_db)
    await _generate_one_note(learner, prompt="topic one")
    await _generate_one_note(learner, prompt="topic two")

    resp = await learner.get("/api/ai-tutor/notes")
    assert resp.status_code == 200
    notes = resp.json()["notes"]

    assert len(notes) == 2
    assert [n["title"] for n in notes] == ["topic two", "topic one"]  # newest first
    assert all("html_content" not in n for n in notes)


async def test_list_notes_is_scoped_to_the_caller(client_factory, test_db):
    owner = await _learner(client_factory, test_db)
    await _generate_one_note(owner, prompt="owners note")

    stranger = await _seed_and_login_user(test_db, client_factory, role=7, name="Stranger")
    resp = await stranger.get("/api/ai-tutor/notes")
    assert resp.json()["notes"] == []


async def test_get_note_returns_full_content(client_factory, test_db):
    learner = await _learner(client_factory, test_db)
    body = await _generate_one_note(learner, prompt="full preview")

    resp = await learner.get(f"/api/ai-tutor/notes/{body['note_id']}")
    assert resp.status_code == 200
    note = resp.json()["note"]
    assert note["html_content"] == "<html>notes</html>"
    assert note["title"] == "full preview"


async def test_get_note_404s_for_another_users_note(client_factory, test_db):
    """The IDOR this closes: note_id is just a Mongo _id string — without
    the owner scope any authenticated user could read anyone else's notes
    by guessing/enumerating ids."""
    owner = await _learner(client_factory, test_db)
    body = await _generate_one_note(owner, prompt="private notes")

    stranger = await _seed_and_login_user(test_db, client_factory, role=7, name="Stranger")
    resp = await stranger.get(f"/api/ai-tutor/notes/{body['note_id']}")
    assert resp.status_code == 404


async def test_get_note_404s_for_malformed_id(client_factory, test_db):
    learner = await _learner(client_factory, test_db)
    resp = await learner.get("/api/ai-tutor/notes/not-a-valid-object-id")
    assert resp.status_code == 404


async def test_delete_note_removes_it_and_cleans_up_imagekit(client_factory, test_db):
    learner = await _learner(client_factory, test_db)
    body = await _generate_one_note(learner, upload_return={"url": "https://cdn.test/n.pdf", "file_id": "fid-del"})

    with patch(_DELETE_IMAGEKIT_PATCH) as imagekit_delete:
        resp = await learner.delete(f"/api/ai-tutor/notes/{body['note_id']}")
    assert resp.status_code == 200
    imagekit_delete.assert_called_once_with("fid-del")

    from bson import ObjectId
    assert await test_db.selfLearnerNotes.find_one({"_id": ObjectId(body["note_id"])}) is None
    assert (await learner.get(f"/api/ai-tutor/notes/{body['note_id']}")).status_code == 404


async def test_delete_note_stranger_cannot_delete_another_users_note(client_factory, test_db):
    owner = await _learner(client_factory, test_db)
    body = await _generate_one_note(owner)

    stranger = await _seed_and_login_user(test_db, client_factory, role=7, name="Stranger")
    with patch(_DELETE_IMAGEKIT_PATCH) as imagekit_delete:
        resp = await stranger.delete(f"/api/ai-tutor/notes/{body['note_id']}")
    assert resp.status_code == 404
    imagekit_delete.assert_not_called()

    from bson import ObjectId
    assert await test_db.selfLearnerNotes.find_one({"_id": ObjectId(body["note_id"])}) is not None


async def test_delete_note_succeeds_even_if_imagekit_cleanup_fails(client_factory, test_db):
    """An orphaned PDF on ImageKit is a storage-cost concern, not a
    correctness one — the Mongo record (what the student sees as "the
    note") must still be deleted."""
    learner = await _learner(client_factory, test_db)
    body = await _generate_one_note(learner)

    with patch(_DELETE_IMAGEKIT_PATCH, side_effect=RuntimeError("imagekit down")):
        resp = await learner.delete(f"/api/ai-tutor/notes/{body['note_id']}")
    assert resp.status_code == 200

    from bson import ObjectId
    assert await test_db.selfLearnerNotes.find_one({"_id": ObjectId(body["note_id"])}) is None
