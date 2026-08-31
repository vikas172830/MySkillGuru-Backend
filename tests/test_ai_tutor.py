from unittest.mock import patch

from tests.test_security_fixes import _seed_and_login_user

_CLAUDE_PATCH = "app.api.routers.ai_tutor.generate_html"
_PDF_PATCH = "app.api.routers.ai_tutor.render_html_to_pdf"
_UPLOAD_PATCH = "app.api.routers.ai_tutor.upload_file_to_imagekit"


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
    with patch(_CLAUDE_PATCH, return_value=("<html>notes</html>", {"total_tokens": 200})), \
         patch(_PDF_PATCH, return_value=b"%PDF-fake"), \
         patch(_UPLOAD_PATCH, return_value={"url": "https://cdn.test/notes.pdf", "file_id": "fid2"}):
        created = await learner.post("/api/ai-tutor/generate-notes", data={"prompt": "photosynthesis"})
    assert created.status_code == 202
    job_id = created.json()["jobId"]

    status = await learner.get(f"/api/ai-tutor/generate-notes/status/{job_id}")
    assert status.status_code == 200
    assert status.json()["status"] == "completed"
