"""
Regression suite for security fixes that still apply to MySkillGuru's
scoped-down surface (self_learner accounts only — no institute/faculty
product line). Also the shared test-helper module other test files import
from (PASSWORD, _seed_and_login_user).
"""

import json
import uuid

from app.core.security import hash_password
from app.models.user import create_user_document
from app.services.job_store import set_job
from tests.conftest import login

PASSWORD = "TestPass123!"


async def _seed_and_login_user(test_db, client_factory, role: int, name: str = "Test User"):
    """Bypasses /register (which requires an already-authenticated caller
    for every role, and leaves self_learner pending approval) purely to get
    a second/third distinct logged-in user quickly. The registration flow
    itself is covered separately in test_myskillguru_auth.py."""
    email = f"user-{uuid.uuid4().hex[:10]}@test.local"
    user_doc = create_user_document(
        {"fullName": name, "email": email, "role": role, "is_active": True}, hash_password(PASSWORD)
    )
    await test_db["users"].insert_one(user_doc)
    client = await client_factory()
    await login(client, email, PASSWORD)
    return client


# ============================================================
# Job-status endpoints scoped to the requesting user
# ============================================================

async def test_roadmap_job_status_scoped_to_owner(client_factory, test_db):
    from app.api.routers.roadmap import ROADMAP_JOB_PREFIX

    owner = await _seed_and_login_user(test_db, client_factory, role=7, name="Learner A")
    other = await _seed_and_login_user(test_db, client_factory, role=7, name="Learner B")

    owner_user_doc = await test_db["users"].find_one({"fullName": "Learner A"})
    job_id = str(uuid.uuid4())
    await set_job(ROADMAP_JOB_PREFIX, job_id, {"status": "processing", "user_id": str(owner_user_doc["_id"])})

    denied = await other.get(f"/api/self-learner/roadmap/status/{job_id}")
    assert denied.status_code == 404

    allowed = await owner.get(f"/api/self-learner/roadmap/status/{job_id}")
    assert allowed.status_code == 200
    assert allowed.json()["status"] == "processing"


# ============================================================
# Exception handler preserves custom dict detail content
# ============================================================

async def test_exception_handler_preserves_unrecognized_dict_detail():
    from fastapi import HTTPException

    from app.main import http_exception_handler

    exc = HTTPException(status_code=400, detail={"foo": "bar"})
    response = await http_exception_handler(None, exc)

    assert response.status_code == 400
    body = json.loads(response.body)
    assert body["foo"] == "bar"
    assert body["error"] != "Request failed"
    assert "bar" in body["error"]


# ============================================================
# CSRF hardening — SameSite=Strict on the auth cookie
# ============================================================

async def test_login_cookie_is_samesite_strict(client, test_db):
    email = f"cookie-test-{uuid.uuid4().hex[:8]}@test.local"
    await test_db["users"].insert_one(
        create_user_document({"fullName": "Cookie Test", "email": email, "role": 7, "is_active": True}, hash_password(PASSWORD))
    )

    resp = await client.post("/login", json={"email": email, "password": PASSWORD})
    assert resp.status_code == 200

    set_cookie_headers = resp.headers.get_list("set-cookie")
    assert any("samesite=strict" in h.lower() for h in set_cookie_headers)
