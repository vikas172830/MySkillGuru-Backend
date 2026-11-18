# ============================================================
# Coverage for the public MySkillGuru registration endpoint
# (app/api/routers/myskillguru_auth.py) — the unauthenticated counterpart
# to /register's self_learner path, reused via the same
# _register_self_learner() helper so both produce identical accounts:
# role 7, active immediately (auto-activation is deliberate for now — the
# superadmin approval panel doesn't exist in this project; see the TEMPORARY
# note in app/api/routers/auth.py::_register_self_learner).
# ============================================================
import uuid

from app.core.security import verify_password
from tests.conftest import login

PASSWORD = "TestPass123!"


def _email(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}@test.local"


async def test_myskillguru_register_requires_no_auth(client):
    # The whole point: unlike /register, this must work with zero session.
    resp = await client.post("/myskillguru/register", json={
        "role": "self_learner", "fullName": "New Learner", "email": _email("mcg"), "password": PASSWORD,
    })
    assert resp.status_code == 200
    assert "log in" in resp.json()["message"].lower()


async def test_myskillguru_register_creates_active_self_learner(client, test_db):
    email = _email("mcg")
    resp = await client.post("/myskillguru/register", json={
        "role": "self_learner", "fullName": "New Learner", "email": email, "password": PASSWORD,
    })
    assert resp.status_code == 200

    user = await test_db["users"].find_one({"email": email})
    assert user is not None
    assert user["role"] == 7
    assert user["is_active"] is True
    assert verify_password(PASSWORD, user["password_hash"])


async def test_myskillguru_register_rejects_duplicate_email(client, test_db):
    email = _email("mcg")
    first = await client.post("/myskillguru/register", json={
        "role": "self_learner", "fullName": "First", "email": email, "password": PASSWORD,
    })
    assert first.status_code == 200

    second = await client.post("/myskillguru/register", json={
        "role": "self_learner", "fullName": "Second", "email": email, "password": PASSWORD,
    })
    assert second.status_code == 400


async def test_myskillguru_register_rejects_short_password(client):
    resp = await client.post("/myskillguru/register", json={
        "role": "self_learner", "fullName": "X", "email": _email("mcg"), "password": "abc",
    })
    assert resp.status_code == 422


async def test_myskillguru_register_rejects_blank_fullname(client):
    resp = await client.post("/myskillguru/register", json={
        "role": "self_learner", "fullName": "   ", "email": _email("mcg"), "password": PASSWORD,
    })
    assert resp.status_code == 422


async def test_myskillguru_register_rejects_invalid_email(client):
    resp = await client.post("/myskillguru/register", json={
        "role": "self_learner", "fullName": "X", "email": "not-an-email", "password": PASSWORD,
    })
    assert resp.status_code == 422


async def test_myskillguru_register_rejects_wrong_role(client):
    # This endpoint is self-learner-only — institute/faculty/etc. must go
    # through the authenticated /register flow instead.
    resp = await client.post("/myskillguru/register", json={
        "role": "institute", "fullName": "X", "email": _email("mcg"), "password": PASSWORD,
        "institute": {"institute_name": "Sneaky Institute"},
    })
    assert resp.status_code == 422


async def test_myskillguru_registered_account_can_login_immediately(client):
    email = _email("mcg")
    await client.post("/myskillguru/register", json={
        "role": "self_learner", "fullName": "Instant Learner", "email": email, "password": PASSWORD,
    })

    resp = await client.post("/login", json={"email": email, "password": PASSWORD})
    assert resp.status_code == 200


async def test_deactivated_account_cannot_login(client, test_db):
    email = _email("mcg")
    await client.post("/myskillguru/register", json={
        "role": "self_learner", "fullName": "Deactivated", "email": email, "password": PASSWORD,
    })
    await test_db["users"].update_one({"email": email}, {"$set": {"is_active": False}})

    denied = await client.post("/login", json={"email": email, "password": PASSWORD})
    assert denied.status_code == 403
