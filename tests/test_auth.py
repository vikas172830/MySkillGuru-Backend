import uuid

from app.core.security import hash_password
from app.models.user import create_user_document
from tests.conftest import login, register

PASSWORD = "TestPass123!"


def _email(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}@test.local"


# ============================================================
# REGISTER — schema-level validation (422s from RegisterPayload)
# ============================================================

async def test_register_requires_authenticated_caller(client):
    resp = await client.post("/register", json={
        "role": "self_learner", "fullName": "X", "email": _email("x"), "password": PASSWORD,
    })
    assert resp.status_code == 401


async def test_register_rejects_unknown_role(superadmin_client):
    resp = await superadmin_client.post("/register", json={
        "role": "hacker", "fullName": "X", "email": _email("x"), "password": PASSWORD,
    })
    assert resp.status_code == 422


async def test_register_rejects_short_password(superadmin_client):
    resp = await superadmin_client.post("/register", json={
        "role": "self_learner", "fullName": "X", "email": _email("x"), "password": "abc",
    })
    assert resp.status_code == 422


async def test_register_rejects_blank_fullname(superadmin_client):
    resp = await superadmin_client.post("/register", json={
        "role": "self_learner", "fullName": "   ", "email": _email("x"), "password": PASSWORD,
    })
    assert resp.status_code == 422


async def test_register_rejects_invalid_email_format(superadmin_client):
    resp = await superadmin_client.post("/register", json={
        "role": "self_learner", "fullName": "X", "email": "not-an-email", "password": PASSWORD,
    })
    assert resp.status_code == 422


async def test_register_institute_requires_institute_name(superadmin_client):
    resp = await superadmin_client.post("/register", json={
        "role": "institute", "fullName": "X", "email": _email("inst"), "password": PASSWORD,
        "institute": {},
    })
    assert resp.status_code == 422


# ============================================================
# DUPLICATES
# ============================================================

async def test_register_duplicate_email_rejected(superadmin_client):
    email = _email("dup")
    payload = {"role": "self_learner", "fullName": "X", "email": email, "password": PASSWORD}
    first = await superadmin_client.post("/register", json=payload)
    assert first.status_code == 200

    second = await superadmin_client.post("/register", json=payload)
    assert second.status_code == 400
    assert "already exists" in second.json()["error"].lower()


# ============================================================
# LOGIN
# ============================================================

async def test_login_rejects_missing_password(client):
    resp = await client.post("/login", json={"email": "x@test.local"})
    assert resp.status_code == 422


async def test_login_rejects_invalid_email_format(client):
    resp = await client.post("/login", json={"email": "not-an-email", "password": PASSWORD})
    assert resp.status_code == 422


async def test_login_rejects_unknown_user(client):
    resp = await client.post("/login", json={"email": _email("nobody"), "password": PASSWORD})
    assert resp.status_code == 404


async def test_login_rejects_wrong_password(test_db, client):
    email = _email("wrongpw")
    await test_db["users"].insert_one(
        create_user_document({"fullName": "X", "email": email, "role": 1, "is_active": True}, hash_password(PASSWORD))
    )
    resp = await client.post("/login", json={"email": email, "password": "WrongPass123!"})
    assert resp.status_code == 401


async def test_login_success_returns_user(test_db, client):
    email = _email("gooduser")
    await test_db["users"].insert_one(
        create_user_document({"fullName": "Good User", "email": email, "role": 1, "is_active": True}, hash_password(PASSWORD))
    )
    resp = await client.post("/login", json={"email": email, "password": PASSWORD})
    assert resp.status_code == 200
    assert resp.json()["user"]["email"] == email


# ============================================================
# /me and /logout
# ============================================================

async def test_me_requires_auth(client):
    resp = await client.get("/me")
    assert resp.status_code == 401


async def test_logout_clears_cookie(superadmin_client):
    resp = await superadmin_client.post("/logout")
    assert resp.status_code == 200

    after = await superadmin_client.get("/me")
    assert after.status_code == 401
