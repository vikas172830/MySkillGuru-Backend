"""Coverage for the hardening ported from the LMS: server-side token
revocation (logout / password change), inactive-account rejection, and the
fail-fast settings validators."""

import pytest
from bson import ObjectId

from app.core.config import Settings, settings
from tests.test_security_fixes import PASSWORD, _seed_and_login_user


async def _replay_client(client_factory, token: str):
    """A fresh client carrying a previously issued token (simulates a stolen
    or stale cookie that the browser no longer holds)."""
    c = await client_factory()
    c.cookies.set(settings.JWT_COOKIE_NAME, token)
    return c


async def test_logout_revokes_the_token_server_side(client_factory, test_db):
    learner = await _seed_and_login_user(test_db, client_factory, role=7, name="Revoke Me")
    token = learner.cookies.get(settings.JWT_COOKIE_NAME)
    assert token

    assert (await learner.get("/me")).status_code == 200
    assert (await learner.post("/logout")).status_code == 200

    replay = await _replay_client(client_factory, token)
    resp = await replay.get("/me")
    assert resp.status_code == 401


async def test_new_login_after_logout_works(client_factory, test_db):
    learner = await _seed_and_login_user(test_db, client_factory, role=7, name="Relogin")
    await learner.post("/logout")

    user = await test_db["users"].find_one({"fullName": "Relogin"})
    again = await client_factory()
    resp = await again.post("/login", json={"email": user["email"], "password": PASSWORD})
    assert resp.status_code == 200
    assert (await again.get("/me")).status_code == 200


async def test_password_change_revokes_existing_sessions(client_factory, test_db):
    learner = await _seed_and_login_user(test_db, client_factory, role=7, name="Change Pw")
    old_token = learner.cookies.get(settings.JWT_COOKIE_NAME)

    resp = await learner.put(
        "/profile/change-password",
        json={"currentPassword": PASSWORD, "newPassword": "BrandNewPass456!"},
    )
    assert resp.status_code == 200

    replay = await _replay_client(client_factory, old_token)
    assert (await replay.get("/me")).status_code == 401


async def test_inactive_account_is_rejected(client_factory, test_db):
    learner = await _seed_and_login_user(test_db, client_factory, role=7, name="Soon Inactive")
    await test_db["users"].update_one({"fullName": "Soon Inactive"}, {"$set": {"is_active": False}})
    assert (await learner.get("/me")).status_code == 401


async def test_deleted_account_is_rejected(client_factory, test_db):
    learner = await _seed_and_login_user(test_db, client_factory, role=7, name="Soon Deleted")
    await test_db["users"].delete_one({"fullName": "Soon Deleted"})
    assert (await learner.get("/me")).status_code == 401


async def test_token_with_malformed_subject_is_rejected(client_factory, test_db):
    from app.core.security import create_access_token

    token = create_access_token("not-an-objectid", {"role": 7})
    c = await _replay_client(client_factory, token)
    assert (await c.get("/me")).status_code == 401


def test_production_rejects_weak_jwt_secret():
    with pytest.raises(ValueError):
        Settings(_env_file=None, ENV="production", JWT_SECRET_KEY="change-me-in-production")
    with pytest.raises(ValueError):
        Settings(_env_file=None, ENV="production", JWT_SECRET_KEY="short")


def test_production_accepts_strong_jwt_secret():
    s = Settings(_env_file=None, ENV="production", JWT_SECRET_KEY="x" * 48)
    assert s.is_production


def test_development_allows_default_jwt_secret():
    Settings(_env_file=None, ENV="development", JWT_SECRET_KEY="change-me-in-production")


def test_wildcard_cors_is_rejected():
    with pytest.raises(ValueError):
        Settings(_env_file=None, CORS_ORIGINS="*")
