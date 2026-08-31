from tests.test_security_fixes import PASSWORD


async def test_get_profile_requires_auth(client):
    resp = await client.get("/profile")
    assert resp.status_code == 401


async def test_get_profile_self_learner(client_factory, test_db):
    from tests.test_security_fixes import _seed_and_login_user

    learner = await _seed_and_login_user(test_db, client_factory, role=7, name="Profile Learner")
    resp = await learner.get("/profile")
    assert resp.status_code == 200
    assert resp.json()["role"] == 7


async def test_update_profile_rejects_empty_body(client_factory, test_db):
    from tests.test_security_fixes import _seed_and_login_user

    learner = await _seed_and_login_user(test_db, client_factory, role=7, name="Empty Body Learner")
    resp = await learner.put("/profile", json={})
    assert resp.status_code == 400


async def test_update_profile_success(client_factory, test_db):
    from tests.test_security_fixes import _seed_and_login_user

    learner = await _seed_and_login_user(test_db, client_factory, role=7, name="Update Learner")
    resp = await learner.put("/profile", json={"fullName": "New Name", "language": "hindi"})
    assert resp.status_code == 200

    fetched = await learner.get("/profile")
    assert fetched.json()["fullName"] == "New Name"
    assert fetched.json()["language"] == "hindi"


async def test_update_profile_self_learner_cannot_set_color(client_factory, test_db):
    from tests.test_security_fixes import _seed_and_login_user

    learner = await _seed_and_login_user(test_db, client_factory, role=7, name="Color Learner")
    resp = await learner.put("/profile", json={"color": "#123456"})
    assert resp.status_code == 403


async def test_change_password_rejects_missing_field(client_factory, test_db):
    from tests.test_security_fixes import _seed_and_login_user

    learner = await _seed_and_login_user(test_db, client_factory, role=7, name="Pwd Learner 1")
    resp = await learner.put("/profile/change-password", json={"currentPassword": PASSWORD})
    assert resp.status_code == 422


async def test_change_password_rejects_weak_new_password(client_factory, test_db):
    from tests.test_security_fixes import _seed_and_login_user

    learner = await _seed_and_login_user(test_db, client_factory, role=7, name="Pwd Learner 2")
    resp = await learner.put(
        "/profile/change-password", json={"currentPassword": PASSWORD, "newPassword": "abc"}
    )
    assert resp.status_code == 422


async def test_change_password_rejects_wrong_current(client_factory, test_db):
    from tests.test_security_fixes import _seed_and_login_user

    learner = await _seed_and_login_user(test_db, client_factory, role=7, name="Pwd Learner 3")
    resp = await learner.put(
        "/profile/change-password", json={"currentPassword": "WrongOne123!", "newPassword": "NewStrongPass1!"}
    )
    assert resp.status_code == 401


async def test_change_password_success_and_relogin(client_factory, test_db):
    from tests.test_security_fixes import _seed_and_login_user

    learner = await _seed_and_login_user(test_db, client_factory, role=7, name="Pwd Learner 4")
    me = (await learner.get("/profile")).json()
    email = me["email"]

    resp = await learner.put(
        "/profile/change-password", json={"currentPassword": PASSWORD, "newPassword": "NewStrongPass1!"}
    )
    assert resp.status_code == 200

    new_client = await client_factory()
    relogin = await new_client.post("/login", json={"email": email, "password": "NewStrongPass1!"})
    assert relogin.status_code == 200


async def test_imagekit_auth_returns_signature(client_factory, test_db):
    from tests.test_security_fixes import _seed_and_login_user

    learner = await _seed_and_login_user(test_db, client_factory, role=7, name="ImageKit Learner")
    resp = await learner.get("/imagekit-auth")
    assert resp.status_code == 200
    body = resp.json()
    assert "signature" in body and "token" in body and "expire" in body
