import uuid

from bson import ObjectId

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


async def test_register_faculty_rejects_manual_color(superadmin_client, client_factory):
    institute_email = _email("institute")
    await register(
        superadmin_client, role="institute", fullName="Inst Admin", email=institute_email,
        password=PASSWORD, institute={"institute_name": "Test Institute"},
    )
    institute_client = await client_factory()
    await login(institute_client, institute_email, PASSWORD)

    resp = await institute_client.post("/register", json={
        "role": "faculty", "fullName": "Fac", "email": _email("fac"), "password": PASSWORD,
        "school_id": str(ObjectId()), "color": "#123456",
    })
    assert resp.status_code == 422


async def test_register_faculty_requires_school_id(superadmin_client, client_factory):
    institute_email = _email("institute")
    await register(
        superadmin_client, role="institute", fullName="Inst Admin", email=institute_email,
        password=PASSWORD, institute={"institute_name": "Test Institute"},
    )
    institute_client = await client_factory()
    await login(institute_client, institute_email, PASSWORD)

    resp = await institute_client.post("/register", json={
        "role": "faculty", "fullName": "Fac", "email": _email("fac"), "password": PASSWORD,
    })
    assert resp.status_code == 422


# ============================================================
# REGISTER — full role hierarchy happy paths (superadmin -> institute ->
# faculty -> institute_student; tutor -> tutor_student; self_learner)
# ============================================================

async def test_full_registration_hierarchy(superadmin_client, client_factory):
    # institute (by superadmin)
    institute_email = _email("institute")
    await register(
        superadmin_client, role="institute", fullName="Inst Admin", email=institute_email,
        password=PASSWORD, institute={"institute_name": "Test Institute"},
    )
    institute_client = await client_factory()
    await login(institute_client, institute_email, PASSWORD)

    # faculty (by institute)
    faculty_email = _email("faculty")
    await register(
        institute_client, role="faculty", fullName="Faculty One", email=faculty_email,
        password=PASSWORD, school_id=str(ObjectId()),
    )
    faculty_client = await client_factory()
    await login(faculty_client, faculty_email, PASSWORD)

    # institute_student (by faculty)
    student_resp = await register(
        faculty_client, role="institute_student", fullName="Student One", email=_email("unused"),
        password=PASSWORD, school_id=str(ObjectId()), programme_id=str(ObjectId()), roll_no="R001",
        contact_no="9999999999", enrollment_no="ENR001",
    )
    assert "college_email" in student_resp

    # tutor (self-signup, caller = superadmin here since /register always
    # needs *some* authenticated caller; _register_tutor ignores who calls)
    tutor_email = _email("tutor")
    await register(superadmin_client, role="tutor", fullName="Tutor One", email=tutor_email, password=PASSWORD)

    # tutor accounts start inactive pending approval
    login_resp = await (await client_factory()).post("/login", json={"email": tutor_email, "password": PASSWORD})
    assert login_resp.status_code == 403


async def test_institute_student_duplicate_roll_no_rejected(superadmin_client, client_factory):
    institute_email = _email("institute")
    await register(
        superadmin_client, role="institute", fullName="Inst Admin", email=institute_email,
        password=PASSWORD, institute={"institute_name": "Test Institute"},
    )
    institute_client = await client_factory()
    await login(institute_client, institute_email, PASSWORD)

    programme_id = str(ObjectId())
    first = await institute_client.post("/register", json={
        "role": "institute_student", "fullName": "Alpha Student", "email": _email("unused"),
        "password": PASSWORD, "school_id": str(ObjectId()), "programme_id": programme_id, "roll_no": "R100",
        "contact_no": "9999999999", "enrollment_no": "ENR100A",
    })
    assert first.status_code == 200

    second = await institute_client.post("/register", json={
        "role": "institute_student", "fullName": "Beta Student", "email": _email("unused"),
        "password": PASSWORD, "school_id": str(ObjectId()), "programme_id": programme_id, "roll_no": "R100",
        "contact_no": "9999999999", "enrollment_no": "ENR100B",
    })
    assert second.status_code == 400
    assert "roll number" in second.json()["error"].lower()


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


# ============================================================
# BULK STUDENT ENROLLMENT
# ============================================================

async def test_bulk_enrollment_rejects_empty_list(superadmin_client):
    resp = await superadmin_client.post("/bulk-student-enrollment", json={"students": []})
    assert resp.status_code == 422


async def test_bulk_enrollment_partial_failure_reporting(superadmin_client, client_factory):
    institute_email = _email("institute")
    await register(
        superadmin_client, role="institute", fullName="Inst Admin", email=institute_email,
        password=PASSWORD, institute={"institute_name": "Test Institute"},
    )
    institute_client = await client_factory()
    await login(institute_client, institute_email, PASSWORD)

    school_id = str(ObjectId())
    programme_id = str(ObjectId())
    resp = await institute_client.post("/bulk-student-enrollment", json={
        "students": [
            {"fullName": "Good Student", "email": "unused1@test.local", "school_id": school_id, "programme_id": programme_id, "roll_no": "B001", "contact_no": "9999999999", "enrollment_no": "ENRB001"},
            {"fullName": "Bad Student"},  # missing required fields -> should fail, not abort the batch
        ]
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["success_count"] == 1
    assert body["failed_count"] == 1
