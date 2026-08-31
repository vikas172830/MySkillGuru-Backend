# ============================================================
# AUTH ROUTER
# Roles: superadmin(1), institute(2), faculty(3),
#        institute_student(4), tutor(5), tutor_student(6), self_learner(7)
# Ported from controllers/auth_controller.py + routes/auth_routes.py
# ============================================================

import re
from typing import Any, Dict

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Response
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.api.deps import get_current_identity, get_current_user
from app.core.rate_limit import login_rate_limit
from app.core.security import create_access_token, hash_password, set_access_cookie, unset_access_cookie, verify_password
from app.db.mongodb import get_database
from app.models.user import create_user_document, serialize_user
from app.schemas.auth import LoginRequest, RegisterPayload

router = APIRouter(tags=["auth"])

# ============================================================
# CONSTANTS
# ============================================================

ROLE_NAME_TO_NUMBER = {
    "superadmin": 1,
    "institute": 2,
    "faculty": 3,
    "institute_student": 4,
    "tutor": 5,
    "tutor_student": 6,
    "self_learner": 7,
}

PENDING_APPROVAL_ROLES = {5, 7}  # tutor, self_learner


def _validate_color(color) -> str:
    if isinstance(color, str) and re.match(r"^#([A-Fa-f0-9]{6})$", color):
        return color
    return "#FF7F10"


def _validate_language(language) -> str:
    if isinstance(language, str) and language.lower() in ["english", "hindi"]:
        return language.lower()
    return "english"


def _resolve_role(data: dict):
    role_name = data.get("role")

    if not role_name or not isinstance(role_name, str):
        return None, ({"error": "role is required and must be a string (e.g. 'tutor')"}, 400)

    role_number = ROLE_NAME_TO_NUMBER.get(role_name.strip().lower())
    if role_number is None:
        return None, ({"error": f"Unknown role '{role_name}'. Valid roles: {list(ROLE_NAME_TO_NUMBER.keys())}"}, 400)

    return role_number, None


# ============================================================
# ROLE-SPECIFIC REGISTRATION HANDLERS (private)
#
# MySkillGuru only ever creates self_learner (7) accounts — the
# institute/faculty/institute_student/tutor/tutor_student handlers that
# used to live here (roles 2/3/4/5/6) were removed along with the rest of
# the institute/tutor product surface this scoped-down copy doesn't serve.
# ============================================================

async def _register_self_learner(db, data, password_hash):
    language = _validate_language(data.get("language"))

    user_doc = create_user_document(
        {
            "fullName": data["fullName"],
            "email": data["email"].lower(),
            "role": 7,
            "phone": data.get("phone"),
            "hasCOAccess": False,
            "hasQPGAccess": False,
            # TEMPORARY: auto-activate on signup. The original approval gate
            # (is_active=False -> superadmin flips it on) has no way to be
            # cleared in this scoped-down copy — the super-admin panel that
            # approved pending accounts was removed along with the rest of
            # the institute/tutor UI. Re-enable the gate (is_active: False)
            # once there's a real approval path again.
            "is_active": True,
            "color": "#FF7F10",
            "language": language,
        },
        password_hash,
    )

    await db["users"].insert_one(user_doc)

    return {"message": "Registration successful. You can log in now."}, 201


# ============================================================
# REGISTER
# ============================================================

@router.post("/register")
async def register(
    payload: RegisterPayload,
    identity: dict = Depends(get_current_identity),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    # role/fullName/email/password/faculty-color/institute-details shape are
    # now enforced by the RegisterPayload discriminated union (422 instead of
    # the previous custom 400 messages for those specific cases) — see
    # app/schemas/auth.py. Per-role business-rule checks (caller authorization,
    # duplicate email, duplicate roll number, etc.) stay in the handlers below
    # unchanged, since those depend on DB state Pydantic can't see.
    data = payload.model_dump(exclude_unset=True)
    email = payload.email
    password = payload.password

    try:
        role, role_error = _resolve_role(data)
        if role_error:
            body, code = role_error
            raise HTTPException(status_code=code, detail=body["error"])

        if await db["users"].find_one({"email": email.lower()}):
            raise HTTPException(status_code=400, detail="A user with this email already exists")

        password_hash = hash_password(password)

        if role == 7:
            body, code = await _register_self_learner(db, data, password_hash)
        else:
            body, code = {"error": "Registration not supported for this role"}, 400

        if code >= 400:
            raise HTTPException(status_code=code, detail=body.get("error"))
        return body

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Registration failed")


# ============================================================
# LOGIN
# ============================================================

@router.post("/login", dependencies=[Depends(login_rate_limit)])
async def login(
    response: Response,
    payload: LoginRequest,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    try:
        email = payload.email
        password = payload.password

        user = await db["users"].find_one({"email": email, "is_deleted": False})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        if not verify_password(password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        if not user.get("is_active", True):
            if user.get("role") in PENDING_APPROVAL_ROLES:
                raise HTTPException(status_code=403, detail="Your account is pending superadmin approval. Please wait.")
            raise HTTPException(status_code=403, detail="Your account has been deactivated. Please contact your administrator.")

        access_token = create_access_token(identity=str(user["_id"]), additional_claims={"role": user["role"]})

        serialized_user = serialize_user(user)
        serialized_user["hasCOAccess"] = user.get("hasCOAccess", False)
        serialized_user["hasQPGAccess"] = user.get("hasQPGAccess", False)

        set_access_cookie(response, access_token)

        return {"message": "Login successful", "user": serialized_user}

    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception:
        raise HTTPException(status_code=500, detail="Login failed")


# ============================================================
# GET CURRENT USER (/me)
# ============================================================

@router.get("/me")
async def get_me(
    identity: dict = Depends(get_current_identity),
    user: dict = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    try:
        role = identity.get("role")
        user_id = identity["user_id"]

        serialized_user = serialize_user(user)
        serialized_user["hasCOAccess"] = user.get("hasCOAccess", False)
        serialized_user["hasQPGAccess"] = user.get("hasQPGAccess", False)

        if role == 2:
            institute = await db["instituteDetails"].find_one(
                {"user_id": ObjectId(user_id), "is_deleted": {"$ne": True}}
            )
            if institute:
                serialized_user["institute_id"] = str(institute["_id"])
                serialized_user["banner_url"] = institute.get("banner_url", "")
                serialized_user["logo_url"] = institute.get("logo_url", "")

        elif role == 3:
            faculty = await db["facultyDetails"].find_one(
                {"user_id": ObjectId(user_id), "is_deleted": {"$ne": True}}
            )
            if faculty:
                institute = await db["instituteDetails"].find_one(
                    {"_id": ObjectId(faculty["institute_id"]), "is_deleted": {"$ne": True}}
                )
                if institute:
                    serialized_user["institute_id"] = str(institute["_id"])
                    serialized_user["banner_url"] = institute.get("banner_url", "")
                    serialized_user["logo_url"] = institute.get("logo_url", "")

        elif role == 5:
            tutor = await db["tutorDetails"].find_one(
                {"user_id": ObjectId(user_id), "is_deleted": {"$ne": True}}
            )
            if tutor:
                serialized_user["tutor_id"] = str(tutor["_id"])
                serialized_user["coaching_name"] = tutor.get("coaching_name", "")

        elif role == 4:
            student = await db["studentDetails"].find_one(
                {"user_id": ObjectId(user_id), "is_deleted": {"$ne": True}}
            )
            if student:
                serialized_user["institute_id"] = str(student.get("institute_id", ""))

        elif role == 6:
            student = await db["studentDetails"].find_one(
                {"user_id": ObjectId(user_id), "is_deleted": {"$ne": True}}
            )
            if student:
                serialized_user["tutor_id"] = str(student.get("tutor_id", ""))

        return {"role": role, "user": serialized_user}

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to fetch user")


# ============================================================
# LOGOUT
# ============================================================

@router.post("/logout")
async def logout(response: Response, identity: dict = Depends(get_current_identity)):
    unset_access_cookie(response)
    return {"message": "Logged out successfully"}
