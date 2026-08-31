# ============================================================
# PROFILE ROUTER
# Ported from routes/profile_routes.py + controllers/profile_controller.py
#
# Role map: 1 superadmin, 2 institute, 3 faculty, 4 institute_student,
#           5 tutor, 6 tutor_student, 7 self_learner
# ============================================================

import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Query
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.api.deps import get_current_identity
from app.core.security import hash_password, verify_password
from app.db.mongodb import get_database
from app.models.user import serialize_doc
from app.schemas.profile import ChangePasswordRequest, ProfileUpdateRequest
from app.services.imagekit import get_imagekit_auth_params
from app.utils.cascade import cascade_institute_access

router = APIRouter(tags=["profile"])


def _validate_color(color) -> str:
    if isinstance(color, str) and re.match(r"^#([A-Fa-f0-9]{6})$", color):
        return color
    return "#FF7F10"


def _validate_language(language) -> str:
    if isinstance(language, str) and language.lower() in ["english", "hindi", "bengali"]:
        return language.lower()
    return "english"


def _common_user_fields(data: dict) -> dict:
    fields = {}
    if "fullName" in data and data["fullName"]:
        fields["fullName"] = data["fullName"].strip()
    if "phone" in data:
        fields["phone"] = data["phone"]
    if "profileImage" in data:
        fields["profileImage"] = data["profileImage"]
    if "language" in data:
        fields["language"] = _validate_language(data["language"])
    return fields


# ============================================================
# PROFILE (any logged-in user)
# ============================================================

@router.get("/profile")
async def get_profile(
    identity: dict = Depends(get_current_identity),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    user_id = identity["user_id"]
    user = await db["users"].find_one({"_id": ObjectId(user_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    role = user.get("role")
    profile = {
        "id": str(user["_id"]),
        "fullName": user.get("fullName"),
        "email": user.get("email"),
        "phone": user.get("phone"),
        "role": role,
        "profileImage": user.get("profileImage"),
        "color": user.get("color", "#FF7F10"),
        "language": user.get("language", "english"),
        "is_active": user.get("is_active", True),
        "hasCOAccess": user.get("hasCOAccess", False),
        "hasQPGAccess": user.get("hasQPGAccess", False),
    }

    if role == 2:
        institute = await db["instituteDetails"].find_one({"user_id": ObjectId(user_id)})
        if institute:
            profile["institute_profile"] = serialize_doc(institute)

    elif role == 3:
        faculty = await db["facultyDetails"].find_one({"user_id": ObjectId(user_id)})
        if faculty:
            profile["faculty_profile"] = serialize_doc(faculty)
            institute = await db["instituteDetails"].find_one({"_id": faculty.get("institute_id")})
            if institute:
                profile["institute_id"] = str(institute["_id"])
                profile["institute_name"] = institute.get("institute_name", "")

    elif role == 4:
        student = await db["studentDetails"].find_one({"user_id": ObjectId(user_id)})
        if student:
            profile["student_profile"] = serialize_doc(student)
            profile["institute_id"] = str(student.get("institute_id", ""))

    elif role == 5:
        tutor = await db["tutorDetails"].find_one({"user_id": ObjectId(user_id)})
        if tutor:
            profile["tutor_profile"] = serialize_doc(tutor)

    elif role == 6:
        student = await db["studentDetails"].find_one({"user_id": ObjectId(user_id)})
        if student:
            profile["student_profile"] = serialize_doc(student)
            profile["tutor_id"] = str(student.get("tutor_id", ""))

    return profile


@router.put("/profile")
async def update_profile(
    payload: ProfileUpdateRequest,
    identity: dict = Depends(get_current_identity),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    data = payload.model_dump(exclude_unset=True)
    if not data:
        raise HTTPException(status_code=400, detail="No data provided")

    user_id = identity["user_id"]
    user = await db["users"].find_one({"_id": ObjectId(user_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    role = user.get("role")
    now = datetime.now(timezone.utc)
    user_fields = _common_user_fields(data)

    if role == 1:
        if "color" in data:
            user_fields["color"] = _validate_color(data["color"])
        message = "Superadmin profile updated successfully"

    elif role == 2:
        if "color" in data:
            color = _validate_color(data["color"])
            user_fields["color"] = color

            institute = await db["instituteDetails"].find_one({"user_id": ObjectId(user_id)})
            if institute:
                institute_id = institute["_id"]
                await db["instituteDetails"].update_one(
                    {"_id": institute_id}, {"$set": {"color": color, "updated_at": now}}
                )
                await cascade_institute_access(db, institute_id, ObjectId(user_id), color=color)

        institute_fields = {}
        allowed_institute_fields = [
            "institute_name", "short_name", "institute_code", "email", "phone", "website",
            "address_line1", "address_line2", "city", "state", "country", "pincode",
            "affiliation", "accreditation", "established_year", "logo_url", "banner_url",
            "description",
        ]
        for field in allowed_institute_fields:
            if field in data:
                institute_fields[field] = data[field]

        if institute_fields:
            institute_fields["updated_at"] = now
            await db["instituteDetails"].update_one({"user_id": ObjectId(user_id)}, {"$set": institute_fields})

        message = "Institute profile updated successfully"

    elif role == 3:
        if "color" in data:
            raise HTTPException(status_code=403, detail="Faculty color is inherited from institute and cannot be changed")

        faculty_fields = {}
        allowed_faculty_fields = [
            "designation", "qualification", "experience_years", "bio",
            "profile_image", "specialization", "joining_date", "employee_code",
        ]
        for field in allowed_faculty_fields:
            if field in data:
                faculty_fields[field] = data[field]

        if faculty_fields:
            faculty_fields["updated_at"] = now
            await db["facultyDetails"].update_one({"user_id": ObjectId(user_id)}, {"$set": faculty_fields})

        message = "Faculty profile updated successfully"

    elif role == 4:
        if "color" in data:
            raise HTTPException(status_code=403, detail="Students cannot update color")

        student_fields = {}
        for field in ["roll_number", "enrollment_number", "year", "bio"]:
            if field in data:
                student_fields[field] = data[field]

        if student_fields:
            student_fields["updated_at"] = now
            await db["studentDetails"].update_one(
                {"user_id": ObjectId(user_id), "role": 4}, {"$set": student_fields}
            )

        message = "Student profile updated successfully"

    elif role == 5:
        if "color" in data:
            user_fields["color"] = _validate_color(data["color"])

        tutor_fields = {}
        for field in ["bio", "qualification", "experience", "subject_specialization"]:
            if field in data:
                tutor_fields[field] = data[field]

        if tutor_fields:
            tutor_fields["updated_at"] = now
            await db["tutorDetails"].update_one({"user_id": ObjectId(user_id)}, {"$set": tutor_fields})

        message = "Tutor profile updated successfully"

    elif role == 6:
        if "color" in data:
            raise HTTPException(status_code=403, detail="Students cannot update color")

        student_fields = {}
        for field in ["bio", "year"]:
            if field in data:
                student_fields[field] = data[field]

        if student_fields:
            student_fields["updated_at"] = now
            await db["studentDetails"].update_one(
                {"user_id": ObjectId(user_id), "role": 6}, {"$set": student_fields}
            )

        message = "Tutor student profile updated successfully"

    elif role == 7:
        if "color" in data:
            raise HTTPException(status_code=403, detail="MySkillGuru learners cannot update color")
        message = "Profile updated successfully"

    else:
        raise HTTPException(status_code=400, detail=f"Profile update not supported for role {role}")

    if user_fields:
        user_fields["updated_at"] = now
        await db["users"].update_one({"_id": ObjectId(user_id)}, {"$set": user_fields})

    return {"message": message}


@router.put("/profile/change-password")
async def change_password(
    payload: ChangePasswordRequest,
    identity: dict = Depends(get_current_identity),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    user_id = identity["user_id"]
    user = await db["users"].find_one({"_id": ObjectId(user_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not verify_password(payload.currentPassword, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect current password")

    new_hash = hash_password(payload.newPassword)
    await db["users"].update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"password_hash": new_hash, "updated_at": datetime.now(timezone.utc)}},
    )

    return {"message": "Password updated successfully"}



# ============================================================
# IMAGEKIT — client upload auth signature (relocated from answers.py,
# which owned it only incidentally; used by profile picture uploads).
# ============================================================

@router.get("/imagekit-auth")
async def imagekit_auth():
    return get_imagekit_auth_params()
