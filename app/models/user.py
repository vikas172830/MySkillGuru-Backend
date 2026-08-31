from datetime import datetime, timezone
from typing import Any, Dict
import re

# -----------------------------------
# Role constants (RBAC)
# -----------------------------------

SUPERADMIN = 1
INSTITUTE = 2
FACULTY = 3
INSTITUTE_STUDENT = 4
TUTOR = 5
TUTOR_STUDENT = 6
SELF_LEARNER = 7

ROLE_DETAILS = {
    SUPERADMIN: {"name": "superadmin", "description": "Platform Super Administrator"},
    INSTITUTE: {"name": "institute", "description": "Institute Administrator"},
    FACULTY: {"name": "faculty", "description": "Faculty Member"},
    INSTITUTE_STUDENT: {"name": "institute_student", "description": "Institute Student"},
    TUTOR: {"name": "tutor", "description": "Private Tutor"},
    TUTOR_STUDENT: {"name": "tutor_student", "description": "Tutor Student"},
    SELF_LEARNER: {"name": "self_learner", "description": "MySkillGuru Learner"},
}

ALLOWED_ROLES = {
    SUPERADMIN, INSTITUTE, FACULTY, INSTITUTE_STUDENT, TUTOR, TUTOR_STUDENT, SELF_LEARNER,
}

# -----------------------------------
# Validation constants
# -----------------------------------

EMAIL_REGEX = r"^[\w\.-]+@[\w\.-]+\.\w+$"
COLOR_REGEX = r"^#([A-Fa-f0-9]{6})$"
ALLOWED_LANGUAGES = ["english", "hindi"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# -----------------------------------
# Validators
# -----------------------------------

def _validate_email(email: str) -> str:
    if not email or not re.match(EMAIL_REGEX, email):
        raise ValueError("Valid email is required")
    return email.lower().strip()


def _validate_password(password: str) -> str:
    if not password or len(password) < 6:
        raise ValueError("Password must be at least 6 characters")
    return password


def _validate_color(color: str) -> str:
    if not isinstance(color, str) or not re.match(COLOR_REGEX, color):
        return "#FF7F10"
    return color


def _validate_language(language: str) -> str:
    if not language:
        return "english"
    language = language.strip().lower()
    if language not in ALLOWED_LANGUAGES:
        raise ValueError("Invalid language")
    return language


# -----------------------------------
# Create user document
# -----------------------------------

def create_user_document(data: Dict[str, Any], password_hash: str) -> Dict[str, Any]:
    fullName = data.get("fullName")
    role = data.get("role")

    if not fullName:
        raise ValueError("fullName is required")

    if role not in ALLOWED_ROLES:
        raise ValueError("Invalid role")

    email = _validate_email(data.get("email"))

    hasCOAccess = bool(data.get("hasCOAccess", data.get("hascoAccess", False)))
    hasQPGAccess = bool(data.get("hasQPGAccess", False))
    hasMySkillGuruAccess = bool(data.get("hasMySkillGuruAccess", False))

    color = _validate_color(data.get("color", "#FF7F10"))
    language = _validate_language(data.get("language", "english"))

    return {
        "fullName": fullName.strip(),
        "email": email,
        "password_hash": password_hash,

        "role": role,
        "phone": data.get("phone"),

        "is_active": bool(data.get("is_active", True)),
        "is_deleted": False,
        "last_login": None,

        "hasCOAccess": hasCOAccess,
        "hasQPGAccess": hasQPGAccess,
        "hasMySkillGuruAccess": hasMySkillGuruAccess,

        "color": color,
        "language": language,

        "created_at": _utcnow(),
        "updated_at": _utcnow(),
    }


# -----------------------------------
# Update user document
# -----------------------------------

def update_user_document(data: Dict[str, Any]) -> Dict[str, Any]:
    update_fields: Dict[str, Any] = {"updated_at": _utcnow()}

    if "fullName" in data:
        update_fields["fullName"] = data["fullName"].strip()

    if "email" in data:
        update_fields["email"] = _validate_email(data["email"])

    if "phone" in data:
        update_fields["phone"] = data["phone"]

    if "role" in data:
        if data["role"] not in ALLOWED_ROLES:
            raise ValueError("Invalid role")
        update_fields["role"] = data["role"]

    if "is_active" in data:
        update_fields["is_active"] = bool(data["is_active"])

    if "is_deleted" in data:
        update_fields["is_deleted"] = bool(data["is_deleted"])

    if "last_login" in data:
        update_fields["last_login"] = data["last_login"]

    if "hasCOAccess" in data:
        update_fields["hasCOAccess"] = bool(data["hasCOAccess"])
    elif "hascoAccess" in data:
        update_fields["hasCOAccess"] = bool(data["hascoAccess"])

    if "hasQPGAccess" in data:
        update_fields["hasQPGAccess"] = bool(data["hasQPGAccess"])

    if "hasMySkillGuruAccess" in data:
        update_fields["hasMySkillGuruAccess"] = bool(data["hasMySkillGuruAccess"])

    if "color" in data:
        update_fields["color"] = _validate_color(data["color"])

    if "language" in data:
        update_fields["language"] = _validate_language(data["language"])

    return {"$set": update_fields}


def update_password_document(password_hash: str) -> Dict[str, Any]:
    return {"$set": {"password_hash": password_hash, "updated_at": _utcnow()}}


# -----------------------------------
# GLOBAL SERIALIZER (SAFE)
# -----------------------------------

def serialize_doc(doc):
    """Recursively converts ObjectId -> string."""
    from bson import ObjectId

    if isinstance(doc, list):
        return [serialize_doc(d) for d in doc]

    if isinstance(doc, dict):
        new_doc = {}
        for k, v in doc.items():
            if isinstance(v, ObjectId):
                new_doc[k] = str(v)
            else:
                new_doc[k] = serialize_doc(v)
        return new_doc

    return doc


# -----------------------------------
# Serialize user (clean API response)
# -----------------------------------

def serialize_user(doc: Dict[str, Any]) -> Dict[str, Any] | None:
    if not doc:
        return None

    role = doc.get("role")
    role_info = ROLE_DETAILS.get(role, {})

    return {
        "id": str(doc["_id"]),
        "fullName": doc.get("fullName"),
        "email": doc.get("email"),

        "role": role,
        "role_name": role_info.get("name"),
        "role_description": role_info.get("description"),

        "hasCOAccess": bool(doc.get("hasCOAccess", False)),
        "hasQPGAccess": bool(doc.get("hasQPGAccess", False)),
        "hasMySkillGuruAccess": bool(doc.get("hasMySkillGuruAccess", False)),

        "phone": doc.get("phone"),
        "is_active": doc.get("is_active", True),

        "color": doc.get("color", "#FF7F10"),
        "language": doc.get("language", "english"),

        "last_login": doc.get("last_login"),
        "created_at": doc.get("created_at"),
    }
