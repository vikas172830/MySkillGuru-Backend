from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, field_validator

from app.models.user import _validate_password

# All PUT /profile* endpoints are genuine partial updates: only fields
# present get processed, unknown fields are silently ignored by the
# handlers, and the caller's *server-side* role (not client input) decides
# which fields matter. extra="allow" preserves that flexibility; these
# schemas add real type-safety for the well-known fields without forcing
# any of them to be required.


class ProfileUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    fullName: Optional[str] = None
    phone: Optional[str] = None
    profileImage: Optional[str] = None
    language: Optional[str] = None
    color: Optional[str] = None


class ChangePasswordRequest(BaseModel):
    currentPassword: str
    newPassword: str

    @field_validator("currentPassword")
    @classmethod
    def current_not_blank(cls, v: str) -> str:
        if not v:
            raise ValueError("currentPassword is required")
        return v

    @field_validator("newPassword")
    @classmethod
    def new_password_strength(cls, v: str) -> str:
        # NOTE: the original endpoint had no strength check at all on
        # newPassword (any non-empty string was accepted) — added here since
        # it's a direct, low-risk improvement matching /register's rule.
        return _validate_password(v)


class InstituteUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    institute: Dict[str, Any] = {}
    hasCOAccess: Optional[bool] = None
    hasQPGAccess: Optional[bool] = None
    hasMySkillGuruAccess: Optional[bool] = None


class FacultyUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    fullName: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    password: Optional[str] = None
    designation: Optional[str] = None
    qualification: Optional[str] = None
    experience_years: Optional[int] = None
    specialization: Optional[str] = None
    employee_code: Optional[str] = None
    joining_date: Optional[str] = None
    bio: Optional[str] = None
    is_active: Optional[bool] = None
    school_id: Optional[str] = None


class TutorUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    fullName: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    is_active: Optional[bool] = None


class SelfLearnerUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    fullName: Optional[str] = None
    phone: Optional[str] = None
    is_active: Optional[bool] = None
