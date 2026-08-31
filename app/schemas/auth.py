import re
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.user import EMAIL_REGEX, _validate_password


def _valid_email(v: str) -> str:
    if not v or not re.match(EMAIL_REGEX, v):
        raise ValueError("Valid email is required")
    return v.lower().strip()


class _RegisterBase(BaseModel):
    # Every _register_* handler in auth.py accepts extra optional passthrough
    # fields (institute's short_name/institute_code, faculty's designation/
    # qualification, student's father_name/dob/..., etc.) and forwards them
    # via **data into the existing app/models/*.py builder functions, which
    # already do their own .get()-based defaulting. Modeling every one of
    # those here would duplicate that layer for no validation benefit, so
    # extra="allow" preserves the original flexibility; only the fields that
    # actually gate behavior (role, required IDs, credentials) are typed.
    model_config = ConfigDict(extra="allow")

    fullName: str
    email: str
    password: str
    phone: Optional[str] = None
    color: Optional[str] = None
    language: Optional[str] = None

    @field_validator("fullName")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("fullName is required")
        return v

    @field_validator("email")
    @classmethod
    def valid_email(cls, v: str) -> str:
        return _valid_email(v)

    @field_validator("password")
    @classmethod
    def valid_password(cls, v: str) -> str:
        return _validate_password(v)


class RegisterSuperadmin(_RegisterBase):
    role: Literal["superadmin"]


class RegisterInstitute(_RegisterBase):
    role: Literal["institute"]
    institute: Dict[str, Any]
    hasCOAccess: Optional[bool] = False
    hasQPGAccess: Optional[bool] = False
    hasMySkillGuruAccess: Optional[bool] = False

    @field_validator("institute")
    @classmethod
    def institute_needs_name(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if not v or not v.get("institute_name"):
            raise ValueError("Institute details are required in the 'institute' field")
        return v


class RegisterFaculty(_RegisterBase):
    role: Literal["faculty"]
    school_id: str

    @field_validator("color")
    @classmethod
    def no_manual_color(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            raise ValueError("Faculty color is inherited from institute and cannot be set manually")
        return v


class RegisterInstituteStudent(_RegisterBase):
    role: Literal["institute_student"]
    school_id: str
    programme_id: str
    roll_no: str
    # Institute students get an auto-generated college email/password, so
    # (unlike every other role) a self-supplied password is optional here.
    password: Optional[str] = None

    @field_validator("password")
    @classmethod
    def valid_password(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        return _validate_password(v)


class RegisterTutor(_RegisterBase):
    role: Literal["tutor"]


class RegisterTutorStudent(_RegisterBase):
    role: Literal["tutor_student"]


class RegisterSelfLearner(_RegisterBase):
    role: Literal["self_learner"]


RegisterPayload = Annotated[
    Union[
        RegisterSuperadmin,
        RegisterInstitute,
        RegisterFaculty,
        RegisterInstituteStudent,
        RegisterTutor,
        RegisterTutorStudent,
        RegisterSelfLearner,
    ],
    Field(discriminator="role"),
]


class LoginRequest(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def valid_email(cls, v: str) -> str:
        return _valid_email(v)

    @field_validator("password")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v:
            raise ValueError("Password is required")
        return v


class BulkStudentEnrollmentRequest(BaseModel):
    # Individual rows deliberately stay untyped dicts: _register_institute_student
    # already validates each row's required fields and reports per-row
    # failures without aborting the rest of the batch — an all-or-nothing
    # Pydantic model per row would break that partial-success behavior.
    students: List[Dict[str, Any]]

    @field_validator("students")
    @classmethod
    def non_empty(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not v:
            raise ValueError("students array is required")
        return v
