from typing import Any, Dict, Optional

from pydantic import BaseModel, field_validator, model_validator


class PreAssessmentRequest(BaseModel):
    subject: str

    @field_validator("subject")
    @classmethod
    def valid_subject(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("subject is required")
        if len(v) > 200:
            raise ValueError("subject must be 200 characters or fewer")
        return v


class CreateRoadmapRequest(BaseModel):
    subject: str
    goal: Optional[str] = ""
    skill_level: str = "Beginner"
    daily_study_time: str = "1 Hour"
    revision_frequency: str = "Every Week"
    assessment_score: Optional[Any] = None
    doc_id: Optional[str] = None
    custom_instruction: Optional[str] = None

    @field_validator("subject")
    @classmethod
    def valid_subject(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("subject is required")
        if len(v) > 200:
            raise ValueError("subject must be 200 characters or fewer")
        return v

    @field_validator("custom_instruction")
    @classmethod
    def valid_custom_instruction(cls, v: Optional[str]) -> Optional[str]:
        v = (v or "").strip()
        if len(v) > 1000:
            raise ValueError("custom_instruction must be 1000 characters or fewer")
        return v or None

    @field_validator("goal")
    @classmethod
    def valid_goal(cls, v: Optional[str]) -> str:
        v = (v or "").strip()
        if len(v) > 500:
            raise ValueError("goal must be 500 characters or fewer")
        return v


class UpdateSubtopicRequest(BaseModel):
    subtopic_key: str
    completed: bool = True

    @field_validator("subtopic_key")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v:
            raise ValueError("subtopic_key is required")
        return v


class SubmitQuizRequest(BaseModel):
    week: int = 1
    answers: Dict[str, Any] = {}


class GenerateAutoTestRequest(BaseModel):
    week: int = 1
    mcq_percent: float = 100
    subjective_percent: float = 0
    practical_percent: float = 0
    question_count: int = 10
    custom_prompt: Optional[str] = None

    @field_validator("question_count")
    @classmethod
    def valid_count(cls, v: int) -> int:
        if v < 1 or v > 50:
            raise ValueError("question_count must be between 1 and 50")
        return v

    @field_validator("mcq_percent", "subjective_percent", "practical_percent")
    @classmethod
    def valid_percent(cls, v: float) -> float:
        if v < 0 or v > 100:
            raise ValueError("percentages must be between 0 and 100")
        return v

    @field_validator("custom_prompt")
    @classmethod
    def valid_custom_prompt(cls, v: Optional[str]) -> Optional[str]:
        v = (v or "").strip()
        if len(v) > 500:
            raise ValueError("custom_prompt must be 500 characters or fewer")
        return v or None

    @model_validator(mode="after")
    def percentages_sum_to_100(self) -> "GenerateAutoTestRequest":
        total = self.mcq_percent + self.subjective_percent + self.practical_percent
        if round(total) != 100:
            raise ValueError(f"mcq_percent + subjective_percent + practical_percent must sum to 100 (got {total})")
        return self


class EvaluatePracticeAnswerRequest(BaseModel):
    week: int = 1
    subtopic_idx: int = 0
    question_idx: int
    student_answer: str

    @field_validator("student_answer")
    @classmethod
    def valid_answer(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("student_answer is required")
        if len(v) > 4000:
            raise ValueError("student_answer must be 4000 characters or fewer")
        return v
