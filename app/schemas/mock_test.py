from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict


class MockTestCreateRequest(BaseModel):
    # app.models.mock_test.build_create_document already does full defensive
    # validation/defaulting (invalid difficulty/questionTypes silently fall
    # back to safe defaults rather than erroring) — extra="allow" plus loose
    # optional typing here preserves that, while still guaranteeing basic
    # shapes (e.g. questionCount really is an int) before that runs.
    model_config = ConfigDict(extra="allow")

    # mode: "subject" (default, original practice-test flow — untouched) |
    # "roadmap" (reuses the Auto Test config shape across a week range).
    mode: str = "subject"

    subject_id: Optional[str] = None
    subjectId: Optional[str] = None
    subjectName: Optional[str] = None
    topic: Optional[str] = None
    instructions: Optional[str] = None
    questionCount: Optional[int] = None
    marksPerQuestion: Optional[float] = None
    duration: Optional[int] = None
    difficulty: Optional[str] = None
    negativeMarking: Optional[bool] = None
    negativeMarks: Optional[float] = None
    questionTypes: Optional[Any] = None
    scheduleDate: Optional[str] = None

    # roadmap mode only
    roadmap_id: Optional[str] = None
    week_start: Optional[int] = None
    week_end: Optional[int] = None
    mcq_percent: Optional[float] = None
    subjective_percent: Optional[float] = None
    practical_percent: Optional[float] = None
    custom_prompt: Optional[str] = None


class MockTestSubmitRequest(BaseModel):
    answers: Dict[str, Any]
