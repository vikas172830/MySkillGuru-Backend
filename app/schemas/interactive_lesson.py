"""Versioned contract for AI-generated, domain-aware guided lessons.

The model supplies content-only JSON. The frontend owns every interaction;
arbitrary HTML, JavaScript, and model-generated component code are never run.
"""

from typing import Annotated, List, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator


class LessonModel(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


class LessonMission(LessonModel):
    goal: str = Field(min_length=1, max_length=500)
    whyItMatters: str = Field(min_length=1, max_length=1000)
    estimatedMinutes: int = Field(ge=5, le=240)
    successCriteria: List[str] = Field(min_length=2, max_length=6)


class AnchorExample(LessonModel):
    title: str = Field(min_length=1, max_length=140)
    context: str = Field(min_length=1, max_length=1200)
    whyChosen: str = Field(min_length=1, max_length=800)


class KeyTerm(LessonModel):
    term: str = Field(min_length=1, max_length=100)
    meaning: str = Field(min_length=1, max_length=600)
    example: str = Field(default="", max_length=500)


class LessonSummary(LessonModel):
    keyTakeaways: List[str] = Field(min_length=3, max_length=8)
    masteryChecklist: List[str] = Field(min_length=2, max_length=8)
    nextStep: str = Field(min_length=1, max_length=600)


class VisualGridRow(LessonModel):
    label: str = Field(min_length=1, max_length=80)
    values: List[str] = Field(min_length=1, max_length=6)


class GridVisualAid(LessonModel):
    kind: Literal["grid", "table"]
    title: str = Field(min_length=1, max_length=140)
    purpose: str = Field(min_length=1, max_length=700)
    columnHeaders: List[str] = Field(min_length=1, max_length=6)
    rows: List[VisualGridRow] = Field(min_length=1, max_length=6)
    interactionPrompt: str = Field(min_length=1, max_length=500)
    caption: str = Field(min_length=1, max_length=700)

    @model_validator(mode="after")
    def rows_match_columns(self):
        column_count = len(self.columnHeaders)
        if any(len(row.values) != column_count for row in self.rows):
            raise ValueError("every visual row must match the number of column headers")
        return self


class VisualItem(LessonModel):
    label: str = Field(min_length=1, max_length=100)
    value: str = Field(default="", max_length=160)
    description: str = Field(min_length=1, max_length=500)
    level: int = Field(default=0, ge=0, le=5)


class ItemVisualAid(LessonModel):
    kind: Literal["sequence", "flow", "timeline", "hierarchy"]
    title: str = Field(min_length=1, max_length=140)
    purpose: str = Field(min_length=1, max_length=700)
    items: List[VisualItem] = Field(min_length=2, max_length=10)
    interactionPrompt: str = Field(min_length=1, max_length=500)
    caption: str = Field(min_length=1, max_length=700)

    @model_validator(mode="after")
    def hierarchy_levels_are_ordered(self):
        if self.kind != "hierarchy":
            return self
        if self.items[0].level != 0:
            raise ValueError("a hierarchy must start at level 0")
        for previous, current in zip(self.items, self.items[1:]):
            if current.level > previous.level + 1:
                raise ValueError("hierarchy levels cannot skip a parent level")
        return self


VisualAid = Annotated[
    Union[GridVisualAid, ItemVisualAid],
    Field(discriminator="kind"),
]


class ConceptBlock(LessonModel):
    type: Literal["concept"]
    title: str = Field(min_length=1, max_length=140)
    simpleExplanation: str = Field(min_length=1, max_length=2500)
    whyItMatters: str = Field(min_length=1, max_length=1200)
    realWorldConnection: str = Field(default="", max_length=1200)


class MentalModelBlock(LessonModel):
    type: Literal["mental_model"]
    title: str = Field(min_length=1, max_length=140)
    analogy: str = Field(min_length=1, max_length=1200)
    explanation: str = Field(min_length=1, max_length=1800)
    remember: str = Field(min_length=1, max_length=500)


class WorkedStep(LessonModel):
    label: str = Field(min_length=1, max_length=120)
    action: str = Field(min_length=1, max_length=800)
    explanation: str = Field(min_length=1, max_length=1200)
    result: str = Field(default="", max_length=600)


class WorkedExampleBlock(LessonModel):
    type: Literal["worked_example"]
    title: str = Field(min_length=1, max_length=140)
    scenario: str = Field(min_length=1, max_length=1600)
    exampleReason: str = Field(min_length=1, max_length=800)
    steps: List[WorkedStep] = Field(min_length=2, max_length=10)
    outcome: str = Field(min_length=1, max_length=1000)


class WalkthroughStep(LessonModel):
    focus: str = Field(min_length=1, max_length=160)
    action: str = Field(min_length=1, max_length=800)
    state: str = Field(default="", max_length=900)
    why: str = Field(min_length=1, max_length=1000)


class GuidedWalkthroughBlock(LessonModel):
    type: Literal["guided_walkthrough"]
    title: str = Field(min_length=1, max_length=140)
    purpose: str = Field(min_length=1, max_length=1000)
    example: str = Field(min_length=1, max_length=1000)
    exampleReason: str = Field(min_length=1, max_length=800)
    steps: List[WalkthroughStep] = Field(min_length=2, max_length=12)
    conclusion: str = Field(min_length=1, max_length=1200)


class FormulaStep(LessonModel):
    label: str = Field(min_length=1, max_length=120)
    expression: str = Field(min_length=1, max_length=500)
    result: str = Field(default="", max_length=300)
    explanation: str = Field(min_length=1, max_length=1000)


class FormulaWalkthroughBlock(LessonModel):
    type: Literal["formula_walkthrough"]
    title: str = Field(min_length=1, max_length=140)
    formula: str = Field(min_length=1, max_length=500)
    explanation: str = Field(min_length=1, max_length=1200)
    steps: List[FormulaStep] = Field(min_length=1, max_length=10)


class ComparisonColumn(LessonModel):
    heading: str = Field(min_length=1, max_length=100)
    points: List[str] = Field(min_length=1, max_length=8)


class ComparisonBlock(LessonModel):
    type: Literal["comparison"]
    title: str = Field(min_length=1, max_length=140)
    columns: List[ComparisonColumn] = Field(min_length=2, max_length=4)
    conclusion: str = Field(min_length=1, max_length=800)


class CodeStep(LessonModel):
    line: int = Field(ge=1, le=500)
    focus: str = Field(min_length=1, max_length=160)
    state: str = Field(default="", max_length=600)
    explanation: str = Field(min_length=1, max_length=1000)


class CodeWalkthroughBlock(LessonModel):
    type: Literal["code_walkthrough"]
    title: str = Field(min_length=1, max_length=140)
    language: str = Field(default="text", max_length=40)
    code: str = Field(min_length=1, max_length=6000)
    purpose: str = Field(min_length=1, max_length=1000)
    steps: List[CodeStep] = Field(min_length=1, max_length=20)
    outcome: str = Field(min_length=1, max_length=800)


class MistakeItem(LessonModel):
    mistake: str = Field(min_length=1, max_length=700)
    whyItHappens: str = Field(min_length=1, max_length=900)
    correction: str = Field(min_length=1, max_length=900)


class CommonMistakesBlock(LessonModel):
    type: Literal["common_mistakes"]
    title: str = Field(min_length=1, max_length=140)
    items: List[MistakeItem] = Field(min_length=1, max_length=6)


class PracticalActivityBlock(LessonModel):
    type: Literal["practical_activity"]
    title: str = Field(min_length=1, max_length=140)
    instructions: str = Field(min_length=1, max_length=1200)
    steps: List[str] = Field(min_length=2, max_length=10)
    expectedOutcome: str = Field(min_length=1, max_length=1000)
    reflectionQuestion: str = Field(min_length=1, max_length=700)


class CaseStudyBlock(LessonModel):
    type: Literal["case_study"]
    title: str = Field(min_length=1, max_length=140)
    scenario: str = Field(min_length=1, max_length=1800)
    facts: List[str] = Field(min_length=2, max_length=8)
    decision: str = Field(min_length=1, max_length=1000)
    recommendedApproach: str = Field(min_length=1, max_length=1200)
    reasoning: str = Field(min_length=1, max_length=1600)


class DebuggingLabBlock(LessonModel):
    type: Literal["debugging_lab"]
    title: str = Field(min_length=1, max_length=140)
    scenario: str = Field(min_length=1, max_length=1000)
    brokenExample: str = Field(min_length=1, max_length=4000)
    hints: List[str] = Field(min_length=1, max_length=5)
    solution: str = Field(min_length=1, max_length=4000)
    explanation: str = Field(min_length=1, max_length=1200)


class ParameterOption(LessonModel):
    value: str = Field(min_length=1, max_length=100)
    label: str = Field(default="", max_length=120)
    effect: str = Field(min_length=1, max_length=1200)


class ParameterExplorerBlock(LessonModel):
    type: Literal["parameter_explorer"]
    title: str = Field(min_length=1, max_length=140)
    prompt: str = Field(min_length=1, max_length=1000)
    parameterLabel: str = Field(min_length=1, max_length=100)
    options: List[ParameterOption] = Field(min_length=2, max_length=8)


class AssessmentBlock(LessonModel):
    title: str = Field(min_length=1, max_length=140)
    question: str = Field(min_length=1, max_length=1500)
    options: List[str] = Field(min_length=2, max_length=6)
    correctAnswerIndex: int = Field(ge=0, le=5)
    explanation: str = Field(min_length=1, max_length=1500)

    @model_validator(mode="after")
    def answer_points_to_an_option(self):
        if self.correctAnswerIndex >= len(self.options):
            raise ValueError("correctAnswerIndex must point to an existing option")
        return self


class PredictionBlock(AssessmentBlock):
    type: Literal["prediction"]


class QuickCheckBlock(AssessmentBlock):
    type: Literal["quick_check"]


LessonBlock = Annotated[
    Union[
        ConceptBlock,
        MentalModelBlock,
        WorkedExampleBlock,
        GuidedWalkthroughBlock,
        FormulaWalkthroughBlock,
        ComparisonBlock,
        CodeWalkthroughBlock,
        CommonMistakesBlock,
        PracticalActivityBlock,
        CaseStudyBlock,
        DebuggingLabBlock,
        ParameterExplorerBlock,
        PredictionBlock,
        QuickCheckBlock,
    ],
    Field(discriminator="type"),
]

lesson_block_adapter = TypeAdapter(LessonBlock)


class InteractiveLesson(LessonModel):
    schemaVersion: Literal[4] = 4
    language: Literal["English"] = "English"
    domain: Literal["technology", "business", "quantitative", "science", "humanities", "general"]
    title: str = Field(min_length=1, max_length=180)
    mission: LessonMission
    prerequisites: List[str] = Field(min_length=1, max_length=6)
    learningOutcomes: List[str] = Field(min_length=2, max_length=6)
    keyTerms: List[KeyTerm] = Field(min_length=2, max_length=10)
    anchorExample: AnchorExample
    visualAid: VisualAid
    blocks: List[LessonBlock] = Field(min_length=5, max_length=12)
    summary: LessonSummary

    @model_validator(mode="after")
    def has_guidance_and_retrieval(self):
        types = {block.type for block in self.blocks}
        if not ({"worked_example", "guided_walkthrough"} & types):
            raise ValueError("lesson requires a worked example or guided walkthrough")
        if not ({"quick_check", "prediction"} & types):
            raise ValueError("lesson requires a prediction or quick check")
        return self
