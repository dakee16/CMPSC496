from pydantic import BaseModel
from typing import List, Literal, Optional

ExpectedType = Literal["code", "int", "float", "bool", "string"]


class StepItem(BaseModel):
    question_id: str          # e.g. "Q1"
    step_id: str              # e.g. "Step 1"
    prompt: str               # the question shown to the student
    expected_type: ExpectedType = "string"
    skill: str = "unspecified"
    rubric: Optional[str] = None


class DecomposeOutput(BaseModel):
    steps: List[StepItem]


class EvalResult(BaseModel):
    correct: bool
    short_reason: str         # used directly as the hint when wrong
    correct_answer: Optional[str] = None  # revealed after two failed attempts