from pydantic import BaseModel
from typing import List, Literal, Optional

ExpectedType = Literal["code", "int", "float", "bool", "string"]


class StepItem(BaseModel):
    question_id: str
    step_id: str
    prompt: str
    expected_type: ExpectedType = "string"
    skill: str = "unspecified"
    rubric: Optional[str] = None


class DecomposeOutput(BaseModel):
    steps: List[StepItem]


class EvalResult(BaseModel):
    correct: bool
    short_reason: str
    correct_answer: Optional[str] = None