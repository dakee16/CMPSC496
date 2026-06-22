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
    canonical: Optional[str] = None   # ONE runnable line for this step
    indent: int = 0                   # block depth (0=def, 1=body, 2=inside loop/if)
    reference: Optional[str] = None 

class DecomposeOutput(BaseModel):
    steps: List[StepItem]


class EvalResult(BaseModel):
    correct: bool
    short_reason: str
    correct_answer: Optional[str] = None
    divergent: bool = False   # correct, but NOT the canonical line — offer replan