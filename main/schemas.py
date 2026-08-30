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
    divergent: bool = False   # correct, but NOT the canonical line - offer replan

# ── answer-checking types ────────────────────────────────────────────────
# Grading verdicts were previously a bare {"correct": bool, "tier": str} dict,
# which could not express "we could not tell" - so infrastructure failure was
# indistinguishable from a wrong student. These types make that distinction
# representable, and make every verdict carry its provenance.

GradeVerdict = Literal["correct", "incorrect", "indeterminate"]

# Which stage produced the verdict. execution-* are deterministic (a real run
# decided it); llm-judge is explicitly an opinion; system means we never got as
# far as judging the student at all.
GradeTier = Literal[
    "syntax",                # did not parse
    "policy",                # rejected by the safety policy before running
    "execution-reference",   # ran with the trusted reference tail
    "execution-adapted",     # ran with a CALIBRATED adapted tail
    "execution-final",       # last chunk: whole function, no borrowed tail
    "llm-judge",             # execution could not attribute fault
    "system",                # precondition/infrastructure failure
]

ExecutionOutcome = Literal[
    "pass",
    "wrong_output",       # ran cleanly, produced the wrong answer
    "runtime_error",      # student code raised
    "timeout",            # wall-clock/CPU limit hit
    "policy_violation",   # blocked by the AST safety policy
    "harness_error",      # OUR failure, never evidence about the student
]

# How an accepted answer got into the session's prefix.
AnswerProvenance = Literal["student", "revealed_reference"]


class ExecutionResult(BaseModel):
    """One classified run of candidate code. Replaces the ambiguous
    ok/fraction/error blob for answer-checking decisions."""
    outcome: ExecutionOutcome
    passed: int = 0
    total: int = 0
    # Internal only - never serialised to the browser.
    failures: List[dict] = []
    internal_error: Optional[str] = None


class GradeResult(BaseModel):
    """The single shape every grading path returns."""
    verdict: GradeVerdict
    tier: GradeTier
    # True only when a real execution decided it. An LLM opinion is never
    # deterministic, and neither is a system failure.
    deterministic: bool
    # Safe to show a student: no stack traces, no oracle contents, no reference.
    student_reason: str
    # Stable machine-readable code for logs/analytics, e.g. "blank_answer".
    reason_code: str
    # Internal diagnostics; stripped at the API boundary.
    failures: List[dict] = []
    internal_detail: Optional[str] = None
    # Correct, but not the canonical approach - the caller may offer a replan.
    divergent: bool = False
    # Infrastructure failure must not cost the student an attempt.
    consume_attempt: bool = True
    execution_outcome: Optional[ExecutionOutcome] = None
