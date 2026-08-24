"""Versioned benchmark case schema.

A case's expected label must be derivable WITHOUT the grader. Every generator
in corpus.py therefore proves its label by executing the assembled candidate
against the oracle directly, or by construction (blank/policy/no-op).
"""
from typing import List, Literal, Optional

from pydantic import BaseModel

SCHEMA_VERSION = "1.0.1"

ExpectedLabel = Literal["correct", "incorrect", "ambiguous"]
CreationSource = Literal["deterministic", "manually_reviewed", "model_generated"]

AnswerCategory = Literal[
    "exact_reference", "renamed_variables", "alternate_implementation",
    "divergent_interface", "clean_logical_error", "boundary_off_by_one",
    "no_op", "blank", "syntax_error", "runtime_error", "timeout",
    "policy_violation", "adapter_bypass", "wrong_after_divergent_prefix",
    "wrong_after_revealed_prefix", "ambiguous_tier4",
]


class AcceptedItem(BaseModel):
    code: str
    provenance: Literal["student", "revealed_reference"] = "student"


class BenchmarkCase(BaseModel):
    schema_version: str = SCHEMA_VERSION
    case_id: str
    slug: str
    content_hash: str
    decomposition_id: str
    chunk_index: int
    accepted_prefix: List[AcceptedItem] = []
    student_answer: str
    expected: ExpectedLabel
    category: AnswerCategory
    source: CreationSource
    rationale: str


class CaseResult(BaseModel):
    case_id: str
    slug: str
    chunk_index: int
    category: AnswerCategory
    expected: ExpectedLabel
    actual: str
    tier: str
    deterministic: bool
    reason_code: str
    divergent: bool = False
    adapter_attempted: bool = False
    adapter_calibrated: bool = False
    antibypass_rejected: bool = False
    judge_agreement: Optional[bool] = None
    judge_confidence: Optional[float] = None
    latency_ms: float = 0.0
    consumed_attempt: bool = True
    system_error: Optional[str] = None
