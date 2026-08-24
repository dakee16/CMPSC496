"""
grader.py — in-context grading for chunk-based decomposition.

grade_chunk() evaluates a student's multi-line answer to ONE sub-question,
given the surrounding chunks and the student's accepted prefix. Tiers run
cheapest-and-most-certain first.

  Tier 1  syntactic  — does prefix + student chunk compile?
  Tier 2  execution  — does header + prefix + student chunk + reference tail
                       pass the main oracle tests? (approach-agnostic)
  (Tiers 3+4 — interface-adaptive retry and LLM judge — added next.)
"""
import ast
import re
import textwrap

from main.schemas import StepItem
from main.identity import get_resolved_entry



def _indent_body(code: str) -> str:
    return textwrap.indent(code.rstrip(), "    ")


def _header_for(problem: dict) -> str:
    resolved = get_resolved_entry(problem)
    name, params = resolved["entry_name"], resolved["params"]
    return f"def {name or 'solve'}({', '.join(params)}):"


def _assigned_names(code: str) -> set:
    """Variable names the code binds (assignments, loop targets, etc.)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    return {n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}


def _read_write_names(code: str) -> tuple:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set(), set()
    reads, writes = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            (reads if isinstance(n.ctx, ast.Load) else writes).add(n.id)
    return reads, writes


# NOTE: the legacy _adapt_tail()/_llm_judge()/JUDGE_SYSTEM helpers were
# removed. They implemented a SECOND, different grading algorithm (single
# judge, no calibration, no anti-bypass). All grading now goes through
# main.grading.grade_submission().

def grade_chunk(problem: dict, chunks: list[StepItem], index: int,
                student_code: str, accepted_prefix: list[str]) -> dict:
    """Research/offline wrapper around the ONE grading engine.

    Keeps the old outward dict shape that research_agent.py expects, but has no
    grading logic of its own: it builds a session-shaped record and delegates to
    main.grading.grade_submission(). Two implementations produced inconsistent
    research and UI verdicts — notably a different (single-judge) Tier 4 and a
    now-removed in-process exec() path. There is only one engine now.
    """
    from main.grading import grade_submission

    session = {
        "slug": problem.get("slug", ""), "title": problem.get("title", ""),
        "description": problem.get("description", ""),
        "solution": problem.get("solution", ""),
        "header": _header_for(problem),
        "chunks": [{"step_id": c.step_id, "prompt": c.prompt,
                    "expected_type": c.expected_type, "reference": c.reference or ""}
                   for c in chunks],
        "index": index,
        "accepted": [{"step_id": "", "code": a, "provenance": "student"}
                     for a in accepted_prefix],
    }
    r = grade_submission(session, student_code)
    return {"correct": r.verdict == "correct", "tier": r.tier,
            "reason": r.student_reason, "failures": r.failures,
            "verdict": r.verdict, "deterministic": r.deterministic,
            "indeterminate": r.verdict == "indeterminate",
            "reason_code": r.reason_code, "divergent": r.divergent}
