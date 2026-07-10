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
from sandbox import get_oracle_tests, passes_tests, _extract_signature
from main.ollama_client import chat

MODEL = "qwen2.5:7b-instruct"


def _indent_body(code: str) -> str:
    return textwrap.indent(code.rstrip(), "    ")


def _header_for(problem: dict) -> str:
    name, params = _extract_signature(problem.get("solution", ""))
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


def _adapt_tail(problem: dict, header: str, upto: str,
                reference_tail: str, student_outputs: set) -> str:
    """Regenerate the remaining body to complete the STUDENT's code. May
    restructure freely, but must build on the student's produced values."""
    desc = problem.get("description") or problem.get("title", "")
    sys = ("You complete a partially-written Python function. Return ONLY the "
           "remaining body lines — no function header, no markdown, no explanation.")
    user = (
        f"PROBLEM:\n{desc}\n\n"
        f"The student has written this so far:\n{header}\n{textwrap.indent(upto, '    ')}\n\n"
        f"For reference, the remaining logic was originally written as:\n{reference_tail}\n\n"
        f"Variables the student's code produced (build on these — use them, do NOT "
        f"recompute or reassign them): {', '.join(sorted(student_outputs)) or '(none)'}.\n"
        "Write the REMAINING body lines that finish the function correctly, continuing "
        "from the student's code. Structure it however works, but it must return the "
        "correct final answer using the student's variables.\n"
        "Output only the remaining body lines at column 0 (no header, no extra indent)."
    )
    raw = chat(MODEL, sys, [{"role": "user", "content": user}], temperature=0.2)
    raw = re.sub(r'```[\w]*\n?', '', raw)
    return re.sub(r'```', '', raw).strip()

JUDGE_SYSTEM = """\
You are a strict but fair code evaluator. A student answered one sub-question
of a larger coding problem. Execution-based grading could not give a definitive
verdict. Your job: decide correct or incorrect, and give the student useful
feedback. Be concise. Never reveal the reference implementation.
"""


def _llm_judge(problem: dict, header: str, student_upto: str,
               chunk: StepItem, full_code: str,
               execution_failures: list | None = None) -> dict:
    """Tier 4: LLM semantic judge. Fires when execution can't settle the verdict."""
    desc = problem.get("description") or problem.get("title", "")
    fail_note = ""
    if execution_failures:
        fail_note = "\nExecution found these failures (so the code is not fully correct):\n"
        fail_note += "\n".join(f"  input {f['input']} → expected {f['expected']}, got {f['got']}"
                               for f in execution_failures[:3])

    user = (
        f"PROBLEM:\n{desc}\n\n"
        f"SUB-QUESTION the student was answering:\n{chunk.prompt}\n\n"
        f"Function assembled so far:\n{header}\n{textwrap.indent(student_upto, '    ')}\n"
        f"{fail_note}\n"
        "Does the student's code correctly answer the sub-question?\n"
        "Consider: does it compile, does it logically solve the sub-question, "
        "and would it allow a correct completion of the overall function?\n\n"
        'Return JSON only: {"correct": true/false, "reason": "one sentence for the student"}'
    )
    raw = chat(MODEL, JUDGE_SYSTEM, [{"role": "user", "content": user}],
               temperature=0.0, fmt="json")
    try:
        from main.run_phase1 import parse_json
        data = parse_json(raw)
        return {"correct": bool(data.get("correct")), "tier": "llm-judge",
                "reason": data.get("reason", "No reason given."), "failures": []}
    except Exception:
        return {"correct": False, "tier": "llm-judge",
                "reason": "Could not parse judge response.", "failures": []}


def grade_chunk(problem: dict, chunks: list[StepItem], index: int,
                student_code: str, accepted_prefix: list[str]) -> dict:
    """Grade the student's answer to chunks[index].

    problem          — the problem dict (needs 'solution' for the oracle).
    chunks           — full ordered list of StepItem chunks (carry .reference).
    index            — which chunk is being answered.
    student_code     — the student's multi-line answer for THIS chunk.
    accepted_prefix  — the student's accepted answers to chunks[0..index-1],
                       each a string, in order.

    Returns {"correct": bool, "tier": str, "reason": str, "failures": [...]}.
    """
    header = _header_for(problem)
    student_code = (student_code or "").strip()

    if not student_code:
        return {"correct": False, "tier": "syntactic",
                "reason": "No answer submitted.", "failures": []}

    # ── Tier 1: syntactic — prefix + this chunk must compile ──
    prefix_body = "\n".join(p.rstrip() for p in accepted_prefix if p.strip())
    upto = (prefix_body + "\n" + student_code).strip() if prefix_body else student_code
    candidate_syntax = header + "\n" + _indent_body(upto)
    try:
        compile(candidate_syntax, "<chunk>", "exec")
    except SyntaxError as e:
        return {"correct": False, "tier": "syntactic",
                "reason": f"Code doesn't parse: {e.msg} (line {e.lineno}).",
                "failures": []}

    # ── Tier 2: execution — full candidate with reference tail must pass oracle ──
    tail_refs = [(chunks[j].reference or "") for j in range(index + 1, len(chunks))]
    tail_body = "\n".join(r.rstrip() for r in tail_refs if r.strip())
    full_body = "\n".join(b for b in (upto, tail_body) if b.strip())
    full_code = header + "\n" + _indent_body(full_body)

    # Ensure we have a solution for oracle generation — fetch from Supabase if missing
    if not problem.get("solution", "").strip():
        try:
            from supabase import create_client
            import os
            sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
            res_db = sb.table("problems").select("solution").eq(
                "slug", problem.get("slug", "")).single().execute()
            if res_db.data and res_db.data.get("solution"):
                problem = {**problem, "solution": res_db.data["solution"]}
        except Exception:
            pass

    tests = get_oracle_tests(problem)
    res = None
    if not tests:
        try:
            compile(full_code, "<smoke>", "exec")
            ns = {}
            exec(compile(full_code, "<smoke>", "exec"), ns)  # noqa: S102
            # If it executes without error, we can't verify correctness — defer to judge
        except SyntaxError as e:
            return {"correct": False, "tier": "syntactic",
                    "reason": f"Code doesn't compile: {e.msg} (line {e.lineno}).",
                    "failures": []}
        except Exception:
            pass  # runtime errors during exec are expected (no inputs yet)
        return _llm_judge(problem, header, upto, chunks[index], full_code)

    m = re.search(r'def\s+(\w+)', full_code)
    res = passes_tests(full_code, tests, entry_name=m.group(1) if m else None)
    if res["ok"] and res["fraction"] == 1.0:
        return {"correct": True, "tier": "execution",
                "reason": "Correct — passes all tests (any valid approach accepted).",
                "failures": []}
        
        
    # ── Tier 3: interface-adaptive retry ──
    if index < len(chunks) - 1 and tail_body.strip():
        student_outputs = _assigned_names(student_code)
        adapted_tail = _adapt_tail(problem, header, upto, tail_body, student_outputs)
        if adapted_tail:
            reads, writes = _read_write_names(adapted_tail)
            if (reads & student_outputs) and not (writes & student_outputs):
                adapted_full = header + "\n" + _indent_body(
                    "\n".join(b for b in (upto, adapted_tail) if b.strip()))
                m2 = re.search(r'def\s+(\w+)', adapted_full)
                res2 = passes_tests(adapted_full, tests, entry_name=m2.group(1) if m2 else None)
                if res2["ok"] and res2["fraction"] == 1.0:
                    return {"correct": True, "tier": "execution-adapted",
                            "reason": "Correct — your approach differs from the model "
                                      "solution, but it works once the rest builds on it.",
                            "failures": [], "_adapted_tail": adapted_tail}

    # Tier 4: execution exhausted — fall back to LLM judge for final verdict.
    return _llm_judge(problem, header, upto, chunks[index], full_code,
                      execution_failures=res.get("failures", []) if res else [])