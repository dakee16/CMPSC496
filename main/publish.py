"""
publish.py — prepare a teacher's problems BEFORE any student sees them.

This is the fix for the failure that used to land in a student's face. Oracle
generation, mutation validation and decomposition all used to run when a STUDENT
pressed Start: slow, paid, and roughly a coin-flip on whether it worked, with an
HTTP 500 as the failure mode. All of it now happens once, at upload, in front of
the teacher — who can actually do something about it.

Two consequences worth stating plainly:

  * A problem is `ready` only when it has a STRONG mutation-validated oracle AND
    a gated decomposition. Anything less is never offered to a student, because
    an unready problem cannot be graded and would dead-end them.
  * The student request path no longer generates anything. It reads a prepared
    decomposition and a cached oracle, or it refuses.

Preparation is SLOW by nature (mutation testing, LLM counterexample search), so
prepare_assignment_stream() yields one event per problem as it finishes rather
than making the teacher stare at a blank page for minutes.
"""
from .gates import assert_serveable
from .identity import content_hash, get_resolved_entry
from .schemas import StepItem


def _reason(exc: Exception) -> str:
    """A teacher-readable sentence. Never a traceback — this reaches a UI."""
    from .run_phase1 import (DecompositionUnavailableError, NoOracleTestsError,
                             OracleNotStrongError)
    if isinstance(exc, OracleNotStrongError):
        return ("The generated tests were not strong enough to grade this "
                "reliably. Try making the problem statement more specific about "
                "edge cases.")
    if isinstance(exc, NoOracleTestsError):
        return ("No usable test cases could be generated. This usually means the "
                "inputs aren't simple values (lists, numbers, strings).")
    if isinstance(exc, DecompositionUnavailableError):
        return ("Could not split this problem into steps that hold together. "
                "You can split it yourself, or simplify the solution.")
    msg = str(exc).splitlines()[0] if str(exc).strip() else exc.__class__.__name__
    return msg[:300]


def prepare_problem(problem: dict) -> dict:
    """Make one problem student-ready. Returns {slug, ready, chunks, error}.

    Never raises: a failure here is ordinary teacher feedback, not a server
    fault, and one bad problem must not abort an upload of twenty."""
    from tests.sandbox import get_oracle_tests, is_oracle_strong
    from .run_phase1 import get_chunk_decomposition

    slug = problem.get("slug", "?")

    def fail(msg):
        return {"slug": slug, "ready": False, "chunks": 0, "error": msg}

    if not (problem.get("solution") or "").strip():
        return fail("no solution provided")

    # The entry point must be resolvable and actually runnable, or every later
    # stage is measuring the wrong function.
    try:
        resolved = get_resolved_entry(problem)
    except Exception as e:
        return fail(f"could not read the function: {_reason(e)}")
    if not resolved.get("entry_name"):
        return fail("could not find the function to test")
    if not resolved.get("confirmed"):
        return fail("the solution could not be run — check that it executes")

    # ORACLE. The slow part: generate inputs, compute expected outputs from the
    # teacher's own solution, then mutation-test the resulting suite.
    try:
        tests = get_oracle_tests(problem)
    except Exception as e:
        return fail(f"test generation failed: {_reason(e)}")
    if not tests:
        return fail("No usable test cases could be generated. This usually "
                    "means the inputs aren't simple values.")
    try:
        if not is_oracle_strong(problem):
            return fail("The generated tests were not strong enough to grade "
                        "this reliably.")
    except Exception as e:
        return fail(_reason(e))

    # DECOMPOSITION, gated. get_chunk_decomposition runs the same serve boundary
    # a student request would have, so "ready" means exactly what it says.
    try:
        decomp = get_chunk_decomposition(problem)
    except Exception as e:
        return fail(_reason(e))

    return {"slug": slug, "ready": True,
            "chunks": len(decomp.get("chunks") or []),
            "n_tests": len(tests), "error": None}


def prepare_assignment_stream(problems: list[dict]):
    """Yield one dict per problem as preparation finishes, then a summary.

    A generator so the upload page can show progress: preparing twenty problems
    is minutes of work, and a silent wait is indistinguishable from a hang."""
    total = len(problems)
    yield {"event": "start", "total": total}
    ready = 0
    for i, p in enumerate(problems, 1):
        yield {"event": "preparing", "index": i, "total": total,
               "slug": p.get("slug", "?"), "title": p.get("title", "")}
        res = prepare_problem(p)
        ready += 1 if res["ready"] else 0
        yield {"event": "prepared", "index": i, "total": total, **res}
    yield {"event": "done", "total": total, "ready": ready, "failed": total - ready}


def save_manual_decomposition(problem: dict, header: str,
                              chunks: list[dict]) -> dict:
    """Teacher-authored split, for a problem auto-decomposition could not do.

    Goes through assert_serveable — THE SAME GATE as a generated one. A
    hand-written decomposition is not automatically trustworthy: it can still
    contain a chunk that does no work, which would let a student skip a step and
    still be marked correct. Raises on rejection so the teacher sees why."""
    from .run_phase1 import _load_pool, _save_pool, _serialize

    items = [StepItem(question_id=problem.get("slug", "problem"),
                      step_id=c.get("step_id") or f"Part {i + 1}",
                      prompt=(c.get("prompt") or "").strip(),
                      expected_type="code",
                      reference=(c.get("reference") or ""))
             for i, c in enumerate(chunks)]
    decomp = {"header": header, "chunks": items}
    assert_serveable(problem, decomp)              # raises if not serveable

    pool = _load_pool()
    key = content_hash(problem)
    pool.setdefault(key, []).append(_serialize(decomp))
    _save_pool(pool)
    return {"ready": True, "chunks": len(items)}
