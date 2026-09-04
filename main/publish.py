"""
publish.py - prepare a teacher's problems BEFORE any student sees them.

This is the fix for the failure that used to land in a student's face. Oracle
generation, mutation validation and decomposition all used to run when a STUDENT
pressed Start: slow, paid, and roughly a coin-flip on whether it worked, with an
HTTP 500 as the failure mode. All of it now happens once, at upload, in front of
the teacher - who can actually do something about it.

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
    """A teacher-readable sentence. Never a traceback - this reaches a UI."""
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


# The gates a problem passes, in the order prepare_problem applies them. This
# is the list an instructor reads as a checklist while fixing a blocked problem,
# so the labels say what a stage MEANS to them, not what it is called here.
PREPARE_STAGES = (
    ("parses",   "Reads as one Python function with a docstring"),
    ("runs",     "The solution runs"),
    ("tests",    "Test cases could be generated from it"),
    ("strength", "Those tests are strong enough to grade with"),
    ("steps",    "Splits into steps a student can work through"),
)


def checklist(stage: str | None, error: str | None = None) -> list[dict]:
    """The stages as pass / fail / not-reached, for the teacher's fix panel.

    `stage` is where preparation stopped, or None if it finished. Everything
    before the failure passed by construction - preparation is strictly
    sequential and never reaches a later gate without clearing the earlier
    ones - and everything after it is genuinely UNKNOWN, which is why those are
    "pending" rather than a second kind of failure."""
    names = [s for s, _ in PREPARE_STAGES]
    stop = names.index(stage) if stage in names else len(names)
    return [{"id": sid, "label": label,
             "state": "ok" if i < stop else "fail" if i == stop else "pending",
             "error": error if i == stop else None}
            for i, (sid, label) in enumerate(PREPARE_STAGES)]


# Every message below is written either by prepare_problem() or by
# assignments.parse_assignment_file(), so mapping one back to its stage is the
# inverse of a table this package owns rather than a guess about arbitrary text.
# It is needed because a stored problem row keeps only `prepare_error`: a retry
# reports its stage exactly, and this recovers the stage for a row that was
# prepared before any retry ran.
_ERROR_STAGE = (
    ("no solution provided",         "parses"),
    ("could not read the function",  "parses"),
    ("could not find the function",  "parses"),
    ("not valid python",             "parses"),
    ("no function found",            "parses"),
    ("has no docstring",             "parses"),
    ("class-based solutions",        "parses"),
    ("duplicate slug",               "parses"),
    ("block is empty",               "parses"),
    ("must be lowercase letters",    "parses"),
    ("could not be run",             "runs"),
    ("test generation failed",       "tests"),
    ("no usable test cases",         "tests"),
    ("not strong enough",            "strength"),
    ("could not split this problem", "steps"),
)


def stage_of_error(error: str | None) -> str | None:
    """Which gate a stored prepare_error stopped at, or None if it is not a
    failure at all. Unrecognised text falls through to the LAST stage: the
    earlier gates are the ones we could have named, so the honest reading of an
    unknown failure is that it got past them."""
    text = (error or "").strip().lower()
    if not text:
        return None
    for needle, stage in _ERROR_STAGE:
        if needle in text:
            return stage
    return PREPARE_STAGES[-1][0]


def prepare_problem(problem: dict) -> dict:
    """Make one problem student-ready. Returns {slug, ready, chunks, stage,
    error}, where `stage` names the gate it stopped at - None once it passed.

    Never raises: a failure here is ordinary teacher feedback, not a server
    fault, and one bad problem must not abort an upload of twenty."""
    from tests.sandbox import get_oracle_tests, is_oracle_strong
    from .run_phase1 import get_chunk_decomposition

    slug = problem.get("slug", "?")

    def fail(stage, msg):
        return {"slug": slug, "ready": False, "chunks": 0,
                "stage": stage, "error": msg}

    if not (problem.get("solution") or "").strip():
        return fail("parses", "no solution provided")

    # The entry point must be resolvable and actually runnable, or every later
    # stage is measuring the wrong function.
    try:
        resolved = get_resolved_entry(problem)
    except Exception as e:
        return fail("parses", f"could not read the function: {_reason(e)}")
    if not resolved.get("entry_name"):
        return fail("parses", "could not find the function to test")
    if not resolved.get("confirmed"):
        return fail("runs", "the solution could not be run - check that it executes")

    # ORACLE. The slow part: generate inputs, compute expected outputs from the
    # teacher's own solution, then mutation-test the resulting suite.
    try:
        tests = get_oracle_tests(problem)
    except Exception as e:
        return fail("tests", f"test generation failed: {_reason(e)}")
    if not tests:
        return fail("tests", "No usable test cases could be generated. This "
                             "usually means the inputs aren't simple values.")
    try:
        if not is_oracle_strong(problem):
            return fail("strength", "The generated tests were not strong enough "
                                    "to grade this reliably.")
    except Exception as e:
        return fail("strength", _reason(e))

    # DECOMPOSITION, gated. get_chunk_decomposition runs the same serve boundary
    # a student request would have, so "ready" means exactly what it says.
    try:
        decomp = get_chunk_decomposition(problem)
    except Exception as e:
        return fail("steps", _reason(e))

    return {"slug": slug, "ready": True,
            "chunks": len(decomp.get("chunks") or []),
            "n_tests": len(tests), "stage": None, "error": None}


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

    Goes through assert_serveable - THE SAME GATE as a generated one. A
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


if __name__ == "__main__":
    # The checklist is the only logic here that runs with no oracle, no model
    # and no database, and it is what the teacher's fix panel is drawn from.
    #   python -m main.publish
    ok = checklist(None)
    assert [c["state"] for c in ok] == ["ok"] * len(PREPARE_STAGES), ok
    assert all(c["error"] is None for c in ok)

    mid = checklist("tests", "test generation failed: timeout")
    assert [c["state"] for c in mid] == ["ok", "ok", "fail", "pending", "pending"], mid
    # The reason is attached to the gate that failed, and to nothing else - a
    # message repeated on every row reads as five separate problems.
    assert [c["error"] for c in mid].count("test generation failed: timeout") == 1

    # An unknown stage reads as "finished", never as a sixth failing row.
    assert "fail" not in [c["state"] for c in checklist("nonsense", "x")], \
        "an unrecognised stage must not invent a failing row"

    assert stage_of_error(None) is None and stage_of_error("  ") is None
    for text, want in (
            ("function 'f' has no docstring. The docstring IS the...", "parses"),
            ("not valid Python: invalid syntax (line 3)", "parses"),
            ("the solution could not be run - check that it executes", "runs"),
            ("No usable test cases could be generated.", "tests"),
            ("The generated tests were not strong enough to grade this.", "strength"),
            ("Could not split this problem into steps that hold together.", "steps"),
            ("something nobody has ever written", "steps")):
        assert stage_of_error(text) == want, (text, stage_of_error(text))

    print("publish.py self-check OK")
