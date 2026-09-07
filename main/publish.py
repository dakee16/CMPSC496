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
    ("waiting on your review",       "strength"),
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


def prepare_problem(problem: dict, emit=None) -> dict:
    """Make one problem student-ready. Returns {slug, ready, chunks, stage,
    error}, where `stage` names the gate it stopped at - None once it passed.

    Never raises: a failure here is ordinary teacher feedback, not a server
    fault, and one bad problem must not abort an upload of twenty.

    `emit`, when given, narrates the run in main/live_playground.py's event
    vocabulary so a teacher can WATCH this problem being prepared instead of
    watching a spinner - see main/prepare_bus.py for why the watching tab
    mirrors this run rather than starting its own. It never changes behaviour
    and never affects the return value; leaving it None is the production path,
    and a failure inside a watcher must never fail an upload."""
    from tests.sandbox import get_oracle_tests, is_oracle_certified
    from .run_phase1 import get_chunk_decomposition

    slug = problem.get("slug", "?")
    emit = emit or (lambda ev: None)

    def fail(stage, msg):
        # The same sentence the teacher's row shows, on the transcript too - a
        # watcher whose stream just stopped cannot tell "failed" from "hung".
        emit({"type": "blocked", "at": stage, "error_type": "PrepareFailed",
              "message": msg})
        return {"slug": slug, "ready": False, "chunks": 0,
                "stage": stage, "error": msg}

    emit({"type": "stage", "name": "start",
          "label": f"Preparing {problem.get('title') or slug}"})
    emit({"type": "ground_truth", "code": problem.get("solution", "")})

    if not (problem.get("solution") or "").strip():
        return fail("parses", "no solution provided")

    # The entry point must be resolvable and actually runnable, or every later
    # stage is measuring the wrong function.
    emit({"type": "stage", "name": "entry",
          "label": "Resolving the entry point (which function to call)"})
    try:
        resolved = get_resolved_entry(problem)
    except Exception as e:
        return fail("parses", f"could not read the function: {_reason(e)}")
    if not resolved.get("entry_name"):
        return fail("parses", "could not find the function to test")
    emit({"type": "entry", "entry_name": resolved.get("entry_name"),
          "params": resolved.get("params", [])})
    if not resolved.get("confirmed"):
        return fail("runs", "the solution could not be run - check that it executes")

    # ORACLE. The slow part: generate inputs, compute expected outputs from the
    # teacher's own solution, then mutation-test the resulting suite.
    emit({"type": "stage", "name": "oracle_gen",
          "label": "Generating oracle tests - the model proposes INPUTS only, "
                   "the teacher's own solution computes every expected output"})
    try:
        tests = get_oracle_tests(problem, emit=emit)
    except Exception as e:
        return fail("tests", f"test generation failed: {_reason(e)}")
    if not tests:
        return fail("tests", "No usable test cases could be generated. This "
                             "usually means the inputs aren't simple values.")
    # The verdict, read back from what validation just persisted, so a watcher
    # sees the same numbers the badge on the upload row will show.
    try:
        from main.identity import content_hash as _ch
        from main.oracle_store import load_cache as _lc, verdict_event
        _entry = _lc().get(_ch(problem))
        if isinstance(_entry, dict) and "strong" in _entry:
            emit(verdict_event(_entry))
    except Exception:
        pass                            # narration must never fail preparation

    try:
        if not is_oracle_certified(problem):
            # A4 - "not strong" is now two different situations, and only one of
            # them is a failure the instructor can fix by editing the problem.
            #
            #   weak          even in the best case the tests miss too much
            #   needs_review  the tests may well be fine; a handful of
            #                 deliberate errors could not be judged either way,
            #                 and a person has to look
            #
            # The second is NOT the instructor writing a bad problem, so it must
            # not be reported as one. It gets its own outcome, and the upload
            # page offers to walk them through it.
            from main.identity import content_hash
            from main.oracle_store import load_cache
            verdict = (load_cache().get(content_hash(problem)) or {})
            if verdict.get("status") == "needs_review":
                return {"slug": slug, "ready": False, "chunks": 0,
                        "stage": "strength", "needs_review": True,
                        "undetermined": verdict.get("undetermined", 0),
                        "kill_rate_lower": verdict.get("kill_rate_lower", 0.0),
                        "kill_rate_upper": verdict.get("kill_rate_upper", 0.0),
                        # Still an error string, because that column is the
                        # only thing a stored row keeps and a row with no
                        # explanation reads as an unexplained failure later.
                        # The wording says who has to act and that the problem
                        # itself may be fine.
                        "error": ("Waiting on your review: a few checks on the "
                                  "generated tests came back inconclusive.")}
            return fail("strength", "The generated tests were not strong enough "
                                    "to grade this reliably.")
    except Exception as e:
        return fail("strength", _reason(e))

    # DECOMPOSITION, gated. get_chunk_decomposition runs the same serve boundary
    # a student request would have, so "ready" means exactly what it says.
    emit({"type": "stage", "name": "decomposition",
          "label": "Decomposing into steps - exactly the path a student request "
                   "takes, including every retry and Gate 1"})
    try:
        decomp = get_chunk_decomposition(problem)
    except Exception as e:
        return fail("steps", _reason(e))

    chunks = decomp.get("chunks") or []
    emit({"type": "chunks", "header": decomp.get("header", ""),
          "chunks": [{"step_id": c.step_id, "prompt": c.prompt,
                      "reference": c.reference or ""} for c in chunks]})
    emit({"type": "stage", "name": "finished", "label": "Ready for students"})

    return {"slug": slug, "ready": True,
            "chunks": len(chunks),
            "n_tests": len(tests), "stage": None, "error": None}


def prepare_assignment_stream(problems: list[dict], emit_for=None):
    """Yield one dict per problem as preparation finishes, then a summary.

    A generator so the upload page can show progress: preparing twenty problems
    is minutes of work, and a silent wait is indistinguishable from a hang.

    `emit_for`, when given, is called with a slug and returns either an `emit`
    callback for that problem's narration or None. That indirection is what lets
    the upload open one watchable channel per problem (main/prepare_bus.py)
    without this module knowing anything about channels, HTTP or who is
    watching."""
    total = len(problems)
    yield {"event": "start", "total": total}
    ready = 0
    review: list[dict] = []
    for i, p in enumerate(problems, 1):
        # The channel is opened BEFORE the row is announced. The other order has
        # a real gap in it: the upload page draws the row, the teacher clicks it
        # immediately - which is the whole point of the feature - and the
        # watcher arrives at an address nothing has opened yet.
        emit = emit_for(p.get("slug", "?")) if emit_for else None
        yield {"event": "preparing", "index": i, "total": total,
               "slug": p.get("slug", "?"), "title": p.get("title", "")}
        res = prepare_problem(p, emit=emit)
        ready += 1 if res["ready"] else 0
        if res.get("needs_review"):
            review.append({"slug": res["slug"], "title": p.get("title", ""),
                           "undetermined": res.get("undetermined", 0),
                           "kill_rate_lower": res.get("kill_rate_lower", 0.0),
                           "kill_rate_upper": res.get("kill_rate_upper", 0.0)})
        yield {"event": "prepared", "index": i, "total": total, **res}
    # `blocked` is the count the teacher can act on by editing the problem.
    # Review problems are held back from students too, but the fix is a person
    # looking at a test suite, not a rewrite - so they are counted separately
    # and `failed` no longer conflates the two.
    yield {"event": "done", "total": total, "ready": ready,
           "failed": total - ready - len(review),
           "review": review,
           "n_review": len(review)}


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
            ("Waiting on your review: a few checks came back inconclusive.", "strength"),
            ("Could not split this problem into steps that hold together.", "steps"),
            ("something nobody has ever written", "steps")):
        assert stage_of_error(text) == want, (text, stage_of_error(text))

    # The summary event is what the upload page counts on, and its whole point
    # is that "held for review" is NOT "failed". Stub preparation so the three
    # outcomes are exercised without a model, a sandbox or a database.
    _outcomes = {
        "a": {"slug": "a", "ready": True, "chunks": 3, "n_tests": 9,
              "stage": None, "error": None},
        "b": {"slug": "b", "ready": False, "chunks": 0, "stage": "tests",
              "error": "test generation failed"},
        "c": {"slug": "c", "ready": False, "chunks": 0, "stage": "strength",
              "needs_review": True, "undetermined": 2,
              "kill_rate_lower": 0.67, "kill_rate_upper": 1.0,
              "error": "Waiting on your review: ..."},
    }
    # Patched in THIS module's globals - run as __main__ that is the namespace
    # prepare_assignment_stream actually resolves the name in.
    _real = prepare_problem
    globals()["prepare_problem"] = lambda p: _outcomes[p["slug"]]
    try:
        events = list(prepare_assignment_stream(
            [{"slug": s, "title": s.upper()} for s in ("a", "b", "c")]))
    finally:
        globals()["prepare_problem"] = _real

    done = events[-1]
    assert done["event"] == "done" and done["total"] == 3, done
    assert done["ready"] == 1, done
    # The one held for review must not be counted as a failure: a teacher told
    # "2 need attention" goes and edits a problem that may need no edit at all.
    assert done["failed"] == 1, done
    assert done["n_review"] == 1 and len(done["review"]) == 1, done
    assert done["review"][0] == {"slug": "c", "title": "C", "undetermined": 2,
                                 "kill_rate_lower": 0.67,
                                 "kill_rate_upper": 1.0}, done["review"]
    assert done["ready"] + done["failed"] + done["n_review"] == done["total"]
    # And the review row still carries its flag on the per-problem event, which
    # is what picks the third badge instead of "needs work".
    prepared = [e for e in events if e["event"] == "prepared"]
    assert [bool(e.get("needs_review")) for e in prepared] == [False, False, True]

    print("publish.py self-check OK")
