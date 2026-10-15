"""test_free_rejection.py - a wrong step is rejected without the model when the
teacher's remaining steps read nothing an earlier step made.

THE REPORT (13 Oct, HW4 student-path audit). Of 82 wrong steps, 35 were sent to
the paid model tiers - which exist because a failure joined to the teacher's
remaining steps might be only a naming mismatch. For 19 of them the remaining
steps read nothing an earlier step made (`return 'Cleared cache!'`), so there
was nothing to mismatch: the failure was the student's own.

THE TRAP, found while building it: a tail that carries on INSIDE a loop starts
indented, the name reader cannot parse it and reports "reads nothing", and a
correct renamed step was convicted (HW3 calculateExpressions). Such a tail
must count as depending on earlier steps.

REAL: grade_submission, sessions, execution. FAKED: the model tiers find
nothing (and a model call fails the test).
"""
import types

import pytest

SOLUTION = ("def average(nums):\n"
            "    if not nums:\n"
            "        return None\n"
            "    total = 0\n"
            "    for n in nums:\n"
            "        total += n\n"
            "    return total / len(nums)\n")
PROBLEM = {"slug": "average", "title": "Average", "solution": SOLUTION, "entry_hint": "average",
           "description": "The average of nums, or None for an empty list."}
_ns = {}
exec(SOLUTION, _ns)
TESTS = [{"input": [x], "expected": _ns["average"](x)} for x in ([], [4], [1, 2, 3], [-2, 2], [5, 5, 8])]
INDEPENDENT = [("if not nums:\n    return None"),
               ("total = 0\nfor n in nums:\n    total += n\nreturn total / len(nums)")]
DEPENDENT = [("if not nums:\n    return None\ntotal = 0"),
             ("for n in nums:\n    total += n\nreturn total / len(nums)")]
# Like HW3's calculateExpressions roadmap 3: step 1 opens the loop and starts
# its body, step 2 carries on inside it - so the tail starts indented.
IN_LOOP = [("if not nums:\n    return None\ntotal = 0\ncount = 0\nfor n in nums:\n    count += 1"),
           ("    total += n"), ("return total / count")]


@pytest.fixture
def grade(monkeypatch, tmp_path):
    from main import grading, identity, ollama_client, sessions
    monkeypatch.setenv("MICROTUTOR_SESSION_DB", str(tmp_path / "s.sqlite3"))
    monkeypatch.setenv("MICROTUTOR_TRACE_FILE", str(tmp_path / "t.jsonl"))
    monkeypatch.setattr(identity, "_RESOLVED_PATH", str(tmp_path / "r.json"))
    asked = []
    monkeypatch.setattr(ollama_client, "_openai_chat",
                        lambda *a, **k: asked.append(1) or (_ for _ in ()).throw(RuntimeError("no model")))
    monkeypatch.setattr(grading, "_request_adaptation", lambda *a, **k: asked.append(1) or ("", []))
    monkeypatch.setattr(grading, "_request_completion", lambda *a, **k: asked.append(1) or "")

    def run(steps, at, code, before=()):
        decomp = {"header": "def average(nums):", "chunks": [
            types.SimpleNamespace(step_id=f"Part {i + 1}", expected_type="code", prompt="p",
                                  reference=r) for i, r in enumerate(steps)]}
        sid = sessions.create_session(dict(PROBLEM), decomp, "h-avg", student_id="stu")["session_id"]
        for i, b in enumerate(before):
            s = sessions.load_session(sid)
            sessions.commit_outcome(sid, f"p{i}", s["revision"], {"verdict": "correct",
                                    "tier": "execution-reference"}, accept_code=b, provenance="student")
        asked.clear()
        grading._VERDICT_MEMO.clear()
        return grading.grade_submission(sessions.load_session(sid), code,
                                        oracle_loader=lambda p: list(TESTS))
    run.asked = asked
    return run


def test_a_wrong_step_with_an_independent_tail_is_rejected_for_free(grade):
    r = grade(INDEPENDENT, 0, "if nums:\n    return None")           # inverted guard
    assert (r.verdict, r.tier, r.deterministic) == ("incorrect", "execution-reference", True), r
    assert r.reason_code == "reference_fail_independent"
    assert grade.asked == [], "no model was needed to say so"
    assert r.failing_cases, "the case it gets wrong is shown"


def test_a_right_step_with_an_independent_tail_is_still_right(grade):
    r = grade(INDEPENDENT, 0, "if len(nums) == 0:\n    return None")
    assert r.verdict == "correct", r


def test_a_correct_step_the_tail_cannot_read_is_not_convicted(grade):
    """It skips the `total` the teacher's next step reads - correct, but the
    tail cannot run on it. That is the model tiers' case, never a verdict."""
    r = grade(DEPENDENT, 0, "if not nums:\n    return None\nrunning = 0")
    assert r.verdict != "incorrect", r


def test_a_tail_inside_a_loop_is_never_read_as_independent(grade):
    """The trap: the tail starts indented. A correct step 1 with its own name
    for the tally must not be convicted because the tail could not be parsed."""
    r = grade(IN_LOOP, 0, "if not nums:\n    return None\nrunning = 0\nk = 0\nfor n in nums:\n    k += 1")
    assert r.verdict != "incorrect", r
