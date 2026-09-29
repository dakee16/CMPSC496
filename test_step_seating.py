"""test_step_seating.py - a step that sits inside a loop is seated where THIS
student's code leaves room for it, never forced into a place that cannot parse.

THE REPORT (measured on the live site, 28 Sep). advanced-calculator-replace-
variables had roadmaps whose step 2 continues the TEACHER's loop from step 1 (its
reference starts `    elif ...`, four columns in). Every step-2 answer was
re-seated to that depth. But a student who wrote their OWN complete step 1 had
no loop left open, so whatever they typed - even `return None` or `pass` - came
back "Your code doesn't parse: unexpected indent, on line 1 of your answer".
Eleven such verdicts, all to one student, each one correct code marked wrong.
Two of that problem's five saved roadmaps are shaped like this.

The rule: the reference's depth is still tried FIRST, so every answer that
parsed before is seated exactly as before. Only when that depth cannot parse
after this student's own accepted steps is another depth tried, nearest first.
Code that parses at no depth is still their syntax error.

Self-contained: inline problem, inline oracle, no model.
"""
import types

import pytest

SOLUTION = ("def f(words):\n    out = []\n    for w in words:\n        if w == 'x':\n"
            "            out.append(w)\n        else:\n            out.append(w.upper())\n"
            "    return out\n")
PROBLEM = {"slug": "seat-f", "title": "f", "description": "Upper-case every word but x.",
           "solution": SOLUTION}
STEPS = [dict(step_id="Part 1", expected_type="code", prompt="Keep the x words.",
              reference="out = []\nfor w in words:\n    if w == 'x':\n        out.append(w)"),
         dict(step_id="Part 2", expected_type="code", prompt="Upper-case the others.",
              reference="    else:\n        out.append(w.upper())"),
         dict(step_id="Part 3", expected_type="code", prompt="Hand back the result.",
              reference="return out")]
ORACLE = [{"input": [["x", "a"]], "expected": ["x", "A"]},
          {"input": [[]], "expected": []},
          {"input": [["b", "x"]], "expected": ["B", "x"]}]


@pytest.fixture
def grade_step2(monkeypatch, tmp_path):
    monkeypatch.setenv("MICROTUTOR_SESSION_DB", str(tmp_path / "s.sqlite3"))
    from main import grading, identity, ollama_client, sessions
    monkeypatch.setattr(identity, "_RESOLVED_PATH", str(tmp_path / "r.json"))
    grading._VERDICT_MEMO.clear()

    def no_model(*_a, **_k):
        raise AssertionError("seating must never reach a model")
    monkeypatch.setattr(ollama_client, "_openai_chat", no_model)
    monkeypatch.setattr(grading, "_request_adaptation", lambda *a, **k: ("", []))
    monkeypatch.setattr(grading, "_request_completion", lambda *a, **k: "")

    def _grade(step1, step2):
        decomp = {"header": "def f(words):",
                  "chunks": [types.SimpleNamespace(**c) for c in STEPS]}
        sid = sessions.create_session(dict(PROBLEM), decomp, "h-seat",
                                      student_id="stu")["session_id"]
        s = sessions.load_session(sid)
        sessions.commit_outcome(sid, "pre-0", s["revision"],
                                {"verdict": "correct", "tier": "execution-reference"},
                                accept_code=grading.align_submission(s, step1),
                                provenance="student")
        s = sessions.load_session(sid)
        return (grading.grade_submission(s, step2, oracle_loader=lambda p: list(ORACLE)),
                grading.align_submission(s, step2))
    yield _grade
    grading._VERDICT_MEMO.clear()


# Their own step 1 did the whole job with no loop left open.
DONE_IN_STEP_1 = "out = [w if w == 'x' else w.upper() for w in words]"


def test_the_reported_case_is_not_a_syntax_error(grade_step2):
    r, seated = grade_step2(DONE_IN_STEP_1, "pass")
    assert r.tier != "syntax", r.student_reason
    assert r.verdict == "correct", r
    assert seated == "pass", "seated at the only depth their own code leaves open"


def test_the_teachers_shape_is_seated_exactly_as_before(grade_step2):
    """Their step 1 left the loop open like the teacher's: the reference depth
    fits, so nothing about this answer changes."""
    r, seated = grade_step2(STEPS[0]["reference"], "else:\n    out.append(w.upper())")
    assert r.verdict == "correct", r
    assert seated == "    else:\n        out.append(w.upper())"


def test_code_that_parses_nowhere_is_still_their_syntax_error(grade_step2):
    r, _ = grade_step2(DONE_IN_STEP_1, "out.append(w.upper()")
    assert (r.verdict, r.tier) == ("incorrect", "syntax"), r


def test_when_both_depths_parse_the_references_depth_wins(grade_step2):
    """Inside their open loop or after it would both parse; the reference's
    depth is the one the step was written for, so nothing that worked before
    may move."""
    _, seated = grade_step2(STEPS[0]["reference"], "out.append(w.upper())")
    assert seated == "    out.append(w.upper())"
