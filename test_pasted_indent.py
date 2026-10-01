"""test_pasted_indent.py - a block pasted into the code box with its first line
deeper than the rest is TOLD where the indentation is wrong - never fixed for
the student, and never silently run somewhere they did not mean.

THE REPORT (29-30 Sep, the GPT student audit, then the live data). The code
box opens every step with the caret after a few spaces of its own
(frontend/student.js: the function body's indent plus the step's), so the box
lines up with the frozen code above it. A PASTED block keeps those spaces on
its first line only:

        if len(calcStack) == 1:       <- the box's 4 + the pasted 4
        return calcStack.pop()
    return None

Two things went wrong. Usually: "Your indentation doesn't line up on line 2"
- pointing at the wrong line. Once, worse: the extra spaces made the first
line legal INSIDE the previous step's loop, so it ran there and the student was
told their answer was wrong (calculate, 23 Sep). A first version of this fix
moved the line back for them; Sanan's call (30 Sep): tell them, don't fix it.

Not said when the step's own code starts deeper and then comes back out
(continuing a loop, then leaving it) - a correct answer has that shape too.

Self-contained: inline problems, inline oracles, no model.
"""
import types

from main.indent import unpad_first_line
from test_step_seating import ORACLE, PROBLEM, STEPS, grade_step2  # noqa: F401

TEACHER_STEP_1 = STEPS[0]["reference"]


def _paste(code: str, pad: int) -> str:
    """What the box holds after pasting `code` at the caret, `pad` spaces in."""
    return " " * pad + code


def _grade_step1(code, tmp_path, monkeypatch, steps=STEPS, problem=PROBLEM, oracle=ORACLE,
                 header="def f(words):"):
    from main import grading, identity, sessions
    monkeypatch.setenv("MICROTUTOR_SESSION_DB", str(tmp_path / "s.sqlite3"))
    monkeypatch.setattr(identity, "_RESOLVED_PATH", str(tmp_path / "r.json"))
    monkeypatch.setattr(grading, "_request_adaptation", lambda *a, **k: ("", []))
    monkeypatch.setattr(grading, "_request_completion", lambda *a, **k: "")
    grading._VERDICT_MEMO.clear()
    decomp = {"header": header, "chunks": [types.SimpleNamespace(**c) for c in steps]}
    sid = sessions.create_session(dict(problem), decomp, "h-paste",
                                  student_id="stu")["session_id"]
    return grading.grade_submission(sessions.load_session(sid), code,
                                    oracle_loader=lambda p: list(oracle))


def test_a_pasted_block_is_told_which_line_is_off(tmp_path, monkeypatch):
    r = _grade_step1(_paste(TEACHER_STEP_1, 4), tmp_path, monkeypatch)
    assert (r.verdict, r.tier) == ("incorrect", "syntax"), r
    assert r.student_reason.startswith("Indentation error on line 1 of your answer"), r.student_reason
    assert "`out = []`" in r.student_reason


def test_the_line_number_counts_blank_lines_before_it(tmp_path, monkeypatch):
    r = _grade_step1("\n\n" + _paste(TEACHER_STEP_1, 4), tmp_path, monkeypatch)
    assert "line 3 of your answer" in r.student_reason, r.student_reason


def test_a_paste_that_would_run_inside_the_previous_loop_is_told_not_run(grade_step2):
    """The silent case: the extra spaces put `out.append` INSIDE step 1's
    loop, where it parses - it used to run there and be marked wrong."""
    r, _ = grade_step2(TEACHER_STEP_1, _paste("else:\n    out.append(w.upper())", 8))
    assert (r.verdict, r.tier) == ("incorrect", "syntax"), r
    assert "Indentation error on line 1" in r.student_reason, r.student_reason


def test_a_pasted_def_line_is_told_about_the_def_line(tmp_path, monkeypatch):
    r = _grade_step1(_paste("def f(words):\n    out = []", 4), tmp_path, monkeypatch)
    assert "already written for you" in r.student_reason, r.student_reason


def test_code_lined_up_correctly_is_graded_as_before(grade_step2):
    for step2 in ("    else:\n        out.append(w.upper())",
                  "else:\n    out.append(w.upper())"):
        r, _ = grade_step2(TEACHER_STEP_1, step2)
        assert r.verdict == "correct", (step2, r.student_reason)


def test_a_step_that_continues_a_loop_and_then_leaves_it_is_not_flagged(tmp_path, monkeypatch):
    """Its own reference starts deeper and comes back out, so the student's
    correct answer has the same shape."""
    steps = [dict(step_id="Part 1", expected_type="code", prompt="p1",
                  reference="out = []\nfor w in words:\n    w = w.strip()"),
             dict(step_id="Part 2", expected_type="code", prompt="p2",
                  reference="    out.append(w if w == 'x' else w.upper())\nreturn out")]
    from main import grading, sessions
    r1 = _grade_step1("out = []\nfor w in words:\n    w = w.strip()", tmp_path, monkeypatch,
                      steps=steps)
    assert r1.verdict == "correct", r1
    decomp = {"header": "def f(words):", "chunks": [types.SimpleNamespace(**c) for c in steps]}
    sid = sessions.create_session(dict(PROBLEM), decomp, "h-out", student_id="stu")["session_id"]
    s = sessions.load_session(sid)
    sessions.commit_outcome(sid, "p0", s["revision"], r1.model_dump(),
                            accept_code=grading.align_submission(s, "out = []\nfor w in words:\n    w = w.strip()"),
                            provenance="student")
    grading._VERDICT_MEMO.clear()
    r2 = grading.grade_submission(sessions.load_session(sid),
                                  "    out.append(w if w == 'x' else w.upper())\nreturn out",
                                  oracle_loader=lambda p: list(ORACLE))
    assert r2.verdict == "correct", (r2.tier, r2.student_reason)


def test_only_the_pasted_shape_is_ever_flagged():
    assert unpad_first_line("    x = 1\ny = 2") == "x = 1\ny = 2"
    assert unpad_first_line("        else:\n    c") == "    else:\n    c"
    for same in ("x = 1\ny = 2",                  # flat
                 # ORDINARY code that opens a block and carries on at its own
                 # level. A first version flagged this shape: replayed on the
                 # live submissions it would have failed 34 correct answers.
                 "    if self.top is None:\n        return None\n    return self.top.value",
                 "try:\n    float(txt)\nexcept ValueError:\n    return False",
                 "    for a in b:\n    c",              # broken - but Python says so itself
                 "    x = 1\n# a note at the margin\n    y = 2",
                 "    total = add(1,\n2)\n    return total",   # inside a bracket
                 "    x = 1\n    y = 2",          # evenly indented
                 "for a in b:\n    c",            # a real block
                 "    # note\ny = 2",             # a comment's indent is harmless
                 "\tx = 1\ny = 2",                # tabs: no guessing widths
                 "x = 1"):                        # one line
        assert unpad_first_line(same) == same, same
