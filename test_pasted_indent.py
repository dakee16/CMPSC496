"""test_pasted_indent.py - a block pasted into the editor is graded as the code
the student pasted, not as the editor's leftover spaces plus that code.

THE REPORT (29 Sep, GPT's student audit, then the live data). The code box
opens every step with the caret after a few spaces of its own
(frontend/student.js: the function body's indent plus the step's), so the box
lines up with the frozen code above it. A PASTED block keeps those spaces on
its first line only:

        self.states = {}          <- the editor's 4 + the pasted 4
    report = {}
    calcObj = Calculator()

and came back "Your indentation doesn't line up on line 2 of your answer". On
the live site: 3 correct answers from 2 students, plus the audit's paste.

The rule: the answer is seated exactly as before whenever it parses anywhere.
Only when it parses nowhere is its first line moved back to the depth of the
lines under it (one level above them if it opens a block) and tried again.
Code that still parses nowhere is still their syntax error.

Self-contained: inline problem, inline oracle, no model (the seating test's).
"""
from main.indent import align_to_chunk, unpad_first_line
from test_step_seating import STEPS, grade_step2  # noqa: F401

TEACHER_STEP_1 = STEPS[0]["reference"]


def _paste(code: str, pad: int) -> str:
    """What the box holds after pasting `code` at the caret, `pad` spaces in."""
    return " " * pad + code


def test_a_pasted_step_is_kept_as_the_code_the_student_meant(grade_step2):
    """Step 1 is top level, so the box opens 4 spaces in. What is stored as
    their accepted step 1 must be the code they pasted - or step 2, built on
    it, cannot parse."""
    r, _ = grade_step2(_paste(TEACHER_STEP_1, 4), "    else:\n        out.append(w.upper())")
    assert r.verdict == "correct", (r.tier, r.student_reason)


def test_the_audits_exact_shape_is_accepted_at_step_1(tmp_path, monkeypatch):
    """First line at the editor's 4 + the paste's own 4, the rest at 4."""
    import types
    from main import grading, identity, sessions
    from test_step_seating import ORACLE, PROBLEM
    monkeypatch.setenv("MICROTUTOR_SESSION_DB", str(tmp_path / "s.sqlite3"))
    monkeypatch.setattr(identity, "_RESOLVED_PATH", str(tmp_path / "r.json"))
    grading._VERDICT_MEMO.clear()
    pasted = _paste("\n".join("    " + ln for ln in TEACHER_STEP_1.splitlines()), 4)
    assert pasted.startswith("        out = []\n    for w in words:")
    decomp = {"header": "def f(words):",
              "chunks": [types.SimpleNamespace(**c) for c in STEPS]}
    sid = sessions.create_session(dict(PROBLEM), decomp, "h-paste",
                                  student_id="stu")["session_id"]
    r = grading.grade_submission(sessions.load_session(sid), pasted,
                                 oracle_loader=lambda p: list(ORACLE))
    assert r.verdict == "correct", (r.tier, r.student_reason)


def test_a_pasted_block_opener_inside_the_loop_is_accepted(grade_step2):
    """Step 2 sits inside the teacher's loop: the box opens 8 spaces in, and
    the pasted `else:` is a block opener, so it goes a level above its body."""
    r, seated = grade_step2(TEACHER_STEP_1, _paste("else:\n    out.append(w.upper())", 8))
    assert r.verdict == "correct", (r.tier, r.student_reason)
    assert seated == "    else:\n        out.append(w.upper())"


def test_code_that_parsed_before_is_seated_exactly_as_before(grade_step2):
    for step2 in ("    else:\n        out.append(w.upper())",
                  "else:\n    out.append(w.upper())"):
        _r, seated = grade_step2(TEACHER_STEP_1, step2)
        assert seated == align_to_chunk(step2, STEPS[1])


def test_code_broken_in_its_own_right_is_still_a_syntax_error(grade_step2):
    # The pasted shape, AND a body that is not indented under its `if`.
    r, _ = grade_step2(TEACHER_STEP_1, _paste("else:\nout.append(w.upper())", 8))
    assert (r.verdict, r.tier) == ("incorrect", "syntax"), r


def test_only_the_pasted_shape_is_ever_touched():
    assert unpad_first_line("    x = 1\ny = 2") == "x = 1\ny = 2"
    assert unpad_first_line("    for a in b:\n    c") == "for a in b:\n    c"
    for same in ("x = 1\ny = 2",                  # flat
                 "    x = 1\n    y = 2",          # evenly indented
                 "for a in b:\n    c",            # a real block
                 "    # note\ny = 2",             # a comment's indent is harmless
                 "\tx = 1\ny = 2",                # tabs: no guessing widths
                 "x = 1"):                        # one line
        assert unpad_first_line(same) == same, same
