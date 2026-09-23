"""test_shape_gates.py - answers that are not positioned to run at all.

Every tier in main/grading.py was built to judge whether an answer's LOGIC is
right. None of them looked at its SHAPE - whether the code is even placed to
execute - and a real student hit that gap in his first hour. The failures all
look the same from the student's side: nothing runs, nothing binds, no tier can
attribute anything, so it falls through to the judges, which cannot convict,
and the answer is "We could not confirm this step."

ORDER IS THE POINT OF THIS FILE, not the individual checks - those are already
unit-tested in grading's own self-check. An audit on the live site found
`comments_only` never firing: a body of pure comments makes the assembled
function EMPTY, which is an IndentationError, so the syntax gate answered
first with "your indentation doesn't line up on line 1 - `# Count each value
in d.`. The line above it opens a block..." about a single comment with no line
above it. The check was sound and sat one gate too late, and testing it alone
could not have shown that. So every case here goes through grade_submission.

Self-contained: the oracle is inline via grade_submission(oracle_loader=...),
because the real pool and problem file are gitignored and a test that skips on
a fresh checkout guards nothing.
"""
import types

import pytest

SOLUTION = ("def invert(d):\n"
            "    counts = {}\n"
            "    for value in d.values():\n"
            "        counts[value] = counts.get(value, 0) + 1\n"
            "    result = {}\n"
            "    for key, value in d.items():\n"
            "        if counts[value] == 1:\n"
            "            result[value] = key\n"
            "    return result\n")
PROBLEM = {"slug": "invert", "title": "Invert", "solution": SOLUTION,
           "description": "Map each value that appears once back to its key."}
STEPS = [{"step_id": "Part 1", "expected_type": "code",
          "prompt": "Count how many times each value appears, and keep it.",
          "reference": "counts = {}\nfor value in d.values():\n"
                       "    counts[value] = counts.get(value, 0) + 1"},
         {"step_id": "Part 2", "expected_type": "code",
          "prompt": "Pair each value that appeared once with its key.",
          "reference": "result = {}\nfor key, value in d.items():\n"
                       "    if counts[value] == 1:\n        result[value] = key"},
         {"step_id": "Part 3", "expected_type": "code",
          "prompt": "Hand it back.", "reference": "return result"}]
ORACLE = [{"input": [{"a": 1, "b": 2}], "expected": {1: "a", 2: "b"}},
          {"input": [{"a": 1, "b": 1}], "expected": {}},
          {"input": [{}], "expected": {}},
          {"input": [{"x": 5, "y": 5, "z": 9}], "expected": {9: "z"}}]


@pytest.fixture
def grade(monkeypatch, tmp_path):
    monkeypatch.setenv("MICROTUTOR_SESSION_DB", str(tmp_path / "s.sqlite3"))
    from main import grading, identity, ollama_client, sessions
    # grade_submission persists into the TRACKED resolved_entries.json otherwise.
    monkeypatch.setattr(identity, "_RESOLVED_PATH", str(tmp_path / "resolved.json"))

    def no_model(*_a, **_k):
        raise AssertionError("a shape check must never reach a model")
    for mod in (ollama_client, grading):
        if hasattr(mod, "chat"):
            monkeypatch.setattr(mod, "chat", no_model, raising=False)

    def _grade(code, student="stu"):
        decomp = {"header": "def invert(d):",
                  "chunks": [types.SimpleNamespace(**c) for c in STEPS]}
        sid = sessions.create_session(dict(PROBLEM), decomp, "h-shape",
                                      student_id=student)["session_id"]
        return grading.grade_submission(sessions.load_session(sid), code,
                                        oracle_loader=lambda p: list(ORACLE))
    return _grade


@pytest.mark.parametrize("label,code", [
    # THE AUDIT'S CASE. Both of these were answered with indentation advice
    # about a line that has nothing above it.
    ("two comments", "# Count each value in d.\n# Keep only values that occur once."),
    ("one comment", "# Count each value in d."),
    ("a comment and blank lines", "\n# count them\n\n"),
])
def test_an_answer_with_no_code_is_named_as_such(grade, label, code):
    res = grade(code, "stu-" + label.replace(" ", "-"))
    assert res.reason_code == "comments_only", (label, res.reason_code,
                                                res.student_reason)
    assert "no code" in res.student_reason, res.student_reason
    # The wrong answer this used to give, in the words it used to give it.
    assert "indentation" not in res.student_reason.lower(), res.student_reason


@pytest.mark.parametrize("label,code", [
    ("the def line again", "def invert(d):\n    counts = {}\n    return counts"),
    ("a whole class", "class Inverter:\n    def invert(self, d):\n"
                      "        counts = {}\n        return counts"),
])
def test_script_shaped_code_in_a_body_box_is_named(grade, label, code):
    res = grade(code, "stu-" + label.replace(" ", "-"))
    assert res.reason_code == "redefined_function", (label, res.reason_code)
    assert "already written for you" in res.student_reason, res.student_reason


def test_a_real_indentation_fault_is_still_an_indentation_fault(grade):
    """The gate that was answering first still has to answer for its own case -
    moving `comments_only` in front of it must not have taken this with it."""
    res = grade("counts = {}\n  for value in d.values():\n        pass")
    assert res.verdict == "incorrect"
    assert res.reason_code in ("indentation_error", "syntax_error"), res.reason_code


def test_an_ordinary_correct_answer_is_untouched(grade):
    """None of these gates may stand between a student and a correct answer."""
    res = grade(STEPS[0]["reference"])
    assert res.verdict == "correct", (res.reason_code, res.student_reason)
    assert res.tier == "execution-reference", res.tier
