"""test_failing_case_difference.py - a failing case shows WHERE the answers differ.

THE REPORT (15 Oct, GPT's audit of HW4 put). A wrong `>=` step was rightly
rejected, but the case it was shown read

    expected: [None, 'INSERTED: CONTENT ID: 1 SIZE: 40 ...', ... ...<truncated>
    you gave: [None, 'INSERTED: CONTENT ID: 1 SIZE: 40 ...', ... ...<truncated>

- both cut at 160 characters, both identical up to the cut. The sandbox
shortens each value on its own, and a long run of calls that only goes wrong
at its end (the exact fit, last) hides the one thing the case is shown for.

REAL: grade_submission, sessions, the sandbox run and the rendering of the
case. FAKED: nothing is asked of a model (it would fail the test).
"""
import types

import pytest

SHELF = '''"""Shelves"""


class Shelf:
    """
        >>> s = Shelf(2)
        >>> s.add('pen')
        'ADDED: pen - the shelf now holds 1 items'
    """
    # --- steps: add ---
    def __init__(self, size):
        self.size = size
        self.items = []

    def add(self, name):
        """Put name on the shelf and say so; when the shelf is already full,
        say there is no room and change nothing."""
        if len(self.items) >= self.size:
            return 'FULL: there is no room for ' + name + ' on this shelf'
        self.items.append(name)
        return 'ADDED: ' + name + ' - the shelf now holds ' + str(len(self.items)) + ' items'
'''
STEPS = ["if len(self.items) >= self.size:\n"
         "    return 'FULL: there is no room for ' + name + ' on this shelf'",
         "self.items.append(name)\n"
         "return 'ADDED: ' + name + ' - the shelf now holds ' + str(len(self.items)) + ' items'"]
NAMES = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
# Five fit, the sixth does not: the one difference a `>` makes is the LAST output.
CALLS = [["new", 5]] + [["add", n] for n in NAMES]
EXPECTED = ([None] + [f"ADDED: {n} - the shelf now holds {i} items" for i, n in enumerate(NAMES[:5], 1)]
            + ["FULL: there is no room for foxtrot on this shelf"])


@pytest.fixture
def grade(monkeypatch, tmp_path):
    from main import grading, identity, ollama_client, sessions
    from main.assignments import parse_assignment_file
    monkeypatch.setenv("MICROTUTOR_SESSION_DB", str(tmp_path / "s.sqlite3"))
    monkeypatch.setenv("MICROTUTOR_TRACE_FILE", str(tmp_path / "t.jsonl"))
    monkeypatch.setattr(identity, "_RESOLVED_PATH", str(tmp_path / "r.json"))
    monkeypatch.setattr(ollama_client, "_openai_chat",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no model")))
    for name in ("_request_adaptation", "_request_completion"):
        monkeypatch.setattr(grading, name, lambda *a, **k: (_ for _ in ()).throw(AssertionError("no model")))
    problem = parse_assignment_file(SHELF, "shelf.py")["problems"][0]

    def run(code):
        decomp = {"header": "def add(self, name):", "chunks": [
            types.SimpleNamespace(step_id=f"Part {i + 1}", expected_type="code", prompt="p",
                                  reference=r) for i, r in enumerate(STEPS)]}
        sid = sessions.create_session(dict(problem), decomp, "h-shelf", student_id="stu")["session_id"]
        grading._VERDICT_MEMO.clear()
        return grading.grade_submission(sessions.load_session(sid), code,
                                        oracle_loader=lambda p: [{"input": [CALLS], "expected": EXPECTED}])
    return run


def test_the_case_shown_names_the_first_output_that_differs(grade):
    r = grade("if len(self.items) > self.size:\n"
              "    return 'FULL: there is no room for ' + name + ' on this shelf'")
    assert r.verdict == "incorrect", r
    (case,) = r.failing_cases
    # The cut summaries are the same up to their cut - the bug...
    assert "truncated" in case, case
    # ...so the difference itself has to be said, beside the call that made it.
    assert "x.add('foxtrot')" in case.split("first differ", 1)[-1], case
    assert "FULL: there is no room for foxtrot" in case and "ADDED: foxtrot" in case, case


def test_a_short_case_is_shown_as_it_was(grade):
    """Nothing is cut, so nothing is added: the two lines already show it."""
    from main import grading
    out = grading._render_case({"slug": "x"}, {"input": [1]},
                               {"got": {"repr": "[1, 2]", "type": "list"}, "expected": {"repr": "[1, 3]", "type": "list"}, "at": 1,
                                "got_at": {"repr": "2", "type": "int"}, "expected_at": {"repr": "3", "type": "int"}})
    assert "first differ" not in out, out
