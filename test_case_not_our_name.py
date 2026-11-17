"""test_case_not_our_name.py - the case a student is shown is about THEIR
mistake, never about a name of ours they did not use.

THE REPORT (19 Oct, GPT's HW4 round 3, mruEvict). The step was saved as
`evicted` where the teacher's next step reads `node`. Joined as written, every
call crashed on `node`, and the case shown ended "you gave: <crashed:
NameError>" - a crash in OUR lines, about a name the student never chose. When
the names pair one-to-one, the case comes from the run with THEIR names in the
rest of the solution (bridge.renamed_failure) - the difference that is theirs.

REAL: parse_assignment_file, grade_submission, the bridge, execution. FAKED:
the model tiers find nothing.
"""
import types

import pytest

SRC = '''"""Piles"""


class Pile:
    """
        >>> p = Pile()
        >>> p.push(1)
        >>> p.take()
        1
    """
    # --- steps: take ---
    def __init__(self):
        self.items = []

    def push(self, x):
        self.items.append(x)

    def take(self):
        """Remove and return the item on top, or None when it is empty."""
        if not self.items:
            return None
        top = self.items.copy()[-1]
        self.items = self.items[:-1]
        return top
'''
# `.copy()` is a call, so the grader cannot recompute `top` for a student who
# named it otherwise - as HW4's `node`, which the next step changes in place.
STEPS = ["if not self.items:\n    return None\ntop = self.items.copy()[-1]\nself.items = self.items[:-1]",
         "return top"]
CALLS = [["new"], ["push", 1], ["push", 2], ["take"], ["take"], ["take"]]
ORACLE = [{"input": [CALLS], "expected": [None, None, None, 2, 1, None]}]


@pytest.fixture
def grade(monkeypatch, tmp_path):
    from main import grading, identity, ollama_client, sessions
    from main.assignments import parse_assignment_file
    monkeypatch.setenv("MICROTUTOR_SESSION_DB", str(tmp_path / "s.sqlite3"))
    monkeypatch.setenv("MICROTUTOR_TRACE_FILE", str(tmp_path / "t.jsonl"))
    monkeypatch.setattr(identity, "_RESOLVED_PATH", str(tmp_path / "r.json"))
    monkeypatch.setattr(ollama_client, "_openai_chat",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no model")))
    monkeypatch.setattr(grading, "_request_adaptation", lambda *a, **k: ("", []))
    monkeypatch.setattr(grading, "_request_completion", lambda *a, **k: "")
    # As on HW4's long tests: no tracing QUESTION can be built, so a CASE is
    # what the student gets - the path this file is about.
    monkeypatch.setattr(grading, "_with_diagnosis", lambda result, *a, **k: result)
    problem = parse_assignment_file(SRC, "pile.py")["problems"][0]

    def run(code):
        decomp = {"header": "def take(self):", "chunks": [
            types.SimpleNamespace(step_id=f"Part {i + 1}", expected_type="code", prompt="p",
                                  reference=r) for i, r in enumerate(STEPS)]}
        sid = sessions.create_session(dict(problem), decomp, "h-pile", student_id="stu")["session_id"]
        grading._VERDICT_MEMO.clear()
        return grading.grade_submission(sessions.load_session(sid), code,
                                        oracle_loader=lambda p: list(ORACLE))
    return run


def test_a_renamed_wrong_step_is_shown_its_own_mistake_not_our_name(grade):
    # `last` for the teacher's `top`, and the wrong end removed.
    r = grade("if not self.items:\n    return None\nlast = self.items.copy()[-1]\nself.items = self.items[1:]")
    assert r.verdict == "indeterminate" and r.failing_cases, (r.reason_code, r.student_reason)
    case = r.failing_cases[0]
    assert "NameError" not in case, case
    assert "own names" in r.student_reason, r.student_reason


def test_the_same_names_still_show_the_ordinary_case(grade):
    r = grade("if not self.items:\n    return None\ntop = self.items.copy()[-1]\nself.items = self.items[1:]")
    assert r.verdict != "correct" and r.failing_cases, r
    assert "NameError" not in r.failing_cases[0]


def test_a_name_crash_past_the_cut_is_still_seen():
    """HW4's long cases: the summary is cut at 160 characters and the crash sat
    past it, only in the first-difference item."""
    from main.grading import _crashed_on_a_name
    cut = {"index": 1, "got": {"repr": "[None, 'INSERTED ...<truncated>", "type": "list"},
           "at": 13, "got_at": {"repr": "'!NameError\\x00name node is not defined'", "type": "str"}}
    assert _crashed_on_a_name([cut])
    assert not _crashed_on_a_name([{"index": 1, "got": {"repr": "[1, 2]", "type": "list"}}])
