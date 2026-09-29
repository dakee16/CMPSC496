"""test_early_return_message.py - a step that returns too early is told so.

THE AUDIT (29 Sep, _isNumber step 1 of 2). `return True` ran fine and ended the
function, and the student was told "Check that the lines you wrote actually
run - code inside something that never happens is never reached": advice for
a different mistake. Every step but the last is asked to keep its result for
the next step, so that is what the message now says. A step that binds nothing
and does not return (`print(words)`) keeps the old message.

REAL: grade_submission end to end. FAKED: the two model tiers, answering
nothing, so the step lands where the audit's did - not confirmed, no attempt.
"""
import types

import pytest

from test_step_seating import ORACLE, PROBLEM, STEPS


@pytest.fixture
def grade_step1(monkeypatch, tmp_path):
    monkeypatch.setenv("MICROTUTOR_SESSION_DB", str(tmp_path / "s.sqlite3"))
    from main import grading, identity, ollama_client, sessions
    monkeypatch.setattr(identity, "_RESOLVED_PATH", str(tmp_path / "r.json"))
    monkeypatch.setattr(ollama_client, "_openai_chat",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("model")))
    monkeypatch.setattr(grading, "_request_adaptation", lambda *a, **k: ("", []))
    monkeypatch.setattr(grading, "_request_completion", lambda *a, **k: "")
    grading._VERDICT_MEMO.clear()

    def _grade(code):
        decomp = {"header": "def f(words):",
                  "chunks": [types.SimpleNamespace(**c) for c in STEPS]}
        sid = sessions.create_session(dict(PROBLEM), decomp, "h-early",
                                      student_id="stu")["session_id"]
        return grading.grade_submission(sessions.load_session(sid), code,
                                        oracle_loader=lambda p: list(ORACLE))
    yield _grade
    grading._VERDICT_MEMO.clear()


def test_a_step_that_returns_is_told_to_keep_its_result(grade_step1):
    r = grade_step1("return words")
    assert (r.verdict, r.consume_attempt) == ("indeterminate", False), r
    assert "returns from the function" in r.student_reason, r.student_reason
    assert "keep it for the next step" in r.student_reason
    assert "never reached" not in r.student_reason


def test_a_step_that_binds_nothing_without_returning_keeps_the_old_message(grade_step1):
    r = grade_step1("print(words)")
    assert r.verdict == "indeterminate", r
    assert "never reached" in r.student_reason, r.student_reason
    assert "returns from the function" not in r.student_reason
