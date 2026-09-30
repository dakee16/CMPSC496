"""test_unstarted_loop_message.py - a step that only sets things up, where the
step is meant to START the loop the next step carries on inside, is told so.

THE REPORT (30 Sep). On get-postfix and calculateExpressions a student
answered step 1 ("prepare everything needed...") with the set-up alone - what
it asked for. The next step continues INSIDE a loop that step 1 is meant to
start, so their step joined to it could not even be read ("unexpected
indent"), nothing could confirm it, and they were told to "check what your code
produces against what the step asks for": no hint that a loop was expected.

The verdict does not change - still "could not confirm", still no attempt
used. Only the reason does, and only when the shape proves it: the next step
sits deeper than this one, and joined to their step it does not parse.
Everything goes through grade_submission; only the model proposals are
stubbed (to propose nothing), so tiers 3-4 run and find nothing.
"""
import types

import pytest

from test_restart_reroute import ORACLE, PROBLEM
from test_reword_pools import LOOPED
from test_student_open_never_generates import ROADMAP


@pytest.fixture
def grade(monkeypatch, tmp_path):
    monkeypatch.setenv("MICROTUTOR_SESSION_DB", str(tmp_path / "s.sqlite3"))
    from main import grading, identity, ollama_client, sessions
    monkeypatch.setattr(identity, "_RESOLVED_PATH", str(tmp_path / "r.json"))

    def no_model(*_a, **_k):
        raise RuntimeError("model disabled in this test")
    monkeypatch.setattr(ollama_client, "_openai_chat", no_model)
    monkeypatch.setattr(ollama_client, "_ollama_chat", no_model)
    monkeypatch.setattr(grading, "_request_adaptation", lambda *a, **k: ("", []))
    monkeypatch.setattr(grading, "_request_completion", lambda *a, **k: "")

    def run(code, roadmap=LOOPED):
        decomp = {"header": roadmap["header"],
                  "chunks": [types.SimpleNamespace(**c) for c in roadmap["chunks"]]}
        sid = sessions.create_session(dict(PROBLEM), decomp, "h-loop",
                                      student_id="stu")["session_id"]
        grading._VERDICT_MEMO.clear()
        return grading.grade_submission(sessions.load_session(sid), code,
                                        oracle_loader=lambda p: list(ORACLE))
    return run


def _hint(r):
    return "start" in r.student_reason and "repetition" in r.student_reason


def test_set_up_alone_is_told_the_step_should_start_the_loop(grade):
    r = grade("counts = {}")
    assert (r.verdict, r.tier) == ("indeterminate", "unconfirmed"), r
    assert r.consume_attempt is False
    assert _hint(r), r.student_reason


def test_a_step_that_does_start_the_loop_is_not_told_to(grade):
    r = grade("counts = {}\nfor c in txt:\n    c = c.upper()")
    assert r.verdict != "correct" and not _hint(r), r


def test_no_loop_hint_when_the_next_step_does_not_carry_on_inside_one(grade):
    r = grade("letters = sorted(txt)", roadmap=ROADMAP)
    assert r.tier == "unconfirmed" and not _hint(r), r


def test_the_teachers_own_step_is_still_accepted(grade):
    r = grade(LOOPED["chunks"][0]["reference"])
    assert r.verdict == "correct", r
