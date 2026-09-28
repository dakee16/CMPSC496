"""test_adapter_evidence.py - tier 3 shows a student a failing case only when
its own proof (calibration) actually ran the rewrite on that case.

THE GAP. Tier 3 has a model rewrite the teacher's remaining steps in the
student's names, and trusts the rewrite only once it passes on the TEACHER'S own
steps (calibration). Where the teacher's step returns early, the rewrite never
runs during calibration - calibration says nothing about those inputs - yet a
wrong answer there was shown to the student as THEIR failing case. Measured on
the 116 saved roadmap steps (28 Sep): 57 skip the rewrite on some tests, and
calculateExpressions step 2 skips it on 32 of 40.

Everything goes through grade_submission; only the two model proposals are
stubbed, so every gate runs in its real order.
"""
import types

import pytest

SOLUTION = ("def f(xs):\n    if not xs:\n        return []\n"
            "    out = sorted(xs)\n    return out\n")
PROBLEM = {"slug": "sort-t3e", "title": "f", "description": "The items in order.",
           "solution": SOLUTION}
STEPS = [dict(step_id="Part 1", expected_type="code",
              prompt="Put the items in order and keep them for the next step.",
              reference="if not xs:\n    return []\nout = sorted(xs)"),
         dict(step_id="Part 2", expected_type="code", prompt="Hand them back.",
              reference="return out")]
ORACLE = [{"input": [[]], "expected": []},
          {"input": [[3, 1]], "expected": [1, 3]},
          {"input": [[2, 2]], "expected": [2, 2]}]
# Right wherever calibration runs it; wrong on [] - which the teacher's early
# return means calibration never runs it on.
REWRITE = "return sorted(srt) if srt else None"
ALIASES = [{"teacher": "out", "student": "srt"}]
CORRECT = "srt = sorted(xs, reverse=True)"      # a different, correct shape
BUGGY = "srt = sorted(set(xs))"                 # loses repeated items


@pytest.fixture
def grade(monkeypatch, tmp_path):
    monkeypatch.setenv("MICROTUTOR_SESSION_DB", str(tmp_path / "s.sqlite3"))
    from main import grading, identity, ollama_client, sessions
    monkeypatch.setattr(identity, "_RESOLVED_PATH", str(tmp_path / "r.json"))
    grading._VERDICT_MEMO.clear()

    def no_model(*_a, **_k):                  # the diagnosis question falls back
        raise RuntimeError("model disabled in this test")
    monkeypatch.setattr(ollama_client, "_openai_chat", no_model)
    monkeypatch.setattr(ollama_client, "_ollama_chat", no_model)
    monkeypatch.setattr(grading, "_request_adaptation",
                        lambda *a, **k: (REWRITE, list(ALIASES)))

    def run(code, completion=""):
        monkeypatch.setattr(grading, "_request_completion", lambda *a, **k: completion)
        decomp = {"header": "def f(xs):",
                  "chunks": [types.SimpleNamespace(**c) for c in STEPS]}
        sid = sessions.create_session(dict(PROBLEM), decomp, "h-t3e",
                                      student_id="stu")["session_id"]
        grading._VERDICT_MEMO.clear()
        return grading.grade_submission(sessions.load_session(sid), code,
                                        oracle_loader=lambda p: list(ORACLE))
    return run


def _mentions_empty(result):
    return any("f([])" in c for c in (result.failing_cases or []))


def test_a_correct_step_is_not_shown_a_case_calibration_never_ran(grade):
    r = grade(CORRECT)
    assert r.consume_attempt is False and r.verdict != "incorrect", r
    assert r.tier != "execution-adapted" and not _mentions_empty(r), r


def test_a_buggy_step_is_still_shown_the_case_calibration_did_run(grade):
    r = grade(BUGGY)
    assert (r.verdict, r.tier) == ("indeterminate", "execution-adapted"), r
    assert any("f([2, 2])" in c for c in r.failing_cases), r.failing_cases
    assert not _mentions_empty(r), r.failing_cases


def test_tier_4_can_still_confirm_the_correct_step(grade):
    r = grade(CORRECT, completion="return sorted(srt)")
    assert (r.verdict, r.tier) == ("correct", "execution-completed"), r
