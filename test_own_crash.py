"""test_own_crash.py - a crash in the student's OWN lines is caught at the step.

THE REPORT. A real student wrote `self._expr` for `self.__expr` in step 1 of
Calculator.calculate. Execution could not confirm it; the AI judges, which read
code and never run it, accepted it; the broken step was locked in, and he then
failed step 2 nine times against a crash he could no longer edit. The same
happened to him on two more problems. Replayed over every real submission, the
rule below catches those steps on the first try, with zero false failures.

WHY A CRASH MAY BE FAILED WHEN A WRONG VALUE MAY NOT. At a middle step a
different-looking value can be a correct different approach; a crash cannot be
undone by anything written after it. So a crash in their own lines, the first
time those lines run, is a verdict about their code alone. What is NOT counted
is pinned here too: a crash the teacher's version also has, and a step whose
earlier, judge-accepted neighbour is the one crashing (that step gets pointed
at instead - "C").

Self-contained: inline class, inline oracle, no model.
"""
import types

import pytest

PREFIX = ("class Box:\n"
          "    def __init__(self):\n"
          "        self.__v = None\n"
          "        self.__items = []\n"
          "\n"
          "    def put(self, v):\n"
          "        self.__v = v\n"
          "\n"
          "    def get(self):\n")
SOLUTION = "def get(self):\n    v = self.__v\n    return v\n"
PROBLEM = {"slug": "box-get", "title": "Box.get", "solution": SOLUTION,
           "description": "Return what was put in the box.",
           "context_prefix": PREFIX, "context_suffix": "", "context_indent": 8,
           "entry_hint": "get", "group_title": "Box"}
ORACLE = [{"input": [[["get"]]], "expected": [None]},
          {"input": [[["put", 5], ["get"]]], "expected": [None, 5]}]
STEPS = [dict(step_id="Part 1", expected_type="code",
              prompt="Read the stored value and keep it.", reference="v = self.__v"),
         dict(step_id="Part 2", expected_type="code",
              prompt="Hand it back.", reference="return v")]


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("MICROTUTOR_SESSION_DB", str(tmp_path / "s.sqlite3"))
    from main import grading, identity, ollama_client, sessions
    monkeypatch.setattr(identity, "_RESOLVED_PATH", str(tmp_path / "r.json"))
    calls = []

    def no_model(*_a, **_k):
        calls.append(1)
        raise RuntimeError("model disabled in this test")
    monkeypatch.setattr(ollama_client, "_openai_chat", no_model)

    def session(problem=PROBLEM, steps=STEPS, accepted=()):
        decomp = {"header": "def get(self):",
                  "chunks": [types.SimpleNamespace(**c) for c in steps]}
        sid = sessions.create_session(dict(problem), decomp, "h-crash",
                                      student_id="stu")["session_id"]
        for n, (code, tier) in enumerate(accepted):
            s = sessions.load_session(sid)
            sessions.commit_outcome(sid, f"pre-{n}", s["revision"],
                                    {"verdict": "correct", "tier": tier},
                                    accept_code=code, provenance="student")
        return sessions.load_session(sid)

    def grade(code, oracle=ORACLE, **kw):
        return grading.grade_submission(session(**kw), code,
                                        oracle_loader=lambda p: list(oracle))
    return type("E", (), {"grade": grade, "calls": calls})


def test_the_reported_typo_fails_at_step_one_and_names_the_line(env):
    r = env.grade("v = self._v")
    assert (r.verdict, r.tier, r.reason_code) == \
        ("incorrect", "execution-crash", "own_code_crash"), r
    assert r.consume_attempt, "the student's own crash uses an attempt"
    assert "line 1 of your answer" in r.student_reason, r.student_reason
    assert "no attribute '_v'" in r.student_reason, r.student_reason
    assert env.calls == [], "caught without any model"


def test_correct_code_is_never_caught(env):
    r = env.grade("v = self.__v")
    assert r.tier != "execution-crash" and r.verdict == "correct", r


def test_a_crash_the_teachers_version_also_has_is_not_counted(env):
    """pop() on an empty stack may be MEANT to raise - the oracle says so.

    The sequence has to FAIL somewhere else for this to be tested at all: when
    a student's crash is the only output and it matches, the case simply
    passes and the rule is never consulted. Here their step alone gets the
    first call right (the intended IndexError) and the last one wrong (their
    return is not written yet)."""
    prefix = PREFIX.replace("        self.__v = v\n", "        self.__items.append(v)\n")
    prob = {**PROBLEM, "context_prefix": prefix,
            "solution": "def get(self):\n    v = self.__items[0]\n    return v\n"}
    steps = [dict(STEPS[0], reference="v = self.__items[0]"), STEPS[1]]
    oracle = [{"input": [[["get"], ["put", 5], ["get"]]],
               "expected": ["!IndexError", None, 5]}]
    r = env.grade("v = self.__items[0]", oracle=oracle, problem=prob, steps=steps)
    assert r.tier != "execution-crash", r


def test_a_crash_in_an_earlier_judge_accepted_step_points_there(env):
    """C. Step 2 is fine; step 1, accepted by the judges unrun, is what
    crashes. The attempt is not charged to step 2, and they are sent to
    Rework on step 1."""
    r = env.grade("return v", accepted=[("v = self._v", "llm-judge")])
    assert (r.verdict, r.reason_code) == ("indeterminate", "earlier_step_crash"), r
    assert not r.consume_attempt
    assert "step 1" in r.student_reason and "Rework" in r.student_reason, r.student_reason


def test_a_wrong_final_answer_behind_a_judge_accepted_step_mentions_it(env):
    """C, the wrong-values case: nothing crashed, so no step can be blamed -
    the attempt counts, but the earlier unrun step is named as a suspect."""
    r = env.grade("return v + 1 if v else 0",
                  accepted=[("v = self.__v", "llm-judge")])
    assert r.verdict == "incorrect", r
    assert "Step 1 was accepted earlier without being fully confirmed" in r.student_reason, \
        r.student_reason


def test_nested_or_recursive_code_is_left_alone():
    """The two shapes where a crash can be an artifact of the missing later
    steps: lines inside a block that continues after them, and recursion."""
    from main.grading import _entry_def
    nested = ("def f(xs):\n"
              "    for x in xs:\n"
              "        y = x.upper()\n"          # their line, line 3
              "        z = y\n"
              "    return z\n")
    assert _entry_def(nested, 3, 3) is None
    recursive = ("def f(n):\n"
                 "    r = f(n - 1)\n"
                 "    return r\n")
    assert _entry_def(recursive, 2, 2) is None
    flat = ("def f(xs):\n"
            "    y = xs.upper()\n"
            "    return y\n")
    assert _entry_def(flat, 2, 2)["name"] == "f"
