"""test_crash_check_gaps.py - the crash check sees every crash in the student's
own lines: not only those in the first five failing tests, and in block tests.

THE GAPS (replayed over all 468 real submissions, 28 Sep).
  * A run reported at most FIVE failing tests, and the crash check read only
    those. An unfinished middle step gives plain wrong answers on most tests, so
    a crash in test 7 was never seen: calculator-is-number's
    `for i in splitted[0]:` crashes with IndexError on blank input, and three
    submissions got "couldn't confirm" instead of that line. 305 of 339
    non-passing class-method runs were cut off this way.
  * Block tests were skipped outright. One real step called the `calculate`
    PROPERTY as if it were a method - that line crashes every time it runs - and
    the old AI judge accepted it.
Replayed with both closed: 62 steps failed instead of 57, and none of the 62 is
a step tier 1, tier 2 or an early finish proves correct.

Self-contained: inline problems, inline oracles, no model.
"""
import types

import pytest

from test_own_crash import PROBLEM as BOX, STEPS as BOX_STEPS

FN = {"slug": "first-item", "title": "first", "description": "The first item, or None.",
      "solution": "def first(xs):\n    item = xs[0] if xs else None\n    return item\n"}
FN_STEPS = [dict(step_id="Part 1", expected_type="code", prompt="Pick the first item "
                 "(or nothing) and keep it for the next step.",
                 reference="item = xs[0] if xs else None"),
            dict(step_id="Part 2", expected_type="code", prompt="Hand it back.",
                 reference="return item")]
# Five tests the unfinished step gets plainly wrong (it returns nothing yet),
# THEN the one it crashes on.
FIVE_WRONG_THEN_EMPTY = [{"input": [[k]], "expected": k} for k in range(1, 6)] + \
                        [{"input": [[]], "expected": None}]
BOX_FIVE_WRONG_THEN_EMPTY = [{"input": [[["put", k], ["get"]]], "expected": [None, k]}
                             for k in range(1, 6)] + [{"input": [[["get"]]], "expected": [None]}]
# `self.__v + 0`: fine once something is in the box, a TypeError on an empty one.
CRASHES_WHEN_EMPTY = "v = self.__v + 0"


@pytest.fixture
def grade(monkeypatch, tmp_path):
    monkeypatch.setenv("MICROTUTOR_SESSION_DB", str(tmp_path / "s.sqlite3"))
    from main import grading, identity, ollama_client, sessions
    monkeypatch.setattr(identity, "_RESOLVED_PATH", str(tmp_path / "r.json"))
    grading._VERDICT_MEMO.clear()

    def no_model(*_a, **_k):
        raise RuntimeError("model disabled in this test")
    monkeypatch.setattr(ollama_client, "_openai_chat", no_model)
    monkeypatch.setattr(ollama_client, "_ollama_chat", no_model)

    def run(code, oracle, problem=BOX, steps=BOX_STEPS, header="def get(self):"):
        decomp = {"header": header, "chunks": [types.SimpleNamespace(**c) for c in steps]}
        sid = sessions.create_session(dict(problem), decomp, "h-" + problem["slug"],
                                      student_id="stu")["session_id"]
        return grading.grade_submission(sessions.load_session(sid), code,
                                        oracle_loader=lambda p: list(oracle))
    return run


def test_a_method_crash_after_five_wrong_answers_is_caught(grade):
    r = grade(CRASHES_WHEN_EMPTY, BOX_FIVE_WRONG_THEN_EMPTY)
    assert (r.verdict, r.tier, r.reason_code) == \
        ("incorrect", "execution-crash", "own_code_crash"), r
    assert "line 1 of your answer" in r.student_reason and "TypeError" in r.student_reason


def test_a_function_crash_after_five_wrong_answers_is_caught(grade):
    r = grade("item = xs[0]", FIVE_WRONG_THEN_EMPTY, problem=FN, steps=FN_STEPS,
              header="def first(xs):")
    assert (r.verdict, r.tier) == ("incorrect", "execution-crash"), r
    assert "IndexError" in r.student_reason, r.student_reason


def test_the_student_still_sees_at_most_three_cases(grade):
    r = grade(CRASHES_WHEN_EMPTY, BOX_FIVE_WRONG_THEN_EMPTY)
    assert len(r.failing_cases or []) <= 3


def test_a_crash_in_a_block_test_is_caught(grade):
    r = grade(CRASHES_WHEN_EMPTY, [{"input": ["x = Box()\nx.get()"], "expected": [None, None]}])
    assert (r.verdict, r.tier) == ("incorrect", "execution-crash"), r


def test_a_block_crash_after_an_earlier_call_is_not_counted(grade):
    """The second get() crashes, but the first one ran their lines already -
    so something they have not written yet could have mattered."""
    block = "x = Box()\nx.put(1)\nx.get()\nx.put(None)\nx.get()"
    r = grade(CRASHES_WHEN_EMPTY, [{"input": [block], "expected": [None, None, 1, None, None]}])
    assert r.tier != "execution-crash", r


def test_a_block_statement_that_calls_it_twice_or_loops_is_not_counted(grade):
    for block in ("x = Box()\n[x.get(), x.get()]",
                  "x = Box()\nfor _ in range(2):\n    x.get()"):
        r = grade(CRASHES_WHEN_EMPTY, [{"input": [block], "expected": [None, None]}])
        assert r.tier != "execution-crash", (block, r)


def test_correct_code_is_still_never_caught(grade):
    oracle = BOX_FIVE_WRONG_THEN_EMPTY + [{"input": ["x = Box()\nx.get()"], "expected": [None, None]}]
    r = grade("v = self.__v", oracle)
    assert r.verdict == "correct" and r.tier != "execution-crash", r
