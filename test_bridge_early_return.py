"""test_bridge_early_return.py - a correct step that ALSO finishes the easy
case early is matched on the inputs it did not finish.

THE REPORT (18 Oct, GPT's HW4 round 2, __contains__ after a rebuild). The
step searched from the tail and returned False at once on a miss; the
roadmap's step left `current` as None on a miss for step 2 to return False.
On every hit the two held the same node - but on a miss the student's name
never reached the end of the step (it had returned), was recorded as
"missing", and "missing" against None is a mismatch. No pairing, three model
calls, "could not confirm". An input their code already finished has nothing
for the next step to continue: it is no evidence either way. The oracle run
of the whole composite still decides, exactly as before.

REAL: grade_submission, the bridge, sessions, execution. FAKED: the model
tiers (a call fails the test).
"""
import types

import pytest

SOLUTION = ("def find(items, target):\n"
            "    i = 0\n"
            "    hit = None\n"
            "    while i < len(items):\n"
            "        if items[i] == target:\n"
            "            hit = i\n"
            "            break\n"
            "        i += 1\n"
            "    if hit is None:\n"
            "        return -1\n"
            "    return hit\n")
PROBLEM = {"slug": "find-hit", "title": "Find", "solution": SOLUTION, "entry_hint": "find",
           "description": "The index of target in items (all different), or -1."}
STEPS = ["i = 0\nhit = None\nwhile i < len(items):\n    if items[i] == target:\n"
         "        hit = i\n        break\n    i += 1",
         "if hit is None:\n    return -1\nreturn hit"]
ORACLE = [{"input": [[4, 5, 6], 4], "expected": 0}, {"input": [[4, 5, 6], 6], "expected": 2},
          {"input": [[4, 5, 6], 9], "expected": -1}, {"input": [[], 1], "expected": -1},
          {"input": [[7], 7], "expected": 0}]
# From the other end, set up differently, and the miss finished here.
BACKWARDS = ("pos = len(items) - 1\nwhile pos >= 0 and items[pos] != target:\n"
             "    pos -= 1\nif pos < 0:\n    return -1")


@pytest.fixture
def grade(monkeypatch, tmp_path):
    from main import grading, identity, ollama_client, sessions
    monkeypatch.setenv("MICROTUTOR_SESSION_DB", str(tmp_path / "s.sqlite3"))
    monkeypatch.setenv("MICROTUTOR_TRACE_FILE", str(tmp_path / "t.jsonl"))
    monkeypatch.setattr(identity, "_RESOLVED_PATH", str(tmp_path / "r.json"))
    boom = lambda *a, **k: (_ for _ in ()).throw(AssertionError("a model was asked"))
    monkeypatch.setattr(ollama_client, "_openai_chat", boom)
    monkeypatch.setattr(grading, "_request_adaptation", boom)
    monkeypatch.setattr(grading, "_request_completion", boom)

    def run(code):
        decomp = {"header": "def find(items, target):", "chunks": [
            types.SimpleNamespace(step_id=f"Part {i + 1}", expected_type="code", prompt="p",
                                  reference=r) for i, r in enumerate(STEPS)]}
        sid = sessions.create_session(dict(PROBLEM), decomp, "h-find", student_id="stu")["session_id"]
        grading._VERDICT_MEMO.clear()
        return grading.grade_submission(sessions.load_session(sid), code,
                                        oracle_loader=lambda p: list(ORACLE))
    return run


def test_a_step_that_finishes_the_miss_early_is_matched_on_the_hits(grade):
    r = grade(BACKWARDS)
    assert (r.verdict, r.tier) == ("correct", "execution-bridged"), (r.tier, r.student_reason)


def test_finishing_the_miss_WRONG_is_still_not_accepted(grade, monkeypatch):
    """The skipped inputs are no evidence FOR a pairing either: the composite
    run on every test is what accepts, and a wrong early answer fails it."""
    from main import grading
    monkeypatch.setattr(grading, "_request_adaptation", lambda *a, **k: ("", []))
    monkeypatch.setattr(grading, "_request_completion", lambda *a, **k: "")
    r = grade(BACKWARDS.replace("return -1", "return 0"))
    assert r.verdict != "correct", r


def test_inputs_that_were_finished_are_skipped_but_never_all_of_them():
    from main import bridge
    u = repr(bridge.UNBOUND)
    ref = {"hit": ("0", "2", "None", "None")}
    stu = {"pos": ("0", "2", u, u), "other": ("1", "2", u, u)}
    assert bridge._candidates(ref, stu, early={2, 3}) == {"hit": ["pos"]}
    assert bridge._candidates(ref, stu) == {"hit": []}, "without the early inputs known, as before"
    assert bridge._candidates(ref, stu, early={0, 1, 2, 3}) == {"hit": []}, "no evidence is no match"
