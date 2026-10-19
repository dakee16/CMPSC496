"""test_free_rejection.py - a wrong step is rejected without the model when the
teacher's remaining steps read nothing an earlier step made.

THE REPORT (13 Oct, HW4 student-path audit). Of 82 wrong steps, 35 were sent to
the paid model tiers - which exist because a failure joined to the teacher's
remaining steps might be only a naming mismatch. For 19 of them the remaining
steps read nothing an earlier step made (`return 'Cleared cache!'`), so there
was nothing to mismatch: the failure was the student's own.

THE TRAP, found while building it: a tail that carries on INSIDE a loop starts
indented, the name reader cannot parse it and reports "reads nothing", and a
correct renamed step was convicted (HW3 calculateExpressions). Such a tail
must count as depending on earlier steps.

REAL: grade_submission, sessions, execution. FAKED: the model tiers find
nothing (and a model call fails the test).
"""
import types

import pytest

SOLUTION = ("def average(nums):\n"
            "    if not nums:\n"
            "        return None\n"
            "    total = 0\n"
            "    for n in nums:\n"
            "        total += n\n"
            "    return total / len(nums)\n")
PROBLEM = {"slug": "average", "title": "Average", "solution": SOLUTION, "entry_hint": "average",
           "description": "The average of nums, or None for an empty list."}
_ns = {}
exec(SOLUTION, _ns)
TESTS = [{"input": [x], "expected": _ns["average"](x)} for x in ([], [4], [1, 2, 3], [-2, 2], [5, 5, 8])]
INDEPENDENT = [("if not nums:\n    return None"),
               ("total = 0\nfor n in nums:\n    total += n\nreturn total / len(nums)")]
DEPENDENT = [("if not nums:\n    return None\ntotal = 0"),
             ("for n in nums:\n    total += n\nreturn total / len(nums)")]
# Like HW3's calculateExpressions roadmap 3: step 1 opens the loop and starts
# its body, step 2 carries on inside it - so the tail starts indented.
IN_LOOP = [("if not nums:\n    return None\ntotal = 0\ncount = 0\nfor n in nums:\n    count += 1"),
           ("    total += n"), ("return total / count")]


@pytest.fixture
def grade(monkeypatch, tmp_path):
    from main import grading, identity, ollama_client, sessions
    monkeypatch.setenv("MICROTUTOR_SESSION_DB", str(tmp_path / "s.sqlite3"))
    monkeypatch.setenv("MICROTUTOR_TRACE_FILE", str(tmp_path / "t.jsonl"))
    monkeypatch.setattr(identity, "_RESOLVED_PATH", str(tmp_path / "r.json"))
    asked = []
    monkeypatch.setattr(ollama_client, "_openai_chat",
                        lambda *a, **k: asked.append(1) or (_ for _ in ()).throw(RuntimeError("no model")))
    monkeypatch.setattr(grading, "_request_adaptation", lambda *a, **k: asked.append(1) or ("", []))
    monkeypatch.setattr(grading, "_request_completion", lambda *a, **k: asked.append(1) or "")

    def run(steps, at, code, before=()):
        decomp = {"header": "def average(nums):", "chunks": [
            types.SimpleNamespace(step_id=f"Part {i + 1}", expected_type="code", prompt="p",
                                  reference=r) for i, r in enumerate(steps)]}
        sid = sessions.create_session(dict(PROBLEM), decomp, "h-avg", student_id="stu")["session_id"]
        for i, b in enumerate(before):
            s = sessions.load_session(sid)
            sessions.commit_outcome(sid, f"p{i}", s["revision"], {"verdict": "correct",
                                    "tier": "execution-reference"}, accept_code=b, provenance="student")
        asked.clear()
        grading._VERDICT_MEMO.clear()
        return grading.grade_submission(sessions.load_session(sid), code,
                                        oracle_loader=lambda p: list(TESTS))
    run.asked = asked
    return run


def test_a_wrong_step_with_an_independent_tail_is_rejected_for_free(grade):
    r = grade(INDEPENDENT, 0, "if nums:\n    return None")           # inverted guard
    assert (r.verdict, r.tier, r.deterministic) == ("incorrect", "execution-reference", True), r
    assert r.reason_code == "reference_fail_independent"
    assert grade.asked == [], "no model was needed to say so"
    assert r.failing_cases, "the case it gets wrong is shown"


def test_a_right_step_with_an_independent_tail_is_still_right(grade):
    r = grade(INDEPENDENT, 0, "if len(nums) == 0:\n    return None")
    assert r.verdict == "correct", r


def test_a_correct_step_the_tail_cannot_read_is_not_convicted(grade):
    """It skips the `total` the teacher's next step reads - correct, but the
    tail cannot run on it. That is the model tiers' case, never a verdict."""
    r = grade(DEPENDENT, 0, "if not nums:\n    return None\nrunning = 0")
    assert r.verdict != "incorrect", r


def test_a_tail_inside_a_loop_is_never_read_as_independent(grade):
    """The trap: the tail starts indented. A correct step 1 with its own name
    for the tally must not be convicted because the tail could not be parsed."""
    r = grade(IN_LOOP, 0, "if not nums:\n    return None\nrunning = 0\nk = 0\nfor n in nums:\n    k += 1")
    assert r.verdict != "incorrect", r


DRAIN = ("def drain(n):\n"
         "    if n < 0:\n"
         "        return 'BAD'\n"
         "    while n != 0:\n"
         "        n -= 1\n"
         "    return 'DONE'\n")


def test_a_step_that_lets_a_later_loop_run_forever_is_not_told_it_loops(monkeypatch, tmp_path):
    """GPT's audit of HW4 put (15 Oct): a missing guard step let the
    teacher's eviction loop run forever, and the student - whose answer has no
    loop - was told "Your step makes the solution take too long - it may loop
    forever"."""
    from main import grading, identity, ollama_client, sessions
    monkeypatch.setenv("MICROTUTOR_SESSION_DB", str(tmp_path / "s.sqlite3"))
    monkeypatch.setenv("MICROTUTOR_TRACE_FILE", str(tmp_path / "t.jsonl"))
    monkeypatch.setattr(identity, "_RESOLVED_PATH", str(tmp_path / "r.json"))
    monkeypatch.setattr(ollama_client, "_openai_chat",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no model")))
    problem = {"slug": "drain", "title": "Drain", "solution": DRAIN, "entry_hint": "drain",
               "description": "Count n down to zero and say DONE; 'BAD' for a negative n."}
    steps = ["if n < 0:\n    return 'BAD'", "while n != 0:\n    n -= 1\nreturn 'DONE'"]
    decomp = {"header": "def drain(n):", "chunks": [
        types.SimpleNamespace(step_id=f"Part {i + 1}", expected_type="code", prompt="p",
                              reference=r) for i, r in enumerate(steps)]}
    sid = sessions.create_session(dict(problem), decomp, "h-drain", student_id="stu")["session_id"]
    grading._VERDICT_MEMO.clear()
    # A guard on the wrong side: -1 gets past it and the loop never ends.
    # (`pass` alone is now named as no code before anything runs.)
    r = grading.grade_submission(sessions.load_session(sid), "if n > 100:\n    return 'BAD'",
                                 oracle_loader=lambda p: [
        {"input": [3], "expected": "DONE"}, {"input": [0], "expected": "DONE"},
        {"input": [-1], "expected": "BAD"}])
    assert (r.verdict, r.deterministic) == ("incorrect", True), r
    said = r.student_reason
    assert "never" in said and "later" in said, said
    assert "Your step makes the solution take too long" not in said, said


FIND = ("def find(items, target):\n"
        "    i = 0\n"
        "    while i < len(items):\n"
        "        if items[i] == target:\n"
        "            return i\n"
        "        i += 1\n"
        "    return -1\n")
FIND_STEPS = ["i = 0", "while i < len(items):\n    if items[i] == target:\n        return i\n"
                       "    i += 1\nreturn -1"]


@pytest.mark.parametrize("tail, depends", [
    (FIND_STEPS[1], True),                          # reads i, then sets it
    ("total += 5\nreturn total", True),             # += reads before it sets
    ("count += 1\nreturn 'ok'", True),              # ...even with no other read
    ("node = self.head\nwhile node:\n    node = node.next\nreturn node", False),
    ("for n in items:\n    print(n)\nreturn 'done'", False),
    ("return 'Cleared cache!'", False),
])
def test_a_name_the_tail_reads_before_it_sets_it_is_an_earlier_steps(tail, depends):
    """GPT's audit of HW4 __contains__ (15 Oct): the tail `while current is not
    None: ... current = current.next` READS current and also SETS it, and
    subtracting every name it sets read it as depending on nothing - so a
    tail-first `probe = self.tail` was rejected for free, never reaching the
    tiers that can follow another route."""
    from main import grading
    header = "def f(self, items, target):"
    assert grading._tail_reads_earlier_names({"slug": "x", "solution": header + "\n    pass"},
                                             header, tail) is depends


def test_searching_from_the_other_end_is_not_rejected_for_free(monkeypatch, tmp_path):
    from main import grading, identity, ollama_client, sessions
    monkeypatch.setenv("MICROTUTOR_SESSION_DB", str(tmp_path / "s.sqlite3"))
    monkeypatch.setenv("MICROTUTOR_TRACE_FILE", str(tmp_path / "t.jsonl"))
    monkeypatch.setattr(identity, "_RESOLVED_PATH", str(tmp_path / "r.json"))
    monkeypatch.setattr(ollama_client, "_openai_chat",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no model")))
    monkeypatch.setattr(grading, "_request_adaptation", lambda *a, **k: ("", []))
    monkeypatch.setattr(grading, "_request_completion", lambda *a, **k: "")
    problem = {"slug": "find", "title": "Find", "solution": FIND, "entry_hint": "find",
               "description": "The index of target in items, or -1."}
    decomp = {"header": "def find(items, target):", "chunks": [
        types.SimpleNamespace(step_id=f"Part {i + 1}", expected_type="code", prompt="p",
                              reference=r) for i, r in enumerate(FIND_STEPS)]}
    sid = sessions.create_session(dict(problem), decomp, "h-find", student_id="stu")["session_id"]
    grading._VERDICT_MEMO.clear()
    r = grading.grade_submission(sessions.load_session(sid), "i = len(items) - 1", oracle_loader=lambda p: [
        {"input": [[4, 5, 6], 4], "expected": 0}, {"input": [[4, 5, 6], 9], "expected": -1},
        {"input": [[], 1], "expected": -1}])
    assert r.reason_code != "reference_fail_independent", r
