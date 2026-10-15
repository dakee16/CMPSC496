"""test_recompute_rule.py - a correct step that never stores a value the
teacher's next step reads is graded for free, and an empty step never is.

THE REPORT (13-14 Oct, HW4 update). Step 2 of the teacher's update is
`available_space = self.remainingSpace + self.head.value.size` and a check
against it; step 3 reads `available_space`. A student who wrote the same check
rearranged - `content.size - self.head.value.size > self.remainingSpace` - had
written a correct step, but the run with the teacher's step 3 died on
NameError and it went to the paid model tiers (2 calls). Matching the student's
values (the research approach - Clara, SarfGen; our bridges) cannot help: the
value never exists in their code. An earlier idea - compute any missing value
from the teacher's line - accepted `pass` in 14 places.

The rule (grading._recomputed_values): recompute the teacher's line only when
it is that name's one binding, at the top level of THE STEP BEING GRADED, is
plain arithmetic, the step ALSO does visible work, and the student's step is
not empty.

REAL: grade_submission, sessions, execution. FAKED: the model tiers find
nothing (and a model call is recorded).
"""
import types

import pytest

SOLUTION = ("def fit(cap, used, size):\n"
            "    if size < 0:\n"
            "        return 'BAD'\n"
            "    room = cap - used\n"
            "    if size > room:\n"
            "        return 'FULL'\n"
            "    left = room - size\n"
            "    return left\n")
PROBLEM = {"slug": "fit", "title": "Fit", "solution": SOLUTION, "entry_hint": "fit",
           "description": "How much room is left after adding size, 'FULL' if it "
                          "does not fit, 'BAD' for a negative size."}
_ns = {}
exec(SOLUTION, _ns)
TESTS = [{"input": list(x), "expected": _ns["fit"](*x)}
         for x in ((10, 2, 3), (10, 2, 8), (10, 2, 9), (5, 5, 0), (5, 0, -1), (7, 3, 5), (9, 1, 8))]
STEPS = ["if size < 0:\n    return 'BAD'",
         "room = cap - used\nif size > room:\n    return 'FULL'",
         "left = room - size\nreturn left"]
# The step ONLY sets the value: recomputing it would do the student's work.
ONLY_SETS = ["if size < 0:\n    return 'BAD'", "room = cap - used",
             "if size > room:\n    return 'FULL'\nleft = room - size\nreturn left"]


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

    def run(steps, code, before):
        decomp = {"header": "def fit(cap, used, size):", "chunks": [
            types.SimpleNamespace(step_id=f"Part {i + 1}", expected_type="code", prompt="p",
                                  reference=r) for i, r in enumerate(steps)]}
        sid = sessions.create_session(dict(PROBLEM), decomp, "h-fit", student_id="stu")["session_id"]
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


def test_the_check_rearranged_without_the_value_is_right_and_free(grade):
    r = grade(STEPS, "if size + used > cap:\n    return 'FULL'", before=[STEPS[0]])
    assert (r.verdict, r.deterministic) == ("correct", True), r
    assert grade.asked == [], "no model was needed"


def test_an_empty_step_is_never_accepted(grade):
    for code in ("pass", "# nothing yet\npass"):
        r = grade(STEPS, code, before=[STEPS[0]])
        assert r.verdict != "correct", (code, r)


def test_a_wrong_check_is_not_accepted(grade):
    r = grade(STEPS, "if size > cap:\n    return 'FULL'", before=[STEPS[0]])
    assert r.verdict != "correct", r


def test_a_step_that_only_sets_the_value_is_not_done_for_the_student(grade):
    """Its whole job is the value. A step that skips it is not completed by
    recomputing it - and the student's own name for it still works (bridges)."""
    from main import grading
    assert grading._recomputed_values(
        PROBLEM, [{"reference": r} for r in ONLY_SETS], "def fit(cap, used, size):",
        ONLY_SETS[0] + "\nx = 1", "\n".join(ONLY_SETS[2:]), 1, "x = 1") == []
    r = grade(ONLY_SETS, "space = cap - used", before=[ONLY_SETS[0]])
    assert r.verdict == "correct", "renamed, it is still right"


@pytest.mark.parametrize("why, step1, step2, step3", [
    ("a method call can change things", STEPS[0],
     "room = cap.bit_length()\nif size > room:\n    return 'FULL'", "return room - size"),
    ("set twice is not this step's alone", "room = 0\nif size < 0:\n    return 'BAD'",
     "room = cap - used\nif size > room:\n    return 'FULL'", "return room - size"),
    ("set only on some paths", STEPS[0],
     "if used:\n    room = cap - used\nif size > cap:\n    return 'FULL'", "return room - size"),
    ("filled in by the step", STEPS[0],
     "room = [cap]\nroom.append(used)\nif size > cap:\n    return 'FULL'", "return room[0] - size"),
    ("changed in place by the step", STEPS[0],
     "room = [cap]\nroom[0] = cap - used\nif size > cap:\n    return 'FULL'", "return room[0] - size"),
])
def test_what_is_never_recomputed(why, step1, step2, step3):
    from main import grading
    steps = [step1, step2, step3]
    mine = "if size + used > cap:\n    return 'FULL'"
    assert grading._recomputed_values(
        PROBLEM, [{"reference": r} for r in steps], "def fit(cap, used, size):",
        STEPS[0] + "\n" + mine, step3, 1, mine) == [], why
    # ...and the same call with the teacher's plain line IS recomputed, so each
    # case above is refused for its own reason.
    plain = [STEPS[0], "room = cap - used\nif size > room:\n    return 'FULL'", step3]
    assert grading._recomputed_values(
        PROBLEM, [{"reference": r} for r in plain], "def fit(cap, used, size):",
        STEPS[0] + "\n" + mine, step3, 1, mine) == ["room = cap - used"]


def test_an_empty_step_is_refused_before_anything_runs():
    """Visible work the tests happen never to reach would otherwise let `pass`
    through: the empty step is refused outright."""
    from main import grading
    chunks = [{"reference": r} for r in STEPS]
    args = (PROBLEM, chunks, "def fit(cap, used, size):", STEPS[0] + "\npass", STEPS[2], 1)
    assert grading._recomputed_values(*args, "pass") == []
    assert grading._recomputed_values(*args, "if size + used > cap:\n    return 'FULL'") == \
        ["room = cap - used"]
