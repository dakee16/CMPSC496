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
    return "end inside" in r.student_reason and "loop" in r.student_reason


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


# ── THE GUARD (30 Sep): no tier may ACCEPT a step the next one cannot continue.
# Sean's calculateExpressions step 1 - set-up only - was accepted three times by
# tier 4, which wrote the missing loop itself; then no step 2 could join it.
# Reproduced on his real roadmap, and here in miniature: the completion below
# is what tier 4 writes, and without the guard it is accepted.
SHOUT_SOLUTION = ("def shout(txt):\n    words = txt.split(';')\n    out = []\n"
                  "    for w in words:\n        w = w.strip()\n        if not w:\n"
                  "            continue\n        if w.isdigit():\n"
                  "            out.append(int(w) * 2)\n        else:\n"
                  "            out.append(w.upper())\n    return out\n")
SHOUT = {"slug": "shout", "title": "Shout", "solution": SHOUT_SOLUTION,
         "description": "Split on ';', drop blanks, double numbers, upper-case words."}
_ns = {}
exec(SHOUT_SOLUTION, _ns)
SHOUT_ORACLE = [{"input": i, "expected": _ns["shout"](*i)}
                for i in (["a;b"], ["1; x ;;2"], [""], ["hi;3;there"], [" ; "], ["7"])]
SHOUT_STEPS = {"header": "def shout(txt):", "chunks": [
    {"step_id": "Part 1", "expected_type": "code", "prompt": "p1",
     "reference": "words = txt.split(';')\nout = []\nfor w in words:\n"
                  "    w = w.strip()\n    if not w:\n        continue"},
    {"step_id": "Part 2", "expected_type": "code", "prompt": "p2",
     "reference": "    if w.isdigit():\n        out.append(int(w) * 2)\n"
                  "    else:\n        out.append(w.upper())"},
    {"step_id": "Part 3", "expected_type": "code", "prompt": "p3",
     "reference": "return out"}]}
TIER4_FINISH = ("for w in words:\n    w = w.strip()\n    if not w:\n        continue\n"
                "    if w.isdigit():\n        out.append(int(w) * 2)\n    else:\n"
                "        out.append(w.upper())\nreturn out")


@pytest.fixture
def guarded(monkeypatch, tmp_path):
    monkeypatch.setenv("MICROTUTOR_SESSION_DB", str(tmp_path / "s.sqlite3"))
    from main import grading, identity, ollama_client, sessions
    monkeypatch.setattr(identity, "_RESOLVED_PATH", str(tmp_path / "r.json"))
    monkeypatch.setattr(ollama_client, "_openai_chat",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no model")))
    asked = []
    monkeypatch.setattr(grading, "_request_adaptation",
                        lambda *a, **k: asked.append("tier 3") or ("", []))
    monkeypatch.setattr(grading, "_request_completion",
                        lambda *a, **k: asked.append("tier 4") or TIER4_FINISH)

    def run(code, roadmap=SHOUT_STEPS, problem=SHOUT, oracle=SHOUT_ORACLE):
        decomp = {"header": roadmap["header"],
                  "chunks": [types.SimpleNamespace(**c) for c in roadmap["chunks"]]}
        sid = sessions.create_session(dict(problem), decomp, "h-guard",
                                      student_id="stu")["session_id"]
        grading._VERDICT_MEMO.clear()
        asked.clear()
        return grading.grade_submission(sessions.load_session(sid), code,
                                        oracle_loader=lambda p: list(oracle))
    run.asked = asked
    return run


def test_a_set_up_only_step_is_never_accepted_by_a_model_tier(guarded):
    r = guarded("words = txt.split(';')\nout = []")
    assert (r.verdict, r.tier, r.consume_attempt) == ("indeterminate", "unconfirmed", False), r
    assert guarded.asked == [], "the model tiers must not even be asked"
    assert "`for` loop" in r.student_reason and "using your own names" in r.student_reason


def test_the_message_names_the_kind_of_loop(guarded):
    from test_bridge_renaming import WHILE
    r = guarded("counts = {}", roadmap=WHILE, problem=PROBLEM, oracle=ORACLE)
    assert "`while` loop" in r.student_reason, r.student_reason


def test_returning_early_keeps_its_own_message_in_this_shape(guarded):
    r = guarded("return []")
    assert "returns from the function" in r.student_reason, r.student_reason
    assert guarded.asked == []


def test_a_step_that_starts_the_loop_still_reaches_every_tier(guarded):
    r = guarded("words = txt.split(';')\nout = []\nfor w in words:\n"
                "    w = w.strip()\n    if not w:\n        continue")
    assert r.verdict == "correct", r


def test_the_teachers_message_never_leaks_their_line(guarded):
    r = guarded("words = txt.split(';')\nout = []")
    assert "for w in words" not in r.student_reason
