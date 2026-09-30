"""test_bridge_renaming.py - a correct step that names its variables differently
is confirmed without a model, even when the step ends INSIDE a loop.

THE GAP (30 Sep, from a student report on get-postfix). Tier 2 matches names by
the VALUES they hold at the end of the step. On roadmaps cut inside a loop there
is no "end of the step" to look at: the snapshot ran after the whole loop, and
a step that leaves the loop's advancing to the next step never finishes - the
teacher's own step 1 timed out at 5s. Tier 2 also read no names at all from a
next step that starts indented. So a correct step 1 that said `operator_stack`
where the teacher said `postfixStack` could only be confirmed by a model.

Now the names are paired by how each is first SET UP (`x = {}` and `y = {}`,
`s = Stack()` and `t = Stack()`), the teacher's remaining steps are run with the
student's names swapped in, and only a pass on EVERY test confirms it.
Everything goes through grade_submission; the model tiers propose nothing.
"""
import types

import pytest

from test_restart_reroute import ORACLE, PROBLEM
from test_reword_pools import LOOPED

# Like get-postfix: step 1 only moves on past what it skips; step 2 does the
# rest - so step 1 alone never finishes on a letter.
WHILE = {"header": "def frequency(txt):", "chunks": [
    {"step_id": "Part 1", "expected_type": "code",
     "prompt": "Write code that starts going through each character, skipping "
               "anything that is not a letter, and keep the tally for the next step.",
     "reference": "counts = {}\ni = 0\nwhile i < len(txt):\n    ch = txt[i]\n"
                  "    if not ch.isalpha():\n        i += 1\n        continue"},
    {"step_id": "Part 2", "expected_type": "code",
     "prompt": "For each letter, add one to its tally, and keep the result for the next step.",
     "reference": "    counts[ch] = counts.get(ch, 0) + 1\n    i += 1"},
    {"step_id": "Part 3", "expected_type": "code",
     "prompt": "Write code that returns how many times each letter appears.",
     "reference": "return counts"}]}
RENAMED = WHILE["chunks"][0]["reference"].replace("counts", "tally")


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

    def run(code, roadmap=WHILE):
        decomp = {"header": roadmap["header"],
                  "chunks": [types.SimpleNamespace(**c) for c in roadmap["chunks"]]}
        sid = sessions.create_session(dict(PROBLEM), decomp, "h-rename",
                                      student_id="stu")["session_id"]
        grading._VERDICT_MEMO.clear()
        return grading.grade_submission(sessions.load_session(sid), code,
                                        oracle_loader=lambda p: list(ORACLE))
    return run


def test_a_renamed_step_that_ends_inside_a_while_loop_is_confirmed(grade):
    r = grade(RENAMED)
    assert (r.verdict, r.tier) == ("correct", "execution-bridged"), r


def test_a_renamed_step_that_ends_inside_a_for_loop_is_confirmed(grade):
    r = grade(LOOPED["chunks"][0]["reference"].replace("counts", "tally"), roadmap=LOOPED)
    assert (r.verdict, r.tier) == ("correct", "execution-bridged"), r


def test_a_different_set_up_is_not_paired(grade):
    r = grade(RENAMED.replace("tally = {}", "tally = []"))
    assert r.verdict != "correct", r


def test_the_same_set_up_with_wrong_logic_is_not_confirmed(grade):
    """Pairing only proposes: the tests decide. Skipping the letters instead of
    the rest sets `tally` up exactly like `counts`, and still fails."""
    r = grade(RENAMED.replace("if not ch.isalpha()", "if ch.isalpha()"))
    assert r.verdict != "correct", r


def test_the_teachers_own_names_still_take_the_ordinary_path(grade):
    r = grade(WHILE["chunks"][0]["reference"])
    assert (r.verdict, r.tier) == ("correct", "execution-reference"), r


def test_renaming_touches_variables_only_and_keeps_every_column():
    from main.bridge import _renamed
    tail = ("    counts[ch] = counts.get(ch, 0) + 1\n"
            "    self.counts = f(counts=counts)\n"
            "return counts")
    assert _renamed(tail, {"counts": "tally"}) == (
        "    tally[ch] = tally.get(ch, 0) + 1\n"
        "    self.counts = f(counts=tally)\n"
        "return tally")
