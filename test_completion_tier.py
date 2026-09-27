"""test_completion_tier.py - tier 4 accepts only what EXECUTION confirms.

Replaced the two LLM judges (2026-09-26). The judges gave an
opinion from reading code; measured on every real acceptance, 25 of the 31
students who later finished had to redo the step they passed, and 41 more were
stuck behind one. The model is now asked only to WRITE the rest of the function
on top of the student's step; the full oracle and a blank-out check decide.

Everything here goes through grade_submission with the proposal stubbed, so the
gates are tested in the order they really run.
"""
import types

import pytest

SOLUTION = ("def f(txt):\n    counts = {}\n    for ch in txt:\n"
            "        counts[ch] = counts.get(ch, 0) + 1\n    return counts\n")
PROBLEM = {"slug": "freq-t3", "title": "f", "description": "Count each character.",
           "solution": SOLUTION}
STEPS = [dict(step_id="Part 1", expected_type="code", prompt="Tally the characters.",
              reference="counts = {}\nfor ch in txt:\n    counts[ch] = counts.get(ch, 0) + 1"),
         dict(step_id="Part 2", expected_type="code", prompt="Hand back the tally.",
              reference="return counts")]
ORACLE = [{"input": ["aab"], "expected": {"a": 2, "b": 1}},
          {"input": ["xyzx"], "expected": {"x": 2, "y": 1, "z": 1}},
          {"input": [""], "expected": {}}]
# A different SHAPE - a sorted list, not a dict - so no renaming can match it to
# the teacher's names and it genuinely reaches the model tiers.
MINE = "letters = sorted(txt)"


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("MICROTUTOR_SESSION_DB", str(tmp_path / "s.sqlite3"))
    from main import grading, identity, ollama_client, sessions
    monkeypatch.setattr(identity, "_RESOLVED_PATH", str(tmp_path / "r.json"))
    grading._VERDICT_MEMO.clear()
    asked = []

    def no_network(*_a, **_k):
        raise AssertionError("tier 4 must go through _request_completion")
    monkeypatch.setattr(ollama_client, "_openai_chat", no_network)
    # Tier 3 runs first and, here, proposes nothing usable - these tests are
    # about what tier 4 does once it gets its turn (test_adapter_tier.py covers
    # tier 3 and the hand-over).
    monkeypatch.setattr(grading, "_request_adaptation", lambda *a, **k: ("", []))

    def propose(text):
        def _stub(problem, header, upto, student_code, remaining, ref_tail,
                  evidence, temperature=0.0, step_prompt=""):
            asked.append({"evidence": evidence, "remaining": remaining,
                          "step_prompt": step_prompt})
            if isinstance(text, Exception):
                raise text
            return text
        monkeypatch.setattr(grading, "_request_completion", _stub)

    def grade(code=MINE):
        decomp = {"header": "def f(txt):",
                  "chunks": [types.SimpleNamespace(**c) for c in STEPS]}
        sid = sessions.create_session(dict(PROBLEM), decomp, "h-t3",
                                      student_id="stu")["session_id"]
        return grading.grade_submission(sessions.load_session(sid), code,
                                        oracle_loader=lambda p: list(ORACLE))
    yield type("E", (), {"grade": grade, "propose": propose, "asked": asked})
    grading._VERDICT_MEMO.clear()


def test_a_completion_that_uses_their_work_is_accepted(env):
    env.propose("return {c: letters.count(c) for c in letters}")
    r = env.grade()
    assert (r.verdict, r.tier, r.reason_code) == \
        ("correct", "execution-completed", "completed_pass"), r


def test_the_model_is_shown_their_real_values_and_the_remaining_steps(env):
    env.propose("return {c: letters.count(c) for c in letters}")
    env.grade()
    ev = env.asked[0]["evidence"]
    assert "letters = ['a', 'a', 'b']" in ev, ev      # measured, not guessed
    assert env.asked[0]["remaining"] == ["Hand back the tally."]
    # ...and what THEIR step was asked to do, so a value that contradicts it
    # can be noticed instead of quietly worked around.
    assert env.asked[0]["step_prompt"] == "Tally the characters."


def test_a_completion_that_redoes_the_work_is_refused(env):
    """It passes the tests, but it passes them just as well with their values
    blanked out - so it never used their step."""
    env.propose("out = {}\nfor c in txt:\n    out[c] = out.get(c, 0) + 1\nreturn out")
    r = env.grade()
    assert (r.verdict, r.tier) == ("indeterminate", "unconfirmed"), r
    assert r.consume_attempt is False


def test_a_completion_that_only_glances_at_their_variable_is_refused(env):
    """Reads their variable once, then redoes everything itself. Deleting their
    step would make it crash (the name is gone) and LOOK like their work
    mattered - which is exactly why a completion that reads their names is
    tested by blanking the values, not by deleting the step."""
    env.propose("_ = letters\nout = {}\nfor c in txt:\n    out[c] = out.get(c, 0) + 1\nreturn out")
    assert env.grade().tier == "unconfirmed"


def test_a_completion_that_rebuilds_their_variable_is_refused(env):
    env.propose("letters = list(txt)\nreturn {c: letters.count(c) for c in letters}")
    assert env.grade().tier == "unconfirmed"


def test_a_completion_that_repairs_a_wrong_step_is_refused(env):
    """THE DANGEROUS CASE. Their step is wrong - set() throws the repeats
    away - and a completion quietly rebuilds the variable to cover for it.
    It passes the tests AND still depends on their value (blanked out, it
    breaks), so only the no-rebinding rule stands between this and accepting
    a wrong step."""
    env.propose("letters = sorted(txt) if letters else []\n"
                "return {c: letters.count(c) for c in letters}")
    r = env.grade("letters = sorted(set(txt))")
    assert r.tier == "unconfirmed", r


def test_adding_to_a_variable_they_set_up_empty_is_allowed(env):
    """`+=` carries an accumulator on - but only one their step left EMPTY.
    Adding to a value it already computed is patching it: see the read-only
    tests at the bottom of this file."""
    env.propose("extra += letters\nreturn {c: extra.count(c) for c in extra}")
    assert env.grade("letters = sorted(txt)\nextra = []").tier == "execution-completed"


def test_adding_to_a_value_they_already_computed_is_refused(env):
    env.propose("letters += []\nreturn {c: letters.count(c) for c in letters}")
    assert env.grade().tier == "unconfirmed"


def test_a_completion_that_fails_the_tests_is_not_accepted(env):
    env.propose("return {c: 1 for c in letters}")
    r = env.grade()
    assert r.tier == "unconfirmed" and r.consume_attempt is False, r


def test_an_outage_is_ours_not_theirs(env):
    env.propose(ConnectionError("provider down"))
    r = env.grade()
    assert (r.verdict, r.reason_code) == ("indeterminate", "completion_unavailable"), r
    assert r.consume_attempt is False


def test_identical_code_gets_an_identical_verdict_without_asking_again(env):
    env.propose("return {c: letters.count(c) for c in letters}")
    first = env.grade()
    n = len(env.asked)
    assert env.grade() == first
    assert len(env.asked) == n, "a repeat must be answered from the memo"


# ── A step that only GUARDS (binds no names) ─────────────────────────────
# Measured on a real, correct `calculate` guard: it could never be confirmed,
# because blanking out "the names it produced" blanks nothing, so the check
# always read as "their work does not matter". A guard is tested by DELETING it.
# The teacher's step 1 also computes `n`, which step 2 reads - so a student
# whose step 1 is ONLY the guard leaves the teacher's step 2 nothing to read,
# and the step genuinely reaches the model tiers (the shape of the real `calculate` case).
GUARD_SOLUTION = ("def g(txt):\n    if not isinstance(txt, str):\n        return None\n"
                  "    n = len(txt)\n    return n\n")
GUARD_STEPS = [dict(step_id="Part 1", expected_type="code", prompt="Reject non-text.",
                    reference="if not isinstance(txt, str):\n    return None\nn = len(txt)"),
               dict(step_id="Part 2", expected_type="code", prompt="Measure it.",
                    reference="return n")]
GUARD_ORACLE = [{"input": ["abc"], "expected": 3}, {"input": [5], "expected": None}]


def _grade_guard(env_mod, code):
    from main import grading, sessions
    decomp = {"header": "def g(txt):",
              "chunks": [types.SimpleNamespace(**c) for c in GUARD_STEPS]}
    prob = {"slug": "guard", "title": "g", "description": "Length of text.",
            "solution": GUARD_SOLUTION}
    sid = sessions.create_session(prob, decomp, "h-g", student_id="stu")["session_id"]
    return grading.grade_submission(sessions.load_session(sid), code,
                                    oracle_loader=lambda p: list(GUARD_ORACLE))


THEIR_GUARD = "if not isinstance(txt, str):\n    return None"


def test_a_guard_step_is_confirmed_when_removing_it_breaks_things(env):
    """The right continuation here is the teacher's own remaining code - which
    must not be refused for being the teacher's code."""
    env.propose("return len(txt)")
    r = _grade_guard(env, THEIR_GUARD)
    assert r.tier == "execution-completed", r
    assert r.verdict == "correct", r


def test_a_completion_that_brings_its_own_guard_is_refused(env):
    """Delete their guard and it still passes - the completion did the job."""
    env.propose("if not isinstance(txt, str):\n    return None\nreturn len(txt)")
    r = _grade_guard(env, THEIR_GUARD + "\nprint('checked')")
    assert r.tier == "unconfirmed", r


def test_reusing_their_loop_variable_name_is_not_overwriting(env):
    """Measured on a real, correct `invert` step: the completion passed 10/10
    and was refused because it reused the student's loop variable's NAME."""
    env.propose("out = {}\nfor c in letters:\n    out[c] = letters.count(c)\nreturn out")
    assert env.grade("letters = []\nfor c in txt:\n    letters.append(c)").tier \
        == "execution-completed"


def test_an_indented_completion_is_used(env):
    env.propose("    return {c: letters.count(c) for c in letters}")
    assert env.grade().tier == "execution-completed"


def test_a_completion_that_merely_calls_their_helper_is_checked_properly(env):
    """Their step defines a helper; the completion calls it but also does all
    the work itself. Blanking a FUNCTION used to crash (type(f)() cannot build
    one), which read as 'their work mattered' for free."""
    env.propose("helper(txt)\nout = {}\nfor c in txt:\n    out[c] = out.get(c, 0) + 1\nreturn out")
    assert env.grade("def helper(s):\n    return sorted(s)").tier == "unconfirmed"


def test_a_completion_may_not_do_the_current_steps_work(env):
    """The step only set things up; the completion did the step's real work
    itself. Measured on a real calculateExpressions answer: 30 lines standing
    in for a 2-line remaining step."""
    env.propose("counts = {}\n" + "\n".join(f"n{i} = {i}" for i in range(12))
                + "\nfor c in letters:\n    counts[c] = counts.get(c, 0) + 1\nreturn counts")
    assert env.grade().tier == "unconfirmed"


# ── THEIR FINISHED VALUES ARE READ-ONLY ──────────────────────────────────
# The residual risk of this tier: a step that is slightly WRONG, and a
# completion that quietly PATCHES the value (`n += 1`, `chars.insert(...)`)
# before using it. Every test passes, and blanking their values still breaks
# it - because the patch builds on their value - so the step would be accepted.
# The rule: a completion may only READ what their step produced. The one
# exception is a value their step merely set up EMPTY (`counts = {}`), which
# carrying on is the whole point of. Checked by running their code, no model.
LEN_SOLUTION = "def g(txt):\n    n = len(txt)\n    return n\n"
LEN_STEPS = [dict(step_id="Part 1", expected_type="code", prompt="Measure the text.",
                  reference="n = len(txt)"),
             dict(step_id="Part 2", expected_type="code", prompt="Hand it back.",
                  reference="return n")]
LEN_ORACLE = [{"input": ["abc"], "expected": 3}, {"input": [""], "expected": 0},
              {"input": ["hello"], "expected": 5}]


def _grade_len(code):
    from main import grading, sessions
    decomp = {"header": "def g(txt):",
              "chunks": [types.SimpleNamespace(**c) for c in LEN_STEPS]}
    prob = {"slug": "len-q4", "title": "g", "description": "Length of text.",
            "solution": LEN_SOLUTION}
    sid = sessions.create_session(prob, decomp, "h-len", student_id="stu")["session_id"]
    return grading.grade_submission(sessions.load_session(sid), code,
                                    oracle_loader=lambda p: list(LEN_ORACLE))


def test_an_off_by_one_patched_with_plus_equals_is_refused(env):
    env.propose("n += 1\nreturn n")
    r = _grade_len("n = len(txt) - 1")
    assert r.tier == "unconfirmed", r


def test_a_list_patched_in_place_is_refused(env):
    env.propose("if txt:\n    chars.insert(0, txt[0])\nreturn len(chars)")
    r = _grade_len("chars = list(txt)[1:]")
    assert r.tier == "unconfirmed", r


WORDS_SOLUTION = ("def w(txt):\n    words = txt.split()\n    counts = {}\n"
                  "    for x in words:\n        counts[x] = counts.get(x, 0) + 1\n"
                  "    return counts\n")
WORDS_STEPS = [dict(step_id="Part 1", expected_type="code",
                    prompt="Split the text into words and start an empty tally.",
                    reference="words = txt.split()\ncounts = {}"),
               dict(step_id="Part 2", expected_type="code", prompt="Tally them and hand it back.",
                    reference="for x in words:\n    counts[x] = counts.get(x, 0) + 1\nreturn counts")]
WORDS_ORACLE = [{"input": ["b a b"], "expected": {"a": 1, "b": 2}},
                {"input": [""], "expected": {}}, {"input": ["c"], "expected": {"c": 1}}]


def test_filling_a_tally_their_step_set_up_empty_is_allowed(env):
    """The exception. Their `tally = {}` is a setup, not a result: filling it
    is what the next step is FOR. (Their words are sorted, so the teacher's
    own steps cannot confirm it and it genuinely reaches this tier.)"""
    from main import grading, sessions
    env.propose("for x in parts:\n    tally[x] = tally.get(x, 0) + 1\nreturn tally")
    decomp = {"header": "def w(txt):",
              "chunks": [types.SimpleNamespace(**c) for c in WORDS_STEPS]}
    prob = {"slug": "words-q4", "title": "w", "description": "Count words.",
            "solution": WORDS_SOLUTION}
    sid = sessions.create_session(prob, decomp, "h-words", student_id="stu")["session_id"]
    r = grading.grade_submission(sessions.load_session(sid),
                                 "parts = sorted(txt.split())\ntally = {}",
                                 oracle_loader=lambda p: list(WORDS_ORACLE))
    assert r.tier == "execution-completed", r
