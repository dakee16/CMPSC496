"""test_adapter_tier.py - tier 3 (the calibrated adapter), repaired, and how it
hands over to tier 4.

THE MEASUREMENT (2026-09-27). The live adapter was run once on each of 137 real
steps that reached it: 86% of its tries were thrown away before running and it
never accepted one. Four causes, each pinned below:
  * the model's name pairs came back the other way round from what the checker
    insisted on (54 of 136) - and the whole try was discarded;
  * the model echoed the example pair from its own instructions (43 of 137);
  * the "pasted the teacher's code" rule matched a tail found ANYWHERE in the
    solution, so a bare `return True` counted (34);
  * one unusable pair discarded the whole try.
It also paid for four tries at temperature 0; the second was a word-for-word
copy of the first 17 times in 25. It now gets one.

WHY TIER 4 STILL RUNS AFTER TIER 3 FINDS FAILING CASES. Those cases come from a
model's rewrite: a strong hint, not proof. Tier 4 can overrule only with proof,
and is never shown the cases (they would invite a patch that hides the bug).

Everything goes through grade_submission with only the two model proposals
stubbed, so the gates run in their real order.
"""
import types

import pytest

# The teacher tallies into a dict; this student builds sorted (char, count)
# PAIRS instead. Different shape, so neither the teacher's own tail (tier 1)
# nor matching names by value (tier 2) can confirm it - it genuinely reaches
# tier 3 - and yet `dict(pairs)` works on the teacher's dict too, so a rewrite
# CAN be proven on the teacher's own steps.
SOLUTION = ("def f(txt):\n    counts = {}\n    for ch in txt:\n"
            "        counts[ch] = counts.get(ch, 0) + 1\n    return counts\n")
PROBLEM = {"slug": "freq-t3a", "title": "f", "description": "Count each character.",
           "solution": SOLUTION}
STEPS = [dict(step_id="Part 1", expected_type="code", prompt="Tally the characters.",
              reference="counts = {}\nfor ch in txt:\n    counts[ch] = counts.get(ch, 0) + 1"),
         dict(step_id="Part 2", expected_type="code", prompt="Hand back the tally.",
              reference="return counts")]
ORACLE = [{"input": ["aab"], "expected": {"a": 2, "b": 1}},
          {"input": ["xyzx"], "expected": {"x": 2, "y": 1, "z": 1}},
          {"input": [""], "expected": {}}]
PAIRS = "pairs = sorted((c, txt.count(c)) for c in set(txt))"
# Wrong: every count is 1. Still pairs, still the same shape.
WRONG_PAIRS = "pairs = sorted((c, 1) for c in set(txt))"
GOOD_TAIL = "return dict(pairs)"


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("MICROTUTOR_SESSION_DB", str(tmp_path / "s.sqlite3"))
    from main import grading, identity, ollama_client, sessions, trace
    trace._SINK.clear()
    real_request = grading._request_adaptation
    monkeypatch.setattr(identity, "_RESOLVED_PATH", str(tmp_path / "r.json"))
    grading._VERDICT_MEMO.clear()

    def no_network(*_a, **_k):
        raise AssertionError("a model call escaped the stubs")
    monkeypatch.setattr(ollama_client, "_openai_chat", no_network)
    log = types.SimpleNamespace(adapter=0, completion=[])

    def adapter(tail, aliases=()):
        def _stub(problem, header, upto, reference_tail, student_outputs):
            log.adapter += 1
            if isinstance(tail, Exception):
                raise tail
            return tail, list(aliases)
        monkeypatch.setattr(grading, "_request_adaptation", _stub)

    def completion(text=""):
        def _stub(problem, header, upto, student_code, remaining, ref_tail,
                  evidence, temperature=0.0, step_prompt=""):
            log.completion.append({"evidence": evidence, "ref": ref_tail})
            return text
        monkeypatch.setattr(grading, "_request_completion", _stub)

    completion("")                      # tier 4 finds nothing unless told

    def grade(code=PAIRS, problem=PROBLEM):
        decomp = {"header": "def f(txt):",
                  "chunks": [types.SimpleNamespace(**c) for c in STEPS]}
        sid = sessions.create_session(dict(problem), decomp, "h-t3a",
                                      student_id="stu")["session_id"]
        return grading.grade_submission(sessions.load_session(sid), code,
                                        oracle_loader=lambda p: list(ORACLE))

    def routes():
        return [e for e in trace.events() if e.get("kind") == "route"]

    def adapter_outcomes():             # tier 3 runs first, so its try is first
        return [e["outcome"] for e in trace.events() if e.get("kind") == "adapter"][:1]
    yield types.SimpleNamespace(grade=grade, adapter=adapter, completion=completion,
                                log=log, routes=routes, adapter_outcomes=adapter_outcomes,
                                real_request=real_request)
    grading._VERDICT_MEMO.clear()


# ── the four measured bugs ───────────────────────────────────────────────

def test_a_proven_rewrite_is_accepted(env):
    env.adapter(GOOD_TAIL, [{"teacher": "counts", "student": "pairs"}])
    r = env.grade()
    assert (r.verdict, r.tier, r.reason_code) == ("correct", "execution-adapted", "adapted_pass"), r
    assert env.log.completion == [], "an acceptance ends it - tier 4 is not asked"


def test_a_pair_written_the_other_way_round_still_counts(env):
    """54 of 136 real pairs came back reversed, and each threw the try away."""
    env.adapter(GOOD_TAIL, [{"teacher": "pairs", "student": "counts"}])
    assert env.grade().tier == "execution-adapted"


def test_an_echoed_example_or_an_expression_is_ignored_not_fatal(env):
    env.adapter(GOOD_TAIL, [{"teacher": "n", "student": "m"},
                            {"teacher": "counts", "student": "pairs"},
                            {"teacher": "self.top.value", "student": "pairs"}])
    assert env.grade().tier == "execution-adapted"


def test_a_short_tail_found_elsewhere_in_the_solution_is_not_a_paste(env):
    """Measured: `return True` was refused as "the teacher's code" because the
    solution returned True somewhere else."""
    prob = {**PROBLEM, "solution": SOLUTION.replace(
        "    return counts\n", "    return counts  # not return dict(pairs)\n")}
    env.adapter(GOOD_TAIL, [{"teacher": "counts", "student": "pairs"}])
    assert env.grade(problem=prob).tier == "execution-adapted"


def test_the_teachers_tail_handed_back_unchanged_is_refused(env):
    """Not an adaptation - it is tier 1 again, and tier 1 already failed."""
    env.adapter("return counts", [{"teacher": "counts", "student": "pairs"}])
    r = env.grade()
    assert (r.verdict, r.tier) == ("indeterminate", "unconfirmed"), r


def test_a_tail_sent_as_a_list_of_lines_is_read_as_code(env, monkeypatch):
    """70 of 88 real answers put the tail in a JSON LIST of lines. Read as
    text it became a list literal that does nothing, and failed every time.
    Goes through the real request function; only the model's reply is fake."""
    from main import grading
    reply = {"adapted_tail": ["out = dict(pairs)", "return out"],
             "aliases": [{"teacher": "counts", "student": "pairs"}]}
    monkeypatch.setattr(grading, "_request_adaptation", env.real_request)
    monkeypatch.setattr(grading, "chat", lambda *a, **k: __import__("json").dumps(reply))
    assert env.grade().tier == "execution-adapted"


def test_one_try_only(env):
    env.adapter("   ")                              # unusable
    env.grade()
    assert env.log.adapter == 1


def test_tier3_may_not_add_onto_their_variable(env):
    """`+=` is tier 4's, for carrying an accumulator on; a rewrite of the
    teacher's remaining code never needed it in 137 real tries."""
    env.adapter("pairs += []\nreturn dict(pairs)", [{"teacher": "counts", "student": "pairs"}])
    assert env.grade().tier != "execution-adapted"
    assert env.adapter_outcomes() == ["unsafe"], "refused by the rule, before running"


def test_a_rewrite_may_not_change_what_their_step_computed(env):
    """Read-only, as in tier 4 (see test_completion_tier.py): a rewrite that
    edits their finished value could be patching a wrong step."""
    env.adapter("pairs.sort()\nreturn dict(pairs)", [{"teacher": "counts", "student": "pairs"}])
    env.grade()
    assert env.adapter_outcomes() == ["changes_their_values"], env.adapter_outcomes()


def test_a_rewrite_that_ignores_their_work_is_refused(env):
    env.adapter("out = {}\nfor c in txt:\n    out[c] = out.get(c, 0) + 1\nreturn out")
    assert env.grade().tier != "execution-adapted"


# ── handing over to tier 4 ───────────────────────────────────────────────

def test_failing_cases_are_shown_when_tier4_cannot_prove_the_step(env):
    env.adapter(GOOD_TAIL, [{"teacher": "counts", "student": "pairs"}])
    r = env.grade(WRONG_PAIRS)
    assert (r.verdict, r.tier, r.reason_code) == \
        ("indeterminate", "execution-adapted", "adapted_evidence_only"), r
    assert r.consume_attempt is False and r.failing_cases, r
    assert len(env.log.completion) >= 1, "tier 4 must still get its look"
    assert [e["final_route"] for e in env.routes()] == ["execution-adapted"]


BOX = ("class Box:\n    def __init__(self):\n        self.__v = None\n\n"
       "    def put(self, v):\n        self.__v = v\n\n    def get(self):\n")
BOX_PROBLEM = {"slug": "box-t3", "title": "Box.get", "description": "Return what was put in.",
               "solution": "def get(self):\n    v = self.__v\n    return v\n",
               "context_prefix": BOX, "context_suffix": "", "context_indent": 8,
               "entry_hint": "get", "group_title": "Box"}
BOX_STEPS = [dict(step_id="Part 1", expected_type="code", prompt="Read the stored value.",
                  reference="v = self.__v"),
             dict(step_id="Part 2", expected_type="code", prompt="Hand it back.",
                  reference="return v")]
BOX_ORACLE = [{"input": [[["get"]]], "expected": [None]},
              {"input": [[["put", 5], ["get"]]], "expected": [None, 5]}]


def test_the_rewrites_own_crash_is_never_shown_as_their_failing_case(env, monkeypatch):
    """Measured on a real, correct `calculate` guard: the rewrite read a name
    only the TEACHER's step creates, crashed with NameError on the student's
    work, and a method's crash is recorded as a value - so it read as a wrong
    answer and was about to be shown to the student as theirs."""
    from main import grading, sessions
    env.adapter("return v if True else pair", [])      # passes on the teacher's steps
    decomp = {"header": "def get(self):",
              "chunks": [types.SimpleNamespace(**c) for c in BOX_STEPS]}
    sid = sessions.create_session(dict(BOX_PROBLEM), decomp, "h-box", student_id="stu")["session_id"]
    r = grading.grade_submission(sessions.load_session(sid), "pair = (self.__v,)",
                                 oracle_loader=lambda p: list(BOX_ORACLE))
    assert r.tier != "execution-adapted" and not r.failing_cases, r
    assert env.adapter_outcomes() == ["no_acquittal_wrong_output"], env.adapter_outcomes()


def test_tier4_is_never_shown_tier3s_failing_cases(env):
    env.adapter(GOOD_TAIL, [{"teacher": "counts", "student": "pairs"}])
    env.grade(WRONG_PAIRS)
    for asked in env.log.completion:
        assert "expected" not in asked["evidence"], asked["evidence"]


def test_tier4_can_overrule_with_proof_and_the_overrule_is_logged(env):
    """A program built on their step that passes every test and breaks without
    it beats a failing rewrite. Here the student's step is right and the
    REWRITE is what is wrong - faithful on the teacher's dict, which is all
    calibration can check, and wrong on the student's pairs."""
    env.adapter("return dict(pairs) if isinstance(pairs, dict) else {c: 1 for c, n in pairs}",
                [{"teacher": "counts", "student": "pairs"}])
    env.completion("return dict(pairs)")
    r = env.grade()
    assert (r.verdict, r.tier) == ("correct", "execution-completed"), r
    routes = env.routes()
    assert [(e["final_route"], e.get("overruled")) for e in routes] == \
        [("execution-completed", "execution-adapted")], routes


def test_adapter_trouble_still_leaves_tier4_its_look(env):
    env.adapter(ConnectionError("provider hiccup"))
    env.completion("return dict(pairs)")
    assert env.grade().tier == "execution-completed"


# ── Q5: the names Python already provides ────────────────────────────────

def test_type_and_hasattr_are_not_undefined(env):
    for code in ("pairs = sorted((c, txt.count(c)) for c in set(txt)) if type(txt) == str else []",
                 "pairs = sorted((c, txt.count(c)) for c in set(txt)) if hasattr(txt, 'count') else []"):
        r = env.grade(code)
        assert r.reason_code != "undefined_name", (code, r.student_reason)


def test_getattr_is_refused_honestly(env):
    r = env.grade("pairs = sorted((c, getattr(txt, 'count')(c)) for c in set(txt))")
    assert r.verdict == "incorrect", r
    assert "getattr" in r.student_reason and "isn't allowed" in r.student_reason, r.student_reason
    assert "isn't defined" not in r.student_reason
