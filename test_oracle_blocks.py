"""What a hidden-test BLOCK may be, and when a cached verdict must be redone.

Every case here is one that shipped: each assertion below failed against the
code as it stood, on HW3's own Calculator and AdvancedCalculator. They are
written against the real assignment file rather than a fixture because the bugs
were all in reading a real class - name mangling, a decorator, a @property -
and a hand-made fixture is exactly where they hid.
"""
import ast
import json
import pathlib

import pytest

from main import context, mutation
from main.assignments import parse_assignment_file
from main.identity import get_resolved_entry
from main.oracle_store import is_stale, oracle_features
from tests.sandbox import _useless_block, run_solution

HW3 = pathlib.Path(__file__).parent / "assignment_hw3.py"


@pytest.fixture(scope="module")
def problems():
    parsed = parse_assignment_file(HW3.read_text(), "assignment_hw3.py")
    return {p["slug"]: p for p in parsed["problems"]}


# ── what a block is allowed to look at ───────────────────────────────────

def test_a_name_python_hides_is_not_offered_as_readable(problems):
    """Calculator's only internal is `self.__expr`, mangled to _Calculator__expr.

    A block runs at module level, where that mangling does not happen, so every
    block offered `__expr` observed AttributeError and nothing else. Twelve of
    them were generated, cached, and counted as a suite."""
    assert context.fixed_internals(problems["calculator-calculate"]) == set()
    # ...and with nothing to read, blocks do not apply to it at all.
    assert oracle_features(problems["calculator-calculate"]) == {"calls"}


def test_a_class_whose_given_code_fixes_public_state_still_gets_blocks(problems):
    """The opposite case, which must keep working: AdvancedCalculator's
    `states` and `expressions` are set by given code the student never writes,
    so they are contract, and the reset-on-failure rule can only be seen there."""
    p = problems["advanced-calculator-calculate-expressions"]
    assert context.fixed_internals(p) == {"states", "expressions"}
    assert "blocks/4" in oracle_features(p)


def test_ordinary_python_is_not_reaching_inside_the_object(problems):
    """`.keys()` on a local dict is not an attempt to grade implementation
    detail. Rejecting it left two AdvancedCalculator methods with zero blocks
    while their permitted set was non-empty."""
    p = problems["advanced-calculator-calculate-expressions"]
    ok = ("C = AdvancedCalculator()\n"
          "C.setExpression('a = 5;return a')\n"
          "C.calculateExpressions()\n"
          "sorted(C.states.keys())")
    assert context.block_is_permitted(p, ok)


def test_a_bare_read_of_an_unfixed_attribute_is_still_refused(problems):
    """The rule the check exists for has to survive the relaxation above."""
    p = problems["advanced-calculator-calculate-expressions"]
    assert not context.block_is_permitted(
        p, "C = AdvancedCalculator()\nC._my_own_helper")


# ── a block that cannot observe anything is not a test ───────────────────

def test_an_observation_that_is_an_object_is_refused():
    """Its memory address differs in every subprocess, so the reference
    disagrees with ITSELF: every mutant scores as killed and the suite is
    certified STRONG while testing nothing."""
    assert _useless_block(["x = Calculator()\nx"],
                          [None, "<Calculator object at 0x103c48050>"])


def test_a_block_that_only_ever_errors_is_refused():
    """Same list for every possible implementation - it can neither kill a
    mutant nor fail a student."""
    assert _useless_block(["x = Calculator()\nx.calculate()\ny"],
                          [None, "!TypeError", "!NameError"])
    assert _useless_block(["x = Calculator()\nx.setExpr('1')"], [None, None])


def test_a_block_that_observes_something_is_kept():
    assert not _useless_block(["x = Stack()\nx.push(1)\nn = x.top\nx.pop()\n"
                               "n.next is None"],
                              [None, None, None, 1, True])


def test_an_ordinary_call_list_is_never_judged_as_a_block():
    """Including a plain function whose first argument is a string."""
    assert not _useless_block([[["push", 2], ["pop"]]], [None, 2])
    # A plain function whose argument is a string looks exactly like a block,
    # and its expected value is a bool rather than a list of observations.
    assert not _useless_block(["racecar"], True, method=False)


# ── the free counterexample search must not skip string functions ────────

def test_boundary_probes_survive_a_plain_string_argument():
    """`is_palindrome('racecar')` looks exactly like a block. Judging on shape
    alone threw away the probes AND the whole sweep for every such problem."""
    tests = [{"input": ["racecar"], "expected": True}]
    assert mutation._probe_inputs(tests)          # no problem = plain function
    assert mutation._sweep_inputs(tests)


def test_blocks_are_never_varied_as_arguments(problems):
    """A block is a program: there is nothing to vary, and mutating one would
    produce broken Python rather than a new case."""
    p = problems["stack-pop"]
    tests = [{"input": ["x = Stack()\nx.push(1)"], "expected": [None, None]}]
    assert mutation._drop_blocks(tests, True) == []
    for cand in mutation._probe_inputs(tests, p) + mutation._sweep_inputs(tests, problem=p):
        assert not isinstance(cand[0], str), cand


# ── the free search must speak the problem's own input language ──────────

def test_a_methods_free_search_produces_replayable_call_sequences(problems):
    """Varying a method's ONE argument produced `[[49, 33, 42]]` - a sequence
    whose first call is the integer 49. The driver cannot read that as a call,
    both programs die the same way, and no variant could ever be a
    counterexample: every class problem reached the paid search having had no
    free search at all."""
    p = problems["advanced-calculator-calculate-expressions"]
    tests = [{"input": [[["setExpression", "a = 5;return a"],
                         ["calculateExpressions"]]], "expected": [None, {}]}]
    cands = mutation._probe_inputs(tests, p) + mutation._sweep_inputs(tests, problem=p)
    assert cands
    for c in cands:
        assert len(c) == 1, c
        if isinstance(c[0], str):
            # A block: real Python, and only touching what it is allowed to.
            ast.parse(c[0])
            assert context.block_is_permitted(p, c[0]), c[0]
            continue
        for call in c[0]:
            assert isinstance(call, list) and call and isinstance(call[0], str), call


def test_the_free_search_also_observes_the_state_a_run_left_behind(problems):
    """A call sequence compares return values, and every check still surviving
    on calculateExpressions returns None either way - the whole difference is
    in `states`. The sweep therefore emits each run twice: once as calls, once
    as a block that reads the fixed internals afterwards."""
    p = problems["advanced-calculator-calculate-expressions"]
    tests = [{"input": [[["setExpression", "a = 5;return a"],
                         ["calculateExpressions"]]], "expected": [None, {}]}]
    blocks = [c[0] for c in mutation._sweep_inputs(tests, problem=p)
              if isinstance(c[0], str)]
    assert blocks
    assert all(b.rstrip().endswith("o.states") for b in blocks), blocks[0]


def test_no_blocks_are_derived_for_a_class_with_no_fixed_state(problems):
    """Calculator keeps everything in a name Python hides, so there is nothing
    a block may read and the sweep must not invent one."""
    p = problems["calculator-calculate"]
    assert mutation._as_block(p, [["setExpr", "1 + 2"], ["calculate"]]) is None
    tests = [{"input": [[["setExpr", "1 + 2"], ["calculate"]]],
              "expected": [None, 3.0]}]
    assert not [c for c in mutation._sweep_inputs(tests, problem=p)
                if isinstance(c[0], str)]


def test_a_property_is_read_not_called_in_a_derived_block(problems):
    """`x.calculate()` on a property raises before the method runs a line."""
    p = problems["advanced-calculator-calculate-expressions"]
    b = mutation._as_block(p, [["setExpression", "a = 5"], ["calculateExpressions"]])
    assert "o.setExpression('a = 5')" in b
    assert "o.calculateExpressions()" in b


def test_the_free_search_reaches_the_method_with_nothing_set_up(problems):
    """The guard branch on a fresh object is where the survivors live, and no
    recorded sequence ever starts there."""
    p = problems["advanced-calculator-calculate-expressions"]
    tests = [{"input": [[["setExpression", "a = 5;return a"],
                         ["calculateExpressions"]]], "expected": [None, {}]}]
    assert [["new"], ["calculateExpressions"]] in [
        c[0] for c in mutation._probe_inputs(tests, p)]


def test_the_free_search_feeds_malformed_arguments(problems):
    """Truncated, emptied and doubled - structural corruption, which is the
    only kind a generic sweep can know about."""
    p = problems["advanced-calculator-calculate-expressions"]
    tests = [{"input": [[["setExpression", "a = 5;return a"],
                         ["calculateExpressions"]]], "expected": [None, {}]}]
    args = {call[1] for c in mutation._sweep_inputs(tests, problem=p)
            for call in c[0] if len(call) > 1 and isinstance(call[1], str)}
    assert "" in args, args
    assert any(a and a != "a = 5;return a" and "a = 5;return a".startswith(a)
               for a in args), args


def test_the_free_search_is_seeded_and_repeatable(problems):
    """Two runs must produce byte-identical candidates, or a verdict stops
    being repeatable."""
    p = problems["advanced-calculator-calculate-expressions"]
    tests = [{"input": [[["setExpression", "a = 5;return a"],
                         ["calculateExpressions"]]], "expected": [None, {}]}]
    assert (mutation._sweep_inputs(tests, problem=p)
            == mutation._sweep_inputs(tests, problem=p))
    assert mutation._probe_inputs(tests, p) == mutation._probe_inputs(tests, p)


# ── a decorated method survives a retry ──────────────────────────────────

def test_retrying_a_property_does_not_double_its_decorator(problems):
    """`property(property(f))` is not callable, so the whole class raised
    TypeError on read - and the retry path saved it."""
    p = problems["calculator-calculate"]
    module = context.module_with_method(p, p["solution"])
    assert module.count("@property\n    def calculate") == 1
    ns = {}
    exec(compile(module, "<t>", "exec"), ns)
    c = ns["Calculator"]()
    c.setExpr("1 + 2")
    assert c.calculate == 3.0


def test_a_retried_property_reparses_to_the_same_problem(problems):
    """The retry re-parses the spliced module, so a decorator that accumulated
    here would be stored back into Supabase."""
    p = problems["calculator-calculate"]
    module = context.module_with_method(p, p["solution"])
    again = parse_assignment_file(module, "x.py")["problems"]
    got = next(q for q in again if q["slug"] == "calculator-calculate")
    assert got["solution"].splitlines()[:2] == ["@property", "def calculate(self):"]


# ── the probe must watch the line it was told about ──────────────────────

def test_the_probe_reads_the_tree_the_mutant_index_came_from(problems):
    """Mutant indices are positions in the BARE METHOD's ast.walk. Handing the
    probe the assembled module made every index address unrelated code, so all
    35 sites came back 'shape not instrumentable' and every class problem
    reported verdict=unknown, reached=0."""
    p = problems["calculator-calculate"]
    entry = get_resolved_entry(p)["entry_name"]
    inputs = [[[["setExpr", "2 - 3 * 4"], ["calculate"]]],
              [[["setExpr", "7 ^ 2"], ["calculate"]]]]
    site = next(m for m in mutation.generate_mutants(p["solution"])
                if m.get("kind") == "cmp")
    from main.probe import probe_site
    pr = probe_site(p["solution"], entry, inputs, site["index"], site["kind"],
                    site.get("slot", 0),
                    wrap=lambda src: mutation._runnable(p, src))
    assert pr["ok"], pr.get("error")
    assert pr["reached"] > 0, "the edited line never ran - wrong tree again"
    assert pr["verdict"] != "unknown"


# ── a mutant nobody can ever kill must never be generated ────────────────

def test_a_trailing_return_none_is_never_mutated(problems):
    """`calculateExpressions` ends in `return None`. Deleting it cannot change
    anything - a function that falls off the end returns None - so it was one
    survivor that provably could not be killed by any test, sitting in the
    denominator and holding the verdict down for good."""
    p = problems["advanced-calculator-calculate-expressions"]
    last = p["solution"].rstrip().splitlines()[-1].strip()
    assert last.startswith("return None"), "fixture moved; pick another problem"
    labels = [m["label"] for m in mutation.generate_mutants(p["solution"])]
    assert not [l for l in labels
                if l.startswith(f"line {len(p['solution'].splitlines())}")
                and "return None" in l], labels


def test_a_return_none_that_leaves_early_is_still_mutated(problems):
    """The rule is about the LAST statement of a function body. A `return None`
    that exits a branch is real behaviour and must keep its mutant."""
    p = problems["advanced-calculator-calculate-expressions"]
    labels = [m["label"] for m in mutation.generate_mutants(p["solution"])]
    assert len([l for l in labels if "remove `return None`" in l]) >= 5, labels


# ── a verdict is stale when the generators CHANGED, either way ───────────

def test_a_verdict_reached_with_a_generator_that_no_longer_applies_is_stale(problems):
    """The one-way check only noticed generators being ADDED. Calculator's
    twelve dead blocks were removed by a FIX, and a one-way check would have
    kept them - and their verdict - forever."""
    p = problems["calculator-calculate"]
    stale = {"strong": False, "status": "needs_review", "final_tests": [],
             "features": ["blocks", "calls"]}
    assert is_stale(stale, p)
    assert not is_stale({**stale, "features": ["calls"]}, p)


def test_an_unvalidated_entry_is_not_stale(problems):
    assert not is_stale(None, problems["calculator-calculate"])
    assert not is_stale([], problems["calculator-calculate"])


# ── an instructor's acceptance is not part of the verdict ────────────────

def test_certified_honours_an_acceptance_only_on_needs_review():
    from main.oracle_store import certified
    base = {"strong": False, "final_tests": [], "accepted_by": "dr-who"}
    assert certified({**base, "status": "needs_review"})
    assert not certified({**base, "status": "weak"})


def test_revalidating_keeps_the_acceptance(tmp_path, monkeypatch, problems):
    """/teacher/problems/accept writes the acceptance and then re-prepares,
    which lands back in get_oracle_tests. A plain overwrite there deleted the
    thing the endpoint had just recorded, while its response still reported
    success from an in-memory copy."""
    import tests.sandbox as sandbox
    from main.identity import content_hash
    from main.oracle_store import verdict_entry

    p = problems["stack-push"]
    key = content_hash(p)
    store = {key: {"strong": False, "status": "needs_review", "final_tests": [],
                   "accepted_by": "dr-who", "accepted_at": "2026-09-08T00:00:00+00:00",
                   "features": ["calls"]}}
    monkeypatch.setattr(sandbox, "_load_cache", lambda: dict(store))
    monkeypatch.setattr(sandbox, "_save_cache", lambda c: store.update(c))
    monkeypatch.setattr(sandbox, "make_oracle_tests", lambda pr, n=12: [
        {"input": [[["push", 1], ["pop"]]], "expected": [None, 1]}])
    monkeypatch.setattr(sandbox, "verdict_entry", verdict_entry)
    monkeypatch.setattr("main.mutation.validate_oracle", lambda pr, t, emit=None: {
        "final_tests": t, "strong": False, "kill_rate": 0.5,
        "kill_rate_direct": 0.5, "status": "needs_review", "needs_review": True,
        "undetermined": 1, "mutants": []})

    sandbox.get_oracle_tests(p)
    assert store[key]["accepted_by"] == "dr-who", "the acceptance was erased"
    assert store[key]["status"] == "needs_review"
    assert store[key]["features"] == sorted(oracle_features(p))


# ── the generated suites must actually drive the error paths ─────────────

def test_error_path_sequences_kill_what_valid_input_cannot(problems):
    """The measurement behind the prompt change. Calculator's undetermined
    checks all sit on guard branches, and no generated sequence ever fed input
    that reached one."""
    p = problems["calculator-calculate"]
    entry = get_resolved_entry(p)["entry_name"]
    seqs = [[["calculate"]],                                   # nothing set up
            [["setExpr", "5 +"], ["calculate"]],               # unparseable
            [["setExpr", "1 / 0"], ["calculate"]]]             # division by zero
    inputs = [[s] for s in seqs]
    ref = context.reference_program(p)
    base = run_solution(ref, inputs, entry_name=entry)
    assert base["ok"]
    guards = ("remove `if not isinstance", "remove `if postfix is None",
              "remove `if right == 0")
    killed = 0
    for m in mutation.generate_mutants(p["solution"]):
        if not any(g in m["label"] for g in guards):
            continue
        out = run_solution(mutation._runnable(p, m["code"]), inputs,
                           entry_name=entry)
        if not out["ok"] or any(mutation._norm(a) != mutation._norm(b)
                                for a, b in zip(out["results"], base["results"])):
            killed += 1
    assert killed >= 3, f"only {killed} guard mutants died to bad input"


# ── the decomposer needs the class it is writing inside ──────────────────

def test_the_decomposer_is_shown_the_class_but_not_the_body_it_must_write(problems):
    """It is asked for a method BODY and was given the docstring and the `def`
    line - so for calculateExpressions it had to guess that the input arrives
    on self.expressions, that _replaceVariables exists and what it returns on
    bad input, and that a Calculator is what evaluates. It guessed, and the
    assembled body was gated against an oracle built from the real class."""
    p = problems["advanced-calculator-calculate-expressions"]
    around = context.surrounding_class(p)
    assert "def calculateExpressions(self):" in around
    assert "YOUR CHUNKS GO HERE" in around
    # the siblings the chunks actually run against
    assert "def _replaceVariables(self, expr):" in around
    assert "class Calculator:" in around
    # ...but never this method's own body
    assert "report['_return_'] = value" not in around
    compile(around, "<t>", "exec")


def test_a_plain_function_gets_no_class_context(problems):
    assert context.surrounding_class({"entry_hint": "f", "solution": "def f(): pass"}) == ""


# ── gate 2: a sub-question states a goal, never the method ───────────────

def _step(prompt, reference):
    from types import SimpleNamespace
    return SimpleNamespace(step_id="Part 1", prompt=prompt, reference=reference)


def test_a_prompt_that_names_the_mechanism_is_rejected(problems):
    """Showing the decomposer the reference is necessary - it cannot match an
    oracle built from that code otherwise - and it pulls the PROMPTS towards
    describing it. Nothing checked the prompts, so the pull won."""
    from main.gates import check_prompts
    p = problems["advanced-calculator-calculate-expressions"]
    out = check_prompts([_step("Initialize the states and report dictionary.",
                               "self.states = {}")], p)
    assert out["status"] == "fail" and "Initialize" in out["summary"]


def test_a_prompt_in_the_problems_own_words_is_accepted(problems):
    from main.gates import check_prompts
    p = problems["advanced-calculator-calculate-expressions"]
    out = check_prompts([_step("Work through the statements in order and give "
                               "back a report of what each one left behind.",
                               "self.states = {}\nreport = {}")], p)
    assert out["status"] == "pass", out["summary"]


def test_a_name_only_the_solution_uses_is_a_leak(problems):
    from main.gates import check_prompts
    p = problems["stack-pop"]
    out = check_prompts([_step("Return whatever removedNode was holding.",
                               "removedNode = self.top")], p)
    assert out["status"] == "fail" and "removedNode" in out["summary"]
