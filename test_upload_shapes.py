"""test_upload_shapes.py - kinds of assignment no one has uploaded yet either
prepare correctly or fail at upload with the real reason - never pass
preparation and then fail every student.

THE AUDIT (6 Oct, after HW4). Common CMPSC 132 shapes, run through the real
preparation with the model faked:
  - a GENERATOR came back STRONG and ready while the teacher's own solution
    failed 5 of 5 of its tests: every answer was `<generator object at 0x..>`,
    an address that differs in every process. A random answer would have done
    the same. Every student, every attempt, wrong.
  - an answer that depends on SET ORDER was recorded under a random hash seed
    and graded under seed 0, so a correct student could not match it.
  - a program testing `a + b`, `a == b` or `list(it)` was dropped for not
    naming __add__ / __eq__ / __next__ - HW4's `c[x]` bug, for operators.

REAL: preparation (publish.prepare_problem), test generation, the runner, the
counterexample search, the grader's runner. FAKED: the model, which proposes
inputs only and is never reached.
"""
import json

import pytest

from main import context, mutation
from main.assignments import parse_assignment_file
from tests import sandbox


def _problem(src):
    out = parse_assignment_file(src, "x.py")
    assert out["errors"] == [], out["errors"]
    return out["problems"]


GENERATOR = '''"""G"""
def countdown(n):
    """Yield n, n-1, ..., 1."""
    while n > 0:
        yield n
        n -= 1
'''
# The import sits INSIDE the function: one at the top of the file never
# reaches the code that runs (a separate, open problem - see the report).
RANDOM = '''"""R"""
def roll(n):
    """A random number, plus n."""
    import random
    if n < 1:
        return None
    return random.randint(1, 10 ** 9) + n
'''
STABLE = '''"""T"""
def double(n):
    """Twice n, or 0 for negatives."""
    if n < 0:
        return 0
    return n * 2 + 0
'''
SET_ORDER = '''"""S"""
def uniq(words):
    """The distinct words."""
    if not words:
        return []
    return list(set(words))
'''
VECTOR = '''"""V"""
class Vector:
    """
        >>> a = Vector(1, 2)
        >>> b = Vector(3, 4)
        >>> a + b
        Vector(4, 6)
        >>> a == Vector(1, 2)
        True
    """
    # --- steps: __add__, __eq__ ---
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

    def __add__(self, other):
        """Add two vectors."""
        if not isinstance(other, Vector):
            return None
        return Vector(self.x + other.x, self.y + other.y)

    def __eq__(self, other):
        """Same coordinates."""
        return isinstance(other, Vector) and self.x == other.x and self.y == other.y
'''


@pytest.fixture
def model(monkeypatch, tmp_path):
    """The model proposes `inputs`; nothing else is reachable."""
    from main import identity, ollama_client
    monkeypatch.setenv("MICROTUTOR_ORACLE_CACHE", str(tmp_path / "o.json"))
    monkeypatch.setenv("MICROTUTOR_TRACE_FILE", str(tmp_path / "t.jsonl"))
    monkeypatch.setattr(identity, "_RESOLVED_PATH", str(tmp_path / "r.json"))
    monkeypatch.setattr(ollama_client, "_openai_chat",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("real model")))
    monkeypatch.setattr(mutation, "chat", lambda *a, **k: "{}")
    state = {"inputs": "[[3], [0], [5], [1], [2]]", "programs": []}
    monkeypatch.setattr(sandbox, "chat", lambda *a, **k: json.dumps(
        {"inputs_src": state["inputs"], "sequences": [], "programs": state["programs"]}))
    return state


# ── an answer has to be the same answer twice ───────────────────────────

def test_a_generator_problem_fails_at_upload_and_says_why(model):
    from main import publish
    r = publish.prepare_problem(_problem(GENERATOR)[0])
    assert (r["ready"], r["stage"]) == (False, "tests"), r
    assert "a generator" in r["error"] and "changes from run to run" in r["error"]


def test_a_random_answer_is_never_a_test(model):
    from main.context import reference_program
    p = _problem(RANDOM)[0]
    once = sandbox.run_solution(reference_program(p), [[5]], entry_name="roll")
    assert isinstance(once["results"][0], int), "it runs - it just never agrees"
    assert sandbox.make_oracle_tests(p) == []


def test_a_stable_answer_keeps_every_test(model):
    assert len(sandbox.make_oracle_tests(_problem(STABLE)[0])) == 5


def test_set_order_answers_are_recorded_the_way_students_are_graded(model):
    from main.execution import classify_run
    from main.identity import get_resolved_entry
    model["inputs"] = "[[['apple', 'pear', 'fig', 'kiwi', 'plum']], [['a', 'b', 'c', 'd']], [[]]]"
    p = _problem(SET_ORDER)[0]
    tests = sandbox.make_oracle_tests(p)
    assert len(tests) == 3
    run = classify_run(p["solution"], tests, entry_name=get_resolved_entry(p)["entry_name"])
    assert (run.passed, run.total) == (3, 3), run.failures


def test_a_counterexample_must_also_be_the_same_answer_twice(model):
    p = _problem(RANDOM)[0]
    mutant = p["solution"].replace("n < 1", "n < 2")
    assert mutation._first_disagreement(p["solution"], mutant, "roll",
                                        [[5], [1]], set()) is None
    q = _problem(STABLE)[0]
    found = mutation._first_disagreement(q["solution"], q["solution"].replace("* 2", "* 3"),
                                         "double", [[5]], set())
    assert found == {"input": [5], "expected": 10}, "a real difference still counts"


# ── operators ────────────────────────────────────────────────────────────

def test_a_program_using_the_operator_counts_for_its_method(model):
    add = next(p for p in _problem(VECTOR) if p["entry_hint"] == "__add__")
    model["programs"] = ["a = Vector(1, 2)\nb = a + Vector(0, 1)\nb.x\nb.y",
                         "a = Vector(1, 2)\na.x"]
    kept = sandbox._generate_blocks(add, "Vector", "__add__", context.class_methods(add))
    assert kept == [model["programs"][0]]
    assert context.doctest_covers(add), "the teacher's `a + b` example covers it"
    eq = next(p for p in _problem(VECTOR) if p["entry_hint"] == "__eq__")
    assert context.doctest_covers(eq)


def test_iterating_counts_for_next():
    assert context.block_exercises("c = Countdown(3)\nlist(c)", "__next__")
    assert context.block_exercises("t = 0\nfor x in Countdown(3):\n    t += x\nt", "__iter__")
    assert not context.block_exercises("c = Countdown(3)\nc.n", "__next__")


def test_an_address_is_never_a_test_even_if_it_repeats():
    """Run-twice catches an address that moves; this catches one that does
    not (a process laid out the same way twice)."""
    gen = "<generator object countdown at 0x10a1b2c30>"
    assert sandbox._useless_block([3], gen, method=False)
    assert sandbox._useless_block([3], [1, gen], method=False)
    assert not sandbox._useless_block(["racecar"], True, method=False)


# ── the class format, as the upload page shows it ────────────────────────

def test_the_class_example_on_the_upload_page_is_a_working_assignment(model):
    """Served beside the template; HW4 came in wrong because nothing on the
    page showed a class. It has to prepare like anything else would."""
    import doctest
    import types as _t
    from test_auth_routes import client
    served = client().get("/assignment_template").json()
    src = served["class_example"]
    ns = _t.ModuleType("example")
    exec(src, ns.__dict__)
    assert doctest.testmod(ns).failed == 0, "its own examples pass"
    probs = _problem(src)
    assert [p["entry_hint"] for p in probs] == ["deposit", "withdraw"]
    for p in probs:
        assert len(mutation.generate_mutants(p["solution"])) >= mutation._MIN_MUTANTS
        assert context.doctest_covers(p), p["slug"]
        seed = context.calls_from_docstring(p["group_description"], "BankAccount")
        run = sandbox.run_solution(context.reference_program(p), [[seed]],
                                   entry_name=context.SEQ_ENTRY)
        assert run["results"] == [[None, 150, "Insufficient funds", 120]]
    assert "TEMPLATE" not in src and served["content"].startswith('"""Week 1')


# ── None is an answer ────────────────────────────────────────────────────

SAFE_DIVIDE = '''"""D"""
def safe_divide(a, b):
    """Return a divided by b, or None if b is 0."""
    if b == 0:
        return None
    return a / b
'''
PRINTS = '''"""P"""
def show_double(n):
    """Print twice n."""
    if n < 0:
        print(0)
    else:
        print(n * 2)
'''


def test_return_none_on_bad_input_is_tested(model):
    """7 Oct: every None answer was dropped, so (5, 0) never became a test and
    a student with no zero check passed 4/4."""
    from main.execution import classify_run
    model["inputs"] = "[[6, 3], [10, 4], [5, 0], [0, 7], [-8, 2], [1, 0]]"
    tests = sandbox.make_oracle_tests(_problem(SAFE_DIVIDE)[0])
    assert {"input": [5, 0], "expected": None} in tests and len(tests) == 6
    forgot = classify_run("def safe_divide(a, b):\n    return a / b\n", tests,
                          entry_name="safe_divide")
    assert forgot.passed < forgot.total, "no zero check is caught now"
    right = classify_run(_problem(SAFE_DIVIDE)[0]["solution"], tests, entry_name="safe_divide")
    assert right.passed == right.total


def test_a_solution_that_returns_nothing_fails_at_upload_and_says_why(model):
    from main import publish
    r = publish.prepare_problem(_problem(PRINTS)[0])
    assert (r["ready"], r["stage"]) == (False, "tests"), r
    assert "prints its answer instead of returning it" in r["error"]
