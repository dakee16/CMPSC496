"""
test_class_problems.py - a problem that is one METHOD of a class, end to end.

The pipeline's oracle contract is "call the entry with these args, compare the
return value". That works for a pure method and cannot express a stateful one:
push(2) returns None, and pop() only means 6 after three pushes. main/context.py
closes that by making the SEQUENCE the argument - it injects a driver that
replays a recorded run against one instance - so the rest of the pipeline keeps
the single contract it already has.

What is worth testing here is not "does a stack work". It is the four places a
class problem could silently be handled as if it were a plain function:

  * the module a method lives in must be REBUILT the same way everywhere, or a
    correct submission is graded against a program it never had a chance
    against;
  * the entry point must be the driver, never the bare method;
  * two methods of the same class must not share a cache key;
  * a stateful bug must actually be CAUGHT - the whole point of the exercise.

No network, no LLM, no database: the reference implementations below are the
ground truth, exactly as a teacher's uploaded file would be.
"""
import os
import tempfile
import textwrap

# Redirect the two regenerable caches BEFORE anything imports them. Resolving an
# entry point and validating an oracle both WRITE, and a test run that leaves the
# working tree dirty - with fixture problems that exist nowhere but this file -
# is a test that costs more than it pays.
os.environ["MICROTUTOR_ORACLE_CACHE"] = os.path.join(
    tempfile.mkdtemp(prefix="microtutor-test-"), "tests_cache.json")

import main.identity as _identity                                    # noqa: E402
_identity._RESOLVED_PATH = os.path.join(
    tempfile.mkdtemp(prefix="microtutor-test-"), "resolved_entries.json")

from main.assignments import parse_assignment_file
from main.context import (SEQ_ENTRY, build_program, calls_from_docstring,
                          class_methods, doctest_covers, entry_name,
                          header_of, is_method, reference_program,
                          solution_body)
from main.identity import content_hash, get_resolved_entry
from tests.sandbox import passes_tests, run_solution

# A handout shaped like HW3: a helper class the exercise builds on, a class
# whose specification lives on the CLASS docstring and carries a recorded
# doctest run, and method bodies marked with the handout's own TODO comment.
HW = textwrap.dedent('''
    """Homework - a stack."""


    class Node:
        def __init__(self, value):
            self.value = value
            self.next = None


    class Stack:
        """A LIFO stack built on linked Nodes.

        >>> x=Stack(); x.push(2); x.push(4); x.push(6)
        >>> x.pop()
        6
        >>> len(x)
        2
        >>> x.peek()
        4
        >>> x.isEmpty()
        False
        """

        def __init__(self):
            self.top = None
            self.count = 0

        def push(self, value):
            # YOUR CODE STARTS HERE
            node = Node(value)
            node.next = self.top
            self.top = node
            self.count += 1

        def pop(self):
            # YOUR CODE STARTS HERE
            node = self.top
            self.top = node.next
            self.count -= 1
            return node.value

        def peek(self):
            # YOUR CODE STARTS HERE
            return self.top.value

        def isEmpty(self):
            # YOUR CODE STARTS HERE
            return self.top is None

        def __len__(self):
            # YOUR CODE STARTS HERE
            return self.count
    ''')


def _problems():
    parsed = parse_assignment_file(HW, "hw3.py")
    assert parsed["errors"] == [], parsed["errors"]
    return {p["entry_hint"]: p for p in parsed["problems"]}


def test_a_class_becomes_one_problem_per_method():
    probs = _problems()
    # Node contributes nothing: it is a helper the exercise builds on, not a
    # broken problem, and __init__ is scaffolding rather than the exercise.
    assert list(probs) == ["push", "pop", "peek", "isEmpty", "__len__"]
    assert [probs[m]["member_order"] for m in probs] == [0, 1, 2, 3, 4]
    assert all(p["group_title"] == "Stack" for p in probs.values())
    # The specification lives on the class, so every method inherits it.
    assert all("LIFO stack" in p["description"] for p in probs.values())


def test_the_entry_point_is_the_driver_not_the_method():
    push = _problems()["push"]
    assert is_method(push)
    assert entry_name(push) == SEQ_ENTRY
    resolved = get_resolved_entry(push)
    assert resolved["entry_name"] == SEQ_ENTRY
    # Confirmation runs the WHOLE assembled module, which is also the only real
    # check that the class still compiles around the method.
    assert resolved["confirmed"] is True
    # What the STUDENT works under is a different thing from what runs.
    assert header_of(push) == "def push(self, value):"


def test_the_assembled_module_keeps_the_class_around_the_method():
    pop = _problems()["pop"]
    program = build_program(pop, solution_body(pop))
    compile(program, "<t>", "exec")
    assert "class Node" in program, "the helper class was dropped"
    assert "def peek" in program, "the rest of the class was dropped"
    assert class_methods(pop) == ["__init__", "push", "pop", "peek",
                                  "isEmpty", "__len__"]
    # A body is seated at the class's own depth, not at four columns.
    assert "        node = self.top" in program


def test_each_method_gets_its_own_cache_key():
    probs = _problems()
    keys = {m: content_hash(p) for m, p in probs.items()}
    assert len(set(keys.values())) == len(keys), keys


def _oracle(problem, sequences):
    """Tests the way make_oracle_tests builds them: the model supplies CALLS,
    the teacher's own module supplies every expected value."""
    inputs = [[s] for s in sequences]
    base = run_solution(reference_program(problem), inputs,
                        entry_name=entry_name(problem))
    assert base["ok"], base
    return [{"input": i, "expected": e} for i, e in zip(inputs, base["results"])]


def test_the_teachers_own_doctest_is_read_back_as_a_sequence():
    pop = _problems()["pop"]
    seed = calls_from_docstring(pop["description"], "Stack")
    assert seed == [["new"], ["push", 2], ["push", 4], ["push", 6],
                    ["pop"], ["len"], ["peek"], ["isEmpty"]]
    # And running it reproduces the numbers the teacher wrote next to it.
    got = _oracle(pop, [seed])[0]["expected"]
    assert got == [None, None, None, None, 6, 2, 4, False], got


def test_a_stateful_bug_is_caught():
    """The reason any of this exists.

    Every one of these mutations returns the right value from the call under
    test and is wrong only in the state it leaves behind - which is precisely
    what one isolated call, on a fresh instance, cannot see."""
    pop = _problems()["pop"]
    tests = _oracle(pop, [
        calls_from_docstring(pop["description"], "Stack"),
        [["push", 1], ["pop"], ["isEmpty"]],
        [["pop"]],                                   # popping an empty stack
        [["push", 5], ["push", 7], ["pop"], ["pop"], ["pop"]],
    ])
    entry = entry_name(pop)

    assert passes_tests(reference_program(pop), tests,
                        entry_name=entry)["fraction"] == 1.0

    # pop() returns the right value every time and forgets to shrink the stack.
    forgets_count = build_program(
        pop, "node = self.top\nself.top = node.next\nreturn node.value")
    assert passes_tests(forgets_count, tests, entry_name=entry)["fraction"] < 1.0

    # pop() removes nothing: the value is right, the stack is unchanged.
    removes_nothing = build_program(
        pop, "self.count -= 1\nreturn self.top.value")
    assert passes_tests(removes_nothing, tests, entry_name=entry)["fraction"] < 1.0


def test_how_a_method_fails_is_part_of_its_behaviour():
    """A mutant that turns a clean pop into a different exception has to be
    killed, not written off as a crashed harness."""
    pop = _problems()["pop"]
    tests = _oracle(pop, [[["pop"]], [["push", 1], ["pop"], ["pop"]]])
    assert any("!" in str(t["expected"]) for t in tests), tests
    raises_wrong_thing = build_program(
        pop, "if self.top is None:\n    raise ValueError('empty')\n"
             "node = self.top\nself.top = node.next\n"
             "self.count -= 1\nreturn node.value")
    assert passes_tests(raises_wrong_thing, tests,
                        entry_name=entry_name(pop))["fraction"] < 1.0


def test_a_trivial_method_is_certified_by_the_teachers_doctest():
    """The gate for a method too trivial to mutate.

    `return self.count` has no plausible wrong single-point implementation, so
    mutation testing generates nothing and can never reach a verdict - which is
    "we cannot measure", not "this is weak". The teacher's own `>>>` example is
    the second, narrower basis for trust, and it is kept as its own status
    rather than written into `strong`."""
    from main.mutation import evaluate_oracle
    from main.oracle_store import certified

    probs = _problems()
    verdicts = {}
    for name in ("__len__", "peek", "isEmpty", "push", "pop"):
        p = probs[name]
        report = evaluate_oracle(
            p, _oracle(p, [calls_from_docstring(p["description"], "Stack")]))
        verdicts[name] = report["status"]
        assert certified({"strong": report["strong"],
                          "status": report["status"]}), (name, report["status"])

    # The two that CAN be mutation-tested are, and are not let through on the
    # doctest: a method with enough mutants is judged on them.
    assert verdicts["push"] == "strong", verdicts
    assert verdicts["pop"] == "strong", verdicts
    # The three one-liners earn it the other way.
    assert verdicts["__len__"] == "doctest_verified", verdicts
    assert verdicts["peek"] == "doctest_verified", verdicts
    assert verdicts["isEmpty"] == "doctest_verified", verdicts


def test_a_trivial_method_the_doctest_never_calls_stays_blocked():
    """The rule has to be able to say no, or it is not a gate.

    Same trivial method, with the one example that reaches it removed from the
    class docstring: nothing now states what peek() should do, so nothing
    certifies it."""
    from main.mutation import evaluate_oracle
    from main.oracle_store import certified

    hw = HW.replace("    >>> x.peek()\n    4\n", "")
    probs = {p["entry_hint"]: p
             for p in parse_assignment_file(hw, "hw3.py")["problems"]}
    peek = probs["peek"]
    assert not doctest_covers(peek)
    report = evaluate_oracle(
        peek, _oracle(peek, [calls_from_docstring(peek["description"], "Stack")]))
    assert report["status"] == "insufficient_mutants", report["status"]
    assert not certified({"strong": report["strong"], "status": report["status"]})


def test_statement_deletion_is_generated_and_killed():
    """Straight-line stateful code offers almost nothing to an operator-flipping
    mutator - push and pop yielded two mutants each, under the floor, so their
    oracles could never earn a verdict however good they were. Deletion is also
    the operator that matches the bug: "forgot to decrement the count" is a
    removed statement."""
    from main.mutation import _MIN_MUTANTS, generate_mutants

    pop = _problems()["pop"]
    mutants = generate_mutants(pop["solution"])
    labels = [m["label"] for m in mutants]
    assert len(mutants) >= _MIN_MUTANTS, labels
    assert any("remove `self.count -= 1`" in l for l in labels), labels

    tests = _oracle(pop, [
        calls_from_docstring(pop["description"], "Stack"),
        [["push", 1], ["pop"], ["isEmpty"]],
        [["push", 5], ["push", 7], ["pop"], ["pop"]],
    ])
    for m in mutants:
        code = build_program(pop, solution_body({"solution": m["code"]}))
        res = passes_tests(code, tests, entry_name=entry_name(pop))
        assert (not res["ok"]) or res["fraction"] < 1.0, f"survived: {m['label']}"


def test_a_plain_function_is_untouched():
    """The whole flat path must behave exactly as it did before any of this."""
    parsed = parse_assignment_file(textwrap.dedent('''
        def is_leap_year(year):
            """True when `year` is a leap year."""
            return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        '''), "a.py")
    p = parsed["problems"][0]
    assert not is_method(p)
    assert entry_name(p) == "is_leap_year"
    assert header_of(p) == "", "a plain function builds its own header"
    assert class_methods(p) == []
    assert reference_program(p) == p["solution"]
    # Byte-for-byte what grading._assemble used to concatenate on its own.
    assert build_program(p, "return 1", "def f(n):") == "def f(n):\n    return 1"
