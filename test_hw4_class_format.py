"""test_hw4_class_format.py - an assignment of linked classes (HW4, 6 Oct) can
be prepared: uploaded as methods, tested through the objects it hands around.

THE REPORT. HW4 (a cache built on a doubly linked list) came back 0/10 ready,
every problem "No usable test cases could be generated". Each answer had been
written as a standalone `def cache_list_put(self, ...)` the classes called, and
a standalone problem is tested with plain values - `self` is an object.

Converted to methods (HW3's format) it still could not have worked, for four
reasons this file pins, each one measured on the real HW4 file:
  - every run began `CacheList()`, and CacheList takes a size: every test of
    every problem crashed before its first call, the teacher's included;
  - a test could not read `tail` or `previous` - the links HW4 is about - and
    the test writers were never told those exist, or how to build the object;
  - a program testing `c[x]` or `5 in c` was dropped for not naming
    __getitem__ / __contains__;
  - the teacher's examples hand objects around, so as a call list they shrank
    to `CacheList(200); clear()` - and Cache.insert and Cache.__setitem__, too
    short to mutate, had nothing left to be trusted on: never ready.

REAL: the upload parser, the run driver, the permitted-attribute rule, the
test writers' prompts and filter, the free probes, the example block, the
feature record. Inline files; the model is faked and never reached.
"""
import json

from main import context, mutation, oracle_store
from main.assignments import parse_assignment_file
from tests import sandbox
from tests.sandbox import run_solution

FILE = '''"""Linked boxes"""

class Node:
    def __init__(self, value):
        self.value = value
        self.next = None
        self.previous = None


class Chain:
    """
        >>> c = Chain()
        >>> c.add(1)
        >>> c.add(2)
        >>> c.head.value
        2
    """
    # --- steps: add ---
    def __init__(self):
        self.head = None
        self.tail = None

    def add(self, value):
        """Put value at the front, keeping both links."""
        node = Node(value)
        node.next = self.head
        if self.head is None:
            self.tail = node
        else:
            self.head.previous = node
        self.head = node


class Holder:
    """
        >>> h = Holder()
        >>> h.put(3)
        >>> h.put(4)
    """
    # --- steps: put ---
    def __init__(self):
        self.chain = Chain()

    def put(self, value):
        """Add value to the chain it holds."""
        self.chain.add(value)


class Shelf:
    """
        >>> s = Shelf(2)
        >>> a = Node(5)
        >>> s.put(a)
        >>> 5 in s
        True
        >>> s.chain.head.value
        5
    """
    # --- steps: put, __contains__ ---
    def __init__(self, cap):
        self.cap = cap
        self.chain = Chain()

    def put(self, node):
        """Put the node's value on the chain, while there is room."""
        if self.cap > 0:
            self.chain.add(node.value)
            self.cap -= 1

    def __contains__(self, value):
        """Is value on the chain?"""
        node = self.chain.head
        while node is not None:
            if node.value == value:
                return True
            node = node.next
        return False


class Tally:
    """
        >>> Tally().count(Chain())
        0
    """
    # --- steps: count ---
    def count(self, chain):
        """How many values the chain holds."""
        n, node = 0, chain.head
        while node is not None:
            n, node = n + 1, node.next
        return n


class Loose:
    """A class with no steps line: its methods past the scaffolding are
    exercises, by the parser's own default."""
    def __init__(self):
        self.made = 1

    def tweak(self):
        """Set the secret."""
        self.secret = 2
'''

AS_FUNCTIONS = '''"""Linked boxes"""

class Chain:
    def __init__(self):
        self.head = None

    def add(self, value):
        return chain_add(self, value)


# --- problem: chain-add ---
def chain_add(self, value):
    """Put value at the front."""
    self.head = value
'''


def _problems():
    out = parse_assignment_file(FILE, "boxes.py")
    assert out["errors"] == [], out["errors"]
    return {p["slug"]: p for p in out["problems"]}


def _run(problem, *inputs):
    r = run_solution(context.reference_program(problem), [list(i) for i in inputs],
                     entry_name=context.SEQ_ENTRY)
    assert r["ok"], r
    return r["results"]


def _prompts(monkeypatch, problem, programs=()):
    """What the call-list writer and the block writer are asked, and the
    blocks the second keeps."""
    asked = []

    def chat(model, system, messages, **_k):
        asked.append(messages[0]["content"])
        return json.dumps({"sequences": [], "programs": list(programs)})
    monkeypatch.setattr(sandbox, "chat", chat)
    cls, target = problem["group_title"], problem["entry_hint"]
    sandbox._generate_call_sequences(problem, 4)
    kept = sandbox._generate_blocks(problem, cls, target, context.class_methods(problem))
    calls = next(a for a in asked if "CALL SEQUENCES" in a)
    blocks = next(a for a in asked if "short Python programs" in a)
    return (calls, blocks), kept


# ── the upload ───────────────────────────────────────────────────────────

def test_a_standalone_problem_taking_self_is_refused_at_upload_with_the_fix():
    out = parse_assignment_file(AS_FUNCTIONS, "boxes.py")
    assert out["problems"] == []
    err = out["errors"][0]["error"]
    assert "takes `self`" in err and "steps" in err, err


# ── building the object ──────────────────────────────────────────────────

def test_a_run_that_builds_with_arguments_runs():
    shelf = _problems()["shelf-contains"]
    assert _run(shelf, [[["new", 1], ["__contains__", 5]]]) == [[None, False]]
    # ...and one that never says "new" records the error per call; it does not
    # void the run (a TypeError raised before the first call used to).
    out = _run(shelf, [[["__contains__", 5]]])[0]
    assert out[0].startswith(context.ERROR_PREFIX + "TypeError")


def test_the_test_writers_are_told_how_to_build_it(monkeypatch):
    shelf = _problems()["shelf-put"]
    asked, _ = _prompts(monkeypatch, shelf)
    calls, blocks = asked
    assert 'Begin EVERY sequence with ["new", ...] supplying them, e.g. ["new", 2]' in calls
    assert "build a Shelf(2) itself" in blocks
    assert '["new", 2]' in mutation._method_input_spec(shelf)
    # ...and the free probe of the method "on a fresh object" builds it that way.
    seed = [{"input": [[["new", 2], ["put", 1]]], "expected": [None, None]}]
    assert mutation._sequence_probes(shelf, seed)[0][0][0] == ["new", 2]


def test_a_class_built_with_no_arguments_is_asked_for_as_before(monkeypatch):
    asked, _ = _prompts(monkeypatch, _problems()["holder-put"])
    assert all("alone is an error" not in a for a in asked)
    assert "build a Holder() itself" in asked[1]


# ── what a test may read ─────────────────────────────────────────────────

def test_a_block_may_read_what_given_code_in_any_class_fixes():
    holder = _problems()["holder-put"]
    assert context.block_is_permitted(
        holder, "h = Holder()\nh.put(1)\nh.put(2)\nh.chain.tail.previous.value")
    assert context.block_is_permitted(holder, "h = Holder()\nh.chain.head.next")


def test_nothing_a_student_writes_is_offered():
    holder = _problems()["holder-put"]
    assert not context.block_is_permitted(holder, "x = Loose()\nx.tweak()\nx.secret"), \
        "Loose.tweak is an exercise (no steps line), not given"
    assert context.block_is_permitted(holder, "x = Loose()\nx.made")


def test_the_test_writers_are_told_the_names_further_along(monkeypatch):
    """Both are told "...you MAY read: ... Nothing else." - a wider permission
    they are never told about would never be used."""
    holder = _problems()["holder-put"]
    spec = mutation._method_input_spec(holder)
    assert "You may read: chain; and, on the objects those hold, " in spec
    assert "previous" in spec and "secret" not in spec
    asked, _ = _prompts(monkeypatch, holder)
    assert "you MAY read: chain; and, on the objects those hold, " in asked[1]


def test_names_are_read_off_the_object_only_when_they_live_on_it():
    """The free derived block reads each name straight off the object, so it
    must get this class's own names only: `o.previous` on a Holder - like
    `o.top` on HW3's Calculator - is AttributeError on every run."""
    holder = _problems()["holder-put"]
    assert context.fixed_internals(holder) == {"chain"}
    block = mutation._as_block(holder, [["put", 1]])
    assert "o.chain" in block and "o.previous" not in block and "o.head" not in block


def test_the_record_of_which_tests_a_problem_gets_is_unchanged():
    """Decided by the class's OWN given methods, as before - widening what a
    block may read must not turn a saved, certified suite stale (it would have
    for HW3's three Calculator problems, which were live)."""
    chain = _problems()["chain-add"]
    assert context.fixed_internals(chain) == {"head", "tail"}
    assert oracle_store.oracle_features(chain) == {"calls", "blocks/4"}
    # Like Calculator: none of its OWN given code fixes anything, so it was
    # tested by calls alone - and must stay so, though Node and Chain now count.
    tally = _problems()["tally-count"]
    assert context.fixed_internals(tally) == set()
    assert oracle_store.oracle_features(tally) == {"calls"}
    assert mutation._as_block(tally, [["count", None]]) is None


# ── a method reached through an operator ─────────────────────────────────

def test_a_program_using_the_operator_counts_for_its_method(monkeypatch):
    shelf = _problems()["shelf-contains"]
    program = "s = Shelf(1)\ns.put(Node(3))\n3 in s\n4 in s"
    _, kept = _prompts(monkeypatch, shelf, [program, "s = Shelf(1)\ns.cap"])
    assert kept == [program], "`3 in s` IS __contains__; `s.cap` reaches nothing"


# ── the teacher's own examples ───────────────────────────────────────────

def test_examples_that_hand_objects_around_are_kept_whole(monkeypatch):
    shelf = _problems()["shelf-contains"]
    taught = context.doctest_block(shelf)
    assert taught and "s.put(a)" in taught and "5 in s" in taught
    assert _run(shelf, [taught]) == [[None, None, None, True, 5]]
    # The method is too short to mutate much; the example is what it is trusted on.
    assert context.doctest_covers(shelf)
    monkeypatch.setattr(sandbox, "chat", lambda *a, **k: "{}")
    assert [taught] in sandbox._generate_call_sequences(shelf, 4), "it is in the suite"


def test_examples_a_call_list_holds_are_left_as_they_were():
    """Every HW3 class but three: nothing new in the suite."""
    assert context.doctest_block(_problems()["holder-put"]) is None
