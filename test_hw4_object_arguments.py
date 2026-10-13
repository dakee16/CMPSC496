"""test_hw4_object_arguments.py - a method handed an object the file defines is
tested with real objects, and a run the teacher's own code crashes on is never
a test.

THE REPORT (13 Oct, HW4 live data). 3 of HW4's 10 problems were held "waiting
on your review", and the 7 ready ones were worse than they looked:
  - `put(content, ...)` takes a ContentItem. A call list holds only plain
    values, and neither test writer was told what a ContentItem is, so they
    passed dicts - {"id": 1000, "size": 10, ...}. The teacher's code crashed
    on them and the crash was saved as the answer: 60 of cache-list-update's
    64 tests, 57 of cache-insert's 66. Only the exception TYPE is compared, so
    a correct __getitem__ that reads `content.cid` before hashing crashed with
    AttributeError where the teacher's said TypeError - graded INCORRECT, an
    attempt used, on its last step.
  - With nothing reaching the real work, update's size check was reached by 1
    test of 64, and the evicts' unlinking by none that could see it: 5 checks
    undecided. Programs that BUILD ContentItems settle 4 of them (measured).

Box and Item are that shape in miniature: `find` hashes first, as
Cache.__getitem__ does.

REAL: the upload parser, the test writers' prompts and their filter, the run
driver, the counterexample acceptance, the feature record, grade_submission.
FAKED: the model - it proposes inputs only, never reached by grading.
"""
import json
import types

import pytest

from main import context, mutation, oracle_store
from main.assignments import parse_assignment_file
from tests import sandbox

FILE = '''"""Boxes of items"""

class Item:
    """
        >>> hash(Item(3, 10, "a"))
        1
    """
    # --- steps: __hash__ ---
    def __init__(self, cid, size, tag):
        self.cid = cid
        self.size = size
        self.tag = tag

    def __hash__(self):
        """The shelf an item goes on: its cid modulo 2."""
        return self.cid % 2


class Box:
    """
        >>> b = Box(50)
        >>> a = Item(1, 10, "a")
        >>> b.put(a)
        'PUT 1'
        >>> b.find(a)
        'FOUND 1'
        >>> b.find(Item(2, 5, "b"))
        'MISSING'
    """
    # --- steps: put, find, swap ---
    def __init__(self, room):
        self.room = room
        self.shelves = [[], []]

    def put(self, item):
        """Shelve item if it fits: 'PUT <cid>', or 'FULL' when it does not."""
        if item.size > self.room:
            return 'FULL'
        self.room -= item.size
        self.shelves[hash(item)].append(item.cid)
        return f'PUT {item.cid}'

    def find(self, item):
        """Look on item's shelf for its cid: 'FOUND <cid>' or 'MISSING'."""
        shelf = self.shelves[hash(item)]
        if item.cid in shelf:
            return f'FOUND {item.cid}'
        return 'MISSING'

    def swap(self, cid, item):
        """Put item in place of the shelved cid: 'SWAPPED <new cid>', or
        'MISSING' when cid is not shelved."""
        if not any(cid in shelf for shelf in self.shelves):
            return 'MISSING'
        for shelf in self.shelves:
            if cid in shelf:
                shelf.remove(cid)
        self.shelves[hash(item)].append(item.cid)
        return f'SWAPPED {item.cid}'


class Tray:
    """
        >>> t = Tray()
        >>> t.push(1)
        >>> t.pop()
        1
    """
    # --- steps: pop ---
    def __init__(self):
        self.items = []

    def push(self, x):
        self.items.append(x)

    def pop(self):
        """Remove and return the last item; IndexError when there is none."""
        if not self.items:
            raise IndexError('empty tray')
        return self.items.pop()
'''

# What the writers sent for HW4, in miniature: dicts where an Item goes.
AS_DICTS = [[["new", 50], ["put", {"cid": 1, "size": 10, "tag": "a"}],
             ["find", {"cid": 1, "size": 10, "tag": "a"}]],
            [["new", 50], ["find", {"cid": 2, "size": 5, "tag": "b"}]]]
PLAIN = [["new", 50]]
BUILDS = "b = Box(50)\na = Item(3, 10, 'a')\nb.put(a)\nb.find(a)\nb.find(Item(4, 1, 'b'))"
DICT_IN_A_PROGRAM = "b = Box(50)\nb.find({'cid': 2, 'size': 1, 'tag': 'x'})"
CRASHES = "b = Box(50)\nb.put(Item(1, 10, 'a'))\nb.find(None)"

# A correct find that reads the cid before hashing - HW4's false conviction.
CID_FIRST = ("cid = item.cid\nshelf = self.shelves[hash(item)]\nif cid in shelf:\n"
             "    return f'FOUND {cid}'\nreturn 'MISSING'")
ALWAYS_MISSING = "shelf = self.shelves[hash(item)]\nreturn 'MISSING'"
# HW4's update in miniature: the teacher's code answers 'MISSING' before it
# ever touches the item, so a dict there crashes nothing - until a correct
# answer reads the item first.
SWAP_DICT = [["new", 50], ["swap", 7, {"cid": 1, "size": 1, "tag": "a"}]]
ITEM_FIRST = ("new = self.shelves[hash(item)]\nif not any(cid in s for s in self.shelves):\n"
              "    return 'MISSING'\nfor s in self.shelves:\n    if cid in s:\n"
              "        s.remove(cid)\nnew.append(item.cid)\nreturn f'SWAPPED {item.cid}'")


def _problems():
    out = parse_assignment_file(FILE, "boxes.py")
    assert out["errors"] == [], out["errors"]
    return {p["slug"]: p for p in out["problems"]}


@pytest.fixture(autouse=True)
def _scratch(monkeypatch, tmp_path):
    """Every test here resolves entry points and may touch the caches: all of it
    goes to temp files, never the repo's resolved_entries.json or data/."""
    from main import identity
    monkeypatch.setenv("MICROTUTOR_ORACLE_CACHE", str(tmp_path / "o.json"))
    monkeypatch.setenv("MICROTUTOR_SESSION_DB", str(tmp_path / "s.db"))
    monkeypatch.setenv("MICROTUTOR_TRACE_FILE", str(tmp_path / "t.jsonl"))
    monkeypatch.setattr(identity, "_RESOLVED_PATH", str(tmp_path / "r.json"))


@pytest.fixture
def env(monkeypatch, tmp_path):
    from main import grading, identity, ollama_client
    monkeypatch.setenv("MICROTUTOR_ORACLE_CACHE", str(tmp_path / "o.json"))
    monkeypatch.setenv("MICROTUTOR_SESSION_DB", str(tmp_path / "s.db"))
    monkeypatch.setenv("MICROTUTOR_TRACE_FILE", str(tmp_path / "t.jsonl"))
    monkeypatch.setattr(identity, "_RESOLVED_PATH", str(tmp_path / "r.json"))
    monkeypatch.setattr(ollama_client, "_openai_chat",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no model")))
    asked = []

    def writers(model, system, messages, **_k):
        asked.append(messages[0]["content"])
        return json.dumps({"sequences": AS_DICTS + [PLAIN],
                           "programs": [BUILDS, DICT_IN_A_PROGRAM, CRASHES]})
    monkeypatch.setattr(sandbox, "chat", writers)
    grading._VERDICT_MEMO.clear()
    return asked


def _suite(problem):
    return sandbox.make_oracle_tests(problem)


def _grade_last_step(problem, tests, body):
    """`body` as a one-step roadmap's only step - its last - through grade_submission."""
    from main import grading, sessions
    from main.identity import content_hash
    header = problem["solution"].splitlines()[0]
    chunk = types.SimpleNamespace(step_id="Part 1", expected_type="code",
                                  prompt="Write the method.",
                                  reference=context.solution_body(problem))
    sid = sessions.create_session(dict(problem), {"header": header, "chunks": [chunk]},
                                  content_hash(problem), student_id="stu")["session_id"]
    grading._VERDICT_MEMO.clear()
    return grading.grade_submission(sessions.load_session(sid), body,
                                    oracle_loader=lambda p: list(tests))


# ── the suite ────────────────────────────────────────────────────────────

def test_a_dict_in_place_of_an_object_is_never_a_test(env):
    tests = _suite(_problems()["box-find"])
    assert tests, "the plain run and the program that builds Items remain"
    assert not any(isinstance(c, list) and any(isinstance(a, dict) for a in c[1:])
                   for t in tests if isinstance(t["input"][0], list)
                   for c in t["input"][0]), [t["input"] for t in tests]
    assert [BUILDS] in [t["input"] for t in tests]
    assert [DICT_IN_A_PROGRAM] not in [t["input"] for t in tests]


def test_a_dict_the_teachers_code_never_touches_is_still_not_a_test(env, monkeypatch):
    p = _problems()["box-swap"]
    monkeypatch.setattr(sandbox, "chat", lambda *a, **k: json.dumps(
        {"sequences": [SWAP_DICT, PLAIN], "programs": [
            "b = Box(50)\nb.put(Item(1, 10, 'a'))\nb.swap(1, Item(4, 2, 'b'))\nb.swap(9, Item(5, 2, 'c'))"]}))
    tests = _suite(p)
    assert [SWAP_DICT] not in [t["input"] for t in tests], "no crash to drop it - the dict rule must"
    r = _grade_last_step(p, tests, ITEM_FIRST)
    assert (r.verdict, r.tier) == ("correct", "execution-final"), r.student_reason


def test_a_run_the_teachers_code_crashes_on_is_not_a_test(env):
    tests = _suite(_problems()["box-find"])
    assert [CRASHES] not in [t["input"] for t in tests]
    assert not any(isinstance(o, str) and o.startswith(context.ERROR_PREFIX)
                   for t in tests for o in t["expected"]), "nothing expects a crash"


def test_unless_the_method_raises_on_purpose(monkeypatch):
    """Tray.pop says it raises IndexError, and does: that crash is the spec."""
    monkeypatch.setattr(sandbox, "chat", lambda *a, **k: json.dumps(
        {"sequences": [[["pop"]], [["push", 2], ["pop"], ["pop"]]], "programs": []}))
    tests = _suite(_problems()["tray-pop"])
    assert [[["pop"]]] in [t["input"] for t in tests]
    assert any(o.startswith(context.ERROR_PREFIX + "IndexError")
               for t in tests for o in t["expected"] if isinstance(o, str))


def test_a_method_handed_nothing_keeps_every_test(monkeypatch):
    """The rule reaches only problems whose methods are handed the file's
    objects; Item.__hash__ builds only itself."""
    monkeypatch.setattr(sandbox, "chat", lambda *a, **k: json.dumps(
        {"sequences": [[["new", 3, 10, "a"], ["__hash__"]]], "programs": []}))
    assert context.argument_classes(_problems()["item-hash"]) == []
    assert _suite(_problems()["item-hash"])


# ── grading with it ──────────────────────────────────────────────────────

def test_a_correct_answer_that_fails_differently_on_a_dict_is_correct(env):
    p = _problems()["box-find"]
    tests = _suite(p)
    r = _grade_last_step(p, tests, CID_FIRST)
    assert (r.verdict, r.tier) == ("correct", "execution-final"), r.student_reason


def test_the_teachers_answer_is_correct_and_a_wrong_one_is_still_caught(env):
    p = _problems()["box-find"]
    tests = _suite(p)
    assert _grade_last_step(p, tests, context.solution_body(p)).verdict == "correct"
    r = _grade_last_step(p, tests, ALWAYS_MISSING)
    assert r.verdict == "incorrect", "the program that builds Items still catches it"


# ── the writers, and the search ──────────────────────────────────────────

def test_the_writers_are_told_how_to_build_what_a_method_is_handed(env):
    p = _problems()["box-find"]
    assert context.argument_classes(p) == [
        {"name": "Item", "params": ["cid", "size", "tag"], "example": "Item(1, 10, 'a')"}]
    _suite(p)
    calls = next(a for a in env if "CALL SEQUENCES" in a)
    programs = next(a for a in env if "short Python programs" in a)
    assert "never pass a JSON object" in calls
    assert "Item(cid, size, tag), e.g. Item(1, 10, 'a')" in programs
    assert "ORDINARY USE" in programs and "Never pass None, a dict" in programs
    spec = mutation._method_input_spec(p)
    assert "Item(cid, size, tag), e.g. Item(1, 10, 'a')" in spec
    assert "never pass a JSON object" in spec


def test_a_class_handed_nothing_is_asked_exactly_as_before(env):
    """Box's prompts change; a class whose examples build nothing else does not."""
    from test_hw4_class_format import _problems as boxes
    holder = boxes()["holder-put"]
    assert context.argument_classes(holder) == []
    sandbox._generate_call_sequences(holder, 4)
    assert all("JSON object" not in a and "ORDINARY USE" not in a for a in env)


def test_the_search_never_keeps_one_either():
    """A counterexample becomes a permanent test, so the same rule holds there.
    Each run outside the spec DOES separate the mutant - only the rule stops it."""
    p = _problems()["box-swap"]
    original = context.reference_program(p)
    at = original.index("def swap")
    gone = original[:at] + original[at:].replace("return 'MISSING'", "return 'GONE'", 1)
    unshelved = original[:at] + original[at:].replace(
        "self.shelves[hash(item)].append(item.cid)", "pass", 1)
    off = sandbox.off_spec(p)
    for mutant, outside, inside in (
            # a dict the teacher's code never touches
            (gone, [SWAP_DICT], ["b = Box(50)\nb.swap(9, Item(1, 1, 'a'))"]),
            # a run the teacher's code crashes on (IndexError; the mutant's differs)
            (unshelved, ["b = Box(50)\nb.put(Item(1, 10, 'a'))\nb.swap(1, 5)"],
             ["b = Box(50)\nb.put(Item(1, 10, 'a'))\nb.swap(1, Item(2, 1, 'b'))"
              "\nb.find(Item(2, 1, 'b'))"])):
        assert mutation._first_disagreement(original, mutant, context.SEQ_ENTRY,
                                            [outside], set(), True) is not None, \
            "without the rule it IS a witness - so the rule is what refuses it"
        assert mutation._first_disagreement(original, mutant, context.SEQ_ENTRY,
                                            [outside], set(), True, off_spec=off) is None
        found = mutation._first_disagreement(original, mutant, context.SEQ_ENTRY,
                                             [inside], set(), True, off_spec=off)
        assert found and found["input"] == inside


# ── the suites saved before this ─────────────────────────────────────────

def test_hw4_shaped_suites_are_redone_and_no_others():
    p = _problems()
    assert "objects/1" in oracle_store.oracle_features(p["box-find"])
    assert oracle_store.is_stale({"strong": True, "features": ["blocks/4", "calls"]},
                                 p["box-find"])
    assert "objects/1" not in oracle_store.oracle_features(p["item-hash"])
    assert "objects/1" not in oracle_store.oracle_features(p["tray-pop"])
