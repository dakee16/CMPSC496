"""test_replan_class_method.py - a rebuild from the student's plan works for a
METHOD, not only a plain function.

WHY THIS EXISTS (18 Oct). "These steps don't match my plan" was asked for on
HW4 __contains__ - a method of a class - after GPT's audit: an approved plan to
search from the TAIL was handed the teacher's head-first steps. main/reroute.py
was built and tested on plain functions (`frequency`, `invert`). If it could
not assemble a method inside its class, the button would only ever say "we
can't break your approach into steps" on exactly the problems it is for.

REAL: reroute.build - the oracle gate on the proposed body, the decomposer's
own gates (assembly, necessity, shape, wording), the class context around the
method. FAKED: the model's two answers (a tail-first body, and its split).
"""
import json

import pytest

SRC = '''"""Chains"""


class Node:
    def __init__(self, value):
        self.value = value
        self.next = None
        self.previous = None


class Chain:
    """
        >>> c = Chain()
        >>> c.add(1)
        >>> c.has(1)
        True
    """
    # --- steps: has ---
    def __init__(self):
        self.head = None
        self.tail = None

    def add(self, value):
        node = Node(value)
        node.next = self.head
        if self.head is None:
            self.tail = node
        else:
            self.head.previous = node
        self.head = node

    def has(self, value):
        """Whether value is in the chain."""
        current = self.head
        while current is not None:
            if current.value == value:
                return True
            current = current.next
        return False
'''
TAIL_FIRST = ("current = self.tail\nwhile current is not None:\n"
              "    if current.value == value:\n        return True\n"
              "    current = current.previous\nreturn False")
SPLIT = {"subproblems": [
    {"prompt": "Write code that picks the node at the far end of the chain to begin "
               "from, and keep it for the next step.",
     "reference": "current = self.tail"},
    {"prompt": "Write code that checks each node, moving toward the front, and says "
               "whether the value is there.",
     "reference": "while current is not None:\n    if current.value == value:\n"
                  "        return True\n    current = current.previous\nreturn False"}]}
PLAN = {"nodes": [{"id": "s", "kind": "start", "label": "start at the tail"},
                  {"id": "l", "kind": "loop", "label": "move to the previous node"},
                  {"id": "b", "kind": "branch", "label": "value matches"},
                  {"id": "r", "kind": "return", "label": "return True"},
                  {"id": "f", "kind": "return", "label": "return False"},
                  {"id": "e", "kind": "end", "label": "done"}], "edges": []}


@pytest.fixture
def chain(tmp_path, monkeypatch):
    from main import identity, ollama_client, reroute, run_phase1
    from main.assignments import parse_assignment_file
    from main.identity import content_hash
    from main.oracle_store import oracle_features, save_cache
    monkeypatch.setenv("MICROTUTOR_ORACLE_CACHE", str(tmp_path / "o.json"))
    monkeypatch.setenv("MICROTUTOR_TRACE_FILE", str(tmp_path / "t.jsonl"))
    monkeypatch.setattr(identity, "_RESOLVED_PATH", str(tmp_path / "r.json"))
    problem = parse_assignment_file(SRC, "chain.py")["problems"][0]
    calls = [[["new"], ["has", 1]], [["new"], ["add", 1], ["add", 2], ["has", 1], ["has", 5]],
             [["new"], ["add", 3], ["has", 3], ["add", 4], ["has", 3], ["has", 4]]]
    expected = [[None, False], [None, None, None, True, False],
                [None, None, True, None, True, True]]
    save_cache({content_hash(problem): {
        "strong": True, "status": "strong", "kill_rate": 1.0, "kill_rate_direct": 1.0,
        "features": sorted(oracle_features(problem)),
        "final_tests": [{"input": [c], "expected": e} for c, e in zip(calls, expected)]}})
    asked = []

    def fake_chat(model, system, messages, **kw):
        asked.append((system or "")[:30])
        if "subproblem" in (system or "").lower():
            return json.dumps(SPLIT)
        return json.dumps({"body": TAIL_FIRST})
    for mod in (ollama_client, reroute, run_phase1):
        monkeypatch.setattr(mod, "chat", fake_chat, raising=False)
    reroute._NO_ROUTE.clear()
    return problem, asked


def test_a_tail_first_plan_gets_tail_first_steps_for_a_method(chain):
    from main import reroute
    problem, asked = chain
    out = reroute.build(problem, reroute.effective_header(problem), PLAN)
    refs = [c.reference for c in out["chunks"]]
    assert refs[0].strip() == "current = self.tail", refs
    assert "current.previous" in refs[-1], refs
    assert asked, "the model was asked for a proposal and its split"


def test_a_proposal_that_fails_the_tests_is_never_served_for_a_method(chain, monkeypatch):
    """The gate that makes a rebuild safe holds for a method too: a body that
    gets the tests wrong is thrown away, never cut into steps."""
    from main import ollama_client, reroute, run_phase1
    problem, _ = chain
    wrong = TAIL_FIRST.replace("return False", "return True")
    for mod in (ollama_client, reroute, run_phase1):
        monkeypatch.setattr(mod, "chat", lambda *a, **k: json.dumps({"body": wrong}),
                            raising=False)
    with pytest.raises(reroute.RouteUnavailable):
        reroute.build(problem, reroute.effective_header(problem), PLAN)
