"""test_roadmap_pool_cost.py - opening a problem that already has enough saved
roadmaps does not pay for a new one.

THE MEASUREMENT. On 25 Sep, the busiest normal day, building fresh roadmaps
was the largest normal cost: $2.50 of $6.71, 97 model calls. Every live problem
but two already had 5 or more saved, gated roadmaps on the server, yet a rule
built a new one on 40% of opens anyway and appended it - so the pool only ever
grew (invert had 10) and the cost never stopped.

The fakes are the network edges only: the oracle (a fixed, certified test set)
and the fresh generator (decompose_into_chunks, the one function here that
calls the model). The choosing, and the serve gate every served roadmap goes
through, are the real code.
"""
import json
import random
import sys
import types

import pytest

SOLUTION = ("def frequency(txt):\n    counts = {}\n    for ch in txt:\n"
            "        counts[ch] = counts.get(ch, 0) + 1\n    return counts\n")
PROBLEM = {"slug": "freq-pool", "title": "Letter frequency",
           "description": "Count each character.", "solution": SOLUTION}
ORACLE = [{"input": ["aab"], "expected": {"a": 2, "b": 1}},
          {"input": ["xyzx"], "expected": {"x": 2, "y": 1, "z": 1}},
          {"input": [""], "expected": {}}]
ROADMAP = {"header": "def frequency(txt):", "chunks": [
    {"step_id": "Part 1", "prompt": "Tally the characters.", "expected_type": "code",
     "reference": "counts = {}\nfor ch in txt:\n    counts[ch] = counts.get(ch, 0) + 1"},
    {"step_id": "Part 2", "prompt": "Hand back the tally.", "expected_type": "code",
     "reference": "return counts"}]}


@pytest.fixture
def pool(monkeypatch, tmp_path):
    monkeypatch.setenv("MICROTUTOR_ORACLE_CACHE", str(tmp_path / "o.json"))
    from main import identity, ollama_client, run_phase1
    from tests import sandbox
    monkeypatch.setattr(identity, "_RESOLVED_PATH", str(tmp_path / "r.json"))

    def no_network(*_a, **_k):
        raise AssertionError("a test reached the real model provider")
    monkeypatch.setattr(ollama_client, "_openai_chat", no_network)
    monkeypatch.setattr(ollama_client, "_ollama_chat", no_network)
    fakes = {"get_oracle_tests": lambda p, **k: list(ORACLE),
             "is_oracle_certified": lambda p: True}
    for name, fake in fakes.items():
        real = getattr(sandbox, name)
        for mod in list(sys.modules.values()):
            if getattr(mod, name, None) is real:
                monkeypatch.setattr(mod, name, fake)

    path = tmp_path / "pool.json"
    monkeypatch.setattr(run_phase1, "_CHUNK_POOL_PATH", str(path))
    built = []

    def fresh(problem, *_a, **_k):
        built.append(problem["slug"])
        return run_phase1._deserialize(ROADMAP)
    monkeypatch.setattr(run_phase1, "decompose_into_chunks", fresh)
    # Every coin flip comes up "build a new one" - the old rule's worst case.
    monkeypatch.setattr(random, "random", lambda: 0.0)

    def saved(n):
        path.write_text(json.dumps({identity.content_hash(PROBLEM): [ROADMAP] * n}))

    def open_problem():
        return run_phase1.get_chunk_decomposition(dict(PROBLEM))

    def count():
        return len(json.loads(path.read_text())[identity.content_hash(PROBLEM)])
    return types.SimpleNamespace(saved=saved, open=open_problem, built=built, count=count)


def test_a_full_pool_is_served_without_building_a_new_roadmap(pool):
    pool.saved(5)
    for _ in range(20):
        assert [c.step_id for c in pool.open()["chunks"]] == ["Part 1", "Part 2"]
    assert pool.built == [], f"built {len(pool.built)} roadmaps for a problem that had 5"
    assert pool.count() == 5, "the pool must stop growing once it is full"


def test_a_short_pool_still_fills_up_to_five_then_stops(pool):
    pool.saved(3)
    for _ in range(10):
        pool.open()
    assert len(pool.built) == 2 and pool.count() == 5


def test_a_problem_with_no_saved_roadmap_gets_one(pool):
    pool.saved(0)
    pool.open()
    assert pool.built == ["freq-pool"] and pool.count() == 1
