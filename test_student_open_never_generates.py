"""test_student_open_never_generates.py - a student opening a problem can never
start test-set generation, whatever state the saved test set is in.

THE GAP THIS PINS. Grading reads the test set read-only
(oracle_store.load_strong_cached_oracle). Opening a problem did not: every open
goes through run_phase1.get_chunk_decomposition, whose serve gate
(gates.assert_serveable) and roadmap builder read it with
tests.sandbox.get_oracle_tests - the WRITE path, which regenerates and
mutation-tests when the saved set is outdated (oracle_features() changed since it
was validated) or missing (the problem's content moved). reroute.build read it
the same way. That is the class of work that cost $600 over 24-25 Sep - about
585 calls for one class problem - and it ran while a student waited.

WHAT IS REAL. The pool, the serve gate, the decomposer, reroute, and the oracle
cache lookup are the real code over a real cache file. Only the provider is
faked: it answers the roadmap split and the reroute proposal when a test allows
them, and records every call so any test-set generation fails the test by name.
"""
import json

import pytest

from test_restart_reroute import ORACLE, PROBLEM

ROADMAP = {"header": "def frequency(txt):", "chunks": [
    {"step_id": "Part 1", "prompt": "Count each letter and keep the tally.",
     "expected_type": "code",
     "reference": "counts = {}\nfor ch in txt:\n    if ch.isalpha():\n"
                  "        counts[ch] = counts.get(ch, 0) + 1"},
    {"step_id": "Part 2", "prompt": "Hand back what you counted.",
     "expected_type": "code", "reference": "return counts"}]}
SPLIT = json.dumps({"subproblems": [
    {"prompt": c["prompt"], "reference": c["reference"]} for c in ROADMAP["chunks"]]})
# The modules whose model calls ARE test-set generation or mutation testing.
GENERATION = ("tests.sandbox", "main.mutation")


@pytest.fixture
def world(tmp_path, monkeypatch):
    from main import identity, ollama_client, run_phase1
    from main.identity import content_hash

    cache = tmp_path / "oracles.json"
    pool = tmp_path / "pool.json"
    monkeypatch.setenv("MICROTUTOR_ORACLE_CACHE", str(cache))
    monkeypatch.setenv("MICROTUTOR_TRANSCRIPTS", str(tmp_path / "tr.json"))
    monkeypatch.setenv("MICROTUTOR_TRACE_FILE", str(tmp_path / "t.jsonl"))
    monkeypatch.setattr(identity, "_RESOLVED_PATH", str(tmp_path / "r.json"))
    # Read once at import, so the env var alone would not move it.
    monkeypatch.setattr(run_phase1, "_CHUNK_POOL_PATH", str(pool))

    calls, allowed = [], {}

    def provider(model, system, messages, *a, **k):
        import sys
        f = sys._getframe(1)            # the feature that asked, not this fake
        while f and f.f_globals.get("__name__") == "main.ollama_client":
            f = f.f_back
        who = f"{f.f_globals.get('__name__')}.{f.f_code.co_name}"
        calls.append(who)
        for prefix, answer in allowed.items():
            if prefix in (system or "").lower():
                return answer
        raise AssertionError(f"model call that should not happen: {who}")

    monkeypatch.setattr(ollama_client, "_openai_chat", provider)
    monkeypatch.setattr(ollama_client, "_ollama_chat", provider)

    def saved_tests(state):
        entry = {"slug": PROBLEM["slug"], "strong": True, "status": "strong",
                 "kill_rate": 1.0, "kill_rate_direct": 1.0,
                 "features": ["calls"], "final_tests": ORACLE}
        if state == "outdated":
            # What a change to oracle_features() does to every saved set: the
            # verdict was reached with a different set of generators.
            entry["features"] = ["calls", "blocks/3"]
        cache.write_text(json.dumps({} if state == "missing"
                                    else {content_hash(PROBLEM): entry}))

    def saved_roadmaps(n):
        pool.write_text(json.dumps({content_hash(PROBLEM): [ROADMAP] * n}))

    def generation():
        return [c for c in calls if c.startswith(GENERATION)]

    return type("W", (), {"cache": cache, "pool": pool, "calls": calls,
                          "allowed": allowed, "saved_tests": staticmethod(saved_tests),
                          "saved_roadmaps": staticmethod(saved_roadmaps),
                          "generation": staticmethod(generation)})


def test_an_outdated_test_set_still_serves_the_saved_roadmap_for_free(world):
    """Grading already accepts an outdated set, so opening must too - the
    student sees exactly what they saw before, and nothing is paid for."""
    from main.run_phase1 import get_chunk_decomposition
    world.saved_tests("outdated")
    world.saved_roadmaps(5)
    before = world.cache.read_text()

    out = get_chunk_decomposition(dict(PROBLEM))

    assert [c.step_id for c in out["chunks"]] == ["Part 1", "Part 2"]
    assert world.calls == [], world.calls
    assert world.cache.read_text() == before, "opening must never rewrite the test set"


def test_a_missing_test_set_is_refused_not_generated(world):
    from main.run_phase1 import NoOracleTestsError, get_chunk_decomposition
    world.saved_tests("missing")
    world.saved_roadmaps(5)

    with pytest.raises(NoOracleTestsError):
        get_chunk_decomposition(dict(PROBLEM))
    assert world.calls == [], world.calls
    assert world.cache.read_text() == "{}"


def test_filling_the_pool_pays_for_a_roadmap_but_never_for_a_test_set(world):
    """Roadmaps are built at upload (fill_pool), never on open. Even there,
    building one reads the saved test set and never regenerates it."""
    from main.identity import content_hash
    from main.run_phase1 import fill_pool
    world.saved_tests("outdated")
    world.saved_roadmaps(4)
    world.allowed["subproblem"] = SPLIT
    before = world.cache.read_text()

    assert fill_pool(dict(PROBLEM)) == 5
    assert world.generation() == [], world.generation()
    assert set(world.calls) == {"main.run_phase1.decompose_into_chunks"}, world.calls
    assert len(json.loads(world.pool.read_text())[content_hash(PROBLEM)]) == 5, \
        "the new roadmap was built and saved"
    assert world.cache.read_text() == before


def test_a_reroute_with_an_outdated_test_set_never_generates(world):
    from main import reroute
    world.saved_tests("outdated")
    world.allowed["subproblem"] = SPLIT
    body = "\n".join(ln[4:] for ln in PROBLEM["solution"].splitlines()[1:])
    world.allowed[""] = json.dumps({"body": body})        # the proposal
    before = world.cache.read_text()
    plan = {"nodes": [{"id": "s", "kind": "start", "label": "take the text"},
                      {"id": "l", "kind": "loop", "label": "each letter"},
                      {"id": "r", "kind": "return", "label": "the counts"}],
            "edges": []}

    out = reroute.build(PROBLEM, reroute.effective_header(PROBLEM), plan)

    assert len(out["chunks"]) == 2
    assert world.generation() == [], world.generation()
    assert set(world.calls) <= {"main.reroute.propose_solution",
                                "main.run_phase1.decompose_into_chunks"}, world.calls
    assert world.cache.read_text() == before
