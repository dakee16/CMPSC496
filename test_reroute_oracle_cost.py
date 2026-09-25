"""test_reroute_oracle_cost.py - a rebuilt roadmap is gated with the TEACHER'S
tests and never pays for a new oracle.

THE BUG THIS PINS, and what it cost. reroute.build hands the decomposer the
teacher's problem with the student-shaped solution swapped in. The oracle cache
is keyed by content and content includes `solution`, so every rebuild MISSED
the cache and generated + mutation-tested a whole new oracle. For a METHOD the
swapped-in solution is the entire assignment file, so mutation testing planted
291 mutants across every method in HW3 (3-44 for the one actually being
solved) and asked gpt-4o about each survivor with the file quoted twice.
Measured on the real advanced-calculator-is-variable: 585 calls of ~12,400
tokens for one rebuild. The OpenAI export for 24 Sep 2026 shows 15,557 requests
averaging 10,788 tokens in / 135 out - that shape - and $325 in a day on a key
nothing else uses. After the fix the same rebuild makes 2 calls.

WHY THE EXISTING REROUTE TESTS NEVER SAW IT. test_restart_reroute's fixture
fakes get_oracle_tests everywhere, which is exactly the function this bug lives
behind. These tests leave it REAL: only the provider is faked, and any call the
fix should have prevented fails the test by name.
"""
import json

import pytest

from test_restart_reroute import PROBLEM, SOLUTION, ORACLE, SLUG

REROUTED_SOLUTION = ("def frequency(txt):\n"
                     "    counts = {}\n"
                     "    for ch in sorted(txt):\n"
                     "        if ch.isalpha():\n"
                     "            counts[ch] = counts.get(ch, 0) + 1\n"
                     "    return counts\n")


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A real oracle cache holding a CERTIFIED oracle for the teacher's problem,
    and a provider that refuses every call unless a test opts one in."""
    from main import identity, ollama_client
    from main.identity import content_hash

    cache = tmp_path / "oracles.json"
    cache.write_text(json.dumps({content_hash(PROBLEM): {
        "slug": SLUG, "strong": True, "status": "strong", "kill_rate": 1.0,
        "features": ["calls"], "final_tests": ORACLE}}))
    monkeypatch.setenv("MICROTUTOR_ORACLE_CACHE", str(cache))
    monkeypatch.setattr(identity, "_RESOLVED_PATH", str(tmp_path / "r.json"))

    calls = []
    allowed = {}

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
    return type("W", (), {"cache": cache, "calls": calls, "allowed": allowed})


def test_a_rerouted_problem_reads_the_teachers_tests_for_free(world):
    from tests.sandbox import get_oracle_tests, is_oracle_certified
    before = world.cache.read_text()
    rerouted = {**PROBLEM, "solution": REROUTED_SOLUTION, "oracle_from": PROBLEM}

    assert get_oracle_tests(rerouted) == ORACLE
    assert is_oracle_certified(rerouted) is True
    assert world.calls == [], world.calls
    assert world.cache.read_text() == before, \
        "a borrowed oracle must not write a new cache entry"


def test_no_certified_teacher_oracle_means_no_tests_not_a_new_one(world):
    """The fallback must be "cannot certify" -> teacher's roadmap, never
    "generate one" - generating is the expensive thing."""
    from tests.sandbox import get_oracle_tests, is_oracle_certified
    world.cache.write_text("{}")
    rerouted = {**PROBLEM, "solution": REROUTED_SOLUTION, "oracle_from": PROBLEM}

    assert get_oracle_tests(rerouted) == []
    assert is_oracle_certified(rerouted) is False
    assert world.calls == []


def test_a_full_rebuild_only_pays_for_the_proposal_and_the_split(world):
    """The real reroute.build, real decomposer, real gates, real oracle lookup.
    Only the proposal and the decomposition may reach the provider - no oracle
    generation, no mutation testing. (The decomposer may retry once when its
    prompt-wording gate objects; that is bounded by DECOMPOSE_TRIES and is not
    what this pins.)"""
    from main import reroute

    body = "\n".join(ln[4:] for ln in REROUTED_SOLUTION.splitlines()[1:])
    world.allowed["subproblem"] = json.dumps({"subproblems": [
        {"prompt": "Count each letter, in sorted order, and keep the tally.",
         "reference": "counts = {}\nfor ch in sorted(txt):\n"
                      "    if ch.isalpha():\n"
                      "        counts[ch] = counts.get(ch, 0) + 1"},
        {"prompt": "Hand back what you counted.", "reference": "return counts"}]})
    world.allowed[""] = json.dumps({"body": body})       # the proposal

    plan = {"nodes": [{"id": "s", "kind": "start", "label": "take the text"},
                      {"id": "l", "kind": "loop", "label": "sorted letters"},
                      {"id": "r", "kind": "return", "label": "the counts"}],
            "edges": []}
    out = reroute.build(PROBLEM, reroute.effective_header(PROBLEM), plan)

    assert len(out["chunks"]) == 2
    assert set(world.calls) == {"main.reroute.propose_solution",
                                "main.run_phase1.decompose_into_chunks"}, \
        world.calls
    assert len(world.calls) <= 1 + reroute.DECOMPOSE_TRIES, world.calls
