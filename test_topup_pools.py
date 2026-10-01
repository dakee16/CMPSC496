"""test_topup_pools.py - the one-off top-up builds roadmaps for live problems
with 1-4 saved, and nothing else, under a hard dollar cap.

WHAT IS REAL. The pool, the oracle cache, fill_pool, the decomposer and its
gates, and ollama_client itself - the cap wraps the client's own HTTP call, so
it is tested where it sits. Faked: Supabase (the problem rows) and the network
reply, which is billed like the real one.
"""
import json
import types

import pytest
import requests

from test_auth_routes import FakeSupabase
from test_restart_reroute import ORACLE, PROBLEM
from test_student_open_never_generates import ROADMAP, SPLIT


def _problem(name):
    """Distinct content, so a distinct pool and oracle key."""
    return {**PROBLEM, "slug": f"freq-{name}",
            "description": f"{PROBLEM['description']} ({name})"}


FULL, SHORT, NONE, DRAFT = (_problem(n) for n in ("full", "short", "none", "draft"))


@pytest.fixture
def world(tmp_path, monkeypatch):
    from main import identity, run_phase1, topup_pools
    from main.identity import content_hash

    cache, pool = tmp_path / "o.json", tmp_path / "pool.json"
    monkeypatch.setenv("MICROTUTOR_ORACLE_CACHE", str(cache))
    monkeypatch.setenv("MICROTUTOR_TRACE_FILE", str(tmp_path / "t.jsonl"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-never-sent")
    monkeypatch.setattr(identity, "_RESOLVED_PATH", str(tmp_path / "r.json"))
    monkeypatch.setattr(run_phase1, "_CHUNK_POOL_PATH", str(pool))
    # Every kept roadmap also gets step notes - one call of its own, counted
    # and capped in test_step_notes.py / test_fix_pools.py. Off here, so the
    # counts below stay about the builds this tool exists to pay for.
    from main import step_notes
    monkeypatch.setattr(step_notes, "write_notes", lambda problem, d, **_k: d)

    certified = {"strong": True, "status": "strong", "kill_rate": 1.0,
                 "kill_rate_direct": 1.0, "features": ["calls"], "final_tests": ORACLE}
    cache.write_text(json.dumps({content_hash(p): certified
                                 for p in (FULL, SHORT, NONE, DRAFT)}))
    pool.write_text(json.dumps({content_hash(FULL): [ROADMAP] * 5,
                                content_hash(SHORT): [ROADMAP] * 2,
                                content_hash(DRAFT): [ROADMAP]}))
    sb = FakeSupabase()
    sb.problems.extend([{**FULL, "ready": True}, {**SHORT, "ready": True},
                        {**NONE, "ready": True}, {**DRAFT, "ready": False}])
    monkeypatch.setattr(topup_pools, "_sb", lambda: sb)

    calls = []

    class Reply:
        status_code, headers, text = 200, {}, ""

        def __init__(self, body):
            self.body = body

        def raise_for_status(self):
            pass

        def json(self):
            return self.body

    def network(url, json=None, **_k):
        usage = {"prompt_tokens": 4000, "completion_tokens": 300}
        calls.append(usage)
        return Reply({"model": json["model"], "usage": usage,
                      "choices": [{"message": {"content": SPLIT}}]})
    monkeypatch.setattr(requests, "post", network)

    def saved(p):
        return len(run_phase1._load_pool().get(content_hash(p), []))
    return types.SimpleNamespace(calls=calls, saved=saved)


def _dollars(calls):
    from main.ollama_client import LIST_PRICES
    pin, pout = LIST_PRICES["gpt-4o"]
    return sum(u["prompt_tokens"] * pin + u["completion_tokens"] * pout
               for u in calls) / 1e6


def test_a_dry_run_lists_the_short_problem_and_spends_nothing(world, capsys):
    from main import topup_pools
    assert topup_pools.main([]) == 0
    out = capsys.readouterr().out
    assert "freq-short: 2 saved, needs 3" in out
    assert "freq-none" in out and "Reprepare" in out
    assert "freq-full" not in out and "freq-draft" not in out
    assert world.calls == [] and world.saved(SHORT) == 2


def test_apply_brings_only_the_short_problem_up_to_five(world):
    from main import topup_pools
    assert topup_pools.main(["--apply"]) == 0
    assert [world.saved(p) for p in (FULL, SHORT, NONE, DRAFT)] == [5, 5, 0, 1]
    assert len(world.calls) == 3


def test_the_cap_stops_before_the_call_that_could_cross_it(world, capsys):
    """Each call here really costs ~1.3c. The cap must stop the run part way,
    never let the total pass it, and keep every roadmap already paid for."""
    from main import topup_pools
    cap = 0.04
    assert topup_pools.main(["--apply", "--max-dollars", str(cap)]) == 1
    assert 0 < len(world.calls) < 3, world.calls
    assert _dollars(world.calls) <= cap
    assert world.saved(SHORT) == 2 + len(world.calls), "a paid-for build was thrown away"
    assert "STOPPED" in capsys.readouterr().out


def test_the_cap_refuses_a_model_call_from_anything_but_the_roadmap_builder(world):
    from main import ollama_client, topup_pools
    with topup_pools.spend_cap(1.0):
        with pytest.raises(topup_pools.SpendCapReached, match="refused"):
            ollama_client.chat("gpt-4o", "Return json.",
                               [{"role": "user", "content": "json please"}], fmt="json")
    assert world.calls == []
