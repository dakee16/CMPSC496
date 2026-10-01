"""test_fix_pools.py - the one-off that brings saved roadmaps up to 30 Sep's
rules: unsafe cuts replaced, failing wording re-worded, notes added to all.

REAL: the gates, the cut rule, the splitter, readiness (the teacher's code
typed flat through real grading), the pool file, the spend cap. FAKED: the
problem row (in memory), the oracle cache (a file), and OpenAI's HTTP endpoint
- a canned reply per kind of call, with a known token usage.
"""
import json
import re
import types

import pytest
import requests

from test_auth_routes import FakeSupabase
from test_bridge_renaming import WHILE
from test_restart_reroute import ORACLE, PROBLEM
from test_reword_pools import CLEAR, VAGUE

NOTE = {"starts_with": "The text, and what the earlier steps left.",
        "leaves": "How many times each letter appears so far."}


def _wording(n):
    return [f"Write code that handles part {i + 1} of the counting for each letter, "
            f"and keep the result for the next step." for i in range(n - 1)] \
        + ["Write code that returns how many times each letter appears."]


@pytest.fixture
def world(tmp_path, monkeypatch):
    from main import fix_pools, identity, reword_pools, run_phase1
    from main.identity import content_hash
    cache, pool = tmp_path / "o.json", tmp_path / "pool.json"
    monkeypatch.setenv("MICROTUTOR_ORACLE_CACHE", str(cache))
    monkeypatch.setenv("MICROTUTOR_TRACE_FILE", str(tmp_path / "t.jsonl"))
    monkeypatch.setenv("MICROTUTOR_SESSION_DB", str(tmp_path / "s.sqlite3"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-never-sent")
    monkeypatch.setattr(identity, "_RESOLVED_PATH", str(tmp_path / "r.json"))
    monkeypatch.setattr(run_phase1, "_CHUNK_POOL_PATH", str(pool))
    cache.write_text(json.dumps({content_hash(PROBLEM): {
        "strong": True, "status": "strong", "kill_rate": 1.0, "kill_rate_direct": 1.0,
        "features": ["calls"], "final_tests": ORACLE}}))
    pool.write_text(json.dumps({content_hash(PROBLEM): [WHILE, VAGUE, CLEAR]}))
    sb = FakeSupabase()
    sb.problems.append({**PROBLEM, "ready": True})
    monkeypatch.setattr(reword_pools, "_sb", lambda: sb)
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
        system, user = json["messages"][0]["content"], json["messages"][-1]["content"]
        n = len(re.findall(r"STEP \d+", user))
        if "two short notes" in system:
            body = {"notes": [NOTE] * n}
            calls.append("notes")
        else:
            body = {"prompts": _wording(len(re.findall(r"STEP \d+ CODE", user)))}
            calls.append("wording")
        usage = {"prompt_tokens": 1500, "completion_tokens": 200}
        return Reply({"model": json["model"], "usage": usage,
                      "choices": [{"message": {"content": __import__("json").dumps(body)}}]})
    monkeypatch.setattr(requests, "post", network)

    def saved():
        return run_phase1._load_pool()[content_hash(PROBLEM)]
    return types.SimpleNamespace(calls=calls, saved=saved, pool=pool, mod=fix_pools,
                                 file=str(tmp_path / "fix.json"))


def test_listing_names_each_fix_and_spends_nothing(world, capsys):
    before = world.pool.read_text()
    assert world.mod.main([]) == 0
    out = capsys.readouterr().out
    assert "'unsafe': 1" in out and "'wording': 1" in out and "'notes': 2" in out, out
    assert world.calls == [] and world.pool.read_text() == before


def test_a_draft_changes_nothing(world):
    before = world.pool.read_text()
    assert world.mod.main(["--draft", world.file]) == 0
    assert world.pool.read_text() == before
    drafted = json.load(open(world.file))
    assert sorted(r["why"] for r in drafted) == ["notes", "unsafe", "wording+notes"], drafted


def test_apply_fixes_every_roadmap_with_no_model_call(world):
    world.mod.main(["--draft", world.file])
    before = len(world.calls)
    assert world.mod.main(["--apply", world.file]) == 0
    assert len(world.calls) == before, "apply must never call the model"
    from main import splitter
    saved = world.saved()
    assert WHILE not in saved and VAGUE not in saved and CLEAR not in saved
    assert saved and all(world.mod.needs(dict(PROBLEM), e) == [] for e in saved), \
        [world.mod.needs(dict(PROBLEM), e) for e in saved]
    assert not any(splitter.unsafe_cut(PROBLEM, e) for e in saved)
    assert all(c.get("leaves") for e in saved for c in e["chunks"])


def test_apply_refuses_a_roadmap_that_changed_since_the_draft(world):
    from main.identity import content_hash
    world.mod.main(["--draft", world.file])
    edited = {**CLEAR, "chunks": [{**CLEAR["chunks"][0], "prompt": CLEAR["chunks"][0]["prompt"] + " Now."}]
              + CLEAR["chunks"][1:]}
    world.pool.write_text(json.dumps({content_hash(PROBLEM): [WHILE, VAGUE, edited]}))
    assert world.mod.main(["--apply", world.file]) == 1
    assert edited in world.saved(), "the edited roadmap is left exactly as it is"


def test_the_draft_stops_at_its_cap(world, capsys):
    assert world.mod.main(["--draft", world.file, "--max-dollars", "0.005"]) == 1
    assert "STOPPED" in capsys.readouterr().out


def test_drafted_wording_that_still_fails_the_gate_is_refused_at_apply(world, monkeypatch):
    """write_prompts keeps a working split with its best wording even when the
    gate never passed - so --apply checks again, and keeps the old roadmap."""
    import test_fix_pools as me
    monkeypatch.setattr(me, "_wording", lambda n: [
        "Write code that initializes a tally for each letter, and keep the result for "
        "the next step."] * (n - 1) + ["Write code that initializes and returns the tally."])
    world.mod.main(["--draft", world.file])
    assert world.mod.main(["--apply", world.file]) == 1
    assert VAGUE in world.saved() and WHILE in world.saved(), "nothing failing is swapped in"
