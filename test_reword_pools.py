"""test_reword_pools.py - saved steps that copied the wording prompt's example
sentence get new wording; exactly the wording that was read is what ships.

THE MEASUREMENT (server, 30 Sep). prompts.SPLIT_PROMPTS_SYSTEM quoted one
example setup step - "Write code that gets everything ready to work through the
statements, and keep it for the next step." - and the model copied it into 6 of
200 live steps: all 5 calculateExpressions roadmaps and 1 of get-postfix's 3,
one of them a 19-line step that ends in `return report`.

REAL: the pool file, splitter.write_prompts, the prompt gate, the spend cap
(it wraps the client's own HTTP call), the serve gate and grading. FAKED:
Supabase (the problem rows) and OpenAI's HTTP reply, billed like the real one.
"""
import json
import re
import types

import pytest
import requests

from test_auth_routes import FakeSupabase
from test_restart_reroute import ORACLE, PROBLEM
from test_student_open_never_generates import ROADMAP

COPIED = "Write code that gets everything ready to work through the statements, " \
         "and keep it for the next step."
VAGUE = {**ROADMAP, "chunks": [{**ROADMAP["chunks"][0], "prompt": COPIED},
                               ROADMAP["chunks"][1]]}
DRAFTED = ["Write code that works out how many times each letter appears in the "
           "text, and keep the result for the next step.",
           "Write code that returns how many times each letter appears."]


@pytest.fixture
def world(tmp_path, monkeypatch):
    from main import identity, reword_pools, run_phase1
    from main.identity import content_hash

    cache, pool = tmp_path / "o.json", tmp_path / "pool.json"
    monkeypatch.setenv("MICROTUTOR_ORACLE_CACHE", str(cache))
    monkeypatch.setenv("MICROTUTOR_TRACE_FILE", str(tmp_path / "t.jsonl"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-never-sent")
    monkeypatch.setattr(identity, "_RESOLVED_PATH", str(tmp_path / "r.json"))
    monkeypatch.setattr(run_phase1, "_CHUNK_POOL_PATH", str(pool))
    cache.write_text(json.dumps({content_hash(PROBLEM): {
        "strong": True, "status": "strong", "kill_rate": 1.0, "kill_rate_direct": 1.0,
        "features": ["calls"], "final_tests": ORACLE}}))
    pool.write_text(json.dumps({content_hash(PROBLEM): [VAGUE, ROADMAP]}))
    sb = FakeSupabase()
    sb.problems.append({**PROBLEM, "ready": True})
    monkeypatch.setattr(reword_pools, "_sb", lambda: sb)

    calls, wording = [], {"prompts": list(DRAFTED)}

    class Reply:
        status_code, headers, text = 200, {}, ""

        def __init__(self, body):
            self.body = body

        def raise_for_status(self):
            pass

        def json(self):
            return self.body

    def network(url, json=None, **_k):
        usage = {"prompt_tokens": 1500, "completion_tokens": 200}
        calls.append(usage)
        return Reply({"model": json["model"], "usage": usage,
                      "choices": [{"message": {"content": __import__("json").dumps(wording)}}]})
    monkeypatch.setattr(requests, "post", network)

    def saved():
        return run_phase1._load_pool()[content_hash(PROBLEM)]
    return types.SimpleNamespace(calls=calls, wording=wording, saved=saved,
                                 file=str(tmp_path / "reword.json"), pool=pool,
                                 run=reword_pools.main)


def test_listing_finds_only_the_copied_sentence_and_spends_nothing(world, capsys):
    before = world.pool.read_text()
    assert world.run([]) == 0
    out = capsys.readouterr().out
    assert "1 roadmap(s)" in out and "gets everything ready" in out
    assert world.calls == [] and world.pool.read_text() == before


def test_a_draft_shows_old_and_new_wording_and_changes_nothing(world, capsys):
    before = world.pool.read_text()
    assert world.run(["--draft", world.file]) == 0
    drafted = json.load(open(world.file))
    assert len(drafted) == 1 and drafted[0]["new"] == DRAFTED, drafted
    assert drafted[0]["old"][0] == COPIED and drafted[0]["gate"] == "pass"
    assert DRAFTED[0] in capsys.readouterr().out
    assert len(world.calls) == 1 and world.pool.read_text() == before


def test_apply_swaps_in_exactly_the_drafted_wording_with_no_model_call(world, capsys):
    world.run(["--draft", world.file])
    world.wording["prompts"] = ["Something the reviewers never read.", "Nor this."]
    calls = len(world.calls)

    assert world.run(["--apply", world.file]) == 0
    vague, clear = world.saved()
    assert [c["prompt"] for c in vague["chunks"]] == DRAFTED
    assert [c["reference"] for c in vague["chunks"]] == \
        [c["reference"] for c in ROADMAP["chunks"]], "the code must not change"
    assert clear == ROADMAP, "a roadmap that did not copy the sentence is untouched"
    assert len(world.calls) == calls, "apply must never call the model"
    assert "every roadmap is ready" in capsys.readouterr().out


def test_apply_skips_a_roadmap_whose_code_changed_since_the_draft(world):
    world.run(["--draft", world.file])
    pool = json.loads(world.pool.read_text())
    key = next(iter(pool))
    pool[key][0]["chunks"][1]["reference"] = "return dict(counts)"
    world.pool.write_text(json.dumps(pool))

    assert world.run(["--apply", world.file]) == 1
    assert world.saved()[0]["chunks"][0]["prompt"] == COPIED


def test_apply_refuses_wording_the_prompt_gate_rejects(world):
    world.run(["--draft", world.file])
    drafted = json.load(open(world.file))
    drafted[0]["new"][0] = "Initialize counts to an empty dictionary."
    json.dump(drafted, open(world.file, "w"))

    assert world.run(["--apply", world.file]) == 1
    assert world.saved()[0]["chunks"][0]["prompt"] == COPIED


def test_the_draft_stops_at_its_hard_cap(world, capsys):
    assert world.run(["--draft", world.file, "--max-dollars", "0.001"]) == 1
    assert world.calls == [] and "STOPPED" in capsys.readouterr().out
    assert json.load(open(world.file)) == []


def test_readiness_catches_a_step_the_teachers_code_cannot_pass(world):
    from main import reword_pools
    broken = {**ROADMAP, "chunks": [ROADMAP["chunks"][0],
                                    {**ROADMAP["chunks"][1], "reference": "return count"}]}
    assert reword_pools.readiness(dict(PROBLEM), broken)
    assert reword_pools.readiness(dict(PROBLEM), ROADMAP) == []


def test_the_wording_prompt_has_no_setup_sentence_to_copy():
    from main.prompts import SPLIT_PROMPTS_SYSTEM
    assert "gets everything ready" not in re.sub(r"\s+", " ", SPLIT_PROMPTS_SYSTEM)
