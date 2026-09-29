"""test_roadmap_pool_cost.py - a student opening a problem is only ever SERVED
a saved roadmap; the teacher's upload is what builds them, five per problem.

THE MEASUREMENT. On 25 Sep, the busiest normal day, building fresh roadmaps
was the largest normal cost: $2.50 of $6.71, 97 model calls. Every live problem
but two already had 5 or more saved, gated roadmaps on the server, yet a rule
built a new one on 40% of opens anyway and appended it - so the pool only ever
grew (invert had 10) and the cost never stopped.

THEN THE CAP ITSELF LEAKED (28 Sep). Opens topped the pool up to five, and the
"fewer than five?" check ran BEFORE a build that takes seconds, with nothing
locking the file: 30 students opening a new problem together paid for 30 builds
and the pool reached 2 - each save overwrote the others'. So building moved to
upload (run_phase1.fill_pool) and opening became serve-only.

The fakes are the network edges only: the oracle (a fixed, certified test set)
and the fresh generator (decompose_into_chunks, the one function here that
calls the model). The choosing, the filling, the upload pipeline and the serve
gate every served roadmap goes through are the real code.
"""
import json
import random
import sys
import threading
import time
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

    def write(entries):
        path.write_text(json.dumps({identity.content_hash(PROBLEM): entries}))

    def saved(n):
        write([ROADMAP] * n)

    def open_problem():
        return run_phase1.get_chunk_decomposition(dict(PROBLEM))

    def entries():
        return run_phase1._load_pool().get(identity.content_hash(PROBLEM), [])

    def count():
        return len(entries())
    return types.SimpleNamespace(saved=saved, write=write, open=open_problem,
                                 built=built, entries=entries, count=count)


def test_a_full_pool_is_served_without_building_a_new_roadmap(pool):
    pool.saved(5)
    for _ in range(20):
        assert [c.step_id for c in pool.open()["chunks"]] == ["Part 1", "Part 2"]
    assert pool.built == [], f"built {len(pool.built)} roadmaps for a problem that had 5"
    assert pool.count() == 5, "the pool must stop growing once it is full"


def test_a_short_pool_is_served_and_never_topped_up_by_a_student(pool):
    pool.saved(3)
    for _ in range(10):
        pool.open()
    assert pool.built == [] and pool.count() == 3


def test_a_problem_with_no_saved_roadmap_is_refused_not_built(pool):
    from main.run_phase1 import DecompositionUnavailableError
    pool.saved(0)
    with pytest.raises(DecompositionUnavailableError):
        pool.open()
    assert pool.built == []


def test_students_opening_together_build_nothing(pool, monkeypatch):
    """The leak, reproduced: a build takes seconds, and every open that started
    during it saw "fewer than five" too."""
    from main import run_phase1

    def slow(problem, *_a, **_k):
        pool.built.append(problem["slug"])
        time.sleep(0.3)
        return run_phase1._deserialize(ROADMAP)
    monkeypatch.setattr(run_phase1, "decompose_into_chunks", slow)
    pool.saved(1)
    results = []

    def student():
        try:
            pool.open()
            results.append("served")
        except Exception as e:
            results.append(type(e).__name__)
    threads = [threading.Thread(target=student) for _ in range(30)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    assert results == ["served"] * 30, results
    assert pool.built == [], f"students paid for {len(pool.built)} builds"
    assert pool.count() == 1


def test_an_upload_saves_five_roadmaps_and_preparing_again_adds_none(pool):
    from main import publish
    pool.saved(0)
    assert publish.prepare_problem(dict(PROBLEM))["ready"] is True
    assert len(pool.built) == 5 and pool.count() == 5
    # Retry / accept run the same pipeline: a full pool is final.
    assert publish.prepare_problem(dict(PROBLEM))["ready"] is True
    assert len(pool.built) == 5 and pool.count() == 5


def test_an_upload_stops_at_the_first_failed_build(pool, monkeypatch):
    """The money bound. A problem the model cannot split this time costs one
    failed build and one best-effort try - what an upload cost before - not a
    retry per missing roadmap."""
    from main import publish, run_phase1
    tried = []

    def fails(problem, *_a, **_k):
        tried.append("build")
        raise RuntimeError("every try failed a gate")

    def best_fails(problem, *_a, **_k):
        tried.append("best")
        raise run_phase1.DecompositionUnavailableError("nothing safe to serve")
    monkeypatch.setattr(run_phase1, "decompose_into_chunks", fails)
    monkeypatch.setattr(run_phase1, "decompose_into_chunks_best", best_fails)
    # Since 28 Sep a failed build is followed by cuts of the teacher's own code
    # (main/splitter.py, no model build - see test_splitter.py). This pins the
    # bound for the case where even that finds nothing to serve.
    monkeypatch.setattr(run_phase1.splitter, "plan", lambda *a, **k: [])
    pool.saved(0)

    result = publish.prepare_problem(dict(PROBLEM))

    assert result["ready"] is False and result["stage"] == "steps"
    assert tried == ["build", "best"], tried


def test_reprepare_swaps_new_roadmaps_in_with_no_gap_for_students(pool, monkeypatch, tmp_path):
    """Reprepare used to DROP every problem's roadmaps before rebuilding any, and
    students filled the gap by building on open. Opening is serve-only now, so a
    gap would be an error page for as long as the whole assignment takes to
    re-prepare. The old roadmaps are served until the new ones exist."""
    from frontend import api_server
    from main import run_phase1
    from test_auth_routes import PW, client, register
    monkeypatch.setenv("MICROTUTOR_TRANSCRIPTS", str(tmp_path / "tr.json"))
    c = client()
    sb = api_server.get_supabase()
    register(c, "prof@psu.edu", PW)
    sb.students[0]["role"] = "teacher"
    c.post("/logout")
    c.post("/login", json={"username": "prof@psu.edu", "password": PW})
    sb.problems.append({**PROBLEM, "assignment_id": "a-1", "ready": True,
                        "prepare_error": None})
    old = {**ROADMAP, "chunks": [{**ch, "prompt": "OLD " + ch["prompt"]}
                                 for ch in ROADMAP["chunks"]]}
    pool.write([old] * 5)

    during = []

    def fresh(problem, *_a, **_k):
        if not during:                  # a student opens it mid-rebuild, once
            during.append("?")
            try:
                during[0] = pool.open()["chunks"][0].prompt
            except Exception as e:
                during[0] = type(e).__name__
        pool.built.append(problem["slug"])
        return run_phase1._deserialize(ROADMAP)
    monkeypatch.setattr(run_phase1, "decompose_into_chunks", fresh)

    r = c.post("/teacher/assignments/a-1/reprepare")

    assert r.status_code == 200
    done = [json.loads(ln) for ln in r.text.splitlines() if ln.strip()][-1]
    assert done.get("event") == "done" and done.get("ready") == 1, done
    assert during[0].startswith("OLD"), f"a student mid-rebuild got: {during[0]}"
    prompts = [e["chunks"][0]["prompt"] for e in pool.entries()]
    assert len(prompts) == 5 and not any(p.startswith("OLD") for p in prompts), prompts


def test_two_uploads_saving_at_once_keep_both_sets(pool, monkeypatch):
    """One file holds every problem's roadmaps, and a save is load-modify-save.
    Two unlocked writers keep only the last save. The read is slowed down here
    so the two saves always overlap, instead of only on an unlucky day."""
    from main import identity, run_phase1
    other = {**PROBLEM, "slug": "freq-other", "description": "Count characters."}
    real_load = run_phase1._load_pool

    def slow_load():
        got = real_load()
        time.sleep(0.2)
        return got
    monkeypatch.setattr(run_phase1, "_load_pool", slow_load)
    pool.saved(0)

    threads = [threading.Thread(target=run_phase1.fill_pool, args=(dict(p),))
               for p in (PROBLEM, other)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    saved = real_load()
    assert [len(saved.get(identity.content_hash(p), [])) for p in (PROBLEM, other)] == [5, 5]


def test_a_pool_write_that_dies_part_way_keeps_every_saved_roadmap(pool, monkeypatch):
    """Saving opened the file for writing - emptying it - and then wrote. A
    failure in between left a truncated file that loads as {}: every problem's
    roadmaps gone at once, and students are only ever served saved ones."""
    from main import run_phase1
    pool.saved(5)
    real_dump = json.dump

    def dies_part_way(obj, f, **_k):
        f.write('{"trunc')
        raise OSError("disk full")
    monkeypatch.setattr(json, "dump", dies_part_way)
    run_phase1._save_pool({"some-other-problem": []})
    monkeypatch.setattr(json, "dump", real_dump)

    assert pool.count() == 5
