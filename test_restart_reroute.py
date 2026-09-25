"""test_restart_reroute.py - what opening a problem costs after Start over.

THE BUG THIS PINS, reported from testing and reproduced here: a student pressed
Start over on `stack` and `calculator` and both problems became unusable - the
plan panel and the chat span for ever, on every later open.

Nothing crashed. Three things lined up:

  * a restart retires the grading session, so /decompose_chunks stops RESUMING
    and takes the full decomposition path again;
  * mt_graphs is append-only, and latest_plan_graph did not read the restart
    marker - so the plan from the run they had just discarded still chose the
    roadmap, and for these problems it chose a REBUILT one;
  * reroute.build was bounded by a count (3 attempts) and not by a clock, and
    each attempt is a proposal plus a five-try decomposition: eighteen
    sequential model calls, measured, with no timeout anywhere in Caddy or
    uvicorn to cut the request off. All of it before a session exists, so
    nothing remembered it had happened and the next open paid again.

The fakes here are the network edges only - Supabase, the model, the pool.
Every decision under test (the marker, the budget, the memo) is the real code.
"""
import datetime as dt
import json
import os
import tempfile
import uuid

import pytest
from fastapi.testclient import TestClient

SOLUTION = ("def frequency(txt):\n"
            "    counts = {}\n"
            "    for ch in txt:\n"
            "        if ch.isalpha():\n"
            "            counts[ch] = counts.get(ch, 0) + 1\n"
            "    return counts\n")
SLUG = "frequency"
PROBLEM = {"slug": SLUG, "title": "Letter frequency",
           "description": "Count how many times each letter appears.",
           "solution": SOLUTION}
ORACLE = [{"input": ["aab"], "expected": {"a": 2, "b": 1}},
          {"input": ["hello world"], "expected": {"h": 1, "e": 1, "l": 3,
                                                  "o": 2, "w": 1, "r": 1,
                                                  "d": 1}},
          {"input": [""], "expected": {}}]
# No loop: the student planned to recurse, the teacher looped. That difference
# is exactly what reroute exists for, so this plan DOES ask for a rebuild.
DIVERGENT_PLAN = {"nodes": [{"id": "s", "kind": "start", "label": "take the text"},
                            {"id": "b", "kind": "branch",
                             "label": "if the text is empty, answer nothing"},
                            {"id": "r", "kind": "return",
                             "label": "otherwise add the first letter to the "
                                      "answer for the rest"}],
                  "edges": [{"src": "s", "dst": "b"}, {"src": "b", "dst": "r"}]}


class FakeTable:
    """Enough postgrest to honour eq/gt/order/limit - the marker comparison is
    the thing under test, so a fake that ignored `gt` would pass either way."""

    def __init__(self, rows):
        self.rows, self._eq, self._gt = rows, [], []
        self._order = self._limit = self._pending = self._update = None

    def select(self, *_a, **_k): return self
    def eq(self, c, v): self._eq.append((c, v)); return self
    def gt(self, c, v): self._gt.append((c, v)); return self
    def order(self, c, desc=False): self._order = (c, desc); return self
    def limit(self, n): self._limit = n; return self
    @property
    def not_(self): return self
    def is_(self, *_a, **_k): return self
    def insert(self, d): self._pending = d; return self
    def upsert(self, d, **_k): self._pending = d; return self
    def update(self, d): self._update = d; return self

    def execute(self):
        if self._update is not None:
            patch, self._update = self._update, None
            hits = list(self._match())
            for r in hits:
                r.update(patch)
            return type("R", (), {"data": hits})()
        if self._pending is not None:
            payload, self._pending = self._pending, None
            out = []
            for row in (payload if isinstance(payload, list) else [payload]):
                row = dict(row)
                row.setdefault("id", str(uuid.uuid4()))
                self.rows.append(row)
                out.append(row)
            return type("R", (), {"data": out})()
        out = list(self._match())
        if self._order:
            col, desc = self._order
            out.sort(key=lambda r: (r.get(col) or ""), reverse=desc)
        return type("R", (), {"data": out[:self._limit] if self._limit else out})()

    def _match(self):
        return (r for r in self.rows
                if all(r.get(c) == v for c, v in self._eq)
                and all((r.get(c) or "") > v for c, v in self._gt))


class FakeSupabase:
    def __init__(self): self.tables = {}
    def table(self, name): return FakeTable(self.tables.setdefault(name, []))
    def rows(self, name): return self.tables.setdefault(name, [])


def _stamp(minutes_ago=0):
    return (dt.datetime.now(dt.timezone.utc)
            - dt.timedelta(minutes=minutes_ago)).isoformat()


@pytest.fixture
def env(monkeypatch, tmp_path):
    """A signed-in student, one problem, and a counter on every model call."""
    # The session store and the entry-point cache are WRITTEN by this path, so
    # both are redirected before anything imports them (grading.py's own
    # self-check makes the same move for resolved_entries.json).
    monkeypatch.setenv("MICROTUTOR_SESSION_DB", str(tmp_path / "sessions.sqlite3"))
    import main.auth as auth
    from main import identity, ollama_client, reroute, run_phase1
    from frontend import api_server
    monkeypatch.setattr(identity, "_RESOLVED_PATH", str(tmp_path / "resolved.json"))

    calls = []

    def fake_chat(model, system, messages, **kw):
        calls.append(system.splitlines()[0][:40] if system else "")
        # The proposal is the teacher's own body, so it clears the oracle gate
        # and the run reaches the decomposer - where the junk below fails, which
        # is the expensive path this test is about.
        return json.dumps({"body": "\n".join(
            ln[4:] for ln in SOLUTION.splitlines()[1:] if ln.strip())})

    for mod in (ollama_client, reroute, run_phase1):
        monkeypatch.setattr(mod, "chat", fake_chat, raising=False)
    # The last line of defence: anything that still reaches the real provider
    # fails loudly here instead of spending credits - or, with none left,
    # failing as if the code under test were broken. That is how this suite
    # was found to be online: tests.sandbox generates oracle inputs through
    # its OWN imported copy of chat, which none of the patches above reach.
    def no_network(*a, **k):
        raise AssertionError("a test reached the real model provider")
    monkeypatch.setattr(ollama_client, "_openai_chat", no_network)
    monkeypatch.setattr(ollama_client, "_ollama_chat", no_network)
    import tests.sandbox as sandbox
    # EVERY copy of the name: run_phase1 and gates import get_oracle_tests at
    # load time, so patching sandbox alone left the decomposer's gates
    # generating a real oracle.
    import sys
    # The necessity gate also asks whether that oracle was mutation-validated,
    # which is a lookup in the real cache: ORACLE stands in for a validated one.
    fakes = {"get_oracle_tests": lambda p, **k: list(ORACLE),
             "is_oracle_certified": lambda p: True}
    for name, fake in fakes.items():
        real = getattr(sandbox, name)
        for mod in list(sys.modules.values()):
            if getattr(mod, name, None) is real:
                monkeypatch.setattr(mod, name, fake)
    # The teacher's own roadmap is not what is being measured: make the
    # fallback instant so every model call counted below belongs to reroute.
    monkeypatch.setattr(api_server, "get_chunk_decomposition", lambda p: {
        "header": "def frequency(txt):",
        "chunks": [type("C", (), {"step_id": "Part 1", "prompt": "count them",
                                  "expected_type": "code",
                                  "reference": "counts = {}"})(),
                   type("C", (), {"step_id": "Part 2", "prompt": "hand it back",
                                  "expected_type": "code",
                                  "reference": "return counts"})()]})
    monkeypatch.setattr(api_server, "_slug_published", lambda slug: True)
    monkeypatch.setattr(run_phase1, "load_problems", lambda limit=500: [dict(PROBLEM)])
    reroute._NO_ROUTE.clear()

    auth.SESSION_SECRET = "test-secret-not-a-real-key"
    auth.ALLOWED_DOMAINS = ("psu.edu",)
    auth.ALLOWED_EMAILS = frozenset()
    auth._failures.clear()
    sb = FakeSupabase()
    api_server.set_supabase(sb)
    sb.rows("problems").append(dict(PROBLEM))

    c = TestClient(api_server.app)
    c.post("/register", json={"username": "abc123@psu.edu",
                              "password": "a-good-enough-password",
                              "first_name": "Test", "last_name": "Student"})
    student = c.get("/auth/me").json()["student_id"]
    return type("Env", (), {"client": c, "sb": sb, "student": student,
                            "calls": calls, "reroute": reroute})()


def _open(env):
    calls_before = len(env.calls)
    r = env.client.post("/decompose_chunks", json={
        "slug": SLUG, "title": PROBLEM["title"],
        "description": PROBLEM["description"]})
    assert r.status_code == 200, r.text
    return r.json(), len(env.calls) - calls_before


def _plan(env, at):
    env.sb.rows("mt_graphs").append(
        {"student_id": env.student, "slug": SLUG, "kind": "plan",
         "created_at": at, "graph": DIVERGENT_PLAN})


def test_a_plan_from_before_the_restart_does_not_choose_the_roadmap(env):
    """THE REPORTED BUG. Start over promises an empty plan; the archive keeps
    the old one, and it was still picking - and paying for - a rebuilt roadmap
    on every open, for ever."""
    _plan(env, _stamp(minutes_ago=10))
    _open(env)                                   # the run they then discarded
    assert env.client.post(f"/problems/{SLUG}/restart").status_code == 200

    for attempt in range(3):                     # ...and it stays free
        _, spent = _open(env)
        assert spent == 0, f"open #{attempt + 1} after Start over cost {spent} model calls"


def test_a_plan_from_this_run_is_still_rerouted_and_is_bounded(env):
    """The feature is intact: a CURRENT divergent plan does ask for a rebuild.
    What changed is the ceiling - it was eighteen sequential model calls."""
    assert env.client.post(f"/problems/{SLUG}/restart").status_code == 200
    _plan(env, _stamp())                         # planned again, after restarting

    _, spent = _open(env)
    ceiling = 3 * (1 + env.reroute.DECOMPOSE_TRIES)
    assert spent > 0, "a current divergent plan should still be rerouted"
    assert spent <= ceiling, f"{spent} model calls on one open (ceiling {ceiling})"


def test_a_failed_reroute_is_not_paid_for_twice(env):
    """All of the cost lands before a session exists, so nothing downstream
    remembered the attempt. Every reopen used to repeat it."""
    _plan(env, _stamp())
    _, first = _open(env)
    assert first > 0
    env.client.post(f"/problems/{SLUG}/restart")
    _plan(env, _stamp())                         # same problem, same plan shape
    _, second = _open(env)
    assert second == 0, f"the same failure was recomputed ({second} calls)"


def test_the_budget_ends_it_without_asking_a_model(env):
    """The clock, not the count. With none left, no attempt starts at all."""
    from main import reroute
    with pytest.raises(reroute.RouteUnavailable) as e:
        reroute.build(dict(PROBLEM), "def frequency(txt):", DIVERGENT_PLAN,
                      budget_seconds=0)
    assert "ran out of time" in e.value.reason
    assert env.calls == [], "the budget was spent on a model call anyway"


def test_our_own_outage_is_not_remembered_as_this_plan_failing(env):
    """A provider failure says nothing about whether this approach can be cut
    into steps - so it must not pin the student to the teacher's roadmap for
    the life of the process."""
    from main import ollama_client, reroute, run_phase1

    def down(*_a, **_k):
        raise RuntimeError("provider unreachable")
    for mod in (ollama_client, reroute, run_phase1):
        setattr(mod, "chat", down)
    with pytest.raises(reroute.RouteUnavailable):
        reroute.build(dict(PROBLEM), "def frequency(txt):", DIVERGENT_PLAN)
    assert reroute._NO_ROUTE == {}, "an outage was remembered as a dead end"
