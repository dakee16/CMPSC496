"""Offline API/integration tests. No credentials, no network, no LLM."""
import json
import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SOLUTION = ("class Solution:\n"
            "    def add(self, a, b):\n"
            "        total = a + b\n"
            "        return total\n")
PROBLEM = {"slug": "adder", "title": "Adder", "description": "Return a + b.",
           "solution": SOLUTION, "difficulty": "Easy"}
TESTS = [{"input": [1, 2], "expected": 3}, {"input": [0, 0], "expected": 0}]
REFS = ["total = a + b", "return total"]


class FakeSupabase:
    """Minimal stand-in: enough surface for the routes we exercise."""
    def __init__(self): self.writes = []
    def table(self, name): self._t = name; return self
    def select(self, *a, **k): return self
    def eq(self, *a, **k): return self
    def limit(self, *a, **k): return self
    def single(self): return self
    def insert(self, row): self.writes.append((self._t, "insert", row)); return self
    def upsert(self, row, **k): self.writes.append((self._t, "upsert", row)); return self
    def execute(self):
        if self._t == "problems":
            return type("R", (), {"data": [PROBLEM]})()
        return type("R", (), {"data": []})()


@pytest.fixture
def api(tmp_path, monkeypatch):
    os.environ["MICROTUTOR_SESSION_DB"] = str(tmp_path / "s.sqlite3")
    import tests.sandbox as sb, main.identity as ident
    cache = tmp_path / "cache.json"
    monkeypatch.setattr(sb, "_CACHE_PATH", str(cache))
    monkeypatch.setattr(ident, "_RESOLVED_PATH", str(tmp_path / "r.json"))
    cache.write_text(json.dumps({ident.content_hash(PROBLEM): {
        "final_tests": TESTS, "strong": True, "kill_rate": 1.0,
        "kill_rate_direct": 1.0, "validated_at": "2026-01-01T00:00:00+00:00",
        "slug": "adder"}}))

    import frontend.api_server as srv
    from fastapi.testclient import TestClient
    from main.schemas import StepItem
    fake = FakeSupabase()
    srv.set_supabase(fake)
    # Decomposition is gated elsewhere; here we inject a known-good one so the
    # API flow is exercised without an LLM.
    chunks = [StepItem(question_id="adder", step_id=f"Part {i+1}", prompt=f"step {i+1}",
                       expected_type="code", reference=r) for i, r in enumerate(REFS)]
    monkeypatch.setattr(srv, "get_chunk_decomposition",
                        lambda p: {"header": "def add(a, b):", "chunks": chunks})
    monkeypatch.setattr(srv, "get_oracle_tests", lambda p: TESTS, raising=False)
    # No network LLM anywhere: no adapter, and a judge that says "not quite".
    import main.grading as g
    monkeypatch.setattr(g, "_request_adaptation", lambda *a, **k: ("", []))
    monkeypatch.setattr(g, "_ask_judge", lambda p, r: (False, "Not quite.", 0.9, "sem"))
    return TestClient(srv.app), srv, fake


def _start(api, student=None):
    client, srv, _ = api
    body = {"slug": "adder", "title": "Adder", "description": "Return a + b.",
            "solution": SOLUTION}
    if student:
        body["student_id"] = student
    r = client.post("/decompose_chunks", json=body)
    assert r.status_code == 200, r.text
    return r.json()


# ── import purity ────────────────────────────────────────────────────────
def test_import_creates_no_client():
    import importlib
    import frontend.api_server as srv
    importlib.reload(srv)
    assert srv._SB is None          # no network/client work at import time


# ── full flow ────────────────────────────────────────────────────────────
def test_decompose_exposes_nothing_private(api):
    data = _start(api)
    blob = json.dumps(data)
    # No secret VALUES, and no forbidden KEYS. ("expected_type" is public, so
    # a naive "expected" substring check would be a false alarm.)
    assert "total = a + b" not in blob and "return total" not in blob
    assert "class Solution" not in blob
    for forbidden in ("reference", "solution", "final_tests", "oracle"):
        assert f'"{forbidden}"' not in blob
    assert set(data["chunks"][0]) == {"step_id", "prompt", "expected_type"}
    assert data["session_id"] and data["decomposition_id"]


def test_full_flow_to_independent_completion(api):
    client, _, fake = api
    d = _start(api, student="stu-1")
    sid = d["session_id"]
    r1 = client.post("/grade_chunk", json={"session_id": sid, "submission_id": "a1",
                                           "student_code": "total = a + b"}).json()
    assert r1["verdict"] == "correct" and r1["index"] == 1 and r1["deterministic"]
    r2 = client.post("/grade_chunk", json={"session_id": sid, "submission_id": "a2",
                                           "student_code": "return total"}).json()
    assert r2["completed"] is True and r2["assisted"] is False
    assert r2["solved_independently"] is True
    m = client.post("/mark_solved", json={"session_id": sid}).json()
    assert m["recorded"] is True and m["solved_independently"] is True
    assert any(w[0] == "solved" for w in fake.writes)


def test_idempotent_retry(api):
    client, _, _ = api
    sid = _start(api)["session_id"]
    p = {"session_id": sid, "submission_id": "same", "student_code": "total = a + b"}
    a = client.post("/grade_chunk", json=p).json()
    b = client.post("/grade_chunk", json=p).json()
    assert a["index"] == b["index"] == 1
    assert b["idempotent_replay"] is True


def test_reveal_then_assisted_completion(api):
    client, _, fake = api
    d = _start(api, student="stu-2")
    sid = d["session_id"]
    r1 = client.post("/grade_chunk", json={"session_id": sid, "submission_id": "b1",
                                           "student_code": "total = 999"}).json()
    assert r1["verdict"] == "incorrect" and "revealed_reference" not in r1
    r2 = client.post("/grade_chunk", json={"session_id": sid, "submission_id": "b2",
                                           "student_code": "total = 999"}).json()
    assert r2["revealed_reference"] == "total = a + b"     # only at the limit
    assert r2["assisted"] is True and r2["index"] == 1
    r3 = client.post("/grade_chunk", json={"session_id": sid, "submission_id": "b3",
                                           "student_code": "return total"}).json()
    assert r3["completed"] and r3["solved_independently"] is False
    m = client.post("/mark_solved", json={"session_id": sid}).json()
    assert m["recorded"] is False and m["assisted"] is True
    assert not any(w[0] == "solved" for w in fake.writes)


def test_stale_index_conflicts(api):
    client, _, _ = api
    sid = _start(api)["session_id"]
    r = client.post("/grade_chunk", json={"session_id": sid, "submission_id": "z",
                                          "student_code": "total = a + b",
                                          "expected_index": 5})
    assert r.status_code == 409


def test_unknown_session_404(api):
    client, _, _ = api
    r = client.post("/grade_chunk", json={"session_id": "forged", "submission_id": "x",
                                          "student_code": "total = a + b"})
    assert r.status_code == 404


def test_student_id_not_accepted_on_grade(api):
    """Identity is bound at session start and cannot be swapped mid-session."""
    client, _, fake = api
    sid = _start(api, student="stu-1")["session_id"]
    r = client.post("/grade_chunk", json={"session_id": sid, "submission_id": "c1",
                                          "student_code": "total = a + b",
                                          "student_id": "attacker"})
    assert r.status_code == 422        # forged identity is rejected, not ignored


def test_no_private_fields_in_grade_response(api):
    client, _, _ = api
    sid = _start(api)["session_id"]
    r = client.post("/grade_chunk", json={"session_id": sid, "submission_id": "d1",
                                          "student_code": "total = 42"}).json()
    blob = json.dumps(r)
    assert "expected" not in blob and "failures" not in blob
    assert "total = a + b" not in blob        # reference not leaked pre-reveal


def test_replan_disabled(api):
    client, _, _ = api
    r = client.post("/replan", json={"slug": "adder", "description": "d",
                                     "accepted_steps": []})
    assert r.status_code == 410
    assert r.json()["detail"]["reason_code"] == "replan_disabled"


# ── concurrency ──────────────────────────────────────────────────────────
def _session_only(tmp_env):
    from main.identity import content_hash
    from main.schemas import StepItem
    from main.sessions import create_session
    chunks = [StepItem(question_id="adder", step_id=f"Part {i+1}", prompt="p",
                       expected_type="code", reference=r) for i, r in enumerate(REFS)]
    return create_session(PROBLEM, {"header": "def add(a, b):", "chunks": chunks},
                          content_hash(PROBLEM))["session_id"]


def test_concurrent_same_submission_id(api):
    """Test A: same session + same submission id from two threads."""
    from main.sessions import begin_submission, commit_outcome, load_session
    sid = _session_only(api)
    out, errs = [], []

    def work():
        try:
            done, s = begin_submission(sid, "dup")
            if done is not None:
                out.append(("replay", done)); return
            time.sleep(0.05)                      # grading happens here
            out.append(("commit", commit_outcome(sid, "dup", s["revision"],
                                                 {"verdict": "correct"},
                                                 accept_code="total = a + b")))
        except Exception as e:
            errs.append(e)

    ts = [threading.Thread(target=work) for _ in range(2)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert not errs, errs
    assert sum(1 for k, _ in out if k == "commit") == 1     # graded once
    assert load_session(sid)["index"] == 1                  # advanced once


def test_concurrent_different_submission_ids(api):
    """Test B: two different submissions racing the same session revision."""
    from main.sessions import (SessionError, begin_submission, commit_outcome,
                               load_session)
    sid = _session_only(api)
    ok, conflict = [], []

    def work(tag):
        done, s = begin_submission(sid, tag)
        time.sleep(0.05)
        try:
            ok.append(commit_outcome(sid, tag, s["revision"], {"verdict": "correct"},
                                     accept_code="total = a + b"))
        except SessionError as e:
            conflict.append(e.reason_code)

    ts = [threading.Thread(target=work, args=(t,)) for t in ("t1", "t2")]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert len(ok) == 1 and conflict == ["session_conflict"]
    assert load_session(sid)["index"] == 1                  # no skipped chunk


def test_no_write_txn_held_during_slow_grade(api):
    """A slow grade must not block another session's write."""
    from main.sessions import begin_submission, commit_outcome
    sid_a, sid_b = _session_only(api), _session_only(api)
    done, sa = begin_submission(sid_a, "slow")
    assert done is None
    t0 = time.time()
    commit_outcome(sid_b, "fast", 0, {"verdict": "correct"},
                   accept_code="total = a + b")     # must not wait on A
    assert time.time() - t0 < 2.0
    commit_outcome(sid_a, "slow", sa["revision"], {"verdict": "correct"},
                   accept_code="total = a + b")


# ── parity ───────────────────────────────────────────────────────────────
def test_api_and_research_agree(api, monkeypatch):
    """Same input, same engine: identical verdict and tier."""
    from main.grading import grade_submission
    from main.schemas import StepItem
    from main.sessions import load_session
    from tests.grader import grade_chunk
    sid = _session_only(api)
    sess = load_session(sid)
    code = "total = a + b"
    engine = grade_submission(sess, code)

    chunks = [StepItem(question_id="adder", step_id=f"Part {i+1}", prompt="p",
                       expected_type="code", reference=r) for i, r in enumerate(REFS)]
    research = grade_chunk(PROBLEM, chunks, 0, code, [])
    assert research["verdict"] == engine.verdict
    assert research["tier"] == engine.tier
    assert research["deterministic"] == engine.deterministic


# ── Tier 3 attribution: NON-FINAL chunk, real calibrated adapter ─────────
# 3 chunks so index 1 is genuinely non-final. The reference tail (`return total`)
# fails against a divergent student interface, which is what forces Tier 3.
REFS3 = ["total = a", "total = total + b", "return total"]
ADAPTER = ("return acc", [{"target": "acc", "source": "total"}])


def _sess3(prefix_code, provenance):
    from main.identity import content_hash
    from main.schemas import StepItem
    from main.sessions import apply_outcome, create_session, load_session
    chunks = [StepItem(question_id="adder", step_id=f"Part {i+1}", prompt="p",
                       expected_type="code", reference=r)
              for i, r in enumerate(REFS3)]
    sid = create_session(PROBLEM, {"header": "def add(a, b):", "chunks": chunks},
                         content_hash(PROBLEM))["session_id"]
    apply_outcome(sid, "pfx", {"verdict": "correct"},
                  accept_code=prefix_code, provenance=provenance)
    return load_session(sid)


@pytest.mark.parametrize("prefix,prov,good,bad", [
    # A: prior prefix divergent but previously accepted
    ("acc = a", "student", "acc = acc + b", "acc = acc * b"),
    # B: prior prefix is a revealed reference
    ("total = a", "revealed_reference", "acc = total + b", "acc = total * b"),
])
def test_tier3_attribution_non_final(api, monkeypatch, prefix, prov, good, bad):
    import main.grading as g
    from main.grading import grade_submission
    monkeypatch.setattr(g, "_request_adaptation", lambda *a, **k: ADAPTER)
    def no_judge(*a, **k):
        raise AssertionError("Tier 4 reached — Tier 3 should have settled this")
    monkeypatch.setattr(g, "_ask_judge", no_judge)

    s = _sess3(prefix, prov)
    assert s["index"] == 1 and s["index"] < len(s["chunks"]) - 1   # NON-final

    # correct current answer -> calibrated pass + knockout fails
    ok = grade_submission(s, good)
    assert ok.tier == "execution-adapted", ok
    assert ok.verdict == "correct" and ok.deterministic is True and ok.divergent

    # wrong current answer -> calibrated clean wrong output convicts THIS chunk
    bad_r = grade_submission(s, bad)
    assert bad_r.tier == "execution-adapted", bad_r
    assert bad_r.verdict == "incorrect" and bad_r.deterministic is True


def test_tier3_uncalibratable_does_not_convict(api, monkeypatch):
    """If the adapter can't calibrate, the failure is unattributable — it must
    reach Tier 4, never a deterministic incorrect."""
    import main.grading as g
    from main.grading import grade_submission
    monkeypatch.setattr(g, "_request_adaptation",
                        lambda *a, **k: ("return 12345", []))   # never calibrates
    monkeypatch.setattr(g, "_ask_judge", lambda p, r: (False, "no", 0.9, "sem"))
    s = _sess3("acc = a", "student")
    r = grade_submission(s, "acc = acc * b")
    assert r.tier == "llm-judge" and r.deterministic is False


# ── forbidden client fields ──────────────────────────────────────────────
@pytest.mark.parametrize("extra", [
    {"student_id": "attacker"}, {"problem": {"solution": "x"}},
    {"solution": "def add(a,b): return 0"}, {"chunks": [{"reference": "x"}]},
    {"accepted_prefix": ["x"]}, {"index": 0},
])
def test_forbidden_fields_rejected(api, extra):
    client, _, _ = api
    sid = _start(api)["session_id"]
    r = client.post("/grade_chunk", json={
        "session_id": sid, "submission_id": "f1",
        "student_code": "total = a + b", **extra})
    assert r.status_code == 422, (extra, r.text)


def test_mark_solved_forbids_extras(api):
    client, _, _ = api
    sid = _start(api)["session_id"]
    r = client.post("/mark_solved", json={"session_id": sid, "student_id": "x"})
    assert r.status_code == 422


# ── retired routes ───────────────────────────────────────────────────────
def test_evaluate_disabled(api):
    client, _, _ = api
    r = client.post("/evaluate", json={"step": {}, "answer": "x", "context": ""})
    assert r.status_code == 410
    assert r.json()["detail"]["reason_code"] == "legacy_evaluate_disabled"


# ── in-flight same-id behaviour ──────────────────────────────────────────
def test_in_flight_same_id_returns_typed_conflict(api):
    """A concurrent twin of the SAME submission gets a typed 409 telling it to
    retry with the same id — never a 500, never a second grade."""
    from main.sessions import begin_submission
    client, _, _ = api
    sid = _start(api)["session_id"]
    claimed, _sess = begin_submission(sid, "inflight")   # simulate twin in progress
    assert claimed is None
    r = client.post("/grade_chunk", json={"session_id": sid,
                                          "submission_id": "inflight",
                                          "student_code": "total = a + b"})
    assert r.status_code == 409
    assert r.json()["detail"]["reason_code"] == "submission_in_progress"
