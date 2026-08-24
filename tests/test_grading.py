"""Offline grading tests. No Supabase, no OpenAI, no Ollama — every LLM call
is monkeypatched and every file is a tmp_path fixture."""
import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SOLUTION = ("class Solution:\n"
            "    def add(self, a, b):\n"
            "        total = a + b\n"
            "        return total\n")
PROBLEM = {"slug": "adder", "title": "Adder", "description": "Return a + b.",
           "solution": SOLUTION}
TESTS = [{"input": [1, 2], "expected": 3}, {"input": [0, 0], "expected": 0},
         {"input": [-1, 5], "expected": 4}]
CHUNK_REFS = ["total = a + b", "return total"]


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Temp session DB + temp oracle cache. Nothing real is touched."""
    os.environ["MICROTUTOR_SESSION_DB"] = str(tmp_path / "s.sqlite3")
    import tests.sandbox as sb
    import main.identity as ident
    cache_file = tmp_path / "cache.json"
    monkeypatch.setattr(sb, "_CACHE_PATH", str(cache_file))
    monkeypatch.setattr(ident, "_RESOLVED_PATH", str(tmp_path / "resolved.json"))
    entry = {"final_tests": TESTS, "strong": True, "kill_rate": 1.0,
             "kill_rate_direct": 1.0, "validated_at": "2026-01-01T00:00:00+00:00",
             "slug": "adder"}
    cache_file.write_text(json.dumps({ident.content_hash(PROBLEM): entry}))
    return tmp_path


def _session(env, refs=CHUNK_REFS):
    from main.identity import content_hash
    from main.schemas import StepItem
    from main.sessions import create_session, load_session
    chunks = [StepItem(question_id="adder", step_id=f"Part {i+1}",
                       prompt=f"step {i+1}", expected_type="code", reference=r)
              for i, r in enumerate(refs)]
    pub = create_session(PROBLEM, {"header": "def add(a, b):", "chunks": chunks},
                         content_hash(PROBLEM))
    return pub, load_session(pub["session_id"])


def _no_llm(monkeypatch):
    """Any model call is a test failure on deterministic paths."""
    import main.grading as g
    def boom(*a, **k): raise AssertionError("LLM called on a deterministic path")
    monkeypatch.setattr(g, "chat", boom)


# 1 — missing/weak oracle blocks and consumes nothing
def test_missing_oracle_blocks(env, monkeypatch):
    from main.grading import grade_submission
    _, s = _session(env)
    other = {**PROBLEM, "solution": SOLUTION + "\n# different"}
    s = {**s, "solution": other["solution"]}
    r = grade_submission(s, "total = a + b")
    assert r.verdict == "indeterminate" and r.tier == "system"
    assert r.consume_attempt is False and r.reason_code == "oracle_missing"


def test_weak_oracle_blocks(env, monkeypatch):
    import json as j
    import tests.sandbox as sb
    from main.grading import grade_submission
    from main.identity import content_hash
    c = j.loads(open(sb._CACHE_PATH).read())
    c[content_hash(PROBLEM)]["strong"] = False
    open(sb._CACHE_PATH, "w").write(j.dumps(c))
    _, s = _session(env)
    r = grade_submission(s, "total = a + b")
    assert r.verdict == "indeterminate" and r.reason_code == "oracle_weak"
    assert r.consume_attempt is False


# 2 — blank / syntax / unsafe are deterministic incorrect
@pytest.mark.parametrize("code,tier,rc", [
    ("", "syntax", "blank_answer"),
    ("total = (a +", "syntax", "syntax_error"),
    ("import os\ntotal = a + b", "policy", "policy_violation"),
    ("total = eval('a+b')", "policy", "policy_violation"),
])
def test_tier1_deterministic(env, monkeypatch, code, tier, rc):
    from main.grading import grade_submission
    _no_llm(monkeypatch)
    _, s = _session(env)
    r = grade_submission(s, code)
    assert (r.verdict, r.tier, r.reason_code) == ("incorrect", tier, rc)
    assert r.deterministic is True


# 3 — infinite loop is contained (not a hang, not a harness error)
def test_infinite_loop_contained(env, monkeypatch):
    from main.grading import grade_submission
    _no_llm(monkeypatch)
    _, s = _session(env, refs=["total = a + b"])       # single => last chunk
    r = grade_submission(s, "while True:\n    pass\ntotal = a + b")
    assert r.verdict == "incorrect" and r.execution_outcome == "timeout"
    assert r.deterministic is True


def test_output_flood_contained(env, monkeypatch):
    from main.grading import grade_submission
    _no_llm(monkeypatch)
    _, s = _session(env, refs=["total = a + b"])
    r = grade_submission(s, "for _ in range(20000):\n    print('x'*200)\ntotal = a + b")
    # Containment is the point: the parent survived and we reached a
    # DETERMINISTIC verdict (either it finished, or it burned its CPU budget).
    # What must never happen is a harness error or an indeterminate.
    assert r.verdict in ("correct", "incorrect") and r.deterministic is True
    assert r.reason_code != "harness_error"


# 4 — non-final correct via reference tail, no LLM
def test_reference_tail_pass(env, monkeypatch):
    from main.grading import grade_submission
    _no_llm(monkeypatch)
    _, s = _session(env)
    r = grade_submission(s, "total = a + b")
    assert (r.verdict, r.tier) == ("correct", "execution-reference")
    assert r.deterministic is True


# 9 — final chunk pass / wrong / crash
@pytest.mark.parametrize("code,verdict,outcome", [
    ("total = a + b\nreturn total", "correct", "pass"),
    ("return 999", "incorrect", "wrong_output"),
    ("return 1/0", "incorrect", "runtime_error"),
])
def test_final_chunk(env, monkeypatch, code, verdict, outcome):
    from main.grading import grade_submission
    _no_llm(monkeypatch)
    _, s = _session(env, refs=["pass"])
    r = grade_submission(s, code)
    assert r.verdict == verdict and r.tier == "execution-final"
    assert r.execution_outcome == outcome and r.deterministic is True


# 6 — an uncalibrated adapter cannot convict; falls through to Tier 4
def test_uncalibrated_adapter_cannot_convict(env, monkeypatch):
    import main.grading as g
    from main.grading import grade_submission
    # adapter that never calibrates (returns nonsense)
    monkeypatch.setattr(g, "_request_adaptation",
                        lambda *a, **k: ("return 12345", []))
    monkeypatch.setattr(g, "_ask_judge", lambda payload, role: (False, "no", 0.9, "x"))
    _, s = _session(env)
    r = grade_submission(s, "acc = a + b")     # renamed var: reference tail fails
    assert r.tier == "llm-judge"               # never execution-adapted
    assert r.deterministic is False


# 5 — renamed variable passes via CALIBRATED adaptation
def test_calibrated_adaptation_pass(env, monkeypatch):
    import main.grading as g
    from main.grading import grade_submission
    monkeypatch.setattr(g, "_request_adaptation",
                        lambda *a, **k: ("return acc", [{"target": "acc", "source": "total"}]))
    _, s = _session(env)
    r = grade_submission(s, "acc = a + b")
    assert (r.verdict, r.tier) == ("correct", "execution-adapted")
    assert r.deterministic is True and r.divergent is True


# 7 — calibrated + clean wrong output => deterministic incorrect
def test_calibrated_wrong_is_incorrect(env, monkeypatch):
    import main.grading as g
    from main.grading import grade_submission
    monkeypatch.setattr(g, "_request_adaptation",
                        lambda *a, **k: ("return acc", [{"target": "acc", "source": "total"}]))
    _, s = _session(env)
    r = grade_submission(s, "acc = a * b")     # wrong, but interface-compatible
    assert r.verdict == "incorrect" and r.tier == "execution-adapted"
    assert r.deterministic is True


# 10/11 — judge agreement labeled non-deterministic; disagreement indeterminate
def test_judge_agreement(env, monkeypatch):
    import main.grading as g
    from main.grading import grade_submission
    monkeypatch.setattr(g, "_request_adaptation", lambda *a, **k: ("", []))
    monkeypatch.setattr(g, "_ask_judge", lambda p, r: (True, "looks right", 0.9, "sem"))
    _, s = _session(env)
    res = grade_submission(s, "acc = a + b")
    assert res.verdict == "correct" and res.tier == "llm-judge"
    assert res.deterministic is False


def test_judge_disagreement_indeterminate(env, monkeypatch):
    import main.grading as g
    from main.grading import grade_submission
    monkeypatch.setattr(g, "_request_adaptation", lambda *a, **k: ("", []))
    calls = iter([(True, "yes", 0.9, "a"), (False, "no", 0.9, "b")])
    monkeypatch.setattr(g, "_ask_judge", lambda p, r: next(calls))
    _, s = _session(env)
    res = grade_submission(s, "acc = a + b")
    assert res.verdict == "indeterminate" and res.consume_attempt is False


def test_judge_outage_indeterminate(env, monkeypatch):
    import main.grading as g
    from main.grading import grade_submission
    monkeypatch.setattr(g, "_request_adaptation", lambda *a, **k: ("", []))
    def down(*a, **k): raise RuntimeError("OpenAI unreachable")
    monkeypatch.setattr(g, "_ask_judge", down)
    _, s = _session(env)
    res = grade_submission(s, "acc = a + b")
    assert res.verdict == "indeterminate" and res.consume_attempt is False
    assert res.tier == "system"


# 13 — sessions never expose references
def test_session_public_view_has_no_reference(env):
    pub, priv = _session(env)
    assert "reference" not in json.dumps(pub)
    assert priv["chunks"][0]["reference"]        # but the server still has it


# 14/15/20 — idempotency, ordering, no double advance
def test_submission_idempotent(env):
    from main.sessions import apply_outcome
    pub, _ = _session(env)
    r1 = apply_outcome(pub["session_id"], "sub-A", {"verdict": "correct"},
                       accept_code="total = a + b")
    r2 = apply_outcome(pub["session_id"], "sub-A", {"verdict": "correct"},
                       accept_code="total = a + b")
    assert r1["index"] == 1 and r2["index"] == 1 and r2["idempotent_replay"]


def test_unknown_session_rejected(env):
    from main.sessions import SessionError, load_session
    with pytest.raises(SessionError) as e:
        load_session("forged-session-id")
    assert e.value.reason_code == "session_not_found"


# 16/17 — reveal only at the limit, marks assisted; indeterminate advances nothing
def test_reveal_marks_assisted(env):
    from main.sessions import apply_outcome
    pub, _ = _session(env)
    apply_outcome(pub["session_id"], "s1", {"verdict": "incorrect"})
    st = apply_outcome(pub["session_id"], "s2", {"verdict": "incorrect"},
                       accept_code="total = a + b", provenance="revealed_reference")
    assert st["assisted"] is True and st["index"] == 1
    assert st["solved_independently"] is False


def test_indeterminate_consumes_nothing(env):
    from main.sessions import apply_outcome, load_session
    pub, _ = _session(env)
    st = apply_outcome(pub["session_id"], "s1", {"verdict": "indeterminate"},
                       consume_attempt=False)
    assert st["attempts"] == 0 and st["index"] == 0
    assert load_session(pub["session_id"])["index"] == 0


# 19 — no private material in the public grade payload
def test_public_payload_is_clean(env, monkeypatch):
    from main.grading import grade_submission
    _no_llm(monkeypatch)
    _, s = _session(env, refs=["pass"])          # last chunk => deterministic
    r = grade_submission(s, "return a * b")      # wrong -> has internal failures
    public = {"verdict": r.verdict, "tier": r.tier,
              "deterministic": r.deterministic, "reason": r.student_reason}
    blob = json.dumps(public)
    assert "expected" not in blob and "total = a + b" not in blob
    assert r.failures            # internally we DO keep them
