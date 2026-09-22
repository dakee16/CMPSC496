"""test_tier_trace.py - every acquittal is traced, exactly once, under its tier.

WHY THIS EXISTS. Deleting the LLM judges is gated on one measurement: how often
tier 4 is what actually acquitted a submission. main/trace.py was given a
durable sink specifically so that history survives a restart and the question
can be answered - but two of the `correct` exits in main/grading.py did not
record a route event at all:

  * execution-final's ordinary last-chunk pass, which is how EVERY problem
    finishes, recorded nothing;
  * execution-reference recorded only in the rare subcase where a declaration
    had to be hoisted.

llm-judge, meanwhile, always recorded. So a tier census read off the trace
showed the judges as a far larger share of `correct` than they are - and the
census is the thing that decides whether the judges can go, so under-counting
the deterministic tiers argues for keeping them. The verdicts were never wrong;
only the record of how they were reached was.

SELF-CONTAINED ON PURPOSE. The real pool and problem file are gitignored, so a
test resting on them skips on a fresh checkout and guards nothing. grading takes
an `oracle_loader`, so the oracle here is inline and every other part - the
sandbox, the bridge, the session store, the trace sink - is the real thing. No
model is reachable: `chat` raises, so a path that needs one fails loudly.
"""
import json
import types

import pytest

SOLUTION = ("def total(nums):\n"
            "    running = 0\n"
            "    for n in nums:\n"
            "        running += n\n"
            "    return running\n")
PROBLEM = {"slug": "total", "title": "Total", "description": "Add the numbers.",
           "solution": SOLUTION}
STEPS = [{"step_id": "Part 1", "expected_type": "code",
          "prompt": "Add the numbers up, keeping the running total.",
          "reference": "running = 0\nfor n in nums:\n    running += n"},
         {"step_id": "Part 2", "expected_type": "code",
          "prompt": "Hand back the total.", "reference": "return running"}]
ORACLE = [{"input": [[1, 2, 3]], "expected": 6}, {"input": [[]], "expected": 0},
          {"input": [[5]], "expected": 5}, {"input": [[-1, 1]], "expected": 0},
          {"input": [[10, -2, 7]], "expected": 15}]


@pytest.fixture
def env(monkeypatch, tmp_path):
    """A real session store and a real trace file, both under tmp_path."""
    trace_file = tmp_path / "model_trace.jsonl"
    monkeypatch.setenv("MICROTUTOR_TRACE_FILE", str(trace_file))
    monkeypatch.setenv("MICROTUTOR_SESSION_DB", str(tmp_path / "sessions.sqlite3"))
    from main import grading, identity, ollama_client, sessions, trace
    # grade_submission persists an entry into the TRACKED resolved_entries.json
    # otherwise - grading.py's own self-check makes the same move.
    monkeypatch.setattr(identity, "_RESOLVED_PATH", str(tmp_path / "resolved.json"))

    def no_model(*_a, **_k):
        raise AssertionError("a model call escaped a deterministic tier")
    for mod in (ollama_client, grading):
        if hasattr(mod, "chat"):
            monkeypatch.setattr(mod, "chat", no_model, raising=False)
    # The sink attaches lazily and caches that it did, so a previous test's
    # handler would otherwise still be the one receiving events.
    monkeypatch.setattr(trace, "_ATTACHED", False)
    monkeypatch.setattr(trace, "_LOG", __import__("logging").getLogger(
        "microtutor.trace.test"), raising=False)
    trace._LOG.propagate = False
    trace._LOG.setLevel(10)

    def routes_since(n):
        if not trace_file.exists():
            return []
        events = [json.loads(l) for l in trace_file.read_text().splitlines()]
        return [e for e in events[n:] if e.get("kind") == "route"]

    def written():
        return (len(trace_file.read_text().splitlines())
                if trace_file.exists() else 0)

    def fresh_session(student):
        decomp = {"header": "def total(nums):",
                  "chunks": [types.SimpleNamespace(**c) for c in STEPS]}
        pub = sessions.create_session(dict(PROBLEM), decomp, "hash-total",
                                      student_id=student)
        return pub["session_id"]

    return types.SimpleNamespace(grading=grading, sessions=sessions,
                                 routes_since=routes_since, written=written,
                                 fresh_session=fresh_session)


def _grade(env, session_id, code):
    """Grade one submission and return (result, the route events it produced)."""
    before = env.written()
    s = env.sessions.load_session(session_id)
    res = env.grading.grade_submission(s, code, oracle_loader=lambda p: list(ORACLE))
    return res, env.routes_since(before)


@pytest.mark.parametrize("label,code,tier", [
    # The reference verbatim: the commonest acquittal there is.
    ("verbatim", STEPS[0]["reference"], "execution-reference"),
    # Renamed local: the deterministic bridge, no model (see main/bridge.py).
    ("renamed", STEPS[0]["reference"].replace("running", "acc"),
     "execution-bridged"),
])
def test_a_non_final_acquittal_is_traced_once_under_its_own_tier(env, label, code, tier):
    res, routes = _grade(env, env.fresh_session("stu-" + label), code)
    assert res.verdict == "correct", (res.verdict, res.reason_code)
    assert res.tier == tier, res.tier
    assert [(e["final_route"], e["verdict"]) for e in routes] == [(tier, "correct")], routes


def test_the_last_chunk_pass_is_traced(env):
    """How every problem finishes. It recorded nothing at all."""
    sid = env.fresh_session("stu-final")
    env.sessions.apply_outcome(sid, "sub-1", {"verdict": "correct"},
                               accept_code=STEPS[0]["reference"])
    res, routes = _grade(env, sid, STEPS[1]["reference"])
    assert res.verdict == "correct" and res.tier == "execution-final", (res.verdict, res.tier)
    assert [(e["final_route"], e["verdict"]) for e in routes] == \
        [("execution-final", "correct")], routes


def test_no_acquittal_is_counted_twice(env):
    """Two explicit traces on one path would inflate that tier's share just as
    surely as a missing one deflates it."""
    for label, code in [("a", STEPS[0]["reference"]),
                        ("b", STEPS[0]["reference"].replace("running", "acc"))]:
        res, routes = _grade(env, env.fresh_session("stu-dup-" + label), code)
        assert res.verdict == "correct"
        assert len(routes) == 1, f"{res.tier} traced {len(routes)} times"


def test_a_conviction_is_still_deterministic_and_reference_free(env):
    """Not a tracing claim - the guard that this file's edits to the grader did
    not touch what a verdict IS. Blank and undefined-name convict without any
    model, and no tier here may be a judge."""
    for code in ("", "running = acc"):
        res, _ = _grade(env, env.fresh_session("stu-bad-" + str(len(code))), code)
        assert res.verdict == "incorrect", (code, res.verdict)
        assert res.tier == "syntax", (code, res.tier)
