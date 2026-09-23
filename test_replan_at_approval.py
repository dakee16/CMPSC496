"""test_replan_at_approval.py - the roadmap is chosen when the plan exists.

THE BUG THIS PINS, measured on the real routes before the fix: a first-time
student never got a rerouted roadmap, whatever they planned.

  first open, no plan yet   -> 0 model calls (nothing to reroute from)
  after the plan is written -> 0 model calls, same session resumed

The order was the whole problem. /decompose_chunks is called from start(p) the
instant a student clicks into a problem - before they have planned - so
latest_plan_graph() found nothing, the teacher's roadmap was taken and the
session was created. Planning then happened, and openGate() called loadSteps(),
which only fetches the PROMPTS of the session that already exists. Nothing
revisited the roadmap, and nothing could have: find_resumable() returns early on
a later /decompose_chunks by design, because a resumed session must keep its own
chunks. So follows_reference() was never called with a plan in hand.

/replan asks the question at approval instead - the first moment a plan exists
and the last moment no code has been written.

The fakes here are the network edges only (Supabase, the model, the pool);
every decision under test is the real code.
"""
import pytest

from test_restart_reroute import (env, _open, _plan, _stamp, SLUG,  # noqa: F401
                                  PROBLEM, DIVERGENT_PLAN, SOLUTION)


@pytest.fixture(autouse=True)
def _keep_the_oracle_cache_clean(tmp_path, monkeypatch):
    """The oracle cache is a TRACKED file and this path writes to it.

    reroute.build reaches get_oracle_tests, which is the WRITE path: a problem
    it has not seen gets a new entry appended to data/oracles/tests_cache.json.
    The fixture problem here is synthetic, so running these tests left a
    synthetic entry in a file that is committed - the same trap
    resolved_entries.json sets, which grading.py's own self-check sidesteps the
    same way."""
    monkeypatch.setenv("MICROTUTOR_ORACLE_CACHE", str(tmp_path / "oracles.json"))


@pytest.fixture
def env2(env, monkeypatch):
    """`env`, but with a model that can also answer the DECOMPOSER.

    test_restart_reroute's fake returns a proposal for every call, which is all
    its own tests need - they measure cost and failure. A test that asserts a
    reroute SUCCEEDS has to get past the decomposition too, so this answers
    whichever of the two is being asked. Everything it returns is the teacher's
    own code, split at the obvious seam, so what the real gates then check -
    the oracle at 100%, assembly, necessity, shape - is genuinely checked."""
    import json
    from main import ollama_client, reroute, run_phase1

    body = "\n".join(ln[4:] for ln in SOLUTION.splitlines()[1:] if ln.strip())

    def fake_chat(model, system, messages, **kw):
        env.calls.append((system or "").splitlines()[0][:40])
        if "subproblem" in (system or "").lower():
            return json.dumps({"subproblems": [
                {"prompt": "Count how many times each letter appears, and keep "
                           "that tally for the next step.",
                 "reference": "counts = {}\nfor ch in txt:\n"
                              "    if ch.isalpha():\n"
                              "        counts[ch] = counts.get(ch, 0) + 1"},
                {"prompt": "Hand back what you counted.",
                 "reference": "return counts"}]})
        return json.dumps({"body": body})

    for mod in (ollama_client, reroute, run_phase1):
        monkeypatch.setattr(mod, "chat", fake_chat, raising=False)
    return env


def _approve_design(env, at=None):
    """What /design_review/plan writes when it accepts a plan."""
    env.sb.rows("mt_designs").append(
        {"student_id": env.student, "slug": SLUG, "approved": True,
         "created_at": at or _stamp()})


def _replan(env):
    before = len(env.calls)
    r = env.client.post("/replan", json={"slug": SLUG})
    assert r.status_code == 200, r.text
    return r.json(), len(env.calls) - before


def test_a_first_time_student_now_gets_their_own_roadmap(env2):
    """The reported case, in the order a real student goes in."""
    env = env2
    first, spent = _open(env)
    assert spent == 0, "nothing to reroute from before a plan exists"
    teacher_steps = len(first["chunks"])

    # They plan, and it is approved. NOW the roadmap can be decided.
    _plan(env, _stamp())
    _approve_design(env)
    out, spent_on_replan = _replan(env)

    assert out["rerouted"] is True, out
    assert spent_on_replan > 0, "a rebuild that cost nothing did not happen"
    assert out["session_id"] != first["session_id"], \
        "a rebuilt roadmap needs its own session"
    assert out["chunks"], "the rebuilt roadmap came back empty"
    print(f"\n  teacher roadmap: {teacher_steps} steps -> "
          f"rebuilt: {len(out['chunks'])} steps, {spent_on_replan} model calls")


def test_the_new_session_is_what_a_later_open_resumes(env2):
    """The old session is retired, so find_resumable cannot pick between two."""
    env = env2
    first, _ = _open(env)
    _plan(env, _stamp()); _approve_design(env)
    out, _ = _replan(env)
    assert out["rerouted"] is True
    again, spent = _open(env)
    assert again["session_id"] == out["session_id"], \
        "reopening must resume the REBUILT session, not the retired one"
    assert spent == 0, "and resuming must not pay for the roadmap again"


def test_no_plan_means_no_reroute_and_no_cost(env):
    """Approved, but nothing planned in this run - the teacher's roadmap stands."""
    first, _ = _open(env)
    _approve_design(env)
    out, spent = _replan(env)
    assert out == {"rerouted": False}, out
    assert spent == 0, "a student with no plan must not pay for a rebuild"


def test_an_unapproved_student_is_refused(env):
    """The step prompts are the answer split up; /replan hands back chunks, so
    it carries the same gate /session_steps does."""
    _open(env)
    _plan(env, _stamp())
    r = env.client.post("/replan", json={"slug": SLUG})
    assert r.status_code == 403, r.text
    assert r.json()["detail"]["reason_code"] == "design_not_approved"


def test_work_already_accepted_is_carried_forward_not_lost(env2):
    """A student can be rerouted AFTER they have started, and their code comes
    with them.

    The steps they had accepted were graded against chunks that stop existing,
    so they cannot be carried as ACCEPTED - but the code is theirs and retyping
    it is not a thing to ask. It comes back as `carried`, for the page to seed
    the editor with; bridge.find then reports how far it reaches, so one
    submission can answer several of the new steps at once."""
    from main import sessions
    env = env2
    first, _ = _open(env)
    sessions.apply_outcome(first["session_id"], "sub-1", {"verdict": "correct"},
                           accept_code="counts = {}")
    _plan(env, _stamp()); _approve_design(env)
    out, spent = _replan(env)
    assert out["rerouted"] is True, out
    assert spent > 0, "a rebuild that cost nothing did not happen"
    assert "counts = {}" in out["carried"], out.get("carried")
    assert out["session_id"] != first["session_id"]


def test_code_that_could_not_be_confirmed_can_trigger_a_rebuild(env2):
    """THE CASE PROSE CANNOT EXPRESS. A plan that follows the teacher's route
    still gets a rebuild if what they actually WROTE is a different shape -
    and `code` is only ever sent after grading could not confirm the step."""
    from main.context import solution_body
    from main.graphs import code_graph
    from main import reroute
    env = env2
    _open(env)
    # A plan the router reads as the SAME route, so nothing here comes from it.
    same = code_graph(solution_body(dict(PROBLEM)),
                      reroute.effective_header(dict(PROBLEM)))
    env.sb.rows("mt_graphs").append(
        {"student_id": env.student, "slug": SLUG, "kind": "plan",
         "created_at": _stamp(), "graph": same})
    _approve_design(env)
    assert _replan(env)[0] == {"rerouted": False}, "the plan alone says nothing"

    # ...now send what they wrote. Two nested loops where the step has one.
    divergent = ("letters = list(txt)\nseen = []\nfor ch in letters:\n"
                 "    n = 0\n    for other in letters:\n"
                 "        if other == ch:\n            n += 1\n"
                 "    if n == 1:\n        seen.append(ch)")
    before = len(env.calls)
    r = env.client.post("/replan", json={"slug": SLUG, "code": divergent})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["rerouted"] is True, out
    assert len(env.calls) - before > 0


def test_a_plan_that_follows_the_teacher_is_not_rebuilt(env):
    """Errs to 'same route': being wrong that way costs nothing."""
    from main.context import solution_body
    from main.graphs import code_graph
    from main import reroute
    _open(env)
    same = code_graph(solution_body(dict(PROBLEM)),
                      reroute.effective_header(dict(PROBLEM)))
    env.sb.rows("mt_graphs").append(
        {"student_id": env.student, "slug": SLUG, "kind": "plan",
         "created_at": _stamp(), "graph": same})
    _approve_design(env)
    out, spent = _replan(env)
    assert out == {"rerouted": False}, out
    assert spent == 0, "matching the teacher's route must be free"
