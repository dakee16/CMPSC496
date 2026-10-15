"""test_roadmap_score.py - roadmaps are graded, the best three different ones
are served, and a problem none of whose roadmaps reaches the bar waits on its
teacher instead of going live.

THE REPORT (14 Oct). A problem served every saved roadmap at random: up to
seven, some the same cut saved five times (HW4 update), one whose first step
was one line and whose second fifteen (HW4 __contains__). Sanan: "add a
rubric/grading system for the roadmaps ... top 3 will make it into the final
3 roadmaps (also add a cutoff grade ...)". Agreed after measuring it on every
saved HW3 and HW4 roadmap: three, or as many different ones as the method
allows; "needs review" only when not one reaches the bar; copies dropped;
problems already live are only pointed out, never hidden.

REAL: main/roadmap_score, run_phase1 (_usable, fill_pool), the splitter,
publish.prepare_problem, the teacher routes and the review view. FAKED: the
model (canned cuts and wording - test_splitter's env), the oracle (a fixed
certified set) and Supabase.
"""
import pytest

from test_splitter import LOPSIDED, PROBLEM, _cut, env  # noqa: F401

# Three different cuts of the teacher's code that reach the bar, and two that
# do not: everything in one step, and a 16-line step before the last three.
GOOD = [_cut(0, 4, 10, 16), _cut(0, 6, 10, 16), _cut(0, 2, 6, 10, 16)]
TWO_STEP = _cut(0, 16)


def _d(entry):
    from main import run_phase1
    return run_phase1._deserialize(entry)


def _points(entry):
    from main import roadmap_score
    return roadmap_score.score(PROBLEM, _d(entry))["points"]


def test_the_grade_puts_lopsided_roadmaps_under_the_bar_and_even_ones_over(env):
    from main import roadmap_score
    assert all(_points(g) >= roadmap_score.CUTOFF for g in GOOD), [_points(g) for g in GOOD]
    lop = roadmap_score.score(PROBLEM, _d(LOPSIDED))
    assert lop["points"] < roadmap_score.CUTOFF, lop
    assert lop["parts"]["balance"] < 5 and lop["parts"]["biggest step"] < 10, \
        "it says WHY: one step doing nearly everything"
    assert _points(TWO_STEP) < roadmap_score.CUTOFF


def test_the_best_three_different_roadmaps_are_served(env):
    from main import identity, run_phase1
    unclear = {**GOOD[0], "chunks": [{**c, "prompt": "Do it."} for c in GOOD[0]["chunks"]]}
    pool = [LOPSIDED, unclear, GOOD[2], GOOD[0], GOOD[0], GOOD[1], TWO_STEP]
    run_phase1._add_to_pool(identity.content_hash(PROBLEM), pool, replace=True)
    served = run_phase1._usable(dict(PROBLEM), run_phase1._load_pool()[identity.content_hash(PROBLEM)])
    assert sorted(map(_points, served), reverse=True) == list(map(_points, served)), "best first"
    assert len(served) == 3 and all(s in GOOD for s in served), "three different, all over the bar"
    assert unclear not in served, "of two copies of a cut, the clearly worded one"


def test_a_live_problem_with_none_over_the_bar_still_serves_its_best(env):
    from main import identity, run_phase1
    run_phase1._add_to_pool(identity.content_hash(PROBLEM), [LOPSIDED, TWO_STEP], replace=True)
    served = run_phase1._usable(dict(PROBLEM), run_phase1._load_pool()[identity.content_hash(PROBLEM)])
    assert served == [TWO_STEP, LOPSIDED], "never hidden - ranked"
    assert run_phase1.get_chunk_decomposition(dict(PROBLEM))["chunks"]


def test_an_upload_none_of_whose_roadmaps_reaches_the_bar_waits_on_its_teacher(env, monkeypatch):
    from main import publish, roadmap_score
    env.builder(LOPSIDED)
    monkeypatch.setattr(env.splitter, "MAX_STEP_LINES", 50)
    monkeypatch.setattr(env.splitter, "plan", lambda *a, **k: [_d(TWO_STEP)])
    out = publish.prepare_problem(dict(PROBLEM))
    assert out["ready"] is False and out["stage"] == "steps", out
    assert out["error"].startswith("Waiting on your review: no roadmap reached the bar"), out
    assert publish.stage_of_error(out["error"]) == "steps"
    saved = env.saved()
    assert saved and max(map(_points, saved)) < roadmap_score.CUTOFF, \
        "the best of what was built is kept, for the teacher to look at"


def test_rebuilding_never_makes_a_problem_worse(env, monkeypatch):
    from main import identity, run_phase1
    run_phase1._add_to_pool(identity.content_hash(PROBLEM), GOOD, replace=True)
    env.builder(LOPSIDED)
    monkeypatch.setattr(env.splitter, "plan", lambda *a, **k: [])
    run_phase1.fill_pool(dict(PROBLEM), replace=True)
    assert env.saved() == GOOD, "a worse new set is not swapped in"


def test_copies_are_not_saved_and_the_splitter_words_only_its_best(env, monkeypatch):
    from main import identity, roadmap_score, run_phase1
    monkeypatch.setattr(env.splitter, "MAX_STEP_LINES", 50)
    run_phase1._add_to_pool(identity.content_hash(PROBLEM), [GOOD[0]], replace=True)
    env.builder(GOOD[0])                 # the model gives the same cut again
    assert run_phase1.fill_pool(dict(PROBLEM)) == 3
    saved = env.saved()
    assert saved[0] == GOOD[0] and len(saved) == 3
    assert all(not roadmap_score._same(_d(a), _d(b)) for i, a in enumerate(saved) for b in saved[i + 1:])
    assert env.log.worded == 2, "two cuts worded - the two it saved - of the eight it found"
    assert all(_points(s) >= roadmap_score.CUTOFF for s in saved)


def test_a_model_roadmap_under_the_bar_is_not_saved_beside_good_ones(env, monkeypatch):
    from main import run_phase1
    monkeypatch.setattr(env.splitter, "MAX_STEP_LINES", 50)
    env.builder([GOOD[0], LOPSIDED])
    run_phase1.fill_pool(dict(PROBLEM))
    saved = env.saved()
    assert GOOD[0] in saved and LOPSIDED not in saved, [_points(s) for s in saved]


def test_the_splitter_offers_every_step_count_so_three_reach_the_bar(env, monkeypatch):
    """Fewest-steps-first offered this method four 2-step cuts (16 + 3 lines and
    the like) and one 3-step cut: only two would have reached the bar."""
    from main import roadmap_score, run_phase1
    monkeypatch.setattr(env.splitter, "MAX_STEP_LINES", 50)
    env.builder(RuntimeError("the model could not split it"))
    assert run_phase1.fill_pool(dict(PROBLEM)) == 3
    assert all(_points(s) >= roadmap_score.CUTOFF for s in env.saved())


# ── through the routes ───────────────────────────────────────────────────

@pytest.fixture
def teacher(env, monkeypatch, tmp_path):
    from frontend import api_server
    from main.identity import content_hash
    from main.oracle_store import save_cache
    from test_auth_routes import teacher_client
    from test_splitter import ORACLE
    monkeypatch.setenv("MICROTUTOR_TRANSCRIPTS", str(tmp_path / "tr.json"))
    c, sb = teacher_client()
    row = {"slug": PROBLEM["slug"], "title": PROBLEM["title"], "description": PROBLEM["description"],
           "solution": PROBLEM["solution"], "assignment_id": "a-1", "ready": False,
           "prepare_error": None}
    sb.problems.append(row)
    save_cache({content_hash(PROBLEM): {"strong": True, "status": "strong", "kill_rate": 1.0,
                                        "kill_rate_direct": 1.0, "features": ["calls"],
                                        "final_tests": ORACLE,
                                        "breakdown": {"mutants": []}}})
    assert content_hash({**row, **api_server._context_of(row)}) == content_hash(PROBLEM)
    return c, sb, row


def test_the_teacher_sees_the_roadmaps_and_publishes_with_the_best(env, teacher):
    from main import identity, publish, run_phase1
    c, sb, row = teacher
    run_phase1._add_to_pool(identity.content_hash(PROBLEM), [LOPSIDED, TWO_STEP], replace=True)
    row["prepare_error"] = publish.roadmap_review_message(_points(TWO_STEP))
    v = c.get(f"/teacher/review/a-1/{PROBLEM['slug']}").json()
    R = v["roadmaps"]
    assert v["in_review"] and R["review"] and not R["below"]
    assert R["best"] == _points(TWO_STEP) and [r["points"] for r in R["list"]] == \
        [_points(TWO_STEP), _points(LOPSIDED)]
    assert R["list"][0]["steps"][0].startswith("Write code that handles part 1")

    d = c.post("/teacher/problems/accept_roadmaps",
               json={"assignment_id": "a-1", "slug": PROBLEM["slug"]}).json()
    assert d["ready"] is True and d["accepted_by"], d
    assert row["ready"] is True and row["prepare_error"] is None
    assert publish.roadmaps_accepted(dict(PROBLEM))
    # Preparing again respects the decision for this version - no second
    # review - though it still builds nothing better (no cut reaches the bar).
    env.builder(LOPSIDED)
    import main.splitter as splitter
    real = splitter.plan
    splitter.plan = lambda *a, **k: [_d(TWO_STEP)]
    try:
        assert publish.prepare_problem(dict(PROBLEM))["ready"] is True
    finally:
        splitter.plan = real


def test_a_live_problem_under_the_bar_is_pointed_out_and_stays_live(env, teacher, monkeypatch):
    from main import identity, run_phase1
    c, sb, row = teacher
    row["ready"] = True
    run_phase1._add_to_pool(identity.content_hash(PROBLEM), [LOPSIDED], replace=True)
    listed = c.get("/teacher/assignments/a-1/problems").json()["problems"]
    (mine,) = [p for p in listed if p["slug"] == PROBLEM["slug"]]
    assert mine["roadmaps_below"] == _points(LOPSIDED)
    v = c.get(f"/teacher/review/a-1/{PROBLEM['slug']}").json()
    assert v["roadmaps"]["below"] and v["queue"] == [{"slug": PROBLEM["slug"], "title": PROBLEM["title"]}]

    env.builder(LOPSIDED)                # a rebuild that finds nothing better
    monkeypatch.setattr(env.splitter, "plan", lambda *a, **k: [])
    d = c.post("/teacher/problems/rebuild_roadmaps",
               json={"assignment_id": "a-1", "slug": PROBLEM["slug"]}).json()
    assert row["ready"] is True, "a live problem is never hidden by a rebuild"
    assert c.get(f"/teacher/review/a-1/{PROBLEM['slug']}").json()["roadmaps"]["below"], \
        "...and still pointed out"

    env.builder(GOOD)                    # ...and one that does
    d = c.post("/teacher/problems/rebuild_roadmaps",
               json={"assignment_id": "a-1", "slug": PROBLEM["slug"]}).json()
    assert d["ready"] is True and row["ready"] is True, d
    listed = c.get("/teacher/assignments/a-1/problems").json()["problems"]
    assert [p["roadmaps_below"] for p in listed if p["slug"] == PROBLEM["slug"]] == [None]


def test_a_student_cannot_rebuild_or_accept_roadmaps(env, teacher):
    from test_auth_routes import PW, client, register
    s = client()
    register(s, "abc123@psu.edu", PW)
    for path in ("/teacher/problems/rebuild_roadmaps", "/teacher/problems/accept_roadmaps"):
        assert s.post(path, json={"assignment_id": "a-1", "slug": PROBLEM["slug"]}).status_code == 403


def test_an_edit_to_a_live_problem_is_kept_though_its_roadmaps_are_under_the_bar(env, teacher, monkeypatch):
    """HW4 __contains__: its only cut of the teacher's code is 1 line then 15.
    The teacher adds an example to its docstring and prepares it again - the
    edit must be saved, and the problem stay live, flagged."""
    from main import identity, run_phase1
    c, sb, row = teacher
    row["ready"] = True
    run_phase1._add_to_pool(identity.content_hash(PROBLEM), [LOPSIDED], replace=True)
    env.builder(LOPSIDED)
    monkeypatch.setattr(env.splitter, "plan", lambda *a, **k: [_d(TWO_STEP)])
    edited = PROBLEM["solution"].replace("def score(words):\n",
                                         'def score(words):\n    """Score the words."""\n')
    d = c.post("/teacher/problems/retry", json={"assignment_id": "a-1", "slug": PROBLEM["slug"],
                                                "source": edited, "keep_live": True}).json()
    saved = [p for p in sb.problems if p["slug"] == PROBLEM["slug"]][-1]
    assert d["ready"] is True and saved["ready"] is True, d
    assert '"""Score the words."""' in saved["solution"], "the edit is saved"
    listed = c.get("/teacher/assignments/a-1/problems").json()["problems"]
    assert [p["roadmaps_below"] for p in listed if p["slug"] == PROBLEM["slug"]][-1] is not None
