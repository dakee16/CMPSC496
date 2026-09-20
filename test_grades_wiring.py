"""The two silent breaks behind an empty grade sheet.

Both are invisible by construction. A denominator of zero renders as "nothing
to grade" rather than as an error, and an archive insert that never happens
leaves no trace anywhere - so a finished assignment with real submissions
behind it looked exactly like an assignment nobody had started.
"""
import ast
import inspect
import os
import pathlib

import pytest

from main.assignments import parse_assignment_file
from main.identity import content_hash
from main.sessions import CONTEXT_FIELDS, context_of

# Kept out of the repo (see main/handback.py). Missing means skip, not error.
HW3 = pathlib.Path(os.environ.get("MICROTUTOR_HW3")
                   or pathlib.Path(__file__).parent / "assignment_hw3.py")


@pytest.fixture(scope="module")
def problems():
    if not HW3.exists():
        pytest.skip(f"no assignment fixture at {HW3} (set MICROTUTOR_HW3)")
    parsed = parse_assignment_file(HW3.read_text(), HW3.name)
    return {p["slug"]: p for p in parsed["problems"]}


def _as_db_row(p):
    """The problem as `problems` stores it: context in a jsonb blob beside the
    flat columns, exactly what _group_columns writes."""
    row = {k: p[k] for k in ("slug", "title", "description", "solution")}
    row["ready"] = True
    row["context"] = {k: p[k] for k in CONTEXT_FIELDS if k in p}
    for k in ("group_slug", "group_title", "group_description"):
        row[k] = p.get(k)
    return row


# ── the denominator ──────────────────────────────────────────────────────

def test_a_method_read_back_from_the_database_keeps_its_identity(problems):
    """content_hash folds context_prefix and context_suffix into the key, so a
    method rebuilt without them hashes as a plain function and matches nothing
    in the chunk pool - which is how every class problem counted 0 steps."""
    for slug in ("stack-pop", "advanced-calculator-calculate-expressions"):
        p = problems[slug]
        row = _as_db_row(p)
        assert content_hash({**row, **context_of(row)}) == content_hash(p), slug


def test_dropping_the_context_changes_the_hash(problems):
    """The failure this is guarding against, stated directly - otherwise the
    test above passes for a build where context_of returns {} for everything."""
    p = problems["stack-pop"]
    thin = {k: p[k] for k in ("slug", "title", "description", "solution")}
    assert content_hash(thin) != content_hash(p)


def test_the_grade_sheet_asks_for_the_context_column(problems):
    """assignment_problems must SELECT context, not just the flat columns."""
    from main import grades
    src = inspect.getsource(grades.assignment_problems)
    assert "context" in src, "the pool key cannot be rebuilt without it"
    assert "context_of" in src, "...and it has to be flattened back on"


def test_context_of_survives_the_shapes_a_database_returns():
    assert context_of({}) == {}
    assert context_of({"context": None}) == {}
    assert context_of({"context": "not json"}) == {}
    assert context_of({"context": '{"context_prefix": "x"}'}) == {"context_prefix": "x"}
    assert context_of({"context": {"nonsense": 1}}) == {}, "unknown keys dropped"


# ── the spine row ────────────────────────────────────────────────────────

def test_every_archive_writer_is_actually_called():
    """save_session_start was the one writer of the five that nothing called.
    Four were wired when identity landed and it was missed, and no test, log or
    error could have said so."""
    root = pathlib.Path(__file__).parent
    archive = root / "main" / "archive.py"
    writers = [n.name for n in ast.parse(archive.read_text()).body
               if isinstance(n, ast.FunctionDef) and n.name.startswith("save_")]
    assert len(writers) >= 5, writers

    callers = "\n".join(
        f.read_text() for f in [*(root / "main").glob("*.py"),
                                *(root / "frontend").glob("*.py")]
        if f != archive)
    unused = [w for w in writers if w not in callers]
    assert not unused, f"archive writers nothing calls: {unused}"


def test_the_session_route_opens_the_spine_row():
    """The denominator, the 'turned up' column and save_session_end's UPDATE
    all hang off this row existing."""
    src = (pathlib.Path(__file__).parent / "frontend" / "api_server.py").read_text()
    route = src.split("def decompose_chunks_route")[1].split("\n@app.")[0]
    assert "save_session_start" in route
    assert "create_session" in route
    assert route.index("create_session") < route.index("save_session_start"), \
        "the row records what the session actually became"


# ── a finished problem has to look finished ──────────────────────────────

def _student_js():
    return (pathlib.Path(__file__).parent / "frontend" / "student.js").read_text()


def _api_src():
    return (pathlib.Path(__file__).parent / "frontend" / "api_server.py").read_text()


def test_completing_with_help_is_recorded_somewhere():
    """/mark_solved writes the `solved` table ONLY for an independent solve, so
    a problem finished with a shown answer was recorded nowhere at all: the
    list held it at "In progress" forever and the assignment bar never moved -
    one screen after the app told the student it was "recorded as solved with
    help"."""
    route = _api_src().split("def get_solved")[1].split("\n@app.")[0]
    assert "mt_sessions" in route, "the only place an assisted finish is recorded"
    assert "completed_at" in route, "...and only a FINISHED session counts"
    assert '"assisted"' in route, "the two claims are returned apart"


def test_mark_solved_still_means_they_did_it_themselves():
    """The guard against fixing the above by widening `solved`. That table is
    read by /history to say "you solved this before", and it has to keep
    meaning the student's own work."""
    route = _api_src().split("def mark_solved")[1].split("\n@app.")[0]
    assert "if independent:" in route
    i = route.index("if independent:")
    assert 'table("solved")' in route[i:], "the write must stay behind the guard"


def test_start_over_is_not_a_loop_back_to_the_congratulations_screen():
    """A finished problem now opens on its congratulations screen, which offers
    "Start over". /restart deletes nothing - the `solved` row is permanent - so
    reading that row alone sent the student straight back to the screen they had
    just left. The restart marker already scopes designs and messages; `solved`
    has to be read through it too."""
    route = _api_src().split('@app.get("/history/{slug}")')[1].split("\n@app.")[0]
    after = route[route.index('table("solved")'):]
    assert "if since:" in after, "the restart marker must narrow the solved flag"
    assert "completed_at" in after, "...and only a session finished SINCE it counts"


def test_the_student_list_counts_finished_problems_not_just_independent_ones():
    """Three separate places count progress - the assignment card, the problem
    header, and each class group. All three must agree, or a bar moves while
    the number beside it does not."""
    js = _student_js()
    assert "function isDone(slug)" in js
    # The invariant, stated directly: nothing counts progress by asking SOLVED
    # alone. statusOf is the one place that may, because distinguishing the two
    # is its whole job.
    # Exactly two lines may ask SOLVED directly: isDone, which is the
    # predicate, and statusOf, whose whole job is telling the two apart.
    for i, ln in enumerate(js.splitlines(), 1):
        if "SOLVED.has(" not in ln:
            continue
        if "ASSISTED.has(" in ln or 'return "solved"' in ln:
            continue
        raise AssertionError(f"line {i} counts independent solves only: {ln.strip()}")
    # ...and all three progress counts go through the predicate.
    counts = [ln for ln in js.splitlines()
              if "isDone" in ln and ".filter(" in ln]
    assert len(counts) == 3, counts


def test_an_assisted_finish_reads_as_done_not_as_in_progress():
    js = _student_js()
    assert 'ASSISTED.has(slug)) return "helped"' in js
    # ONE word. Nothing is revealed to a student any more, so an assisted
    # finish is shown as "Solved" too - what must not happen is it reading as
    # unfinished work. The two sets still differ underneath.
    assert '"helped": "Solved"' in js or 'helped: "Solved"' in js
    # ...and the state has a colour of its own, or it renders unstyled.
    css = (pathlib.Path(__file__).parent / "frontend" / "ui.css").read_text()
    assert ".stat.helped{" in css


def test_finishing_records_the_kind_of_finish_it_actually_was():
    """The server is about to report the same split back, and disagreeing with
    it for one screen is how a problem reads Solved until the next reload."""
    js = _student_js()
    fn = js.split("async function finish(")[1].split("\n}")[0]
    assert "res.solved_independently ? SOLVED : ASSISTED" in fn


# ── a repeat of the same answer must not be a worse answer ───────────────
#
# The submission id is a hash of the code (frontend/student.js), so resubmitting
# an identical answer is answered from submissions.result_json and never reaches
# the grader again. Anything the route attached to its RESPONSE rather than to
# that row therefore vanished on the second try: a diagnosis built on the
# student's own measured values degraded into the bare "we could not confirm
# this step", and the two recovery buttons it drives went with it.

def test_the_route_records_what_the_student_was_shown():
    """The evidence goes into the committed row, not just onto the response."""
    route = _api_src().split("def grade_chunk_route")[1].split("\n@app.")[0]
    committed = route.split("state = commit_outcome(")[0].split("graded = {")[1]
    for field in ("needs_diagnosis", "diagnosis", "failing_cases",
                  "failed_total", "revealed_reference"):
        assert field in committed, \
            f"{field} is not persisted, so a replay loses it"
    assert "commit_outcome(\n            req.session_id, req.submission_id," \
           " session[\"revision\"], graded," in route, \
        "the committed dict must be the one that carries the evidence"


def test_the_response_is_read_back_out_of_the_record():
    """Both paths build the same body. A replay has `state` and no `result`, so
    anything still read off `result` here is a field the replay cannot have."""
    route = _api_src().split("def grade_chunk_route")[1].split("\n@app.")[0]
    tail = route[route.index('body = {"verdict"'):]
    assert "state[key]" in tail, "the optional fields must come from the record"
    for field in ("needs_diagnosis", "diagnosis", "failing_cases",
                  "failed_total", "revealed_reference"):
        assert f"result.{field}" not in tail, \
            f"{field} is still read off the live GradeResult, which a replay " \
            f"does not have"


def test_an_identical_resubmission_keeps_its_diagnosis(tmp_path):
    """End to end through the real store: what was committed comes back."""
    import types

    from main.sessions import (begin_submission, commit_outcome,
                               create_session, stored_result)

    db = str(tmp_path / "sessions.sqlite3")
    n = lambda **kw: types.SimpleNamespace(**kw)
    opened = create_session(
        {"slug": "frequency", "title": "Frequency", "description": "d",
         "solution": "def frequency(txt):\n    return {}"},
        {"header": "def frequency(txt):",
         "chunks": [n(step_id="Part 1", prompt="count them",
                      expected_type="code", reference="counts = {}"),
                    n(step_id="Part 2", prompt="return them",
                      expected_type="code", reference="return counts")]},
        "hash-frequency", student_id="student-1", db_path=db)
    sid = opened["session_id"]

    graded = {
        "verdict": "indeterminate", "tier": "execution-adapted",
        "deterministic": False, "divergent": False,
        "reason": "We could not confirm this step.",
        "needs_diagnosis": True,
        "diagnosis": "For 'aab', your step leaves `unique` holding {'a', 'b'}. "
                     "What does that tell you about the text it came from?",
        "failing_cases": ["frequency('aab')\n\nexpected: {'a': 2, 'b': 1}"],
        "failed_total": 4,
    }
    prior, session = begin_submission(sid, "sub-abc123", db_path=db)
    assert prior is None
    # An indeterminate verdict accepts nothing and spends no attempt.
    commit_outcome(sid, "sub-abc123", session["revision"], graded,
                   consume_attempt=False, db_path=db)

    replay = stored_result(sid, "sub-abc123", db_path=db)
    assert replay["idempotent_replay"] is True
    assert replay["needs_diagnosis"] is True
    assert replay["diagnosis"] == graded["diagnosis"]
    assert replay["failing_cases"] == graded["failing_cases"]
    assert replay["failed_total"] == 4
    # ...and the replay still costs nothing and moves nothing.
    assert replay["index"] == 0 and replay["attempts"] == 0

    # begin_submission takes the same route for a concurrent twin.
    again, _ = begin_submission(sid, "sub-abc123", db_path=db)
    assert again["diagnosis"] == graded["diagnosis"]


def test_the_page_shows_the_cases_on_an_indeterminate_verdict():
    """grading.adapted_evidence_only returns `indeterminate` WITH cases, and
    says "the case below is worth tracing by hand". The branch that renders it
    dropped them, so the sentence pointed at nothing."""
    js = _student_js()
    generic = next(l for l in js.splitlines()
                   if 'res.verdict === "indeterminate"' in l and "needs_diagnosis" in l)
    after = js[js.index(generic):]
    assert "failingCasesHTML(res)" in after.split("\n")[1], \
        "the bare indeterminate branch must pass the cases through"
    diag = js.split("if (res.needs_diagnosis && res.diagnosis){")[1][:400]
    assert "failingCasesHTML(res)" in diag, \
        "a diagnosis that arrived with evidence must show the evidence"
