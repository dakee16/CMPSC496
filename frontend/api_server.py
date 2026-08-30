"""
api_server.py - FastAPI bridge between Next.js web UI and local LLM pipeline.
Place this file in your microprog_phase1/ folder and run:
    pip install fastapi uvicorn
    uvicorn api_server:app --port 8000 --reload
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json
from dotenv import load_dotenv
import bcrypt
import os
from supabase import create_client

# Must run before anything reads os.environ below. It happened to work only
# because importing main.run_phase1 (next block) loads its own .env as a
# side effect first -- reorder those imports, or drop that import, and the
# create_client() call two lines down starts raising KeyError.
load_dotenv()

from main.run_phase1 import eval_step, parse_json, decompose_into_chunks, replan_from_prefix, get_chunk_decomposition
from main.schemas import StepItem

app = FastAPI(title="MicroTutor API", version="1.0")

# Single lazy client boundary. Importing this module must perform NO network or
# client construction, so tests can import the app and inject a fake without
# credentials. No route may build its own client.
_SB = None


def get_supabase():
    global _SB
    if _SB is None:
        _SB = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    return _SB


def set_supabase(client):
    """Test seam: inject a fake and skip credentials entirely."""
    global _SB
    _SB = client


_UUID_RE = __import__("re").compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", 2)


def resolve_student(value: str | None) -> str | None:
    """Turn whatever the demo sent into a real students.id UUID.

    students.id, student_interactions.student_id and solved.student_id are all
    uuid columns, but the demo role picker has no login and sends a typed NAME.
    Inserting that raised `invalid input syntax for type uuid` on every write,
    which the logging path then swallowed - progress silently vanished.

    A UUID passes through untouched. A name is looked up, and created on a miss,
    so a demo student keeps the same id across sessions and their history adds
    up. TEMPORARY: real identity arrives with PSU-email auth, at which point the
    JWT subject replaces this entirely."""
    name = (value or "").strip()
    if not name:
        return None
    if _UUID_RE.match(name):
        return name
    sb = get_supabase()
    try:
        hit = sb.table("students").select("id").eq("username", name).limit(1).execute().data
        if hit:
            return hit[0]["id"]
        # No password: a demo row cannot be logged into, only referenced.
        made = sb.table("students").insert(
            {"username": name, "password_hash": "!demo-no-login"}).execute().data
        return made[0]["id"] if made else None
    except Exception as e:
        # Identity is optional - anonymous sessions are valid. Never fail a
        # student's request because we could not name them.
        print(f"  ⚠️  could not resolve student {name!r}: {e}")
        return None


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # allow any frontend origin (for dev)
    allow_credentials=True,
    allow_methods=["*"],          # allow GET, POST, PUT, etc.
    allow_headers=["*"],          # allow any headers
)



class DecomposeRequest(BaseModel):
    slug: str
    description: str
    # Uploaded problems carry the professor's reference solution in the request;
    # curated ones are looked up in the DB by slug. Either way a solution is
    # REQUIRED before the pipeline runs - see decompose_chunks_route.
    title: str | None = None
    solution: str | None = None
    # Identity is bound ONCE, here, when the session begins. /grade_chunk never
    # accepts it again - it must not be changeable mid-session.
    student_id: str | None = None


class EvaluateRequest(BaseModel):
    step: dict
    answer: str
    context: str = ""

class ReplanRequest(BaseModel):
    slug: str
    description: str
    accepted_steps: list[dict]

class ChunkRequest(BaseModel, extra="forbid"):
    """The client is NOT authoritative. It may send only an opaque session id,
    a stable submission id, its code, and (optionally) the index it believes it
    is on so a stale UI can be detected. Solution, chunks, references, accepted
    prefix and the real index all live server-side."""
    session_id: str
    submission_id: str
    student_code: str
    expected_index: int | None = None

class AuthRequest(BaseModel):
    username: str
    password: str

class LogInteractionRequest(BaseModel):
    student_id: str
    slug: str
    chunk_index: int
    attempt_number: int
    student_code: str
    verdict: bool
    tier: str
    reason: str

class TutorChatRequest(BaseModel, extra="forbid"):
    """The tutor is given the problem SLUG, never a solution. The server looks
    up only the public fields; the reference solution never enters this path."""
    slug: str
    messages: list[dict] = []
    chunk_prompt: str | None = None


class MarkSolvedRequest(BaseModel, extra="forbid"):
    """Only a completed grading session may claim a solve. Student, slug and
    independence are DERIVED from it - the client cannot assert any of them."""
    session_id: str


class LiveRunRequest(BaseModel):
    # Either a curated slug (solution looked up in the DB) or a fully custom
    # problem with the ground truth pasted in. Same rule as /decompose_chunks:
    # no ground truth, no run.
    slug: str
    title: str | None = None
    description: str | None = None
    solution: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "message": "MicroTutor API running"}


# NOTE: POST /decompose (the older step-based flow) was removed. It had zero
# callers anywhere in the repo and its backing function decompose_validated()
# had none outside that route, so it was dead by construction -- and it served
# ungated material: it built a problem dict with no "solution", so
# get_oracle_tests() returned [] and the "skipped" status was accepted as pass,
# with neither an oracle-strength nor a necessity check. /decompose_chunks is
# the live path. /replan is NOT dead (tests/test_replan.py uses its backing
# function) and is flagged for gating, not deletion.


@app.post("/evaluate")
def evaluate(req: EvaluateRequest):
    """DISABLED. This was the old step-based, LLM-ONLY answer evaluator: it
    judged a student's answer with no execution, no oracle and no gates - a
    second answer-checking implementation with different behavior. It has no
    callers. All answer checking goes through /grade_chunk."""
    raise HTTPException(status_code=410, detail={
        "reason_code": "legacy_evaluate_disabled",
        "message": "This endpoint has been retired; use /grade_chunk."})


@app.post("/replan")
def replan(req: ReplanRequest):
    """DISABLED. This route served ungated material: it built a problem dict
    with no solution, so get_oracle_tests() returned [] and replan_from_prefix()
    accepted status "skipped" as success - no oracle-strength check and no
    necessity gate. It is closed rather than left open while it is rebuilt
    behind the same serve boundary as /decompose_chunks."""
    raise HTTPException(status_code=410, detail={
        "reason_code": "replan_disabled",
        "message": "Replanning is temporarily unavailable."})


@app.post("/decompose_chunks")
def decompose_chunks_route(req: DecomposeRequest):
    try:
        problem = {"slug": req.slug, "title": req.title or req.slug,
                   "description": req.description,
                   "solution": (req.solution or "").strip()}

        # Curated problems keep their ground truth in the DB; uploads send it in
        # the request. Only look it up when the request didn't carry one.
        if not problem["solution"]:
            from main.run_phase1 import load_problems
            problems = load_problems(limit=500)
            full = next((p for p in problems if p.get("slug") == req.slug), None)
            if full:
                problem["solution"] = (full.get("solution") or "").strip()

        # No ground truth means no oracle, which means no mutation validation and
        # no Gate 1 - the decomposition would be served unvalidated. That silent
        # degradation is exactly what uploads used to do; it is now a hard error.
        if not problem["solution"]:
            raise HTTPException(
                status_code=400,
                detail=(f"No reference solution for '{req.slug}'. A problem cannot "
                        f"be decomposed without ground truth - oracle tests, "
                        f"mutation validation and the necessity gate all depend "
                        f"on it. Supply `solution` with the request."))

        result = get_chunk_decomposition(problem)
        # Register a server-owned session. From here the browser never sees a
        # reference, the solution, or oracle data again.
        from main.identity import content_hash
        from main.sessions import create_session
        # Identity is bound ONCE, here, and resolved to a real students.id so
        # every later write (interactions, solved) has a valid uuid.
        public = create_session(problem, result, content_hash(problem),
                                student_id=resolve_student(
                                    getattr(req, "student_id", None)))

        # NO oracle pre-warm here. get_oracle_tests() is the WRITE path: on a
        # miss it generates inputs and runs mutation testing, minutes of paid
        # work triggered by a student pressing Start. Preparation now happens
        # once at teacher-upload time (main/publish.py), and a problem that did
        # not survive it is never offered to a student in the first place.
        return public          # session_id, decomposition_id, header, PUBLIC chunks
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Decomposition unavailable: {e}")


@app.post("/grade_chunk")
def grade_chunk_route(req: ChunkRequest):
    """Grade one submission against a SERVER-OWNED session.

    All grading logic lives in main.grading.grade_submission - this route only
    loads the session, enforces request-level preconditions, applies the
    attempt/reveal policy, and strips private material from the response."""
    from main.grading import grade_submission
    from main.sessions import MAX_ATTEMPTS, SessionError, load_session

    try:
        session = load_session(req.session_id)
    except SessionError as e:
        code = 409 if e.reason_code in ("session_completed", "session_expired",
                                        "session_inactive") else 404
        raise HTTPException(status_code=code,
                            detail={"reason_code": e.reason_code, "message": str(e)})

    # Stale UI: the browser thinks it is on a different chunk than the server.
    if req.expected_index is not None and req.expected_index != session["index"]:
        raise HTTPException(status_code=409, detail={
            "reason_code": "stale_index",
            "message": "Your page is out of date - reload to continue.",
            "index": session["index"]})

    # Reserve first. A concurrent twin of this exact submission is told to
    # retry with the SAME id rather than being graded a second time.
    from main.sessions import begin_submission, commit_outcome
    prior, session = begin_submission(req.session_id, req.submission_id)
    if prior is not None:
        if prior.get("__in_flight__"):
            raise HTTPException(status_code=409, detail={
                "reason_code": "submission_in_progress",
                "message": "This answer is still being graded - retry with the "
                           "same submission id."})
        return prior                            # stored result, graded once

    # Graded with NO write transaction held, so a slow grade blocks nobody.
    from main.sessions import release_submission
    try:
        result = grade_submission(session, req.student_code)
    except Exception as e:                      # never let our fault convict
        # Hand the reservation back. Without this the row stays claimed with no
        # result, and every retry of this submission id 409s "still being
        # graded" forever - the student can neither advance nor retry.
        release_submission(req.session_id, req.submission_id)
        raise HTTPException(status_code=503, detail={
            "reason_code": "grader_unavailable", "message": str(e)[:200]})

    # ── attempt / reveal policy (server-owned) ──
    accept_code, provenance, reveal_ref = None, "student", None
    if result.verdict == "correct":
        accept_code = req.student_code
    elif result.verdict == "incorrect" and session["attempts"] + 1 >= MAX_ATTEMPTS:
        # Second failure: reveal THIS chunk's reference, record its provenance,
        # mark the session assisted, and move on.
        reveal_ref = session["chunks"][session["index"]].get("reference", "")
        accept_code, provenance = reveal_ref, "revealed_reference"

    try:
        state = commit_outcome(
            req.session_id, req.submission_id, session["revision"], {
                "verdict": result.verdict, "tier": result.tier,
                "deterministic": result.deterministic,
                "reason": result.student_reason, "divergent": result.divergent},
            accept_code=accept_code, provenance=provenance,
            consume_attempt=result.consume_attempt)
    except SessionError as e:
        # Same reasoning as the grader failure above: the CAS lost, nothing was
        # recorded, so the reservation must not outlive the attempt.
        release_submission(req.session_id, req.submission_id)
        raise HTTPException(status_code=409, detail={
            "reason_code": e.reason_code, "message": str(e)})

    # Log authoritatively here - the browser no longer reports its own verdicts.
    bound_student = session.get("student_id")
    if bound_student and result.verdict != "indeterminate":
        try:
            # student_interactions, NOT interactions. `interactions` is the OLD
            # research table (step_id/agent_level/answer/hint_shown/score) from
            # the weak/normal/strong agent experiment; it has no attempt_number,
            # so every insert here failed with PGRST204 and was swallowed by the
            # except below - this path had been logging nothing at all.
            get_supabase().table("student_interactions").insert({
                "student_id": bound_student, "problem_slug": session["slug"],
                "chunk_index": session["index"], "attempt_number": state["attempts"],
                "student_code": (req.student_code or "")[:4000],
                "verdict": result.verdict == "correct", "tier": result.tier,
                "reason": result.student_reason[:500]}).execute()
        except Exception as e:
            print(f"  ⚠️  interaction log failed: {e}")

    # PUBLIC response: no oracle data, no failures, no future references, no
    # adapted tail, no internal exception text.
    body = {"verdict": result.verdict, "tier": result.tier,
            "deterministic": result.deterministic, "reason": result.student_reason,
            "divergent": result.divergent, "index": state["index"],
            "attempts": state["attempts"], "assisted": state["assisted"],
            "completed": state["completed"],
            "solved_independently": state["solved_independently"],
            "total_chunks": state["total_chunks"],
            "idempotent_replay": state.get("idempotent_replay", False)}
    if reveal_ref is not None:
        body["revealed_reference"] = reveal_ref     # only ever at the limit
    return body


# ── playground (read-only showcase) ───────────────────────────────────────
# These serve the step-through demo in frontend/playground.html. They REPLAY
# cached results and never trigger validation, decomposition, or any LLM call
# a demo must not stall for minutes on a click. The only things computed on read
# are the AST mutants and their knockouts, which are deterministic, model-free
# and sub-second; they are recomputed because evaluate_oracle's per-mutant
# breakdown is not currently persisted (only the aggregate rates are).

@app.get("/playground/problems")
def playground_problems():
    """Every problem with a cached verdict, for the showcase picker."""
    from tests.sandbox import _load_cache
    out = []
    for entry in _load_cache().values():
        if isinstance(entry, dict) and "strong" in entry:
            out.append({"slug": entry.get("slug", ""),
                        "strong": bool(entry["strong"]),
                        "kill_rate_direct": entry.get("kill_rate_direct", 0.0),
                        "n_tests": len(entry.get("final_tests", []))})
    return sorted(out, key=lambda r: (not r["strong"], r["slug"]))


@app.get("/playground/{slug}")
def playground_detail(slug: str):
    """One problem's full journey, assembled from cache. Never recomputes an
    oracle or a decomposition; returns nulls the UI renders as 'not yet'."""
    import json as _json
    import os as _os
    from main.identity import content_hash, get_resolved_entry
    from main.mutation import _disagrees, generate_mutants
    from tests.sandbox import _load_cache, passes_tests, run_solution

    row = get_supabase().table("problems").select(
        "slug, title, description, difficulty, solution").eq("slug", slug).execute().data
    if not row:
        raise HTTPException(status_code=404, detail=f"Problem '{slug}' not found.")
    problem = row[0]
    solution = problem.get("solution") or ""
    # The solution is needed BELOW to recompute mutants, but it must not be
    # returned. Everything sent back is built from `public_problem`.
    public_problem = {k: v for k, v in problem.items() if k != "solution"}

    cached = _load_cache().get(content_hash(problem))
    if not (isinstance(cached, dict) and "strong" in cached):
        return {"problem": public_problem, "oracle": None, "mutants": [],
                "chunks": None, "necessity": [], "grading": None,
                "note": "not yet validated"}

    tests = cached.get("final_tests", [])
    entry = get_resolved_entry(problem)["entry_name"]

    # Stage 2 - prefer the breakdown persisted at validation time; only fall
    # back to recomputing for entries written before it was stored.
    breakdown = cached.get("breakdown")
    if isinstance(breakdown, dict) and breakdown.get("mutants"):
        # "killed" means the suite AS HANDED IN caught it; killed_on_retry means
        # it only died after the search added a test, i.e. it slipped past the
        # suite the student would have faced.
        mutants = [{"label": m["label"], "caught": m["status"] == "killed",
                    "status": m["status"]}
                   for m in breakdown["mutants"]]
        source = "cached"
    else:
        base = run_solution(solution, [t["input"] for t in tests], entry_name=entry)
        expected = base["results"] if base["ok"] else []
        mutants = []
        for m in generate_mutants(solution):
            caught = bool(expected) and _disagrees(
                m["code"], entry, [t["input"] for t in tests], expected)
            mutants.append({"label": m["label"], "caught": caught,
                            "status": "killed" if caught else "survived"})
        source = "recomputed"

    # Stages 3/4 - pooled decomposition, if one exists.
    chunks, necessity = None, []
    pool_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                              "main", "chunk_pool.json")
    if _os.path.exists(pool_path):
        try:
            pool = _json.load(open(pool_path))
        except Exception:
            pool = {}
        pooled = (pool.get(content_hash(problem)) or [None])[0]
        if pooled:
            chunks = pooled["chunks"]
            header = pooled["header"]
            body = [c.get("reference", "") for c in chunks]
            for i, c in enumerate(chunks):
                knocked = list(body)
                knocked[i] = "pass"
                code = header + "\n" + "\n".join(
                    "    " + ln for ref in knocked for ln in (ref or "").splitlines())
                res = passes_tests(code, tests, entry_name=entry)
                necessity.append({
                    "step_id": c.get("step_id", f"Part {i + 1}"),
                    "broke": (not res["ok"]) or res["fraction"] < 1.0,
                    "passed": res.get("passed", 0), "total": res.get("total", 0)})

    # Stage 5 - a right and a wrong submission, decided by EXECUTION only.
    grading = None
    if tests:
        wrong = "def _f(*a, **k):\n    return None"
        ok = passes_tests(solution, tests, entry_name=entry)
        bad = passes_tests(wrong, tests, entry_name="_f")
        grading = {
            "correct": {"tier": "Tier 2 - executed against the oracle",
                        "verdict": ok["ok"] and ok["fraction"] == 1.0,
                        "passed": ok.get("passed", 0), "total": ok.get("total", 0)},
            "incorrect": {"tier": "Tier 2 - executed against the oracle",
                          "verdict": bad["ok"] and bad["fraction"] == 1.0,
                          "passed": bad.get("passed", 0), "total": bad.get("total", 0)},
        }

    # Why is it weak? Without this the UI can show "100% caught" beside a WEAK
    # badge - true but self-contradictory-looking - when the real reason is that
    # the solution yielded too few mutants to judge at all.
    from main.mutation import _MIN_MUTANTS, CUTOFF_1_KILL_RATE
    weak_reason = None
    if not cached["strong"]:
        if len(mutants) < _MIN_MUTANTS:
            weak_reason = (f"only {len(mutants)} way(s) to break this solution could be "
                           f"found - too few to judge the tests fairly "
                           f"(need at least {_MIN_MUTANTS})")
        else:
            weak_reason = (f"{round(cached.get('kill_rate_direct', 0.0) * 100)}% of cheaters "
                           f"caught is below the {round(CUTOFF_1_KILL_RATE * 100)}% bar")

    # Chunk PROMPTS are public; chunk REFERENCES are the answer to each step and
    # never leave the server. The showcase only ever needed the prompts and the
    # knockout result, so dropping the references costs it nothing.
    if chunks:
        chunks = [{k: v for k, v in c.items() if k != "reference"} for c in chunks]

    return {"problem": public_problem,
            "oracle": {"n_tests": len(tests), "tests": tests[:6],
                       "kill_rate_direct": cached.get("kill_rate_direct", 0.0),
                       "kill_rate": cached.get("kill_rate", 0.0),
                       "strong": bool(cached["strong"]),
                       "weak_reason": weak_reason,
                       "validated_at": cached.get("validated_at", "")},
            "mutants": mutants, "breakdown": breakdown,
            "breakdown_source": source, "chunks": chunks,
            "necessity": necessity, "grading": grading, "note": None}


# ── playground LIVE (real pipeline run, streamed) ─────────────────────────
# The read-only endpoints above replay cache; this one runs the REAL pipeline
# (fresh oracle generation, mutation testing + repair, verdict, decomposition,
# Gate 1) and streams every step as a newline-delimited JSON event the moment
# it happens. Served with fetch()+ReadableStream on the frontend - the
# one-directional SSE pattern, delivered over POST because the ground-truth
# solution rides in the request body (EventSource can only GET).
#
# COSTS REAL MONEY per click: same OpenAI usage as a warmup pass on one
# problem. It also persists its verdict to tests/tests_cache.json exactly as
# warmup would, so a live run is never wasted work.

@app.post("/playground/live")
def playground_live(req: LiveRunRequest):
    from fastapi.responses import StreamingResponse

    problem = {"slug": req.slug, "title": req.title or req.slug,
               "description": req.description or "",
               "solution": (req.solution or "").strip()}

    # Same contract as /decompose_chunks: curated problems keep their ground
    # truth in the DB; uploads carry it in the request. No ground truth = no
    # oracle = nothing to watch - hard error, not a degraded run.
    if not problem["solution"] or not problem["description"]:
        row = get_supabase().table("problems").select(
            "slug, title, description, solution").eq("slug", req.slug).execute().data
        if row:
            problem["solution"] = problem["solution"] or (row[0].get("solution") or "").strip()
            problem["description"] = problem["description"] or (row[0].get("description") or "")
            problem["title"] = req.title or row[0].get("title") or req.slug
    if not problem["solution"]:
        raise HTTPException(
            status_code=400,
            detail=(f"No reference solution for '{req.slug}'. A live run cannot "
                    f"start without ground truth - oracle tests, mutation "
                    f"validation and Gate 1 all depend on it. Supply `solution` "
                    f"with the request."))

    from main.live_playground import ndjson_stream
    return StreamingResponse(
        ndjson_stream(problem),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/problems")
def list_problems(limit: int = 100, difficulty: str = None):
    """List problems from Supabase with optional difficulty filter."""
    try:
        query = get_supabase().table("problems").select(
            "id, slug, title, difficulty, topic_tags"
        ).limit(limit)
        if difficulty:
            query = query.eq("difficulty", difficulty)
        res = query.execute()
        return {"problems": res.data, "count": len(res.data)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# PUBLIC problem columns. `solution` is deliberately absent and must stay that
# way: this route is unauthenticated and the student UI calls it on every
# problem click, so selecting `solution` here handed the reference answer to
# anyone who asked. /decompose_chunks reads the solution server-side from the
# database instead - the browser never needs to carry it.
_PUBLIC_PROBLEM_COLS = "id, slug, title, difficulty, description, topic_tags"


@app.get("/problems/{slug}")
def get_problem(slug: str):
    """Fetch a single problem by slug. PUBLIC fields only - never the solution."""
    try:
        res = get_supabase().table("problems").select(
            _PUBLIC_PROBLEM_COLS).eq("slug", slug).single().execute()
        if not res.data:
            raise HTTPException(status_code=404, detail=f"Problem '{slug}' not found.")
        return res.data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── assignments: teacher upload, student browse ──────────────────────────
# The trust boundary runs straight through this section. Teacher routes may see
# solutions and preparation errors; student routes may see neither, and may only
# ever list problems that are `ready` - an unready problem cannot be graded, so
# offering one would dead-end the student.

class AssignmentUpload(BaseModel, extra="forbid"):
    """The file arrives as text, not multipart: the browser can read it with
    FileReader, which avoids adding python-multipart as a dependency."""
    filename: str = "assignment.py"
    content: str
    teacher_name: str = "demo-teacher"


class ManualSplitRequest(BaseModel, extra="forbid"):
    slug: str
    header: str
    chunks: list[dict]


@app.get("/assignment_template")
def assignment_template():
    """A starter file a teacher can download, edit and re-upload."""
    from main.assignments import TEMPLATE
    return {"filename": "assignment.py", "content": TEMPLATE}


@app.post("/teacher/assignments")
def upload_assignment(req: AssignmentUpload):
    """Parse an assignment file and PREPARE every problem, streaming progress.

    Preparation (oracle generation, mutation validation, decomposition) is
    minutes of work, so this streams NDJSON as each problem finishes instead of
    holding the connection silent. Same one-directional pattern as
    /playground/live, delivered over POST because the file rides in the body."""
    import json as _json
    from fastapi.responses import StreamingResponse
    from main.assignments import AssignmentParseError, parse_assignment_file
    from main.publish import prepare_assignment_stream

    try:
        parsed = parse_assignment_file(req.content, req.filename)
    except AssignmentParseError as e:
        raise HTTPException(status_code=400, detail={
            "reason_code": "unparseable_assignment", "message": str(e)})

    sb = get_supabase()
    row = sb.table("assignments").insert({
        "name": parsed["name"], "teacher_name": req.teacher_name,
        "source_file": req.filename}).execute().data
    assignment_id = row[0]["id"]

    def events():
        yield _json.dumps({"event": "parsed", "assignment_id": assignment_id,
                           "name": parsed["name"],
                           "n_problems": len(parsed["problems"]),
                           "parse_errors": parsed["errors"]}) + "\n"
        # Problems the FILE got wrong never reach preparation; record them so the
        # teacher sees one combined list rather than two.
        for bad in parsed["errors"]:
            try:
                sb.table("problems").upsert({
                    "slug": bad["slug"], "title": bad["slug"],
                    "description": "", "solution": "",
                    "assignment_id": assignment_id, "ready": False,
                    "prepare_error": bad["error"]},
                    on_conflict="assignment_id,slug").execute()
            except Exception:
                pass

        for ev in prepare_assignment_stream(parsed["problems"]):
            if ev.get("event") == "prepared":
                src = next((p for p in parsed["problems"]
                            if p["slug"] == ev["slug"]), None)
                if src:
                    try:
                        sb.table("problems").upsert({
                            "slug": src["slug"], "title": src["title"],
                            "description": src["description"],
                            "solution": src["solution"],
                            "assignment_id": assignment_id,
                            "ready": bool(ev["ready"]),
                            "prepare_error": ev.get("error")},
                            on_conflict="assignment_id,slug").execute()
                    except Exception as e:
                        # A problem that did not SAVE is not ready, whatever
                        # preparation decided. Reporting it as ready while the
                        # row is missing is exactly how an assignment showed
                        # green badges and still sat at 0 / 0 - the failure has
                        # to reach the teacher, not a swallowed warning field.
                        ev = {**ev, "ready": False,
                              "error": f"prepared, but could not be saved: {e}"[:300]}
            yield _json.dumps(ev) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/assignments")
def list_assignments():
    """Assignments with a ready-count. Safe for students AND teachers."""
    sb = get_supabase()
    rows = sb.table("assignments").select(
        "id, name, teacher_name, created_at").order(
        "created_at", desc=True).execute().data or []
    probs = sb.table("problems").select(
        "assignment_id, ready").not_.is_("assignment_id", "null").execute().data or []
    counts = {}
    for p in probs:
        c = counts.setdefault(p["assignment_id"], {"total": 0, "ready": 0})
        c["total"] += 1
        c["ready"] += 1 if p["ready"] else 0
    return {"assignments": [
        {**a, **counts.get(a["id"], {"total": 0, "ready": 0})} for a in rows]}


@app.get("/assignments/{assignment_id}/problems")
def assignment_problems(assignment_id: str):
    """STUDENT view. Ready problems only, public columns only.

    No solution, no prepare_error, and nothing that isn't ready - a student must
    never be handed a problem the grader cannot actually grade."""
    res = get_supabase().table("problems").select(
        _PUBLIC_PROBLEM_COLS).eq("assignment_id", assignment_id).eq(
        "ready", True).execute()
    return {"problems": res.data or [], "count": len(res.data or [])}


@app.get("/teacher/assignments/{assignment_id}/problems")
def teacher_assignment_problems(assignment_id: str):
    """TEACHER view. Every problem including the ones that failed, with reasons.

    Still no solution: the teacher already has it in their own file, and not
    sending it keeps one fewer copy of the answer key moving over the wire."""
    res = get_supabase().table("problems").select(
        _PUBLIC_PROBLEM_COLS + ", ready, prepare_error").eq(
        "assignment_id", assignment_id).execute()
    rows = res.data or []
    return {"problems": rows,
            "ready": sum(1 for r in rows if r["ready"]),
            "failed": sum(1 for r in rows if not r["ready"]),
            "count": len(rows)}


@app.post("/teacher/split")
def manual_split(req: ManualSplitRequest):
    """Teacher-authored decomposition for a problem the model could not split.

    Goes through the SAME serve gate as a generated one - a hand-written split
    can still contain a step that does no work, which would let a student skip
    it and be marked correct."""
    from main.publish import save_manual_decomposition

    sb = get_supabase()
    row = sb.table("problems").select(
        "slug, title, description, solution").eq("slug", req.slug).execute().data
    if not row:
        raise HTTPException(status_code=404, detail=f"Problem '{req.slug}' not found.")
    problem = row[0]
    try:
        out = save_manual_decomposition(problem, req.header, req.chunks)
    except Exception as e:
        # A gate rejection is teacher feedback, not a server fault.
        raise HTTPException(status_code=400, detail={
            "reason_code": "split_rejected",
            "message": str(e).splitlines()[0][:300]})
    sb.table("problems").update(
        {"ready": True, "prepare_error": None}).eq("slug", req.slug).execute()
    return out



@app.post("/register")
def register(req: AuthRequest):
    pw_hash = bcrypt.hashpw(req.password.encode(), bcrypt.gensalt()).decode()

    try:
        result = get_supabase().table("students").insert({
            "username": req.username,
            "password_hash": pw_hash
        }).execute()
    except Exception:
        raise HTTPException(status_code=400, detail="Username already taken")

    if not result.data:
        raise HTTPException(status_code=400, detail="Could not create account")

    row = result.data[0]
    return {"student_id": row["id"], "username": row["username"]}


@app.post("/login")
def login(req: AuthRequest):

    result = get_supabase().table("students").select("*").eq("username", req.username).execute()

    if not result.data:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    row = result.data[0]
    if not bcrypt.checkpw(req.password.encode(), row["password_hash"].encode()):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    return {"student_id": row["id"], "username": row["username"]}


@app.get("/solved/{student_id}")
def get_solved(student_id: str):

    result = get_supabase().table("solved").select("problem_slug").eq("student_id", student_id).execute()
    slugs = [r["problem_slug"] for r in (result.data or [])]
    return {"slugs": slugs}


@app.post("/tutor_chat")
def tutor_chat(req: TutorChatRequest):
    """Socratic tutor for the problem currently open on the student page.

    Deliberately NOT session-bound: it never grades, never advances a session,
    never consumes an attempt, and never sees a solution, a chunk reference or
    an oracle test. It only reads the public title/description the student can
    already see."""
    from main.tutor import MAX_TURNS, reply

    if len(req.messages) > MAX_TURNS * 2:
        raise HTTPException(status_code=400, detail={
            "reason_code": "conversation_too_long",
            "message": "This conversation is very long — start a fresh one."})

    row = get_supabase().table("problems").select(
        "slug, title, description").eq("slug", req.slug).execute().data
    if not row:
        raise HTTPException(status_code=404, detail={
            "reason_code": "problem_not_found",
            "message": f"Unknown problem '{req.slug}'."})

    try:
        out = reply(row[0], req.messages, req.chunk_prompt)
    except Exception as e:
        # A tutor outage is not a judgement about the student.
        raise HTTPException(status_code=503, detail={
            "reason_code": "tutor_unavailable",
            "message": "The tutor is unavailable right now. Try again shortly.",
            "detail": str(e)[:120]})
    return out


@app.post("/mark_solved")
def mark_solved(req: MarkSolvedRequest):
    """Derive the solve from a completed session. The old {student_id, slug}
    form let a browser mark any problem solved for any student, including
    incomplete or assisted work."""
    from main.sessions import session_snapshot

    snap = session_snapshot(req.session_id)
    if snap is None:
        raise HTTPException(status_code=404, detail={
            "reason_code": "session_not_found", "message": "Unknown session."})
    if snap["state"] != "completed":
        raise HTTPException(status_code=409, detail={
            "reason_code": "session_incomplete",
            "message": "That session isn't finished."})
    if not snap["student_id"]:
        return {"ok": False, "recorded": False, "reason": "anonymous session"}

    independent = not snap["assisted"]
    if independent:
        get_supabase().table("solved").upsert(
            {"student_id": snap["student_id"], "problem_slug": snap["slug"]},
            on_conflict="student_id,problem_slug").execute()
    return {"ok": True, "recorded": independent,
            "solved_independently": independent, "assisted": snap["assisted"]}


@app.post("/log_interaction")
def log_interaction(req: LogInteractionRequest):

    data = req.model_dump()
    data["problem_slug"] = data.pop("slug")
    get_supabase().table("student_interactions").insert(data).execute()
    return {"ok": True}
