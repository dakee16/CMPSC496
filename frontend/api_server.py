"""
api_server.py - FastAPI bridge between Next.js web UI and local LLM pipeline.
Place this file in your microprog_phase1/ folder and run:
    pip install fastapi uvicorn
    uvicorn api_server:app --port 8000 --reload
"""
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import json
from dotenv import load_dotenv
import os
from supabase import create_client

# Must run before anything reads os.environ below. It happened to work only
# because importing main.run_phase1 (next block) loads its own .env as a
# side effect first -- reorder those imports, or drop that import, and the
# create_client() call two lines down starts raising KeyError.
load_dotenv()

from main import auth as auth_mod
from main.run_phase1 import eval_step, parse_json, decompose_into_chunks, replan_from_prefix, get_chunk_decomposition
from main.schemas import StepItem

app = FastAPI(title="MicroTutor API", version="1.0")

# Local http development cannot set Secure cookies; anything else must. Behind
# the VPN is not an exception - the VPN carries every other student too, so a
# session riding over plain http is readable by them, not by the internet.
_COOKIE_SECURE = os.environ.get("MICROTUTOR_ENV", "dev") != "dev"


def _require_auth_configured():
    """Refuse to issue sessions without a signing key, rather than signing
    every cookie with "" - which would let anyone mint one."""
    if not auth_mod.is_configured():
        raise HTTPException(status_code=503, detail={
            "reason_code": "auth_not_configured",
            "message": "Sign-in is not configured on this server."})

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


# ── who is asking ────────────────────────────────────────────────────────
# THE ONE PLACE the rest of the app answers "who is this". Everything below
# reads the signed cookie; nothing reads a name or a role out of a request
# body. resolve_student() used to sit here and did the opposite - it took the
# typed name the demo role-picker sent and looked it up, creating the row on a
# miss, which meant anyone could act as anyone by typing their name.

def current_claims(request: Request | None) -> dict | None:
    """Verified session claims, or None if signed out."""
    from main.auth import SESSION_COOKIE, read_session
    if request is None:
        return None
    return read_session(request.cookies.get(SESSION_COOKIE, ""))


def current_student(request: Request | None = None) -> str | None:
    """The signed-in student's students.id, or None.

    The id, not the username: an address can be reassigned when a student
    changes their name, and re-pointing years of saved work at the wrong
    person is not a recoverable mistake."""
    claims = current_claims(request)
    return claims["sub"] if claims else None


def require_student(request: Request) -> dict:
    """Claims, or 401. Use on anything that writes a student's own work."""
    claims = current_claims(request)
    if not claims:
        raise HTTPException(status_code=401, detail={
            "reason_code": "not_signed_in", "message": "Please sign in."})
    return claims


def require_teacher(request: Request) -> dict:
    """Claims, or 401/403. The role comes from the students row the cookie was
    signed from, never from the browser - the old gate let anyone reach the
    upload screen by picking "Instructor" on a radio button."""
    claims = require_student(request)
    if claims.get("role") != "teacher":
        raise HTTPException(status_code=403, detail={
            "reason_code": "not_a_teacher",
            "message": "This page is for instructors."})
    return claims


# The pages are served BY this app (see the StaticFiles mount at the bottom of
# this file), so in production the browser is same-origin and CORS never comes
# into it. This block exists only for opening frontend/*.html straight off disk
# during development.
#
# `allow_origins=["*"]` is gone and cannot come back: the CORS spec forbids the
# wildcard together with credentials, so with a session cookie in play every
# browser would silently refuse to send it and every request would look
# signed-out for no visible reason.
_DEV_ORIGINS = [o.strip() for o in os.environ.get(
    "MICROTUTOR_ALLOWED_ORIGINS",
    "http://localhost:8000,http://127.0.0.1:8000,"
    "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]

# `null` - the origin a page opened straight off disk sends - is deliberately
# NOT in this list, and adding it is a trap worth naming. It does let the
# request through, so sign-in answers 200 and the page looks like it worked.
# But the session cookie is SameSite=Lax and (in dev) not Secure, so the
# browser stores nothing for an opaque origin: the very next request is signed
# out, and the user is bounced back to the login form with no error to read.
# A loud "could not reach the server" beats a login that silently un-happens.
# Serve the pages from this app instead - http://localhost:8000/login.html -
# which is same-origin, needs no CORS at all, and is how it is deployed.

app.add_middleware(
    CORSMiddleware,
    allow_origins=_DEV_ORIGINS,
    allow_credentials=True,       # required for the session cookie
    allow_methods=["*"],
    allow_headers=["*"],
)



class DecomposeRequest(BaseModel, extra="forbid"):
    # forbid, like ChunkRequest: an unrecognised field is a stale client, and
    # the one that used to be here was student_id. Ignoring it silently is how
    # a page keeps sending an identity that stopped meaning anything.
    slug: str
    description: str
    # Uploaded problems carry the professor's reference solution in the request;
    # curated ones are looked up in the DB by slug. Either way a solution is
    # REQUIRED before the pipeline runs - see decompose_chunks_route.
    title: str | None = None
    solution: str | None = None
    # NO student_id. Identity is bound ONCE when the session begins, and it is
    # taken from the session cookie - never from the body, which the student
    # controls and used to be able to set to anyone's name.


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

class RegisterRequest(AuthRequest):
    """Registration also carries a name. Separate from AuthRequest so /login
    cannot quietly accept fields it would then ignore - and so the one route
    that stores a name is the only one that can be handed one."""
    first_name: str = ""
    last_name: str = ""

class TutorChatRequest(BaseModel, extra="forbid"):
    """The tutor is given the problem SLUG, never a solution. The server looks
    up only the public fields; the reference solution never enters this path."""
    slug: str
    messages: list[dict] = []
    chunk_prompt: str | None = None
    # Whether this student's design has been approved by /design_review. It only
    # selects the tutor's POSTURE (push back vs. help), so a forged `true` costs
    # nothing worse than a friendlier tutor - it can never reveal a solution,
    # since the tutor is not given one in either mode. The coding-UI lock itself
    # is currently enforced client-side; making it server-authoritative needs an
    # identity to key the approval to, which arrives with the PSU auth work.
    design_ok: bool = False


class PlanGraphRequest(BaseModel, extra="forbid"):
    """Re-extract the plan graph from the chat so far.

    `current` is the last graph the browser was given. It round-trips through
    the client because the plan graph has nowhere to live yet - it belongs to a
    STUDENT, and there is no authenticated student to key it to until the PSU
    login lands. main/archive.py holds the storage that replaces this."""
    slug: str
    messages: list[dict] = []
    current: dict | None = None


class GraphsRequest(BaseModel, extra="forbid"):
    """Both graphs plus their comparison, for a finished (or in-progress)
    session. The code graph is derived server-side from the session's accepted
    answers - the client cannot assert what code it wrote."""
    session_id: str
    plan: dict | None = None


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
def decompose_chunks_route(req: DecomposeRequest, request: Request):
    claims = require_student(request)
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
        # Identity is bound ONCE, here, from the signed cookie - so every later
        # write (interactions, solved) carries a students.id the student could
        # not have chosen.
        public = create_session(problem, result, content_hash(problem),
                                student_id=claims["sub"])

        # NO oracle pre-warm here. get_oracle_tests() is the WRITE path: on a
        # miss it generates inputs and runs mutation testing, minutes of paid
        # work triggered by a student pressing Start. Preparation now happens
        # once at teacher-upload time (main/publish.py), and a problem that did
        # not survive it is never offered to a student in the first place.
        return public          # session_id, decomposition_id, header, PUBLIC chunks
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Decomposition unavailable: {e}")


@app.post("/grade_chunk")
def grade_chunk_route(req: ChunkRequest, request: Request):
    """Grade one submission against a SERVER-OWNED session.

    All grading logic lives in main.grading.grade_submission - this route only
    loads the session, enforces request-level preconditions, applies the
    attempt/reveal policy, and strips private material from the response."""
    from main.grading import align_submission, grade_submission
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
        # Store what was GRADED, not what was typed. grade_submission re-seats a
        # submission at its chunk's indent depth (main/indent.py); storing the raw
        # text instead would put a flat answer into the accepted prefix and break
        # the NEXT step's assembly, one chunk after the mistake.
        accept_code = align_submission(session, req.student_code)
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

    # Permanent archive. Inert until sign-in exists (current_student() is None,
    # every writer short-circuits), and best-effort forever after: this runs
    # AFTER commit_outcome, so an archive outage can never cost a graded answer.
    # This is what makes the wrong attempts - the ones the session store drops
    # on expiry - answer "where do students go wrong".
    from main.archive import save_session_end, save_submission
    _student = current_student(request)
    save_submission(get_supabase() if _student else None, _student, session,
                    session["index"], state["attempts"],
                    req.student_code, {"verdict": result.verdict,
                                       "tier": result.tier,
                                       "deterministic": result.deterministic,
                                       "reason": result.student_reason})
    if state["completed"]:
        save_session_end(get_supabase() if _student else None, _student,
                         req.session_id, state)

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
    # teacher_name is gone: the author is the signed-in instructor, taken from
    # the session cookie in upload_assignment().


class ManualSplitRequest(BaseModel, extra="forbid"):
    slug: str
    header: str
    chunks: list[dict]


class ProblemRetryRequest(BaseModel, extra="forbid"):
    """One corrected problem, re-prepared on its own.

    `slug` and `assignment_id` identify the EXISTING row - the pair the problems
    table is unique on. `source` is the instructor's corrected Python for that
    one problem; the slug inside it, if any, is ignored (see the route)."""
    assignment_id: str
    slug: str
    source: str


@app.get("/assignment_template")
def assignment_template():
    """A starter file a teacher can download, edit and re-upload."""
    from main.assignments import TEMPLATE
    return {"filename": "assignment.py", "content": TEMPLATE}


@app.post("/teacher/assignments")
def upload_assignment(req: AssignmentUpload, request: Request):
    """Parse an assignment file and PREPARE every problem, streaming progress.

    Preparation (oracle generation, mutation validation, decomposition) is
    minutes of work, so this streams NDJSON as each problem finishes instead of
    holding the connection silent. Same one-directional pattern as
    /playground/live, delivered over POST because the file rides in the body."""
    import json as _json
    from fastapi.responses import StreamingResponse
    from main.assignments import AssignmentParseError, parse_assignment_file
    from main.publish import prepare_assignment_stream

    # Preparation is minutes of paid model work per problem, and it publishes
    # what students then see. Both are reasons this cannot be reachable by
    # anyone who typed "Instructor" into a radio button.
    teacher = require_teacher(request)

    try:
        parsed = parse_assignment_file(req.content, req.filename)
    except AssignmentParseError as e:
        raise HTTPException(status_code=400, detail={
            "reason_code": "unparseable_assignment", "message": str(e)})

    sb = get_supabase()
    # The signed-in instructor, not req.teacher_name - the browser used to name
    # its own author, so an assignment's owner was whatever it claimed to be.
    row = sb.table("assignments").insert({
        "name": parsed["name"], "teacher_name": teacher["username"],
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
                    # The block's own text, so the instructor can fix THIS
                    # problem in place. Stored empty before, which is why the
                    # only remedy for a bad problem used to be re-uploading the
                    # whole file - there was no copy of it anywhere.
                    "description": "", "solution": bad.get("source", ""),
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
def teacher_assignment_problems(assignment_id: str, request: Request):
    """TEACHER view. Every problem including the ones that failed, with reasons.

    Still no solution: the teacher already has it in their own file, and not
    sending it keeps one fewer copy of the answer key moving over the wire.
    The reasons themselves are instructor-only, so the route is role-gated -
    prepare_error quotes the source file back."""
    require_teacher(request)
    res = get_supabase().table("problems").select(
        _PUBLIC_PROBLEM_COLS + ", ready, prepare_error").eq(
        "assignment_id", assignment_id).execute()
    rows = res.data or []
    return {"problems": rows,
            "ready": sum(1 for r in rows if r["ready"]),
            "failed": sum(1 for r in rows if not r["ready"]),
            "count": len(rows)}


@app.get("/teacher/assignments/{assignment_id}/grades")
def teacher_grades(assignment_id: str, request: Request):
    """The grade sheet for one assignment: every student, one row each.

    Instructor-only, and it has to be - this hands back the whole class's
    standing, which is the one thing a student must never be able to read about
    anyone but themselves. The arithmetic lives in main/grades.py; this route
    only checks the role and passes the client through."""
    from main.grades import grade_sheet

    require_teacher(request)
    return grade_sheet(get_supabase(), assignment_id)


@app.get("/teacher/assignments/{assignment_id}/transcript/{student_id}")
def teacher_transcript(assignment_id: str, student_id: str, request: Request):
    """One student's whole record on one assignment, as a text file.

    Plain text on purpose: it is read once, next to a grade book, and a format
    that opens in any editor beats one that needs the app that wrote it.
    Content-Disposition is what makes the browser save it rather than render it
    in a tab the instructor then has to copy out of."""
    from fastapi.responses import PlainTextResponse
    from main.grades import transcript

    require_teacher(request)
    try:
        filename, body = transcript(get_supabase(), assignment_id, student_id)
    except LookupError:
        raise HTTPException(status_code=404, detail={
            "reason_code": "student_not_found",
            "message": "That student no longer exists."}) from None
    return PlainTextResponse(body, headers={
        "Content-Disposition": f'attachment; filename="{filename}"',
        # The filename is derived from a student name the student typed, so it
        # is quoted above and kept to alphanumerics by transcript() itself; a
        # header cannot carry a raw name safely.
        "Cache-Control": "no-store"})


@app.get("/teacher/problems/{slug}/source")
def teacher_problem_source(slug: str, assignment_id: str, request: Request):
    """One problem's own text, plus WHICH preparation gate it stopped at.

    Instructor-only, and the one route that deliberately returns `solution`:
    this is the teacher's own file coming back to them to correct, which is the
    whole point of the fix-and-retry panel. Every other route keeps it server-
    side. Keyed on (assignment_id, slug) because that pair - not slug alone - is
    what the problems table is unique on."""
    from main.publish import checklist, stage_of_error

    require_teacher(request)
    rows = get_supabase().table("problems").select(
        "slug, title, description, solution, ready, prepare_error").eq(
        "assignment_id", assignment_id).eq("slug", slug).execute().data
    if not rows:
        raise HTTPException(status_code=404, detail={
            "reason_code": "problem_not_found",
            "message": f"No problem '{slug}' in this assignment."})
    p = rows[0]
    stage = None if p["ready"] else stage_of_error(p.get("prepare_error"))
    return {"slug": p["slug"], "title": p["title"], "ready": bool(p["ready"]),
            "error": p.get("prepare_error"),
            "source": p.get("solution") or "",
            "checklist": checklist(stage, p.get("prepare_error"))}


@app.post("/teacher/problems/retry")
def teacher_problem_retry(req: ProblemRetryRequest, request: Request):
    """Re-prepare ONE problem from text the instructor just corrected.

    The same pipeline an upload runs (main/publish.prepare_problem), on one
    problem instead of twenty, so a fixed problem does not require re-uploading
    the file and re-preparing everything beside it. Slow for the same reason an
    upload is - it regenerates the oracle and the decomposition - so the page
    shows it as work in progress.

    The SLUG IS NOT TAKEN FROM THE EDITED TEXT. The instructor is correcting an
    existing problem, so its identity is the row's; letting the marker line
    rename it would silently create a second problem and leave the broken one
    in place."""
    from main.assignments import AssignmentParseError, parse_assignment_file
    from main.publish import checklist, prepare_problem

    # Minutes of paid model work, and it decides what students are served.
    require_teacher(request)

    def blocked(stage, message):
        """A refusal the teacher can act on - never a 4xx. The panel redraws its
        checklist from this exactly as it does from a finished run."""
        return {"ready": False, "stage": stage, "error": message,
                "checklist": checklist(stage, message)}

    # Parse first: a source that cannot be read costs nothing to reject, and
    # spending an oracle run to discover a missing docstring is pure waste.
    try:
        parsed = parse_assignment_file(req.source, f"{req.slug}.py")
    except AssignmentParseError as e:
        return blocked("parses", str(e))
    if parsed["problems"]:
        problem = parsed["problems"][0]
    elif parsed["errors"]:
        return blocked("parses", parsed["errors"][0]["error"])
    else:
        return blocked("parses", "no problem found in this text")
    if len(parsed["problems"]) > 1:
        return blocked("parses", "this is one problem's text - it defines "
                                 f"{len(parsed['problems'])} problems. Give "
                                 "each its own entry.")

    problem["slug"] = req.slug                     # identity is the row's
    result = prepare_problem(problem)

    sb = get_supabase()
    try:
        sb.table("problems").upsert({
            "slug": req.slug, "title": problem["title"],
            "description": problem["description"],
            "solution": problem["solution"],
            "assignment_id": req.assignment_id,
            "ready": bool(result["ready"]),
            "prepare_error": result.get("error")},
            on_conflict="assignment_id,slug").execute()
    except Exception as e:
        # Same rule as the upload path: a problem that did not SAVE is not
        # ready, whatever preparation decided. Reporting success over a missing
        # row is how an assignment shows a green badge and still serves nothing.
        return blocked("steps", f"prepared, but could not be saved: {e}"[:300])

    return {"ready": bool(result["ready"]), "stage": result.get("stage"),
            "error": result.get("error"), "chunks": result.get("chunks", 0),
            "n_tests": result.get("n_tests", 0),
            "checklist": checklist(result.get("stage"), result.get("error"))}


@app.post("/teacher/split")
def manual_split(req: ManualSplitRequest, request: Request):
    """Teacher-authored decomposition for a problem the model could not split.

    Goes through the SAME serve gate as a generated one - a hand-written split
    can still contain a step that does no work, which would let a student skip
    it and be marked correct."""
    from main.publish import save_manual_decomposition

    # This flips `ready` - it decides what students are served.
    require_teacher(request)
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



# ══════════════════════════════════════════════════════════════════════════
# SIGN-IN - username (PSU email) + password, against Supabase.
#
# Microsoft/Entra sign-in is ON HOLD until PSU IT issues Azure credentials;
# main/psu_auth.py keeps that work intact and unwired. The site itself sits
# behind the college VPN, which is what limits it to students and faculty;
# these routes are what tell one of them from another. See main/auth.py.

def _session_response(body: dict, student: dict) -> JSONResponse:
    """Answer with the account, and set the cookie that proves it.

    The id is in the body only so the UI can show it; every server-side use
    reads it back out of the signed cookie, which the page cannot forge and
    (HttpOnly) cannot even read."""
    from main.auth import cookie_kwargs, issue_session
    resp = JSONResponse(body)
    resp.set_cookie(auth_mod.SESSION_COOKIE, issue_session(student),
                    **cookie_kwargs(secure=_COOKIE_SECURE))
    return resp


def _account(student: dict) -> dict:
    return {"student_id": student["id"], "username": student["username"],
            "name": auth_mod.full_name(student),
            "role": student.get("role") or "student"}


@app.post("/register")
def register(req: RegisterRequest):
    """Create an account. Always as a student - see main/auth.py for why role
    is not something the browser gets to ask for."""
    _require_auth_configured()
    try:
        row = auth_mod.register_student(get_supabase(), req.username, req.password,
                                        req.first_name, req.last_name)
    except auth_mod.AuthError as e:
        if e.detail:
            print(f"  ⚠️  registration refused: {e.detail}")
        raise HTTPException(status_code=400, detail={
            "reason_code": "registration_refused", "message": str(e)})
    return _session_response(_account(row), row)


@app.post("/login")
def login(req: AuthRequest):
    _require_auth_configured()
    try:
        row = auth_mod.authenticate(get_supabase(), req.username, req.password)
    except auth_mod.RateLimited as e:
        # Before the AuthError clause: RateLimited subclasses it, so the order
        # of these two is what decides whether the limit is visible at all.
        print(f"  ⚠️  sign-in throttled: {e.detail}")
        raise HTTPException(status_code=429, detail={
            "reason_code": "too_many_attempts", "message": str(e)},
            headers={"Retry-After": str(e.retry_after)})
    except auth_mod.AuthError as e:
        raise HTTPException(status_code=401, detail={
            "reason_code": "bad_credentials", "message": str(e)})
    return _session_response(_account(row), row)


@app.post("/logout")
def logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(auth_mod.SESSION_COOKIE, path="/")
    return resp


@app.get("/auth/me")
def auth_me(request: Request):
    """Who the cookie says you are. The pages call this on load, so a cookie
    that expired mid-lab bounces to sign-in instead of failing later on a
    write the student thought had been saved."""
    claims = require_student(request)
    return {"student_id": claims["sub"], "username": claims["username"],
            "name": claims["name"], "role": claims.get("role", "student")}


@app.get("/solved")
def get_solved(request: Request):
    """The signed-in student's solves.

    Was /solved/{student_id}, which handed anyone else's progress to anyone
    who could type a uuid."""
    claims = require_student(request)
    result = get_supabase().table("solved").select("problem_slug").eq(
        "student_id", claims["sub"]).execute()
    return {"slugs": [r["problem_slug"] for r in (result.data or [])]}


@app.get("/history/{slug}")
def student_problem_history(slug: str, request: Request):
    """The signed-in student's own past work on one problem.

    main/archive.py has been WRITING this since sign-in landed - chat turns,
    designs, graph snapshots, every graded submission - and until now nothing
    read any of it back. So reopening a finished problem showed an empty chat,
    an empty plan graph and a locked editor, and the student reasonably
    concluded their work had been thrown away. It never was; there was simply
    no route to ask for it.

    Only ever the CALLER's own rows: student_id comes from the cookie, never
    from the query, so this cannot be pointed at a classmate. No reference
    solution and no chunk reference is read here, so replaying a transcript
    cannot leak an answer the student was not already given."""
    from main.archive import student_history
    from main.graphs import compare

    claims = require_student(request)
    sb = get_supabase()

    empty = {"slug": slug, "found": False, "solved": False,
             "design_approved": False, "messages": [], "plan": None,
             "code": None, "comparison": None}
    try:
        h = student_history(sb, claims["sub"], slug)
    except Exception as e:
        # A missing archive must never stop a problem from OPENING. Degrade to
        # "nothing recorded" - which is exactly how it behaved before this
        # route existed - rather than 500 the page that calls it.
        print(f"  \u26a0\ufe0f  history unavailable for {slug}: {str(e)[:160]}")
        return empty

    # Snapshots, newest wins: save_graph appends a row every time the plan
    # changes, and what the student wants back is the last one they saw.
    latest = {}
    for g in h.get("graphs") or []:
        if g.get("graph"):
            latest[g.get("kind")] = g["graph"]
    plan, code = latest.get("plan"), latest.get("code")

    msgs = [{"role": m["role"], "content": m["content"], "at": m.get("created_at")}
            for m in (h.get("messages") or [])
            if m.get("phase") == "tutor" and m.get("role") in ("user", "assistant")]

    solved = bool(sb.table("solved").select("problem_slug").eq(
        "student_id", claims["sub"]).eq("problem_slug", slug).execute().data)

    return {"slug": slug,
            "found": bool(msgs or plan or code or h.get("designs")),
            "solved": solved,
            # Approved ONCE is approved: the gate exists to make a student plan
            # before coding, and they already did that for this problem. Making
            # them re-upload the same diagram to reread their own finished work
            # would be a toll, not a lesson.
            "design_approved": any(d.get("approved") for d in (h.get("designs") or [])),
            "messages": msgs, "plan": plan, "code": code,
            # Recomputed rather than stored: compare() is deterministic and free,
            # and the snapshot rows hold the two graphs but not their diff.
            "comparison": compare(plan, code) if (plan and code) else None}


@app.post("/tutor_chat")
def tutor_chat(req: TutorChatRequest, request: Request):
    """Socratic tutor for the problem currently open on the student page.

    Deliberately NOT session-bound: it never grades, never advances a session,
    never consumes an attempt, and never sees a solution, a chunk reference or
    an oracle test. It only reads the public title/description the student can
    already see."""
    from main.tutor import MAX_TURNS, reply

    if len(req.messages) > MAX_TURNS * 2:
        raise HTTPException(status_code=400, detail={
            "reason_code": "conversation_too_long",
            "message": "This conversation is very long - start a fresh one."})

    row = get_supabase().table("problems").select(
        "slug, title, description").eq("slug", req.slug).execute().data
    if not row:
        raise HTTPException(status_code=404, detail={
            "reason_code": "problem_not_found",
            "message": f"Unknown problem '{req.slug}'."})

    try:
        out = reply(row[0], req.messages, req.chunk_prompt, req.design_ok)
    except Exception as e:
        # A tutor outage is not a judgement about the student.
        raise HTTPException(status_code=503, detail={
            "reason_code": "tutor_unavailable",
            "message": "The tutor is unavailable right now. Try again shortly.",
            "detail": str(e)[:120]})

    # Archive only the NEW turns - the student's last message and this reply.
    # The client resends the whole history every call, so writing all of it
    # would grow the transcript quadratically.
    from main.archive import save_messages
    _student = current_student(request)
    if _student:
        tail = [m for m in req.messages[-1:] if m.get("role") == "user"]
        save_messages(get_supabase(), _student, req.slug, "tutor",
                      tail + [{"role": "assistant", "content": out["reply"]}])
    return out


# ══════════════════════════════════════════════════════════════════════════
# MICROSOFT / ENTRA SIGN-IN - ON HOLD
#
# The full PKCE flow lives in main/psu_auth.py, written and self-tested. It is
# waiting on Azure credentials from PSU IT, which is not our schedule, so the
# live path is username + password (main/auth.py) behind the college VPN.
#
# When the credentials land it plugs in ABOVE, not here: /login and /register
# already set the session cookie every other route reads, so SSO only has to
# find or create the students row for the verified PSU address and call the
# same auth.issue_session(). current_student() and require_teacher() do not
# change. Usernames are already PSU emails, which is the same string Entra
# returns as `preferred_username` - so that lookup is a join, not a migration.
# ══════════════════════════════════════════════════════════════════════════


@app.post("/plan_graph")
def plan_graph_route(req: PlanGraphRequest):
    """Grow the student's plan graph from what they have said in chat.

    Same guarantee as /tutor_chat: public title/description only, never a
    solution. This is a drawing, never a gate - so unlike /grade_chunk it fails
    soft, returning the previous graph rather than an error the page has to
    handle mid-conversation."""
    from main.graphs import plan_graph

    row = get_supabase().table("problems").select(
        "slug, title, description").eq("slug", req.slug).execute().data
    if not row:
        raise HTTPException(status_code=404, detail={
            "reason_code": "problem_not_found",
            "message": f"Unknown problem '{req.slug}'."})
    # plan_graph() never raises by contract - a failed extraction returns the
    # graph that was already on screen.
    fresh = plan_graph(row[0], req.messages, req.current)
    # A graph read off the student's DRAWN design must not be silently replaced
    # by a thinner one scraped from chat prose. merge_plan keeps whichever
    # captured more of their plan.
    if (req.current or {}).get("meta", {}).get("source") == "design":
        from main.graphs import merge_plan
        return merge_plan(req.current, fresh)
    return fresh


@app.post("/graphs")
def graphs_route(req: GraphsRequest, request: Request):
    """The dual-graph payload: plan, code, and what differs between them.

    Uses session_snapshot rather than load_session because this is most useful
    on a COMPLETED session, which load_session deliberately refuses."""
    from main.graphs import build_both
    from main.sessions import session_snapshot

    session = session_snapshot(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail={
            "reason_code": "session_not_found",
            "message": "That session no longer exists."})
    out = build_both(session, req.plan)
    out["completed"] = session["state"] == "completed"

    # Snapshot both graphs as they stood at this moment. Snapshots, not an
    # update: a plan the student revised three times mid-problem is the finding,
    # and overwriting a single row would destroy the evidence of it.
    from main.archive import save_graph
    _student = current_student(request)
    if _student:
        for kind in ("plan", "code"):
            save_graph(get_supabase(), _student, session["slug"], kind,
                       out[kind], req.session_id)
    return out


@app.post("/design_review")
async def design_review(request: Request,
                        slug: str = Form(...),
                        history: str = Form("[]"),
                        design: UploadFile = File(...)):
    """Review a student's uploaded design before they may write any code.

    Same guarantee as /tutor_chat: the reviewer is handed only the public
    title/description, never the reference solution. Multipart rather than JSON
    because the payload is a file; `history` is the prior review conversation,
    JSON-encoded, so a resubmit is judged against what was asked last round."""
    from main.design_review import DesignRejected, review_design

    row = get_supabase().table("problems").select(
        "slug, title, description").eq("slug", slug).execute().data
    if not row:
        raise HTTPException(status_code=404, detail={
            "reason_code": "problem_not_found",
            "message": f"Unknown problem '{slug}'."})

    try:
        prior = json.loads(history)
        if not isinstance(prior, list):
            raise ValueError
    except Exception:
        prior = []

    try:
        blob = await design.read()
        out = review_design(row[0], blob, design.content_type or "", prior)
    except DesignRejected as e:
        # The upload itself was wrong - a validation message for the student,
        # not a judgement on their design, and no model call was made.
        raise HTTPException(status_code=400, detail={
            "reason_code": "design_rejected", "message": str(e)})
    except Exception as e:
        raise HTTPException(status_code=503, detail={
            "reason_code": "reviewer_unavailable",
            "message": "The design reviewer is unavailable right now. Try again "
                       "shortly.",
            "detail": str(e)[:120]})

    # On approval, read the plan graph off the DRAWING itself. Without this a
    # student who draws a careful flowchart and types little gets an empty plan
    # graph - punishing exactly the behaviour this gate exists to encourage.
    # Only on approval, so it is one extra vision call per problem, not per try.
    if out.get("approved"):
        from main.graphs import graph_from_design
        out["plan_graph"] = graph_from_design(row[0], blob,
                                              design.content_type or "")

    # The diagram itself goes to private object storage; only its path is kept
    # in the row. A rejected design is archived exactly like an approved one -
    # the rejected ones are where the teaching signal is.
    from main.archive import save_design, save_messages
    _student = current_student(request)
    if _student:
        save_design(get_supabase(), _student, slug, blob,
                    design.content_type or "", out)
        save_messages(get_supabase(), _student, slug, "design",
                      [{"role": "assistant", "content": out["reply"]}])
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


# POST /log_interaction is gone. It had no callers - /grade_chunk writes
# student_interactions itself, from the graded outcome - and it took student_id
# and `verdict` straight from the body, so any browser could file a passing
# attempt under any student's id.


# The pages are served from this app so the browser is same-origin with the
# API: the session cookie is sent on every fetch with no CORS involved, and
# SameSite=lax is enough. Mounted LAST because it claims "/" - anything
# declared after it would be shadowed by a 404 for a missing file.
#
# Behind the VPN this is the whole deployment: one uvicorn process, one origin,
# nothing else to configure. Put TLS in front of it and set MICROTUTOR_ENV to
# anything but "dev" so the cookie goes out Secure.
from pathlib import Path

from fastapi.staticfiles import StaticFiles

app.mount("/", StaticFiles(directory=Path(__file__).parent, html=True),
          name="frontend")
