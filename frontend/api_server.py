"""
api_server.py - FastAPI bridge between Next.js web UI and local LLM pipeline.
Place this file in your microprog_phase1/ folder and run:
    pip install fastapi uvicorn
    uvicorn api_server:app --port 8000 --reload
"""
from datetime import datetime, timezone

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
    # The plan the page has drawn from this chat. Sent so the tutor can put a
    # release past the REAL gate before promising anything - see /tutor_chat.
    plan: dict | None = None
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


class PlanSubmitRequest(BaseModel, extra="forbid"):
    """Submit the plan the PAGE drew from this student's chat, in place of an
    uploaded picture. Same gate, same bar - see main/design_review.review_plan_graph
    for why this is not a way around it."""
    slug: str
    graph: dict
    history: list[dict] = []
    messages: list[dict] = []


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
    # Playground knob overrides, keyed by PLAYGROUND_PARAMS keys. Values are
    # clamped server-side to each knob's registered bounds; unknown keys are
    # dropped. Omitting this runs the pipeline exactly as production does.
    params: dict | None = None


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

        # The module a METHOD lives in is loaded here, server-side, and never
        # taken from the request. context_prefix contains the teacher's
        # implementations of the class's other methods, so a browser allowed to
        # supply it is a browser that can read it back.
        from main.sessions import CONTEXT_FIELDS
        ctx = get_supabase().table("problems").select(
            "context, title, description").eq(
            "slug", req.slug).limit(1).execute().data
        if ctx:
            # THE DESCRIPTION IS PART OF THE ORACLE CACHE KEY (identity.py:56),
            # and it was being taken from the REQUEST. That makes the key depend
            # on a client value: a stale tab, or a description edited in the
            # database after a page load, moves the key, and the problem answers
            # "oracle_missing" for a reason nothing on screen can explain.
            # Verified a no-op on the current data - all 11 hash identically
            # either way - so this closes the hole without moving any key.
            if ctx[0].get("description") is not None:
                problem["description"] = ctx[0]["description"]
            if ctx[0].get("title"):
                problem["title"] = ctx[0]["title"]
            if isinstance(ctx[0].get("context"), dict):
                problem.update({k: v for k, v in ctx[0]["context"].items()
                                if k in CONTEXT_FIELDS})

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

        # Open the archive's spine row. create_session writes to the SERVER's
        # own SQLite store, which is where grading reads from; mt_sessions is
        # the durable record everything else joins against, and it was the one
        # writer of the five in main/archive.py that nothing ever called.
        #
        # Every consequence was silent. main/grades.step_counts reads
        # total_chunks from here for its denominator and found none, so a
        # finished assignment reported "nothing to grade". save_session_end
        # UPDATEs this row, so closing a session that was never opened wrote
        # nothing and raised nothing. And the grade sheet's "turned up" column
        # is a set built from these rows, so a student who worked for an hour
        # without submitting read as missing.
        from main.archive import save_session_start
        # No email: the cookie carries sub / username / name / role and no
        # address, and student_email is a denormalised convenience beside the
        # student_id that actually identifies the row. Writing the username
        # into a column named for an address would be worse than leaving it
        # null - it reads as fact to whoever queries it next.
        save_session_start(get_supabase(), claims["sub"],
                           {**public, "slug": req.slug,
                            "content_hash": content_hash(problem)})

        # NO oracle pre-warm here. get_oracle_tests() is the WRITE path: on a
        # miss it generates inputs and runs mutation testing, minutes of paid
        # work triggered by a student pressing Start. Preparation now happens
        # once at teacher-upload time (main/publish.py), and a problem that did
        # not survive it is never offered to a student in the first place.
        # THE STEPS ARE HELD BACK until the design is accepted (the
        # instructors' requirement). The count and each step's indent still
        # travel, so the page can show how many there are and size itself;
        # only the PROMPTS - which are the answer, split up - are withheld.
        # /session_steps hands them over once the gate is passed.
        if not _design_approved(claims["sub"], req.slug):
            public = {**public, "steps_locked": True,
                      "chunks": [{**c, "prompt": ""} for c in public["chunks"]]}
        return public          # session_id, decomposition_id, header, PUBLIC chunks
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Decomposition unavailable: {e}")


def _design_approved(student_id: str | None, slug: str) -> bool:
    """Has this student had a design accepted for this problem?

    The one fact the STEPS are gated on. Read from mt_designs rather than
    trusted from the browser: the step prompts are a decomposition of the
    answer, and "1. work out whether the stack is empty / 2. return it" read
    before designing is the shape of the solution, handed over. A page that
    merely hides them is a page whose network tab shows them."""
    if not student_id:
        return False
    try:
        rows = (get_supabase().table("mt_designs").select("approved")
                .eq("student_id", student_id).eq("slug", slug)
                .eq("approved", True).limit(1).execute().data)
        return bool(rows)
    except Exception:
        # An unreachable archive must not hand out the steps.
        return False


@app.get("/session_steps/{session_id}")
def session_steps(session_id: str, request: Request):
    """The step prompts, once the design gate has been passed.

    Called by the page the moment a design is approved. Re-checks the approval
    here rather than believing the caller."""
    from main.sessions import public_chunks, session_snapshot

    claims = require_student(request)
    snap = session_snapshot(session_id)
    if snap is None:
        raise HTTPException(status_code=404, detail={
            "reason_code": "session_not_found", "message": "Unknown session."})
    if snap.get("student_id") and snap["student_id"] != claims["sub"]:
        raise HTTPException(status_code=403, detail={
            "reason_code": "not_your_session",
            "message": "That session belongs to someone else."})
    if not _design_approved(claims["sub"], snap.get("slug", "")):
        raise HTTPException(status_code=403, detail={
            "reason_code": "design_not_approved",
            "message": "Your design has not been accepted yet."})
    return {"chunks": public_chunks(snap["chunks"]),
            "total_chunks": len(snap["chunks"])}


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
    elif (result.verdict == "incorrect" and MAX_ATTEMPTS is not None
          and session["attempts"] + 1 >= MAX_ATTEMPTS):
        # Only reachable when a limit is configured. MAX_ATTEMPTS is None by
        # default (main/sessions.py): the reference is never revealed, and a
        # student keeps their own attempt at every step.
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
    # ONE failing case, already rendered for a human (main/grading.failing_case).
    # The exception to "no failures": the suite stays hidden, but a student told
    # only "wrong on at least one case" has been given a shrug, not a hint. The
    # page keeps it behind a disclosure they have to open.
    if result.failing_case:
        body["failing_case"] = result.failing_case
    return body


# ── playground (read-only showcase) ───────────────────────────────────────
# These serve the step-through demo in frontend/playground.html. They REPLAY
# cached results and never trigger validation, decomposition, or any LLM call
# a demo must not stall for minutes on a click. The only things computed on read
# are the AST mutants and their knockouts, which are deterministic, model-free
# and sub-second; they are recomputed because evaluate_oracle's per-mutant
# breakdown is not currently persisted (only the aggregate rates are).

@app.get("/playground/problems")
def playground_problems(request: Request):
    """Every problem in Supabase, with its whole pipeline story in one row:
    did preparation succeed, where did it stop, is there a validated oracle,
    how many pooled decompositions exist. This is the troubleshooter's list -
    teacher-gated, because the page it feeds also shows ground truths."""
    require_teacher(request)
    from main.identity import content_hash
    from main.publish import stage_of_error
    from main.run_phase1 import _load_pool
    from tests.sandbox import _load_cache

    sb = get_supabase()
    rows = sb.table("problems").select(
        "slug, title, difficulty, description, solution, ready, prepare_error,"
        " assignment_id, context, group_slug, group_title, group_description"
        ).execute().data or []
    asg = {a["id"]: a["name"] for a in (sb.table("assignments").select(
        "id, name").execute().data or [])}
    cache = _load_cache()
    pool = _load_pool()

    out = []
    for p in rows:
        # Must include the class context: content_hash folds context_prefix and
        # context_suffix in, so hashing description+solution alone never matches
        # a method's cached verdict - every class problem read "no oracle yet".
        key = content_hash({"description": p.get("description") or "",
                            "solution": p.get("solution") or "",
                            **_context_of(p)})
        oracle = cache.get(key)
        # Three states, not two. "review" is a suite that may well be fine but
        # carries checks nothing could decide; calling that "weak" sends the
        # instructor rewriting a problem that might need no change at all.
        oracle_state = (None if not (isinstance(oracle, dict) and "strong" in oracle)
                        else "strong" if oracle["strong"]
                        # A one-line getter generates no mutants at all, so it
                        # can never be strong. Shown as its own state rather
                        # than as "weak": nothing about it is weak, and sending
                        # the instructor to rewrite a correct `return self.count`
                        # is the one thing this list must not do.
                        else "doctest" if oracle.get("status") == "doctest_verified"
                        else "review" if oracle.get("status") == "needs_review"
                        else "weak")
        out.append({
            "slug": p["slug"], "title": p.get("title") or p["slug"],
            "difficulty": p.get("difficulty"),
            "assignment": asg.get(p.get("assignment_id"), ""),
            "ready": bool(p.get("ready")),
            "prepare_error": p.get("prepare_error"),
            # Which gate preparation stopped at ("parses"/"runs"/"tests"/
            # "strength"/"steps"), or None once it passed - same mapping the
            # teacher's fix panel uses.
            "stage": None if p.get("ready") else stage_of_error(p.get("prepare_error")),
            "has_solution": bool((p.get("solution") or "").strip()),
            "oracle": oracle_state,
            "kill_rate_direct": (oracle or {}).get("kill_rate_direct")
                                if oracle_state else None,
            "kill_rate_lower": (oracle or {}).get("kill_rate_lower"),
            "kill_rate_upper": (oracle or {}).get("kill_rate_upper"),
            "undetermined": (oracle or {}).get("undetermined"),
            "pool_entries": len(pool.get(key) or []),
        })
    # Broken first - this list exists to find them.
    return {"problems": sorted(out, key=lambda r: (r["ready"], r["slug"]))}


@app.get("/playground/params")
def playground_params(request: Request):
    """The tunable-knob registry, straight from the pipeline. The page builds
    its sliders from this, so bounds and defaults have exactly one home
    (main/live_playground.PLAYGROUND_PARAMS)."""
    require_teacher(request)
    from main.live_playground import current_params
    return {"params": current_params()}


@app.get("/playground/problem/{slug}")
def playground_problem(slug: str, request: Request):
    """One problem WITH its ground truth, for the troubleshooter's left panel.
    Teacher-gated for exactly that reason - this is the one read path that
    hands a solution to a browser."""
    require_teacher(request)
    from main.identity import content_hash
    from main.run_phase1 import _load_pool
    from tests.sandbox import _load_cache

    row = get_supabase().table("problems").select(
        "slug, title, description, difficulty, solution, ready, prepare_error, "
        "context, group_slug, group_title, group_description"
    ).eq("slug", slug).execute().data
    if not row:
        raise HTTPException(status_code=404, detail=f"Problem '{slug}' not found.")
    p = row[0]
    key = content_hash({"description": p.get("description") or "",
                        "solution": p.get("solution") or "",
                        **_context_of(p)})
    oracle = _load_cache().get(key)
    if not (isinstance(oracle, dict) and "strong" in oracle):
        oracle = None
    return {"slug": p["slug"], "title": p.get("title") or p["slug"],
            "description": p.get("description") or "",
            "difficulty": p.get("difficulty"),
            "solution": p.get("solution") or "",
            "ready": bool(p.get("ready")),
            "prepare_error": p.get("prepare_error"),
            "oracle": None if oracle is None else {
                "strong": bool(oracle["strong"]),
                "kill_rate_direct": oracle.get("kill_rate_direct", 0.0),
                "n_tests": len(oracle.get("final_tests", [])),
                "validated_at": oracle.get("validated_at", "")},
            "pool_entries": len(_load_pool().get(key) or [])}


@app.get("/playground/{slug}")
def playground_detail(slug: str, request: Request):
    """One problem's full journey, assembled from cache. Never recomputes an
    oracle or a decomposition; returns nulls the UI renders as 'not yet'.

    Teacher-gated like every other playground route. It withholds `solution`,
    which is what made it look safe - but it returns the decomposition, and a
    decomposition carries every step's private reference answer. Ungated, the
    complete answer key was one guessed slug away from any signed-out visitor."""
    import json as _json
    import os as _os
    from main.identity import content_hash, get_resolved_entry
    from main.mutation import _disagrees, generate_mutants
    from tests.sandbox import _load_cache, passes_tests, run_solution

    require_teacher(request)
    row = get_supabase().table("problems").select(
        "slug, title, description, difficulty, solution, context, group_slug, "
        "group_title, group_description").eq("slug", slug).execute().data
    if not row:
        raise HTTPException(status_code=404, detail=f"Problem '{slug}' not found.")
    # get_resolved_entry and the mutant replay below both EXECUTE this problem,
    # so it has to carry its class context or a method resolves to a bare
    # `def pop(self)` that cannot run.
    problem = {**row[0], **_context_of(row[0])}
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
    if cached.get("status") == "doctest_verified":
        weak_reason = None                  # not weak - see oracle_store.certified
    elif not cached["strong"]:
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

@app.get("/teacher/prepare/live/{assignment_id}/{slug}")
def prepare_live_mirror(assignment_id: str, slug: str, request: Request):
    """Watch one problem being prepared by an upload that is ALREADY running.

    The mirror half of /playground/live. That route STARTS a pipeline run; this
    one attaches to the run an upload is in the middle of, replaying what has
    happened so far and then following along - see main/prepare_bus.py for why
    a second run would be the wrong answer (it costs a second set of model calls
    and races the first one to write the same oracle cache entry).

    Teacher-gated for the same reason /playground/live is: the transcript
    carries the ground truth, the oracle suite and every chunk reference, which
    together are the complete answer key."""
    import json as _json
    from fastapi.responses import StreamingResponse
    from main.prepare_bus import is_open, key, subscribe

    require_teacher(request)
    k = key(assignment_id, slug)
    if not is_open(k):
        raise HTTPException(status_code=404, detail={
            "reason_code": "no_run_to_watch",
            "message": (f"Nothing is preparing '{slug}' right now. A run can "
                        f"only be watched while it is happening - if the upload "
                        f"has finished, open the problem in the playground to "
                        f"run it again.")})

    def events():
        # None is a heartbeat: an idle connection has to put something on the
        # wire or a proxy closes it, and the readers already skip blank lines.
        for ev in subscribe(k):
            yield "\n" if ev is None else _json.dumps(ev) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/teacher/problems/{assignment_id}/{slug}/transcript")
def problem_transcript(assignment_id: str, slug: str, request: Request):
    """The saved narration of the run that blocked this problem.

    The after-the-fact half of /teacher/prepare/live: that one attaches to a run
    in progress, this one replays a run that has already finished. Only problems
    that did NOT pass have one - see main/transcripts.py.

    Teacher-gated like every other route that carries a transcript: it contains
    the ground truth, the oracle suite and the chunk references."""
    from main import transcripts

    require_teacher(request)
    saved = transcripts.load(assignment_id, slug)
    if not saved:
        raise HTTPException(status_code=404, detail={
            "reason_code": "no_transcript",
            "message": (f"No saved run for '{slug}'. Transcripts are kept only "
                        f"for problems that did not pass, and are cleared once "
                        f"a problem is fixed.")})
    return saved


@app.post("/playground/live")
def playground_live(req: LiveRunRequest, request: Request):
    from fastapi.responses import StreamingResponse

    # Teacher-gated: the stream narrates the ground truth, the oracle suite and
    # every chunk reference - the complete answer key for the problem.
    require_teacher(request)

    problem = {"slug": req.slug, "title": req.title or req.slug,
               "description": req.description or "",
               "solution": (req.solution or "").strip()}

    # Same contract as /decompose_chunks: curated problems keep their ground
    # truth in the DB; uploads carry it in the request. No ground truth = no
    # oracle = nothing to watch - hard error, not a degraded run.
    # The context blob is fetched UNCONDITIONALLY, not only when the solution is
    # missing. A method problem without it is not a degraded problem, it is the
    # WRONG problem: is_method() goes false, main/context.py stops assembling the
    # class around the method, and the entry point resolves to `pop` instead of
    # the sequence driver. Every class problem then dies at oracle generation
    # with "no usable test cases", because a bare `def pop(self)` cannot be run.
    row = get_supabase().table("problems").select(
        "slug, title, description, solution, context, group_slug, group_title, "
        "group_description").eq("slug", req.slug).execute().data
    if row:
        problem["solution"] = problem["solution"] or (row[0].get("solution") or "").strip()
        problem["description"] = problem["description"] or (row[0].get("description") or "")
        problem["title"] = req.title or row[0].get("title") or req.slug
        problem.update(_context_of(row[0]))
    if not problem["solution"]:
        raise HTTPException(
            status_code=400,
            detail=(f"No reference solution for '{req.slug}'. A live run cannot "
                    f"start without ground truth - oracle tests, mutation "
                    f"validation and Gate 1 all depend on it. Supply `solution` "
                    f"with the request."))

    from main.live_playground import ndjson_stream
    return StreamingResponse(
        ndjson_stream(problem, req.params),
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
_PUBLIC_PROBLEM_COLS = ("id, slug, title, difficulty, description, topic_tags,"
                        " group_slug, group_title, group_description,"
                        " group_order, member_order")
# `context` is NOT in that list and must never be added to it. It holds the
# module a method was carved out of, which includes the teacher's reference
# implementations of the class's OTHER methods - for Stack.pop it contains a
# finished push(). Serving it to a student hands them four of the five answers.


def _stored_problem(assignment_id, slug: str) -> dict | None:
    """The stored row for one problem, with its class context flattened back on.

    Returns None when there is no row or it is not a class problem - a plain
    function needs no context and its source is already a whole module."""
    try:
        row = get_supabase().table("problems").select(
            "slug, title, description, solution, context, group_slug, "
            "group_title, group_description").eq(
            "assignment_id", assignment_id).eq("slug", slug).execute().data
    except Exception:
        return None
    if not row:
        return None
    merged = {**row[0], **_context_of(row[0])}
    return merged if merged.get("context_prefix") else None


def _context_of(row: dict) -> dict:
    """The class-context fields from a stored problem row, flattened back onto
    the problem dict main/context.py expects.

    The inverse of _group_columns. It exists because reading a method problem
    back WITHOUT these is silently catastrophic rather than merely lossy: the
    problem still looks valid, so nothing errors - it just stops being a method
    and starts being an unrunnable bare function. Any route that loads a problem
    for execution must go through here."""
    from main.sessions import context_of
    return context_of(row)


def _group_columns(problem: dict) -> dict:
    """The class-group columns for one problem row, or all-null for a plain
    function. See migrations/007_class_groups.sql for why the grouping is
    columns and the context is one jsonb."""
    from main.sessions import CONTEXT_FIELDS
    context = {k: problem[k] for k in CONTEXT_FIELDS if k in problem}
    return {"group_slug": problem.get("group_slug"),
            "group_title": problem.get("group_title"),
            "group_description": problem.get("group_description"),
            "group_order": problem.get("group_order"),
            "member_order": problem.get("member_order"),
            "context": context or None}


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


class ProblemAcceptRequest(BaseModel, extra="forbid"):
    """An instructor taking responsibility for the checks nothing could decide."""
    assignment_id: str
    slug: str


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

    # One watchable channel per problem, so a teacher can open any row in a new
    # tab and watch the pipeline reason about that one problem. The upload owns
    # channel lifetime; main/publish.py knows only that it was handed an `emit`.
    from main.prepare_bus import finish as bus_finish, key as bus_key
    from main.prepare_bus import open_channel, publish as bus_publish
    from main import transcripts
    opened: list[str] = []
    # The same events, kept per problem so a run that did NOT pass can be
    # replayed later. The bus is live-only and dies with the request, which is
    # why "watch" was useless the moment an upload finished - and an hour later
    # is exactly when an instructor sits down to fix things.
    taped: dict[str, list] = {}

    def emit_for(slug: str):
        k = bus_key(assignment_id, slug)
        open_channel(k)
        opened.append(k)
        tape = taped.setdefault(slug, [])

        def emit(ev):
            tape.append(ev)
            bus_publish(k, ev)
        return emit

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

        try:
            for ev in prepare_assignment_stream(parsed["problems"],
                                                emit_for=emit_for):
                if ev.get("event") == "prepared":
                    # A finished problem releases its watchers. Without this, a
                    # tab that opened mid-run sits on an open connection for
                    # ever: the stream's only other ending is the whole run.
                    bus_finish(bus_key(assignment_id, ev.get("slug", "?")))
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
                                "prepare_error": ev.get("error"),
                                **_group_columns(src)},
                                on_conflict="assignment_id,slug").execute()
                        except Exception as e:
                            # A problem that did not SAVE is not ready, whatever
                            # preparation decided. Reporting it as ready while
                            # the row is missing is exactly how an assignment
                            # showed green badges and still sat at 0 / 0 - the
                            # failure has to reach the teacher, not a swallowed
                            # warning field.
                            ev = {**ev, "ready": False,
                                  "error": f"prepared, but could not be saved: {e}"[:300]}
                    # AFTER the save, so `ready` here is the outcome actually
                    # recorded - a problem that prepared but failed to store is
                    # not ready, and its transcript has to be kept too.
                    try:
                        transcripts.record(assignment_id, ev.get("slug", "?"),
                                           taped.pop(ev.get("slug", "?"), []), ev)
                    except Exception:
                        pass          # evidence is a nicety; never fail an upload for it
                yield _json.dumps(ev) + "\n"
        finally:
            # A disconnect or a crash must not leave watchers hanging on a run
            # that is never going to send them anything again.
            for k in opened:
                bus_finish(k)

    return StreamingResponse(events(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/teacher/assignments/{assignment_id}/reprepare")
def reprepare_assignment(assignment_id: str, request: Request,
                         scope: str = "all"):
    """Prepare an existing assignment again, from what is already stored - no
    file, no re-upload. scope="all" does every problem; scope="blocked" does
    only the ones that did not come out ready.

    The rows carry the solution AND the class context, which is everything
    preparation needs. Asking for the .py file again was a leftover from when
    they did not: the server kept the parsed problems, not the file, so the
    only way back was to make the instructor find it.

    Streams the same events an upload does, so the page renders it with the same
    code. Existing decompositions for these problems are dropped first: the
    point of re-preparing is to get new ones, and serving a pooled split from
    before the change would quietly defeat that."""
    import json as _json
    from fastapi.responses import StreamingResponse
    from main.publish import prepare_assignment_stream
    from main.prepare_bus import finish as bus_finish, key as bus_key
    from main.prepare_bus import open_channel, publish as bus_publish
    from main.identity import content_hash
    from main.run_phase1 import _load_pool, _save_pool
    from main import transcripts

    require_teacher(request)
    sb = get_supabase()
    rows = sb.table("problems").select(
        "slug, title, description, solution, ready, prepare_error, context, "
        "group_slug, group_title, group_description").eq(
        "assignment_id", assignment_id).execute().data or []
    if not rows:
        raise HTTPException(status_code=404, detail={
            "reason_code": "assignment_empty",
            "message": "This assignment has no problems stored to prepare."})

    # Re-splitting is the slow part of preparation, and a problem that is
    # already ready gets the same answer for it. After a fix aimed at the ones
    # that failed, doing only those is the whole point.
    if scope == "blocked":
        rows = [r for r in rows if not r.get("ready")]
        if not rows:
            raise HTTPException(status_code=409, detail={
                "reason_code": "nothing_blocked",
                "message": ("Every problem in this assignment is already "
                            "ready, so there is nothing to re-prepare.")})

    problems = [{**r, **_context_of(r)} for r in rows]
    problems = [p for p in problems if (p.get("solution") or "").strip()]
    if not problems:
        raise HTTPException(status_code=409, detail={
            "reason_code": "no_solutions",
            "message": ("None of these problems has a stored solution, so there "
                        "is nothing to prepare. Upload the .py file instead.")})

    # Drop the old splits for exactly these problems. Verdicts are keyed by
    # content and stay - re-running mutation testing on unchanged text would
    # cost minutes and reach the same answer.
    pool = _load_pool()
    for pr in problems:
        pool.pop(content_hash(pr), None)
    _save_pool(pool)

    opened: list[str] = []
    taped: dict[str, list] = {}

    def emit_for(slug: str):
        k = bus_key(assignment_id, slug)
        open_channel(k)
        opened.append(k)
        tape = taped.setdefault(slug, [])

        def emit(ev):
            tape.append(ev)
            bus_publish(k, ev)
        return emit

    def events():
        yield _json.dumps({"event": "parsed", "assignment_id": assignment_id,
                           "name": "", "n_problems": len(problems),
                           "parse_errors": []}) + "\n"
        try:
            for ev in prepare_assignment_stream(problems, emit_for=emit_for):
                if ev.get("event") == "prepared":
                    bus_finish(bus_key(assignment_id, ev.get("slug", "?")))
                    src = next((x for x in problems
                                if x["slug"] == ev["slug"]), None)
                    if src:
                        try:
                            sb.table("problems").upsert({
                                "slug": src["slug"],
                                "title": src.get("title") or src["slug"],
                                "description": src.get("description") or "",
                                "solution": src.get("solution") or "",
                                "assignment_id": assignment_id,
                                "ready": bool(ev["ready"]),
                                "prepare_error": ev.get("error")},
                                on_conflict="assignment_id,slug").execute()
                        except Exception as e:
                            ev = {**ev, "ready": False,
                                  "error": f"prepared, but could not be saved: {e}"[:300]}
                    try:
                        transcripts.record(assignment_id, ev.get("slug", "?"),
                                           taped.pop(ev.get("slug", "?"), []), ev)
                    except Exception:
                        pass
                yield _json.dumps(ev) + "\n"
        finally:
            for k in opened:
                bus_finish(k)

    return StreamingResponse(events(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


def _is_published(assignment_id) -> bool:
    """Is this assignment visible to students? True when the column is absent,
    so a database that has not run the migration behaves exactly as before."""
    try:
        row = get_supabase().table("assignments").select("published").eq(
            "id", assignment_id).execute().data
    except Exception:
        return True                    # no column yet - nothing is hidden
    return bool(row and row[0].get("published", True) is not False)


class PublishToggleRequest(BaseModel, extra="forbid"):
    published: bool


@app.post("/teacher/assignments/{assignment_id}/published")
def set_assignment_published(assignment_id: str, req: PublishToggleRequest,
                             request: Request):
    """Show or hide a whole assignment from students, instantly.

    Deliberately NOT the same thing as un-preparing it. Every verdict, every
    decomposition and every cached oracle survives, so this is reversible in one
    click - which is what makes it usable as an emergency stop when something
    looks wrong mid-term."""
    require_teacher(request)
    try:
        get_supabase().table("assignments").update(
            {"published": bool(req.published)}).eq("id", assignment_id).execute()
    except Exception as e:
        raise HTTPException(status_code=409, detail={
            "reason_code": "no_published_column",
            "message": (f"Could not change visibility: {e}. If the `published` "
                        f"column has not been added to the assignments table "
                        f"yet, run the migration first.")[:300]})
    return {"assignment_id": assignment_id, "published": bool(req.published)}


@app.get("/assignments")
def list_assignments():
    """Assignments with a ready-count. Safe for students AND teachers."""
    sb = get_supabase()
    rows = sb.table("assignments").select(
        "id, name, teacher_name, created_at, published").order(
        "created_at", desc=True).execute().data or []
    probs = sb.table("problems").select(
        "assignment_id, ready").not_.is_("assignment_id", "null").execute().data or []
    counts = {}
    for p in probs:
        c = counts.setdefault(p["assignment_id"], {"total": 0, "ready": 0})
        c["total"] += 1
        c["ready"] += 1 if p["ready"] else 0
    return {"assignments": [
        # `published` defaults to True for rows written before the column
        # existed - an assignment that was visible must not vanish because a
        # migration ran.
        {**a, "published": a.get("published", True) is not False,
         **counts.get(a["id"], {"total": 0, "ready": 0})} for a in rows]}


@app.get("/assignments/{assignment_id}/problems")
def assignment_problems(assignment_id: str):
    """STUDENT view. Ready problems only, public columns only.

    No solution, no prepare_error, and nothing that isn't ready - a student must
    never be handed a problem the grader cannot actually grade."""
    sb = get_supabase()
    # An UNPUBLISHED assignment shows a student nothing, whatever its problems
    # say. The flag is the instructor's emergency stop: it hides the whole set
    # instantly without destroying a single verdict or decomposition, so
    # publishing again is a flag flip rather than another hour of preparation.
    if not _is_published(assignment_id):
        return {"problems": [], "count": 0, "published": False}
    res = sb.table("problems").select(
        _PUBLIC_PROBLEM_COLS).eq("assignment_id", assignment_id).eq(
        "ready", True).execute()
    return {"problems": res.data or [], "count": len(res.data or []),
            "published": True}


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


def _findings_for(row: dict) -> dict | None:
    """Which specific checks decided this problem's verdict, in the instructor's
    terms - or None when it was never validated.

    A "check" is one deliberate single-line edit to the solution. Caught means a
    test noticed; inconclusive means nothing could separate the two programs and
    a person has to look. The line number is relative to the problem's own text,
    which is exactly what the fix panel puts in its editor."""
    import re
    import re as _re
    from main.identity import content_hash
    from tests.sandbox import _load_cache

    entry = _load_cache().get(content_hash(
        {"description": row.get("description") or "",
         "solution": row.get("solution") or "",
         **_context_of(row)}))
    if not (isinstance(entry, dict) and "strong" in entry):
        return None
    checks = ((entry.get("breakdown") or {}).get("mutants")) or []

    def one(m):
        label = m.get("label") or ""
        hit = _re.match(r"\s*line\s+(\d+)\s*:\s*(.*)", label)
        what = (hit.group(2) if hit else label).strip()
        # The stored labels are written for the engine ("remove `x = 1`",
        # "is -> is not"), and an instructor reads "remove ..." as an
        # instruction to go and remove it. Say what was DONE, in the past
        # tense, so it reads as a report rather than a suggestion.
        edit = re.sub(r"^remove\s+", "", what)
        if what.startswith("remove "):
            phrasing = f"deleting {edit}"
        elif "->" in what:
            a, b = [x.strip() for x in what.split("->", 1)]
            phrasing = f"changing {a} to {b}"
        else:
            phrasing = f"changing {what}"
        pr = m.get("probe") or {}
        return {"line": int(hit.group(1)) if hit else None,
                "what": what, "phrasing": phrasing, "label": label,
                # Which of the possible meanings the evidence points at, when
                # there is evidence. Absent for verdicts written before this was
                # persisted - the UI then lists the possibilities unnarrowed.
                "probe": pr.get("verdict"),
                "reached": pr.get("reached"),
                "differed": pr.get("differed"),
                "settled": m.get("status") in ("killed", "killed_on_retry")}

    return {"status": entry.get("status") or
                      ("strong" if entry.get("strong") else "weak"),
            "lower": entry.get("kill_rate_lower"),
            "upper": entry.get("kill_rate_upper"),
            "n_tests": len(entry.get("final_tests") or []),
            "checks": [one(m) for m in checks]}


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
        "slug, title, description, solution, ready, prepare_error, context, "
        "group_slug, group_title, group_description").eq(
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
            "checklist": checklist(stage, p.get("prepare_error")),
            # The per-check detail, read from the STORED VERDICT rather than
            # from the run's narration. It has to come from here: a re-run whose
            # verdict is already cached skips validation entirely and therefore
            # emits no mutant events at all, which is exactly how an instructor
            # ended up staring at "a few checks came back inconclusive" with no
            # way to find out which ones.
            "findings": _findings_for(p)}


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
    from main import context
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
        # A CLASS problem is stored as a bare method, and parsing a bare method
        # yields a plain function: no class around it, entry point `pop` instead
        # of the sequence driver, and a solution that cannot run at all - which
        # surfaced as "no usable test cases" on every retry of a method. Splice
        # the edited method back into its class first, so the retry re-parses
        # the same shape the upload did.
        stored = _stored_problem(req.assignment_id, req.slug)
        module = context.module_with_method(stored, req.source) if stored else None
        parsed = parse_assignment_file(module or req.source, f"{req.slug}.py")
    except AssignmentParseError as e:
        return blocked("parses", str(e))

    if module:
        # The spliced module re-parses the WHOLE class, so pick this problem out
        # by slug rather than taking the first of eleven.
        problem = next((p for p in parsed["problems"] if p["slug"] == req.slug), None)
        if problem is None:
            return blocked("parses", parsed["errors"][0]["error"] if parsed["errors"]
                           else "that text no longer defines this problem")
    elif parsed["problems"]:
        problem = parsed["problems"][0]
        if len(parsed["problems"]) > 1:
            return blocked("parses", "this is one problem's text - it defines "
                                     f"{len(parsed['problems'])} problems. Give "
                                     "each its own entry.")
    elif parsed["errors"]:
        return blocked("parses", parsed["errors"][0]["error"])
    else:
        return blocked("parses", "no problem found in this text")

    problem["slug"] = req.slug                     # identity is the row's
    # Taped for the same reason the upload tapes: if this retry ALSO does not
    # pass, the instructor needs the new evidence, not the run from an hour ago.
    # A retry that succeeds clears the old transcript - transcripts.record does
    # that itself when `ready` is true.
    from main import transcripts
    tape: list = []
    result = prepare_problem(problem, emit=tape.append)
    try:
        transcripts.record(req.assignment_id, req.slug, tape, result)
    except Exception:
        pass

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


@app.post("/teacher/problems/accept")
def teacher_problem_accept(req: ProblemAcceptRequest, request: Request):
    """Publish a NEEDS-REVIEW problem anyway, on the instructor's judgement.

    The one place a person can overrule the strength gate, and it is deliberately
    narrow:

      * only from `needs_review`. A `weak` verdict fails even in its best case,
        so there is no judgement to make and this refuses it.
      * the acceptance is NAMED and dated in the oracle cache, and `strong` stays
        false. A person agreeing with a measurement does not change the
        measurement; what changed is who is answerable for it.
      * everything after the strength gate still runs. A needs-review problem was
        never decomposed - preparation stops before it - so this re-runs
        preparation, and the decomposition and its necessity gate must still
        pass on their own. Accepting the tests is not accepting anything else.
    """
    from main.identity import content_hash
    from main.oracle_store import load_cache, save_cache
    from main.publish import checklist, prepare_problem
    from main import transcripts
    from datetime import datetime, timezone

    teacher = require_teacher(request)

    stored = _stored_problem(req.assignment_id, req.slug)
    if stored is None:
        row = get_supabase().table("problems").select(
            "slug, title, description, solution, context, group_slug, "
            "group_title, group_description").eq(
            "assignment_id", req.assignment_id).eq("slug", req.slug).execute().data
        if not row:
            raise HTTPException(status_code=404, detail={
                "reason_code": "problem_not_found",
                "message": f"No problem '{req.slug}' in this assignment."})
        stored = {**row[0], **_context_of(row[0])}

    cache = load_cache()
    key = content_hash(stored)
    entry = cache.get(key)
    if not (isinstance(entry, dict) and "strong" in entry):
        raise HTTPException(status_code=409, detail={
            "reason_code": "not_validated",
            "message": (f"'{req.slug}' has no verdict to accept yet. Prepare it "
                        f"first.")})
    if entry.get("status") != "needs_review":
        raise HTTPException(status_code=409, detail={
            "reason_code": "not_reviewable",
            "message": (f"'{req.slug}' is '{entry.get('status')}', not waiting on "
                        f"review. Only an inconclusive result can be accepted - "
                        f"a weak one fails even in the best case, so there is "
                        f"nothing to judge.")})

    cache[key] = {**entry,
                  "accepted_by": teacher.get("username") or "instructor",
                  "accepted_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    save_cache(cache)

    tape: list = []
    result = prepare_problem(stored, emit=tape.append)
    try:
        transcripts.record(req.assignment_id, req.slug, tape, result)
    except Exception:
        pass

    try:
        get_supabase().table("problems").upsert({
            "slug": req.slug, "title": stored.get("title") or req.slug,
            "description": stored.get("description") or "",
            "solution": stored.get("solution") or "",
            "assignment_id": req.assignment_id,
            "ready": bool(result["ready"]),
            "prepare_error": result.get("error")},
            on_conflict="assignment_id,slug").execute()
    except Exception as e:
        return {"ready": False, "stage": "steps",
                "error": f"accepted, but could not be saved: {e}"[:300],
                "checklist": checklist("steps", "could not be saved")}

    return {"ready": bool(result["ready"]), "stage": result.get("stage"),
            "error": result.get("error"), "chunks": result.get("chunks", 0),
            "n_tests": result.get("n_tests", 0),
            "accepted_by": cache[key]["accepted_by"],
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
        "slug, title, description, solution, context, group_slug, group_title, "
        "group_description").eq("slug", req.slug).execute().data
    if not row:
        raise HTTPException(status_code=404, detail=f"Problem '{req.slug}' not found.")
    # save_manual_decomposition runs assert_serveable, which executes the
    # problem against its oracle - so the class context has to ride along.
    row = [{**row[0], **_context_of(row[0])}]
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
    sb = get_supabase()
    done = {r["problem_slug"] for r in
            (sb.table("solved").select("problem_slug")
             .eq("student_id", claims["sub"]).execute().data or [])}

    # ...and every session that actually FINISHED. /mark_solved writes the
    # `solved` table ONLY for an independent solve, so a problem completed with
    # a shown answer was recorded nowhere: the list held it at "In progress"
    # forever and the assignment bar never moved, one screen after the app had
    # told the student it was "recorded as solved with help". Nothing was.
    #
    # Read here rather than widening `solved`, because mt_sessions already
    # carries the distinction (migrations/004_archive.sql) and the two claims
    # are different: `solved` means they did it themselves, and that is what
    # the independent list must keep meaning.
    #
    # It cannot inflate a grade. The grade sheet counts passing SUBMISSIONS
    # (main/grades.tally) and never reads either of these.
    assisted, opened, last_slug = set(), set(), None
    try:
        rows = (sb.table("mt_sessions")
                .select("slug, solved_independently, completed_at, started_at")
                .eq("student_id", claims["sub"])
                .order("started_at", desc=True).execute().data or [])
        # The most recently OPENED problem, for the "Continue where you left
        # off" affordance on the assignment list. That used to come from a
        # timestamp in localStorage; started_at has always been the same fact,
        # recorded server-side.
        last_slug = rows[0]["slug"] if rows else None
        for r in rows:
            opened.add(r["slug"])
            if r.get("completed_at"):
                (done if r.get("solved_independently") else assisted).add(r["slug"])
    except Exception as e:
        # A student's progress list must still render if the archive is
        # unreachable - it degrades to the independent solves it always showed.
        print(f"  ⚠️  /solved: could not read mt_sessions: {str(e)[:160]}")
    # `opened` is what "In progress" now means, and it comes from HERE rather
    # than from a note in localStorage. The old note was per-browser: it
    # survived a server wipe (so a cleared account still showed work in
    # progress), it did not follow a student to another machine, and it was
    # invisible to the instructor. mt_sessions has recorded every problem
    # opened since sign-in landed; nothing new is stored to make this work.
    return {"slugs": sorted(done), "assisted": sorted(assisted - done),
            "opened": sorted(opened - done - assisted), "last_slug": last_slug}


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

    # WHERE THIS STUDENT'S HISTORY BEGINS. A restart writes a marker rather
    # than deleting anything (see /problems/{slug}/restart), so everything
    # before the newest marker is still archived for the instructor and simply
    # not replayed to the student.
    since = max((m.get("created_at") or "") for m in (h.get("messages") or [])
                if m.get("phase") == "restart") if any(
        m.get("phase") == "restart" for m in (h.get("messages") or [])) else ""

    def _after(row):
        return not since or (row.get("created_at") or "") > since

    # Snapshots, newest wins: save_graph appends a row every time the plan
    # changes, and what the student wants back is the last one they saw.
    latest = {}
    for g in h.get("graphs") or []:
        if g.get("graph") and _after(g):
            latest[g.get("kind")] = g["graph"]
    plan, code = latest.get("plan"), latest.get("code")

    msgs = [{"role": m["role"], "content": m["content"], "at": m.get("created_at")}
            for m in (h.get("messages") or [])
            if m.get("phase") == "tutor" and m.get("role") in ("user", "assistant")
            and _after(m)]

    solved = bool(sb.table("solved").select("problem_slug").eq(
        "student_id", claims["sub"]).eq("problem_slug", slug).execute().data)

    return {"slug": slug,
            "found": bool(msgs or plan or code
                          or [d for d in (h.get("designs") or []) if _after(d)]),
            "solved": solved,
            # Approved ONCE is approved: the gate exists to make a student plan
            # before coding, and they already did that for this problem. Making
            # them re-upload the same diagram to reread their own finished work
            # would be a toll, not a lesson.
            "design_approved": any(d.get("approved") for d in (h.get("designs") or [])
                                   if _after(d)),
            "messages": msgs, "plan": plan, "code": code,
            # Recomputed rather than stored: compare() is deterministic and free,
            # and the snapshot rows hold the two graphs but not their diff.
            "comparison": compare(plan, code) if (plan and code) else None}


@app.post("/problems/{slug}/restart")
def restart_problem(slug: str, request: Request):
    """Start this problem over, as if the student had never opened it.

    NOTHING IS DELETED. The old conversation, designs and graphs stay in the
    archive and still appear in the instructor's transcript - a student who
    went round three times is the finding, and a restart that erased it would
    hide exactly the thing worth seeing. What changes is where the STUDENT's
    history begins: a marker row is written, and /history only replays what
    came after the newest one.

    The marker is an mt_messages row with phase='restart'. A third phase needs
    no migration, and /history already keeps only phase='tutor' rows, so the
    marker can never surface as a chat bubble."""
    claims = require_student(request)
    sb = get_supabase()
    row = sb.table("problems").select("slug").eq("slug", slug).limit(1).execute().data
    if not row:
        raise HTTPException(status_code=404, detail={
            "reason_code": "problem_not_found",
            "message": f"Unknown problem '{slug}'."})
    try:
        sb.table("mt_messages").insert({
            "session_id": None, "student_id": claims["sub"], "slug": slug,
            "phase": "restart", "role": "system",
            "content": "student restarted this problem",
            "created_at": datetime.now(timezone.utc).isoformat()}).execute()
    except Exception as e:
        print(f"  ⚠️  restart failed for {slug}: {type(e).__name__}: {str(e)[:200]}")
        raise HTTPException(status_code=503, detail={
            "reason_code": "restart_failed",
            "message": "Could not restart this problem. Try again.",
            "detail": f"{type(e).__name__}: {str(e)[:100]}"})
    return {"slug": slug, "restarted": True}


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

    # ── THE TUTOR DOES NOT GET TO PROMISE WHAT THE GATE WILL SAY ──────────
    #
    # Twice now a student was told "sounds like you have a plan", submitted it,
    # and was rejected seconds later - once for an off-by-one, once for doing
    # the lowercase/strip AFTER the loop. Two graders reading the same plan and
    # disagreeing is not something a shared rubric fixed, because they were
    # never reading it at the same moment.
    #
    # So before the tutor releases anyone, the ACTUAL gate reviews the plan the
    # page has drawn, and if it objects the student hears that objection now -
    # from the tutor, in the chat, while they are still thinking - instead of
    # after a submission. One judgement, delivered once.
    #
    # Costs one extra call, and only on the turn that would have released them.
    # Skipped when the page sent no plan, or when the gate is already open.
    if out.get("ready") and not req.design_ok and (req.plan or {}).get("nodes"):
        try:
            from main.design_review import review_plan_graph
            verdict = review_plan_graph(row[0], req.plan, [],
                                        chat_log=req.messages)
            if not verdict.get("approved"):
                out = {**out, "ready": False, "reply": verdict["reply"],
                       "held_by_review": True}
        except Exception:
            # The gate being unreachable must not strand a student mid-chat.
            # They keep the tutor's reply; the real gate still runs on submit.
            pass

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
def plan_graph_route(req: PlanGraphRequest, request: Request):
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
        fresh = merge_plan(req.current, fresh)

    # PERSIST IT. save_graph was reached from exactly one place - /graphs, which
    # only runs when a student FINISHES a problem - so the plan of anyone who
    # stopped partway was never written down. /history then found no snapshot
    # and the graph came back empty on reopen, which read as the work having
    # been thrown away. It is a snapshot per change by design (main/archive.py):
    # a plan revised three times is the finding, not noise.
    _student = current_student(request)
    if _student and fresh.get("nodes"):
        from main.archive import save_graph
        prev = [n.get("id") for n in ((req.current or {}).get("nodes") or [])]
        if [n.get("id") for n in fresh["nodes"]] != prev:   # only real changes
            save_graph(get_supabase(), _student, req.slug, "plan", fresh)
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
                        chat: str = Form("[]"),
                        design: UploadFile = File(...)):
    """Review a student's uploaded design before they may write any code.

    Same guarantee as /tutor_chat: the reviewer is handed only the public
    title/description, never the reference solution. Multipart rather than JSON
    because the payload is a file; `history` is the prior review conversation,
    JSON-encoded, so a resubmit is judged against what was asked last round.

    `chat` is the TUTOR conversation, so the reviewer judges the design together
    with what the student has already explained. Without it the two graders
    apply two different bars to one plan: the tutor tells a student their plan
    is workable and sends them to draw it, and the reviewer - which had never
    seen that conversation - sends them back for a step they explained in the
    message above."""
    from main.design_review import DesignRejected, review_design

    row = get_supabase().table("problems").select(
        "slug, title, description").eq("slug", slug).execute().data
    if not row:
        raise HTTPException(status_code=404, detail={
            "reason_code": "problem_not_found",
            "message": f"Unknown problem '{slug}'."})

    def _turns(raw):
        """Client-supplied transcript, or nothing. Never a 4xx: a malformed
        history must degrade to a cold review, not refuse the upload."""
        try:
            got = json.loads(raw)
            return got if isinstance(got, list) else []
        except Exception:
            return []

    prior, tutor_chat = _turns(history), _turns(chat)

    try:
        blob = await design.read()
        out = review_design(row[0], blob, design.content_type or "", prior,
                            chat_log=tutor_chat)
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


@app.get("/assignments/{assignment_id}/handback")
def assignment_handback(assignment_id: str, request: Request):
    """The student's own copy of the assignment file, with their answers in it.

    What they were handed, completed - same classes, same docstrings, same
    helper code, their body under each `def` at the right indent. It is a
    RUNNABLE file, so any problem they have not finished keeps the teacher's
    original body rather than a hole; the header at the top names which is
    which, so nothing shown is passed off as their own work.

    Their OWN work only: student_id comes from the cookie, so this cannot be
    pointed at a classmate. It reads no chunk references and no oracle, and a
    problem the student never opened contributes the same text the assignment
    already handed them - so this discloses nothing they were not given."""
    from fastapi.responses import Response

    from main.handback import build_handback
    from main.sessions import completed_answers

    claims = require_student(request)
    sb = get_supabase()

    asg = sb.table("assignments").select("id, name, source_file").eq(
        "id", assignment_id).limit(1).execute().data
    if not asg:
        raise HTTPException(status_code=404, detail={
            "reason_code": "assignment_not_found",
            "message": "That assignment does not exist."})

    rows = sb.table("problems").select(
        "slug, title, description, solution, context, group_title, "
        "group_order, member_order").eq("assignment_id", assignment_id).execute().data or []
    if not rows:
        raise HTTPException(status_code=404, detail={
            "reason_code": "no_problems",
            "message": "That assignment has no problems to hand back."})

    problems = [{**r, **_context_of(r),
                 "order": (r.get("group_order") or 0) * 100
                          + (r.get("member_order") or 0)}
                for r in rows]
    problems.sort(key=lambda p: p["order"])

    answers = completed_answers(claims["sub"], [p["slug"] for p in problems])
    if not answers:
        raise HTTPException(status_code=409, detail={
            "reason_code": "nothing_completed",
            "message": "Finish at least one problem and your file will be "
                       "ready to download."})

    name = (claims.get("name") or claims.get("username") or "").strip()
    text = build_handback(
        problems, {s: a["code"] for s, a in answers.items()},
        assignment_name=asg[0].get("name") or "Assignment",
        student_name=name,
        revealed_slugs={s for s, a in answers.items() if a["assisted"]})

    # The teacher's own filename, so what lands in Downloads is recognisably the
    # file they were given rather than a slug nobody chose.
    stem = (asg[0].get("source_file") or "assignment.py").rsplit("/", 1)[-1]
    if stem.endswith(".py"):
        stem = stem[:-3]
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in stem)[:60]
    who = "".join(c for c in name.split("@")[0] if c.isalnum() or c in "-_")[:40]
    filename = f"{safe}{'_' + who if who else ''}.py"

    return Response(
        content=text, media_type="text/x-python",
        headers={"Content-Disposition": f'attachment; filename="{filename}"',
                 # The file changes as they finish more problems.
                 "Cache-Control": "no-store"})


@app.post("/design_review/plan")
def design_review_plan(req: PlanSubmitRequest, request: Request):
    """Submit the plan graph the page built from this student's own chat.

    The gate used to accept exactly one thing, an uploaded picture - so a
    student whose plan was already drawn on screen had to screenshot that
    drawing and upload it back to the same app. Same reviewer, same rubric, same
    archive row as /design_review; the plan simply arrives as structure instead
    of as a photo, which also means no vision call."""
    from main.design_review import DesignRejected, review_plan_graph

    row = get_supabase().table("problems").select(
        "slug, title, description").eq("slug", req.slug).execute().data
    if not row:
        raise HTTPException(status_code=404, detail={
            "reason_code": "problem_not_found",
            "message": f"Unknown problem '{req.slug}'."})

    try:
        out = review_plan_graph(row[0], req.graph, req.history,
                                chat_log=req.messages)
    except DesignRejected as e:
        raise HTTPException(status_code=400, detail={
            "reason_code": "design_rejected", "message": str(e)})
    except Exception as e:
        raise HTTPException(status_code=503, detail={
            "reason_code": "reviewer_unavailable",
            "message": "The design reviewer is unavailable right now. Try again "
                       "shortly.",
            "detail": str(e)[:120]})

    # Archived exactly like an uploaded design, with no bytes: save_design skips
    # the storage upload for an empty blob and still writes the row, which is
    # what /history reads to decide `design_approved` on reopen. Without this a
    # student who passed the gate this way would be asked to pass it again.
    from main.archive import save_design, save_messages
    _student = current_student(request)
    if _student:
        save_design(get_supabase(), _student, req.slug, b"",
                    "application/x-plan-graph", out)
        save_messages(get_supabase(), _student, req.slug, "design",
                      [{"role": "assistant", "content": out["reply"]}])
        if out.get("approved"):
            # The plan that PASSED the gate is the one worth keeping, and from
            # here it is frozen - see the student page. Snapshot it now so a
            # reopen restores the approved plan rather than the last thing the
            # chat scraper happened to produce.
            from main.archive import save_graph
            save_graph(get_supabase(), _student, req.slug, "plan", req.graph)
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
