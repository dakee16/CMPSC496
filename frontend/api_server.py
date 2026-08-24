"""
api_server.py — FastAPI bridge between Next.js web UI and local LLM pipeline.
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
from tests.grader import grade_chunk
from main.schemas import StepItem

app = FastAPI(title="MicroTutor API", version="1.0")
_sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


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
    # REQUIRED before the pipeline runs — see decompose_chunks_route.
    title: str | None = None
    solution: str | None = None


class EvaluateRequest(BaseModel):
    step: dict
    answer: str
    context: str = ""
    
class ReplanRequest(BaseModel):
    slug: str
    description: str
    accepted_steps: list[dict]
    
class ChunkRequest(BaseModel):
    problem: dict
    chunks: list[dict]
    index: int
    student_code: str
    accepted_prefix: list[str] = []

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

class MarkSolvedRequest(BaseModel):
    student_id: str
    slug: str


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
    if not req.answer or not req.answer.strip():
        req.answer = "__BLANK__"
    try:
        step = StepItem(
            question_id=req.step.get("step_id", "Step 1"),
            step_id=req.step.get("step_id", "Step 1"),
            prompt=req.step.get("prompt", ""),
            expected_type=req.step.get("expected_type", "code"),
            rubric=req.step.get("rubric", ""),
            canonical=req.step.get("canonical") or None,
            indent=int(req.step.get("indent", 0) or 0),
        )
        result = eval_step(step, req.answer, req.context)
        return {
            "correct": result.correct,
            "short_reason": result.short_reason,
            "correct_answer": result.correct_answer or "",
            "divergent": result.divergent,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/replan")
def replan(req: ReplanRequest):
    try:
        accepted = [StepItem(**s) for s in req.accepted_steps]
        problem = {"slug": req.slug, "title": req.slug, "description": req.description}
        new_steps = replan_from_prefix(problem, accepted)
        return {
            "steps": [
                {
                    "step_id": s.step_id,
                    "prompt": s.prompt,
                    "expected_type": s.expected_type,
                    "rubric": s.rubric or "",
                    "canonical": s.canonical or "",
                    "indent": s.indent,
                }
                for s in new_steps
            ]
        }
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    
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
        # no Gate 1 — the decomposition would be served unvalidated. That silent
        # degradation is exactly what uploads used to do; it is now a hard error.
        if not problem["solution"]:
            raise HTTPException(
                status_code=400,
                detail=(f"No reference solution for '{req.slug}'. A problem cannot "
                        f"be decomposed without ground truth — oracle tests, "
                        f"mutation validation and the necessity gate all depend "
                        f"on it. Supply `solution` with the request."))

        result = get_chunk_decomposition(problem)

        # Pre-warm oracle tests (guaranteed to have a solution by this point).
        try:
            from tests.sandbox import get_oracle_tests
            get_oracle_tests(problem)  # generates + caches, result discarded
            print(f"  ✅ Oracle pre-warmed for {problem.get('slug')}")
        except Exception as e:
            print(f"  ⚠️  Oracle pre-warm failed: {e}")
        return {
            "header": result["header"],
            "chunks": [
                {"step_id": c.step_id, "prompt": c.prompt,
                 "expected_type": c.expected_type, "reference": c.reference or ""}
                for c in result["chunks"]
            ]
        }
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Decomposition unavailable: {e}")


@app.post("/grade_chunk")
def grade_chunk_route(req: ChunkRequest):
    try:
        slug = req.problem.get("slug")
        if slug and not (req.problem.get("solution") or "").strip():
            from main.run_phase1 import load_problems
            problems = load_problems(limit=500)
            full = next((p for p in problems if p.get("slug") == slug), None)
            if full:
                req.problem["solution"] = (full.get("solution") or "").strip()
        if not (req.problem.get("solution") or "").strip():
            # Grading tiers 1-2 run the oracle; without ground truth they would
            # silently fall back to weaker LLM-only judgement.
            raise HTTPException(
                status_code=400,
                detail=f"No reference solution for '{slug}'. Cannot grade without "
                       f"ground truth — send `solution` with the problem.")
        chunks = [StepItem(question_id=req.problem.get("slug", "q"),
                           step_id=c.get("step_id", f"Part {i+1}"),
                           prompt=c.get("prompt", ""),
                           expected_type=c.get("expected_type", "code"),
                           reference=c.get("reference", ""))
                  for i, c in enumerate(req.chunks)]
        result = grade_chunk(req.problem, chunks, req.index,
                             req.student_code, req.accepted_prefix)
        return {"correct": result["correct"], "tier": result["tier"],
                "reason": result["reason"],
                "failures": result.get("failures", [])[:3]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# ── playground (read-only showcase) ───────────────────────────────────────
# These serve the step-through demo in frontend/playground.html. They REPLAY
# cached results and never trigger validation, decomposition, or any LLM call —
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

    row = _sb.table("problems").select(
        "slug, title, description, difficulty, solution").eq("slug", slug).execute().data
    if not row:
        raise HTTPException(status_code=404, detail=f"Problem '{slug}' not found.")
    problem = row[0]
    solution = problem.get("solution") or ""

    cached = _load_cache().get(content_hash(problem))
    if not (isinstance(cached, dict) and "strong" in cached):
        return {"problem": problem, "oracle": None, "mutants": [],
                "chunks": None, "necessity": [], "grading": None,
                "note": "not yet validated"}

    tests = cached.get("final_tests", [])
    entry = get_resolved_entry(problem)["entry_name"]

    # Stage 2 — prefer the breakdown persisted at validation time; only fall
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

    # Stages 3/4 — pooled decomposition, if one exists.
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

    # Stage 5 — a right and a wrong submission, decided by EXECUTION only.
    grading = None
    if tests:
        wrong = "def _f(*a, **k):\n    return None"
        ok = passes_tests(solution, tests, entry_name=entry)
        bad = passes_tests(wrong, tests, entry_name="_f")
        grading = {
            "correct": {"tier": "Tier 2 — executed against the oracle",
                        "verdict": ok["ok"] and ok["fraction"] == 1.0,
                        "passed": ok.get("passed", 0), "total": ok.get("total", 0)},
            "incorrect": {"tier": "Tier 2 — executed against the oracle",
                          "verdict": bad["ok"] and bad["fraction"] == 1.0,
                          "passed": bad.get("passed", 0), "total": bad.get("total", 0)},
        }

    # Why is it weak? Without this the UI can show "100% caught" beside a WEAK
    # badge — true but self-contradictory-looking — when the real reason is that
    # the solution yielded too few mutants to judge at all.
    from main.mutation import _MIN_MUTANTS, CUTOFF_1_KILL_RATE
    weak_reason = None
    if not cached["strong"]:
        if len(mutants) < _MIN_MUTANTS:
            weak_reason = (f"only {len(mutants)} way(s) to break this solution could be "
                           f"found — too few to judge the tests fairly "
                           f"(need at least {_MIN_MUTANTS})")
        else:
            weak_reason = (f"{round(cached.get('kill_rate_direct', 0.0) * 100)}% of cheaters "
                           f"caught is below the {round(CUTOFF_1_KILL_RATE * 100)}% bar")

    return {"problem": problem,
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
# it happens. Served with fetch()+ReadableStream on the frontend — the
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
    # oracle = nothing to watch — hard error, not a degraded run.
    if not problem["solution"] or not problem["description"]:
        row = _sb.table("problems").select(
            "slug, title, description, solution").eq("slug", req.slug).execute().data
        if row:
            problem["solution"] = problem["solution"] or (row[0].get("solution") or "").strip()
            problem["description"] = problem["description"] or (row[0].get("description") or "")
            problem["title"] = req.title or row[0].get("title") or req.slug
    if not problem["solution"]:
        raise HTTPException(
            status_code=400,
            detail=(f"No reference solution for '{req.slug}'. A live run cannot "
                    f"start without ground truth — oracle tests, mutation "
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
        from supabase import create_client
        import os
        sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
        query = sb.table("problems").select(
            "id, slug, title, difficulty, topic_tags"
        ).limit(limit)
        if difficulty:
            query = query.eq("difficulty", difficulty)
        res = query.execute()
        return {"problems": res.data, "count": len(res.data)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/problems/{slug}")
def get_problem(slug: str):
    """Fetch a single problem by slug from Supabase."""
    try:
        from supabase import create_client
        import os
        sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
        res = sb.table("problems").select(
            "id, slug, title, difficulty, description, topic_tags, solution"
        ).eq("slug", slug).single().execute()
        if not res.data:
            raise HTTPException(status_code=404, detail=f"Problem '{slug}' not found.")
        return res.data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


@app.post("/register")
def register(req: AuthRequest):
    pw_hash = bcrypt.hashpw(req.password.encode(), bcrypt.gensalt()).decode()

    try:
        result = _sb.table("students").insert({
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

    result = _sb.table("students").select("*").eq("username", req.username).execute()

    if not result.data:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    row = result.data[0]
    if not bcrypt.checkpw(req.password.encode(), row["password_hash"].encode()):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    return {"student_id": row["id"], "username": row["username"]}


@app.get("/solved/{student_id}")
def get_solved(student_id: str):
     
    result = _sb.table("solved").select("problem_slug").eq("student_id", student_id).execute()
    slugs = [r["problem_slug"] for r in (result.data or [])]
    return {"slugs": slugs}


@app.post("/mark_solved")
def mark_solved(req: MarkSolvedRequest):
 
    _sb.table("solved").upsert({
        "student_id": req.student_id,
        "problem_slug": req.slug
    }, on_conflict="student_id,problem_slug").execute()
    return {"ok": True}


@app.post("/log_interaction")
def log_interaction(req: LogInteractionRequest):

    data = req.model_dump()
    data["problem_slug"] = data.pop("slug")
    _sb.table("student_interactions").insert(data).execute()
    return {"ok": True}