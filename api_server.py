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

from run_phase1 import decompose_validated, eval_step, parse_json, decompose_into_chunks, replan_from_prefix
from grader import grade_chunk
from schemas import StepItem

app = FastAPI(title="MicroTutor API", version="1.0")
from fastapi.middleware.cors import CORSMiddleware

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


@app.get("/health")
def health():
    return {"status": "ok", "message": "MicroTutor API running"}


@app.post("/decompose")
def decompose(req: DecomposeRequest):
    try:
        steps = decompose_validated(
            {"slug": req.slug, "title": req.slug, "description": req.description}
        )
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
                for s in steps
            ]
        }
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


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
        problem = {"slug": req.slug, "title": req.slug, "description": req.description}
        # Fetch the full problem dict (needs solution for oracle)
        from run_phase1 import load_problems
        problems = load_problems(limit=500)
        full = next((p for p in problems if p.get("slug") == req.slug), None)
        if full:
            problem["solution"] = full.get("solution", "")
        result = decompose_into_chunks(problem)
        
        # Pre-warm oracle tests
        if problem.get("solution", "").strip():
            try:
                from sandbox import get_oracle_tests
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
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/grade_chunk")
def grade_chunk_route(req: ChunkRequest):
    try:
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