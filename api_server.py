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

from run_phase1 import decompose_question, eval_step, parse_json
from schemas import StepItem

app = FastAPI(title="MicroTutor API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://*.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class DecomposeRequest(BaseModel):
    slug: str
    description: str


class EvaluateRequest(BaseModel):
    step: dict
    answer: str
    context: str = ""


@app.get("/health")
def health():
    return {"status": "ok", "message": "MicroTutor API running"}


@app.post("/decompose")
def decompose(req: DecomposeRequest):
    try:
        steps = decompose_question(req.slug, req.description)
        return {
            "steps": [
                {
                    "step_id": s.step_id,
                    "prompt": s.prompt,
                    "expected_type": s.expected_type,
                    "rubric": s.rubric or "",
                }
                for s in steps
            ]
        }
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate")
def evaluate(req: EvaluateRequest):
    try:
        step = StepItem(
            question_id=req.step.get("step_id", "Step 1"),
            step_id=req.step.get("step_id", "Step 1"),
            prompt=req.step.get("prompt", ""),
            expected_type=req.step.get("expected_type", "code"),
            rubric=req.step.get("rubric", ""),
        )
        result = eval_step(step, req.answer, req.context)
        return {
            "correct": result.correct,
            "short_reason": result.short_reason,
            "correct_answer": result.correct_answer or "",
        }
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))