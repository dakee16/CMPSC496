import json
import os
import re
from dotenv import load_dotenv
from pydantic import ValidationError
from supabase import create_client

from ollama_client import chat
from schemas import DecomposeOutput, EvalResult, StepItem
from prompts import DECOMPOSE_SYSTEM, EVAL_SYSTEM
from student_agent import get_student_answer

load_dotenv()

MODEL  = "qwen2.5:7b-instruct"
AGENTS = ["weak", "normal", "strong"]

# ---------------------------------------------------------------------------
# Supabase client
# ---------------------------------------------------------------------------

supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_KEY"],
)

# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def parse_json(text: str) -> dict:
    """Extract the first complete JSON object from a string."""
    text = text.strip()
    start = text.find("{")
    if start == -1:
        raise ValueError("No '{' found in model output.")
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("No matching '}' found in model output.")


# ---------------------------------------------------------------------------
# Code helpers
# ---------------------------------------------------------------------------

def normalize_code(text: str) -> str:
    """Collapse whitespace so spacing never affects grading."""
    text = re.sub(r'\s*(//=|//|\*\*|==|!=|<=|>=|%|[+\-*/<>])\s*', r' \1 ', text)
    return re.sub(r'  +', ' ', text).strip()


def strip_comments(text: str) -> str:
    """Remove # placeholder comments from a code answer."""
    lines = [re.sub(r'\s*#.*$', '', line) for line in text.splitlines()]
    return '\n'.join(line for line in lines if line.strip()).strip()


def reconstruct_solution(
    steps: list[StepItem],
    answers: list[str],
    problem_title: str,
) -> str:
    """
    Ask the LLM to assemble individual step answers into a coherent
    Python function. This gives score_answer something meaningful to compare.
    """
    step_lines = "\n".join(
        f"Step {i+1} ({s.prompt}): {a}"
        for i, (s, a) in enumerate(zip(steps, answers))
    )
    prompt = (
        f"You are assembling a Python solution from individual micro-step answers.\n\n"
        f"Problem: {problem_title}\n\n"
        f"Step answers collected:\n{step_lines}\n\n"
        "Combine these into a single clean Python function. "
        "Fix indentation and structure. Return ONLY the Python code, no explanation."
    )
    raw = chat(
        MODEL,
        "You are a Python code assembler. Return only clean Python code.",
        [{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    raw = re.sub(r'```[\w]*\n?', '', raw)
    raw = re.sub(r'```', '', raw)
    return raw.strip()


# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------

def load_problems(limit: int = 500) -> list[dict]:
    """Fetch problems from Supabase."""
    res = (
        supabase.table("problems")
        .select("id, slug, title, difficulty, description, solution")
        .limit(limit)
        .execute()
    )
    return res.data


def save_steps(problem_id: str, steps: list[StepItem]) -> list[str]:
    """Save decomposed steps to Supabase. Returns list of step UUIDs."""
    step_ids = []
    for step in steps:
        num = step.step_id.split()[-1]
        row = {
            "problem_id":    problem_id,
            "step_number":   int(num) if num.isdigit() else 0,
            "prompt":        step.prompt,
            "expected_type": step.expected_type,
            "rubric":        step.rubric or "",
        }
        res = supabase.table("steps").insert(row).execute()
        step_ids.append(res.data[0]["id"])
    return step_ids


def save_interaction(
    step_uuid: str,
    agent: str,
    attempt: int,
    answer: str,
    correct: bool,
    hint: str | None,
    final_answer: str | None = None,
    score: float | None = None,
) -> None:
    """Log one agent interaction to Supabase."""
    supabase.table("interactions").insert({
        "step_id":      step_uuid,
        "agent_level":  agent,
        "attempt":      attempt,
        "answer":       answer,
        "correct":      correct,
        "hint_shown":   hint or "",
        "final_answer": final_answer or "",
        "score":        score,
    }).execute()


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

def decompose_question(question_id: str, question_text: str) -> list[StepItem]:
    """Ask the LLM to break one question into ordered micro-steps."""
    user_msg = (
        f"QUESTION_ID: {question_id}\n"
        f"PROBLEM:\n{question_text}\n\n"
        "Return JSON only using this schema:\n"
        '{"steps": [{"step_id":"Step 1","prompt":"...","expected_type":"code","rubric":"..."}, ...]}'
    )
    raw = chat(
        MODEL,
        DECOMPOSE_SYSTEM,
        [{"role": "user", "content": user_msg}],
        temperature=0.2,
        fmt="json",
    )
    try:
        data = parse_json(raw)
    except (ValueError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Decomposition produced invalid JSON: {e}\nRaw:\n{raw[:500]}")
    try:
        parsed = DecomposeOutput.model_validate({
            "steps": [
                {
                    "question_id":   question_id,
                    "step_id":       s.get("step_id", f"Step {idx + 1}"),
                    "prompt":        s.get("prompt", "").strip(),
                    "expected_type": s.get("expected_type", "string"),
                    "skill":         "auto-decomposed",
                    "rubric":        s.get("rubric"),
                }
                for idx, s in enumerate(data.get("steps", []))
                if s.get("prompt")
            ]
        })
    except ValidationError as e:
        raise RuntimeError(f"Decomposition schema mismatch: {e}")
    return parsed.steps


def eval_step(step: StepItem, student_answer: str, context: str) -> EvalResult:
    """Grade one student answer against a micro-step."""
    if step.expected_type == "code":
        student_answer = normalize_code(student_answer)
    user_msg = (
        f"MICRO-STEP:\n{step.prompt}\n\n"
        f"RUBRIC (what correct looks like for THIS step only):\n{step.rubric or 'Not provided'}\n\n"
        f"EXPECTED_TYPE: {step.expected_type}\n\n"
        f"PRIOR CONTEXT (validated facts for this question only):\n{context or 'None yet.'}\n\n"
        f"STUDENT ANSWER:\n{student_answer}\n\n"
        'Return JSON only: {"correct": true/false, "short_reason": "...", "correct_answer": "..."}'
    )
    raw = chat(
        MODEL,
        EVAL_SYSTEM,
        [{"role": "user", "content": user_msg}],
        temperature=0.0,
        fmt="json",
    )
    try:
        data = parse_json(raw)
        return EvalResult.model_validate(data)
    except (ValueError, json.JSONDecodeError, ValidationError) as e:
        raise RuntimeError(f"Evaluation produced invalid output: {e}\nRaw:\n{raw[:400]}")


def score_answer(
    reconstructed: str,
    ground_truth: str,
    problem_title: str,
) -> float:
    """
    Score a reconstructed full solution against ground truth.
    Returns 0.0 to 1.0.
    """
    prompt = (
        f"You are a code grader comparing two Python solutions for: {problem_title}\n\n"
        f"GROUND TRUTH SOLUTION:\n{ground_truth[:1500]}\n\n"
        f"STUDENT SOLUTION:\n{reconstructed[:1500]}\n\n"
        "Score the student solution from 0.0 to 1.0:\n"
        "  1.0 = functionally equivalent, handles all cases\n"
        "  0.7 = mostly correct, minor logic issues\n"
        "  0.4 = correct approach but incomplete or has bugs\n"
        "  0.1 = shows understanding but mostly wrong\n"
        "  0.0 = completely wrong, empty, or unrelated\n\n"
        "Focus on LOGIC and CORRECTNESS, not style.\n"
        'Return JSON only: {"score": 0.0, "reason": "one sentence"}'
    )
    raw = chat(
        MODEL,
        "You are a strict but fair code grader. Return only JSON.",
        [{"role": "user", "content": prompt}],
        temperature=0.0,
        fmt="json",
    )
    try:
        data = parse_json(raw)
        score = float(data.get("score", 0.0))
        reason = data.get("reason", "")
        print(f"  📝 Score reason: {reason}")
        return score
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Per-agent loop
# ---------------------------------------------------------------------------

def run_agent(
    problem: dict,
    steps: list[StepItem],
    step_uuids: list[str],
    agent: str,
) -> None:
    """Run one agent through all steps, reconstruct solution, then score."""
    print(f"\n  ── {agent.upper()} AGENT ──")
    local_context     = ""
    collected_answers = []

    for step, step_uuid in zip(steps, step_uuids):
        print(f"\n  {step.step_id}: {step.prompt}")

        # ── Attempt 1 ────────────────────────────────────────────────────
        ans = get_student_answer(step.prompt, local_context, agent)
        print(f"  Answer: {ans}")

        try:
            result = eval_step(step, ans, local_context)
        except RuntimeError as e:
            print(f"  ⚠️  Eval error — skipping.\n  {e}")
            collected_answers.append(ans)
            continue

        # Log attempt 1
        if step_uuid:
            try:
                save_interaction(step_uuid, agent, 1, ans, result.correct, None)
            except Exception as e:
                print(f"  ⚠️  DB log failed: {e}")

        if result.correct:
            print("  ✅ Correct.")
            local_context += f"- {step.step_id}: {step.prompt} | Answer: {ans}\n"
            collected_answers.append(ans)
            continue

        # ── Attempt 2 ────────────────────────────────────────────────────
        print(f"  ❌ Incorrect. Hint: {result.short_reason}")
        ans2 = get_student_answer(step.prompt, local_context, agent)
        print(f"  Answer: {ans2}")

        try:
            result2 = eval_step(step, ans2, local_context)
        except RuntimeError as e:
            print(f"  ⚠️  Eval error — skipping.\n  {e}")
            collected_answers.append(ans2)
            continue

        # Log attempt 2
        if step_uuid:
            try:
                save_interaction(
                    step_uuid, agent, 2, ans2,
                    result2.correct, result.short_reason,
                )
            except Exception as e:
                print(f"  ⚠️  DB log failed: {e}")

        if result2.correct:
            print("  ✅ Correct.")
            local_context += f"- {step.step_id}: {step.prompt} | Answer: {ans2}\n"
            collected_answers.append(ans2)
        else:
            correct_ans = result2.correct_answer or ""
            if step.expected_type == "code":
                correct_ans = strip_comments(correct_ans)
            print(f"  ❌ Still incorrect. ✅ Correct answer: {correct_ans}")
            collected_answers.append(correct_ans)
            local_context += f"- {step.step_id}: {step.prompt} | Answer: {correct_ans}\n"

    # ── Reconstruct full solution then score ─────────────────────────────
    ground_truth = problem.get("solution", "")

    if ground_truth:
        print(f"\n  🔧 Reconstructing solution from {len(collected_answers)} step answers…")
        reconstructed = reconstruct_solution(steps, collected_answers, problem["title"])
        print(f"  📊 Scoring {agent.upper()} against ground truth…")
        score = score_answer(reconstructed, ground_truth, problem["title"])
        print(f"  📊 {agent.upper()} final score: {score:.2f}")

        # Store final_answer + score on the last step's interaction row
        last_uuid = next((u for u in reversed(step_uuids) if u), None)
        if last_uuid:
            try:
                (
                    supabase.table("interactions")
                    .update({"final_answer": reconstructed, "score": score})
                    .eq("step_id", last_uuid)
                    .eq("agent_level", agent)
                    .execute()
                )
            except Exception as e:
                print(f"  ⚠️  Could not save score to DB: {e}")
    else:
        print(f"\n  ⚠️  No ground truth — skipping score for {agent}.")


# ---------------------------------------------------------------------------
# Per-problem orchestrator
# ---------------------------------------------------------------------------

def run_question(problem: dict) -> None:
    """Decompose one problem and run all three agents through it."""
    problem_id = problem["id"]
    slug       = problem["slug"]
    qtext      = problem["description"] or problem["title"]

    print(f"\n{'=' * 60}")
    print(f"  {slug}  [{problem['difficulty']}]")
    print(f"{'=' * 60}")
    print(f"Problem: {problem['title']}\n")
    print("[Decomposing into micro-steps…]\n")

    try:
        steps = decompose_question(slug, qtext)
    except RuntimeError as e:
        print(f"⚠️  Could not decompose — skipping.\n{e}\n")
        return

    print(f"Generated {len(steps)} steps.\n")

    # Save steps once per problem
    try:
        step_uuids = save_steps(problem_id, steps)
    except Exception as e:
        print(f"⚠️  Could not save steps to Supabase: {e}")
        step_uuids = [None] * len(steps)

    # Run all three agents independently
    for agent in AGENTS:
        run_agent(problem, steps, step_uuids, agent)

    print(f"\n✅ Done: {slug}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading problems from Supabase…")
    problems = load_problems(limit=500)
    if not problems:
        print("⚠️  No problems found. Run upload_to_supabase.py first.")
        return

    print(f"Loaded {len(problems)} problems.\n")

    for problem in problems:
        run_question(problem)

    print("\nDone. All problems complete.")


if __name__ == "__main__":
    main()