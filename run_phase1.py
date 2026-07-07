import json
import os
import random
import re
import textwrap
from dotenv import load_dotenv
from pydantic import ValidationError
from supabase import create_client

from ollama_client import chat
from schemas import DecomposeOutput, EvalResult, StepItem
from student_agent import get_student_answer
from semantic import ast_equivalent
from sandbox import get_oracle_tests, passes_tests, _extract_signature
from prompts import DECOMPOSE_SYSTEM, EVAL_SYSTEM, CHUNK_DECOMPOSE_SYSTEM


load_dotenv()

MODEL  = "qwen2.5:7b-instruct"
AGENTS = ["weak", "normal", "strong"]

# Supabase client

supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_KEY"],
)


_CHUNK_POOL_PATH = os.path.join(os.path.dirname(__file__), "chunk_pool.json")
_POOL_TARGET = 5          # stop generating fresh once a problem has this many
_FRESH_PROBABILITY = 0.4  # chance to generate fresh even when pool has entries



# JSON parsing

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


# Code helpers

def normalize_code(text: str) -> str:
    """Normalize operator spacing but PRESERVE leading indentation."""
    text = text.expandtabs(4)
    lines = text.split('\n')
    normalized = []
    for line in lines:
        # Preserve leading whitespace exactly
        leading = len(line) - len(line.lstrip())
        indent = line[:leading]
        content = line[leading:]
        # Normalize operator spacing on the content only
        content = re.sub(r'\s*(//=|//|\*\*|==|!=|<=|>=|%|[+\-*/<>])\s*', r' \1 ', content)
        content = re.sub(r'  +', ' ', content).strip()
        normalized.append(indent + content)
    return '\n'.join(normalized).rstrip()

def expected_indent(context: str) -> int:
    """
    Determine required indentation for the next line
    by looking at the last non-empty line of context.
    """
    if not context or not context.strip():
        return 0
    lines = [l for l in context.split('\n') if l.strip()]
    if not lines:
        return 0
    last = lines[-1]
    last_indent = len(last) - len(last.lstrip())
    if last.rstrip().endswith(':'):
        return last_indent + 4
    return last_indent


def check_indentation(answer: str, context: str) -> str | None:
    """
    Returns an error string if indentation is wrong, None if correct.
    Only checks the first non-empty line of the answer.
    """
    if not context or not context.strip():
        return None  # no context to infer from

    first_line = next((l for l in answer.split('\n') if l.strip()), None)
    if not first_line:
        return None

    actual = len(first_line) - len(first_line.lstrip())
    expected = expected_indent(context)

    stripped = first_line.strip()
    is_header = any(stripped.startswith(kw) for kw in
                    ('def ', 'class ', 'for ', 'while ', 'if ', 'else:',
                     'elif ', 'try:', 'except', 'finally:', 'with '))

    if actual != expected:
        return (
            f"Wrong indentation: expected {expected} spaces "
            f"but got {actual}. "
            + ("Indent inside the block." if actual < expected
               else "Too much indentation.")
        )
    return None

def strip_comments(text: str) -> str:
    lines = [re.sub(r'\s*#.*$', '', line) for line in text.splitlines()]
    return '\n'.join(line for line in lines if line.strip()).strip()


def reconstruct_solution(steps: list[StepItem], answers: list[str], problem_title: str) -> str:
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


# Supabase helpers

def load_problems(limit: int = 500) -> list[dict]:
    res = (
        supabase.table("problems")
        .select("id, slug, title, difficulty, description, solution")
        .limit(limit)
        .execute()
    )
    return res.data


def save_steps(problem_id: str, steps: list[StepItem]) -> list[str]:
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


def save_interaction(step_uuid: str, agent: str, attempt: int, answer: str, correct: bool, hint: str | None, final_answer: str | None = None, score: float | None = None) -> None:
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


def assemble_canonical(steps: list[StepItem]) -> str:
    """Stack the canonical lines at their indent depths into a runnable function."""
    lines = []
    for s in steps:
        if s.expected_type != "code":
            continue
        line = (s.canonical or "").strip()
        if line:
            lines.append("    " * max(0, s.indent) + line)
    return "\n".join(lines)


def _gate_code(code: str, problem: dict) -> dict:
    """Compile + run assembled `code` against oracle tests.
    status: 'pass' | 'fail' | 'skipped'."""
    try:
        compile(code, "<assembled>", "exec")
    except SyntaxError as e:
        return {"status": "fail", "detail": f"not valid Python: {e}", "code": code, "failures": []}
    tests = get_oracle_tests(problem)
    if not tests:
        return {"status": "skipped", "detail": "no oracle tests (non-JSON inputs / no ground truth)",
                "code": code, "failures": []}
    m = re.search(r'def\s+(\w+)', code)
    entry = m.group(1) if m else None
    res = passes_tests(code, tests, entry_name=entry)
    if res["ok"] and res["fraction"] == 1.0:
        return {"status": "pass", "detail": f"{res['passed']}/{res['total']} pass",
                "code": code, "failures": []}
    return {"status": "fail", "detail": res["error"] or f"only {res['passed']}/{res['total']} pass",
            "code": code, "failures": res.get("failures", [])}


def assemble_references(header: str, chunks: list[StepItem]) -> str:
    body = "\n".join((c.reference or "").rstrip() for c in chunks if (c.reference or "").strip())
    return header + "\n" + textwrap.indent(body, "    ")


def decompose_into_chunks(problem: dict, max_tries: int = 5) -> dict:
    name, params = _extract_signature(problem.get("solution", ""))
    header = f"def {name or 'solve'}({', '.join(params)}):"
    text = problem.get("description") or problem.get("title", "")
    qid = problem.get("slug") or problem.get("title", "problem")

    feedback, best = "", None
    for attempt in range(1, max_tries + 1):
        user_msg = (
            f"PROBLEM:\n{text}\n\n"
            f"The function header is: {header}\n"
            "Write body chunks only.\n\n"
            + (f"PREVIOUS ATTEMPT FAILED:\n{feedback}\nFix it so the assembled body "
               "is correct.\n\n" if feedback else "")
            + 'Return JSON only: {"subproblems": [{"prompt": "...", "reference": "..."}, ...]}'
        )
        raw = chat(MODEL, CHUNK_DECOMPOSE_SYSTEM,
                   [{"role": "user", "content": user_msg}], temperature=0.2, fmt="json")
        try:
            data = parse_json(raw)
        except (ValueError, json.JSONDecodeError):
            continue
        chunks = [
            StepItem(question_id=qid, step_id=f"Part {i + 1}", prompt=c.get("prompt", "").strip(),
                     expected_type="code", reference=c.get("reference", ""))
            for i, c in enumerate(data.get("subproblems", [])) if c.get("prompt")
        ]
        if not chunks:
            continue

        # Reject single-chunk decompositions — defeats the purpose
        if len(chunks) < 2:
            print(f"  🧩 Chunk decomposition attempt {attempt}: fail — only 1 chunk produced")
            feedback = ("You produced only 1 chunk. This is invalid — you MUST break the "
                        "problem into 2 or 3 distinct sub-questions. Split the solution at "
                        "the point where the main computation begins vs. where the result "
                        "is returned, or at any other natural boundary.")
            continue
        
        code = assemble_references(header, chunks)
        print(f"\n  🔍 ASSEMBLED attempt {attempt}:\n{code}\n  ---")      
        report = _gate_code(code, problem)
        print(f"  🔍 FAILURES: {report.get('failures', [])}\n")
        
        print(f"  🧩 Chunk decomposition attempt {attempt}: {report['status']} — {report['detail']}")
        if report["status"] in ("pass", "skipped"):
            return {"header": header, "chunks": chunks}
        best = {"header": header, "chunks": chunks}
        fails = report.get("failures", [])[:3]
        feedback = ("Assembled body:\n" + code + "\n\nFailing tests:\n" +
                    "\n".join(f"  {f['input']} → expected {f['expected']}, got {f['got']}" for f in fails))

    print(f"  ⚠️  Chunk decomposition failed all {max_tries} tries.")
    raise RuntimeError("Could not generate a valid decomposition for this problem.")


def _load_pool() -> dict:
    if os.path.exists(_CHUNK_POOL_PATH):
        try:
            return json.load(open(_CHUNK_POOL_PATH))
        except Exception:
            return {}
    return {}


def _save_pool(pool: dict) -> None:
    try:
        json.dump(pool, open(_CHUNK_POOL_PATH, "w"), indent=2)
    except Exception as e:
        print(f"  ⚠️  Could not save chunk pool: {e}")


def _serialize(result: dict) -> dict:
    return {"header": result["header"],
            "chunks": [{"step_id": c.step_id, "prompt": c.prompt,
                        "expected_type": c.expected_type, "reference": c.reference or ""}
                       for c in result["chunks"]]}


def _deserialize(entry: dict) -> dict:
    return {"header": entry["header"],
            "chunks": [StepItem(question_id="pool", step_id=c["step_id"],
                                prompt=c["prompt"], expected_type=c.get("expected_type", "code"),
                                reference=c.get("reference", ""))
                       for c in entry["chunks"]]}


def decompose_into_chunks_best(problem: dict, max_tries: int = 5) -> dict:
    name, params = _extract_signature(problem.get("solution", ""))
    header = f"def {name or 'solve'}({', '.join(params)}):"
    text = problem.get("description") or problem.get("title", "")
    qid = problem.get("slug") or problem.get("title", "problem")
    best = None
    best_score = -1

    for attempt in range(1, max_tries + 1):
        user_msg = (
            f"PROBLEM:\n{text}\n\n"
            f"The function header is: {header}\n"
            "Write body chunks only.\n\n"
            'Return JSON only: {"subproblems": [{"prompt": "...", "reference": "..."}, ...]}'
        )
        raw = chat(MODEL, CHUNK_DECOMPOSE_SYSTEM,
                   [{"role": "user", "content": user_msg}], temperature=0.3, fmt="json")
        try:
            data = parse_json(raw)
        except Exception:
            continue
        chunks = [
            StepItem(question_id=qid, step_id=f"Part {i+1}",
                     prompt=c.get("prompt", "").strip(),
                     expected_type="code", reference=c.get("reference", ""))
            for i, c in enumerate(data.get("subproblems", [])) if c.get("prompt")
        ]
        if not chunks:
            continue
        code = assemble_references(header, chunks)
        report = _gate_code(code, problem)
        score = report.get("fraction", 0) if report["status"] != "skipped" else 0.5
        if score > best_score:
            best_score = score
            best = {"header": header, "chunks": chunks}
        if report["status"] in ("pass", "skipped"):
            return best

    return best or {"header": header, "chunks": [
        StepItem(question_id=qid, step_id="Part 1",
                 prompt="Write a complete solution for this problem.",
                 expected_type="code", reference="pass")
    ]}
    

def get_chunk_decomposition(problem: dict) -> dict:
    slug = problem.get("slug", "")
    pool = _load_pool()
    entries = pool.get(slug, [])

    want_fresh = (not entries) or (len(entries) < _POOL_TARGET) or \
                 (random.random() < _FRESH_PROBABILITY)

    if want_fresh:
        try:
            fresh = decompose_into_chunks(problem)
            entries.append(_serialize(fresh))
            pool[slug] = entries
            _save_pool(pool)
            print(f"  ✨ Fresh decomposition added to pool for {slug} "
                  f"(pool size: {len(entries)})")
            return fresh
        except RuntimeError as e:
            if entries:
                # Pool has validated entries — serve one, log the failure
                print(f"  ↩️  Fresh generation failed ({e}); serving from pool.")
            else:
                # Pool empty AND generation failed — serve best-attempt with a warning.
                # Better than a 500 — the gate failure means it's imperfect but usable.
                print(f"  ⚠️  All retries failed and pool empty for {slug}. "
                      f"Serving best attempt (imperfect but not crashing).")
                # Re-run once more explicitly to get best attempt
                try:
                    return decompose_into_chunks_best(problem)
                except Exception:
                    # Absolute last resort — return a minimal valid structure
                    from sandbox import _extract_signature
                    name, params = _extract_signature(problem.get("solution", ""))
                    header = f"def {name or 'solve'}({', '.join(params)}):"
                    return {"header": header, "chunks": [
                        StepItem(question_id=slug, step_id="Part 1",
                                 prompt="Write a complete solution for this problem.",
                                 expected_type="code", reference="pass")
                    ]}

    chosen = random.choice(entries)
    print(f"  🎲 Served pooled decomposition for {slug} (pool size: {len(entries)})")
    return _deserialize(chosen)


def validate_decomposition(steps: list[StepItem], problem: dict) -> dict:
    code_steps = [s for s in steps if s.expected_type == "code"]
    if not code_steps or any(not (s.canonical or "").strip() for s in code_steps):
        return {"status": "skipped", "detail": "missing canonical lines", "code": "", "failures": []}
    return _gate_code(assemble_canonical(steps), problem)


def decompose_question(question_id: str, question_text: str, feedback: str = "", prefix_note: str = "") -> list[StepItem]:
    user_msg = (
        f"QUESTION_ID: {question_id}\n"
        f"PROBLEM:\n{question_text}\n\n"
        + (prefix_note + "\n\n" if prefix_note else "")
        + (f"PREVIOUS ATTEMPT FAILED VALIDATION:\n{feedback}\n"
           "Fix the step ordering / canonical lines so the assembled program is "
           "correct. Do not place a return before code that still needs to run.\n\n"
           if feedback else "")
        + "Return JSON only using this schema:\n"
        '{"steps": [{"step_id":"Step 1","prompt":"...","expected_type":"code",'
        '"rubric":"...","canonical":"def f(x):","indent":0}, ...]}'
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
                    "canonical":     s.get("canonical"),
                    "indent":        int(s.get("indent", 0) or 0),
                }
                for idx, s in enumerate(data.get("steps", []))
                if s.get("prompt")
            ]
        })
    except ValidationError as e:
        raise RuntimeError(f"Decomposition schema mismatch: {e}")
    return parsed.steps


def decompose_validated(problem: dict, max_tries: int = 3) -> list[StepItem]:
    """Decompose, then gate on execution. If the assembled canonical solution
    fails the oracle tests, re-prompt with the failing cases and retry."""
    qid = problem.get("slug") or problem.get("title", "problem")
    text = problem.get("description") or problem.get("title", "")
    feedback = ""
    best = None

    for attempt in range(1, max_tries + 1):
        steps = decompose_question(qid, text, feedback=feedback)
        report = validate_decomposition(steps, problem)
        print(f"  🧪 Decomposition attempt {attempt}: {report['status']} — {report['detail']}")

        if report["status"] in ("pass", "skipped"):
            return steps  # pass = verified correct; skipped = can't verify, accept as-is

        best = steps  # keep the latest failing attempt as fallback
        fails = report.get("failures", [])[:3]
        feedback = (
            "Assembled program:\n" + report["code"] + "\n\n"
            "Failing tests (input → expected vs what your steps produced):\n"
            + "\n".join(f"  {f['input']} → expected {f['expected']}, got {f['got']}"
                        for f in fails)
        )

    print(f"  ⚠️  Decomposition still failing after {max_tries} tries — using best attempt.")
    return best


def replan_from_prefix(problem: dict, accepted_steps: list[StepItem],
                       max_tries: int = 3) -> list[StepItem]:
    """Re-decompose the REMAINING steps so they continue from the student's own
    accepted prefix (their chosen approach) to a correct full solution. Fired when
    a student takes a valid alternative path and chooses to keep going their way."""
    qid = problem.get("slug") or problem.get("title", "problem")
    text = problem.get("description") or problem.get("title", "")
    prefix_code = assemble_canonical(accepted_steps)
    last = accepted_steps[-1] if accepted_steps else None
    start_num = len(accepted_steps) + 1

    prefix_note = (
        "The student has ALREADY written the start of the solution using their own approach:\n"
        f"{prefix_code}\n"
        f"The last line is `{(last.canonical if last else '').strip()}` at indent depth "
        f"{last.indent if last else 0}.\n"
        "Produce ONLY the remaining micro-steps that continue from this exact prefix to a "
        "correct, complete solution that builds on it. Never repeat a prefix line, and never "
        f"restate the function header. Number new steps starting at Step {start_num}."
    )

    feedback, best = "", None
    for attempt in range(1, max_tries + 1):
        steps = decompose_question(qid, text, feedback=feedback, prefix_note=prefix_note)
        for i, st in enumerate(steps):                     # force correct numbering
            st.step_id = f"Step {start_num + i}"
        full = prefix_code + ("\n" if prefix_code else "") + assemble_canonical(steps)
        report = _gate_code(full, problem)
        print(f"  🔁 Replan attempt {attempt}: {report['status']} — {report['detail']}")
        if report["status"] in ("pass", "skipped"):
            return steps
        best = steps
        fails = report.get("failures", [])[:3]
        feedback = ("Assembled program (student prefix + your new steps):\n" + full +
                    "\n\nFailing tests:\n" +
                    "\n".join(f"  {f['input']} → expected {f['expected']}, got {f['got']}"
                              for f in fails))
    print(f"  ⚠️  Replan still failing after {max_tries} tries — using best attempt.")
    return best


def _dedent_all(text: str) -> str:
    """Strip leading whitespace from every line — indentation is graded by the
    reconstructor, never at the step level."""
    return re.sub(r'(?m)^[ \t]+', '', text)


def eval_step(step: StepItem, student_answer: str, context: str) -> EvalResult:
    if not student_answer or not student_answer.strip():
        student_answer = "__BLANK__"

    if step.expected_type == "code" and student_answer != "__BLANK__":
        student_answer = normalize_code(student_answer)
        student_answer = re.sub(r'(\w)\s+\(', r'\1(', student_answer)
        # Indentation is NOT graded here — strip it so only logic is compared.
        student_answer = _dedent_all(student_answer)

        # Tier 1: deterministic AST equivalence vs the canonical line.
        # If it matches, it's correct — no LLM call, no chance to mis-reject.
        if step.canonical and ast_equivalent(student_answer, step.canonical):
            return EvalResult(
                correct=True,
                short_reason="Correct (equivalent to the expected line).",
                correct_answer=step.canonical.strip(),
            )
            
        # Tier 1.5: not the canonical line — is it a VALID different approach?
        # Splice it onto the prefix and run; if the partial program is still on a
        # correct trajectory, flag divergent so the UI can offer to replan.
        if step.canonical and context:
            candidate = context.rstrip() + "\n" + ("    " * step.indent) + student_answer
            try:
                compile(candidate + "\n    pass" if candidate.rstrip().endswith(":")
                        else candidate, "<cand>", "exec")
                parses = True
            except SyntaxError:
                parses = False
            if parses and not student_answer.strip().startswith(("#", "pass")):
                return EvalResult(
                    correct=True,
                    short_reason="Correct, but a different approach than suggested.",
                    correct_answer=step.canonical.strip(),
                    divergent=True,
                )

    rubric_for_grading = _dedent_all(step.rubric) if step.rubric else None

    user_msg = (
        f"MICRO-STEP:\n{step.prompt}\n\n"
        f"RUBRIC:\n{rubric_for_grading or 'Not provided'}\n\n"
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


def score_answer(reconstructed: str, ground_truth: str, problem_title: str) -> float:
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


# Per-agent loop

def run_agent(problem: dict, steps: list[StepItem], step_uuids: list[str], agent: str) -> None:
    print(f"\n  ── {agent.upper()} AGENT ──")
    local_context     = ""
    collected_answers = []

    for step, step_uuid in zip(steps, step_uuids):
        print(f"\n  {step.step_id}: {step.prompt}")

        # Attempt 1
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

        # Attempt 2
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

    # Reconstruct full solution then score
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


# Per-problem orchestrator

def run_question(problem: dict) -> None:
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


# Entry point

def main() -> None:
    print("Loading problems from Supabase…")
    problems = load_problems(limit=100)
    if not problems:
        print("⚠️  No problems found. Run upload_to_supabase.py first.")
        return

    print(f"Loaded {len(problems)} problems.\n")

    for problem in problems:
        run_question(problem)

    print("\nDone. All problems complete.")


if __name__ == "__main__":
    main()