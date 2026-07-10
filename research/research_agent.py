"""
research_agent.py — simulated 3-agent research pipeline over the new chunk-based
tutoring system. Same architecture as the live tutor (chunks, 4-tier grader, oracle
gate, pool) but with weak/normal/strong Qwen agents typing instead of a human.

Same temperature across all three agents to isolate model-capability as the
independent variable.
"""
import os
import sys
import time
from datetime import datetime

from dotenv import load_dotenv
from supabase import create_client

from main.run_phase1 import get_chunk_decomposition
from tests.grader import grade_chunk
from main.ollama_client import chat
from .student_agent import get_student_answer, AGENTS

load_dotenv()
SB = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

# ── Agent configuration: SAME temperature, only model size varies ──
UNIFIED_TEMPERATURE = 0.3


def log_interaction(slug: str, chunk_index: int, chunk_prompt: str, agent_level: str,
                    attempt: int, code: str, result: dict, revealed: bool = False):
    """Save one attempt row to Supabase."""
    try:
        SB.table("research_interactions").insert({
            "problem_slug": slug,
            "chunk_index": chunk_index,
            "chunk_prompt": chunk_prompt[:500],
            "agent_level": agent_level,
            "attempt": attempt,
            "student_code": code[:4000],
            "correct": bool(result.get("correct")),
            "tier": result.get("tier"),
            "reason": (result.get("reason") or "")[:500],
            "failures": result.get("failures", [])[:3],
            "revealed": revealed,
        }).execute()
    except Exception as e:
        print(f"    ⚠️  DB log failed: {e}")


def run_agent_on_problem(problem: dict, chunks_result: dict, agent_level: str) -> dict:
    """Run one agent through all chunks of one problem. Two attempts per chunk;
    reveal reference on second failure. Returns summary stats for the poster."""
    slug = problem["slug"]
    chunks = chunks_result["chunks"]
    chunks_serialized = [
        {"step_id": c.step_id, "prompt": c.prompt, "expected_type": c.expected_type,
         "reference": c.reference or ""}
        for c in chunks
    ]

    accepted_prefix = []
    stats = {"chunks_passed_a1": 0, "chunks_passed_a2": 0, "chunks_revealed": 0}

    from .student_agent import AGENTS
    print(f"\n  ── {agent_level.upper()} ({AGENTS[agent_level]}) ──")

    for idx, chunk in enumerate(chunks):
        print(f"    Part {idx+1}: {chunk.prompt[:80]}")

        # Attempt 1
        code1 = get_student_answer(chunk.prompt, accepted_prefix, agent_level)
        result1 = grade_chunk(problem, chunks, idx, code1, accepted_prefix)
        log_interaction(slug, idx, chunk.prompt, agent_level, 1, code1, result1)
        print(f"      A1 [{result1['tier']}] {'✅' if result1['correct'] else '❌'}")

        if result1["correct"]:
            accepted_prefix.append(code1)
            stats["chunks_passed_a1"] += 1
            continue

        # Attempt 2 with hint
        code2 = get_student_answer(chunk.prompt, accepted_prefix, agent_level, hint=result1.get("reason"))
        result2 = grade_chunk(problem, chunks, idx, code2, accepted_prefix)
        log_interaction(slug, idx, chunk.prompt, agent_level, 2, code2, result2)
        print(f"      A2 [{result2['tier']}] {'✅' if result2['correct'] else '❌'}")

        if result2["correct"]:
            accepted_prefix.append(code2)
            stats["chunks_passed_a2"] += 1
        else:
            # Reveal reference and move on
            reference = chunk.reference or "pass"
            accepted_prefix.append(reference)
            stats["chunks_revealed"] += 1
            # Log the reveal as a synthetic attempt 2 row with revealed=True
            # (already logged the actual attempt above; this marks the reveal event)
            print(f"      🔓 Reference revealed")

    return stats


def already_done(slug: str) -> bool:
    """Skip problems already fully processed (resumability)."""
    try:
        res = SB.table("research_interactions").select("id", count="exact") \
            .eq("problem_slug", slug).execute()
        # A finished problem has >= 3 agents × 2+ chunks × 1+ attempts = at least 6 rows
        return (res.count or 0) >= 6
    except Exception:
        return False


'''def load_problems() -> list[dict]:
    """Load all problems from Supabase with their ground-truth solutions."""
    res = SB.table("problems").select("slug, title, description, difficulty, solution").execute()
    return res.data or []'''

def load_problems() -> list[dict]:
    """TEMP: small slice for validation before the full run."""
    res = SB.table("problems").select(
        "slug, title, description, difficulty, solution"
    ).in_("slug", ["palindrome-number", "two-sum", "roman-to-integer"]).execute()
    return res.data or []

def main():
    problems = load_problems()
    print(f"Loaded {len(problems)} problems from Supabase.\n")

    start_time = time.time()
    processed, skipped, failed = 0, 0, 0

    for i, problem in enumerate(problems, 1):
        slug = problem["slug"]
        elapsed = (time.time() - start_time) / 60

        print(f"\n{'═'*70}")
        print(f"[{i}/{len(problems)}] {slug}  ({elapsed:.1f} min elapsed)")
        print('═'*70)

        if already_done(slug):
            print(f"  ⏭️  Already processed — skipping.")
            skipped += 1
            continue

        # Get chunk decomposition (uses pool + gate + fallback)
        try:
            chunks_result = get_chunk_decomposition(problem)
        except Exception as e:
            print(f"  ⚠️  Decomposition failed hard: {e}")
            failed += 1
            continue

        if not chunks_result.get("chunks"):
            print(f"  ⚠️  No chunks produced — skipping.")
            failed += 1
            continue

        # Run all three agents
        for agent_level in ["weak", "normal", "strong"]:
            try:
                run_agent_on_problem(problem, chunks_result, agent_level)
            except Exception as e:
                print(f"  ⚠️  Agent {agent_level} crashed: {e}")

        processed += 1
        print(f"\n  ✅ {slug} done. ({processed} processed, {skipped} skipped, {failed} failed)")

    total_min = (time.time() - start_time) / 60
    print(f"\n{'═'*70}\nRUN COMPLETE — {processed} problems in {total_min:.1f} min "
          f"({skipped} skipped, {failed} failed)\n{'═'*70}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏸️  Paused. Rerun to resume from where you left off.")
        sys.exit(0)