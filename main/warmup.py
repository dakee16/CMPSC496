"""
warmup.py — offline oracle precompute.

tests.sandbox.get_oracle_tests() mutation-tests an oracle the first time anyone
asks for it, and that pass can include LLM counterexample search for surviving
mutants. That cost belongs offline. The architecture promises students only ever
hit the grading path and never wait through a retry loop — but grader.py:178 and
run_phase1.py:221 both route into get_oracle_tests(), so today the first student
to touch a problem pays for its validation inline.

Run this before students do, and that first-touch cost is already spent:

    python -m main.warmup

Safe to re-run: problems whose oracle is already cached and flagged strong are
skipped, so an interrupted run resumes where it left off.
"""
import os
import sys
import time

from dotenv import load_dotenv
from supabase import create_client

from tests.sandbox import _is_validated, _load_cache, get_oracle_tests

from .identity import content_hash

load_dotenv()
SB = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


def load_problems() -> list[dict]:
    """Every problem with its ground-truth solution.

    Same query research/research_agent.py uses — note that its copy is currently
    pinned to a 3-slug TEMP slice, so this deliberately re-states the full-set
    version rather than importing it. Warm-up wants everything."""
    res = SB.table("problems").select(
        "slug, title, description, difficulty, solution"
    ).execute()
    return res.data or []


def warm_up_oracles(problems: list[dict] | None = None) -> dict:
    """Force oracle validation for every problem, so no student ever triggers it.

    Calling get_oracle_tests() is the whole mechanism: the validation wiring
    already lives there. This adds no validation logic of its own — it iterates,
    skips what is already done, and keeps one bad problem from killing the run.

    Returns {total, already_strong, already_weak, newly_validated, failed,
    elapsed_sec}, where `failed` carries the slug and reason for each error."""
    problems = load_problems() if problems is None else problems
    summary = {"total": len(problems), "already_strong": 0, "already_weak": 0,
               "newly_validated": 0, "failed": []}
    start = time.time()

    for i, problem in enumerate(problems, 1):
        slug = problem.get("slug", "?")
        head = f"  [{i}/{len(problems)}] {slug}"

        # Pure cache read — never triggers a validation pass. Keyed by content,
        # so two problems sharing a slug can't be mistaken for each other.
        cached = _load_cache().get(content_hash(problem))
        if _is_validated(cached) and cached["strong"]:
            summary["already_strong"] += 1
            print(f"{head}: SKIP (cached strong)")
            continue

        # Already carries a verdict but did not clear the bar. get_oracle_tests
        # validates once and only once, so re-calling it would do nothing.
        if _is_validated(cached):
            summary["already_weak"] += 1
            print(f"{head}: SKIP (already validated, still WEAK — needs a look)")
            continue

        t0 = time.time()
        try:
            tests = get_oracle_tests(problem)
        except Exception as e:
            summary["failed"].append({"slug": slug, "error": f"{type(e).__name__}: {e}"})
            print(f"{head}: FAILED — {type(e).__name__}: {e}")
            continue

        if not tests:
            summary["failed"].append({"slug": slug, "error": "no oracle tests generated"})
            print(f"{head}: FAILED — no oracle tests generated")
            continue

        # Must key by content, not slug: a slug lookup can land on a
        # pre-migration orphan entry, which is a bare list, not a dict.
        entry = _load_cache().get(content_hash(problem))
        if not isinstance(entry, dict):
            entry = {}
        summary["newly_validated"] += 1
        print(f"{head}: VALIDATED {len(tests)} tests, "
              f"kill_rate={entry.get('kill_rate', 0.0):.2f} "
              f"(direct {entry.get('kill_rate_direct', 0.0):.2f}) "
              f"{'STRONG' if entry.get('strong') else 'WEAK'} "
              f"in {time.time() - t0:.1f}s")

    summary["elapsed_sec"] = round(time.time() - start, 1)
    return summary


def main():
    problems = load_problems()
    print(f"Warming up oracles for {len(problems)} problems.\n")
    summary = warm_up_oracles(problems)

    print(f"\n{'=' * 70}")
    print(f"  total            : {summary['total']}")
    print(f"  already strong   : {summary['already_strong']} (skipped)")
    print(f"  already weak     : {summary['already_weak']} (skipped, needs attention)")
    print(f"  newly validated  : {summary['newly_validated']}")
    print(f"  failed           : {len(summary['failed'])}")
    for f in summary["failed"]:
        print(f"      {f['slug']}: {f['error']}")
    print(f"  elapsed          : {summary['elapsed_sec']}s")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏸️  Paused. Rerun to resume — validated oracles stay cached.")
        sys.exit(0)
