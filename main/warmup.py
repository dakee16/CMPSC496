"""
warmup.py - offline oracle precompute.

tests.sandbox.get_oracle_tests() mutation-tests an oracle the first time anyone
asks for it, and that pass can include LLM counterexample search for surviving
mutants. That cost belongs offline. The architecture promises students only ever
hit the grading path and never wait through a retry loop - but grader.py:178 and
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

from tests.sandbox import _is_validated, _load_cache, _save_cache, get_oracle_tests

from .identity import content_hash

load_dotenv()
SB = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


def load_problems() -> list[dict]:
    """Every problem with its ground-truth solution.

    Same query research/research_agent.py uses - note that its copy is currently
    pinned to a 3-slug TEMP slice, so this deliberately re-states the full-set
    version rather than importing it. Warm-up wants everything."""
    res = SB.table("problems").select(
        "slug, title, description, difficulty, solution"
    ).execute()
    return res.data or []


_VERDICT_KEYS = ("strong", "kill_rate", "kill_rate_direct", "validated_at",
                 "insufficient_mutants", "status")


def _clear_stale_verdicts() -> int:
    """Drop the trust VERDICT from every cached entry, keeping its tests.

    Needed whenever the scoring basis changes (e.g. STRONG moving from the
    post-repair kill_rate to kill_rate_direct): old verdicts were computed under
    a formula that no longer applies. Clearing rather than deleting the cache is
    deliberate - final_tests are still a perfectly good starting suite, and
    re-validating from them is both cheaper and fairer than regenerating.

    Clearing beats a force flag here because the authoritative skip lives inside
    get_oracle_tests(), which returns early on any entry carrying a verdict. A
    flag on warm_up_oracles alone would be bypassed there; removing the verdict
    makes BOTH layers re-validate with no flag threading."""
    cache = _load_cache()
    cleared = 0
    for key, entry in cache.items():
        if isinstance(entry, dict) and any(k in entry for k in _VERDICT_KEYS):
            cache[key] = {k: v for k, v in entry.items() if k not in _VERDICT_KEYS}
            cleared += 1
    _save_cache(cache)
    return cleared


def warm_up_oracles(problems: list[dict] | None = None,
                    force: bool = False) -> dict:
    """Force oracle validation for every problem, so no student ever triggers it.

    Calling get_oracle_tests() is the whole mechanism: the validation wiring
    already lives there. This adds no validation logic of its own - it iterates,
    skips what is already done, and keeps one bad problem from killing the run.

    force=True first strips every cached verdict (keeping the tests), so each
    problem is re-scored from scratch under the current formula. Resumability is
    preserved: a problem re-scored during the run carries a fresh verdict, so a
    restart correctly skips it rather than redoing it.

    Returns {total, already_strong, already_weak, newly_validated, blocked,
    failed, elapsed_sec}. `blocked` holds problems with no usable oracle (they
    are neither strong nor weak); `failed` holds hard errors."""
    problems = load_problems() if problems is None else problems
    if force:
        n = _clear_stale_verdicts()
        print(f"  [warmup] force: cleared {n} stale verdict(s), kept their tests\n")
    summary = {"total": len(problems), "already_strong": 0, "already_weak": 0,
               "newly_validated": 0, "blocked": [], "failed": []}
    start = time.time()

    for i, problem in enumerate(problems, 1):
        slug = problem.get("slug", "?")
        head = f"  [{i}/{len(problems)}] {slug}"

        # Pure cache read - never triggers a validation pass. Keyed by content,
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
            print(f"{head}: SKIP (already validated, still WEAK - needs a look)")
            continue

        t0 = time.time()
        try:
            tests = get_oracle_tests(problem)
        except Exception as e:
            summary["failed"].append({"slug": slug, "error": f"{type(e).__name__}: {e}"})
            print(f"{head}: FAILED - {type(e).__name__}: {e}")
            continue

        if not tests:
            # No executable ground truth (linked-list/tree inputs, in-place
            # mutators, unimportable annotations). Neither strong nor weak -
            # there is simply nothing to score. Reported, never crashes the run.
            summary["blocked"].append({"slug": slug, "reason": "no usable oracle tests"})
            print(f"{head}: BLOCKED - no usable oracle tests "
                  f"({time.time() - t0:.1f}s)")
            continue

        # Must key by content, not slug: a slug lookup can land on a
        # pre-migration orphan entry, which is a bare list, not a dict.
        entry = _load_cache().get(content_hash(problem))
        if not isinstance(entry, dict):
            entry = {}
        if entry.get("insufficient_mutants"):
            summary["blocked"].append({"slug": slug,
                                       "reason": "insufficient mutants to score"})
            print(f"{head}: BLOCKED - solution too trivial to mutate "
                  f"({time.time() - t0:.1f}s)")
            continue
        summary["newly_validated"] += 1
        print(f"{head}: {'STRONG' if entry.get('strong') else 'WEAK  '} "
              f"kill_rate_direct={entry.get('kill_rate_direct', 0.0):.2f} "
              f"(post_repair {entry.get('kill_rate', 0.0):.2f}) "
              f"{len(tests)} tests in {time.time() - t0:.1f}s")

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
        print("\n\n⏸️  Paused. Rerun to resume - validated oracles stay cached.")
        sys.exit(0)
