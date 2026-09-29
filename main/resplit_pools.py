"""resplit_pools.py - ONE-OFF: replace saved roadmaps that have a step over
splitter.MAX_STEP_LINES with roadmaps cut from the teacher's own code.

Measured on the server (28 Sep): every roadmap for calculateExpressions (8) and
calculator-get-postfix (7), and 3 of 6 for calculator-calculate, had a step of
23-48 lines. New uploads are covered by fill_pool; this fixes what is already
saved, once, from the teacher side.

    docker compose exec app python -m main.resplit_pools                  # dry run, $0
    docker compose exec app python -m main.resplit_pools --apply          # $1 cap
    docker compose exec app python -m main.resplit_pools --apply --max-dollars 2

Per problem: roadmaps within the limit are KEPT; the rest are replaced by the
splitter's, in one swap, so students are served the old set until the new one
is saved. Where the teacher's code cannot be cut to the limit at all
(get-postfix's best is 24 lines), a saved roadmap is replaced only by a split
whose biggest step is smaller than its own. The only model calls are the
splitter's step wording (~1c per roadmap). Don't run it during an upload - the
pool lock is per process, and this is a second process.
"""
import sys

from . import splitter
from .identity import content_hash
from .run_phase1 import (_POOL_TARGET, _add_to_pool, _load_pool,
                         _reading_saved_tests, _sb, _serialize, _usable)
from .sessions import context_of
from .topup_pools import SpendCapReached, spend_cap


def sizes(entry: dict) -> list[int]:
    return [splitter.step_lines(c.get("reference")) for c in entry["chunks"]]


def targets() -> list[tuple]:
    """(problem, kept entries, oversized entries) for every ready problem with
    at least one saved roadmap over the limit."""
    rows = _sb().table("problems").select(
        "slug, title, description, solution, context, assignment_id, "
        "group_slug, group_title, group_description").eq("ready", True).execute().data
    pool = _load_pool()
    out = []
    for row in rows or []:
        if not (row.get("solution") or "").strip():
            continue
        problem = {**row, **context_of(row)}            # the key students read
        usable = _usable(problem, pool.get(content_hash(problem), []))
        big = [e for e in usable if max(sizes(e), default=0) > splitter.MAX_STEP_LINES]
        if big:
            out.append((problem, [e for e in usable if e not in big], big))
    return out


def main(argv: list[str]) -> int:
    apply = "--apply" in argv
    cap = float(argv[argv.index("--max-dollars") + 1]) if "--max-dollars" in argv else 1.0
    plans = []
    for problem, kept, big in targets():
        slug = problem["slug"]
        new = splitter.plan(_reading_saved_tests(problem), want=_POOL_TARGET - len(kept))
        worst = min(max(sizes(e)) for e in big)
        new = [d for d in new if splitter.biggest_step(d) < worst]
        print(f"{slug}: keep {len(kept)} {[sizes(e) for e in kept]}")
        print(f"    replace {len(big)} {[sizes(e) for e in big]}")
        print(f"    with {len(new)} cut from the teacher's code "
              f"{[[splitter.step_lines(c.reference) for c in d['chunks']] for d in new]}")
        if new:
            plans.append((problem, kept, new))
    print(f"{len(plans)} problem(s) to change, {sum(len(n) for _, _, n in plans)} "
          f"roadmap(s) to word: ~${0.01 * sum(len(n) for _, _, n in plans):.2f}, "
          f"at most {2 * sum(len(n) for _, _, n in plans)} model calls.")
    if not apply:
        print("Dry run - nothing changed, $0. Add --apply to write the new roadmaps.")
        return 0

    spent, stopped = [0.0], None
    try:
        with spend_cap(cap) as spent:
            for problem, kept, new in plans:
                worded = [splitter.write_prompts(_reading_saved_tests(problem), d)
                          for d in new]
                _add_to_pool(content_hash(problem),
                             kept + [_serialize(d) for d in worded], replace=True)
                print(f"  {problem['slug']}: now {len(kept) + len(worded)} roadmap(s), "
                      f"biggest step {max(max(sizes(e)) for e in kept + [_serialize(d) for d in worded])}")
    except SpendCapReached as e:
        stopped = e
    if stopped:
        print(f"STOPPED: {stopped}. Problems finished before this are saved; "
              f"the rest are unchanged.")
    print(f"Spent ${spent[0]:.3f} of the ${cap:.2f} cap.")
    return 1 if stopped else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
