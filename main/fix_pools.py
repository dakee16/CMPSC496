"""fix_pools.py - ONE-OFF (30 Sep): bring every saved roadmap up to today's rules.

Three things are wrong with roadmaps saved before 30 Sep, all measured on the
live pool that day:

  unsafe   a step carries on inside a block a student's own code could not
           continue (splitter.unsafe_cut): get-postfix's 3, cut inside its
           `while` tokenizer (2 of 2 students stuck), and 2 of replace-
           variables', whose step 2 starts `elif` inside the teacher's own if.
           REPLACED by splits of the teacher's code that obey the rule.
  wording  a question fails today's prompt gate (17 of 79): mid-loop steps
           worded "prepare", steps that never say they hand a result on,
           questions naming the solution's variables. RE-WORDED, code unchanged.
  notes    no "starts with / leaves for the next step" lines (all of them -
           main/step_notes.py). ADDED, code and wording unchanged.

THREE STEPS, so what ships is exactly what was read (as main/reword_pools):

    docker compose exec app python -m main.fix_pools                          # list, $0
    docker compose exec app python -m main.fix_pools --draft /data/fix.json   # <= $2.00
    docker compose exec app python -m main.fix_pools --apply /data/fix.json   # $0

--draft is the only step that calls the model (wording ~0.7c a try, notes ~1c a
try, at most 3 tries each), under a hard cap. It prints old and new and changes
nothing. --apply makes no model call: per problem it swaps in only drafted
roadmaps that still pass every gate - the prompt gate, the notes gate, the cut
rule, and readiness (the serve gate, and the teacher's code typed flat accepted
at every step) - and only where the roadmaps drafted from are still saved
unchanged. Students mid-problem keep their session's roadmap. Don't run --apply
during an upload: the pool lock is per process.
"""
import collections
import json
import os
import sys

from . import splitter, step_notes
from .gates import check_notes, check_prompts
from .identity import content_hash
from .reword_pools import _ready_problems, fingerprint, readiness
from .run_phase1 import (_add_to_pool, _deserialize, _load_pool, _reading_saved_tests,
                         _serialize, _usable)
from .topup_pools import SpendCapReached, spend_cap

_COST = {"unsafe": 0.017, "wording": 0.017, "notes": 0.010}   # $ per roadmap, one try each


def needs(problem: dict, entry: dict) -> list[str]:
    """Which of the three this saved roadmap needs - [] when none."""
    why = []
    if splitter.unsafe_cut(problem, entry):
        return ["unsafe"]
    if check_prompts(_deserialize(entry)["chunks"], problem)["status"] != "pass":
        why.append("wording")
    if not all(c.get("starts_with") and c.get("leaves") for c in entry["chunks"]):
        why.append("notes")
    return why


def targets() -> list[tuple]:
    """(problem, [(entry, needs)]) for every ready problem with work to do."""
    pool, out = _load_pool(), []
    for key, problem in _ready_problems().items():
        work = [(e, needs(problem, e)) for e in _usable(problem, pool.get(key, []))]
        work = [(e, w) for e, w in work if w]
        if work:
            out.append((problem, work))
    return out


def valid(problem: dict, entry: dict) -> list[str]:
    """Why this drafted roadmap may not be served - [] when it may."""
    d = _deserialize(entry)
    why = []
    if check_prompts(d["chunks"], problem)["status"] != "pass":
        why.append("prompt gate")
    if any(c.starts_with or c.leaves for c in d["chunks"]) and \
            check_notes(d["chunks"], problem, entry.get("header") or "")["status"] != "pass":
        why.append("notes gate")
    if splitter.unsafe_cut(problem, entry):
        why.append("cut rule")
    return why or readiness(problem, entry)


def _show(rec: dict) -> None:
    print(f"\n{rec['slug']} - {rec['why']} ({len(rec['old'])} old -> {len(rec['new'])} new)")
    for n, e in enumerate(rec["new"], 1):
        print(f"  new roadmap {n}: step sizes "
              f"{[splitter.step_lines(c.get('reference')) for c in e['chunks']]}")
        for i, c in enumerate(e["chunks"], 1):
            print(f"    step {i}: {c.get('prompt')}")
            print(f"       starts with: {c.get('starts_with') or '-'}")
            print(f"       leaves:      {c.get('leaves') or '-'}")


def _finish(problem: dict, d: dict, reword: bool) -> dict:
    if reword:
        d = splitter.write_prompts(problem, d)
    return _serialize(step_notes.write_notes(problem, d))


def draft(path: str, cap: float) -> int:
    work, records, spent, stopped = targets(), [], [0.0], None
    try:
        with spend_cap(cap) as spent:
            for problem, entries in work:
                tested = _reading_saved_tests(problem)
                key = content_hash(problem)
                unsafe = [e for e, w in entries if "unsafe" in w]
                if unsafe:
                    new = [_finish(tested, d, True)
                           for d in splitter.plan(tested, want=len(unsafe))]
                    records.append({"key": key, "slug": problem["slug"], "why": "unsafe",
                                    "old": [fingerprint(e) for e in unsafe], "new": new})
                    _show(records[-1])
                for e, w in entries:
                    if "unsafe" in w:
                        continue
                    records.append({"key": key, "slug": problem["slug"], "why": "+".join(w),
                                    "old": [fingerprint(e)],
                                    "new": [_finish(tested, _deserialize(e), "wording" in w)]})
                    _show(records[-1])
    except SpendCapReached as e:
        stopped = e
    with open(path, "w") as f:
        json.dump(records, f, indent=1)
    if stopped:
        print(f"\nSTOPPED: {stopped}. The {len(records)} drafted so far are saved.")
    print(f"\nDrafted {len(records)} change(s) into {path}. Spent ${spent[0]:.3f} of the "
          f"${cap:.2f} cap. Nothing in the pool changed; read the above, then --apply {path}.")
    return 1 if stopped else 0


def apply(path: str) -> int:
    # Readiness grades for real and no model is called, so none of it belongs
    # in the live trace (the 29 Sep fake-spend lesson).
    os.environ["MICROTUTOR_TRACE_FILE"] = ""
    problems, pool = _ready_problems(), _load_pool()
    by_key = collections.defaultdict(list)
    for r in json.load(open(path)):
        by_key[r["key"]].append(r)
    refused, changed = 0, 0
    for key, recs in by_key.items():
        problem = problems.get(key)
        entries = list(pool.get(key, [])) if problem else []
        for r in recs:
            here = {fingerprint(e) for e in entries}
            if not problem or not set(r["old"]) <= here:
                refused += 1
                print(f"  {r['slug']}: NOT applied - the roadmap it was drafted from "
                      f"changed or is gone.")
                continue
            good = []
            for n in r["new"]:
                why = valid(problem, n)
                if why:
                    print(f"  {r['slug']}: a drafted roadmap refused - {why[0][:150]}")
                else:
                    good.append(n)
            if not good:
                refused += 1
                print(f"  {r['slug']}: NOT applied - no drafted roadmap passed; the old one stays.")
                continue
            entries = [e for e in entries if fingerprint(e) not in r["old"]] + good
            changed += 1
        if entries != pool.get(key, []):
            _add_to_pool(key, entries, replace=True)
            print(f"  {problem['slug']}: saved {len(entries)} roadmap(s).")
    print(f"\nApplied {changed} change(s); {refused} not applied.")
    return 1 if refused else 0


def main(argv: list[str]) -> int:
    if "--apply" in argv:
        return apply(argv[argv.index("--apply") + 1])
    if "--draft" in argv:
        cap = float(argv[argv.index("--max-dollars") + 1]) if "--max-dollars" in argv else 2.00
        return draft(argv[argv.index("--draft") + 1], cap)
    total, count = 0.0, collections.Counter()
    for problem, entries in targets():
        for _e, w in entries:
            count.update(w)
            total += max(_COST[x] for x in w)
        print(f"  {problem['slug']:42} " + ", ".join(
            f"{n} {w}" for w, n in collections.Counter(x for _e, w in entries for x in w).items()))
    print(f"\n{sum(count.values())} fix(es): {dict(count)}. About ${total:.2f} to draft "
          f"(more if the gates ask for retries; capped at $2.00). Nothing changed, $0. "
          f"Next: --draft FILE.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
