"""reword_pools.py - ONE-OFF: new wording for saved steps that start the loop
the next step carries on inside, but are not worded that way (gates.opens_unsaid).
The code of every step stays exactly as is.

THE MEASUREMENT (server, 30 Sep). prompts.SPLIT_PROMPTS_SYSTEM quoted one
example setup step - "Write code that gets everything ready to work through the
statements, and keep it for the next step." - and the model copied it into 6 of
200 live steps: all 5 calculateExpressions roadmaps and 1 of get-postfix's 3.
One of them is a 19-line step that ends in `return report`. The example is now
a rule. Then a student report (30 Sep) showed the copy was only one symptom:
"prepares everything needed to work through the expression" hid the same loop,
and no word search found it - so roadmaps are now found by their shape.

THREE STEPS, so the wording that ships is exactly the wording that was read:

    docker compose exec app python -m main.reword_pools                            # list, $0
    docker compose exec app python -m main.reword_pools --draft /data/reword.json  # <= $0.50
    docker compose exec app python -m main.reword_pools --apply /data/reword.json  # $0

--draft is the only step that calls the model (splitter.write_prompts, ~0.7c a
try, at most 3 tries a roadmap), under a hard cap. It prints old and new wording,
saves both to the file, and changes nothing. --apply makes no model call: it
swaps in exactly the drafted wording - a second draft could differ from what was
read - only where the roadmap's code is still what was drafted from and the
wording still passes the prompt gate. Then every roadmap of each problem it
touched must still be ready: the serve gate, and the teacher's code typed flat
accepted at every step. Students mid-problem keep their session's wording.
Don't run --apply during an upload: the pool lock is per process.
"""
import collections
import hashlib
import json
import os
import sys
import textwrap

from . import grading, splitter
from .gates import assert_serveable, check_prompts, opens_unsaid
from .identity import content_hash
from .run_phase1 import (_add_to_pool, _deserialize, _load_pool,
                         _reading_saved_tests, _sb, _usable)
from .sessions import CONTEXT_FIELDS, context_of
from .topup_pools import SpendCapReached, spend_cap

def fingerprint(entry: dict) -> str:
    """The roadmap exactly as drafted from - its code AND the wording being
    replaced. The pool holds roadmaps with the same code and different wording;
    code alone would re-word those too."""
    raw = json.dumps([entry.get("header"),
                      [[c.get("reference"), c.get("prompt")] for c in entry["chunks"]]])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _ready_problems() -> dict:
    rows = _sb().table("problems").select(
        "slug, title, description, solution, context, assignment_id, "
        "group_slug, group_title, group_description").eq("ready", True).execute().data
    problems = [{**r, **context_of(r)} for r in rows or [] if (r.get("solution") or "").strip()]
    return {content_hash(p): p for p in problems}           # the key students read


def _unsaid(entry: dict) -> list[int]:
    """Steps that start the loop the next step continues, without saying so.
    Found by SHAPE, not by words: the first re-word searched for the copied
    sentence and missed get-postfix's "prepares everything needed to work
    through the expression" - the roadmap in the 30 Sep student report."""
    return opens_unsaid(_deserialize(entry)["chunks"])


def affected(problems: dict) -> list[tuple]:
    """(problem, roadmap) for every served roadmap with such a step."""
    pool = _load_pool()
    return [(p, e) for key, p in problems.items() for e in _usable(p, pool.get(key, []))
            if _unsaid(e)]


def readiness(problem: dict, entry: dict) -> list[str]:
    """What stops this roadmap being served - [] when nothing does: the serve
    gate, then the teacher's own code typed flat through real grading, step by
    step. No model is asked: a step only tiers 3-4 could accept counts as not
    ready, because a student there could not be graded without one."""
    try:
        assert_serveable(_reading_saved_tests(problem), _deserialize(entry))
    except Exception as e:
        return [f"serve gate: {e}"[:200]]
    session = {**{k: problem.get(k) for k in ("slug", "title", "description", "solution")},
               "context": {k: problem[k] for k in CONTEXT_FIELDS if k in problem},
               "header": entry["header"], "accepted": [],
               "chunks": [{"step_id": c.get("step_id"), "prompt": c.get("prompt"),
                           "reference": c.get("reference") or ""} for c in entry["chunks"]]}
    real = grading._request_adaptation, grading._request_completion

    def no_model(*_a, **_k):
        raise RuntimeError("the readiness check asks no model")
    grading._request_adaptation = grading._request_completion = no_model
    try:
        for k, c in enumerate(session["chunks"]):
            s, typed = {**session, "index": k}, textwrap.dedent(c["reference"])
            grading._VERDICT_MEMO.clear()
            r = grading.grade_submission(s, typed)
            if r.verdict != "correct":
                return [f"{c['step_id']}: the teacher's own code got {r.verdict} ({r.tier})"]
            session["accepted"].append({"code": grading.align_submission(s, typed),
                                        "tier": r.tier})
    finally:
        grading._request_adaptation, grading._request_completion = real
    return []


def _show(record: dict) -> None:
    print(f"\n{record['slug']} (roadmap {record['fingerprint']}, prompt gate: {record['gate']})")
    for n, (old, new) in enumerate(zip(record["old"], record["new"]), 1):
        print(f"  step {n}\n    old: {old}\n    new: {new}")


def draft(path: str, cap: float) -> int:
    targets = affected(_ready_problems())
    records, spent, stopped = [], [0.0], None
    try:
        with spend_cap(cap) as spent:
            for problem, e in targets:
                worded = splitter.write_prompts(_reading_saved_tests(problem), _deserialize(e))
                records.append({"slug": problem["slug"], "key": content_hash(problem),
                                "fingerprint": fingerprint(e),
                                "old": [c.get("prompt") for c in e["chunks"]],
                                "new": [c.prompt for c in worded["chunks"]],
                                "gate": check_prompts(worded["chunks"], problem)["status"]})
                _show(records[-1])
    except SpendCapReached as e:
        stopped = e
    with open(path, "w") as f:
        json.dump(records, f, indent=1)
    if stopped:
        print(f"\nSTOPPED: {stopped}. The {len(records)} drafted so far are saved.")
    print(f"\nDrafted {len(records)} of {len(targets)} roadmap(s) into {path}. "
          f"Spent ${spent[0]:.3f} of the ${cap:.2f} cap. Nothing in the pool changed; "
          f"read the wording above, then --apply {path}.")
    return 1 if stopped else 0


def apply(path: str) -> int:
    # Readiness grades for real, and no model call is made here, so nothing in
    # this run belongs in the live trace (Sanan's 29 Sep warning: fake verdicts
    # and costs landed there from a check run in the container).
    os.environ["MICROTUTOR_TRACE_FILE"] = ""
    problems, pool = _ready_problems(), _load_pool()
    by_key = collections.defaultdict(list)
    for r in json.load(open(path)):
        by_key[r["key"]].append(r)
    skipped, touched = 0, []
    for key, recs in by_key.items():
        problem, wanted, entries = problems.get(key), {r["fingerprint"]: r for r in recs}, []
        for e in pool.get(key, []) if problem else []:
            r = wanted.get(fingerprint(e))
            if r and len(r["new"]) == len(e["chunks"]):
                new = {**e, "chunks": [{**c, "prompt": p} for c, p in zip(e["chunks"], r["new"])]}
                if check_prompts(_deserialize(new)["chunks"], problem)["status"] == "pass":
                    entries.append(new)
                    r["applied"] = True
                    continue
            entries.append(e)
        for r in recs:
            if not r.get("applied"):
                skipped += 1
                print(f"  {r['slug']}: NOT applied - its code changed since the draft, "
                      f"it is gone, or its new wording fails the prompt gate.")
        if any(r.get("applied") for r in recs):
            _add_to_pool(key, entries, replace=True)
            touched.append(key)
            print(f"  {problems[key]['slug']}: new wording on "
                  f"{sum(bool(r.get('applied')) for r in recs)} roadmap(s).")
    not_ready = [f"{problems[key]['slug']}: {why}" for key in touched
                 for e in _load_pool().get(key, []) for why in readiness(problems[key], e)]
    for line in not_ready:
        print(f"  NOT READY - {line}")
    if touched and not not_ready:
        print("Readiness: every roadmap is ready (serve gate + the teacher's code "
              "typed flat, accepted at every step).")
    return 1 if skipped or not_ready else 0


def main(argv: list[str]) -> int:
    if "--apply" in argv:
        return apply(argv[argv.index("--apply") + 1])
    if "--draft" in argv:
        cap = float(argv[argv.index("--max-dollars") + 1]) if "--max-dollars" in argv else 0.50
        return draft(argv[argv.index("--draft") + 1], cap)
    targets = affected(_ready_problems())
    for problem, e in targets:
        for i in _unsaid(e):
            print(f"  {problem['slug']} (roadmap {fingerprint(e)}) step {i + 1}: "
                  f"{e['chunks'][i].get('prompt')}")
    print(f"{len(targets)} roadmap(s) with a step that starts a loop but is not "
          f"worded that way: about "
          f"${0.007 * len(targets):.2f} to re-word, at most {3 * len(targets)} model "
          f"calls. Nothing changed, $0. Next: --draft FILE.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
