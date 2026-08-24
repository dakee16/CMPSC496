"""Manual-review queue. Nothing here is scored until a human approves it.

Correct candidates must independently pass the STRONG oracle as COMPLETE
functions via the direct labeling executor (never the grader). Incorrect
candidates carry independent failure evidence. Ambiguous items explain why a
forced correct/incorrect label would be wrong.
"""
import hashlib
import json
import os
import re

from main.execution import classify_run
from main.identity import content_hash
from .corpus import assemble

QUEUE_PATH = "evaluation/review_queue.json"


def _verify(header, refs, idx, candidate, tests):
    """Independent evidence: swap the chunk into the full reference solution and
    run the oracle directly."""
    bodies = list(refs); bodies[idx] = candidate
    code = assemble(header, bodies)
    m = re.search(r"def\s+(\w+)", code)
    r = classify_run(code, tests, entry_name=m.group(1) if m else None)
    return r.outcome, f"{r.passed}/{r.total}"


def build(problems, pool, cache):
    items = []

    def add(slug, h, idx, code, cat, expected, why, prefix=None, evidence=None):
        items.append({
            "case_id": "rq-" + hashlib.sha1(
                f"{h}:{idx}:{cat}:{code}".encode()).hexdigest()[:14],
            "slug": slug, "content_hash": h, "chunk_index": idx,
            "accepted_prefix": prefix or [], "candidate_code": code,
            "category": cat, "expected": expected, "author_rationale": why,
            "oracle_evidence": evidence,
            "reviewer_status": "pending",       # NEVER auto-approved
            "reviewer_notes": "",
        })

    for h, problem in problems.items():
        ents = pool.get(h) or []
        e = cache.get(h)
        if not ents or not isinstance(e, dict) or not e.get("strong"):
            continue
        header = ents[0]["header"]
        refs = [c.get("reference", "") for c in ents[0]["chunks"]]
        tests = e["final_tests"]
        slug = e.get("slug", "?")
        last = len(refs) - 1

        # ── alternate_implementation: same semantics, different construction.
        # Only admitted if the COMPLETE function passes the oracle.
        alt = None
        r = refs[last]
        if r.strip().startswith("return ") and " if " not in r:
            expr = r.strip()[len("return "):]
            alt = f"_alt = ({expr})\nreturn _alt"
        if alt:
            out, ev = _verify(header, refs, last, alt, tests)
            if out == "pass":
                add(slug, h, last, alt, "alternate_implementation", "correct",
                    "same value via an explicit temporary; verified as a complete "
                    "function by the direct executor", 
                    [{"code": x, "provenance": "student"} for x in refs[:last]],
                    {"outcome": out, "passed": ev})

        # ── divergent_interface: renames the LAST binding so the fixed
        # reference tail cannot consume it -> must route through Tier 3.
        if last > 0:
            prev = refs[last - 1]
            mm = re.match(r"\s*([A-Za-z_]\w*)\s*=", prev)
            if mm:
                nm = mm.group(1)
                div = re.sub(rf"\b{nm}\b", nm + "_alt", prev)
                out, ev = _verify(header, refs[:last - 1] + [div] + refs[last:],
                                  last - 1, div, tests)
                add(slug, h, last - 1, div, "divergent_interface",
                    "correct" if out == "pass" else "ambiguous",
                    f"renames '{nm}' so the fixed reference tail cannot consume "
                    f"it; correctness depends on a calibrated adapter",
                    [{"code": x, "provenance": "student"} for x in refs[:last - 1]],
                    {"outcome": out, "passed": ev})

        # ── adapter_bypass: a chunk that does nothing while a malicious tail
        # could recompute everything. Anti-bypass knockout must reject it.
        add(slug, h, max(0, last - 1), "pass", "adapter_bypass", "incorrect",
            "no-op chunk; any adapter that still passes is doing the student's "
            "work and must be rejected by the knockout check",
            [{"code": x, "provenance": "student"} for x in refs[:max(0, last - 1)]],
            {"outcome": "constructed", "passed": "n/a"})

        # ── ambiguous_tier4: partial/structural answer with no single defensible
        # deterministic label.
        add(slug, h, 0, "# I think we need a loop here\npass", "ambiguous_tier4",
            "ambiguous",
            "a comment plus a no-op states an approach without implementing it; "
            "forcing correct/incorrect would be arbitrary — the point is whether "
            "the grader abstains rather than guesses",
            [], {"outcome": "n/a", "passed": "n/a"})
    return items


def main():
    from dotenv import load_dotenv
    load_dotenv(".env")
    from supabase import create_client
    cache = json.load(open("tests/tests_cache.json"))
    pool = json.load(open("main/chunk_pool.json"))
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    rows = sb.table("problems").select(
        "slug,title,description,difficulty,solution").execute().data or []
    problems = {content_hash(p): p for p in rows if content_hash(p) in pool}
    items = build(problems, pool, cache)
    json.dump({"schema": "review-queue/1", "items": items},
              open(QUEUE_PATH, "w"), indent=2)
    from collections import Counter
    c = Counter(i["category"] for i in items)
    print(f"  queued {len(items)} PENDING items -> {QUEUE_PATH}")
    for k, v in sorted(c.items()):
        print(f"    {k:<26} {v}")
    print(f"  approved: {sum(1 for i in items if i['reviewer_status']=='approved')}")


if __name__ == "__main__":
    main()
