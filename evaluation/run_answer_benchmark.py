"""Offline answer-checking benchmark.

Never mutates production state: the oracle cache and chunk pool are COPIED to a
temp dir, and the session DB is a temp file. The real shared grading engine
(main.grading.grade_submission) is used — the point is to measure it, not a
re-implementation of it.
"""
import argparse
import csv
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from collections import Counter, defaultdict

from .schema import SCHEMA_VERSION, CaseResult


def _load_frozen(workdir):
    """Copy production caches into a sandbox and point the code at the copies."""
    import main.identity as ident
    import tests.sandbox as sb
    src_cache, src_pool = "tests/tests_cache.json", "main/chunk_pool.json"
    dst_cache = os.path.join(workdir, "tests_cache.json")
    shutil.copy(src_cache, dst_cache)
    pool = json.load(open(src_pool)) if os.path.exists(src_pool) else {}
    sb._CACHE_PATH = dst_cache
    ident._RESOLVED_PATH = os.path.join(workdir, "resolved.json")
    os.environ["MICROTUTOR_SESSION_DB"] = os.path.join(workdir, "sessions.sqlite3")
    return json.load(open(dst_cache)), pool


def _problems_from_db(hashes):
    """Problem dicts are needed for grading; pull them from Supabase once."""
    import os as _os
    from dotenv import load_dotenv
    from supabase import create_client
    load_dotenv(".env")
    sb = create_client(_os.environ["SUPABASE_URL"], _os.environ["SUPABASE_KEY"])
    rows = sb.table("problems").select(
        "slug, title, description, difficulty, solution").execute().data or []
    from main.identity import content_hash
    return {content_hash(p): p for p in rows if content_hash(p) in hashes}


def run(args):
    workdir = tempfile.mkdtemp(prefix="mt_bench_")
    cache, pool = _load_frozen(workdir)
    strong = {k: v for k, v in cache.items()
              if isinstance(v, dict) and v.get("strong") and v.get("final_tests")}
    usable = {k for k in pool if k in strong}
    problems = _problems_from_db(usable)
    if args.slugs:
        want = set(args.slugs.split(","))
        problems = {h: p for h, p in problems.items() if p["slug"] in want}

    from .corpus import build_cases
    cases = sorted(build_cases(problems, pool, cache), key=lambda c: c.case_id)
    if args.limit:
        cases = cases[:args.limit]

    os.makedirs(args.output_dir, exist_ok=True)
    raw_path = os.path.join(args.output_dir, "results_private.jsonl")
    done = set()
    if args.resume and os.path.exists(raw_path):
        for line in open(raw_path):
            try: done.add(json.loads(line)["case_id"])
            except Exception: pass
        print(f"  resuming: {len(done)} case(s) already recorded")

    import main.grading as g
    from main.grading import grade_submission
    from main.identity import content_hash
    if args.offline:
        # Deterministic pass: no adapter generation, no judge. A case that
        # needs them is recorded as such rather than silently mocked correct.
        g._request_adaptation = lambda *a, **k: ("", [])
        g._ask_judge = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("tier4 skipped in --offline"))

    results = []
    with open(raw_path, "a") as sink:
        for i, c in enumerate(cases, 1):
            if c.case_id in done:
                continue
            problem = problems[c.content_hash]
            dec = pool[c.content_hash][0]
            session = {
                "slug": c.slug, "title": problem.get("title", ""),
                "description": problem.get("description", ""),
                "solution": problem.get("solution", ""),
                "header": dec["header"], "chunks": dec["chunks"],
                "index": c.chunk_index,
                "accepted": [a.model_dump() for a in c.accepted_prefix],
            }
            t0 = time.perf_counter()
            try:
                r = grade_submission(session, c.student_answer)
                res = CaseResult(
                    case_id=c.case_id, slug=c.slug, chunk_index=c.chunk_index,
                    category=c.category, expected=c.expected, actual=r.verdict,
                    tier=r.tier, deterministic=r.deterministic,
                    reason_code=r.reason_code, divergent=r.divergent,
                    consumed_attempt=r.consume_attempt,
                    latency_ms=(time.perf_counter() - t0) * 1000)
            except Exception as e:
                res = CaseResult(
                    case_id=c.case_id, slug=c.slug, chunk_index=c.chunk_index,
                    category=c.category, expected=c.expected, actual="error",
                    tier="system", deterministic=False, reason_code="runner_error",
                    consumed_attempt=False, system_error=repr(e)[:200],
                    latency_ms=(time.perf_counter() - t0) * 1000)
            results.append(res)
            sink.write(res.model_dump_json() + "\n"); sink.flush()
            if i % 25 == 0:
                print(f"  {i}/{len(cases)}")
    if args.resume and done:
        results = [CaseResult(**json.loads(l)) for l in open(raw_path)]
    metrics(results, cases, args.output_dir)
    shutil.rmtree(workdir, ignore_errors=True)
    return results


def metrics(results, cases, outdir):
    unamb = [r for r in results if r.expected != "ambiguous"]
    amb = [r for r in results if r.expected == "ambiguous"]
    conf = Counter((r.expected, r.actual) for r in unamb)
    n = len(unamb) or 1
    hits = sum(1 for r in unamb if r.expected == r.actual)
    fa = [r for r in unamb if r.expected == "incorrect" and r.actual == "correct"]
    fr = [r for r in unamb if r.expected == "correct" and r.actual == "incorrect"]
    ind = [r for r in unamb if r.actual == "indeterminate"]
    bad_consume = [r for r in results
                   if r.actual in ("indeterminate", "error") and r.consumed_attempt]
    by_cat = defaultdict(lambda: [0, 0])
    for r in unamb:
        by_cat[r.category][1] += 1
        by_cat[r.category][0] += int(r.expected == r.actual)
    by_tier = Counter(r.tier for r in unamb)
    lat = defaultdict(list)
    for r in results:
        lat[r.tier].append(r.latency_ms)

    summary = {
        "schema_version": SCHEMA_VERSION, "total_cases": len(results),
        "unambiguous": len(unamb), "ambiguous_excluded": len(amb),
        "accuracy": round(hits / n, 4),
        "false_acceptance_rate": round(len(fa) / n, 4),
        "false_rejection_rate": round(len(fr) / n, 4),
        "indeterminate_rate": round(len(ind) / n, 4),
        "deterministic_coverage": round(
            sum(1 for r in unamb if r.deterministic) / n, 4),
        "attempts_wrongly_consumed": len(bad_consume),
        "system_errors": sum(1 for r in results if r.system_error),
        "confusion": {f"{k[0]}->{k[1]}": v for k, v in conf.items()},
        "tier_distribution": dict(by_tier),
        "accuracy_by_category": {k: {"acc": round(v[0] / v[1], 4), "n": v[1]}
                                 for k, v in sorted(by_cat.items())},
        "latency_ms_by_tier": {k: {"p50": round(statistics.median(v), 1),
                                   "max": round(max(v), 1), "n": len(v)}
                               for k, v in lat.items()},
        "misclassified": [{"case_id": r.case_id, "slug": r.slug,
                           "chunk": r.chunk_index, "category": r.category,
                           "expected": r.expected, "actual": r.actual,
                           "tier": r.tier, "reason_code": r.reason_code}
                          for r in unamb if r.expected != r.actual],
    }
    json.dump(summary, open(os.path.join(outdir, "summary.json"), "w"), indent=2)
    with open(os.path.join(outdir, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "slug", "chunk", "category", "expected", "actual",
                    "tier", "deterministic", "reason_code", "latency_ms"])
        for r in results:
            w.writerow([r.case_id, r.slug, r.chunk_index, r.category, r.expected,
                        r.actual, r.tier, r.deterministic, r.reason_code,
                        round(r.latency_ms, 1)])
    with open(os.path.join(outdir, "report.md"), "w") as f:
        f.write(f"# Answer-checking benchmark (schema {SCHEMA_VERSION})\n\n"
                f"- cases: {len(results)} ({len(unamb)} scored, {len(amb)} ambiguous)\n"
                f"- accuracy: **{summary['accuracy']:.3f}**\n"
                f"- false acceptance: {summary['false_acceptance_rate']:.4f}\n"
                f"- false rejection: {summary['false_rejection_rate']:.4f}\n"
                f"- indeterminate: {summary['indeterminate_rate']:.4f}\n"
                f"- attempts wrongly consumed: {summary['attempts_wrongly_consumed']}\n\n"
                f"## Tier distribution\n\n")
        for k, v in sorted(by_tier.items()):
            f.write(f"- `{k}`: {v}\n")
        f.write("\n## Accuracy by category\n\n")
        for k, v in summary["accuracy_by_category"].items():
            f.write(f"- `{k}`: {v['acc']:.3f} (n={v['n']})\n")
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("misclassified", "confusion")}, indent=2))
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(prog="evaluation.run_answer_benchmark")
    p.add_argument("--offline", action="store_true", default=True)
    p.add_argument("--live-judge", dest="offline", action="store_false")
    p.add_argument("--slugs"); p.add_argument("--limit", type=int)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--output-dir", default="evaluation/out")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args(argv)
    import random
    random.seed(a.seed)
    run(a)


if __name__ == "__main__":
    main()
