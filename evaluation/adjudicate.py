"""Three-way adjudication of a benchmark run. Read-only: never touches the
corpus, the grader, or production caches.

  offline_adjudicated      offline mode could settle it; scored normally.
  model_dependent_deferred it reached the ADAPTER/JUDGE boundary and stopped
                           only because --offline disables model calls. Not a
                           false rejection and not a grader error — unmeasured.
  genuine_system_error     our infrastructure failed for some other reason.

The distinction that matters for renamed-variables cases: reaching Tier 3 and
being deferred is expected offline; being DETERMINISTICALLY rejected before the
adapter boundary is a real failure.
"""
import json
import sys
from collections import Counter, defaultdict

# Offline mode makes _ask_judge raise, so this reason_code means "we got as far
# as needing a model and stopped there".
_DEFERRED_CODES = {"judge_unavailable"}
_SYSTEM_CODES = {"harness_error", "runner_error", "oracle_missing",
                 "oracle_weak", "oracle_unvalidated", "oracle_malformed",
                 "oracle_load_failed", "no_current_chunk"}


def bucket(r: dict) -> str:
    if r.get("system_error"):
        return "genuine_system_error"
    rc = r.get("reason_code", "")
    if rc in _DEFERRED_CODES:
        return "model_dependent_deferred"
    if rc in _SYSTEM_CODES or r.get("actual") == "error":
        return "genuine_system_error"
    return "offline_adjudicated"


def main(path, cases_path=None):
    rows = [json.loads(l) for l in open(path)]
    by = defaultdict(list)
    for r in rows:
        by[bucket(r)].append(r)

    adj = by["offline_adjudicated"]
    scored = [r for r in adj if r["expected"] != "ambiguous"]
    hits = [r for r in scored if r["expected"] == r["actual"]]
    miss = [r for r in scored if r["expected"] != r["actual"]]
    fa = [r for r in miss if r["expected"] == "incorrect" and r["actual"] == "correct"]
    fr = [r for r in miss if r["expected"] == "correct" and r["actual"] == "incorrect"]
    bad_consume = [r for r in rows if r["actual"] in ("indeterminate", "error")
                   and r.get("consumed_attempt")]

    n = len(scored) or 1
    print(f"TOTAL CASES: {len(rows)}")
    for k in ("offline_adjudicated", "model_dependent_deferred", "genuine_system_error"):
        print(f"  {k:<26} {len(by[k]):>4}  ({len(by[k])/len(rows)*100:.1f}%)")
    print(f"\nADJUDICABLE SUBSET: {len(scored)} scored")
    print(f"  accuracy            : {len(hits)/n:.4f}")
    print(f"  false acceptance    : {len(fa)/n:.4f}  ({len(fa)} cases)")
    print(f"  false rejection     : {len(fr)/n:.4f}  ({len(fr)} cases)")
    print(f"  deterministic        : {sum(1 for r in scored if r['deterministic'])/n:.4f}")
    print(f"  attempts wrongly consumed on indeterminate/error: {len(bad_consume)}")

    print("\nVERDICT x TIER (all rows)")
    for k, v in sorted(Counter((r["actual"], r["tier"]) for r in rows).items()):
        print(f"  {k[0]:<14} {k[1]:<20} {v}")
    print("\nREASON CODES")
    for k, v in sorted(Counter(r["reason_code"] for r in rows).items(),
                       key=lambda kv: -kv[1]):
        print(f"  {k:<28} {v}")

    print("\nBY CATEGORY  (adjudicated acc | deferred | sys)")
    cat = defaultdict(lambda: [0, 0, 0, 0])   # hit, scored, deferred, sys
    for r in rows:
        b = bucket(r); c = cat[r["category"]]
        if b == "offline_adjudicated" and r["expected"] != "ambiguous":
            c[1] += 1; c[0] += int(r["expected"] == r["actual"])
        elif b == "model_dependent_deferred": c[2] += 1
        elif b == "genuine_system_error": c[3] += 1
    for k, (h, s, d, e) in sorted(cat.items()):
        acc = f"{h/s:.3f}" if s else "  n/a"
        print(f"  {k:<32} {acc} (n={s:>3})  deferred={d:>3}  sys={e}")

    if miss:
        print(f"\nMISCLASSIFIED ({len(miss)}):")
        for r in miss[:40]:
            print(f"  {r['slug'][:34]:<34} ch{r['chunk_index']} {r['category']:<28} "
                  f"exp={r['expected']:<9} got={r['actual']:<13} "
                  f"tier={r['tier']:<20} {r['reason_code']}")
    else:
        print("\nMISCLASSIFIED: none")

    # A renamed-variables case rejected BEFORE the adapter boundary is a real
    # failure, not a deferral. Surface those explicitly.
    pre_adapter = [r for r in rows if r["category"] == "renamed_variables"
                   and bucket(r) == "offline_adjudicated"
                   and r["actual"] == "incorrect"]
    print(f"\nrenamed_variables rejected BEFORE the adapter boundary "
          f"(real failures): {len(pre_adapter)}")
    for r in pre_adapter[:15]:
        print(f"  {r['slug'][:34]:<34} ch{r['chunk_index']} tier={r['tier']:<20} "
              f"{r['reason_code']}")
    return by


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "evaluation/out/results_private.jsonl")
