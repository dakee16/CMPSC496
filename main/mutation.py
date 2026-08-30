"""
mutation.py - mutation testing to measure oracle test-suite strength.

An oracle suite is only as good as the wrong answers it can reject. Today the
inputs are LLM-generated and the expected values come from running the
ground-truth solution, so nothing proves the suite would catch a WRONG student
answer: if every cached Palindrome Number test happens to expect True, a
student submitting `return True` scores 100%.

This module measures that gap directly. It builds "mutants" - mechanical,
single-point edits of the ground-truth solution - and checks whether the oracle
notices. A mutant the oracle cannot tell apart from the original is a concrete
hole in the suite.

Two hard rules:
  1. Mutant generation is pure `ast` and never calls a model. It grades our own
     machinery, so it must be exactly repeatable.
  2. The LLM is used in exactly ONE place - proposing candidate INPUTS for a
     mutant that survived - and never gets a say in whether two outputs differ.
     That is always decided by executing both programs and comparing results.
"""
import ast
import copy
import json
import random

from .identity import get_resolved_entry
from .ollama_client import chat
from tests.sandbox import (
    GEN_MODEL,
    _first_json_obj,
    _norm,
    make_oracle_tests,
    run_solution,
)

# ── CUTOFFS ───────────────────────────────────────────────────────────────
# PLACEHOLDERS. Every number below is a first guess, NOT a calibrated value.
# They must be re-tuned against the full 100-problem set later in the project
# (measure the kill-rate distribution across all problems, then pick the
# threshold that separates genuinely weak oracles from strong ones). Do not
# cite these as results and do not treat them as final.
CUTOFF_1_KILL_RATE = 0.85                   # kill_rate at/above this ⇒ oracle is STRONG
CUTOFF_2_MAX_EXPAND_ROUNDS = 3              # max validate_oracle rounds before giving up
CUTOFF_4_MAX_COUNTEREXAMPLE_CANDIDATES = 5  # LLM input guesses per surviving mutant

# Mutants can loop forever where the original did not (e.g. `num //= 10`
# mutated to `num *= 10`), so they get a tighter leash than a real solution.
_MUTANT_TIMEOUT = 5.0

# Ceiling on the free boundary probes tried before falling back to the model.
_MAX_PROBE_INPUTS = 12

# PLACEHOLDER, calibratable. Size of the secondary deterministic sweep used to
# prove a survivor genuinely equivalent. Bigger = more confident exclusions and
# slower runs; tune once the kill-rate distribution over the 100-problem set is
# known. This sweep never calls the LLM.
_EQUIVALENCE_SWEEP_SIZE = 200

# PLACEHOLDER, calibratable. Below this many mutants a solution is too trivial
# for the kill rate to mean anything - two-sum once scored STRONG off 2 mutants.
# Such a result is flagged insufficient_mutants and is never strong.
_MIN_MUTANTS = 3

# Fixed seed: the sweep must be exactly repeatable, like mutant generation.
_SWEEP_SEED = 20260823


# ── mutation operators (deterministic, no model call) ─────────────────────

_CMP_FLIP = {
    ast.Lt: ast.LtE, ast.LtE: ast.Lt,       # boundary / off-by-one
    ast.Gt: ast.GtE, ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,   # negation
    ast.In: ast.NotIn, ast.NotIn: ast.In,   # containment negation - the
    # one comparison a hash-map/set-lookup solution (two-sum, contains-
    # duplicate, ...) actually uses, so without this a solution with no
    # <, >, +, -, *, // and no int/bool literals yields ZERO mutants.
}
_BOOL_FLIP = {ast.And: ast.Or, ast.Or: ast.And}
_BIN_FLIP = {
    ast.Add: ast.Sub, ast.Sub: ast.Add,
    ast.Mult: ast.Div, ast.Div: ast.Mult,
    ast.FloorDiv: ast.Mult,                 # digit-stripping loops are everywhere here
}
_OP_SYMBOL = {
    ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
    ast.Eq: "==", ast.NotEq: "!=", ast.And: "and", ast.Or: "or",
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//",
    ast.In: "in", ast.NotIn: "not in",
}


def _sites(tree: ast.AST) -> list[tuple[int, str, str]]:
    """Find every mutable point as (walk_index, kind, label).

    Positions are indices into `ast.walk`, whose order is deterministic for a
    given tree - and `copy.deepcopy` preserves that order - so an index found
    on the original tree addresses the same node in any copy of it."""
    found = []
    for i, node in enumerate(ast.walk(tree)):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            old = type(node.ops[0])
            if old in _CMP_FLIP:
                new = _CMP_FLIP[old]
                found.append((i, "cmp", f"line {line}: {_OP_SYMBOL[old]} -> {_OP_SYMBOL[new]}"))
        elif isinstance(node, ast.BoolOp):
            old = type(node.op)
            if old in _BOOL_FLIP:
                new = _BOOL_FLIP[old]
                found.append((i, "bool", f"line {line}: {_OP_SYMBOL[old]} -> {_OP_SYMBOL[new]}"))
        elif isinstance(node, ast.BinOp):
            old = type(node.op)
            if old in _BIN_FLIP:
                new = _BIN_FLIP[old]
                found.append((i, "bin", f"line {line}: {_OP_SYMBOL[old]} -> {_OP_SYMBOL[new]}"))
        elif isinstance(node, ast.Constant):
            # bool must be checked first: isinstance(True, int) is True.
            if isinstance(node.value, bool):
                found.append((i, "const", f"line {line}: {node.value} -> {not node.value}"))
            elif isinstance(node.value, (int, float)):
                found.append((i, "const", f"line {line}: {node.value} -> {node.value + 1}"))
    return found


def _apply(node: ast.AST, kind: str) -> None:
    """Apply this site's single edit in place."""
    if kind == "cmp":
        node.ops[0] = _CMP_FLIP[type(node.ops[0])]()
    elif kind == "bool":
        node.op = _BOOL_FLIP[type(node.op)]()
    elif kind == "bin":
        node.op = _BIN_FLIP[type(node.op)]()
    elif kind == "const":
        node.value = (not node.value) if isinstance(node.value, bool) else node.value + 1


def generate_mutants(solution_code: str) -> list[dict]:
    """Mechanical single-point edits of `solution_code`, one mutant per site.

    Pure AST rewriting - deterministic, no model call, no side effects on the
    input. Covers comparison flips (< <= > >= == !=), and/or flips, arithmetic
    flips (+ - * / //), int/float bumps (+1) and boolean literal flips.
    Returns [{"code": mutant_source, "label": "line 3: < -> <="}, ...]."""
    try:
        tree = ast.parse(solution_code)
    except SyntaxError:
        return []

    original_src = ast.unparse(tree)
    mutants = []
    for index, kind, label in _sites(tree):
        mutated = copy.deepcopy(tree)
        _apply(list(ast.walk(mutated))[index], kind)
        try:
            code = ast.unparse(mutated)
        except Exception:
            continue
        if code != original_src:            # a no-op edit is not a mutant
            mutants.append({"code": code, "label": label})
    return mutants


# ── counterexample search: LLM proposes INPUTS, execution decides ─────────

def _key(inp) -> str:
    """Stable dedupe key for an input argument-list."""
    return json.dumps(inp, sort_keys=True, default=str)


def _candidate_inputs(problem: dict, original: str, mutant_code: str,
                      n: int = CUTOFF_4_MAX_COUNTEREXAMPLE_CANDIDATES,
                      emit=None) -> list[list]:
    """Ask the LLM for up to n input argument-lists that might make the two
    programs disagree. INPUTS ONLY - the model never reports outputs, and its
    opinion about them is never read.

    `emit`, when given, is called with progress-event dicts for live UIs.
    It never changes behaviour - leaving it None is the production path."""
    resolved = get_resolved_entry(problem)
    name, params = resolved["entry_name"], resolved["params"]
    sig = f"{name}({', '.join(params)})" if name else problem.get("title", "")
    prompt = (
        f"Problem: {problem.get('title','')}\n\n"
        f"Description:\n{(problem.get('description') or '')[:600]}\n\n"
        f"Function: {sig}\n\n"
        f"PROGRAM A (correct):\n{original}\n\n"
        f"PROGRAM B (a single-point edit of A):\n{mutant_code}\n\n"
        f"Find inputs where A and B return DIFFERENT values. Look at exactly "
        f"what the edit changed and target the code path it sits on - a "
        f"boundary value, a sign change, an empty or single-element case.\n"
        f"Every input must still satisfy the problem's stated constraints.\n"
        f"Each input is a JSON array of the {len(params)} positional "
        f"argument(s) in order: {', '.join(params) or 'unknown'}.\n"
        f"Give {n} candidates, most likely first. Do NOT report outputs.\n"
        f'Return JSON only: {{"inputs": [[arg1, ...], ...]}}'
    )
    if emit:
        emit({"type": "llm_asking", "detail": f"asking model for up to {n} "
              f"inputs that might make the two programs disagree"})
    raw = chat(GEN_MODEL, "You generate test inputs as strict JSON. No prose.",
               [{"role": "user", "content": prompt}], temperature=0.3, fmt="json")
    data = _first_json_obj(raw) or {}
    inputs = [i if isinstance(i, list) else [i]
              for i in data.get("inputs", []) if i is not None]
    inputs = inputs[:n]
    if emit:
        emit({"type": "llm_candidates", "inputs": inputs})
    return inputs


def _probe_inputs(tests: list) -> list[list]:
    """Boundary variants of the inputs we already have: one argument at a time
    pushed to 0/±1/its neighbours, a list or string emptied, a bool flipped.

    Free, deterministic, and it catches the off-by-one and sign mutants the
    model reliably fails to think of - so `likely_equivalent` is only reached
    after these have been tried too."""
    out = []
    for inp in [t["input"] for t in tests][:2]:
        for i, arg in enumerate(inp):
            if isinstance(arg, bool):
                variants = [not arg]
            elif isinstance(arg, (int, float)):
                variants = [0, 1, -1, -arg, arg + 1, arg - 1]
            elif isinstance(arg, (list, str)):
                variants = [type(arg)(), arg[:1]]
            else:
                continue
            for v in variants:
                out.append([*inp[:i], v, *inp[i + 1:]])
    return list({_key(c): c for c in out}.values())[:_MAX_PROBE_INPUTS]


def _sweep_inputs(tests: list, n: int = _EQUIVALENCE_SWEEP_SIZE) -> list[list]:
    """A broad, deterministic, type-directed input sweep - no LLM involved.

    Shapes are taken from the inputs we already have, then each argument is
    varied far more widely than _probe_inputs does: signs, zeros, boundaries,
    long and empty sequences, duplicates, sorted and reversed orders. Seeded,
    so two runs produce byte-identical sweeps."""
    seeds = [t["input"] for t in tests]
    if not seeds:
        return []
    rng = random.Random(_SWEEP_SEED)
    shape = seeds[0]

    def values_for(arg):
        if isinstance(arg, bool):
            return [True, False]
        if isinstance(arg, int):
            return ([0, 1, -1, 2, -2, 7, 10, -10, 99, -99, 100, 121, -121,
                     1000, -1000, 2147483647, -2147483648]
                    + [rng.randint(-5000, 5000) for _ in range(24)])
        if isinstance(arg, float):
            return [0.0, 1.0, -1.0, 0.5, -0.5, 1e6, -1e6] + \
                   [rng.uniform(-1000, 1000) for _ in range(12)]
        if isinstance(arg, str):
            base = ["", "a", "ab", "aba", "abc", "aa", "Z", "zZ", "0", "123",
                    "racecar", "ab ba", "!@#", "aeiou", "x" * 40]
            return base + ["".join(rng.choice("abcxyz01 ")
                                   for _ in range(rng.randint(0, 12)))
                           for _ in range(16)]
        if isinstance(arg, list):
            inner = arg[0] if arg else 0
            if isinstance(inner, str):
                pool = [[], ["a"], ["a", "b"], ["a", "a"], ["ab", "ba"]]
            else:
                pool = [[], [0], [1], [-1], [1, 1], [0, 0], [1, 2, 3],
                        [3, 2, 1], [-1, -2, -3], [5, 5, 5], [2, 7, 11, 15],
                        list(range(10)), list(range(10, 0, -1))]
            return pool + [[rng.randint(-50, 50) for _ in range(rng.randint(0, 8))]
                           for _ in range(16)]
        return [arg]

    pools = [values_for(a) for a in shape]
    out, guard = [], 0
    while len(out) < n and guard < n * 20:
        guard += 1
        cand = [rng.choice(p) for p in pools]
        out.append(cand)
    # Dedupe but keep order - deterministic either way.
    return list({_key(c): c for c in out}.values())[:n]


def _proves_equivalent(original: str, mutant_code: str, entry: str | None,
                       tests: list) -> bool:
    """POSITIVE evidence that a mutation changed nothing.

    Runs both programs across the broad deterministic sweep and requires exact
    agreement on EVERY input - including agreeing on which inputs raise. Only
    that earns exclusion from the denominator. A sweep that cannot run (either
    program failing to execute at all) proves nothing and returns False."""
    inputs = _sweep_inputs(tests)
    if not inputs:
        return False
    orig = run_solution(original, inputs, entry_name=entry, timeout=_MUTANT_TIMEOUT)
    if not orig["ok"]:
        return False                        # no baseline ⇒ nothing proven
    mut = run_solution(mutant_code, inputs, entry_name=entry, timeout=_MUTANT_TIMEOUT)
    if not mut["ok"]:
        return False                        # died where the original ran ⇒ not equivalent
    return all(_norm(a) == _norm(b)
               for a, b in zip(orig["results"], mut["results"]))


def _first_disagreement(original: str, mutant_code: str, entry: str | None,
                        candidates: list, seen: set) -> dict | None:
    """Run BOTH programs on `candidates` and compare the real results. Returns
    the first genuine disagreement as {"input", "expected"}, else None."""
    candidates = [c for c in candidates if _key(c) not in seen]
    if not candidates:
        return None

    orig = run_solution(original, candidates, entry_name=entry, timeout=_MUTANT_TIMEOUT)
    if not orig["ok"]:
        return None                         # can't trust these inputs at all
    mut = run_solution(mutant_code, candidates, entry_name=entry, timeout=_MUTANT_TIMEOUT)

    for i, inp in enumerate(candidates):
        expected = orig["results"][i]
        if isinstance(expected, dict) and "__error__" in expected:
            continue                        # original fails here → not a valid oracle test
        got = mut["results"][i] if mut["ok"] else {"__error__": mut["error"]}
        if _norm(got) != _norm(expected):
            return {"input": inp, "expected": expected}
    return None


def _disagrees(code: str, entry: str | None, inputs: list, expected: list) -> bool:
    """True if `code` differs from the original's recorded outputs on any input,
    or dies/hangs where the original did not."""
    if not inputs:
        return False
    run = run_solution(code, inputs, entry_name=entry, timeout=_MUTANT_TIMEOUT)
    if not run["ok"]:
        return True
    return any(_norm(a) != _norm(b) for a, b in zip(run["results"], expected))


def _counterexample_test(problem: dict, original: str, mutant_code: str,
                         entry: str | None, seen: set, tests: list,
                         emit=None) -> dict | None:
    """One last attempt to kill a survivor: free boundary probes first, then the
    model's suggested inputs. Either way the verdict comes from executing both
    programs - the model only ever supplies inputs.

    `emit`, when given, narrates each phase for live UIs; None (the default,
    and the production path) changes nothing."""
    probes = _probe_inputs(tests)
    if emit:
        emit({"type": "probes", "inputs": probes,
              "detail": "free deterministic boundary probes (no LLM cost)"})
    found = _first_disagreement(original, mutant_code, entry, probes, seen)
    if found:
        if emit:
            emit({"type": "disagreement", "source": "probe",
                  "input": found["input"], "expected": found["expected"]})
        return found
    if emit:
        emit({"type": "probes_exhausted",
              "detail": "no probe made the programs disagree; asking the model"})
    found = _first_disagreement(
        original, mutant_code, entry,
        _candidate_inputs(problem, original, mutant_code, emit=emit), seen)
    if emit:
        if found:
            emit({"type": "disagreement", "source": "llm",
                  "input": found["input"], "expected": found["expected"]})
        else:
            emit({"type": "search_empty",
                  "detail": "none of the model's inputs made the programs "
                            "disagree - this alone proves nothing"})
    return found


# ── one full pass ─────────────────────────────────────────────────────────

def _detailed_disagreement(code: str, entry: str | None,
                           inputs: list, expected: list) -> dict:
    """Same verdict as _disagrees, plus per-test detail for live UIs.

    Semantics MUST stay identical to _disagrees: a crash/hang where the
    original ran counts as killed, and any normalised mismatch counts as
    killed. Returns {"killed": bool, "crashed": bool, "error": str | None,
    "per_test": [{"input", "expected", "got", "pass"}]}."""
    if not inputs:
        return {"killed": False, "crashed": False, "error": None, "per_test": []}
    run = run_solution(code, inputs, entry_name=entry, timeout=_MUTANT_TIMEOUT)
    if not run["ok"]:
        return {"killed": True, "crashed": True, "error": run.get("error"),
                "per_test": [{"input": i, "expected": e, "got": None,
                              "pass": False} for i, e in zip(inputs, expected)]}
    per_test, killed = [], False
    for inp, exp, got in zip(inputs, expected, run["results"]):
        ok = _norm(got) == _norm(exp)
        killed = killed or not ok
        per_test.append({"input": inp, "expected": exp, "got": got, "pass": ok})
    return {"killed": killed, "crashed": False, "error": None, "per_test": per_test}


def evaluate_oracle(problem: dict, oracle_tests: list, emit=None) -> dict:
    """Score `oracle_tests` against every mutant of the problem's solution.

    `emit`, when given, is called with structured progress events so a live UI
    can watch the pass happen (mutants, per-test results, retries, running
    kill rate). Leaving it None - the production path - changes nothing.

    A mutant is killed when it disagrees with the ORIGINAL on any test, or
    crashes/hangs where the original did not. Every survivor then lands in
    exactly one of three buckets:

      killed_on_retry    the counterexample search found a real distinguishing
                         input -> it becomes a new oracle test, mutant counts
                         as KILLED (numerator +1, denominator +1)
      proven_equivalent  the broad deterministic sweep found EXACT agreement on
                         every one of _EQUIVALENCE_SWEEP_SIZE inputs. Positive
                         evidence, so excluded from the denominator entirely
      unresolved         neither -> counts AGAINST the oracle (denominator +1,
                         numerator +0)

    The distinction that matters: "the search found nothing" is not evidence of
    equivalence, it is absence of evidence, and treating it as proof is what
    fabricated the old near-universal 1.00 scores. Only the sweep's positive
    result buys an exclusion. Search errors are unresolved, never excluded.

        kill_rate = (killed + killed_on_retry) / (total - proven_equivalent)

    A solution yielding fewer than _MIN_MUTANTS mutants is reported
    insufficient_mutants and is never strong, however the ratio comes out.
    `kill_rate_direct` is the same ratio BEFORE counterexample repair - the
    strength of the suite exactly as handed in, for comparing two suites."""
    solution = problem.get("solution", "")
    entry = get_resolved_entry(problem)["entry_name"]
    mutants = generate_mutants(solution)
    if emit:
        emit({"type": "mutants", "total": len(mutants),
              "mutants": [{"index": i, "label": m["label"], "code": m["code"]}
                          for i, m in enumerate(mutants)]})

    tests = list(oracle_tests)              # working set grows; caller's list untouched
    base = run_solution(solution, [t["input"] for t in tests], entry_name=entry)
    if not base["ok"]:
        if emit:
            emit({"type": "base_run_failed", "error": base["error"]})
        return {"kill_rate": 0.0, "kill_rate_direct": 0.0, "strong": False,
                "insufficient_mutants": len(mutants) < _MIN_MUTANTS,
                "status": "error", "total_mutants": len(mutants), "killed": 0,
                "killed_on_retry": 0, "killed_direct": 0,
                "proven_equivalent": 0, "unresolved": 0,
                "new_tests": [], "mutants": [], "error": base["error"]}
    expected = base["results"]              # ground truth, parallel to `tests`

    seen = {_key(t["input"]) for t in tests}
    status = [None] * len(mutants)
    new_tests = []

    # Phase 1 - the oracle exactly as handed in. Kept separate from the repair
    # pass below so `killed_direct` measures THIS suite, not one already
    # improved by an earlier mutant's counterexample (which would make the
    # number depend on mutant order).
    inputs_p1 = [t["input"] for t in tests]
    for i, m in enumerate(mutants):
        if emit:
            # Detailed path: single execution, verdict semantics identical to
            # _disagrees (crash ⇒ killed, any normalised mismatch ⇒ killed).
            d = _detailed_disagreement(m["code"], entry, inputs_p1, expected)
            if d["killed"]:
                status[i] = "killed"
            done = i + 1
            emit({"type": "mutant_result", "phase": 1, "index": i,
                  "label": m["label"], "killed": d["killed"],
                  "crashed": d["crashed"], "error": d["error"],
                  "per_test": d["per_test"]})
            emit({"type": "tally", "phase": 1, "processed": done,
                  "total": len(mutants), "killed_so_far": status.count("killed"),
                  "kill_rate_so_far": (status.count("killed") / done)})
        else:
            if _disagrees(m["code"], entry, inputs_p1, expected):
                status[i] = "killed"
    killed_direct = status.count("killed")
    if emit:
        emit({"type": "phase1_done", "killed_direct": killed_direct,
              "total": len(mutants),
              "survivors": [i for i, s in enumerate(status) if not s]})

    # Phase 2 - one repair attempt per survivor.
    for i, m in enumerate(mutants):
        if status[i]:
            continue
        # Per-mutant emit wrapper so every event carries its mutant's identity.
        em = ((lambda d, _i=i, _l=m["label"]:
               emit({**d, "index": _i, "label": _l})) if emit else None)
        if em:
            em({"type": "retry_start",
                "detail": "survived the suite as handed in - one repair attempt"})
        # A test found for an earlier survivor may already cover this one - free.
        if _disagrees(m["code"], entry, [t["input"] for t in new_tests],
                      [t["expected"] for t in new_tests]):
            status[i] = "killed_on_retry"
            if em:
                em({"type": "killed_by_earlier_counterexample",
                    "detail": "a test added for an earlier survivor already "
                              "kills this one - free"})
                em({"type": "mutant_final", "status": "killed_on_retry"})
            continue
        try:
            found = _counterexample_test(problem, solution, m["code"], entry,
                                         seen, tests, emit=em)
        except Exception as e:
            # Model unreachable - we could not even ask. Never silently excluded:
            # an unanswered question counts against the oracle.
            print(f"  ⚠️  counterexample search failed ({type(e).__name__}); "
                  f"{m['label']} left UNRESOLVED")
            status[i] = "unresolved"
            if em:
                em({"type": "search_error", "error": f"{type(e).__name__}: {e}"})
                em({"type": "mutant_final", "status": "unresolved"})
            continue
        if found:
            seen.add(_key(found["input"]))
            new_tests.append(found)
            status[i] = "killed_on_retry"
            if em:
                em({"type": "oracle_grown", "new_test": found,
                    "n_tests": len(tests) + len(new_tests),
                    "detail": "real disagreement found - added as a new oracle "
                              "test, mutant killed on retry"})
                em({"type": "mutant_final", "status": "killed_on_retry"})
        else:
            # The LLM-guided search came up empty. That alone is NOT evidence of
            # equivalence - it is just as likely a gap the search missed. Demand
            # positive proof from the broad deterministic sweep; anything less
            # stays UNRESOLVED and counts against us.
            if em:
                em({"type": "equivalence_sweep_start",
                    "sweep_size": _EQUIVALENCE_SWEEP_SIZE,
                    "detail": "searched hard and found nothing - that is not "
                              "proof of harmlessness. Running the broad "
                              "deterministic sweep to demand positive evidence"})
            equivalent = _proves_equivalent(solution, m["code"], entry, tests)
            status[i] = "proven_equivalent" if equivalent else "unresolved"
            if em:
                em({"type": "equivalence_sweep_result", "equivalent": equivalent,
                    "detail": ("exact agreement on every sweep input - proven "
                               "equivalent, excluded from the score"
                               if equivalent else
                               "sweep could not prove equivalence - UNRESOLVED, "
                               "counts AGAINST the oracle")})
                em({"type": "mutant_final", "status": status[i]})

    results = [{"label": m["label"], "status": s} for m, s in zip(mutants, status)]
    killed = status.count("killed") + status.count("killed_on_retry")
    killed_on_retry = status.count("killed_on_retry")
    proven_equivalent = status.count("proven_equivalent")
    unresolved = status.count("unresolved")
    total = len(mutants)

    # Only PROVEN equivalence leaves the denominator. Unresolved survivors stay
    # in it and drag the rate down - when in doubt, it counts against us.
    denom = total - proven_equivalent
    kill_rate = killed / denom if denom else 0.0

    # THE RATE THAT DECIDES TRUST. kill_rate above is the POST-REPAIR figure: it
    # credits the suite for counterexamples the search had to add during this
    # very pass, which is how the all-True palindrome oracle scored 1.00 STRONG
    # while missing every negative input. kill_rate_direct is the suite exactly
    # as it entered this pass. A repaired suite still improves - the new tests
    # are kept - but it earns STRONG only on a LATER pass, once those tests are
    # part of the suite it starts with. STRONG is always about the as-handed-in
    # suite. (No mutants ⇒ 0.0, never a free perfect score.)
    kill_rate_direct = killed_direct / denom if denom else 0.0

    # A solution too trivial to mutate cannot earn a verdict at all.
    insufficient = total < _MIN_MUTANTS
    strong = (not insufficient) and kill_rate_direct >= CUTOFF_1_KILL_RATE

    out = {"kill_rate": kill_rate,
           "kill_rate_direct": kill_rate_direct,
           "strong": strong,
           "insufficient_mutants": insufficient,
           "status": ("insufficient_mutants" if insufficient
                      else "strong" if strong else "weak"),
           "total_mutants": total, "killed": killed,
           "killed_on_retry": killed_on_retry, "killed_direct": killed_direct,
           "proven_equivalent": proven_equivalent, "unresolved": unresolved,
           "new_tests": new_tests, "mutants": results, "error": None}
    if emit:
        emit({"type": "evaluation_done", **{k: out[k] for k in (
            "kill_rate", "kill_rate_direct", "strong", "insufficient_mutants",
            "status", "total_mutants", "killed", "killed_on_retry",
            "killed_direct", "proven_equivalent", "unresolved")}})
    return out


# ── orchestrator ──────────────────────────────────────────────────────────

def validate_oracle(problem: dict, initial_tests: list,
                    max_rounds: int = CUTOFF_2_MAX_EXPAND_ROUNDS,
                    emit=None) -> dict:
    """Evaluate the oracle, and while it is still weak pull in a fresh batch of
    LLM-generated + ground-truth-verified tests and try again, up to
    max_rounds. Stops as soon as the oracle is STRONG.

    Returns the final evaluation plus `rounds` and `final_tests` (the full
    grown suite - persist this to keep the improvement)."""
    tests = list(initial_tests)
    result, rnd = None, 0

    for rnd in range(1, max_rounds + 1):
        if emit:
            emit({"type": "round_start", "round": rnd,
                  "max_rounds": max_rounds, "n_tests": len(tests)})
        result = evaluate_oracle(problem, tests, emit=emit)
        tests += result["new_tests"]
        if emit:
            emit({"type": "round_summary", "round": rnd,
                  "kill_rate_direct": result["kill_rate_direct"],
                  "kill_rate": result["kill_rate"],
                  "cutoff": CUTOFF_1_KILL_RATE, "status": result["status"],
                  "n_tests": len(tests)})
        print(f"  [mutation] round {rnd}: "
              f"kill_rate_direct={result['kill_rate_direct']:.2f} "
              f"(GATES vs {CUTOFF_1_KILL_RATE}) "
              f"post_repair={result['kill_rate']:.2f} "
              f"killed={result['killed']}/{result['total_mutants']} "
              f"(on_retry={result['killed_on_retry']}) "
              f"unresolved={result['unresolved']} "
              f"proven_equiv={result['proven_equivalent']} "
              f"{result['status'].upper()}")
        if result["strong"] or rnd == max_rounds:
            break

        seen = {_key(t["input"]) for t in tests}
        if emit:
            emit({"type": "expanding_suite", "round": rnd,
                  "detail": "still weak - generating a fresh batch of "
                            "ground-truth-verified tests and re-running"})
        fresh = [t for t in make_oracle_tests(problem) if _key(t["input"]) not in seen]
        if not fresh:
            print("  [mutation] no new tests available; stopping early")
            if emit:
                emit({"type": "expansion_empty",
                      "detail": "no new distinct tests could be generated; "
                                "stopping early"})
            break
        tests += fresh
        if emit:
            emit({"type": "suite_expanded", "added": len(fresh),
                  "n_tests": len(tests),
                  "new_tests": fresh})

    return {**result, "rounds": rnd, "final_tests": tests}