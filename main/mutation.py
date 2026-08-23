"""
mutation.py — mutation testing to measure oracle test-suite strength.

An oracle suite is only as good as the wrong answers it can reject. Today the
inputs are LLM-generated and the expected values come from running the
ground-truth solution, so nothing proves the suite would catch a WRONG student
answer: if every cached Palindrome Number test happens to expect True, a
student submitting `return True` scores 100%.

This module measures that gap directly. It builds "mutants" — mechanical,
single-point edits of the ground-truth solution — and checks whether the oracle
notices. A mutant the oracle cannot tell apart from the original is a concrete
hole in the suite.

Two hard rules:
  1. Mutant generation is pure `ast` and never calls a model. It grades our own
     machinery, so it must be exactly repeatable.
  2. The LLM is used in exactly ONE place — proposing candidate INPUTS for a
     mutant that survived — and never gets a say in whether two outputs differ.
     That is always decided by executing both programs and comparing results.
"""
import ast
import copy
import json

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


# ── mutation operators (deterministic, no model call) ─────────────────────

_CMP_FLIP = {
    ast.Lt: ast.LtE, ast.LtE: ast.Lt,       # boundary / off-by-one
    ast.Gt: ast.GtE, ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,   # negation
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
}


def _sites(tree: ast.AST) -> list[tuple[int, str, str]]:
    """Find every mutable point as (walk_index, kind, label).

    Positions are indices into `ast.walk`, whose order is deterministic for a
    given tree — and `copy.deepcopy` preserves that order — so an index found
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

    Pure AST rewriting — deterministic, no model call, no side effects on the
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
                      n: int = CUTOFF_4_MAX_COUNTEREXAMPLE_CANDIDATES) -> list[list]:
    """Ask the LLM for up to n input argument-lists that might make the two
    programs disagree. INPUTS ONLY — the model never reports outputs, and its
    opinion about them is never read."""
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
        f"what the edit changed and target the code path it sits on — a "
        f"boundary value, a sign change, an empty or single-element case.\n"
        f"Every input must still satisfy the problem's stated constraints.\n"
        f"Each input is a JSON array of the {len(params)} positional "
        f"argument(s) in order: {', '.join(params) or 'unknown'}.\n"
        f"Give {n} candidates, most likely first. Do NOT report outputs.\n"
        f'Return JSON only: {{"inputs": [[arg1, ...], ...]}}'
    )
    raw = chat(GEN_MODEL, "You generate test inputs as strict JSON. No prose.",
               [{"role": "user", "content": prompt}], temperature=0.3, fmt="json")
    data = _first_json_obj(raw) or {}
    inputs = [i if isinstance(i, list) else [i]
              for i in data.get("inputs", []) if i is not None]
    return inputs[:n]


def _probe_inputs(tests: list) -> list[list]:
    """Boundary variants of the inputs we already have: one argument at a time
    pushed to 0/±1/its neighbours, a list or string emptied, a bool flipped.

    Free, deterministic, and it catches the off-by-one and sign mutants the
    model reliably fails to think of — so `likely_equivalent` is only reached
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
                         entry: str | None, seen: set, tests: list) -> dict | None:
    """One last attempt to kill a survivor: free boundary probes first, then the
    model's suggested inputs. Either way the verdict comes from executing both
    programs — the model only ever supplies inputs."""
    found = _first_disagreement(original, mutant_code, entry, _probe_inputs(tests), seen)
    if found:
        return found
    return _first_disagreement(original, mutant_code, entry,
                               _candidate_inputs(problem, original, mutant_code), seen)


# ── one full pass ─────────────────────────────────────────────────────────

def evaluate_oracle(problem: dict, oracle_tests: list) -> dict:
    """Score `oracle_tests` against every mutant of the problem's solution.

    A mutant is killed when it disagrees with the ORIGINAL on any test, or
    crashes/hangs where the original did not. Survivors get one counterexample
    search; a disagreement found there becomes a new oracle test (and kills the
    mutant), otherwise the mutant is judged "likely equivalent" — behaviorally
    identical to the original, not a real gap — and is dropped from the
    denominator so it cannot count against the oracle.

    Returns kill_rate, strong, total_mutants, killed, likely_equivalent,
    new_tests and per-mutant results. `kill_rate_direct` is the same ratio
    BEFORE counterexample repair — the strength of the suite exactly as handed
    in, which is the number to use when comparing two suites."""
    solution = problem.get("solution", "")
    entry = get_resolved_entry(problem)["entry_name"]
    mutants = generate_mutants(solution)

    tests = list(oracle_tests)              # working set grows; caller's list untouched
    base = run_solution(solution, [t["input"] for t in tests], entry_name=entry)
    if not base["ok"]:
        return {"kill_rate": 0.0, "kill_rate_direct": 0.0, "strong": False,
                "total_mutants": len(mutants), "killed": 0, "killed_direct": 0,
                "likely_equivalent": 0, "new_tests": [], "mutants": [],
                "error": base["error"]}
    expected = base["results"]              # ground truth, parallel to `tests`

    seen = {_key(t["input"]) for t in tests}
    status = [None] * len(mutants)
    new_tests = []

    # Phase 1 — the oracle exactly as handed in. Kept separate from the repair
    # pass below so `killed_direct` measures THIS suite, not one already
    # improved by an earlier mutant's counterexample (which would make the
    # number depend on mutant order).
    for i, m in enumerate(mutants):
        if _disagrees(m["code"], entry, [t["input"] for t in tests], expected):
            status[i] = "killed"
    killed_direct = status.count("killed")

    # Phase 2 — one repair attempt per survivor.
    for i, m in enumerate(mutants):
        if status[i]:
            continue
        # A test found for an earlier survivor may already cover this one — free.
        if _disagrees(m["code"], entry, [t["input"] for t in new_tests],
                      [t["expected"] for t in new_tests]):
            status[i] = "killed_on_retry"
            continue
        try:
            found = _counterexample_test(problem, solution, m["code"], entry, seen, tests)
        except Exception as e:
            # Model unreachable — we could not even ask, so we must NOT claim
            # equivalence. Left as a survivor, counted against the oracle.
            print(f"  ⚠️  counterexample search failed ({type(e).__name__}); "
                  f"{m['label']} left as survived")
            status[i] = "survived"
            continue
        if found:
            seen.add(_key(found["input"]))
            new_tests.append(found)
            status[i] = "killed_on_retry"
        else:
            status[i] = "likely_equivalent"

    results = [{"label": m["label"], "status": s} for m, s in zip(mutants, status)]
    killed = status.count("killed") + status.count("killed_on_retry")
    equivalent = status.count("likely_equivalent")
    total = len(mutants)
    denom = total
    kill_rate = killed / denom if denom else 0.0
    return {"kill_rate": kill_rate,
            # No mutants means the oracle was never tested, not that it aced the
            # test. Must match kill_rate's 0.0 above -- returning 1.0 here made a
            # zero-mutant problem report a perfect direct score alongside a 0.00
            # kill_rate, which is self-contradictory.
            "kill_rate_direct": killed_direct / denom if denom else 0.0,
            "strong": kill_rate >= CUTOFF_1_KILL_RATE,
            "total_mutants": total, "killed": killed, "killed_direct": killed_direct,
            "likely_equivalent": equivalent, "new_tests": new_tests,
            "mutants": results, "error": None}


# ── orchestrator ──────────────────────────────────────────────────────────

def validate_oracle(problem: dict, initial_tests: list,
                    max_rounds: int = CUTOFF_2_MAX_EXPAND_ROUNDS) -> dict:
    """Evaluate the oracle, and while it is still weak pull in a fresh batch of
    LLM-generated + ground-truth-verified tests and try again, up to
    max_rounds. Stops as soon as the oracle is STRONG.

    Returns the final evaluation plus `rounds` and `final_tests` (the full
    grown suite — persist this to keep the improvement)."""
    tests = list(initial_tests)
    result, rnd = None, 0

    for rnd in range(1, max_rounds + 1):
        result = evaluate_oracle(problem, tests)
        tests += result["new_tests"]
        print(f"  [mutation] round {rnd}: kill_rate={result['kill_rate']:.2f} "
              f"(direct {result['kill_rate_direct']:.2f}) "
              f"killed={result['killed']}/{result['total_mutants']} "
              f"equivalent={result['likely_equivalent']} "
              f"{'STRONG' if result['strong'] else 'WEAK'}")
        if result["strong"] or rnd == max_rounds:
            break

        seen = {_key(t["input"]) for t in tests}
        fresh = [t for t in make_oracle_tests(problem) if _key(t["input"]) not in seen]
        if not fresh:
            print("  [mutation] no new tests available; stopping early")
            break
        tests += fresh

    return {**result, "rounds": rnd, "final_tests": tests}
