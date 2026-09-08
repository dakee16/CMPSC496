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

from .equivalence import proves_harmless
from .probe import (NEVER_REACHED, NO_INFECTION, PROPAGATION, UNKNOWN,
                    probe_site)
from .context import (build_program, doctest_covers, is_method,
                      solution_body)
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

# B3. How many decisions the probe must have watched before "it never behaved
# differently" is worth stopping the round loop over. Three observations is a
# coincidence; hundreds is a pattern. Deliberately far below the 1,470 seen on
# combination-sum and far above the handful a barely-exercised line produces.
_B3_MIN_DECISIONS = 25

# What each probe finding MEANS, in one line, for whoever reads the transcript.
# Three very different situations that used to look identical from outside.
_PROBE_DETAIL = {
    NEVER_REACHED: "no oracle test executes this line at all - this is a gap "
                   "in the test suite, not an equivalence question. Write a "
                   "test that reaches it and the mutant resolves itself.",
    NO_INFECTION:  "the line runs, but the edit never changed the decision it "
                   "makes - not once across the whole suite. Strong evidence "
                   "it is harmless, though not proof.",
    PROPAGATION:   "the line runs AND the edit really did change the decision, "
                   "but the difference never reached the answer. Could be "
                   "genuinely harmless, could be a real bug the output format "
                   "cannot show. Worth a look.",
    UNKNOWN:       "the probe could not run, so nothing was learned here.",
}


# ── mutation operators (deterministic, no model call) ─────────────────────

_CMP_FLIP = {
    ast.Lt: ast.LtE, ast.LtE: ast.Lt,       # boundary / off-by-one
    ast.Gt: ast.GtE, ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,   # negation
    ast.In: ast.NotIn, ast.NotIn: ast.In,   # containment negation - the
    # one comparison a hash-map/set-lookup solution (two-sum, contains-
    # duplicate, ...) actually uses, so without this a solution with no
    # <, >, +, -, *, // and no int/bool literals yields ZERO mutants.
    ast.Is: ast.IsNot, ast.IsNot: ast.Is,   # `x is None` guards
}
_BOOL_FLIP = {ast.And: ast.Or, ast.Or: ast.And}
_BIN_FLIP = {
    ast.Add: ast.Sub, ast.Sub: ast.Add,
    ast.Mult: ast.Div, ast.Div: ast.Mult,
    ast.FloorDiv: ast.Mult,                 # digit-stripping loops are everywhere here
    ast.Mod: ast.FloorDiv,                  # `n % 10` vs `n // 10` - digit extraction
    ast.Pow: ast.Mult,                      # `x ** k` vs `x * k`
    # Bit twiddling: hamming distance and power-of-two checks are built on it,
    # and without these they yield mutants only from their loop bounds.
    ast.BitXor: ast.BitAnd, ast.BitAnd: ast.BitOr, ast.BitOr: ast.BitAnd,
    ast.LShift: ast.RShift, ast.RShift: ast.LShift,
}
_OP_SYMBOL = {
    ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
    ast.Eq: "==", ast.NotEq: "!=", ast.And: "and", ast.Or: "or",
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//",
    ast.In: "in", ast.NotIn: "not in", ast.Is: "is", ast.IsNot: "is not",
    ast.Mod: "%", ast.Pow: "**",
    ast.BitXor: "^", ast.BitAnd: "&", ast.BitOr: "|",
    ast.LShift: "<<", ast.RShift: ">>",
}


def _sites(tree: ast.AST) -> list[tuple[int, str, str, int]]:
    """Find every mutable point as (walk_index, kind, label, slot).

    Positions are indices into `ast.walk`, whose order is deterministic for a
    given tree - and `copy.deepcopy` preserves that order - so an index found
    on the original tree addresses the same node in any copy of it. `slot` is
    the position WITHIN that node for the one kind that has several (a chained
    comparison's ops); every other kind ignores it."""
    found = []
    for i, node in enumerate(ast.walk(tree)):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Compare):
            # Every operator in the chain, not just the first. The old guard
            # was `len(node.ops) == 1`, which skipped `0 <= i < n` entirely -
            # and a bounds check is exactly where an off-by-one hides.
            for slot, op in enumerate(node.ops):
                old = type(op)
                if old in _CMP_FLIP:
                    new = _CMP_FLIP[old]
                    found.append((i, "cmp",
                                  f"line {line}: {_OP_SYMBOL[old]} -> {_OP_SYMBOL[new]}",
                                  slot))
        elif isinstance(node, ast.BoolOp):
            old = type(node.op)
            if old in _BOOL_FLIP:
                new = _BOOL_FLIP[old]
                found.append((i, "bool",
                              f"line {line}: {_OP_SYMBOL[old]} -> {_OP_SYMBOL[new]}", 0))
        elif isinstance(node, (ast.BinOp, ast.AugAssign)):
            # AugAssign holds its operator DIRECTLY (`node.op`), it is not
            # wrapped in a BinOp - so matching only BinOp left every `+=`,
            # `-=`, `*=` and `//=` in the codebase unmutated. Those are the
            # accumulator updates first-year solutions are mostly made of:
            # `total += digit`, `n //= 10`, `count -= 1`.
            old = type(node.op)
            if old in _BIN_FLIP:
                new = _BIN_FLIP[old]
                aug = "=" if isinstance(node, ast.AugAssign) else ""
                found.append((i, "bin",
                              f"line {line}: {_OP_SYMBOL[old]}{aug} -> "
                              f"{_OP_SYMBOL[new]}{aug}", 0))
        elif isinstance(node, ast.Constant):
            # bool must be checked first: isinstance(True, int) is True.
            if isinstance(node.value, bool):
                found.append((i, "const", f"line {line}: {node.value} -> {not node.value}", 0))
            elif isinstance(node.value, (int, float)):
                found.append((i, "const", f"line {line}: {node.value} -> {node.value + 1}", 0))

        # STATEMENT DELETION. Every operator above rewrites an expression, so a
        # solution made of straight-line statements - which is what a stateful
        # method IS - offered almost nothing to mutate: HW3's push and pop are
        # four lines each and yielded two mutants apiece, under the floor, so
        # their oracles could never earn a verdict at all however good they were.
        #
        # It is also the operator that matches the bug: "forgot to decrement the
        # count" is a removed statement, not a flipped operator, and it is the
        # single most common way a stateful method is wrong.
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            # A block of one cannot lose its only statement - that unparses to
            # an invalid function, not to a mutant. Module and ClassDef are
            # skipped because deleting a whole def is not a single-point edit
            # of the logic under test.
            if (not isinstance(block, list) or len(block) < 2
                    or isinstance(node, (ast.Module, ast.ClassDef))):
                continue
            for pos, stmt in enumerate(block):
                if isinstance(stmt, ast.Pass):
                    continue
                if (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
                        and isinstance(stmt.value.value, str)):
                    continue                 # a docstring is not behaviour, and
                                             # removing it survives every test
                text = ast.unparse(stmt).splitlines()[0]
                if len(text) > 40:
                    text = text[:37] + "..."
                found.append((i, f"del:{field}",
                              f"line {getattr(stmt, 'lineno', line)}: "
                              f"remove `{text}`", pos))
    return found


def _apply(node: ast.AST, kind: str, slot: int = 0) -> None:
    """Apply this site's single edit in place."""
    if kind.startswith("del:"):
        del getattr(node, kind[4:])[slot]
    elif kind == "cmp":
        node.ops[slot] = _CMP_FLIP[type(node.ops[slot])]()
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
    flips (+ - * / //), int/float bumps (+1), boolean literal flips, and the
    deletion of any statement from a block that holds more than one.
    Returns [{"code": mutant_source, "label": "line 3: < -> <="}, ...]."""
    try:
        tree = ast.parse(solution_code)
    except SyntaxError:
        return []

    original_src = ast.unparse(tree)
    walked = list(ast.walk(tree))
    # Parent links, so a mutated CONSTANT can be judged through the comparison
    # that encloses it: `remain < 0` -> `remain < 1` edits the Constant, but
    # the thing whose behaviour changed is the Compare around it.
    parent = {id(c): p for p in walked for c in ast.iter_child_nodes(p)}
    mutants = []
    for index, kind, label, slot in _sites(tree):
        mutated = copy.deepcopy(tree)
        mut_walked = list(ast.walk(mutated))
        mut_node = mut_walked[index]
        _apply(mut_node, kind, slot)

        # A3 - never generate a mutant we can PROVE changes nothing.
        #
        # Skipping beats generating-then-excusing: a mutant that never exists
        # cannot inflate the total, cannot be mistaken for a survivor, and
        # costs nothing downstream to judge. The alternative (create it, then
        # remove it from the denominator later) reaches the same rate by a
        # longer route and leaves phantom rows in the playground.
        #
        # proves_harmless is deliberately conservative: it answers True only
        # for the guarded-`elif` shape it can actually prove, and False for
        # everything it cannot reason about. It is the only check whose answer
        # DELETES a mutant, so a false positive would hide a real bug - see
        # main/equivalence.py, whose self-check pins the three ways earlier
        # versions of it got this wrong.
        #
        # Both mutation kinds that can alter a comparison are checked: the
        # operator itself (`< -> <=`) and a constant inside it (`< 0 -> < 1`).
        # Those are the same edit to the same predicate by two routes, and
        # skipping only the first left the second to be generated and then
        # puzzled over.
        orig_cmp = mut_cmp = None
        if kind == "cmp":
            orig_cmp, mut_cmp = walked[index], mut_node
        elif kind == "const":
            p_o, p_m = parent.get(id(walked[index])), None
            if isinstance(p_o, ast.Compare):
                # same position in the copied tree
                p_m = {id(c): p for p in mut_walked
                       for c in ast.iter_child_nodes(p)}.get(id(mut_node))
                orig_cmp, mut_cmp = p_o, p_m
        if orig_cmp is not None and isinstance(mut_cmp, ast.Compare) \
                and proves_harmless(tree, orig_cmp, mut_cmp):
            continue

        try:
            code = ast.unparse(mutated)
        except Exception:
            continue
        if code != original_src:            # a no-op edit is not a mutant
            # The site coordinates ride along so main/probe.py can instrument
            # the SAME expression later without re-deriving which one changed.
            mutants.append({"code": code, "label": label,
                            "index": index, "kind": kind, "slot": slot})
    return mutants


# ── counterexample search: LLM proposes INPUTS, execution decides ─────────

def _key(inp) -> str:
    """Stable dedupe key for an input argument-list."""
    return json.dumps(inp, sort_keys=True, default=str)


_DIVERGENCE_SYSTEM = (
    "You locate where two near-identical programs diverge. Answer as strict "
    "JSON only, no prose.")


def _ask_divergence_point(problem: dict, original: str, mutant_code: str,
                          emit=None) -> tuple[str | None, list]:
    """A5 tier 2 - ask the model for the DIVERGENCE POINT, not for an input.

    Measured, not assumed. Asking for inputs fails on this class of mutant no
    matter how the request is worded: on reverse_integer's overflow guard, the
    plain prompt missed, a hint that the edit was on an internal variable
    missed, and explicit step-by-step inversion instructions missed. Three
    independent tries of the SAME prompt at temperature 0.6 returned nearly
    byte-identical lists - so retrying buys nothing either.

    What worked first time was asking a different question. The model reliably
    names the variable and the exact value where the two branches part
    (`result`, 2147483648); it is the ARITHMETIC of working backwards to an
    input that it fumbles - it once reversed 2147483647 instead of the
    2147483648 it had just correctly identified, one sentence earlier.

    So the labour splits along the grain: the model does the semantics, and
    _candidates_from_target does the exact computation, which code gets right
    every time and can be unit-tested.

    Returns (variable_name, [candidate divergence values])."""
    user = (
        f"Problem: {problem.get('title','')}\n\n"
        f"PROGRAM A (correct):\n{original}\n\n"
        f"PROGRAM B (a single-point edit of A):\n{mutant_code}\n\n"
        "Do NOT give me inputs to the function. Identify the divergence point "
        "only.\n"
        "The edit may sit on an INTERNAL variable rather than on an argument. "
        "Name that variable, and give the EXACT value(s) of it for which A and "
        "B take different branches or produce different results. Usually this "
        "is a single boundary value.\n"
        'Return JSON only: {"variable": "<name>", "values": [<number>, ...]}')
    if emit:
        emit({"type": "llm_asking",
              "detail": "asking the model WHERE the two programs diverge - "
                        "the variable and its boundary value, not an input"})
    raw = chat(GEN_MODEL, _DIVERGENCE_SYSTEM, [{"role": "user", "content": user}],
               temperature=0.2, fmt="json")
    data = _first_json_obj(raw) or {}
    var = data.get("variable")
    values = [v for v in (data.get("values") or [])
              if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if emit:
        emit({"type": "divergence_point", "variable": var, "values": values})
    return (var if isinstance(var, str) else None), values[:4]


def _candidates_from_target(shape: list, target) -> list[list]:
    """Inputs that might drive an internal variable to `target`.

    The exact half of A5. Given the argument SHAPE of a real oracle test and a
    value the internal state must reach, derive candidate arguments by the
    transformations these problems actually use - the digits reversed (which is
    literally reverse_integer's computation), the sign flipped, the value
    itself, and its immediate neighbours.

    This is deliberately a small, honest set rather than a general inverter.
    Inverting an arbitrary function is not something we can do, and pretending
    otherwise would produce confident wrong answers. When none of these
    separate the two programs the caller learns nothing and the mutant stays
    undetermined, which is the correct outcome for "we could not tell"."""
    if not isinstance(target, int) or isinstance(target, bool):
        return []
    # Neighbours FIRST, then transform each of them - not the other way round.
    #
    # The model is reliable about which variable and roughly which boundary,
    # and unreliable by exactly one: asked about `> 2147483647 -> > 2147483648`
    # it has named the divergence value as 2147483648 on one run and 2147483647
    # on another. Transforming only the value it gave makes the whole search
    # hostage to that off-by-one; transforming its neighbours too costs a
    # handful of extra executions and absorbs it.
    derived = set()
    for base in (target, target + 1, target - 1):
        reversed_digits = int(str(abs(base))[::-1] or "0")
        derived.update({base, -base, reversed_digits, -reversed_digits})
    out = []
    for value in derived:
        for pos, arg in enumerate(shape):
            if isinstance(arg, bool) or not isinstance(arg, int):
                continue
            out.append([*shape[:pos], value, *shape[pos + 1:]])
    # Dedupe, keep order - the search must be repeatable.
    return list({_key(c): c for c in out}.values())


def _candidate_inputs(problem: dict, original: str, mutant_code: str,
                      n: int | None = None,
                      emit=None) -> list[list]:
    """Ask the LLM for up to n input argument-lists that might make the two
    programs disagree. INPUTS ONLY - the model never reports outputs, and its
    opinion about them is never read.

    `emit`, when given, is called with progress-event dicts for live UIs.
    It never changes behaviour - leaving it None is the production path."""
    # Late-bound rather than a def-time default: the playground tunes these
    # module constants per run, and a default frozen at import time would
    # silently ignore the override while the UI claimed it was in effect.
    if n is None:
        n = CUTOFF_4_MAX_COUNTEREXAMPLE_CANDIDATES
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


def _drop_blocks(tests: list, method: bool) -> list:
    """`tests` without the block PROGRAMS. A block is a program, not an argument
    list: there is nothing to vary, and mutating one would produce broken Python
    rather than a new case.

    `method` is what makes this safe. Only a METHOD problem can have a block,
    and a method's ordinary input is a LIST of calls - so a bare string in the
    first slot identifies a block unambiguously there. Deciding on the shape
    alone, with no idea what kind of problem it belongs to, threw away every
    test of every plain function whose first argument happens to be a string:
    is_palindrome('racecar') looks exactly like a block. Those problems lost
    both the free boundary probes and the whole deterministic sweep, so their
    mutants went to the paid search unexamined and came back undetermined."""
    if not method:
        return tests
    return [t for t in tests
            if not (t.get("input") and isinstance(t["input"][0], str))]


def _probe_inputs(tests: list, method: bool = False) -> list[list]:
    """Boundary variants of the inputs we already have: one argument at a time
    pushed to 0/±1/its neighbours, a list or string emptied, a bool flipped.

    Free, deterministic, and it catches the off-by-one and sign mutants the
    model reliably fails to think of - so `likely_equivalent` is only reached
    after these have been tried too."""
    tests = _drop_blocks(tests, method)
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


def _sweep_inputs(tests: list, n: int | None = None,
                  method: bool = False) -> list[list]:
    """A broad, deterministic, type-directed input sweep - no LLM involved.

    Shapes are taken from the inputs we already have, then each argument is
    varied far more widely than _probe_inputs does: signs, zeros, boundaries,
    long and empty sequences, duplicates, sorted and reversed orders. Seeded,
    so two runs produce byte-identical sweeps."""
    tests = _drop_blocks(tests, method)
    if n is None:                       # late-bound: see _candidate_inputs
        n = _EQUIVALENCE_SWEEP_SIZE
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


# A1 REMOVED _proves_equivalent().
#
# It ran the generated sweep and, on full agreement, marked a mutant
# `proven_equivalent` - which took it OUT of the denominator entirely. That is
# a proof-strength conclusion drawn from a sample, and the sample cannot bear
# it: reverse_integer's `> 2147483647 -> > 2147483648` is a real off-by-one in
# an overflow guard, and it was excused because none of the 183 generated
# inputs happened to produce the one internal value where the two differ. It
# shipped as STRONG at 100%. Sixty-six mutants across eighteen problems were
# excused the same way.
#
# Only main/equivalence.proves_harmless may remove a mutant now, because it
# returns a proof rather than an absence of counterexamples - and A3 applies it
# at GENERATION time, so those mutants are never created in the first place.
# The sweep survives below as what it always actually was: a free way to hunt
# for counterexamples, run BEFORE anything paid (B2).

# B1 - how many inputs share one subprocess and one timeout.
#
# The sweep used to run as a single batch. Some generated inputs make a
# solution loop forever (negative candidates for combination-sum, a target of
# two billion), and one such input killed the process and discarded ALL 183
# results - which is why combination-sum's kill rate never moved across six
# rounds. Batching bounds the damage to one chunk instead of everything.
_CHUNK = 16


def _runnable(problem: dict, source: str) -> str:
    """A method source made executable inside the class it belongs to.

    Everything below this point runs plain code strings through the sandbox, so
    seating happens ONCE here rather than being threaded through eight run
    helpers. A plain function is returned untouched."""
    if not is_method(problem):
        return source
    return build_program(problem, solution_body({"solution": source}))


def _first_disagreement(original: str, mutant_code: str, entry: str | None,
                        candidates: list, seen: set) -> dict | None:
    """Run BOTH programs on `candidates` and compare the real results. Returns
    the first genuine disagreement as {"input", "expected"}, else None.

    Runs in chunks so a single non-terminating input costs its own chunk rather
    than the whole search - see _CHUNK."""
    candidates = [c for c in candidates if _key(c) not in seen]
    if not candidates:
        return None

    for start in range(0, len(candidates), _CHUNK):
        part = candidates[start:start + _CHUNK]
        orig = run_solution(original, part, entry_name=entry,
                            timeout=_MUTANT_TIMEOUT)
        if not orig["ok"]:
            continue                        # this chunk hangs the ORIGINAL;
                                            # it proves nothing either way
        mut = run_solution(mutant_code, part, entry_name=entry,
                           timeout=_MUTANT_TIMEOUT)
        for i, inp in enumerate(part):
            expected = orig["results"][i]
            if isinstance(expected, dict) and "__error__" in expected:
                continue                    # original fails here → not a valid
                                            # oracle test, so not a witness
            # A mutant that hangs where the original completed IS a real
            # difference, and the whole chunk failing is the only signal we get.
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
    method = is_method(problem)
    probes = _probe_inputs(tests, method)
    if emit:
        emit({"type": "probes", "inputs": probes,
              "detail": "free deterministic boundary probes (no LLM cost)"})
    found = _first_disagreement(original, mutant_code, entry, probes, seen)
    if found:
        if emit:
            emit({"type": "disagreement", "source": "probe",
                  "input": found["input"], "expected": found["expected"]})
        return found

    # B2 - the broad free sweep runs HERE, before anything is paid for.
    #
    # It used to run last, after the model call, purely because it was framed
    # as an equivalence proof rather than a search. But running two programs
    # and comparing answers is one operation that settles both questions at
    # once: a disagreement is a counterexample, and total agreement is
    # evidence of harmlessness. So the free version goes first, and the model
    # is only paid once ~180 deterministic inputs have come up empty.
    sweep = _sweep_inputs(tests, method=method)
    if emit:
        emit({"type": "sweep", "n": len(sweep),
              "detail": f"{len(sweep)} generated inputs, still free "
                        f"(deterministic, no LLM cost)"})
    found = _first_disagreement(original, mutant_code, entry, sweep, seen)
    if found:
        if emit:
            emit({"type": "disagreement", "source": "sweep",
                  "input": found["input"], "expected": found["expected"]})
        return found

    if emit:
        emit({"type": "probes_exhausted",
              "detail": "no free input made the programs disagree; asking the model"})
    found = _first_disagreement(
        original, mutant_code, entry,
        _candidate_inputs(problem, original, mutant_code, emit=emit), seen)
    if found:
        if emit:
            emit({"type": "disagreement", "source": "llm",
                  "input": found["input"], "expected": found["expected"]})
        return found

    # ── A5 tier 2 - ask a different question, then do the arithmetic here ──
    #
    # Tier 1 asked for inputs and got boundary values that are not inputs. That
    # fails whenever the edited line works on a COMPUTED value: separating the
    # programs then needs the pre-image of a boundary under the function's own
    # computation, and that is exact arithmetic rather than reasoning.
    #
    # Only two tiers, and no repeats. Both were settled by experiment: three
    # identical tries returned near-identical answers, and a ladder of
    # progressively stronger hints (name the variable / work backwards / invert
    # explicitly) missed every time. Changing WHAT is asked worked; changing
    # how firmly, or how often, did not.
    shape = next((t["input"] for t in tests if t.get("input")), None)
    if shape is None:
        if emit:
            emit({"type": "search_empty",
                  "detail": "no oracle input to derive candidates from"})
        return None
    try:
        var, targets = _ask_divergence_point(problem, original, mutant_code,
                                             emit=emit)
    except Exception as e:
        if emit:
            emit({"type": "search_error", "error": f"{type(e).__name__}: {e}"})
        return None
    for target in targets:
        derived = _candidates_from_target(shape, target)
        if not derived:
            continue
        if emit:
            emit({"type": "inverted_candidates", "variable": var,
                  "target": target, "inputs": derived[:8],
                  "detail": f"model says {var} must reach {target}; these are "
                            f"the inputs that computation could come from"})
        found = _first_disagreement(original, mutant_code, entry, derived, seen)
        if found:
            if emit:
                emit({"type": "disagreement", "source": "inversion",
                      "input": found["input"], "expected": found["expected"]})
            return found

    if emit:
        emit({"type": "search_empty",
              "detail": "no input from either tier separated the programs - "
                        "that alone proves nothing, so this stays undetermined"})
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
      undetermined       nothing separated the two programs - not the free
                         probes, not the generated sweep, not the model. That
                         is ambiguity, not proof of harmlessness, so it is
                         REPORTED rather than resolved (see A4's range below)

    The distinction that matters: "the search found nothing" is not evidence of
    equivalence, it is absence of evidence. Nothing is excused on that basis any
    more (A1); mutants that CAN be proved harmless are never generated (A3).

        kill_rate_lower = killed_direct / total              (all undetermined are bugs)
        kill_rate_upper = killed_direct / (total - undetermined)   (all harmless)

    A solution yielding fewer than _MIN_MUTANTS mutants is reported
    insufficient_mutants and is never strong, however the ratio comes out.
    `kill_rate_direct` is the same ratio BEFORE counterexample repair - the
    strength of the suite exactly as handed in, for comparing two suites."""
    # The METHOD alone, kept under its own name for the whole function. Mutant
    # indices are positions in THIS tree, and `solution` is about to become the
    # assembled module - a different tree with eight times as many nodes, in
    # which those indices address unrelated code. See the probe call below.
    bare = problem.get("solution", "")
    entry = get_resolved_entry(problem)["entry_name"]
    # Mutants are generated from the METHOD ALONE, then seated back into its
    # class. Mutating the assembled module instead would spend the budget
    # damaging Node.__init__, the sibling methods the teacher wrote, and the
    # injected driver - none of which is the code this problem asks a student
    # for, so a suite that missed those edits would be called weak for no
    # reason. A plain function is its own module, so both are the same string.
    mutants = generate_mutants(bare)
    if emit:
        # Unseated on purpose: the playground shows the one-line edit, not the
        # whole class wrapped around it.
        emit({"type": "mutants", "total": len(mutants),
              "mutants": [{"index": i, "label": m["label"], "code": m["code"]}
                          for i, m in enumerate(mutants)]})
    solution = _runnable(problem, bare)
    mutants = [{**m, "code": _runnable(problem, m["code"])} for m in mutants]

    tests = list(oracle_tests)              # working set grows; caller's list untouched
    base = run_solution(solution, [t["input"] for t in tests], entry_name=entry)
    if not base["ok"]:
        if emit:
            emit({"type": "base_run_failed", "error": base["error"]})
        return {"kill_rate": 0.0, "kill_rate_direct": 0.0, "strong": False,
                "insufficient_mutants": len(mutants) < _MIN_MUTANTS,
                "status": "error", "total_mutants": len(mutants), "killed": 0,
                "killed_on_retry": 0, "killed_direct": 0,
                "proven_equivalent": 0, "unresolved": 0, "undetermined": 0,
                "kill_rate_lower": 0.0, "kill_rate_upper": 0.0,
                "needs_review": False,
                "new_tests": [], "mutants": [], "error": base["error"]}
    expected = base["results"]              # ground truth, parallel to `tests`

    seen = {_key(t["input"]) for t in tests}
    status = [None] * len(mutants)
    probes = [None] * len(mutants)
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
            status[i] = "undetermined"
            if em:
                em({"type": "search_error", "error": f"{type(e).__name__}: {e}"})
                em({"type": "mutant_final", "status": "undetermined"})
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
            # A1 - the search came up empty, and that is where it stops.
            #
            # Every free input and then the model failed to separate the two
            # programs. That is genuinely ambiguous: the mutant may be harmless,
            # or it may be a real bug whose one distinguishing input nothing
            # reached. The old code resolved the ambiguity by running the sweep
            # and calling full agreement a proof, which is how a real overflow
            # bug got excused. We now say what is true - we do not know - and
            # let A4's range carry it to a human.
            status[i] = "undetermined"

            # A2a - now ask WHY it survived, by watching the edited line while
            # the real oracle tests run. This changes no verdict (A1: evidence
            # never excuses); it turns a bare "undetermined" into one of three
            # findings a person can actually act on. See main/probe.py.
            #
            # `bare`, not `solution`. The index came from generate_mutants(bare)
            # and only means something in that tree: handing it the assembled
            # module made every index address unrelated code - a Compare became
            # a FunctionDef, a Constant became an Assign - so _instrument
            # refused all 35 sites and every class problem reported
            # verdict=unknown, reached=0. The instrumented method is seated by
            # `wrap` afterwards, which is a no-op for a plain function.
            pr = probe_site(bare, entry, [t["input"] for t in tests],
                            m.get("index", -1), m.get("kind", ""),
                            m.get("slot", 0),
                            wrap=lambda src: _runnable(problem, src))
            probes[i] = pr
            if em:
                em({"type": "probe_result", **pr,
                    "detail": _PROBE_DETAIL.get(pr["verdict"], "")})
                em({"type": "mutant_final", "status": "undetermined",
                    "detail": "no free input and no model-suggested input "
                              "separated the two programs. That is not proof "
                              "of harmlessness, so this one is reported as "
                              "undetermined rather than guessed either way."})

    results = [{"label": m["label"], "status": s,
                **({"probe": p} if p else {})}
               for m, s, p in zip(mutants, status, probes)]
    killed = status.count("killed") + status.count("killed_on_retry")
    killed_on_retry = status.count("killed_on_retry")
    undetermined = status.count("undetermined")
    total = len(mutants)

    # ── A4 - report what is known, as a RANGE ────────────────────────────
    #
    # An undetermined mutant is exactly that: we could not tell whether it is a
    # real bug or a harmless edit. The old code forced it into one of two
    # verdicts and stated the guess as fact - `unresolved` counted it as a
    # confirmed bug, `proven_equivalent` deleted it as confirmed harmless. Both
    # are guesses wearing a verdict's clothing.
    #
    # So bound it instead. The two ends are the two ways the ambiguity could
    # resolve, and the truth is somewhere between:
    #
    #   lower = every undetermined mutant is a REAL BUG      (worst case)
    #   upper = every undetermined mutant is HARMLESS        (best case)
    #
    # Combination-sum was reported "67%, WEAK". The truthful statement was
    # "between 67% and 100%, and we could not tell which".
    kill_rate_lower = killed_direct / total if total else 0.0
    scored = total - undetermined
    kill_rate_upper = killed_direct / scored if scored else 0.0

    # THE DECISION. If the WORST case already clears the bar, the undetermined
    # mutants cannot change the answer and no human ever needs to look. If even
    # the BEST case fails, it is weak whatever they turn out to be. Only when
    # the bar falls between the two ends does the ambiguity actually decide the
    # verdict - and that is the one case worth a person's time.
    #
    # This is what keeps the review queue small: modelled over the old cache,
    # 34 of 48 problems cleared on the lower bound alone.
    strong = kill_rate_lower >= CUTOFF_1_KILL_RATE
    hopeless = kill_rate_upper < CUTOFF_1_KILL_RATE

    # A solution too trivial to mutate cannot earn a verdict at all. Measured
    # against the mutants that actually carry the score. (A6: this used to test
    # `total`, which is a different number the moment anything leaves the
    # denominator - rotate-list passed a floor of 3 on 9 generated mutants and
    # was then scored 1/1 = 100% STRONG on the single one that remained.)
    insufficient = total < _MIN_MUTANTS or (not hopeless and scored < _MIN_MUTANTS)
    if insufficient:
        strong = False

    # A method too trivial to mutate, whose behaviour the TEACHER wrote down.
    #
    # `return self.count` has no plausible wrong single-point implementation, so
    # there is nothing for a mutation operator to generate and no kill rate that
    # could ever mean anything - insufficient_mutants here is "we cannot
    # measure", not "this is weak". Blocking on it would leave three of HW3's
    # five Stack methods permanently unservable however good their tests are.
    #
    # So a second, narrower basis for trust, and deliberately NOT folded into
    # `strong`: strong keeps meaning "cleared mutation testing". This means "a
    # person stated what this method does, in a `>>>` example, and that
    # statement is in the suite" - see context.doctest_covers. It applies ONLY
    # when mutation testing could not reach a verdict; a method with enough
    # mutants is judged on them, doctest or no doctest.
    doctest_verified = insufficient and doctest_covers(problem)

    verdict = ("doctest_verified" if doctest_verified
               else "insufficient_mutants" if insufficient
               else "strong" if strong
               else "weak" if hopeless
               else "needs_review")

    # kill_rate_direct stays the headline number - the suite exactly as handed
    # in, worst case. kill_rate is the post-repair figure, which credits tests
    # this very pass had to add and so can never earn STRONG on its own.
    kill_rate_direct = kill_rate_lower
    kill_rate = killed / total if total else 0.0

    out = {"kill_rate": kill_rate,
           "kill_rate_direct": kill_rate_direct,
           "kill_rate_lower": kill_rate_lower,
           "kill_rate_upper": kill_rate_upper,
           "strong": strong,
           "needs_review": verdict == "needs_review",
           "insufficient_mutants": insufficient,
           "status": verdict,
           "total_mutants": total, "killed": killed,
           "killed_on_retry": killed_on_retry, "killed_direct": killed_direct,
           # Kept at 0: nothing is excused after A1, and A3 stops the provable
           # ones being generated. Retained so older readers of this dict do
           # not KeyError.
           "proven_equivalent": 0,
           "undetermined": undetermined, "unresolved": undetermined,
           "new_tests": new_tests, "mutants": results, "error": None}
    if emit:
        emit({"type": "evaluation_done", **{k: out[k] for k in (
            "kill_rate", "kill_rate_direct", "strong", "insufficient_mutants",
            "status", "total_mutants", "killed", "killed_on_retry",
            "killed_direct", "undetermined", "kill_rate_lower",
            "kill_rate_upper", "needs_review")}})
    return out


# ── orchestrator ──────────────────────────────────────────────────────────

def validate_oracle(problem: dict, initial_tests: list,
                    max_rounds: int | None = None,
                    emit=None) -> dict:
    """Evaluate the oracle, and while it is still weak pull in a fresh batch of
    LLM-generated + ground-truth-verified tests and try again, up to
    max_rounds. Stops as soon as the oracle is STRONG.

    Returns the final evaluation plus `rounds` and `final_tests` (the full
    grown suite - persist this to keep the improvement)."""
    if max_rounds is None:              # late-bound: see _candidate_inputs
        max_rounds = CUTOFF_2_MAX_EXPAND_ROUNDS
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
              f"undetermined={result['undetermined']} "
              f"{result['status'].upper()}")
        if result["strong"] or rnd == max_rounds:
            break

        # B3 - stop when the SURVIVORS cannot be killed, not when the score
        # stops moving.
        #
        # The first version of this rule stopped as soon as a round failed to
        # improve the number, which is wrong: fresh test batches are generated
        # by a model and genuinely vary, so a batch that misses in round 2 can
        # hit in round 3. The repair phase has really earned kills that way.
        #
        # What DOES justify stopping is evidence about the mutants themselves.
        # If every remaining survivor was probed and the edited line never once
        # behaved differently across the whole suite, more tests of the same
        # kind are not going to separate them - combination-sum bought five
        # rounds of exactly that. A survivor whose line was never REACHED is
        # the opposite case: more tests are precisely what it needs, so those
        # keep the loop running.
        probes = [m.get("probe") for m in result["mutants"]
                  if m["status"] == "undetermined"]
        if probes and all(p and p.get("verdict") == NO_INFECTION
                          and p.get("reached", 0) >= _B3_MIN_DECISIONS
                          for p in probes):
            msg = (f"every remaining survivor was watched across "
                   f"{sum(p['reached'] for p in probes)} decisions and never "
                   f"once behaved differently - more tests cannot separate "
                   f"them, stopping")
            print(f"  [mutation] {msg}")
            if emit:
                emit({"type": "stopping_early", "reason": "no_infection",
                      "detail": msg})
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

if __name__ == "__main__":
    # ── self-check ────────────────────────────────────────────────────────
    # Pure and free: AST rewriting, the seeded sweep, and the scoring
    # arithmetic. No oracle, no subprocess, no model call.
    #     python -m main.mutation
    #
    # Every case below is a real observation, not an invented example. The
    # operator cases are lines from assignment_20.py that generated ZERO
    # mutants before; the scoring cases are verdicts read out of the live
    # cache; the divergence cases are bugs that shipped into three separate
    # prototypes of the structural equivalence check.

    def _labels(src):
        return [m["label"] for m in generate_mutants(src)]

    # ── operators the generator was blind to (A0) ─────────────────────────
    # AugAssign stores its operator in node.op rather than wrapping a BinOp,
    # so matching only BinOp missed every += -= *= //= in the codebase.
    for op, want in (("+=", "+= -> -="), ("-=", "-= -> +="),
                     ("*=", "*= -> /="), ("//=", "//= -> *=")):
        src = f"def f(n):\n    t = 0\n    t {op} n\n    return t\n"
        assert any(want in l for l in _labels(src)), (op, _labels(src))

    # Binary operators that had no entry in the flip table.
    for src, want in (("def f(a,b):\n    return a % b\n",  "% -> //"),
                      ("def f(a,b):\n    return a ** b\n", "** -> *"),
                      ("def f(a,b):\n    return a ^ b\n",  "^ -> &"),
                      ("def f(a,b):\n    return a << b\n", "<< -> >>")):
        assert any(want in l for l in _labels(src)), (want, _labels(src))

    assert any("is -> is not" in l
               for l in _labels("def f(x):\n    return x is None\n"))

    # A chained comparison is ONE Compare node carrying several operators. The
    # old guard was `len(node.ops) == 1`, which skipped `0 <= i < n` entirely -
    # and a bounds check is exactly where an off-by-one hides.
    chained = _labels("def f(i, n):\n    return 0 <= i < n\n")
    assert any("<= -> <" in l for l in chained), chained
    assert any(": < -> <=" in l for l in chained), chained

    # is_armstrong: two mutable sites on one line, both previously invisible.
    # It produced 2 mutants - under _MIN_MUTANTS - so it could never be STRONG
    # however good its tests were.
    armstrong = ("def f(n):\n    digits = str(n)\n    power = len(digits)\n"
                 "    total = 0\n    for ch in digits:\n"
                 "        total += int(ch) ** power\n    return total == n\n")
    assert len(_labels(armstrong)) >= _MIN_MUTANTS, _labels(armstrong)

    # Every mutant must be valid Python that actually differs from the source.
    for m in generate_mutants(armstrong):
        ast.parse(m["code"])
        assert m["code"] != ast.unparse(ast.parse(armstrong))

    # ── the seeded sweep is reproducible ──────────────────────────────────
    # Two runs must produce byte-identical sweeps, or a verdict stops being
    # repeatable and two identical problems can disagree.
    seed_tests = [{"input": [3, 7], "expected": 10}]
    assert _sweep_inputs(seed_tests) == _sweep_inputs(seed_tests)
    assert len(_sweep_inputs(seed_tests, n=25)) <= 25

    # ── scoring arithmetic: A4's range and A6's floor ────────────────────
    def _score(total, killed_direct, undetermined):
        """The lines evaluate_oracle uses to reach a verdict, in isolation."""
        lower = killed_direct / total if total else 0.0
        scored = total - undetermined
        upper = killed_direct / scored if scored else 0.0
        strong = lower >= CUTOFF_1_KILL_RATE
        hopeless = upper < CUTOFF_1_KILL_RATE
        insufficient = total < _MIN_MUTANTS or (not hopeless and scored < _MIN_MUTANTS)
        return ("insufficient" if insufficient else "strong" if strong
                else "weak" if hopeless else "needs_review"), lower, upper

    # No ambiguity at all: the range collapses to a point.
    assert _score(10, 10, 0)[0] == "strong"
    assert _score(10,  5, 0)[0] == "weak"

    # combination-sum: 4 killed, 2 undetermined of 6. Worst case 67% (fails),
    # best case 100% (passes) - so the undetermined ones actually decide it.
    verdict, lo, hi = _score(6, 4, 2)
    assert verdict == "needs_review" and lo == 4/6 and hi == 1.0, (verdict, lo, hi)

    # THE RULE THAT KEEPS THE QUEUE SMALL: if the WORST case already clears the
    # bar, the undetermined mutants cannot change the answer - no human needed.
    assert _score(20, 18, 2)[0] == "strong", "18/20 worst case = 90% >= 85%"

    # And if the BEST case still fails, it is weak whatever they turn out to be.
    assert _score(20, 5, 3)[0] == "weak", "5/17 best case = 29% < 85%"

    # A6: a floor on the mutants actually carrying the score. rotate-list had 9
    # generated and 8 that never resolved - one mutant cannot decide a verdict.
    assert _score(9, 1, 8)[0] == "insufficient"
    assert _score(5, 1, 4)[0] == "insufficient"
    # ...but only when the ambiguity is what shrank it. An honestly weak
    # 9-mutant suite stays weak rather than hiding behind "insufficient".
    assert _score(9, 1, 0)[0] == "weak"
    # Too trivial to mutate at all, as always.
    assert _score(2, 2, 0)[0] == "insufficient"

    # ── divergence sets: three bugs that shipped into prototypes ──────────
    # The structural check (A2b) proves equivalence by deriving WHERE two
    # predicates disagree, then showing every such value is unreachable. All
    # three failures below were false answers about that first step.
    def _divergence(orig_src, mut_src, var):
        """Search band derived from the CONSTANTS, never hardcoded."""
        consts = [n.value
                  for src in (orig_src, mut_src)
                  for n in ast.walk(ast.parse(src, mode="eval"))
                  if isinstance(n, ast.Constant) and isinstance(n.value, int)]
        lo, hi = min(consts) - 2, max(consts) + 2
        f_o = eval(f"lambda {var}: {orig_src}")
        f_m = eval(f"lambda {var}: {mut_src}")
        return [v for v in range(lo, hi + 1) if f_o(v) != f_m(v)]

    # Prototype 3 hardcoded range(-10000, 10001). Any constant outside it
    # returned [] - read as "they never differ" - read as EQUIVALENT. The last
    # case is reverse_integer's real overflow bug, which it excused.
    assert _divergence("remain < 0", "remain <= 0", "remain") == [0]
    assert _divergence("remain < 0", "remain < 1", "remain") == [0]
    assert _divergence("r < 20000", "r <= 20000", "r") == [20000]
    assert _divergence("r > 2147483647", "r > 2147483648", "r") == [2147483648]
    assert _divergence("x >= 100", "x > 100", "x") == [100]

    # The derived band must be complete, not merely lucky: a search 500,000
    # values wider finds nothing more.
    for o, m, v in (("r < 20000", "r <= 20000", "r"),
                    ("r > 2147483647", "r > 2147483648", "r")):
        f_o, f_m = eval(f"lambda {v}: {o}"), eval(f"lambda {v}: {m}")
        c = [n.value for s in (o, m) for n in ast.walk(ast.parse(s, mode="eval"))
             if isinstance(n, ast.Constant)]
        wide = [x for x in range(min(c) - 500_000, max(c) + 500_001)
                if f_o(x) != f_m(x)]
        assert wide == _divergence(o, m, v), (o, m, wide)

    # ── A5: the exact half - deriving inputs from a divergence target ────
    # The model supplies `target`; this must turn it into candidate arguments
    # without any model involvement. reverse_integer is the case it exists for:
    # result must reach 2147483648, and the input that produces it is that
    # number's digits reversed.
    derived = _candidates_from_target([123], 2147483648)
    assert [8463847412] in derived, derived
    # ...and it must survive the model being off by one, which it observably is.
    off_by_one = _candidates_from_target([123], 2147483647)
    assert [8463847412] in off_by_one, "neighbours are transformed too"
    assert [2147483648] in derived, "the boundary itself is still worth trying"
    assert [-8463847412] in derived, "sign flips are cheap and often right"

    # It must substitute into the right argument position, and leave the others.
    two = _candidates_from_target([5, 7], 21)
    assert [12, 7] in two and [5, 12] in two, two
    # A non-integer argument is never substituted into.
    mixed = _candidates_from_target([[1, 2], 9], 21)
    assert all(c[0] == [1, 2] for c in mixed), mixed
    # Repeatable, and nothing derived from a non-integer target.
    assert _candidates_from_target([1], 21) == _candidates_from_target([1], 21)
    assert _candidates_from_target([1], "x") == []
    assert _candidates_from_target([1], True) == []

    print("mutation.py self-check OK")
