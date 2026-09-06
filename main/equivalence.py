"""
equivalence.py - proving a mutant harmless by reading the code, not running it.

Mutation testing's oldest problem: some mutants are behaviourally identical to
the original, so no test can ever kill them. Deciding that in general is
undecidable (it reduces to program equivalence), and no amount of sampling
inputs settles it - "we tried 183 inputs and saw no difference" is not the same
claim as "no difference exists". Believing otherwise is what let a real overflow
bug in reverse_integer sit excused in the oracle cache while the problem was
reported STRONG.

So this module does something narrower and sound. For one specific, extremely
common shape it produces an actual PROOF:

    if remain == 0:        <- a guard that always exits
        ...
        return
    elif remain < 0:       <- so this line never sees remain == 0
        ...

A mutation of that `elif` test which differs from the original ONLY at
`remain == 0` cannot change behaviour, because that value cannot arrive there.
Two facts are derived independently and then compared:

    where do the two tests disagree?     -> divergence_set()
    what can the variable NOT be here?   -> excluded_at()

If every disagreement is excluded, the mutation is proved harmless.

WHY THIS IS A PROOF AND NOT A SAMPLE. Both tests are threshold comparisons on
one integer. Each flips exactly once, at its own constant, and never changes
again - so two of them can only disagree in the band BETWEEN their constants.
Enumerating a band that spans both constants is therefore exhaustive, not a
sample. The band is derived from the constants themselves; hardcoding a range
silently returns "no disagreement" for any constant outside it, which reads as
"equivalent" and excuses a real bug. That mistake shipped into a prototype and
is pinned by the self-check below.

USED FROM TWO PLACES, DELIBERATELY THE SAME MATHS:
  * generation time - mutation.generate_mutants skips a mutant this proves
    harmless, so it is never created and never has to be judged.
  * survivor time   - a safety net for anything that reached the checker
    another way.

SAFETY. This is the only check whose answer REMOVES a mutant from the score, so
a false positive hides a real bug and inflates a reported number. Every
uncertain case must return False. Three prototypes of this function were wrong
in exactly that direction; all three failures are self-check cases below.
"""
import ast

# Statements after which control cannot fall through to the elif below.
_EXITS = (ast.Return, ast.Raise, ast.Break, ast.Continue)

# Margin around the constants when enumerating. Two threshold predicates can
# only disagree between their flip points, which sit at the constants, so any
# positive margin suffices; 2 is cheap insurance against off-by-one.
_BAND_MARGIN = 2

# A band wider than this means the two constants are far apart, which for a
# single-point mutation should not happen (an operator flip keeps the constant,
# a constant bump moves it by one). Refuse rather than enumerate millions.
_MAX_BAND = 10_000


def _int_const(node) -> int | None:
    """The integer value of a constant node, or None.

    Handles the unary-minus wrapper: `-5000` parses as UnaryOp(USub, 5000),
    NOT as Constant(-5000), so matching only Constant silently refuses every
    negative threshold - and negative bounds are exactly where off-by-ones
    live. Rejects bools, which are ints in Python and would make `True` look
    like the number 1."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _int_const(node.operand)
        return None if inner is None else -inner
    if isinstance(node, ast.Constant) and isinstance(node.value, int) \
            and not isinstance(node.value, bool):
        return node.value
    return None


def _simple_threshold(test) -> tuple[str, int] | None:
    """Is this `NAME <op> INTCONST` with a single comparison operator?

    Returns (variable name, constant) or None. Anything else - two variables,
    a chained comparison, a float, a call - is not something this module can
    reason about, and must fall through to the caller's other checks."""
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1
            and isinstance(test.left, ast.Name)):
        return None
    const = _int_const(test.comparators[0])
    if const is None:
        return None
    return test.left.id, const


def divergence_set(orig_test, mut_test) -> set[int] | None:
    """Every value of the variable where the two tests disagree.

    Returns None when the pair is not two simple integer thresholds on the SAME
    variable, or when the band would be unreasonably large - both meaning "I
    cannot answer", never "they agree"."""
    a, b = _simple_threshold(orig_test), _simple_threshold(mut_test)
    if a is None or b is None or a[0] != b[0]:
        return None
    var, (c1, c2) = a[0], (a[1], b[1])
    lo, hi = min(c1, c2) - _BAND_MARGIN, max(c1, c2) + _BAND_MARGIN
    if hi - lo > _MAX_BAND:
        return None
    # Compile once, then evaluate against a namespace holding just the variable.
    # fix_missing_locations because these nodes are being re-parented into a
    # fresh Expression, which compile() requires to carry positions.
    try:
        code_o = compile(ast.fix_missing_locations(ast.Expression(orig_test)),
                         "<orig>", "eval")
        code_m = compile(ast.fix_missing_locations(ast.Expression(mut_test)),
                         "<mut>", "eval")
    except Exception:
        return None
    out = set()
    for v in range(lo, hi + 1):
        try:
            if eval(code_o, {}, {var: v}) != eval(code_m, {}, {var: v}):
                out.add(v)
        except Exception:
            return None                 # cannot evaluate ⇒ cannot conclude
    return out


def excluded_at(tree, test_node) -> set[int]:
    """Values the tested variable provably cannot hold when `test_node` runs.

    Looks for `if VAR == CONST:` whose body always exits, with `test_node` as
    its `elif` test. `test_node` MUST be a node from `tree` - identity is what
    ties the elif to its guard, and a node from a second parse of the same
    source will silently match nothing. That bug shipped once.
    """
    shape = _simple_threshold(test_node)
    if shape is None:
        return set()
    var = shape[0]
    excluded = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not node.body:
            continue
        # test_node must be THIS if's elif test
        if not (len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If)
                and node.orelse[0].test is test_node):
            continue
        if not isinstance(node.body[-1], _EXITS):
            continue                    # falls through ⇒ elif still reachable
        guard = node.test
        if not (isinstance(guard, ast.Compare) and len(guard.ops) == 1
                and isinstance(guard.ops[0], ast.Eq)
                and isinstance(guard.left, ast.Name) and guard.left.id == var):
            continue
        const = _int_const(guard.comparators[0])
        if const is not None:
            excluded.add(const)
    return excluded


def proves_harmless(tree, orig_test, mut_test) -> bool:
    """Is replacing `orig_test` with `mut_test` provably behaviour-preserving?

    `orig_test` must be a node inside `tree` (see excluded_at). Returns True
    only on a proof; every uncertainty returns False."""
    diverge = divergence_set(orig_test, mut_test)
    if diverge is None:
        return False                    # could not reason about the pair
    if not diverge:
        return False                    # identical behaviour ⇒ not a mutant at
                                        # all; generate_mutants drops those, and
                                        # calling it "harmless" would hide a
                                        # no-op edit rather than report it
    return diverge <= excluded_at(tree, orig_test)


if __name__ == "__main__":
    # ── self-check ────────────────────────────────────────────────────────
    #     python -m main.equivalence
    # Pure: no oracle, no subprocess, no model. Every case below is either a
    # real shape from assignment_20.py or a bug that shipped into a prototype.

    def _tests(src, mut_src):
        """Parse a guarded elif and its mutant, returning (tree, orig, mut)."""
        tree = ast.parse(src)
        mut = ast.parse(mut_src)
        o = tree.body[0].orelse[0].test
        m = mut.body[0].orelse[0].test
        return tree, o, m

    GUARD = ("if remain == {c}:\n    x = 1\n    return\n"
             "elif remain {op}:\n    return\n")

    # ── divergence: derived from the constants, never a hardcoded range ────
    # A prototype hardcoded range(-10000, 10001). Any constant outside it came
    # back empty, which reads as "never differ", which reads as EQUIVALENT.
    # The last two cases are reverse_integer's real overflow bug.
    for op_o, op_m, want in (("< 0", "<= 0", {0}),
                             ("< 0", "< 1", {0}),
                             ("< 20000", "<= 20000", {20000}),
                             ("> 2147483647", "> 2147483648", {2147483648}),
                             (">= 100", "> 100", {100}),
                             ("< -5000", "<= -5000", {-5000})):
        _, o, m = _tests(GUARD.format(c=0, op=op_o), GUARD.format(c=0, op=op_m))
        got = divergence_set(o, m)
        assert got == want, (op_o, op_m, got, want)

    # Completeness: a search 500,000 values wider finds nothing more.
    for op_o, op_m in (("< 20000", "<= 20000"), ("> 2147483647", "> 2147483648")):
        _, o, m = _tests(GUARD.format(c=0, op=op_o), GUARD.format(c=0, op=op_m))
        band = divergence_set(o, m)
        co = compile(ast.Expression(o), "<o>", "eval")
        cm = compile(ast.Expression(m), "<m>", "eval")
        consts = [c for n in (o, m) for c in [_int_const(n.comparators[0])]]
        wide = {v for v in range(min(consts) - 500_000, max(consts) + 500_001)
                if eval(co, {}, {"remain": v}) != eval(cm, {}, {"remain": v})}
        assert wide == band, (op_o, op_m, wide, band)

    # Shapes it must refuse rather than guess at.
    tree = ast.parse("if a == 0:\n    return\nelif a < b:\n    return\n")
    mut = ast.parse("if a == 0:\n    return\nelif a <= b:\n    return\n")
    assert divergence_set(tree.body[0].orelse[0].test,
                          mut.body[0].orelse[0].test) is None      # two variables
    tree = ast.parse("if a == 0:\n    return\nelif a < 0.5:\n    return\n")
    mut = ast.parse("if a == 0:\n    return\nelif a <= 0.5:\n    return\n")
    assert divergence_set(tree.body[0].orelse[0].test,
                          mut.body[0].orelse[0].test) is None      # floats
    tree = ast.parse("if a == 0:\n    return\nelif 0 <= a < 9:\n    return\n")
    mut = ast.parse("if a == 0:\n    return\nelif 0 <= a <= 9:\n    return\n")
    assert divergence_set(tree.body[0].orelse[0].test,
                          mut.body[0].orelse[0].test) is None      # chained

    # ── excluded_at: the guard must actually guard ────────────────────────
    t, o, _ = _tests(GUARD.format(c=0, op="< 0"), GUARD.format(c=0, op="<= 0"))
    assert excluded_at(t, o) == {0}
    t, o, _ = _tests(GUARD.format(c=7, op="< 0"), GUARD.format(c=7, op="<= 0"))
    assert excluded_at(t, o) == {7}, "the guard's constant, not a hardcoded 0"

    # A guard that FALLS THROUGH excludes nothing - the elif still sees 0.
    fall = ("if remain == 0:\n    x = 1\nelif remain < 0:\n    return\n")
    tf = ast.parse(fall)
    assert excluded_at(tf, tf.body[0].orelse[0].test) == set()

    # A guard on a DIFFERENT variable excludes nothing.
    other = ("if other == 0:\n    return\nelif remain < 0:\n    return\n")
    to = ast.parse(other)
    assert excluded_at(to, to.body[0].orelse[0].test) == set()

    # Identity matters: a node from a SECOND parse matches no guard. This is
    # the bug that made a prototype prove nothing at all, silently.
    t, o, _ = _tests(GUARD.format(c=0, op="< 0"), GUARD.format(c=0, op="<= 0"))
    stranger = ast.parse(GUARD.format(c=0, op="< 0")).body[0].orelse[0].test
    assert excluded_at(t, stranger) == set(), \
        "a node from another parse must not match - callers must pass their own"

    # ── proves_harmless: the two facts combined ───────────────────────────
    # combination-sum's two real survivors.
    for op_m in ("<= 0", "< 1"):
        t, o, m = _tests(GUARD.format(c=0, op="< 0"), GUARD.format(c=0, op=op_m))
        assert proves_harmless(t, o, m) is True, op_m

    # Same edit, but the guard falls through -> NOT harmless.
    tf = ast.parse("if remain == 0:\n    x = 1\nelif remain < 0:\n    return\n")
    mf = ast.parse("if remain == 0:\n    x = 1\nelif remain <= 0:\n    return\n")
    assert proves_harmless(tf, tf.body[0].orelse[0].test,
                           mf.body[0].orelse[0].test) is False

    # Divergence OUTSIDE the excluded set -> NOT harmless. The guard excludes 0
    # but these disagree at 5, which can happen.
    t5 = ast.parse("if r == 0:\n    return\nelif r < 5:\n    return\n")
    m5 = ast.parse("if r == 0:\n    return\nelif r < 6:\n    return\n")
    assert proves_harmless(t5, t5.body[0].orelse[0].test,
                           m5.body[0].orelse[0].test) is False

    # A guard excluding the WRONG value -> NOT harmless.
    t7 = ast.parse("if r == 7:\n    return\nelif r < 0:\n    return\n")
    m7 = ast.parse("if r == 7:\n    return\nelif r <= 0:\n    return\n")
    assert proves_harmless(t7, t7.body[0].orelse[0].test,
                           m7.body[0].orelse[0].test) is False

    # An edit that changes nothing at all is not "harmless", it is a non-mutant.
    t, o, _ = _tests(GUARD.format(c=0, op="< 0"), GUARD.format(c=0, op="< 0"))
    same = ast.parse(GUARD.format(c=0, op="< 0")).body[0].orelse[0].test
    assert proves_harmless(t, o, same) is False

    print("equivalence.py self-check OK")
