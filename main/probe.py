"""
probe.py - watching a mutated line from the inside, while the real tests run.

A mutant that survives the oracle tells you almost nothing on its own. Three
completely different things produce that same silence, and they need opposite
responses:

    1. REACHABILITY   no test ever runs that line. The edit cannot matter
                      because the edited code never happens. This is a hole in
                      the TEST SUITE, not an equivalence question, and the fix
                      is concrete: write a test that gets there.

    2. INFECTION      the line runs, but the edit never changes the decision it
                      makes. `remain < 0` and `remain <= 0` differ only when
                      remain is 0; if that value never arrives, the two are
                      making the identical call every single time.

    3. PROPAGATION    the line runs AND the edit changes the decision - but the
                      difference is swallowed before it reaches the output. The
                      palindrome check runs one extra loop pass comparing a
                      character to itself, which can never be unequal.

The pipeline only ever compared final outputs, so all three collapsed into one
indistinguishable "the tests still passed". This module tells them apart by
instrumenting the mutated expression and running the ORACLE TESTS - inputs
already known to be valid - rather than guessing new ones from argument types.

Two things that matter about that choice. It observes REAL states: 1,470
genuine decisions on combination-sum, against 183 fabricated inputs most of
which the problem forbids. And it never runs hostile inputs, so the timeouts
that plagued the generated sweep simply do not arise.

HONEST SCOPE. "The line never behaved differently across this suite" is
evidence, not proof - a 48th test could still separate them. Nothing here
removes a mutant from the score; that needs main/equivalence.proves_harmless.
This exists to tell a human WHICH of the three happened.

SANDBOXED. The instrumented solution runs in a subprocess with a timeout, like
everything else that executes uploaded code. An instrumented solution is still
teacher code and can still loop forever; running it in-process would hang the
API server.
"""
import ast
import copy
import json
import os
import subprocess
import sys
import tempfile

# Wall-clock leash for one probed run of the whole suite. Generous compared to
# a single mutant run because this executes every oracle test once.
PROBE_TIMEOUT = 20.0

# Classification labels. These are the vocabulary the playground and the
# instructor-facing copy are written against.
NEVER_REACHED = "never_reached"
NO_INFECTION = "no_infection"
PROPAGATION = "propagation"
UNKNOWN = "unknown"


_HARNESS = r'''
import json, sys
payload = json.load(open(sys.argv[1]))
_S = {"reached": 0, "differed": 0}
ns = {"_S": _S, "List": list, "Dict": dict, "Set": set, "Tuple": tuple,
      "Optional": None, "Any": None}
try:
    exec(compile(payload["code"], "<probed>", "exec"), ns)
except Exception as e:
    print(json.dumps({"ok": False, "error": "exec: %s" % e})); raise SystemExit
fn = ns.get(payload["entry"])
if fn is None:
    for v in ns.values():
        if callable(v) and getattr(v, "__module__", None) is None:
            fn = v
            break
if fn is None:
    print(json.dumps({"ok": False, "error": "entry not found"})); raise SystemExit
for args in payload["inputs"]:
    try:
        fn(*args)
    except Exception:
        pass          # a raising input still exercised the line up to that point
print(json.dumps({"ok": True, **_S}))
'''


def _recorder_src(kind, orig_op, mut_op, orig_const=None, mut_const=None):
    """Source for `_p`, generated per site so both operands are bound ONCE.

    The naive instrumentation - emitting `_p(a < b, a <= b)` - evaluates the
    operands twice, which changes behaviour for anything with a side effect and
    doubles the cost of anything expensive. Passing the operands and applying
    both operators inside keeps evaluation exactly as often as the original."""
    if kind == "const":
        # The comparison keeps its operator; only the literal moved.
        return (f"def _p(a):\n"
                f"    _S['reached'] += 1\n"
                f"    o = a {orig_op} {orig_const!r}\n"
                f"    m = a {orig_op} {mut_const!r}\n"
                f"    if o != m: _S['differed'] += 1\n"
                f"    return o\n")
    return (f"def _p(a, b):\n"
            f"    _S['reached'] += 1\n"
            f"    o = a {orig_op} b\n"
            f"    m = a {mut_op} b\n"
            f"    if o != m: _S['differed'] += 1\n"
            f"    return o\n")


def _instrument(tree, index, kind, slot):
    """Replace the site at `index` with a call to `_p`, and return the
    recorder source to prepend. Returns (new_tree, recorder_src) or None when
    the shape is not one we can instrument safely."""
    # Deferred: mutation.py imports THIS module for the probe, so importing it
    # at module scope would close a cycle. The tables are only needed here.
    from .mutation import _BIN_FLIP, _CMP_FLIP, _OP_SYMBOL

    work = copy.deepcopy(tree)
    walked = list(ast.walk(work))
    if index >= len(walked):
        return None
    node = walked[index]
    call = lambda *args: ast.Call(func=ast.Name(id="_p", ctx=ast.Load()),
                                  args=list(args), keywords=[])
    parent = {id(c): p for p in walked for c in ast.iter_child_nodes(p)}

    if kind == "cmp" and isinstance(node, ast.Compare) and len(node.ops) == 1:
        o = type(node.ops[0])
        if o not in _CMP_FLIP:
            return None
        rec = _recorder_src("cmp", _OP_SYMBOL[o], _OP_SYMBOL[_CMP_FLIP[o]])
        new = call(node.left, node.comparators[0])
    elif kind == "bin" and isinstance(node, ast.BinOp):
        o = type(node.op)
        if o not in _BIN_FLIP:
            return None
        rec = _recorder_src("bin", _OP_SYMBOL[o], _OP_SYMBOL[_BIN_FLIP[o]])
        new = call(node.left, node.right)
    elif kind == "bin" and isinstance(node, ast.AugAssign):
        # `x += y` needs a STATEMENT rewrite, not an expression swap: it holds
        # its operator directly and its target is a store. Rewritten as
        # `x = _p(x, y)`, which reads the target once and writes it once, same
        # as the original. A0 made augmented assignment a large share of all
        # mutants, so skipping this shape would blind the probe to most of them.
        o = type(node.op)
        if o not in _BIN_FLIP or not isinstance(node.target, ast.Name):
            return None
        rec = _recorder_src("bin", _OP_SYMBOL[o], _OP_SYMBOL[_BIN_FLIP[o]])
        load_target = ast.Name(id=node.target.id, ctx=ast.Load())
        replacement = ast.Assign(
            targets=[ast.Name(id=node.target.id, ctx=ast.Store())],
            value=call(load_target, node.value))
        holder = parent.get(id(node))
        if holder is None:
            return None
        for field, value in ast.iter_fields(holder):
            if isinstance(value, list):
                for i, item in enumerate(value):
                    if item is node:
                        value[i] = replacement
                        ast.fix_missing_locations(work)
                        return work, rec
        return None
    elif kind == "const" and isinstance(node, ast.Constant):
        p = parent.get(id(node))
        if not (isinstance(p, ast.Compare) and len(p.ops) == 1
                and p.comparators[0] is node):
            # A bare constant elsewhere - a loop bound, an initialiser. Its
            # value differs at the site by construction, so there is no
            # infection question to answer; only reachability, which the
            # caller can still learn from `reached`.
            return None
        o = type(p.ops[0])
        rec = _recorder_src("const", _OP_SYMBOL[o], None,
                            node.value, node.value + 1)
        new = call(p.left)
        node = p                                  # replace the whole Compare
    else:
        return None

    holder = parent.get(id(node))
    if holder is None:
        return None
    for field, value in ast.iter_fields(holder):
        if value is node:
            setattr(holder, field, new)
            ast.fix_missing_locations(work)
            return work, rec
        if isinstance(value, list):
            for i, item in enumerate(value):
                if item is node:
                    value[i] = new
                    ast.fix_missing_locations(work)
                    return work, rec
    return None


def probe_site(solution_src: str, entry: str | None, inputs: list,
               index: int, kind: str, slot: int = 0,
               timeout: float = PROBE_TIMEOUT) -> dict:
    """Run the oracle tests with the mutated line instrumented.

    Returns {"ok", "reached", "differed", "verdict", "error"}. `verdict` is one
    of NEVER_REACHED / NO_INFECTION / PROPAGATION / UNKNOWN - and UNKNOWN
    whenever anything at all went wrong, because a probe that could not run
    must never be mistaken for a probe that saw no difference."""
    fail = {"ok": False, "reached": 0, "differed": 0, "verdict": UNKNOWN}
    try:
        tree = ast.parse(solution_src)
    except SyntaxError as e:
        return {**fail, "error": f"parse: {e}"}
    built = _instrument(tree, index, kind, slot)
    if built is None:
        return {**fail, "error": "shape not instrumentable"}
    work, recorder = built
    try:
        code = recorder + "\n" + ast.unparse(work)
    except Exception as e:
        return {**fail, "error": f"unparse: {e}"}

    payload = {"code": code, "entry": entry, "inputs": inputs}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as pf:
        json.dump(payload, pf, default=str)
        path = pf.name
    try:
        proc = subprocess.run([sys.executable, "-c", _HARNESS, path],
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {**fail, "error": f"timeout after {timeout}s"}
    finally:
        os.unlink(path)
    if proc.returncode != 0:
        return {**fail, "error": (proc.stderr or "nonzero exit").strip()[:200]}
    try:
        got = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return {**fail, "error": "unparseable probe output"}
    if not got.get("ok"):
        return {**fail, "error": got.get("error", "probe failed")}

    reached, differed = got["reached"], got["differed"]
    verdict = (NEVER_REACHED if reached == 0
               else NO_INFECTION if differed == 0
               else PROPAGATION)
    return {"ok": True, "reached": reached, "differed": differed,
            "verdict": verdict, "error": None}


if __name__ == "__main__":
    # ── self-check ────────────────────────────────────────────────────────
    #     python -m main.probe
    # Runs real subprocesses, but no model and no oracle generation.
    from .mutation import _sites

    def _site_for(src, needle):
        """The site whose label contains `needle`."""
        tree = ast.parse(src)
        for index, kind, label, slot in _sites(tree):
            if needle in label:
                return index, kind, slot
        raise AssertionError(f"no site matching {needle!r} in\n{src}")

    # ── condition 2: reached, but the edit never changes the decision ─────
    # combination-sum's real shape. `remain < 0` is reached constantly and
    # never sees remain == 0, the only value where `<` and `<=` disagree.
    COMBO = ("def combinationSum(candidates, target):\n"
             "    result = []\n"
             "    def backtrack(remain, comb, start):\n"
             "        if remain == 0:\n"
             "            result.append(list(comb))\n"
             "            return\n"
             "        elif remain < 0:\n"
             "            return\n"
             "        for i in range(start, len(candidates)):\n"
             "            comb.append(candidates[i])\n"
             "            backtrack(remain - candidates[i], comb, i)\n"
             "            comb.pop()\n"
             "    backtrack(target, [], 0)\n"
             "    return result\n")
    i, k, s = _site_for(COMBO, "< -> <=")
    r = probe_site(COMBO, "combinationSum",
                   [[[1, 2, 3], 4], [[2, 3, 5], 8], [[1, 3, 4], 7]], i, k, s)
    assert r["ok"], r
    assert r["reached"] > 50, r          # the line runs constantly
    assert r["differed"] == 0, r         # and the edit never changes a decision
    assert r["verdict"] == NO_INFECTION, r

    # ── condition 3: reached, DID differ, output unchanged ────────────────
    # is_palindrome_string. The extra loop pass compares a character to itself.
    PAL = ("def is_palindrome_string(text):\n"
           "    cleaned = ''\n"
           "    for ch in text.lower():\n"
           "        if ch.isalnum():\n"
           "            cleaned += ch\n"
           "    left = 0\n"
           "    right = len(cleaned) - 1\n"
           "    while left < right:\n"
           "        if cleaned[left] != cleaned[right]:\n"
           "            return False\n"
           "        left += 1\n"
           "        right -= 1\n"
           "    return True\n")
    i, k, s = _site_for(PAL, "< -> <=")
    r = probe_site(PAL, "is_palindrome_string",
                   [["racecar"], ["abba"], ["hello"], ["A man, a plan, a canal: Panama"]],
                   i, k, s)
    assert r["ok"], r
    assert r["reached"] > 0 and r["differed"] > 0, r
    assert r["verdict"] == PROPAGATION, r

    # ── condition 1: the line never runs ──────────────────────────────────
    GUARD = ("def average(items):\n"
             "    if len(items) < 1:\n"
             "        return 0\n"
             "    return sum(items) / len(items)\n")
    i, k, s = _site_for(GUARD, "< -> <=")
    r = probe_site(GUARD, "average", [[[1, 2, 3]], [[4, 5]]], i, k, s)
    assert r["ok"] and r["reached"] > 0, r   # the guard itself IS evaluated
    # ...but with no empty list among the inputs it never takes the branch, so
    # the comparison never separates. That is no_infection, not never_reached -
    # never_reached needs a line no test executes at all.
    assert r["verdict"] == NO_INFECTION, r

    # ── augmented assignment, the shape that needs a statement rewrite ────
    ACC = ("def digit_sum(n):\n"
           "    n = abs(n)\n"
           "    total = 0\n"
           "    while n > 0:\n"
           "        total += n % 10\n"
           "        n //= 10\n"
           "    return total\n")
    i, k, s = _site_for(ACC, "+= -> -=")
    r = probe_site(ACC, "digit_sum", [[1234], [905], [7]], i, k, s)
    assert r["ok"], r
    assert r["reached"] > 0, r
    assert r["differed"] > 0, r          # + and - genuinely differ here
    assert r["verdict"] == PROPAGATION, r

    # ── a probe that cannot run must never look like agreement ────────────
    HANG = ("def f(n):\n"
            "    while n > 0:\n"
            "        pass\n"
            "    return n\n")
    i, k, s = _site_for(HANG, "> -> >=")
    r = probe_site(HANG, "f", [[5]], i, k, s, timeout=3.0)
    assert not r["ok"] and r["verdict"] == UNKNOWN, r
    assert r["differed"] == 0 and r["reached"] == 0, \
        "a failed probe must report nothing, not zero-differences"

    # A shape we decline to instrument reports UNKNOWN, never a verdict.
    BOOLOP = "def f(a, b):\n    return a and b\n"
    tree = ast.parse(BOOLOP)
    idx = next(i for i, n in enumerate(ast.walk(tree)) if isinstance(n, ast.BoolOp))
    r = probe_site(BOOLOP, "f", [[1, 2]], idx, "bool", 0)
    assert not r["ok"] and r["verdict"] == UNKNOWN, r

    print("probe.py self-check OK")
