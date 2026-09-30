"""
grading.py - the answer-checking state machine.

One orchestration function, grade_submission(), owns every verdict. Routes must
not re-implement any part of it.

Three rules shape the whole design:

  * Execution decides, opinion is last. A verdict is deterministic only when a
    real run attributed the fault to the student.
  * Our failure is never evidence about the student. Infrastructure trouble
    returns `indeterminate` and costs no attempt - it never defaults to wrong.
  * CORRECT CODE MUST NEVER BE MARKED WRONG. Differently named, differently
    structured, redundant, slow, or written ahead of the step it was asked for -
    all correct. A false `incorrect` costs a student an attempt and is the one
    unacceptable outcome; a false "we could not tell" costs them nothing.

WHAT MAY CONVICT, AND WHAT MAY ONLY ACQUIT. The third rule is enforced by
splitting the tiers on exactly that line, because "be careful" is not a
mechanism. `incorrect` may only come from evidence about the student's own code
that no later chunk can repair, and that never involves comparing their
intermediate state to the teacher's:

    blank answer, syntax, indentation, policy violation, an undefined name,
    and - on the LAST chunk only - the real oracle.

Everything below is an ACQUITTAL PATH. Each may return `correct`, and none may
return `incorrect`:

    execution-reference  the teacher's tail runs as-is against their work
    execution-bridged    same tail, byte for byte, with their names matched to
                         it BY VALUE (main/bridge.py). Deterministic, no model.
    execution-adapted    a model rewrites the tail in their names; calibration,
                         the full oracle and a knockout must all still pass.
                         When it runs cleanly and comes out WRONG, the cases
                         are shown - evidence, never a verdict.
    execution-completed  a model writes its own finish on top of their step;
                         the full oracle and a knockout must still pass

(llm-judge - two judges reading code without running it - is retired: it
acquitted 37 real steps that went on to trap their students.)

A tier that cannot acquit falls through, and running out of tiers is
`indeterminate` with no attempt spent. What that costs is the ability to tell a
student their non-final step is wrong on anything other than their own code -
which was never something we could do soundly. Identical submissions came back
correct at 1am and incorrect at 4:58am because a model was deciding it.
"""
import ast
import difflib
import json
import re
import textwrap

from . import bridge
from .execution import classify_run
from .identity import get_resolved_entry
from .indent import align_to, align_to_chunk, base_indent, unpad_first_line
from .ollama_client import GRADING_MODEL, chat
from . import trace
from .schemas import GradeResult
from .context import build_program
from .sessions import accepted_prefix, problem_of

# Verdict memo, so identical code gets an identical verdict. Tiers 3 and 4 are
# model calls, and a model that wavers turns one student's answer into
# `correct` and an identical answer into `cannot verify` - a milder version of
# the 1am/4:58am flip, but the same complaint. Keyed by what was actually
# graded, never by submission id, and bounded so a long-lived process cannot
# grow without limit. In-process is sufficient: start.sh pins uvicorn to one
# worker (three other stores already depend on that).
_VERDICT_MEMO: dict = {}
_MEMO_LIMIT = 512


def _memo_key(problem, idx, upto, student_code):
    import hashlib
    raw = f"{problem.get('slug','')}\x00{idx}\x00{upto}\x00{student_code}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _trace(fn, *a, **k):
    """Call a tracing hook defensively. Telemetry must never be able to change
    a verdict, so the failure is swallowed AT THE SEAM as well as inside the
    sink - a caller that patches or wraps a hook cannot break grading."""
    try:
        fn(*a, **k)
    except Exception:
        pass


def _indent(code: str) -> str:
    return "\n".join("    " + ln if ln.strip() else ln for ln in code.splitlines())


def _assemble(problem: dict, header: str, *bodies: str) -> str:
    """The runnable program for this problem with `bodies` as its implementation.

    Delegates to main/context.py rather than concatenating here, because a
    METHOD is not `header + indented body` at all - it is the whole module it
    was carved out of, with the body dropped in at the class's own depth. That
    module has to be assembled the SAME way here as it was when the oracle was
    generated, or a correct submission fails against tests it never had a chance
    against. One assembler, one shape."""
    body = "\n".join(b.rstrip() for b in bodies if b and b.strip())
    return build_program(problem, body, header)


def _parse_body(code: str):
    """Parse BODY code (which may contain `return`) by wrapping it in a function.

    Parsing it standalone raises SyntaxError on any `return`, which silently
    made _tail_is_sane reject every valid adapter and made _names return an
    empty set - quietly disabling the clobber and anti-bypass checks."""
    return ast.parse("def _w():\n" + _indent(code))


def _names(code: str, ctx) -> set:
    try:
        tree = _parse_body(code)
    except SyntaxError:
        return set()
    return {n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ctx)}


# Builtins a coding exercise may reference without defining them first. Kept
# deliberately small: a bare name that is not here and not in scope is far more
# often a typo or the wrong parameter name than a builtin the student meant.
_SAFE_BUILTINS = frozenset({
    "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes",
    "callable", "chr", "complex", "dict", "divmod", "enumerate", "filter",
    "float", "format", "frozenset", "hash", "hex", "int", "isinstance",
    "issubclass", "iter", "len", "list", "map", "max", "min", "next", "object",
    "oct", "ord", "pow", "print", "range", "repr", "reversed", "round", "set",
    "slice", "sorted", "str", "sum", "tuple", "zip",
    # Missing, they convicted valid code as "`type` isn't defined" - a real
    # student lost attempts to hasattr. getattr/setattr/delattr are NOT here on
    # purpose: they reach any attribute by a string built at run time, which
    # walks straight past the policy's ban on `.__class__` and friends, so
    # execution._BANNED_NAMES refuses them with an honest "isn't allowed here".
    "type", "id", "hasattr", "super",
    "True", "False", "None", "NotImplemented", "Ellipsis", "__name__",
    # typing aliases the execution harness injects into the run namespace
    "List", "Dict", "Optional", "Tuple", "Set", "Any", "Union", "Callable",
    "Iterable", "Iterator",
    # EXCEPTIONS. Their absence failed the most ordinary thing a student can
    # write: `except ValueError:` was read as a reference to an undefined name,
    # so Calculator._isNumber - whose whole job is try/float/except - was
    # rejected on its first chunk. A bare exception name is never the "typed the
    # type name instead of the variable" mistake this gate is looking for.
    "ArithmeticError", "AssertionError", "AttributeError", "EOFError",
    "Exception", "IndentationError", "IndexError", "KeyError", "LookupError",
    "MemoryError", "NameError", "NotImplementedError", "OverflowError",
    "RecursionError", "RuntimeError", "StopIteration", "SyntaxError",
    "TypeError", "UnboundLocalError", "ValueError", "ZeroDivisionError",
})

# These ARE in _SAFE_BUILTINS so `list(map(...))` etc. are fine, but using one
# as a bare value (`max(list)`, `list.pop(...)`) is almost always a student
# reaching for the input list and typing the type name instead.
_BUILTIN_TYPE_NAMES = frozenset({"list", "dict", "set", "tuple", "frozenset"})


def _bound_names(tree) -> set:
    """Every name the parsed body BINDS - assignment targets, loop and
    comprehension targets, `with ... as`, walrus, function/class defs and their
    parameters, `except ... as`, `global`/`nonlocal`, and import aliases."""
    bound = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            bound.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(n.name)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            bound.update(n.names)
        elif isinstance(n, ast.Import):
            for a in n.names:
                bound.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                bound.add(a.asname or a.name)
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            a = n.args
            for grp in (a.posonlyargs, a.args, a.kwonlyargs):
                bound.update(x.arg for x in grp)
            if a.vararg:
                bound.add(a.vararg.arg)
            if a.kwarg:
                bound.add(a.kwarg.arg)
    return bound


def _bare_builtin_types(tree) -> set:
    """Builtin type names read as a plain value rather than called: `list` in
    `max(list)` or `list.pop(...)`, but NOT `list` in `list(map(...))`."""
    parent = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
    bad = set()
    for n in ast.walk(tree):
        if (isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                and n.id in _BUILTIN_TYPE_NAMES):
            par = parent.get(n)
            if not (isinstance(par, ast.Call) and par.func is n):
                bad.add(n.id)
    return bad


def _render_case(problem: dict, test: dict, failure: dict) -> str:
    """One failing case, written as the program that produced it.

    "Your solution runs but gives the wrong answer on at least one case" is a
    shrug: it tells a student they are wrong and nothing about where to look.
    This renders the case as runnable lines they can trace by hand."""
    from .context import is_method

    inp = (test or {}).get("input") or []
    lines, labels = [], []      # labels[i]: the call that produced output i
    if is_method(problem) and inp and isinstance(inp[0], str):
        lines.append(inp[0].rstrip())          # a block test IS a program
    elif is_method(problem) and inp and isinstance(inp[0], list):
        cls = problem.get("group_title") or "Solution"
        lines.append(f"x = {cls}()")
        for call in inp[0]:
            if not (isinstance(call, list) and call):
                labels.append(None)
                continue
            name, args = str(call[0]), call[1:]
            rendered = ", ".join(repr(a) for a in args)
            if name == "new":
                lines[0] = f"x = {cls}({rendered})"
                labels.append(lines[0])
            elif name in ("len", "str", "bool"):
                lines.append(f"{name}(x)")
                labels.append(lines[-1])
            else:
                lines.append(f"x.{name}({rendered})")
                labels.append(lines[-1])
    else:
        name = get_resolved_entry(problem)["entry_name"] or "solution"
        lines.append(f"{name}({', '.join(repr(a) for a in inp)})")

    out = ["\n".join(lines)]
    if failure.get("error"):
        out.append(f"\nyour code crashed: {failure['error']}")
    else:
        # Expected first: where the teacher's method raises on purpose, the
        # same exception from theirs is the behaviour asked for, not a crash.
        # Listed as one, it sent a student (29 Sep, _isNumber) after an
        # `x._getPostfix()` error their `return True` could not have caused.
        expected, raises = _crashes_in(failure.get("expected"), expected=True)
        got, crashes = _crashes_in(failure.get("got"), same=set(raises))
        out.append(f"\nexpected: {expected}")
        out.append(f"you gave: {got}")
        # WHICH CALL CRASHED, AND WHY, in words. The recorded value is only the
        # exception's name; a student shown `'!AttributeError'` spent nine
        # attempts on a one-character typo the message would have named.
        # Outputs line up one-to-one with the calls rendered above (the
        # construction line records nothing), so each crash is put beside its
        # own call when that holds, and listed plainly when it does not.
        for pos, msg in crashes:
            where = (labels[pos] if pos is not None and 0 <= pos < len(labels)
                     and labels[pos] else "your code")
            out.append(f"  {where} crashed: {msg}")
    return "\n".join(out)


_CRASH = re.compile(r"""(['"])!([A-Za-z_]\w*)(?:\\x00((?:(?!\1)[^\\]|\\.)*))?\1""")


# ── A CRASH IN THE STUDENT'S OWN LINES ───────────────────────────────────
#
# At a middle step a wrong-LOOKING result proves nothing: a correct answer that
# takes a different route holds different values too, which is why every other
# non-final failure here is "could not confirm". A CRASH is different. Once a
# line of theirs raises, no later step can undo it - so every finished program
# built on this step crashes there as well, and that is a verdict about their
# code alone. A real student lost nine attempts to `self._expr` for
# `self.__expr` after the judges, who read code and never run it, let it
# through.
#
# Read off the run grading ALREADY makes of their code with nothing of the
# teacher's after it (the "already finished?" check), so no run and no model is
# added. Only counted when it would crash in ANY completion of their step:
#   * their step sits at the top level of the function - not inside a loop,
#     branch, try or with, where code after it could run first or catch it;
#   * the crash is the FIRST time their lines ever run in that test - the
#     function is not recursive, is called directly, and was not called
#     earlier in the sequence - so nothing they have not written yet could
#     have set things up differently;
#   * the teacher's version does not raise there too (pop() on an empty stack
#     may be meant to);
#   * it is an ordinary exception, not recursion depth or memory.
# Anything that fails a condition is simply not counted, and grading carries on
# exactly as before.

# Accepted WITHOUT execution confirming the student's step on its own terms:
# by the retired judges, the retired adapter, or a model-written completion
# (which can still paper over a subtle bug). A later crash or failure inside
# one of these is pointed back at it.
_UNCONFIRMED_ACCEPTS = {"llm-judge", "execution-adapted", "execution-completed"}

_NOT_THEIR_FAULT = {"RecursionError", "MemoryError", "TimeoutError",
                    "KeyboardInterrupt", "SystemExit"}


def _changed_lines(before: str, after: str):
    """1-based (first, last) lines of `after` that are not in `before`, found
    by trimming the shared head and tail. Any coincidental match only SHRINKS
    the range, which can hide a crash but never blame a line that is not theirs."""
    a, b = before.splitlines(), after.splitlines()
    h = 0
    while h < min(len(a), len(b)) and a[h] == b[h]:
        h += 1
    t = 0
    while t < min(len(a), len(b)) - h and a[-1 - t] == b[-1 - t]:
        t += 1
    return (h + 1, len(b) - t) if len(b) - t >= h + 1 else None


def _step_ranges(problem, header, codes):
    """Line range of each accepted step (and the one being answered, last) in
    the assembled program."""
    out = []
    for k in range(len(codes)):
        out.append(_changed_lines(_assemble(problem, header, *codes[:k]),
                                  _assemble(problem, header, *codes[:k + 1]))
                   if codes[k].strip() else None)
    return out


def _entry_def(program: str, first: int, last: int):
    """The function their lines live in, and whether they sit at its top level
    with nothing re-entrant about it. None when any condition fails."""
    try:
        tree = ast.parse(program)
    except SyntaxError:
        return None
    fns = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
           and n.lineno <= first and (n.end_lineno or 0) >= last]
    if not fns:
        return None
    fn = max(fns, key=lambda n: n.lineno)            # innermost
    for st in fn.body:                               # top level only
        s0, s1 = st.lineno, st.end_lineno or st.lineno
        if s0 < first <= s1 or s0 <= last < s1 and s0 < first:
            return None                              # their lines are nested
    names = lambda n: {x.attr if isinstance(x, ast.Attribute) else x.id
                       for x in ast.walk(n) if isinstance(x, (ast.Attribute, ast.Name))}
    if fn.name in names(ast.Module(body=fn.body, type_ignores=[])):
        return None                                  # recursive
    # Every function that can reach this one, directly or through another.
    fdefs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
             and n is not fn]
    callers, grew = set(), True
    while grew:
        grew = False
        for d in fdefs:
            if d.name not in callers and names(d) & ({fn.name} | callers):
                callers.add(d.name)
                grew = True
    starts = {fn.lineno} | {d.lineno for d in fn.decorator_list}
    return {"name": fn.name, "starts": starts, "callers": callers}


def _mentions(node, names: set) -> int:
    return sum(1 for n in ast.walk(node)
               if isinstance(n, ast.Attribute) and n.attr in names
               or isinstance(n, ast.Name) and n.id in names)


def _first_run_here(calls, pos, same, callers, block) -> bool:
    """Is call `pos` a DIRECT call of the method, and the first time anything
    in the test could have run it? A call list says so by name. A block
    statement must be a simple one that names the method exactly once - no
    loop, no second call - with nothing before it naming the method or a
    method that calls it."""
    if not block:
        return (str((calls[pos] or [""])[0]) in same
                and not any(str((c or [""])[0]) in same | callers for c in calls[:pos]))
    st = calls[pos]
    return (isinstance(st, (ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign))
            and _mentions(st, same) == 1 and not _mentions(st, callers)
            and not any(_mentions(s, same | callers) for s in calls[:pos]))


def _crash_sites(problem, tests, failures, entry):
    """(test index, 'Type: message', innermost line in the entry function) for
    each crash the conditions above allow."""
    from .context import DUNDER_CALL, is_method
    same = {entry["name"], DUNDER_CALL.get(entry["name"], entry["name"])}
    out = []
    for f in failures or []:
        i = f.get("index")
        if not (isinstance(i, int) and 0 <= i < len(tests or [])):
            continue
        t = tests[i]
        if not is_method(problem):
            exp = t.get("expected")
            if not f.get("error") or (isinstance(exp, dict) and "__error__" in exp):
                continue
            kind, where = f["error"].split(":", 1)[0], f.get("where") or []
            msg = f["error"]
        else:
            inp = t.get("input") or []
            block = bool(inp) and isinstance(inp[0], str)
            if not (inp and (isinstance(inp[0], list) or block)):
                continue
            if block:
                # A BLOCK is a program, one recorded output per top-level
                # statement. Dunders are left out: Python calls them without
                # naming them (`if x:`, `x == y`), so "first call" is unknowable.
                if entry["name"].startswith("__"):
                    continue
                try:
                    calls = ast.parse(inp[0]).body
                except SyntaxError:
                    continue
            else:
                calls = inp[0]
            text = _shown(f.get("got"))
            try:
                expected = ast.literal_eval(_shown(f.get("expected")))
            except Exception:
                expected = None
            hit = None
            for m in _CRASH.finditer(text):
                pos = _index_at(text, m.start())
                if pos is None or not (0 <= pos < len(calls)) or not m.group(3):
                    continue
                if not _first_run_here(calls, pos, same, entry["callers"], block):
                    continue
                if isinstance(expected, list) and pos < len(expected) and \
                        isinstance(expected[pos], str) and expected[pos].startswith("!"):
                    continue                         # the teacher's raises too
                try:
                    raw = ast.literal_eval(m.group(1) + m.group(3) + m.group(1))
                except Exception:
                    continue
                message, _, frames = raw.partition("\x00@")
                hit = (m.group(2), f"{m.group(2)}: {message}",
                       [tuple(int(x) for x in fr.split(":")) for fr in frames.split(",") if ":" in fr])
                break
            if hit is None:
                continue
            kind, msg, where = hit
        if kind in _NOT_THEIR_FAULT:
            continue
        inner = [ln for first, ln in where if first in entry["starts"]]
        if inner:
            out.append((i, msg, inner[-1]))
    return out


def _crash_verdict(problem, header, session, student_code, tests, res, is_last):
    """A GradeResult when the run `res` of their code shows a crash that is
    theirs to fix, else None. Two answers:

      * in the lines of the step being answered -> `incorrect`, attempt used,
        and the line named;
      * in an EARLIER step the judges accepted without running it -> no
        attempt, and they are pointed at that step and Rework (C). The step
        they are on may be fine; charging it would be charging the wrong step.
    """
    try:
        codes = [a.get("code") or "" for a in session.get("accepted") or []]
        tiers = [a.get("tier") for a in session.get("accepted") or []]
        ranges = _step_ranges(problem, header, codes + [student_code])
        mine = ranges[-1]
        if mine is None:
            return None
        program = _assemble(problem, header, *(codes + [student_code]))
        entry = _entry_def(program, *mine)
        if entry is None:
            return None
        sites = _crash_sites(problem, tests, res.failures, entry)
        lines = program.splitlines()
        for i, msg, ln in sites:
            shown = failing_cases(problem, tests,
                                  [f for f in res.failures if f.get("index") == i])
            quoted = lines[ln - 1].strip() if 0 < ln <= len(lines) else ""
            if mine[0] <= ln <= mine[1]:
                n = ln - mine[0] + 1
                return _ok("incorrect", "execution-crash",
                           f"Your step crashed on line {n} of your answer"
                           + (f" (`{quoted}`)" if quoted else "")
                           + f": {msg}. Fix that line and try again.",
                           "own_code_crash", execution_outcome="runtime_error",
                           failures=res.failures, failing_cases=shown,
                           failed_total=_failed_total(res, len(shown)))
            for k, r in enumerate(ranges[:-1]):
                if r and r[0] <= ln <= r[1] and tiers[k] in _UNCONFIRMED_ACCEPTS:
                    return _ok("indeterminate", "execution-crash",
                               f"The crash is in step {k + 1}, which was accepted "
                               f"earlier without being fully confirmed"
                               + (f" (`{quoted}`)" if quoted else "")
                               + f": {msg}. Your attempt was not used - reopen "
                               f"step {k + 1} with Rework and fix it there.",
                               "earlier_step_crash", deterministic=True,
                               consume_attempt=False, execution_outcome="runtime_error",
                               failures=res.failures, failing_cases=shown,
                               failed_total=_failed_total(res, len(shown)))
    except Exception:
        return None                     # a hint is never worth breaking grading
    return None


_CRASH_CUT = re.compile(r"""['"]!([A-Za-z_]\w*)\\x00.*$""", re.S)


def _crashes_in(v, expected: bool = False, same: set = frozenset()):
    """A recorded value as a person should read it, plus the crashes inside it.

    A call that raised is recorded as "!Name" + NUL + message (context.py,
    ERROR_SEP). Shown raw that is `'!AttributeError\\x00...'`; shown here it is
    <crashed: AttributeError> in the list, and the message comes back as
    (position, "AttributeError: 'Calculator' object has no attribute '_expr'")
    for the line underneath. An EXPECTED crash reads <raises IndexError>: that
    is the teacher's method raising on purpose, not a fault - and so is theirs
    raising the same exception at the same position (`same`, the (position,
    name) pairs of the expected value, which is what `expected=True` returns in
    place of crashes)."""
    text = _shown(v)
    crashes, pos = [], [-1]

    def swap(m):
        name, raw = m.group(2), m.group(3)
        idx = _index_at(text, m.start())
        if expected:
            crashes.append((idx, name))
        elif idx is None or (idx, name) not in same:
            msg = name
            if raw:
                try:
                    import ast as _a
                    text_ = _a.literal_eval(m.group(1) + raw + m.group(1))
                    msg = f"{name}: {text_.partition(chr(0) + '@')[0]}"
                except Exception:
                    msg = f"{name}: {raw.split(chr(92) + 'x00@')[0]}"
            crashes.append((idx, msg))
            return f"<crashed: {name}>"
        return f"<raises {name}>"

    text = _CRASH.sub(swap, text)
    # Long outputs are cut to a length cap, and the cut can land inside a
    # crash's message - which left `"!AttributeError\\x00'Advanced...<truncated>`
    # on screen. A crash the cap cut through is still a crash.
    cut = _CRASH_CUT.search(text)
    if cut:
        text = text[:cut.start()] + (f"<raises {cut.group(1)}>" if expected
                                     else f"<crashed: {cut.group(1)}>") + " ..."
    return text, crashes


def _index_at(text: str, offset: int):
    """Which top-level list element `offset` falls in, or None if `text` is not
    a flat list. Only for pairing a crash with the call that produced it."""
    if not text.startswith("["):
        return None
    depth, idx, quote = 0, 0, None
    for i, ch in enumerate(text[:offset]):
        if quote:
            if ch == "\\":
                continue
            if ch == quote and text[i - 1] != "\\":
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch in "[({":
            depth += 1
        elif ch in "])}":
            depth -= 1
        elif ch == "," and depth == 1:
            idx += 1
    return idx


def _count_outputs(v) -> int:
    if isinstance(v, dict) and "len" in v:
        return int(v.get("len") or -1)
    return len(v) if isinstance(v, list) else -1


def _shown(v) -> str:
    """A recorded value as a person should read it.

    execution.brief() wraps every value as {"repr", "type", "len"} so a 7MB
    result can never cross the IPC boundary. That is the right thing to send
    between processes and the wrong thing to show a student, who would be told
    their answer was `{'repr': '[None, None, 4]', 'type': 'list', 'len': 3}`."""
    if isinstance(v, dict) and "repr" in v and "type" in v:
        return str(v["repr"])
    return repr(v)


# How many failing cases a student may see at once.
#
# It was ONE, on the argument that a student who could read the whole suite
# would write code that satisfies the tests instead of the problem. That
# argument is about the SUITE, not about the number one, and one case turned out
# to be too few to debug from: told only that invert({'a':1,'b':1}) came back
# wrong, a student cannot see whether their code keeps repeated values or drops
# the wrong one of the pair - two cases distinguish those, one does not. Three
# is enough to show the shape of a mistake and far short of an answer key, and
# the suite stays hidden behind it either way.
MAX_SHOWN_CASES = 3


def failing_cases(problem: dict, tests: list, failures: list,
                  limit: int = MAX_SHOWN_CASES) -> list[str]:
    """Up to `limit` failing cases, each rendered for a human.

    A case that cannot be rendered is SKIPPED rather than aborting the list: a
    hint is never worth an error, and one awkward input must not cost the
    student the two cases that would have told them something."""
    out = []
    for f in failures or []:
        if len(out) >= limit:
            break
        i = f.get("index")
        if not (isinstance(i, int) and 0 <= i < len(tests or [])):
            continue
        try:
            out.append(_render_case(problem, tests[i], f))
        except Exception:
            continue
    return out


def _failed_total(res, shown: int) -> int:
    """How many oracle cases the submission actually failed.

    NOT len(shown): the sandbox reports at most five wrong-output failures
    (main/execution.py), so the list is a sample and `total - passed` is the
    count. Falls back to what was shown when the run reported no totals, which
    is the only honest answer there - better to under-report than to claim a
    number the run did not produce."""
    missed = int(getattr(res, "total", 0) or 0) - int(getattr(res, "passed", 0) or 0)
    return max(missed, shown)


_SYNTAX_LINE = re.compile(r"\s*\(line (\d+)\)\s*$")


def _syntax_message(raw: str, problem: dict, prefix: str,
                    student_code: str) -> str:
    """A parse error the student can act on.

    The compiler sees the WHOLE assembled module - the teacher's classes, the
    accepted steps above, the injected driver - so it reports the line number in
    that module. On HW3 a typo on the second line the student typed came back as
    "invalid syntax (line 94)", and there is no line 94 anywhere they can see.
    Nothing about their own code was in the message.

    Translated to a line of THEIR submission, and the offending line is quoted,
    because "If self.top is not None:" beside the words "invalid syntax" is the
    whole explanation - a capital I. An unmappable number is dropped rather than
    guessed at: no number at all beats a wrong one."""
    m = _SYNTAX_LINE.search(raw or "")
    base = _SYNTAX_LINE.sub("", raw or "").strip() or "invalid syntax"
    lines = (student_code or "").splitlines()
    if not m or not lines:
        return f"Your code doesn't parse: {base}."

    # Everything the assembler puts ABOVE the student's own first line.
    from .context import is_method
    if is_method(problem):
        before = len((problem.get("context_prefix") or "").splitlines())
    else:
        before = 1                                   # the def line
    before += len(prefix.splitlines()) if prefix.strip() else 0

    n = int(m.group(1)) - before                     # 1-based within their code
    if not (1 <= n <= len(lines)):
        return f"Your code doesn't parse: {base}."
    quoted = lines[n - 1].strip()
    where = f"line {n} of your answer"
    return (f"Your code doesn't parse: {base}, on {where}"
            + (f" - `{quoted}`." if quoted else "."))


# Python's own wording for an indentation fault, translated into the edit the
# student has to make. The raw message is accurate and useless to them:
# "expected an indented block after 'for' statement on line 1" describes the
# parser's state, not what to type.
_INDENT_ADVICE = (
    ("expected an indented block",
     "The line above it opens a block, so the line under it has to sit further "
     "in - that is what puts it inside."),
    ("unexpected indent",
     "It sits further in than the line above it, but nothing above it opens a "
     "block."),
    ("unindent does not match",
     "It sits between two levels - it has to line up with one of the blocks "
     "already open above it."),
)


def _indent_message(student_code: str) -> str | None:
    """An indentation fault, named as one and pointed at the student's own line.

    Kept apart from _syntax_message because it is the one parse error with a
    mechanical fix, and because the generic wording actively misleads here: "your
    code doesn't parse" sends a student hunting for a typo through code that is
    spelled perfectly and merely lines up wrong.

    ONLY THE INSIDE OF THE ANSWER IS EVER JUDGED. Which column the block as a
    whole starts at is not the student's to get right - they were never told it,
    and align_submission() has already re-seated the submission before this runs
    (see main/indent.py) - so a fault here can only be lines disagreeing with
    each other, and the last sentence says so rather than leaving them to wonder
    whether the step wanted some depth they failed to guess.

    Parsed inside a synthetic `def` for the same reason _parse_body is: a body
    holding `return` does not parse on its own. Returns None for anything that
    is not an indentation fault - an ordinary syntax error is _syntax_message's.
    """
    if not (student_code or "").strip():
        return None          # a blank answer is blank_answer, not a bad indent
    try:
        ast.parse("def _w():\n" + _indent(student_code))
    except IndentationError as e:
        lines = (student_code or "").splitlines()
        # Line 1 of the wrapper is the synthetic def, so their own numbering is
        # one less. An unmappable number is dropped rather than guessed at.
        n = (e.lineno or 0) - 1
        on_line = 1 <= n <= len(lines)
        quoted = lines[n - 1].strip() if on_line else ""
        raw = (e.msg or "").lower()
        head = ("Your indentation doesn't line up"
                + (f" on line {n} of your answer" if on_line else "")
                + (f" - `{quoted}`." if quoted else "."))
        advice = next((a for k, a in _INDENT_ADVICE if k in raw), None)
        return " ".join([
            head,
            advice or f"Python says: {e.msg}.",
            "You don't have to match the step's own indentation - that is added "
            "for you - but the lines inside your answer have to agree with each "
            "other.",
        ])
    except SyntaxError:
        return None
    return None


def _module_names(problem: dict, header: str = "") -> set:
    """Top-level names the assembled program already defines.

    `header` MATTERS FOR A PLAIN FUNCTION, and leaving it out convicted every
    recursive submission. build_program() for a non-method is the def line plus
    the body, so without the def line the assembled module defines nothing at
    all - and the function's OWN name is therefore not in scope. A student
    writing `return n * factorial(n - 1)` was told "`factorial` isn't defined",
    which is both wrong and the exact opposite of what they needed to hear.
    Python binds a function's name before its body ever runs, so a function can
    always call itself.

    The scope gate asks "does every name this step reads resolve to something
    real", and for a METHOD the answer depends on the module it was carved out
    of - which the gate was not looking at. `Stack.push` begins
    `node = Node(value)`, and `Node` is a class the teacher GAVE the student at
    the top of the same file; `Calculator._getPostfix` opens by constructing a
    `Stack()`, defined 150 lines above it. Both came back "isn't defined", on
    the first chunk, for exactly the code the handout leads them to write.

    Read off the assembled program rather than tracked separately, so it cannot
    drift from what the student's code will actually run beside. Failure is an
    empty set: the gate then behaves as it did before, which is conservative in
    the direction of a false rejection but never of a false pass."""
    from .context import build_program
    try:
        tree = ast.parse(build_program(problem, "pass", header))
    except Exception:
        return set()
    out = set()
    for n in tree.body:
        if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            out.add(n.name)
        elif isinstance(n, ast.Assign):
            out |= {t.id for t in n.targets if isinstance(t, ast.Name)}
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            out.add(n.target.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            out |= {(a.asname or a.name).split(".")[0] for a in n.names}
    return out


def _header_name(header: str) -> str:
    """The name on the def line the student is writing under, or ""."""
    try:
        tree = ast.parse((header or "").strip() + "\n    pass")
    except SyntaxError:
        return ""
    fn = tree.body[0] if tree.body else None
    return fn.name if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) else ""


def _has_no_statements(student_code: str) -> bool:
    """Their answer is comments and nothing else.

    Comments are not run, so this is an empty answer wearing the shape of a
    full one - and the blank check above cannot see it, because the text is not
    blank. It then behaves exactly like the `def` line bug reported on
    2026-09-22: nothing executes, nothing binds, no tier can attribute
    anything, and it reaches the judges - which cannot convict - so the student
    is told "we could not confirm this step" about an answer that contains no
    code to confirm.

    Sound without any reference: a submission with no statements cannot be a
    correct answer to a coding step, whatever the reference says. Parsed with a
    sentinel appended because a body of pure comments raises IndentationError
    on its own rather than parsing to something empty.

    Deliberately narrow: a docstring, a `pass`, or an `if` whose body never
    runs are all STATEMENTS and are left to the tiers below. Only the genuinely
    statement-free answer is named here."""
    try:
        tree = ast.parse("def _w():\n" + _indent(student_code) + "\n    pass")
    except SyntaxError:
        return False             # does not parse: _syntax_message's business
    fn = tree.body[0] if tree.body else None
    if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    return len(fn.body) == 1     # the sentinel, and nothing of their own


def _redefines_enclosing(student_code: str, header: str) -> str | None:
    """They typed the `def` line again, inside the body it already opened.

    REPORTED BY A REAL STUDENT (2026-09-22, _isNumber): the editor shows
    `def _isNumber(self, txt):` as a frozen first line and the box below it is
    the BODY, but writing a whole function is what you do everywhere else in
    Python, so they wrote `def _isNumber(txt):` again and indented their work
    under it. Nothing then runs: the body's only statement defines an inner
    function that is never called, so the method binds nothing and returns None.

    That is invisible to every tier below. It parses, it violates no policy,
    its names are all in scope (the inner def rebinds the parameters), and no
    value can be matched because none was produced - so it fell through the
    deterministic tiers to the LLM judges, which cannot convict, and came back
    "We could not confirm this step." Measured: their logic was RIGHT - the same
    body without the def line grades `correct` at execution-reference. They spent
    twenty minutes on a line the page could have named instantly.

    ONLY AN EXACT REDEFINITION OF THE ENCLOSING FUNCTION. A nested helper under
    any other name is legitimate Python and is left alone, and recursion is a
    Call rather than a FunctionDef, so a recursive answer never matches here -
    that distinction matters, recursion has been false-convicted before."""
    name = _header_name(header)
    if not name:
        return None                  # no def line to repeat
    try:
        tree = ast.parse("def _w():\n" + _indent(student_code))
    except SyntaxError:
        return None                  # a parse error is _syntax_message's business
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return (f"The `def {name}(...)` line is already written for you - it is "
                    f"the first line shown above the box. This step is the code that "
                    f"goes INSIDE it, so write just those lines and leave the `def` "
                    f"line out.")
    return None


def _header_params(header: str) -> set:
    """Parameter names of the def line the STUDENT is actually writing under.

    THE ONE THING resolved["params"] cannot supply for a method. A method's
    resolved entry point is the injected call-sequence driver, so its parameter
    list is ["calls"] - the driver's - while the student is writing the body of
    `def pop(self):`. The scope gate took the driver's list, so `self` was not
    in scope, and the very first chunk of every class problem came back
    "This step uses `self`, which isn't defined" - the teacher's own reference
    answer included. Every problem in HW3 is a method.

    Parsed from the header rather than pattern-matched so that defaults,
    *args/**kwargs and keyword-only parameters all resolve. Returns an empty set
    for anything that is not a def line, which leaves plain functions exactly as
    they were - their resolved params are already the right answer."""
    try:
        tree = ast.parse((header or "").strip() + "\n    pass")
    except SyntaxError:
        return set()
    fn = tree.body[0] if tree.body else None
    if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return set()
    a = fn.args
    names = {x.arg for x in [*a.posonlyargs, *a.args, *a.kwonlyargs]}
    if a.vararg:
        names.add(a.vararg.arg)
    if a.kwarg:
        names.add(a.kwarg.arg)
    return names


def _has_star_import(code: str) -> bool:
    """Does this body do `from X import *`?

    A star import binds names this module cannot enumerate, so every name the
    step reads might legitimately come from it. Live, `from math import *`
    followed by `sqrt(nums[0])` was failed as "`sqrt` isn't defined" - valid
    Python, allowed by the execution policy (math is in _ALLOWED_IMPORTS), and
    convicted by the gate in front of it. The gate cannot answer this question,
    so it must decline to answer it rather than guess wrong."""
    try:
        tree = _parse_body(code)
    except SyntaxError:
        return False
    return any(isinstance(n, ast.ImportFrom)
               and any(a.name == "*" for a in n.names)
               for n in ast.walk(tree))


def _scope_violation(student_code: str, in_scope: set,
                     star_import: bool = False) -> tuple[str, str] | None:
    """Deterministic pre-LLM gate: every name this step READS must resolve to
    something real - a function parameter, a variable an earlier accepted step
    produced, or a safe builtin. A step that reads an undefined name (a typo, or
    the wrong parameter name) is the student's own error and is failed here,
    before any model is consulted. Without this, the resulting NameError surfaces
    later as `ownership ambiguous` and is handed to the LLM judge, which grades
    intent and green-lights nonsense like `max(list)`.

    Returns (student_message, reason_code) on a violation, else None. A parse
    failure returns None - syntax is classified elsewhere, and so does a star
    import anywhere in the accepted prefix or this step: see _has_star_import.
    """
    if star_import:
        return None
    try:
        tree = _parse_body(student_code)
    except SyntaxError:
        return None
    bound = _bound_names(tree)
    reads = {n.id for n in ast.walk(tree)
             if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    # `x += 1` reads x before writing it; the AST marks the target Store-only.
    for n in ast.walk(tree):
        if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name):
            reads.add(n.target.id)

    unknown = sorted(x for x in (reads - bound)
                     if x not in in_scope and x not in _SAFE_BUILTINS)
    if unknown:
        shown = ", ".join(f"`{u}`" for u in unknown[:3])
        return (f"This step uses {shown}, which isn't defined. Use the "
                f"function's parameters or a value from an earlier step.",
                "undefined_name")

    bare = sorted(x for x in _bare_builtin_types(tree)
                  if x not in bound and x not in in_scope)
    if bare:
        # NAME THE VARIABLE THEY ALMOST CERTAINLY MEANT. "Did you mean one of
        # the function's parameters?" was a fixed sentence, and on the case it
        # fires most - `new_dict = {}` ... `return dict` - it is pointing at
        # the wrong thing entirely: the parameter is `txt`, and the answer is a
        # variable an earlier step produced. A student who follows that
        # sentence goes looking in the one place the fix is not.
        near = difflib.get_close_matches(bare[0], sorted(in_scope | bound),
                                         n=1, cutoff=0.6)
        return (f"This step uses `{bare[0]}` as a value, but that is Python's "
                + (f"built-in type. Did you mean `{near[0]}`?" if near else
                   "built-in type. Use one of the function's parameters, or a "
                   "value from an earlier step."),
                "builtin_type_as_value")
    return None


def align_submission(session: dict, student_code: str) -> str:
    """Re-seat a submission at the indent depth of the chunk it answers.

    A chunk may begin part-way through a loop or conditional body, in which case
    its reference is stored indented and a flat answer stitched at column 0 lands
    OUTSIDE the block - see main/indent.py for the failure this prevents. The
    student was never told what depth to type at, so it is not theirs to get
    wrong.

    Public and pure: /grade_chunk calls it to store exactly the text that
    grade_submission judged, so the accepted prefix and the graded code can
    never drift apart.
    """
    chunks, idx = session["chunks"], session["index"]
    if idx >= len(chunks):
        return (student_code or "").strip()
    seated = align_to_chunk(student_code, chunks[idx])
    want = base_indent(chunks[idx].get("reference") or "")
    # THE REFERENCE'S DEPTH ASSUMES THE TEACHER'S STRUCTURE. A step that
    # continues the teacher's loop is seated four columns in - but a student who
    # wrote their OWN complete earlier step has no loop left open there, and
    # every answer they could type came back "unexpected indent, on line 1".
    # Measured live on replace-variables: 11 correct answers marked wrong, one
    # student. The reference depth is still tried FIRST, so an answer that
    # parsed before is seated exactly as before; only when it cannot parse
    # after THIS student's own accepted steps is another depth tried, nearest
    # first. Code that parses at no depth stays at the reference depth and is
    # their syntax error.
    if not seated or _seat_fits(session, seated):
        return seated
    for depth in sorted(range(0, want + 12, 4), key=lambda d: (abs(d - want), d)):
        if want and depth != want and _seat_fits(session, align_to(student_code, depth)):
            return align_to(student_code, depth)
    # LAST, and only when nothing above parses: a block pasted into the
    # editor's own starting indent (indent.unpad_first_line).
    unpadded = unpad_first_line(student_code)
    if unpadded != student_code:
        again = align_submission(session, unpadded)
        if again and _seat_fits(session, again):
            return again
    return seated


def _seat_fits(session: dict, code: str) -> bool:
    """Does `code` parse where it lands, after this student's accepted steps?
    An answer ending in a block opener (`for w in words:`) is left for the next
    step to fill, so it is tried with a placeholder body as well."""
    prefix = "\n".join(accepted_prefix(session))
    last = [ln for ln in code.splitlines() if ln.strip()][-1]
    body = " " * (len(last) - len(last.lstrip()) + 4) + "pass"
    for trial in (code, code + "\n" + body):
        try:
            compile(_assemble(problem_of(session), session.get("header") or "",
                              prefix, trial), "<seat>", "exec")
            return True
        except (SyntaxError, ValueError):
            continue
        except Exception:
            return True             # not a question of depth - leave it be
    return False


def _ok(verdict, tier, reason, code, **kw) -> GradeResult:
    # Deterministic by default: every path here except the LLM judge and the
    # system failures reaches its verdict by actually running code. Those two
    # pass deterministic=False explicitly.
    kw.setdefault("deterministic", True)
    return GradeResult(verdict=verdict, tier=tier, student_reason=reason,
                       reason_code=code, **kw)


def _system(reason_code: str, detail: str | None = None) -> GradeResult:
    """Our fault. Never consumes an attempt, never convicts."""
    return _ok("indeterminate", "system",
               "The grader could not safely decide this one. Your attempt was "
               "not used - please try again.", reason_code,
               deterministic=False, consume_attempt=False, internal_detail=detail)


def _provider_down(reason_code: str, detail: str | None = None) -> GradeResult:
    """The model provider did not answer.

    Identical guarantees to _system(), namely indeterminate with no attempt
    consumed, but it NAMES the cause. "The grader could not decide" reads as
    a fault in
    the student's answer; an outage is not, and a student whose answer may well
    be correct deserves to know the difference. `detail` stays internal; only
    the sentence below ever reaches the browser."""
    return _ok("indeterminate", "system",
               "OpenAI is down - the service we use to check this step isn't "
               "responding right now. Your attempt was NOT used. Please try "
               "again in a moment.", reason_code,
               deterministic=False, consume_attempt=False, internal_detail=detail)


# ── Deferred initializer hoist - deterministic, before Tier 3 ────────────

def _is_literal_declaration(value, problem: dict, header: str = "") -> bool:
    """May this right-hand side be restated anywhere, without computing?

    True for a literal (`{}`, `[]`, `0`, `{'+': 1, '-': 1}`) and for a call
    taking NO arguments (`Stack()`, `set()`, `dict()`). An empty constructor is
    a literal wearing a name: it depends on nothing, so it means the same thing
    at any point in the function.

    False for everything else, and the argument list is what does the work.
    `sum(values)`, `len(nums)` and `list(values)` are all calls that read the
    student's problem and answer part of it; `Stack()` cannot. See
    _hoistable_declarations for the submission this was written against."""
    if isinstance(value, ast.Call):
        if value.args or value.keywords:
            return False
        return (isinstance(value.func, ast.Name)
                and value.func.id in (_module_names(problem, header)
                                      | _SAFE_BUILTINS))
    try:
        ast.literal_eval(value)
        return True
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False

def _hoistable_declarations(problem: dict, chunks: list, header: str,
                            upto: str, ref_tail: str) -> list[str]:
    """Declarations the trusted tail assumes, that the student has not made YET.

    The tail is the teacher's own code for the remaining steps, so it reads
    whatever the reference had bound by this point - including names the
    reference created long before it needed them. Calculator._getPostfix opens
    chunk 1, the tokenizer, with `postfixStack = Stack()` and
    `precedence = {...}` and touches neither until chunk 3. A student who defers
    both to chunk 3 has written a correct tokenizer, but the tail reads them as
    if they exist, so the composed program died on NameError and the submission
    fell through to the Tier 3 adapter. Across 24 live trials of the IDENTICAL
    student code that adapter returned a clean calibrated rewrite 12 times and,
    the other 12, also invented a self-referencing alias
    ({"target": "postfixStack", "source": "postfixStack"}) that its alias
    check rightly refused - so WHICH TIER decided a correct answer was a coin flip on
    the model's mood rather than on anything the student did.

    Returns the lines to prepend to the tail, in the reference's own order, or
    [] to change nothing. It never judges: it only restates a declaration the
    tail was always entitled to assume, and the tests and the pass/fail
    comparison are untouched.

    ALL OR NOTHING, AND ONLY LITERAL DECLARATIONS. `postfixStack = Stack()` and
    `prec = {...}` mean the same thing wherever they run. `total = n * 2` does
    not - `n` was computed somewhere specific in the reference's own control
    flow, and moving that line early either raises a NameError that reads as a
    grader bug or, far worse, binds silently to some unrelated `n` in the
    student's own code and makes a wrong answer look right. One name that cannot
    be settled this way disqualifies the whole submission, which then takes the
    Tier 3 path exactly as it does today.

    "READS ONLY PARAMETERS" WAS NOT A STRICT ENOUGH TEST, and the gap was
    reproducible rather than theoretical: for a reference whose first chunk is
    `result = sum(values)`, that line reads nothing but the function's own
    parameter, so it qualified - and a student submitting literally `pass`
    received correct/deterministic=True, with the grader having computed the
    answer on their behalf. Two rules close it:

      LITERAL ONLY   the right-hand side must be a literal, or a call with NO
                     arguments (`Stack()`, `set()`). `sum(values)` is a call
                     WITH an argument and can compute; `{}` and `{'+': 1}`
                     cannot. This is what separates declaring a container from
                     filling one.
      GAP ONLY       hoisting may fill gaps beside the student's work, never
                     stand in for all of it. If the tail reads nothing the
                     student actually bound, there is no work to fill a gap in,
                     and the submission is refused here.
    """
    tail_free = (_names(ref_tail, ast.Load) - _names(ref_tail, ast.Store)
                 - _header_params(header) - _module_names(problem, header)
                 - _SAFE_BUILTINS)
    needed = tail_free - _names(upto, ast.Store)
    if not needed:
        return []            # the student bound everything the tail reads
    if not (tail_free & _names(upto, ast.Store)):
        return []            # GAP ONLY - the student supplied none of it
    try:
        tree = _parse_body("\n".join((c.get("reference") or "") for c in chunks))
    except SyntaxError:
        return []
    found = []
    for name in needed:
        matches = [n for n in ast.walk(tree)
                   if isinstance(n, ast.Assign) and len(n.targets) == 1
                   and isinstance(n.targets[0], ast.Name)
                   and n.targets[0].id == name]
        if len(matches) != 1:
            return []        # nowhere, or several places: not ours to guess at
        if not _is_literal_declaration(matches[0].value, problem, header):
            return []        # computes something: not ours to hand over
        found.append(matches[0])
    # The reference's own order, in case one declaration is ever written in
    # terms of another - and so the same submission always yields the same tail.
    return [ast.unparse(n)
            for n in sorted(found, key=lambda n: (n.lineno, n.col_offset))]


# ── TIER 3 - CALIBRATED ADAPTATION ───────────────────────────────────────
#
# A model rewrites the teacher's remaining steps so they read the STUDENT's
# names. Before it may say anything about the student, the rewrite must pass
# the full oracle after the teacher's OWN steps (calibration) - so when it then
# runs cleanly on the student's work and comes out wrong, that is evidence
# about their step and not about a broken rewrite. Evidence, not a verdict: the
# failing cases are shown and no attempt is used.
#
# MEASURED ON 137 REAL STEPS (2026-09-27, the live code, one real call each):
# 86% of tries were thrown away before running and none was ever accepted.
# Four causes, each fixed below where it lives:
#   * the model filed its name pairs the other way round from what the checker
#     insisted on - 54 of 136 pairs;
#   * it echoed the example pair from these instructions verbatim - 43 of 137;
#   * a "pasted the teacher's code" rule matched a tail found ANYWHERE in the
#     solution, so a bare `return True` counted - 34;
#   * one unusable pair threw away the whole try, though pairs feed only
#     calibration.
# Replayed with the fixes, the same recorded answers showed 44 students their
# failing cases instead of 2. They still confirmed no correct step - that is
# what tier 4 does well, and why it runs after this one.
#
# ONE TRY. The model runs at temperature 0: a second try repeated the first
# word for word 17 times in 25, yet 136 of 141 real steps paid for all four.
MAX_ADAPT_TRIES = 1

_ADAPT_SYSTEM = (
    "You rewrite the remaining part of a partially written Python function so "
    "that it works with the student's variables. Return STRICT JSON only, with "
    'two keys: "adapted_tail" and "aliases". adapted_tail is ONE string holding '
    "the remaining body code, lines separated by newlines - no def line, no "
    "imports, no markdown - and it must read the STUDENT's variables. aliases "
    "is a list with one object per variable of "
    "the original remaining logic that holds the same thing as one of the "
    'student\'s variables; each object has the key "teacher" (the original\'s '
    'variable name) and the key "student" (the student\'s variable name). '
    "Plain variable names only, never expressions. Use an empty list when "
    "there are none.")


def _request_adaptation(problem, header, upto, reference_tail, student_outputs):
    """Ask for a tail that builds on the student's interface. Isolated so tests
    can substitute it without a network call."""
    user = (f"PROBLEM:\n{(problem.get('description') or '')[:600]}\n\n"
            f"Function header: {header}\n\n"
            f"Code so far (the student's own approach):\n{upto}\n\n"
            f"The remaining logic was originally written as:\n{reference_tail}\n\n"
            f"Variables the student's code produced: {sorted(student_outputs) or 'none'}\n\n"
            "Rewrite the remaining logic so it builds on the student's variables. "
            "Do not restate their work and do not recompute the answer from scratch.")
    raw = chat(GRADING_MODEL, _ADAPT_SYSTEM, [{"role": "user", "content": user}],
               temperature=0, fmt="json")
    data = json.loads(raw)
    tail = data.get("adapted_tail") or ""
    # A LIST OF LINES is still the tail. Measured: 70 of 88 real answers came
    # back that way, and str() of the list is a list literal - valid Python
    # that does nothing - so every one of them failed calibration.
    if isinstance(tail, list) and all(isinstance(ln, str) for ln in tail):
        tail = "\n".join(tail)
    return (tail if isinstance(tail, str) else ""), data.get("aliases") or []


def _valid_aliases(aliases, student_names: set, teacher_names: set) -> list:
    """`student = teacher` lines for CALIBRATION, the only place they are used.

    WHICH SIDE IS WHICH is read off the names, never off the key the model
    filed them under - 54 of 136 real pairs came back the other way round, and
    each one used to throw away the whole try. A pair naming nothing real on
    either side (an echoed example, an expression like `self.top.value`) is
    DROPPED rather than fatal. That is safe because a pair can only help the
    rewrite pass on the teacher's work; a verdict still needs it to pass on
    the student's own, where no pair is applied at all."""
    out = []
    for a in aliases if isinstance(aliases, list) else []:
        if not isinstance(a, dict):
            continue
        t = a.get("teacher", a.get("target"))
        s = a.get("student", a.get("source"))
        if not (isinstance(t, str) and isinstance(s, str)
                and t.isidentifier() and s.isidentifier()):
            continue
        if s in student_names and t in teacher_names:
            out.append(f"{s} = {t}")
        elif t in student_names and s in teacher_names:
            out.append(f"{t} = {s}")
    return out


def _calibrate(problem, header, trusted_prefix, alias_lines, tail, tests, entry):
    """An adapter must prove itself on TRUSTED work before it may judge a
    student. A random LLM tail that fails proves nothing about the student:
    it may simply be a broken tail. Only a tail that passes here has earned
    the right to produce a verdict."""
    cand = _assemble(problem, header, trusted_prefix, "\n".join(alias_lines), tail)
    return classify_run(cand, tests, entry_name=entry).outcome == "pass"


_REACHED = "_mt_rewrite_reached"


def _tail_reached(problem, header, trusted_prefix, alias_lines, tests, entry) -> set:
    """Indices of the tests on which calibration actually RAN the rewrite: the
    same program with a tail that raises on arrival. A test it passes returned
    before the tail. Every crash comes back (execution reports them all), so
    the set is complete; a run our machinery could not finish vouches for none.
    ponytail: per test, not per call - a sequence whose rewrite ran on one call
    counts as covered for all of them."""
    probe = _assemble(problem, header, trusted_prefix, "\n".join(alias_lines),
                      f'raise RuntimeError("{_REACHED}")')
    res = classify_run(probe, tests, entry_name=entry)
    return {f.get("index") for f in res.failures or []
            if _REACHED in (f.get("error") or _shown(f.get("got")))}


def _squash(code: str) -> str:
    return re.sub(r"\s+", "", code or "")


def _step_outputs(student_code: str) -> set:
    """Every name their step produces: what it assigns, and any function or
    class it DEFINES. Defs are not Store names, so they were missed - a
    completion calling their helper was then tested by deleting the step, and
    the NameError read as "their work mattered" whatever it actually did."""
    out = _names(student_code, ast.Store)
    try:
        out |= {n.name for n in _parse_body(student_code).body[0].body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    except SyntaxError:
        pass
    return out


def _their_work_matters(problem, header, prefix, upto, tail, outputs, tests, entry) -> bool:
    """Does the finished program BREAK without the student's step?

    Deleting the step is foolable when the tail reads their names: it just
    raises NameError, which reads as "it broke without them" even when the tail
    did all the work. So a tail that reads their names gets their VALUES blanked
    instead (a function becomes one that does nothing - `type(f)()` cannot
    build a function, and crashing read as dependence for free). A tail that
    reads none of them - their step only guards or returns - has nothing to
    blank, so the step is DELETED. A run our machinery could not complete
    proves nothing either way, so it never counts as breaking."""
    if bridge.free_names(tail) & outputs:
        neutral = "\n".join(
            f"{n} = (lambda *_a, **_k: None) if callable({n}) else type({n})()"
            for n in sorted(outputs))
        program = _assemble(problem, header, upto, neutral, tail)
    else:
        program = _assemble(problem, header, prefix, tail)
    return classify_run(program, tests, entry_name=entry).outcome not in ("pass", "harness_error")


# THEIR FINISHED VALUES ARE READ-ONLY. The residual risk of both model tiers:
# a step that is slightly WRONG and a continuation that quietly PATCHES the
# value before using it - `n += 1` on an off-by-one, `chars.insert(0, ...)` on
# a list missing its first item. Every test passes, and blanking their values
# still breaks it, because the patch builds on their value - so the wrong step
# would be accepted. A continuation may therefore only READ what their step
# produced, except a value the step merely set up EMPTY (`counts = {}`,
# `total = 0`, `stack = Stack()`), which carrying on is the point of.
# Methods that change the object they are called on; the course's own classes
# add push/enqueue/dequeue.
_MUTATORS = frozenset({
    "append", "extend", "insert", "pop", "remove", "clear", "sort", "reverse",
    "update", "setdefault", "popitem", "add", "discard", "intersection_update",
    "difference_update", "symmetric_difference_update",
    "push", "enqueue", "dequeue",
})


def _changed_in_place(tail: str, names: set) -> set:
    """Which of `names` the tail CHANGES without rebinding them: `n += 1`,
    `xs[i] = ...`, `node.next = ...`, `del d[k]`, `xs.append(...)`."""
    try:
        tree = _parse_body(tail)
    except SyntaxError:
        return set()

    def root(t):
        while isinstance(t, (ast.Subscript, ast.Attribute)):
            t = t.value
        return t.id if isinstance(t, ast.Name) else None

    hit = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.AugAssign):
            hit.add(root(n.target))
        elif isinstance(n, (ast.Assign, ast.AnnAssign, ast.Delete)):
            targets = n.targets if isinstance(n, (ast.Assign, ast.Delete)) else [n.target]
            for t in targets:
                for sub in ast.walk(t):
                    if isinstance(sub, (ast.Subscript, ast.Attribute)) \
                            and isinstance(sub.ctx, (ast.Store, ast.Del)):
                        hit.add(root(sub))
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                and n.func.attr in _MUTATORS:
            hit.add(root(n.func.value))
    return (hit - {None}) & names


def _only_set_up_empty(problem, header, upto, names, tests, entry) -> bool:
    """Did their code leave every one of `names` EMPTY - equal to a fresh
    `type(x)()` - on every input that reached the end of it? Measured by
    running their code, never guessed; anything unreadable is not empty."""
    names = sorted(names)
    fresh = [f"_mt_fresh_{i}" for i in range(len(names))]
    probe = "\n".join(f"try:\n    {f} = type({n})()\nexcept Exception:\n"
                      f"    {f} = ('<no empty form>',)" for f, n in zip(fresh, names))
    try:
        sig = bridge.capture(problem, header, upto + "\n" + probe, names + fresh,
                             [t["input"] for t in tests], entry)
    except Exception:
        return False
    if not sig:
        return False
    unbound = repr(bridge.UNBOUND)
    return all(a == b for n, f in zip(names, fresh)
               for a, b in zip(sig[n], sig[f]) if a != unbound)


def _patches_their_values(problem, header, upto, tail, protected, tests, entry) -> bool:
    changed = _changed_in_place(tail, protected)
    return bool(changed) and not _only_set_up_empty(problem, header, upto, changed,
                                                    tests, entry)


def _run_crashed(res) -> bool:
    """Did a run CRASH somewhere its expected output does not, rather than
    finish with a wrong answer?

    A method's crash is recorded as a value ("!NameError..."), so its run
    classifies as wrong_output like any clean wrong answer. Measured on a real,
    correct `calculate` guard: tier 3's rewrite read `calcStack`, which the
    teacher's step creates and the step never asked for, crashed with NameError
    on the student's work - and those cases would have been shown to the
    student as THEIR failing cases. A crash is the rewrite's, not evidence."""
    for f in res.failures or []:
        if f.get("error"):
            return True
        if len(_CRASH.findall(_shown(f.get("got")))) > \
                len(_CRASH.findall(_shown(f.get("expected")))):
            return True
    return False


def _tier3(problem, session, header, prefix, student_code, upto, ref_tail,
           tests, entry, outputs, corr=None):
    """An acceptance, an evidence-only result (failing cases, no attempt), a
    harness failure - or None when it proved nothing. Records no route: the
    caller, _model_tiers, decides what the student sees and records it once."""
    idx = session["index"]
    trusted_prefix = "\n".join((session["chunks"][j].get("reference") or "")
                               for j in range(idx + 1))
    prefix_names = _names(upto, ast.Store) | set(get_resolved_entry(problem)["params"])
    student_names = outputs | prefix_names
    teacher_names = _names(trusted_prefix, ast.Store) | prefix_names
    # A loop variable is a counter, not a result of theirs: a rewrite reusing
    # `value` for its own loop is not overwriting their work, and one that
    # secretly redoes the loop is caught by _their_work_matters.
    protected = outputs - bridge.loop_targets(student_code)

    for attempt in range(1, MAX_ADAPT_TRIES + 1):
        try:
            with trace.model_call(corr, GRADING_MODEL, "adapter", attempt=attempt):
                tail, aliases = _request_adaptation(problem, header, upto, ref_tail, outputs)
            tail = textwrap.dedent(tail).strip("\n")
        except Exception:
            # Model trouble. Tier 4 is next and asks for itself; if the
            # provider is really down, that is where the student hears so.
            _trace(trace.record_adapter, corr, GRADING_MODEL, attempt, "malformed")
            return None
        # THE TEACHER'S TAIL HANDED BACK UNCHANGED is not an adaptation: it is
        # tier 1 again, and tier 1 already failed - so its "failing cases"
        # would be the teacher's names missing from a correct different
        # approach. Compared with the TAIL, not with the whole solution: that
        # matched any fragment found anywhere, so `return True` counted.
        # No `+=` on their names here (allow_add=False): tier 4 needs it to
        # carry an accumulator on; a rewrite of the teacher's code never did,
        # in 137 real tries.
        if not _tail_is_sane(tail, protected, "", allow_add=False) \
                or _squash(tail) == _squash(ref_tail):
            _trace(trace.record_adapter, corr, GRADING_MODEL, attempt, "unsafe")
            continue
        if _patches_their_values(problem, header, upto, tail, protected, tests, entry):
            _trace(trace.record_adapter, corr, GRADING_MODEL, attempt, "changes_their_values")
            continue
        alias_lines = _valid_aliases(aliases, student_names, teacher_names)
        if not _calibrate(problem, header, trusted_prefix, alias_lines, tail,
                          tests, entry):
            _trace(trace.record_adapter, corr, GRADING_MODEL, attempt, "calibration_failed")
            continue
        # The alias lines belong to calibration only. They map the teacher's
        # names onto the student's; here the student's names already exist.
        cand = classify_run(_assemble(problem, header, upto, tail), tests, entry_name=entry)
        if cand.outcome == "pass":
            if not _their_work_matters(problem, header, prefix, upto, tail,
                                       outputs, tests, entry):
                _trace(trace.record_adapter, corr, GRADING_MODEL, attempt, "bypass_rejected")
                continue
            _trace(trace.record_adapter, corr, GRADING_MODEL, attempt, "accepted")
            return _ok("correct", "execution-adapted",
                       "Correct - your step works with the rest of the solution.",
                       "adapted_pass", execution_outcome="pass", divergent=True)
        if cand.outcome == "harness_error":
            return _system("harness_error", cand.internal_error)
        if cand.outcome == "wrong_output" and not _run_crashed(cand):
            # THE EVIDENCE WITHOUT THE VERDICT. A calibrated rewrite ran
            # cleanly on their work and the finished answer came out wrong.
            # Worth SHOWING, not worth convicting on: the cases are concrete
            # and checkable by hand, but that the fault is theirs rests on the
            # model having re-expressed the tail faithfully, which calibration
            # does not establish. Costs no attempt.
            #
            # ONLY CASES THE PROOF COVERED. Where the teacher's step returns
            # early, calibration never ran the rewrite, so it proves nothing
            # there - and a case from there could blame a correct step. 57 of
            # 116 saved steps skip it on some tests (calculateExpressions step
            # 2: 32 of 40). None covered, nothing to show.
            reached = _tail_reached(problem, header, trusted_prefix,
                                    alias_lines, tests, entry)
            covered = [f for f in cand.failures if f.get("index") in reached]
            if not covered:
                _trace(trace.record_adapter, corr, GRADING_MODEL, attempt,
                       "evidence_unproven")
                continue
            # The sentence is derived from what was actually shown: failing
            # _cases skips any case it cannot render, and may skip them all.
            shown = failing_cases(problem, tests, covered)
            _trace(trace.record_adapter, corr, GRADING_MODEL, attempt, "evidence_only")
            return _ok(
                "indeterminate", "execution-adapted",
                "We ran your step together with the rest of the solution and the "
                "finished answer came out wrong on at least one case. We can't "
                "be certain the fault is in this step, so your attempt was not "
                "used"
                + (" - but the case below is worth tracing by hand." if shown
                   else ". Try your step on a small input of your own and check "
                        "what it hands on to the rest of the solution."),
                "adapted_evidence_only" if shown else "adapted_evidence_unrenderable",
                deterministic=False, consume_attempt=False,
                execution_outcome="wrong_output", failures=covered,
                # A floor, never the run's total: the uncovered failures are
                # exactly the ones we cannot vouch for.
                failing_cases=shown, failed_total=max(len(covered), len(shown)))
        _trace(trace.record_adapter, corr, GRADING_MODEL, attempt,
               f"no_acquittal_{cand.outcome}")
    return None


# ── TIER 4 - A COMPLETION THE TESTS DECIDE ────────────────────────────────
#
# Replaced the two LLM judges (2026-09-26).
#
# The judges were asked "is this step correct?" - an opinion, formed by reading
# code, which is the one thing a model is worst at. Measured on every real
# acceptance they made: of 31 students who later finished, 25 had to REDO the
# step the judges had passed, and 41 more were left stuck behind it. Only 6
# kept it. Tier 3 is sound but narrow: it has to rewrite the TEACHER'S
# remaining code and prove it on the teacher's own steps first (calibration),
# which a student who took a different route can never satisfy.
#
# The model is now asked for what it is GOOD at - writing code that carries on
# from someone else's - and never for a verdict. It is shown what the student's
# variables really hold on real inputs (measured, not guessed) and asked to
# write the remaining steps on top of them. Then EXECUTION decides:
#   * the finished program must pass every oracle test, and
#   * with the student's values blanked out it must FAIL - so the completion
#     genuinely used their work instead of quietly redoing it, and
#   * it may not rebind a name their step produced (it may add to it).
# No completion that survives all three -> "could not confirm", which costs no
# attempt and comes with a question built on their own values. Known residue:
# a completion can still compensate for a subtle bug (an off-by-one undone
# later). That is why a step accepted here is tracked, and a later crash or
# failure inside it is pointed back at it (see _crash_verdict).
MAX_COMPLETE_TRIES = 2

_COMPLETE_SYSTEM = (
    "You finish a student's partially written Python function. Return STRICT "
    'JSON only: {"completion": "<the remaining body lines>"}. '
    "Body lines only - no def line, no imports, no markdown fences. "
    "BUILD ON THE STUDENT'S CODE EXACTLY AS IT IS: use the variables their code "
    "produced, with the values it really gives them (you are shown those "
    "values, measured by running it). Never reassign, rebuild, recompute or "
    "change a variable their code produced - only read it, except one their "
    "code merely set up empty (like counts = {}), which you may fill. Do not redo "
    "their work. If what their code produced cannot lead to a correct answer "
    "without changing it, return an empty completion - do not work around it.")


def _lines(code: str) -> int:
    """Lines that do something - blank lines and comments do not count."""
    return sum(1 for ln in (code or "").splitlines()
               if ln.strip() and not ln.strip().startswith("#"))


def _their_values(problem, header, upto, tests, entry, limit=5) -> str:
    """What the student's own variables hold on a few short real inputs -
    measured by running their code, never guessed. Empty when nothing honest
    can be shown."""
    try:
        names = sorted(bridge.stores(upto) - bridge.loop_targets(upto))[:6]
        if not names or not tests:
            return ""
        inputs = sorted((t["input"] for t in tests), key=lambda i: len(repr(i)))[:limit]
        got = bridge.capture(problem, header, upto, names, inputs, entry,
                             human=True, with_owners=True)
        sig, owners = got if isinstance(got, tuple) else (got, [])
        if not sig or not owners:
            return ""
        lines = []
        for i, inp in enumerate(inputs):
            pos = [p for p, o in enumerate(owners) if o == i]
            shown = None
            for p in reversed(pos):                 # the latest snapshot
                vals = {n: sig[n][p] for n in names
                        if sig[n][p] not in (None, bridge.UNBOUND)}
                if vals:
                    shown = vals
                    break
            if shown:
                lines.append(f"  input {', '.join(repr(a) for a in inp)}: "
                             + "; ".join(f"{n} = {str(v)[:200]}" for n, v in shown.items()))
        return "\n".join(lines)
    except Exception:
        return ""


def _request_completion(problem, header, upto, student_code, remaining,
                        reference_tail, evidence, temperature=0.0,
                        step_prompt: str = "") -> str:
    """Ask for the remaining body on top of the student's code. Isolated so
    tests can substitute it without a network call."""
    steps = "\n".join(f"  {k}. {p}" for k, p in enumerate(remaining, 1) if p)
    user = (f"PROBLEM:\n{(problem.get('description') or '')[:600]}\n\n"
            f"Function header: {header}\n\n"
            f"CODE SO FAR (the last part is the student's newest step):\n{upto}\n\n"
            + (f"WHAT THAT NEWEST STEP WAS ASKED TO DO:\n{step_prompt}\n\n" if step_prompt else "")
            + (f"WHAT THEIR VARIABLES HOLD after that code, measured:\n{evidence}\n\n"
               if evidence else "")
            + (f"THE REMAINING STEPS TO WRITE:\n{steps}\n\n" if steps else "")
            + f"A COMPLETE CORRECT SOLUTION, written by someone who named and "
              f"shaped things differently from this student - use it only to "
              f"see what the function must do:\n{reference_tail}\n\n"
            "First compare the student's code and measured values with what the "
            "newest step was asked to do. If they contradict it on any input "
            "shown, OR the step does not yet do everything it was asked, return "
            "an empty completion - never write the newest step's missing work "
            "yourself. Otherwise write ONLY the remaining steps, building on the "
            "student's variables, so the finished function is correct.")
    raw = chat(GRADING_MODEL, _COMPLETE_SYSTEM, [{"role": "user", "content": user}],
               temperature=temperature, fmt="json")
    return str(json.loads(raw).get("completion", "") or "")


def _tail_is_sane(tail: str, current_outputs: set, solution: str,
                  allow_add: bool = True) -> bool:
    """Structural checks before the tail is allowed anywhere near a verdict.
    `allow_add` lets it `+=` onto a name in current_outputs (tier 4 only)."""
    if not tail.strip():
        return False
    try:
        tree = _parse_body(tail)
    except SyntaxError:
        return False
    # index 0 is the synthetic wrapper from _parse_body; anything else defining
    # a function or class means the tail smuggled in a header or a whole solution.
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                and getattr(n, "name", "") != "_w":
            return False
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            return False
    # Must not overwrite what the current chunk produced.
    #
    # COMPREHENSION VARIABLES ARE NOT OVERWRITES. In Python 3 the `p` of
    # `{p[0]: p[1] for p in pairs}` is sealed inside the comprehension and
    # cannot touch a `p` the student is using, but ast marks it Store - so a
    # perfectly good adapted tail was thrown away whenever the model happened to
    # pick a letter the student had also used. Measured: the same tail with `q`
    # instead of `p` was accepted. That is a coin flip on a variable name, and
    # part of why this tier's reliability looked like ~50%.
    # ADDING TO a name is not overwriting it: `total += x` on a total their step
    # started is how an accumulation carries on, and refusing it threw away
    # correct completions. Plain rebinding (`counts = ...`) is still refused.
    augmented = {n.target.id for n in ast.walk(tree)
                 if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name)} \
        if allow_add else set()
    rebound = _names(tail, ast.Store) - bridge._comprehension_vars(tree) - augmented
    if rebound & current_outputs:
        return False
    # NO "must not be the teacher's code" rule. It used to refuse any tail found
    # verbatim in the solution, which also refused the RIGHT answer whenever
    # the student's step matched the teacher's closely enough that the correct
    # continuation IS the teacher's remaining code - measured on a real, correct
    # `calculate` guard. Cheating by pasting a solution that ignores the student
    # is caught where it belongs: _their_work_matters. (Tier 3 refuses the
    # teacher's tail handed back UNCHANGED - see _tier3 - which is narrower.)
    return True



def _returns_early(code: str) -> bool:
    """Does this step's own code `return` - outside any def or lambda of its
    own? A step that is not the last and returns has ended the function."""
    try:
        tree = ast.parse(textwrap.dedent(code or ""))
    except SyntaxError:
        return False

    def walk(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Return):
                return True
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                      ast.Lambda, ast.ClassDef)) and walk(child):
                return True
        return False
    return walk(tree)


def _with_diagnosis(result, problem, chunk, header, chunks, idx, upto,
                    student_code, tests, entry, ambient):
    """`result` plus a question built on a real input, when one can be found.

    Defensive end to end: diagnosis is help, and help must never be able to
    change a verdict or take down a grading response. Anything that goes wrong
    here leaves the result exactly as it arrived - the student still gets their
    indeterminate, still spends no attempt, and simply gets no question."""
    try:
        from . import diagnose
        example = diagnose.counterexample(problem, header, chunks, idx, upto,
                                          tests, entry, ambient)
        if example is None:
            # NOTHING TO TRACE BECAUSE NOTHING RAN. counterexample() gives up
            # here, and what the student was then left with was the bare "we
            # could not confirm this step" - the least useful sentence we have,
            # at the moment it explains the least. Saying what we OBSERVED
            # covers every shape that parses without executing at once: a step
            # inside `if __name__ == '__main__':`, a nested def that is never
            # called, a branch that cannot be taken. No rule per habit, and no
            # claim that their answer is wrong - it still costs no attempt.
            if diagnose.bound_nothing(upto) and hasattr(result, "model_copy"):
                # UNLESS IT RAN AND RETURNED. `return True` at step 1 of 2 ran
                # fine and ended the function, and was told to check that its
                # lines "actually run" (29 Sep audit, _isNumber) - advice about
                # a different mistake. Every step but the last is asked to keep
                # its result for the next step; say that.
                if _returns_early(student_code):
                    return result.model_copy(update={"student_reason":
                        "Your code for this step returns from the function, so "
                        "the steps after it never run. This isn't the last "
                        "step - work out the result and keep it for the next "
                        "step instead of returning it. Your attempt was not used."})
                return result.model_copy(update={"student_reason":
                    "We ran your code for this step and it left no values "
                    "behind for the next step to use. Check that the lines you "
                    "wrote actually run - code inside something that never "
                    "happens is never reached. Your attempt was not used."})
            return result
        msg = diagnose.question(problem, chunk, student_code, example)
    except Exception:
        return result
    return result.model_copy(update={"needs_diagnosis": True, "diagnosis": msg}) \
        if hasattr(result, "model_copy") else result


# ── the state machine ────────────────────────────────────────────────────

def grade_submission(session: dict, student_code: str,
                     oracle_loader=None, case_id: str | None = None) -> GradeResult:
    """Grade one submission against a SERVER-OWNED session. The only entry
    point; routes must add no grading logic of their own."""
    import uuid
    from .oracle_store import OracleUnusableError, load_strong_cached_oracle
    loader = oracle_loader or load_strong_cached_oracle
    # Correlation id per grading ATTEMPT. Idempotent replays never reach here
    # begin_submission() returns the stored result first - so a retry cannot
    # produce a duplicate completed-attempt trace.
    corr = case_id or f"grade-{uuid.uuid4().hex[:12]}"

    chunks, idx = session["chunks"], session["index"]
    if idx >= len(chunks):
        return _system("no_current_chunk")
    chunk = chunks[idx]
    problem = problem_of(session)
    header = session["header"]
    is_last = idx == len(chunks) - 1

    # PRECONDITION - a STRONG cached oracle, read-only. Grading must never
    # generate one, and must never proceed without one.
    try:
        tests = loader(problem)
    except OracleUnusableError as e:
        return _ok("indeterminate", "system",
                   "This problem isn't ready for grading yet. Your attempt was "
                   "not used.", e.reason_code, deterministic=False,
                   consume_attempt=False, internal_detail=str(e))
    except Exception as e:
        return _system("oracle_load_failed", repr(e)[:200])

    resolved = get_resolved_entry(problem)
    entry = resolved["entry_name"]
    prefix = "\n".join(accepted_prefix(session))
    # Re-seat the answer at this chunk's depth BEFORE anything reads it. A step
    # that continues inside a loop is stitched four columns in, and the student
    # was never told that, so a flat answer is not a wrong answer.
    student_code = align_submission(session, student_code)

    # ── TIER 1 - static policy + compile. No LLM here, ever. ──
    if not student_code:
        return _ok("incorrect", "syntax", "No answer submitted.", "blank_answer")
    # ── COMMENTS ARE NOT CODE. BEFORE THE PARSE, and that ordering is the whole
    #    point: an answer of pure comments makes the assembled function body
    #    EMPTY, which is an IndentationError, so the syntax gate below got there
    #    first and said "your indentation doesn't line up on line 1 - `# Count
    #    each value in d.`. The line above it opens a block..." - about a single
    #    comment with no line above it. Reported by an audit on the live site,
    #    reproduced here on the real `invert` problem. Sitting after the parse,
    #    this check could never fire for the one input it was written for.
    #    It needs no parse of its own: _has_no_statements appends a sentinel.
    if _has_no_statements(student_code):
        return _ok("incorrect", "syntax",
                   "There is no code in this answer - comments and blank lines "
                   "are not run, so there is nothing here to answer the step yet.",
                   "comments_only")
    upto = "\n".join(b for b in (prefix, student_code) if b.strip())
    probe = classify_run(_assemble(problem, header, upto), [], entry_name=entry)
    if probe.outcome == "policy_violation":
        return _ok("incorrect", "policy",
                   f"That answer uses something not allowed here - "
                   f"{probe.internal_error}.", "policy_violation",
                   execution_outcome="policy_violation")
    if (probe.internal_error or "").startswith("syntax:"):
        # Indentation is asked FIRST. It is the one parse error with a
        # mechanical fix, and "your code doesn't parse" is the wrong sentence
        # for code whose only fault is which column it starts in.
        indent_fault = _indent_message(student_code)
        if indent_fault:
            return _ok("incorrect", "syntax", indent_fault, "indentation_error")
        return _ok("incorrect", "syntax",
                   _syntax_message(probe.internal_error[7:].strip(),
                                   problem, prefix, student_code),
                   "syntax_error")

    # ── THE DEF LINE IS ALREADY THERE. Static, deterministic, no reference and
    #    no model - and it has to run BEFORE the scope gate, because an inner
    #    def rebinds the parameters and so looks perfectly in scope. ──
    redefined = _redefines_enclosing(student_code, header)
    if redefined is not None:
        return _ok("incorrect", "syntax", redefined, "redefined_function")

    # ── SCOPE GATE - names the step READS must already exist. Deterministic,
    #    runs before any execution tier or LLM. Catches the wrong parameter
    #    name / typo that would otherwise crash and be excused as our fault. ──
    # The header's own parameters are unioned in, not substituted: for a plain
    # function the two agree, and for a METHOD the resolved params belong to the
    # injected driver rather than to the def the student is writing under.
    in_scope = (_names(prefix, ast.Store) | set(resolved["params"])
                | _header_params(header) | _module_names(problem, header))
    # A star import ANYWHERE in what has run so far - an earlier accepted step
    # or this one - makes the set of defined names unknowable, so the gate
    # declines rather than convicts. Checked on `upto`, not student_code: the
    # import may sit in a step accepted three submissions ago.
    scope = _scope_violation(student_code, in_scope, _has_star_import(upto))
    if scope is not None:
        return _ok("incorrect", "syntax", scope[0], scope[1])

    # ── LAST CHUNK - whole function, no borrowed tail ──
    if is_last:
        res = classify_run(_assemble(problem, header, upto), tests, entry_name=entry)
        if res.outcome == "pass":
            # TRACED LIKE EVERY OTHER ACQUITTAL. This is the ordinary way a
            # problem finishes and it recorded nothing, while llm-judge
            # acquittals always recorded - so a tier census read off the trace
            # showed the judges as a far larger share of `correct` than they
            # are. That census is what decides whether tier 4 can be deleted,
            # so under-counting the deterministic tiers argues the wrong way.
            _trace(trace.record_route, corr, "execution-final", "correct",
                   early=False)
            return _ok("correct", "execution-final",
                       "Correct - your full solution passes every test.",
                       "final_pass", execution_outcome="pass",
                       divergent=False)
        if res.outcome == "harness_error":
            return _system("harness_error", res.internal_error)
        msg = {"wrong_output": "Your solution runs but gives the wrong answer on "
                               "at least one case.",
               "runtime_error": "Your solution crashes while running.",
               "timeout": "Your solution took too long - it may loop forever.",
               "policy_violation": "That answer uses something not allowed here."}
        crashed = _crash_verdict(problem, header, session, student_code,
                                 tests, res, is_last=True)
        if crashed is not None:
            return crashed
        shown = failing_cases(problem, tests, res.failures)
        judged = [k + 1 for k, a in enumerate(session.get("accepted") or [])
                  if a.get("tier") in _UNCONFIRMED_ACCEPTS]
        if judged:
            # C, the rare case: a wrong ANSWER (not a crash) with a step behind
            # it that was accepted without being fully confirmed. We cannot tell which step is
            # at fault, so the attempt still counts - but they are told the
            # earlier step is a suspect and that Rework can reopen it.
            s_ = "s" if len(judged) > 1 else ""
            steps_ = ", ".join(str(k) for k in judged)
            hint = (f" Step{s_} {steps_} {'were' if s_ else 'was'} accepted "
                    f"earlier without being fully confirmed - if this "
                    f"step looks right to you, the problem may be there "
                    f"(reopen it with Rework).")
        else:
            hint = ""
        return _ok("incorrect", "execution-final",
                   msg.get(res.outcome, "Your solution didn't pass.") + hint,
                   f"final_{res.outcome}", execution_outcome=res.outcome,
                   failures=res.failures, failing_cases=shown,
                   failed_total=_failed_total(res, len(shown)))

    # ── ALREADY FINISHED? - student code only, no borrowed tail ──
    # A student who arrives with the whole solution wrote it into step 1, was
    # told "correct", and was then asked for step 2 - which their own code
    # already contained. Answering `pass` there produced "Solved. Nice work."
    # Being walked through a step you have visibly already answered teaches
    # nothing and reads as the grader not following along.
    #
    # THE SAME ORACLE THE LAST CHUNK USES, on the same student-only code, so
    # this is the final check arriving early rather than a new kind of
    # judgement. Nothing is borrowed: if their code needs the teacher's tail to
    # pass, it does not pass here and the normal flow continues below.
    #
    # A FAILURE HERE IS NOT A VERDICT. Most non-final steps will fail it, for
    # the ordinary reason that they are not finished - that is what `continue`
    # means, and reading it as anything else would convict every correct
    # partial answer. It can only ever end the problem early, never end it
    # badly.
    #
    # No extra gate is needed for "they must have planned first": the editor is
    # locked until the design gate passes (api_server._design_approved), so
    # every submission that reaches here has already been through it.
    whole = classify_run(_assemble(problem, header, upto), tests, entry_name=entry)
    if whole.outcome == "pass":
        _trace(trace.record_route, corr, "execution-final", "correct",
               early=True, covers=len(chunks) - idx)
        return _ok("correct", "execution-final",
                   "Correct - and that completes the whole problem. The rest of "
                   "the steps are already answered by what you wrote.",
                   "final_pass_early", execution_outcome="pass",
                   covers_chunks=len(chunks) - idx)

    # ── A CRASH IN THEIR OWN LINES - read off the run just made, which has
    #    nothing of the teacher's after it. See _crash_verdict. ──
    crashed = _crash_verdict(problem, header, session, student_code, tests,
                             whole, is_last=False)
    if crashed is not None:
        _trace(trace.record_route, corr, "execution-crash", crashed.verdict)
        return crashed

    # ── NON-LAST - trusted reference tail ──
    ref_tail = "\n".join((chunks[j].get("reference") or "")
                         for j in range(idx + 1, len(chunks)))
    # The tail may read a name the reference declared earlier than the student
    # chose to. Settle that here, for free, instead of letting a NameError send
    # a correct answer to the Tier 3 model - computed ONCE, so the ordinary case
    # (nothing needed) costs nothing and the fixable one is fixed on the first
    # and only run.
    hoisted = _hoistable_declarations(problem, chunks, header, upto, ref_tail)
    if hoisted:
        ref_tail = "\n".join(hoisted) + "\n" + ref_tail
    res = classify_run(_assemble(problem, header, upto, ref_tail), tests, entry_name=entry)
    if res.outcome == "pass":
        # ALWAYS, not just when something was hoisted. `hoisted` stays on the
        # event so telemetry can still count how often the ordering difference
        # is real, but gating the whole event on it meant the commonest
        # acquittal of all was invisible - see the note on execution-final
        # above. No model is consulted on this path either way.
        _trace(trace.record_route, corr, "execution-reference", "correct",
               hoisted=hoisted or [])
        return _ok("correct", "execution-reference",
                   "Correct - your step works with the rest of the solution.",
                   "reference_pass_hoisted" if hoisted else "reference_pass",
                   execution_outcome="pass")
    if res.outcome == "harness_error":
        return _system("harness_error", res.internal_error)
    if res.outcome == "policy_violation":
        return _ok("incorrect", "policy", "That answer uses something not "
                   "allowed here.", "policy_violation",
                   execution_outcome="policy_violation")

    # ── TIER 2.5 - DETERMINISTIC VALUE BRIDGE, before any model ──
    # A fixed reference tail can fail purely because it expected different
    # variable names. That is a naming difference, not a mistake, and matching
    # names by the VALUES they hold settles it by execution alone - no model, no
    # rewriting of the teacher's code, the same answer every time. It is tried
    # before Tier 3 because Tier 3 was measured at ~50% on exactly this case.
    # See main/bridge.py for why a bridge may only ever acquit.
    ambient = (set(resolved["params"]) | _header_params(header)
               | _module_names(problem, header) | _SAFE_BUILTINS)
    # Values first; when that finds nothing - no snapshot can be taken inside
    # a loop the next step carries on - pairing names by their set-up.
    bridged = bridge.find(problem, header, chunks, idx, upto, tests, entry, ambient) \
        or bridge.rename(problem, header, chunks, idx, upto, tests, entry, ambient)
    if bridged:
        covers = bridged["boundary"] - idx + 1
        _trace(trace.record_route, corr, "execution-bridged", "correct",
               mapping=bridged["mapping"], boundary=bridged["boundary"],
               by=bridged.get("by", "values"))
        # SAME SENTENCE AS EXECUTION-REFERENCE, deliberately. This used to say
        # "you named things differently to our version", which tells the
        # student a reference solution exists and that theirs was measured
        # against it - and an audit found it firing on the ORDINARY approach
        # too, so it was not even describing an unusual answer. Which tier
        # acquitted is our business; what they need to know is that their step
        # works. The tier is still on the result for telemetry.
        return _ok("correct", "execution-bridged",
                   "Correct - your step works with the rest of the solution."
                   if covers == 1 else
                   f"Correct - and you have already written what the next "
                   f"{covers - 1} step(s) asked for, so we have marked those "
                   f"done too.",
                   "bridged_pass", execution_outcome="pass",
                   divergent=True, covers_chunks=covers)

    result = _model_tiers(problem, session, chunk, header, prefix, student_code,
                          upto, ref_tail, tests, entry, corr)
    generic = result.student_reason
    # ── COULD NOT CONFIRM -> ASK, rather than leave them on a step nobody
    #    named a problem with. An indeterminate verdict does not advance the
    #    session, so every tier being acquit-only would otherwise strand a
    #    student who IS wrong: no verdict and no way forward, which is worse
    #    than the false conviction that discipline removed.
    #
    #    Only when execution actually looked. `tier == "system"` is our
    #    machinery failing - a provider outage, a broken harness - and a
    #    counterexample drawn from that would be inventing a problem in the
    #    student's code to explain one in ours.
    if result.verdict == "indeterminate" and result.tier != "system":
        result = _with_diagnosis(result, problem, chunk, header, chunks, idx,
                                 upto, student_code, tests, entry,
                                 set(resolved["params"]) | _header_params(header)
                                 | _module_names(problem, header) | _SAFE_BUILTINS)
    # ── A LOOP THEY WERE MEANT TO START. Reported 30 Sep: step 1 said "prepare
    #    everything needed", a student wrote the set-up alone, and was told to
    #    check their code against the step - which it matched. The next step
    #    carries on INSIDE a loop this step starts, so joined to theirs it
    #    cannot even be read. Said only when the shape proves it, and only in
    #    place of the generic sentence - a more specific one (returned early,
    #    never reached) keeps priority. Verdict and attempt stay as they were. ──
    if result.tier == "unconfirmed" and result.student_reason == generic \
            and idx + 1 < len(chunks) \
            and base_indent(chunks[idx + 1].get("reference") or "") \
            > base_indent(chunk.get("reference") or "") \
            and "unexpected indent" in (res.internal_error or ""):
        result = result.model_copy(update={"student_reason": _LOOP_NOT_STARTED})
    return result


_LOOP_NOT_STARTED = (
    "We could not confirm this step, so your attempt was not used. The next step "
    "carries on inside a repetition (a loop) that this step is meant to start, "
    "and your code does not start one yet. Have this step also start going "
    "through the items - the next step continues from inside it.")


def _model_tiers(problem, session, chunk, header, prefix, student_code, upto,
                 ref_tail, tests, entry, corr=None) -> GradeResult:
    """Tier 3, then tier 4, and the ONE route event for whatever came of them.

    TIER 4 RUNS EVEN WHEN TIER 3 FOUND FAILING CASES. Those cases come from a
    model's rewrite, so they are a strong hint and not proof - and the same
    code always gets the same answer, so a correct step shown a wrong hint
    could only move on by being changed. Tier 4 can overrule it only with
    proof: a program built on their step that passes every test and breaks
    without it. It is NOT shown the cases: told exactly which inputs fail, a
    model is invited to write the patch that hides the mistake. An overrule is
    traced (`overruled`) so each one can be reviewed."""
    idx = session["index"]
    # Identical code, identical verdict - see _VERDICT_MEMO. Everything from
    # here down can consult a model, and this is the last point before that.
    memo_key = _memo_key(problem, idx, upto, student_code)
    if memo_key in _VERDICT_MEMO:
        return _VERDICT_MEMO[memo_key]

    def _remember(result):
        if len(_VERDICT_MEMO) >= _MEMO_LIMIT:
            _VERDICT_MEMO.clear()       # cheap bound; correctness never depends
        _VERDICT_MEMO[memo_key] = result
        return result

    outputs = _step_outputs(student_code)
    adapted = _tier3(problem, session, header, prefix, student_code, upto,
                     ref_tail, tests, entry, outputs, corr)
    if adapted is not None and adapted.verdict == "correct":
        _trace(trace.record_route, corr, "execution-adapted", "correct")
        return _remember(adapted)
    if adapted is not None and adapted.tier == "system":
        _trace(trace.record_route, corr, "system", "indeterminate")
        return adapted                  # ours - not remembered

    completed = _complete(problem, session, chunk, header, prefix, student_code,
                          upto, ref_tail, tests, entry, outputs, corr)
    if completed is not None and completed.verdict == "correct":
        if adapted is not None:
            _trace(trace.record_route, corr, "execution-completed", "correct",
                   overruled="execution-adapted")
            completed = completed.model_copy(update={"internal_detail":
                "overruled tier 3's failing cases: " + " | ".join(
                    (adapted.failing_cases or [])[:2])[:500]})
        else:
            _trace(trace.record_route, corr, "execution-completed", "correct")
        return _remember(completed)
    if completed is not None:
        # Ours (provider down, harness error). Not remembered: the next try
        # should ask again rather than replay an outage. Tier 3's evidence,
        # when there is some, is still the most useful thing to show.
        if adapted is not None:
            _trace(trace.record_route, corr, "execution-adapted", "indeterminate")
            return adapted
        _trace(trace.record_route, corr, "system", "indeterminate")
        return completed
    if adapted is not None:
        _trace(trace.record_route, corr, "execution-adapted", "indeterminate")
        return _remember(adapted)
    _trace(trace.record_route, corr, "unconfirmed", "indeterminate")
    return _remember(_ok(
        "indeterminate", "unconfirmed",
        "We could not confirm this step, so your attempt was not used. Check "
        "what your code produces against what the step asks for.",
        "unconfirmed", deterministic=False, consume_attempt=False))


def _complete(problem, session, chunk, header, prefix, student_code, upto,
              ref_tail, tests, entry, outputs, corr=None):
    """Tier 4 - see the note above MAX_COMPLETE_TRIES. An acceptance, one of
    our own failures, or None when no completion proved anything."""
    idx = session["index"]
    chunks = session["chunks"]
    remaining = [(chunks[j].get("prompt") or "") for j in range(idx + 1, len(chunks))]
    remaining_lines = sum(_lines(chunks[j].get("reference") or "")
                          for j in range(idx + 1, len(chunks)))
    # A loop variable is a counter, not a result of theirs: a completion reusing
    # `value` for its own loop is not overwriting their work. Measured - a real
    # completion passing 10/10 was refused for exactly that.
    protected = outputs - bridge.loop_targets(student_code)
    evidence = _their_values(problem, header, upto, tests, entry)
    # The WHOLE reference body as the example, not just the steps after this
    # one: a name the teacher created in an earlier step (`calcStack = Stack()`)
    # is otherwise invisible, and the completion reads it without creating it.
    reference = problem.get("solution") or ref_tail

    for attempt in range(1, MAX_COMPLETE_TRIES + 1):
        try:
            with trace.model_call(corr, GRADING_MODEL, "completion", attempt=attempt):
                tail = _request_completion(problem, header, upto, student_code,
                                           remaining, reference, evidence,
                                           temperature=0.0 if attempt == 1 else 0.4,
                                           step_prompt=chunk.get("prompt") or "")
            # Models often return the body still indented as it sits in the
            # function; it is spliced in at the right depth by _assemble.
            tail = textwrap.dedent(tail).strip("\n")
        except (ValueError, TypeError, AttributeError):
            _trace(trace.record_adapter, corr, GRADING_MODEL, attempt, "malformed")
            continue
        except Exception as e:
            # The provider, not the student.
            return _provider_down("completion_unavailable", repr(e)[:200])
        if not _tail_is_sane(tail, protected, problem.get("solution", "")):
            _trace(trace.record_adapter, corr, GRADING_MODEL, attempt, "unsafe")
            continue
        # SIZE BUDGET - the completion writes the REMAINING steps, not this
        # one. Measured: a step that only set things up ("split the statements
        # AND evaluate each one", answered with the split alone) was accepted
        # because the completion wrote the whole evaluation loop itself - 30
        # lines standing in for a 2-line remaining step. A different approach
        # may need a few more lines than the teacher's; it does not need the
        # current step's work done for it.
        if _lines(tail) > 2 * remaining_lines + 4:
            _trace(trace.record_adapter, corr, GRADING_MODEL, attempt, "oversized")
            continue
        if _patches_their_values(problem, header, upto, tail, protected, tests, entry):
            _trace(trace.record_adapter, corr, GRADING_MODEL, attempt, "changes_their_values")
            continue
        cand = classify_run(_assemble(problem, header, upto, tail), tests, entry_name=entry)
        if cand.outcome == "harness_error":
            return _system("harness_error", cand.internal_error)
        if cand.outcome != "pass":
            _trace(trace.record_adapter, corr, GRADING_MODEL, attempt, f"failed_{cand.outcome}")
            continue
        # THEIR WORK MUST MATTER - see _their_work_matters.
        if not _their_work_matters(problem, header, prefix, upto, tail,
                                   outputs, tests, entry):
            _trace(trace.record_adapter, corr, GRADING_MODEL, attempt, "bypass_rejected")
            continue
        _trace(trace.record_adapter, corr, GRADING_MODEL, attempt, "accepted")
        return _ok("correct", "execution-completed",
                   "Correct - your step works with the rest of the solution.",
                   "completed_pass", execution_outcome="pass", divergent=True,
                   deterministic=False)
    return None


if __name__ == "__main__":
    # Self-check for the deterministic scope gate. Pure - no oracle, no model.
    # Run with:  python -m main.grading
    params = {"nums"}

    flagged = [
        ("max_element = max(list)", params, "builtin_type_as_value"),
        ("list.pop(list.index(top))", params | {"top"}, "builtin_type_as_value"),
        ("x = dict", params, "builtin_type_as_value"),
        ("total = arr[0]", params, "undefined_name"),
        ("return maxx", params, "undefined_name"),
        ("total += n", params, "undefined_name"),        # n never bound
    ]
    for code, scope, expect_code in flagged:
        got = _scope_violation(code, set(scope))
        assert got is not None and got[1] == expect_code, (code, got)

    # ...and it names the variable in scope, not "the function's parameters".
    # `frequency(txt)` builds `new_dict` and finishes `return dict`: the
    # parameter is txt, and pointing there sends them the wrong way.
    _ret = _scope_violation("return dict", {"txt", "new_dict"})
    assert _ret[1] == "builtin_type_as_value" and "`new_dict`" in _ret[0], _ret
    # No near match, no guess - the generic sentence still names both places.
    _far = _scope_violation("max_element = max(list)", params)
    assert "earlier step" in _far[0] and "`list`" in _far[0], _far

    clean = [
        ("first = max(nums)", params),
        ("return max(nums)", params),
        ("seen = list(map(int, nums))", params),          # list(...) is a call
        ("d = dict()", params),
        ("total = 0", params),
        ("total = total + n\nn = 1", params),             # bound within the step
        ("for i in range(len(nums)):\n    total += nums[i]", params | {"total"}),
        ("out = [x for x in nums if x > 0]", params),
        ("import math\nr = math.sqrt(nums[0])", params),
        ("return sorted(nums)[-2]", params),
    ]
    for code, scope in clean:
        got = _scope_violation(code, set(scope))
        assert got is None, (code, got)

    # A syntax fragment is classified elsewhere, not here.
    assert _scope_violation("elif x:", params) is None

    # ── what the gate must know about a METHOD ───────────────────────────
    # Each of these rejected the teacher's OWN reference answer on chunk 1 of a
    # class problem, which is every problem in HW3.
    assert _header_params("def pop(self):") == {"self"}
    assert _header_params("def push(self, value):") == {"self", "value"}
    assert _header_params("def f(a, b=2, *rest, k=1, **kw):") == \
        {"a", "b", "rest", "k", "kw"}
    assert _header_params("") == set() and _header_params("not a def") == set()

    # `self` is the method's, never the injected driver's ["calls"].
    meth = {"self"}
    assert _scope_violation("if self.top is None:\n    return None", meth) is None
    # ...and a typo of it is still the student's error.
    assert _scope_violation("if slef.top is None:\n    return None",
                            meth)[1] == "undefined_name"

    # ── A FUNCTION MAY CALL ITSELF ───────────────────────────────────────
    # Every recursive submission was convicted: build_program for a plain
    # function is the def line plus the body, and _module_names was calling it
    # WITHOUT the def line, so the assembled module defined nothing and the
    # function's own name was not in scope. `return n * factorial(n - 1)` came
    # back "`factorial` isn't defined".
    _fact_h = "def factorial(n):"
    _fact_p = {"slug": "fact", "entry_hint": "factorial",
               "solution": _fact_h + "\n    return 1"}
    assert _module_names(_fact_p, _fact_h) == {"factorial"}, \
        _module_names(_fact_p, _fact_h)
    _fact_scope = {"n"} | _module_names(_fact_p, _fact_h)
    assert _scope_violation("return n * factorial(n - 1)", _fact_scope) is None
    # ...and a misspelling of it is still the student's error.
    assert _scope_violation("return factorail(n - 1)",
                            _fact_scope)[1] == "undefined_name"

    # A builtin exception is not an undefined name. Calculator._isNumber is
    # try/float/except and was failed for naming ValueError.
    assert _scope_violation("try:\n    float(t)\nexcept ValueError:\n    pass",
                            {"t"}) is None

    # A class the teacher GAVE the student resolves: Stack.push opens with
    # `node = Node(value)`, and Calculator._getPostfix constructs a Stack().
    _mod = {"slug": "t", "description": "d", "entry_hint": "push",
            "group_title": "Stack",
            "context_prefix": "class Node:\n    pass\n\n\nclass Stack:\n"
                              "    def push(self, value):\n",
            "context_suffix": "", "context_indent": 8,
            "solution": "pass"}
    assert "Node" in _module_names(_mod) and "Stack" in _module_names(_mod)
    assert _module_names({"slug": "x", "solution": "def f():\n    pass"}) is not None
    assert _scope_violation("node = Node(value)",
                            {"self", "value"} | _module_names(_mod)) is None

    # ── indentation is named as indentation, not as a typo ───────────────
    # Each of these used to come back "Your code doesn't parse: ...", which
    # sends a student looking for a misspelling through correctly spelled code.
    for code, needle in (
            ("for x in nums:\ntotal += x", "has to sit further in"),
            ("total = 0\n    total += 1", "nothing above it opens a block"),
            ("if a:\n    b = 1\n  c = 2", "between two levels")):
        said = _indent_message(code)
        assert said and needle in said, (code, said)
        assert "line" in said, said            # it must point AT a line
        # ...and it must never imply the outer depth was theirs to get right:
        # align_submission has already re-seated it.
        assert "added for you" in said, said
    # The whole block sitting at the wrong column is NOT a fault - main/indent.py
    # re-seats it - and an ordinary syntax error belongs to _syntax_message.
    assert _indent_message("        total = 0\n        total += 1") is None
    assert _indent_message("return max(nums") is None
    assert _indent_message("") is None

    # ── failing cases: capped, counted honestly, and never fatal ─────────
    # Rendering a case resolves the problem's entry point, and main/identity.py
    # PERSISTS that resolution - so running this self-check would otherwise file
    # two fixture problems in resolved_entries.json, which the repo tracks. A
    # throwaway path keeps the check from editing the project it is checking.
    import os
    import tempfile

    from main import identity as _identity
    _identity._RESOLVED_PATH = os.path.join(tempfile.mkdtemp(), "resolved.json")

    _p = {"slug": "t", "entry_hint": "f", "solution": "def f(n):\n    return n"}
    _tests = [{"input": [i], "expected": i} for i in range(6)]
    _fails = [{"index": i, "got": {"repr": "0", "type": "int"},
               "expected": {"repr": str(i), "type": "int"}} for i in range(5)]
    shown = failing_cases(_p, _tests, _fails)
    assert len(shown) == MAX_SHOWN_CASES, shown
    assert all("expected:" in c and "you gave:" in c for c in shown), shown
    assert failing_cases(_p, _tests, _fails, limit=1) == shown[:1]
    assert failing_cases(_p, _tests, []) == []
    # A failure the renderer cannot place is SKIPPED, so the list can come back
    # empty even though the run reported failures. That is what made the
    # adapted tier promise "the case below" over nothing at all.
    assert failing_cases(_p, _tests, [{"index": None, "error": "boom"}]) == []
    assert failing_cases(_p, _tests, [{"index": 99, "error": "boom"}]) == []

    # An index the suite does not have is skipped, not raised on, and must not
    # cost the student the cases that WOULD have told them something.
    assert len(failing_cases(_p, _tests, [{"index": 99}] + _fails)) == MAX_SHOWN_CASES

    # The count is the run's, never the length of the sample: execution.py caps
    # what comes back, so "3 shown" must not be reported as "3 wrong".
    class _R:
        total, passed = 12, 5
    assert _failed_total(_R(), 3) == 7
    # ...and with no totals to read, what was shown is the only honest floor.
    class _None:
        total, passed = 0, 0
    assert _failed_total(_None(), 3) == 3

    # ── the deferred initializer hoist ──────────────────────────────────
    # The shape of Calculator._getPostfix: the reference declares its stack and
    # its precedence table in chunk 1 and reads neither until chunk 3, so a
    # student who declares them in chunk 3 - which is where they are actually
    # used - handed the trusted tail a NameError and a coin flip between tiers.
    _hdr = "def to_postfix(tokens):"
    _refs = [
        'terms = [t.strip() for t in tokens]\n'
        'stack = []\n'
        'prec = {"+": 1, "-": 1, "*": 2, "/": 2}',

        'out = []\n'
        'for t in terms:\n'
        '    if t not in prec:\n'
        '        out.append(t)\n'
        '    else:\n'
        '        while stack and prec[stack[-1]] >= prec[t]:\n'
        '            out.append(stack.pop())\n'
        '        stack.append(t)',

        'while stack:\n'
        '    out.append(stack.pop())\n'
        'return out',
    ]
    _sess = {"slug": "postfix", "title": "Postfix", "description": "shunting-yard",
             "solution": _hdr + "\n" + _indent("\n".join(_refs)),
             "header": _hdr, "index": 0, "accepted": [],
             "chunks": [{"prompt": f"step {i}", "reference": r}
                        for i, r in enumerate(_refs, 1)]}
    _postfix_tests = [
        {"input": [["3", "+", "4", "*", "2"]], "expected": ["3", "4", "2", "*", "+"]},
        {"input": [["8", "/", "2", "/", "2"]], "expected": ["8", "2", "/", "2", "/"]},
    ]
    _deferred = "terms = [t.strip() for t in tokens]"   # stack/prec left to step 3
    _tail = "\n".join(_refs[1:])
    _prob = problem_of(_sess)

    # Exactly the two names the tail cannot supply itself, in the reference's
    # own order. The tail's own locals - t, out - are bound by the tail and must
    # never show up here.
    assert _hoistable_declarations(_prob, _sess["chunks"], _hdr, _deferred, _tail) \
        == ["stack = []", "prec = {'+': 1, '-': 1, '*': 2, '/': 2}"]
    # A name the student bound HERSELF is hers, whatever it holds. A genuinely
    # different interface is Tier 3's to adapt, never this gate's to paper over.
    assert _hoistable_declarations(_prob, _sess["chunks"], _hdr,
                                   _deferred + "\nstack = 0\nprec = {}", _tail) == []
    # NEGATIVE, and the more important half: `size` is computed from another
    # local, so that line does not mean the same thing anywhere else. One name
    # that cannot be settled disqualifies the submission - nothing is hoisted.
    # `size` is deliberately NOT len(terms). An earlier version of this fixture
    # used `size = len(terms)`, which makes `out` a copy of `terms` on every
    # input - so the value bridge correctly matched them and the fixture stopped
    # testing the fall-through it exists for. Dropping the last token keeps the
    # two genuinely different.
    _neg_refs = ["terms = [t.strip() for t in tokens]\nsize = len(terms) - 1",
                 "out = terms[:size]",
                 "return out"]
    _neg = {**_sess, "solution": _hdr + "\n" + _indent("\n".join(_neg_refs)),
            "chunks": [{"prompt": f"step {i}", "reference": r}
                       for i, r in enumerate(_neg_refs, 1)]}
    assert _hoistable_declarations(problem_of(_neg), _neg["chunks"], _hdr,
                                   _deferred, "\n".join(_neg_refs[1:])) == []

    # End to end, against real runs. A model call is a bug on BOTH paths: the
    # hoist must never need one, and the negative fixture must not be quietly
    # rescued by one either - it has to take the same road it takes today.
    def _no_model(*a, **k):
        raise AssertionError("no model may be consulted here")

    _request_completion, chat = _no_model, _no_model

    _graded = grade_submission(_sess, _deferred, oracle_loader=lambda p: _postfix_tests)
    assert _graded.verdict == "correct", (_graded.verdict, _graded.student_reason)
    assert _graded.tier == "execution-reference", _graded.tier
    assert _graded.deterministic is True and _graded.execution_outcome == "pass"
    assert _graded.reason_code == "reference_pass_hoisted", _graded.reason_code
    # Deterministic means deterministic: identical code, identical verdict,
    # every time. That was the whole complaint - 24 identical submissions, 12
    # of them decided by a judge because the adapter model wavered.
    for _ in range(3):
        assert grade_submission(_sess, _deferred,
                                oracle_loader=lambda p: _postfix_tests) == _graded

    # ...and the un-hoistable fixture still falls through to the model tiers
    # untouched, where the stubs above make it land on the outage path.
    _fell = grade_submission(_neg, _deferred,
                             oracle_loader=lambda p: [{"input": [[" 3 ", "+"]],
                                                       "expected": ["3"]},
                                                      {"input": [["a", "b", "c"]],
                                                       "expected": ["a", "b"]}])
    assert _fell.reason_code == "completion_unavailable", _fell.reason_code

    # ── the value bridge, end to end, with no model reachable ────────────
    # THE SUBMISSION THIS WHOLE TIER EXISTS FOR. The teacher wrote `counts`,
    # the student wrote `new_dict`, and identical code came back correct at 1am
    # and incorrect twice at 4:58am because an LLM was being asked to rewrite
    # the teacher's tail and managed it about half the time. _no_model is still
    # installed above, so a model call anywhere on this path fails the check.
    _fh = "def frequency(txt):"
    _frefs = ["counts = {}",
              "for ch in txt:\n    if ch.isalpha():\n"
              "        counts[ch] = counts.get(ch, 0) + 1",
              "return counts"]
    _fsess = {"slug": "frequency", "title": "f", "description": "Count letters.",
              "solution": _fh + "\n" + _indent("\n".join(_frefs)), "header": _fh,
              "index": 0, "accepted": [],
              "chunks": [{"step_id": f"Part {i + 1}", "prompt": f"step {i + 1}",
                          "reference": r} for i, r in enumerate(_frefs)]}
    _ftests = [{"input": ["hello"], "expected": {"h": 1, "e": 1, "l": 2, "o": 1}},
               {"input": ["aab"], "expected": {"a": 2, "b": 1}},
               {"input": [""], "expected": {}}]
    _fl = lambda p: _ftests

    _b = grade_submission(_fsess, "new_dict = {}", oracle_loader=_fl)
    assert _b.verdict == "correct", (_b.verdict, _b.student_reason)
    assert _b.tier == "execution-bridged" and _b.deterministic is True, _b.tier
    assert _b.covers_chunks == 1, _b.covers_chunks
    # Deterministic means deterministic. This is the assertion the bug report was.
    for _ in range(5):
        assert grade_submission(_fsess, "new_dict = {}", oracle_loader=_fl) == _b

    # AHEAD: step 2's work done inside step 1 is accepted for BOTH steps, so the
    # student is never asked to write the same loop a second time.
    _a = grade_submission(_fsess, "out = {}\nfor ch in txt:\n    if ch.isalpha():"
                                  "\n        out[ch] = out.get(ch, 0) + 1",
                          oracle_loader=_fl)
    assert _a.verdict == "correct" and _a.covers_chunks == 2, _a

    # The teacher's own name needs no bridge and must not take this path.
    assert grade_submission(_fsess, "counts = {}",
                            oracle_loader=_fl).tier == "execution-reference"

    # A WRONG submission finds no bridge - and that is never a conviction. With
    # the model stubbed out it lands on the outage path; what matters is that it
    # is not `incorrect` and costs no attempt.
    _w = grade_submission(_fsess, "new_dict = []", oracle_loader=_fl)
    assert _w.verdict != "incorrect", _w.verdict
    assert _w.consume_attempt is False, "a bridge that is not found costs nothing"

    # ── A FINISHED SOLUTION ENDS THE PROBLEM, WHATEVER STEP IT ARRIVES AT ──
    # Live, a student wrote the whole of `frequency` into step 1, was told
    # "correct", and was then asked for step 2 - which their own code already
    # contained. Answering `pass` there produced "Solved. Nice work."
    _whole = ("from collections import Counter\n"
              "q = dict(Counter(c for c in txt.lower() if c.isalpha()))\n"
              "return q")
    _w = grade_submission(_fsess, _whole, oracle_loader=_fl)
    assert _w.verdict == "correct" and _w.tier == "execution-final", _w
    assert _w.reason_code == "final_pass_early", _w.reason_code
    assert _w.covers_chunks == len(_frefs), _w.covers_chunks
    # Nothing is borrowed: the same student-only oracle the LAST chunk runs.
    # So a partial answer cannot trip it, however correct that answer is...
    assert grade_submission(_fsess, _frefs[0],
                            oracle_loader=_fl).tier == "execution-reference"
    # ...and neither `return {}` nor a helper that is never called may COMPLETE
    # the problem. Note what is NOT asserted: that they are rejected. In this
    # fixture chunk 0 is `counts = {}`, so a student who writes `q = {}` beside
    # a dead helper really has answered it, and the bridge accepting that is
    # correct - dead code is not a fault. The claim here is only that neither
    # ends the problem early.
    for _partial in ("return {}", "def helper(s):\n    return {}\nq = {}"):
        _p = grade_submission(_fsess, _partial, oracle_loader=_fl)
        assert _p.reason_code != "final_pass_early", (_partial, _p.reason_code)
        assert _p.verdict != "incorrect", "an unfinished step is not a conviction"

    # ── TIER 3: A COMPLETION ACQUITS ONLY IF EXECUTION AGREES ────────────
    # The model proposes the rest of the function; the oracle and the blank-out
    # check decide. No network - the proposal is stubbed.
    _real_req = _request_completion
    _csess = {"slug": "freq3", "title": "f", "description": "Count letters.",
              "solution": "def f(txt):\n    counts = {}\n    for ch in txt:\n"
                          "        counts[ch] = counts.get(ch, 0) + 1\n    return counts",
              "header": "def f(txt):", "index": 0, "accepted": [],
              "chunks": [{"prompt": "tally", "reference": "counts = {}\nfor ch in txt:\n"
                          "    counts[ch] = counts.get(ch, 0) + 1"},
                         {"prompt": "return it", "reference": "return counts"}]}
    _ct = [{"input": ["aab"], "expected": {"a": 2, "b": 1}},
           {"input": [""], "expected": {}}]
    # A DIFFERENT SHAPE (a sorted list, not a dict of counts), so no renaming
    # can match it to the teacher's names and it genuinely reaches this tier.
    _mine = "letters = sorted(txt)"
    try:
        # A completion that USES their work: acquitted, by execution.
        _request_completion = lambda *a, **k: "return {c: letters.count(c) for c in letters}"
        _VERDICT_MEMO.clear()
        _v = grade_submission(_csess, _mine, oracle_loader=lambda p: _ct)
        assert (_v.verdict, _v.tier) == ("correct", "execution-completed"), _v
        # A completion that REDOES the work ignores their values: the blank-out
        # check sees it still passing and refuses it.
        _request_completion = lambda *a, **k: ("out = {}\nfor c in txt:\n"
                                               "    out[c] = out.get(c, 0) + 1\nreturn out")
        _VERDICT_MEMO.clear()
        _v = grade_submission(_csess, _mine, oracle_loader=lambda p: _ct)
        assert (_v.verdict, _v.tier) == ("indeterminate", "unconfirmed"), _v
        assert _v.consume_attempt is False, "not confirming costs no attempt"
        # A completion that REBUILDS their variable is refused before it runs.
        _request_completion = lambda *a, **k: ("letters = list(txt)\n"
                                               "return {c: letters.count(c) for c in letters}")
        _VERDICT_MEMO.clear()
        _v = grade_submission(_csess, _mine, oracle_loader=lambda p: _ct)
        assert _v.tier == "unconfirmed", _v
    finally:
        _request_completion = _real_req
        _VERDICT_MEMO.clear()

    # ── THEY TYPED THE def LINE AGAIN ───────────────────────────────────
    # Reported by a real student on _isNumber: the frozen first line already
    # says `def _isNumber(self, txt):` and the box below it is the BODY, but
    # writing a whole function is the habit, so they wrote the def again and
    # indented their work under it. Nothing ran, every deterministic tier let it
    # through, and the LLM judges - which cannot convict - answered "we could
    # not confirm this step". Their logic was RIGHT: the same body without the
    # def line grades correct at execution-reference.
    _H = "def _isNumber(self, txt):"
    _caught = [
        ("def _isNumber(txt):\n    return True", _H),
        # ...even nested inside a block, which is how it looks once indented.
        ("if txt:\n    def _isNumber(t):\n        return True", _H),
        ("def frequency(txt):\n    return {}", "def frequency(txt):"),
    ]
    for code, header in _caught:
        assert _redefines_enclosing(code, header) is not None, code

    # ONLY an exact redefinition of the enclosing function. Everything here is
    # ordinary Python and must pass untouched - especially RECURSION, which
    # calls the name but never redefines it, and which this grader has
    # false-convicted before by a different route.
    _clean = [
        ("if n <= 1:\n    return 1\nreturn factorial(n - 1) * n", "def factorial(n):"),
        ("def _digits(s):\n    return s.isdigit()\nanswer = _digits(txt)", _H),
        ("answer = txt.strip().isdigit()", _H),
        ("f = lambda x: x.isdigit()\nanswer = f(txt)", _H),
        ("", _H),
        ("def _isNumber(txt:", _H),        # unparseable: _syntax_message's job
        ("def _isNumber(txt):\n    return True", ""),   # no def line to repeat
    ]
    for code, header in _clean:
        assert _redefines_enclosing(code, header) is None, code

    # ── COMMENTS ARE NOT CODE ───────────────────────────────────────────
    # The same dead end as the def-line bug, from the other direction: text
    # that is not blank but runs nothing, so no tier can attribute anything and
    # the judges - which cannot convict - answer "we could not confirm this".
    for _code in ("# work out whether it is a number\n# then save the answer",
                  "   \n  \n"):
        assert _has_no_statements(_code), repr(_code)
    # Anything that IS a statement belongs to the tiers below, including ones
    # that happen to do nothing at runtime - those are not empty answers.
    for _code in ("# decide\nis_number = txt.isdigit()",
                  '\'\'\'decide if it is a number\'\'\'',
                  "pass",
                  "if False:\n    is_number = True",
                  "is_number = True",
                  "if x"):                      # unparseable: _syntax_message's
        assert not _has_no_statements(_code), repr(_code)

    print("grading.py scope-gate self-check OK")
