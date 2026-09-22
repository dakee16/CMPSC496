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
    execution-adapted    a model rewrites the tail; calibration, the full oracle
                         and a neutralised knockout must all still pass
    llm-judge            two independent judges both say it is right

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

from . import bridge
from .execution import classify_run
from .identity import get_resolved_entry
from .indent import align_to_chunk
from .ollama_client import GRADING_MODEL, chat
from . import trace
from .schemas import GradeResult
from .context import build_program
from .sessions import accepted_prefix, problem_of

# Retries of the Tier 3 adapter. Raised from 2 once that tier became
# acquit-only, because the two situations are not the same kind of thing.
# Re-running a JUDGE until it agrees with itself manufactures confidence: the
# output is an opinion nothing checks, so repetition converges on the model's
# favourite answer (at a 70% bias, an agreed verdict is wrong 84% of the time).
# Re-running the ADAPTER is a SEARCH: every candidate must still pass
# calibration, the full oracle and the anti-bypass check before it can acquit,
# so a bad one is discarded by execution rather than believed. Failure now
# costs nothing but latency, which makes another look worth taking.
MAX_ADAPT_TRIES = 4

# Verdict memo, so identical code gets an identical verdict. Tier 3 and Tier 4
# are model calls, and a model that wavers turns one student's answer into
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
    lines = []
    if is_method(problem) and inp and isinstance(inp[0], str):
        lines.append(inp[0].rstrip())          # a block test IS a program
    elif is_method(problem) and inp and isinstance(inp[0], list):
        cls = problem.get("group_title") or "Solution"
        lines.append(f"x = {cls}()")
        for call in inp[0]:
            if not (isinstance(call, list) and call):
                continue
            name, args = str(call[0]), call[1:]
            rendered = ", ".join(repr(a) for a in args)
            if name == "new":
                lines[0] = f"x = {cls}({rendered})"
            elif name in ("len", "str", "bool"):
                lines.append(f"{name}(x)")
            else:
                lines.append(f"x.{name}({rendered})")
    else:
        name = get_resolved_entry(problem)["entry_name"] or "solution"
        lines.append(f"{name}({', '.join(repr(a) for a in inp)})")

    out = ["\n".join(lines)]
    if failure.get("error"):
        out.append(f"\nit raised: {failure['error']}")
    else:
        out.append(f"\nexpected: {_shown(failure.get('expected'))}")
        out.append(f"you gave: {_shown(failure.get('got'))}")
    return "\n".join(out)


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
    return align_to_chunk(student_code, chunks[idx])


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
    ({"target": "postfixStack", "source": "postfixStack"}) that _valid_aliases
    rightly refused - so WHICH TIER decided a correct answer was a coin flip on
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


# ── Tier 3: calibrated adaptation ────────────────────────────────────────

_ADAPT_SYSTEM = (
    "You complete a partially written Python function. Return STRICT JSON only: "
    '{"adapted_tail": "<remaining body lines>", "aliases": [{"target": "n", "source": "m"}]}. '
    "adapted_tail is body code only - no def line, no imports, no markdown. "
    "aliases map a name the tail needs (target) to a name the earlier code already "
    "produced (source). Identifiers only, no expressions.")


def _request_adaptation(problem, header, upto, reference_tail, student_outputs):
    """Ask for a tail that builds on the student's interface. Isolated so tests
    can substitute it without a network call."""
    user = (f"PROBLEM:\n{(problem.get('description') or '')[:600]}\n\n"
            f"Function header: {header}\n\n"
            f"Code so far (the student's own approach):\n{upto}\n\n"
            f"The remaining logic was originally written as:\n{reference_tail}\n\n"
            f"Names the student's code produced: {sorted(student_outputs) or 'none'}\n\n"
            "Rewrite the remaining logic so it builds on the student's names. "
            "Do not restate their work and do not recompute the answer from scratch.")
    raw = chat(GRADING_MODEL, _ADAPT_SYSTEM, [{"role": "user", "content": user}],
               temperature=0, fmt="json")
    data = json.loads(raw)
    return data.get("adapted_tail", ""), data.get("aliases", []) or []


def _valid_aliases(aliases, allowed_targets, allowed_sources):
    """Only `target = source`, both plain identifiers, both in scope."""
    out = []
    for a in aliases:
        t, s = (a or {}).get("target", ""), (a or {}).get("source", "")
        if not (isinstance(t, str) and isinstance(s, str)):
            return None
        if not (t.isidentifier() and s.isidentifier()):
            return None
        if allowed_targets and t not in allowed_targets:
            return None
        if allowed_sources and s not in allowed_sources:
            return None
        out.append(f"{t} = {s}")
    return out


def _tail_is_sane(tail: str, current_outputs: set, solution: str) -> bool:
    """Structural checks before the tail is allowed anywhere near a verdict."""
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
    if (_names(tail, ast.Store) - bridge._comprehension_vars(tree)) & current_outputs:
        return False
    # Must not simply be the reference solution pasted back in.
    body = re.sub(r"\s+", "", tail)
    if body and body in re.sub(r"\s+", "", solution or ""):
        return False
    return True


def _calibrate(problem, header, trusted_prefix, alias_lines, tail, tests, entry):
    """An adapter must prove itself on TRUSTED work before it may judge a
    student. A random LLM tail that fails proves nothing about the student:
    it may simply be a broken tail. Only a tail that passes here has earned
    the right to produce a verdict."""
    cand = _assemble(problem, header, trusted_prefix, "\n".join(alias_lines), tail)
    return classify_run(cand, tests, entry_name=entry).outcome == "pass"


# ── Tier 4: dual LLM judge ───────────────────────────────────────────────

_JUDGE_SYSTEM = (
    "You judge ONE step of a student's partial solution. Return STRICT JSON: "
    '{"correct": true/false, "reason": "<one sentence for the student>", '
    '"failing_input": "<required when correct is false: a concrete input value '
    'on which this code produces a different RESULT than the step requires, or '
    'empty string if you cannot name one>", '
    '"confidence": 0.0-1.0, "evidence_category": "<short label>"}. '
    # THE ONLY GROUND FOR CONVICTION IS A DIFFERENT RESULT. The judge is handed
    # the reference, so it drifts into marking any deviation from it wrong:
    # `txt.replace(" ", "")` before an isalpha() check was failed live as
    # "unnecessary" - redundant, yes, and identical in output, so the student
    # was told correct code was incorrect. Style, efficiency, redundancy,
    # naming and structure are not this judge's business, and a rule saying so
    # is only half of it: failing_input is the half that has teeth, because
    # "unnecessary" cannot name an input where the answer differs. See
    # _tier4 for what happens when the field comes back empty.
    "THE ONLY REASON TO ANSWER correct=false IS THAT THE CODE PRODUCES A "
    "DIFFERENT RESULT than this step requires. Before answering false, name "
    "the input in failing_input and satisfy yourself that the student's code "
    "really does produce something different on it. If their code reaches the "
    "same result by a longer, redundant, slower, differently-named or "
    "differently-shaped route than the reference, that is CORRECT - say so. "
    "Extra work that changes nothing is not an error. Differing from the "
    "reference is not an error. Only a different answer is an error. "
    "Never quote the reference solution, hidden tests, or internal code in reason. "
    # ...AND NEVER DESCRIBE IT EITHER. The rule above says "quote", and the
    # model complied with it exactly: asked about `counts = []` it answered
    # "it should be a dictionary to map letters to their counts", which quotes
    # nothing and hands over the container AND what it maps - point 1 of the
    # plan rubric, the thing the tutor holds a student at the design gate to
    # work out for themselves. Saying it in prose is the same disclosure.
    "Do not describe the correct approach either - naming the data structure, "
    "the algorithm, or what the student should have written instead is the "
    "same disclosure as quoting it. Say what their code DOES and where it "
    "stops matching the step as asked; never say what it should do.")


# The answer-leak redaction that used to live here is gone with the verdict it
# guarded. It rewrote a judge's sentence when an INCORRECT explanation named the
# data structure the student was being held at the design gate to work out for
# themselves. A judge can no longer return incorrect, and its own docstring made
# the point that a CORRECT verdict cannot leak - it describes code the student
# has already written. The tutor keeps its own guard (tutor._handed_over), which
# is where that logic belongs and where it is still exercised.


def _ask_judge(payload: str, role: str):
    raw = chat(GRADING_MODEL, _JUDGE_SYSTEM + f"\nYou are the {role}.",
               [{"role": "user", "content": payload}], temperature=0, fmt="json")
    d = json.loads(raw)
    return (bool(d["correct"]), str(d.get("reason", ""))[:300],
            float(d.get("confidence", 0.0)), str(d.get("evidence_category", ""))[:60],
            str(d.get("failing_input", ""))[:200].strip())


def _tier4(problem, chunk, upto, student_code, why, evidence, corr=None) -> GradeResult:
    payload = (f"PROBLEM:\n{(problem.get('description') or '')[:600]}\n\n"
               f"STEP ASKED:\n{chunk['prompt']}\n\n"
               f"ACCEPTED SO FAR:\n{upto}\n\n"
               f"STUDENT'S ANSWER FOR THIS STEP:\n{student_code}\n\n"
               f"PRIVATE REFERENCE FOR THIS STEP:\n{chunk.get('reference','')}\n\n"
               f"EXECUTION EVIDENCE: {evidence}\nWHY EXECUTION WAS INCONCLUSIVE: {why}")
    try:
        with trace.model_call(corr, GRADING_MODEL, "judge", role="primary"):
            a_ok, a_reason, a_conf, a_cat, a_input = _ask_judge(payload, "primary judge")
        _trace(trace.record_judge, corr, GRADING_MODEL, "primary", a_ok, a_conf)
        # THE SECOND JUDGE IS NOT SHOWN THE FIRST ONE'S ANSWER. It used to be:
        # the payload carried the first judge's verdict and reason appended to
        # it, which anchors the second on the very answer it is supposed to
        # check independently. Two anchored samples agreeing is close to no
        # evidence at all, and their agreement was the whole basis for acting.
        # Same payload, different role, no cross-talk. The self-check at the
        # bottom of this file guards against it being reintroduced.
        with trace.model_call(corr, GRADING_MODEL, "judge", role="verifier"):
            b_ok, b_reason, b_conf, b_cat, b_input = _ask_judge(
                payload, "independent verifier")
        _trace(trace.record_judge, corr, GRADING_MODEL, "verifier", b_ok, b_conf)
    except Exception as e:
        # This try wraps ONLY the two model calls, so anything landing here is
        # a provider failure - unreachable, timed out, or malformed output.
        _trace(trace.record_route, corr, "system", "indeterminate")
        return _provider_down("judge_unavailable", repr(e)[:200])

    if a_ok != b_ok or min(a_conf, b_conf) < 0.6:
        _trace(trace.record_route, corr, "llm-judge", "indeterminate")
        return _ok("indeterminate", "llm-judge",
                   "This one needs a closer look - we couldn't decide "
                   "confidently, so your attempt was not used.",
                   "judge_disagreement", deterministic=False,
                   consume_attempt=False,
                   internal_detail=f"a={a_ok}/{a_conf} b={b_ok}/{b_conf}")

    # A JUDGE MAY ACQUIT, NEVER CONVICT.
    #
    # This is the rule the 1am/4:58am flip came down to. Everything above this
    # line is an opinion: no run attributed anything to the student, and the
    # tier exists precisely because execution could not. An opinion that costs a
    # student an attempt is the one thing this module is not allowed to do, and
    # the failure is not hypothetical - correct-but-redundant code was failed
    # live as "unnecessary" by judges that agreed with each other.
    #
    # Requiring a nameable failing_input (below) narrowed that, but it is still
    # the model deciding what counts as a failing input. So a `false` here now
    # ends the same way an "I don't know" does: indeterminate, no attempt spent,
    # and the student is asked to look again. What is LOST is the ability to
    # tell a student their non-final step is wrong on a model's say-so, which
    # was never a thing we could do soundly. The final chunk still runs the real
    # oracle and still convicts on it.
    if not a_ok:
        _trace(trace.record_route, corr, "llm-judge", "indeterminate")
        return _ok("indeterminate", "llm-judge",
                   "We could not confirm this step. Your attempt was not used - "
                   "check it against your plan and try again.",
                   "judge_no_acquittal", deterministic=False,
                   consume_attempt=False,
                   internal_detail=f"agreed incorrect a={a_input!r} b={b_input!r}")

    # AN "INCORRECT" THAT CANNOT NAME A FAILING INPUT IS NOT A CONVICTION.
    # Execution already failed to decide this submission - that is why we are
    # here - so the judge's sentence is all the evidence there is, and a
    # sentence like "this removes spaces first, which is unnecessary" is an
    # observation about style wearing a verdict's clothes. Requiring a concrete
    # input is what separates the two: code that is merely redundant has none
    # to give, because there is no input on which it answers differently.
    # Indeterminate rather than correct - we have not shown them right either -
    # and it costs no attempt, in keeping with this module's rule that our own
    # inability to decide is never evidence about the student.
    # Both judges, independently, said this step is right. That is an ACQUITTAL
    # and nothing more: it is still flagged deterministic=False, because no run
    # attributed anything, and the final chunk will check the whole solution
    # against the real oracle regardless.
    _trace(trace.record_route, corr, "llm-judge", "correct")
    return _ok("correct", "llm-judge", a_reason,
               f"judge_{a_cat or 'agreed'}", deterministic=False)


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

    # ── COMMENTS ARE NOT CODE. Same dead end as the gate below, from the other
    #    direction: text that is not blank but runs nothing. ──
    if _has_no_statements(student_code):
        return _ok("incorrect", "syntax",
                   "There is no code in this answer - comments and blank lines "
                   "are not run, so there is nothing here to answer the step yet.",
                   "comments_only")

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
        shown = failing_cases(problem, tests, res.failures)
        return _ok("incorrect", "execution-final",
                   msg.get(res.outcome, "Your solution didn't pass."),
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
    bridged = bridge.find(problem, header, chunks, idx, upto, tests, entry,
                          set(resolved["params"]) | _header_params(header)
                          | _module_names(problem, header) | _SAFE_BUILTINS)
    if bridged:
        covers = bridged["boundary"] - idx + 1
        _trace(trace.record_route, corr, "execution-bridged", "correct",
               mapping=bridged["mapping"], boundary=bridged["boundary"])
        return _ok("correct", "execution-bridged",
                   "Correct - you named things differently to our version, and "
                   "your step works with the rest of the solution."
                   if covers == 1 else
                   f"Correct - and you have already written what the next "
                   f"{covers - 1} step(s) asked for, so we have marked those "
                   f"done too.",
                   "bridged_pass", execution_outcome="pass",
                   divergent=True, covers_chunks=covers)

    result = _tier3(problem, session, chunk, header, prefix, student_code, upto,
                    ref_tail, tests, entry, res, corr)
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
    return result


def _tier3(problem, session, chunk, header, prefix, student_code, upto,
           ref_tail, tests, entry, ref_res, corr=None) -> GradeResult:
    """Adapt the tail to the student's interface - but only a CALIBRATED
    adapter may influence a verdict."""
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

    trusted_prefix = "\n".join((session["chunks"][j].get("reference") or "")
                               for j in range(idx + 1))
    current_outputs = _names(student_code, ast.Store)
    prefix_names = _names(upto, ast.Store) | set(
        get_resolved_entry(problem)["params"])
    trusted_names = _names(trusted_prefix, ast.Store) | prefix_names
    evidence = f"reference-tail {ref_res.outcome} ({ref_res.passed}/{ref_res.total})"

    for attempt in range(1, MAX_ADAPT_TRIES + 1):
        try:
            with trace.model_call(corr, GRADING_MODEL, "adapter", attempt=attempt):
                tail, aliases = _request_adaptation(
                    problem, header, upto, ref_tail, current_outputs)   # noqa: E501
        except Exception:
            _trace(trace.record_adapter, corr, GRADING_MODEL, attempt, "malformed")
            break                                   # model trouble -> Tier 4
        alias_lines = _valid_aliases(aliases, current_outputs | prefix_names,
                                     trusted_names)
        if alias_lines is None or not _tail_is_sane(
                tail, current_outputs, problem.get("solution", "")):
            _trace(trace.record_adapter, corr, GRADING_MODEL, attempt, "unsafe")
            continue

        # CALIBRATION - prove the adapter on trusted work first.
        if not _calibrate(problem, header, trusted_prefix, alias_lines, tail,
                          tests, entry):
            _trace(trace.record_adapter, corr, GRADING_MODEL, attempt, "calibration_failed")
            continue                                # uncalibrated: prove nothing

        # The alias bridge belongs to CALIBRATION only. It maps the trusted
        # reference's names onto the tail's interface; the student already
        # produces those names, so injecting it here would assign from an
        # undefined reference name and raise NameError on every run.
        cand = classify_run(_assemble(problem, header, upto, tail), tests, entry_name=entry)
        if cand.outcome == "pass":
            # ANTI-BYPASS - blank out what the student's chunk PRODUCED, keeping
            # the names, and require the composite to break. Deleting the chunk
            # instead is foolable: the tail then reads a name that no longer
            # exists, raises NameError, and "it broke without them" reads as
            # necessity even when the tail was doing all the work. Rebinding
            # each name to an empty value of its own type removes the VALUES
            # while leaving the interface intact, so only a tail that genuinely
            # used their work survives.
            #
            # Unlike the bridge below it, an adapted tail is model-written code
            # that CAN compute, so this check still earns its keep here.
            neutral = "\n".join(f"{n} = type({n})()" for n in sorted(current_outputs))
            ko = classify_run(
                _assemble(problem, header, upto, neutral, tail),
                tests, entry_name=entry)
            if ko.outcome == "pass":
                _trace(trace.record_adapter, corr, GRADING_MODEL, attempt, "bypass_rejected")
                continue                            # bypassing adapter: reject
            _trace(trace.record_adapter, corr, GRADING_MODEL, attempt, "accepted")
            _trace(trace.record_route, corr, "execution-adapted", "correct")
            return _remember(_ok(
                "correct", "execution-adapted",
                "Correct - your approach differs from ours, but it works.",
                "adapted_pass", execution_outcome="pass", divergent=True))
        if cand.outcome == "harness_error":
            return _system("harness_error", cand.internal_error)
        if cand.outcome == "wrong_output":
            # THE EVIDENCE WITHOUT THE VERDICT. A calibrated tail ran cleanly on
            # their work and the finished solution came out wrong somewhere.
            # That is worth SHOWING and not worth CONVICTING on: the cases are
            # concrete and checkable by hand, while the claim that the fault is
            # theirs rests on a model having re-expressed the tail faithfully,
            # which calibration does not establish.
            #
            # Without this a wrong answer got the bare "we could not confirm
            # this step", which is the worst of both - no verdict AND no way
            # forward - and a student cannot advance past a step they keep
            # failing. Costs no attempt, so being wrong about it is free.
            shown = failing_cases(problem, tests, cand.failures)
            _trace(trace.record_adapter, corr, GRADING_MODEL, attempt, "evidence_only")
            _trace(trace.record_route, corr, "execution-adapted", "indeterminate")
            # THE SENTENCE IS DERIVED FROM THE EVIDENCE, NEVER WRITTEN BESIDE
            # IT. failing_cases() SKIPS any case it cannot render - a failure
            # carrying no usable index, an input whose rendering raises - and it
            # may skip every one of them. This message promised "the case below"
            # either way, so a student was told to trace a case that was never
            # sent: the worst version of this text, because it reads as the page
            # having dropped the one useful thing in it. One name for the rule
            # this broke - never promise evidence that is not there - and the
            # promise is now a function of `shown` rather than a constant.
            return _remember(_ok(
                "indeterminate", "execution-adapted",
                "We ran your step together with the rest of the solution and the "
                "finished answer came out wrong on at least one case. We can't "
                "be certain the fault is in this step, so your attempt was not "
                "used"
                + (" - but the case below is worth tracing by hand." if shown
                   else ". Try your step on a small input of your own and check "
                        "what it hands on to the rest of the solution."),
                "adapted_evidence_only" if shown else "adapted_evidence_unrenderable",
                deterministic=False,
                consume_attempt=False, execution_outcome="wrong_output",
                failures=cand.failures, failing_cases=shown,
                failed_total=_failed_total(cand, len(shown))))
        # ANYTHING ELSE IS NOT A CONVICTION, and this is the change that
        # matters most in this function. It used to return `incorrect` on
        # wrong_output, which means a student was failed on the strength of code
        # a language model wrote: calibration proves the tail is a correct
        # continuation of the REFERENCE's own earlier chunks, never that it was
        # faithfully re-expressed in the student's vocabulary. A subtly wrong
        # re-expression produces wrong answers that are not the student's. So
        # this tier may now only ever acquit, and a failure falls through.
        _trace(trace.record_adapter, corr, GRADING_MODEL, attempt,
               f"no_acquittal_{cand.outcome}")

    return _remember(_tier4(
        problem, chunk, upto, student_code,
        "no calibrated adapter produced attributable evidence", evidence, corr))


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

    # ── never promise evidence that is not there ─────────────────────────
    # The sentence is derived from `shown`, so the two cannot disagree. Checked
    # on the source because reaching this branch for real needs a model, an
    # oracle and four subprocesses; what must hold is that neither wording is a
    # constant sitting next to the other.
    import inspect as _inspect
    _t3 = _inspect.getsource(_tier3)
    _promise = "the case below is worth tracing by hand"
    assert _promise in _t3, "the with-evidence wording is gone"
    _line = next(l for l in _t3.splitlines() if _promise in l)
    assert "if shown" in _line, \
        "the promise must be conditional on the cases actually existing"
    assert '"adapted_evidence_only" if shown' in _t3, \
        "the two outcomes must be distinguishable in telemetry"
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

    _request_adaptation, chat = _no_model, _no_model

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
    assert _fell.reason_code == "judge_unavailable", _fell.reason_code

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

    # ── A JUDGE MAY ACQUIT, NEVER CONVICT ────────────────────────────────
    # Live, a student's step 1 on `frequency` was failed with "the code
    # incorrectly removes spaces before checking for alphabetic characters,
    # which is unnecessary". Redundant, yes - and identical in output, because
    # isalpha() already skips spaces. Correct code, marked wrong for not being
    # lean, by two judges that agreed with each other.
    #
    # Requiring a nameable failing input narrowed that but left the model
    # deciding what counts as one. The rule below does not depend on the model
    # agreeing to anything: a `false` from this tier cannot reach a student as
    # `incorrect` at all, whatever it says and however confident it is. No
    # network - the judge is stubbed.
    _real_ask = _ask_judge
    def _stub(ok, reason, failing_input):
        return lambda payload, role: (ok, reason, 0.9, "style", failing_input)

    _p, _c = {"description": "Count letters."}, {"prompt": "Prepare.", "reference": "counts = {}"}
    try:
        # Style complaint, no failing input: not a conviction, no attempt spent.
        _ask_judge = _stub(False, "This removes spaces first, which is unnecessary.", "")
        _v = _tier4(_p, _c, "", "code", "why", "evidence")
        assert _v.verdict == "indeterminate", _v.verdict
        assert _v.consume_attempt is False, "an undecided verdict costs no attempt"
        # ...and NEITHER IS A CONFIDENT ONE THAT NAMES AN INPUT. This is the
        # case that used to convict. Both judges agree, both are sure, both can
        # point at 'a1b' - and it is still an opinion about code no run
        # attributed anything to, so it still costs the student nothing.
        _ask_judge = _stub(False, "It counts a character it should skip.", "'a1b'")
        _v = _tier4(_p, _c, "", "code", "why", "evidence")
        assert _v.verdict == "indeterminate", _v.verdict
        assert _v.consume_attempt is False, "an opinion may never cost an attempt"
        assert _v.reason_code == "judge_no_acquittal", _v.reason_code
        # An ACQUITTAL is what this tier is still for.
        _ask_judge = _stub(True, "This prepares the count correctly.", "")
        _v = _tier4(_p, _c, "", "code", "why", "evidence")
        assert _v.verdict == "correct" and _v.deterministic is False, _v
    finally:
        _ask_judge = _real_ask
    assert "DIFFERENT RESULT" in _JUDGE_SYSTEM, \
        "the judge must be told that only a different answer is an error"
    # The verifier must not be shown the primary's answer: two anchored samples
    # agreeing is not independent agreement, and agreement was the whole basis
    # for acting on this tier at all.
    import inspect as _inspect
    assert "PRIMARY JUDGMENT" not in _inspect.getsource(_tier4), \
        "the second judge must not be anchored on the first"

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
