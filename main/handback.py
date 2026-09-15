"""
handback.py - the student's own file back, with their answers in the holes.

WHAT THIS PRODUCES. The teacher's original assignment file, byte for byte,
except that every method the student solved now contains THEIR code instead of
the reference. Same classes, same docstrings, same helper code, same order - so
what they download is the thing they were given, completed, and it runs.

WHY IT IS REBUILT RATHER THAN PATCHED. The uploaded file's TEXT is not stored
anywhere: `assignments.source_file` holds a filename and nothing else. But every
problem row carries `context_prefix` and `context_suffix` - the module before
its body and the module after it, from main/assignments.py - so the original is
recoverable exactly, from any single problem, and the two halves also give the
body's line span for free:

    body starts at   len(context_prefix.splitlines())
    body ends at     total_lines - len(context_suffix.splitlines())

That is derived from stored data, not from re-parsing, so a file this module
rebuilds cannot drift from the file grading assembled.

INDENTATION comes from the same place execution gets it: `context_indent` is
the column the method's body sits at, and accepted chunk code is held at column
0 (main/indent.py), exactly as build_program() expects. So `pop`'s answer lands
under `def pop`, at the class's own depth, with nothing to line up by hand.
"""
from datetime import datetime, timezone


class HandbackUnsafe(RuntimeError):
    """A blank file could not be blanked, so it must not be served.

    The only error this module raises. Everything else here degrades - an
    unparseable solution falls back to its stored text, a missing span is
    skipped - because a file that shows a bit less is still useful. A file that
    shows a bit MORE is the one failure mode that is not recoverable, so it is
    the one that stops."""


def _indent(text: str, n: int) -> str:
    pad = " " * n
    return "\n".join(pad + ln if ln.strip() else ln
                     for ln in (text or "").splitlines())


# The marker a handout leaves where the student is meant to write. Byte-for-byte
# what main/assignments._TODO_MARK looks for, so a file this module blanks and a
# file a teacher hands out say the same thing in the same words.
STUB_MARK = "# YOUR CODE STARTS HERE"

# ...and `pass` under it, so the file still COMPILES with holes in it. A blank
# body is a SyntaxError, and a starter file you cannot run is not a starter
# file - the student is meant to be able to open it and run the doctests.
STUB = STUB_MARK + "\npass"


def blank_body_lines(problem: dict) -> list[str]:
    """The stub, at this problem's own body indent - what fills a hole the
    student has not filled themselves."""
    return _indent(STUB, int(problem.get("context_indent") or 8)).splitlines()


def flat_with_body(problem: dict, body: str | None) -> str:
    """A plain-function problem with `body` in place of the entry function's own.

    `body` is at column 0 - what the chunk store holds - and is seated here at
    the function's own indent, the same way build_program() seats it to run.
    None asks for the stub instead.

    Everything else survives: decorators, the `def` line, the docstring that IS
    the problem statement, any helper the teacher grouped into the same block,
    and anything after the function. Only the entry function's own body is the
    exercise, so only it is replaced.

    THIS IS ALSO A FIX. The flat path used to emit `answers[slug]` as the whole
    problem - but an accepted answer is a BODY, so a downloaded file of plain
    functions came out as bare statements with no `def` line above them and did
    not compile. The class path never had the bug because its `def` lives in
    context_prefix and only the body is ever spliced; this makes the two agree.

    Falls back to the stored source when it will not parse: a file that shows
    what it has beats one that shows an error, and a solution that does not
    parse could never have been prepared in the first place."""
    import ast

    src = problem.get("solution") or ""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src.rstrip()
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not funcs or not funcs[-1].body:
        return src.rstrip()
    fn, lines = funcs[-1], src.splitlines()      # helpers first, entry point last
    start = fn.body[0].lineno - 1
    # The docstring is the problem statement, not the implementation: keep it.
    if (isinstance(fn.body[0], ast.Expr)
            and isinstance(fn.body[0].value, ast.Constant)
            and isinstance(fn.body[0].value.value, str) and len(fn.body) > 1):
        start = fn.body[1].lineno - 1
    if not 0 <= start < len(lines):
        return src.rstrip()
    indent = len(lines[start]) - len(lines[start].lstrip())
    filling = _indent(body if (body or "").strip() else STUB, indent).splitlines()
    return "\n".join(lines[:start] + filling + lines[fn.end_lineno:]).rstrip()


def blank_solution(problem: dict) -> str:
    """A plain-function problem with its body replaced by the stub."""
    return flat_with_body(problem, None)


def solution_body_lines(problem: dict) -> list[str]:
    """The teacher's own body for this problem, as it sits in the file.

    Read out of the stored method source rather than re-derived, and returned at
    FILE indent so it can be spliced straight back in - this is what fills the
    holes of every problem the student did not finish."""
    from .context import solution_body
    body = solution_body(problem) or "pass"
    return _indent(body, int(problem.get("context_indent") or 8)).splitlines()


def original_lines(problems: list[dict]) -> list[str] | None:
    """The teacher's file, rebuilt from any ONE problem's stored context.

    Every problem in an assignment carries the same module split at a different
    point, so one is enough; the first with a usable context wins. None when the
    assignment has no class problems (a flat file of functions has no context to
    rebuild from - see build_handback for that path)."""
    for p in problems:
        prefix = p.get("context_prefix")
        if not prefix:
            continue
        return (prefix.splitlines()
                + solution_body_lines(p)
                + (p.get("context_suffix") or "").splitlines())
    return None


def body_span(problem: dict, total: int) -> tuple[int, int] | None:
    """Where this problem's body sits in the rebuilt file, end-exclusive."""
    prefix = problem.get("context_prefix")
    if not prefix:
        return None
    a = len(prefix.splitlines())
    b = total - len((problem.get("context_suffix") or "").splitlines())
    return (a, b) if 0 <= a <= b <= total else None


def _banner(assignment: str, student: str, filled: list, revealed: list,
            missing: list, blank: bool = False) -> list[str]:
    """A truthful header. It names what is the student's own work and what is
    not, because a file that silently mixes the two is a file a teacher cannot
    mark and a student cannot learn from.

    `blank` says which of the two files this is, and the difference is not
    cosmetic: an unfinished problem is the TEACHER'S code in a download and a
    hole in the working copy, so a header claiming "left as given" over a file
    full of stubs would be a lie about the thing the student is looking at."""
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    out = ['"""', f"{assignment}", ""]
    if student:
        out.append(f"{'Working copy for' if blank else 'Completed by'}: {student}")
    out += [f"{'Generated:  ' if blank else 'Downloaded: '}  {when}", ""]
    out.append(f"Your own answers ({len(filled)}): "
               + (", ".join(filled) if filled else "none yet" if blank else "none"))
    if revealed:
        out.append(f"Answered with the shown solution ({len(revealed)}): "
                   + ", ".join(revealed))
    if missing:
        out.append((f"Still to write ({len(missing)}): " if blank
                    else f"Not attempted, left as given ({len(missing)}): ")
                   + ", ".join(missing))
    out += ["",
            (f"Each one of those is a `{STUB_MARK}` below. Everything else is "
             f"the file exactly as it was handed out." if blank
             else "Everything else is the file exactly as it was handed out."),
            '"""']
    return out


def _replace_function(lines: list[str], problem: dict,
                      body: str | None) -> tuple[list[str], bool]:
    """`lines` with this plain function's body replaced by `body`, or the stub.

    For a problem carried in a file that was rebuilt from a CLASS's context
    there is no body_span to work from - that number comes from the problem's
    own context_prefix, and a loose function has none. So the function is found
    by parsing the text we actually have and matching the entry point by name.

    Returns (lines, replaced). A miss leaves the text untouched, which is the
    right direction for the completing download and the WRONG one for the blank
    view - an untouched function there is the teacher's answer. The caller
    raises rather than serving that, so a shape this cannot handle becomes a
    loud failure instead of a quiet disclosure."""
    import ast

    name = (problem.get("entry_hint") or "").strip()
    src = "\n".join(lines)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return lines, False
    fn = next((n for n in tree.body
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name == name), None)
    if fn is None or not fn.body:
        return lines, False
    start = fn.body[0].lineno - 1
    if (isinstance(fn.body[0], ast.Expr)
            and isinstance(fn.body[0].value, ast.Constant)
            and isinstance(fn.body[0].value.value, str) and len(fn.body) > 1):
        start = fn.body[1].lineno - 1      # the docstring IS the statement
    if not 0 <= start < len(lines):
        return lines, False
    indent = len(lines[start]) - len(lines[start].lstrip())
    filling = _indent(body if (body or "").strip() else STUB, indent).splitlines()
    return lines[:start] + filling + lines[fn.end_lineno:], True


def build_handback(problems: list[dict], answers: dict,
                   assignment_name: str = "Assignment",
                   student_name: str = "",
                   revealed_slugs: set | None = None,
                   blank_unanswered: bool = False) -> str:
    """The finished file.

    `problems`   every problem in the assignment, with its context fields.
    `answers`    {slug: body at column 0} - what the student had accepted.
    `revealed`   slugs whose answer came from pressing through to the shown
                 solution; named in the banner, never silently presented as the
                 student's own.

    `blank_unanswered` decides what fills a problem the student has NOT
    finished, and it is the difference between two genuinely different files:

      False  the teacher's own body. Right for the end-of-term download: what
             lands in Downloads is the handout, completed, and it runs.
      True   a `# YOUR CODE STARTS HERE` stub. Required for anything a student
             reads WHILE STILL WORKING - the same file with the teacher's body
             in it would hand them the answer to the next problem, which is the
             one thing this whole system exists to not do.

    Splices are applied BOTTOM-UP against spans measured on the pristine file,
    because replacing a body changes the line count and would move every span
    below it."""
    revealed_slugs = revealed_slugs or set()
    lines = original_lines(problems)

    if lines is None:
        # A flat file of plain functions: there is no shared module to splice
        # into, so the file IS its problems, in order - preceded by the top of
        # the file, which belongs to no problem and is stored beside them
        # (main/assignments.module_preamble). Without it the rebuild silently
        # dropped the imports and module constants the problems below need.
        preamble = next((p.get("module_preamble") for p in problems
                         if (p.get("module_preamble") or "").strip()), "")
        out = [preamble, ""] if preamble else []
        for p in sorted(problems, key=lambda x: x.get("order") or 0):
            code = answers.get(p.get("slug"))
            if code:
                out.append(flat_with_body(p, code))
            else:
                out.append(blank_solution(p) if blank_unanswered
                           else (p.get("solution") or "").rstrip())
            out.append("")
        body = "\n".join(out)
        filled = sorted(s for s in answers if s not in revealed_slugs)
        head = _banner(assignment_name, student_name, filled,
                       sorted(revealed_slugs),
                       sorted({p["slug"] for p in problems} - set(answers)),
                       blank=blank_unanswered)
        return "\n".join(head) + "\n\n" + body

    total = len(lines)
    planned, unplaced = [], []
    for p in problems:
        slug = p.get("slug")
        answered = slug in answers
        if not answered and not blank_unanswered:
            continue                      # keep the teacher's body
        span = body_span(p, total)
        if span is None:
            # NO CONTEXT, so there is no hole in the rebuilt module to splice
            # into - this problem is a plain function sitting beside the class,
            # and `lines` was rebuilt from the CLASS's context, which carries
            # that function's full body along with everything else in the file.
            #
            # Skipping it, which is what this did, left the teacher's working
            # implementation in a file titled "the rest is left blank for you".
            # One mixed assignment - any class plus any loose function - and the
            # answer came back out through the starter view. Found by audit.
            unplaced.append(p)
            continue
        planned.append((span,
                        _indent(answers[slug],
                                int(p.get("context_indent") or 8)).splitlines()
                        if answered else blank_body_lines(p)))

    # Bottom-up: every span was measured on the pristine file.
    for (a, b), replacement in sorted(planned, key=lambda x: x[0][0],
                                      reverse=True):
        lines[a:b] = replacement

    # The loose functions, found by their own `def` in the rebuilt text rather
    # than by a stored span they do not have. Done AFTER the class splices and
    # re-measured each time, because every splice above moved the line numbers.
    for p in unplaced:
        lines, replaced = _replace_function(
            lines, p, answers.get(p.get("slug")) if p.get("slug") in answers
            else None)
        # FAIL LOUD, NEVER QUIETLY. In blank mode an un-replaced function is the
        # teacher's body sitting in a file the student is told is blank; a
        # traceback the instructor sees beats a disclosure nobody sees.
        if blank_unanswered and not replaced:
            raise HandbackUnsafe(
                f"could not blank '{p.get('slug')}' - refusing to serve a "
                f"starter file that still contains a reference solution")

    all_slugs = {p.get("slug") for p in problems}
    filled = sorted(s for s in answers if s not in revealed_slugs)
    head = _banner(assignment_name, student_name, filled,
                   sorted(revealed_slugs & set(answers)),
                   sorted(all_slugs - set(answers)),
                   blank=blank_unanswered)
    return "\n".join(head) + "\n\n" + "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    import os

    from .assignments import parse_assignment_file  # noqa: F401

    # ── the blank file, on a synthetic assignment that ships with the repo ──
    # Deliberately NOT gated on the HW3 fixture below: this is the file a
    # student reads while they are still working, so the thing it must never do
    # - show them the reference body of a problem they have not solved - has to
    # be checked on every run, not only on a machine that happens to have a real
    # assignment lying next to it.
    CLASSES = '''"""LAB9 - Stacks"""


class Node:
    """A link in the stack. Given to you."""
    def __init__(self, value):
        self.value, self.next = value, None


class Stack:
    """A LIFO stack.

    >>> x = Stack(); x.push(2); x.push(4)
    >>> x.pop()
    4
    """
    def __init__(self):
        self.top = None

    def push(self, value):
        """Put value on the top of the stack."""
        node = Node(value)
        node.next = self.top
        self.top = node

    def pop(self):
        """Take the top value off the stack and return it."""
        v = self.top.value
        self.top = self.top.next
        return v

    def isEmpty(self):
        """Return True when nothing is on the stack."""
        return self.top is None
'''
    cls_probs = parse_assignment_file(CLASSES, "lab9.py")["problems"]
    assert [p["slug"] for p in cls_probs] == \
        ["stack-push", "stack-pop", "stack-is-empty"], cls_probs
    mine = "node = Node(value)\nnode.next = self.top\nself.top = node"
    working = build_handback(cls_probs, {"stack-push": mine}, "LAB9", "A Student",
                             blank_unanswered=True)

    # THE WHOLE POINT: no reference body for anything they have not solved.
    for leaked in ("self.top = self.top.next", "return self.top is None",
                   "v = self.top.value"):
        assert leaked not in working, f"reference leaked into the working copy: {leaked}"
    # ...their own answer IS there, under its own def, at the class's depth.
    assert "\n        node.next = self.top\n" in working, working
    # ...every unsolved method is a hole, and the file still RUNS with holes in
    # it, or it is not a starter file.
    assert working.count("    " + STUB_MARK) == 2, working
    compile(working, "<working>", "exec")
    ns = {}
    exec(working, ns)
    s_ = ns["Stack"]()
    s_.push(7)
    assert s_.top.value == 7, "the student's own push must actually work"
    assert s_.pop() is None, "an unwritten method is a stub, not the answer"
    # Code the student was GIVEN is never blanked - Node and __init__ are not
    # exercises, and a starter file without them cannot run at all.
    assert "self.value, self.next = value, None" in working
    assert "self.top = None" in working

    # The completing mode still exists and still completes - but NOTHING SERVES
    # IT TODAY. /handback used to, and that handed a student who had finished
    # one problem the teacher's body for every problem they had not. It is kept
    # for the one case that can justify it - a runnable "completed file" after
    # an assignment has closed - and a caller that wants it has to ask for it.
    completing = build_handback(cls_probs, {"stack-push": mine}, "LAB9", "A Student")
    assert "return self.top is None" in completing
    assert STUB_MARK not in completing
    assert "Not attempted, left as given" in completing
    assert "Still to write (2)" in working, working

    # ── A CLASS AND A LOOSE FUNCTION IN ONE FILE ──────────────────────────
    # The file is rebuilt from the CLASS's stored context, which carries the
    # loose function's full body with it - and that function has no span of its
    # own to splice into, so it used to be skipped and its reference answer went
    # out inside a file headed "the rest is left blank for you". Found by audit.
    MIXED = CLASSES.rstrip() + (
        "\n\n\ndef running_total(nums):\n"
        '    """Return the running totals of nums."""\n'
        "    out, total = [], 0\n"
        "    for n in nums:\n"
        "        total += n\n"
        "        out.append(total)\n"
        "    return out\n")
    mixed = parse_assignment_file(MIXED, "lab9.py")["problems"]
    assert "running-total" in [p["slug"] for p in mixed], \
        [p["slug"] for p in mixed]
    both = build_handback(mixed, {"stack-push": mine}, "LAB9", "A",
                          blank_unanswered=True)
    assert "out.append(total)" not in both, "the loose function kept its answer"
    assert "def running_total(nums):" in both, "...and must still be IN the file"
    assert "Return the running totals of nums." in both, "the statement stays"
    compile(both, "<mixed>", "exec")
    # The student's own answer to a loose function lands under its own def.
    theirs = build_handback(mixed, {"running-total": "return nums"}, "LAB9", "A",
                            blank_unanswered=True)
    assert "\n    return nums\n" in theirs, theirs
    compile(theirs, "<mixed2>", "exec")
    # A shape this cannot blank must RAISE, never quietly ship the answer.
    broken = [dict(p, entry_hint="not_a_real_name") if p["slug"] == "running-total"
              else p for p in mixed]
    try:
        build_handback(broken, {}, "LAB9", "A", blank_unanswered=True)
        raise AssertionError("an unblankable file was served")
    except HandbackUnsafe:
        pass
    # ...but the completing download is untouched by that guard.
    assert "out.append(total)" in build_handback(broken, {"stack-push": mine},
                                                 "LAB9", "A")

    # ── the flat path: same rule, no shared module to splice into ──────────
    FLAT = '''"""Week 1"""

# --- problem: is-leap-year ---
def is_leap_year(year):
    """Return True when year is a leap year."""
    if year % 400 == 0:
        return True
    return year % 4 == 0


# --- problem: double-it ---
def double_it(n):
    """Return n doubled."""
    return n * 2
'''
    flat_probs = parse_assignment_file(FLAT, "week1.py")["problems"]
    flat = build_handback(flat_probs, {"double-it": "return n + n"}, "Week 1", "A",
                          blank_unanswered=True)
    assert "year % 400" not in flat, "reference leaked on the flat path"
    assert "return n + n" in flat, "their own answer must survive"
    # The docstring IS the problem statement, so it is kept, not blanked with
    # the body it sits above.
    assert "Return True when year is a leap year." in flat, flat
    # Counted INDENTED, so the banner's own mention of the marker is not one.
    assert flat.count("    " + STUB_MARK) == 1, flat
    assert "def is_leap_year(year):" in flat, "the def line must survive"
    assert "def double_it(n):" in flat and "\n    return n + n" in flat, \
        "an accepted answer is a BODY - it has to land under its own def"
    compile(flat, "<flat>", "exec")

    # A student who has finished nothing is exactly who needs the starter file.
    starter = build_handback(cls_probs, {}, "LAB9", "A", blank_unanswered=True)
    compile(starter, "<starter>", "exec")
    assert starter.count("    " + STUB_MARK) == 3, starter
    assert "none yet" in starter

    print("handback.py blank-file self-check OK")

    # The fixture is a real assignment file, kept out of the repo. Without it
    # there is nothing to rebuild, so skip rather than fail: a self-check that
    # breaks when a sample file moves teaches nothing about this module.
    FIXTURE = os.environ.get("MICROTUTOR_HW3", "assignment_hw3.py")
    if not os.path.exists(FIXTURE):
        print(f"handback.py fixture self-check SKIPPED (no {FIXTURE})")
        raise SystemExit(0)
    src = open(FIXTURE).read()
    probs = parse_assignment_file(src, FIXTURE)["problems"]

    # ── the rebuilt file IS the original ──────────────────────────────────
    rebuilt = "\n".join(original_lines(probs))
    assert rebuilt.rstrip() == src.rstrip(), "rebuild is not byte-identical"

    # ── one answer lands under its own def, at the right depth ────────────
    out = build_handback(probs, {"stack-pop": "return 42"},
                         "HW3", "A Student")
    assert "def pop(self):" in out
    seg = out[out.index("def pop(self):"):]
    seg = seg[:seg.index("def peek")]
    assert "\n        return 42\n" in seg, seg
    assert "self.top = node.next" not in seg, "the teacher's body must be gone"
    # ...and every OTHER method is untouched.
    assert "node.next = self.top" in out, "push must still be the original"
    compile(out, "<handback>", "exec")

    # ── several answers at once, spans must not drift ─────────────────────
    many = build_handback(probs, {
        "stack-pop": "return 1", "stack-push": "return 2",
        "stack-is-empty": "return 3",
        "advanced-calculator-calculate-expressions": "return 4"}, "HW3", "S")
    compile(many, "<handback>", "exec")
    ns = {}
    exec(many, ns)
    st = ns["Stack"]()
    assert st.pop() == 1 and st.push(0) == 2 and st.isEmpty() == 3, \
        "each answer must land in ITS OWN method"
    assert ns["AdvancedCalculator"]().calculateExpressions() == 4

    # ── an unfinished assignment still downloads, and still runs ──────────
    none_done = build_handback(probs, {}, "HW3", "S")
    compile(none_done, "<handback>", "exec")
    assert "Not attempted, left as given (11)" in none_done

    # ── the banner does not pass a shown answer off as their own ──────────
    mixed = build_handback(probs, {"stack-pop": "return 1", "stack-peek": "return 2"},
                           "HW3", "S", revealed_slugs={"stack-peek"})
    assert "Your own answers (1): stack-pop" in mixed, mixed[:400]
    assert "Answered with the shown solution (1): stack-peek" in mixed

    print("handback.py self-check OK")
