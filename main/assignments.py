"""
assignments.py - parse a teacher's assignment file into problems.

THE FORMAT is a plain Python file, because that is what a CS instructor writes
anyway when they make a solution key:

    \"\"\"Week 3 - Loops and Strings\"\"\"          <- assignment name

    # --- problem: palindrome-number ---        <- slug marker
    def is_palindrome(x: int) -> bool:
        \"\"\"Given an integer x, return True if x reads the same
        forwards and backwards, else False.\"\"\"          <- shown to STUDENTS
        if x < 0:
            return False
        s = str(x)
        return s == s[::-1]                              <- ground truth, PRIVATE

Why a .py file rather than JSON or YAML: the teacher's own editor checks the
syntax before they upload, multi-line code needs no escaping, and there is no
new dependency. The docstring/body split maps exactly onto the two things the
backend needs and must keep apart - the public statement and the private
solution.

ENTRY POINT: within one problem block the LAST top-level function is the entry
point and any earlier ones are helpers. That mirrors main/identity.py's
_mirror_resolve(), which mirrors the execution harness, so all three agree on
which function is "the problem".

This module is PURE: it parses text and returns dicts. No I/O, no database
writes, no model calls - so it can be tested on its own.
"""
import ast
import re
import textwrap

# A slug marker opens a problem block. Everything up to the next marker (or EOF)
# belongs to it, so a problem may carry helper functions without them being
# mistaken for separate problems.
_MARKER = re.compile(
    r"^[ \t]*#[ \t]*-{2,}[ \t]*problem[ \t]*:[ \t]*([A-Za-z0-9][\w.-]*)[ \t]*-*[ \t]*$",
    re.MULTILINE)

_SLUG_OK = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class AssignmentParseError(ValueError):
    """The file could not be read as an assignment at all.

    Distinct from a per-problem error: those are collected and reported so the
    teacher can fix one problem without losing the rest of the upload."""


def _slugify(name: str) -> str:
    """camelCase / snake_case function name -> kebab-case slug."""
    s = re.sub(r"(?<!^)(?=[A-Z])", "-", name).replace("_", "-").lower()
    return re.sub(r"-{2,}", "-", s).strip("-")


def _titleize(slug: str) -> str:
    return " ".join(w.capitalize() for w in slug.split("-"))


def _dedent_block(src: str) -> str:
    """Blocks are already top-level; just trim blank edges."""
    return "\n".join(ln.rstrip() for ln in src.strip("\n").splitlines()).strip("\n")


# Methods that are scaffolding in every class we have seen, never the exercise:
# the constructor and the two printers. Everything else is a candidate step.
_SCAFFOLD = {"__init__", "__str__", "__repr__", "__new__"}

# The marker a handout leaves where the student is meant to write. When a
# teacher fills these in and uploads the solved file, the comments usually
# survive - and they are then an EXACT record of which methods were the
# exercise, which no heuristic can match.
_TODO_MARK = re.compile(r"#\s*YOUR\s+CODE\s+STARTS\s+HERE", re.IGNORECASE)

# ...and the explicit override, for a file where they did not survive:
#     # --- steps: push, pop, peek ---
_STEPS_MARK = re.compile(
    r"^[ \t]*#[ \t]*-{2,}[ \t]*steps?[ \t]*:[ \t]*([^\n]+?)[ \t]*-*[ \t]*$",
    re.MULTILINE)


def _method_span(src_lines: list[str], node: ast.FunctionDef) -> tuple[int, int]:
    """Line span of one method INCLUDING its decorators, 0-based, end-exclusive."""
    start = node.lineno - 1
    for d in node.decorator_list:
        start = min(start, d.lineno - 1)
    return start, node.end_lineno


def _class_steps(cls: ast.ClassDef, class_src: str,
                 module_lines: list[str]) -> list[ast.FunctionDef]:
    """Which methods of this class are the exercise, in source order.

    Three sources, most explicit first. A teacher who says nothing gets every
    method except the scaffolding, which is right far more often than it is
    wrong - and the upload page lists what was chosen, so a wrong guess is
    visible rather than silent."""
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)]

    named = _STEPS_MARK.search(class_src)
    if named:
        want = {w.strip() for w in named.group(1).split(",") if w.strip()}
        picked = [m for m in methods if m.name in want]
        if picked:
            return picked

    # MODULE lines, not class_src: ast line numbers are module-relative, and
    # slicing them into a class-relative list read a window from the wrong part
    # of the file - which is how Stack came back as {__init__, __str__, isEmpty}
    # instead of the five methods the handout actually leaves blank.
    todo = []
    for m in methods:
        a, b = _method_span(module_lines, m)
        if any(_TODO_MARK.search(ln) for ln in module_lines[a:b]):
            todo.append(m)
    if todo:
        return todo

    return [m for m in methods if m.name not in _SCAFFOLD]


def _class_problems(cls: ast.ClassDef, module_src: str, preamble: str,
                    index: int) -> tuple[list[dict], list[dict]]:
    """One class -> one problem GROUP whose steps are its methods.

    Returns (problems, errors). Each method becomes an ordinary problem the
    existing pipeline can prepare on its own: `solution` is just that method,
    dedented to top level, so decomposition splits the METHOD and not the whole
    class. What makes it runnable is `context_prefix`/`context_suffix` - the
    module and the rest of the class, with a hole where this method's body goes.
    Everything downstream that assembles a program fills that hole instead of
    concatenating a bare function."""
    lines = module_src.splitlines()
    class_a, class_b = cls.lineno - 1, cls.end_lineno
    class_src = "\n".join(lines[class_a:class_b])
    group_slug = _slugify(cls.name)
    group_doc = (ast.get_docstring(cls) or "").strip()

    steps = _class_steps(cls, class_src, lines)
    if not steps:
        # A class with nothing to implement is a HELPER the exercise builds on -
        # HW3's Node is exactly this - not a broken problem. It stays in the
        # file (every method-problem carries the whole module as context, so it
        # is still in scope) and simply contributes no steps of its own.
        return [], []

    problems, errors = [], []
    for pos, m in enumerate(steps):
        a, b = _method_span(lines, m)
        method_src = textwrap.dedent("\n".join(lines[a:b])).rstrip()
        slug = f"{group_slug}-{_slugify(m.name.strip('_') or m.name)}"
        if not _SLUG_OK.match(slug):
            errors.append({"slug": slug, "error": f"method '{m.name}' does not "
                           f"make a usable slug", "source": method_src})
            continue

        # The program around this method: everything before it, and everything
        # after. The student's body is dropped in between at the class's own
        # indent depth, so the rest of the class - and any helper class the file
        # defines above it - is in scope exactly as the teacher wrote it.
        body_a = m.body[0].lineno - 1
        if (isinstance(m.body[0], ast.Expr)
                and isinstance(m.body[0].value, ast.Constant)
                and isinstance(m.body[0].value.value, str) and len(m.body) > 1):
            body_a = m.body[1].lineno - 1          # keep the docstring in prefix
        problems.append({
            "slug": slug,
            "title": m.name,
            # A method's own docstring is its statement; classes like Stack put
            # the whole specification on the CLASS instead, so fall back to it
            # rather than refusing a perfectly well documented exercise.
            "description": (ast.get_docstring(m) or "").strip() or group_doc,
            "solution": method_src,
            "entry_hint": m.name,
            "order": index + pos,
            "context_prefix": "\n".join(lines[:body_a]),
            "context_suffix": "\n".join(lines[m.end_lineno:]),
            "context_indent": (len(lines[body_a]) - len(lines[body_a].lstrip())
                               if body_a < len(lines) else 8),
            "group_slug": group_slug,
            "group_title": cls.name,
            "group_description": group_doc,
            "group_kind": "class",
            "group_order": index,
            "member_order": pos,
            "member_of": len(steps),
        })
    if not group_doc and not any(p["description"] for p in problems):
        errors.append({"slug": group_slug,
                       "error": f"class '{cls.name}' and its methods have no "
                                f"docstring. The docstring IS the problem "
                                f"statement students see, so one is required.",
                       "source": class_src})
        return [], errors
    return problems, errors


def _problem_from_block(slug: str | None, src: str, index: int) -> dict:
    """Turn one block of source into a problem dict, or raise ValueError."""
    src = _dedent_block(src)
    if not src.strip():
        raise ValueError("block is empty")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        raise ValueError(f"not valid Python: {e.msg} (line {e.lineno})") from None

    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not funcs:
        # A class reaching HERE means it was inside a `# --- problem: ---`
        # block, which asks for one problem; a class is a GROUP of them and is
        # handled by _class_problems on the top-level scan instead.
        if any(isinstance(n, ast.ClassDef) for n in tree.body):
            raise ValueError(
                "a class is a group of problems, one per method - take it out "
                "of the '# --- problem: ---' block and leave it at the top "
                "level of the file, where each method becomes its own step")
        raise ValueError("no function found - each problem needs one")

    entry = funcs[-1]                    # helpers first, entry point last
    doc = ast.get_docstring(entry)
    if not doc or not doc.strip():
        raise ValueError(
            f"function '{entry.name}' has no docstring. The docstring IS the "
            f"problem statement students see, so it is required")

    slug = slug or _slugify(entry.name)
    if not _SLUG_OK.match(slug):
        raise ValueError(
            f"slug '{slug}' must be lowercase letters, digits and hyphens")

    return {"slug": slug, "title": _titleize(slug),
            "description": doc.strip(), "solution": src,
            "entry_hint": entry.name, "order": index}


def parse_assignment_file(text: str, filename: str = "assignment.py") -> dict:
    """Parse an assignment file.

    Returns {"name", "problems": [...], "errors": [{"slug", "error"}]}.

    A malformed PROBLEM never fails the whole upload - it lands in `errors` so
    the teacher is told exactly which one to fix while the rest proceed. Only an
    unusable FILE raises."""
    if not text or not text.strip():
        raise AssignmentParseError("the file is empty")

    # Assignment name: module docstring, else the filename.
    try:
        doc = (ast.get_docstring(ast.parse(text)) or "").strip()
    except SyntaxError as e:
        raise AssignmentParseError(
            f"the file is not valid Python: {e.msg} (line {e.lineno})") from None
    name = doc.splitlines()[0].strip() if doc else ""
    if not name:
        name = (re.sub(r"\.py$", "", filename).replace("_", " ").strip()
                or "Untitled assignment")

    marks = list(_MARKER.finditer(text))
    blocks: list[tuple[str | None, str]] = []
    if marks:
        for i, m in enumerate(marks):
            end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
            blocks.append((m.group(1), text[m.end():end]))
    else:
        # No markers: every top-level function is its own problem, and every
        # top-level CLASS is a group of them - one per method it asks the
        # student to write. Forgiving for a simple file, but it cannot group
        # loose helpers - hence the hint below.
        tree = ast.parse(text)
        lines = text.splitlines(keepends=True)
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
        if not funcs and not classes:
            raise AssignmentParseError(
                "no problems found. Each problem is a function whose docstring "
                "is the problem statement, or a class whose methods are the "
                "exercise; group helpers under a "
                "'# --- problem: some-slug ---' marker")
        for i, fn in enumerate(funcs):
            start = fn.lineno - 1
            for d in fn.decorator_list:                # keep decorators with it
                start = min(start, d.lineno - 1)
            nxt = [n.lineno - 1 for n in tree.body if n.lineno - 1 > start]
            end = min(nxt) if nxt else len(lines)
            blocks.append((None, "".join(lines[start:end])))

    problems, errors, seen = [], [], set()

    # Classes are expanded on the WHOLE file, not per block: a method needs the
    # module around it to run (HW3's Calculator uses Stack, which is a different
    # class entirely), and that context is only available here.
    if not marks:
        for ci, cls in enumerate(classes):
            got, bad = _class_problems(cls, text, "", ci * 100)
            problems.extend(got)
            errors.extend(bad)
            seen.update(p["slug"] for p in got)
    for i, (slug, src) in enumerate(blocks):
        # The block's own text rides along with its error. Without it a problem
        # that failed to parse had its source stored nowhere, so the only way to
        # fix one was to re-upload the whole file; the teacher page can now put
        # this exact text in front of the instructor to correct in place.
        text = _dedent_block(src)
        try:
            p = _problem_from_block(slug, src, i)
        except ValueError as e:
            # A file that also defines classes is a real assignment, and its
            # loose `run_tests()` / `main()` plumbing is not a problem the
            # teacher forgot to document. Only complain about an undocumented
            # function when it was the only thing that could have been one.
            if not marks and classes and "has no docstring" in str(e):
                continue
            errors.append({"slug": slug or f"block {i + 1}", "error": str(e),
                           "source": text})
            continue
        if p["slug"] in seen:
            errors.append({"slug": p["slug"], "error": "duplicate slug in this file",
                           "source": text})
            continue
        seen.add(p["slug"])
        problems.append(p)

    if not problems and not errors:
        raise AssignmentParseError("no problems found in the file")
    return {"name": name, "problems": problems, "errors": errors}


# The starter file a teacher downloads. It MUST survive preparation itself
# the first thing a teacher does is upload this, and an example that fails the
# oracle-strength gate would look like the product is broken.
#
# That gate needs at least _MIN_MUTANTS (3) distinct ways to break the solution,
# or it cannot tell a strong test suite from a lucky one. A one-liner like
# `sum(nums)` yields ONE mutant and is correctly rejected as ungradeable - so
# both examples below deliberately carry several comparisons and constants.
# Measured with main.mutation.generate_mutants: 11 and 6 respectively.
TEMPLATE = '''"""Week 1 - Warm-up"""

# --- problem: is-leap-year ---
def is_leap_year(year):
    """Given a year, return True if it is a leap year and False otherwise.

    A year is a leap year when it is divisible by 4, except that years
    divisible by 100 are not leap years, unless they are also divisible
    by 400.

    Example: 2024 -> True, 1900 -> False, 2000 -> True
    """
    if year % 400 == 0:
        return True
    if year % 100 == 0:
        return False
    return year % 4 == 0


# --- problem: second-largest ---
def second_largest(nums):
    """Given a list of at least two distinct integers, return the second
    largest value in the list.

    Example: second_largest([4, 1, 9, 7]) -> 7
    """
    best = nums[0]
    second = nums[1]
    if second > best:
        best, second = second, best
    for n in nums[2:]:
        if n > best:
            best, second = n, best
        elif n > second:
            second = n
    return second
'''


if __name__ == "__main__":
    # Self-check: the shapes that matter, including the failure modes.
    r = parse_assignment_file(TEMPLATE, "week1.py")
    assert r["name"] == "Week 1 - Warm-up", r["name"]
    assert [p["slug"] for p in r["problems"]] == ["is-leap-year", "second-largest"]
    assert r["problems"][0]["description"].startswith("Given a year")
    assert "year % 400" in r["problems"][0]["solution"]
    assert "Week 1" not in r["problems"][0]["solution"]      # no assignment-doc leak
    assert r["errors"] == []

    # helpers stay with their problem when a marker groups them
    grouped = ('# --- problem: two-sum ---\n'
               'def _seen(nums):\n    return {v: i for i, v in enumerate(nums)}\n\n'
               'def two_sum(nums, target):\n'
               '    """Return indices of the two numbers adding to target."""\n'
               '    d = _seen(nums)\n    return [0, 1]\n')
    g = parse_assignment_file(grouped, "x.py")
    assert len(g["problems"]) == 1, g
    assert g["problems"][0]["entry_hint"] == "two_sum"        # last func wins
    assert "_seen" in g["problems"][0]["solution"]            # helper preserved

    # markerless files: one function per problem, slug derived from the name
    m = parse_assignment_file('def isPalindrome(x):\n    """Is x a palindrome?"""\n    return True\n')
    assert m["problems"][0]["slug"] == "is-palindrome", m["problems"][0]["slug"]

    # a bad problem is reported, not fatal; good ones still come through
    mixed = ('# --- problem: ok-one ---\n'
             'def ok_one(x):\n    """Doc."""\n    return x\n\n'
             '# --- problem: no-doc ---\n'
             'def no_doc(x):\n    return x\n')
    mx = parse_assignment_file(mixed, "m.py")
    assert [p["slug"] for p in mx["problems"]] == ["ok-one"]
    assert mx["errors"][0]["slug"] == "no-doc" and "docstring" in mx["errors"][0]["error"]
    # The failed block's own text comes back with it, or there is nothing for
    # the teacher's fix-and-retry panel to open.
    assert "def no_doc(x):" in mx["errors"][0]["source"], mx["errors"][0]

    for bad, why in ((" ", "empty"), ("def f(:\n  pass", "valid Python"),
                     ("x = 1\n", "no problems")):
        try:
            parse_assignment_file(bad, "b.py")
            raise AssertionError(f"expected failure for {why}")
        except AssignmentParseError as e:
            assert why in str(e), (why, str(e))

    print("assignments.py self-check OK")
