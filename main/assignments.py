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
        # A class-based solution is legitimate Python, but the header the
        # decomposer builds is a bare `def`. Refuse clearly rather than let it
        # fail later inside decomposition where the teacher cannot see why.
        if any(isinstance(n, ast.ClassDef) for n in tree.body):
            raise ValueError(
                "class-based solutions are not supported in an assignment file; "
                "write a plain top-level function instead")
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
        # No markers: every top-level function is its own problem. Forgiving for
        # a simple file, but it cannot group helpers - hence the hint below.
        tree = ast.parse(text)
        lines = text.splitlines(keepends=True)
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        if not funcs:
            raise AssignmentParseError(
                "no problems found. Each problem is a function whose docstring "
                "is the problem statement; group helpers under a "
                "'# --- problem: some-slug ---' marker")
        for i, fn in enumerate(funcs):
            start = fn.lineno - 1
            for d in fn.decorator_list:                # keep decorators with it
                start = min(start, d.lineno - 1)
            end = funcs[i + 1].lineno - 1 if i + 1 < len(funcs) else len(lines)
            blocks.append((None, "".join(lines[start:end])))

    problems, errors, seen = [], [], set()
    for i, (slug, src) in enumerate(blocks):
        # The block's own text rides along with its error. Without it a problem
        # that failed to parse had its source stored nowhere, so the only way to
        # fix one was to re-upload the whole file; the teacher page can now put
        # this exact text in front of the instructor to correct in place.
        text = _dedent_block(src)
        try:
            p = _problem_from_block(slug, src, i)
        except ValueError as e:
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
