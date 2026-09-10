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
import ast
from datetime import datetime, timezone


def _indent(text: str, n: int) -> str:
    pad = " " * n
    return "\n".join(pad + ln if ln.strip() else ln
                     for ln in (text or "").splitlines())


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
            missing: list) -> list[str]:
    """A truthful header. It names what is the student's own work and what is
    not, because a file that silently mixes the two is a file a teacher cannot
    mark and a student cannot learn from."""
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    out = ['"""', f"{assignment}", ""]
    if student:
        out.append(f"Completed by: {student}")
    out += [f"Downloaded:   {when}", ""]
    out.append(f"Your own answers ({len(filled)}): "
               + (", ".join(filled) if filled else "none"))
    if revealed:
        out.append(f"Answered with the shown solution ({len(revealed)}): "
                   + ", ".join(revealed))
    if missing:
        out.append(f"Not attempted, left as given ({len(missing)}): "
                   + ", ".join(missing))
    out += ["",
            "Everything else is the file exactly as it was handed out.",
            '"""']
    return out


def build_handback(problems: list[dict], answers: dict,
                   assignment_name: str = "Assignment",
                   student_name: str = "",
                   revealed_slugs: set | None = None) -> str:
    """The finished file.

    `problems`   every problem in the assignment, with its context fields.
    `answers`    {slug: body at column 0} - what the student had accepted.
                 A slug that is absent keeps the teacher's original body, so a
                 partly finished assignment still downloads and still runs.
    `revealed`   slugs whose answer came from pressing through to the shown
                 solution; named in the banner, never silently presented as the
                 student's own.

    Splices are applied BOTTOM-UP against spans measured on the pristine file,
    because replacing a body changes the line count and would move every span
    below it."""
    revealed_slugs = revealed_slugs or set()
    lines = original_lines(problems)

    if lines is None:
        # A flat file of plain functions: there is no shared module to splice
        # into, so the file IS its problems, in order.
        out = []
        for p in sorted(problems, key=lambda x: x.get("order") or 0):
            code = answers.get(p.get("slug"))
            out.append(code.rstrip() if code else (p.get("solution") or "").rstrip())
            out.append("")
        body = "\n".join(out)
        filled = sorted(s for s in answers if s not in revealed_slugs)
        head = _banner(assignment_name, student_name, filled,
                       sorted(revealed_slugs),
                       sorted({p["slug"] for p in problems} - set(answers)))
        return "\n".join(head) + "\n\n" + body

    total = len(lines)
    planned = []
    for p in problems:
        slug = p.get("slug")
        if slug not in answers:
            continue                      # keep the teacher's body
        span = body_span(p, total)
        if span is None:
            continue
        planned.append((span, _indent(answers[slug],
                                      int(p.get("context_indent") or 8)
                                      ).splitlines()))

    # Bottom-up: every span was measured on the pristine file.
    for (a, b), replacement in sorted(planned, key=lambda x: x[0][0],
                                      reverse=True):
        lines[a:b] = replacement

    all_slugs = {p.get("slug") for p in problems}
    filled = sorted(s for s in answers if s not in revealed_slugs)
    head = _banner(assignment_name, student_name, filled,
                   sorted(revealed_slugs & set(answers)),
                   sorted(all_slugs - set(answers)))
    return "\n".join(head) + "\n\n" + "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    from .assignments import parse_assignment_file

    src = open("assignment_hw3.py").read()
    probs = parse_assignment_file(src, "assignment_hw3.py")["problems"]

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
