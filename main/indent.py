"""
indent.py - seating a submission at the depth its chunk actually occupies.

A decomposition may split the solution ANYWHERE, and the split lands mid-block
often: chunk 1 of "reverse the digits" opens a `while`, and chunk 2 is the
`n //= 10` that closes it out. The reference for such a chunk is stored ALREADY
INDENTED - that is the only way the stacked references form a runnable program
and clear the generation gate in run_phase1.assemble_references().

The student is shown an empty editor and a one-line prompt. Nothing in that
says their answer lands four columns in, and no student would guess it, so they
type it flat. Flat is then where it was stitched, which silently moved
`n //= 10` OUT of the loop and hung the grader on an infinite loop the student
never wrote. Every tier saw a timeout, so a correct answer came back as "your
solution took too long" - or worse, went to the LLM judge to have an opinion
formed about code that was never assembled the way the student meant it.

The conclusion this module encodes: indentation is not the student's to get
right, because they were never told what it should be. The chunk's own
reference records the depth, and every submission is re-seated to it before
anything reads it.

Pure, deterministic and free of I/O, so grading and the route that stores the
accepted answer can each call it and get byte-identical results.
"""

# Python 3 forbids mixing tabs and spaces for indentation, and the execution
# sandbox would reject it anyway. Expanding first means a student who pressed
# Tab is measured on the same scale as one who pressed space four times.
TAB_WIDTH = 4


def _leading(line: str) -> int:
    """Columns of indentation on one already-expanded line."""
    return len(line) - len(line.lstrip(" "))


def _expand(code: str) -> list[str]:
    return (code or "").expandtabs(TAB_WIDTH).splitlines()


def base_indent(code: str) -> int:
    """The column this block as a whole sits at.

    Measured from the SHALLOWEST non-blank line, which is the one that carries
    the block's own depth; anything deeper belongs to structure inside it.
    """
    depths = [_leading(ln) for ln in _expand(code) if ln.strip()]
    return min(depths) if depths else 0


def dedent_block(code: str) -> str:
    """Strip the indentation common to every non-blank line, keeping the
    block's internal shape intact.

    Blank lines are emptied rather than left holding stray trailing spaces, and
    the blank lines around the block are dropped entirely - a leading newline
    from the editor must not become a blank first line of the function body.
    """
    lines = _expand(code)
    filled = [i for i, ln in enumerate(lines) if ln.strip()]
    if not filled:
        return ""
    cut = base_indent(code)
    body = lines[filled[0]:filled[-1] + 1]
    # rstrip as well: trailing spaces are invisible in the editor and carry no
    # meaning in Python, but they make two identical answers compare unequal.
    return "\n".join(ln[cut:].rstrip() if ln.strip() else "" for ln in body)


def align_to(code: str, columns: int) -> str:
    """Re-seat `code` so its outermost line starts at `columns`.

    Whatever the student typed - flat, half-indented, or exactly right - comes
    out at the one depth that makes it land where the prompt meant it to.
    """
    body = dedent_block(code)
    if not body:
        return ""
    pad = " " * max(0, columns)
    return "\n".join(pad + ln if ln.strip() else "" for ln in body.split("\n"))


def align_to_chunk(code: str, chunk: dict) -> str:
    """Re-seat a submission at the depth of the chunk it answers.

    The chunk's reference is the authority: it was written to stack with its
    neighbours into a program that passed the generation gate, so its own
    indentation is the depth this step genuinely occupies.
    """
    return align_to(code, base_indent((chunk or {}).get("reference") or ""))


if __name__ == "__main__":
    # Self-check. Pure - no session, no oracle, no model.
    #   python -m main.indent
    assert base_indent("a = 1") == 0
    assert base_indent("    n //= 10") == 4
    assert base_indent("") == 0
    assert base_indent("\n   \n") == 0
    # The shallowest line carries the depth; nesting below it does not.
    assert base_indent("    while n:\n        n -= 1") == 4
    # A `for` header plus body, flat: depth 0, not 4.
    assert base_indent("for i in nums:\n    total += i") == 0

    assert dedent_block("        x = 1\n        y = 2") == "x = 1\ny = 2"
    # Internal shape survives the dedent.
    assert dedent_block("    while n:\n        n -= 1") == "while n:\n    n -= 1"
    # Blank lines come back empty, never as trailing whitespace.
    assert dedent_block("    a = 1\n   \n    b = 2") == "a = 1\n\nb = 2"

    # The bug this module exists for: a flat answer to a step that lives
    # inside a loop.
    chunk = {"reference": "    n //= 10"}
    assert align_to_chunk("n //= 10", chunk) == "    n //= 10"
    # Already correct in, unchanged out - idempotent.
    assert align_to_chunk("    n //= 10", chunk) == "    n //= 10"
    # Over-indented guesses are corrected too, not compounded.
    assert align_to_chunk("            n //= 10", chunk) == "    n //= 10"
    # Tabs measure the same as four spaces.
    assert align_to_chunk("\tn //= 10", chunk) == "    n //= 10"
    # A multi-line answer keeps its own nesting, re-seated as a unit.
    assert align_to_chunk("if n:\n    n -= 1", chunk) == "    if n:\n        n -= 1"
    # A top-level chunk is left where it is.
    assert align_to_chunk("return rev", {"reference": "return rev"}) == "return rev"
    # Nothing in, nothing out - the blank-answer check downstream still fires.
    assert align_to_chunk("   \n  ", chunk) == ""
    assert align_to_chunk("", {}) == ""
    # Editor newlines around the answer are dropped, not carried into the body.
    assert align_to_chunk("\n\nn //= 10\n\n", chunk) == "    n //= 10"
    # ...but a blank line INSIDE the answer is kept, so the shape is preserved.
    assert align_to_chunk("a = 1\n\nb = 2", {"reference": ""}) == "a = 1\n\nb = 2"

    print("indent.py self-check OK")
