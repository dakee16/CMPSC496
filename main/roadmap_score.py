"""roadmap_score.py - a grade out of 100 for a roadmap, and the three best.

14 Oct, Sanan: "add a rubric/grading system for the roadmaps, and grade them,
top 3 will make it into the final 3 roadmaps (also add a cutoff grade ...)".
Before this, a problem served every saved roadmap at random - up to seven, some
exact copies - and HW4's __contains__ served one whose first step was one line
and whose second was fifteen.

THE RUBRIC. Pure AST work on the teacher's own lines: no model, no test run, so
it is cheap enough to rank the saved roadmaps on every open.

    biggest step     30   8 code lines or fewer full marks, 20 or more none
    balance          25   how far the biggest step is over an even share of
                          the method: none, full marks; all of it, none. One
                          step doing everything (1 line, then 15) is what
                          this catches - a short setup step beside longer
                          ones is normal and costs little
    number of steps  15   about one step per 5 lines, between 2 and 5
    clean starts     15   steps that begin at the method's own level, not
                          halfway inside a loop
    values handed on 15   names one step sets and a later step reads, each
                          counted once; each is a place a correct student's own
                          naming can need the model to grade (3 points each)

A roadmap with a fault (gates.roadmap_faults: it rewrites the teacher's code,
uses something students may not, or its questions are unclear) is capped just
under CUTOFF: it can never make the cut, and is served only while nothing
better exists.

THE CUTOFF. A problem is served the best three roadmaps at or above CUTOFF.
When preparing a problem finds none (main/publish.py), it waits on the
teacher's review instead of going live - measured 14 Oct, that is the long
methods no cut can make small (HW3's calculate, calculateExpressions,
getPostfix). Problems already live are only ever ranked here, never hidden.
"""
import ast

CUTOFF = 60
TOP = 3
_SMALL, _BIG = 8, 20


def _refs(decomposition: dict) -> list[str]:
    return [((c.get("reference") if isinstance(c, dict) else getattr(c, "reference", "")) or "")
            for c in decomposition.get("chunks") or []]


def _code_lines(ref: str) -> int:
    return sum(1 for ln in ref.splitlines() if ln.strip() and not ln.strip().startswith("#"))


def _handed_on(refs: list[str]) -> int:
    """Names set in one step and read in a later one, each counted once - a
    running total carried through four cuts is one value, not four.
    Read off the WHOLE body, by line - a step that carries on inside a loop
    cannot be parsed on its own."""
    lines, step_of = [], []
    for i, r in enumerate(refs):
        for ln in r.split("\n"):
            lines.append(ln)
            step_of.append(i)
    try:
        tree = ast.parse("def _w():\n" + "\n".join("    " + ln for ln in lines))
    except SyntaxError:
        return 0
    stored, loaded = {}, {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and 2 <= n.lineno <= len(lines) + 1:
            step = step_of[n.lineno - 2]
            (stored if isinstance(n.ctx, ast.Store) else loaded).setdefault(n.id, set()).add(step)
    return sum(1 for name, at in stored.items()
               if any(s > min(at) for s in loaded.get(name, ())))


def score(problem: dict, decomposition: dict, wording: bool = True,
          teachers: bool = True) -> dict:
    """{points, parts, sizes, faults}. `wording` False for a cut whose questions
    are not written yet (the splitter's plan): it is ranked on its code alone."""
    from .gates import roadmap_faults
    from .indent import base_indent
    refs = _refs(decomposition)
    sizes = [_code_lines(r) for r in refs] or [0]
    total, big = sum(sizes), max(sizes)
    ideal = min(5, max(2, round(total / 5))) if total >= 2 else 1
    carried = _handed_on(refs)
    parts = {
        "biggest step": 30 if big <= _SMALL else max(0.0, 30 * (_BIG - big) / (_BIG - _SMALL)),
        "balance": (25 * (1 - (big / total - 1 / len(sizes)) / (1 - 1 / len(sizes)))
                    if len(sizes) > 1 and total else 25),
        "number of steps": max(0, 15 - 7 * abs(len(sizes) - ideal)),
        "clean starts": (15 * sum(1 for r in refs[1:] if not base_indent(r)) / (len(refs) - 1)
                         if len(refs) > 1 else 15),
        "values handed on": max(0, 15 - 3 * carried),
    }
    parts = {k: round(v) for k, v in parts.items()}
    faults = roadmap_faults(problem, decomposition, teachers=teachers, wording=wording)
    points = sum(parts.values())
    if faults:
        points = min(points, CUTOFF - 1)
    return {"points": points, "parts": parts, "sizes": sizes, "faults": faults,
            "handed_on": carried}


def _same(a: dict, b: dict) -> bool:
    return [r.strip() for r in _refs(a)] == [r.strip() for r in _refs(b)]


def rank(problem: dict, entries: list, deserialize=lambda e: e) -> list[tuple]:
    """(entry, score) for each DIFFERENT roadmap, best first. Copies - the same
    cut of the same code, saved more than once - count once, as the best of
    them (the same cut can be saved once unclearly worded, once clearly)."""
    scored = [(e, deserialize(e)) for e in entries]
    scored = [(e, d, score(problem, d)) for e, d in scored]
    # A roadmap without a fault always ranks above one with: an unclear
    # question is not made up for by tidy code.
    scored.sort(key=lambda eds: (bool(eds[2]["faults"]), -eds[2]["points"]))
    out, kept = [], []
    for e, d, sc in scored:
        if not any(_same(d, k) for k in kept):
            kept.append(d)
            out.append((e, sc))
    return out


def best(ranked: list[tuple]) -> list:
    """What a student may be served: the best TOP at or above CUTOFF - or, for
    a problem with none (a live one, or one its teacher published anyway), the
    best TOP without a fault, and only failing that the best TOP there are
    (13 Oct: an unclear roadmap only while no clear one exists). Never empty
    while anything is ranked."""
    good = [e for e, s in ranked if s["points"] >= CUTOFF]
    clear = [e for e, s in ranked if not s["faults"]]
    return (good or clear or [e for e, _ in ranked])[:TOP]


def best_points(ranked: list[tuple]) -> int:
    return max((s["points"] for _, s in ranked), default=0)
