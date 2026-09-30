"""
splitter.py - roadmaps cut from the teacher's own code, with no model deciding
where the cuts go.

WHY. The model-built roadmaps for two problems had one giant step every time:
measured on the server (28 Sep), all 8 for calculateExpressions and all 7 for
calculator-get-postfix had a step of 35-48 lines. calculateExpressions is one
`for` loop that is 88% of the code, so no balanced split exists unless a step
starts INSIDE the loop body - and the model never cut there. Cutting the
teacher's code at statement boundaries found 61 balanced splits of that
problem which pass every serve gate. This module is that search, made
permanent: fill_pool uses it whenever a model roadmap has a step over
MAX_STEP_LINES or cannot be built at all, and main.resplit_pools used it once
to replace the oversized roadmaps already saved.

WHAT IS AND IS NOT A MODEL CALL. Where to cut is decided here, deterministically,
from the teacher's own statements - nothing to hallucinate, and every candidate
goes through gates.assert_serveable, the same serve boundary as everything
else. Only the wording of each step's question is written by a model
(write_prompts), because it is prose; it is then checked by the plain-code
prompt gate (gates.check_prompts), exactly as the builder's own prompts are.

HOW A SPLIT IS CHOSEN.
  * Cuts go only where a statement starts, at any depth, never before
    `elif`/`else`/`except`/`finally` (a step cannot begin mid-clause).
  * Every step must be seatable (its first line is its shallowest - see
    indent.is_seatable) and every prefix must be a program (shape gate).
  * The FEWEST steps whose every step fits in MAX_STEP_LINES win, topped up
    from ONE more step when that count yields fewer than five. When no
    split of up to MAX_STEPS steps fits, the smallest possible biggest step
    wins, then the fewest steps. Among equals, the most even split goes first.
"""
import ast
import json

from .context import header_of, solution_body
from .gates import assert_serveable, check_prompts
from .identity import get_resolved_entry
from .indent import base_indent, is_seatable
from .ollama_client import DECOMPOSE_MODEL, chat
from .prompts import SPLIT_PROMPTS_SYSTEM
from .schemas import StepItem

# No step may be longer than this (code lines: not blank, not comments).
# 14 is the biggest step any model roadmap has for the 11 problems that were
# fine; the three that were not had 20-48.
MAX_STEP_LINES = 20
MAX_STEPS = 5
# How many candidates to put through the serve gate (which runs the oracle)
# for one step count before trying the next. Bounds a teacher's upload time.
_GATE_BUDGET = 40
_CLAUSE_WORDS = ("elif", "else", "except", "finally")


def step_lines(reference: str) -> int:
    """Lines that do something - blank lines and comments do not count."""
    return sum(1 for ln in (reference or "").splitlines()
               if ln.strip() and not ln.strip().startswith("#"))


def biggest_step(decomposition: dict) -> int:
    chunks = (decomposition or {}).get("chunks") or []
    return max((step_lines(getattr(c, "reference", None) if not isinstance(c, dict)
                           else c.get("reference")) for c in chunks), default=0)


def too_big(decomposition: dict) -> bool:
    return biggest_step(decomposition) > MAX_STEP_LINES


def header_for(problem: dict) -> str:
    """The def line the student works under - the builder's own rule."""
    resolved = get_resolved_entry(problem)
    return header_of(problem) or \
        f"def {resolved['entry_name'] or 'solve'}({', '.join(resolved['params'])}):"


def _cut_points(lines: list[str]) -> list[int]:
    """Line indices where a step may begin: every statement start, at any
    depth, except the first line and clause lines."""
    src = "def _w():\n" + "\n".join("    " + ln if ln.strip() else ln for ln in lines)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    starts = {n.lineno - 2 for n in ast.walk(tree)
              if isinstance(n, ast.stmt) and n.lineno >= 2}
    return sorted(i for i in starts
                  if 0 < i < len(lines)
                  and not lines[i].strip().startswith(_CLAUSE_WORDS))


def _candidates(lines: list[str], k: int, limit: int) -> list[list[int]]:
    """Every way to cut `lines` into k steps, each 1..limit code lines and
    seatable, as lists of start indices (the first is always 0)."""
    cuts = _cut_points(lines)
    code = [0]
    for ln in lines:
        code.append(code[-1] + (1 if ln.strip() and not ln.strip().startswith("#") else 0))
    n = len(lines)
    seat = {}

    def ok(a, b):
        size = code[b] - code[a]
        if not 1 <= size <= limit:
            return False
        if (a, b) not in seat:
            seat[(a, b)] = is_seatable("\n".join(lines[a:b]))
        return seat[(a, b)]

    out = []

    def walk(start, left, acc):
        if left == 1:
            if ok(start, n):
                out.append(acc)
            return
        for c in cuts:
            if c <= start:
                continue
            if code[c] - code[start] > limit:
                break
            if ok(start, c):
                walk(c, left - 1, acc + [c])
    walk(0, k, [0])

    def evenness(starts):
        sizes = [code[b] - code[a] for a, b in zip(starts, starts[1:] + [n])]
        return (max(sizes), sum(s * s for s in sizes))
    return sorted(out, key=evenness)


def _chunks(lines: list[str], starts: list[int], prompts=None) -> list[StepItem]:
    bounds = starts + [len(lines)]
    refs = ["\n".join(lines[a:b]).rstrip() for a, b in zip(bounds, bounds[1:])]
    return [StepItem(question_id="split", step_id=f"Part {i + 1}",
                     prompt=(prompts[i] if prompts else f"Step {i + 1}."),
                     expected_type="code", reference=r)
            for i, r in enumerate(refs)]


def plan(problem: dict, want: int = 5) -> list[dict]:
    """Up to `want` decompositions of the teacher's own code that pass the
    serve gate, per the rules in the module docstring. Prompts are
    placeholders here - write_prompts fills them. No model call."""
    header = header_for(problem)
    lines = [ln.rstrip() for ln in solution_body(problem).splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    total = sum(1 for ln in lines if ln.strip() and not ln.strip().startswith("#"))
    if total < 2:
        return []

    def passing(k, limit):
        found = []
        for starts in _candidates(lines, k, limit)[:_GATE_BUDGET]:
            decomp = {"header": header, "chunks": _chunks(lines, starts)}
            try:
                assert_serveable(problem, decomp)
            except Exception:
                continue
            found.append(decomp)
            if len(found) >= want:
                break
        return found

    # Fewest steps first; a step count that yields fewer than `want` passing
    # splits is topped up from the next one, so a problem still gets variety.
    for limit in [MAX_STEP_LINES] + list(range(MAX_STEP_LINES + 1, total + 1)):
        found, first = [], None
        for k in range(2, MAX_STEPS + 1):
            if first is not None and k > first + 1:
                break             # topping up adds at most one step: more only
                                  # chops the setup into 2-line pieces
            got = passing(k, limit)[:want - len(found)]
            if got and first is None:
                first = k
            found += got
            if len(found) >= want:
                break
        if found:                 # within MAX_STEP_LINES, else the smallest biggest step
            return found
    return []


def write_prompts(problem: dict, decomposition: dict, tries: int = 3) -> dict:
    """The question a student sees for each step - the ONE model call here.
    Checked by the prompt gate; a failure buys a retry with the gate's own
    words (two retries: measured, one was not enough for a setup step), and
    after that the working split is kept with its best wording, as the builder
    does (a split that is right beats no split over phrasing)."""
    chunks = decomposition["chunks"]
    # A step that starts indented continues a block an earlier step opened -
    # in these problems, the loop. Said here, because the model did not infer
    # it: measured, none of the first worded in-loop steps said "for each".
    steps = "\n\n".join(
        f"STEP {i + 1} CODE"
        + (" (RUNS ONCE PER ITEM - inside the repetition an earlier step started)"
           if base_indent(c.reference) else "")
        # ...and the step that STARTS that repetition, which was worded as
        # set-up and answered as set-up (student report, 30 Sep).
        + (" (STARTS A REPETITION - the next step carries on inside it)"
           if i + 1 < len(chunks)
           and base_indent(chunks[i + 1].reference) > base_indent(c.reference) else "")
        + f":\n{c.reference}" for i, c in enumerate(chunks))
    feedback = ""
    for _ in range(tries):
        user = (f"PROBLEM:\n{(problem.get('description') or '')[:1500]}\n\n"
                f"The function header is: {decomposition['header']}\n\n"
                f"This solution is already split into {len(chunks)} steps, in "
                f"order. Write the question a student sees for each step.\n\n"
                f"{steps}\n\n"
                + (f"YOUR LAST PROMPTS WERE REJECTED:\n{feedback}\n\n" if feedback else "")
                + f'Return JSON only: {{"prompts": [...]}} with exactly '
                  f"{len(chunks)} strings, one per step, in order.")
        raw = chat(DECOMPOSE_MODEL, SPLIT_PROMPTS_SYSTEM,
                   [{"role": "user", "content": user}], temperature=0.2, fmt="json")
        try:
            prompts = [str(p).strip() for p in json.loads(raw).get("prompts", [])]
        except (ValueError, AttributeError):
            continue
        if len(prompts) != len(chunks) or not all(prompts):
            feedback = f"You returned {len(prompts)} prompts; exactly {len(chunks)} are needed."
            continue
        chunks = [StepItem(**{**c.model_dump(), "prompt": p}) for c, p in zip(chunks, prompts)]
        gate = check_prompts(chunks, problem)
        if gate["status"] == "pass":
            break
        feedback = gate["summary"]
    return {**decomposition, "chunks": chunks}


def build(problem: dict, want: int = 5) -> list[dict]:
    """`want` served-ready decompositions cut from the teacher's code, worded.
    Model calls: write_prompts only, at most 2 per decomposition."""
    return [write_prompts(problem, d) for d in plan(problem, want)]
