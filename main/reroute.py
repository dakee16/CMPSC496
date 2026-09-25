"""reroute.py - a roadmap for a student whose approach is not the teacher's.

THE PROBLEM. The chunks a student is graded against are cut from the TEACHER'S
solution. A student who plans recursion and is handed a roadmap cut from a loop
is being asked, step by step, for code they never intended to write - and every
tier in main/grading.py will correctly decline to confirm any of it. Their plan
was reviewed and approved, and then quietly stopped mattering.

WHAT THIS DOES, AND THE LINE IT DOES NOT CROSS. It regenerates the ROADMAP -
the sequence of steps - and never the ground truth. The oracle is untouched: it
is the teacher's, it was mutation-validated, and it remains the only thing that
decides whether code is correct. A model proposes a solution in the student's
own shape and that proposal is then made to EARN its place:

    passes the hidden tests, 100%        it is a correct solution - the oracle
                                          said so, not the model
    splits into chunks that reassemble   the steps are a real decomposition
    every chunk is load-bearing          no step can be skipped
    every chunk is answerable            gates.shape_failures - see it for the
                                          two ways a split cannot be answered

Fail any of those and there is no roadmap, and the student is told that in
those terms. A generated roadmap that clears them all is verified by exactly the
standard the teacher's own decomposition is held to, which is what makes it safe
to grade against.

WHY THIS IS NOT /replan. That route existed, served regenerated steps, and was
disabled with a 410 (frontend/api_server.py) because it skipped these gates -
it built a problem dict with no solution, so the oracle came back empty and
"skipped" was accepted as success. Everything here runs through
gates.assert_serveable, which refuses on a missing or weak oracle rather than
treating either as a pass.

THE MODEL IS USED TWICE AND DECIDES NOTHING. Once to read the student's plan
graph and write a solution in its shape; once, inside the existing decomposer,
to split it. Execution checks both. The graph itself is model-extracted from the
student's own words (main/graphs.py, which notes that it "never gates anything")
- and that stays true here, because a bad extraction produces a solution that
either fails the oracle and is thrown away, or passes it and is therefore
correct regardless of how faithfully it read them.
"""
import json
import time

from .gates import shape_failures
from .ollama_client import GRADING_MODEL, chat


def follows_reference(problem: dict, header: str, graph: dict) -> bool:
    """Does the student's plan have the same SHAPE as the teacher's solution?

    Compares control-flow kinds only - loop, branch, return - never labels,
    because one side is prose and the other is Python. That is the same
    comparison graphs.compare() makes between a plan and submitted code, used
    here one stage earlier, against the teacher's code instead of the student's.

    THIS IS THE ROUTER, AND IT IS DELIBERATELY NOT THE REVIEWER.
    main/design_review.py is SOLUTION-BLIND on purpose - its own prompt says so,
    because "a reviewer holding one answer rejects the others", and a plan is
    approved on whether it is coherent and workable. This function does see the
    solution, and it never speaks to the student and never blocks approval. It
    only picks WHICH roadmap to build.

    Errs towards True - "same route" - because being wrong that way costs
    nothing: the student gets the teacher's roadmap, and if their code really
    does diverge the grader declines to confirm it and the tutor asks, exactly
    as it does for anyone else. Being wrong the other way spends a minute of
    model work rebuilding a roadmap they did not need."""
    from .context import solution_body
    from .graphs import _signature, code_graph

    try:
        body = solution_body(problem) if problem.get("solution") else ""
        if not body.strip():
            return True
        theirs = _signature(graph or {})
        ours = _signature(code_graph(body, header))
    except Exception:
        return True                     # cannot tell -> do not rebuild anything
    if not theirs or not ours:
        return True
    # CONTROL FLOW ONLY. A plan is a sketch: nobody writes "and then I assign it
    # to a variable", so the teacher's graph carries `step` nodes for every
    # assignment that a prose plan will never mention. Comparing those made
    # every plan look divergent. What actually distinguishes two ROUTES is
    # whether they loop, whether they branch, and whether they return - so
    # recursion planned against a loop shows up as a missing `loop`, which is
    # precisely the case this exists for.
    flow = ("loop", "branch", "return")
    if not (set(ours) & set(flow)) or not (set(theirs) & set(flow)):
        # Nothing to compare. A solution that does not parse comes back as a
        # single `step` node labelled "does not parse yet" rather than as an
        # error, so an emptiness check on the NODES is not enough - it has to be
        # on the control flow. Without this, an unreadable teacher solution
        # looked like a divergent student. A plan with no control flow at all is
        # too thin to count, and counting it would reroute every terse plan.
        return True
    # HOW MANY, not just WHICH. Comparing the SET of kinds was too blunt to see
    # the case this exists for: measured on the real `invert` problem, a student
    # who planned two parallel lists, a rescan-to-count and index matching
    # reduced to {branch, loop, return} - and so did the teacher's dictionary of
    # counts. Same vocabulary, different algorithm, and the router called it the
    # same route and handed over the teacher's roadmap. With counts, that plan
    # reads (2 branches, 1 loop, 1 return) against the teacher's (1, 2, 1) and
    # is correctly seen as divergent.
    #
    # STILL ERRS TOWARDS "SAME ROUTE" on everything it cannot see. A plan is
    # prose and prose is lossy - measured on a real one, the plan said ONE loop
    # where the student's own code went on to have THREE, because nobody writes
    # "and then I nest a second loop inside the first". So counts here are a
    # floor, not a description: they catch a plan that is plainly a different
    # shape and will still miss a terse plan that happens to compress to the
    # teacher's numbers. Being wrong that way costs a rebuilt roadmap the
    # student did not need, which the oracle gate makes harmless; the code-side
    # check in grading.py is what catches what prose cannot express.
    return [theirs.count(k) for k in flow] == [ours.count(k) for k in flow]


def diverges_from_roadmap(chunks: list, idx: int, student_code: str,
                          header: str) -> bool:
    """Is what they have WRITTEN a different shape from the roadmap they are on?

    THE ONE COMPARISON THAT CAN BE STRICT, because both sides are code. The
    plan-side check in follows_reference() has to stay loose: a plan is prose,
    and measured on a real one the student wrote "scan the whole list and count"
    where their own code went on to have THREE loops. Nobody writes "and then I
    nest a second loop". So counts there are a floor.

    Here there is no prose. Their submission is compared against the teacher's
    reference for the steps up to and including this one - like with like, same
    point in the solution - so the counts mean what they say. Measured on the
    real `invert` roadmap: the student who kept a `unique_values` list reads
    (2 branches, 2 loops) against the teacher's step 1 (0, 1) and is seen as
    divergent, while an ordinary answer and the same answer WITH ITS VARIABLES
    RENAMED both read (0, 1) and are left alone. Renaming cannot move control
    flow, which is what makes this safe to be strict about.

    CALLED ONLY WHEN GRADING COULD NOT CONFIRM THE STEP, never on its own. That
    is the whole safety story: "could not confirm" already means no tier could
    attribute anything, which is exactly the state a genuinely different
    approach produces. A submission that was ACCEPTED is never asked - and it
    matters that it is not, because a correct answer can still be a different
    shape (the same student's accepted attempt reads (1, 2) against (0, 1)).

    Errs to False - "same shape" - on everything it cannot read, for the same
    reason follows_reference errs to True: a rebuild nobody needed costs a
    minute, and refusing to rebuild costs nothing that was not already lost."""
    flow = ("loop", "branch", "return")
    try:
        from .graphs import _signature, code_graph
        # THIS STEP AGAINST THIS STEP. Comparing their submission to the
        # reference for every chunk UP TO here read a perfectly ordinary step 2
        # as divergent - one step of theirs against two of the teacher's is not
        # like with like. A student whose step-1 answer also does step 2's work
        # never reaches this: being ahead is something bridge.find ACCEPTS.
        chunk = chunks[idx] if chunks and 0 <= idx < len(chunks) else None
        mine_src = (chunk.get("reference") if isinstance(chunk, dict)
                    else getattr(chunk, "reference", "")) or ""
        if not (student_code or "").strip() or not mine_src.strip():
            return False
        mine = _signature(code_graph(mine_src, header))
        yours = _signature(code_graph(student_code, header))
    except Exception:
        return False
    if not (set(yours) & set(flow)):
        # THEIR side is what has to be readable. An answer with no loop, branch
        # or return says nothing about their approach - it is far likelier to be
        # unfinished than to be a different algorithm, and rebuilding a roadmap
        # around it spends a minute to learn nothing.
        return False
    return [yours.count(k) for k in flow] != [mine.count(k) for k in flow]


# HOW LONG ONE OPEN MAY SPEND HERE. build()'s own docstring has always said
# the honest failure should be fast rather than a long spinner ending in the
# same sentence - and the code did not hold to it: three attempts of a proposal
# plus a five-try decomposition is up to EIGHTEEN sequential model calls, which
# measured at minutes on one student's click. Nothing else cuts it off either;
# there is no timeout in Caddy or uvicorn, so the browser simply span.
#
# Checked between attempts, so the true ceiling is this plus the attempt in
# flight. The inner decomposer also gets fewer tries than its own default,
# because its retries are the bulk of those eighteen calls.
BUDGET_SECONDS = 60.0
DECOMPOSE_TRIES = 2

# A FAILURE IS REMEMBERED, so one slow open is not every open. What a reroute
# attempt depends on is this problem and the SHAPE of the plan, and all the
# expensive work happens BEFORE a session exists - so nothing downstream
# recorded that it had already been tried, and every reopen paid for it again.
# That is what turned one Start over into a multi-minute wait on every later
# open of the same problem.
#
# Failures only, and only ones our own machinery did not cause: a SUCCESS ends
# in a session that reopening resumes instead of coming back here, and a model
# outage is not a fact about this plan (see the transient flag in build). The
# consequence of a remembered failure is that the student keeps the teacher's
# roadmap, which is exactly the behaviour before any of this existed.
# Bounded, and in-process is sufficient - one uvicorn worker, see start.sh.
_NO_ROUTE: dict[tuple, str] = {}
_NO_ROUTE_LIMIT = 512


def _route_key(problem: dict, graph: dict):
    """What a reroute outcome depends on, or None when it cannot be keyed."""
    try:
        from .graphs import _signature
        from .identity import content_hash
        return (content_hash(problem), tuple(_signature(graph or {})))
    except Exception:
        return None


def _remember_no_route(key, reason: str) -> None:
    if key is None:
        return
    if len(_NO_ROUTE) >= _NO_ROUTE_LIMIT:
        _NO_ROUTE.clear()           # cheap bound; correctness never depends
    _NO_ROUTE[key] = reason


class RouteUnavailable(Exception):
    """No roadmap could be built for this approach.

    Carries `reason` for logs. What a STUDENT is told is the sentence in
    student_message(), which blames the tooling rather than their idea - it is
    our decomposition that failed, and saying otherwise would tell a student
    with a perfectly good unusual approach that their thinking was rejected."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def student_message() -> str:
    """What the student sees when no roadmap could be built.

    Never "your method is not viable". A failure here is ours - the approach may
    be excellent and simply not something we can cut into checkable steps - and
    the two options below are both real: the second is the STRONGEST check in
    the system, not a consolation prize."""
    return ("We can't break your approach into steps to check as you go. "
            "That's a limit on our side, not a problem with your idea. "
            "You can write the whole solution and we'll check it against the "
            "real tests, or talk it through with the tutor and map it out "
            "together.")


_SYSTEM = """\
You are writing a reference solution to a programming problem, IN THE SHAPE the
student has already planned. Their plan was reviewed and approved; your job is
to express it as working code, not to improve on it or replace it.

RULES:
- Follow their plan's structure. If they planned recursion, write recursion. If
  they planned two passes, write two passes. A solution that ignores their shape
  is useless here even if it is correct.
- Write the FUNCTION BODY only - no def line, no imports, no comments, no
  markdown, no explanation. The def line already exists and is given to you.
- USE THE PARAMETER NAMES FROM THAT def LINE. They are the names the student is
  writing under; inventing your own produces code that refers to variables which
  do not exist.
- FOR RECURSION, CALL THE FUNCTION BY ITS OWN NAME, from the given def line. It
  is already bound - a function can always call itself - so you never need to
  define a second one. Writing `def helper(...)` and recursing into that is
  wrong here: it is a whole function, not a body, and the outer one would then
  return nothing.
- It must be complete and correct on its own.

Return STRICT JSON: {"body": "<the function body, newline separated>"}"""


def effective_header(problem: dict) -> str:
    """The def line the student is actually writing under. NEVER "".

    `context.header_of` returns the empty string for a plain function - it is
    only meant to name a METHOD's own def line - and a caller that passes that
    straight to build_program gets the body back with NO def line wrapped round
    it. The consequences are not cosmetic:

      * the assembled "solution" is whatever the model wrote at module level,
      * so a proposal that renamed the function to `letter_count` became a
        module defining `letter_count`,
      * and sandbox.resolve_entry, finding no `frequency`, falls back to the
        last function defined - and cheerfully scored the rename 10/10.

    Measured exactly that way: Gate 1 reported 10/10 for a body that is 0/10
    under the real header, and a roadmap built on `text`/`letter_count` was
    served to a student writing `def frequency(txt)`. The oracle gate is the
    entire safety argument of this module, so the header it runs under cannot
    be left to a function that returns "" by design.

    Built the same way run_phase1.decompose_into_chunks builds it, so a
    regenerated route and the teacher's are assembled identically."""
    from .context import header_of
    from .identity import get_resolved_entry

    head = header_of(problem)
    if head.strip():
        return head
    resolved = get_resolved_entry(problem)
    name = resolved["entry_name"] or "solve"
    return f"def {name}({', '.join(resolved['params'])}):"


def _defines_a_function(body: str) -> bool:
    """Does this 'body' actually open with a def or class?

    The prompt asks for a function BODY and says so three times; models write a
    whole function anyway. Under a correct header that merely nests - harmless
    but useless, since the outer function then returns None and the oracle
    rejects it - so this is belt to the header's braces. Worth having both:
    the failure it guards is silent, and the check is free."""
    import ast
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return False
    return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.ClassDef)) for n in tree.body)


def _plan_text(graph: dict) -> str:
    """The student's plan graph as the ordered prose they put into it."""
    nodes = (graph or {}).get("nodes") or []
    return "\n".join(f"{i + 1}. [{n.get('kind', 'step')}] {n.get('label', '')}"
                     for i, n in enumerate(nodes)
                     if n.get("kind") not in ("start", "end"))


def propose_solution(problem: dict, header: str, graph: dict,
                     chat_log: list | None = None) -> str:
    """A candidate body in the student's shape. UNVERIFIED - the caller gates it."""
    plan = _plan_text(graph)
    said = "\n".join(m.get("content", "")[:300] for m in (chat_log or [])
                     if m.get("role") == "user")[:1500]
    user = (f"PROBLEM:\n{(problem.get('description') or '')[:900]}\n\n"
            f"THE def LINE THAT ALREADY EXISTS - use its name and its "
            f"parameters:\n{header}\n\n"
            f"THE STUDENT'S APPROVED PLAN:\n{plan}\n\n"
            + (f"WHAT THEY SAID ABOUT IT:\n{said}\n\n" if said.strip() else "")
            + "Write the body that implements THEIR plan.")
    raw = chat(GRADING_MODEL, _SYSTEM, [{"role": "user", "content": user}],
               temperature=0, fmt="json")
    return str(json.loads(raw).get("body", "")).strip("\n")


def build(problem: dict, header: str, graph: dict, chat_log: list | None = None,
          max_tries: int = 3, budget_seconds: float = BUDGET_SECONDS):
    """A gated decomposition following the student's plan.

    Raises RouteUnavailable if none can be produced. Every attempt is checked by
    EXECUTION, never by the model that wrote it:

      1. the proposed body passes the hidden tests 100%   (tests/sandbox)
      2. its decomposition assembles, is load-bearing and
         is answerable                                    (gates.assert_serveable
                                                           + shape_failures)

    `max_tries` is small on purpose. This runs while a student waits, and each
    attempt is a solution proposal plus a full decomposition with its own
    internal retries - so the honest failure is fast rather than a long spinner
    ending in the same sentence. `budget_seconds` is what makes that true: the
    count alone bounded the number of model calls at eighteen and their wall
    clock at nothing. A failure is remembered (_NO_ROUTE) so the next open of
    the same problem with the same plan shape is free."""
    # Local imports: run_phase1 imports gates, and this module is imported from
    # the route layer, so top-level imports here would close a cycle.
    from .context import build_program
    from .run_phase1 import decompose_into_chunks
    from tests.sandbox import get_oracle_tests, passes_tests
    from .identity import get_resolved_entry

    # ALREADY KNOWN NOT TO WORK? Checked before the oracle read, which is the
    # first thing here that costs anything.
    key = _route_key(problem, graph)
    if key is not None and key in _NO_ROUTE:
        raise RouteUnavailable(f"{_NO_ROUTE[key]} (already tried)")

    tests = get_oracle_tests(problem)
    if not tests:
        raise RouteUnavailable("no oracle tests; nothing could be verified")
    entry = get_resolved_entry(problem)["entry_name"]
    # NOT the caller's header - see effective_header. An empty one silently
    # turns the oracle gate below into a test of whatever the model happened to
    # name its function.
    header = effective_header(problem)

    last, transient = "no attempt completed", False
    deadline = time.monotonic() + budget_seconds
    for attempt in range(1, max_tries + 1):
        # BETWEEN ATTEMPTS, not inside one: an attempt that has already paid
        # for a proposal may as well finish and be gated.
        if time.monotonic() >= deadline:
            last = f"ran out of time after {attempt - 1} attempt(s): {last}"
            break
        try:
            body = propose_solution(problem, header, graph, chat_log)
        except Exception as e:
            # OUR trouble, not this plan's: a provider outage says nothing
            # about whether this approach can be cut into steps, so it must not
            # be the reason a student is pinned to the teacher's roadmap for
            # the rest of the process's life.
            last, transient = f"proposal failed: {e!r}"[:200], True
            continue
        if not body.strip():
            last = "proposal was empty"
            continue
        if _defines_a_function(body):
            last = "proposal was a whole function, not a body"
            continue

        # GATE 1 - THE ORACLE. This is the whole safety argument: a solution
        # that passes the teacher's mutation-validated tests is correct, and it
        # is the tests that say so. Nothing downstream may soften this.
        res = passes_tests(build_program(problem, body, header), tests,
                           entry_name=entry)
        if not (res["ok"] and res["fraction"] == 1.0):
            last = (f"proposal failed the oracle "
                    f"({res.get('passed', 0)}/{res.get('total', 0)})")
            continue

        # GATE 2 - THE SAME DECOMPOSER, ON THEIR SOLUTION. Handing it `solution`
        # is what makes this their route rather than the teacher's: the
        # decomposer splits the code it is given.
        try:
            # `oracle_from`: gate with the TEACHER'S tests, never a new oracle
            # built from this solution - see tests/sandbox._borrowed_tests for
            # the $325 day that doing otherwise produced.
            decomp = decompose_into_chunks(
                {**problem, "solution": build_program(problem, body, header),
                 "oracle_from": problem},
                max_tries=DECOMPOSE_TRIES)
        except Exception as e:
            last = f"could not be split into steps: {e!r}"[:200]
            continue

        # GATE 3 - ANSWERABLE. decompose_into_chunks already applies this, but
        # it is re-checked here because a roadmap that reaches a student
        # unanswerable costs them an attempt on code they typed correctly.
        shape = shape_failures(problem, decomp.get("header", header),
                               decomp.get("chunks") or [])
        if shape:
            last = f"split is not answerable: {shape[0]}"[:200]
            continue
        return decomp

    if not transient:
        _remember_no_route(key, last)
    raise RouteUnavailable(last)


if __name__ == "__main__":
    # Self-check.  python -m main.reroute
    # Pure: no model, no oracle, no subprocess. The gated path needs all three
    # and is exercised against the real frequency oracle in the scratch runs.
    _g = {"nodes": [{"id": "s", "kind": "start", "label": "begin"},
                    {"id": "a", "kind": "branch", "label": "if n is 0, answer 1"},
                    {"id": "b", "kind": "step",
                     "label": "otherwise multiply n by the answer for n-1"},
                    {"id": "r", "kind": "return", "label": "give that back"},
                    {"id": "e", "kind": "end", "label": "done"}]}
    _t = _plan_text(_g)
    assert "if n is 0" in _t and "multiply n" in _t, _t
    # start/end are scaffolding in the graph schema, not steps the student wrote.
    assert "begin" not in _t and "done" not in _t, _t
    assert _plan_text({}) == "" and _plan_text(None) == ""

    # The student-facing sentence must never blame their approach. A student
    # with a good unusual idea being told it "is not viable" is the same class
    # of error as marking correct code wrong, aimed at their thinking.
    _m = student_message()
    for _bad in ("not viable", "not possible", "invalid", "wrong", "unsupported",
                 "can't be solved", "your method"):
        assert _bad not in _m.lower(), _bad
    assert "limit on our side" in _m and "real tests" in _m, _m

    _e = RouteUnavailable("oracle said 3/10")
    assert _e.reason == "oracle said 3/10"

    # ── the router ───────────────────────────────────────────────────────
    _loop_prob = {"slug": "f", "entry_hint": "frequency",
                  "solution": "def frequency(txt):\n    counts = {}\n"
                              "    for ch in txt:\n        if ch.isalpha():\n"
                              "            counts[ch] = counts.get(ch, 0) + 1\n"
                              "    return counts"}
    _hdr = "def frequency(txt):"
    # A plan with the same shape as the teacher's - loop, branch, return.
    _same = {"nodes": [{"id": "a", "kind": "loop", "label": "go through the text"},
                       {"id": "b", "kind": "branch", "label": "if it is a letter"},
                       {"id": "c", "kind": "return", "label": "give back the tally"}]}
    assert follows_reference(_loop_prob, _hdr, _same) is True
    # A RECURSIVE plan against an iterative solution has no loop at all. This is
    # the case the whole module exists for.
    _rec = {"nodes": [{"id": "a", "kind": "branch", "label": "if the text is empty"},
                      {"id": "b", "kind": "step", "label": "solve the rest, then add one"},
                      {"id": "c", "kind": "return", "label": "give that back"}]}
    assert follows_reference(_loop_prob, _hdr, _rec) is False

    # EVERY uncertainty answers "same route", because being wrong that way costs
    # a student nothing - they get the teacher's roadmap and the ordinary tiers.
    assert follows_reference({"slug": "x"}, _hdr, _rec) is True   # no solution
    assert follows_reference(_loop_prob, _hdr, {}) is True        # no plan
    assert follows_reference(_loop_prob, _hdr, None) is True
    assert follows_reference({"slug": "x", "solution": "!!not python"},
                             _hdr, _rec) is True

    # ── WHAT THEY WROTE, AGAINST THE STEP THEY WROTE IT FOR ──────────────
    # diverges_from_roadmap is the strict half, and it may only ever be strict
    # because both sides are code. Cases measured on the real `invert` roadmap.
    _H = "def invert(d):"
    _CH = [{"reference": "counts = {}\nfor value in d.values():\n"
                         "    counts[value] = counts.get(value, 0) + 1"},
           {"reference": "result = {}\nfor key, value in d.items():\n"
                         "    if counts[value] == 1:\n        result[value] = key"},
           {"reference": "return result"}]
    # The reported case: two parallel lists, a rescan to count, a unique list.
    _theirs = ("keys = list(d.keys())\nvalues = list(d.values())\n"
               "unique_values = []\nfor v in values:\n    count = 0\n"
               "    for other in values:\n        if other == v:\n"
               "            count += 1\n    if count == 1:\n"
               "        unique_values.append(v)")
    assert diverges_from_roadmap(_CH, 0, _theirs, _H)
    # ...and everything ordinary is left alone. RENAMING especially: it cannot
    # move control flow, which is what makes strictness safe here.
    for _idx, _code in [
            (0, "counts = {}\nfor v in d.values():\n"
                "    counts[v] = counts.get(v, 0) + 1"),
            (0, "tally = {}\nfor item in d.values():\n"
                "    tally[item] = tally.get(item, 0) + 1"),
            # one step of theirs against ONE of the teacher's, not all of them
            # so far - comparing against chunks[:idx+1] read this as divergent.
            (1, "result = {}\nfor k, v in d.items():\n"
                "    if counts[v] == 1:\n        result[v] = k"),
            (2, "return result"),
            (0, "counts = dict()"),      # no control flow: too thin to judge
            (0, ""),
    ]:
        assert not diverges_from_roadmap(_CH, _idx, _code, _H), _code

    print("reroute.py self-check OK")
