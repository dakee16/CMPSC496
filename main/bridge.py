"""bridge.py - matching a student's variables to the reference's BY VALUE.

THE PROBLEM. A non-final chunk is graded by stitching the student's code to the
teacher's reference code for the remaining steps and running the result. That
works until the student picks different names. The teacher wrote `counts = {}`
and the student wrote `new_dict = {}`; the reference tail reads `counts`, so the
composite dies on NameError and a perfectly correct answer looks broken.

WHAT USED TO HAPPEN. An LLM was asked to REWRITE the teacher's tail in the
student's vocabulary. Measured at ~50% on identical input: across 24 live trials
of one submission it produced a clean rewrite 12 times and a self-referencing
alias the other 12, so WHICH TIER decided a correct answer was a coin flip on
the model's mood. Identical code came back correct at 1am and incorrect at
4:58am.

WHAT HAPPENS HERE. Nothing is rewritten. The teacher's tail is used byte for
byte, and the only thing synthesised is a line like

    counts = new_dict

An identifier-to-identifier assignment cannot count, loop, sum or branch, so -
unlike a rewritten tail - it is structurally incapable of doing the student's
work for them. That is the property that makes this safe, and it is why no
necessity check is needed on this path (see grading._hoistable_declarations for
the one place a gap CAN be filled, and the two rules that bound it).

HOW THE MAPPING IS FOUND: by running both and comparing values, never by
comparing names. The reference's own chunks are executed over the oracle inputs
and every name the tail needs is recorded; the student's accepted prefix plus
this chunk is executed over the SAME inputs; names whose value sequences match
are candidates. Ambiguity is not resolved by guessing - every injective
assignment is tried, in a deterministic order, and the ORACLE decides.

THE ASYMMETRY THAT MAKES IT SOUND. A bridge may ACQUIT, never CONVICT. A found
bridge whose composite passes is proof the student's step works with the rest of
the teacher's solution. A bridge that cannot be found proves nothing at all -
the student may simply be doing something else - so failure here returns
"cannot verify" and the caller must not read it as evidence. Sampling N oracle
inputs and seeing agreement is evidence, not a proof of equivalence, and this
module never pretends otherwise: the evidence is only ever used to acquit.

FOUR THINGS THAT BREAK NAIVE VALUE MATCHING, all found by testing:

  * OBJECTS HAVE ADDRESSES. repr(node) is '<Node object at 0x10033cc20>' and
    differs every run, so a Stack/Node state would never match itself. _ENCODER
    walks objects structurally instead - see it for the cycle and property
    handling.
  * UNBOUND LOCALS. A loop variable does not exist when the loop never runs, so
    one empty-string input raised and destroyed the whole snapshot. Every name
    is captured under its own guard.
  * HASH RANDOMISATION. Set and dict iteration order changes per process, so two
    runs of one program disagree. Pinned in execution._sanitized_env; this
    module additionally sorts sets and dict items when encoding.
  * RECURSION. Appending a capture to a recursive function captures the INNER
    frames: factorial(3) came back {'sub': {'sub': {'sub': 1}}} instead of 2.
    Recursive submissions are refused here rather than mis-measured.

SCOPE. Plain functions AND methods. A method looked out of reach at first - its
state lives partly in `self` across a whole recorded call sequence, and its
entry point is the injected driver rather than the method itself, so there is no
single dictionary of locals at the boundary. Both facts are true and neither
turned out to matter. The capture is appended to the METHOD body, so the driver
records one snapshot per CALL and a signature is taken over (input, call) pairs
instead of over inputs; `self` is the same object on both sides and so never
needs bridging at all, leaving only locals to match. `owners` carries each
position back to the input that produced it, for callers that need to talk about
an input rather than a call.

A method's snapshot leaves OUT OF BAND - appended to a module-level list, with
the driver shadowed by a wrapper that returns it - rather than on the method's
return value. Hijacking the return worked for most methods and was fatal for the
ones whose return type is contractual: `__len__` must return a non-negative int,
so handed a dict it raised on every call and the signature came back unbound at
every position, which does not read as a broken probe but as a student whose
variables hold nothing. A slot is claimed on ENTRY and filled on EXIT, so a call
that returns before the boundary - `pop` on an empty stack - still occupies a
position and the two sides stay in step.

What IS still declined: recursion, in either form - `factorial(n - 1)` or
`self.push(value)` - because the capture would then read an inner frame rather
than the boundary, and `from x import *`, because the probe cannot enumerate
what is in scope. Declining is free: the caller falls through to the tiers
below, which is a false "cannot verify" and costs a student nothing.
"""
import ast
import itertools

from .context import SEQ_ENTRY, build_program, is_method
from .execution import run_student_code

# How many candidate assignments may be tried before giving up. Each one is a
# full oracle run (~30ms locally), and the count is factorial in the number of
# indistinguishable names: 4 names with 4 candidates each is 24 assignments,
# 6 is 720, 8 is 40,320. The cap is a time bound, and hitting it is a
# "cannot verify", never an "incorrect" - search exhaustion is our limit, not
# evidence about the student.
MAX_ASSIGNMENTS = 24

# Marks a capture that actually reached the end of the body. Without it an
# early `return` inside the student's chunk is indistinguishable from a captured
# state: the function returns THEIR value and the probe reads it as the snapshot.
CAPTURE_KEY = "__mt_cap__"
UNBOUND = "__mt_unbound__"

# Injected at MODULE level of the probe program, so it is exempt from the AST
# policy the same way context.SEQ_ENTRY is - see execution._policed_nodes. It
# uses only dir/getattr/type/sorted/repr, none of which the policy bans.
_ENCODER = '''
def _mt_state(v, _d=0, _seen=None):
    """A value as address-free, order-stable, structural text.

    Sets and dict items are SORTED because iteration order is not part of the
    value, and comparing two processes on it would report a difference that does
    not exist. Objects become their class name plus their public non-callable
    attributes, so two Nodes holding the same numbers compare equal even though
    their addresses never will. Cycles - `node.next = node` - terminate on an
    identity set, and a property that raises is recorded as unreadable rather
    than taking the whole snapshot down."""
    if _d > 12:
        return "<deep>"
    if _seen is None:
        _seen = frozenset()
    if v is None or isinstance(v, (bool, int, float, str, bytes)):
        return v
    if id(v) in _seen:
        return "<cycle>"
    _seen = _seen | {id(v)}
    if isinstance(v, (list, tuple)):
        inner = [_mt_state(x, _d + 1, _seen) for x in v]
        return {"t": type(v).__name__, "v": inner}
    if isinstance(v, (set, frozenset)):
        return {"t": "set", "v": sorted(repr(_mt_state(x, _d + 1, _seen)) for x in v)}
    if isinstance(v, dict):
        return {"t": "dict", "v": sorted(
            (repr(k), repr(_mt_state(x, _d + 1, _seen))) for k, x in v.items())}
    fields = []
    for a in sorted(dir(v)):
        if a.startswith("_"):
            continue
        try:
            x = getattr(v, a)
        except Exception:
            fields.append((a, "<unreadable>"))
            continue
        if callable(x):
            continue
        fields.append((a, repr(_mt_state(x, _d + 1, _seen))))
    return {"t": type(v).__name__, "f": fields}
'''


def _indent(code: str, n: int = 4) -> str:
    pad = " " * n
    return "\n".join(pad + ln if ln.strip() else ln for ln in code.splitlines())


def _tree(body: str):
    return ast.parse("def _mt_w():\n" + _indent(body or "pass"))


def _own_scope(nodes):
    """Every node in `nodes`, WITHOUT descending into a nested def or class.

    A nested `def` binds its NAME in this scope and its locals in a different
    one, so the two must not be collected together. The def/class node itself is
    still yielded - the caller wants `.name` - but its body belongs to another
    scope and is not walked."""
    for n in nodes:
        yield n
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        yield from _own_scope(ast.iter_child_nodes(n))


def stores(body: str) -> set:
    """Names this body binds AT ITS OWN SCOPE.

    SCOPE-AWARE, which it was not. `ast.walk` descends into a nested `def`, so a
    student who wrote a helper had that helper's LOCALS collected as their own
    state. Measured live on a step whose body was

        def tally(s):
            running = 0
            for c in s:
                running += 1
            return running
        size = tally(txt)

    which came back {'c', 'running', 'size', 'tally'}. `running` and `c` live in
    tally's frame and do not exist at this boundary at all, so the capture
    appended to the OUTER body recorded them as UNBOUND - and main/diagnose.py
    put `running is __mt_unbound__` in front of the student as their own state.
    Two visible defects from one miscollection, which is why the fix is here
    rather than at either screen.

    The synthetic `_mt_w` wrapper _tree adds needs no special case now: the walk
    starts inside its body, so its name is never in reach."""
    try:
        t = _tree(body)
    except SyntaxError:
        return set()
    out = set()
    for n in _own_scope(t.body[0].body):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            out.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
    return out


def _targets(node) -> set:
    """Names a binding target introduces - a loop variable, a `with ... as`."""
    return {n.id for n in ast.walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}


def _comprehension_vars(node) -> set:
    """Names bound by comprehension generators anywhere inside `node`.

    `[x for x in items]` binds `x` to the comprehension alone - it is not a free
    variable and it cannot collide with anything outside. Missing this reported
    every comprehension variable as a name the bridge had to supply."""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp,
                          ast.GeneratorExp)):
            for gen in n.generators:
                out |= _targets(gen.target)
    return out


def _reads(node) -> set:
    """Every name this node reads, counting an AugAssign target as a read.

    `total += n` marks `total` Store-only in the AST. Reading that literally
    makes an accumulator look self-sufficient, so a tail that accumulates into a
    name the reference declared earlier would never have that name bridged - and
    the same misreading is why grading._hoistable_declarations does not hoist a
    deferred `total = 0` when the tail does `total += n`."""
    out = {n.id for n in ast.walk(node)
           if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    for n in ast.walk(node):
        if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name):
            out.add(n.target.id)
    return out - _comprehension_vars(node)


def free_names(body: str) -> set:
    """Names this body reads BEFORE it binds them - what a bridge must supply.

    Order matters, and a set difference cannot express it. `while lo <= hi:` …
    `lo = mid + 1` both reads and writes `lo`, but the read happens first, on
    the loop test, so `lo` has to come from somewhere: subtracting all stores
    would call that tail self-sufficient and silently bridge nothing.

    Statements are walked in order against a growing `bound` set. A compound
    statement (for/while/if/try/with) is treated atomically - every name inside
    it is tested against what was bound BEFORE it, and its own bindings count
    only afterwards. That is deliberately conservative in the safe direction:
    a loop body can run zero times or many, so a name it binds cannot be assumed
    bound on entry. Over-reporting a free name costs at most a bridge attempt
    that finds nothing; under-reporting one silently drops a name the tail needs.
    """
    try:
        tree = _tree(body)
    except SyntaxError:
        return set()

    free = set()

    def walk(stmts, bound: set) -> set:
        """Record free reads in `stmts`; return what is bound after them."""
        bound = set(bound)
        for s in stmts:
            if isinstance(s, (ast.For, ast.AsyncFor)):
                # The iterable is read in the OUTER scope; the target is bound
                # for the body. `for ch in txt` makes `ch` the loop's own, so it
                # is never a name the bridge has to supply.
                free.update(_reads(s.iter) - bound)
                inner = bound | _targets(s.target)
                walk(s.body, inner)
                walk(s.orelse, inner)
                bound |= _targets(s.target) | _all_stores(s.body)
            elif isinstance(s, ast.While):
                free.update(_reads(s.test) - bound)
                walk(s.body, bound)
                walk(s.orelse, bound)
                bound |= _all_stores(s.body)
            elif isinstance(s, ast.If):
                free.update(_reads(s.test) - bound)
                after_t = walk(s.body, bound)
                after_f = walk(s.orelse, bound) if s.orelse else bound
                # Only what BOTH branches bind is certainly bound afterwards.
                bound |= (after_t & after_f) - bound
            elif isinstance(s, ast.Try):
                walk(s.body, bound)
                for h in s.handlers:
                    if h.type is not None:
                        free.update(_reads(h.type) - bound)
                    walk(h.body, bound | ({h.name} if h.name else set()))
                walk(s.orelse, bound)
                walk(s.finalbody, bound)
                bound |= _all_stores(s.body)
            elif isinstance(s, (ast.With, ast.AsyncWith)):
                for item in s.items:
                    free.update(_reads(item.context_expr) - bound)
                    if item.optional_vars is not None:
                        bound |= _targets(item.optional_vars)
                walk(s.body, bound)
                bound |= _all_stores(s.body)
            elif isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef,
                                ast.ClassDef)):
                # Its body does not run here, and its free variables are closed
                # over rather than read at this point. Only the name is bound.
                bound.add(s.name)
            else:
                free.update(_reads(s) - bound)
                bound |= _all_stores(s)
        return bound

    walk(tree.body[0].body, set())
    return free


def _all_stores(nodes) -> set:
    """Every name bound anywhere inside a node or list of nodes."""
    if not isinstance(nodes, list):
        nodes = [nodes]
    out = set()
    for node in nodes:
        out |= {n.id for n in ast.walk(node)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        for n in ast.walk(node):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.ClassDef)) and n.name != "_mt_w":
                out.add(n.name)
    return out


def calls(body: str, name: str) -> bool:
    """Does this body call `name`? Used only to detect recursion.

    BOTH call shapes, because a method recurses through an attribute:
    `factorial(n - 1)` is a Name, `self.push(value)` is an Attribute, and
    matching only the first let a recursive method through to a capture that
    would have read an inner frame."""
    if not name:
        return False
    try:
        t = _tree(body)
    except SyntaxError:
        return False
    for n in ast.walk(t):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        if isinstance(f, ast.Name) and f.id == name:
            return True
        if isinstance(f, ast.Attribute) and f.attr == name:
            return True
    return False


def has_star_import(body: str) -> bool:
    try:
        t = _tree(body)
    except SyntaxError:
        return False
    return any(isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names)
               for n in ast.walk(t))


def is_applicable(problem: dict, entry: str, bodies: list) -> bool:
    """Can value-matching say anything trustworthy about this submission?

    Declines anything RECURSIVE (the capture would read an inner frame instead
    of the boundary: factorial(3) came back {'sub': {'sub': {'sub': 1}}}) and
    anything importing * (the probe cannot enumerate what is in scope).

    A METHOD IS FINE, which it was not at first. The objection was that its
    state spans `self` across a whole recorded call sequence and its entry point
    is the injected driver rather than the method - so there is no single
    boundary dictionary. Both are true and neither matters: the capture is
    appended to the METHOD body, so the driver records one snapshot per CALL,
    and the signature is simply taken over (input, call) pairs instead of over
    inputs. `self` needs no bridging at all - it is the same object on both
    sides, and only LOCALS can differ in name."""
    for b in bodies:
        if b and (calls(b, entry) or has_star_import(b)):
            return False
        # A method's own name is not `entry` (that is the driver), so recursion
        # inside one has to be looked for separately.
        if b and is_method(problem) and calls(b, _method_name(problem)):
            return False
    return True


def _method_name(problem: dict) -> str:
    """The name of the method being written, or "" for a plain function."""
    for line in reversed((problem.get("context_prefix") or "").splitlines()):
        s = line.strip()
        if s.startswith(("def ", "async def ")):
            return s.split("(")[0].split()[-1]
    return ""


def loop_targets(body: str) -> set:
    """Names bound only as a loop or comprehension variable.

    Not state a student would describe as theirs - `ch` in `for ch in txt` is
    machinery, and showing it back to them beside their real variables is noise.
    main/diagnose.py uses this to decide what is worth putting on screen."""
    try:
        t = _tree(body)
    except SyntaxError:
        return set()
    out = set(_comprehension_vars(t))
    for n in ast.walk(t):
        if isinstance(n, (ast.For, ast.AsyncFor)):
            out |= _targets(n.target)
    return out - (stores(body) - out)


def _capture_program(problem: dict, header: str, body: str, names: list,
                     human: bool = False) -> str:
    """`body`, then a guarded snapshot of `names`, as a runnable program.

    Each name is captured in its OWN try/except. A single shared try would let
    one unbound loop variable - on the empty-string input, say - discard the
    whole snapshot, which is how this was first written and why every signature
    came back empty."""
    # `human` swaps the structural encoding for plain repr. The encoding exists
    # to make two PROCESSES comparable; it is unreadable to a person, and a
    # student shown {'t': 'dict', 'v': [("'a'", '1')]} has been handed noise
    # instead of their own {'a': 1}. Matching uses the encoded form, the tutor
    # uses this one, and neither is asked to do the other's job.
    enc = "repr({n})" if human else "_mt_state({n})"
    lines = [f"{CAPTURE_KEY} = {{}}"]
    for n in sorted(names):
        lines.append(f"try:\n    {CAPTURE_KEY}[{n!r}] = " + enc.format(n=n) + "\n"
                     f"except Exception:\n    {CAPTURE_KEY}[{n!r}] = {UNBOUND!r}")

    if not is_method(problem):
        lines.append(f"return {{{CAPTURE_KEY!r}: {CAPTURE_KEY}}}")
        full = (body.rstrip() + "\n" if body.strip() else "") + "\n".join(lines)
        return _ENCODER + "\n" + build_program(problem, full, header)

    # A METHOD SENDS ITS SNAPSHOT OUT OF BAND, and does not touch its own return
    # value. Hijacking the return worked for most methods and was fatal for the
    # ones whose return type is contractual: `__len__` must return a
    # non-negative int, so handed a dict it raised on EVERY call and the
    # signature came back unbound at every position - which does not read as a
    # broken probe, it reads as a student whose variables hold nothing.
    #
    # Appending to a module-level list instead leaves the method's real
    # behaviour completely intact, which also means the rest of the recorded
    # call sequence behaves as it actually would. The driver is then shadowed by
    # a wrapper that runs the original and hands back the log. Redefining it is
    # safe: `_mt_run_calls` is skipped by the AST policy by MODULE-LEVEL name
    # (execution._policed_nodes), and this definition is module-level, so the
    # wrapper is exempt exactly as the original is.
    # A SLOT IS CLAIMED ON ENTRY AND FILLED ON EXIT, so a call that returns
    # before the boundary still occupies a position. Appending only at the end
    # would silently shorten the log for whichever side returned early - `pop`
    # on an empty stack does exactly that - and the two signatures would then be
    # compared out of step, matching one call against another.
    lines.append(f"_mt_log[-1] = {{{CAPTURE_KEY!r}: {CAPTURE_KEY}}}")
    full = ("_mt_log.append(None)\n"
            + (body.rstrip() + "\n" if body.strip() else "")
            + "\n".join(lines))
    wrapper = f'''

_mt_inner = {SEQ_ENTRY}


def {SEQ_ENTRY}(calls):
    """Replay the sequence for its side effects, then return what was seen."""
    del _mt_log[:]
    try:
        _mt_inner(calls)
    except Exception:
        pass
    return list(_mt_log)
'''
    return (_ENCODER + "\n_mt_log = []\n"
            + build_program(problem, full, header) + wrapper)


def capture(problem: dict, header: str, body: str, names: list,
            inputs: list, entry: str, human: bool = False,
            with_owners: bool = False):
    """{name: value-signature across every input}, or None if unusable.

    A signature is the tuple of that name's encoded value on each input, so two
    names match only when they agree EVERYWHERE, not on a lucky case. An input
    on which the body returned early contributes the returned value instead,
    which keeps a partial match from looking total."""
    if not names:
        return ({}, []) if with_owners else {}
    status, results, _ = run_student_code(
        _capture_program(problem, header, body, names, human), inputs,
        entry_name=entry)
    if status != "ok" or len(results) != len(inputs):
        return (None, []) if with_owners else None
    # A METHOD yields one snapshot per CALL, not one per input: the driver
    # replays a recorded sequence against a single instance and returns the list
    # of per-call results, so the capture appears wherever that method was
    # called. Flattening those into the same position list makes a method's
    # signature the same kind of object as a function's - a tuple compared
    # element by element - and the alignment holds because both sides replay the
    # identical sequence. A position where the method was not called, or where
    # the body returned before reaching the end, records None on BOTH sides.
    # `owners[p]` is the index of the input that produced signature position p.
    # For a function that is the identity; for a method one input contributes as
    # many positions as it has calls, so anything wanting to talk about an INPUT
    # - main/diagnose.py picking a counterexample - needs the map rather than
    # assuming position == input.
    per_input, owners = [], []
    for idx_in, r in enumerate(results):
        if isinstance(r, list):
            for x in r:
                per_input.append(x[CAPTURE_KEY] if isinstance(x, dict)
                                 and CAPTURE_KEY in x else None)
                owners.append(idx_in)
        elif isinstance(r, dict) and CAPTURE_KEY in r:
            per_input.append(r[CAPTURE_KEY]); owners.append(idx_in)
        else:
            per_input.append(None); owners.append(idx_in)   # returned early
    # In human mode the child already returned repr(value), so re-repring it
    # here would show a student their dict wrapped in quotes. Matching needs the
    # extra repr to make the encoded structures comparable as text.
    if human:
        sig = {n: tuple((row or {}).get(n, UNBOUND) for row in per_input)
               for n in names}
    else:
        sig = {n: tuple(repr((row or {}).get(n, UNBOUND)) for row in per_input)
               for n in names}
    return (sig, owners) if with_owners else sig


def _candidates(ref_sig: dict, stu_sig: dict) -> dict:
    """{reference name: [student names whose signature matches exactly]}.

    A name whose value is UNBOUND on every input matches nothing: an absent
    variable is not a value, and treating two absences as equal would bridge
    names that were never there."""
    out = {}
    for r, sig in ref_sig.items():
        if all(v == repr(UNBOUND) for v in sig):
            out[r] = []
            continue
        out[r] = sorted(s for s, ss in stu_sig.items() if ss == sig)
    return out


def assignments(cands: dict, limit: int = MAX_ASSIGNMENTS) -> list[dict]:
    """Injective reference->student mappings, deterministically ordered, capped.

    Sorted names and sorted candidate lists make the order a function of the
    submission alone, so the same code always produces the same verdict - which
    was the whole complaint. Two names with identical signatures are
    interchangeable in the composite anyway; the enumeration exists for the
    cases where they are not (one of them aliases a parameter, say)."""
    keys = sorted(cands)
    if not keys or any(not cands[k] for k in keys):
        return []
    out = []
    for combo in itertools.product(*(cands[k] for k in keys)):
        if len(set(combo)) != len(combo):
            continue                        # not injective
        out.append(dict(zip(keys, combo)))
        if len(out) >= limit:
            break
    return out


def find(problem: dict, header: str, chunks: list, idx: int, upto: str,
         tests: list, entry: str, ambient: set) -> dict | None:
    """Search for a bridge that makes the student's work pass the real oracle.

    `ambient` is everything already in scope without being bridged - the
    function's parameters, module-level names, safe builtins - supplied by the
    caller so this module never has to import the grader back.

    EVERY BOUNDARY AT OR AFTER `idx` IS TRIED, not just idx, and that is what
    handles a student who is AHEAD. Someone who did step 2's work inside step 1
    has a state that matches no name at boundary 1 and every name at boundary 2;
    matching only at their current step would call that correct answer a
    divergence. The first boundary that yields a passing composite wins, and
    `boundary` comes back in the result so the caller can advance the session
    past the steps they have already written.

    Returns None when no bridge is found. THAT IS NOT EVIDENCE OF ANYTHING. It
    means this mechanism could not confirm the submission - the student may be
    correct by another route, may be ahead in a way the oracle cannot see, or
    may simply have hit the assignment cap. The caller must route it to "cannot
    verify", never to "incorrect"."""
    from .execution import classify_run              # local: avoid a cycle

    refs = [(c.get("reference") or "") for c in chunks]
    if not is_applicable(problem, entry, [upto] + refs):
        return None
    inputs = [t["input"] for t in tests]
    if not inputs:
        return None

    stu_sig = capture(problem, header, upto, sorted(stores(upto)), inputs, entry)
    if stu_sig is None:
        return None                 # their code did not run cleanly; not ours to judge

    for boundary in range(idx, len(refs) - 1):
        tail = "\n".join(r for r in refs[boundary + 1:] if r.strip())
        if not tail.strip():
            continue
        needed = sorted(free_names(tail) - ambient)
        if not needed:
            continue                # the tail is self-sufficient; nothing to bridge
        ref_sig = capture(problem, header,
                          "\n".join(r for r in refs[:boundary + 1] if r.strip()),
                          needed, inputs, entry)
        if ref_sig is None:
            continue
        for mapping in assignments(_candidates(ref_sig, stu_sig)):
            body = "\n".join(p for p in (upto, bridge_lines(mapping), tail)
                             if p.strip())
            res = classify_run(build_program(problem, body, header), tests,
                               entry_name=entry)
            if res.outcome == "pass":
                return {"boundary": boundary, "mapping": mapping,
                        "bridge": bridge_lines(mapping),
                        "ahead": boundary > idx}
    return None


def bridge_lines(mapping: dict) -> str:
    """The mapping as ONE simultaneous assignment.

    `lo, hi = hi, lo`, never `lo = hi` then `hi = lo`. Sequential assignment
    clobbers: a student who used the reference's own two names with their roles
    swapped - correct code - scored 2/4 sequentially and 4/4 simultaneously.
    Under the acquit-only rule that mistake would have shown up as a false
    "cannot verify" rather than a false "incorrect", but it would have thrown
    away a correct answer either way."""
    pairs = [(r, s) for r, s in sorted(mapping.items()) if r != s]
    if not pairs:
        return ""
    return (", ".join(r for r, _ in pairs) + " = "
            + ", ".join(s for _, s in pairs))


if __name__ == "__main__":
    # Self-check.  python -m main.bridge
    # Runs real subprocesses (the capture has to be real to mean anything) but
    # never a model. Every case below is one that was actually measured.
    AMB = {"txt", "nums", "d", "len", "range", "set", "sorted", "sum", "max",
           "min", "enumerate", "str", "int", "list", "dict", "print", "abs"}

    # ── pure AST helpers ────────────────────────────────────────────────
    assert stores("a = 1\nfor b in x:\n    pass") == {"a", "b"}
    # A NESTED DEF IS A DIFFERENT SCOPE. Its name is bound here; its locals are
    # not, and collecting them made diagnose show a helper's `running` as the
    # student's own variable - holding UNBOUND, because the outer frame never
    # had it. Both of those reached a student's screen.
    _helper = ("def tally(s):\n"
               "    running = 0\n"
               "    for c in s:\n"
               "        running += 1\n"
               "    return running\n"
               "size = tally(txt)")
    assert stores(_helper) == {"tally", "size"}, stores(_helper)
    # A class body is a scope too, and its methods are scopes inside that.
    assert stores("class Box:\n    n = 0\n\n"
                  "    def put(self, v):\n        self.v = v\n"
                  "b = Box()") == {"Box", "b"}
    # ...and the ordinary case is unchanged: a comprehension and a `with` are
    # NOT separate scopes for this purpose, and a lambda binds only its name.
    assert stores("out = [y for y in xs]") == {"out", "y"}
    assert stores("with open(p) as fh:\n    data = fh.read()") == {"fh", "data"}
    assert stores("f = lambda q: q + 1") == {"f"}
    # AugAssign is a READ as well as a write - the bug that hides an accumulator
    assert "total" in free_names("total += n")
    # Read on the loop test, written in the body: `lo` must still be free, or a
    # binary-search tail looks self-sufficient and nothing gets bridged.
    _bs = free_names("while lo <= hi:\n    mid = (lo + hi) // 2\n    lo = mid + 1")
    assert {"lo", "hi"} <= _bs, _bs
    # ...but a name bound BEFORE it is read is not free.
    assert "lo" not in free_names("lo = 0\nwhile lo <= hi:\n    lo += 1")
    assert "total" not in free_names("total = 0\ntotal += n")
    assert calls("sub = factorial(n - 1)", "factorial")
    assert not calls("if n == 0:\n    return 1", "factorial")
    assert has_star_import("from math import *")
    assert not has_star_import("from math import sqrt")

    # Simultaneous, never sequential: `lo = hi` then `hi = lo` loses hi.
    assert bridge_lines({"lo": "hi", "hi": "lo"}) == "hi, lo = lo, hi"
    assert bridge_lines({"counts": "new_dict"}) == "counts = new_dict"
    assert bridge_lines({"counts": "counts"}) == ""      # identity needs no line

    # Enumeration is injective, deterministic and capped.
    assert assignments({"a": ["x"], "b": ["x"]}) == []    # only x: not injective
    assert assignments({"a": ["x", "y"], "b": ["x", "y"]}) == [
        {"a": "x", "b": "y"}, {"a": "y", "b": "x"}]
    assert assignments({"a": ["x"], "b": []}) == []       # one name unmatched
    _big = {c: [f"s{i}" for i in range(8)] for c in "abcdefgh"}
    assert len(assignments(_big)) == MAX_ASSIGNMENTS      # capped, never 40,320

    # ── the real thing, end to end ──────────────────────────────────────
    H, E = "def frequency(txt):", "frequency"
    REFS = ["counts = {}",
            "for ch in txt:\n    if ch.isalpha():\n        counts[ch] = counts.get(ch, 0) + 1",
            "return counts"]
    CH = [{"reference": r} for r in REFS]
    T = [{"input": ["hello"], "expected": {"h": 1, "e": 1, "l": 2, "o": 1}},
         {"input": ["aab"], "expected": {"a": 2, "b": 1}},
         {"input": [""], "expected": {}}]
    P = {"slug": "frequency", "entry_hint": "frequency",
         "description": "Count letters.",
         "solution": H + "\n" + _indent("\n".join(REFS))}

    # THE MOTIVATING FAILURE: a pure rename. This is the submission that came
    # back correct at 1am and incorrect at 4:58am.
    got = find(P, H, CH, 0, "new_dict = {}", T, E, AMB)
    assert got and got["mapping"] == {"counts": "new_dict"}, got
    assert got["boundary"] == 0 and not got["ahead"], got

    # ...and it is DETERMINISTIC, which was the entire complaint.
    for _ in range(3):
        assert find(P, H, CH, 0, "new_dict = {}", T, E, AMB) == got

    # A decoy with an identical signature is resolved by trying both and letting
    # the oracle decide, not by guessing from the names.
    assert find(P, H, CH, 0, "new_dict = {}\nseen = {}", T, E, AMB)

    # AHEAD: step 2's work done inside step 1 matches a LATER boundary.
    _ahead = find(P, H, CH, 0,
                  "out = {}\nfor ch in txt:\n    if ch.isalpha():\n"
                  "        out[ch] = out.get(ch, 0) + 1", T, E, AMB)
    assert _ahead and _ahead["ahead"] and _ahead["boundary"] == 1, _ahead

    # A GENUINELY WRONG submission finds no bridge - and that is NOT a
    # conviction, it is the caller's cue to say "cannot verify".
    assert find(P, H, CH, 0, "new_dict = []", T, E, AMB) is None
    # Neither does an empty one.
    assert find(P, H, CH, 0, "pass", T, E, AMB) is None

    # ── the four traps ──────────────────────────────────────────────────
    # 1. An unbound loop variable on the empty input must not destroy the
    #    snapshot: `ch` does not exist when txt is "".
    _sig = capture(P, H, "acc = {}\nfor ch in txt:\n    acc[ch] = 1",
                   ["acc", "ch"], [t["input"] for t in T], E)
    assert _sig is not None and len(_sig["acc"]) == 3, _sig
    assert _sig["ch"][2] == repr(UNBOUND), _sig["ch"]     # "" leaves it unbound
    assert _sig["acc"][0] != _sig["acc"][1], "distinct inputs, distinct state"

    # 2. Objects encode structurally - no addresses, stable across processes.
    OH, OE = "def build(n):", "build"
    OP = {"slug": "o", "entry_hint": "build",
          "solution": "class Node:\n    def __init__(self, v):\n        self.value = v\n"
                      "        self.next = None\n\n" + OH + "\n    pass"}
    _o = capture(OP, "class Node:\n    def __init__(self, v):\n        self.value = v\n"
                     "        self.next = None\n\n" + OH,
                 "node = Node(n)\nnode.next = node", ["node"], [[1], [2]], OE)
    assert _o is not None and "0x" not in _o["node"][0], _o
    assert "cycle" in _o["node"][0], "a self-referencing node must terminate"
    assert _o["node"][0] != _o["node"][1], "different values, different signature"
    for _ in range(3):                       # stable across processes
        assert capture(OP, "class Node:\n    def __init__(self, v):\n"
                           "        self.value = v\n        self.next = None\n\n" + OH,
                       "node = Node(n)\nnode.next = node", ["node"],
                       [[1], [2]], OE) == _o

    # 3. RECURSION is refused, not mis-measured. factorial(3) would capture
    #    {'sub': {'sub': {'sub': 1}}} - an inner frame, not the boundary state.
    assert not is_applicable({"slug": "f"}, "factorial",
                             ["sub = factorial(n - 1)"])
    # 4. A METHOD IS IN SCOPE. The capture is appended to the method body, so
    #    the driver records one snapshot per CALL and the signature is taken
    #    over (input, call) pairs. `self` never needs bridging - it is the same
    #    object on both sides - so only locals are matched.
    _meth = {"slug": "m", "group_title": "S", "context_indent": 8,
             "context_suffix": "\n",
             "context_prefix": "class Node:\n    def __init__(self, v):\n"
                               "        self.value = v\n        self.next = None\n\n"
                               "class S:\n    def __init__(self):\n"
                               "        self.top = None\n\n"
                               "    def push(self, value):\n",
             "solution": "pass"}
    assert is_applicable(_meth, "_mt_run_calls", ["n = Node(value)"])
    # A DUNDER WITH A RETURN-TYPE CONTRACT IS FINE NOW. It was not when the
    # snapshot rode out on the return value: `__len__` must return a
    # non-negative int, so handed a dict it raised on every call and the
    # signature came back unbound everywhere - which reads as "this student's
    # variables hold nothing", the most dangerous shape of wrong answer this
    # module could produce. The method's own return value is untouched today.
    _dunder = {**_meth, "context_prefix":
               _meth["context_prefix"].replace("def push(self, value):",
                                               "def __len__(self):")}
    assert _method_name(_dunder) == "__len__"
    assert is_applicable(_dunder, "_mt_run_calls", ["count = 0"])
    # ...but recursion inside one is still refused, and a method's own name is
    #    not the entry point, so it has to be looked for separately.
    assert _method_name(_meth) == "push", _method_name(_meth)
    assert not is_applicable(_meth, "_mt_run_calls", ["self.push(value)"])

    _mrefs = ["node = Node(value)\nnode.next = self.top", "self.top = node"]
    _mch = [{"reference": r} for r in _mrefs]
    _mtests = [{"input": [[["new"], ["push", 2], ["push", 4]]],
                "expected": [None, None, None]}]
    # A renamed LOCAL inside a method bridges exactly as one in a function does.
    _mgot = find(_meth, "def push(self, value):", _mch, 0,
                 "n = Node(value)\nn.next = self.top", _mtests,
                 "_mt_run_calls", {"self", "value", "Node", "S"})
    assert _mgot and _mgot["mapping"] == {"node": "n"}, _mgot

    # ...and an ordinary plain function is in scope.
    assert is_applicable(P, E, ["counts = {}"])

    print("bridge.py self-check OK")
