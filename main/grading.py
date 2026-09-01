"""
grading.py - the answer-checking state machine.

One orchestration function, grade_submission(), owns every verdict. Routes must
not re-implement any part of it.

Two rules shape the whole design:
  * Execution decides, opinion is last. A verdict is deterministic only when a
    real run attributed the fault to the student.
  * Our failure is never evidence about the student. Infrastructure trouble
    returns `indeterminate` and costs no attempt - it never defaults to wrong.
"""
import ast
import json
import re

from .execution import classify_run
from .identity import get_resolved_entry
from .indent import align_to_chunk
from .ollama_client import GRADING_MODEL, chat
from . import trace
from .schemas import GradeResult
from .sessions import MAX_ATTEMPTS, accepted_prefix, problem_of

MAX_ADAPT_TRIES = 2


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


def _assemble(header: str, *bodies: str) -> str:
    body = "\n".join(b.rstrip() for b in bodies if b and b.strip())
    return header + "\n" + (_indent(body) if body.strip() else "    pass")


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


def _scope_violation(student_code: str, in_scope: set) -> tuple[str, str] | None:
    """Deterministic pre-LLM gate: every name this step READS must resolve to
    something real - a function parameter, a variable an earlier accepted step
    produced, or a safe builtin. A step that reads an undefined name (a typo, or
    the wrong parameter name) is the student's own error and is failed here,
    before any model is consulted. Without this, the resulting NameError surfaces
    later as `ownership ambiguous` and is handed to the LLM judge, which grades
    intent and green-lights nonsense like `max(list)`.

    Returns (student_message, reason_code) on a violation, else None. A parse
    failure returns None - syntax is classified elsewhere."""
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
        return (f"This step uses `{bare[0]}` as a value, but that is Python's "
                f"built-in type. Did you mean one of the function's parameters?",
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
    if _names(tail, ast.Store) & current_outputs:
        return False
    # Must not simply be the reference solution pasted back in.
    body = re.sub(r"\s+", "", tail)
    if body and body in re.sub(r"\s+", "", solution or ""):
        return False
    return True


def _calibrate(header, trusted_prefix, alias_lines, tail, tests, entry):
    """An adapter must prove itself on TRUSTED work before it may judge a
    student. A random LLM tail that fails proves nothing about the student:
    it may simply be a broken tail. Only a tail that passes here has earned
    the right to produce a verdict."""
    cand = _assemble(header, trusted_prefix, "\n".join(alias_lines), tail)
    return classify_run(cand, tests, entry_name=entry).outcome == "pass"


# ── Tier 4: dual LLM judge ───────────────────────────────────────────────

_JUDGE_SYSTEM = (
    "You judge ONE step of a student's partial solution. Return STRICT JSON: "
    '{"correct": true/false, "reason": "<one sentence for the student>", '
    '"confidence": 0.0-1.0, "evidence_category": "<short label>"}. '
    "Never quote the reference solution, hidden tests, or internal code in reason.")


def _ask_judge(payload: str, role: str):
    raw = chat(GRADING_MODEL, _JUDGE_SYSTEM + f"\nYou are the {role}.",
               [{"role": "user", "content": payload}], temperature=0, fmt="json")
    d = json.loads(raw)
    return (bool(d["correct"]), str(d.get("reason", ""))[:300],
            float(d.get("confidence", 0.0)), str(d.get("evidence_category", ""))[:60])


def _tier4(problem, chunk, upto, student_code, why, evidence, corr=None) -> GradeResult:
    payload = (f"PROBLEM:\n{(problem.get('description') or '')[:600]}\n\n"
               f"STEP ASKED:\n{chunk['prompt']}\n\n"
               f"ACCEPTED SO FAR:\n{upto}\n\n"
               f"STUDENT'S ANSWER FOR THIS STEP:\n{student_code}\n\n"
               f"PRIVATE REFERENCE FOR THIS STEP:\n{chunk.get('reference','')}\n\n"
               f"EXECUTION EVIDENCE: {evidence}\nWHY EXECUTION WAS INCONCLUSIVE: {why}")
    try:
        with trace.model_call(corr, GRADING_MODEL, "judge", role="primary"):
            a_ok, a_reason, a_conf, a_cat = _ask_judge(payload, "primary judge")
        _trace(trace.record_judge, corr, GRADING_MODEL, "primary", a_ok, a_conf)
        with trace.model_call(corr, GRADING_MODEL, "judge", role="verifier"):
            b_ok, b_reason, b_conf, b_cat = _ask_judge(
                payload + f"\n\nPRIMARY JUDGMENT: correct={a_ok} reason={a_reason}",
                "independent verifier")
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
    _trace(trace.record_route, corr, "llm-judge", "correct" if a_ok else "incorrect")
    return _ok("correct" if a_ok else "incorrect", "llm-judge", a_reason,
               f"judge_{a_cat or 'agreed'}", deterministic=False)


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
    probe = classify_run(_assemble(header, upto), [], entry_name=entry)
    if probe.outcome == "policy_violation":
        return _ok("incorrect", "policy",
                   f"That answer uses something not allowed here - "
                   f"{probe.internal_error}.", "policy_violation",
                   execution_outcome="policy_violation")
    if (probe.internal_error or "").startswith("syntax:"):
        return _ok("incorrect", "syntax",
                   f"Your code doesn't parse: {probe.internal_error[7:].strip()}.",
                   "syntax_error")

    # ── SCOPE GATE - names the step READS must already exist. Deterministic,
    #    runs before any execution tier or LLM. Catches the wrong parameter
    #    name / typo that would otherwise crash and be excused as our fault. ──
    in_scope = _names(prefix, ast.Store) | set(resolved["params"])
    scope = _scope_violation(student_code, in_scope)
    if scope is not None:
        return _ok("incorrect", "syntax", scope[0], scope[1])

    # ── LAST CHUNK - whole function, no borrowed tail ──
    if is_last:
        res = classify_run(_assemble(header, upto), tests, entry_name=entry)
        if res.outcome == "pass":
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
        return _ok("incorrect", "execution-final",
                   msg.get(res.outcome, "Your solution didn't pass."),
                   f"final_{res.outcome}", execution_outcome=res.outcome,
                   failures=res.failures)

    # ── NON-LAST - trusted reference tail ──
    ref_tail = "\n".join((chunks[j].get("reference") or "")
                         for j in range(idx + 1, len(chunks)))
    res = classify_run(_assemble(header, upto, ref_tail), tests, entry_name=entry)
    if res.outcome == "pass":
        return _ok("correct", "execution-reference",
                   "Correct - your step works with the rest of the solution.",
                   "reference_pass", execution_outcome="pass")
    if res.outcome == "harness_error":
        return _system("harness_error", res.internal_error)
    if res.outcome == "policy_violation":
        return _ok("incorrect", "policy", "That answer uses something not "
                   "allowed here.", "policy_violation",
                   execution_outcome="policy_violation")

    # A fixed reference tail can fail purely because it expected different
    # variable names. That is not the student's fault, so we do NOT convict
    # here - we try a calibrated adapter instead.
    return _tier3(problem, session, chunk, header, prefix, student_code, upto,
                  ref_tail, tests, entry, res, corr)


def _tier3(problem, session, chunk, header, prefix, student_code, upto,
           ref_tail, tests, entry, ref_res, corr=None) -> GradeResult:
    """Adapt the tail to the student's interface - but only a CALIBRATED
    adapter may influence a verdict."""
    idx = session["index"]
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
        if not _calibrate(header, trusted_prefix, alias_lines, tail, tests, entry):
            _trace(trace.record_adapter, corr, GRADING_MODEL, attempt, "calibration_failed")
            continue                                # uncalibrated: prove nothing

        # The alias bridge belongs to CALIBRATION only. It maps the trusted
        # reference's names onto the tail's interface; the student already
        # produces those names, so injecting it here would assign from an
        # undefined reference name and raise NameError on every run.
        cand = classify_run(_assemble(header, upto, tail), tests, entry_name=entry)
        if cand.outcome == "pass":
            # ANTI-BYPASS - knock out ONLY the student chunk; if it still
            # passes, the tail was doing the student's work for them.
            ko = classify_run(_assemble(header, prefix, "pass", tail),
                              tests, entry_name=entry)
            if ko.outcome == "pass":
                _trace(trace.record_adapter, corr, GRADING_MODEL, attempt, "bypass_rejected")
                continue                            # bypassing adapter: reject
            _trace(trace.record_adapter, corr, GRADING_MODEL, attempt, "accepted")
            _trace(trace.record_route, corr, "execution-adapted", "correct")
            return _ok("correct", "execution-adapted",
                       "Correct - your approach differs from ours, but it works.",
                       "adapted_pass", execution_outcome="pass", divergent=True)
        if cand.outcome == "wrong_output":
            # Calibrated tail + clean run + wrong answers => the student's step
            # is genuinely wrong. This is the only adapter path that convicts.
            return _ok("incorrect", "execution-adapted",
                       "Your step runs, but the finished solution gives the "
                       "wrong answer.", "adapted_wrong_output",
                       execution_outcome="wrong_output", failures=cand.failures)
        if cand.outcome == "harness_error":
            return _system("harness_error", cand.internal_error)
        break     # crash/timeout: ownership ambiguous -> Tier 4

    return _tier4(problem, chunk, upto, student_code,
                  "no calibrated adapter produced attributable evidence", evidence,
                  corr)


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

    print("grading.py scope-gate self-check OK")
