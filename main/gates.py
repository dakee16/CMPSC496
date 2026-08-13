"""
gates.py — validation gates a decomposition must clear before it is served.

These run at DECOMPOSITION time, on reference material, never on a student's
answer. Nothing here belongs in the grading path.

Gate 1 — necessity. Full-assembly validation only proves the chunks work
TOGETHER. It cannot tell you whether each chunk actually carries weight. On
Palindrome Number a chunk handling negatives (`if x < 0: return False`) becomes
dead weight the moment a later chunk compares against a reversed `abs(x)` — the
negative case is absorbed downstream, so a student could answer "pass" for that
chunk and still be marked correct. That defeats chunking: an early mistake must
be caught, not silently absorbed.

The check is a knockout. Replace chunk i's reference with a no-op, reassemble
with every other chunk untouched, and run the oracle. A load-bearing chunk's
removal BREAKS the assembly. If the assembly still passes, chunk i was never
load-bearing and the whole decomposition is rejected — not patched — exactly as
a full-assembly gate failure is.

Note the gate is only as sharp as the oracle behind it: a knockout can only be
detected by a test that exercises the removed chunk's code path. Oracle strength
(main/mutation.py) and this gate reinforce each other.

Gate 2 will live here as well.
"""
from .identity import get_resolved_entry
from tests.sandbox import get_oracle_tests, is_oracle_strong, passes_tests

_NOOP = "pass"


def _knock_out(chunk):
    """A copy of `chunk` with its reference replaced by a no-op. Never mutates
    the original — the caller's chunks must survive the check untouched."""
    if hasattr(chunk, "model_copy"):                 # pydantic StepItem
        return chunk.model_copy(update={"reference": _NOOP})
    import copy
    clone = copy.copy(chunk)
    clone.reference = _NOOP
    return clone


def _outcome(code: str, tests: list, entry: str | None) -> tuple[bool, str]:
    """Run a knocked-out assembly. Returns (chunk_was_necessary, description).

    A crash counts as evidence the chunk was necessary, not as an inconclusive
    result: the assembly is broken without it either way."""
    try:
        compile(code, "<knockout>", "exec")
    except SyntaxError as e:
        return True, f"crashed: not valid Python without it ({e})"

    res = passes_tests(code, tests, entry_name=entry)
    if not res["ok"]:
        return True, f"crashed: {res['error']}"
    if res["fraction"] == 1.0:
        return False, f"still passed all {res['total']} oracle tests"
    return True, f"failed {res['total'] - res['passed']}/{res['total']} oracle tests"


def check_necessity(header: str, chunks: list, problem: dict) -> dict:
    """Gate 1. Every chunk's reference must be load-bearing.

    For each chunk in turn: no-op its reference, reassemble with the others
    unchanged, run the oracle. Returns {status, passed, per_chunk, summary}.

    status is one of:
      "pass"             every chunk proved necessary
      "fail"             at least one chunk was not load-bearing
      "skipped"          no oracle tests exist at all (matches _gate_code's
                         long-standing policy for non-JSON-input problems)
      "oracle_not_strong"  PRECONDITION FAILURE — see below

    The precondition matters. A knockout can only be detected by a test that
    exercises the removed chunk's code path, so a weak oracle produces FALSE
    REJECTIONS: a genuinely necessary chunk looks redundant simply because
    nothing tests the case it handles. Measured example — knocking out the
    negative-number chunk of the real Palindrome decomposition fails exactly
    ONE oracle test; against the old all-positive suite that correct
    decomposition would have been thrown away. So Gate 1 refuses to render a
    verdict on an oracle that has not cleared mutation testing. "I cannot
    evaluate this" and "this decomposition is bad" are different answers and
    callers must not conflate them."""
    # Local import: run_phase1 imports this module, so a top-level import here
    # would be circular.
    from .run_phase1 import assemble_references

    tests = get_oracle_tests(problem)
    if not tests:
        return {"status": "skipped", "passed": True, "per_chunk": [],
                "summary": "no oracle tests available — necessity could not be checked"}

    # get_oracle_tests above already validated on a miss, so this reads the
    # stored verdict. False here means the oracle was validated and came back
    # WEAK — regenerating the decomposition cannot fix that.
    if not is_oracle_strong(problem):
        slug = problem.get("slug") or problem.get("title", "<unnamed problem>")
        return {"status": "oracle_not_strong", "passed": False, "per_chunk": [],
                "summary": (
                    f"CANNOT EVALUATE — the oracle for '{slug}' is not "
                    f"mutation-validated strong, so a knockout result would be "
                    f"unreliable: a necessary chunk can look redundant when no "
                    f"test exercises its code path. Strengthen the oracle "
                    f"(main/mutation.py / python -m main.warmup) before gating "
                    f"this decomposition. This is NOT a necessity failure.")}

    entry = get_resolved_entry(problem)["entry_name"]
    per_chunk, dead = [], []

    for i, chunk in enumerate(chunks):
        knocked = [_knock_out(c) if j == i else c for j, c in enumerate(chunks)]
        necessary, detail = _outcome(assemble_references(header, knocked), tests, entry)
        step_id = getattr(chunk, "step_id", f"Part {i + 1}")
        per_chunk.append({"index": i, "step_id": step_id,
                          "necessary": necessary, "outcome": detail})
        if not necessary:
            dead.append((step_id, getattr(chunk, "prompt", "")))

    if not dead:
        return {"status": "pass", "passed": True, "per_chunk": per_chunk,
                "summary": f"all {len(chunks)} chunks are load-bearing"}

    named = ", ".join(s for s, _ in dead)
    summary = (
        f"NECESSITY FAILURE — {named} {'is' if len(dead) == 1 else 'are'} not "
        f"load-bearing. Replacing "
        f"{'its' if len(dead) == 1 else 'their'} reference with `pass` still "
        f"passes every oracle test, which means a student could skip "
        f"{'that chunk' if len(dead) == 1 else 'those chunks'} entirely and "
        f"still be marked correct.\n"
        + "\n".join(f"  - {sid}: \"{prompt[:90]}\"" for sid, prompt in dead)
        + "\n\nRewrite the decomposition so every chunk does work no other chunk "
          "repeats. Do not let a later chunk re-handle a case an earlier chunk "
          "already covers (for example, do not neutralise an earlier sign check "
          "by using abs() downstream)."
    )
    return {"status": "fail", "passed": False, "per_chunk": per_chunk,
            "summary": summary}
