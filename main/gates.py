"""
gates.py - validation gates a decomposition must clear before it is served.

These run at DECOMPOSITION time, on reference material, never on a student's
answer. Nothing here belongs in the grading path.

Gate 1 - necessity. Full-assembly validation only proves the chunks work
TOGETHER. It cannot tell you whether each chunk actually carries weight. On
Palindrome Number a chunk handling negatives (`if x < 0: return False`) becomes
dead weight the moment a later chunk compares against a reversed `abs(x)` - the
negative case is absorbed downstream, so a student could answer "pass" for that
chunk and still be marked correct. That defeats chunking: an early mistake must
be caught, not silently absorbed.

The check is a knockout. Replace chunk i's reference with a no-op, reassemble
with every other chunk untouched, and run the oracle. A load-bearing chunk's
removal BREAKS the assembly. If the assembly still passes, chunk i was never
load-bearing and the whole decomposition is rejected - not patched - exactly as
a full-assembly gate failure is.

Note the gate is only as sharp as the oracle behind it: a knockout can only be
detected by a test that exercises the removed chunk's code path. Oracle strength
(main/mutation.py) and this gate reinforce each other.

Gate 2 will live here as well.
"""
from .identity import get_resolved_entry
from tests.sandbox import get_oracle_tests, is_oracle_strong, passes_tests

_NOOP = "pass"

# A decomposition below this is degenerate, not "small". The whole point of
# chunking is that a student's early mistake is caught rather than absorbed.
_MIN_CHUNKS = 2

# References that do nothing. Such a chunk cannot possibly be load-bearing, so
# it would fail necessity anyway - catching it here gives a precise error
# instead of a confusing knockout result.
_NOOP_REFERENCES = {"", "pass", "...", "None", "return"}


def _is_noop_reference(ref: str) -> bool:
    """True if the reference is empty or does nothing once comments are gone."""
    body = "\n".join(ln for ln in (ref or "").splitlines()
                     if ln.strip() and not ln.strip().startswith("#"))
    return body.strip() in _NOOP_REFERENCES


def assert_serveable(problem: dict, decomposition: dict) -> dict:
    """THE serve boundary. Nothing reaches a student except through here.

    Gating used to live inside one generator, so every other path that produced
    a decomposition - the best-effort fallback, replan - served ungated. The
    shared mechanism was the status string "skipped" meaning "accept" in those
    places while meaning "block" inside decompose_into_chunks. This function is
    the single place that meaning is decided, and here "skipped"/no-oracle is
    always FAILURE.

    Enforced in order, raising a typed exception on the first failure:
      a. the oracle exists (NoOracleTestsError) and is STRONG
         (OracleNotStrongError) - STRONG per is_oracle_strong, which now keys
         off kill_rate_direct; never reimplemented here
      b. at least _MIN_CHUNKS chunks (DecompositionUnavailableError)
      c. no chunk reference is a no-op (DecompositionUnavailableError)
      d. check_necessity passes (typed by its own status)

    Returns `decomposition` unchanged so callers can `return assert_serveable(...)`."""
    # Local import: run_phase1 imports this module, so a top-level import here
    # would be circular - same reason as the assemble_references import below.
    from .run_phase1 import (DecompositionUnavailableError, NoOracleTestsError,
                             OracleNotStrongError)

    slug = problem.get("slug") or problem.get("title") or "<unnamed problem>"
    chunks = (decomposition or {}).get("chunks") or []
    header = (decomposition or {}).get("header", "")

    # (a) An oracle that does not exist cannot certify anything. This is the
    #     case that used to arrive as "skipped" and get accepted.
    if not get_oracle_tests(problem):
        raise NoOracleTestsError(
            f"'{slug}' has no oracle tests, so nothing about this decomposition "
            f"has actually been verified. Refusing to serve unverified material.")
    if not is_oracle_strong(problem):
        raise OracleNotStrongError(
            f"'{slug}' has an oracle that did not clear mutation testing, so a "
            f"necessity verdict on it would be unreliable. Strengthen the oracle "
            f"(python -m main.warmup) before serving this decomposition.")

    # (b) Degenerate output.
    if len(chunks) < _MIN_CHUNKS:
        raise DecompositionUnavailableError(
            f"'{slug}' produced {len(chunks)} chunk(s); at least {_MIN_CHUNKS} "
            f"are required. A single chunk is the whole problem restated, not a "
            f"decomposition.")

    # (c) Do-nothing references.
    noop = [getattr(c, "step_id", f"Part {i + 1}") for i, c in enumerate(chunks)
            if _is_noop_reference(getattr(c, "reference", ""))]
    if noop:
        raise DecompositionUnavailableError(
            f"'{slug}' has chunk(s) with a do-nothing reference: {', '.join(noop)}. "
            f"A chunk that does nothing can never be load-bearing.")

    # (d) Necessity. Its own statuses carry the right exception type.
    nec = check_necessity(header, chunks, problem)
    if nec["status"] == "oracle_not_strong":
        raise OracleNotStrongError(nec["summary"])
    if nec["status"] == "skipped":
        raise NoOracleTestsError(nec["summary"])
    if nec["status"] != "pass":
        raise DecompositionUnavailableError(nec["summary"])

    return decomposition


def _knock_out(chunk):
    """A copy of `chunk` with its reference replaced by a no-op. Never mutates
    the original - the caller's chunks must survive the check untouched."""
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
      "oracle_not_strong"  PRECONDITION FAILURE - see below

    The precondition matters. A knockout can only be detected by a test that
    exercises the removed chunk's code path, so a weak oracle produces FALSE
    REJECTIONS: a genuinely necessary chunk looks redundant simply because
    nothing tests the case it handles. Measured example - knocking out the
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
                "summary": "no oracle tests available - necessity could not be checked"}

    # get_oracle_tests above already validated on a miss, so this reads the
    # stored verdict. False here means the oracle was validated and came back
    # WEAK - regenerating the decomposition cannot fix that.
    if not is_oracle_strong(problem):
        slug = problem.get("slug") or problem.get("title", "<unnamed problem>")
        return {"status": "oracle_not_strong", "passed": False, "per_chunk": [],
                "summary": (
                    f"CANNOT EVALUATE - the oracle for '{slug}' is not "
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
        f"NECESSITY FAILURE - {named} {'is' if len(dead) == 1 else 'are'} not "
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
