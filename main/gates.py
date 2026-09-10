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

Gate 2 - prompt leakage. A sub-question must state a GOAL and never the
METHOD, because a prompt that describes the code has written the student's
answer for them. That rule has always been in CHUNK_DECOMPOSE_SYSTEM, and
nothing ever checked it: the assembly gate reads the references and the
necessity gate knocks them out, so the PROMPT - the only part a student
actually reads - was the one artefact no gate looked at. Asking twice does not
hold it either. Once the decomposer is shown the reference implementation (it
must be, or it cannot match an oracle built from that code) the pull towards
describing it is strong enough that "Initialize the states and report
dictionary" comes back with the rules quoted directly above it.

So it is measured instead. Two signals, both cheap and both deterministic:
a word that names the mechanism, and an identifier that only the CODE knows -
one the statement never uses, which is therefore a name the student was
supposed to arrive at.
"""
import ast
import re

from .identity import get_resolved_entry
from tests.sandbox import (get_oracle_tests, is_oracle_certified,
                           passes_tests)

_NOOP = "pass"

# A decomposition below this is degenerate, not "small". The whole point of
# chunking is that a student's early mistake is caught rather than absorbed.
_MIN_CHUNKS = 2

# References that do nothing. Such a chunk cannot possibly be load-bearing, so
# it would fail necessity anyway - catching it here gives a precise error
# instead of a confusing knockout result.
_NOOP_REFERENCES = {"", "pass", "...", "None", "return"}


# Words that name the MECHANISM. Deliberately short: a false positive here
# costs a retry on a decomposition that was fine, so only the phrasings the
# system prompt already calls out by name are listed.
_METHOD_WORDS = re.compile(
    r"\b(initiali[sz]\w*|iterat\w*|loop\s+(?:over|through)|traverse"
    r"|append\s+to|set\s+\w+\s+to"
    r"|(?:create|build|make|use)\s+(?:a|an|the)\s+"
    r"(?:dict\w*|list|set|array|stack|queue|counter|variable|loop|pointer))\b",
    re.I)


# Names that are ORDINARY ENGLISH as well as common variable names. A prompt
# saying "keep the result for the next step" is describing an outcome, not
# leaking `result` - but the leak check cannot tell those apart by spelling, and
# a false positive here burns a retry on a decomposition that was fine. The list
# is deliberately short: every word on it is one a person would write in a
# sentence without thinking about code.
_ORDINARY = frozenset({
    "result", "results", "value", "values", "answer", "answers", "count",
    "total", "sum", "number", "numbers", "item", "items", "step", "steps",
    "check", "input", "output", "data", "text", "word", "words", "line",
    "lines", "current", "first", "last", "next", "index", "length", "size",
    "name", "names", "key", "keys", "start", "end", "empty", "found",
})


def _identifiers(src: str) -> set:
    """Every name the reference code uses - variables and attributes alike."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set()
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
    return out


# A prompt that says what the chunk leaves for the next one. Deliberately loose
# on wording and strict on WHEN it is required - see _hands_off_needed.
_HANDOFF = re.compile(
    r"next step|next chunk|for the next|later step|keep(s|ing)? (the|it|its)\b"
    r"|stor(e|es|ing) (the|it|its)\b|without returning|do(es)? not return"
    r"|ready to\b|leave(s|ing)? (the|it|its)\b|hand(s|ing)? (it|the)"
    # A model says this a dozen ways. These two came back from real runs and
    # were flagged as ambiguous when they are not: "link it to the current top
    # so that it's READY TO become the new top", and "checks if the word starts
    # with a letter and CAPTURES whether it begins correctly". A miss here costs
    # a retry on a decomposition that was already clear, which is the one way
    # this gate can make things worse rather than better.
    r"|captur(e|es|ing)\b|so (that )?it (can|is|will)\b"
    r"|(previous|earlier|first|preceding) (step|chunk|part)",
    re.I)


def _returns(reference: str) -> bool:
    """Does this chunk's own code finish the method?"""
    try:
        tree = ast.parse("def _w():\n" + "\n".join(
            "    " + ln for ln in (reference or "").splitlines()))
    except SyntaxError:
        return True                      # unparseable: judged elsewhere, not here
    return any(isinstance(n, ast.Return) for n in ast.walk(tree))


def check_prompts(chunks: list, problem: dict) -> dict:
    """Gate 2. {"status": "pass"|"fail", "summary": str}.

    A name the STATEMENT already uses is the problem's own vocabulary and is
    fine - HW3's docstring says "store the result in states", so a prompt may
    say states. A name only the code knows is the student's decision, and
    putting it in the question makes the decision for them."""
    said = set(re.findall(r"[a-z_]\w*",
                          ((problem.get("description") or "") + " " +
                           (problem.get("group_description") or "")).lower()))
    bad = []
    for i, c in enumerate(chunks):
        prompt = getattr(c, "prompt", "") or ""
        step = getattr(c, "step_id", "?")
        reference = getattr(c, "reference", "") or ""

        # THE MISLEADING-STEP CHECK, and the only one here that is about what a
        # student would DO rather than what they are told. A chunk that is not
        # last and whose own code never returns must leave its result for the
        # next chunk - and if the prompt does not say so, it reads as the whole
        # job. "Determine if the stack is empty by checking if it has a top
        # node" is answered `return self.top is None` by any reasonable
        # student, which is then marked wrong for returning too early, with
        # nothing on screen explaining why.
        #
        # Decidable rather than guessed: a chunk whose reference DOES return is
        # a step that legitimately finishes something, and is never flagged.
        if i < len(chunks) - 1 and not _returns(reference) \
                and not _HANDOFF.search(prompt):
            bad.append(f'{step}: does not finish the method, but does not say '
                       f'what it leaves for the next step - a student will '
                       f'answer it as if it were the whole job. End it with '
                       f'something like "...and keep the result for the next '
                       f'step."')
            continue

        hit = _METHOD_WORDS.search(prompt)
        if hit:
            bad.append(f'{step}: "{hit.group(0)}" states HOW, not what to achieve')
            continue
        leaked = sorted(n for n in _identifiers(getattr(c, "reference", "") or "")
                        if len(n) > 2 and n.lower() not in said
                        and n.lower() not in _ORDINARY
                        and re.search(rf"\b{re.escape(n)}\b", prompt))
        if leaked:
            bad.append(f'{step}: names {", ".join(leaked)} - that is a name only '
                       f'the solution uses, so the student is told what to call it')
    if not bad:
        return {"status": "pass", "summary": "prompts state goals, not methods"}
    return {"status": "fail",
            "summary": ("These sub-questions describe the SOLUTION instead of "
                        "asking for it:\n  " + "\n  ".join(bad) +
                        "\nRewrite the prompts only - say what each chunk must "
                        "ACHIEVE, in the problem's own words. Leave the "
                        "reference code exactly as it is.")}


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
         (OracleNotStrongError) - CERTIFIED per is_oracle_certified, which keys
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
    if not is_oracle_certified(problem):
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
    if not is_oracle_certified(problem):
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
        necessary, detail = _outcome(
            assemble_references(problem, header, knocked), tests, entry)
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


if __name__ == "__main__":
    # Gate 2 is pure - no oracle, no model, no subprocess - so it self-checks.
    from types import SimpleNamespace as _N

    _step = lambda i, prompt, ref: _N(step_id=f"Part {i}", prompt=prompt,
                                      reference=ref)
    _prob = {"description": "Run every statement in order and return a report "
                            "dictionary, or None if any statement is invalid. "
                            "Store the result in states."}

    # The two shapes that came back from the model with the rules quoted
    # directly above them, which is why this gate exists at all.
    out = check_prompts([_step(1, "Initialize the states and report dictionary.",
                               "self.states = {}")], _prob)
    assert out["status"] == "fail" and "Initialize" in out["summary"], out
    out = check_prompts([_step(1, "Iterate over each statement.",
                               "for s in x:\n    pass")], _prob)
    assert out["status"] == "fail", out

    # A name the STATEMENT uses is the problem's own vocabulary, not a leak.
    ok = check_prompts([_step(1, "Record what each statement leaves in states, "
                                 "and give back the report dictionary.",
                              "self.states = {}\nreport = {}")], _prob)
    assert ok["status"] == "pass", ok
    # ...a name only the CODE knows is the decision the student should make.
    out = check_prompts([_step(1, "Build calcObj and use it.",
                               "calcObj = Calculator()")], _prob)
    assert out["status"] == "fail" and "calcObj" in out["summary"], out
    # Short names are not leaks - `x` and `i` match half the English language.
    assert check_prompts([_step(1, "Return the total you accumulated.",
                                "x = 0\nfor i in y:\n    x += i")],
                         _prob)["status"] == "pass"
    # An unparseable reference must not crash the gate; it fails elsewhere.
    assert check_prompts([_step(1, "Do the thing.", "def (")],
                         _prob)["status"] == "pass"
    assert check_prompts([], _prob)["status"] == "pass"

    print("gates.py self-check OK")
