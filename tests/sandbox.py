"""
sandbox.py - deterministic execution grading for MicroTutor.

Runs a Python solution against (input, expected) test cases in an isolated
subprocess with a timeout. Resolves the entry point whether the code is a
bare function  (def is_palindrome(x): ...)  or LeetCode-style
(class Solution: def isPalindrome(self, x): ...).
"""
import json
import os
import subprocess
import sys
import tempfile
import re
from datetime import datetime, timezone
from main.ollama_client import chat, OPENAI_MODEL

GEN_MODEL = OPENAI_MODEL   # system role: oracle test-input generation
                           # (main/mutation.py imports GEN_MODEL from here)

# Harness that runs INSIDE the child process. Reads a JSON payload file (argv[1]):
#   {"code": "<python>", "inputs": [[arg1, arg2], ...], "entry_name": "optional"}
# Writes one JSON line to stdout:
#   {"ok": true, "results": [...]}   or   {"ok": false, "error": "..."}
_HARNESS = r'''
import json, re, sys
try:
    import resource
except ImportError:
    resource = None
from typing import (List, Dict, Optional, Tuple, Set, Any, Union, Callable,
                    Iterable, Iterator)

def resolve_entry(ns, entry_name, helpers=()):
    if entry_name and callable(ns.get(entry_name)):
        return ns[entry_name]
    Sol = ns.get("Solution")
    if isinstance(Sol, type):
        inst = Sol()
        # dir() is ALPHABETICAL, so a helper can outrank the real entry point:
        # searchRange() calls binarySearch(), and "b" < "s". Skip any method a
        # sibling invokes as self.<name>(...) -- those are helpers by
        # definition. main/identity.py applies the identical filter so the two
        # cannot disagree.
        for m in dir(inst):
            if not m.startswith("_") and m not in helpers and callable(getattr(inst, m)):
                return getattr(inst, m)
        for m in dir(inst):
            if not m.startswith("_") and callable(getattr(inst, m)):
                return getattr(inst, m)
    funcs = [v for k, v in ns.items()
             if not k.startswith("__") and callable(v) and hasattr(v, "__code__")]
    return funcs[-1] if funcs else None

def main():
    payload = json.load(open(sys.argv[1]))
    if resource:
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
        except Exception:
            pass
    # exec() runs the payload's code against THIS dict as its globals - a
    # bare `import typing` at the top of this harness script would NOT be
    # visible inside exec(), since that import lives in the harness's own
    # module globals, not in `ns`. Reference solutions routinely use type
    # hints like `nums: List[int]`, which crash with a bare NameError the
    # instant the function is defined unless these names are pre-seeded here.
    ns = {"List": List, "Dict": Dict, "Optional": Optional, "Tuple": Tuple,
          "Set": Set, "Any": Any, "Union": Union, "Callable": Callable,
          "Iterable": Iterable, "Iterator": Iterator}
    try:
        exec(compile(payload["code"], "<solution>", "exec"), ns)
    except Exception as e:
        print(json.dumps({"ok": False, "error": "exec: " + repr(e)})); return
    helpers = set(re.findall(r"self\.(\w+)\s*\(", payload["code"]))
    fn = resolve_entry(ns, payload.get("entry_name"), helpers)
    if fn is None:
        print(json.dumps({"ok": False, "error": "no entry point found"})); return
    results = []
    for args in payload["inputs"]:
        try:
            results.append(fn(*args))
        except Exception as e:
            results.append({"__error__": repr(e)})
    print(json.dumps({"ok": True, "results": results}, default=str))

main()
'''


def run_solution(code: str, inputs: list, entry_name: str | None = None,
                 timeout: float = 8.0) -> dict:
    """Run `code` against a list of argument-lists. Each input is a list of
    positional args. Returns {"ok": bool, "results": [...]} or
    {"ok": False, "error": "..."}. A result is the return value, or
    {"__error__": "..."} if that call raised."""
    payload = {"code": code, "inputs": inputs, "entry_name": entry_name}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as pf:
        json.dump(payload, pf)
        payload_path = pf.name
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _HARNESS, payload_path],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout after {timeout}s (possible infinite loop)"}
    finally:
        os.unlink(payload_path)
    if proc.returncode != 0:
        return {"ok": False, "error": (proc.stderr or "nonzero exit").strip()[:300]}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return {"ok": False, "error": "unparseable harness output: " + proc.stdout[:200]}


def _norm(x):
    """Tuples and lists compare equal (JSON round-trips tuples to lists)."""
    if isinstance(x, (list, tuple)):
        return [_norm(i) for i in x]
    return x


def passes_tests(code: str, tests: list, entry_name: str | None = None,
                 timeout: float = 8.0) -> dict:
    """tests = [{"input": [args...], "expected": value}, ...].
    Returns {"ok", "passed", "total", "fraction", "failures", "error"}."""
    inputs = [t["input"] for t in tests]
    run = run_solution(code, inputs, entry_name=entry_name, timeout=timeout)
    if not run["ok"]:
        return {"ok": False, "passed": 0, "total": len(tests),
                "fraction": 0.0, "failures": [], "error": run["error"]}
    passed, failures = 0, []
    for t, got in zip(tests, run["results"]):
        if isinstance(got, dict) and "__error__" in got:
            failures.append({"input": t["input"], "expected": t["expected"], "got": got["__error__"]})
        elif _norm(got) == _norm(t["expected"]):
            passed += 1
        else:
            failures.append({"input": t["input"], "expected": t["expected"], "got": got})
    total = len(tests)
    return {"ok": True, "passed": passed, "total": total,
            "fraction": passed / total if total else 0.0,
            "failures": failures[:5], "error": None}


# ── oracle test generation: LLM makes INPUTS, ground-truth makes EXPECTED ──

def _extract_signature(solution: str) -> tuple[str | None, list[str]]:
    """Return (entry_name, param_names_without_self) from a solution string.
    Handles both class Solution methods and bare functions."""
    for m in re.finditer(r'def\s+(\w+)\s*\(([^)]*)\)', solution):
        name, params = m.group(1), m.group(2)
        if name.startswith("__"):
            continue
        parts = [p.strip() for p in params.split(",") if p.strip()]
        parts = [p for p in parts if p != "self"]
        names = [p.split(":")[0].split("=")[0].strip() for p in parts]
        return name, names
    return None, []


def _first_json_obj(text: str) -> dict | None:
    """Extract the first complete JSON object by brace-depth matching (tolerant
    of prose or trailing junk around it)."""
    start = text.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
            continue
        if ch == '"': in_str = True
        elif ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


def _clean_sequences(raw: list) -> list[list]:
    """Keep only well-formed call sequences: a list of [name, *json-args].

    The model is asked for a nested structure, and a nested structure is exactly
    what it gets subtly wrong - a bare string instead of a one-element list, a
    call with no name. A malformed sequence is dropped rather than repaired,
    because a repaired guess would silently become part of the oracle."""
    out = []
    for seq in raw if isinstance(raw, list) else []:
        if not isinstance(seq, list) or not seq:
            continue
        calls = []
        for call in seq:
            if isinstance(call, str):
                call = [call]                     # "pop" is a call with no args
            if (isinstance(call, list) and call
                    and isinstance(call[0], str) and call[0]):
                calls.append(call)
        if calls:
            out.append(calls)
    return out


def _generate_call_sequences(problem: dict, n: int) -> list[list]:
    """Test inputs for a METHOD: sequences of calls, not argument lists.

    push(2) returns None and pop() only means 6 after three pushes, so the unit
    the oracle compares has to be a whole run against one object. The teacher's
    own `>>>` examples are the first sequence - they are a recorded oracle
    someone already thought about - and the model grows the rest around them."""
    from main.context import calls_from_docstring, class_methods

    cls = problem.get("group_title") or "Solution"
    target = problem.get("entry_hint") or problem.get("title", "")
    methods = [m for m in class_methods(problem) if m != "__init__"]
    seed = calls_from_docstring(problem.get("description") or "", cls) \
        or calls_from_docstring(problem.get("group_description") or "", cls)

    callable_names = ", ".join(m for m in methods if not m.startswith("__")) or "none"
    dunders = [m for m in methods if m.startswith("__")]
    extra = ""
    if "__len__" in dunders:
        extra += '- Use ["len"] to call len(obj).\n'
    if "__str__" in dunders or "__repr__" in dunders:
        extra += '- Use ["str"] to call str(obj).\n'

    prompt = (
        f"Class: {cls}\n\n"
        f"Specification:\n{(problem.get('group_description') or problem.get('description') or '')[:900]}\n\n"
        f"The method under test is {cls}.{target}.\n"
        f"Methods you may call: {callable_names}\n"
        f"{extra}\n"
        f"Generate {n} diverse CALL SEQUENCES. Each sequence runs against a "
        f"fresh {cls}() and is a JSON array of calls; each call is "
        f'["method_name", arg1, arg2, ...] with no arguments beyond what the '
        f"method takes.\n\n"
        f"CRITICAL:\n"
        f"- EVERY sequence must actually call {target} at least once - a "
        f"sequence that never reaches it tests nothing.\n"
        f"- Build up state first. A sequence that calls {target} on an empty "
        f"object is a good edge case, but it must not be the only kind.\n"
        f"- Vary length: some 2-3 calls, some 8-12, interleaving different "
        f"methods so ordering bugs show up.\n"
        f"- Only JSON-serializable arguments (numbers, strings, booleans, "
        f"arrays). Integers within -1000..1000, strings under 30 chars.\n"
        f"- Do NOT include expected results. Only the calls.\n\n"
        f'Return JSON only: {{"sequences": [[["push", 2], ["pop"]], ...]}}'
    )

    generated = []
    for temp in (0.2, 0.6):
        raw = chat(GEN_MODEL, "You generate call sequences as strict JSON. No prose.",
                   [{"role": "user", "content": prompt}], temperature=temp, fmt="json")
        generated = _clean_sequences((_first_json_obj(raw) or {}).get("sequences", []))
        if generated:
            break

    # The seed goes FIRST and is never dropped: it is the one sequence in the
    # suite whose expected values a human has already checked by hand.
    sequences = ([seed] if seed else []) + generated
    if not sequences:
        print(f"  ⚠️  sequence-gen empty for {problem.get('slug','?')}")
    # One positional argument - the whole sequence - because the driver's
    # signature is _mt_run_calls(calls). See main/context.py.
    return [[seq] for seq in sequences]


def generate_test_inputs(problem: dict, n: int = 10) -> list[list]:
    """Ask the LLM for n diverse input argument-lists (edge cases included).
    INPUTS ONLY - never expected outputs. Retries once if the model returns junk."""
    # Local import: main.identity imports this module, so a top-level import
    # here would be circular.
    from main.context import is_method
    from main.identity import get_resolved_entry
    if is_method(problem):
        return _generate_call_sequences(problem, n)
    resolved = get_resolved_entry(problem)
    name, params = resolved["entry_name"], resolved["params"]
    sig = f"{name}({', '.join(params)})" if name else problem.get("title", "")
    desc_text = (problem.get('description') or '')[:800]
    prompt = (
        f"Problem: {problem.get('title','')}\n\n"
        f"Description:\n{desc_text}\n\n"
        f"Function: {sig}\n"
        f"It takes {len(params)} argument(s): {', '.join(params) or 'unknown'}.\n\n"
        f"Generate {n} diverse test INPUTS that satisfy ALL constraints and "
        f"preconditions stated in the problem description. "
        f"For example, if the problem guarantees exactly one solution exists, "
        f"every input you generate MUST have exactly one valid solution. "
        f"If the problem says inputs are non-negative, never generate negatives.\n"
        f"Include edge cases (empty, single-element, minimal values) that still "
        f"respect the problem's constraints.\n"
        f"Each input is a JSON array of the positional arguments in order.\n"
        f"Use only JSON-serializable values (numbers, strings, booleans, arrays, objects).\n"
        f"Keep values reasonable: integers within -1000000000..1000000000, "
        f"strings under 50 chars, arrays under 20 items. Never emit extremely large numbers.\n"
        f'Return JSON only: {{"inputs": [[arg1, ...], ...]}}'
        f"CRITICAL for correctness:\n"
        f"- Never generate inputs where multiple valid answers exist "
        f"(e.g. for Two Sum, never use arrays where more than one pair sums to target).\n"
        f"- Never generate inputs with duplicate values unless the problem "
        f"explicitly requires handling duplicates.\n"
        f"- Every generated input must have exactly ONE correct output.\n"
    )
    raw = ""
    for temp in (0.2, 0.5):
        raw = chat(GEN_MODEL, "You generate test inputs as strict JSON. No prose.",
                   [{"role": "user", "content": prompt}], temperature=temp, fmt="json")
        data = _first_json_obj(raw) or {}
        inputs = [i if isinstance(i, list) else [i]
                  for i in data.get("inputs", []) if i is not None]
        if inputs:
            return inputs
    print(f"  ⚠️  input-gen empty for {problem.get('slug','?')}; raw head: {raw[:160]!r}")
    return []


def _count_valid_pairs(nums: list, target: int) -> int:
    """Count how many distinct index pairs sum to target."""
    count = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                count += 1
    return count


def _is_ambiguous_output(inp: list, out) -> bool:
    """Detect inputs where multiple valid outputs exist for common problem patterns."""
    if out is None:
        return True
    # Two Sum pattern: array + target, output is index pair
    if (len(inp) == 2 and isinstance(inp[0], list) and isinstance(inp[1], int)
            and isinstance(out, list) and len(out) == 2):
        if _count_valid_pairs(inp[0], inp[1]) != 1:
            return True
        # Also reject duplicate values that break pre-built dict approach
        nums = inp[0]
        if len(nums) != len(set(nums)):
            return True
    return False


def make_oracle_tests(problem: dict, n: int = 12) -> list[dict]:
    """Generate inputs, run ground-truth to compute expected outputs.
    Filters out ambiguous inputs (multiple valid answers, duplicates that
    break common approaches) so the gate only tests unambiguous cases."""
    # For a METHOD this is the whole module the method lives in, not the bare
    # `def` - a method has no ground truth outside its class.
    from main.context import reference_program
    solution = reference_program(problem)
    if not solution.strip():
        return []
    from main.identity import get_resolved_entry
    name = get_resolved_entry(problem)["entry_name"]

    # Generate more inputs than needed so we have room to filter
    inputs = generate_test_inputs(problem, n=n)
    if not inputs:
        return []

    run = run_solution(solution, inputs, entry_name=name)
    if not run["ok"]:
        return []

    tests = []
    for inp, out in zip(inputs, run["results"]):
        if isinstance(out, dict) and "__error__" in out:
            continue
        if out is None:
            continue
        if _is_ambiguous_output(inp, out):
            continue
        tests.append({"input": inp, "expected": out})

    return tests


# Oracle data now lives under data/oracles/ and is owned by main.oracle_store.
from main.oracle_store import (OracleUnusableError, cache_path,  # noqa: E402
                               entry_tests as _entry_tests,
                               is_validated as _is_validated,
                               load_cache as _load_cache_impl,
                               load_strong_cached_oracle,
                               save_cache as _save_cache_impl,
                               verdict_entry)
_CACHE_PATH = cache_path()

# A cache entry is:
#   {"final_tests": [{"input", "expected"}, ...],   # the suite, possibly grown
#    "strong": bool,                                # cleared mutation testing?
#    "kill_rate": float, "kill_rate_direct": float, # see main/mutation.py
#    "validated_at": "<UTC ISO-8601>"}
# Entries written before mutation testing existed are a bare list of tests.
# Those are treated as unvalidated and upgraded in place on first access.


def _load_cache() -> dict:
    """Delegates to main.oracle_store (backend-owned data)."""
    return _load_cache_impl()


def _save_cache(cache: dict) -> None:
    """Delegates to main.oracle_store (backend-owned data)."""
    _save_cache_impl(cache)




def get_oracle_tests(problem: dict, n: int = 10, emit=None) -> list[dict]:
    """Cached oracle tests for `problem`, mutation-tested before they are
    trusted. Tests are generated (or loaded) and then handed to
    validate_oracle, which may GROW the suite with counterexamples that kill
    surviving mutants; the grown suite and its verdict are persisted together.

    Validation runs at most once per problem - an entry that already carries a
    verdict is returned as-is, weak or strong. Always returns a plain list of
    tests, so existing callers are unaffected.

    `emit`, when given, narrates the run for a live UI - the same callback and
    the same event vocabulary main/live_playground.py uses, so a teacher
    watching an upload sees the mutation detail rather than a spinner. It is
    accepted HERE rather than in the caller because this function owns the one
    boundary the caller cannot see: whether the tests were generated or read
    from cache, and when generation ends and validation begins. Never changes
    behaviour; leaving it None is the production path."""
    emit = emit or (lambda ev: None)
    from main.identity import content_hash
    slug = problem.get("slug", "")          # for humans reading the logs only
    key = content_hash(problem)             # cache identity: content, not title
    cache = _load_cache()
    entry = cache.get(key)

    if _is_validated(entry):
        # status, not strong=False - see the verdict print below.
        print(f"  [oracle] {slug}: validation SKIPPED (cached status="
              f"{entry.get('status') or ('strong' if entry['strong'] else 'weak')}"
              f", kill_rate={entry.get('kill_rate', 0):.2f})")
        cached_tests = _entry_tests(entry)
        # A watcher must be told the suite was REUSED. Silence here reads as a
        # mutation stage that never started, which looks like a hang.
        emit({"type": "oracle_tests", "tests": cached_tests, "origin": "cached"})
        return cached_tests

    tests = _entry_tests(entry) if entry is not None else make_oracle_tests(problem, n=n)
    if not tests:
        return []
    emit({"type": "oracle_tests", "tests": tests,
          "origin": "cached (old format)" if entry is not None else "fresh"})

    # Local import: main.mutation imports this module, so a top-level import
    # here would be circular.
    from main.mutation import validate_oracle

    origin = "cached (old format)" if entry is not None else "freshly generated"
    print(f"  [oracle] {slug or '?'}: validation RUNNING on {len(tests)} "
          f"{origin} tests")
    from main.mutation import CUTOFF_1_KILL_RATE
    emit({"type": "stage", "name": "mutation",
          "label": f"Mutation testing - deterministically breaking the ground "
                   f"truth one edit at a time and checking the oracle notices "
                   f"(STRONG needs kill_rate_direct \u2265 {CUTOFF_1_KILL_RATE})"})
    report = validate_oracle(problem, tests, emit=emit)
    # Shape owned by main.oracle_store, not built here: main.live_playground
    # writes the same entry from the same report, and two hand-built copies
    # drift the moment a field is added (A4 added four).
    validated = verdict_entry(report, slug)
    # Print the STATUS, not the `strong` bit. The bit has only two values and
    # there are three outcomes, so a needs_review problem printed as WEAK while
    # the very next line - and the stored verdict - said needs_review. A log
    # that contradicts the verdict sends someone chasing a bug that is not there.
    # The range is printed with it, because for needs_review the two bounds are
    # the whole reason a human is being asked.
    status = (validated.get("status") or
              ("strong" if validated["strong"] else "weak")).upper()
    span = ""
    if validated.get("kill_rate_lower") != validated.get("kill_rate_upper"):
        span = (f" [{validated['kill_rate_lower']:.2f}-"
                f"{validated['kill_rate_upper']:.2f}, "
                f"{validated.get('undetermined', 0)} undetermined]")
    print(f"  [oracle] {slug or '?'}: {len(tests)} -> "
          f"{len(validated['final_tests'])} tests, "
          f"kill_rate={validated['kill_rate']:.2f} "
          f"(direct {validated['kill_rate_direct']:.2f}) "
          f"{status}{span}")

    cache[key] = validated          # verdict_entry already carries the slug
    _save_cache(cache)
    return validated["final_tests"]


def is_oracle_certified(problem: dict) -> bool:
    """May this problem's oracle be graded with?

    Renamed from is_oracle_strong: "strong" is now one of TWO ways to earn it,
    and a predicate named after one of its branches reads as a stricter promise
    than it makes. The verdict itself is unchanged and still available - see
    oracle_store.certified for the two claims and why they are kept apart.

    Reads the cached verdict, running validation once to populate it if it is
    missing. Takes the problem dict, not a slug: the cache is keyed by content
    now, and a slug can no longer identify an entry - that ambiguity is exactly
    the collision this change removes."""
    from main.identity import content_hash
    from main.oracle_store import certified
    slug = problem.get("slug", "")
    entry = _load_cache().get(content_hash(problem))
    if _is_validated(entry):
        print(f"  [oracle] {slug}: validation SKIPPED "
              f"(cached status={entry.get('status') or ('strong' if entry['strong'] else 'weak')})")
    else:
        get_oracle_tests(problem)               # validates and persists the verdict
        entry = _load_cache().get(content_hash(problem))
    return certified(entry)


# ── self-test: run `python sandbox.py` ──
if __name__ == "__main__":
    bare = "def is_palindrome(x):\n    s = str(x)\n    return s == s[::-1]"
    leet = ("class Solution:\n"
            "    def isPalindrome(self, x):\n"
            "        if x < 0:\n            return False\n"
            "        s = str(x)\n        return s == s[::-1]")
    tests = [{"input": [121], "expected": True},
             {"input": [-121], "expected": False},
             {"input": [10],  "expected": False},
             {"input": [0],   "expected": True}]
    print("bare function :", passes_tests(bare, tests))
    print("class Solution:", passes_tests(leet, tests))
    loop = "def f(x):\n    while True:\n        pass"
    print("infinite loop :", passes_tests(loop, [{"input": [1], "expected": 1}], timeout=2.0))
    if "--oracle" in sys.argv:
        prob = {
            "slug": "palindrome-number",
            "title": "Palindrome Number",
            "description": "Given an integer x, return True if x is a palindrome integer, False otherwise.",
            "solution": leet,
        }
        print("\noracle tests (LLM inputs + ground-truth expected):")
        for t in make_oracle_tests(prob, n=8):
            print("  ", t)

class OracleUnusableError(RuntimeError):
    """No STRONG cached oracle for this problem, so it cannot be graded.

    Carries `reason_code` so callers can distinguish missing / unvalidated /
    weak / malformed without parsing the message."""

    def __init__(self, message: str, reason_code: str):
        super().__init__(message)
        self.reason_code = reason_code


def load_strong_cached_oracle(problem: dict) -> list[dict]:
    """READ-ONLY oracle access for the answer-checking path.

    get_oracle_tests() is the WRITE path: on a miss it generates tests, runs
    mutation testing and can block for minutes. That is correct for warm-up and
    fatal for grading - a student pressing Submit must never trigger it, and a
    weak or absent oracle must never be silently accepted as a basis for a
    verdict. This function only ever reads the cache.

    Returns the tests when the content-hash entry exists and is strong.
    Raises OracleUnusableError otherwise. Never generates, never validates,
    never writes."""
    from main.identity import content_hash

    entry = _load_cache().get(content_hash(problem))
    slug = problem.get("slug") or problem.get("title") or "<unnamed problem>"

    if entry is None:
        raise OracleUnusableError(
            f"No cached oracle for '{slug}'. It must be validated "
            f"(python -m main.warmup) before it can be graded.", "oracle_missing")
    if isinstance(entry, list):
        # Pre-migration slug-keyed entry: tests but no verdict.
        raise OracleUnusableError(
            f"'{slug}' has a legacy cache entry with no verdict.", "oracle_unvalidated")
    if not isinstance(entry, dict) or "strong" not in entry:
        raise OracleUnusableError(
            f"'{slug}' has no validation verdict.", "oracle_unvalidated")
    if not entry["strong"]:
        raise OracleUnusableError(
            f"'{slug}' has an oracle that did not clear mutation testing.",
            "oracle_weak")

    tests = entry.get("final_tests")
    if not isinstance(tests, list) or not tests:
        raise OracleUnusableError(
            f"'{slug}' is marked strong but stores no tests.", "oracle_malformed")
    for t in tests:
        if not (isinstance(t, dict) and "input" in t and "expected" in t):
            raise OracleUnusableError(
                f"'{slug}' has a malformed cached test.", "oracle_malformed")
    return tests
