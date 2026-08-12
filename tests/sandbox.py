"""
sandbox.py — deterministic execution grading for MicroTutor.

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
import json, sys
try:
    import resource
except ImportError:
    resource = None

def resolve_entry(ns, entry_name):
    if entry_name and callable(ns.get(entry_name)):
        return ns[entry_name]
    Sol = ns.get("Solution")
    if isinstance(Sol, type):
        inst = Sol()
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
    ns = {}
    try:
        exec(compile(payload["code"], "<solution>", "exec"), ns)
    except Exception as e:
        print(json.dumps({"ok": False, "error": "exec: " + repr(e)})); return
    fn = resolve_entry(ns, payload.get("entry_name"))
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


def generate_test_inputs(problem: dict, n: int = 10) -> list[list]:
    """Ask the LLM for n diverse input argument-lists (edge cases included).
    INPUTS ONLY — never expected outputs. Retries once if the model returns junk."""
    name, params = _extract_signature(problem.get("solution", ""))
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
    solution = problem.get("solution", "")
    if not solution.strip():
        return []
    name, _ = _extract_signature(solution)

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


_CACHE_PATH = os.path.join(os.path.dirname(__file__), "tests_cache.json")

# A cache entry is:
#   {"final_tests": [{"input", "expected"}, ...],   # the suite, possibly grown
#    "strong": bool,                                # cleared mutation testing?
#    "kill_rate": float, "kill_rate_direct": float, # see main/mutation.py
#    "validated_at": "<UTC ISO-8601>"}
# Entries written before mutation testing existed are a bare list of tests.
# Those are treated as unvalidated and upgraded in place on first access.


def _load_cache() -> dict:
    if os.path.exists(_CACHE_PATH):
        try:
            return json.load(open(_CACHE_PATH))
        except Exception:
            pass
    return {}


def _save_cache(cache: dict) -> None:
    try:
        json.dump(cache, open(_CACHE_PATH, "w"), indent=2)
    except Exception:
        pass


def _entry_tests(entry) -> list[dict]:
    """The test list out of a cache entry, old format (bare list) or new."""
    if isinstance(entry, list):
        return entry
    if isinstance(entry, dict):
        return entry.get("final_tests", [])
    return []


def _is_validated(entry) -> bool:
    """A prior validation left a verdict here — don't spend another pass."""
    return isinstance(entry, dict) and "strong" in entry


def get_oracle_tests(problem: dict, n: int = 10) -> list[dict]:
    """Cached oracle tests for `problem`, mutation-tested before they are
    trusted. Tests are generated (or loaded) and then handed to
    validate_oracle, which may GROW the suite with counterexamples that kill
    surviving mutants; the grown suite and its verdict are persisted together.

    Validation runs at most once per problem — an entry that already carries a
    verdict is returned as-is, weak or strong. Always returns a plain list of
    tests, so existing callers are unaffected."""
    slug = problem.get("slug", "")
    cache = _load_cache()
    entry = cache.get(slug) if slug else None

    if _is_validated(entry):
        print(f"  [oracle] {slug}: validation SKIPPED (cached "
              f"strong={entry['strong']}, kill_rate={entry.get('kill_rate', 0):.2f})")
        return _entry_tests(entry)

    tests = _entry_tests(entry) if entry is not None else make_oracle_tests(problem, n=n)
    if not tests:
        return []

    # Local import: main.mutation imports this module, so a top-level import
    # here would be circular.
    from main.mutation import validate_oracle

    origin = "cached (old format)" if entry is not None else "freshly generated"
    print(f"  [oracle] {slug or '?'}: validation RUNNING on {len(tests)} "
          f"{origin} tests")
    report = validate_oracle(problem, tests)
    validated = {
        "final_tests": report["final_tests"],
        "strong": report["strong"],
        "kill_rate": report["kill_rate"],
        "kill_rate_direct": report["kill_rate_direct"],
        "validated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    print(f"  [oracle] {slug or '?'}: {len(tests)} -> "
          f"{len(validated['final_tests'])} tests, "
          f"kill_rate={validated['kill_rate']:.2f} "
          f"(direct {validated['kill_rate_direct']:.2f}) "
          f"{'STRONG' if validated['strong'] else 'WEAK'}")

    if slug:
        cache[slug] = validated
        _save_cache(cache)
    return validated["final_tests"]


def is_oracle_strong(problem) -> bool:
    """Did this problem's oracle clear mutation testing?

    Reads the cached verdict, running validation once to populate it if it is
    missing. Accepts a problem dict, or a bare slug for a cache-only lookup —
    a slug alone carries no ground-truth solution to mutate, so an unvalidated
    one simply reads as not-strong."""
    if isinstance(problem, str):
        entry = _load_cache().get(problem)
        if not _is_validated(entry):
            print(f"  [oracle] {problem}: no cached verdict (slug lookup only)")
            return False
        print(f"  [oracle] {problem}: validation SKIPPED "
              f"(cached strong={entry['strong']})")
        return bool(entry["strong"])

    slug = problem.get("slug", "")
    entry = _load_cache().get(slug) if slug else None
    if _is_validated(entry):
        print(f"  [oracle] {slug}: validation SKIPPED "
              f"(cached strong={entry['strong']})")
    else:
        get_oracle_tests(problem)               # validates and persists the verdict
        entry = _load_cache().get(slug) if slug else None
    return bool(_is_validated(entry) and entry["strong"])


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