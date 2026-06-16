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
from ollama_client import chat

GEN_MODEL = "qwen2.5:7b-instruct"

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
    prompt = (
        f"Problem: {problem.get('title','')}\n\n"
        f"Description:\n{(problem.get('description') or '')[:800]}\n\n"
        f"Function: {sig}\n"
        f"It takes {len(params)} argument(s): {', '.join(params) or 'unknown'}.\n\n"
        f"Generate {n} diverse test INPUTS, including edge cases "
        f"(zero, negative, empty, single-element, large where relevant).\n"
        f"Each input is a JSON array of the positional arguments in order.\n"
        f"Use only JSON-serializable values (numbers, strings, booleans, arrays, objects).\n"
        f"Keep values reasonable: integers within -1000000000..1000000000, "
        f"strings under 50 chars, arrays under 20 items. Never emit extremely large numbers.\n"
        f'Return JSON only: {{"inputs": [[arg1, ...], ...]}}'
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


def make_oracle_tests(problem: dict, n: int = 10) -> list[dict]:
    """Generate inputs, then run the GROUND-TRUTH to compute expected outputs.
    Returns [{"input": [...], "expected": value}, ...]. Inputs where the ground
    truth itself errors are dropped."""
    solution = problem.get("solution", "")
    if not solution.strip():
        return []
    name, _ = _extract_signature(solution)
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
        tests.append({"input": inp, "expected": out})
    return tests


_CACHE_PATH = os.path.join(os.path.dirname(__file__), "tests_cache.json")

def get_oracle_tests(problem: dict, n: int = 10) -> list[dict]:
    """Cached wrapper around make_oracle_tests, keyed by slug — generate once, reuse."""
    slug = problem.get("slug", "")
    cache = {}
    if os.path.exists(_CACHE_PATH):
        try:
            cache = json.load(open(_CACHE_PATH))
        except Exception:
            cache = {}
    if slug and slug in cache:
        return cache[slug]
    tests = make_oracle_tests(problem, n=n)
    if slug and tests:
        cache[slug] = tests
        try:
            json.dump(cache, open(_CACHE_PATH, "w"), indent=2)
        except Exception:
            pass
    return tests


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