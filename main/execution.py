"""
execution.py — hardened execution path for UNTRUSTED student code.

Deliberately separate from tests/sandbox.py:run_solution(), which stays the
trusted runner for ground-truth solutions and mutation testing. That runner
inherits the parent environment (including .env secrets) and applies no policy
check, which is fine for code we wrote and fatal for code a student typed.

HONEST LIMITATION: this is a hardened LOCAL harness, not a mathematically
secure sandbox. It raises the cost of an attack (AST policy, sanitized env,
isolated mode, temp cwd, rlimits, output caps, wall-clock timeout) but a
determined attacker with arbitrary Python can still find gaps. Production
deployment needs container/OS-level isolation. The interface here is
deliberately narrow so a Docker/gVisor backend can replace the body of
run_student_code() without touching callers.
"""
import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile

from .schemas import ExecutionResult

# Modules a coding exercise legitimately needs. Everything else is refused.
_ALLOWED_IMPORTS = {
    "math", "collections", "itertools", "functools", "heapq", "bisect",
    "string", "re", "random", "typing", "operator", "datetime", "decimal",
    "fractions", "statistics", "copy", "json", "array", "enum", "dataclasses",
}
# Names that hand back arbitrary execution or the filesystem.
_BANNED_NAMES = {
    "eval", "exec", "compile", "open", "input", "__import__", "breakpoint",
    "globals", "locals", "vars", "memoryview", "exit", "quit", "help",
}
_BANNED_ATTRS = {
    "__subclasses__", "__bases__", "__mro__", "__globals__", "__code__",
    "__closure__", "__builtins__", "__loader__", "__reduce__", "__reduce_ex__",
    "__getattribute__", "__dict__", "__class__",
}

# Bound on OUR result payload, not on student print(): the child already
# redirects student stdout/stderr to devnull, so a print() flood never reaches
# the parent. At 64KB this capped nothing harmful and silently TRUNCATED
# legitimate results (combine(n,k) emits ~6.9MB of valid JSON), producing a
# JSONDecodeError that surfaced as harness_error on a child that exited 0.
# Still bounded so a pathological result cannot exhaust the parent.
_MAX_OUTPUT_BYTES = 4 * 1024 * 1024
# Problems whose oracle results exceed the cap are UNSUPPORTED rather than a
# reason to weaken limits for everyone.
_DEFAULT_TIMEOUT = 6.0
_MEM_BYTES = 512 * 1024 * 1024
_CPU_SECONDS = 5


class PolicyViolation(Exception):
    """Student code was refused before it ever ran."""


def check_policy(code: str) -> None:
    """AST safety policy. Raises PolicyViolation with a student-safe message.

    Runs BEFORE execution — a refused program is never executed at all."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return                      # syntax is classified separately, not here

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                if root not in _ALLOWED_IMPORTS:
                    raise PolicyViolation(f"the '{root}' module isn't allowed here")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level or root not in _ALLOWED_IMPORTS:
                raise PolicyViolation(f"the '{root or 'relative'}' module isn't allowed here")
        elif isinstance(node, ast.Name) and node.id in _BANNED_NAMES:
            raise PolicyViolation(f"'{node.id}' isn't allowed here")
        elif isinstance(node, ast.Attribute) and node.attr in _BANNED_ATTRS:
            raise PolicyViolation("that attribute isn't allowed here")


# Child-process harness. Mirrors the trusted harness's entry resolution so a
# student candidate and the reference resolve the same function, but adds
# resource limits and output caps.
_STUDENT_HARNESS = r'''
import json, re, sys, os
try:
    import resource
except ImportError:
    resource = None
from typing import (List, Dict, Optional, Tuple, Set, Any, Union, Callable,
                    Iterable, Iterator)

def limits(mem, cpu):
    if not resource: return
    for what, val in ((resource.RLIMIT_AS, mem), (resource.RLIMIT_CPU, cpu),
                      (resource.RLIMIT_FSIZE, 1 << 20), (resource.RLIMIT_NOFILE, 64)):
        try: resource.setrlimit(what, (val, val))
        except Exception: pass
    try: resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    except Exception: pass

def resolve_entry(ns, entry_name, helpers=()):
    if entry_name and callable(ns.get(entry_name)):
        return ns[entry_name]
    Sol = ns.get("Solution")
    if isinstance(Sol, type):
        inst = Sol()
        for m in dir(inst):
            if not m.startswith("_") and m not in helpers and callable(getattr(inst, m)):
                return getattr(inst, m)
        for m in dir(inst):
            if not m.startswith("_") and callable(getattr(inst, m)):
                return getattr(inst, m)
    funcs = [v for k, v in ns.items()
             if not k.startswith("__") and callable(v) and hasattr(v, "__code__")]
    return funcs[-1] if funcs else None

def norm(x):
    if isinstance(x, (list, tuple)):
        return [norm(i) for i in x]
    return x

def brief(v, cap=160):
    """Bounded diagnostic: never ship the whole value back to the parent."""
    t = type(v).__name__
    try:
        n = len(v)
    except Exception:
        n = None
    r = repr(v)
    if len(r) > cap:
        r = r[:cap] + "...<truncated>"
    return {"repr": r, "type": t, "len": n}

def main():
    payload = json.load(open(sys.argv[1]))
    limits(payload["mem"], payload["cpu"])
    sys.stdin.close()
    devnull = open(os.devnull, "w")
    real_stdout = sys.stdout
    sys.stdout = devnull            # student print() goes nowhere
    sys.stderr = devnull
    ns = {"List": List, "Dict": Dict, "Optional": Optional, "Tuple": Tuple,
          "Set": Set, "Any": Any, "Union": Union, "Callable": Callable,
          "Iterable": Iterable, "Iterator": Iterator}
    def emit(obj):
        sys.stdout = real_stdout
        print(json.dumps(obj, default=str))
    try:
        exec(compile(payload["code"], "<student>", "exec"), ns)
    except Exception as e:
        emit({"status": "exec_error", "error": repr(e)[:300]}); return
    helpers = set(re.findall(r"self\.(\w+)\s*\(", payload["code"]))
    fn = resolve_entry(ns, payload.get("entry_name"), helpers)
    if fn is None:
        emit({"status": "no_entry"}); return
    tests = payload.get("tests")
    if tests is not None:
        # COMPARE IN-CHILD. A correct 6.9MB return value must never cross the
        # IPC boundary: only a compact verdict does. The old protocol shipped
        # the entire result back and drowned in its own serialization.
        passed, failures, raised = 0, [], False
        for i, t in enumerate(tests):
            try:
                got = fn(*t["input"])
            except Exception as e:
                raised = True
                failures.append({"index": i, "error": repr(e)[:200]})
                continue
            if norm(got) == norm(t["expected"]):
                passed += 1
            elif len(failures) < 5:
                failures.append({"index": i, "got": brief(got),
                                 "expected": brief(t["expected"])})
        emit({"status": "compared", "passed": passed, "total": len(tests),
              "failures": failures, "raised": raised})
        return
    results, raised = [], False
    for args in payload["inputs"]:
        try:
            results.append(fn(*args))
        except Exception as e:
            results.append({"__error__": repr(e)[:200]}); raised = True
    emit({"status": "ok", "results": results, "raised": raised})

main()
'''


def _sanitized_env() -> dict:
    """A minimal environment. Critically, this does NOT inherit the parent's
    variables, so .env secrets (SUPABASE_KEY, OPENAI_API_KEY) are unreachable
    from student code."""
    return {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8",
            "PYTHONIOENCODING": "utf-8", "HOME": "/nonexistent"}


def run_student_code(code: str, inputs: list, entry_name: str | None = None,
                     timeout: float = _DEFAULT_TIMEOUT,
                     tests: list | None = None) -> tuple[str, list, str | None]:
    """Execute untrusted `code`. Returns (status, results, internal_error).

    status is one of: ok | policy | syntax | exec_error | no_entry | timeout |
    harness_error. Callers classify into ExecutionOutcome; this stays low-level
    so the classification lives in one place (classify_run)."""
    try:
        check_policy(code)
    except PolicyViolation as e:
        return "policy", [], str(e)
    try:
        compile(code, "<student>", "exec")
    except SyntaxError as e:
        return "syntax", [], f"{e.msg} (line {e.lineno})"

    workdir = tempfile.mkdtemp(prefix="mt_student_")
    payload_path = os.path.join(workdir, "payload.json")
    try:
        with open(payload_path, "w") as f:
            json.dump({"code": code, "inputs": inputs, "entry_name": entry_name,
                       "tests": tests, "mem": _MEM_BYTES, "cpu": _CPU_SECONDS}, f)
        proc = subprocess.run(
            [sys.executable, "-I", "-S", "-c", _STUDENT_HARNESS, payload_path],
            capture_output=True, text=True, timeout=timeout,
            cwd=workdir, env=_sanitized_env(), stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return "timeout", [], f"exceeded {timeout}s"
    except Exception as e:                       # our failure, not the student's
        return "harness_error", [], repr(e)[:200]
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    raw_out = proc.stdout or ""
    if len(raw_out) > _MAX_OUTPUT_BYTES:
        # Typed, explicit — never a silent truncation that looks like a parse bug.
        return ("harness_error", [],
                f"result payload {len(raw_out):,}B exceeds {_MAX_OUTPUT_BYTES:,}B "
                f"cap; problem unsupported at this execution budget")
    out = raw_out.strip()
    if not out:
        # Killed by a resource limit (OOM/CPU) leaves no output. That is the
        # student's program dying, not our harness failing.
        if proc.returncode and proc.returncode < 0:
            sig = -proc.returncode
            # SIGXCPU/SIGKILL = a resource ceiling the student's own program hit.
            # Any OTHER signal is an unknown termination: that is OUR problem and
            # must stay harness_error/indeterminate, never consuming an attempt.
            if sig in (9, 24):
                return "timeout", [], f"killed by signal {sig} (resource limit)"
            return "harness_error", [], f"child terminated by signal {sig}"
        return ("exec_error", [], (proc.stderr or "no output")[:200]) if proc.returncode \
            else ("harness_error", [], "empty harness output")
    try:
        data = json.loads(out.splitlines()[-1])
    except Exception:
        return "harness_error", [], "unparseable harness output"

    st = data.get("status")
    if st == "compared":
        return "compared", [data], None
    if st == "ok":
        return "ok", data.get("results", []), None
    if st == "exec_error":
        return "exec_error", [], data.get("error")
    if st == "no_entry":
        return "no_entry", [], "no entry point found"
    return "harness_error", [], f"unknown status {st!r}"


def _norm(x):
    if isinstance(x, (list, tuple)):
        return [_norm(i) for i in x]
    return x


def classify_run(code: str, tests: list, entry_name: str | None = None,
                 timeout: float = _DEFAULT_TIMEOUT) -> ExecutionResult:
    """Run untrusted code against oracle tests and return a CLASSIFIED result.

    Comparison happens INSIDE the child; only a bounded control message crosses
    the IPC boundary. A legitimately huge return value therefore passes as long
    as its computation stays inside the CPU/memory/wall budget. JSON only —
    never pickle."""
    status, payload, err = run_student_code(
        code, [t["input"] for t in tests], entry_name=entry_name,
        timeout=timeout, tests=tests)

    if status == "policy":
        return ExecutionResult(outcome="policy_violation", total=len(tests),
                               internal_error=err)
    if status == "syntax":
        return ExecutionResult(outcome="runtime_error", total=len(tests),
                               internal_error=f"syntax: {err}")
    if status == "timeout":
        return ExecutionResult(outcome="timeout", total=len(tests), internal_error=err)
    if status in ("exec_error", "no_entry"):
        return ExecutionResult(outcome="runtime_error", total=len(tests),
                               internal_error=err)
    if status != "compared":
        return ExecutionResult(outcome="harness_error", total=len(tests),
                               internal_error=err or f"unexpected status {status!r}")

    d = payload[0]
    passed, total = d.get("passed", 0), d.get("total", len(tests))
    failures = d.get("failures", [])
    if passed == total and total:
        return ExecutionResult(outcome="pass", passed=passed, total=total)
    return ExecutionResult(
        outcome="runtime_error" if d.get("raised") else "wrong_output",
        passed=passed, total=total, failures=failures[:5])
