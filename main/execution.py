"""
execution.py - hardened execution path for UNTRUSTED student code.

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
import os
import shutil
import subprocess
import sys
import tempfile

from .context import SEQ_ENTRY
from .pyvalue import SOURCE as _PYVALUE_SRC, dumps as _dumps, loads as _loads
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


def _policed_nodes(tree: ast.AST):
    """Every node in `tree` EXCEPT the injected call-sequence driver.

    What `code` holds is the whole assembled MODULE, never the student's
    fragment alone: a method problem is its entire class, plus the driver that
    replays call sequences and runs block tests. That driver legitimately needs
    `import ast`, `globals()`, `eval`, `exec` and `compile` - every one of which
    this policy bans. Walking it refused OUR OWN code and reported it to the
    student as their violation ("the 'ast' module isn't allowed here"), which
    made every class problem ungradeable: even the teacher's own reference
    solution was rejected. A student only ever saw it once their answer PARSED,
    because an unparseable one returns above before reaching here - so the
    message always landed on whichever edit happened to fix their syntax.

    Skipped by MODULE-LEVEL name, and that is what stops it being a bypass. A
    student's chunk is always spliced into a function body or a class method,
    so nothing they write is ever a direct child of the module: a
    `def _mt_run_calls` typed into an answer is nested, and stays policed."""
    stack = [n for n in ast.iter_child_nodes(tree)
             if not (isinstance(n, ast.FunctionDef) and n.name == SEQ_ENTRY)]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(ast.iter_child_nodes(node))


def check_policy(code: str) -> None:
    """AST safety policy. Raises PolicyViolation with a student-safe message.

    Runs BEFORE execution - a refused program is never executed at all."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return                      # syntax is classified separately, not here

    for node in _policed_nodes(tree):
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


# ONE DEFINITION, TWO USES, for the same reason pyvalue.SOURCE is shared.
# The child compares in-process (it must - a 7MB result cannot cross the
# IPC boundary) and main/mutation.py compares in the parent when it decides
# whether a mutant was killed. As two separate functions they DRIFTED, and
# the drift is what made stack-push ungradeable: see the docstring below.
_NORM_SRC = r'''
def norm(x):
    """Put a value on the same footing as a STORED expected value.

    Expected values crossed pyvalue.mt_lit on their way into the oracle cache,
    and mt_lit has no literal form for an arbitrary object, so it writes
    repr(str(v)) - a Node came back out as the STRING 'Node(5)'. Comparison then
    put a live Node beside that string and they never matched.

    Measured on the real stack-push oracle: 2 of its 15 tests read x.top, whose
    value is a Node, so the TEACHER'S OWN implementation scored 13/15 and the
    problem was ungradeable for everybody. Mutation testing could not see it
    either - those two tests were unkillable, so they made the oracle look
    weaker rather than making the bug visible.

    Anything mt_lit would have stringified is stringified here too."""
    if isinstance(x, (list, tuple)):
        return [norm(i) for i in x]
    if isinstance(x, dict):
        return {norm(k): norm(v) for k, v in x.items()}
    if isinstance(x, (set, frozenset)):
        return {norm(i) for i in x}
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    return str(x)
'''

exec(_NORM_SRC, globals())          # parent-side norm, same definition
_norm = norm                        # the name main/mutation.py imports


# Child-process harness. Mirrors the trusted harness's entry resolution so a
# student candidate and the reference resolve the same function, but adds
# resource limits and output caps.
#
# Payload and result are PYTHON LITERALS, not JSON: JSON has no integer dict
# keys, no tuples and no sets, so a problem keyed by year received '2019' where
# it wrote 2019 and every correct solution raised KeyError. The encoder is
# prepended from main/pyvalue.py - it cannot be imported here, because this runs
# under `-I`, which strips the script directory from sys.path.
_STUDENT_HARNESS = _PYVALUE_SRC + _NORM_SRC + r'''
import ast as _ast, json, re, sys, os
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
    payload = _ast.literal_eval(open(sys.argv[1]).read())
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
        print(mt_lit(obj))
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
    from student code.

    PYTHONHASHSEED IS PINNED, and that is a correctness fix rather than a
    hardening one. Python randomizes string hashing per process, so set and
    dict iteration order changes between runs of the SAME program: six runs of
    `list(set(txt.split()))` produced six different orderings. Two things break
    on that. An oracle whose expected value came from one such run fails a
    correct solution on the next - a flaky verdict that looks exactly like a
    student bug. And main/bridge.py compares captured variable values across
    two processes, so an unpinned seed makes a correct student's state never
    match the reference's.

    Zero rather than a random constant: the value has to be the same in every
    process that will ever be compared, including oracle generation months
    earlier, so it cannot be drawn at runtime."""
    return {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8",
            "PYTHONIOENCODING": "utf-8", "HOME": "/nonexistent",
            "PYTHONHASHSEED": "0"}


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
    payload_path = os.path.join(workdir, "payload.txt")
    try:
        with open(payload_path, "w") as f:
            f.write(_dumps({"code": code, "inputs": inputs,
                            "entry_name": entry_name, "tests": tests,
                            "mem": _MEM_BYTES, "cpu": _CPU_SECONDS}))
        # -P -s -S, NOT -I. `-I` is `-E -P -s` together, and the `-E` half makes
        # the interpreter ignore every PYTHON* variable - including
        # PYTHONHASHSEED, which _sanitized_env pins for the correctness reason
        # documented there. Under -I the pin was silently a no-op: six runs of
        # one correct program still produced six different set orderings.
        #
        # Dropping -E costs nothing HERE, and only here, because the child's
        # environment is not inherited - it is the dict _sanitized_env builds,
        # holding five variables and no secrets. -E exists to defend against a
        # hostile ambient environment; there isn't one to defend against.
        # -P (no cwd on sys.path) and -s (no user site-packages) are the halves
        # that actually isolate, and both are kept.
        proc = subprocess.run(
            [sys.executable, "-P", "-s", "-S", "-c", _STUDENT_HARNESS, payload_path],
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
        # Typed, explicit - never a silent truncation that looks like a parse bug.
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
        data = _loads(out.splitlines()[-1])
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


def classify_run(code: str, tests: list, entry_name: str | None = None,
                 timeout: float = _DEFAULT_TIMEOUT) -> ExecutionResult:
    """Run untrusted code against oracle tests and return a CLASSIFIED result.

    Comparison happens INSIDE the child; only a bounded control message crosses
    the IPC boundary. A legitimately huge return value therefore passes as long
    as its computation stays inside the CPU/memory/wall budget. JSON only -
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


if __name__ == "__main__":
    # Self-check.  python -m main.execution
    # Real subprocesses, no model, no oracle cache.

    # ── the sandbox is deterministic across processes ─────────────────────
    # PYTHONHASHSEED randomises set and dict iteration order per process, so
    # two runs of ONE correct program disagreed and an oracle built from the
    # first failed the second. Pinning it only works because the interpreter is
    # launched with -P -s -S rather than -I: `-I` implies `-E`, which makes
    # every PYTHON* variable - including the pin - a silent no-op.
    _code = "def f(t):\n    return {'w': list(set(t.split()))}"
    _seen = {repr(run_student_code(_code, [["the cat sat on the mat by a door"]],
                                   entry_name="f")[1]) for _ in range(6)}
    assert len(_seen) == 1, f"set order varies across processes: {_seen}"

    # ...and dropping -E must not have opened the sandbox up.
    _t = [{"input": [1], "expected": 1}]
    for _src, _why in (
            ("import os\ndef f(x):\n    return 1", "a banned import"),
            ("def f(x):\n    return open('/etc/passwd')", "open()"),
            ("import main.grading\ndef f(x):\n    return 1", "the app's own package"),
            ("def f(x):\n    return eval('1')", "eval")):
        assert classify_run(_src, _t, entry_name="f").outcome == "policy_violation", _why
    assert classify_run("def f(x):\n    return x", _t, entry_name="f").outcome == "pass"

    # ── an object is compared the way it was STORED ───────────────────────
    # mt_lit has no literal form for an arbitrary object, so an expected value
    # that is one was written as repr(str(v)) and read back as a STRING. The
    # live object was then compared against that string and never matched: on
    # the real stack-push oracle the teacher's own implementation scored 13/15,
    # and the two tests were unkillable, so mutation testing never saw it.
    class _N:
        def __repr__(self):
            return "Node(5)"
    assert _norm(_N()) == "Node(5)"
    assert _norm([1, _N()]) == [1, "Node(5)"]
    assert _norm({"a": _N()}) == {"a": "Node(5)"}
    # Values that DO have a literal form are untouched - including the ones
    # pyvalue.py exists to preserve.
    for _v in (0, 1.5, True, False, None, "s", [1, 2], {2019: 1}, {1, 2}):
        assert _norm(_v) == _v, _v
    assert _norm((1, 2)) == [1, 2], "a tuple is stored as a list"

    print("execution.py self-check OK")
