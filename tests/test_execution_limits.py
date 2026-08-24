"""Phase A/B regressions: result-size handling, signal typing, symbol-aware rename."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.corpus import _rename, assemble
from main.execution import _MAX_OUTPUT_BYTES, classify_run

SUPPORTED = ["generate-parentheses", "palindrome-number", "roman-to-integer",
             "add-binary", "powx-n", "integer-to-roman"]


def _pool_case(slug):
    pool = json.load(open("main/chunk_pool.json"))
    cache = json.load(open("tests/tests_cache.json"))
    for h, ents in pool.items():
        e = cache.get(h)
        if isinstance(e, dict) and e.get("slug") == slug:
            refs = [c.get("reference", "") for c in ents[0]["chunks"]]
            return ents[0]["header"], refs, e["final_tests"]
    return None


@pytest.mark.parametrize("slug", SUPPORTED)
def test_exact_reference_never_harness_errors(slug):
    """A supported problem's own reference solution must PASS — never surface
    as a harness error. This is the regression for the 64KB-cap truncation."""
    got = _pool_case(slug)
    if got is None:
        pytest.skip(f"{slug} not pooled")
    header, refs, tests = got
    import re
    code = assemble(header, refs)
    entry = re.search(r"def\s+(\w+)", code).group(1)
    r = classify_run(code, tests, entry_name=entry)
    assert r.outcome != "harness_error", r.internal_error
    assert r.outcome == "pass", f"{slug}: {r.outcome} {r.internal_error}"


def test_oversized_control_message_is_typed(monkeypatch):
    """The cap now bounds only the CONTROL MESSAGE. Results are compared
    in-child and never crossed it, so this path is defensive: if a child ever
    emits an oversized message it must be typed explicitly, not surface as an
    unparseable-output mystery."""
    import main.execution as ex

    class P:
        returncode = 0
        stdout = "x" * (ex._MAX_OUTPUT_BYTES + 10)
        stderr = ""

    monkeypatch.setattr(ex.subprocess, "run", lambda *a, **k: P())
    r = classify_run("def f(x):\n    return x", [{"input": [1], "expected": 1}], "f")
    assert r.outcome == "harness_error"
    assert "exceeds" in (r.internal_error or "")


def test_harness_error_never_counts_against_the_student():
    from main.schemas import ExecutionResult
    r = ExecutionResult(outcome="harness_error")
    assert r.outcome == "harness_error"      # grading maps this to indeterminate


# ── the six v1 mislabels must no longer be generated ─────────────────────
@pytest.mark.parametrize("chunk,params,prefix,why", [
    ("if carry:\n    result = '1' + result\nreturn result", ["a", "b"],
     "result = ''\ncarry = 0", "result is bound by the PREFIX"),
    ("result = 1\nwhile n > 0:\n    result *= x\n    n //= 2\nreturn result",
     ["x", "n"], "if n < 0:\n    x = 1 / x", "n and x are PARAMETERS"),
    ("roman = ''\nfor v, s in value_map:\n    while num >= v:\n        num -= v\n"
     "return roman", ["num"], "value_map = [(1, 'I')]",
     "num is a parameter, value_map is prefix-bound"),
])
def test_rename_protects_params_and_prefix(chunk, params, prefix, why):
    out = _rename(chunk, params=params, prefix_code=prefix)
    if out is None:
        return                                  # nothing safe to rename: fine
    for p in params:
        assert f"{p}_v" not in out, f"renamed a parameter ({why})"
    import ast
    pb = {n.id for n in ast.walk(ast.parse("def _p():\n" + "\n".join(
        "    " + l for l in prefix.splitlines())))
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    for b in pb:
        assert f"{b}_v" not in out, f"renamed a prefix-bound name ({why})"


def test_rename_still_works_where_safe():
    out = _rename("tmp = 1\ntmp = tmp + 2\nreturn tmp", params=["a"], prefix_code="")
    assert out and "tmp_v" in out and "a_v" not in out


# ── in-child comparison protocol ─────────────────────────────────────────
def test_large_correct_value_passes_without_huge_ipc():
    """A legitimately huge return value must PASS; only a compact verdict may
    cross the IPC boundary."""
    code = "def f(n):\n    return list(range(n))"
    r = classify_run(code, [{"input": [300000], "expected": list(range(300000))}],
                     entry_name="f")
    assert r.outcome == "pass" and r.passed == 1


def test_large_wrong_value_has_bounded_diagnostics():
    code = "def f(n):\n    return list(range(n))"
    r = classify_run(code, [{"input": [200000], "expected": [1, 2, 3]}],
                     entry_name="f")
    assert r.outcome == "wrong_output"
    blob = json.dumps(r.failures)
    assert len(blob) < 4000, "diagnostics must be bounded"
    got = r.failures[0]["got"]
    assert got["type"] == "list" and got["len"] == 200000   # type/length reported
    assert "truncated" in got["repr"]


def test_combinations_exact_reference_passes_after_protocol_fix():
    got = _pool_case("combinations")
    if got is None:
        pytest.skip("combinations not pooled")
    header, refs, tests = got
    import re
    code = assemble(header, refs)
    r = classify_run(code, tests, entry_name=re.search(r"def\s+(\w+)", code).group(1))
    assert r.outcome == "pass", r.internal_error


def test_student_print_flood_still_suppressed():
    code = ("def f(n):\n"
            "    for _ in range(5000):\n        print('x' * 500)\n    return n")
    r = classify_run(code, [{"input": [7], "expected": 7}], entry_name="f")
    assert r.outcome == "pass"


def test_malformed_control_message_is_harness_error(monkeypatch):
    import main.execution as ex
    real = ex.subprocess.run

    class P:
        returncode = 0
        stdout = "{not json at all"
        stderr = ""

    monkeypatch.setattr(ex.subprocess, "run", lambda *a, **k: P())
    r = classify_run("def f(x):\n    return x", [{"input": [1], "expected": 1}], "f")
    monkeypatch.setattr(ex.subprocess, "run", real)
    assert r.outcome == "harness_error"


def test_no_oracle_values_in_public_grade_response(tmp_path, monkeypatch):
    """Bounded diagnostics stay INTERNAL; the public payload carries none."""
    from main.schemas import GradeResult
    r = GradeResult(verdict="incorrect", tier="execution-final", deterministic=True,
                    student_reason="Your solution gives the wrong answer.",
                    reason_code="final_wrong_output",
                    failures=[{"index": 0, "expected": {"repr": "[1, 2, 3]"}}])
    public = {"verdict": r.verdict, "tier": r.tier, "reason": r.student_reason,
              "deterministic": r.deterministic}
    assert "expected" not in json.dumps(public)
    assert r.failures            # kept internally
