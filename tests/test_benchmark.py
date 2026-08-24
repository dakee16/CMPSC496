"""Offline tests for the benchmark harness itself."""
import hashlib
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.schema import SCHEMA_VERSION, BenchmarkCase, CaseResult

SOLUTION = ("class Solution:\n    def add(self, a, b):\n"
            "        total = a + b\n        return total\n")
PROBLEM = {"slug": "adder", "title": "Adder", "description": "Return a + b.",
           "solution": SOLUTION}
TESTS = [{"input": [1, 2], "expected": 3}, {"input": [4, 5], "expected": 9}]
REFS = ["total = a + b", "return total"]


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    import main.identity as ident, tests.sandbox as sb
    os.environ["MICROTUTOR_SESSION_DB"] = str(tmp_path / "s.sqlite3")
    cf = tmp_path / "cache.json"
    monkeypatch.setattr(sb, "_CACHE_PATH", str(cf))
    monkeypatch.setattr(ident, "_RESOLVED_PATH", str(tmp_path / "r.json"))
    h = ident.content_hash(PROBLEM)
    cache = {h: {"final_tests": TESTS, "strong": True, "kill_rate": 1.0,
                 "kill_rate_direct": 1.0, "slug": "adder",
                 "validated_at": "2026-01-01T00:00:00+00:00"}}
    cf.write_text(json.dumps(cache))
    pool = {h: [{"header": "def add(a, b):",
                 "chunks": [{"step_id": f"Part {i+1}", "prompt": "p",
                             "expected_type": "code", "reference": r}
                            for i, r in enumerate(REFS)]}]}
    return {h: PROBLEM}, pool, cache, tmp_path


def test_schema_validates_and_rejects_bad_label():
    c = BenchmarkCase(case_id="a", slug="s", content_hash="h", decomposition_id="d",
                      chunk_index=0, student_answer="x", expected="correct",
                      category="exact_reference", source="deterministic",
                      rationale="why")
    assert c.schema_version == SCHEMA_VERSION
    with pytest.raises(Exception):
        BenchmarkCase(case_id="a", slug="s", content_hash="h", decomposition_id="d",
                      chunk_index=0, student_answer="x", expected="maybe",
                      category="exact_reference", source="deterministic",
                      rationale="why")


def test_deterministic_generation_and_reproducibility(frozen):
    from evaluation.corpus import build_cases
    problems, pool, cache, _ = frozen
    a = build_cases(problems, pool, cache)
    b = build_cases(problems, pool, cache)
    assert a and [c.case_id for c in a] == [c.case_id for c in b]
    assert [c.student_answer for c in a] == [c.student_answer for c in b]


def test_labels_are_independent_of_grader(frozen):
    """Ground truth comes from executing the assembled candidate, not from the
    grader. Verify the exact-reference case really does pass the oracle."""
    from main.execution import classify_run
    from main.identity import get_resolved_entry
    from evaluation.corpus import assemble
    problems, pool, cache, _ = frozen
    entry = get_resolved_entry(PROBLEM)["entry_name"]
    ok = classify_run(assemble("def add(a, b):", REFS), TESTS, entry_name=entry)
    assert ok.outcome == "pass"
    bad = classify_run(assemble("def add(a, b):", ["total = a * b", "return total"]),
                       TESTS, entry_name=entry)
    assert bad.outcome == "wrong_output"


def test_every_case_has_rationale_and_source(frozen):
    from evaluation.corpus import build_cases
    problems, pool, cache, _ = frozen
    for c in build_cases(problems, pool, cache):
        assert c.rationale and c.source == "deterministic"
        assert c.expected in ("correct", "incorrect", "ambiguous")


def test_metrics_math(tmp_path):
    from evaluation.run_answer_benchmark import metrics
    mk = lambda e, a, cat="blank": CaseResult(
        case_id=e + a + cat, slug="s", chunk_index=0, category=cat, expected=e,
        actual=a, tier="execution-final", deterministic=True, reason_code="r")
    rs = [mk("correct", "correct"), mk("incorrect", "incorrect"),
          mk("incorrect", "correct", "no_op"),      # false accept
          mk("correct", "incorrect", "syntax_error")]  # false reject
    s = metrics(rs, [], str(tmp_path))
    assert s["accuracy"] == 0.5
    assert s["false_acceptance_rate"] == 0.25
    assert s["false_rejection_rate"] == 0.25
    assert len(s["misclassified"]) == 2


def test_ambiguous_excluded_from_accuracy(tmp_path):
    from evaluation.run_answer_benchmark import metrics
    mk = lambda e, a: CaseResult(case_id=e + a, slug="s", chunk_index=0,
                                 category="ambiguous_tier4", expected=e, actual=a,
                                 tier="llm-judge", deterministic=False,
                                 reason_code="r")
    s = metrics([mk("correct", "correct"), mk("ambiguous", "incorrect")], [],
                str(tmp_path))
    assert s["unambiguous"] == 1 and s["ambiguous_excluded"] == 1
    assert s["accuracy"] == 1.0


def test_public_report_has_no_private_material(tmp_path):
    from evaluation.run_answer_benchmark import metrics
    r = CaseResult(case_id="c1", slug="adder", chunk_index=0, category="blank",
                   expected="incorrect", actual="incorrect", tier="syntax",
                   deterministic=True, reason_code="blank_answer")
    metrics([r], [], str(tmp_path))
    for name in ("summary.json", "results.csv", "report.md"):
        blob = open(os.path.join(str(tmp_path), name)).read()
        assert "total = a + b" not in blob      # no reference
        assert "class Solution" not in blob     # no solution
        assert '"expected": 3' not in blob      # no oracle test values
        assert "adapted_tail" not in blob


def test_runner_uses_shared_engine():
    """The benchmark must measure the real engine, not a copy."""
    import inspect
    from evaluation import run_answer_benchmark as rb
    src = inspect.getsource(rb.run)
    assert "from main.grading import grade_submission" in src
    assert "grade_submission(session" in src


def test_resume_does_not_duplicate(tmp_path):
    raw = tmp_path / "results_private.jsonl"
    r = CaseResult(case_id="dup", slug="s", chunk_index=0, category="blank",
                   expected="incorrect", actual="incorrect", tier="syntax",
                   deterministic=True, reason_code="blank_answer")
    raw.write_text(r.model_dump_json() + "\n")
    done = {json.loads(l)["case_id"] for l in open(raw)}
    assert done == {"dup"}        # a resumed run skips exactly these


def test_production_files_untouched_by_import():
    """Importing the benchmark must not rewrite production caches."""
    before = [hashlib.md5(open(p, "rb").read()).hexdigest()
              for p in ("tests/tests_cache.json", "main/chunk_pool.json")]
    import importlib
    import evaluation.run_answer_benchmark as rb
    importlib.reload(rb)
    after = [hashlib.md5(open(p, "rb").read()).hexdigest()
             for p in ("tests/tests_cache.json", "main/chunk_pool.json")]
    assert before == after
