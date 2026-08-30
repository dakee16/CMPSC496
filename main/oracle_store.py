"""
oracle_store.py - READ-ONLY access to validated oracle data.

Relocated out of tests/ because it is backend runtime, not test scaffolding:
the grading path depends on it. The cached data now lives under data/oracles/.

The contract that matters: grading may READ a STRONG oracle and may never
generate, validate or modify one. Oracle authoring belongs to the warm-up /
problem-publishing workflow, never to a student request - a student pressing
Submit must not be able to trigger minutes of mutation testing, and a weak or
absent oracle must never be silently accepted as a basis for a verdict.
"""
import json
import os

# Backend data location. Overridable for local development.
DEFAULT_ORACLE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "oracles", "tests_cache.json")


def cache_path() -> str:
    return os.environ.get("MICROTUTOR_ORACLE_CACHE") or DEFAULT_ORACLE_PATH


class OracleUnusableError(RuntimeError):
    """No STRONG cached oracle for this problem, so it cannot be graded.

    `reason_code` distinguishes missing / unvalidated / weak / malformed so the
    API can respond precisely instead of emitting a generic failure."""

    def __init__(self, message: str, reason_code: str):
        super().__init__(message)
        self.reason_code = reason_code


def load_cache() -> dict:
    path = cache_path()
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception:
            pass
    return {}


def save_cache(cache: dict) -> None:
    """Authoring-side only. Never called from a grading request."""
    path = cache_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(cache, open(path, "w"), indent=2)
    except Exception as e:
        print(f"  ⚠️  Could not save oracle cache: {e}")


def entry_tests(entry) -> list[dict]:
    """Tests out of a cache entry, old format (bare list) or new (dict)."""
    if isinstance(entry, list):
        return entry
    if isinstance(entry, dict):
        return entry.get("final_tests", [])
    return []


def is_validated(entry) -> bool:
    """A prior validation left a verdict here."""
    return isinstance(entry, dict) and "strong" in entry


def load_strong_cached_oracle(problem: dict) -> list[dict]:
    """READ-ONLY oracle access for the answer-checking path.

    Returns the tests when the content-hash entry exists and is STRONG.
    Raises OracleUnusableError otherwise. Never generates, never validates,
    never writes."""
    from .identity import content_hash

    entry = load_cache().get(content_hash(problem))
    slug = problem.get("slug") or problem.get("title") or "<unnamed problem>"

    if entry is None:
        raise OracleUnusableError(
            f"No cached oracle for '{slug}'. It must be validated "
            f"(python -m main.warmup) before it can be graded.", "oracle_missing")
    if isinstance(entry, list):
        raise OracleUnusableError(
            f"'{slug}' has a legacy cache entry with no verdict.",
            "oracle_unvalidated")
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
