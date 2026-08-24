"""
identity.py — one answer to "which function is this problem?" and "which cache
entry is it?"

Two failure modes this closes:

  * DIFFERENT FUNCTIONS SELECTED. _extract_signature() was called independently
    in sandbox.py, run_phase1.py, mutation.py and grader.py — four regex guesses
    that were never checked against each other, nor against what the execution
    harness actually runs. A solution carrying a helper function could leave the
    oracle testing one function while the gate assembled another. Here the entry
    point is resolved once by MIRRORING the harness's own resolution order,
    CONFIRMED by executing it, cached, and read by everyone else.

  * CACHE COLLISIONS. tests_cache.json and chunk_pool.json were keyed by slug,
    which is title-derived. Two different problems sharing a title — which
    starts happening as professor-uploaded problems flow through the same
    pipeline as curated ones — would silently share cached tests or
    decompositions. Keys now come from problem CONTENT.

content_hash is a CACHE KEY ONLY. slug remains the human-facing/DB identifier
everywhere else.
"""
import ast
import hashlib
import json
import os

from tests.sandbox import _extract_signature, run_solution

# Repo root, alongside the other regenerable caches.
_RESOLVED_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resolved_entries.json")

# Confirmation only needs to prove the function is real and callable, so it can
# be impatient — we throw away the output either way.
_CONFIRM_TIMEOUT = 3.0


def _norm(text: str) -> str:
    """Trim the edges and trailing whitespace, keep interior indentation —
    two solutions differing only in trailing spaces are the same problem, but
    two differing in indentation are not the same Python."""
    return "\n".join(line.rstrip() for line in (text or "").strip().splitlines())


def content_hash(problem: dict) -> str:
    """Stable cache key derived from what the problem IS.

    Title, slug and difficulty are deliberately excluded: they change without
    the problem changing, and two genuinely different problems can share a
    title. CACHE KEY ONLY — never a primary identifier."""
    payload = (_norm(problem.get("description", "")) + "\x00"
               + _norm(problem.get("solution", "")))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _params_of(fn: ast.FunctionDef) -> list[str]:
    return [a.arg for a in fn.args.args if a.arg != "self"]


def _mirror_resolve(solution: str) -> tuple[str | None, list[str]]:
    """Mirror of resolve_entry() inside sandbox.py's execution harness.

    Given no explicit entry_name the harness picks, in order:
      1. a `Solution` class -> its first PUBLIC method in dir() order, which is
         ALPHABETICAL, not definition order;
      2. otherwise funcs[-1] — the LAST top-level function defined.

    Both quirks are mirrored on purpose. Whatever this names is what the
    harness would have chosen on its own, so "resolved" and "executed" cannot
    disagree. Note this differs from _extract_signature(), which returns the
    FIRST def it regex-matches — that mismatch was the bug."""
    try:
        tree = ast.parse(solution)
    except SyntaxError:
        return _extract_signature(solution)      # regex fallback, better than nothing

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Solution":
            methods = [n for n in node.body
                       if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")]
            if methods:
                # A method invoked as self.<name>(...) by a sibling is a HELPER,
                # not the entry point. dir()'s alphabetical order happily picks
                # the helper: searchRange() calls binarySearch(), and "b" < "s",
                # so the oracle got built around the helper and a student would
                # be asked to implement the wrong function entirely. Drop any
                # method a sibling calls, THEN fall back to alphabetical so this
                # still matches the harness (which applies the same filter).
                called = {n.func.attr for n in ast.walk(node)
                          if isinstance(n, ast.Call)
                          and isinstance(n.func, ast.Attribute)
                          and isinstance(n.func.value, ast.Name)
                          and n.func.value.id == "self"}
                entries = [m for m in methods if m.name not in called] or methods
                first = min(entries, key=lambda n: n.name)
                return first.name, _params_of(first)

    funcs = [n for n in tree.body
             if isinstance(n, ast.FunctionDef) and not n.name.startswith("__")]
    if funcs:
        return funcs[-1].name, _params_of(funcs[-1])         # harness takes funcs[-1]
    return _extract_signature(solution)


def resolve_entry_point(problem: dict) -> dict:
    """Resolve the entry function and CONFIRM it by actually running it.

    Confirmation is not a correctness check — placeholder arguments will often
    raise inside the function, and that is fine: an exception proves the
    function was found and called. Only a harness that cannot find an entry
    point at all, or a solution that will not exec, counts as unconfirmed.

    Returns {"entry_name", "params", "confirmed"}."""
    solution = problem.get("solution", "") or ""
    name, params = _mirror_resolve(solution)
    confirmed = False

    if name:
        run = run_solution(solution, [[0] * len(params)],
                           entry_name=name, timeout=_CONFIRM_TIMEOUT)
        if run["ok"]:
            confirmed = True
        elif "timeout" in (run.get("error") or "").lower():
            # It hung inside the function — which proves it is real and ran.
            confirmed = True

    return {"entry_name": name, "params": params, "confirmed": confirmed}


def _load() -> dict:
    if os.path.exists(_RESOLVED_PATH):
        try:
            return json.load(open(_RESOLVED_PATH))
        except Exception:
            pass
    return {}


def _save(cache: dict) -> None:
    try:
        json.dump(cache, open(_RESOLVED_PATH, "w"), indent=2)
    except Exception as e:
        print(f"  ⚠️  Could not save resolved entries: {e}")


def get_resolved_entry(problem: dict) -> dict:
    """The resolved entry point for this problem, resolving and persisting on a
    miss. Every module reads its entry point through here, so they cannot pick
    different functions."""
    key = content_hash(problem)
    cache = _load()
    hit = cache.get(key)
    if isinstance(hit, dict) and "entry_name" in hit:
        return hit

    resolved = resolve_entry_point(problem)
    # slug is stored for humans reading the file — it is NOT the key.
    cache[key] = {**resolved, "slug": problem.get("slug", "")}
    _save(cache)
    return cache[key]
