"""Deterministic labeled-answer generation.

Ground truth never comes from the grader. Two independent sources only:
  * EXECUTION — assemble the full reference solution with one chunk replaced by
    the candidate and run it against the cached oracle directly. Passing every
    test means the answer is genuinely correct for that chunk; failing means it
    is genuinely wrong. This is the grader's *input*, not its verdict.
  * CONSTRUCTION — blank, `pass`, policy-violating and syntactically invalid
    answers are wrong by definition and need no execution.
"""
import ast
import hashlib
import json
import os

from main.execution import classify_run
from main.identity import get_resolved_entry
from main.mutation import generate_mutants
from .schema import AcceptedItem, BenchmarkCase


def _indent(code: str) -> str:
    return "\n".join("    " + ln if ln.strip() else ln for ln in code.splitlines())


def assemble(header: str, bodies) -> str:
    body = "\n".join(b.rstrip() for b in bodies if b and b.strip())
    return header + "\n" + (_indent(body) if body.strip() else "    pass")


def _truth(header, refs, idx, candidate, tests, entry) -> str:
    """INDEPENDENT ground truth: swap one chunk, run the oracle directly."""
    bodies = list(refs)
    bodies[idx] = candidate
    out = classify_run(assemble(header, bodies), tests, entry_name=entry)
    return "correct" if out.outcome == "pass" else "incorrect"


def _rename(code: str, params=(), prefix_code: str = "", suffix="_v") -> str | None:
    """Symbol-aware, semantics-preserving rename.

    v1 renamed every name the chunk BOUND, which mislabeled 6 cases: it renamed
    function PARAMETERS (powx-n: n, x) and names already bound by the accepted
    PREFIX (add-binary: result), producing read-before-assignment NameErrors.
    Those answers genuinely crash, so "correct" was the wrong label and the
    grader was right to reject them.

    Renames only names that are (a) bound in THIS chunk, (b) not parameters,
    (c) not bound by the prefix, (d) not read before their first binding here,
    and never touches imports, attributes, function names or builtins.
    """
    import builtins
    import re as _re
    try:
        tree = ast.parse("def _w():\n" + _indent(code))
    except SyntaxError:
        return None

    bound, first_load, first_store, order = set(), {}, {}, 0
    imported, func_names = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for al in n.names:
                imported.add((al.asname or al.name).split(".")[0])
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_names.add(n.name)
        elif isinstance(n, ast.Name):
            order += 1
            if isinstance(n.ctx, ast.Store):
                bound.add(n.id); first_store.setdefault(n.id, order)
            else:
                first_load.setdefault(n.id, order)

    prefix_bound = set()
    if prefix_code.strip():
        try:
            pt = ast.parse("def _p():\n" + _indent(prefix_code))
            prefix_bound = {x.id for x in ast.walk(pt)
                            if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store)}
        except SyntaxError:
            return None

    safe = set()
    for name in bound:
        if name in set(params) or name in prefix_bound:
            continue                       # parameter / prefix-owned
        if name in imported or name in func_names or name in dir(builtins):
            continue                       # import, def, builtin
        if name in first_load and first_load[name] < first_store.get(name, 10**9):
            continue                       # read before assignment here
        safe.add(name)
    if not safe:
        return None

    out = code
    for nm in sorted(safe, key=len, reverse=True):
        out = _re.sub(rf"\b{nm}\b", nm + suffix, out)
    return out if out != code else None


def build_cases(problems: dict, pool: dict, cache: dict, limit_per_chunk=None):
    """problems: hash -> problem dict. Returns [BenchmarkCase]."""
    cases = []

    def add(slug, h, did, idx, answer, expected, cat, src, why, prefix=None):
        cid = hashlib.sha1(f"{h}:{idx}:{cat}:{answer}".encode()).hexdigest()[:16]
        cases.append(BenchmarkCase(
            case_id=cid, slug=slug, content_hash=h, decomposition_id=did,
            chunk_index=idx, student_answer=answer, expected=expected,
            category=cat, source=src, rationale=why,
            accepted_prefix=prefix or []))

    for h, problem in problems.items():
        entries = pool.get(h) or []
        cached = cache.get(h)
        if not entries or not isinstance(cached, dict):
            continue
        dec = entries[0]
        header, chunks = dec["header"], dec["chunks"]
        refs = [c.get("reference", "") for c in chunks]
        tests = cached["final_tests"]
        slug = cached.get("slug", "?")
        did = hashlib.sha1(json.dumps(refs).encode()).hexdigest()[:12]
        entry = get_resolved_entry(problem)["entry_name"]

        for idx, ref in enumerate(refs):
            if not ref.strip():
                continue
            pre = [AcceptedItem(code=r) for r in refs[:idx]]

            # 1 exact reference — correct by the gate's own construction
            add(slug, h, did, idx, ref, "correct", "exact_reference",
                "deterministic", "verbatim reference chunk", pre)

            # 8/7/9/12/11 — wrong by construction, no execution needed
            add(slug, h, did, idx, "", "incorrect", "blank",
                "deterministic", "empty answer is wrong by definition", pre)
            add(slug, h, did, idx, "pass", "incorrect", "no_op",
                "deterministic", "no-op cannot implement the step", pre)
            add(slug, h, did, idx, ref + "\n    if (", "incorrect", "syntax_error",
                "deterministic", "unbalanced paren cannot parse", pre)
            add(slug, h, did, idx, "import os\n" + ref, "incorrect",
                "policy_violation", "deterministic", "banned import", pre)
            add(slug, h, did, idx, "while True:\n    pass\n" + ref, "incorrect",
                "timeout", "deterministic", "unconditional infinite loop", pre)
            add(slug, h, did, idx, "raise ValueError('boom')\n" + ref, "incorrect",
                "runtime_error", "deterministic", "unconditional raise", pre)

            # 2 renamed variables — semantics-preserving, label known a priori
            ren = _rename(ref, params=get_resolved_entry(problem)["params"],
                          prefix_code="\n".join(refs[:idx]))
            # ADMISSION GATE: the transformed complete function must pass the
            # STRONG oracle via the direct labeling executor — never the grader.
            if ren and _truth(header, refs, idx, ren, tests, entry) == "correct":
                add(slug, h, did, idx, ren, "correct", "renamed_variables",
                    "deterministic",
                    "symbol-aware rename; transformed function verified against "
                    "the oracle by the direct executor", pre)

            # 5/6 mutation-derived errors, each PROVEN wrong by direct execution
            n_mut = 0
            for m in generate_mutants("def _w():\n" + _indent(ref)):
                cand = "\n".join(ln[4:] if ln.startswith("    ") else ln
                                 for ln in m["code"].splitlines()[1:])
                if not cand.strip() or cand == ref:
                    continue
                if _truth(header, refs, idx, cand, tests, entry) != "incorrect":
                    continue                       # mutant is equivalent; skip
                cat = ("boundary_off_by_one"
                       if any(t in m["label"] for t in ("-> <=", "-> >=", "0 -> 1",
                                                        "1 -> 2", "-> >", "-> <"))
                       else "clean_logical_error")
                add(slug, h, did, idx, cand, "incorrect", cat, "deterministic",
                    f"mutation {m['label']} proven to fail the oracle", pre)
                n_mut += 1
                if n_mut >= (limit_per_chunk or 2):
                    break

            # 14/15 wrong answer AFTER a divergent / revealed prefix (non-final)
            if idx > 0:
                prev_ren = _rename(refs[idx - 1],
                                   params=get_resolved_entry(problem)["params"],
                                   prefix_code="\n".join(refs[:idx - 1]))
                if prev_ren and prev_ren != refs[idx - 1]:
                    div = [AcceptedItem(code=r) for r in refs[:idx - 1]] + \
                          [AcceptedItem(code=prev_ren)]
                    add(slug, h, did, idx, "pass", "incorrect",
                        "wrong_after_divergent_prefix", "deterministic",
                        "no-op current chunk after an accepted divergent prefix",
                        div)
                rev = [AcceptedItem(code=r) for r in refs[:idx - 1]] + \
                      [AcceptedItem(code=refs[idx - 1], provenance="revealed_reference")]
                add(slug, h, did, idx, "pass", "incorrect",
                    "wrong_after_revealed_prefix", "deterministic",
                    "no-op current chunk after a revealed-reference prefix", rev)
    return cases
