"""review_help.py - what an instructor reviewing a problem is shown.

A problem lands in review when preparation's mutation check could not decide:
some deliberate one-line breaks of the teacher's solution went unnoticed by
every generated test. The engine knows WHICH changes slipped past and on which
line; it does not know, in words, what case the tests are missing. So:

  * one model call per problem (capped, remembered) puts each change into a
    plain sentence and proposes example STATEMENTS that might expose them;
  * then, for free, those statements are run on the teacher's code and on every
    missed change. The teacher's code supplies each answer - the model's opinion
    of what the output is is never asked for - and a change counts as a REAL GAP
    only when a statement actually gives it a different answer. No case found
    is said as exactly that ("may be harmless"), never as proof.

The examples are offered as doctest text to add to the problem's docstring,
where they become tests (context.doctest_block / examples_as_inputs).

9 Oct, after HW4's `update` sat in review with nothing on screen but "a few
checks came back inconclusive".
"""
from __future__ import annotations

import ast
import json
import os

from .context import _entry_def, is_method, reference_program
from .ollama_client import OPENAI_MODEL, chat

MODEL = OPENAI_MODEL
_MAX_TOKENS = 1500                 # a full answer is ~400 tokens
_SHOW = 900                        # characters of statement / class doc sent


# ── the store: one answer per problem version ────────────────────────────

def _path() -> str:
    from .oracle_store import cache_path
    return os.path.join(os.path.dirname(cache_path()) or ".", "review_help.json")


def _load() -> dict:
    try:
        with open(_path()) as f:
            return json.load(f)
    except Exception:
        return {}


def _save(store: dict) -> None:
    path = _path()
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(store, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# v2 (13 Oct): separate test cases, each with what it catches. Answers stored
# in the v1 shape are simply not found, and are written again once.
_VERSION = "v2:"


def cached(key: str) -> dict | None:
    return _load().get(_VERSION + key)


# ── marked harmless, by the teacher (14 Oct) ─────────────────────────────
# A missed change no test CAN catch - HW4's mruEvict unlinks a `previous` that
# is always already empty - is not a gap. The teacher says so once, by name,
# for this problem version, and it stops being counted.

def dismissed(key: str) -> dict:
    """{label: {"by", "at"}} for this problem version."""
    return _load().get("dismissed:" + key) or {}


def dismiss(key: str, label: str, by: str) -> dict:
    from datetime import datetime, timezone
    store = _load()
    marks = store.get("dismissed:" + key) or {}
    marks[label] = {"by": by, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    store["dismissed:" + key] = marks
    _save(store)
    return marks


# ── running example statements, free ─────────────────────────────────────

_RUNNER = '''

def _mt_examples():
    import ast as _a
    ns = dict(globals())
    out = []
    for src in __STATEMENTS__:
        try:
            tree = _a.parse(src)
            if len(tree.body) == 1 and isinstance(tree.body[0], _a.Expr):
                v = eval(compile(_a.Expression(tree.body[0].value), "<ex>", "eval"), ns)
                out.append(None if v is None else repr(v))
            else:
                exec(compile(tree, "<ex>", "exec"), ns)
                out.append(None)
        except Exception as exc:
            out.append("!" + type(exc).__name__ + ": " + str(exc)[:200])
            break
    return out
'''


def run_examples(program: str, statements: list[str]) -> list | None:
    """What each statement shows when run after `program`, the way a doctest
    shows it: repr of an expression's value, nothing for None or a statement.
    An error stops the run and is recorded as "!Name: message". None when the
    program itself cannot run."""
    from tests.sandbox import run_solution
    src = program + _RUNNER.replace("__STATEMENTS__", repr(list(statements)))
    r = run_solution(src, [[]], entry_name="_mt_examples")
    if not r.get("ok") or not r.get("results"):
        return None
    got = r["results"][0]
    return got if isinstance(got, list) else None


def doctest_text(statements: list[str], shown: list) -> str:
    """The statements and the teacher's answers, as docstring examples."""
    lines = []
    for src, out in zip(statements, shown):
        first, *rest = src.splitlines() or [""]
        lines.append(f">>> {first}")
        lines += [f"... {ln}" for ln in rest]
        if out is not None:
            lines += [ln if ln.strip() else "<BLANKLINE>" for ln in str(out).splitlines()]
    return "\n".join(lines)


def insert_examples(solution: str, text: str, entry: str | None = None) -> str | None:
    """`solution` with `text` added at the END of its exercise's docstring,
    indented to match. None when it has no docstring to add to."""
    try:
        tree = ast.parse(solution)
    except SyntaxError:
        return None
    fn = _entry_def(tree, entry)
    if fn is None or not fn.body or not (
            isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant)
            and isinstance(fn.body[0].value.value, str)):
        return None
    doc = fn.body[0]
    lines = solution.splitlines()
    pad = " " * (fn.col_offset + 4)
    close_i = doc.end_lineno - 1
    closing = lines[close_i]
    quote = '"""' if closing.rstrip().endswith('"""') else "'''"
    body = [pad + ln if ln.strip() else "" for ln in text.splitlines()]
    if closing.strip() == quote:                      # quotes on their own line
        new = lines[:close_i] + [""] + body + lines[close_i:]
    else:                                             # """One line."""
        head = closing.rstrip()[:-3].rstrip()
        new = lines[:close_i] + [head, ""] + body + [pad + quote] + lines[close_i + 1:]
    return "\n".join(new) + ("\n" if solution.endswith("\n") else "")


def _programs(problem: dict, labels: list[str]) -> tuple[str, dict]:
    """The teacher's runnable program, and one per missed change by label."""
    from .mutation import _runnable, generate_mutants
    by_label = {m["label"]: m["code"] for m in generate_mutants(problem.get("solution") or "")}
    return (reference_program(problem),
            {lb: _runnable(problem, by_label[lb]) for lb in labels if lb in by_label})


def check(problem: dict, statements: list[str], labels: list[str]) -> dict:
    """Run the statements on the teacher's code and on every missed change.
    Free. {ok, shown, doctest, caught: {label: bool}, reason}."""
    ref, broken = _programs(problem, labels)
    shown = run_examples(ref, statements)
    if shown is None:
        return {"ok": False, "reason": "the examples could not run"}
    if any(isinstance(s, str) and s.startswith("!") for s in shown):
        bad = next(s for s in shown if isinstance(s, str) and s.startswith("!"))
        return {"ok": False, "reason": f"your solution raised {bad[1:]} on them"}
    caught = {}
    for lb, prog in broken.items():
        got = run_examples(prog, statements)
        caught[lb] = got != shown
    return {"ok": True, "shown": shown, "doctest": doctest_text(statements, shown),
            "caught": caught}


# ── the one model call ───────────────────────────────────────────────────

def _prompt(problem: dict, labels: list[str]) -> str:
    cls = problem.get("group_title")
    numbered = "\n".join(f"{i}. {lb}" for i, lb in enumerate(labels, 1))
    return (
        f"A teacher's solution{f' (a method of class {cls})' if cls else ''}:\n\n"
        f"{problem.get('solution') or ''}\n\n"
        + (f"The class it belongs to says:\n{(problem.get('group_description') or '')[:_SHOW]}\n\n"
           if cls else "")
        + f"Each of these one-line changes to it was made on purpose, and NO "
        f"existing test noticed:\n{numbered}\n\n"
        f"For each change, write for the teacher, in plain words with no jargon:\n"
        f'- "title": one short sentence starting "A student could", saying what '
        f"mistake this change stands for.\n"
        f'- "missing": a few words naming the kind of test case that would tell '
        f'it apart, e.g. "a replacement that is too big to fit".\n'
        f"Then write \"tests\": one small test case per change (fewer if one "
        f"case covers several). Each is a list of up to 8 Python statements, run "
        f"in order in a fresh session where the file's classes and functions "
        f"exist, that would show a different result if that change were made. "
        f"Build every object from scratch inside each test. End with plain "
        f"expressions whose values reveal the result. Do NOT write outputs or "
        f"comments.\n\n"
        f'Return JSON only: {{"changes": [{{"n": 1, "title": "...", "missing": '
        f'"..."}}], "tests": [["x = ...", "x.method(...)"], ["y = ...", "y.f()"]]}}')


def explain(problem: dict, key: str, labels: list[str]) -> dict:
    """The review help for this problem version: remembered after the first
    time, so a page reopened does not pay again."""
    store = _load()
    if _VERSION + key in store:
        return store[_VERSION + key]
    raw = chat(MODEL, "You explain test gaps to teachers and write example code, "
                      "as strict JSON. No prose.",
               [{"role": "user", "content": _prompt(problem, labels)}],
               temperature=0.2, fmt="json", max_tokens=_MAX_TOKENS)
    try:
        data = json.loads(raw)
    except Exception:
        data = {}
    said = {c.get("n"): c for c in (data.get("changes") or []) if isinstance(c, dict)}
    changes = [{"label": lb,
                "title": str((said.get(i) or {}).get("title") or "").strip()[:200] or None,
                "missing": str((said.get(i) or {}).get("missing") or "").strip()[:160] or None}
               for i, lb in enumerate(labels, 1)]
    raw_tests = data.get("tests") or ([data["examples"]] if data.get("examples") else [])
    tests, seen, ran, why = [], set(), False, "no test cases were suggested"
    for t in raw_tests[:8]:
        stmts = t.get("statements") if isinstance(t, dict) else t
        stmts = [x for x in (stmts or []) if isinstance(x, str) and x.strip()][:8]
        try:
            for x in stmts:
                ast.parse(x)
        except SyntaxError:
            continue
        if not stmts:
            continue
        v = check(problem, stmts, labels)
        if not v["ok"]:
            why = v["reason"]
            continue
        ran = True
        catches = [lb for lb in labels if v["caught"].get(lb)]
        if catches and v["doctest"] not in seen:
            seen.add(v["doctest"])
            tests.append({"doctest": v["doctest"], "catches": catches,
                          "statements": stmts, "shown": v["shown"]})
    # They go into the docstring as ONE run, so they are offered only if they
    # show the same answers run together as run apart (a class counting its
    # objects would not).
    if tests:
        together = run_examples(_programs(problem, [])[0], [x for t in tests for x in t["statements"]])
        if together != [x for t in tests for x in t["shown"]]:
            tests, ran, why = [], False, "the test cases change each other's answers when run together"
    tests = [{"doctest": t["doctest"], "catches": t["catches"]} for t in tests]
    caught = {lb: any(lb in t["catches"] for t in tests) for lb in labels}
    examples = ({"ok": True, "doctest": "\n\n".join(t["doctest"] for t in tests), "caught": caught}
                if ran else {"ok": False, "reason": why})
    out = {"changes": changes, "tests": tests, "examples": examples}
    store[_VERSION + key] = out
    _save(store)
    return out
