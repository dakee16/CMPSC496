"""test_given_code.py - a plain function may use what its file gives it: an
import or a constant at the top of the file, or a helper beside it.

THE AUDIT (6 Oct). None of these ever reached a run:
  - `import math` at the top of the file: the teacher's own solution raised
    NameError on every test, and the upload failed "no usable test cases";
  - `import math` inside the problem's block: the upload said READY, and then
    every student - the teacher's own code included - was told "This step uses
    `math`, which isn't defined" on the last step;
  - a helper inside the block, which the upload page tells teachers to do:
    the splitter cut the HELPER's body into steps, and nothing could be served.

REAL: the parser, test generation and the strength check, the splitter, and
grading every step of every roadmap exactly as a student's answer is graded.
FAKED: the model - it proposes inputs only, and is never reached by grading.
"""
import json
import types

import pytest

from main import context, mutation
from main.assignments import parse_assignment_file
from tests import sandbox

TOP_IMPORT = '''"""Geometry"""
import math

LIMIT = 1000


def hyp(a, b):
    """The hypotenuse of a right triangle with sides a and b, to 2 places, or
    None when a side is negative or over LIMIT."""
    if a < 0 or b < 0 or a > LIMIT or b > LIMIT:
        return None
    s = a * a + b * b
    return round(math.sqrt(s), 2)
'''
IMPORT_IN_BLOCK = '''"""Geometry"""
# --- problem: hyp ---
import math


def hyp(a, b):
    """The hypotenuse of a right triangle with sides a and b, to 2 places, or
    None when a side is negative."""
    if a < 0 or b < 0:
        return None
    s = a * a + b * b
    return round(math.sqrt(s), 2)
'''
HELPER_IN_BLOCK = '''"""Geometry"""
# --- problem: hyp ---
def sq(x):
    return x * x


def hyp(a, b):
    """The hypotenuse of a right triangle with sides a and b, to 2 places, or
    None when a side is negative."""
    if a < 0 or b < 0:
        return None
    s = sq(a) + sq(b)
    return round(s ** 0.5, 2)
'''
SHAPES = {"top import": TOP_IMPORT, "import in block": IMPORT_IN_BLOCK,
          "helper in block": HELPER_IN_BLOCK}


@pytest.fixture
def env(monkeypatch, tmp_path):
    from main import grading, identity, ollama_client
    monkeypatch.setenv("MICROTUTOR_ORACLE_CACHE", str(tmp_path / "o.json"))
    monkeypatch.setenv("MICROTUTOR_SESSION_DB", str(tmp_path / "s.db"))
    monkeypatch.setenv("MICROTUTOR_TRACE_FILE", str(tmp_path / "t.jsonl"))
    monkeypatch.setattr(identity, "_RESOLVED_PATH", str(tmp_path / "r.json"))
    monkeypatch.setattr(ollama_client, "_openai_chat",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no model")))
    monkeypatch.setattr(mutation, "chat", lambda *a, **k: "{}")
    monkeypatch.setattr(sandbox, "chat", lambda *a, **k: json.dumps(
        {"inputs_src": "[[3, 4], [5, 12], [1, 1], [-1, 2], [0, 0], [8, 15], [2, -7]]"}))
    grading._VERDICT_MEMO.clear()


def _problem(src):
    out = parse_assignment_file(src, "geometry.py")
    assert out["errors"] == [], out["errors"]
    return out["problems"][0]


def _certified(p):
    """The suite the real generator makes, saved as passing the strength
    check - the fake model cannot run the paid search that would get it there,
    and the strength check is not what this file is about."""
    from main.identity import content_hash
    from main.oracle_store import save_cache
    tests = sandbox.make_oracle_tests(p)
    save_cache({content_hash(p): {"strong": True, "status": "strong", "kill_rate": 1.0,
                                  "kill_rate_direct": 1.0, "features": ["calls"],
                                  "final_tests": tests}})
    return tests


def _grade_all_steps(problem, roadmap, tests, answers=None):
    """Every step of `roadmap`, answered in order, through grade_submission."""
    from main import grading, sessions
    from main.identity import content_hash
    chunks = [types.SimpleNamespace(**(c if isinstance(c, dict) else c.model_dump()))
              for c in roadmap["chunks"]]
    sid = sessions.create_session(dict(problem), {"header": roadmap["header"], "chunks": chunks},
                                  content_hash(problem), student_id="stu")["session_id"]
    verdicts = []
    for i, c in enumerate(chunks):
        code = (answers or {}).get(i, c.reference)
        s = sessions.load_session(sid)
        grading._VERDICT_MEMO.clear()
        r = grading.grade_submission(s, code, oracle_loader=lambda p: list(tests))
        verdicts.append((r.verdict, r.tier, r.student_reason))
        if r.verdict != "correct":
            break
        sessions.commit_outcome(sid, f"x{i}", s["revision"], r.model_dump(),
                                accept_code=grading.align_submission(s, code),
                                provenance="student")
    return verdicts


@pytest.mark.parametrize("shape", SHAPES)
def test_the_teachers_own_steps_are_accepted_on_every_roadmap(env, shape):
    from main import splitter
    p = _problem(SHAPES[shape])
    tests = _certified(p)
    # 7 inputs; the 2 whose answer is None are dropped by make_oracle_tests,
    # as for every problem. The other 5 ran - they were 0 before.
    assert len(tests) == 5, "the teacher's own code runs"
    roadmaps = splitter.plan(p)
    assert roadmaps, "the exercise's own body is what gets split"
    for rm in roadmaps:
        got = _grade_all_steps(p, rm, tests)
        assert all(v == "correct" for v, _, _ in got) and len(got) == len(rm["chunks"]), \
            (shape, [c.reference if hasattr(c, "reference") else c["reference"]
                     for c in rm["chunks"]], got)


@pytest.mark.parametrize("shape", SHAPES)
def test_a_wrong_answer_is_still_wrong(env, shape):
    from main import splitter
    p = _problem(SHAPES[shape])
    tests = _certified(p)
    rm = splitter.plan(p)[0]
    last = len(rm["chunks"]) - 1
    wrong = {last: rm["chunks"][last]["reference"].replace(", 2)", ", 1)")
             if isinstance(rm["chunks"][last], dict)
             else rm["chunks"][last].reference.replace(", 2)", ", 1)")}
    got = _grade_all_steps(p, rm, tests, wrong)
    assert got[-1][0] == "incorrect", got[-1]


def test_mutation_testing_sees_the_import_too(env):
    """Every mutant of a NameError-ing reference dies identically - a suite
    run that way measures nothing."""
    p = _problem(TOP_IMPORT)
    v = mutation.validate_oracle(p, sandbox.make_oracle_tests(p), max_rounds=1)
    assert v["kill_rate_direct"] >= 0.5, v


def test_a_problem_with_nothing_beside_it_runs_exactly_as_before():
    plain = {"slug": "double", "solution": "def double(n):\n    return n * 2\n",
             "entry_hint": "double", "module_preamble": '"""Week 1"""'}
    assert context.given_code(plain) == ""
    assert context.build_program(plain, "return n * 2", "def double(n):") == \
        "def double(n):\n    return n * 2"
    assert context.reference_program(plain) == plain["solution"]


def test_bare_statements_at_the_top_are_not_run_on_every_test():
    p = {"slug": "f", "entry_hint": "f", "solution": "def f(n):\n    return n\n",
         "module_preamble": '"""T"""\nimport math\nprint("hello")\nX = 3'}
    given = context.given_code(p)
    assert "import math" in given and "X = 3" in given and "print" not in given


def test_a_mistake_inside_a_helper_can_still_be_caught(env):
    """A test run of a broken copy must not get a second, unbroken copy of the
    block's helper appended - it would replace the broken one, and every
    mistake made inside a helper would look harmless."""
    p = _problem(HELPER_IN_BLOCK)
    broken = p["solution"].replace("return x * x", "return x + x")
    ref = sandbox.run_solution(context.reference_program(p), [[3, 4]], entry_name="hyp")
    mut = sandbox.run_solution(mutation._runnable(p, broken), [[3, 4]], entry_name="hyp")
    assert ref["results"] == [5.0] and mut["results"] != ref["results"], (ref, mut)
