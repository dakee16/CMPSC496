"""test_splitter.py - roadmaps cut from the teacher's own code keep every step
small, including for a problem that is one big loop, and uploads use them when
the model's roadmap has a giant step or cannot be built at all.

THE MEASUREMENT (server, 28 Sep). Every model-built roadmap for
calculateExpressions (8) and calculator-get-postfix (7) had a step of 35-48
lines: the teacher's solution is one loop, and the model never cut inside it.
Cutting the teacher's code at statement boundaries, through the same serve gate,
gave calculateExpressions [7,15,16,2]-style roadmaps with no step over 20.

Here the limit is lowered to 6 lines so a small inline problem has the same
shape: a loop that is most of the code. The fakes are the network edges only -
the oracle (inline, certified) and the model (stubbed per test). The cutting,
the serve gate, the upload pipeline and grading are the real code.
"""
import json
import re
import sys
import types

import pytest

SOLUTION = ("def score(words):\n"
            "    total = 0\n"
            "    seen = []\n"
            "    for w in words:\n"
            "        w = w.strip().lower()\n"
            "        if not w:\n"
            "            continue\n"
            "        if w in seen:\n"
            "            total -= 1\n"
            "            continue\n"
            "        seen.append(w)\n"
            "        if w.isdigit():\n"
            "            total += int(w)\n"
            "        elif len(w) > 3:\n"
            "            total += 2\n"
            "        else:\n"
            "            total += 1\n"
            "    if total < 0:\n"
            "        total = 0\n"
            "    return total\n")
PROBLEM = {"slug": "score-split", "title": "Score", "solution": SOLUTION,
           "description": "Score a list of words: numbers count their value, long "
                          "words 2, short words 1, repeats cost 1, never below 0."}
INPUTS = [[["a", "bb", "ccc"]], [["hello", "a"]], [["7", "hello"]], [[" ", "a"]],
          [["a", "a", "a"]], [["x", "x", "x", "x"]], [[]], [["12", "12"]],
          [["Word", "word"]], [["abcd"]]]
_ns = {}
exec(SOLUTION, _ns)
ORACLE = [{"input": i, "expected": _ns["score"](*i)} for i in INPUTS]
LOPSIDED = {"header": "def score(words):", "chunks": [
    {"step_id": "Part 1", "prompt": "Work out the score, and keep it for the next step.",
     "expected_type": "code",
     "reference": "\n".join(ln[4:] for ln in SOLUTION.splitlines()[1:-1])},
    {"step_id": "Part 2", "prompt": "Return the score.", "expected_type": "code",
     "reference": "return total"}]}
LIMIT = 6


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("MICROTUTOR_ORACLE_CACHE", str(tmp_path / "o.json"))
    monkeypatch.setenv("MICROTUTOR_SESSION_DB", str(tmp_path / "s.sqlite3"))
    from main import grading, identity, ollama_client, run_phase1, splitter
    from tests import sandbox
    monkeypatch.setattr(identity, "_RESOLVED_PATH", str(tmp_path / "r.json"))

    def no_network(*_a, **_k):
        raise AssertionError("a test reached the real model provider")
    monkeypatch.setattr(ollama_client, "_openai_chat", no_network)
    monkeypatch.setattr(ollama_client, "_ollama_chat", no_network)
    fakes = {"get_oracle_tests": lambda p, **k: list(ORACLE),
             "is_oracle_certified": lambda p: True}
    for name, fake in fakes.items():
        real = getattr(sandbox, name)
        for mod in list(sys.modules.values()):
            if getattr(mod, name, None) is real:
                monkeypatch.setattr(mod, name, fake)
    monkeypatch.setattr(run_phase1, "_CHUNK_POOL_PATH", str(tmp_path / "pool.json"))
    monkeypatch.setattr(splitter, "MAX_STEP_LINES", LIMIT)
    monkeypatch.setattr(grading, "_request_adaptation", lambda *a, **k: ("", []))
    monkeypatch.setattr(grading, "_request_completion", lambda *a, **k: "")
    log = types.SimpleNamespace(worded=0, built=0, best=0)

    def wording(model, system, messages, **_k):
        log.worded += 1
        n = len(re.findall(r"STEP \d+ CODE", messages[0]["content"]))
        # Every non-final step also says it goes through each word: a step that
        # starts the loop must say so or the prompt gate rejects it (30 Sep).
        return json.dumps({"prompts": [f"Write code that handles part {i + 1} of the "
                                       f"scoring as it goes through each word, and "
                                       f"keep the result for the next step."
                                       for i in range(n - 1)]
                           + ["Write code that returns the final score."]})
    monkeypatch.setattr(splitter, "chat", wording)

    def builder(result):
        def build(problem, *_a, **_k):
            log.built += 1
            if isinstance(result, Exception):
                raise result
            return run_phase1._deserialize(result)
        monkeypatch.setattr(run_phase1, "decompose_into_chunks", build)

    def best(*_a, **_k):
        log.best += 1
        raise run_phase1.DecompositionUnavailableError("nothing")
    monkeypatch.setattr(run_phase1, "decompose_into_chunks_best", best)

    def saved():
        return run_phase1._load_pool().get(identity.content_hash(PROBLEM), [])
    return types.SimpleNamespace(splitter=splitter, run_phase1=run_phase1, grading=grading,
                                 log=log, builder=builder, saved=saved)


def _biggest(entry):
    return max(sum(1 for ln in c["reference"].splitlines()
                   if ln.strip() and not ln.strip().startswith("#")) for c in entry["chunks"])


def test_a_loop_heavy_problem_is_cut_inside_the_loop_within_the_limit(env):
    plans = env.splitter.plan(env.run_phase1._reading_saved_tests(dict(PROBLEM)))
    assert plans, "no split found"
    for d in plans:
        assert env.splitter.biggest_step(d) <= LIMIT, \
            [env.splitter.step_lines(c.reference) for c in d["chunks"]]
    assert any(c.reference.startswith("    ") for d in plans for c in d["chunks"]), \
        "a loop that is most of the code can only be split inside the loop"


def test_the_teachers_own_steps_pass_grading_typed_flat(env):
    """What a student gets must be answerable: the teacher's own code, typed
    with no indentation, is accepted at every step - through real grading."""
    import textwrap
    from main import sessions
    for d in env.splitter.plan(env.run_phase1._reading_saved_tests(dict(PROBLEM))):
        sid = sessions.create_session(dict(PROBLEM), d, "h-split", student_id="s")["session_id"]
        for c in d["chunks"]:
            s = sessions.load_session(sid)
            typed = textwrap.dedent(c.reference)
            env.grading._VERDICT_MEMO.clear()
            r = env.grading.grade_submission(s, typed, oracle_loader=lambda p: list(ORACLE))
            assert r.verdict == "correct", (c.step_id, r.student_reason)
            sessions.commit_outcome(sid, f"x-{c.step_id}", s["revision"], r.model_dump(),
                                    accept_code=env.grading.align_submission(s, typed),
                                    provenance="student")


def test_the_model_only_words_the_steps_once_each(env):
    built = env.splitter.build(env.run_phase1._reading_saved_tests(dict(PROBLEM)), want=3)
    assert built and env.log.worded == len(built)
    assert all(c.prompt.startswith("Write code that") for d in built for c in d["chunks"])


def test_an_upload_with_a_giant_step_gets_split_roadmaps_instead(env):
    from main import publish
    env.builder(LOPSIDED)
    assert publish.prepare_problem(dict(PROBLEM))["ready"] is True
    saved = env.saved()
    assert saved and all(_biggest(e) <= LIMIT for e in saved), [_biggest(e) for e in saved]
    assert env.log.built == 1, "the model's oversized roadmap stops the model builds"


def test_an_upload_the_model_cannot_split_is_split_from_the_teachers_code(env):
    from main import publish
    env.builder(RuntimeError("every try failed a gate"))
    assert publish.prepare_problem(dict(PROBLEM))["ready"] is True
    assert env.saved() and env.log.built == 1 and env.log.best == 0


def test_a_normal_upload_never_uses_the_splitter(env, monkeypatch):
    from main import publish
    small = {"header": "def score(words):", "chunks": [
        {**LOPSIDED["chunks"][0], "reference": LOPSIDED["chunks"][0]["reference"]},
        LOPSIDED["chunks"][1]]}
    monkeypatch.setattr(env.splitter, "MAX_STEP_LINES", 50)
    env.builder(small)
    called = []
    monkeypatch.setattr(env.splitter, "build", lambda *a, **k: called.append(1) or [])
    assert publish.prepare_problem(dict(PROBLEM))["ready"] is True
    assert called == [] and env.log.built == 5


def test_resplit_replaces_only_oversized_roadmaps(env, monkeypatch):
    from main import identity, resplit_pools
    ok = {"header": "def score(words):", "chunks": [{"step_id": "Part 1", "prompt": "p",
          "expected_type": "code", "reference": "x = 1"}]}
    key = identity.content_hash(PROBLEM)
    env.run_phase1._add_to_pool(key, [ok, LOPSIDED])
    monkeypatch.setattr(resplit_pools, "targets", lambda: [(dict(PROBLEM), [ok], [LOPSIDED])])

    assert resplit_pools.main([]) == 0
    assert env.saved() == [ok, LOPSIDED] and env.log.worded == 0, "a dry run changes nothing"

    assert resplit_pools.main(["--apply"]) == 0
    saved = env.saved()
    assert saved[0] == ok and LOPSIDED not in saved and len(saved) >= 2
    assert all(_biggest(e) <= LIMIT for e in saved[1:])


def test_a_step_inside_the_loop_is_marked_for_the_wording(env, monkeypatch):
    """Measured: worded without the mark, none of the in-loop steps said
    "for each" - the student could not tell the step runs once per item."""
    seen = []
    monkeypatch.setattr(env.splitter, "chat", lambda m, s, msgs, **k: seen.append(msgs[0]["content"])
                        or json.dumps({"prompts": []}))
    plans = env.splitter.plan(env.run_phase1._reading_saved_tests(dict(PROBLEM)))
    d = next(d for d in plans if any(c.reference.startswith("    ") for c in d["chunks"]))
    env.splitter.write_prompts(dict(PROBLEM), d)
    for i, c in enumerate(d["chunks"]):
        marked = f"STEP {i + 1} CODE (RUNS ONCE PER ITEM" in seen[0]
        assert marked == c.reference.startswith(" "), (i, c.reference[:40])


def test_rejected_wording_is_retried_with_the_gates_reasons(env, monkeypatch):
    replies = iter([["Write code that initializes everything, and keep it for the next step.",
                     "Return the score."]] * 2
                   + [["Write code that gets everything ready, and keep it for the next step.",
                       "Write code that returns the final score."]])
    asked = []

    def chat(m, s, msgs, **k):
        asked.append(msgs[0]["content"])
        return json.dumps({"prompts": next(replies)})
    monkeypatch.setattr(env.splitter, "chat", chat)
    from main.schemas import StepItem
    d = {"header": "def score(words):", "chunks": [
        StepItem(question_id="t", step_id="Part 1", prompt="", expected_type="code",
                 reference="total = 0"),
        StepItem(question_id="t", step_id="Part 2", prompt="", expected_type="code",
                 reference="return total")]}
    out = env.splitter.write_prompts(dict(PROBLEM), d)
    assert len(asked) == 3 and "REJECTED" in asked[1] and "initiali" in asked[1]
    assert out["chunks"][0].prompt.startswith("Write code that gets everything ready")
