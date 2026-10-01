"""test_relevant_example.py - when a step cannot be confirmed, the example a
student is shown is one their code actually gets wrong.

THE REPORT (1 Oct, Ashwin, get-postfix). His tokenizer broke only on negative
numbers ("5 +-3"). Replayed: on his roadmap he got "could not confirm" five
times with nothing to go on, and on the roadmap the 30 Sep fix serves the
example chosen was `setExpr('2 - 1')` - unrelated. The teacher's tokenizer
keeps a `prev` his never had, so EVERY input looked like one his state could
not explain, and the shortest won.

Now the example is chosen from the tests his step, run with the rest of the
solution, actually failed; when none of those is short enough to trace by hand,
the failing case itself is shown, hedged exactly as tier 3's evidence is.

Miniature of the same shape. REAL: grade_submission, the diagnosis's choice of
input. FAKED: the model tiers (nothing found) and the sentence the diagnosis
model would write around the chosen input.
"""
import types

import pytest

SOLUTION = ("def tokens_of(txt):\n"
            "    tokens = []\n"
            "    ops = '+-*/'\n"
            "    prev = None\n"
            "    for part in txt.split():\n"
            "        if part in ops and not (part == '-' and prev in (None, 'op')):\n"
            "            tokens.append(part)\n"
            "            prev = 'op'\n"
            "        elif part == '-':\n"
            "            prev = 'neg'\n"
            "        else:\n"
            "            tokens.append(('-' if prev == 'neg' else '') + part)\n"
            "            prev = 'num'\n"
            "    if tokens and tokens[-1] in ops:\n"
            "        return None\n"
            "    return ' '.join(tokens)\n")
PROBLEM = {"slug": "tokens-of", "title": "Tokens", "solution": SOLUTION,
           "description": "Split a spaced expression into tokens; a minus at the start "
                          "or after an operator belongs to the number after it. A "
                          "trailing operator gives None."}
_ns = {}
exec(SOLUTION, _ns)
INPUTS = ["1", "1 + 2", "3 * 4 - 5", "8 /", "2 * 3 + 4 - 1 * 6", "- 7", "2 * - 3"]
ORACLE = [{"input": [i], "expected": _ns["tokens_of"](i)} for i in INPUTS]
# Like get-postfix: the teacher's set-up holds a constant the later steps read
# (`ops`, there postfixStack/precedence) that the student never wrote - so their
# state never "explains" it, on any input. The grader supplies it to run them.
STEPS = [
    {"step_id": "Part 1", "expected_type": "code", "prompt": "p1",
     "reference": "tokens = []\nops = '+-*/'"},
    {"step_id": "Part 2", "expected_type": "code", "prompt": "p2",
     "reference": "\n".join(ln[4:] for ln in SOLUTION.splitlines()[3:13])},
    {"step_id": "Part 3", "expected_type": "code", "prompt": "p3",
     "reference": "if tokens and tokens[-1] in ops:\n    return None\nreturn ' '.join(tokens)"}]
NO_SIGNS = ("for part in txt.split():\n"
            "    tokens.append(part)")


@pytest.fixture
def step2(monkeypatch, tmp_path):
    monkeypatch.setenv("MICROTUTOR_SESSION_DB", str(tmp_path / "s.sqlite3"))
    from main import diagnose, grading, identity, ollama_client, sessions
    monkeypatch.setattr(identity, "_RESOLVED_PATH", str(tmp_path / "r.json"))
    monkeypatch.setattr(ollama_client, "_openai_chat",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no model")))
    monkeypatch.setattr(grading, "_request_adaptation", lambda *a, **k: ("", []))
    monkeypatch.setattr(grading, "_request_completion", lambda *a, **k: "")
    chosen = []
    monkeypatch.setattr(diagnose, "question",
                        lambda problem, chunk, code, example: chosen.append(example["input"]) or "Q?")

    def run(code, before="tokens = []"):
        decomp = {"header": "def tokens_of(txt):",
                  "chunks": [types.SimpleNamespace(**c) for c in STEPS]}
        sid = sessions.create_session(dict(PROBLEM), decomp, "h-ex", student_id="stu")["session_id"]
        s = sessions.load_session(sid)
        sessions.commit_outcome(sid, "p0", s["revision"], {"verdict": "correct",
                                "tier": "execution-reference"}, accept_code=before,
                                provenance="student")
        grading._VERDICT_MEMO.clear()
        chosen.clear()
        return grading.grade_submission(sessions.load_session(sid), code,
                                        oracle_loader=lambda p: list(ORACLE))
    run.chosen = chosen
    return run


def test_the_example_is_an_input_their_code_gets_wrong(step2):
    r = step2(NO_SIGNS)
    assert r.verdict == "indeterminate" and r.consume_attempt is False, r
    shown = step2.chosen or [c for c in r.failing_cases]
    assert shown, "something concrete must be shown"
    text = repr(shown)
    assert "- 7" in text or "* - 3" in text or "8 /" in text, \
        f"the example must be one their code fails, not {text}"
    assert "'1'" not in repr(step2.chosen), "the shortest input is not the bug"


def test_a_correct_step_is_still_accepted(step2):
    r = step2(STEPS[1]["reference"], before=STEPS[0]["reference"])
    assert r.verdict == "correct", (r.tier, r.student_reason)
