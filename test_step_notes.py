"""test_step_notes.py - every step shows what it starts with, what it must
leave for the next one, and where it sits in a loop.

THE REPORTS (30 Sep). "These kinds of problems seem a little difficult to know
what the problem wants" (a student to Dr. Saha). Two students answered "prepare
everything needed" with the set-up alone, because nothing said step 1 had to
START the loop step 2 carries on inside.

So a step now carries two short notes, written once per roadmap by a model and
gated like its question (no method, no name only the solution uses), plus a
loop note derived from the steps' depths - free, and always right.

REAL: the notes gate, the save/load of a roadmap, the session, the route a
student opens a problem through. FAKED: the model (canned notes) and Supabase.
"""
import json
import types

import pytest

from test_restart_reroute import env, _open  # noqa: F401
from test_reword_pools import LOOPED

GOOD = [{"starts_with": "The text.", "leaves": "Each letter of the text, one at a time."},
        {"starts_with": "One letter of the text.", "leaves": "The tally, one higher for it."},
        {"starts_with": "The tally of every letter.", "leaves": "How many times each letter appears."}]
LEAKY = [dict(GOOD[0], leaves="The letter_counts so far.")] + GOOD[1:]


def _roadmap():
    """LOOPED, with its tally named like code (`letter_counts`) - the kind of
    name a note must never hand a student."""
    from main.run_phase1 import _deserialize
    return _deserialize({**LOOPED, "chunks": [
        {**c, "reference": c["reference"].replace("counts", "letter_counts"),
         "prompt": c["prompt"] or "Start going through each character, and keep the "
                                  "tally for the next step."}
        for c in LOOPED["chunks"]]})


@pytest.fixture
def notes(monkeypatch):
    from main import step_notes
    replies, asked = [], []

    def chat(model, system, messages, **_k):
        asked.append(messages[0]["content"])
        r = replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return json.dumps({"notes": r})
    monkeypatch.setattr(step_notes, "chat", chat)
    return types.SimpleNamespace(replies=replies, asked=asked, mod=step_notes)


PROBLEM = {"slug": "frequency", "title": "Letter frequency",
           "description": "Count how many times each letter appears.",
           "solution": "def frequency(txt):\n    letter_counts = {}\n    for ch in txt:\n"
                       "        if ch.isalpha():\n            letter_counts[ch] = letter_counts.get(ch, 0) + 1\n"
                       "    return letter_counts\n"}


def test_good_notes_reach_every_step_and_survive_saving(notes):
    from main.run_phase1 import _deserialize, _serialize
    notes.replies.append(GOOD)
    d = notes.mod.write_notes(PROBLEM, _roadmap())
    again = _deserialize(_serialize(d))
    assert [(c.starts_with, c.leaves) for c in again["chunks"]] == \
        [(n["starts_with"], n["leaves"]) for n in GOOD]


def test_a_note_naming_the_solutions_variable_is_retried_with_the_reason(notes):
    notes.replies += [LEAKY, GOOD]
    d = notes.mod.write_notes(PROBLEM, _roadmap())
    assert d["chunks"][0].leaves == GOOD[0]["leaves"]
    assert "letter_counts" in notes.asked[1] and "REJECTED" in notes.asked[1]


def test_notes_that_never_pass_leave_the_roadmap_as_it_was(notes):
    notes.replies += [LEAKY, LEAKY, LEAKY]
    before = _roadmap()
    d = notes.mod.write_notes(PROBLEM, before)
    assert d is before and all(c.starts_with is None for c in d["chunks"])


def test_a_provider_outage_keeps_the_roadmap_without_notes(notes):
    notes.replies.append(RuntimeError("OpenAI 503"))
    before = _roadmap()
    assert notes.mod.write_notes(PROBLEM, before) is before
    assert len(notes.asked) == 1, "an outage is not retried as if the notes were wrong"


def test_the_loop_note_says_where_each_step_sits():
    from main.sessions import public_chunks
    shown = public_chunks(LOOPED["chunks"])
    assert "Start going through the items in this step" in shown[0]["loop_note"]
    assert "runs inside the loop" in shown[1]["loop_note"]
    assert shown[2]["loop_note"] == ""
    assert all("reference" not in c for c in shown), "no code may reach the browser"


def test_a_student_opening_the_problem_sees_the_notes(env, monkeypatch, notes):
    """Through /decompose_chunks, the route the student page calls."""
    from frontend import api_server
    notes.replies.append(GOOD)
    noted = notes.mod.write_notes(PROBLEM, _roadmap())
    monkeypatch.setattr(api_server, "get_chunk_decomposition", lambda p: noted)
    body, _ = _open(env)
    first = body["chunks"][0]
    assert (first["starts_with"], first["leaves"]) == (GOOD[0]["starts_with"], GOOD[0]["leaves"])
    assert "Start going through the items" in first["loop_note"]
    assert "counts" not in json.dumps(body["chunks"])


def test_plain_english_that_happens_to_be_a_variable_is_fine():
    """1 Oct, the first live draft: every get-postfix and employee-update note
    was refused for "tokens", "previous year's records", "updated records" and
    "initialized structures" - ordinary words the teacher also used as names."""
    from main.gates import check_notes
    from main.schemas import StepItem
    code = "previous = d[year - 1]\nupdated = {}\ntokens = []\npostfixStack = []"
    chunks = [StepItem(question_id="t", step_id="Part 1", prompt="p", reference=code,
                       starts_with="The previous year's records and initialized structures.",
                       leaves="The updated records, and the tokens of the expression.")]
    assert check_notes(chunks, {"description": "Update the records."})["status"] == "pass"
    chunks[0] = chunks[0].model_copy(update={"leaves": "The postfixStack, filled."})
    assert check_notes(chunks, {"description": "Update the records."})["status"] == "fail"
