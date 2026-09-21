"""test_reopen_step.py - going back to a step that was already accepted.

Reported from testing: a student passed step 1 of the coding stage, then found
a bug in that step, and there was no way back to it. The only route to changing
an earlier answer was Start over, which retires the whole problem - plan, chat
and every other step with it.

The session store is the whole of the logic (the route around it proves
ownership and the design gate, and grades nothing), so this exercises it
directly against real SQLite in a temp file.
"""
import json
import types

import pytest

from main import sessions


def _chunks(n=3):
    return [types.SimpleNamespace(step_id=f"Part {i + 1}", prompt=f"step {i + 1}",
                                  expected_type="code", reference=f"line{i} = {i}")
            for i in range(n)]


@pytest.fixture
def live(tmp_path):
    """A session with steps 1 and 2 accepted and step 3 still to go."""
    db = str(tmp_path / "sessions.sqlite3")
    problem = {"slug": "frequency", "title": "Letter frequency",
               "description": "Count the letters.", "solution": "def f(x):\n    return x"}
    pub = sessions.create_session(
        problem, {"header": "def f(x):", "chunks": _chunks()}, "hash-1",
        student_id="student-1", db_path=db)
    sid = pub["session_id"]
    for step, code in ((0, "counts = {}"), (1, "for ch in txt:\n    pass")):
        sub = f"sub-{step}"
        sessions.begin_submission(sid, sub, db)
        s = sessions.load_session(sid, db)
        sessions.commit_outcome(sid, sub, s["revision"], {"verdict": "correct"},
                                accept_code=code, db_path=db)
    return types.SimpleNamespace(db=db, sid=sid)


def test_it_hands_back_the_step_and_drops_what_rested_on_it(live):
    s = sessions.load_session(live.sid, live.db)
    assert s["index"] == 2 and len(s["accepted"]) == 2

    state = sessions.reopen_step(live.sid, 0, db_path=live.db)

    assert state["index"] == 0
    assert state["attempts"] == 0
    assert state["completed"] is False
    # THE LATER ANSWER COMES BACK, so the page can keep it as a draft: the
    # student reworking step 1 must not have to retype the step 2 they wrote.
    assert [d["index"] for d in state["dropped"]] == [0, 1]
    assert state["dropped"][1]["code"] == "for ch in txt:\n    pass"

    after = sessions.load_session(live.sid, live.db)
    assert after["index"] == 0
    # Dropped, because the accepted prefix is what every later step was graded
    # against - an acceptance resting on code that has since changed is not
    # evidence any more.
    assert after["accepted"] == []
    assert after["revision"] == s["revision"] + 1


def test_it_keeps_the_steps_before_the_one_reopened(live):
    sessions.reopen_step(live.sid, 1, db_path=live.db)
    after = sessions.load_session(live.sid, live.db)
    assert after["index"] == 1
    assert [a["code"] for a in after["accepted"]] == ["counts = {}"]


def test_the_submissions_are_untouched(live):
    """Nothing is deleted anywhere in this system: the instructor's transcript
    still holds every version, and main/grades.tally reads a step as solved
    when ANY submission for it was correct - so reworking never withdraws
    credit already earned."""
    import sqlite3
    sessions.reopen_step(live.sid, 0, db_path=live.db)
    conn = sqlite3.connect(live.db)
    rows = conn.execute("SELECT submission_id, result_json FROM submissions"
                        " WHERE session_id=?", (live.sid,)).fetchall()
    conn.close()
    assert len(rows) == 2, rows
    assert all(json.loads(r[1])["verdict"] == "correct" for r in rows)


def test_a_step_you_have_not_reached_is_not_reopenable(live):
    for bad in (2, 3, -1):
        with pytest.raises(sessions.SessionError) as e:
            sessions.reopen_step(live.sid, bad, db_path=live.db)
        assert e.value.reason_code == "step_not_reopenable"
    assert sessions.load_session(live.sid, live.db)["index"] == 2, "a refusal moved it"


def test_a_finished_problem_is_start_overs_business(live):
    """Un-completing a session would have to unpick the solved flag and the
    reflection stage with it, so a completed one is refused by name."""
    sub = "sub-last"
    sessions.begin_submission(live.sid, sub, live.db)
    s = sessions.load_session(live.sid, live.db)
    sessions.commit_outcome(live.sid, sub, s["revision"], {"verdict": "correct"},
                            accept_code="return counts", db_path=live.db)
    assert sessions.session_snapshot(live.sid, live.db)["state"] == "completed"
    with pytest.raises(sessions.SessionError) as e:
        sessions.reopen_step(live.sid, 0, db_path=live.db)
    assert e.value.reason_code == "session_inactive"


def test_an_unknown_session_is_not_created_by_asking(live):
    with pytest.raises(sessions.SessionError) as e:
        sessions.reopen_step("no-such-session", 0, db_path=live.db)
    assert e.value.reason_code == "session_not_found"
