"""test_reopen_resubmit.py - going back to a step and sending the same answer
again accepts it again, for real.

THE REPORT (1 Oct, Ashwin, live). Step 1 accepted at 04:19:16; "Change an
earlier step: Step 1" at 04:19:27 put the session back on step 1; the same code
sent again at 04:19:29 came back "correct" - and every step 2 after that was
"Your page is out of date - reload to continue", reload or not.

Why: the submission id is the session, the step and the code (student.js), so
the resend matched the stored result from BEFORE the reopen, and a stored
result is handed back without being committed again. The page moved to step 2;
the session never left step 1.

REAL: /grade_chunk, /reopen_step, the session store, grading (inline oracle in
the real cache file). FAKED: Supabase (in memory); the model is switched off.
"""
import json
import types

import pytest

from frontend import api_server
from test_auth_routes import client, register
from test_plan_gate_outage import Archive
from test_restart_reroute import ORACLE, PROBLEM
from test_student_open_never_generates import ROADMAP


@pytest.fixture
def site(monkeypatch, tmp_path):
    from main import grading, identity, ollama_client, sessions
    from main.identity import content_hash
    monkeypatch.setenv("MICROTUTOR_SESSION_DB", str(tmp_path / "s.sqlite3"))
    monkeypatch.setenv("MICROTUTOR_TRACE_FILE", str(tmp_path / "t.jsonl"))
    cache = tmp_path / "o.json"
    cache.write_text(json.dumps({content_hash(PROBLEM): {
        "strong": True, "status": "strong", "kill_rate": 1.0, "kill_rate_direct": 1.0,
        "features": ["calls"], "final_tests": ORACLE}}))
    monkeypatch.setenv("MICROTUTOR_ORACLE_CACHE", str(cache))
    monkeypatch.setattr(identity, "_RESOLVED_PATH", str(tmp_path / "r.json"))
    monkeypatch.setattr(ollama_client, "_openai_chat",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no model")))
    grading._VERDICT_MEMO.clear()
    c = client()
    sb = Archive()
    api_server.set_supabase(sb)
    register(c, "stu@psu.edu")
    student = sb.students[0]["id"]
    sb.tables["mt_designs"].append({"student_id": student, "slug": PROBLEM["slug"],
                                    "approved": True, "round": 1,
                                    "created_at": "2026-10-01T00:00:00+00:00"})
    decomp = {"header": ROADMAP["header"],
              "chunks": [types.SimpleNamespace(**c_) for c_ in ROADMAP["chunks"]]}
    sid = sessions.create_session(dict(PROBLEM), decomp, content_hash(PROBLEM),
                                  student_id=student)["session_id"]

    def submit(code, at):
        return c.post("/grade_chunk", json={"session_id": sid, "student_code": code,
                                            "submission_id": f"{sid}:{at}:{hash(code)}",
                                            "expected_index": at})
    return types.SimpleNamespace(
        c=c, sid=sid, submit=submit,
        index=lambda: sessions.session_snapshot(sid)["index"],
        reopen=lambda i: c.post("/reopen_step", json={"session_id": sid, "index": i}))


STEP_1 = ROADMAP["chunks"][0]["reference"]
STEP_2 = ROADMAP["chunks"][1]["reference"]


def test_the_same_answer_after_reopening_is_accepted_again(site):
    assert site.submit(STEP_1, 0).json()["verdict"] == "correct" and site.index() == 1
    assert site.reopen(0).status_code == 200 and site.index() == 0

    again = site.submit(STEP_1, 0)
    assert again.status_code == 200 and again.json()["verdict"] == "correct"
    assert site.index() == 1, "the page moves on, so the session must move on too"

    nxt = site.submit(STEP_2, 1)
    assert nxt.status_code == 200, nxt.text
    assert nxt.json()["verdict"] == "correct"


def test_a_lost_response_is_still_replayed_not_graded_twice(site):
    """The case the replay exists for: graded and committed, the response
    lost, the same submission sent again for the step the page still shows."""
    first = site.submit(STEP_1, 0).json()
    assert site.index() == 1
    again = site.submit(STEP_1, 0).json()
    assert again["verdict"] == first["verdict"] and again.get("idempotent_replay") is True
    assert site.index() == 1, "a replay must not advance the session a second time"
