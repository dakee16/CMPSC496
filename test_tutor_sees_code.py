"""test_tutor_sees_code.py - once coding is open, the tutor sees the student's
OWN code: their accepted steps and what is in their code box. Never ours.

THE REPORT (30 Sep, a student to Dr. Saha): "The tutor is no help either since
I don't think the tutor can see the entire code file." Right - it saw the chat
and the step's question, nothing else, so "why doesn't this work?" could only
be answered by guessing.

REAL: /tutor_chat, the design gate it reads, the session store the accepted
steps come from, the tutor's prompt assembly. FAKED: Supabase (in memory) and
the model (it records the system prompt it was sent).
"""
import json
import types

import pytest

from frontend import api_server
from test_auth_routes import client, register
from test_plan_gate_outage import SLUG, Archive

REFERENCE = "total = 0\nfor n in nums:\n    total += n"
ACCEPTED = "running = 0\nfor value in nums:\n    running += value"
DRAFT = "return runing"


@pytest.fixture
def site(monkeypatch, tmp_path):
    from main import sessions, tutor
    monkeypatch.setenv("MICROTUTOR_SESSION_DB", str(tmp_path / "s.sqlite3"))
    monkeypatch.setenv("MICROTUTOR_TRACE_FILE", str(tmp_path / "t.jsonl"))
    seen = []

    def chat(model, system, messages, **_k):
        seen.append(system)
        return json.dumps({"reply": "Which line hands back the total?", "ready": False})
    monkeypatch.setattr(tutor, "chat", chat)
    c = client()
    sb = Archive()
    api_server.set_supabase(sb)
    register(c, "stu@psu.edu")
    sid = sb.students[0]["id"]
    problem = {"slug": SLUG, "title": "Sum a list", "description": "Add up the numbers.",
               "solution": "def total(nums):\n" + "\n".join("    " + ln for ln in
                                                          (REFERENCE + "\nreturn total").splitlines())}
    sb.problems.append({k: problem[k] for k in ("slug", "title", "description")})
    decomp = {"header": "def total(nums):", "chunks": [
        types.SimpleNamespace(step_id="Part 1", prompt="Add them up, keep it.", expected_type="code",
                              reference=REFERENCE),
        types.SimpleNamespace(step_id="Part 2", prompt="Hand back the total.", expected_type="code",
                              reference="return total")]}
    sess = sessions.create_session(problem, decomp, "h-tutor", student_id=sid)["session_id"]
    s = sessions.load_session(sess)
    sessions.commit_outcome(sess, "x0", s["revision"], {"verdict": "correct", "tier": "execution-bridged"},
                            accept_code=ACCEPTED, provenance="student")

    def approve():
        sb.tables["mt_designs"].append({"student_id": sid, "slug": SLUG, "approved": True,
                                        "round": 1, "created_at": "2026-09-30T00:00:00+00:00"})

    def ask(code=DRAFT):
        r = c.post("/tutor_chat", json={"slug": SLUG, "code": code, "messages": [
            {"role": "user", "content": "Why does my code not work?"}]})
        assert r.status_code == 200, r.text
        return seen[-1]
    return types.SimpleNamespace(ask=ask, approve=approve)


def test_once_coding_is_open_the_tutor_sees_their_steps_and_their_box(site):
    site.approve()
    system = site.ask()
    assert ACCEPTED in system, "their accepted step"
    assert DRAFT in system, "what is in their code box"
    assert "never rewrite or complete" in system


def test_the_tutor_never_sees_the_teachers_step(site):
    site.approve()
    system = site.ask()
    assert "total += n" not in system and "for n in nums" not in system


def test_before_the_plan_is_approved_no_code_is_sent(site):
    system = site.ask()
    assert ACCEPTED not in system and DRAFT not in system


def test_an_empty_box_still_shows_their_accepted_steps(site):
    site.approve()
    system = site.ask(code="")
    assert ACCEPTED in system
