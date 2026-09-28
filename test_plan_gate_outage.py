"""test_plan_gate_outage.py - when the plan reviewer is down, a student is let
through to code, with the plan recorded as NOT reviewed, instead of the whole
class being locked out.

THE LOCKOUT. The editor stays locked until the AI reviewer approves a plan. When
the review call failed - an OpenAI outage, credits run out, the monthly limit
reached - both review routes answered 503 "The design reviewer is unavailable
right now", and no student could write a line of code until OpenAI came back.
Grading already degrades ("OpenAI is down - your attempt was NOT used"); the
plan gate did not. Our failure is never the student's.

REAL: both routes, the reviewer up to its model call, the archive writes, and
_design_approved - the check the steps, grading and tutor routes all read.
FAKED: Supabase (tables kept in memory) and the model provider.
"""
import collections
import json
import types

import pytest

from frontend import api_server
from test_auth_routes import FakeSupabase, FakeTable, client, register

SLUG = "sum-list"
PLAN = {"nodes": [{"id": "s", "kind": "start", "label": "take the list"},
                  {"id": "l", "kind": "loop", "label": "each number"},
                  {"id": "r", "kind": "return", "label": "the running total"}],
        "edges": []}
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


class Table(FakeTable):
    def __getattr__(self, _name):        # gt, order, in_ ...: filters not needed here
        return lambda *a, **k: self


class Archive(FakeSupabase):
    """FakeSupabase, except every table keeps what is written to it."""

    def __init__(self):
        super().__init__()
        self.tables = collections.defaultdict(list)

    def table(self, name):
        known = {"students": self.students, "solved": self.solved,
                 "problems": self.problems}
        return Table(known[name] if name in known else self.tables[name])


@pytest.fixture
def student(monkeypatch, tmp_path):
    from main import graphs, ollama_client
    monkeypatch.setenv("MICROTUTOR_TRACE_FILE", str(tmp_path / "t.jsonl"))
    calls, answer = [], []

    def provider(*_a, **_k):
        calls.append(1)
        if answer:
            return answer[0]
        raise RuntimeError("OpenAI 429: You exceeded your current quota")
    monkeypatch.setattr(ollama_client, "_openai_chat", provider)
    monkeypatch.setattr(ollama_client, "_ollama_chat", provider)
    read_drawing = []
    monkeypatch.setattr(graphs, "graph_from_design",
                        lambda *a, **k: read_drawing.append(1) or {})

    c = client()
    sb = Archive()
    api_server.set_supabase(sb)
    register(c, "stu@psu.edu")
    sid = sb.students[0]["id"]
    sb.problems.append({"slug": SLUG, "title": "Sum a list",
                        "description": "Add up the numbers in a list."})
    sb.tables["mt_messages"].append({
        "student_id": sid, "slug": SLUG, "phase": "tutor", "role": "user",
        "content": "I'll loop over the list and keep a running total.",
        "created_at": "2026-09-28T00:00:00+00:00"})
    return types.SimpleNamespace(c=c, sb=sb, sid=sid, calls=calls, answer=answer,
                                 read_drawing=read_drawing)


def _designs(student):
    return student.sb.tables["mt_designs"]


def test_a_plan_sent_while_the_reviewer_is_down_lets_the_student_code(student):
    from main.design_review import UNREVIEWED_REPLY
    r = student.c.post("/design_review/plan", json={"slug": SLUG, "graph": PLAN})

    assert r.status_code == 200, r.text
    out = r.json()
    assert (out["approved"], out.get("unreviewed")) == (True, True), out
    assert out["reply"] == UNREVIEWED_REPLY
    assert student.calls, "the reviewer must really have been asked first"
    assert api_server._design_approved(student.sid, SLUG) is True
    assert (_designs(student)[-1]["approved"], _designs(student)[-1]["reviewer_reply"]) \
        == (True, UNREVIEWED_REPLY)


def test_a_drawing_sent_while_the_reviewer_is_down_lets_the_student_code(student):
    r = student.c.post("/design_review", data={"slug": SLUG},
                       files={"design": ("plan.png", PNG, "image/png")})

    assert r.status_code == 200, r.text
    assert (r.json()["approved"], r.json().get("unreviewed")) == (True, True)
    assert api_server._design_approved(student.sid, SLUG) is True
    # Reading the drawing into a plan graph is a second AI call, into the
    # same outage - it is skipped rather than made to fail.
    assert student.read_drawing == []


def test_a_bad_upload_is_still_refused_not_let_through(student):
    r = student.c.post("/design_review", data={"slug": SLUG},
                       files={"design": ("plan.txt", b"my plan", "text/plain")})

    assert r.status_code == 400 and r.json()["detail"]["reason_code"] == "design_rejected"
    assert student.calls == [] and _designs(student) == []
    assert api_server._design_approved(student.sid, SLUG) is False


def test_a_working_reviewer_still_decides(student):
    student.answer.append(json.dumps({
        "reply": "What does your plan hand back for an empty list?",
        "approved": False}))
    out = student.c.post("/design_review/plan", json={"slug": SLUG, "graph": PLAN}).json()

    assert out["approved"] is False and not out.get("unreviewed")
    assert api_server._design_approved(student.sid, SLUG) is False


def test_the_teachers_report_says_the_plan_was_not_reviewed():
    from main.design_review import UNREVIEWED_REPLY
    from main.report import _designs as render
    html = render([{"round": 1, "approved": True, "reviewer_reply": UNREVIEWED_REPLY,
                    "mime": "application/x-plan-graph"}])
    assert "not reviewed" in html and ">approved<" not in html
