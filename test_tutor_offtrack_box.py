"""test_tutor_offtrack_box.py - the "This approach may not get you to the right
answer" box appears only when the plan reviewer agrees.

WHY (30 Sep). The box was the tutor's own guess (tutor.py "offtrack"); the plan
reviewer - the gate that actually unlocks coding - was never asked. On
calculator-is-number it warned students off try float(txt) / except ValueError,
which IS the teacher's solution, and the reviewer approved the same plan a
minute later. Since 20 Sep: 18 clicks from 5 students; all 5 who clicked "keep
going" finished the problem. /tutor_chat already asks the reviewer before the
tutor may say "ready"; the box now goes through the same gate.

REAL: the route, the tutor, the reviewer's parsing and approval rule. FAKED:
Supabase (in memory) and the model provider.
"""
import json
import types

import pytest

from frontend import api_server
from test_auth_routes import client, register
from test_plan_gate_outage import PLAN, SLUG, Archive

SAYS = "I'd turn the text into a number and see whether that works."


@pytest.fixture
def chat(monkeypatch, tmp_path):
    from main import ollama_client
    monkeypatch.setenv("MICROTUTOR_TRACE_FILE", str(tmp_path / "t.jsonl"))
    state = types.SimpleNamespace(offtrack=True, reviewer="rejects", reviews=0)

    def provider(model, system, messages, *a, **k):
        if "PLAN_STEPS" in system:                          # the plan reviewer
            state.reviews += 1
            if state.reviewer == "down":
                raise RuntimeError("OpenAI 503")
            ok = state.reviewer == "approves"
            return json.dumps({"reply": "That plan works." if ok else
                               "What does it do with ' 7 '?", "approved": ok,
                               "trace": "walked ' 3.5 ' -> float -> True; 'x' -> "
                                        "ValueError -> False; None -> TypeError"
                                        " -> False" if ok else ""})
        if '"offtrack"' in system:                          # the tutor
            return json.dumps({"reply": "What does your plan do with ' 7 '?",
                               "offtrack": state.offtrack, "ready": False,
                               # tutor.reply drops a flag with a shorter reason
                               "offtrack_reason": "the plan never handles text "
                                                  "that has spaces around it"})
        return json.dumps({"proposed": [], "rejected": []})  # quick checks
    monkeypatch.setattr(ollama_client, "_openai_chat", provider)
    monkeypatch.setattr(ollama_client, "_ollama_chat", provider)

    c = client()
    sb = Archive()
    api_server.set_supabase(sb)
    register(c, "stu@psu.edu")
    sb.problems.append({"slug": SLUG, "title": "Is it a number",
                        "description": "Say whether the text is a number."})

    def say(plan=PLAN):
        body = {"slug": SLUG, "messages": [{"role": "user", "content": SAYS}]}
        if plan is not None:
            body["plan"] = plan
        r = c.post("/tutor_chat", json=body)
        assert r.status_code == 200, r.text
        return r.json()
    return types.SimpleNamespace(state=state, say=say, sb=sb,
                                 sid=sb.students[0]["id"])


def test_no_box_when_the_reviewer_approves_the_plan(chat):
    chat.state.reviewer = "approves"
    out = chat.say()
    assert out["offtrack"] is False and chat.state.reviews == 1, out


def test_the_box_when_the_reviewer_also_rejects(chat):
    out = chat.say()
    assert out["offtrack"] is True and chat.state.reviews == 1, out


def test_no_box_before_there_is_a_plan(chat):
    for plan in (None, {"nodes": [], "edges": []}):     # nothing drawn yet
        out = chat.say(plan=plan)
        assert out["offtrack"] is False and chat.state.reviews == 0, (plan, out)


def test_no_box_when_the_reviewer_is_unreachable(chat):
    chat.state.reviewer = "down"
    out = chat.say()
    assert out["offtrack"] is False and chat.state.reviews == 1, out


def test_the_reviewer_is_asked_only_when_the_tutor_raises_the_flag(chat):
    chat.state.offtrack = False
    out = chat.say()
    assert out["offtrack"] is False and chat.state.reviews == 0, out


def test_no_box_and_no_call_once_the_plan_gate_is_open(chat):
    chat.sb.tables["mt_designs"].append({"student_id": chat.sid, "slug": SLUG,
                                         "approved": True,
                                         "created_at": "2026-09-30T00:00:00+00:00"})
    out = chat.say()
    assert out["offtrack"] is False and chat.state.reviews == 0, out
