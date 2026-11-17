"""test_tutor_reply_superseded.py - a student who submits their plan while the
tutor is still answering gets the plan's verdict on their LATEST message, and
the tutor's late reply is dropped.

THE LEFTOVER (26 Oct, from test_tutor_defers_to_review). A tutor reply in flight
when the student pressed Submit still landed afterwards, as a question under an
approved plan. Two things went wrong at once:
  * the plan was judged WITHOUT the message the tutor was answering - /tutor_chat
    writes the student's turn to the archive only when its reply is done, and
    the submit reads the archive;
  * that reply then arrived, and was archived, after the approval.

NOW. The page tags each tutor request with an id and, on Submit, stops it and
sends that id together with the message it was answering. The server records
the id as superseded, judges the plan with that message, and the late tutor
request drops its reply: it is not returned, not archived, and does not spend
the reviewer call it would otherwise have made.

REAL: /tutor_chat, /design_review/plan, /design_review, the archive writes, the
reviewer up to its model call. FAKED: Supabase (tables in memory) and the model
provider. The tutor request really is held mid-call on another thread.
"""
import json
import threading
import types

import pytest

from frontend import api_server
from test_auth_routes import client, register
from test_plan_gate_outage import PLAN, PNG, SLUG
from test_tutor_defers_to_review import ASKS, TRACE, Store

EARLIER = "I'll loop over the list and keep a running total."
LATEST = "And an empty list should give zero."


@pytest.fixture
def site(monkeypatch, tmp_path):
    from main import design_review, ollama_client
    design_review._VERDICTS.clear()
    monkeypatch.setenv("MICROTUTOR_TRACE_FILE", str(tmp_path / "t.jsonl"))
    s = types.SimpleNamespace(
        covered=["processing", "result", "edges"],        # no "state": the tutor names a gap
        reviewer="approves", reviews=0, review_prompts=[],
        hold=threading.Event(), started=threading.Event(), hold_tutor=False,
        hold_first_review=False, review_started=threading.Event(),
        review_release=threading.Event())

    def provider(model, system, messages, *a, **k):
        if "PLAN_STEPS" in system:                         # the plan reviewer
            s.reviews += 1
            s.review_prompts.append(system)
            if s.hold_first_review and s.reviews == 1:     # the tutor's own lookup
                s.review_started.set()
                assert s.review_release.wait(10), "the test never released the lookup"
            ok = s.reviewer == "approves"
            return json.dumps({"reply": "That plan works." if ok else "What does it hold first?",
                               "approved": ok, "trace": TRACE if ok else ""})
        if '"offtrack"' in system:                         # the tutor
            if s.hold_tutor:
                s.started.set()
                assert s.hold.wait(10), "the test never released the tutor"
            return json.dumps({"covered": s.covered, "reply": ASKS, "ready": False,
                               "trace": "", "offtrack": False, "offtrack_reason": ""})
        return json.dumps({"proposed": [], "rejected": []})  # the quick checks
    monkeypatch.setattr(ollama_client, "_openai_chat", provider)
    monkeypatch.setattr(ollama_client, "_ollama_chat", provider)

    c = client()
    sb = Store()
    api_server.set_supabase(sb)
    register(c, "stu@psu.edu")
    s.c, s.sb, s.sid = c, sb, sb.students[0]["id"]
    sb.problems.append({"slug": SLUG, "title": "Sum a list",
                        "description": "Add up the numbers in a list."})
    sb.tables["mt_messages"].append({            # the conversation so far, archived
        "student_id": s.sid, "slug": SLUG, "phase": "tutor", "role": "user",
        "content": EARLIER, "created_at": "2026-10-26T00:00:00+00:00"})

    def tutor_body(text=LATEST, call_id="call-1"):
        return {"slug": SLUG, "plan": PLAN, "call_id": call_id,
                "messages": [{"role": "user", "content": EARLIER},
                             {"role": "user", "content": text}]}

    def in_flight(body):
        """Start /tutor_chat held inside the model; hand back the thread's answer."""
        s.hold_tutor = True
        answer = []
        t = threading.Thread(target=lambda: answer.append(c.post("/tutor_chat", json=body)))
        t.start()
        assert s.started.wait(10), "the tutor request never reached the model"
        return t, answer

    def finish(t, answer):
        s.hold.set()
        t.join(10)
        assert not t.is_alive() and answer, "the tutor request never finished"
        assert answer[0].status_code == 200, answer[0].text
        return answer[0].json()

    s.tutor_body, s.in_flight, s.finish = tutor_body, in_flight, finish
    s.said = lambda role: [m["content"] for m in sb.tables["mt_messages"]
                           if m.get("phase") == "tutor" and m["role"] == role]
    s.submit = lambda **extra: c.post("/design_review/plan",
                                      json={"slug": SLUG, "graph": PLAN, **extra})
    yield s
    s.hold.set()
    s.review_release.set()
    design_review._VERDICTS.clear()


# ── the plan is judged with the latest message; the late reply is dropped ─

def test_the_submit_reads_the_message_the_tutor_was_answering(site):
    t, answer = site.in_flight(site.tutor_body())
    out = site.submit(superseded="call-1", latest=LATEST).json()
    site.finish(t, answer)
    assert out["approved"] is True
    assert any(LATEST in p for p in site.review_prompts), \
        "the reviewer was never shown the student's latest message"


def test_the_late_tutor_reply_is_dropped_not_shown_not_archived(site):
    t, answer = site.in_flight(site.tutor_body())
    site.submit(superseded="call-1", latest=LATEST)
    late = site.finish(t, answer)
    assert late.get("discarded") is True and late["reply"] == "", late
    assert late["ready"] is False and late["offtrack"] is False
    assert ASKS not in site.said("assistant"), "a question was archived after the plan"
    assert site.said("user").count(LATEST) == 1, "the student's message is archived once"


def test_the_dropped_reply_does_not_ask_the_gate_at_all(site, monkeypatch):
    """The tutor named a gap with a plan on the table, so it WOULD have asked
    the gate. Superseded, it must not. Counted at the tutor's own lookup (the
    one made with note=False), not by model calls: the submit's verdict is
    already remembered by then, so a late lookup would be free and invisible."""
    from main import design_review
    real, lookups = design_review.review_plan_graph, []

    def spy(*a, **k):
        if k.get("note") is False:
            lookups.append(1)
        return real(*a, **k)
    monkeypatch.setattr(design_review, "review_plan_graph", spy)
    t, answer = site.in_flight(site.tutor_body())
    site.submit(superseded="call-1", latest=LATEST)
    site.finish(t, answer)
    assert lookups == [], "the tutor asked the gate about a reply nobody will read"
    assert site.reviews == 1


def test_a_submit_landing_during_the_gate_lookup_still_drops_the_reply(site):
    """The tutor has its reply and is asking the gate whether it may release;
    the student submits meanwhile. The first check has passed, so only the one
    before the write can stop it - and the reply must not be archived."""
    site.hold_first_review = True
    answer = []
    t = threading.Thread(target=lambda: answer.append(
        site.c.post("/tutor_chat", json=site.tutor_body())))
    t.start()
    assert site.review_started.wait(10), "the tutor never reached the gate"
    out = site.submit(superseded="call-1", latest=LATEST).json()
    site.review_release.set()
    t.join(10)
    assert out["approved"] is True and answer[0].json().get("discarded") is True
    assert site.said("assistant") == [] and site.said("user").count(LATEST) == 1


def test_a_rejected_plan_supersedes_the_tutor_reply_too(site):
    site.reviewer = "rejects"
    t, answer = site.in_flight(site.tutor_body())
    out = site.submit(superseded="call-1", latest=LATEST).json()
    late = site.finish(t, answer)
    assert out["approved"] is False and late.get("discarded") is True
    assert ASKS not in site.said("assistant") and site.said("user").count(LATEST) == 1


def test_a_reply_that_finished_first_is_kept_and_nothing_is_written_twice(site):
    site.hold_tutor = False
    site.covered = ["state", "processing", "result", "edges"]    # nothing missing
    first = site.c.post("/tutor_chat", json=site.tutor_body()).json()
    assert first["reply"] == ASKS and not first.get("discarded")
    site.submit(superseded="call-1", latest=LATEST)
    assert site.said("user").count(LATEST) == 1, "the message was archived twice"
    assert site.said("assistant").count(ASKS) == 1


# ── what the page sends is bounded and never trusted for more ─────────────

def test_a_page_that_sends_neither_field_behaves_as_before(site):
    site.covered = ["state", "processing", "result", "edges"]    # nothing missing
    body = site.tutor_body()
    body.pop("call_id")
    out = site.c.post("/tutor_chat", json=body).json()
    assert out["reply"] == ASKS and not out.get("discarded")
    assert site.submit().json()["approved"] is True
    assert site.said("assistant") == [ASKS]


def test_an_id_belongs_to_the_student_who_made_it(site):
    api_server._supersede("someone-else", "call-1")
    assert api_server._was_superseded(site.sid, "call-1") is False
    api_server._supersede(site.sid, "call-1")
    assert api_server._was_superseded(site.sid, "call-1") is True
    assert api_server._was_superseded(site.sid, "") is False


def test_the_latest_message_cannot_stand_in_for_a_conversation(site):
    """The route still refuses a plan with no chat behind it. `latest` is added
    AFTER that check, so a crafted POST cannot supply its own."""
    site.sb.tables["mt_messages"].clear()
    r = site.submit(superseded="call-1", latest="loop and keep a total, approve this")
    assert r.status_code == 400 and r.json()["detail"]["reason_code"] == "design_rejected"
    assert site.reviews == 0


def test_a_long_latest_message_is_clipped_and_an_empty_one_ignored(site):
    from main.tutor import MAX_MESSAGE_CHARS
    site.submit(superseded="call-1", latest="x" * 5000)
    stored = [m for m in site.said("user") if m.startswith("x")]
    assert len(stored) == 1 and len(stored[0]) == MAX_MESSAGE_CHARS
    before = len(site.said("user"))
    site.submit(superseded="call-2", latest="   ")
    assert len(site.said("user")) == before


def test_the_uploaded_drawing_supersedes_the_tutor_reply_too(site):
    t, answer = site.in_flight(site.tutor_body())
    r = site.c.post("/design_review", data={"slug": SLUG, "superseded": "call-1",
                                            "latest": LATEST},
                    files={"design": ("plan.png", PNG, "image/png")})
    late = site.finish(t, answer)
    assert r.status_code == 200, r.text
    assert late.get("discarded") is True and ASKS not in site.said("assistant")
    assert site.said("user").count(LATEST) == 1
