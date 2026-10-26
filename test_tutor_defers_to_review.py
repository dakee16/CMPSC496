"""test_tutor_defers_to_review.py - the tutor does not keep naming a gap in a
plan the reviewer would approve, and the reviewer gives the same answer to the
same plan wherever it is asked.

THE REPORT (a student, 26 Oct). "Sometimes I am able to submit my plan and have
it accepted even though the AI tutor will tell me that something is missing.
It feels like the Plan Reviewer and the AI tutor are not hard coded to agree."

MEASURED on the live archive (26 Oct): 210 plans were approved. For 79 the
tutor's last message was still a question when the student submitted, and 51
more ended on a statement that did not release them. Those students solved the
problem as often as the rest (91% against 94% for uploaded drawings) with the
same number of wrong submissions per plan (0.9), so the reviewer was right and
the tutor was holding students on points the gate did not hold.

WHY. Two separate model calls judged one plan. The tutor keeps its own
covered/gap/ready verdict every turn; the reviewer, the only gate, reached its
own at submit. /tutor_chat already stopped the tutor RELEASING a plan the
reviewer rejects; nothing stopped it HOLDING one the reviewer approves. And
nothing made the reviewer say the same thing twice about one plan - the same
chat could pass on one click and fail on the next.

NOW. The reviewer is the one authority, both ways. When the tutor names a gap
(or demands the walk-through) and a plan is on the table, the real gate is
asked once - and if it approves, the tutor releases instead. Its verdict is
remembered per student and per conversation, so the submit that follows is the
same answer, with no second model call.

REAL: /tutor_chat and /design_review/plan, the tutor's parsing and gap rule,
the reviewer up to its model call and its approval rule, the archive writes.
FAKED: Supabase (tables in memory) and the model provider.
"""
import json
import re
import types

import pytest

from frontend import api_server
from test_auth_routes import client, register
from test_plan_gate_outage import PLAN, SLUG, Archive, Table

SAYS = "I'll loop over the list and keep a running total."
ALL_FOUR = ["state", "processing", "result", "edges"]
TRACE = "walked [1, 2, 3]: total 0 -> 1 -> 3 -> 6; [] -> 0; matches the statement"
ASKS = "What does the running total start out as?"


class Rows(Table):
    """The shared fake cannot store a LIST insert, which is how the archive
    writes a chat turn pair - so a conversation never reached the submit."""

    def execute(self):
        if isinstance(self._pending, list):
            rows, self._pending = self._pending, None
            self.rows.extend(dict(r) for r in rows)
            return type("R", (), {"data": rows})()
        return super().execute()


class Store(Archive):
    def table(self, name):
        known = {"students": self.students, "solved": self.solved,
                 "problems": self.problems}
        return Rows(known[name] if name in known else self.tables[name])


@pytest.fixture(autouse=True)
def _fresh_memory():
    """The reviewer remembers what it said about a conversation. That memory
    belongs to the process, so a test must start without the last one's."""
    from main import design_review
    design_review._VERDICTS.clear()
    yield
    design_review._VERDICTS.clear()


@pytest.fixture
def site(monkeypatch, tmp_path):
    from main import ollama_client
    monkeypatch.setenv("MICROTUTOR_TRACE_FILE", str(tmp_path / "t.jsonl"))
    s = types.SimpleNamespace(covered=["processing", "result", "edges"],  # no "state"
                              tutor_ready=False, tutor_trace="", reviewer="approves",
                              offtrack=False, proposes=False,
                              reviews=0, notes=0, tutor_calls=0)

    def provider(model, system, messages, *a, **k):
        if "PLAN_STEPS" in system:                              # the plan reviewer
            s.reviews += 1
            if s.reviewer == "down":
                raise RuntimeError("OpenAI 503")
            if s.reviewer == "garbage":
                return "not json at all"
            ok = s.reviewer == "approves"
            return json.dumps({"reply": "That plan works." if ok else
                               "What does your total hold before the first number?",
                               "approved": ok, "trace": TRACE if ok else ""})
        if "ONE optional simplification" in system:             # the necessity note
            s.notes += 1
            return json.dumps({"has_tip": False, "note": ""})
        if '"offtrack"' in system:                              # the tutor
            s.tutor_calls += 1
            return json.dumps({"covered": s.covered, "reply": ASKS, "ready": s.tutor_ready,
                               "trace": s.tutor_trace, "offtrack": s.offtrack,
                               "offtrack_reason": "the plan never says what the "
                                                  "total holds before the first number"
                               if s.offtrack else ""})
        # The quick check: does the student PROPOSE the container they named?
        words = re.search(r"Words: (.*)", messages[0]["content"])
        return json.dumps({"proposed": [w.strip() for w in words.group(1).split(",")]
                           if s.proposes and words else [], "rejected": []})
    monkeypatch.setattr(ollama_client, "_openai_chat", provider)
    monkeypatch.setattr(ollama_client, "_ollama_chat", provider)

    c = client()
    sb = Store()
    api_server.set_supabase(sb)
    register(c, "stu@psu.edu")
    sb.problems.append({"slug": SLUG, "title": "Sum a list",
                        "description": "Add up the numbers in a list."})

    def say(text=SAYS, plan=PLAN, hint=None):
        body = {"slug": SLUG, "messages": [{"role": "user", "content": text}]}
        if plan is not None:
            body["plan"] = plan
        if hint:
            body["offtrack_hint"] = hint
        r = c.post("/tutor_chat", json=body)
        assert r.status_code == 200, r.text
        return r.json()

    def submit():
        r = c.post("/design_review/plan", json={"slug": SLUG, "graph": PLAN})
        assert r.status_code == 200, r.text
        return r.json()
    s.c, s.say, s.submit, s.sb, s.sid = c, say, submit, sb, sb.students[0]["id"]
    return s


# ── the tutor stops holding what the gate would pass ─────────────────────

def test_a_gap_the_reviewer_does_not_share_is_dropped(site):
    """THE REPORT: the tutor asks what is missing; the reviewer approves it."""
    out = site.say()
    assert out["ready"] is True and out.get("released_by_review") is True, out
    assert "?" not in out["reply"] and "missing" in out["reply"], out["reply"]
    assert site.reviews == 1, "the real gate was asked exactly once"


def test_the_page_is_not_sent_the_tutors_private_gap(site):
    for reviewer in ("approves", "rejects"):
        site.reviewer = reviewer
        assert "gap" not in site.say(f"{SAYS} ({reviewer})")


def test_a_gap_the_reviewer_shares_is_still_asked(site):
    site.reviewer = "rejects"
    out = site.say()
    assert out["ready"] is False and out["reply"] == ASKS, out
    assert not out.get("released_by_review") and site.reviews == 1


def test_the_demand_for_a_walk_through_is_a_gap_too(site):
    """`ready` with no hand-trace behind it is held with a canned question."""
    site.covered, site.tutor_ready, site.tutor_trace = ALL_FOUR, True, ""
    out = site.say()
    assert out["ready"] is True and out.get("released_by_review") is True, out
    assert site.reviews == 1


def test_the_canned_question_about_a_new_container_is_a_gap_too(site):
    """A newly named structure is answered with a fixed question - the tutor's
    model is never asked - and it says the starting value is missing."""
    site.proposes = True
    out = site.say("I'll keep a dictionary of the totals.")
    assert site.tutor_calls == 0, "this turn took the canned path"
    assert out["ready"] is True and out.get("released_by_review") is True, out
    assert site.reviews == 1


def test_the_canned_question_stays_when_the_reviewer_shares_the_gap(site):
    site.proposes, site.reviewer = True, "rejects"
    out = site.say("I'll keep a dictionary of the totals.")
    assert site.tutor_calls == 0 and out["ready"] is False, out
    assert not out.get("released_by_review") and "?" in out["reply"]


def test_a_probing_question_with_nothing_missing_is_left_alone(site):
    """The two-question floor is not a gap: nothing is said to be missing, and
    the gate is not paid to second-guess it."""
    site.covered = ALL_FOUR
    out = site.say()
    assert out["ready"] is False and out["reply"] == ASKS, out
    assert site.reviews == 0


def test_the_gate_is_asked_only_once_a_plan_exists(site):
    for plan in (None, {"nodes": [], "edges": []}):
        out = site.say(plan=plan)
        assert out["ready"] is False and not out.get("released_by_review"), (plan, out)
    assert site.reviews == 0


def test_a_question_to_the_tutor_is_answered_not_overridden(site):
    out = site.say("Can you explain what the total means here?")
    assert out["ready"] is False and out["reply"] == ASKS, out
    assert site.reviews == 0


def test_trying_something_else_is_never_released(site):
    out = site.say(hint="the plan never handles an empty list")
    assert out["ready"] is False and not out.get("released_by_review"), out
    assert site.reviews == 0


def test_an_unreachable_reviewer_leaves_the_tutors_reply_alone(site):
    site.reviewer = "down"
    out = site.say()
    assert out["ready"] is False and out["reply"] == ASKS, out
    assert site.reviews == 1


def test_once_the_gate_is_open_nothing_is_asked(site):
    site.sb.tables["mt_designs"].append({"student_id": site.sid, "slug": SLUG,
                                         "approved": True,
                                         "created_at": "2026-10-26T00:00:00+00:00"})
    out = site.say()
    assert not out.get("released_by_review") and site.reviews == 0, out


def test_the_reviewer_is_asked_once_a_turn_however_many_guards_look(site):
    """Tutor ready (one guard) and naming a gap (another) read one verdict."""
    site.reviewer = "rejects"
    site.covered, site.tutor_ready, site.tutor_trace = ALL_FOUR, True, TRACE
    site.say()
    assert site.reviews == 1


def test_an_unreachable_reviewer_is_not_tried_three_times_in_one_turn(site):
    """A failed call is never remembered, so only the turn's own single lookup
    keeps the box guard and the release guard from each calling a dead gate."""
    site.reviewer, site.offtrack = "down", True      # flagged AND naming a gap
    out = site.say()
    assert out["ready"] is False and out["offtrack"] is False, out
    assert site.reviews == 1


# ── the same plan, the same answer ───────────────────────────────────────

def test_the_submit_after_a_release_is_the_same_answer_and_costs_nothing(site):
    assert site.say()["ready"] is True and site.reviews == 1
    out = site.submit()
    assert out["approved"] is True and out["reply"] == "That plan works.", out
    assert site.reviews == 1, "the verdict was remembered, not asked for again"


def test_a_rejection_is_remembered_too(site):
    site.reviewer = "rejects"
    site.say()
    first = site.submit()
    again = site.submit()
    assert first["approved"] is False and again["approved"] is False
    assert first["reply"] == again["reply"] and site.reviews == 1
    assert again["round"] == first["round"] + 1, "every submit is still a round"


def test_a_changed_conversation_is_asked_again(site):
    site.reviewer = "rejects"
    site.say()
    site.reviewer = "approves"
    site.say("It starts at zero, and an empty list gives zero.")
    assert site.reviews == 2


def test_an_answer_that_could_not_be_read_is_not_remembered(site):
    site.say()                                   # puts words on the record
    from main import design_review
    design_review._VERDICTS.clear()
    site.reviewer = "garbage"
    assert site.submit()["approved"] is False
    site.reviewer = "approves"
    assert site.submit()["approved"] is True, "a model slip must not stick"
    assert site.reviews == 3                     # the release, the slip, the retry


def test_one_students_verdict_is_never_another_students(site):
    from main import design_review
    site.say()
    assert len(design_review._VERDICTS) == 1
    site.c.cookies.clear()
    register(site.c, "other@psu.edu")            # same problem, same words
    site.say()
    assert site.reviews == 2 and len(design_review._VERDICTS) == 2


def test_the_round_cap_still_applies_to_a_remembered_verdict(site):
    from main import design_review
    site.say()
    prior = [{"role": "assistant", "content": "no"}] * design_review.MAX_ROUNDS
    out = design_review.review_plan_graph(
        {"slug": SLUG, "description": "Add up the numbers in a list."}, PLAN, prior,
        chat_log=[{"role": "user", "content": SAYS}], who=site.sid)
    assert out["approved"] is False and "back and forth" in out["reply"]


def test_the_optional_note_is_not_bought_at_the_tutors_turn(site):
    """A tip about a collect-then-transform plan is for the submit, not a side
    effect of the tutor asking whether it may release."""
    redundant = {"nodes": [{"id": "a", "kind": "step", "label": "create an empty list of numbers"},
                           {"id": "b", "kind": "loop", "label": "for each number in the list"},
                           {"id": "c", "kind": "step", "label": "create a new dict from the list"},
                           {"id": "d", "kind": "return", "label": "return the dict"}],
                 "edges": []}
    site.say(plan=redundant)
    assert site.notes == 0
