"""test_account_cap.py - one account can be given a hard dollar cap on model
spend without touching anyone else (main/account_cap.py).

WHY (29 Sep). An outside tester was about to drive test@test.com on the live
site, where every chat turn and plan review is a real OpenAI call, and the
OpenAI budget is shared with the real students. A cap on the whole site would
have stopped them too.

REAL: the routes, the middleware that says whose request it is, the model
client up to its HTTP post, and the spend file. FAKED: Supabase (in memory) and
OpenAI's HTTP endpoint - a canned reply that reports a known token usage, so
every dollar below is exact.
"""
import json
import types

import pytest

from frontend import api_server
from test_auth_routes import client, register
from test_plan_gate_outage import PLAN, PNG, SLUG, Archive

CAP = 0.03
USAGE = {"prompt_tokens": 2000, "completion_tokens": 250}
PRICE = {"gpt-4o": (2.50, 10.00), "gpt-4o-mini": (0.15, 0.60)}  # $ per 1M tokens
REPLY = json.dumps({"reply": "What does your plan hand back for an empty list?",
                    "approved": False, "ready": False, "nodes": [], "edges": []})


class _Reply:
    status_code = 200
    headers = {}

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": REPLY}}], "usage": USAGE,
                "model": "gpt-4o"}


# What a capped student SEES under the server's setting is not decided yet. The
# tutor's quick check (tutor.py, "FAILS OPEN ON PURPOSE") treats a refused call
# like an outage and answers with its canned question - every turn, the same
# one - instead of "unavailable". The money is capped either way.
UNDECIDED = ("server setting (gpt-4o): a refused quick check fails open and the "
             "tutor repeats one canned question instead of 'unavailable' - "
             "waiting on Sanan (30 Sep)")


# THE SERVER'S SETTING AND THE CODE'S DEFAULT. The quick check runs on
# MICROTUTOR_MODEL: gpt-4o on the server, gpt-4o-mini by default. It is read once
# at import, so it is set on the module. Until 30 Sep these tests ran only under
# whatever the machine's .env said - and passed only under the default.
@pytest.fixture(params=["gpt-4o", "gpt-4o-mini"], ids=["server-gpt-4o", "default-mini"])
def site(request, monkeypatch, tmp_path):
    from main import account_cap, ollama_client, tutor
    monkeypatch.setattr(tutor, "OPENAI_MODEL", request.param)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("MICROTUTOR_TRACE_FILE", str(tmp_path / "t.jsonl"))
    monkeypatch.setenv("MICROTUTOR_ACCOUNT_CAPS", f"Capped@psu.edu={CAP}")
    monkeypatch.setenv("MICROTUTOR_ACCOUNT_SPEND", str(tmp_path / "spend.json"))
    sent = []

    def openai(url, *a, json=None, **k):
        sent.append((account_cap.ACCOUNT.get(), json["model"]))
        return _Reply()
    monkeypatch.setattr(ollama_client.requests, "post", openai)

    c = client()
    sb = Archive()
    api_server.set_supabase(sb)
    sb.problems.append({"slug": SLUG, "title": "Sum a list",
                        "description": "Add up the numbers in a list."})

    def sign_in(username):
        c.cookies.clear()
        register(c, username)
    def billed():
        """What OpenAI would bill for every call that was really sent."""
        return sum(USAGE["prompt_tokens"] * PRICE[m][0] / 1e6
                   + USAGE["completion_tokens"] * PRICE[m][1] / 1e6 for _, m in sent)
    return types.SimpleNamespace(c=c, sent=sent, sign_in=sign_in, billed=billed,
                                 who=lambda: {a for a, _ in sent},
                                 spent=lambda: account_cap.spent(), model=request.param)


def _chat(site):
    return site.c.post("/tutor_chat", json={"slug": SLUG, "messages": [
        {"role": "user", "content": "I'll loop and keep a running total."}]})


def test_a_capped_account_sees_the_tutor_is_unavailable(site, request):
    if site.model == "gpt-4o":
        request.applymarker(pytest.mark.xfail(strict=True, reason=UNDECIDED))
    site.sign_in("capped@psu.edu")
    answers = [_chat(site).status_code for _ in range(12)]

    assert 200 in answers and answers[-1] == 503, answers
    assert _chat(site).json()["detail"]["reason_code"] == "tutor_unavailable"


def test_a_capped_account_is_stopped_before_its_cap(site):
    site.sign_in("capped@psu.edu")
    answers = [_chat(site).status_code for _ in range(12)]

    assert 503 in answers, answers
    big = sum(m == "gpt-4o" for _, m in site.sent)
    for _ in range(20):                 # keeps trying long after the cap
        _chat(site)
    assert sum(m == "gpt-4o" for _, m in site.sent) == big, \
        "a refused call must never reach OpenAI"
    assert site.billed() <= CAP, (site.billed(), site.sent)
    assert site.spent()["capped@psu.edu"] == pytest.approx(site.billed())


def test_everyone_else_is_untouched(site):
    site.sign_in("capped@psu.edu")
    assert 503 in [_chat(site).status_code for _ in range(12)], "never capped"
    site.sign_in("someone@psu.edu")
    before = len(site.sent)

    assert [_chat(site).status_code for _ in range(10)] == [200] * 10
    assert len(site.sent) > before and site.sent[-1] == ("someone@psu.edu", "gpt-4o")
    assert "someone@psu.edu" not in site.spent()


def test_a_restart_does_not_refill_the_budget(site, tmp_path):
    (tmp_path / "spend.json").write_text(json.dumps({"capped@psu.edu": CAP - 0.001}))
    site.sign_in("capped@psu.edu")
    _chat(site)             # what it SEES is test_a_capped_account_sees_...

    assert all(m != "gpt-4o" for _, m in site.sent) and site.billed() < 0.001


def test_every_student_route_that_calls_the_model_knows_whose_call_it_is(site, monkeypatch):
    """The middleware binds the account above the routes. A route that ran
    its model call outside that binding would be an uncapped hole - this is the
    check that none does, for a sync route, an async one and an upload."""
    monkeypatch.setenv("MICROTUTOR_ACCOUNT_CAPS", "capped@psu.edu=1.00")
    site.sign_in("capped@psu.edu")
    site.c.post("/tutor_chat", json={"slug": SLUG, "plan": PLAN, "messages": [
        {"role": "user", "content": "I'll loop and keep a running total."}]})
    site.c.post("/plan_graph", json={"slug": SLUG, "messages": [
        {"role": "user", "content": "I'll loop and keep a running total."}]})
    site.c.post("/design_review/plan", json={"slug": SLUG, "graph": PLAN})
    site.c.post("/design_review", data={"slug": SLUG},
                files={"design": ("plan.png", PNG, "image/png")})

    assert len(site.sent) >= 4, site.sent
    assert site.who() == {"capped@psu.edu"}, site.sent


def test_no_caps_means_nothing_is_bound_or_written(site, monkeypatch, tmp_path):
    monkeypatch.setenv("MICROTUTOR_ACCOUNT_CAPS", "")
    site.sign_in("capped@psu.edu")

    assert [_chat(site).status_code for _ in range(8)] == [200] * 8
    assert site.who() == {None}
    assert not (tmp_path / "spend.json").exists()


def test_a_malformed_cap_is_reported_not_guessed(capsys, monkeypatch):
    from main import account_cap
    monkeypatch.setenv("MICROTUTOR_ACCOUNT_CAPS", "test@test.com:1.50, a@psu.edu=0.5")
    assert account_cap.caps() == {"a@psu.edu": 0.5}
    assert "ignored 'test@test.com:1.50'" in capsys.readouterr().err
