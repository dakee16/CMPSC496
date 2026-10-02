"""test_db_retry.py - a database read cut off by a network blip is tried once
more; a write never is.

THE REPORT (2 Oct, live): /solved and /assignments answered 500 at the same
instant - "httpcore.ReadError: [Errno 11] Resource temporarily unavailable" -
while 9 other loads of those pages were fine.

REAL: the Supabase client exactly as the site builds it (api_server.
get_supabase, run_phase1._sb), and its PostgREST query code. FAKED: the network
under it - the same error as the live one, then an answer.
"""
import json

import httpx
import pytest

from main.db_retry import RetryReads

JWT_SHAPED = "aaaa.bbbb.cccc"


@pytest.fixture
def db(monkeypatch):
    from frontend import api_server
    from main import run_phase1
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", JWT_SHAPED)
    monkeypatch.setattr(api_server, "_SB", None)
    monkeypatch.setattr(run_phase1, "_SB", None)
    calls, plan = [], []

    def network(request):
        calls.append(request.method)
        step = plan.pop(0) if plan else "ok"
        if step == "blip":
            raise httpx.ReadError("[Errno 11] Resource temporarily unavailable")
        return httpx.Response(200, json=[{"problem_slug": "stack-pop"}],
                              headers={"content-range": "0-0/1"})
    sb = api_server.get_supabase()
    sb.postgrest.session._transport.inner = httpx.MockTransport(network)
    return type("DB", (), {"sb": sb, "calls": calls, "plan": plan,
                           "other": run_phase1._sb})


def test_a_read_cut_off_by_a_blip_is_tried_once_more(db):
    db.plan.append("blip")
    data = db.sb.table("solved").select("problem_slug").eq("student_id", "s").execute().data
    assert data == [{"problem_slug": "stack-pop"}]
    assert db.calls == ["GET", "GET"]


def test_only_once(db):
    db.plan += ["blip", "blip"]
    with pytest.raises(httpx.ReadError):
        db.sb.table("solved").select("problem_slug").execute()
    assert db.calls == ["GET", "GET"]


def test_a_write_is_never_repeated(db):
    """It may already have happened - repeating it could write twice."""
    db.plan.append("blip")
    with pytest.raises(httpx.ReadError):
        db.sb.table("solved").insert({"student_id": "s", "problem_slug": "x"}).execute()
    assert db.calls == ["POST"]


def test_both_live_clients_have_it(db):
    assert isinstance(db.sb.postgrest.session._transport, RetryReads)
    assert isinstance(db.other().postgrest.session._transport, RetryReads)
