"""Student-only records, grade semantics, pagination and private route tests.
No database, model, execution engine or grading session is started.
"""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from frontend.student_routes import student_progress_router
from main import auth, grades, student_progress


class Table:
    def __init__(self, db, name):
        self.db, self.name = db, name
        self.filters, self.orders = [], []
        self.fields, self.bounds = [], (0, 999)

    def select(self, fields):
        self.fields = [f.strip() for f in fields.split(",")]
        return self

    def eq(self, key, value):
        self.filters.append((key, "eq", value))
        return self

    def in_(self, key, values):
        self.filters.append((key, "in", values))
        return self

    def order(self, key, desc=False):
        self.orders.append((key, desc))
        return self

    def range(self, first, last):
        self.bounds = first, last
        return self

    def execute(self):
        self.db.queries.append((self.name, self.fields, self.filters))
        rows = [r for r in self.db.data.get(self.name, []) if all(
            r.get(k) == v if op == "eq" else r.get(k) in v for k, op, v in self.filters)]
        for key, desc in reversed(self.orders):
            rows.sort(key=lambda r: str(r.get(key) or ""), reverse=desc)
        start, end = self.bounds
        return SimpleNamespace(data=[{k:r[k] for k in self.fields if k in r} for r in rows[start:end+1]])


class DB:
    def __init__(self, data):
        self.data, self.queries = data, []

    def table(self, name):
        return Table(self, name)


@pytest.fixture
def db(monkeypatch):
    monkeypatch.setattr(student_progress, "step_counts", lambda _db, ps: {p["slug"]: 3 if p["slug"] != "pending" else 0 for p in ps})
    monkeypatch.setattr(grades, "MAX_ATTEMPTS", None)
    return DB({
        "assignments":[{"id":"a", "name":"Dictionaries", "published":True}, {"id":"hidden", "name":"Secret draft", "published":False}],
        "problems":[{"slug":slug,"title":slug.title(),"assignment_id":assignment,"ready":ready,"solution":"SECRET_REFERENCE","description":"Describe the problem"}
                    for slug,assignment,ready in [("employee","a",True),("new","a",True),("pending","a",True),("private","hidden",True),("unready","a",False)]],
        "mt_sessions":[{"session_id":"1","student_id":"alice","slug":"employee","started_at":"2026-09-11T23:00:00Z"},
                       {"session_id":"2","student_id":"bob","slug":"new","started_at":"2026-09-11T23:00:00Z","completed_at":"2026-09-11T23:20:00Z","solved_independently":True},
                       {"session_id":"3","student_id":"alice","slug":"private","started_at":"2026-09-11T23:00:00Z"}],
        "mt_submissions":[{"id":i,"student_id":student,"slug":slug,"chunk_index":step,"verdict":verdict,"created_at":"2026-09-12T01:00:00Z","code":"PRIVATE_CODE","reason":"PRIVATE_FEEDBACK"}
                          for i,(student,slug,step,verdict) in enumerate([
                              ("alice","employee",0,"incorrect"),("alice","employee",0,"correct"),
                              ("alice","employee",0,"correct"),("alice","employee",1,"incorrect"),
                              ("alice","employee",2,"indeterminate"),("bob","new",0,"correct"),
                              ("alice","private",0,"correct"),("alice","unready",0,"correct")])],
        "solved":[{"student_id":"bob","problem_slug":"new"}]
    })


def snapshot(db, **kwargs):
    return student_progress.progress_snapshot(db, "alice", now=datetime(2026,9,12,12,tzinfo=timezone.utc), **kwargs)


def test_grades_and_activity_only_include_caller_and_visible_ready_problems(db):
    data = snapshot(db, tz="America/New_York")
    assert [a["name"] for a in data["assignments"]] == ["Dictionaries"]
    assert {p["slug"] for p in data["problems"]} == {"employee","new","pending"}
    summary = data["summary"]
    assert summary["earned"] == 1  # repeated passes of the same step do not inflate credit
    assert summary["total"] == 6
    assert summary["percent"] == 17
    assert summary["in_progress"] == 1
    assert summary["completed"] == 0  # Bob's completion must not count for Alice
    assert summary["weekly_submissions"] == 4  # indeterminate is not a graded attempt
    assert summary["active_days"] == 1
    assert data["activity"][-2]["submissions"] == 4  # 01:00 UTC is the prior local day
    assert data["activity"][-1]["submissions"] == 0
    assert data["next_up"][0] == "employee"
    p = next(p for p in data["problems"] if p["slug"] == "employee")
    assert [s["status"] for s in p["steps"]] == ["passed","needs_work","not_graded"]
    assert "SECRET_REFERENCE" not in str(data) and "PRIVATE_CODE" not in str(data)
    assert "PRIVATE_FEEDBACK" not in str(data) and "bob" not in str(data)
    for table, fields, filters in db.queries:
        if table in ("mt_sessions","mt_submissions","solved"):
            assert ("student_id","eq","alice") in filters
        if table == "mt_submissions":
            assert "code" not in fields and "reason" not in fields


def test_unknown_denominator_is_not_a_zero_grade(db):
    data = snapshot(db)
    p = next(p for p in data["problems"] if p["slug"] == "pending")
    assert p["percent"] is None and p["total"] == 0 and p["steps"] == []
    assert data["summary"]["ungraded_problems"] == 1


def test_completion_with_help_is_separate_from_earned_credit(db, monkeypatch):
    monkeypatch.setattr(grades,"MAX_ATTEMPTS",2)
    db.data["mt_sessions"][0].update(completed_at="2026-09-12T02:00:00Z",solved_independently=False)
    db.data["mt_submissions"].append({"id":99,"student_id":"alice","slug":"employee","chunk_index":1,"verdict":"incorrect"})
    data=snapshot(db)
    p=next(p for p in data["problems"] if p["slug"]=="employee")
    assert p["status"]=="helped" and p["solved"]==1 and p["shown"]==1
    assert p["percent"]==33 and p["steps"][1]["status"]=="shown"
    assert data["summary"]["completed"]==1 and data["summary"]["independent"]==0


def test_paginated_history_keeps_the_last_passing_submission(db):
    db.data["mt_submissions"]=[{"id":str(i).zfill(4),"student_id":"alice","slug":"employee","chunk_index":0,"verdict":"incorrect"} for i in range(1001)]
    db.data["mt_submissions"].append({"id":"1001","student_id":"alice","slug":"employee","chunk_index":0,"verdict":"correct"})
    data=snapshot(db)
    assert data["summary"]["earned"]==1
    assert next(p for p in data["problems"] if p["slug"]=="employee")["attempts"]==1002


def test_empty_published_set_never_reads_student_or_reference_records(db):
    db.data["assignments"]=[]
    data=snapshot(db)
    assert data["summary"]["percent"] is None
    assert data["problems"]==[] and data["recent"]==[] and data["next_up"]==[]
    assert {table for table,_,_ in db.queries}=={"assignments"}


def route_client(db, monkeypatch):
    monkeypatch.setattr(auth,"SESSION_SECRET","student-progress-test-only")
    def require(request):
        claims=auth.read_session(request.cookies.get(auth.SESSION_COOKIE,""))
        if not claims:
            raise HTTPException(status_code=401)
        return claims
    app=FastAPI()
    app.include_router(student_progress_router(lambda:db,require))
    return TestClient(app)


def sign_in(client, name):
    client.cookies.set(auth.SESSION_COOKIE,auth.issue_session({"id":name,"username":name+"@psu.edu","first_name":name,"last_name":"Student","role":"student"}))


def test_private_route_rejects_missing_or_forged_cookies(db, monkeypatch):
    client=route_client(db,monkeypatch)
    assert client.get("/student/progress").status_code==401
    client.cookies.set(auth.SESSION_COOKIE,"forged")
    assert client.get("/student/progress").status_code==401
    assert db.queries==[]


def test_route_ignores_supplied_student_id_and_returns_no_store(db, monkeypatch):
    client=route_client(db,monkeypatch)
    sign_in(client,"alice")
    response=client.get("/student/progress?student_id=bob&timezone=America/New_York")
    assert response.status_code==200
    assert response.json()["summary"]["earned"]==1
    assert "no-store" in response.headers["cache-control"]
    assert client.get("/student/progress?timezone=invalid/zone").status_code==400


def test_db_failure_returns_unavailable_not_fabricated_zero_grades(db, monkeypatch):
    client=route_client(db,monkeypatch)
    sign_in(client,"alice")
    monkeypatch.setattr(db,"table",lambda _name: (_ for _ in ()).throw(RuntimeError("private db details")))
    response=client.get("/student/progress")
    assert response.status_code==503
    assert "summary" not in response.json()
    assert "private db details" not in response.text
