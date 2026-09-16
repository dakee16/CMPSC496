"""Difficulty counts and teacher-only access using saved, synthetic attempts."""
import pytest
from fastapi.testclient import TestClient

from frontend import api_server
from main import auth
from main.teacher_dashboard import dashboard_snapshot
from tests.test_student_progress import DB


@pytest.fixture
def db():
    people = ["alice", "bob", "cara", "dan", "teacher"]
    return DB({
        "students": [{"id": n, "username": n + "@psu.edu", "first_name": n.title(),
                      "last_name": "Student", "role": "teacher" if n == "teacher" else "student"}
                     for n in people],
        "assignments": [{"id": "a", "name": "Dictionaries", "published": True},
                        {"id": "b", "name": "Numbers", "published": True},
                        {"id": "hidden", "name": "Draft", "published": False}],
        "problems": [{"slug": slug, "title": slug.title(), "assignment_id": a,
                      "ready": ready, "solution": "SECRET"}
                     for slug, a, ready in [("invert", "a", True), ("count", "a", True),
                                            ("sum", "b", True), ("draft", "hidden", True),
                                            ("unready", "a", False)]],
        "mt_sessions": [{"session_id": n, "student_id": n, "slug": "invert",
                         "started_at": "2026-09-15T12:00:00Z"} for n in people],
        "mt_submissions": [{"id": i, "session_id": student, "student_id": student,
                            "slug": slug, "chunk_index": step, "verdict": verdict,
                            "reason": "Check the empty input.", "tier": "execution-final",
                            "code": "PRIVATE_CODE", "created_at": f"2026-09-15T12:{i:02}:00Z"}
                           for i, (student, slug, step, verdict) in enumerate([
                               ("alice", "invert", 0, "incorrect"),
                               ("alice", "invert", 0, "incorrect"),
                               ("alice", "invert", 1, "incorrect"),
                               ("bob", "invert", 0, "incorrect"),
                               ("bob", "invert", 0, "correct"),
                               ("cara", "invert", 0, "correct"),
                               ("dan", "invert", 0, "indeterminate"),
                               ("teacher", "invert", 0, "incorrect"),
                               ("alice", "draft", 0, "incorrect"),
                               ("alice", "unready", 0, "incorrect")])],
    })


def test_students_not_retries_define_difficulty_and_recovery(db):
    result = dashboard_snapshot(db)
    p = result["problems"][0]
    assert p["slug"] == "invert"
    assert (p["attempted"], p["needs_help"], p["recovered"], p["difficulty_percent"]) == (3, 1, 1, 67)
    assert [(s["number"], s["attempted"], s["needs_help"], s["recovered"]) for s in p["steps"]] == [(1, 3, 1, 1), (2, 1, 1, 0)]
    assert p["follow_up"][0]["steps"] == [1, 2]
    assert p["follow_up"][0]["name"] == "Alice Student"
    assert result["summary"] == {"students": 4, "active": 4, "needs_help": 1,
        "problems": 3, "attempted_problems": 1, "indeterminate": 1}
    assert "SECRET" not in str(result) and "PRIVATE_CODE" not in str(result)
    assert {p["slug"] for p in result["problems"]} == {"invert", "count", "sum"}
    assert all("solution" not in fields and "code" not in fields for _, fields, _ in db.queries)


def test_restart_drops_abandoned_session_and_missing_start_keeps_submissions(db):
    db.data["mt_submissions"].append({"id": 40, "session_id": "new-alice", "student_id": "alice",
        "slug": "invert", "chunk_index": 0, "verdict": "correct", "created_at": "2026-09-16T12:00:00Z"})
    p = dashboard_snapshot(db)["problems"][0]
    assert p["needs_help"] == 0
    assert p["recovered"] == 1  # Bob; Alice's old session is no longer current.
    db.data["mt_sessions"].append({"session_id": "new-bob", "student_id": "bob", "slug": "invert",
                                    "started_at": "2026-09-16T13:00:00Z"})
    p = next(p for p in dashboard_snapshot(db)["problems"] if p["slug"] == "invert")
    assert p["attempted"] == 2 and p["recovered"] == 0


def test_filter_empty_problem_and_pagination(db):
    data = dashboard_snapshot(db, "b")
    assert data["summary"]["active"] == 0
    assert data["problems"][0]["difficulty_percent"] is None
    assert len(data["assignments"]) == 2
    with pytest.raises(LookupError):
        dashboard_snapshot(db, "hidden")
    # A final pass beyond PostgREST's row cap must clear the follow-up flag.
    db.data["mt_submissions"] = [{"id": str(i).zfill(4), "student_id": "alice",
        "session_id": "alice", "slug": "invert", "chunk_index": 0,
        "verdict": "incorrect" if i < 1001 else "correct"} for i in range(1002)]
    p = dashboard_snapshot(db)["problems"][0]
    assert p["attempted"] == 1 and p["recovered"] == 1 and p["needs_help"] == 0


def test_real_route_requires_teacher_and_does_not_cache(db, monkeypatch):
    monkeypatch.setattr(api_server, "_SB", db)
    monkeypatch.setattr(auth, "SESSION_SECRET", "teacher-dashboard-test-only")
    monkeypatch.setattr(auth, "ALLOWED_EMAILS", set())
    client = TestClient(api_server.app)
    assert client.get("/teacher/dashboard").status_code == 401
    client.cookies.set(auth.SESSION_COOKIE, "forged")
    assert client.get("/teacher/dashboard").status_code == 401
    for role, status in [("student", 403), ("teacher", 200)]:
        client.cookies.set(auth.SESSION_COOKIE, auth.issue_session({
            "id": "test", "username": "test@psu.edu", "role": role}))
        response = client.get("/teacher/dashboard")
        assert response.status_code == status
        if role == "student":
            assert db.queries == []
    assert response.headers["cache-control"] == "private, no-store"
    assert client.get("/teacher/dashboard?assignment_id=b").json()["summary"]["active"] == 0
    assert client.get("/teacher/dashboard?assignment_id=missing").status_code == 404
    monkeypatch.setattr(db, "table", lambda name: (_ for _ in ()).throw(RuntimeError("PRIVATE DATABASE DETAILS")))
    response = client.get("/teacher/dashboard")
    assert response.status_code == 503 and "PRIVATE DATABASE DETAILS" not in response.text
    assert "summary" not in response.json()
