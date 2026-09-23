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
                            "code": f"CODE_{student}_{step}_{i}",
                            "created_at": f"2026-09-15T12:{i:02}:00Z"}
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


# Each student's own step prompts. Alice and Bob were served DIFFERENT
# decompositions of the same problem, which is real (pooled candidates, or a
# roadmap rebuilt around a student's plan).
PROMPTS = {"alice": ["Build the inverted mapping", "Hand the mapping back"],
           "bob": ["Collect the values first", "Hand the mapping back"],
           "cara": ["Build the inverted mapping", "Hand the mapping back"]}


def test_students_not_retries_define_difficulty_and_recovery(db):
    result = dashboard_snapshot(db, prompts_for=lambda sid: PROMPTS.get(sid, []))
    p = result["problems"][0]
    assert p["slug"] == "invert"
    assert (p["attempted"], p["needs_help"], p["recovered"], p["difficulty_percent"]) == (3, 1, 1, 67)
    assert [(s["number"], s["attempted"], s["needs_help"], s["recovered"]) for s in p["steps"]] == [(1, 3, 1, 1), (2, 1, 1, 0)]
    assert p["follow_up"][0]["steps"] == [1, 2]
    assert p["follow_up"][0]["name"] == "Alice Student"
    assert result["summary"] == {"students": 4, "active": 4, "needs_help": 1,
        "problems": 3, "attempted_problems": 1, "indeterminate": 1}
    assert {p["slug"] for p in result["problems"]} == {"invert", "count", "sum"}

    # THE TEACHER CAN NOW SEE WHAT TO DISCUSS: what each step asked, and the
    # code of the attempt that failed - "Check step 1" alone was a shrug.
    details = p["follow_up"][0]["details"]
    assert [(d["number"], d["prompt"]) for d in details] == [
        (1, "Build the inverted mapping"), (2, "Hand the mapping back")]
    # The attempt that FAILED LAST for that step, which is what the feedback
    # beside it is about - not an earlier one, and not a later correct one.
    assert [d["code"] for d in details] == ["CODE_alice_0_1", "CODE_alice_1_2"]
    assert all(d["reason"] == "Check the empty input." for d in details)

    # The class-wide list names the step by what MOST students were asked, and
    # says when that varied.
    first = next(s for s in p["steps"] if s["number"] == 1)
    assert first["prompt"] == "Build the inverted mapping" and first["prompt_varies"]
    second = next(s for s in p["steps"] if s["number"] == 2)
    assert second["prompt"] == "Hand the mapping back" and not second["prompt_varies"]

    # WHAT STILL NEVER LEAVES: the teacher's reference, code for a step the
    # student has since got right (Bob recovered), and anyone off the roster.
    text = str(result)
    assert "SECRET" not in text
    assert "CODE_bob" not in text and "CODE_cara" not in text
    assert "CODE_teacher" not in text
    assert all("solution" not in fields for _, fields, _ in db.queries)


def test_an_unreadable_session_falls_back_to_the_step_number(db):
    """No prompt is not an error: the page shows "Step N" as before."""
    def broken(_sid):
        return []
    p = dashboard_snapshot(db, prompts_for=broken)["problems"][0]
    assert all(d["prompt"] == "" for d in p["follow_up"][0]["details"])
    assert all(s["prompt"] == "" and not s["prompt_varies"] for s in p["steps"])


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


# ── the review page and "Issue seen" ─────────────────────────────────────
# Every test here writes to the server's own SQLite store (the seen marks, and
# the grading session the review is rebuilt from), so it is pointed at a temp
# file - never at the real data/grading_sessions.sqlite3.
@pytest.fixture(autouse=True)
def _temp_store(tmp_path, monkeypatch):
    monkeypatch.setenv("MICROTUTOR_SESSION_DB", str(tmp_path / "store.sqlite3"))


def test_issue_seen_clears_the_counter_until_the_next_mistake(db):
    from main.teacher_dashboard import mark_seen
    p = dashboard_snapshot(db)["problems"][0]
    assert p["needs_help"] == 1 and p["seen"] == 0

    mark_seen("alice", "invert")
    p = dashboard_snapshot(db)["problems"][0]
    # Out of the counter - so the tile stops asking for attention...
    assert p["needs_help"] == 0 and p["seen"] == 1
    assert dashboard_snapshot(db)["summary"]["needs_help"] == 0
    # ...but still listed, marked, so the instructor can find them again.
    alice = p["follow_up"][0]
    assert alice["name"] == "Alice Student" and alice["seen"] is True
    # Seen is not the same as recovered: Alice has not corrected anything.
    assert p["recovered"] == 1                     # still only Bob

    # A NEW mistake after it was marked brings her back.
    db.data["mt_submissions"].append({"id": 99, "session_id": "alice",
        "student_id": "alice", "slug": "invert", "chunk_index": 1,
        "verdict": "incorrect", "reason": "Still wrong.", "code": "x",
        "created_at": "2999-01-01T00:00:00Z"})
    p = dashboard_snapshot(db)["problems"][0]
    assert p["needs_help"] == 1 and p["follow_up"][0]["seen"] is False

    mark_seen("alice", "invert", seen=False)       # and it can be undone
    assert dashboard_snapshot(db)["problems"][0]["follow_up"][0]["seen"] is False


def test_the_review_rebuilds_the_function_and_marks_the_failing_step(tmp_path):
    """What an instructor needs in one place: the step, what it asked, what
    the student was told, the cases it failed on, the function up to that
    step, and every attempt. Built from a REAL grading session."""
    import types
    from main import sessions
    from main.teacher_dashboard import review_detail

    chunks = [types.SimpleNamespace(step_id="Part 1", prompt="Walk every pair",
                                    expected_type="code",
                                    reference="inverted = {}\nfor k, v in d.items():"),
              types.SimpleNamespace(step_id="Part 2", prompt="Swap each pair",
                                    expected_type="code",
                                    reference="    inverted[v] = k"),
              types.SimpleNamespace(step_id="Part 3", prompt="Hand it back",
                                    expected_type="code", reference="return inverted")]
    sid = sessions.create_session(
        {"slug": "invert", "title": "Invert", "description": "Invert a dict.",
         "solution": "SECRET_SOLUTION"},
        {"header": "def invert(d):", "chunks": chunks}, "h", student_id="alice")["session_id"]
    sessions.begin_submission(sid, "s1")
    s = sessions.load_session(sid)
    sessions.commit_outcome(sid, "s1", s["revision"], {"verdict": "correct"},
                            accept_code="inverted = {}\nfor k, v in d.items():")
    sessions.begin_submission(sid, "s2")
    s = sessions.load_session(sid)
    sessions.commit_outcome(sid, "s2", s["revision"], {
        "verdict": "incorrect", "reason": "Your step runs but the answer is wrong.",
        "failing_cases": ["invert({1: 2})\n\nexpected: {2: 1}\nyou gave: {1: 2}"],
        "failed_total": 3})

    data = DB({
        "problems": [{"slug": "invert", "title": "Invert", "assignment_id": "a",
                      "solution": "SECRET_SOLUTION"}],
        "students": [{"id": "alice", "username": "a@psu.edu", "first_name": "Alice",
                      "last_name": "Student", "role": "student"}],
        "assignments": [{"id": "a", "name": "Dictionaries"}],
        "mt_sessions": [{"session_id": sid, "student_id": "alice", "slug": "invert",
                         "started_at": "2026-09-15T12:00:00Z"}],
        "mt_submissions": [
            {"id": 1, "session_id": sid, "student_id": "alice", "slug": "invert",
             "chunk_index": 0, "verdict": "correct", "reason": "ok",
             "code": "inverted = {}\nfor k, v in d.items():",
             "created_at": "2026-09-15T12:01:00Z"},
            {"id": 2, "session_id": sid, "student_id": "alice", "slug": "invert",
             "chunk_index": 1, "verdict": "incorrect", "reason": "First try wrong.",
             "code": "inverted[k] = v", "created_at": "2026-09-15T12:02:00Z"},
            {"id": 3, "session_id": sid, "student_id": "alice", "slug": "invert",
             "chunk_index": 1, "verdict": "incorrect",
             "reason": "Your step runs but the answer is wrong.",
             # Typed FLAT, as students do; graded seated inside the loop.
             "code": "inverted[k] = k", "created_at": "2026-09-15T12:03:00Z"}]})

    r = review_detail(data, "invert", "alice")
    assert (r["name"], r["title"], r["assignment"]) == ("Alice Student", "Invert", "Dictionaries")
    assert r["header"] == "def invert(d):" and r["total_steps"] == 3
    assert r["open_steps"] == [2]
    # The function as it stood when step 2 failed: step 1, accepted.
    assert r["prefix"] == [{"number": 1, "prompt": "Walk every pair",
                            "code": "inverted = {}\nfor k, v in d.items():"}]
    step = r["step"]
    assert (step["number"], step["prompt"]) == (2, "Swap each pair")
    # The LAST failing attempt, re-seated where the grader put it.
    assert step["code"] == "    inverted[k] = k"
    assert step["failing_cases"] == ["invert({1: 2})\n\nexpected: {2: 1}\nyou gave: {1: 2}"]
    assert step["failed_total"] == 3
    assert [a["reason"] for a in step["attempts"]] == [
        "First try wrong.", "Your step runs but the answer is wrong."]
    assert r["seen"] is False
    # Never the teacher's reference, and never a chunk's reference code.
    assert "SECRET_SOLUTION" not in str(r) and "return inverted" not in str(r)


def test_review_and_seen_routes_are_teacher_only(db, monkeypatch):
    monkeypatch.setattr(api_server, "_SB", db)
    monkeypatch.setattr(auth, "SESSION_SECRET", "teacher-dashboard-test-only")
    monkeypatch.setattr(auth, "ALLOWED_EMAILS", set())
    client = TestClient(api_server.app)
    body = {"slug": "invert", "student_id": "alice", "seen": True}
    assert client.get("/teacher/review?slug=invert&student_id=alice").status_code == 401
    assert client.post("/teacher/issues/seen", json=body).status_code == 401
    client.cookies.set(auth.SESSION_COOKIE, auth.issue_session(
        {"id": "s", "username": "s@psu.edu", "role": "student"}))
    assert client.get("/teacher/review?slug=invert&student_id=alice").status_code == 403
    assert client.post("/teacher/issues/seen", json=body).status_code == 403
    client.cookies.set(auth.SESSION_COOKIE, auth.issue_session(
        {"id": "t", "username": "t@psu.edu", "role": "teacher"}))
    r = client.get("/teacher/review?slug=invert&student_id=alice")
    assert r.status_code == 200 and r.headers["cache-control"] == "private, no-store"
    assert r.json()["step"]["number"] == 2          # alice's latest open step
    assert client.post("/teacher/issues/seen", json=body).json()["seen"] is True
    assert client.get("/teacher/dashboard").json()["problems"][0]["needs_help"] == 0
    assert client.get("/teacher/review?slug=nope&student_id=alice").status_code == 404
