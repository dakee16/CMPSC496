"""test_purge_student.py - deleting a student deletes all of them.

THE GAP (29 Sep, found while wiping the test account). archive.purge_student is
"the one supported way" to delete a student - FERPA requests go through it -
and it missed three places their work lives:

    solved                 the student still showed problems as solved
    student_interactions   every graded step, with their code
    the grading store      on the server's own disk: their code again, and the
                           session the next open hands straight back

REAL: purge_student, the session store (SQLite in tmp), /solved and the
resume lookup a problem open uses. FAKED: Supabase, in memory.
"""
import pytest

from frontend import api_server
from main import archive, sessions
from test_auth_routes import FakeTable, client, register
from test_plan_gate_outage import Archive

TABLES = ("mt_graphs", "mt_designs", "mt_messages", "mt_submissions",
          "mt_sessions", "solved", "student_interactions")


class Table(FakeTable):
    _delete = False

    def __getattr__(self, _name):            # order, in_, gte ...: not needed
        return lambda *a, **k: self

    def delete(self):
        self._delete = True
        return self

    def execute(self):
        if self._delete:
            gone = [r for r in self.rows if all(r.get(c) == v for c, v in self._filters)]
            self.rows[:] = [r for r in self.rows if r not in gone]
            self._filters, self._delete = [], False
            return type("R", (), {"data": gone})()
        return super().execute()


class DB(Archive):
    storage = None                           # no design images in this test

    def table(self, name):
        rows = {"students": self.students, "solved": self.solved,
                "problems": self.problems}.get(name)
        return Table(rows if rows is not None else self.tables[name])


@pytest.fixture
def two_students(monkeypatch, tmp_path):
    monkeypatch.setenv("MICROTUTOR_SESSION_DB", str(tmp_path / "s.sqlite3"))
    c = client()
    db = DB()
    api_server.set_supabase(db)
    register(c, "gone@psu.edu")
    register(c, "stays@psu.edu")
    ids = {s["username"]: s["id"] for s in db.students}
    decomposition = {"header": "def f(x):", "chunks": []}
    for who, sid in ids.items():
        sessions.create_session({"slug": "p1", "title": "P", "description": "d",
                                 "solution": "def f(x):\n    return x"},
                                decomposition, "hash-p1", student_id=sid)
        for t in TABLES:
            db.table(t).insert({"student_id": sid, "slug": "p1",
                                "problem_slug": "p1"}).execute()
    c.cookies.clear()
    c.post("/login", json={"username": "gone@psu.edu",
                           "password": "a-good-enough-password"})
    return c, db, ids


def test_a_purged_student_is_gone_from_everywhere_their_work_lives(two_students):
    c, db, ids = two_students
    gone = ids["gone@psu.edu"]
    assert c.get("/solved").json()["slugs"] == ["p1"]            # before
    assert sessions.find_resumable(gone, "hash-p1") is not None

    removed = archive.purge_student(db, gone)

    for t in TABLES:
        assert not [r for r in db.table(t).select("*").execute().data
                    if r["student_id"] == gone], f"{t} still has the student"
    assert removed["local_sessions"] == 1
    assert sessions.find_resumable(gone, "hash-p1") is None, \
        "the next open would hand back the purged student's own work"
    assert c.get("/solved").json()["slugs"] == []


def test_everyone_else_is_left_alone(two_students):
    _c, db, ids = two_students
    archive.purge_student(db, ids["gone@psu.edu"])

    stays = ids["stays@psu.edu"]
    for t in TABLES:
        assert [r for r in db.table(t).select("*").execute().data
                if r["student_id"] == stays], f"{t} lost someone else's row"
    assert sessions.find_resumable(stays, "hash-p1") is not None
    assert [s["username"] for s in db.students] == ["gone@psu.edu", "stays@psu.edu"], \
        "purging work must not delete the account itself"
