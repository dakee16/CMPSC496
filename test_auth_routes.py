"""
test_auth_routes.py - the sign-in path, against a fake Supabase.

What is worth testing here is not "does bcrypt work" - it is the handful of
places where a browser used to be able to assert who it was:

  * a session cookie is what carries identity, and a forged one is refused;
  * /solved returns YOUR solves, not whoever's id you typed;
  * /decompose_chunks binds the session to the cookie, not to the body;
  * the teacher routes read the role from the DATABASE row.

No network and no credentials: the fake below implements just enough of the
supabase-py chain to answer the queries these routes make.
"""
import uuid

from fastapi.testclient import TestClient

import main.auth as auth
from frontend import api_server


class FakeTable:
    """Enough of the postgrest builder for select/eq/limit/insert/execute."""

    def __init__(self, rows):
        self.rows = rows
        self._filters = []
        self._pending = None

    def select(self, *_a, **_k):
        return self

    def eq(self, col, val):
        self._filters.append((col, val))
        return self

    def limit(self, _n):
        return self

    def order(self, *_a, **_k):
        return self

    def insert(self, data):
        self._pending = data
        return self

    def upsert(self, data, **_k):
        self._pending = data
        return self

    def execute(self):
        if self._pending is not None:
            row = dict(self._pending)
            # The unique index on username, which is what makes a duplicate
            # signup an error rather than a second account.
            if any(r["username"] == row.get("username") for r in self.rows):
                raise RuntimeError("duplicate key value violates "
                                   "students_username_uniq")
            row.setdefault("id", str(uuid.uuid4()))
            self.rows.append(row)
            self._pending = None
            return type("R", (), {"data": [row]})()
        out = [r for r in self.rows
               if all(r.get(c) == v for c, v in self._filters)]
        self._filters = []
        return type("R", (), {"data": out})()


class FakeSupabase:
    def __init__(self):
        self.students = []
        self.solved = []

    def table(self, name):
        return FakeTable({"students": self.students,
                          "solved": self.solved}.get(name, []))


def client():
    auth.SESSION_SECRET = "test-secret-not-a-real-key"
    auth.ALLOWED_DOMAINS = ("psu.edu",)
    api_server.set_supabase(FakeSupabase())
    # TestClient keeps cookies between calls, or every assertion below would
    # pass for the wrong reason.
    return TestClient(api_server.app)


PW = "a-good-enough-password"


def test_register_login_me_and_logout():
    c = client()
    r = c.post("/register", json={"username": "Abc123@PSU.edu", "password": PW})
    assert r.status_code == 200, r.text
    # Lowercased on the way in, so one person cannot become two accounts.
    assert r.json()["username"] == "abc123@psu.edu"
    assert r.json()["role"] == "student"
    assert auth.SESSION_COOKIE in r.cookies

    me = c.get("/auth/me")
    assert me.status_code == 200
    assert me.json()["username"] == "abc123@psu.edu"

    c.post("/logout")
    assert c.get("/auth/me").status_code == 401


def test_register_refuses_non_psu_and_weak_passwords():
    c = client()
    for user, pw in (("someone@gmail.com", PW),
                     ("attacker@notpsu.edu", PW),   # suffix confusion
                     ("abc123@psu.edu", "short")):
        r = c.post("/register", json={"username": user, "password": pw})
        assert r.status_code == 400, f"{user} / {pw} was accepted"


def test_duplicate_registration_is_refused():
    c = client()
    assert c.post("/register", json={"username": "a@psu.edu",
                                     "password": PW}).status_code == 200
    r = c.post("/register", json={"username": "A@psu.edu", "password": PW})
    assert r.status_code == 400, "the same address registered twice"
    assert "already exists" in r.json()["detail"]["message"]


def test_a_broken_insert_does_not_claim_the_account_exists():
    """Every failed insert used to read as "already exists", which is what a
    missing `role` column said to a student who had never registered."""
    c = client()

    class Broken(FakeSupabase):
        def table(self, name):
            t = super().table(name)
            t.execute = lambda: (_ for _ in ()).throw(
                RuntimeError("PGRST204 could not find the 'role' column"))
            return t

    api_server.set_supabase(Broken())
    r = c.post("/register", json={"username": "new@psu.edu", "password": PW})
    assert r.status_code == 400
    assert "already exists" not in r.json()["detail"]["message"]


def test_login_is_case_insensitive_and_rejects_a_wrong_password():
    c = client()
    c.post("/register", json={"username": "abc123@psu.edu", "password": PW})
    c.post("/logout")

    assert c.post("/login", json={"username": " ABC123@psu.edu ",
                                  "password": PW}).status_code == 200
    assert c.post("/login", json={"username": "abc123@psu.edu",
                                  "password": "not it"}).status_code == 401
    assert c.post("/login", json={"username": "nobody@psu.edu",
                                  "password": PW}).status_code == 401


def test_a_forged_cookie_is_not_a_session():
    c = client()
    c.post("/register", json={"username": "abc123@psu.edu", "password": PW})
    good = c.cookies[auth.SESSION_COOKIE]
    body, sig = good.split(".")

    for bad in (f"{body}x.{sig}", f"{body}.{sig[:-1]}A", "junk", ""):
        c.cookies.set(auth.SESSION_COOKIE, bad)
        assert c.get("/auth/me").status_code == 401, f"accepted {bad[:20]!r}"


def test_solved_is_scoped_to_the_signed_in_student():
    c = client()
    sb = api_server.get_supabase()
    c.post("/register", json={"username": "mine@psu.edu", "password": PW})
    mine = c.get("/auth/me").json()["student_id"]

    sb.solved.append({"student_id": mine, "problem_slug": "two-sum"})
    sb.solved.append({"student_id": "someone-else", "problem_slug": "secret"})

    assert c.get("/solved").json()["slugs"] == ["two-sum"]
    # The old route was /solved/{student_id} - anyone's progress for anyone.
    assert c.get("/solved/someone-else").status_code in (404, 405)


def test_signed_out_callers_cannot_start_or_upload():
    c = client()
    assert c.post("/decompose_chunks", json={
        "slug": "two-sum", "description": "d", "solution": "x"}).status_code == 401
    assert c.post("/teacher/assignments", json={
        "filename": "a.py", "content": "x"}).status_code == 401


def test_decompose_refuses_a_student_id_in_the_body():
    """extra="forbid" is what stops the old field being quietly honoured."""
    c = client()
    c.post("/register", json={"username": "abc123@psu.edu", "password": PW})
    r = c.post("/decompose_chunks", json={
        "slug": "two-sum", "description": "d", "solution": "x",
        "student_id": "somebody-else"})
    assert r.status_code == 422, "student_id in the body was accepted"


def test_a_student_cannot_reach_the_teacher_routes():
    c = client()
    c.post("/register", json={"username": "abc123@psu.edu", "password": PW})
    # 403, not 401: signed in, but the DATABASE says student. The old gate was
    # a radio button on the sign-in page.
    assert c.post("/teacher/assignments", json={
        "filename": "a.py", "content": "x"}).status_code == 403
    assert c.get("/teacher/assignments/any-id/problems").status_code == 403


def test_a_teacher_row_reaches_them():
    c = client()
    sb = api_server.get_supabase()
    c.post("/register", json={"username": "prof@psu.edu", "password": PW})
    sb.students[0]["role"] = "teacher"          # what the migration does by hand
    c.post("/logout")
    r = c.post("/login", json={"username": "prof@psu.edu", "password": PW})
    assert r.json()["role"] == "teacher"
    # Past the role gate; the empty fake `problems` table is what it sees next.
    assert c.get("/teacher/assignments/any-id/problems").status_code == 200


def test_no_signing_key_refuses_to_issue_sessions():
    c = client()
    auth.SESSION_SECRET = ""
    try:
        r = c.post("/login", json={"username": "a@psu.edu", "password": PW})
        assert r.status_code == 503, "issued a session with no signing key"
    finally:
        auth.SESSION_SECRET = "test-secret-not-a-real-key"
