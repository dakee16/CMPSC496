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
            # signup an error rather than a second account. Guarded on the
            # column being present at all: `problems` rows have no username,
            # and indexing blindly made every problems upsert raise KeyError -
            # which the routes report as "could not be saved".
            if "username" in row and any(r.get("username") == row["username"]
                                         for r in self.rows):
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
        self.problems = []

    def table(self, name):
        return FakeTable({"students": self.students,
                          "solved": self.solved,
                          "problems": self.problems}.get(name, []))


def client():
    auth.SESSION_SECRET = "test-secret-not-a-real-key"
    auth.ALLOWED_DOMAINS = ("psu.edu",)
    # Module-level, so without this a test that burns failed logins leaks a
    # lockout into whichever test happens to run next.
    auth._failures.clear()
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


def test_repeated_wrong_passwords_lock_the_account():
    c = client()
    c.post("/register", json={"username": "abc123@psu.edu", "password": PW})
    c.post("/logout")

    for _ in range(auth.FAIL_LIMIT):
        assert c.post("/login", json={"username": "abc123@psu.edu",
                                      "password": "not it"}).status_code == 401

    r = c.post("/login", json={"username": "abc123@psu.edu", "password": "not it"})
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) > 0
    # Locked even for the RIGHT password - otherwise the limit only slows down
    # the guesses that were never going to work.
    assert c.post("/login", json={"username": "abc123@psu.edu",
                                  "password": PW}).status_code == 429


def test_the_lock_is_per_account_not_shared():
    """Everyone on the VPN shares an egress IP, so a limit that is not keyed
    per username locks out the whole lab when one person fat-fingers."""
    c = client()
    c.post("/register", json={"username": "victim@psu.edu", "password": PW})
    c.post("/register", json={"username": "bystander@psu.edu", "password": PW})
    c.post("/logout")

    for _ in range(auth.FAIL_LIMIT + 1):
        c.post("/login", json={"username": "victim@psu.edu", "password": "no"})

    assert c.post("/login", json={"username": "victim@psu.edu",
                                  "password": PW}).status_code == 429
    assert c.post("/login", json={"username": "bystander@psu.edu",
                                  "password": PW}).status_code == 200


def test_the_lock_expires_and_a_success_clears_the_count():
    c = client()
    c.post("/register", json={"username": "abc123@psu.edu", "password": PW})
    c.post("/logout")

    for _ in range(auth.FAIL_LIMIT):
        c.post("/login", json={"username": "abc123@psu.edu", "password": "no"})
    assert c.post("/login", json={"username": "abc123@psu.edu",
                                  "password": PW}).status_code == 429

    # Age the recorded failures past the window; the lock must lift itself,
    # with nobody unlocking anything.
    auth._failures["abc123@psu.edu"] = [
        t - auth.FAIL_WINDOW - 1 for t in auth._failures["abc123@psu.edu"]]
    assert c.post("/login", json={"username": "abc123@psu.edu",
                                  "password": PW}).status_code == 200

    # And a success wipes the slate, so four fumbles then a success does not
    # leave the next session one mistake from a lockout.
    for _ in range(auth.FAIL_LIMIT - 1):
        c.post("/login", json={"username": "abc123@psu.edu", "password": "no"})
    assert c.post("/login", json={"username": "abc123@psu.edu",
                                  "password": PW}).status_code == 200
    assert "abc123@psu.edu" not in auth._failures


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


def teacher_client(monkeypatch=None):
    """A signed-in teacher, with one blocked problem to fix."""
    c = client()
    sb = api_server.get_supabase()
    c.post("/register", json={"username": "prof@psu.edu", "password": PW})
    sb.students[0]["role"] = "teacher"
    c.post("/logout")
    c.post("/login", json={"username": "prof@psu.edu", "password": PW})
    sb.problems.append({
        "slug": "is-armstrong", "title": "Is Armstrong",
        "description": "Given n, ...", "assignment_id": "a-1",
        "solution": 'def is_armstrong(n):\n    """Given n, ..."""\n    return n == 0\n',
        "ready": False,
        "prepare_error": "The generated tests were not strong enough to grade "
                         "this reliably."})
    return c, sb


def test_the_fix_panel_gets_the_source_and_the_failing_stage():
    """The instructor's own text comes back, with the gate it stopped at."""
    c, _ = teacher_client()
    r = c.get("/teacher/problems/is-armstrong/source?assignment_id=a-1")
    assert r.status_code == 200, r.text
    d = r.json()
    assert "def is_armstrong" in d["source"], "no text to fix"
    states = [x["state"] for x in d["checklist"]]
    # "not strong enough" is the fourth gate: three cleared, one failed, one
    # never reached. A checklist that failed everything would be a lie.
    assert states == ["ok", "ok", "ok", "fail", "pending"], states
    assert d["checklist"][3]["error"] == d["error"]


def test_a_student_cannot_read_a_problems_source_or_retry_it():
    """`solution` is the answer key. This is the one route that returns it."""
    c = client()
    c.post("/register", json={"username": "abc123@psu.edu", "password": PW})
    assert c.get(
        "/teacher/problems/is-armstrong/source?assignment_id=a-1").status_code == 403
    assert c.post("/teacher/problems/retry", json={
        "assignment_id": "a-1", "slug": "is-armstrong",
        "source": "def f():\n    pass\n"}).status_code == 403


def test_a_retry_that_cannot_parse_spends_nothing_and_keeps_the_stored_text():
    """The parse gate runs BEFORE preparation, so a bad edit costs no model
    work - and must not overwrite the version that is already stored."""
    c, sb = teacher_client()
    before = sb.problems[0]["solution"]
    r = c.post("/teacher/problems/retry", json={
        "assignment_id": "a-1", "slug": "is-armstrong",
        "source": "def is_armstrong(n):\n    return False\n"})   # no docstring
    assert r.status_code == 200, r.text          # teacher feedback, not a 4xx
    d = r.json()
    assert d["ready"] is False and d["stage"] == "parses"
    assert "docstring" in d["error"]
    assert [x["state"] for x in d["checklist"]] == \
        ["fail", "pending", "pending", "pending", "pending"]
    assert sb.problems[0]["solution"] == before, "a bad edit destroyed the source"


def test_a_successful_retry_publishes_the_problem(monkeypatch):
    """The pipeline itself is stubbed - it is minutes of paid model work and is
    covered elsewhere. What is tested here is the ROUTE: that a prepared problem
    is written back as ready, under the row's own slug."""
    c, sb = teacher_client()
    monkeypatch.setattr(
        "main.publish.prepare_problem",
        lambda p: {"slug": p["slug"], "ready": True, "chunks": 4,
                   "n_tests": 9, "stage": None, "error": None})
    r = c.post("/teacher/problems/retry", json={
        "assignment_id": "a-1",
        "slug": "is-armstrong",
        # A marker naming a DIFFERENT problem: identity is the row's, or a
        # rename would quietly create a second problem and leave this one
        # blocked.
        "source": '# --- problem: something-else ---\n'
                  'def is_armstrong(n):\n    """Given n, ..."""\n'
                  '    return n == 0\n'})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["ready"] is True and d["chunks"] == 4
    assert all(x["state"] == "ok" for x in d["checklist"]), d["checklist"]
    saved = [p for p in sb.problems if p["slug"] == "is-armstrong"]
    assert saved and saved[-1]["ready"] is True, "the fix was not saved"
    assert not any(p["slug"] == "something-else" for p in sb.problems), \
        "the edited marker renamed the problem"


def test_no_signing_key_refuses_to_issue_sessions():
    c = client()
    auth.SESSION_SECRET = ""
    try:
        r = c.post("/login", json={"username": "a@psu.edu", "password": PW})
        assert r.status_code == 503, "issued a session with no signing key"
    finally:
        auth.SESSION_SECRET = "test-secret-not-a-real-key"
