"""The scaled-out stores behave like the single-box ones they replace.

Postgres tests need a disposable server, because they truncate its tables:

    TEST_DATABASE_URL=postgresql://... pytest test_scale_out.py

Redis tests run against fakeredis. Without either, those tests skip.
"""
import json
import os
import threading
from types import SimpleNamespace as N

import pytest

from main import admission, docstore, execution, migrate_shared, pg, sessions, shared

TEST_DB = os.environ.get("TEST_DATABASE_URL", "")
needs_postgres = pytest.mark.skipif(not TEST_DB, reason="TEST_DATABASE_URL not set")
PROBLEM = {"slug": "invert", "title": "Invert", "description": "d",
           "solution": "def invert(d):\n    return {}"}
DECOMPOSITION = {"header": "def invert(d):", "chunks": [
    N(step_id=1, prompt="one", expected_type="code", reference="    a = 1"),
    N(step_id=2, prompt="two", expected_type="code", reference="    return a")]}


@pytest.fixture
def database(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", TEST_DB)
    conn = pg.connect()
    try:
        conn.execute("TRUNCATE sessions, submissions, mt_docs RESTART IDENTITY")
    finally:
        conn.close()
    docstore._recent.clear()
    if hasattr(docstore._local, "snapshots"):
        docstore._local.snapshots.clear()


@pytest.fixture
def redis_fake():
    fakeredis = pytest.importorskip("fakeredis")
    shared.set_client(fakeredis.FakeRedis(decode_responses=True))
    yield shared.client()
    shared.set_client(None)


def _new_session(student="s1"):
    return sessions.create_session(PROBLEM, DECOMPOSITION, "hash", student_id=student)


# ── sessions on Postgres ─────────────────────────────────────────────────

@needs_postgres
def test_a_session_runs_start_to_finish(database):
    sid = _new_session()["session_id"]
    done, s = sessions.begin_submission(sid, "sub1")
    assert done is None
    first = sessions.commit_outcome(sid, "sub1", s["revision"], {"verdict": "correct"},
                                    accept_code="    a = 1")
    assert first["index"] == 1 and not first["completed"]

    done, s = sessions.begin_submission(sid, "sub2")
    last = sessions.commit_outcome(sid, "sub2", s["revision"], {"verdict": "correct"},
                                   accept_code="    return a")
    assert last["completed"]
    assert sessions.stored_result(sid, "sub2")["idempotent_replay"] is True
    assert "invert" in sessions.completed_answers("s1", ["invert"])


@needs_postgres
def test_identical_concurrent_submissions_are_graded_once(database):
    sid = _new_session()["session_id"]
    outcomes = []

    def submit():
        outcomes.append(sessions.begin_submission(sid, "same")[0])

    threads = [threading.Thread(target=submit) for _ in range(16)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sum(o is None for o in outcomes) == 1
    assert all(o == {"__in_flight__": True} for o in outcomes if o is not None)


@needs_postgres
def test_only_one_of_two_racing_commits_advances(database):
    sid = _new_session()["session_id"]
    _, s = sessions.begin_submission(sid, "a")
    sessions.begin_submission(sid, "b")
    sessions.commit_outcome(sid, "a", s["revision"], {"verdict": "correct"},
                            accept_code="    a = 1")
    with pytest.raises(sessions.SessionError) as e:
        sessions.commit_outcome(sid, "b", s["revision"], {"verdict": "correct"},
                                accept_code="    b = 2")
    assert e.value.reason_code == "session_conflict"
    assert sessions.session_snapshot(sid)["index"] == 1


@needs_postgres
def test_resume_prefers_the_session_holding_work(database):
    sid = _new_session()["session_id"]
    _, s = sessions.begin_submission(sid, "a")
    sessions.commit_outcome(sid, "a", s["revision"], {"verdict": "correct"},
                            accept_code="    a = 1")
    _new_session()                      # newer, empty
    assert sessions.find_resumable("s1", "hash")["session_id"] == sid


@needs_postgres
def test_a_finished_session_reopens_as_a_copy(database):
    sid = _new_session()["session_id"]
    for name, code in (("a", "    a = 1"), ("b", "    return a")):
        _, s = sessions.begin_submission(sid, name)
        sessions.commit_outcome(sid, name, s["revision"], {"verdict": "correct"},
                                accept_code=code)
    reopened = sessions.reopen_step(sid, 0)
    assert reopened["copied_from"] == sid
    assert sessions.load_session(reopened["session_id"])["index"] == 0


@needs_postgres
def test_a_nul_in_a_client_id_is_just_unknown(database):
    with pytest.raises(sessions.SessionError) as e:
        sessions.load_session("bad\x00id")
    assert e.value.reason_code == "session_not_found"


@needs_postgres
def test_purge_removes_a_students_rows(database):
    sid = _new_session()["session_id"]
    sessions.begin_submission(sid, "a")
    assert sessions.purge_student("s1") == 1
    assert sessions.session_snapshot(sid) is None


# ── the JSON stores on Postgres ──────────────────────────────────────────

@needs_postgres
def test_two_writers_do_not_erase_each_others_entries(database):
    docstore.save_all("t", {"a": 1})
    errors = []
    barrier = threading.Barrier(2)

    def writer(name):
        try:
            docs = docstore.load_all("t")
            barrier.wait()
            docs[name] = name
            barrier.wait()
            docstore.save_all("t", docs)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(n,)) for n in ("x", "y")]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    assert docstore.load_all("t") == {"a": 1, "x": "x", "y": "y"}


@needs_postgres
def test_a_harmless_mark_is_seen_by_every_worker(database):
    from main import review_help
    errors = []
    barrier = threading.Barrier(2)

    def teacher(key):
        try:
            barrier.wait()
            review_help.dismiss(key, "line 3", "dr")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=teacher, args=(k,)) for k in ("p1", "p2")]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    assert review_help.dismissed("p1")["line 3"]["by"] == "dr"
    assert review_help.dismissed("p2")["line 3"]["by"] == "dr"


@needs_postgres
def test_deleting_an_entry_deletes_only_that_entry(database):
    docstore.save_all("t", {"a": 1, "b": 2})
    docs = docstore.load_all("t")
    del docs["a"]
    docstore.save_all("t", docs)
    assert docstore.load_all("t") == {"b": 2}


@needs_postgres
def test_a_write_is_visible_at_once_on_this_worker(database):
    docstore.save_all("t", {"a": 1})
    assert docstore.get("t", "a") == 1
    docs = docstore.load_all("t")
    docs["a"] = 2
    docstore.save_all("t", docs)
    assert docstore.get("t", "a") == 2


@needs_postgres
def test_oracle_values_json_cannot_hold_survive(database):
    from main import oracle_store
    tests = [{"input": [{(0, 1): {2019, }}], "expected": (1, 2)}]
    oracle_store.save_cache({"h": {"strong": True, "final_tests": tests}})
    assert oracle_store.load_entry("h")["final_tests"] == tests
    assert oracle_store.load_cache()["h"]["final_tests"] == tests


@needs_postgres
def test_migration_copies_everything_and_is_repeatable(database, tmp_path, monkeypatch):
    db = str(tmp_path / "old.sqlite3")
    old = sessions.create_session(PROBLEM, DECOMPOSITION, "hash", student_id="s1",
                                  db_path=db)["session_id"]
    sessions.begin_submission(old, "sub1", db_path=db)
    oracle = tmp_path / "oracle.json"
    oracle.write_text(json.dumps({"h": {"strong": True, "final_tests": []}}))
    monkeypatch.setenv("MICROTUTOR_ORACLE_CACHE", str(oracle))
    monkeypatch.setenv("MICROTUTOR_CHUNK_POOL", str(tmp_path / "missing.json"))
    monkeypatch.setenv("MICROTUTOR_TRANSCRIPTS", str(tmp_path / "missing2.json"))

    for _ in range(2):
        assert migrate_shared.main(["--sqlite", db]) == 0
    assert sessions.load_session(old)["student_id"] == "s1"
    assert docstore.load_all("oracle")["h"]["strong"] is True
    conn = pg.connect()
    try:
        assert conn.execute("SELECT count(*) AS n FROM submissions").fetchone()["n"] == 1
    finally:
        conn.close()


# ── state shared through Redis ───────────────────────────────────────────

def test_the_login_budget_is_shared_between_workers(redis_fake):
    from main import auth
    auth._failures.clear()
    for _ in range(3):
        auth._record_failure("u@psu.edu", 1000.0)
    assert not auth._failures, "must be counted in the shared store"
    assert auth._recent_failures("u@psu.edu", 1001.0) == [1000.0] * 3
    auth._clear_failures("u@psu.edu")
    assert auth._recent_failures("u@psu.edu", 1001.0) == []


def test_old_login_failures_age_out(redis_fake):
    from main import auth
    auth._record_failure("u@psu.edu", 1000.0)
    assert auth._recent_failures("u@psu.edu", 1000.0 + auth.FAIL_WINDOW + 1) == []


def test_a_redis_outage_falls_back_to_local_counting():
    from main import auth

    class Down:
        def __getattr__(self, _):
            raise ConnectionError("down")

    shared.set_client(Down())
    auth._failures.clear()
    try:
        auth._record_failure("u@psu.edu", 1000.0)
        assert auth._recent_failures("u@psu.edu", 1001.0) == [1000.0]
    finally:
        auth._failures.clear()
        shared.set_client(None)


def test_a_spend_cap_holds_under_concurrent_workers(redis_fake, monkeypatch):
    from main import account_cap
    monkeypatch.setenv("MICROTUTOR_ACCOUNT_CAPS", "t@psu.edu=0.05")
    account_cap.ACCOUNT.set("t@psu.edu")
    admitted = refused = 0
    for _ in range(20):
        try:
            account_cap.reserve("gpt-4o", [{"content": "hi"}], (2.5, 10.0))
            admitted += 1
        except account_cap.CapReached:
            refused += 1
    account_cap.ACCOUNT.set(None)
    assert admitted == 4 and refused == 16      # each hold is just over 1c
    assert account_cap.spent()["t@psu.edu"] <= 0.05


def test_a_spend_cap_refuses_when_it_cannot_count(monkeypatch):
    from main import account_cap

    class Down:
        def hincrbyfloat(self, *_):
            raise ConnectionError("down")

    shared.set_client(Down())
    monkeypatch.setenv("MICROTUTOR_ACCOUNT_CAPS", "t@psu.edu=1")
    account_cap.ACCOUNT.set("t@psu.edu")
    try:
        with pytest.raises(account_cap.CapReached):
            account_cap.reserve("gpt-4o", [{"content": "hi"}], (2.5, 10.0))
    finally:
        account_cap.ACCOUNT.set(None)
        shared.set_client(None)


def test_a_model_answer_is_remembered_across_workers(redis_fake):
    from main import ollama_client as client
    client._memo_put("k", "answer")
    client._MEMO.clear()                # the other worker has an empty memo
    assert client._memo_get("k") == "answer"


def test_a_watcher_on_another_worker_replays_then_follows(redis_fake):
    from main import prepare_bus as bus
    k = bus.key("asg", "slug")
    bus.open_channel(k)
    bus.publish(k, {"type": "stage", "n": 1})
    assert bus.is_open(k)
    stream = bus.subscribe(k)
    assert next(stream) == {"type": "stage", "n": 1}
    bus.publish(k, {"type": "verdict"})
    bus.finish(k)
    assert [e for e in stream if e] == [{"type": "verdict"}]


def test_a_reupload_does_not_replay_the_old_run(redis_fake):
    from main import prepare_bus as bus
    k = bus.key("asg", "slug")
    bus.open_channel(k)
    bus.publish(k, {"n": "old"})
    bus.open_channel(k)
    bus.publish(k, {"n": "new"})
    bus.finish(k)
    assert [e for e in bus.subscribe(k) if e] == [{"n": "new"}]


def test_a_channel_that_never_ran_ends(redis_fake):
    from main import prepare_bus as bus
    assert not bus.is_open("no/such")
    assert list(bus.subscribe("no/such")) == []


def test_the_fleet_model_rate_is_enforced(redis_fake, monkeypatch):
    monkeypatch.setenv("MICROTUTOR_T_RPM", "2")
    admission.require_rate("model", "MICROTUTOR_T_RPM")
    admission.require_rate("model", "MICROTUTOR_T_RPM")
    with pytest.raises(admission.Busy):
        admission.require_rate("model", "MICROTUTOR_T_RPM")


# ── execution ────────────────────────────────────────────────────────────

GOOD = "def f(x):\n    return {x: (x, 1)}\n"


def _call_runner(monkeypatch, token="secret"):
    from fastapi.testclient import TestClient
    from main.runner_service import app
    client = TestClient(app)
    monkeypatch.setenv("MICROTUTOR_RUNNER_URL", "http://runner")
    monkeypatch.setenv("MICROTUTOR_RUNNER_TOKEN", token)

    def post(url, data, headers, timeout):
        return client.post("/run", content=data, headers=headers)
    monkeypatch.setattr(execution._runner_http, "post", post)


def test_a_remote_run_matches_a_local_run(monkeypatch):
    local = execution.run_student_code(GOOD, [[3]], entry_name="f")
    assert local[0] == "ok"

    from fastapi.testclient import TestClient
    from main.runner_service import app
    client = TestClient(app)
    monkeypatch.setenv("MICROTUTOR_RUNNER_TOKEN", "secret")

    def post(url, data, headers, timeout):
        # The runner must execute locally, as it does in its own container.
        monkeypatch.delenv("MICROTUTOR_RUNNER_URL")
        try:
            return client.post("/run", content=data, headers=headers)
        finally:
            monkeypatch.setenv("MICROTUTOR_RUNNER_URL", "http://runner")

    monkeypatch.setattr(execution._runner_http, "post", post)
    monkeypatch.setenv("MICROTUTOR_RUNNER_URL", "http://runner")
    assert execution.run_student_code(GOOD, [[3]], entry_name="f") == local


def test_the_runner_refuses_a_wrong_token(monkeypatch):
    from fastapi.testclient import TestClient
    from main.runner_service import app
    monkeypatch.setenv("MICROTUTOR_RUNNER_TOKEN", "secret")
    answer = TestClient(app).post("/run", content="{}",
                                  headers={"X-Runner-Token": "nope"})
    assert answer.status_code == 401


def test_the_runner_refuses_everyone_without_a_token_configured(monkeypatch):
    from fastapi.testclient import TestClient
    from main.runner_service import app
    monkeypatch.delenv("MICROTUTOR_RUNNER_TOKEN", raising=False)
    answer = TestClient(app).post("/run", content="{}", headers={"X-Runner-Token": ""})
    assert answer.status_code == 503


def test_an_unreachable_runner_is_our_failure_not_the_students(monkeypatch):
    monkeypatch.setenv("MICROTUTOR_RUNNER_URL", "http://127.0.0.1:9")
    assert execution.run_student_code(GOOD, [[3]], entry_name="f")[0] == "harness_error"


def test_a_syntax_error_never_reaches_the_runner(monkeypatch):
    monkeypatch.setenv("MICROTUTOR_RUNNER_URL", "http://127.0.0.1:9")
    assert execution.run_student_code("def f(:", [[3]], entry_name="f")[0] == "syntax"


def test_overload_is_reported_as_our_failure(monkeypatch):
    monkeypatch.setenv("MICROTUTOR_MAX_RUNS", "1")
    monkeypatch.setattr(execution, "_RUN_QUEUE_WAIT", 0.01)
    with admission.slot("x", "MICROTUTOR_MAX_RUNS", 1):
        status, _, detail = execution.run_student_code(GOOD, [[3]], entry_name="f")
    assert status == "harness_error" and "capacity" in detail


def test_a_slot_is_released_after_an_error(monkeypatch):
    monkeypatch.setenv("MICROTUTOR_T_SLOT", "1")
    with pytest.raises(ValueError):
        with admission.slot("x", "MICROTUTOR_T_SLOT", 0.01):
            raise ValueError
    with admission.slot("x", "MICROTUTOR_T_SLOT", 0.01):
        pass


def test_without_the_new_settings_nothing_is_limited(monkeypatch):
    monkeypatch.delenv("MICROTUTOR_MAX_RUNS", raising=False)
    with admission.slot("x", "MICROTUTOR_MAX_RUNS", 0):
        with admission.slot("x", "MICROTUTOR_MAX_RUNS", 0):
            pass
    assert shared.client() is None and not pg.enabled()
