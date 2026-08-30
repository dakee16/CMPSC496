"""
sessions.py - server-owned grading sessions (SQLite, stdlib only).

The browser used to be authoritative: /decompose_chunks handed it every hidden
reference, and /grade_chunk accepted the problem, solution, chunks, accepted
prefix and chunk index back from it. All of that was editable by the student.

Everything authoritative now lives here, server-side, keyed by an opaque random
session id. The client learns only a session id and the public part of each
chunk (step_id, prompt, expected_type).
"""
import json
import os
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone

_DEFAULT_DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "grading_sessions.sqlite3")

SESSION_TTL_HOURS = 12
MAX_ATTEMPTS = 2          # second failure reveals the reference

# How long a submission may sit RESERVED (claimed, no result) before another
# attempt with the same id may reclaim it. Must exceed the slowest realistic
# grade - Tier 3/4 can run several subprocess executions plus model calls
# so a genuine concurrent twin is never mistaken for a dead one.
SUBMISSION_GRACE_SECONDS = 180


class SessionError(RuntimeError):
    """Session could not be used. `reason_code` distinguishes the cases so the
    API can map them to 404/409 rather than a generic 500."""

    def __init__(self, message: str, reason_code: str):
        super().__init__(message)
        self.reason_code = reason_code


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _is_stale(created_at: str) -> bool:
    """True if a reservation is old enough that its owner is presumed dead.

    An unparseable timestamp is treated as NOT stale: reclaiming on a parse
    failure could double-grade a live submission, which is worse than making
    the student wait."""
    try:
        age = (datetime.now(timezone.utc)
               - datetime.fromisoformat(created_at)).total_seconds()
    except Exception:
        return False
    return age > SUBMISSION_GRACE_SECONDS


def _connect(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or os.environ.get("MICROTUTOR_SESSION_DB") or _DEFAULT_DB
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, timeout=15, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=15000")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id        TEXT PRIMARY KEY,
            student_id        TEXT,
            slug              TEXT,
            content_hash      TEXT NOT NULL,
            decomposition_id  TEXT NOT NULL,
            solution          TEXT NOT NULL,
            description       TEXT,
            title             TEXT,
            header            TEXT NOT NULL,
            chunks_json       TEXT NOT NULL,   -- PRIVATE: includes references
            idx               INTEGER NOT NULL DEFAULT 0,
            accepted_json     TEXT NOT NULL DEFAULT '[]',
            attempts          INTEGER NOT NULL DEFAULT 0,
            assisted          INTEGER NOT NULL DEFAULT 0,
            state             TEXT NOT NULL DEFAULT 'active',
            created_at        TEXT NOT NULL,
            updated_at        TEXT NOT NULL,
            expires_at        TEXT NOT NULL,
            last_submission_id TEXT,
            last_result_json  TEXT,
            revision          INTEGER NOT NULL DEFAULT 0
        )""")
    # Idempotency is persisted per (session, submission) rather than only
    # remembering the latest submission - a retry of an older id must still
    # replay its own stored result instead of being graded again.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS submissions (
            session_id    TEXT NOT NULL,
            submission_id TEXT NOT NULL,
            result_json   TEXT,
            created_at    TEXT NOT NULL,
            PRIMARY KEY (session_id, submission_id)
        )""")
    return conn


def create_session(problem: dict, decomposition: dict, content_hash: str,
                   student_id: str | None = None,
                   db_path: str | None = None) -> dict:
    """Register a gated decomposition. Returns the PUBLIC view only.

    student_id is bound HERE, once, and is never accepted again per-submission:
    identity must not be changeable mid-session."""
    sid = secrets.token_urlsafe(32)
    did = secrets.token_urlsafe(12)
    chunks = [{"step_id": c.step_id, "prompt": c.prompt,
               "expected_type": c.expected_type, "reference": c.reference or ""}
              for c in decomposition["chunks"]]
    now = _now()
    exp = (datetime.now(timezone.utc) + timedelta(hours=SESSION_TTL_HOURS)
           ).isoformat(timespec="seconds")
    conn = _connect(db_path)
    try:
        conn.execute(
            "INSERT INTO sessions (session_id, student_id, slug, content_hash,"
            " decomposition_id, solution, description, title, header, chunks_json,"
            " created_at, updated_at, expires_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (sid, student_id, problem.get("slug", ""), content_hash, did,
             problem.get("solution", ""), problem.get("description", ""),
             problem.get("title", ""), decomposition["header"],
             json.dumps(chunks), now, now, exp))
    finally:
        conn.close()
    return {"session_id": sid, "decomposition_id": did,
            "header": decomposition["header"], "chunks": public_chunks(chunks),
            "total_chunks": len(chunks)}


def public_chunks(chunks: list[dict]) -> list[dict]:
    """Strip references. The ONLY shape that may cross to the browser."""
    return [{"step_id": c["step_id"], "prompt": c["prompt"],
             "expected_type": c.get("expected_type", "code")} for c in chunks]


def _row_to_session(r: sqlite3.Row) -> dict:
    return {"session_id": r["session_id"], "student_id": r["student_id"],
            "slug": r["slug"], "content_hash": r["content_hash"],
            "decomposition_id": r["decomposition_id"], "solution": r["solution"],
            "description": r["description"], "title": r["title"],
            "header": r["header"], "chunks": json.loads(r["chunks_json"]),
            "index": r["idx"], "accepted": json.loads(r["accepted_json"]),
            "attempts": r["attempts"], "assisted": bool(r["assisted"]),
            "state": r["state"], "expires_at": r["expires_at"],
            "last_submission_id": r["last_submission_id"],
            "revision": r["revision"],
            "last_result": json.loads(r["last_result_json"]) if r["last_result_json"] else None}


def load_session(session_id: str, db_path: str | None = None) -> dict:
    conn = _connect(db_path)
    try:
        r = conn.execute("SELECT * FROM sessions WHERE session_id=?",
                         (session_id,)).fetchone()
    finally:
        conn.close()
    if r is None:
        raise SessionError("Unknown session.", "session_not_found")
    s = _row_to_session(r)
    if s["state"] == "completed":
        raise SessionError("This session is already complete.", "session_completed")
    if s["expires_at"] < _now():
        raise SessionError("This session has expired.", "session_expired")
    if s["state"] != "active":
        raise SessionError("This session is no longer active.", "session_inactive")
    return s


def problem_of(session: dict) -> dict:
    """Rebuild the authoritative problem dict from the session - never the
    client's copy."""
    return {"slug": session["slug"], "title": session["title"],
            "description": session["description"], "solution": session["solution"]}


def accepted_prefix(session: dict) -> list[str]:
    return [a["code"] for a in session["accepted"]]


def stored_result(session_id: str, submission_id: str,
                  db_path: str | None = None) -> dict | None:
    """The recorded outcome for this exact submission, if it already ran."""
    conn = _connect(db_path)
    try:
        r = conn.execute("SELECT result_json FROM submissions WHERE session_id=?"
                         " AND submission_id=?", (session_id, submission_id)).fetchone()
    finally:
        conn.close()
    if r and r["result_json"]:
        return {**json.loads(r["result_json"]), "idempotent_replay": True}
    return None


def begin_submission(session_id: str, submission_id: str,
                     db_path: str | None = None) -> tuple[dict | None, dict]:
    """Reserve a submission. Returns (stored_result_or_None, session).

    Deliberately SHORT: it claims the (session, submission) row and reads the
    current revision, then returns. Grading - which runs subprocesses and may
    call an LLM - happens with NO write transaction held, so a slow grade never
    blocks another request. commit_outcome() then does a compare-and-swap."""
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        r = conn.execute("SELECT * FROM sessions WHERE session_id=?",
                         (session_id,)).fetchone()
        if r is None:
            conn.execute("ROLLBACK")
            raise SessionError("Unknown session.", "session_not_found")
        prior = conn.execute("SELECT result_json, created_at FROM submissions"
                             " WHERE session_id=? AND submission_id=?",
                             (session_id, submission_id)).fetchone()
        if prior is not None:
            if prior["result_json"]:
                conn.execute("COMMIT")
                return ({**json.loads(prior["result_json"]), "idempotent_replay": True},
                        _row_to_session(r))
            # Row claimed but no result yet. Two very different situations:
            #
            #   * a concurrent twin of this exact submission is still grading
            #     returning None would grade it a SECOND time and double-advance;
            #   * the attempt that claimed it DIED (grader error, CAS conflict,
            #     killed process) without writing a result or releasing the row.
            #
            # Without the staleness check that second reservation is PERMANENT:
            # every retry of this submission id 409s "still being graded"
            # forever, so the student can neither advance nor retry. Age is what
            # separates the two. release_submission() makes the recoverable
            # failure paths instant; this is the backstop for the ones that
            # cannot run cleanup at all.
            if _is_stale(prior["created_at"]):
                conn.execute("UPDATE submissions SET created_at=? WHERE session_id=?"
                             " AND submission_id=?", (_now(), session_id, submission_id))
                conn.execute("COMMIT")
                return None, _row_to_session(r)
            conn.execute("COMMIT")
            return {"__in_flight__": True}, _row_to_session(r)
        conn.execute("INSERT INTO submissions (session_id, submission_id, result_json,"
                     " created_at) VALUES (?,?,NULL,?)",
                     (session_id, submission_id, _now()))
        conn.execute("COMMIT")
        return None, _row_to_session(r)
    except sqlite3.IntegrityError:
        # Another thread claimed the same submission id first.
        try: conn.execute("ROLLBACK")
        except Exception: pass
        conn.close()
        return {"__in_flight__": True}, load_session(session_id, db_path)
    except SessionError:
        raise
    finally:
        try: conn.close()
        except Exception: pass


def release_submission(session_id: str, submission_id: str,
                       db_path: str | None = None) -> None:
    """Give back a reservation that never produced a result.

    Called when grading failed or the commit lost its compare-and-swap, so the
    student can retry the SAME submission id immediately instead of waiting out
    SUBMISSION_GRACE_SECONDS. Only ever deletes an UNFINISHED row - a graded
    result must survive, or idempotency would be lost and a retry would grade
    and advance the session twice. Never raises: this runs on a failure path
    and must not replace the original error."""
    try:
        conn = _connect(db_path)
    except Exception:
        return
    try:
        conn.execute("DELETE FROM submissions WHERE session_id=? AND submission_id=?"
                     " AND result_json IS NULL", (session_id, submission_id))
    except Exception:
        pass
    finally:
        try: conn.close()
        except Exception: pass


def commit_outcome(session_id: str, submission_id: str, revision: int, result: dict,
                   *, accept_code: str | None = None, provenance: str = "student",
                   consume_attempt: bool = True,
                   db_path: str | None = None) -> dict:
    """Commit a graded submission with compare-and-swap on `revision`.

    If another request advanced the session while we were grading, the CAS
    fails and this raises a typed conflict rather than double-advancing."""
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        r = conn.execute("SELECT * FROM sessions WHERE session_id=?",
                         (session_id,)).fetchone()
        if r is None:
            conn.execute("ROLLBACK")
            raise SessionError("Unknown session.", "session_not_found")
        s = _row_to_session(r)
        if s["revision"] != revision:
            conn.execute("ROLLBACK")
            raise SessionError("This session moved on while your answer was being "
                               "graded.", "session_conflict")

        accepted, idx = list(s["accepted"]), s["index"]
        attempts, assisted, state = s["attempts"], s["assisted"], s["state"]
        if accept_code is not None:
            accepted.append({"step_id": s["chunks"][idx]["step_id"],
                             "code": accept_code, "provenance": provenance})
            idx += 1
            attempts = 0
            if provenance == "revealed_reference":
                assisted = 1
            if idx >= len(s["chunks"]):
                state = "completed"
        elif consume_attempt:
            attempts += 1

        payload = {**result, "index": idx, "attempts": attempts,
                   "assisted": bool(assisted), "state": state,
                   "total_chunks": len(s["chunks"]),
                   "completed": state == "completed",
                   "solved_independently": state == "completed" and not assisted}
        conn.execute(
            "UPDATE sessions SET idx=?, accepted_json=?, attempts=?, assisted=?,"
            " state=?, updated_at=?, last_submission_id=?, last_result_json=?,"
            " revision=revision+1 WHERE session_id=? AND revision=?",
            (idx, json.dumps(accepted), attempts, assisted, state, _now(),
             submission_id, json.dumps(payload), session_id, revision))
        conn.execute("UPDATE submissions SET result_json=? WHERE session_id=?"
                     " AND submission_id=?",
                     (json.dumps(payload), session_id, submission_id))
        conn.execute("COMMIT")
        return payload
    except SessionError:
        raise
    except Exception:
        try: conn.execute("ROLLBACK")
        except Exception: pass
        raise
    finally:
        conn.close()


def apply_outcome(session_id: str, submission_id: str, result: dict, **kw) -> dict:
    """Back-compat shim: reserve then commit in one call."""
    db_path = kw.pop("db_path", None)
    done, s = begin_submission(session_id, submission_id, db_path)
    if done is not None:
        return done
    return commit_outcome(session_id, submission_id, s["revision"], result,
                          db_path=db_path, **kw)


def session_snapshot(session_id: str, db_path: str | None = None) -> dict | None:
    """Read a session in ANY state, including completed/expired.

    load_session() deliberately refuses those, but solved-status needs to
    inspect exactly a completed one."""
    conn = _connect(db_path)
    try:
        r = conn.execute("SELECT * FROM sessions WHERE session_id=?",
                         (session_id,)).fetchone()
    finally:
        conn.close()
    return _row_to_session(r) if r is not None else None
