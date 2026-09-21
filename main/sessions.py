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

from .indent import base_indent

_DEFAULT_DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "grading_sessions.sqlite3")

SESSION_TTL_HOURS = 12
# UNLIMITED ATTEMPTS, and the reference is never revealed. It was 2, after
# which the answer was shown and the session marked "assisted".
#
# The instructors asked for this directly, and it follows the same principle as
# the rest of the system: a student who is handed the answer has learned that
# being stuck produces one. There is no deadline inside a problem, so the only
# thing a limit bought was ending the loop - and ending it by showing the answer
# ends the learning too. A stuck student now has the tutor, the failing case,
# and as many tries as they want.
#
# None means no limit. Kept as a name rather than deleted because the grading
# route and the page both read it, and a number here is how a future
# "reveal after N in an exam" mode would be turned back on.
MAX_ATTEMPTS = None

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
            revision          INTEGER NOT NULL DEFAULT 0,
            context_json      TEXT
        )""")
    # ONE json column, not five. A class-derived problem carries the module it
    # was carved out of (main/context.py); nothing here ever queries inside it,
    # it only has to come back out intact in problem_of(). CREATE TABLE above
    # does not touch a database that already exists, so the column is also added
    # explicitly - a session DB predating class support must keep working.
    try:
        conn.execute("ALTER TABLE sessions ADD COLUMN context_json TEXT")
    except sqlite3.OperationalError:
        pass                                    # already there
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
            " created_at, updated_at, expires_at, context_json)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (sid, student_id, problem.get("slug", ""), content_hash, did,
             problem.get("solution", ""), problem.get("description", ""),
             problem.get("title", ""), decomposition["header"],
             json.dumps(chunks), now, now, exp,
             json.dumps({k: problem[k] for k in CONTEXT_FIELDS if k in problem})))
    finally:
        conn.close()
    return {"session_id": sid, "decomposition_id": did,
            "header": decomposition["header"], "chunks": public_chunks(chunks),
            "total_chunks": len(chunks)}


def find_resumable(student_id: str | None, content_hash: str,
                   db_path: str | None = None) -> dict | None:
    """This student's live session for this problem, or None.

    WHY THIS EXISTS. Opening a problem used to issue a FRESH session every time,
    so a student who answered two of three steps, closed the tab and came back
    found an empty editor on step 1. The work was never lost - it is in the
    session store and in mt_submissions - but nothing looked for it, and the
    page could not replay it into a new session either, because answers a
    session never graded are not that session's to claim.

    So the fix is not to replay anything: it is to hand back the SAME session,
    still holding its own accepted prefix and its own index. Nothing is claimed
    because nothing moved.

    Keyed on content_hash, not slug, for the same reason every other cache here
    is: an edited problem is a different problem, and resuming into a session
    whose chunks were decomposed from the old text would put a student back to
    work on a question that no longer exists.

    Newest first, and only ACTIVE and unexpired - a completed problem starts
    over, which is what reopening one has always meant. Never raises: failing to
    find a session to resume must fall through to making a new one, not error."""
    if not student_id:
        return None
    try:
        conn = _connect(db_path)
    except Exception:
        return None
    try:
        r = conn.execute(
            "SELECT * FROM sessions WHERE student_id=? AND content_hash=?"
            " AND state='active' AND expires_at > ? ORDER BY created_at DESC"
            " LIMIT 1", (student_id, content_hash, _now())).fetchone()
    except Exception:
        return None
    finally:
        conn.close()
    return _row_to_session(r) if r is not None else None


def abandon_active(student_id: str | None, slug: str,
                   db_path: str | None = None) -> int:
    """Retire this student's live sessions for one problem. Returns how many.

    THE OTHER HALF OF RESUME. Once opening a problem resumes the session instead
    of issuing a new one, "Start this problem over" has to end that session or it
    restarts nothing that matters: the chat and the plan go back to empty and the
    student is handed their old accepted steps anyway, on step 3 of 3, with no
    way back to step 1. Before resume existed this route had nothing to do here,
    because the next open threw the session away by itself.

    Keyed on SLUG rather than content_hash, deliberately. A problem edited since
    the session was created hashes differently, and a student pressing restart
    means this problem, all of it, whichever version they started under.

    'abandoned' rather than a delete: mt_submissions still references these rows
    and an instructor's transcript is built from them, so a student who went
    round three times must stay visible as having gone round three times. Every
    reader here already refuses anything that is not 'active' (load_session) or
    not 'completed' (completed_answers), so the new state needs no migration and
    changes no existing query."""
    if not student_id or not slug:
        return 0
    try:
        conn = _connect(db_path)
    except Exception:
        return 0
    try:
        cur = conn.execute(
            "UPDATE sessions SET state='abandoned', updated_at=? WHERE"
            " student_id=? AND slug=? AND state='active'",
            (_now(), student_id, slug))
        return cur.rowcount or 0
    except Exception:
        return 0
    finally:
        conn.close()


def public_session(session: dict) -> dict:
    """The resumable view of a session: what create_session returns, plus where
    the student had got to.

    `accepted` is their OWN code and nothing else - the same text the page has
    been drawing in the frozen listing all along - so this adds no disclosure.
    References stay behind public_chunks, as ever."""
    return {"session_id": session["session_id"],
            "decomposition_id": session["decomposition_id"],
            "header": session["header"],
            "chunks": public_chunks(session["chunks"]),
            "total_chunks": len(session["chunks"]),
            "index": session["index"],
            "resumed": True,
            "accepted": [{"code": a.get("code", ""),
                          "how": ("revealed"
                                  if a.get("provenance") == "revealed_reference"
                                  else "own")}
                         for a in session.get("accepted") or []]}


def public_chunks(chunks: list[dict]) -> list[dict]:
    """Strip references. The ONLY shape that may cross to the browser.

    `indent` is the one number derived from a reference that DOES cross: the
    column this step's code sits at. The grader re-seats submissions there
    anyway (main/indent.py), so this is not load-bearing - it exists so the UI
    can show a step that continues inside a loop AS being inside that loop,
    instead of presenting an empty flat box and letting the student guess.
    It leaks nesting depth and nothing else: no names, no logic, no code.
    """
    return [{"step_id": c["step_id"], "prompt": c["prompt"],
             "expected_type": c.get("expected_type", "code"),
             "indent": base_indent(c.get("reference") or "")} for c in chunks]


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
            "context": json.loads(r["context_json"] or "{}"),
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


# What a class-derived problem carries beyond the plain four. Named once here
# because create_session writes it and problem_of reads it, and the two drifting
# apart would mean a method graded against a program it was not decomposed from.
def context_of(row: dict) -> dict:
    """The class-context fields from a stored problem row, flattened back onto
    the problem dict main/context.py expects.

    Reading a method problem back WITHOUT these is silently catastrophic rather
    than merely lossy: nothing errors, the problem just stops being a method and
    starts being an unrunnable bare function with a different content_hash. Any
    code that loads a problem from the database and then executes it, or looks
    it up by hash, must go through here - main/grades.py did neither, which is
    why a grade sheet could not find the decomposition its own students had
    already been served.

    Lives here beside CONTEXT_FIELDS so there is one implementation;
    frontend/api_server._context_of delegates to it."""
    context = row.get("context") or {}
    if isinstance(context, str):                  # jsonb can come back as text
        try:
            context = json.loads(context)
        except Exception:
            context = {}
    out = {k: context[k] for k in CONTEXT_FIELDS if k in context}
    # The group columns live beside the blob, not inside it, and group_title is
    # what names the class in the sequence driver.
    for k in ("group_slug", "group_title", "group_description"):
        if row.get(k) is not None:
            out[k] = row[k]
    return out


CONTEXT_FIELDS = ("context_prefix", "context_suffix", "context_indent",
                  "entry_hint", "group_slug", "group_title", "group_description",
                  # The top of a FLAT file - imports and module constants, which
                  # belong to no problem and so were stored nowhere. Rides in the
                  # same jsonb blob (no migration), and is deliberately NOT
                  # context_prefix: that field is what makes is_method() true.
                  "module_preamble")


def problem_of(session: dict) -> dict:
    """Rebuild the authoritative problem dict from the session - never the
    client's copy.

    The context fields ride along so main/context.py can assemble the same
    program the oracle was built from. Without them a method problem would grade
    as a bare function and every submission would fail on the class that is not
    there."""
    return {"slug": session["slug"], "title": session["title"],
            "description": session["description"], "solution": session["solution"],
            **(session.get("context") or {})}


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


def stored_result(session_id: str, submission_id: str,
                  db_path: str | None = None) -> dict | None:
    """The result already recorded for this submission, or None.

    READ-ONLY, and it reserves nothing - begin_submission() is still what
    claims a row for a submission that has never been graded. This exists
    because the answer to "have I already graded this?" has to be available
    BEFORE the rules written for new submissions run.

    Those rules are load_session() refusing a completed session and the
    stale-index check, both correct for new work and both wrong for a replay:
    the first successful grade is what completed the session and moved the
    index, so a student whose browser lost that response got 409
    session_completed or 409 stale_index on every retry, with their own passing
    verdict sitting in this table unreachable. An answer that was graded once
    must be recoverable by asking for it again with the same id.

    Not an authorisation boundary - the caller checks ownership first."""
    try:
        conn = _connect(db_path)
    except Exception:
        return None
    try:
        r = conn.execute("SELECT result_json FROM submissions WHERE"
                         " session_id=? AND submission_id=?",
                         (session_id, submission_id)).fetchone()
        if r is None or not r["result_json"]:
            return None          # never seen, or claimed and still in flight
        return {**json.loads(r["result_json"]), "idempotent_replay": True}
    except Exception:
        # A replay that cannot be read is not an error the student can act on;
        # fall through and let the ordinary path speak.
        return None
    finally:
        try: conn.close()
        except Exception: pass


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
                   consume_attempt: bool = True, covers_chunks: int = 1,
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
            # ONE submission may answer more than one step. A student who wrote
            # step 2's work inside step 1 has already done it, and asking them
            # for it again is asking them to write the same lines twice - so
            # main/bridge.find reports how far their code actually reaches and
            # the session advances that far. The code is recorded against the
            # FIRST step and the steps it also covers are recorded as empty, so
            # the accepted prefix still reassembles to exactly what they wrote:
            # repeating the text once per step would duplicate their loop.
            first = idx
            for step in range(first, min(first + max(1, covers_chunks),
                                         len(s["chunks"]))):
                accepted.append({"step_id": s["chunks"][step]["step_id"],
                                 "code": accept_code if step == first else "",
                                 "provenance": provenance})
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


def reopen_step(session_id: str, index: int,
                db_path: str | None = None) -> dict:
    """Put a student back on a step of their own that was already accepted.

    WHY THIS EXISTS. A step was accepted, the student then found a bug in it,
    and there was no way back to it: the only route to changing an earlier
    answer was Start over, which gives up the whole problem. "Accepted, and
    therefore unreachable" is the one state a student cannot work around from
    the page, and it pushes them into a restart they did not want.

    WHAT IT DROPS, AND WHY IT HAS TO. The steps AFTER `index` leave the accepted
    prefix, because that prefix is exactly what they were graded against:
    main/grading.py stitches the student's accepted code onto the teacher's
    remaining reference and runs the result, so an acceptance that rested on
    code which has since changed has stopped being evidence about anything.
    Keeping them would mean a later step certified against a step that no
    longer exists in that form.

    NOTHING IS LOST. No submission row is touched, so the instructor's
    transcript still holds every version (main/archive.py), and the dropped
    code is RETURNED so the page can hand it straight back as a draft - a
    student reworking step 1 should not have to retype step 2 from memory.

    CREDIT ALREADY EARNED IS NOT WITHDRAWN. main/grades.tally reads a step as
    solved when ANY submission for it came back correct, so reworking a step
    adds attempts to the record and never subtracts a grade. Same rule the rest
    of the archive follows: a student who went round twice is the finding.

    Refuses anything but an ACTIVE session and an index genuinely behind the
    current one. A finished problem is Start over's business - un-completing a
    session would have to unpick the solved flag and the reflection stage with
    it - and "reopening" the step already open would drop nothing and reset an
    attempt count for free."""
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        r = conn.execute("SELECT * FROM sessions WHERE session_id=?",
                         (session_id,)).fetchone()
        if r is None:
            conn.execute("ROLLBACK")
            raise SessionError("Unknown session.", "session_not_found")
        s = _row_to_session(r)
        if s["state"] != "active":
            conn.execute("ROLLBACK")
            raise SessionError("This problem is already finished - use Start "
                               "over to work it again.", "session_inactive")
        if not 0 <= index < s["index"]:
            conn.execute("ROLLBACK")
            raise SessionError("That step is not one you have already "
                               "finished.", "step_not_reopenable")
        dropped = [{"index": index + n, "code": a.get("code") or ""}
                   for n, a in enumerate(s["accepted"][index:])]
        # attempts belong to the step being worked, and this is a different one.
        conn.execute(
            "UPDATE sessions SET idx=?, accepted_json=?, attempts=0,"
            " updated_at=?, revision=revision+1 WHERE session_id=?",
            (index, json.dumps(list(s["accepted"])[:index]), _now(), session_id))
        conn.execute("COMMIT")
    except SessionError:
        raise
    except Exception:
        try: conn.execute("ROLLBACK")
        except Exception: pass
        raise
    finally:
        conn.close()
    return {"index": index, "attempts": 0, "assisted": bool(s["assisted"]),
            "completed": False, "total_chunks": len(s["chunks"]),
            "dropped": dropped}


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


def accepted_so_far(student_id: str, slugs: list[str],
                    db_path: str | None = None) -> dict:
    """What this student has had accepted on problems they have NOT finished.

    {slug: {"code": <body at column 0>, "assisted": bool}} for every ACTIVE
    session holding at least one accepted step. completed_answers() is the
    companion and deliberately refuses these: a partial body is not something to
    hand back in a file that has to run.

    For the on-screen copy the calculation is different, though. A student two
    steps into a three-step method opens the file to see what they have built,
    and being shown a `# YOUR CODE STARTS HERE` stub over their own two accepted
    lines reads as the work having been lost - the same wrong signal that
    reopening a problem used to give before sessions resumed. The caller
    compiles the result and falls back to the stub if the half-written body will
    not parse, so this stays a view and never becomes a broken download.

    Never raises: a file that shows a little less is still a useful file."""
    if not student_id or not slugs:
        return {}
    try:
        conn = _connect(db_path)
    except Exception:
        return {}
    try:
        rows = conn.execute(
            "SELECT slug, accepted_json, assisted FROM sessions WHERE"
            " student_id=? AND state='active' AND expires_at > ?"
            " ORDER BY updated_at ASC", (student_id, _now())).fetchall()
    except Exception:
        return {}
    finally:
        conn.close()

    want, out = set(slugs), {}
    for r in rows:                       # ascending, so the newest wins
        if r["slug"] not in want:
            continue
        try:
            accepted = json.loads(r["accepted_json"]) or []
        except Exception:
            continue
        code = "\n".join(a.get("code", "") for a in accepted if a.get("code"))
        if not code.strip():
            continue
        out[r["slug"]] = {
            "code": code,
            "assisted": bool(r["assisted"]) or any(
                a.get("provenance") == "revealed_reference" for a in accepted),
        }
    return out


def completed_answers(student_id: str, slugs: list[str],
                      db_path: str | None = None) -> dict:
    """What this student actually got accepted, per problem.

    {slug: {"code": <body at column 0>, "assisted": bool}} for every COMPLETED
    session they own among `slugs`. Newest completed session per slug wins - a
    student who works a problem twice gets the run they finished last.

    Reads the accepted chunks rather than their submissions, because those are
    two different claims: `accepted_json` is what the session actually built the
    solution out of, including a step that was answered by revealing the
    reference, and that is exactly what belongs in a file that has to RUN. The
    `assisted` flag travels with it so the caller can say so out loud rather
    than passing a shown answer off as the student's own.

    Never raises: a handback that cannot read one session must still hand back
    the rest of the assignment."""
    if not student_id or not slugs:
        return {}
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT slug, accepted_json, assisted, updated_at FROM sessions"
            " WHERE student_id=? AND state='completed'"
            " ORDER BY updated_at ASC", (student_id,)).fetchall()
    except Exception:
        return {}
    finally:
        conn.close()

    want, out = set(slugs), {}
    for r in rows:                       # ascending, so the last write wins
        if r["slug"] not in want:
            continue
        try:
            accepted = json.loads(r["accepted_json"]) or []
        except Exception:
            continue
        code = "\n".join(a.get("code", "") for a in accepted if a.get("code"))
        if not code.strip():
            continue
        out[r["slug"]] = {
            "code": code,
            # The session-level flag is the durable one, but a per-chunk
            # provenance is more precise when it is there.
            "assisted": bool(r["assisted"]) or any(
                a.get("provenance") == "revealed_reference" for a in accepted),
        }
    return out


if __name__ == "__main__":
    # Self-check for RESUME, against a real SQLite store in a temp directory.
    #   python -m main.sessions
    import tempfile
    from types import SimpleNamespace as N

    db = os.path.join(tempfile.mkdtemp(), "sessions.sqlite3")
    prob = {"slug": "invert", "title": "Invert", "description": "d",
            "solution": "def invert(d):\n    return {}"}
    decomp = {"header": "def invert(d):",
              "chunks": [N(step_id="Part 1", prompt="count the values",
                           expected_type="code", reference="counts = {}"),
                         N(step_id="Part 2", prompt="return the inversion",
                           expected_type="code", reference="return {}")]}
    HASH, WHO = "hash-invert", "student-1"

    opened = create_session(prob, decomp, HASH, student_id=WHO, db_path=db)
    sid = opened["session_id"]

    # Nothing answered yet: resumable, and it resumes AT THE START.
    live = find_resumable(WHO, HASH, db_path=db)
    assert live is not None and live["session_id"] == sid
    assert public_session(live)["index"] == 0
    assert public_session(live)["accepted"] == []
    assert public_session(live)["resumed"] is True

    # ...and it must never leak a reference, resumed or not.
    assert all("reference" not in c for c in public_session(live)["chunks"])

    # Answer step 1, the way /grade_chunk does.
    done, row = begin_submission(sid, "sub-1", db_path=db)
    assert done is None
    commit_outcome(sid, "sub-1", row["revision"], {"verdict": "correct"},
                   accept_code="counts = {}", db_path=db)

    # THE WHOLE POINT: reopening the problem finds that session, on step 2, with
    # the student's own line already in it. This is what used to be thrown away.
    back = public_session(find_resumable(WHO, HASH, db_path=db))
    assert back["session_id"] == sid, "a NEW session would claim ungraded steps"
    assert back["index"] == 1, back
    assert back["accepted"] == [{"code": "counts = {}", "how": "own"}], back

    # Someone else's session, and another problem's, are not this one.
    assert find_resumable("student-2", HASH, db_path=db) is None
    assert find_resumable(WHO, "hash-something-else", db_path=db) is None
    # An anonymous caller has nothing to resume and must not be handed a session.
    assert find_resumable(None, HASH, db_path=db) is None

    # A revealed answer is labelled as one when it comes back, so the page
    # cannot redraw it as the student's own work.
    done, row = begin_submission(sid, "sub-2", db_path=db)
    commit_outcome(sid, "sub-2", row["revision"], {"verdict": "correct"},
                   accept_code="return {}", provenance="revealed_reference",
                   db_path=db)

    # That was the last chunk, so the session COMPLETED - and a finished problem
    # starts over, which is what reopening one has always meant.
    assert find_resumable(WHO, HASH, db_path=db) is None, \
        "a completed problem must not resume into its own finished session"
    finished = session_snapshot(sid, db_path=db)
    assert finished["state"] == "completed"
    assert public_session(finished)["accepted"][1]["how"] == "revealed"

    # An EXPIRED session is not resumable either - the reference it was
    # decomposed from may have moved on since.
    stale = create_session(prob, decomp, HASH, student_id=WHO, db_path=db)
    conn = _connect(db)
    try:
        conn.execute("UPDATE sessions SET expires_at=? WHERE session_id=?",
                     ("2000-01-01T00:00:00+00:00", stale["session_id"]))
    finally:
        conn.close()
    assert find_resumable(WHO, HASH, db_path=db) is None

    # ── restart really restarts ───────────────────────────────────────────
    # Resume made this route load-bearing: without it "start over" empties the
    # chat and then hands the student their old steps back.
    fresh = create_session(prob, decomp, HASH, student_id=WHO, db_path=db)
    assert find_resumable(WHO, HASH, db_path=db)["session_id"] == fresh["session_id"]
    # Two rows go: the live one and the expired-but-still-'active' one above.
    # Retiring a session that has aged out is harmless and keeps the table
    # honest about what is still in play.
    assert abandon_active(WHO, "invert", db_path=db) == 2
    assert find_resumable(WHO, HASH, db_path=db) is None, \
        "start over left the old session resumable"
    # The row survives - an instructor's transcript is built from these.
    assert session_snapshot(fresh["session_id"], db_path=db)["state"] == "abandoned"
    # Nobody else's work is touched, and a second press is a no-op.
    assert abandon_active(WHO, "invert", db_path=db) == 0
    assert abandon_active("student-2", "invert", db_path=db) == 0
    assert abandon_active(None, "invert", db_path=db) == 0

    # ── a graded submission stays recoverable ─────────────────────────────
    # The response can be lost on the way back - a dropped connection, a closed
    # laptop. Retrying with the SAME id has to return the verdict that was
    # already reached, INCLUDING after the final chunk completed the session,
    # because completing it is exactly what made the ordinary route refuse.
    replayable = create_session(prob, decomp, "hash-replay", student_id=WHO, db_path=db)["session_id"]
    _prior, _s = begin_submission(replayable, "sub-A", db_path=db)
    assert _prior is None
    commit_outcome(replayable, "sub-A", _s["revision"],
                   {"verdict": "correct", "tier": "execution-reference",
                    "deterministic": True, "reason": "ok", "divergent": False},
                   accept_code="counts = {}", provenance="student",
                   consume_attempt=True, db_path=db)
    _replay = stored_result(replayable, "sub-A", db_path=db)
    assert _replay and _replay["verdict"] == "correct", _replay
    assert _replay["idempotent_replay"] is True, "a replay must say so"
    # Never seen, and claimed-but-not-yet-graded, are both "no result" - a
    # reservation must not read as a verdict or the twin would be answered
    # with a result nobody produced.
    assert stored_result(replayable, "sub-never", db_path=db) is None
    begin_submission(replayable, "sub-B", db_path=db)
    assert stored_result(replayable, "sub-B", db_path=db) is None

    print("sessions.py resume self-check OK")
