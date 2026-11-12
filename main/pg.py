"""pg.py - Postgres for the stores that used to live on one machine's disk.

Set DATABASE_URL and grading sessions (main/sessions.py) plus the oracle cache,
chunk pool and transcripts (main/docstore.py) live here, so any number of app
processes on any number of machines see the same state. Unset, nothing imports
psycopg and the app runs on SQLite and JSON files as before.

connect() returns an object shaped like the sqlite3 connection sessions.py was
written against, so that module's transactions are the same code on both
stores. Point DATABASE_URL at a transaction-mode pooler in production: prepared
statements are off for that reason, and no connection is held during grading.
"""
import atexit
import os
import re
import sqlite3
import threading

_lock = threading.Lock()
_pool = None
_schema_ready = False
_SCHEMA_LOCK_ID = 4961_0001

_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS sessions (
        seq                BIGSERIAL,
        session_id         TEXT PRIMARY KEY,
        student_id         TEXT,
        slug               TEXT,
        content_hash       TEXT NOT NULL,
        decomposition_id   TEXT NOT NULL,
        solution           TEXT NOT NULL,
        description        TEXT,
        title              TEXT,
        header             TEXT NOT NULL,
        chunks_json        TEXT NOT NULL,
        idx                INTEGER NOT NULL DEFAULT 0,
        accepted_json      TEXT NOT NULL DEFAULT '[]',
        attempts           INTEGER NOT NULL DEFAULT 0,
        assisted           INTEGER NOT NULL DEFAULT 0,
        state              TEXT NOT NULL DEFAULT 'active',
        created_at         TEXT NOT NULL,
        updated_at         TEXT NOT NULL,
        expires_at         TEXT NOT NULL,
        last_submission_id TEXT,
        last_result_json   TEXT,
        revision           INTEGER NOT NULL DEFAULT 0,
        context_json       TEXT,
        retired            INTEGER NOT NULL DEFAULT 0
    )""",
    """CREATE TABLE IF NOT EXISTS submissions (
        session_id    TEXT NOT NULL,
        submission_id TEXT NOT NULL,
        result_json   TEXT,
        created_at    TEXT NOT NULL,
        PRIMARY KEY (session_id, submission_id)
    )""",
    "CREATE INDEX IF NOT EXISTS sessions_student_hash"
    " ON sessions (student_id, content_hash)",
    "CREATE INDEX IF NOT EXISTS sessions_student_slug"
    " ON sessions (student_id, slug, state)",
    """CREATE TABLE IF NOT EXISTS mt_docs (
        ns         TEXT NOT NULL,
        key        TEXT NOT NULL,
        value      TEXT NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        PRIMARY KEY (ns, key)
    )""",
)


def enabled() -> bool:
    return bool(os.environ.get("DATABASE_URL", "").strip())


def _get_pool():
    global _pool
    if _pool is None:
        with _lock:
            if _pool is None:
                from psycopg.rows import dict_row
                from psycopg_pool import ConnectionPool
                _pool = ConnectionPool(
                    os.environ["DATABASE_URL"].strip(),
                    min_size=int(os.environ.get("DB_POOL_MIN", 1)),
                    max_size=int(os.environ.get("DB_POOL_MAX", 10)),
                    timeout=10, open=True,
                    kwargs={"autocommit": True, "row_factory": dict_row,
                            "prepare_threshold": None})
                atexit.register(_pool.close)
    return _pool


def ensure_schema() -> None:
    """Create what is missing, once per process. The advisory lock stops two
    workers booting together from racing on CREATE ... IF NOT EXISTS, which
    Postgres does not make safe by itself."""
    global _schema_ready
    if _schema_ready:
        return
    with _get_pool().connection() as raw:
        with raw.transaction():
            raw.execute("SELECT pg_advisory_xact_lock(%s)", (_SCHEMA_LOCK_ID,))
            for statement in _SCHEMA:
                raw.execute(statement)
    _schema_ready = True


def _adapt(value):
    """The two things SQLite accepts in a parameter and Postgres refuses."""
    if isinstance(value, bool):         # the integer flag columns
        return int(value)
    if isinstance(value, str):          # ids come from the client; a NUL in
        return value.replace("\x00", "")  # one must be "not found", not a 500
    return value


class _Result:
    """Rows are read before the cursor closes, so a result outlives it."""

    def __init__(self, rows: list, rowcount: int):
        self._rows, self.rowcount = rows, rowcount

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows


class PgConnection:
    """The part of sqlite3.Connection that sessions.py uses."""

    def __init__(self, raw, pool):
        self._raw, self._pool = raw, pool
        self._lock_next_session = False

    def execute(self, sql: str, params: tuple = ()) -> _Result:
        import psycopg
        if sql.strip().upper() == "BEGIN IMMEDIATE":
            # SQLite's BEGIN IMMEDIATE takes the whole database's write lock
            # before the first read. Postgres's equivalent for these
            # read-check-write transactions is locking the one session row
            # they read first, so only twins of one session ever wait.
            self._lock_next_session = True
            sql = "BEGIN"
        elif self._lock_next_session and re.match(
                r"\s*SELECT\b.*\bFROM sessions\b", sql, re.I | re.S):
            self._lock_next_session = False
            sql += " FOR UPDATE"
        translated = re.sub(r"\browid\b", "seq", sql.replace("?", "%s"))
        params = tuple(_adapt(p) for p in params)
        try:
            with self._raw.cursor() as cur:
                cur.execute(translated, params or None)
                rows = cur.fetchall() if cur.description else []
                return _Result(rows, cur.rowcount)
        except psycopg.errors.UniqueViolation as e:
            raise sqlite3.IntegrityError(str(e)) from e

    def close(self) -> None:
        """Back to the pool, never mid-transaction. Safe to call twice:
        begin_submission's conflict path does."""
        raw, self._raw = self._raw, None
        if raw is None:
            return
        import psycopg
        try:
            if raw.info.transaction_status != psycopg.pq.TransactionStatus.IDLE:
                raw.rollback()
        finally:
            self._pool.putconn(raw)


def connect() -> PgConnection:
    ensure_schema()
    pool = _get_pool()
    return PgConnection(pool.getconn(), pool)
