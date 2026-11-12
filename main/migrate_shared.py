"""migrate_shared.py - copy single-box state into Postgres, once, at cutover.

    DATABASE_URL=... python -m main.migrate_shared [--sqlite PATH]

Reads the JSON files and SQLite database named by the usual environment
variables. Existing Postgres rows are kept: re-running it fills gaps and never
overwrites anything newer. Stop the old app first, or sessions written after
the copy are left behind.
"""
import json
import os
import sys

from . import pg
from .sessions import _connect as _connect_sqlite

_SESSION_COLUMNS = (
    "session_id", "student_id", "slug", "content_hash", "decomposition_id",
    "solution", "description", "title", "header", "chunks_json", "idx",
    "accepted_json", "attempts", "assisted", "state", "created_at",
    "updated_at", "expires_at", "last_submission_id", "last_result_json",
    "revision", "context_json", "retired")
_SUBMISSION_COLUMNS = ("session_id", "submission_id", "result_json", "created_at")

_DOCUMENTS = {
    "oracle": ("MICROTUTOR_ORACLE_CACHE", "data/oracles/tests_cache.json"),
    "pool": ("MICROTUTOR_CHUNK_POOL", "main/chunk_pool.json"),
    "transcripts": ("MICROTUTOR_TRANSCRIPTS", "data/transcripts.json"),
}


def _document_paths() -> dict:
    paths = {ns: os.environ.get(env) or fallback
             for ns, (env, fallback) in _DOCUMENTS.items()}
    # No env var of its own: review_help.py keeps it beside the oracle cache.
    paths["review_help"] = os.path.join(
        os.path.dirname(paths["oracle"]) or ".", "review_help.json")
    return paths


def _insert_rows(conn, table: str, columns: tuple, rows) -> int:
    marks = ",".join("?" * len(columns))
    sql = (f"INSERT INTO {table} ({','.join(columns)}) VALUES ({marks})"
           " ON CONFLICT DO NOTHING")
    return sum(conn.execute(sql, tuple(r[c] for c in columns)).rowcount
               for r in rows)


def _copy_documents(conn) -> None:
    for ns, path in _document_paths().items():
        if not os.path.exists(path):
            print(f"  {ns}: no file at {path}, skipped")
            continue
        with open(path, encoding="utf-8") as f:
            docs = json.load(f)
        added = sum(conn.execute(
            "INSERT INTO mt_docs (ns, key, value) VALUES (?,?,?)"
            " ON CONFLICT DO NOTHING", (ns, k, json.dumps(v))).rowcount
            for k, v in docs.items())
        print(f"  {ns}: {added} of {len(docs)} entries added")


def _copy_sessions(conn, path: str) -> None:
    if not os.path.exists(path):
        print(f"  sessions: no database at {path}, skipped")
        return
    # Through sessions.py, which adds any column an older database lacks.
    source = _connect_sqlite(path)
    try:
        sessions = source.execute("SELECT * FROM sessions ORDER BY rowid").fetchall()
        submissions = source.execute("SELECT * FROM submissions").fetchall()
    finally:
        source.close()
    # In insertion order, so "newest session" still means the same thing.
    added = _insert_rows(conn, "sessions", _SESSION_COLUMNS, sessions)
    print(f"  sessions: {added} of {len(sessions)} rows added")
    added = _insert_rows(conn, "submissions", _SUBMISSION_COLUMNS, submissions)
    print(f"  submissions: {added} of {len(submissions)} rows added")


def main(argv: list[str]) -> int:
    if not pg.enabled():
        print("DATABASE_URL is not set.")
        return 1
    sqlite_path = (argv[argv.index("--sqlite") + 1] if "--sqlite" in argv else
                   os.environ.get("MICROTUTOR_SESSION_DB",
                                  "data/grading_sessions.sqlite3"))
    conn = pg.connect()
    try:
        conn.execute("BEGIN")
        _copy_documents(conn)
        _copy_sessions(conn, sqlite_path)
        conn.execute("COMMIT")
    finally:
        conn.close()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
