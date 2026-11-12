"""docstore.py - the JSON stores, one Postgres row per entry.

The oracle cache, chunk pool and transcripts are each one JSON file that callers
load whole, change one entry of, and save whole. On a shared database that
pattern would let two machines erase each other's work, so save_all() writes
only the entries THIS thread changed since its own load_all().

Student-path reads use get(), which fetches one entry and remembers it briefly.
Grading reads an oracle on every submission; parsing every problem's tests for
that is what a whole-file load cost.
"""
import hashlib
import json
import threading

from cachetools import TTLCache

from . import pg

CACHE_SECONDS = 30

_local = threading.local()
_lock = threading.Lock()
_recent: TTLCache = TTLCache(maxsize=4096, ttl=CACHE_SECONDS)


def enabled() -> bool:
    return pg.enabled()


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _snapshots() -> dict:
    if not hasattr(_local, "snapshots"):
        _local.snapshots = {}
    return _local.snapshots


def load_all(ns: str) -> dict:
    conn = pg.connect()
    try:
        rows = conn.execute("SELECT key, value FROM mt_docs WHERE ns=?",
                            (ns,)).fetchall()
    finally:
        conn.close()
    _snapshots()[ns] = {r["key"]: _digest(r["value"]) for r in rows}
    return {r["key"]: json.loads(r["value"]) for r in rows}


def get(ns: str, key: str, default=None):
    """One entry, possibly up to CACHE_SECONDS old. Misses are not remembered,
    so a problem published a moment ago is found immediately."""
    with _lock:
        text = _recent.get((ns, key))
    if text is None:
        conn = pg.connect()
        try:
            row = conn.execute("SELECT value FROM mt_docs WHERE ns=? AND key=?",
                               (ns, key)).fetchone()
        finally:
            conn.close()
        if row is None:
            return default
        text = row["value"]
        with _lock:
            _recent[(ns, key)] = text
    return json.loads(text)


def save_all(ns: str, docs: dict) -> None:
    """Persist what changed against this thread's last load_all(ns).

    Entries deleted are deleted only if that load saw them. With no prior load
    every entry is written and nothing is deleted, the safe reading of a caller
    that never looked."""
    before = _snapshots().get(ns)
    encoded = {k: json.dumps(v) for k, v in docs.items()}
    changed = {k: t for k, t in encoded.items()
               if before is None or before.get(k) != _digest(t)}
    removed = [k for k in (before or {}) if k not in encoded]
    if not changed and not removed:
        return

    conn = pg.connect()
    try:
        conn.execute("BEGIN")
        for k, text in changed.items():
            conn.execute(
                "INSERT INTO mt_docs (ns, key, value, updated_at)"
                " VALUES (?,?,?,now()) ON CONFLICT (ns, key) DO UPDATE"
                " SET value=excluded.value, updated_at=now()", (ns, k, text))
        for k in removed:
            conn.execute("DELETE FROM mt_docs WHERE ns=? AND key=?", (ns, k))
        conn.execute("COMMIT")
    finally:
        conn.close()
    _snapshots()[ns] = {k: _digest(t) for k, t in encoded.items()}
    with _lock:
        for k in [*changed, *removed]:
            _recent.pop((ns, k), None)
