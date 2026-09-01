"""
archive.py - the permanent record of what a student did.

STATUS: written and wired, INERT until sign-in exists. Every function here
returns immediately when `student_id` is None, which is what
api_server.current_student() returns today. The moment the PSU auth block is
uncommented, real ids start arriving and these begin writing, with no second
edit at the call sites.

WHAT IS SAVED, AND WHY EACH ONE

  mt_sessions     one row per attempt at a problem - who, which problem, when,
                  and how it ended (independently / with a revealed reference).
                  The spine everything else hangs off.

  mt_submissions  EVERY code submission, not just the accepted one. The wrong
                  attempts are the entire point: "where students go wrong" is
                  a question only the rejected submissions can answer, and the
                  session store throws them away when the session expires.

  mt_messages     the full chat, both phases. Design-review rounds and tutor
                  turns share one table with a `phase` column, because the
                  interesting query - what did they say before they got it
                  right - crosses both.

  mt_designs      each uploaded design, its verdict, and which round it was.
                  The IMAGE goes to object storage and only its path is kept
                  here; a base64 diagram in a JSONB column makes every later
                  query drag megabytes it does not need.

  mt_graphs       snapshots of the plan graph as it grew, plus the final code
                  graph. Snapshots rather than a final state, because a plan
                  that changed three times mid-problem is a finding, and a
                  single final row destroys the evidence of it.

THREE RULES THIS MODULE FOLLOWS

  1. APPEND ONLY. Nothing here updates or deletes a row about student work.
     A correction is a new row with a later timestamp. An archive you can
     rewrite cannot be used to answer "what actually happened".

  2. NEVER BLOCK THE STUDENT. Every write is best-effort: a failure is logged
     and swallowed. Losing an analytics row is a nuisance; a student unable to
     submit because the archive is down is an outage. This is the opposite of
     the rule in grading.py, and deliberately so.

  3. KEYED ON THE ENTRA `oid`, NEVER THE EMAIL. Addresses get reassigned when
     a student changes their name; re-pointing a year of saved work at the
     wrong person is not a recoverable mistake. Email is stored as a display
     convenience and is never a join key.

FERPA: these rows are education records. Whatever retention window the course
settles on, `purge_student()` at the bottom is the one supported way to honour
a deletion request - which is why it is the single exception to rule 1.
"""
import json
from datetime import datetime, timezone

# Paste into the Supabase SQL editor once, before turning sign-in on.
#
# `student_id` is text, not a FK to auth.users: identity comes from Entra, not
# from Supabase Auth, so there is no local user row to point at.
SCHEMA_SQL = """
create table if not exists mt_sessions (
  session_id      text primary key,
  student_id      text not null,
  student_email   text,
  slug            text not null,
  content_hash    text,
  started_at      timestamptz not null default now(),
  completed_at    timestamptz,
  assisted        boolean not null default false,
  solved_independently boolean not null default false,
  total_chunks    integer
);
create index if not exists mt_sessions_student on mt_sessions (student_id, started_at desc);
create index if not exists mt_sessions_slug    on mt_sessions (slug);

create table if not exists mt_submissions (
  id            bigserial primary key,
  session_id    text not null,
  student_id    text not null,
  slug          text not null,
  chunk_index   integer not null,
  attempt       integer not null,
  code          text not null,
  verdict       text not null,          -- correct | incorrect | indeterminate
  tier          text,                   -- which stage decided it
  deterministic boolean,
  reason        text,
  created_at    timestamptz not null default now()
);
create index if not exists mt_submissions_session on mt_submissions (session_id, chunk_index);
create index if not exists mt_submissions_wrong on mt_submissions (slug, chunk_index)
  where verdict = 'incorrect';          -- the "where do they go wrong" query

create table if not exists mt_messages (
  id          bigserial primary key,
  session_id  text,                     -- null: chat before a session exists
  student_id  text not null,
  slug        text not null,
  phase       text not null,            -- design | tutor
  role        text not null,            -- user | assistant
  content     text not null,
  created_at  timestamptz not null default now()
);
create index if not exists mt_messages_thread on mt_messages (student_id, slug, created_at);

create table if not exists mt_designs (
  id            bigserial primary key,
  student_id    text not null,
  slug          text not null,
  round         integer not null,
  mime          text not null,
  storage_path  text,                   -- object storage; NOT the bytes
  byte_size     integer,
  approved      boolean not null,
  reviewer_reply text,
  created_at    timestamptz not null default now()
);
create index if not exists mt_designs_student on mt_designs (student_id, slug, round);

create table if not exists mt_graphs (
  id          bigserial primary key,
  session_id  text,
  student_id  text not null,
  slug        text not null,
  kind        text not null,            -- plan | code
  graph       jsonb not null,
  node_count  integer,
  created_at  timestamptz not null default now()
);
create index if not exists mt_graphs_session on mt_graphs (session_id, kind, created_at);
"""

# Supabase Storage bucket for design uploads. Must be PRIVATE: these are
# student work products, and a public bucket makes every diagram world-readable
# to anyone who guesses a path.
DESIGN_BUCKET = "designs"

MAX_TEXT = 8000          # one archived message/code blob; longer is truncated


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clip(value: str | None) -> str:
    return (value or "")[:MAX_TEXT]


def _write(client, table: str, row: dict) -> bool:
    """One best-effort insert. Rule 2 lives here.

    Returns True on success so a caller can log a rate, but no caller may branch
    on a False in a way that affects the student."""
    if client is None:
        return False
    try:
        client.table(table).insert(row).execute()
        return True
    except Exception as e:
        print(f"  ⚠️  archive: {table} insert failed: {str(e)[:160]}")
        return False


# ── the five writers ─────────────────────────────────────────────────────
# Each takes student_id first and returns immediately when it is None. That
# single guard is what makes this module inert today and live the moment
# current_student() starts returning an id.

def save_session_start(client, student_id: str | None, session: dict,
                       email: str | None = None) -> None:
    """Open the spine row. Called once, when a session is created."""
    if not student_id:
        return
    _write(client, "mt_sessions", {
        "session_id": session["session_id"], "student_id": student_id,
        "student_email": email, "slug": session.get("slug", ""),
        "content_hash": session.get("content_hash"),
        "started_at": _now(),
        "total_chunks": len(session.get("chunks", []))})


def save_session_end(client, student_id: str | None, session_id: str,
                     state: dict) -> None:
    """Close the spine row.

    The one place this module updates rather than inserts, and it is bounded:
    it writes only the three terminal fields, never touching a fact already
    recorded. Rule 1 is about student work, and none of these three are."""
    if not student_id or client is None:
        return
    try:
        client.table("mt_sessions").update({
            "completed_at": _now(),
            "assisted": bool(state.get("assisted")),
            "solved_independently": bool(state.get("solved_independently")),
        }).eq("session_id", session_id).execute()
    except Exception as e:
        print(f"  ⚠️  archive: session close failed: {str(e)[:160]}")


def save_submission(client, student_id: str | None, session: dict,
                    chunk_index: int, attempt: int, code: str,
                    result: dict) -> None:
    """Every submission, right or wrong.

    Called from /grade_chunk AFTER the verdict is committed, so the archive can
    never be the reason a grade fails to record."""
    if not student_id:
        return
    _write(client, "mt_submissions", {
        "session_id": session["session_id"], "student_id": student_id,
        "slug": session.get("slug", ""), "chunk_index": chunk_index,
        "attempt": attempt, "code": _clip(code),
        "verdict": result.get("verdict", ""), "tier": result.get("tier"),
        "deterministic": result.get("deterministic"),
        "reason": _clip(result.get("reason"))[:1000], "created_at": _now()})


def save_messages(client, student_id: str | None, slug: str, phase: str,
                  messages: list[dict], session_id: str | None = None) -> None:
    """Append the newest chat turns.

    Takes only the turns not yet written - the caller passes the tail, not the
    whole history - because re-sending the full transcript on every turn would
    make an n^2 archive out of an n-turn conversation."""
    if not student_id or not messages:
        return
    rows = []
    for m in messages:
        role, content = m.get("role"), m.get("content")
        # Content parts (an uploaded image) are archived by save_design, not
        # here; a base64 blob in a text column helps nobody.
        if role in ("user", "assistant") and isinstance(content, str) \
                and content.strip():
            rows.append({"session_id": session_id, "student_id": student_id,
                         "slug": slug, "phase": phase, "role": role,
                         "content": _clip(content), "created_at": _now()})
    if not rows or client is None:
        return
    try:
        client.table("mt_messages").insert(rows).execute()   # one round trip
    except Exception as e:
        print(f"  ⚠️  archive: message insert failed: {str(e)[:160]}")


def save_design(client, student_id: str | None, slug: str, image_bytes: bytes,
                mime: str, review: dict) -> None:
    """The uploaded design plus its verdict.

    The image goes to private object storage and only its path is recorded.
    An upload failure still records the ROW - knowing a student submitted a
    third design that was rejected matters even if the picture is lost."""
    if not student_id:
        return
    from .design_review import ALLOWED_MIME

    rnd = int(review.get("round", 0))
    ext = ALLOWED_MIME.get(mime, "bin")
    path = f"{student_id}/{slug}/round-{rnd}.{ext}"
    stored = None
    if client is not None and image_bytes:
        try:
            client.storage.from_(DESIGN_BUCKET).upload(
                path, image_bytes, {"content-type": mime, "upsert": "false"})
            stored = path
        except Exception as e:
            print(f"  ⚠️  archive: design upload failed: {str(e)[:160]}")

    _write(client, "mt_designs", {
        "student_id": student_id, "slug": slug, "round": rnd, "mime": mime,
        "storage_path": stored, "byte_size": len(image_bytes or b""),
        "approved": bool(review.get("approved")),
        "reviewer_reply": _clip(review.get("reply"))[:2000],
        "created_at": _now()})


def save_graph(client, student_id: str | None, slug: str, kind: str,
               graph: dict, session_id: str | None = None) -> None:
    """One snapshot of one graph.

    Called each time the plan graph changes and once for the final code graph.
    Snapshots, never an update: a plan that was revised three times is the
    finding, and overwriting would erase it."""
    if not student_id or not graph or not graph.get("nodes"):
        return
    _write(client, "mt_graphs", {
        "session_id": session_id, "student_id": student_id, "slug": slug,
        "kind": kind, "graph": graph, "node_count": len(graph["nodes"]),
        "created_at": _now()})


# ── reading it back ──────────────────────────────────────────────────────

def student_history(client, student_id: str, slug: str | None = None) -> dict:
    """Everything recorded for one student, for the teacher view.

    Reads are allowed to fail loudly - unlike a write, a half-empty report that
    silently looks complete is worse than an error."""
    if client is None:
        return {}
    out: dict = {}
    for table, key in (("mt_sessions", "sessions"),
                       ("mt_submissions", "submissions"),
                       ("mt_messages", "messages"),
                       ("mt_designs", "designs"),
                       ("mt_graphs", "graphs")):
        q = client.table(table).select("*").eq("student_id", student_id)
        if slug:
            q = q.eq("slug", slug)
        order_col = "started_at" if table == "mt_sessions" else "created_at"
        out[key] = q.order(order_col, desc=False).execute().data
    return out


def wrong_answer_report(client, slug: str, chunk_index: int | None = None,
                        limit: int = 200) -> list[dict]:
    """The query this whole module exists to make possible: what did students
    actually submit that was wrong, on this problem, at this step."""
    if client is None:
        return []
    q = (client.table("mt_submissions")
         .select("chunk_index, code, tier, reason, created_at")
         .eq("slug", slug).eq("verdict", "incorrect"))
    if chunk_index is not None:
        q = q.eq("chunk_index", chunk_index)
    return q.order("created_at", desc=True).limit(limit).execute().data


def purge_student(client, student_id: str) -> dict:
    """Delete every record of one student. The ONE exception to append-only.

    Exists because a FERPA deletion request must have a supported path that
    does not involve hand-written SQL against production. Deliberately not
    reachable from any route - it is called deliberately, by a person."""
    if client is None:
        return {}
    removed = {}
    for table in ("mt_graphs", "mt_designs", "mt_messages", "mt_submissions",
                  "mt_sessions"):
        r = client.table(table).delete().eq("student_id", student_id).execute()
        removed[table] = len(r.data or [])
    try:
        files = client.storage.from_(DESIGN_BUCKET).list(student_id)
        if files:
            client.storage.from_(DESIGN_BUCKET).remove(
                [f"{student_id}/{f['name']}" for f in files])
    except Exception as e:
        # Say so rather than reporting a clean purge that left images behind.
        removed["storage_error"] = str(e)[:160]
    return removed


if __name__ == "__main__":
    # The contract that must hold before auth exists: with no student_id every
    # writer is a no-op, and none of them touches the client. A fake client that
    # explodes on use proves it.
    class _Boom:
        def __getattr__(self, name):
            raise AssertionError(f"archive touched the database: .{name}")

    boom, sess = _Boom(), {"session_id": "s1", "slug": "two-sum", "chunks": []}
    save_session_start(boom, None, sess)
    save_session_end(boom, None, "s1", {})
    save_submission(boom, None, sess, 0, 1, "x = 1", {"verdict": "incorrect"})
    save_messages(boom, None, "two-sum", "tutor", [{"role": "user", "content": "hi"}])
    save_design(boom, None, "two-sum", b"x", "image/png", {"round": 1})
    save_graph(boom, None, "two-sum", "plan", {"nodes": [{"id": "a"}]})

    # Empty payloads must also short-circuit, even WITH an id - otherwise every
    # idle turn writes a useless row.
    save_messages(boom, "student-oid", "two-sum", "tutor", [])
    save_graph(boom, "student-oid", "two-sum", "plan", {"nodes": []})

    # Content parts (an uploaded image) must never reach the message table.
    captured = []

    class _Spy:
        def table(self, name):
            return self

        def insert(self, rows):
            captured.extend(rows if isinstance(rows, list) else [rows])
            return self

        def execute(self):
            return type("R", (), {"data": []})()

    save_messages(_Spy(), "oid", "two-sum", "design",
                  [{"role": "user", "content": [{"type": "image_url"}]},
                   {"role": "assistant", "content": "Looks workable."}])
    assert len(captured) == 1, f"image content leaked into mt_messages: {captured}"
    assert captured[0]["content"] == "Looks workable."
    assert captured[0]["phase"] == "design"

    assert len(_clip("x" * 99999)) == MAX_TEXT
    assert json.loads(json.dumps({"nodes": []})) == {"nodes": []}
    for t in ("mt_sessions", "mt_submissions", "mt_messages", "mt_designs",
              "mt_graphs"):
        assert f"create table if not exists {t}" in SCHEMA_SQL, t
    print("archive self-check ok (inert without a student id, as intended)")
