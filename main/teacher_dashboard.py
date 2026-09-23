"""Class difficulty from the existing archive; no model calls or new grades.

Use each student's latest session for each problem. Restarts must not leave
abandoned sessions on the follow-up list. Older work stays in the transcript.

WHAT A TEACHER NEEDS TO ACT ON A "MAY NEED HELP". "Check step 3" and "your
solution gives the wrong answer on at least one case" told them something was
wrong and nothing they could use: not what step 3 asked, and not what the
student wrote. So each unresolved step now carries both - the step's own
PROMPT, and the CODE of the attempt that failed.

The prompt is read from the student's own grading session, never from the
problem: decompositions differ between students (a pooled candidate, or a
roadmap rebuilt around their plan - main/reroute.py), so "step 3" is a
different question for different people and a problem-wide prompt would be
wrong for some of them.

What still never leaves: the teacher's reference solution, and any code
belonging to a step the student has since got right. The route is
teacher-only (api_server.teacher_dashboard -> require_teacher).
"""
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone

from .auth import full_name
from .grades import percent
from .student_progress import _at, _pages


# Long enough for any step a student is asked for; short enough that one
# pasted file cannot make the dashboard payload unbounded.
MAX_CODE_CHARS = 4000


def _session_prompts(session_id):
    """Step prompts for one grading session, from the server's own store.

    Every session row is kept (nothing in main/sessions.py deletes one), so
    this works for finished and expired sessions too. Anything unreadable is
    an empty list, and the page falls back to the step number alone."""
    try:
        from .sessions import session_snapshot
        snap = session_snapshot(session_id)
        return [c.get("prompt") or "" for c in (snap or {}).get("chunks") or []]
    except Exception:
        return []


# ── "Issue seen" ─────────────────────────────────────────────────────────
# An instructor who already knows about a student's mistake can clear it from
# the counter, so a problem's tile stops asking for attention it has had.
#
# A TIMESTAMP, NOT A FLAG. "Seen" covers the mistakes that existed when it was
# pressed, and nothing after: if the student then gets the step wrong AGAIN,
# that is a new thing to look at, and they reappear. A plain flag would let one
# click hide every later struggle on that problem for the rest of the term.
#
# In the server's own SQLite store beside the grading sessions - on the volume
# in production (docker-compose.yml) - so it needs no Supabase migration. One
# row per (student, problem); shared by every instructor, which is the point of
# telling colleagues it has been handled.
def _seen_conn(db_path=None):
    from .sessions import _connect
    conn = _connect(db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS issue_seen ("
                 " student_id TEXT NOT NULL, slug TEXT NOT NULL,"
                 " seen_at TEXT NOT NULL, PRIMARY KEY (student_id, slug))")
    return conn


def seen_marks(db_path=None) -> dict:
    """{(student_id, slug): seen_at}. Unreadable is empty: nobody is hidden."""
    try:
        conn = _seen_conn(db_path)
        try:
            return {(r["student_id"], r["slug"]): r["seen_at"]
                    for r in conn.execute("SELECT * FROM issue_seen")}
        finally:
            conn.close()
    except Exception:
        return {}


def mark_seen(student_id: str, slug: str, seen: bool = True, db_path=None) -> bool:
    """Mark (or un-mark) one student's open issue on one problem as seen."""
    conn = _seen_conn(db_path)
    try:
        if seen:
            conn.execute("INSERT INTO issue_seen (student_id, slug, seen_at)"
                         " VALUES (?,?,?) ON CONFLICT(student_id, slug)"
                         " DO UPDATE SET seen_at=excluded.seen_at",
                         (student_id, slug, datetime.now(timezone.utc).isoformat()))
        else:
            conn.execute("DELETE FROM issue_seen WHERE student_id=? AND slug=?",
                         (student_id, slug))
    finally:
        conn.close()
    return seen


def dashboard_snapshot(client, assignment_id=None, prompts_for=_session_prompts,
                       seen=None):
    assignments = [a for a in _pages(lambda: client.table("assignments").select(
        "id, name, published").order("id")) if a.get("published", True) is not False]
    if assignment_id and assignment_id not in {str(a["id"]) for a in assignments}:
        raise LookupError("Choose a published assignment.")
    ids = [a["id"] for a in assignments if not assignment_id or str(a["id"]) == assignment_id]
    problems = []
    for offset in range(0, len(ids), 100):
        problems.extend(_pages(lambda: client.table("problems").select(
            "slug, title, assignment_id").in_("assignment_id", ids[offset:offset + 100])
            # The teacher's order, same as every other reader.
            .eq("ready", True).order("group_order").order("member_order")
            .order("slug")))
    students = _pages(lambda: client.table("students").select(
        "id, username, first_name, last_name").eq("role", "student").order("id"))
    roster = {s["id"]: full_name(s) for s in students}
    slugs = [p["slug"] for p in problems]
    sessions, submissions = [], []
    for offset in range(0, len(slugs), 100):
        batch = slugs[offset:offset + 100]
        sessions.extend(_pages(lambda: client.table("mt_sessions").select(
            "session_id, student_id, slug, started_at, completed_at, solved_independently")
            .in_("slug", batch).order("session_id")))
        submissions.extend(_pages(lambda: client.table("mt_submissions").select(
            "id, session_id, student_id, slug, chunk_index, verdict, tier, reason, code,"
            " created_at")
            .in_("slug", batch).order("id")))

    def timestamp(value):
        at = _at(value)
        return at.timestamp() if at else 0

    # A session-start archive write can fail while later submissions succeed.
    # Keep those submissions by recovering a start time from their first row.
    visits = {}
    for row in submissions:
        if row["student_id"] not in roster:
            continue
        key = (row["student_id"], row["slug"], row["session_id"])
        at = timestamp(row.get("created_at"))
        if key not in visits or at < timestamp(visits[key].get("started_at")):
            visits[key] = {**row, "started_at": row.get("created_at")}
    for row in sessions:
        if row["student_id"] in roster:
            visits[(row["student_id"], row["slug"], row["session_id"])] = row
    latest = {}
    for key, row in visits.items():
        pair = key[:2]
        order = (timestamp(row.get("started_at")), str(row["session_id"]))
        if pair not in latest or order > latest[pair][0]:
            latest[pair] = (order, row)
    by_problem = defaultdict(dict)
    for (student, slug), (_, row) in latest.items():
        by_problem[slug][student] = {"session": row, "steps": defaultdict(list), "records": []}
    for row in submissions:
        student = by_problem[row["slug"]].get(row["student_id"])
        if student is None or row["session_id"] != student["session"]["session_id"]:
            continue
        student["records"].append(row)
        if row.get("verdict") in ("correct", "incorrect") and row.get("chunk_index", -1) >= 0:
            student["steps"][int(row["chunk_index"])].append(row)

    seen = seen_marks() if seen is None else seen
    prompt_cache = {}

    def prompt(session_id, index):
        if session_id not in prompt_cache:
            prompt_cache[session_id] = prompts_for(session_id)
        found = prompt_cache[session_id]
        return found[index] if 0 <= index < len(found) else ""

    results, active, needs_help = [], set(), set()
    indeterminate = 0
    for problem in problems:
        attempted, struggled, pending, handled = set(), set(), set(), set()
        steps, follow_up, asked = {}, [], defaultdict(Counter)
        visits = by_problem[problem["slug"]]
        active.update(visits)
        for student_id, work in visits.items():
            unresolved = []
            session = work["session"]
            completed = bool(session.get("completed_at") and session.get("solved_independently"))
            indeterminate += sum(r.get("verdict") == "indeterminate" for r in work["records"])
            for index, records in work["steps"].items():
                attempted.add(student_id)
                stat = steps.setdefault(index, {"number": index + 1, "attempted": 0,
                                                "needs_help": 0, "recovered": 0})
                stat["attempted"] += 1
                asked_here = prompt(session["session_id"], index)
                if asked_here:
                    asked[index][asked_here] += 1
                wrong = [r for r in records if r["verdict"] == "incorrect"]
                if not wrong:
                    continue
                struggled.add(student_id)
                if completed or any(r["verdict"] == "correct" for r in records):
                    stat["recovered"] += 1
                else:
                    stat["needs_help"] += 1
                    last = max(wrong, key=lambda r: (timestamp(r.get("created_at")), int(r["id"])))
                    unresolved.append({"number": index + 1, "reason": (last.get("reason") or
                        "An incorrect answer was recorded without further feedback.")[:500],
                        "tier": last.get("tier"), "at": last.get("created_at"),
                        "prompt": asked_here,
                        # The attempt that FAILED, not their latest keystrokes:
                        # it is the code the feedback above is about.
                        "code": (last.get("code") or "")[:MAX_CODE_CHARS]})
            if unresolved:
                pending.add(student_id)
                recent = max(unresolved, key=lambda r: timestamp(r["at"]))
                # Seen only if nothing went wrong AFTER it was marked.
                seen_at = seen.get((student_id, problem["slug"]))
                is_seen = bool(seen_at) and timestamp(recent["at"]) <= timestamp(seen_at)
                if is_seen:
                    handled.add(student_id)
                follow_up.append({"student_id": student_id, "name": roster[student_id],
                    "seen": is_seen,
                    "steps": sorted(s["number"] for s in unresolved),
                    "reason": recent["reason"], "last_activity": recent["at"],
                    "details": [{k: s[k] for k in ("number", "prompt", "reason", "code", "at")}
                                for s in sorted(unresolved, key=lambda s: s["number"])]})
        needs_help.update(pending - handled)
        for index, stat in steps.items():
            common = asked[index].most_common(1)
            stat["prompt"] = common[0][0] if common else ""
            stat["prompt_varies"] = len(asked[index]) > 1
        results.append({**problem, "opened": len(visits), "attempted": len(attempted),
            "needs_help": len(pending - handled), "seen": len(handled),
            "recovered": len(struggled - pending),
            "difficulty_percent": percent(len(struggled), len(attempted)),
            "steps": sorted(steps.values(), key=lambda s: (-s["needs_help"], -s["recovered"], s["number"])),
            # Open issues first; seen ones stay listed, after them.
            "follow_up": sorted(follow_up, key=lambda s: (s["seen"], -len(s["steps"]),
                                                          s["name"].casefold()))})
    results.sort(key=lambda p: (-p["needs_help"], -p["recovered"], p["title"] or p["slug"]))
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "assignments": assignments,
            "assignment_id": assignment_id, "problems": results,
            "summary": {"students": len(roster), "active": len(active),
                        "needs_help": len(needs_help), "problems": len(results),
                        "attempted_problems": sum(p["attempted"] > 0 for p in results),
                        "indeterminate": indeterminate}}


# ── one student, one problem: what an instructor needs to help ───────────
def _local_results(session_id, db_path=None):
    """Graded results the server stored for one session, oldest first. These
    carry what the STUDENT was shown - the failing cases - which the archive
    does not keep."""
    try:
        from .sessions import _connect
        conn = _connect(db_path)
        try:
            rows = conn.execute("SELECT result_json, created_at FROM submissions"
                                " WHERE session_id=? ORDER BY created_at, rowid",
                                (session_id,)).fetchall()
        finally:
            conn.close()
        out = []
        for r in rows:
            try:
                out.append(json.loads(r["result_json"] or "{}"))
            except Exception:
                continue
        return out
    except Exception:
        return []


def review_detail(client, slug, student_id, step=None, db_path=None):
    """Everything about one student's open issue on one problem, or None.

    Built for the instructor's review page, so it answers the questions a
    teacher asks before sitting down with a student: which step, what it
    asked, what the student was told, WHICH CASES it failed on, the whole
    function as it stood, which part of it is the failing attempt, and how
    they have been going at it. Read-only; grades nothing.

    The function is rebuilt from the student's own grading session (its
    accepted prefix is exactly what that step was graded on top of), and the
    failing attempt is re-seated at its step's depth the same way the grader
    did (grading.align_submission) - the archive holds the raw text they typed,
    and showing that flat would draw code they were never graded on.

    `step` picks one of several open steps; the default is the most recent.
    """
    from .grading import align_submission
    from .sessions import session_snapshot

    problem = (client.table("problems").select("slug, title, assignment_id")
               .eq("slug", slug).limit(1).execute().data or [None])[0]
    person = (client.table("students").select("id, username, first_name, last_name")
              .eq("id", student_id).limit(1).execute().data or [None])[0]
    if not problem or not person:
        return None
    assignment = (client.table("assignments").select("id, name")
                  .eq("id", problem["assignment_id"]).limit(1).execute().data or [{}])[0]
    sessions = _pages(lambda: client.table("mt_sessions").select(
        "session_id, student_id, slug, started_at, completed_at")
        .eq("student_id", student_id).eq("slug", slug).order("session_id"))
    rows = _pages(lambda: client.table("mt_submissions").select(
        "id, session_id, chunk_index, verdict, reason, code, created_at")
        .eq("student_id", student_id).eq("slug", slug).order("id"))

    def at(r):
        found = _at(r.get("started_at") or r.get("created_at"))
        return found.timestamp() if found else 0

    # The same "latest session" rule the dashboard uses: a session row, or
    # failing that the earliest submission recorded for it.
    starts = {}
    for r in rows:
        if r["session_id"] not in starts or at(r) < starts[r["session_id"]]:
            starts[r["session_id"]] = at(r)
    for s in sessions:
        starts[s["session_id"]] = at(s)
    if not starts:
        return None
    session_id = max(starts, key=lambda sid: (starts[sid], str(sid)))
    mine = [r for r in rows if r["session_id"] == session_id]

    by_step = defaultdict(list)
    for r in mine:
        if r.get("chunk_index") is not None and r["chunk_index"] >= 0:
            by_step[int(r["chunk_index"])].append(r)
    for attempts in by_step.values():
        attempts.sort(key=lambda r: (at(r), int(r["id"]) if str(r["id"]).isdigit() else 0))
    open_steps = sorted(
        (i for i, a in by_step.items()
         if any(r["verdict"] == "incorrect" for r in a)
         and not any(r["verdict"] == "correct" for r in a)),
        key=lambda i: max(at(r) for r in by_step[i] if r["verdict"] == "incorrect"),
        reverse=True)
    if not open_steps:
        return {"slug": slug, "title": problem.get("title") or slug,
                "assignment_id": problem["assignment_id"],
                "assignment": assignment.get("name") or "",
                "student_id": student_id, "name": full_name(person),
                "open_steps": [], "step": None}
    index = step - 1 if step and (step - 1) in open_steps else open_steps[0]

    snap = session_snapshot(session_id, db_path)
    chunks = (snap or {}).get("chunks") or []
    prompt = lambda i: (chunks[i].get("prompt") or "") if i < len(chunks) else ""
    # The function as it stood: the steps accepted before this one. From the
    # session when it is there; otherwise each earlier step's last correct
    # attempt from the archive.
    if snap:
        prefix = [{"number": i + 1, "prompt": prompt(i), "code": a.get("code") or ""}
                  for i, a in enumerate((snap.get("accepted") or [])[:index])]
    else:
        prefix = []
        for i in range(index):
            good = [r for r in by_step.get(i, []) if r["verdict"] == "correct"]
            prefix.append({"number": i + 1, "prompt": "",
                           "code": (good[-1].get("code") or "") if good else ""})

    attempts = by_step[index]
    failing = [r for r in attempts if r["verdict"] == "incorrect"][-1]
    code = failing.get("code") or ""
    if snap and index < len(chunks):
        try:
            code = align_submission({**snap, "index": index}, code)
        except Exception:
            pass
    # What the student was shown for that attempt: the last failing result the
    # server recorded at this step.
    shown = [r for r in _local_results(session_id, db_path)
             if r.get("verdict") == "incorrect" and r.get("index") == index]
    cases = (shown[-1].get("failing_cases") or []) if shown else []
    seen_at = seen_marks(db_path).get((student_id, slug))
    last_wrong = at(failing)
    return {
        "slug": slug, "title": problem.get("title") or slug,
        "assignment_id": problem["assignment_id"],
        "assignment": assignment.get("name") or "",
        "student_id": student_id, "name": full_name(person),
        "seen": bool(seen_at) and last_wrong <= (_at(seen_at).timestamp() if _at(seen_at) else 0),
        "header": (snap or {}).get("header") or "",
        "total_steps": len(chunks) or None,
        "open_steps": [i + 1 for i in sorted(open_steps)],
        "prefix": prefix,
        "step": {"number": index + 1, "prompt": prompt(index),
                 "reason": failing.get("reason") or "",
                 "code": code[:MAX_CODE_CHARS],
                 "failing_cases": cases[:5],
                 "failed_total": (shown[-1].get("failed_total") if shown else None),
                 "attempts": [{"at": r.get("created_at"), "verdict": r.get("verdict"),
                               "reason": r.get("reason") or "",
                               "code": (r.get("code") or "")[:MAX_CODE_CHARS]}
                              for r in attempts],
                 "first_at": attempts[0].get("created_at"),
                 "last_at": failing.get("created_at")},
    }
