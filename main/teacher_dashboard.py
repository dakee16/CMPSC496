"""Class difficulty from the existing archive; no model calls or new grades.

Use each student's latest session for each problem. Restarts must not leave
abandoned sessions on the follow-up list. Older work stays in the transcript.
"""
from collections import defaultdict
from datetime import datetime, timezone

from .auth import full_name
from .grades import percent
from .student_progress import _at, _pages


def dashboard_snapshot(client, assignment_id=None):
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
            "id, session_id, student_id, slug, chunk_index, verdict, tier, reason, created_at")
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

    results, active, needs_help = [], set(), set()
    indeterminate = 0
    for problem in problems:
        attempted, struggled, pending = set(), set(), set()
        steps, follow_up = {}, []
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
                        "tier": last.get("tier"), "at": last.get("created_at")})
            if unresolved:
                pending.add(student_id)
                recent = max(unresolved, key=lambda r: timestamp(r["at"]))
                follow_up.append({"student_id": student_id, "name": roster[student_id],
                    "steps": sorted(s["number"] for s in unresolved),
                    "reason": recent["reason"], "last_activity": recent["at"]})
        needs_help.update(pending)
        results.append({**problem, "opened": len(visits), "attempted": len(attempted),
            "needs_help": len(pending), "recovered": len(struggled - pending),
            "difficulty_percent": percent(len(struggled), len(attempted)),
            "steps": sorted(steps.values(), key=lambda s: (-s["needs_help"], -s["recovered"], s["number"])),
            "follow_up": sorted(follow_up, key=lambda s: (-len(s["steps"]), s["name"].casefold()))})
    results.sort(key=lambda p: (-p["needs_help"], -p["recovered"], p["title"] or p["slug"]))
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "assignments": assignments,
            "assignment_id": assignment_id, "problems": results,
            "summary": {"students": len(roster), "active": len(active),
                        "needs_help": len(needs_help), "problems": len(results),
                        "attempted_problems": sum(p["attempted"] > 0 for p in results),
                        "indeterminate": indeterminate}}
