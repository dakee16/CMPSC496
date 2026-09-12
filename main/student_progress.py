"""Read-only, caller-scoped progress for the student dashboard and grades.

Reuse the instructor's tally and cached step counts. Never grade, generate
steps, write records, or return reference code / another student's records.
"""
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from .grades import percent, step_counts, tally
from .sessions import context_of


def _pages(query):
    """Read beyond PostgREST's row cap; each page builds a fresh query."""
    result, offset = [], 0
    while True:
        rows = query().range(offset, offset + 499).execute().data or []
        result.extend(rows)
        if len(rows) < 500:
            return result
        offset += 500


def _at(value):
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
    except (ValueError, TypeError):
        return None


def progress_snapshot(client, student_id, tz="UTC", now=None):
    zone = ZoneInfo(tz)
    now = now or datetime.now(timezone.utc)
    today = now.astimezone(zone).date()
    days = {str(today - timedelta(days=i)): {"submissions": 0, "passed": 0, "active": False}
            for i in range(6, -1, -1)}
    assignments = [a for a in _pages(lambda: client.table("assignments").select(
        "id, name, created_at, published").order("created_at", desc=True).order("id"))
        if a.get("published", True) is not False]
    ids = [a["id"] for a in assignments]
    problems = _pages(lambda: client.table("problems").select(
        "slug, title, assignment_id, description, solution, ready, context, "
        "group_slug, group_title, group_description").in_("assignment_id", ids)
        .eq("ready", True).order("slug")) if ids else []
    problems = [{**p, **context_of(p)} for p in problems]
    counts = step_counts(client, problems) if problems else {}
    allowed = {p["slug"] for p in problems}

    # Filters are attached at the database boundary, not just after fetching.
    # Avoid a long URL-sized IN list on the student's potentially large history.
    sessions = _pages(lambda: client.table("mt_sessions").select(
        "slug, started_at, completed_at, solved_independently").eq(
        "student_id", student_id).order("session_id")) if allowed else []
    submissions = _pages(lambda: client.table("mt_submissions").select(
        "slug, chunk_index, verdict, created_at").eq("student_id", student_id)
        .order("id")) if allowed else []
    solves = _pages(lambda: client.table("solved").select("problem_slug").eq(
        "student_id", student_id).order("problem_slug")) if allowed else []
    independent = {s["problem_slug"] for s in solves if s["problem_slug"] in allowed}
    by_session, by_submission = defaultdict(list), defaultdict(list)
    for s in sessions:
        if s["slug"] in allowed:
            by_session[s["slug"]].append(s)
    for s in submissions:
        if s["slug"] in allowed:
            by_submission[s["slug"]].append(s)

    results, recent = [], []
    for p in problems:
        slug = p["slug"]
        records, visits = by_submission[slug], by_session[slug]
        score = tally(counts.get(slug, 0), records)
        completed = [s for s in visits if s.get("completed_at")]
        own = slug in independent or any(s.get("solved_independently") for s in completed)
        status = "solved" if own else "helped" if completed else "progress" if records or visits else "todo"
        timestamps = []
        for s in visits:
            at = _at(s.get("started_at"))
            if at:
                timestamps.append(at)
                day = str(at.astimezone(zone).date())
                if day in days:
                    days[day]["active"] = True
                recent.append({"slug": slug, "kind": "opened", "at": at.isoformat()})
            at = _at(s.get("completed_at"))
            if at:
                timestamps.append(at)
                recent.append({"slug": slug, "kind": "completed", "at": at.isoformat()})
        steps = defaultdict(list)
        for s in records:
            if s.get("chunk_index") is not None:
                steps[int(s["chunk_index"])].append(s)
            at = _at(s.get("created_at"))
            if at:
                timestamps.append(at)
                day = str(at.astimezone(zone).date())
                if s.get("verdict") in ("correct", "incorrect"):
                    if day in days:
                        days[day]["submissions"] += 1
                        days[day]["passed"] += s["verdict"] == "correct"
                        days[day]["active"] = True
                    recent.append({"slug": slug, "kind": "passed" if s["verdict"] == "correct" else "practiced",
                                   "step": int(s.get("chunk_index") or 0) + 1, "at": at.isoformat()})
        detail = []
        for i in range(score["total"]):
            attempts = steps[i]
            step = tally(1, attempts)
            state = "passed" if step["solved"] else "shown" if step["shown"] else (
                "needs_work" if any(a.get("verdict") == "incorrect" for a in attempts) else
                "not_graded" if attempts else "not_started")
            detail.append({"number": i + 1, "status": state,
                           "attempts": sum(a.get("verdict") in ("correct", "incorrect") for a in attempts)})
        results.append({"slug": slug, "title": p.get("title") or slug,
                        "assignment_id": p["assignment_id"], "group": p.get("group_title"),
                        "status": status, **score,
                        "percent": percent(score["solved"], score["total"]),
                        "attempts": sum(s.get("verdict") in ("correct", "incorrect") for s in records),
                        "last_activity": max(timestamps).isoformat() if timestamps else None,
                        "steps": detail})

    def summary(rows):
        total = sum(p["total"] for p in rows)
        earned = sum(p["solved"] for p in rows)
        complete = sum(p["status"] in ("solved", "helped") for p in rows)
        return {"problems": len(rows), "completed": complete,
                "independent": sum(p["status"] == "solved" for p in rows),
                "in_progress": sum(p["status"] == "progress" for p in rows),
                "not_started": sum(p["status"] == "todo" for p in rows),
                "total": total, "earned": earned,
                "shown": sum(p["shown"] for p in rows),
                "remaining": sum(p["missed"] for p in rows),
                "percent": percent(earned, total),
                "completion_percent": percent(complete, len(rows)),
                "ungraded_problems": sum(p["total"] == 0 for p in rows)}

    visible = []
    for a in assignments:
        rows = [p for p in results if p["assignment_id"] == a["id"]]
        if rows:
            visible.append({"id": a["id"], "name": a["name"], **summary(rows)})
    lookup = {p["slug"]: p for p in results}
    recent.sort(key=lambda e: _at(e["at"]), reverse=True)
    next_up = sorted((p for p in results if p["status"] == "progress"),
                     key=lambda p: p["last_activity"] or "", reverse=True)
    next_up += [p for a in visible for p in results if p["assignment_id"] == a["id"] and p["status"] == "todo"]
    return {"generated_at": now.isoformat(), "timezone": tz,
            "summary": {**summary(results), "assignments": len(visible),
                        "completed_assignments": sum(a["completed"] == a["problems"] for a in visible),
                        "active_days": sum(day["active"] for day in days.values()),
                        "weekly_submissions": sum(day["submissions"] for day in days.values())},
            "assignments": visible, "problems": results,
            "activity": [{"date": day, **data} for day, data in days.items()],
            "recent": [{**e, "title": lookup[e["slug"]]["title"],
                        "assignment_id": lookup[e["slug"]]["assignment_id"]} for e in recent[:6]],
            "next_up": [p["slug"] for p in next_up[:3]]}
