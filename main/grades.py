"""
grades.py - the instructor's grade sheet, and one student's transcript.

WHAT A GRADE IS HERE

A problem is worked one STEP at a time, and each step ends in one of three
states:

  * SOLVED   - the student's own code passed the oracle;
  * SHOWN    - they missed twice, so main/sessions.py revealed the reference and
               moved them on (the session is marked assisted from then on);
  * NOT DONE - they never submitted anything for it.

The grade is solved steps over all steps, so a shown step and an untouched step
cost the same. That is the point: the reveal is a teaching device, not partial
credit, and a student who was handed six answers has not done more work than one
who stopped after the first.

WHERE THE NUMBERS COME FROM

Nothing here is computed at grading time and stored. Everything is derived, on
read, from the append-only archive in main/archive.py - mt_sessions for what a
student was served, mt_submissions for what they sent. So a grade can never
drift out of step with the record behind it, and the transcript this module
prints IS the evidence for the number beside it.

THE DENOMINATOR FOR AN UNTOUCHED PROBLEM

A student who never opened a problem has no session, so their own records cannot
say how many steps it had. The count comes instead from the class (any student's
session for that slug) and, failing that, from the decomposition pool - what
they WOULD have been served. A problem nobody has opened and that has never been
decomposed contributes zero steps rather than a guess.

READS ONLY. This module writes nothing, and every route that calls it is
instructor-gated: it returns other people's work by design.
"""
from datetime import datetime, timezone

from .sessions import MAX_ATTEMPTS

# A transcript is read by a person, so code blocks in it are wrapped in rules
# rather than left to run into the prose around them.
_RULE = "-" * 72
_HEAVY = "=" * 72


# ── the tally ────────────────────────────────────────────────────────────

def tally(total_steps: int, submissions: list[dict]) -> dict:
    """One student's standing on one problem.

    `submissions` are that student's mt_submissions rows for this slug, in any
    order. Pure - no client, no I/O - so the rule that decides a grade can be
    exercised without a database.

    A step is SHOWN when it accumulated MAX_ATTEMPTS wrong answers and never a
    right one, because that is exactly the condition under which /grade_chunk
    reveals the reference. Reading it back off the submissions rather than
    storing a flag keeps this in step with the policy: change the attempt limit
    and both change together."""
    by_step: dict[int, list[str]] = {}
    for s in submissions:
        idx = s.get("chunk_index")
        if idx is None:
            continue
        by_step.setdefault(int(idx), []).append(s.get("verdict") or "")

    solved = shown = 0
    for verdicts in by_step.values():
        if "correct" in verdicts:
            solved += 1
        elif verdicts.count("incorrect") >= MAX_ATTEMPTS:
            shown += 1

    # A total below what we have seen means the count is stale or missing; the
    # evidence outranks it. A denominator must never be smaller than its parts.
    total = max(int(total_steps or 0), len(by_step))
    return {"total": total, "solved": solved, "shown": shown,
            "missed": max(0, total - solved - shown),
            "attempted": bool(by_step)}


def percent(solved: int, total: int) -> int | None:
    """Rounded percentage, or None when there is nothing to grade. None rather
    than 0: a student with no gradeable steps has not scored zero, and a 0%
    would be an accusation the data does not support."""
    return round(100 * solved / total) if total else None


# ── reading the archive ──────────────────────────────────────────────────

def _rows(client, table: str, column: str, values: list, select: str) -> list[dict]:
    """One `in` query, or nothing when there is nothing to ask about.

    Supabase turns an empty `in_` list into a filter that matches everything,
    which on mt_submissions is the whole class's work. Guarding here means no
    caller can trip that."""
    if client is None or not values:
        return []
    return client.table(table).select(select).in_(column, values).execute().data or []


def step_counts(client, problems: list[dict]) -> dict[str, int]:
    """slug -> how many steps that problem splits into.

    Class sessions first (that is what students were actually served), pool
    second (what the next one would be). Both are read-only; this never
    triggers a decomposition, which would turn opening a grade sheet into
    minutes of model work."""
    slugs = [p["slug"] for p in problems]
    counts: dict[str, int] = {}
    for row in _rows(client, "mt_sessions", "slug", slugs, "slug, total_chunks"):
        n = int(row.get("total_chunks") or 0)
        if n > counts.get(row["slug"], 0):
            counts[row["slug"]] = n

    from .run_phase1 import pooled_step_count
    for p in problems:
        if not counts.get(p["slug"]):
            try:
                counts[p["slug"]] = pooled_step_count(p)
            except Exception:
                counts[p["slug"]] = 0        # an unreadable pool is not a grade
    return counts


def assignment_problems(client, assignment_id: str) -> list[dict]:
    """Every READY problem in one assignment, with the fields the pool key
    needs. Unready problems are excluded because students were never served
    them, so grading anyone against them would mark work nobody could do."""
    return client.table("problems").select(
        "slug, title, description, solution, ready").eq(
        "assignment_id", assignment_id).eq("ready", True).order(
        "slug").execute().data or []


def roster(client) -> list[dict]:
    """The class, surname first. Students only - an instructor's own practice
    runs are not coursework and would pad the sheet with staff."""
    from .auth import full_name

    rows = client.table("students").select(
        "id, username, first_name, last_name").eq(
        "role", "student").execute().data or []
    for r in rows:
        r["name"] = full_name(r)
    return sorted(rows, key=lambda r: ((r.get("last_name") or "").lower(),
                                       (r.get("first_name") or "").lower(),
                                       r["username"]))


def grade_sheet(client, assignment_id: str) -> dict:
    """One row per student for one assignment.

    Four queries for the whole class, not four per student: a section of thirty
    against an assignment of twenty problems would otherwise be hundreds of
    round trips to render one table."""
    problems = assignment_problems(client, assignment_id)
    slugs = [p["slug"] for p in problems]
    counts = step_counts(client, problems)
    total_steps = sum(counts.get(s, 0) for s in slugs)

    subs: dict[str, dict[str, list[dict]]] = {}
    for row in _rows(client, "mt_submissions", "slug", slugs,
                     "student_id, slug, chunk_index, verdict"):
        subs.setdefault(row["student_id"], {}).setdefault(
            row["slug"], []).append(row)

    # Started but never submitted still counts as turned up, and the sheet has
    # to say so: "missing" beside a name that spent an hour on the problem is
    # the one wrong answer this column can give.
    started = {r["student_id"] for r in
               _rows(client, "mt_sessions", "slug", slugs, "student_id, slug")}

    rows = []
    for st in roster(client):
        mine = subs.get(st["id"], {})
        agg = {"solved": 0, "shown": 0, "missed": 0}
        for slug in slugs:
            t = tally(counts.get(slug, 0), mine.get(slug, []))
            for k in agg:
                agg[k] += t[k]
        rows.append({"student_id": st["id"], "name": st["name"],
                     "username": st["username"],
                     "submitted": bool(mine) or st["id"] in started,
                     **agg, "total": total_steps,
                     "percent": percent(agg["solved"], total_steps)})
    return {"assignment_id": assignment_id, "problems": len(problems),
            "total_steps": total_steps, "students": rows}


# ── the transcript ───────────────────────────────────────────────────────

def _when(iso: str | None) -> str:
    """A timestamp a person can read, or a dash. Never raises on a value the
    database happened to store in another shape - a transcript that fails over
    one odd row is worth less than one with a dash in it."""
    if not iso:
        return "-"
    try:
        return datetime.fromisoformat(str(iso).replace("Z", "+00:00")).strftime(
            "%Y-%m-%d %H:%M UTC")
    except ValueError:
        return str(iso)


def _block(text: str, indent: str = "    ") -> str:
    """Indent a code or chat blob so it cannot be mistaken for transcript
    structure, and never emit nothing where a block was announced."""
    body = (text or "").rstrip()
    if not body:
        return indent + "(empty)"
    return "\n".join(indent + line for line in body.splitlines())


def render_graph(graph: dict | None) -> str:
    """A graph as text, because the transcript is a text file.

    Nodes with their kind, then edges as arrows. The kind is what the plan and
    the code graph have in common (see main/graphs.py), so it is what makes the
    two comparable when they are read one after the other."""
    if not graph or not graph.get("nodes"):
        return "    (none recorded)"
    lines = [f"    {n.get('id', '?'):<6} [{n.get('kind', 'step'):^6}] "
             f"{n.get('label', '')}" for n in graph["nodes"]]
    edges = graph.get("edges") or []
    if edges:
        lines.append("    flow:")
        for e in edges:
            label = f"  ({e['label']})" if e.get("label") else ""
            lines.append(f"      {e.get('src')} -> {e.get('dst')}{label}")
    return "\n".join(lines)


def _design_link(client, path: str | None) -> str:
    """A time-limited link to a design image, or a note saying why there is
    none. The bucket is private and must stay that way - these are student work
    products - so the transcript carries a signed URL, not a path only the
    database can resolve."""
    if not path:
        return "image not stored"
    try:
        from .archive import DESIGN_BUCKET
        signed = client.storage.from_(DESIGN_BUCKET).create_signed_url(
            path, 7 * 24 * 3600)
        return signed.get("signedURL") or signed.get("signedUrl") or path
    except Exception as e:
        # Say what went wrong rather than printing a path that looks like a
        # link and is not one.
        return f"{path}  (no link: {str(e)[:80]})"


def transcript(client, assignment_id: str, student_id: str) -> tuple[str, str]:
    """(filename, text) - everything recorded for one student on one
    assignment, in the order a grader reads it.

    Per problem: what they scored, then the planning conversation, then the
    diagram they uploaded, then both graphs, then every submission with its
    verdict. That order is deliberate - it is the order the student worked in,
    so a grader can see WHERE a wrong answer came from and not only that it was
    wrong."""
    from .archive import student_history
    from .auth import full_name

    student = (client.table("students").select(
        "id, username, first_name, last_name").eq(
        "id", student_id).limit(1).execute().data or [None])[0]
    if student is None:
        raise LookupError(f"no student {student_id}")
    name = full_name(student)

    meta = (client.table("assignments").select("id, name").eq(
        "id", assignment_id).limit(1).execute().data or [{}])[0]
    problems = assignment_problems(client, assignment_id)
    counts = step_counts(client, problems)
    history = student_history(client, student_id)

    def for_slug(key, slug):
        return [r for r in (history.get(key) or []) if r.get("slug") == slug]

    agg = {"solved": 0, "shown": 0, "missed": 0}
    bodies = []
    for i, p in enumerate(problems, 1):
        slug = p["slug"]
        subs = for_slug("submissions", slug)
        t = tally(counts.get(slug, 0), subs)
        for k in agg:
            agg[k] += t[k]

        out = [_HEAVY, f"PROBLEM {i} of {len(problems)}: {p.get('title') or slug}",
               f"slug: {slug}", _HEAVY, "",
               f"Steps: {t['total']} total | {t['solved']} solved | "
               f"{t['shown']} answer shown | {t['missed']} not done", ""]

        sessions = for_slug("sessions", slug)
        if sessions:
            out.append("SESSIONS")
            for s in sessions:
                out.append(f"  started {_when(s.get('started_at'))}  "
                           f"finished {_when(s.get('completed_at'))}  "
                           f"assisted: {'yes' if s.get('assisted') else 'no'}  "
                           f"solved independently: "
                           f"{'yes' if s.get('solved_independently') else 'no'}")
            out.append("")

        msgs = for_slug("messages", slug)
        out.append(f"CHAT ({len(msgs)} turn{'' if len(msgs) == 1 else 's'})")
        if not msgs:
            out.append("    (none recorded)")
        for m in msgs:
            who = "student" if m.get("role") == "user" else "tutor"
            out.append(f"  [{who} - {m.get('phase', '?')} - "
                       f"{_when(m.get('created_at'))}]")
            out.append(_block(m.get("content", ""), "      "))
        out.append("")

        designs = for_slug("designs", slug)
        out.append(f"DESIGN UPLOADS ({len(designs)})")
        if not designs:
            out.append("    (none recorded)")
        for d in designs:
            out.append(f"  round {d.get('round')}  "
                       f"{'approved' if d.get('approved') else 'not approved'}  "
                       f"{d.get('mime', '?')}  {_when(d.get('created_at'))}")
            out.append(f"      image: {_design_link(client, d.get('storage_path'))}")
            if d.get("reviewer_reply"):
                out.append("      reviewer said:")
                out.append(_block(d["reviewer_reply"], "        "))
        out.append("")

        # Newest snapshot of each kind wins: save_graph appends one every time
        # the plan changes, and what is graded is where the student ended up.
        latest = {}
        for g in for_slug("graphs", slug):
            if g.get("graph"):
                latest[g.get("kind")] = g["graph"]
        out += ["PLAN GRAPH (what they said they would do)",
                render_graph(latest.get("plan")), "",
                "CODE GRAPH (what they wrote)",
                render_graph(latest.get("code")), ""]

        out.append(f"CODE SUBMISSIONS ({len(subs)})")
        if not subs:
            out.append("    (none recorded)")
        for row in sorted(subs, key=lambda r: (r.get("chunk_index") or 0,
                                               r.get("attempt") or 0)):
            out.append(f"  step {int(row.get('chunk_index') or 0) + 1}  "
                       f"attempt {row.get('attempt')}  "
                       f"{(row.get('verdict') or '?').upper()}  "
                       f"({row.get('tier') or 'no tier'}, "
                       f"{_when(row.get('created_at'))})")
            out.append("  " + _RULE)
            out.append(_block(row.get("code", ""), "      "))
            out.append("  " + _RULE)
            if row.get("reason"):
                out.append("      grader said: "
                           + " ".join(str(row["reason"]).split())[:400])
            out.append("")
        bodies.append("\n".join(out).rstrip() + "\n")

    total = sum(counts.get(p["slug"], 0) for p in problems)
    pct = percent(agg["solved"], total)
    head = [_HEAVY, "MICROTUTOR TRANSCRIPT", _HEAVY,
            f"Student    : {name} <{student['username']}>",
            f"Assignment : {meta.get('name') or assignment_id}",
            f"Generated  : {_when(datetime.now(timezone.utc).isoformat())}", "",
            f"GRADE      : {agg['solved']} / {total} steps solved"
            + (f"  ({pct}%)" if pct is not None else "  (nothing to grade yet)"),
            f"             {agg['solved']} solved, {agg['shown']} answer shown, "
            f"{agg['missed']} not done",
            "", "A step counts as solved only when the student's own code passed.",
            "A revealed answer scores the same as one never attempted.", ""]

    stem = "".join(c if c.isalnum() else "-" for c in name.lower()).strip("-")
    # A blank line between problems, or the last "(none recorded)" of one runs
    # straight into the next problem's rule and the file reads as one block.
    return (f"transcript-{stem or student_id}.txt",
            "\n".join(head) + "\n" + "\n".join(bodies or ["(no problems)"]))


if __name__ == "__main__":
    # The rule that decides a grade, with no database in sight.
    assert MAX_ATTEMPTS == 2, "the shown-step case below is written against 2"

    none = tally(4, [])
    assert none == {"total": 4, "solved": 0, "shown": 0, "missed": 4,
                    "attempted": False}, none

    mixed = tally(4, [
        {"chunk_index": 0, "verdict": "correct"},
        {"chunk_index": 1, "verdict": "incorrect"},      # one miss, still open
        {"chunk_index": 2, "verdict": "incorrect"},
        {"chunk_index": 2, "verdict": "incorrect"},      # revealed
        {"chunk_index": 3, "verdict": "indeterminate"},  # grader could not say
    ])
    assert mixed["solved"] == 1 and mixed["shown"] == 1, mixed
    assert mixed["missed"] == 2, mixed
    assert mixed["solved"] + mixed["shown"] + mixed["missed"] == mixed["total"]

    # A wrong answer followed by a right one is solved, not shown. Row order
    # must not decide that, since they arrive unsorted.
    late = tally(1, [{"chunk_index": 0, "verdict": "correct"},
                     {"chunk_index": 0, "verdict": "incorrect"},
                     {"chunk_index": 0, "verdict": "incorrect"}])
    assert late == {"total": 1, "solved": 1, "shown": 0, "missed": 0,
                    "attempted": True}, late

    # A stale or missing step count must never produce a negative "not done",
    # nor a denominator smaller than the work already on record.
    stale = tally(1, [{"chunk_index": i, "verdict": "correct"} for i in range(3)])
    assert stale["total"] == 3 and stale["missed"] == 0, stale
    assert tally(0, [])["total"] == 0

    assert percent(3, 4) == 75 and percent(0, 4) == 0
    assert percent(0, 0) is None, "nothing to grade is not a zero"

    g = {"nodes": [{"id": "n0", "kind": "start", "label": "Start"},
                   {"id": "n1", "kind": "loop", "label": "for each digit"}],
         "edges": [{"src": "n0", "dst": "n1", "label": "then"}]}
    text = render_graph(g)
    assert "for each digit" in text and "n0 -> n1  (then)" in text, text
    assert "(none recorded)" in render_graph(None)
    assert "(none recorded)" in render_graph({"nodes": []})

    assert _block("") == "    (empty)"
    assert _block("a\nb", "  ") == "  a\n  b"
    assert _when(None) == "-" and _when("not-a-date") == "not-a-date"

    print("grades.py self-check OK")
