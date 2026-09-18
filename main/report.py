"""report.py - the instructor's readable learning record, as one HTML page.

WHY THIS EXISTS. main/grades.transcript() already assembles the same evidence,
and it is a wall of fixed-width text: a grader looking for "where did this
answer come from" reads 400 lines of rules and ALL-CAPS banners to find three
of them. The evidence was never the problem; the presentation was.

BUILD ONCE, RENDER TWICE. gather() produces ONE record - roster, assignments,
problems, attempts, messages, plans, submissions, graphs - and render_html()
groups it either by student (one student's whole assignment) or by problem
(every student side by side). There is no second grading calculation anywhere
in this module: the numbers come from grades.tally() with the student's own
step count, exactly as the student's own page computes them, so the report and
the site can never disagree.

HTML, NOT PDF, AND THE PDF COMES FROM IT. The page carries a print stylesheet
that expands every collapsed section and breaks pages between students, so the
browser's own Save as PDF produces the printable copy. One renderer, one
artifact: a separately generated PDF would be a second transcript to keep in
step with this one, which is the thing that went wrong the first time.

NOTHING IS INVENTED. A record we do not store renders as "Not recorded" in the
report rather than as zero, absent, or a guess - see _MISSING. What ACADIA does
not capture today is listed once, honestly, in the completeness section.
"""
from datetime import datetime, timezone
from html import escape

from .grades import (assignment_problems, percent, render_graph, step_counts,
                     tally)

# What a field says when the evidence was never captured. Deliberately not "0",
# "none" or "": an unknown step count that reads as zero is a grade nobody
# earned, and an empty chat that reads as silence is a claim we cannot make.
_MISSING = '<span class="missing">Not recorded</span>'

# Evidence ACADIA does not store today. Stated once, in the report, rather than
# quietly rendering as an empty section that looks like it means something.
NOT_CAPTURED = [
    ("Per-attempt step prompts",
     "Step text is read from the decomposition in force now. An attempt worked "
     "under an earlier split shows its own step COUNT, but not the wording it "
     "was given."),
    ("Problem revisions",
     "Problem statements are stored without version history, so an edited "
     "problem shows only its current text."),
    ("Unsubmitted drafts",
     "Autosaved editor drafts live in the student's browser and are never sent, "
     "so only submitted code appears here."),
    ("Instructor notes and manual grade changes",
     "There is no place to record either yet; every number here came from the "
     "grading pipeline."),
]


def _esc(text) -> str:
    return escape("" if text is None else str(text), quote=True)


def _when(iso: str | None) -> str:
    """A timestamp a person can read, or the missing marker."""
    if not iso:
        return ""
    try:
        return datetime.fromisoformat(
            str(iso).replace("Z", "+00:00")).strftime("%d %b %Y, %H:%M UTC")
    except Exception:
        return _esc(iso)


def _sig(problem: dict) -> str:
    """The def line the student wrote under, if it can be read back."""
    try:
        from .context import header_of
        return header_of(problem) or ""
    except Exception:
        return ""


# ── the canonical record ─────────────────────────────────────────────────

def gather(client, assignment_id: str, student_ids: list[str] | None = None) -> dict:
    """Every recorded thing, for one assignment, for the given students.

    `student_ids` None means the whole roster - INCLUDING students with no
    recorded activity, who are the ones an instructor most needs to see. They
    are resolved from the students table rather than inferred from who happens
    to have submitted something.
    """
    from .archive import student_history
    from .auth import full_name

    meta = (client.table("assignments").select("id, name").eq(
        "id", assignment_id).limit(1).execute().data or [{}])[0]
    problems = assignment_problems(client, assignment_id)
    counts = step_counts(client, problems)

    q = client.table("students").select(
        "id, username, first_name, last_name, role").order("username")
    if student_ids:
        q = q.in_("id", student_ids)
    roster = q.execute().data or []

    students = []
    for s in roster:
        history = student_history(client, s["id"])

        def for_slug(key, slug, _h=history):
            return [r for r in (_h.get(key) or []) if r.get("slug") == slug]

        rows, agg, available = [], {"solved": 0, "shown": 0, "missed": 0}, 0
        for p in problems:
            slug = p["slug"]
            subs = for_slug("submissions", slug)
            visits = for_slug("sessions", slug)
            # The student's OWN step count - never the class-wide maximum. See
            # main/student_progress.py for the bug that rule exists to stop.
            own = next((int(v["total_chunks"]) for v in visits
                        if v.get("completed_at") and v.get("total_chunks")), 0) \
                or next((int(v["total_chunks"]) for v in visits
                         if v.get("total_chunks")), 0)
            t = tally(own or counts.get(slug, 0), subs)
            for k in agg:
                agg[k] += t[k]
            available += t["total"]

            graphs = {}
            for g in for_slug("graphs", slug):
                if g.get("graph"):
                    graphs[g.get("kind")] = g["graph"]   # newest of each kind

            done = [v for v in visits if v.get("completed_at")]
            rows.append({
                "problem": p, "tally": t, "sessions": visits,
                "messages": for_slug("messages", slug),
                "designs": for_slug("designs", slug),
                "submissions": sorted(subs, key=lambda r: (
                    r.get("chunk_index") or 0, r.get("attempt") or 0,
                    str(r.get("created_at") or ""))),
                "graphs": graphs,
                "status": ("solved" if any(v.get("solved_independently") for v in done)
                           else "helped" if done
                           else "progress" if (subs or visits) else "todo"),
                "last": max([str(v.get("started_at") or "") for v in visits]
                            + [str(r.get("created_at") or "") for r in subs]
                            + [""]),
            })

        students.append({
            "student": s, "name": full_name(s), "problems": rows,
            "totals": {**agg, "available": available,
                       "percent": percent(agg["solved"], available)},
        })

    return {"assignment": meta, "assignment_id": assignment_id,
            "problems": problems, "students": students,
            "generated_at": datetime.now(timezone.utc).isoformat()}


# ── rendering ────────────────────────────────────────────────────────────

_CSS = """
:root{--ink:#241c17;--muted:#6d625a;--line:#e2d8cc;--bg:#fbf7f1;--card:#fff;
--ok:#1f7a4d;--bad:#a8331f;--warn:#8a6414;--accent:#c2603f}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.6 "DM Sans",-apple-system,Segoe UI,sans-serif}
.wrap{max-width:900px;margin:0 auto;padding:32px 24px 120px}
h1{font-size:30px;margin:0 0 4px;letter-spacing:-.02em}
h2{font-size:22px;margin:40px 0 10px;padding-top:18px;border-top:2px solid var(--line)}
h3{font-size:17px;margin:26px 0 8px}
h4{font-size:15px;margin:18px 0 6px;color:var(--muted);
text-transform:uppercase;letter-spacing:.08em}
p{margin:.5em 0}
.sub{color:var(--muted);margin:0 0 20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:16px 18px;margin:14px 0}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:14px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);
vertical-align:top}
th{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
td.n,th.n{text-align:right}
pre{background:#f4efe8;border:1px solid var(--line);border-radius:8px;
padding:12px 14px;overflow-x:auto;font:13px/1.5 "JetBrains Mono",ui-monospace,monospace;
white-space:pre;margin:8px 0}
.pill{display:inline-block;padding:1px 9px;border-radius:999px;font-size:12px;
font-weight:600;border:1px solid}
.pill.ok{color:var(--ok);border-color:var(--ok);background:#eaf5ef}
.pill.bad{color:var(--bad);border-color:var(--bad);background:#fbeeea}
.pill.warn{color:var(--warn);border-color:var(--warn);background:#fbf3e2}
.pill.mut{color:var(--muted);border-color:var(--line);background:#f4efe8}
.missing{color:var(--muted);font-style:italic}
.meta{color:var(--muted);font-size:13px}
.msg{border-left:3px solid var(--line);padding:2px 0 2px 14px;margin:12px 0}
.msg.student{border-left-color:var(--accent)}
.msg.tutor{border-left-color:#7d92b5}
.msg .who{font-size:12px;font-weight:700;letter-spacing:.06em;
text-transform:uppercase;color:var(--muted)}
.msg .body{white-space:pre-wrap;word-break:break-word}
details{margin:10px 0;border:1px solid var(--line);border-radius:8px;
background:var(--card)}
summary{cursor:pointer;padding:10px 14px;font-weight:600;list-style:none}
summary::-webkit-details-marker{display:none}
summary::before{content:"\\25B8  ";color:var(--muted)}
details[open]>summary::before{content:"\\25BE  "}
.inner{padding:0 14px 14px}
.bar{height:6px;border-radius:4px;background:#ece3d8;overflow:hidden;max-width:220px}
.bar i{display:block;height:100%;background:var(--accent)}
.toolbar{position:sticky;top:0;z-index:5;background:var(--bg);
border-bottom:1px solid var(--line);padding:10px 24px;display:flex;gap:10px;
align-items:center;flex-wrap:wrap}
.toolbar button{font:inherit;font-size:14px;padding:7px 14px;border-radius:8px;
border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer}
.toolbar button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
.toolbar .grow{flex:1}
.student-block{page-break-before:always}
.student-block:first-of-type{page-break-before:auto}
@media print{
  /* EVERY collapsed section opens for print. A report that loses evidence
     because a section was never clicked is not the record it claims to be. */
  details{border:0}
  details>.inner{display:block!important}
  summary{list-style:none}
  .toolbar{display:none}
  body{background:#fff;font-size:11pt}
  .wrap{max-width:none;padding:0}
  pre{white-space:pre-wrap;word-break:break-word}
  h2{page-break-after:avoid}
  tr,pre,.msg{page-break-inside:avoid}
}
"""

_SCRIPT = """
document.addEventListener('click', e => {
  const act = e.target.closest('[data-act]');
  if (!act) return;
  const kind = act.dataset.act;
  if (kind === 'print') window.print();
  else document.querySelectorAll('details').forEach(d => d.open = kind === 'open');
});
/* PRINT MUST NOT LOSE EVIDENCE. CSS alone cannot be trusted to reveal a closed
   <details> - browsers hide its contents through the shadow slot, not through
   anything a stylesheet here can reach - so a report printed with sections
   collapsed silently dropped whole attempts. Open every one before the print
   dialog, and put back exactly what the reader had afterwards. */
let _restore = [];
addEventListener('beforeprint', () => {
  _restore = [...document.querySelectorAll('details')].filter(d => !d.open);
  _restore.forEach(d => d.open = true);
});
addEventListener('afterprint', () => {
  _restore.forEach(d => d.open = false);
  _restore = [];
});
"""


def _pill(status: str) -> str:
    label = {"solved": "Solved independently", "helped": "Completed with help",
             "progress": "In progress", "todo": "Not started"}.get(status, status)
    cls = {"solved": "ok", "helped": "warn", "progress": "mut", "todo": "mut"}
    return f'<span class="pill {cls.get(status, "mut")}">{_esc(label)}</span>'


def _verdict_pill(verdict: str) -> str:
    v = (verdict or "").lower()
    cls = "ok" if v == "correct" else "bad" if v == "incorrect" else "warn"
    return f'<span class="pill {cls}">{_esc(v.upper() or "?")}</span>'


def _credit(t: dict) -> str:
    if not t.get("total"):
        return _MISSING
    pct = percent(t["solved"], t["total"])
    return (f'{t["solved"]} / {t["total"]}'
            + (f' &middot; {pct}%' if pct is not None else ""))


def _messages(rows: list[dict]) -> str:
    if not rows:
        return f"<p>{_MISSING} &mdash; no conversation was captured.</p>"
    out = []
    for m in rows:
        student = m.get("role") == "user"
        who = "Student" if student else "Tutor"
        out.append(
            f'<div class="msg {"student" if student else "tutor"}">'
            f'<div class="who">{who} &middot; {_esc(m.get("phase") or "")} '
            f'&middot; {_when(m.get("created_at"))}</div>'
            f'<div class="body">{_esc(m.get("content"))}</div></div>')
    return "".join(out)


def _submissions(rows: list[dict]) -> str:
    if not rows:
        return f"<p>{_MISSING} &mdash; no code was submitted.</p>"
    out = []
    for r in rows:
        step = int(r.get("chunk_index") or 0) + 1
        out.append(
            f'<h4>Step {step} &middot; attempt {_esc(r.get("attempt"))} '
            f'&middot; {_when(r.get("created_at"))}</h4>'
            # The tier is REAL evidence (which check decided this), so it must
            # not wear the "Not recorded" style - grey italic on a value that
            # exists reads as missing data at a glance.
            + f'<p>{_verdict_pill(r.get("verdict"))} '
            + (f'<span class="meta">decided by {_esc(r["tier"])}</span>'
               if r.get("tier") else _MISSING)
            + "</p>"
            + f'<pre>{_esc(r.get("code")) or _MISSING}</pre>')
        if r.get("reason"):
            out.append(f'<p><strong>Feedback shown to the student:</strong> '
                       f'{_esc(r["reason"])}</p>')
    return "".join(out)


def _designs(rows: list[dict]) -> str:
    if not rows:
        return f"<p>{_MISSING} &mdash; no plan was uploaded as a picture.</p>"
    out = []
    for d in rows:
        ok = "approved" if d.get("approved") else "not approved"
        out.append(
            f'<h4>Plan revision {_esc(d.get("round"))} &middot; {_when(d.get("created_at"))}</h4>'
            f'<p><span class="pill {"ok" if d.get("approved") else "bad"}">{ok}</span> '
            f'{_esc(d.get("mime") or "")} &middot; '
            f'{"image stored" if d.get("storage_path") else "<span class=missing>image not stored</span>"}</p>')
        if d.get("reviewer_reply"):
            out.append(f'<p><strong>Reviewer said:</strong> {_esc(d["reviewer_reply"])}</p>')
    return "".join(out)


def _graphs(graphs: dict) -> str:
    def one(kind, label):
        g = graphs.get(kind)
        if not g:
            return f"<h4>{label}</h4><p>{_MISSING}</p>"
        return f"<h4>{label}</h4><pre>{_esc(render_graph(g))}</pre>"
    return one("plan", "Plan graph &mdash; what they said they would do") \
        + one("code", "Code graph &mdash; what they wrote")


def _attempts_table(sessions: list[dict]) -> str:
    if not sessions:
        return f"<p>{_MISSING} &mdash; this problem was never opened.</p>"
    rows = []
    for i, s in enumerate(sessions, 1):
        end = ("Completed" if s.get("completed_at") else "Not finished")
        rows.append(
            f"<tr><td>{i}</td><td>{_when(s.get('started_at')) or _MISSING}</td>"
            f"<td>{_when(s.get('completed_at')) or _MISSING}</td>"
            f"<td>{_esc(end)}</td>"
            f"<td class='n'>{_esc(s.get('total_chunks')) or _MISSING}</td>"
            f"<td>{'yes' if s.get('solved_independently') else 'no'}</td></tr>")
    return ("<table><thead><tr><th>Attempt</th><th>Started</th><th>Finished</th>"
            "<th>End state</th><th class='n'>Steps in this attempt</th>"
            "<th>Solved independently</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


def _problem_evidence(row: dict) -> str:
    """The full record for one student on one problem."""
    return (
        f"<p><strong>Result:</strong> {_pill(row['status'])} "
        f"&middot; <strong>Step credit:</strong> {_credit(row['tally'])}</p>"
        + _attempts_table(row["sessions"])
        + "<h4>Conversation</h4>" + _messages(row["messages"])
        + "<h4>Plans and design reviews</h4>" + _designs(row["designs"])
        + "<h4>Code submissions</h4>" + _submissions(row["submissions"])
        + _graphs(row["graphs"]))


def _problem_context(p: dict) -> str:
    sig = _sig(p)
    return (
        f'<div class="card"><h4>What the student was asked to do</h4>'
        f'<p><strong>Problem:</strong> {_esc(p.get("title") or p["slug"])} '
        f'<span class="missing">({_esc(p["slug"])})</span></p>'
        + (f"<p><strong>Signature:</strong> <code>{_esc(sig)}</code></p>" if sig else "")
        + f'<pre>{_esc(p.get("description")) or _MISSING}</pre></div>')


def _completeness(record: dict) -> str:
    n = {"attempts": 0, "messages": 0, "designs": 0, "submissions": 0, "graphs": 0}
    for s in record["students"]:
        for r in s["problems"]:
            n["attempts"] += len(r["sessions"])
            n["messages"] += len(r["messages"])
            n["designs"] += len(r["designs"])
            n["submissions"] += len(r["submissions"])
            n["graphs"] += len(r["graphs"])
    rows = "".join(f"<tr><td>{k.title()}</td><td class='n'>{v}</td></tr>"
                   for k, v in n.items())
    gaps = "".join(f"<tr><td>{_esc(t)}</td><td>{_esc(d)}</td></tr>"
                   for t, d in NOT_CAPTURED)
    return (
        "<h2>Export completeness</h2>"
        "<p>Everything below came from one read of the record at the generation "
        "time in the header. “Not recorded” means the evidence was never "
        "captured &mdash; it is not a claim that nothing happened.</p>"
        "<table><thead><tr><th>Included records</th><th class='n'>Count</th>"
        "</tr></thead><tbody>" + rows + "</tbody></table>"
        "<h3>Not captured by ACADIA today</h3>"
        "<table><thead><tr><th>Evidence</th><th>Why it is absent</th></tr></thead>"
        "<tbody>" + gaps + "</tbody></table>")


def render_html(record: dict, mode: str = "student") -> tuple[str, str]:
    """(filename, html). mode: "student" (one per student) or "class"."""
    meta = record["assignment"]
    aname = meta.get("name") or record["assignment_id"]
    when = _when(record["generated_at"])
    stamp = record["generated_at"][:10]

    # TWO FORMS OF THE SAME HEADING, and they are not interchangeable. `title`
    # is plain text and gets escaped once on its way into <title>; `heading` is
    # already HTML, so its dynamic parts are escaped HERE and the entity between
    # them is left alone. Escaping the assembled heading turned the em-dash into
    # a visible "&amp;mdash;", and NOT escaping it put an unescaped student name
    # - which students type themselves at registration - straight into the page.
    if mode == "class":
        body = _class_body(record)
        title = f"Class learning record - {aname}"
        heading = f"Class learning record &mdash; {_esc(aname)}"
        sub = (f"Assignment: {_esc(aname)} &middot; Students: "
               f"{len(record['students'])} &middot; Generated: {when}")
        fname = f"ACADIA_Class_{_slugify(aname)}_{stamp}.html"
    else:
        s = record["students"][0]
        body = _student_body(record, s, with_context=True)
        title = f"Student learning record - {s['name']}"
        heading = f"Student learning record &mdash; {_esc(s['name'])}"
        sub = (f"{_esc(s['student'].get('username'))} &middot; Assignment: "
               f"{_esc(aname)} &middot; Generated: {when}")
        fname = f"ACADIA_Student_{_slugify(s['name'])}_{stamp}.html"

    html = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{_esc(title)}</title><style>{_CSS}</style></head><body>"
            f"<div class='toolbar'><strong>{_esc(aname)}</strong>"
            f"<span class='grow'></span>"
            f"<button data-act='open' type='button'>Expand all</button>"
            f"<button data-act='close' type='button'>Collapse all</button>"
            f"<button data-act='print' class='primary' type='button'>Save as PDF</button>"
            f"</div><div class='wrap'>"
            f"<h1>{heading}</h1><p class='sub'>{sub}</p>"
            f"{body}{_completeness(record)}"
            f"</div><script>{_SCRIPT}</script></body></html>")
    return fname, html


def _slugify(text: str) -> str:
    keep = [c if c.isalnum() else "-" for c in str(text)]
    return "".join(keep).strip("-")[:48] or "report"


def _student_summary(s: dict) -> str:
    t = s["totals"]
    pct = t["percent"]
    rows = "".join(
        f"<tr><td>{_esc(r['problem'].get('title') or r['problem']['slug'])}</td>"
        f"<td>{_pill(r['status'])}</td><td>{_credit(r['tally'])}</td>"
        f"<td class='n'>{len(r['sessions'])}</td>"
        f"<td class='n'>{len(r['submissions'])}</td>"
        f"<td>{_when(r['last']) or _MISSING}</td></tr>"
        for r in s["problems"])
    return (
        f"<div class='card'><p><strong>Step credit:</strong> "
        f"{t['solved']} / {t['available'] or '?'}"
        + (f" &middot; {pct}%" if pct is not None else "")
        + f"</p><div class='bar'><i style='width:{pct or 0}%'></i></div>"
        f"<p class='sub' style='margin-top:10px'>{t['solved']} solved &middot; "
        f"{t['shown']} answer shown &middot; {t['missed']} not done</p></div>"
        "<table><thead><tr><th>Problem</th><th>Status</th><th>Step credit</th>"
        "<th class='n'>Attempts</th><th class='n'>Submissions</th>"
        "<th>Last activity</th></tr></thead><tbody>" + rows + "</tbody></table>")


def _student_body(record: dict, s: dict, with_context: bool) -> str:
    out = ["<h2>At a glance</h2>", _student_summary(s)]
    for i, r in enumerate(s["problems"], 1):
        p = r["problem"]
        out.append(f"<h2>Problem {i}: {_esc(p.get('title') or p['slug'])}</h2>")
        if with_context:
            out.append(_problem_context(p))
        out.append("<details open><summary>Full record &mdash; "
                   f"{len(r['sessions'])} attempt(s), {len(r['submissions'])} "
                   f"submission(s), {len(r['messages'])} message(s)</summary>"
                   f"<div class='inner'>{_problem_evidence(r)}</div></details>")
    return "".join(out)


def _class_body(record: dict) -> str:
    students = record["students"]
    total_solved = sum(s["totals"]["solved"] for s in students)
    total_avail = sum(s["totals"]["available"] for s in students)
    active = sum(1 for s in students
                 if any(r["sessions"] or r["submissions"] for r in s["problems"]))
    pooled = percent(total_solved, total_avail)

    index = "".join(
        f"<tr><td>{_esc(s['name'])}</td>"
        f"<td>{_esc(s['student'].get('username'))}</td>"
        f"<td class='n'>{sum(1 for r in s['problems'] if r['status'] in ('solved','helped'))}</td>"
        f"<td class='n'>{sum(1 for r in s['problems'] if r['status']=='progress')}</td>"
        f"<td class='n'>{sum(1 for r in s['problems'] if r['status']=='todo')}</td>"
        f"<td>{s['totals']['solved']} / {s['totals']['available'] or '?'}</td></tr>"
        for s in students)

    out = [
        "<h2>Class at a glance</h2>",
        f"<div class='card'><table><tbody>"
        f"<tr><td>Students in export</td><td class='n'>{len(students)}</td></tr>"
        f"<tr><td>Students with recorded activity</td>"
        f"<td class='n'>{active} / {len(students)}</td></tr>"
        f"<tr><td>Pooled step credit</td><td class='n'>{total_solved} / "
        f"{total_avail or '?'}" + (f" &middot; {pooled}%" if pooled is not None else "")
        + "</td></tr></tbody></table>"
        "<p class='sub'>Pooled credit is every earned step over every available "
        "step. It is not the average of the students&rsquo; percentages.</p></div>",
        "<h3>Student index</h3>",
        "<table><thead><tr><th>Student</th><th>Identity</th>"
        "<th class='n'>Completed</th><th class='n'>In progress</th>"
        "<th class='n'>Not started</th><th>Step credit</th></tr></thead>"
        "<tbody>" + (index or "<tr><td colspan='6'>No students.</td></tr>")
        + "</tbody></table>",
    ]

    # Problem first, then every student under it - the reading order that makes
    # "how did the class handle THIS task" answerable at a glance.
    for i, p in enumerate(record["problems"], 1):
        out.append(f"<h2>Problem {i}: {_esc(p.get('title') or p['slug'])}</h2>")
        out.append(_problem_context(p))
        rows = "".join(
            f"<tr><td>{_esc(s['name'])}</td><td>{_pill(r['status'])}</td>"
            f"<td>{_credit(r['tally'])}</td>"
            f"<td class='n'>{len(r['sessions'])}</td>"
            f"<td class='n'>{len(r['submissions'])}</td>"
            f"<td>{_when(r['last']) or _MISSING}</td></tr>"
            for s in record["students"]
            for r in s["problems"] if r["problem"]["slug"] == p["slug"])
        out.append("<h3>How the class did</h3><table><thead><tr><th>Student</th>"
                   "<th>Status</th><th>Step credit</th><th class='n'>Attempts</th>"
                   "<th class='n'>Submissions</th><th>Last activity</th>"
                   "</tr></thead><tbody>" + rows + "</tbody></table>")
        out.append("<h3>Full evidence, student by student</h3>")
        for s in record["students"]:
            r = next((x for x in s["problems"]
                      if x["problem"]["slug"] == p["slug"]), None)
            if r is None:
                continue
            out.append(
                f"<details><summary>{_esc(s['name'])} &mdash; "
                f"{len(r['sessions'])} attempt(s), {len(r['submissions'])} "
                f"submission(s)</summary><div class='inner'>"
                f"{_problem_evidence(r)}</div></details>")
    return "".join(out)
