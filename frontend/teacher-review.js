/* One problem, reviewed: who is stuck on it, and, for one student, exactly
   where. Opened from a problem tile on the dashboard (teacher-insights.js).

   Two views, both chosen by the URL so each can be bookmarked, opened in a new
   tab and gone back from:

     ?slug=…                 every student with an open issue, as a plain
                             vertical list: scanning a class means reading
                             down one column, not opening one fold at a time.
     ?slug=…&student=…       that student's issue, the immersive view: the step,
                             what it asked, what they were told, the cases it
                             failed on, their whole function up to that step
                             with the failing attempt in red, and every attempt.

   "Issue seen" is on both. It takes the student out of the problem's counter
   until they get something wrong again; main/teacher_dashboard.py explains why
   it is a timestamp rather than a flag. Nothing here grades anything. */
(() => {
  requireSession("teacher");
  mountHeader({variant: "instructor", active: "Home"});
  const q = new URLSearchParams(location.search);
  const slug = q.get("slug") || "", assignment = q.get("assignment") || "";
  const student = q.get("student") || "", step = q.get("step") || "";
  const host = document.getElementById("reviewHost");
  const notice = document.getElementById("reviewNotice");
  const count = (n, noun = "student") => `${n} ${noun}${n === 1 ? "" : "s"}`;
  const when = iso => {
    const d = new Date(iso);
    return iso && !isNaN(d) ? d.toLocaleString(undefined, {month: "short", day: "numeric", hour: "numeric", minute: "2-digit"}) : "";
  };
  const link = (extra = {}) => "teacher-review.html?" + new URLSearchParams(
    {slug, ...(assignment ? {assignment} : {}), ...extra});
  // Same column the student page draws a function body at (student.js).
  const BODY_INDENT = 4;

  function fail(message) {
    host.setAttribute("aria-busy", "false");
    host.innerHTML = `<div class="insight-empty"><p>${esc(message)}</p><a class="insight-link" href="teacher.html">Back to the dashboard</a></div>`;
  }

  async function setSeen(studentId, seen, button) {
    if (button) setBusy(button, true, seen ? "Marking…" : "Undoing…");
    try {
      const r = await fetch(`${API}/teacher/issues/seen`, {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({slug, student_id: studentId, seen})});
      if (!r.ok) throw new Error();
      notice.replaceChildren();
      await (student ? showStudent : showList)();
    } catch {
      notice.innerHTML = `<div class="banner warn">That could not be saved. Nothing was changed.</div>`;
      if (button) setBusy(button, false);
    }
  }

  const seenButton = (studentId, seen) =>
    `<button class="${seen ? "ghost " : ""}seen-toggle" type="button" data-seen="${esc(studentId)}" data-now="${seen}" aria-pressed="${seen}">${seen ? "Seen · undo" : "Issue seen"}</button>`;
  function wireSeen() {
    host.querySelectorAll("[data-seen]").forEach(b =>
      b.onclick = () => setSeen(b.dataset.seen, b.dataset.now !== "true", b));
  }

  /* ── every student on this problem ─────────────────────────────────── */
  async function showList() {
    const r = await fetch(`${API}/teacher/dashboard${assignment ? `?assignment_id=${encodeURIComponent(assignment)}` : ""}`, {cache: "no-store"});
    if (!r.ok) return fail("This problem could not be loaded. Try again in a moment.");
    const data = await r.json();
    const p = (data.problems || []).find(x => x.slug === slug);
    if (!p) return fail("This problem is not in any published assignment.");
    const aname = (data.assignments || []).find(a => String(a.id) === String(p.assignment_id))?.name || "";
    document.title = `${p.title || p.slug} · Review · ACADIA`;
    setCrumbs([{label: "Dashboard", go: () => location.assign("teacher.html")}, {label: p.title || p.slug}]);
    const open = p.follow_up.filter(s => !s.seen), seen = p.follow_up.filter(s => s.seen);
    const row = s => {
      const first = (s.details && s.details[0]) || {number: s.steps[0], prompt: ""};
      const steps = s.steps.length > 1 ? `Steps ${s.steps.join(", ")}` : `Step ${first.number}`;
      return `<li class="review-student${s.seen ? " is-seen" : ""}">
        <a class="review-student-link" href="${link({student: s.student_id})}">
          <span class="rs-name">${esc(s.name)}${s.seen ? `<span class="rs-badge">Seen</span>` : ""}</span>
          <span class="rs-step">${steps}${first.prompt ? `: ${esc(undash(first.prompt))}` : ""}</span>
          <span class="rs-reason">${esc(undash(s.reason))}</span>
          <span class="rs-when">Last wrong answer ${esc(when(s.last_activity))}</span>
        </a>${seenButton(s.student_id, s.seen)}</li>`;
    };
    host.innerHTML = `<a class="back-link" href="teacher.html">← Dashboard</a>
      <header class="review-head"><p class="eyebrow">${esc(aname)}</p><h1 tabindex="-1">${esc(p.title || p.slug)}</h1>
        <p class="sub">${open.length ? `${count(open.length)} with an open issue` : "No open issues"}${seen.length ? ` · ${seen.length} seen` : ""} · ${count(p.attempted)} tried</p></header>
      ${p.follow_up.length
        ? `<p class="hint">Choose a student to see the step, their code and what went wrong.</p><ul class="review-students">${[...open, ...seen].map(row).join("")}</ul>`
        : `<div class="insight-empty"><p>No students have an uncorrected answer on this problem.</p></div>`}`;
    host.setAttribute("aria-busy", "false");
    wireSeen();
  }

  /* ── one student, immersive ────────────────────────────────────────── */
  // Their function as it stood: the header, every accepted step, then the
  // failing attempt, one row per line. Rows carry the step they came from, and
  // the failing rows are marked in text as well as colour.
  function listing(r) {
    const rows = [];
    if (r.header) rows.push({text: r.header, kind: "head"});
    for (const s of r.prefix) String(s.code || "").split("\n").forEach((l, i) =>
      rows.push({text: l ? " ".repeat(BODY_INDENT) + l : "", kind: "ok", step: i ? "" : s.number}));
    String(r.step.code || "").split("\n").forEach((l, i) =>
      rows.push({text: l ? " ".repeat(BODY_INDENT) + l : "", kind: "fail", step: i ? "" : r.step.number}));
    const first = rows.findIndex(x => x.kind === "fail") + 1, last = rows.length;
    return `<p class="hint" id="listingNote">${first === last ? `Line ${first} is` : `Lines ${first} to ${last} are`} the attempt at step ${r.step.number} that failed, highlighted in red.</p>
      <div class="code-listing" role="group" aria-label="${esc(r.name)}’s function up to step ${r.step.number}" aria-describedby="listingNote">${rows.map((x, i) =>
        `<div class="cl-row cl-${x.kind}"><span class="cl-n" aria-hidden="true">${i + 1}</span><span class="cl-step" aria-hidden="true">${x.step ? `S${x.step}` : ""}</span><code>${esc(x.text) || " "}</code>${x.kind === "fail" ? `<span class="sr-only"> (failing attempt)</span>` : ""}</div>`).join("")}</div>`;
  }

  async function showStudent() {
    const r0 = await fetch(`${API}/teacher/review?slug=${encodeURIComponent(slug)}&student_id=${encodeURIComponent(student)}${step ? `&step=${encodeURIComponent(step)}` : ""}`, {cache: "no-store"});
    if (!r0.ok) return fail(r0.status === 404 ? "No saved work for this student on this problem." : "This student’s work could not be loaded. Try again in a moment.");
    const r = await r0.json();
    document.title = `${r.name} · ${r.title} · ACADIA`;
    setCrumbs([{label: "Dashboard", go: () => location.assign("teacher.html")},
               {label: r.title, go: () => location.assign(link())}, {label: r.name}]);
    const back = `<a class="back-link" href="${link()}">← All students on ${esc(r.title)}</a>`;
    if (!r.step) {
      host.innerHTML = `${back}<header class="review-head"><p class="eyebrow">${esc(r.assignment)} · ${esc(r.title)}</p><h1 tabindex="-1">${esc(r.name)}</h1></header>
        <div class="insight-empty"><p>${esc(r.name)} has no uncorrected answers on this problem.</p></div>`;
      host.setAttribute("aria-busy", "false");
      return;
    }
    const s = r.step, tries = s.attempts.length, wrong = s.attempts.filter(a => a.verdict === "incorrect").length;
    const shown = s.failing_cases.length, total = s.failed_total || shown;
    const download = `${API}/teacher/assignments/${encodeURIComponent(r.assignment_id)}/transcript/${encodeURIComponent(r.student_id)}`;
    host.innerHTML = `${back}
      <header class="review-head review-hero">
        <div><p class="eyebrow">${esc(r.assignment)} · ${esc(r.title)}</p><h1 tabindex="-1">${esc(r.name)}</h1>
          <p class="sub">Stuck on step ${s.number}${r.total_steps ? ` of ${r.total_steps}` : ""} · ${count(tries, "attempt")} on this step, ${wrong} wrong · first tried ${esc(when(s.first_at))}, last wrong ${esc(when(s.last_at))}</p></div>
        <div class="review-actions">${seenButton(r.student_id, r.seen)}<a class="ghost button-link" href="${download}" download>Download their work ↓</a></div>
      </header>
      ${r.open_steps.length > 1 ? `<nav class="step-switch" aria-label="Other open steps">Also stuck on: ${r.open_steps.filter(n => n !== s.number).map(n => `<a href="${link({student: r.student_id, step: n})}">Step ${n}</a>`).join(" ")}</nav>` : ""}
      <div class="review-grid">
        <section class="review-card"><p class="feedback-label">What step ${s.number} asked</p><p class="review-prompt">${s.prompt ? esc(undash(s.prompt)) : "The wording of this step was not recorded."}</p></section>
        <section class="review-card is-warn"><p class="feedback-label">What they were told</p><p>${esc(undash(s.reason))}</p></section>
      </div>
      ${shown ? `<section class="review-card"><h2>Where it goes wrong</h2><p class="hint">${total > shown ? `${shown} of the ${total} cases it failed, ` : "The cases it failed, "}exactly as the student saw them.</p>${s.failing_cases.map(c => `<pre class="code">${esc(c)}</pre>`).join("")}</section>` : ""}
      <section class="review-card"><h2>Their function up to step ${s.number}</h2>${listing(r)}</section>
      <section class="review-card"><h2>Every attempt at step ${s.number}</h2>
        <ol class="attempt-list">${s.attempts.map((a, i) => `<li class="attempt ${esc(a.verdict || "")}">
          <div class="attempt-head"><strong>Attempt ${i + 1}</strong><span class="attempt-verdict">${a.verdict === "incorrect" ? "Wrong" : a.verdict === "correct" ? "Correct" : "Not checked"}</span><span class="attempt-when">${esc(when(a.at))}</span></div>
          ${a.reason ? `<p>${esc(undash(a.reason))}</p>` : ""}
          <pre class="code attempt-code">${esc(a.code)}</pre></li>`).join("")}</ol>
      </section>`;
    host.setAttribute("aria-busy", "false");
    wireSeen();
  }

  if (!slug) return fail("Choose a problem from the dashboard to review it.");
  (student ? showStudent : showList)().catch(() => fail("This page could not be loaded. Try again in a moment."));
})();
