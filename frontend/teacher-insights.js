/* Counts come from saved grading records. This view never grades student work. */
(() => {
  requireSession("teacher");
  if (location.hash === "#list") {location.replace("teacher-assignments.html"); return;}
  mountHeader({variant: "instructor", active: "Home"});
  const host = document.getElementById("classInsights");
  const pick = document.getElementById("insightAssignment");
  const refresh = document.getElementById("refreshInsights");
  const notice = document.getElementById("insightNotice");
  let snapshot = null, selected = null, request = 0, expanded = false;
  const count = (n, noun = "student") => `${n} ${noun}${n === 1 ? "" : "s"}`;
  const assignmentName = p => snapshot.assignments.find(a => String(a.id) === String(p.assignment_id))?.name || "Assignment";
  const gradeLink = p => `grades.html?assignment=${encodeURIComponent(p.assignment_id)}`;

  function detail(p) {
    if (!p) return "";
    const difficult = p.steps.filter(s => s.needs_help || s.recovered);
    return `<section class="insight-detail" aria-labelledby="insightDetailTitle">
      <div class="insight-detail-head"><div><p class="eyebrow">Problem focus</p><h3 id="insightDetailTitle" tabindex="-1">${esc(p.title || p.slug)}</h3><p class="hint">${esc(assignmentName(p))}</p></div><a class="insight-link" href="${gradeLink(p)}">Open gradebook <span aria-hidden="true">↗</span></a></div>
      <div class="insight-detail-grid"><div><h4>Steps worth revisiting</h4>
        ${difficult.length ? `<ol class="insight-steps">${difficult.map(s => `<li><span class="step-number">${s.number}</span><div><strong>Step ${s.number}</strong><span>${count(s.needs_help)} without a passing answer · ${s.recovered} passed after retry</span></div><span class="step-reach">${s.attempted} tried</span></li>`).join("")}</ol>` : `<p class="hint">No incorrect answers recorded for this problem.</p>`}
        <p class="insight-footnote">Step numbers describe positions in a solution. Students may take different approaches; use their feedback to guide the review.</p>
      </div><div><h4>Students to check in with <span class="insight-count">${p.needs_help}</span></h4>
        ${p.follow_up.length ? `<ul class="insight-students">${p.follow_up.map(s => `<li><div class="followup-heading"><strong>${esc(s.name)}</strong><span>Step${s.steps.length === 1 ? "" : "s"} ${s.steps.join(", ")}</span></div><p>${esc(s.reason)}</p><a class="insight-link" href="${API}/teacher/assignments/${encodeURIComponent(p.assignment_id)}/transcript/${encodeURIComponent(s.student_id)}" download>Download work history <span aria-hidden="true">↓</span></a></li>`).join("")}</ul>` : `<div class="insight-clear"><strong>No outstanding misses.</strong><p>Everyone who missed a step has a recorded pass for it in their latest session.</p></div>`}
      </div></div>
    </section>`;
  }

  function render() {
    const {summary: s, problems} = snapshot;
    const tried = problems.filter(p => p.attempted);
    const visible = expanded ? tried : tried.slice(0, 6);
    if (!tried.some(p => p.slug === selected)) selected = tried[0]?.slug;
    const focus = tried.find(p => p.slug === selected);
    const top = tried[0];
    const max = Math.max(1, ...tried.map(p => p.attempted));
    let takeaway = "Student practice will reveal the first teaching priorities.";
    let title = "Waiting for practice";
    if (top?.needs_help) {
      title = "Start the next check-in here";
      takeaway = `<strong>${esc(top.title || top.slug)}</strong> has ${count(top.needs_help)} with missed steps and no passing answer yet, out of ${top.attempted} who tried it.`;
    } else if (top) {
      title = "No outstanding misses";
      takeaway = "Students have passed every step they previously missed in their latest sessions. Unfinished and unattempted steps may still remain.";
    }
    host.innerHTML = `<div class="insight-board">
      <div class="insight-chart"><div class="insight-chart-heading"><h3>Difficulty by problem</h3><span>Number of students</span></div>
        <div class="insight-legend"><span><i class="needs-help"></i>No passing answer yet</span><span><i class="recovered"></i>Passed missed steps</span><span><i class="no-miss"></i>No recorded misses</span></div>
        ${tried.length ? `<div class="insight-chart-axis" aria-hidden="true"><span>0</span><span>${max} students</span></div><div class="insight-bars">${visible.map(p => `<button class="insight-bar-row" type="button" data-insight-problem="${esc(p.slug)}" aria-pressed="${p.slug === selected}" aria-controls="insightFocus" aria-label="${esc(p.title || p.slug)}: ${p.needs_help} without a passing answer, ${p.recovered} passed missed steps, ${p.attempted} students tried. Show details."><span class="insight-bar-label"><strong>${esc(p.title || p.slug)}</strong><small>${esc(assignmentName(p))}</small></span><span class="insight-track" aria-hidden="true"><i class="needs-help" style="width:${p.needs_help / max * 100}%"></i><i class="recovered" style="width:${p.recovered / max * 100}%"></i><i class="no-miss" style="width:${(p.attempted - p.needs_help - p.recovered) / max * 100}%"></i></span><span class="insight-bar-count" aria-hidden="true"><b>${p.needs_help}</b> / ${p.attempted}<small>need follow-up</small></span></button>`).join("")}</div>
        ${tried.length > 6 ? `<button class="ghost insight-show" id="showAllInsights" aria-expanded="${expanded}">${expanded ? "Show priority problems" : `Show all ${tried.length} problems`}</button>` : ""}
        <p class="insight-footnote">Sorted by students needing follow-up. Each student counts once per problem. Select a row to see the steps and feedback.</p>` : `<div class="insight-empty"><span aria-hidden="true">↗</span><h3>${s.problems ? "The first attempts will tell the story." : "Publish an assignment to get started."}</h3><p>${s.problems ? "Once students submit code, this chart will show which problems need a class review. Attempts and feedback are saved as they work." : "Ready problems in published assignments appear here as students start practicing."}</p></div>`}
      </div>
      <aside class="insight-takeaway"><p class="eyebrow">Teaching focus</p><h3>${title}</h3><p>${takeaway}</p>${top?.needs_help ? `<p class="insight-focus-step">Most affected position <strong>Step ${top.steps[0].number}</strong><span>${count(top.steps[0].needs_help)} without a passing answer</span></p><button class="ghost" id="reviewPriority" type="button">Review this problem <span aria-hidden="true">→</span></button>` : ""}<p class="insight-footnote">Based on saved outcomes, not a prediction or a grade.</p></aside>
      <dl class="insight-metrics"><div><dt>Students who started</dt><dd>${s.active}<span> / ${s.students} registered</span></dd></div><div><dt>Students to check in with</dt><dd>${s.needs_help}<span> across these problems</span></dd></div><div><dt>Problems attempted</dt><dd>${s.attempted_problems}<span> / ${s.problems} ready</span></dd></div></dl>
    </div>
    ${s.indeterminate ? `<p class="insight-system-note">${count(s.indeterminate, "submission")} could not be graded. These are excluded from difficulty counts.</p>` : ""}
    <div id="insightFocus">${detail(focus)}</div>`;
    host.querySelectorAll("[data-insight-problem]").forEach(button => {
      button.onclick = () => select(button.dataset.insightProblem);
    });
    const show = document.getElementById("showAllInsights");
    if (show) show.onclick = () => {expanded = !expanded; render(); document.getElementById("showAllInsights").focus();};
    const review = document.getElementById("reviewPriority");
    if (review) review.onclick = () => {
      if (!visible.some(p => p.slug === top.slug)) expanded = false;
      select(top.slug);
      document.getElementById("insightDetailTitle")?.focus();
    };
  }

  function select(slug) {
    selected = slug;
    // Keep the selected button in the DOM so keyboard focus is not lost.
    host.querySelectorAll("[data-insight-problem]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.insightProblem === slug)));
    document.getElementById("insightFocus").innerHTML = detail(snapshot.problems.find(p => p.slug === slug));
  }

  async function load() {
    const id = ++request, filter = pick.value;
    const sameScope = snapshot && (snapshot.assignment_id || "") === filter;
    host.setAttribute("aria-busy", "true");
    refresh.disabled = true;
    refresh.textContent = "Refreshing…";
    notice.textContent = "";
    if (!sameScope) {
      snapshot = null;
      host.innerHTML = `<div class="insight-empty" role="status">Loading class insights…</div>`;
      document.getElementById("insightUpdated").textContent = "";
    }
    try {
      const response = await fetch(`${API}/teacher/dashboard${filter ? `?assignment_id=${encodeURIComponent(filter)}` : ""}`, {cache: "no-store"});
      if (!response.ok) throw new Error("Class insights could not be loaded.");
      const data = await response.json();
      if (id !== request) return;
      snapshot = data;
      pick.innerHTML = `<option value="">All published assignments</option>` + data.assignments.map(a => `<option value="${esc(a.id)}">${esc(a.name)}</option>`).join("");
      pick.value = filter;
      pick.disabled = !data.assignments.length;
      if (!sameScope) {selected = null; expanded = false;}
      render();
      const at = new Date(data.generated_at);
      document.getElementById("insightUpdated").textContent = `Updated ${at.toLocaleString(undefined, {month: "short", day: "numeric", hour: "numeric", minute: "2-digit"})} · All time, latest sessions. Earlier work remains in student work histories.`;
    } catch {
      if (id !== request) return;
      notice.innerHTML = `<div class="banner warn">${sameScope ? "Could not refresh. Showing the last successful update." : "Class insights are temporarily unavailable."} <button id="retryInsights" class="ghost" type="button">Try again</button></div>`;
      if (!sameScope) host.innerHTML = "";
      document.getElementById("retryInsights").onclick = load;
    } finally {
      if (id === request) {
        host.setAttribute("aria-busy", "false");
        refresh.disabled = false;
        refresh.textContent = "Refresh";
      }
    }
  }
  pick.onchange = load;
  refresh.onclick = load;
  load();
})();
