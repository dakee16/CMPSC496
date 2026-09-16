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
    const difficult = p.steps.filter(s => s.needs_help);
    return `<section class="insight-detail" aria-labelledby="insightDetailTitle">
      <div class="insight-detail-head"><div><p class="eyebrow">A closer look</p><h3 id="insightDetailTitle" tabindex="-1">${esc(p.title || p.slug)}</h3><p class="hint">${esc(assignmentName(p))}</p></div><button class="ghost" id="closeInsightDetail" type="button">Close details</button></div>
      <div class="insight-detail-grid"><div><h4>Where are they getting stuck?</h4>
        <p class="insight-explanation">Students write their solution one small step at a time. These steps still have incorrect answers.</p>
        ${difficult.length ? `<ol class="insight-steps">${difficult.map(s => `<li><span class="step-number" aria-hidden="true">${s.number}</span><div><strong>Step ${s.number}</strong><span>${count(s.needs_help)} may need help</span></div></li>`).join("")}</ol>` : `<p class="insight-clear">No uncorrected answers in the saved work.</p>`}
        ${p.recovered ? `<p class="insight-recovery">${count(p.recovered)} corrected their earlier mistakes.</p>` : ""}
        <p class="insight-footnote">Students may use different approaches, so the same step number can cover different code.</p>
        <a class="insight-link" href="${gradeLink(p)}">View grades for this assignment <span aria-hidden="true">↗</span></a>
      </div><div><h4>Who may need help?</h4>
        <p class="insight-explanation">Choose a student to see feedback you can discuss with them.</p>
        ${p.follow_up.length ? `<ul class="insight-students">${p.follow_up.map(s => `<li><details><summary><strong>${esc(s.name)}</strong><span>Check step${s.steps.length === 1 ? "" : "s"} ${s.steps.join(", ")}</span></summary><div class="student-feedback"><p class="feedback-label">Feedback to discuss</p><p>${esc(s.reason)}</p><a class="insight-link" href="${API}/teacher/assignments/${encodeURIComponent(p.assignment_id)}/transcript/${encodeURIComponent(s.student_id)}" download>Download this student’s work <span aria-hidden="true">↓</span></a></div></details></li>`).join("")}</ul>` : `<p class="insight-clear">No students to follow up with on this problem. They may still have unfinished work.</p>`}
      </div></div>
    </section>`;
  }

  function render() {
    const {summary: s, problems} = snapshot;
    const tried = problems.filter(p => p.attempted);
    const visible = expanded ? tried : tried.slice(0, 6);
    if (!tried.some(p => p.slug === selected)) selected = null;
    const focus = tried.find(p => p.slug === selected);
    const top = tried[0];
    const max = Math.max(1, ...tried.map(p => p.needs_help));
    let title = "Waiting for student work";
    let takeaway = "Once students start submitting code, you’ll see which problems they may need help with.";
    if (top?.needs_help) {
      title = `Review ${esc(top.title || top.slug)}`;
      takeaway = `${count(top.needs_help)} may need help with this problem, the most in this selection. A short walkthrough could help them move forward.`;
    } else if (top) {
      title = "No uncorrected answers so far";
      takeaway = "There are no saved mistakes still waiting to be corrected. Students may still have problems to finish or start.";
    }
    host.innerHTML = `<section class="insight-takeaway" aria-labelledby="teachingSuggestion"><div><p class="eyebrow">Suggested next step</p><h3 id="teachingSuggestion">${title}</h3><p>${takeaway}</p></div>${top?.needs_help ? `<button id="reviewPriority" type="button">See who needs help <span aria-hidden="true">→</span></button>` : ""}</section>
      <div class="insight-board"><div class="insight-chart">
        <div class="insight-chart-heading"><h3>Which problems need a review?</h3></div>
        <p class="insight-explanation" id="chartExplanation"><strong>“May need help” means a student submitted an incorrect answer and hasn’t corrected it yet.</strong> Longer bars mean more students may need help.</p>
        ${tried.length ? `<p class="insight-chart-instruction">Choose a problem to see the students and the steps they’re having trouble with.</p><div class="insight-bars" role="group" aria-describedby="chartExplanation">${visible.map(p => `<button class="insight-bar-row" type="button" data-insight-problem="${esc(p.slug)}" aria-expanded="${p.slug === selected}" aria-controls="insightFocus" aria-label="${esc(p.title || p.slug)}: ${count(p.needs_help)} may need help. ${count(p.attempted)} tried this problem. Show details."><span class="insight-bar-label"><strong>${esc(p.title || p.slug)}</strong><small>${esc(assignmentName(p))}</small></span><span class="insight-track" aria-hidden="true"><i style="width:${p.needs_help / max * 100}%"></i></span><span class="insight-bar-count"><strong>${p.needs_help ? `${count(p.needs_help)} may need help` : "No uncorrected answers"}</strong><small>${count(p.attempted)} tried this problem</small></span><span class="insight-row-arrow" aria-hidden="true">→</span></button>`).join("")}</div>
        ${tried.length > 6 ? `<button class="ghost insight-show" id="showAllInsights" aria-expanded="${expanded}">${expanded ? "Show fewer problems" : `Show all ${tried.length} problems`}</button>` : ""}` : `<div class="insight-empty"><h3>${s.problems ? "No answers to review yet" : "Publish an assignment to get started"}</h3><p>${s.problems ? "The chart will appear after students submit their first answers." : "Add and publish an assignment in the Assignments tab. Student progress will appear here."}</p>${s.problems ? "" : `<a class="insight-link" href="teacher-assignments.html">Go to Assignments →</a>`}</div>`}
      </div></div>
      <div id="insightFocus"${focus ? "" : " hidden"}>${detail(focus)}</div>
      <details class="insight-method"><summary>How are these numbers worked out?</summary>
        <ul><li><strong>May need help:</strong> a student has at least one incorrect answer they haven’t corrected yet. This is a reason to check in, not a grade.</li><li><strong>Tried this problem:</strong> a student submitted code and received a correct or incorrect result. Simply opening a problem doesn’t count.</li><li>A student counts once per problem, even if they try many times. We use their most recent attempt at each problem; restarting begins a new attempt.</li><li>Only assignments currently available to students are included. Problems with no checked answers aren’t shown in the chart.</li></ul>
        ${s.indeterminate ? `<p>${count(s.indeterminate, "submission")} could not be checked. ${s.indeterminate === 1 ? "It is" : "They are"} left out of the chart and not counted as a student mistake.</p>` : ""}
      </details>`;
    host.querySelectorAll("[data-insight-problem]").forEach(button => {
      button.onclick = () => select(button.dataset.insightProblem);
    });
    const show = document.getElementById("showAllInsights");
    if (show) show.onclick = () => {expanded = !expanded; render(); document.getElementById("showAllInsights").focus();};
    const review = document.getElementById("reviewPriority");
    if (review) review.onclick = () => select(top.slug);
    wireDetail();
  }

  function wireDetail() {
    const close = document.getElementById("closeInsightDetail");
    if (close) close.onclick = () => {
      const previous = selected;
      selected = null;
      document.getElementById("insightFocus").hidden = true;
      let returnFocus = document.getElementById("showAllInsights");
      host.querySelectorAll("[data-insight-problem]").forEach(button => {
        button.setAttribute("aria-expanded", "false");
        if (button.dataset.insightProblem === previous) returnFocus = button;
      });
      returnFocus?.focus();
    };
  }

  function select(slug) {
    selected = slug;
    host.querySelectorAll("[data-insight-problem]").forEach(button => button.setAttribute("aria-expanded", String(button.dataset.insightProblem === slug)));
    const focus = document.getElementById("insightFocus");
    focus.innerHTML = detail(snapshot.problems.find(p => p.slug === slug));
    focus.hidden = false;
    wireDetail();
    document.getElementById("insightDetailTitle")?.focus();
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
      pick.innerHTML = `<option value="">All assignments</option>` + data.assignments.map(a => `<option value="${esc(a.id)}">${esc(a.name)}</option>`).join("");
      pick.value = filter;
      pick.disabled = !data.assignments.length;
      if (!sameScope) {selected = null; expanded = false;}
      render();
      const at = new Date(data.generated_at);
      document.getElementById("insightUpdated").textContent = `Updated ${at.toLocaleString(undefined, {month: "short", day: "numeric", hour: "numeric", minute: "2-digit"})}.`;
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
