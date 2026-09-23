/* Counts come from saved grading records. This view never grades student work. */
(() => {
  requireSession("teacher");
  if (location.hash === "#list") {location.replace("teacher-assignments.html"); return;}
  mountHeader({variant: "instructor", active: "Home"});
  const host = document.getElementById("classInsights");
  const pick = document.getElementById("insightAssignment");
  const refresh = document.getElementById("refreshInsights");
  const notice = document.getElementById("insightNotice");
  let snapshot = null, request = 0;
  const count = (n, noun = "student") => `${n} ${noun}${n === 1 ? "" : "s"}`;
  const assignmentName = p => snapshot.assignments.find(a => String(a.id) === String(p.assignment_id))?.name || "Assignment";

  // ONE TILE PER PROBLEM, and each is a LINK to that problem's own review page
  // (teacher-review.html) rather than a panel unfolding under the list: a page
  // can be opened in a new tab, bookmarked, and gone back from. A tile is
  // highlighted while any student on it has an open, unseen issue; "Mark
  // done" on the review page is what brings its count back down to zero.
  const reviewLink = p => `teacher-review.html?slug=${encodeURIComponent(p.slug)}&assignment=${encodeURIComponent(p.assignment_id)}`;
  function tile(p) {
    const open = p.needs_help || 0;
    const status = open ? `${count(open)} need${open === 1 ? "s" : ""} help` : "No open issues";
    return `<li><a class="problem-tile${open ? " needs-attention" : ""}" href="${reviewLink(p)}" data-insight-problem="${esc(p.slug)}" aria-label="${esc(p.title || p.slug)}, ${esc(assignmentName(p))}: ${status}.">
      <span class="tile-name">${esc(p.title || p.slug)}</span>
      <span class="tile-assignment" title="${esc(assignmentName(p))}">${esc(assignmentName(p))}</span>
      <span class="tile-count"><strong>${open}</strong><span>${open === 1 ? "student needs help" : "students need help"}</span></span>
      <span class="tile-foot">${count(p.attempted)} tried${p.seen ? ` · ${p.seen} done` : ""}</span>
    </a></li>`;
  }

  function render() {
    const {summary: s, problems} = snapshot;
    const tried = problems.filter(p => p.attempted);
    host.innerHTML = `<div class="insight-board"><div class="insight-chart">
        <div class="insight-chart-heading"><h3>Which problems need a review?</h3></div>
        <p class="insight-explanation" id="chartExplanation"><strong>A highlighted tile has students with an incorrect answer they haven’t corrected yet.</strong> Open a problem to see who they are and exactly where they went wrong.</p>
        ${tried.length ? `<ul class="problem-tiles" aria-describedby="chartExplanation">${tried.map(tile).join("")}</ul>` : `<div class="insight-empty"><h3>${s.problems ? "No answers to review yet" : "Publish an assignment to get started"}</h3><p>${s.problems ? "Problems will appear here after students submit their first answers." : "Add and publish an assignment in the Assignments tab. Student progress will appear here."}</p>${s.problems ? "" : `<a class="insight-link" href="teacher-assignments.html">Go to Assignments →</a>`}</div>`}
      </div></div>
      <details class="insight-method"><summary>How are these numbers worked out?</summary>
        <ul><li><strong>Needs help:</strong> a student has at least one incorrect answer they haven’t corrected yet. This is a reason to check in, not a grade.</li><li><strong>Marked done:</strong> you marked the student done on the problem’s page. It leaves the count until the student gets something wrong again.</li><li><strong>Tried:</strong> a student submitted code and received a correct or incorrect result. Simply opening a problem doesn’t count.</li><li>A student counts once per problem, even if they try many times. We use their most recent attempt at each problem; restarting begins a new attempt.</li><li>Only assignments currently available to students are included. Problems with no checked answers aren’t shown.</li></ul>
        ${s.indeterminate ? `<p>${count(s.indeterminate, "submission")} could not be checked. ${s.indeterminate === 1 ? "It is" : "They are"} left out and not counted as a student mistake.</p>` : ""}
      </details>`;
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
