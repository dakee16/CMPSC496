const S = requireSession("teacher");
mountHeader({variant: "instructor", active: "Grades"});
const $ = id => document.getElementById(id);

/* Which assignment is open survives a reload. An instructor grading a section
   reloads this page more than once, and landing back on the first assignment in
   the list every time is a small tax paid repeatedly. */
const LAST = "mt.grades.assignment";
let ROWS = [];

function gradeCell(r){
  const pct = r.percent === null ? "" : `<span class="pct">${r.percent}%</span>`;
  return `<div class="grade">
      <span class="frac">${r.solved} / ${r.total}</span>${pct}
    </div>
    <div class="split"><b>${r.solved}</b> solved &middot;
      <b>${r.shown}</b> answer shown &middot;
      <b>${r.missed}</b> not done</div>
    ${r.percent === null ? "" : `<div class="grade-track" aria-hidden="true"><i style="width:${Math.max(0, Math.min(100, Number(r.percent) || 0))}%"></i></div>`}`;
}

function table(rows){
  return `<table>
    <thead><tr>
      <th>Student</th><th class="colMail">Email</th><th>Status</th>
      <th>Grade</th><th class="colDl">Transcript</th>
    </tr></thead>
    <tbody>${rows.map(r => `<tr data-id="${esc(r.student_id)}">
      <td class="colName">${esc(r.name)}</td>
      <td class="colMail">${esc(r.username)}</td>
      <td><span class="pill ${r.submitted ? "ok" : "bad"}">${
        r.submitted ? "Submitted" : "Missing"}</span></td>
      <td>${gradeCell(r)}</td>
      <td class="colDl"><button class="ghost" data-act="dl">Download</button></td>
    </tr>`).join("")}</tbody></table>`;
}

/* The transcript is fetched rather than linked. A plain link would send the
   cookie and work, but an expired one would navigate the instructor onto a JSON
   error page; going through fetch() keeps them on this screen and lets ui.js's
   401 handler do what it does everywhere else. */
async function download(btn, r){
  const id = $("pick").value;
  setBusy(btn, true, "Preparing...");
  try {
    const res = await fetch(
      `${API}/teacher/assignments/${encodeURIComponent(id)}/transcript/${
        encodeURIComponent(r.student_id)}`);
    if (!res.ok) throw new Error(`server said ${res.status}`);
    const blob = await res.blob();
    // Filename comes from the server's Content-Disposition, so one place names
    // the file and the saved copy cannot disagree with the header inside it.
    const named = /filename="([^"]+)"/.exec(
      res.headers.get("Content-Disposition") || "");
    const url = URL.createObjectURL(blob);
    const a = Object.assign(document.createElement("a"),
      {href: url, download: named ? named[1] : "transcript.txt"});
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (e) {
    toast(`Could not build that transcript. ${e.message}`, "bad");
  } finally {
    setBusy(btn, false);
  }
}

function empty(line, sub){
  return `<div class="empty">
    <span class="eicon" aria-hidden="true">${BOOK}</span>
    <p>${line}</p><p class="esub">${sub}</p></div>`;
}
const BOOK = `<svg width="19" height="19" viewBox="0 0 24 24" fill="none"
  stroke="currentColor" stroke-width="2" stroke-linecap="round"
  stroke-linejoin="round"><path d="M4 19.5V5a2 2 0 0 1 2-2h13v17H6a2 2 0 0 1-2-1.5z"/>
  <path d="M9 8.5l2 2 4-4"/></svg>`;

async function loadGrades(){
  const id = $("pick").value;
  if (!id) return;
  try { localStorage.setItem(LAST, id); } catch {}
  paintGradeStats(null);
  $("body").innerHTML = skeletonRows(3);
  $("tally").textContent = "";

  let d;
  try {
    const res = await fetch(
      `${API}/teacher/assignments/${encodeURIComponent(id)}/grades`);
    if (!res.ok) throw new Error(`server said ${res.status}`);
    d = await res.json();
  } catch (e) {
    $("body").innerHTML = `<div class="banner bad">Could not load grades.
      ${esc(e.message)} <button class="retry" type="button">Try again</button></div>`;
    $("body").querySelector(".retry").onclick = loadGrades;
    return;
  }

  ROWS = d.students || [];
  paintGradeStats(ROWS);
  $("tally").textContent =
    `${d.problems} problem${d.problems === 1 ? "" : "s"} ready · `
    + `${d.total_steps} step${d.total_steps === 1 ? "" : "s"} to grade`;

  if (!d.total_steps){
    // Zero steps is not zero marks. Saying so beats a table of 0 / 0 rows that
    // reads like the whole class failed.
    $("body").innerHTML = empty("Nothing to grade in this assignment yet.",
      "No problem in it has been split into steps, so there is no denominator.");
    return;
  }
  if (!ROWS.length){
    $("body").innerHTML = empty("No students yet.",
      "Accounts appear here as soon as students register.");
    return;
  }

  $("body").innerHTML = table(ROWS);
  $("body").querySelectorAll('[data-act="dl"]').forEach(b => {
    const r = ROWS.find(x => x.student_id === b.closest("tr").dataset.id);
    b.onclick = () => download(b, r);
  });
}

async function loadAssignments(){
  let list = [];
  try {
    list = (await (await fetch(`${API}/assignments`)).json()).assignments || [];
  } catch {
    $("pick").innerHTML = "<option>Could not reach the server</option>";
    $("pick").disabled = true;
    $("body").innerHTML = `<div class="banner bad">Could not reach the server.
      <button class="retry" type="button">Try again</button></div>`;
    $("body").querySelector(".retry").onclick = () => location.reload();
    return;
  }
  if (!list.length){
    $("pick").innerHTML = "<option>No assignments yet</option>";
    $("pick").disabled = true;
    $("body").innerHTML = empty("No assignments yet.",
      `Upload one on the <a href="teacher.html">Home</a> screen first.`);
    return;
  }
  let last = null;
  try { last = localStorage.getItem(LAST); } catch {}
  $("pick").innerHTML = list.map(a =>
    `<option value="${esc(a.id)}"${String(a.id) === last ? " selected" : ""}
      >${esc(a.name)}</option>`).join("");
  $("pick").onchange = loadGrades;
  loadGrades();
}

loadAssignments();

function paintGradeStats(rows){
  const values = rows ? [rows.length, rows.filter(r => r.submitted).length,
    rows.filter(r => !r.submitted).length] : ["—", "—", "—"];
  $("gradeStats").querySelectorAll("[data-metric]").forEach((el,i) => el.textContent = values[i]);
}
