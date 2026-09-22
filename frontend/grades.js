const S = requireSession("teacher");
mountHeader({variant: "instructor", active: "Grades"});
const $ = id => document.getElementById(id);

/* Which assignment is open survives a reload. An instructor grading a section
   reloads this page more than once, and landing back on the first assignment in
   the list every time is a small tax paid repeatedly. */
const LAST = "mt.grades.assignment";
let ROWS = [];
let gradeRequest=0,visibleAssignment=null;

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
      <td class="colDl"><button class="ghost" data-act="dl">Open record</button></td>
    </tr>`).join("")}</tbody></table>`;
}

/* ---------- the record, full screen ----------

   The preview and the download are the SAME bytes from the same route: an
   instructor reads exactly what they are about to keep, and there is no second
   renderer to drift from this one. The PDF is the browser's own print of that
   page, which is why "Save as PDF" prints the frame rather than asking the
   server for a file it cannot make. */
let reportURL = "", reportName = "report.html";

function openReport(url, title, filename){
  reportURL = url;
  reportName = filename;
  $("reportTitle").textContent = title;
  $("reportFrame").src = url;
  $("reportOverlay").hidden = false;
  document.body.classList.add("reportOpen");
  $("reportClose").focus();
}

function closeReport(){
  $("reportOverlay").hidden = true;
  document.body.classList.remove("reportOpen");
  // Drop the document rather than leave a signed-in transcript rendered in a
  // hidden frame behind whatever the instructor does next.
  $("reportFrame").removeAttribute("src");
}

/* Saving the HTML goes through fetch rather than a plain link for the reason
   every other call here does: a link would navigate the instructor onto a JSON
   error page if the cookie had expired, while fetch keeps them on this screen
   and lets ui.js's 401 handler do what it does everywhere else. */
async function saveHTML(url, fallbackName){
  try {
    const res = await fetch(url + (url.includes("?") ? "&" : "?") + "download=1");
    if (!res.ok) throw new Error(`server said ${res.status}`);
    const blob = await res.blob();
    // The server names the file, so the saved copy and the heading inside it
    // cannot disagree.
    const named = /filename="([^"]+)"/.exec(
      res.headers.get("Content-Disposition") || "");
    const href = URL.createObjectURL(blob);
    const a = Object.assign(document.createElement("a"),
      {href, download: named ? named[1] : fallbackName});
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(href);
  } catch (e) {
    toast(`Could not build that record. ${e.message}`, "bad");
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

async function loadGrades(force=false){
  const id = $("pick").value;
  if (!id) return;
  const request=++gradeRequest;
  try { localStorage.setItem(LAST, id); } catch {}
  if(visibleAssignment!==id){paintGradeStats(null);$("body").innerHTML=skeletonRows(3);$("tally").textContent="";}
  $("body").setAttribute("aria-busy","true");

  let d;
  try {
    const res = await fetch(
      `${API}/teacher/assignments/${encodeURIComponent(id)}/grades`,force===true?{cache:"reload"}:{});
    if (!res.ok) throw new Error(`server said ${res.status}`);
    d = await res.json();
    if(request!==gradeRequest)return;
  } catch (e) {
    if(request!==gradeRequest)return;
    $("body").innerHTML = `<div class="banner bad">Could not load grades.
      ${esc(e.message)} <button class="retry" type="button">Try again</button></div>`;
    $("body").querySelector(".retry").onclick=()=>loadGrades(true);return;
  }finally{if(request===gradeRequest)$("body").setAttribute("aria-busy","false");}
  visibleAssignment=id;
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
  const show = r => openReport(
    `${API}/teacher/assignments/${encodeURIComponent(id)}/report/${
      encodeURIComponent(r.student_id)}`,
    `Learning record for ${r.name}`, `ACADIA_${r.name}.html`);
  $("body").querySelectorAll('[data-act="dl"]').forEach(b => {
    const r = ROWS.find(x => x.student_id === b.closest("tr").dataset.id);
    b.onclick = () => show(r);
  });
  // The GRADE itself opens the record. That cell is the thing an instructor is
  // already looking at when they want to know where a number came from, so it
  // is the natural way in - the button stays for anyone reaching by keyboard.
  $("body").querySelectorAll("tbody tr").forEach(tr => {
    const r = ROWS.find(x => x.student_id === tr.dataset.id);
    if (!r) return;
    const cell = tr.querySelector("td:nth-child(4)");
    if (!cell) return;
    cell.classList.add("gradeOpen");
    cell.tabIndex = 0;
    cell.setAttribute("role", "button");
    cell.setAttribute("aria-label", `Open ${r.name}'s learning record`);
    cell.onclick = () => show(r);
    cell.onkeydown = e => {
      if (e.key === "Enter" || e.key === " "){ e.preventDefault(); show(r); }
    };
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
      `Upload one on the <a href="teacher-assignments.html">Assignments</a> tab first.`);
    return;
  }
  let last = new URLSearchParams(location.search).get("assignment");
  if (!last) { try { last = localStorage.getItem(LAST); } catch {} }
  $("pick").innerHTML = list.map(a =>
    `<option value="${esc(a.id)}"${String(a.id) === last ? " selected" : ""}
      >${esc(a.name)}</option>`).join("");
  $("pick").onchange = loadGrades;
  loadGrades();
}

$("refreshGrades").onclick=()=>loadGrades(true);

$("reportAll").onclick = () => {
  const id = $("pick").value;
  if (!id) return;
  openReport(`${API}/teacher/assignments/${encodeURIComponent(id)}/report`,
             "Class learning record", "ACADIA_Class.html");
};
$("reportClose").onclick = closeReport;
$("reportTab").onclick = () => window.open(reportURL, "_blank", "noopener");
$("reportHtml").onclick = () => saveHTML(reportURL, reportName);
/* Print the FRAME, not this page: the record is the document being saved, and
   printing the grade table around it would produce a PDF of the wrong thing. */
$("reportPdf").onclick = () => {
  const f = $("reportFrame");
  try { f.contentWindow.focus(); f.contentWindow.print(); }
  catch { window.open(reportURL, "_blank", "noopener"); }
};
addEventListener("keydown", e => {
  if (e.key === "Escape" && !$("reportOverlay").hidden) closeReport();
});
window.addEventListener("acadia:cache-update",event=>{
  if(event.detail.url==="/teacher/assignments/"+encodeURIComponent($("pick").value)+"/grades")loadGrades();
});
loadAssignments();

function paintGradeStats(rows){
  const values = rows ? [rows.length, rows.filter(r => r.submitted).length,
    rows.filter(r => !r.submitted).length] : ["-", "-", "-"];
  $("gradeStats").querySelectorAll("[data-metric]").forEach((el,i) => el.textContent = values[i]);
}
