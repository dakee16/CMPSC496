/* The student's private dashboard. PROGRESS only - problems completed,
   assignments finished, days practised. No score, percentage or step credit is
   shown to a student anywhere: this is not the course's LMS, and a number here
   that disagreed with the one there would be a conflict for the student to
   sort out. The instructor's Grades page is where credit lives. */
requireSession("student");
mountHeader({variant: "student", active: "Dashboard"});
const insightEl = id => document.getElementById(id);
let progressData = null;
let loadingProgress = false;
// One word for a finished problem - see STAT_LABEL in student.js for why the
// independent/with-help split is no longer shown to the student.
const statusNames = {solved:"Solved",helped:"Solved",progress:"In progress",todo:"Not started"};
const workLink = (assignment, slug) => `student.html?assignment=${encodeURIComponent(assignment)}${slug ? `&problem=${encodeURIComponent(slug)}` : ""}`;
const barHTML = (value, label) => `<div class="insight-bar" role="progressbar" aria-label="${esc(label)}" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${Number(value) || 0}"><i style="width:${Number(value) || 0}%"></i></div>`;

function activityTime(value){
  const date = new Date(value);
  return Number.isNaN(+date) ? "" : date.toLocaleString(undefined,{month:"short",day:"numeric",hour:"numeric",minute:"2-digit"});
}

function metricHTML(icon, label, value, note){
  return `<div class="card dash-metric"><p>${esc(label)}</p><strong>${esc(value)}</strong><small>${esc(note)}</small></div>`;
}

/* The stored first name, NOT the first word of the display name: with no name
   on the account that display name is standing in with the local part of their
   address, and greeting someone by their user id is worse than greeting them
   by nothing. They can set a real one in Settings. */
function drawGreeting(){
  const host = insightEl("insightsTitle");
  if (!host) return;
  const account = Session.get();
  const name = account && (account.first || "").trim();
  const hour = new Date().getHours();
  const greeting = hour < 12 ? "Good morning" : hour < 18 ? "Good afternoon" : "Good evening";
  host.textContent = name ? `${greeting}, ${name}.` : `${greeting}.`;
}
document.addEventListener("acadia:name-changed", drawGreeting);

function renderDashboard(data){
  const s = data.summary;
  const assignments = new Map(data.assignments.map(a=>[String(a.id),a]));
  const problems = new Map(data.problems.map(p=>[p.slug,p]));
  const next = data.next_up.map(slug=>problems.get(slug)).filter(Boolean);
  const first = next[0];
  drawGreeting();
  const peak = Math.max(1,...data.activity.map(d=>d.submissions));
  const heroTitle = first ? first.title : s.problems ? "Look at how far you've come." : "Your next chapter starts here.";
  const heroSub=first?assignments.get(String(first.assignment_id))?.name||"Your assignment":s.problems?"Every available problem is complete. Revisit an assignment any time.":"Your instructor's published assignments will appear here when they're ready.";
  insightEl("insights").innerHTML = `
    <div class="dash-top">
      <section class="learning-hero" aria-label="Continue learning">
        <div class="hero-copy"><p class="eyebrow">${first ? first.status === "progress" ? "PICK UP WHERE YOU LEFT OFF" : "A GOOD PLACE TO START" : "YOUR LEARNING JOURNEY"}</p><h2>${esc(heroTitle)}</h2><p>${esc(heroSub)}</p>
          <a class="action-link" href="${first ? workLink(first.assignment_id,first.slug) : "student.html"}">${first ? first.status === "progress" ? "Continue learning" : "Start practicing" : "Browse assignments"}${uiIcon("arrow",17)}</a>
        </div>

      </section>
      <section class="card week-card" aria-labelledby="weekTitle">
        <div class="insight-section-title"><h2 id="weekTitle">This week</h2><span class="section-symbol">${uiIcon("chart",18)}</span></div>
        <div class="week-heading"><strong>${s.active_days}<span> / 7</span></strong><span>days of practice</span></div>
        <div class="activity-chart" role="list" aria-label="Submissions in the last seven days">
          ${data.activity.map((d,i)=>{const date=new Date(d.date+"T12:00:00Z");const label=date.toLocaleDateString(undefined,{weekday:"short",timeZone:"UTC"});const full=date.toLocaleDateString(undefined,{month:"short",day:"numeric",timeZone:"UTC"});return `<div class="activity-day ${i===data.activity.length-1 ? "today" : ""}" role="listitem" aria-label="${esc(full)}: ${d.submissions} submissions, ${d.passed} passing" title="${esc(full)} · ${d.submissions} submissions"><span class="count" aria-hidden="true">${d.submissions}</span><div class="activity-bar-space" aria-hidden="true"><i class="activity-bar" style="--height:${Math.round(d.submissions/peak*100)}%"></i></div><span class="day" aria-hidden="true">${esc(label)}</span></div>`;}).join("")}
        </div>
        <p class="week-note">${s.weekly_submissions} submissions · ${s.active_days ? "Every practice day adds up." : "Open a problem to get started."}</p>
      </section>
    </div>
    <section class="dash-metrics" aria-label="Learning overview">
      ${metricHTML("book","Problems completed",`${s.completed} / ${s.problems}`,`${s.independent} solved independently`)}
      ${metricHTML("lab","In progress",s.in_progress,"Problems ready to pick up")}
      ${metricHTML("grid","Assignments finished",`${s.completed_assignments} / ${s.assignments}`,"Published assignments")}
    </section>
    <div class="dash-columns">
      <div class="dash-column">
        <section class="card"><div class="insight-section-title"><h2>Assignment progress</h2><a href="student.html">View all ${uiIcon("arrow",13)}</a></div>
          <div class="assignment-progress-list">${data.assignments.length ? data.assignments.slice(0,5).map(a=>`<div class="assignment-progress-item"><div class="assignment-progress-head"><a href="${workLink(a.id)}">${esc(a.name)}</a><span>${a.completed} / ${a.problems} complete</span></div>${barHTML(a.completion_percent,a.name+" completion")}<div class="assignment-progress-foot"><span>${a.in_progress ? a.in_progress+" in progress" : a.completed===a.problems ? "All problems completed" : "Ready when you are"}</span></div></div>`).join("") : `<div class="insight-empty">No published assignments yet. Check back when your instructor adds one.</div>`}</div>
        </section>
        <section class="card"><div class="insight-section-title"><h2>Up next</h2><a href="student.html">All problems ${uiIcon("arrow",13)}</a></div>
          ${next.length ? `<ol class="next-list">${next.map((p,i)=>`<li><a class="next-link" href="${workLink(p.assignment_id,p.slug)}"><span class="next-number">${String(i+1).padStart(2,"0")}</span><span class="next-copy"><strong>${esc(p.title)}</strong><small>${esc(assignments.get(String(p.assignment_id))?.name || "")} · ${esc(statusNames[p.status])}</small></span>${uiIcon("arrow",16)}</a></li>`).join("")}</ol>` : `<div class="insight-empty">${s.problems ? "You're caught up on every available problem. Revisit a finished problem to reflect on your work." : "Your next steps will appear as soon as an assignment is available."}</div>`}
        </section>
      </div>
      <div class="dash-column">
        <section class="card"><div class="insight-section-title"><h2>Recent activity</h2><span class="section-symbol">${uiIcon("lab",17)}</span></div>
          ${data.recent.length ? `<ol class="activity-list">${data.recent.map(e=>{const label=e.kind==="passed"?`Passed step ${e.step}`:e.kind==="completed"?"Completed a problem":e.kind==="practiced"?`Practiced step ${e.step}`:"Opened a problem";return `<li class="activity-item ${e.kind}"><span class="activity-icon" aria-hidden="true">${e.kind==="passed"||e.kind==="completed"?"✓":"↗"}</span><div class="activity-copy"><a href="${workLink(e.assignment_id,e.slug)}">${esc(e.title)}</a><p>${esc(label)}</p></div><time datetime="${esc(e.at)}" title="${esc(activityTime(e.at))}">${esc(relTime(e.at))}</time></li>`;}).join("")}</ol>` : `<div class="insight-empty">Your first practice session starts the story. Your recent work will appear here.</div>`}
        </section>
        <section class="card"><div class="insight-section-title"><h2>Milestones</h2></div><div class="milestone-list">
          ${[[s.independent>0,"First independent solve",s.independent>0?"You solved a problem with your own code.":"Solve your first problem independently."],[s.active_days>=3,"A steady rhythm",s.active_days>=3?"You practiced on three days this week.":"Practice on three days in a week."],[s.completed_assignments>0,"One chapter complete",s.completed_assignments>0?"You finished an entire assignment.":"Complete every problem in an assignment."]].map(([earned,title,note])=>`<div class="milestone ${earned?"earned":""}"><span aria-hidden="true">${earned?"✓":"○"}</span><div><strong>${esc(title)}<span class="sr-only">${earned?": achieved":": not yet achieved"}</span></strong><small>${esc(note)}</small></div></div>`).join("")}
        </div></section>
      </div>
    </div>`;
}

async function loadStudentProgress(force=false){
  if(loadingProgress)return;
  loadingProgress=true;
  const button=insightEl("refreshProgress");setBusy(button,true,"Refreshing…");
  insightEl("insights").setAttribute("aria-busy","true");
  insightEl("progressNotice").replaceChildren();
  try{
    let timezone="UTC";try{timezone=Intl.DateTimeFormat().resolvedOptions().timeZone||"UTC";}catch{}
    const response=await fetch(`${API}/student/progress?timezone=${encodeURIComponent(timezone)}`,force===true?{cache:"reload"}:{});
    if(!response.ok)throw new Error("Progress unavailable");
    const data=await response.json();
    if(!data.summary||!Array.isArray(data.problems)||!Array.isArray(data.assignments)||!Array.isArray(data.activity)||!Array.isArray(data.recent)||!Array.isArray(data.next_up))throw new Error("Incomplete progress response");
    progressData=data;
    renderDashboard(data);
    insightEl("updatedAt").textContent=`Updated ${new Date(data.generated_at).toLocaleTimeString(undefined,{hour:"numeric",minute:"2-digit"})}`;
  }catch{
    if(!progressData)insightEl("insights").innerHTML='<div class="card insight-empty"><p>Your results could not be loaded. No progress has been changed.</p><button id="retryProgress" type="button">Try again</button></div>';
    else insightEl("progressNotice").innerHTML='<div class="banner warn">Could not refresh your results. You are still seeing the last successful update.</div>';
    const retry=insightEl("retryProgress");if(retry)retry.onclick=()=>loadStudentProgress(true);
  }finally{
    loadingProgress=false;setBusy(button,false);
    insightEl("insights").setAttribute("aria-busy","false");
  }
}

insightEl("refreshProgress").onclick=()=>loadStudentProgress(true);
window.addEventListener("acadia:cache-update",event=>{
  if(event.detail.url!=="/student/progress")return;
  loadStudentProgress();
});
window.addEventListener("acadia:cache-error",event=>{
  if(event.detail.url==="/student/progress"&&progressData)insightEl("progressNotice").innerHTML='<div class="banner warn">Could not refresh. Showing your last saved results.</div>';
});
loadStudentProgress();
