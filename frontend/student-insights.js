/* Private dashboard/gradebook views. Counts come from the server's existing
   grading rules; completion and earned credit are deliberately separate. */
requireSession("student");
const studentView = document.body.dataset.studentView;
mountHeader({variant: "student", active: studentView === "grades" ? "Grades" : "Dashboard"});
const insightEl = id => document.getElementById(id);
let progressData = null;
let loadingProgress = false;
const gradeFilters = {assignment: new URLSearchParams(location.search).get("assignment") || "all", search: "", status: "all", sort: "title"};
const statusNames = {solved:"Solved independently",helped:"Completed with help",progress:"In progress",todo:"Not started",passed:"Passed",shown:"Shown answer",needs_work:"Keep practicing",not_started:"Not attempted",not_graded:"Not graded yet"};
const scoreText = value => value === null || value === undefined ? "—" : `${value}%`;
const workLink = (assignment, slug) => `student.html?assignment=${encodeURIComponent(assignment)}${slug ? `&problem=${encodeURIComponent(slug)}` : ""}`;
const gradeLink = assignment => `student-grades.html?assignment=${encodeURIComponent(assignment)}`;
const statusHTML = state => `<span class="grade-status ${Object.hasOwn(statusNames,state) ? state : "todo"}">${esc(statusNames[state] || "Not started")}</span>`;
const barHTML = (value, label) => `<div class="insight-bar" role="progressbar" aria-label="${esc(label)}" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${Number(value) || 0}"><i style="width:${Number(value) || 0}%"></i></div>`;

function activityTime(value){
  const date = new Date(value);
  return Number.isNaN(+date) ? "" : date.toLocaleString(undefined,{month:"short",day:"numeric",hour:"numeric",minute:"2-digit"});
}

function metricHTML(icon, label, value, note){
  return `<div class="card dash-metric"><span class="dash-metric-icon" aria-hidden="true">${uiIcon(icon,20)}</span><div><p>${esc(label)}</p><strong>${esc(value)}</strong><small>${esc(note)}</small></div></div>`;
}

function renderDashboard(data){
  const s = data.summary;
  const assignments = new Map(data.assignments.map(a=>[String(a.id),a]));
  const problems = new Map(data.problems.map(p=>[p.slug,p]));
  const next = data.next_up.map(slug=>problems.get(slug)).filter(Boolean);
  const first = next[0];
  const account = Session.get();
  const name = account && account.name ? account.name.trim().split(/\s+/)[0] : "";
  const greeting = new Date().getHours()<12 ? "Good morning" : new Date().getHours()<18 ? "Good afternoon" : "Good evening";
  insightEl("insightsTitle").textContent = name ? `${greeting}, ${name}.` : "Make room for progress.";
  const peak = Math.max(1,...data.activity.map(d=>d.submissions));
  const heroTitle = first ? first.title : s.problems ? "Look at how far you've come." : "Your next chapter starts here.";
  const heroSub = first ? `${assignments.get(String(first.assignment_id))?.name || "Your assignment"}. ${first.status === "progress" ? "Pick up your thinking and keep moving forward." : "A fresh problem. A chance to put your ideas to work."}` : s.problems ? "Every available problem is complete. Revisit an assignment or take a look at the steps you've earned." : "Your instructor's published assignments will appear here when they're ready. This is your space to learn, practice, and grow.";
  insightEl("insights").innerHTML = `
    <div class="dash-top">
      <section class="learning-hero" aria-label="Continue learning">
        <div class="hero-copy"><p class="eyebrow">${first ? first.status === "progress" ? "PICK UP WHERE YOU LEFT OFF" : "A GOOD PLACE TO START" : "YOUR LEARNING JOURNEY"}</p><h2>${esc(heroTitle)}</h2><p>${esc(heroSub)}</p>
          <a class="action-link" href="${first ? workLink(first.assignment_id,first.slug) : s.problems ? "student-grades.html" : "student.html"}">${first ? first.status === "progress" ? "Continue learning" : "Start practicing" : s.problems ? "See my grades" : "Browse assignments"}${uiIcon("arrow",17)}</a>
        </div>
        <div class="completion-orbit" style="--completion:${Number(s.completion_percent) || 0}%" role="img" aria-label="${s.completed} of ${s.problems} problems complete">
          <div class="orbit-core"><strong>${scoreText(s.completion_percent)}</strong><span>Complete</span><small>${s.completed} of ${s.problems} problems</small></div>
        </div>
      </section>
      <section class="card week-card" aria-labelledby="weekTitle">
        <div class="insight-section-title"><h2 id="weekTitle">Your week in motion</h2><span class="section-symbol">${uiIcon("chart",18)}</span></div>
        <div class="week-heading"><strong>${s.active_days}<span> / 7</span></strong><span>days of practice</span></div>
        <div class="activity-chart" role="list" aria-label="Submissions in the last seven days">
          ${data.activity.map((d,i)=>{const date=new Date(d.date+"T12:00:00Z");const label=date.toLocaleDateString(undefined,{weekday:"short",timeZone:"UTC"});const full=date.toLocaleDateString(undefined,{month:"short",day:"numeric",timeZone:"UTC"});return `<div class="activity-day ${i===data.activity.length-1 ? "today" : ""}" role="listitem" aria-label="${esc(full)}: ${d.submissions} submissions, ${d.passed} passing" title="${esc(full)} · ${d.submissions} submissions"><span class="count" aria-hidden="true">${d.submissions}</span><div class="activity-bar-space" aria-hidden="true"><i class="activity-bar" style="--height:${Math.round(d.submissions/peak*100)}%"></i></div><span class="day" aria-hidden="true">${esc(label)}</span></div>`;}).join("")}
        </div>
        <p class="week-note">${s.weekly_submissions} graded submissions · ${s.active_days ? "Every practice day adds up." : "Open a problem to get started."}</p>
      </section>
    </div>
    <section class="dash-metrics" aria-label="Learning overview">
      ${metricHTML("book","Problems completed",`${s.completed} / ${s.problems}`,`${s.independent} solved independently`)}
      ${metricHTML("chart","Earned step credit",scoreText(s.percent),`${s.earned} of ${s.total} steps passed`)}
      ${metricHTML("lab","In progress",s.in_progress,"Problems ready to pick up")}
      ${metricHTML("grid","Assignments finished",`${s.completed_assignments} / ${s.assignments}`,"Published assignments")}
    </section>
    <div class="dash-columns">
      <div class="dash-column">
        <section class="card"><div class="insight-section-title"><h2>Assignment progress</h2><a href="student.html">View all ${uiIcon("arrow",13)}</a></div>
          <div class="assignment-progress-list">${data.assignments.length ? data.assignments.slice(0,5).map(a=>`<div class="assignment-progress-item"><div class="assignment-progress-head"><a href="${workLink(a.id)}">${esc(a.name)}</a><span>${a.completed} / ${a.problems} complete</span></div>${barHTML(a.completion_percent,a.name+" completion")}<div class="assignment-progress-foot"><span>${a.in_progress ? a.in_progress+" in progress" : a.completed===a.problems ? "All problems completed" : "Ready when you are"}</span><a href="${gradeLink(a.id)}">${scoreText(a.percent)} step credit</a></div></div>`).join("") : `<div class="insight-empty">No published assignments yet. Check back when your instructor adds one.</div>`}</div>
        </section>
        <section class="card"><div class="insight-section-title"><h2>Your next steps</h2><span class="grade-problem-meta">ONE PROBLEM AT A TIME</span></div>
          ${next.length ? `<ol class="next-list">${next.map((p,i)=>`<li><a class="next-link" href="${workLink(p.assignment_id,p.slug)}"><span class="next-number">${String(i+1).padStart(2,"0")}</span><span class="next-copy"><strong>${esc(p.title)}</strong><small>${esc(assignments.get(String(p.assignment_id))?.name || "")} · ${esc(statusNames[p.status])}</small></span>${uiIcon("arrow",16)}</a></li>`).join("")}</ol>` : `<div class="insight-empty">${s.problems ? "You're caught up on every available problem. Revisit your grades to reflect on your work." : "Your next steps will appear as soon as an assignment is available."}</div>`}
        </section>
      </div>
      <div class="dash-column">
        <section class="card"><div class="insight-section-title"><h2>Recent activity</h2><span class="section-symbol">${uiIcon("lab",17)}</span></div>
          ${data.recent.length ? `<ol class="activity-list">${data.recent.map(e=>{const label=e.kind==="passed"?`Passed step ${e.step}`:e.kind==="completed"?"Completed a problem":e.kind==="practiced"?`Practiced step ${e.step}`:"Opened a problem";return `<li class="activity-item ${e.kind}"><span class="activity-icon" aria-hidden="true">${e.kind==="passed"||e.kind==="completed"?"✓":"↗"}</span><div class="activity-copy"><a href="${workLink(e.assignment_id,e.slug)}">${esc(label)}</a><p>${esc(e.title)}</p><time datetime="${esc(e.at)}">${esc(activityTime(e.at))}</time></div></li>`;}).join("")}</ol>` : `<div class="insight-empty">Your first practice session starts the story. Your recent work will appear here.</div>`}
        </section>
        <section class="card"><div class="insight-section-title"><h2>Small wins, real progress</h2></div><div class="milestone-list">
          ${[[s.independent>0,"First independent solve",s.independent>0?"You solved a problem with your own code.":"Solve your first problem independently."],[s.active_days>=3,"A steady rhythm",s.active_days>=3?"You practiced on three days this week.":"Practice on three days in a week."],[s.completed_assignments>0,"One chapter complete",s.completed_assignments>0?"You finished an entire assignment.":"Complete every problem in an assignment."]].map(([earned,title,note])=>`<div class="milestone ${earned?"earned":""}"><span aria-hidden="true">${earned?"✓":"○"}</span><div><strong>${esc(title)}<span class="sr-only">${earned?" — achieved":" — not yet achieved"}</span></strong><small>${esc(note)}</small></div></div>`).join("")}
        </div></section>
      </div>
    </div>`;
}

function selectedSummary(){
  return gradeFilters.assignment === "all" ? progressData.summary : progressData.assignments.find(a=>String(a.id)===gradeFilters.assignment) || progressData.summary;
}

function renderGrades(data){
  if(gradeFilters.assignment!=="all" && !data.assignments.some(a=>String(a.id)===gradeFilters.assignment)) gradeFilters.assignment="all";
  insightEl("insights").innerHTML = `
    <div class="grade-filters" aria-label="Grade filters">
      <div class="field"><label for="gradeAssignment">Assignment</label><select id="gradeAssignment"><option value="all">All assignments</option>${data.assignments.map(a=>`<option value="${esc(a.id)}">${esc(a.name)}</option>`).join("")}</select></div>
      <div class="field search-field"><label for="gradeSearch">Find a problem</label><input id="gradeSearch" type="search" placeholder="Search problem names…" autocomplete="off"></div>
      <div class="field"><label for="gradeStatus">Status</label><select id="gradeStatus"><option value="all">All statuses</option><option value="progress">In progress</option><option value="todo">Not started</option><option value="solved">Solved independently</option><option value="helped">Completed with help</option></select></div>
      <div class="field"><label for="gradeSort">Sort by</label><select id="gradeSort"><option value="title">Problem name</option><option value="lowest">Lowest grade first</option><option value="recent">Recent activity</option></select></div>
    </div>
    <div class="grades-top" id="gradeOverview"></div>
    <div class="grades-count"><span id="gradeResultCount" role="status" aria-live="polite"></span><button class="ghost" id="clearGradeFilters" type="button">Reset filters</button></div>
    <div id="gradeResults"></div>
    <p class="credit-explainer">Grades reflect independently passed steps across your recorded attempts. Shown answers and unfinished steps earn no credit. Problem completion is tracked separately. “—” means a step count is not available yet; it is not a zero grade.</p>`;
  for(const [key,id] of [["assignment","gradeAssignment"],["search","gradeSearch"],["status","gradeStatus"],["sort","gradeSort"]]){
    const input=insightEl(id);input.value=gradeFilters[key];
    input.addEventListener(key==="search"?"input":"change",()=>{
      gradeFilters[key]=input.value;
      if(key==="assignment") history.replaceState(null,"",gradeFilters.assignment==="all"?location.pathname:`${location.pathname}?assignment=${encodeURIComponent(gradeFilters.assignment)}`);
      renderGradeResults();
    });
  }
  insightEl("clearGradeFilters").onclick=()=>{
    Object.assign(gradeFilters,{assignment:"all",search:"",status:"all",sort:"title"});
    history.replaceState(null,"",location.pathname);
    renderGrades(progressData);
  };
  insightEl("gradeResults").addEventListener("click",event=>{
    const button=event.target.closest("[data-grade-detail]");if(!button)return;
    const row=insightEl(button.getAttribute("aria-controls"));
    row.hidden=!row.hidden;button.setAttribute("aria-expanded",String(!row.hidden));
    button.textContent=row.hidden?"View steps":"Hide steps";
    button.setAttribute("aria-label",button.getAttribute("aria-label").replace(/^(View|Hide)/,row.hidden?"View":"Hide"));
  });
  renderGradeResults();
}

function renderGradeResults(){
  const s=selectedSummary();
  const assignment=gradeFilters.assignment==="all"?null:progressData.assignments.find(a=>String(a.id)===gradeFilters.assignment);
  insightEl("gradeOverview").innerHTML=`<section class="card grade-total"><p class="eyebrow">${assignment?"ASSIGNMENT GRADE":"OVERALL STEP CREDIT"}</p><div class="grade-total-number">${scoreText(s.percent)}</div><p>${s.earned} of ${s.total} available steps passed</p>${barHTML(s.percent,"Earned step credit")}</section><section class="card grade-summary"><h2>${assignment?esc(assignment.name):"A clear view of your work."}</h2><p>You earn credit when your own code passes a step. Keep practicing at your pace—your passing work stays in your record.${s.ungraded_problems?` ${s.ungraded_problems} problem${s.ungraded_problems===1?" is":"s are"} awaiting a step count.`:""}</p><div class="grade-breakdown"><div><strong>${s.earned}</strong><span>Steps passed</span></div><div><strong>${s.shown}</strong><span>Answers shown</span></div><div><strong>${s.remaining}</strong><span>Steps remaining</span></div></div></section>`;
  let rows=progressData.problems.filter(p=>(gradeFilters.assignment==="all"||String(p.assignment_id)===gradeFilters.assignment)&&(gradeFilters.status==="all"||p.status===gradeFilters.status)&&`${p.title} ${p.group||""}`.toLowerCase().includes(gradeFilters.search.trim().toLowerCase()));
  rows.sort((a,b)=>gradeFilters.sort==="lowest"?(a.percent??101)-(b.percent??101)||a.title.localeCompare(b.title):gradeFilters.sort==="recent"?(b.last_activity||"").localeCompare(a.last_activity||""):a.title.localeCompare(b.title));
  insightEl("gradeResultCount").textContent=`${rows.length} problem${rows.length===1?"":"s"} · Select “View steps” for the breakdown`;
  if(!rows.length){
    insightEl("gradeResults").innerHTML=`<div class="card insight-empty"><p>${progressData.problems.length?"No problems match these filters.":"No grades yet. Your published assignments will appear here when they are ready."}</p>${progressData.problems.length?"Try another status or reset the filters.":'<a class="action-link" href="student.html">Browse assignments</a>'}</div>`;
    return;
  }
  let index=0;
  insightEl("gradeResults").innerHTML=progressData.assignments.map(a=>{
    const group=rows.filter(p=>String(p.assignment_id)===String(a.id));if(!group.length)return "";
    return `<section class="card grade-group"><div class="grade-group-header"><div><h2>${esc(a.name)}</h2><p>${a.completed} of ${a.problems} problems complete · ${scoreText(a.percent)} step credit</p></div><a href="${workLink(a.id)}">Open assignment ${uiIcon("arrow",13)}</a></div><div class="grade-table-wrap" role="region" aria-label="${esc(a.name)} grades" tabindex="0"><table class="grade-table"><thead><tr><th scope="col">Problem</th><th scope="col">Status</th><th scope="col">Step credit</th><th scope="col">Grade</th><th scope="col"><span class="sr-only">Details</span></th></tr></thead>${group.map(p=>{const id=`gradeDetail${index++}`;return `<tbody><tr><td><div class="grade-problem-name">${esc(p.title)}</div><div class="grade-problem-meta">${p.last_activity?`Last activity ${esc(activityTime(p.last_activity))}`:"Ready to begin"}</div></td><td>${statusHTML(p.status)}</td><td class="grade-credit">${p.solved} / ${p.total||"—"}</td><td class="grade-score">${scoreText(p.percent)}</td><td><button class="ghost grade-details-toggle" type="button" data-grade-detail aria-expanded="false" aria-controls="${id}" aria-label="View steps for ${esc(p.title)}">View steps</button></td></tr><tr id="${id}" hidden><td class="grade-detail-cell" colspan="5"><div class="grade-detail-inner"><div>${p.steps.length?`<ol class="step-results">${p.steps.map(step=>`<li class="step-result"><strong>Step ${step.number}</strong>${statusHTML(step.status)}<small>${step.attempts} graded attempt${step.attempts===1?"":"s"}</small></li>`).join("")}</ol>`:'<p class="week-note">This problem does not have a recorded step count yet.</p>'}</div><a class="action-link" href="${workLink(p.assignment_id,p.slug)}">${p.status==="solved"||p.status==="helped"?"Practice again":p.status==="progress"?"Continue problem":"Start problem"}${uiIcon("arrow",14)}</a></div></td></tr></tbody>`;}).join("")}</table></div></section>`;
  }).join("");
}

async function loadStudentProgress(){
  if(loadingProgress)return;
  loadingProgress=true;
  const button=insightEl("refreshProgress");setBusy(button,true,"Refreshing…");
  insightEl("insights").setAttribute("aria-busy","true");
  insightEl("progressNotice").replaceChildren();
  try{
    let timezone="UTC";try{timezone=Intl.DateTimeFormat().resolvedOptions().timeZone||"UTC";}catch{}
    const response=await fetch(`${API}/student/progress?timezone=${encodeURIComponent(timezone)}`);
    if(!response.ok)throw new Error("Progress unavailable");
    const data=await response.json();
    if(!data.summary||!Array.isArray(data.problems)||!Array.isArray(data.assignments)||!Array.isArray(data.activity)||!Array.isArray(data.recent)||!Array.isArray(data.next_up))throw new Error("Incomplete progress response");
    progressData=data;
    if(studentView==="grades")renderGrades(data);else renderDashboard(data);
    insightEl("updatedAt").textContent=`Updated ${new Date(data.generated_at).toLocaleTimeString(undefined,{hour:"numeric",minute:"2-digit"})}`;
  }catch{
    if(!progressData)insightEl("insights").innerHTML='<div class="card insight-empty"><p>Your results could not be loaded. No progress has been changed.</p><button id="retryProgress" type="button">Try again</button></div>';
    else insightEl("progressNotice").innerHTML='<div class="banner warn">Could not refresh your results. You are still seeing the last successful update.</div>';
    const retry=insightEl("retryProgress");if(retry)retry.onclick=loadStudentProgress;
  }finally{
    loadingProgress=false;setBusy(button,false);
    insightEl("insights").setAttribute("aria-busy","false");
  }
}

insightEl("refreshProgress").onclick=loadStudentProgress;
loadStudentProgress();
