const S = requireSession("student");
mountHeader({variant: "student", active: "Assignments", wide: true});
const $ = id => document.getElementById(id);

let sessionId = null, chunks = [], idx = 0, accepted = [], editor = null, header = "";
// Columns the function BODY sits at, under its own `def` line. The frozen
// listing and the editor welded to the bottom of it are one continuous piece
// of code, so both are drawn on this scale or they disagree by four columns.
const BODY_INDENT = 4;
// The tutor is scoped to whatever is open on the LEFT. No problem, no chat.
let openProblem = null, chatLog = [], chatBusy = false;
// A student may not write code until they have submitted a DESIGN and the
// reviewer has approved it. Designing before coding IS the pedagogy, so this
// gates the whole coding column, not just the submit button - leaving the
// editor typable invites them to start coding and back-fill a design after.
let tutorReleased = false;
// Non-null while the student is looking back at a step they already finished.
let reviewIdx = null;
let reviewDraft = null;

/* Show or hide the coding half. The design panel and the editor are mutually
   exclusive: exactly one of them occupies the bottom of the code surface, so
   there is never a visible editor that cannot be typed into. */
function applyTutorGate(){
  const wrap = $("editorWrap"), panel = $("designPanel"), code = $("code");
  const btn = $("submit");
  // The steps are part of the answer - see the markup. They appear with the
  // editor, never before it.
  for (const id of ["workStep", "planHead", "probProg", "planHint2", "stepper"]){
    if ($(id)) $(id).hidden = !tutorReleased;
  }
  if ($("planLocked")) $("planLocked").hidden = tutorReleased;
  if (btn){
    if (!tutorReleased) disable(btn, "Submit your design first - the editor "
      + "unlocks once the tutor accepts it.");
    else if (btn.dataset.busy === "1") disable(btn, "Grading your last answer…");
    else enable(btn);
  }
  // CodeMirror ignores the underlying <textarea>'s `disabled` attribute, so
  // once fromTextArea() has run the lock MUST go through setOption("readOnly").
  // "nocursor" also blocks focus. Before the editor exists, fall back to the
  // textarea.
  if (editor){
    editor.setOption("readOnly", tutorReleased ? false : "nocursor");
  } else if (code){
    code.disabled = !tutorReleased;
  }
  if (panel) panel.hidden = tutorReleased;
  const wasHidden = wrap && wrap.hidden;
  const complete = chunks.length > 0 && idx >= chunks.length;
  if (wrap) wrap.hidden = !tutorReleased || reviewIdx !== null || complete;
  // CodeMirror measures itself when it is built, and it is now built inside a
  // wrapper the design gate keeps hidden - so it lays out at zero width and
  // comes back with no gutter, which is what put the frozen listing's "1" hard
  // against the `def`. Re-measure the moment it is actually on screen.
  if (editor && wasHidden && wrap && !wrap.hidden){
    requestAnimationFrame(() => { editor.refresh(); matchGutter(); });
  }
}

/* The unlock is an EVENT, not a silent swap: the editor wipes in from the top
   and a confirmation stays on screen for the rest of the problem. */
function openGate(){
  tutorReleased = true;
  applyTutorGate();
  const wrap = $("editorWrap");
  if (wrap){
    wrap.classList.add("unlocking");
    wrap.addEventListener("animationend",
      () => wrap.classList.remove("unlocking"), {once: true});
  }
  markUnlocked("Design accepted. The editor is unlocked for this problem.");
  freezePlan();
  loadSteps();          // the prompts were withheld until this moment
  toast("Design accepted - the editor is unlocked.", "ok");
  if (editor){
    // The editor has been sized to a hidden box until now. Re-seat the step's
    // indent and re-measure the gutter against a laid-out element.
    requestAnimationFrame(() => {
      editor.refresh();
      matchGutter();
      editor.focus();
    });
  }
}

/* Fetch the step prompts, which the server withholds until the design is
   accepted. Until then `chunks` carries empty prompts - the right count, the
   right indents, and nothing that gives the answer away. */
async function loadSteps(){
  if (!sessionId) return;
  try {
    const r = await fetch(`${API}/session_steps/${encodeURIComponent(sessionId)}`);
    if (!r.ok) return;                    // stay locked rather than guess
    const d = await r.json();
    if (d.chunks && d.chunks.length){
      chunks = d.chunks;
      render();
    }
  } catch (e) {
    /* the editor is open either way; the steps arrive on the next load */
  }
}

/* The standing "editor is unlocked" bar.

   A bar, not a toast, and not show(): a toast is gone in seconds and show()
   writes into #msg, which the next grading verdict overwrites. Reopening a
   problem you had already unlocked showed nothing at all about that, so the
   editor being open looked like luck. It stays on screen for the problem. */
function markUnlocked(text){
  let ok = $("designOK");
  if (!ok){
    ok = document.createElement("div");
    ok.id = "designOK";
    ok.className = "designOK";
    ok.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none"
        stroke="currentColor" stroke-width="2.6" stroke-linecap="round"
        stroke-linejoin="round" aria-hidden="true"><path d="m4 12 5.5 5.5L20 7"/></svg>
      <span></span>`;
    $("workCard").insertBefore(ok, $("workStep"));
  }
  ok.querySelector("span").textContent = text;
}

/* Fix the plan where it stands and say so on the card, so a student who keeps
   chatting is not left wondering why the drawing stopped following them. */
function freezePlan(){
  planFrozen = true;
  const ping = $("planPing"); if (ping) ping.remove();
  if ($("planLock")) $("planLock").hidden = false;
  if ($("planSource")) $("planSource").hidden = true;
  if ($("planHint")) $("planHint").textContent =
    "This is the plan your design was approved on, so it stays as it is now. "
    + "Keep chatting for help while you code - it will not change the drawing.";
}

// Review rounds are kept separate from chatLog: the reviewer judges "did they
// fix what I asked last round", and mixing it with the tutor conversation makes
// both harder for the model to follow.
let designLog = [];
let designFileRef = null;

// The plan graph lives here between calls. It round-trips to the server on each
// refresh because it belongs to a STUDENT and there is nobody to key it to yet
// - see main/archive.py for where it goes once PSU login is live.
let planGraph = null, planSeen = new Set();
// Refreshes run one at a time, in order, and NONE IS EVER DROPPED.
//
// This was a plain `planBusy` flag with `if (planBusy) return;`, which threw
// away any refresh that arrived while one was in flight and never retried it.
// The turn that got thrown away was reliably the LAST one - the message that
// answered the tutor's final question and made it say "sounds like you have a
// plan" - so the student pressed submit and the reviewer judged a graph that
// was missing the very step they had just explained, and sent them back for it.
let planQueue = Promise.resolve();
// FROZEN once the design is approved. The plan that passed the gate is the plan
// the student is now implementing, and letting the chat keep rewriting it means
// the picture beside the editor stops being the thing that was approved - so a
// student asking "why is my loop wrong" could watch their accepted plan quietly
// mutate underneath them. The chat carries on; only the drawing is fixed.
let planFrozen = false;
// Whether the page has already nudged down to the plan card this problem. The
// nudge happens ONCE, when the plan first has something in it.
let planNudged = false;

/* Draw the plan graph, flashing whatever is new since the last draw.

   The card does not YANK the page to itself on every change - it is visible
   from the moment a problem opens (showing its empty state), and later changes
   announce themselves with a highlight plus a "Plan updated" pill on the chat
   panel, which is where the student is actually looking.

   The FIRST time it has anything in it is different, and is the one moment the
   page moves on its own. A student who has just described their approach has no
   reason to believe anything was drawn from it, and the pill alone was being
   missed: it appears on the far side of the screen from the sentence they just
   typed. So the first plan gets a short nudge downwards - enough to bring the
   top of the drawing into view, not so much that the chat leaves it. */
function paintPlan(graph){
  const ids = (graph && graph.nodes || []).map(n => n.id);
  const fresh = ids.filter(id => !planSeen.has(id));
  ids.forEach(id => planSeen.add(id));
  renderGraph($("planLive"), graph,
    "Describe your approach in the chat and your plan will appear here.",
    {height: 420, flash: fresh, legend: !!ids.length});
  syncPlanSubmit();
  if (!fresh.length) return;
  if (planNudged) return showPlanPing();
  planNudged = true;
  nudgeToPlan();
}

/* The one-time nudge. Deliberately NOT scrollIntoView: that centres the card
   and pushes the chat off screen, which is the behaviour the pill replaced. A
   fixed, modest scroll keeps both in frame and stays out of the way of a
   student who is already scrolling themselves. */
function nudgeToPlan(){
  const card = $("planCard");
  if (!card || card.hidden) return;
  const top = card.getBoundingClientRect().top;
  const want = window.innerHeight * 0.55;      // bring its top into the lower half
  if (top <= want) return;                     // already visible enough
  scrollBy({top: Math.min(top - want, 260), behavior: "smooth"});
}

/* "Plan updated", on the chat panel, because the graph is a full page-width
   card below the fold and nobody watches two places at once. */
function showPlanPing(){
  if ($("planPing")) return;
  const b = document.createElement("button");
  b.type = "button";
  b.id = "planPing";
  b.className = "planPing";
  b.textContent = "Plan updated ↓";
  b.onclick = () => {
    b.remove();
    $("planCard").scrollIntoView({behavior: "smooth", block: "center"});
  };
  $("tutorMode").after(b);
}

/* Queue a redraw and hand back a promise for when the queue is empty.

   Callers on the chat path ignore the promise - the drawing must never delay
   the tutor's reply. submitPlanGraph() awaits it, because there the graph stops
   being a drawing and becomes the thing that gets judged. */
function refreshPlanGraph(){
  planQueue = planQueue.then(drawPlanGraph, drawPlanGraph);
  return planQueue;
}

async function drawPlanGraph(){
  // A failure must leave the last good picture on screen.
  //
  // Once the design is approved the plan is FIXED. Re-extracting it from a
  // conversation that is now about debugging would quietly rewrite the plan the
  // student was approved on - and the later the conversation goes, the less it
  // is about the plan at all, so the redraw gets worse as it gets less welcome.
  if (planFrozen || !openProblem) return;
  try {
    const r = await fetch(`${API}/plan_graph`, {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({slug: openProblem.slug,
                            messages: chatLog.concat(designLog),
                            current: planGraph})
    });
    if (!r.ok) return;
    const g = await r.json();
    if (g && g.nodes && g.nodes.length){
      planGraph = g;
      paintPlan(g);
    }
  } catch (e) {
    /* keep the previous drawing */
  }
}

async function showDualGraphs(sid){
  try {
    const r = await fetch(`${API}/graphs`, {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({session_id: sid, plan: planGraph})
    });
    if (!r.ok) return;
    const payload = await r.json();
    $("dualCard").hidden = false;
    renderDual($("dualWrap"), payload);
    // The standalone "Your plan so far" card is now a second, smaller copy of
    // the graph sitting directly above one that shows the same thing beside the
    // code. Retire it: it was there to grow WITH the conversation, and the
    // conversation is over. openProblem() unhides it again for the next problem.
    $("planCard").hidden = true;
    const ping = $("planPing"); if (ping) ping.remove();
    $("dualCard").scrollIntoView({behavior: "smooth", block: "start"});
  } catch (e) {
    /* the comparison is a bonus, never the reason a finished session breaks */
  }
}

/* ---------- the design gate ---------- */

const DESIGN_TYPES = ["image/png", "image/jpeg", "application/pdf"];
const DESIGN_MAX = 10 * 1024 * 1024;

function designMsg(kind, text){
  $("designMsg").innerHTML = text
    ? `<div class="banner pre ${kind}">${esc(text)}</div>` : "";
}

function clearDesign(){
  designFileRef = null;
  $("designFile").value = "";
  $("designPick").hidden = true;
  $("designPreview").innerHTML = "";
  $("designDrop").classList.remove("bad");
  $("designDrop").hidden = false;
  disable($("designBtn"), "Choose a PNG, JPEG or PDF first.");
}

function pickDesign(file){
  if (!file) return;
  if (!DESIGN_TYPES.includes(file.type)){
    $("designDrop").classList.add("bad");
    designMsg("bad", `${file.name} is not a PNG, JPEG or PDF. Export your `
      + `diagram as one of those and try again.`);
    return;
  }
  if (file.size > DESIGN_MAX){
    $("designDrop").classList.add("bad");
    designMsg("bad", `${file.name} is ${fmtBytes(file.size)}. The limit is `
      + `${fmtBytes(DESIGN_MAX)} - a photo at a lower resolution will do.`);
    return;
  }
  designFileRef = file;
  designMsg("", "");
  $("designDrop").classList.remove("bad");
  $("designDrop").hidden = true;
  $("designPick").hidden = false;
  $("designName").textContent = file.name;
  $("designName").title = file.name;
  $("designSize").textContent = fmtBytes(file.size);
  // A thumbnail catches "I attached the wrong photo" before a model call, not
  // after one. A PDF has no cheap preview, so it gets a marker instead.
  const prev = $("designPreview");
  prev.innerHTML = "";
  if (file.type === "application/pdf"){
    prev.innerHTML = `<span class="pdfmark">PDF</span>`;
  } else {
    const img = document.createElement("img");
    img.className = "thumb";
    img.alt = `Preview of ${file.name}`;
    img.src = URL.createObjectURL(file);
    img.onload = () => URL.revokeObjectURL(img.src);
    prev.appendChild(img);
  }
  enable($("designBtn"));
}

async function uploadDesign(){
  const btn = $("designBtn");
  if (!designFileRef) return designMsg("warn", "Choose a PNG, JPEG or PDF first.");
  const fd = new FormData();
  fd.append("slug", openProblem.slug);
  fd.append("history", JSON.stringify(designLog));
  // The tutor conversation too. /plan_graph already builds the plan from
  // chatLog.concat(designLog); the reviewer was the one place still judging
  // the drawing on its own, which is how a plan the tutor had just called
  // workable came back rejected for a step the student had already explained.
  fd.append("chat", JSON.stringify(chatLog));
  fd.append("design", designFileRef);

  setBusy(btn, true, "Reviewing…");
  designMsg("info", "Reading your design. This takes a few seconds.");
  try {
    const r = await fetch(`${API}/design_review`, {method: "POST", body: fd});
    const data = await r.json();
    if (!r.ok){
      const m = (data.detail && data.detail.message)
        || "Could not review that design. Try again.";
      designMsg("bad", m);
      return;
    }
    designLog.push({role: "user", content: "[submitted a design]"});
    designLog.push({role: "assistant", content: data.reply});
    // The chips are anchored under the greeting; once anything else has been
    // said they are stranded in the middle of the transcript.
    hideChips();
    hideFork();            // answered by submitting, whichever way they went
    bubble("me", `Submitted design: ${designFileRef.name}`);
    bubble("bot", data.reply);
    // The verdict belongs in the panel the student is looking at, not only in
    // a chat bubble on the other side of the screen.
    designMsg(data.approved ? "ok" : "warn", data.reply);
    if (data.approved) openGate();
    // A graph read straight off their drawing beats one scraped from chat
    // prose, so it seeds planGraph before any chat-based refresh runs.
    if (data.plan_graph && data.plan_graph.nodes && data.plan_graph.nodes.length){
      planGraph = data.plan_graph;
      paintPlan(planGraph);
    }
    refreshPlanGraph();
  } catch (e) {
    designMsg("bad", "Could not reach the reviewer. Check your connection and "
      + "try again - nothing was submitted.");
  } finally {
    setBusy(btn, false);
    if (!designFileRef) disable(btn, "Choose a PNG, JPEG or PDF first.");
  }
}

/* Submit the plan the page drew from this student's chat, instead of a file.

   Shares everything with uploadDesign() below the network call: the same
   reviewer, the same rubric, the same transcript bookkeeping, the same unlock.
   What it skips is the screenshot round trip. */
async function submitPlanGraph(){
  const btn = $("planSubmitBtn");
  setBusy(btn, true, "Reviewing…");
  designMsg("info", "Reading your plan. This takes a few seconds.");
  try {
    // CATCH THE GRAPH UP FIRST. It is drawn in the background while the student
    // reads each reply, so the last thing they said is routinely not in it yet -
    // and the last thing they said is what made the tutor answer "sounds like
    // you have a plan". Submitting whatever happened to be lying around meant
    // the reviewer judged the conversation minus its final turn and sent them
    // back for the step they had just given. Anywhere else a stale drawing is
    // cosmetic; here it is the submission.
    await refreshPlanGraph();
    if (!planGraph || !(planGraph.nodes || []).length){
      return designMsg("warn", "There is no plan yet - describe your approach in "
        + "the chat and it will appear below.");
    }
    const r = await fetch(`${API}/design_review/plan`, {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({slug: openProblem.slug, graph: planGraph,
                            history: designLog, messages: chatLog})
    });
    const data = await r.json();
    if (!r.ok){
      return designMsg("bad", (data.detail && data.detail.message)
        || "Could not review that plan. Try again.");
    }
    designLog.push({role: "user", content: "[submitted the plan from my chat]"});
    designLog.push({role: "assistant", content: data.reply});
    hideChips();
    hideFork();
    bubble("me", "Submitted my plan from the chat");
    bubble("bot", data.reply);
    designMsg(data.approved ? "ok" : "warn", data.reply);
    if (data.approved) openGate();          // freezes the plan as it stands
  } catch (e) {
    designMsg("bad", "Could not reach the reviewer. Check your connection and "
      + "try again - nothing was submitted.");
  } finally {
    setBusy(btn, false);
  }
}

/* The button only exists when there is something to send. Called from every
   place the plan can change, so it never sits there promising to submit an
   empty drawing. */
function syncPlanSubmit(){
  const row = $("planSubmitRow");
  if (!row) return;
  row.hidden = tutorReleased || !(planGraph && (planGraph.nodes || []).length);
}

/* ---------- starting a problem over ---------- */

let restartPrev = null;      // focus to restore when the dialog closes

function openRestart(){
  if (!openProblem) return;
  restartPrev = document.activeElement;
  $("restartModal").hidden = false;
  $("restartNo").focus();
}

function closeRestart(){
  $("restartModal").hidden = true;
  if (restartPrev && restartPrev.focus) restartPrev.focus();
}

/* Confirmed. The server writes a marker rather than deleting anything, so this
   attempt stays in the instructor's transcript; what the STUDENT gets back is
   an untouched problem. Then just re-open it - start() already rebuilds every
   panel from scratch, so there is no second reset path to keep in step. */
async function doRestart(){
  const btn = $("restartYes");
  const p = openProblem;
  if (!p) return;
  setBusy(btn, true, "Starting over\u2026");
  try {
    const r = await fetch(`${API}/problems/${encodeURIComponent(p.slug)}/restart`,
                          {method: "POST"});
    if (!r.ok){
      let m = "Could not start over. Try again.";
      try { m = (await r.json()).detail.message || m; } catch (e) {}
      return toast(m, "bad");
    }
    closeRestart();
    OPENED.add(p.slug);
    SOLVED.delete(p.slug);
    ASSISTED.delete(p.slug);
    await start(p);
    toast("Started over. This problem is fresh again.", "ok");
  } catch (e) {
    toast("Could not reach the server. Nothing was changed.", "bad");
  } finally {
    setBusy(btn, false);
  }
}

/* ---------- the completed file ----------

   The assignment file back, with their own answers under each `def`. Offered
   as soon as ONE problem is finished rather than only at 100%: the file is
   runnable at every stage - unfinished problems keep the body they were handed
   - and a student who has done four of eleven has something worth keeping. */
function syncHandback(done, total){
  const row = $("handbackRow");
  if (!row) return;
  row.hidden = !done;
  if (!done) return;
  const all = done === total;
  $("handbackTitle").textContent = all
    ? "Your completed file" : "Your file so far";
  $("handbackSub").textContent = all
    ? "Every problem, with your answers in place."
    : `Your answers for ${done} of ${total}. The rest is left exactly as it was `
      + `handed out, so the file still runs.`;
}

async function downloadHandback(){
  const btn = $("handbackBtn");
  if (!openAssign) return;
  setBusy(btn, true, "Building…");
  try {
    const r = await fetch(`${API}/assignments/${openAssign.id}/handback`);
    if (!r.ok){
      let m = "Could not build your file. Try again.";
      try { m = (await r.json()).detail.message || m; } catch (e) {}
      return toast(m, "bad");
    }
    // Blob + a temporary link, not a plain navigation: a navigation to a failed
    // request would leave the student staring at a JSON error page instead of
    // a message they can act on.
    const name = (r.headers.get("Content-Disposition") || "")
      .match(/filename="([^"]+)"/);
    const url = URL.createObjectURL(await r.blob());
    const a = document.createElement("a");
    a.href = url;
    a.download = name ? name[1] : "assignment.py";
    document.body.appendChild(a);
    a.click();
    a.remove();
    // Revoked on the next tick, not immediately: Safari cancels an in-flight
    // download if the object URL disappears in the same frame as the click.
    setTimeout(() => URL.revokeObjectURL(url), 30000);
    toast("Downloaded.", "ok");
  } catch (e) {
    toast("Could not reach the server. Check your connection.", "bad");
  } finally {
    setBusy(btn, false);
  }
}

/* ==========================================================
   Choosing an assignment, then a problem
   ----------------------------------------------------------
   Both screens answer the same question: where was I? The server knows which
   problems this student has SOLVED (GET /solved). It does not record which
   ones were merely opened, or when - so "in progress" and "last worked on"
   come from a local note this browser keeps. That is honest about what it is:
   per-browser, and it never contradicts the server, because a solved problem
   is solved whatever the note says.
   ========================================================== */

const CHEV = `<svg class="chev" width="16" height="16" viewBox="0 0 24 24" fill="none"
  stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"
  aria-hidden="true"><path d="m9 6 6 6-6 6"/></svg>`;

let SOLVED = new Set();      // did it themselves
let ASSISTED = new Set();    // finished it, but a step used the shown answer
// Opened but not finished. From the SERVER (mt_sessions), not from a note this
// browser keeps: the old localStorage version outlived a wiped account, did not
// follow a student to a second machine, and no instructor could see it.
let OPENED = new Set();
// The most recently opened problem, for the "Continue" row. Server-side too.
let LAST_SLUG = null;
/* Both count as DONE. A problem completed with help is not left on the list as
   unfinished work: the student has nothing further to do on it, and telling
   them otherwise sends them back to redo something they have already been
   through. What it is NOT is the same claim - see STAT_LABEL - and it is not a
   grade either, which is counted from passing submissions and never from
   these. */
function isDone(slug){ return SOLVED.has(slug) || ASSISTED.has(slug); }
let ASSIGNMENTS = [];          // enriched with slugs / solved / last
let openAssign = null;         // the assignment currently open
let PROBLEMS = [];             // its problems
let pQuery = "", pFilter = "all", pSort = "alpha";
let landingLinkHandled = false;

const TOUCH_KEY = "mt.touched";
/* RETIRED. "In progress" came from a note this browser kept, which survived a
   wiped account, never followed a student to a second machine, and no
   instructor could see. It is mt_sessions now - see /solved. The old key is
   cleared on load so a stale note cannot outlive the change. */
function clearLegacyTouch(){
  try { localStorage.removeItem(TOUCH_KEY); } catch (e) {}
}

function statusOf(slug){
  if (SOLVED.has(slug)) return "solved";
  if (ASSISTED.has(slug)) return "helped";
  return OPENED.has(slug) ? "progress" : "todo";
}
const STAT_LABEL = {solved: "Solved", helped: "Solved with help",
                    progress: "In progress", todo: "Not started"};
function statMark(state){
  return `<span class="stat ${state}"><span class="dotmark" aria-hidden="true"></span>${
    STAT_LABEL[state]}</span>`;
}
function bar(pct, done){
  return `<span class="bar slim rowbar${done ? " done" : ""}" role="img"
    aria-label="${pct}% solved"><i style="width:${pct}%"></i></span>`;
}

function listError(host, retry){
  host.innerHTML = `<div class="banner bad" style="margin:0">Could not reach the
    server. <button class="retry" type="button">Try again</button></div>`;
  host.querySelector(".retry").onclick = retry;
}

async function loadAssignments(){
  paintStudentStats(null);
  $("assignCard").innerHTML = skeletonRows(2);
  let d, solved;
  try {
    [d, solved] = await Promise.all([
      fetch(`${API}/assignments`).then(r => r.json()),
      // A student with no solves yet is a 200 with an empty list; a failure
      // here must not cost them the assignment list, so it degrades to "none".
      fetch(`${API}/solved`).then(r => r.ok ? r.json() : {slugs: []})
                            .catch(() => ({slugs: []}))
    ]);
  } catch {
    return listError($("assignCard"), loadAssignments);
  }
  SOLVED = new Set(solved.slugs || []);
  ASSISTED = new Set(solved.assisted || []);
  OPENED = new Set(solved.opened || []);
  LAST_SLUG = solved.last_slug || null;

  const list = (d.assignments || []).filter(a => a.ready > 0 && a.published !== false);
  if (!list.length){
    paintStudentStats([]);
    $("assignCard").innerHTML = `<div class="empty">
      <span class="eicon" aria-hidden="true">
        <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor"
             stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M4 5h16v15H4z"/><path d="M8 10h8M8 14h5"/></svg></span>
      <p>Nothing to practice yet.</p>
      <p class="esub">Your instructor has not prepared an assignment. This page
        fills in as soon as one is ready.</p></div>`;
    return;
  }

  // One extra request per assignment, to learn which slugs it contains. With a
  // handful of assignments that is cheaper than a new endpoint; if a course
  // ever has fifty, /assignments should carry the slugs itself.
  ASSIGNMENTS = await Promise.all(list.map(async a => {
    let slugs = [], failed = false;
    try {
      const r = await fetch(`${API}/assignments/${a.id}/problems`);
      if (!r.ok) throw new Error();
      const j = await r.json();
      a._problems = j.problems || [];
      slugs = a._problems.map(p => p.slug);
    } catch {
      // The row still opens - openAssignment() retries the fetch. What it must
      // NOT do is print "0 / 0 solved", which reads as "you have done nothing"
      // when the truth is "we could not ask".
      failed = true;
    }
    const solvedN = slugs.filter(isDone).length;
    const last = LAST_SLUG && slugs.includes(LAST_SLUG) ? 1 : 0;
    return {...a, slugs, solvedN, last, failed};
  }));

  paintStudentStats(ASSIGNMENTS);

  // The "Continue" affordance goes on the one they touched most recently, so
  // the common case - come back, carry on - is one click.
  const resumeId = ASSIGNMENTS.reduce(
    (best, a) => (a.last && (!best || a.last > best.last)) ? a : best, null);

  $("assignCard").innerHTML = `<ul class="rows" id="assignments">${
    ASSIGNMENTS.map(a => {
      const n = a.slugs.length || a.ready;
      const pct = n ? Math.round(a.solvedN / n * 100) : 0;
      // Never surface a number that contradicts the title: the count is the
      // number of problems that are READY, and the ratio is only shown when
      // some did not make it.
      const readyLab = a.total && a.total !== a.ready
        ? `${a.ready} of ${a.total} ready` : `${a.ready} ready`;
      const readyTip = a.total && a.total !== a.ready
        ? `${a.total - a.ready} problem${a.total - a.ready === 1 ? "" : "s"} in this `
          + `assignment could not be prepared, so they are not shown.`
        : "Every problem in this assignment is ready.";
      const resume = !a.failed && resumeId && resumeId.id === a.id;
      return `<li><button class="rowitem${resume ? " resume" : ""}" data-id="${esc(a.id)}">
        <span class="assignment-top"><span class="assignment-glyph" aria-hidden="true">{ }</span>
          <span class="assignment-kind">${resume ? "PICK UP WHERE YOU LEFT OFF" : "PYTHON PRACTICE"}</span></span>
        <span class="rmain"><span class="rname">${esc(a.name)}</span>
          <span class="rmeta" title="${esc(readyTip)}">${readyLab} to practice</span></span>
        <span class="assignment-bottom">
          <span class="assignment-progress">${a.failed ? "Progress unavailable" : `${a.solvedN} of ${n} solved ${bar(pct, pct === 100)}`}</span>
          <span class="go">${resume ? "Continue" : "Open assignment"} ${CHEV}</span>
        </span>
      </button></li>`;
    }).join("")}</ul>`;

  $("assignments").querySelectorAll(".rowitem").forEach(b =>
    b.onclick = () => openAssignment(ASSIGNMENTS.find(a => String(a.id) === b.dataset.id)));
  if (!landingLinkHandled){
    landingLinkHandled = true;
    const params = new URLSearchParams(location.search);
    const requested = params.get("assignment");
    if (requested){
      const assignment = ASSIGNMENTS.find(a => String(a.id) === requested);
      if (assignment){
        await openAssignment(assignment);
        const slug = params.get("problem");
        const problem = PROBLEMS.find(p => p.slug === slug);
        if (problem) await start(problem);
        else if (slug) toast("That problem is not available. Choose another from this assignment.", "warn");
      } else toast("That assignment is not available right now.", "warn");
    }
  }
}

// Summaries use the assignment data already loaded for the cards.
function paintStudentStats(rows){
  const values = rows ? [rows.length, rows.reduce((n,a) => n + (a.ready || 0), 0),
    rows.some(a => a.failed) ? "—" : rows.reduce((n,a) => n + (a.solvedN || 0), 0)] : ["—", "—", "—"];
  $("studentStats").querySelectorAll("[data-metric]").forEach((el,i) => el.textContent = values[i]);
}

function view(which){
  document.body.classList.toggle("in-studio", which === "cSolve");
  setWorkspaceFocus(false);
  setTutorOpen(false, false);
  ["cAssign","cProblem","cSolve"].forEach(id => $(id).hidden = id !== which);
  scrollTo({top: 0, behavior: "smooth"});
}

function goAssignments(){
  history.replaceState(null, "", location.pathname);
  dismissCoach(false);                    // see backToProblems
  document.title = "My assignments · ACADIA";
  setCrumbs([]);
  openAssign = null;
  view("cAssign");
  loadAssignments();                    // solves may have landed since
}

async function openAssignment(a){
  if (!a) return;
  history.replaceState(null, "", `${location.pathname}?assignment=${encodeURIComponent(a.id)}`);
  openAssign = a;
  pQuery = ""; pFilter = "all"; pSort = "alpha";
  $("pfilter").value = "";
  $("psort").value = "alpha";
  $("pchips").querySelectorAll(".chip").forEach(c =>
    c.setAttribute("aria-pressed", String(c.dataset.f === "all")));
  $("assignName").textContent = a.name;
  document.title = `${a.name} · ACADIA`;
  // A breadcrumb in the header, so getting back one level does not mean going
  // all the way Home.
  setCrumbs([{label: a.name}]);
  view("cProblem");

  if (a._problems){ PROBLEMS = a._problems; return renderProblems(); }

  // The list screen's own fetch for this assignment failed (or was never made).
  // Try again here, where there is room to say so and offer a retry.
  PROBLEMS = [];
  $("probSummary").textContent = "";
  $("problems").innerHTML = `<li style="grid-column:1/-1; display:block">${
    skeletonRows(3)}</li>`;
  try {
    const r = await fetch(`${API}/assignments/${a.id}/problems`);
    if (!r.ok) throw new Error();
    a._problems = (await r.json()).problems || [];
  } catch {
    $("problems").innerHTML = `<li style="grid-column:1/-1; display:block">
      <div class="banner bad" style="margin:0">Could not load the problems in
      this assignment. <button class="retry" type="button">Try again</button>
      </div></li>`;
    $("problems").querySelector(".retry").onclick = () => openAssignment(a);
    return;
  }
  PROBLEMS = a._problems;
  renderProblems();
}

/* The problem list: left-aligned names, status on the right, filtered and
   sorted in one place. */
function renderProblems(){
  const q = pQuery.trim().toLowerCase();
  const rank = {progress: 0, todo: 1, helped: 2, solved: 3};

  const rows = PROBLEMS
    .map(p => ({p, state: statusOf(p.slug)}))
    .filter(({p, state}) =>
      // The Solved chip means FINISHED - a student filtering for what they
      // have done should not have half of it hidden behind a distinction they
      // did not ask about.
      (pFilter === "all" || state === pFilter ||
       (pFilter === "solved" && state === "helped")) &&
      (!q || String(p.title || p.slug).toLowerCase().includes(q) ||
             String(p.slug).toLowerCase().includes(q)))
    .sort((a, b) => pSort === "status"
      ? (rank[a.state] - rank[b.state]) ||
        String(a.p.title || a.p.slug).localeCompare(String(b.p.title || b.p.slug))
      : String(a.p.title || a.p.slug).localeCompare(String(b.p.title || b.p.slug)));

  const solvedN = PROBLEMS.filter(p => isDone(p.slug)).length;
  const pct = PROBLEMS.length ? Math.round(solvedN / PROBLEMS.length * 100) : 0;
  $("probSummary").textContent = PROBLEMS.length
    ? `${solvedN} of ${PROBLEMS.length} solved` : "";
  $("probSummaryBar").querySelector("i").style.width = pct + "%";
  $("probSummaryBar").setAttribute("aria-valuenow", String(pct));
  $("probSummaryBar").classList.toggle("done", pct === 100);
  syncHandback(solvedN, PROBLEMS.length);

  if (!PROBLEMS.length){
    $("problems").innerHTML = `<li style="grid-column:1/-1"><div class="empty"
      style="width:100%">
      <p>No problems are ready in this assignment.</p>
      <p class="esub">Your instructor is still preparing them.</p></div></li>`;
    return;
  }
  if (!rows.length){
    $("problems").innerHTML = `<li style="grid-column:1/-1"><div class="empty"
      style="width:100%">
      <p>Nothing matches that.</p>
      <p class="esub">Try a different word, or clear the filter.</p></div></li>`;
    return;
  }

  const rowHTML = ({p, state}) => `
    <button class="rowitem" data-slug="${esc(p.slug)}">
      <span class="rmain">
        <span class="rname">${esc(p.title || p.slug)}</span>
        <span class="rmeta">
          ${p.difficulty ? `<span>${esc(p.difficulty)}</span>` : ""}
          ${(p.topic_tags || []).slice(0, 2).map(t =>
              `<span>${esc(t)}</span>`).join('<span aria-hidden="true">·</span>')}
        </span>
      </span>
      ${statMark(state)}
      ${CHEV}
    </button>`;

  /* A class is ONE assignment with steps, not five unrelated problems. Its
     methods arrive as separate rows carrying the same group_slug, so they are
     folded back into a single card here, in the order the teacher wrote them
     (member_order) rather than the list's own sort - "peek before push" is not
     a thing anyone was asked to do.

     <details> rather than a JS open/closed map: the browser already keeps that
     state, exposes it to a screen reader, and makes the summary keyboard-
     operable for free. */
  const groups = [], byKey = new Map();
  for (const r of rows){
    const key = r.p.group_slug;
    if (!key){ groups.push({solo: r}); continue; }
    let g = byKey.get(key);
    if (!g){
      g = {key, title: r.p.group_title || key, members: []};
      byKey.set(key, g);
      groups.push(g);
    }
    g.members.push(r);
  }
  for (const g of groups)
    if (g.members)
      g.members.sort((a, b) => (a.p.member_order ?? 0) - (b.p.member_order ?? 0));

  $("problems").innerHTML = groups.map(g => {
    if (g.solo) return `<li>${rowHTML(g.solo)}</li>`;
    /* Counted over every offered member of the class, not just the ones
       surviving the current filter - otherwise typing in the search box would
       quietly change the denominator and read as lost progress. */
    const all = PROBLEMS.filter(p => p.group_slug === g.key);
    const done = all.filter(p => isDone(p.slug)).length;
    const complete = done === all.length;
    return `<li class="grouprow"><details class="pgroup"${complete ? "" : " open"}>
      <summary>
        <span class="rmain">
          <span class="rname">${esc(g.title)}</span>
          <span class="rmeta"><span>${all.length} step${
            all.length === 1 ? "" : "s"}</span></span>
        </span>
        <span class="gcount${complete ? " done" : ""}">${done} of ${
          all.length} solved</span>
        ${CHEV}
      </summary>
      <ul class="rows gmembers">${
        g.members.map(m => `<li>${rowHTML(m)}</li>`).join("")}</ul>
    </details></li>`;
  }).join("");

  $("problems").querySelectorAll(".rowitem").forEach(b =>
    b.onclick = () => start(PROBLEMS.find(p => p.slug === b.dataset.slug)));
}

$("pfilter").addEventListener("input", e => { pQuery = e.target.value; renderProblems(); });
$("psort").addEventListener("change", e => { pSort = e.target.value; renderProblems(); });
$("pchips").addEventListener("click", e => {
  const c = e.target.closest(".chip");
  if (!c) return;
  pFilter = c.dataset.f;
  $("pchips").querySelectorAll(".chip").forEach(x =>
    x.setAttribute("aria-pressed", String(x === c)));
  renderProblems();
});

/* Preserve source examples and indentation; prose alone gets reflowed. */
function renderStatement(desc){
  renderLearningText($("statement"), desc, renderExamples);
}

/* "fn(args) -> result", one or more, comma / semicolon / newline separated.
   Returns a node, or null when nothing parsed - the caller then falls back to
   showing the paragraph as plain prose. */
function renderExamples(blob){
  const t = blob.replace(/\s*\n\s*/g, " ").trim();
  const re = /([A-Za-z_]\w*\s*\([^()]*\))\s*(?:->|=>|→)\s*("(?:[^"\\]|\\.)*"|\[[^\]]*\]|\{[^}]*\}|[^,;]+?)(?=\s*[,;]\s*[A-Za-z_]\w*\s*\(|\s*$)/g;
  const pairs = [];
  let m;
  while ((m = re.exec(t))) pairs.push([m[1].trim(), m[2].trim()]);
  if (!pairs.length) return null;
  // Preserve the full source when a nested expression or explanation cannot
  // be completely parsed into input/output pairs.
  if (t.replace(re, "").replace(/[;,]/g, "").trim()) return null;

  const wrap = document.createElement("div");
  wrap.className = "examples";
  wrap.innerHTML = `<p class="ex-label">${pairs.length > 1 ? "Examples" : "Example"}</p>`
    + pairs.map(([inp, out]) =>
        `<div class="ex-case">
           <div class="io"><span>Input</span><code>${esc(inp)}</code></div>
           <div class="io"><span>Output</span><code>${esc(out)}</code></div>
         </div>`).join("");
  return wrap;
}

async function start(p){
  if (!p) return;
  if (openAssign) history.replaceState(null, "", `${location.pathname}?assignment=${encodeURIComponent(openAssign.id)}&problem=${encodeURIComponent(p.slug)}`);
  view("cSolve");
  const name = p.title || p.slug;
  document.title = `${name} · ACADIA`;
  $("probTitle").textContent = name;
  // Opening a problem is what makes it "in progress", and the SERVER records
  // that: /decompose_chunks opens an mt_sessions row, which /solved reads back
  // as `opened`. Marked locally too so the list behind is right immediately,
  // without waiting for a round trip.
  OPENED.add(p.slug);
  const asg = openAssign ? openAssign.name : ($("assignName").textContent || "").trim();
  $("crumb2").textContent = asg ? `${asg}  ›  ${name}` : name;
  setCrumbs([
    ...(openAssign ? [{label: openAssign.name, go: backToProblems}] : []),
    {label: name}
  ]);
  renderStatement(p.description);
  stepperLoading();
  $("stepCount").textContent = "";
  $("prompt").textContent = "";
  $("problemDetails").open = true;
  header = "";
  reviewIdx = null;
  reviewDraft = null;
  $("ctx").style.maxHeight = "";          // a previous finish() may have opened it
  $("submit").style.display = ""; $("backP").style.display = "";
  $("msg").innerHTML = `<div class="banner info">Getting this problem ready…</div>`;
  $("attempts").textContent = "";
  accepted = []; idx = 0;
  openProblem = p;
  tutorReleased = false;
  designLog = [];
  planGraph = null;
  planSeen = new Set();
  planFrozen = false;
  planNudged = false;
  if ($("planLock")) $("planLock").hidden = true;
  if ($("planSource")) $("planSource").hidden = false;
  if ($("planHint")) $("planHint").textContent =
    "This grows as you describe your approach. If a step is missing here, "
    + "it was missing from what you said.";
  syncPlanSubmit();
  const ok = $("designOK"); if (ok) ok.remove();
  const ping = $("planPing"); if (ping) ping.remove();
  clearDesign();
  designMsg("", "");
  // The plan card is visible from the start, showing its empty state. A card
  // that appears out of nowhere three replies in is a card nobody notices.
  $("planCard").hidden = false;
  $("dualCard").hidden = true;
  // NOT resetChat() and NOT paintPlan(null) yet: both assert this is a fresh
  // start, and we do not know that until /history answers. See historyLoading.
  historyLoading();
  applyTutorGate();
  setReviewMode(false);

  // No solution is sent. The server reads the reference from the
  // database, so the browser never holds it.
  let r;
  try {
    r = await fetch(`${API}/decompose_chunks`, {
      method: "POST", headers: {"Content-Type": "application/json"},
      // No student_id: the server binds the session to whoever the cookie
      // says we are. Sending a name here used to let anyone claim anyone.
      body: JSON.stringify({slug: p.slug, title: p.title,
                            description: p.description})
    });
  } catch { return failStart("Could not reach the server."); }
  if (!r.ok) return failStart("This problem could not be started. Try another one.");

  const d = await r.json();
  sessionId = d.session_id; chunks = d.chunks || []; header = d.header || "";
  $("msg").innerHTML = "";
  ensureEditor();
  render();
  applyTutorGate();
  // Nothing to put back, so this IS a fresh start - say hello and draw the
  // empty plan. Skipped when history was restored (it painted both already) and
  // when the student has already moved to another problem.
  if (!await restoreHistory(p)){
    resetChat(p.title || p.slug);
    paintPlan(null);
  }
}

/* Put back what this student already did on this problem.

   Everything below has been archived server-side all along (main/archive.py);
   nothing read it, so reopening a problem showed an empty chat, an empty plan
   graph and a locked editor - and looked exactly like the work had been
   deleted. It never was.

   Deliberately NOT restored: the accepted code and the step position. Those
   belong to a GRADING SESSION, and /decompose_chunks just issued a fresh one -
   replaying old answers into it would claim steps this session never graded.
   Reopening a problem still starts the solving over; what comes back is the
   thinking around it.

   Returns TRUE when it has painted the chat and the plan itself, so the caller
   knows not to overwrite them with a fresh greeting - and true as well when the
   student has moved on mid-fetch, because whatever they moved to owns those
   panels now. False means "nothing here, start clean". */
async function restoreHistory(p){
  let h;
  try {
    const r = await fetch(`${API}/history/${encodeURIComponent(p.slug)}`);
    if (!r.ok) { setRestoring(false); return false; }
    h = await r.json();
  } catch { setRestoring(false); return false; }   // a cold archive is fine
  if (openProblem !== p) return true;              // they moved on; hands off
  if (!h || !h.found) { setRestoring(false); return false; }   // nothing recorded

  if (h.messages && h.messages.length){
    $("clog").innerHTML = "";
    hideChips();
    // They have talked to the tutor about this problem before, so pointing at
    // it now is noise. Remembered, for the same reason.
    dismissCoach(true);
    chatLog = h.messages.map(m => ({role: m.role, content: m.content}));
    h.messages.forEach(m => bubble(m.role === "user" ? "me" : "bot",
                                   m.content, m.at));
    bubble("bot", h.solved
      ? "This is where we got to last time. You have already solved this one - "
        + "work through it again whenever you like."
      : "Picking up where we left off.");
  }

  if (h.plan && h.plan.nodes && h.plan.nodes.length){
    planGraph = h.plan;
    // Seed planSeen FIRST so a restored plan does not flash every node as new,
    // ping "Plan updated" about a conversation that finished days ago, or nudge
    // the page down to a drawing the student has already seen.
    h.plan.nodes.forEach(n => planSeen.add(n.id));
    planNudged = true;
    paintPlan(h.plan);
  }

  // The design gate is passed ONCE per problem. Re-demanding the same diagram
  // to reread your own finished work is a toll, not a lesson.
  //
  // This is stated on a STANDING bar rather than through show(), which writes
  // into #msg and is wiped by the next grading verdict. Coming back to a
  // problem you had already unlocked said nothing at all about why the editor
  // was open, so the state read as an accident.
  if (h.design_approved || h.solved){
    tutorReleased = true;
    applyTutorGate();
    markUnlocked(h.solved
      ? "You solved this before. The editor is unlocked - the steps start again from the top."
      : "Your design was accepted earlier. The editor is unlocked for this problem.");
    loadSteps();      // reopening an unlocked problem gets its steps back too
    // An approved plan stays approved, so it stays fixed across sessions too.
    if (planGraph && (planGraph.nodes || []).length) freezePlan();
  }
  syncPlanSubmit();

  // The finished comparison, if there is one to show.
  if (h.plan && h.code && h.comparison){
    $("dualCard").hidden = false;
    renderDual($("dualWrap"), {plan: h.plan, code: h.code,
                               comparison: h.comparison});
    $("planCard").hidden = true;
  }

  // BOTH panels must be resolved before returning true, or the caller leaves
  // them alone and a skeleton shimmers forever. `found` is true for a student
  // who has ANY row on this problem, and a design row alone is enough - so a
  // half-restored problem is a real state, not a defensive hypothetical.
  if (!(h.messages && h.messages.length)) resetChat(p.title || p.slug);
  if (!(h.plan && h.plan.nodes && h.plan.nodes.length)) paintPlan(null);
  setRestoring(false);
  return true;
}

/* Decomposition never arrived. Drop the loading skeleton so it does not shimmer
   forever under an error message. */
function failStart(msg){
  $("stepper").innerHTML = "";
  $("stepTotal").textContent = "";
  $("probBarLab").textContent = "";
  // This path returns from start() before restoreHistory() ever runs, so the
  // chat and plan skeletons raised on the way in have nobody to clear them.
  // A shimmer that never resolves reads as a hung page on top of a failure the
  // student can actually see and act on.
  if (openProblem) resetChat(openProblem.title || openProblem.slug);
  paintPlan(null);
  setRestoring(false);
  show("bad", msg);
}

function ensureEditor(){
  if (editor) return;
  editor = CodeMirror.fromTextArea($("code"), {
    mode: "python", theme: "dracula", lineNumbers: true,
    indentUnit: 4, tabSize: 4, indentWithTabs: false, lineWrapping: true,
    extraKeys: {
      // VSCode-style multi-cursor. Cmd/Ctrl-D adds the NEXT occurrence of the
      // current selection (press again to keep adding); Cmd/Ctrl-Shift-L
      // selects EVERY occurrence at once. Commands come from keymap/sublime.
      "Cmd-D": "selectNextOccurrence",
      "Ctrl-D": "selectNextOccurrence",
      "Shift-Cmd-L": "findAndSelectAll",
      "Shift-Ctrl-L": "findAndSelectAll",
    },
  });
  editor.setSize(null, 300);
  // The focus ring belongs to the WHOLE code surface, not to the lower half of
  // it - the frozen context and the editor are presented as one box, so a ring
  // around only the editable part would give that away as a lie.
  editor.on("focus", () => $("codestack").classList.add("focus"));
  editor.on("blur",  () => $("codestack").classList.remove("focus"));
  // A freshly created editor defaults to editable - re-assert the current gate
  // so it does not spring open the moment fromTextArea() runs.
  applyTutorGate();
}

/* The frozen half of the code surface: the function header plus every step
   already accepted, drawn to line up exactly with the editor below it.

   Returns how many lines it drew, which is what the editor's line numbering
   has to continue from. */
function renderContext(){
  const lines = [];
  lines.push({text: header || "def solve(...):", cls: "cHead"});
  accepted.forEach(a => a.code.split("\n").forEach(
    l => lines.push({text: l ? " ".repeat(BODY_INDENT) + l : "", cls: "cAccept"})));

  $("ctxCode").innerHTML = lines.map(
    l => `<span class="${l.cls}">${esc(l.text) || "&nbsp;"}</span>`).join("\n");
  $("ctxGutter").textContent = lines.map((_, i) => i + 1).join("\n");

  matchGutter();
  // Keep the newest accepted line in view rather than the top of the function.
  $("ctx").scrollTop = $("ctx").scrollHeight;
  return lines.length;
}

/* Match the frozen gutter to the editor's, to the pixel.

   Measuring the live element is the only thing that stays correct across web
   font loading, browser zoom, and the jump from line 9 to line 10 widening the
   column - all of which would otherwise leave the two halves of one listing
   visibly out of step. */
function matchGutter(){
  if (!editor) return;
  const g = editor.getGutterElement();
  const w = g && g.offsetWidth;
  if (!w) return;
  // `*{box-sizing:border-box}` means width INCLUDES the padding, so the frozen
  // column occupies exactly the gutter's width. The 3px right padding is
  // CodeMirror's own `.CodeMirror-linenumber{padding:0 3px 0 5px}`, which is
  // what puts both sets of digits on the same right edge.
  $("ctxGutter").style.width = w + "px";
  $("ctxGutter").style.paddingRight = "3px";
}

/* Placeholder stepper while the server is still splitting the problem into
   steps, so the row resolves in place rather than popping in from nothing. */
function stepperLoading(){
  $("stepper").innerHTML = [110, 90, 74].map(w => `
    <li><span class="steppill"><span class="n"></span>
      <span class="t"><span class="skel" style="width:${w}px"></span></span></span></li>`).join("");
  $("stepTotal").innerHTML = "&hellip;";
  $("probBarLab").textContent = "";
}

/* The stepper: numbers, state, and a way back into anything finished. */
function renderStepper(){
  $("stepper").innerHTML = chunks.map((c, i) => {
    const done = accepted[i];
    const state = i === idx && reviewIdx === null ? "now"
                : done ? (done.how === "revealed" ? "helped" : "own") : "";
    const mark = done ? (done.how === "revealed" ? "!" : "✓") : (i + 1);
    const tag = done && done.how === "revealed" ? " (shown to you)"
              : done ? " (done)" : i === idx ? " (current)" : " (not started yet)";
    const el = done ? "button" : "span";
    const attrs = done
      ? ` type="button" data-i="${i}"`
      : (i === idx ? ` aria-current="step"` : "");
    return `<li><${el} class="steppill ${state}${reviewIdx === i ? " reviewing" : ""}"${attrs}
        aria-label="Step ${i + 1} of ${chunks.length}${tag}: ${esc(c.prompt)}" title="${esc(c.prompt)}">
        <span class="n" aria-hidden="true">${mark}</span>
        <span class="t">Step ${i + 1}</span>
        <span class="sr-only">Step ${i + 1} of ${chunks.length}${tag}</span>
      </${el}></li>`;
  }).join("");
  $("stepper").querySelectorAll("button.steppill").forEach(b =>
    b.onclick = () => openReview(+b.dataset.i));

  const selected = $("stepper").querySelector(".reviewing, .now");
  if (selected) requestAnimationFrame(() => {
    const row = $("stepper");
    const item = selected.getBoundingClientRect();
    const frame = row.getBoundingClientRect();
    if (item.right > frame.right) row.scrollLeft += item.right - frame.right;
    else if (item.left < frame.left) row.scrollLeft -= frame.left - item.left;
  });

  const doneN = accepted.filter(Boolean).length;
  const pct = chunks.length ? Math.round(doneN / chunks.length * 100) : 0;
  $("probBar").querySelector("i").style.width = pct + "%";
  $("probBar").setAttribute("aria-valuenow", String(pct));
  $("probBar").classList.toggle("done", pct === 100);
  $("probBarLab").textContent = chunks.length
    ? `${doneN} / ${chunks.length} steps` : "";
  $("stepTotal").textContent = chunks.length
    ? `${chunks.length} step${chunks.length === 1 ? "" : "s"}` : "";
}

/* Reviewing swaps the code surface for a read-only listing, so there is never
   an editor on screen that silently refuses to accept the step you are in. */
function setReviewMode(on){
  $("codestack").hidden = on;
  $("workControls").hidden = on;
  $("reviewControls").hidden = !on;
  $("reviewBox").hidden = !on;
}

function openReview(i){
  if (!accepted[i]) return;
  if (reviewIdx === null && editor){
    reviewDraft = {value: editor.getValue(), selections: editor.listSelections()};
  }
  reviewIdx = i;
  const c = chunks[i] || {};
  $("stepCount").innerHTML = `Step ${i + 1} of ${chunks.length}`
    + ` <span class="pill ${accepted[i].how === "revealed" ? "warn" : "ok"}">`
    + `${accepted[i].how === "revealed" ? "shown to you" : "your answer"}</span>`;
  $("prompt").textContent = c.prompt || "";
  $("reviewCode").textContent = accepted[i].code;
  setReviewMode(true);
  renderStepper();
  $("msg").innerHTML = "";
  $("attempts").textContent = "";
  $("backToNow").focus();
}

function leaveReview(){
  reviewIdx = null;
  setReviewMode(false);
  render();
  applyTutorGate();
  if (reviewDraft && editor){
    editor.setValue(reviewDraft.value);
    editor.setSelections(reviewDraft.selections);
    reviewDraft = null;
  }
}

function render(){
  if (reviewIdx !== null) return;
  renderStepper();

  const cur = chunks[idx];
  $("stepCount").textContent = chunks.length
    ? (idx < chunks.length ? `Step ${idx + 1} of ${chunks.length}`
                           : `All ${chunks.length} steps complete`)
    : "";
  $("prompt").textContent = cur ? cur.prompt : "";
  $("attempts").textContent = "";

  const shown = renderContext();
  if (editor) {
    // Numbering continues from the frozen part, so the editor is line N+1 of
    // the same function rather than line 1 of a detached box.
    editor.setOption("firstLineNumber", shown + 1);
    // Seat the caret where this step's code actually starts, on the same
    // scale as the frozen listing above it: the function body's own indent
    // plus whatever nesting the step sits at. `indent` alone is the depth
    // INSIDE the body, so a top-level step came out at column 0 - flush with
    // `def`, four columns left of the accepted line it continues from. The
    // server re-seats whatever arrives (main/indent.py), so the extra column
    // costs nothing; what it buys is an editor that lines up with the code
    // above it instead of contradicting it.
    const pad = " ".repeat(BODY_INDENT + ((cur && cur.indent) || 0));
    editor.setValue(pad);
    editor.setCursor({line: 0, ch: pad.length});
    editor.refresh();
    // CodeMirror re-lays out its gutter after the option change, so the width
    // read during renderContext() above was the PREVIOUS one. Re-measure on the
    // next frame, when the new gutter is on screen.
    requestAnimationFrame(matchGutter);
    if (tutorReleased) editor.focus();   // never pull focus into a locked editor
  }
}

/* Mirror of main/indent.py's align_to_chunk, for DISPLAY only.

   The server re-seats every submission at its step's depth and stores that, so
   a flat answer to a step inside a loop is stored indented. If the frozen
   listing kept showing the raw text the student typed, the picture of their own
   function would disagree with the one being graded - and it is the picture
   they use to write the next step. Same rule, same result, both ends. */
function alignToStep(code, columns){
  const lines = String(code || "").replace(/\t/g, "    ").split("\n");
  const filled = lines.filter(l => l.trim());
  if (!filled.length) return "";
  const cut = Math.min(...filled.map(l => l.length - l.trimStart().length));
  const first = lines.findIndex(l => l.trim());
  let last = lines.length - 1;
  while (last >= 0 && !lines[last].trim()) last--;
  const pad = " ".repeat(Math.max(0, columns || 0));
  return lines.slice(first, last + 1)
    .map(l => l.trim() ? pad + l.slice(cut).replace(/\s+$/, "") : "")
    .join("\n");
}

/* `pre`: the grader writes its reason with real paragraph breaks in it.

   Also the one place the "design accepted" notice is retired. It reports a gate
   that opened before the first step, and once there is a verdict to read it is
   simply an older message sitting above a newer one in the same slot - so the
   verdict takes its place rather than stacking under it. */
function show(kind, text, extra){
  const ok = $("designOK");
  if (ok) ok.remove();
  $("msg").innerHTML = `<div class="banner pre ${kind}">${esc(text)}</div>` + (extra || "");
}

$("submit").onclick = async () => {
  const code = editor ? editor.getValue() : "";
  if (!code.trim()) return show("warn", "Write something first.");
  const btn = $("submit");
  setBusy(btn, true, "Grading…");
  // Stable per submission, so a retry of the same answer is never graded twice.
  const submissionId = `${sessionId}:${idx}:${Date.now()}`;
  let r;
  try {
    r = await fetch(`${API}/grade_chunk`, {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({session_id: sessionId, submission_id: submissionId,
                            student_code: code, expected_index: idx})
    });
  } catch {
    setBusy(btn, false);
    return show("bad", "Could not reach the server. Your attempt was not used.");
  }
  setBusy(btn, false);

  if (!r.ok) {
    let m = "Something went wrong. Your attempt was not used.";
    try { const j = await r.json(); if (j.detail && j.detail.message) m = j.detail.message; } catch {}
    return show("warn", m);
  }

  const res = await r.json();

  // Covers the OpenAI outage case. The server guarantees no attempt was spent.
  if (res.verdict === "indeterminate") return show("warn", res.reason);

  if (res.verdict === "correct") {
    // Store it the way the server did, not the way it was typed.
    accepted[idx] = {code: alignToStep(code, (chunks[idx] || {}).indent), how: "own"};
    idx = res.index;
    if (res.completed) return finish(res);
    render();
    show("ok", res.reason);
    return;
  }

  if (typeof res.revealed_reference === "string") {
    accepted[idx] = {code: res.revealed_reference, how: "revealed"};
    idx = res.index;
    if (res.completed) return finish(res);
    render();
    show("warn", res.reason + "\n\nHere is this step so you can keep going:",
         `<pre class="code" style="margin-top:11px">${esc(res.revealed_reference)}</pre>`);
    return;
  }
  // A wrong answer now comes with the case that caught it, folded away. Closed
  // by default because the point of the step is for them to find it themselves;
  // one click away because "wrong on at least one case" and no case is a shrug,
  // and a student with no way forward stops rather than thinks.
  show("bad", res.reason, failingCaseHTML(res.failing_case));
  // "Attempt 3 of 2" would be a lie now that there is no limit and the answer
  // is never shown. A plain count still tells them where they are without
  // implying a countdown to being given it.
  $("attempts").textContent = res.attempts
    ? `Attempt ${res.attempts}. Keep going - take as many as you need.` : "";
};

/* The disclosure holding one failing case. Empty string when the server sent
   none - a crash or a timeout has no case to show. */
function failingCaseHTML(text){
  if (!text || !String(text).trim()) return "";
  return `<details class="failCase">
      <summary>Show the case it failed</summary>
      <pre class="code">${esc(text)}</pre>
    </details>`;
}

async function finish(res){
  render();
  $("prompt").textContent = "";
  $("editorWrap").hidden = true;
  // With the editor gone the frozen listing IS the finished function, so let it
  // grow instead of scrolling inside a 230px window.
  $("ctx").style.maxHeight = "none";
  renderContext();
  $("submit").style.display = "none";
  show(res.solved_independently ? "ok" : "warn",
       res.solved_independently
         ? "Solved. Every step on your own, nice work."
         : "Problem complete. Some steps used the shown answer, so this is recorded as solved with help.");
  try {
    await fetch(`${API}/mark_solved`, {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({session_id: sessionId})
    });
    // Record it locally too, so going back one screen shows it as done
    // without a round trip to re-read /solved. Which set matters: the server
    // is about to report the same split back, and disagreeing with it for one
    // screen is how a problem reads "Solved" until the next reload.
    if (openProblem){
      (res.solved_independently ? SOLVED : ASSISTED).add(openProblem.slug);
    }
  } catch {}
  // Both graphs, now that there is a finished function to draw the second one
  // from. Awaited last so a slow render never delays the "solved" message.
  await showDualGraphs(sessionId);
}

/* Back to the problem list, with the list redrawn: a problem just solved has
   to show as solved, and the header summary has to move. */
function backToProblems(){
  if (openAssign) history.replaceState(null, "", `${location.pathname}?assignment=${encodeURIComponent(openAssign.id)}`);
  // Leaving is not reading it - the tip is anchored to a chat that is about to
  // go off screen, so it goes with it, but it has not been used up.
  dismissCoach(false);
  $("editorWrap").hidden = false;
  $("ctx").style.maxHeight = "";          // undo the finish() expansion
  if (openAssign){
    document.title = `${openAssign.name} · ACADIA`;
    setCrumbs([{label: openAssign.name}]);
  }
  renderProblems();
  view("cProblem");
}

$("backA").onclick = goAssignments;
$("backP").onclick = backToProblems;
$("backToNow").onclick = leaveReview;

/* ---------- the tutor ----------
   Explains the problem, then makes the student defend an approach. It is
   never sent the solution, the chunk references or the oracle, so it cannot
   leak an answer even if a student talks it into trying. */
const SUGGESTIONS = ["Explain this problem simply",
                     "Here is my approach…",
                     "I am stuck"];

function bubble(who, text, when){
  const b = document.createElement("div");
  b.className = "bub " + who;
  const body = document.createElement("div");
  body.className = "btext learning-text";
  renderLearningText(body, text);
  // Who and when. Faded until hover, but always in the accessibility tree, so
  // the log reads as a conversation rather than a wall of alternating text.
  const meta = document.createElement("span");
  meta.className = "bmeta";
  // `when` is the archived timestamp for a replayed turn. Without it a
  // conversation from last Tuesday would be stamped with the moment the page
  // happened to reload, which is worse than no time at all.
  const at = when ? new Date(when) : new Date();
  meta.textContent = `${who === "me" ? "You" : "Tutor"} · `
    + (isNaN(at) ? "earlier"
       : at.toLocaleTimeString("en-US", {hour: "numeric", minute: "2-digit"}));
  b.append(body, meta);
  $("clog").appendChild(b);
  $("clog").scrollTop = $("clog").scrollHeight;
  return body;                          // callers replace the TEXT, not the meta
}

/* The chips live under the tutor's opening message, where they read as
   answers to it, and go away as soon as there is a real conversation. */
function showChips(){
  if ($("chips")) return;
  const wrap = document.createElement("div");
  wrap.className = "chips";
  wrap.id = "chips";
  wrap.innerHTML = SUGGESTIONS.map(s =>
    `<button type="button" class="chip">${esc(s)}</button>`).join("");
  wrap.addEventListener("click", e => {
    const c = e.target.closest(".chip");
    if (c) sendToTutor(c.textContent);
  });
  $("clog").appendChild(wrap);
}
function hideChips(){
  const c = $("chips");
  if (c) c.remove();
}

/* ---------- the wrong-direction fork ----------

   The tutor is forbidden from announcing a hole in a student's reasoning - it
   asks the question whose honest answer makes them find it. That rule is
   right, and it has one gap: a student can walk a long way down an approach
   that cannot work, answering every question honestly, and only learn that at
   the design gate.

   So the tutor also returns a FLAG, and the flag becomes a choice rather than
   a warning. Carrying on is a genuine option and is offered as one: an
   approach you follow to its dead end and understand is worth more than one
   you abandoned because a machine frowned at it. What the student gets is the
   fact that there is a fork here, and the say over which way to go. */
function showFork(){
  hideFork();
  const wrap = document.createElement("div");
  wrap.className = "chips fork";
  wrap.id = "fork";
  wrap.innerHTML = `<p class="forkq">This approach may not get you to the right
    answer. It is your call.</p>
    <button type="button" class="chip" data-say="I have thought about it - I want to
      keep going with this approach.">Keep going this way</button>
    <button type="button" class="chip" data-say="I would like to try a different
      approach. What should I be asking myself?">Try something else</button>`;
  wrap.addEventListener("click", e => {
    const c = e.target.closest(".chip");
    if (!c) return;
    // sendToTutor() takes the fork down itself, and only once it has decided
    // it is actually sending. Clearing it here instead would lose the choice
    // whenever that call bails out - a turn already in flight, no problem open
    // - leaving the student with neither the buttons nor a reply.
    //
    // The student's CHOICE is what goes into the log, so the rest of the
    // conversation - and the transcript an instructor reads later - shows
    // which way they went and that it was theirs to pick.
    sendToTutor(c.dataset.say.replace(/\s+/g, " ").trim());
  });
  $("clog").appendChild(wrap);
  $("clog").scrollTop = $("clog").scrollHeight;
}
function hideFork(){
  const f = $("fork");
  if (f) f.remove();
}

/* ==========================================================
   The one-time pointer at the tutor.
   ========================================================== */
const COACH_SEEN = "mt.coach.tutor.v1";

function dismissCoach(remember){
  const el = $("coach");
  if (!el) return;
  el.remove();
  window.removeEventListener("resize", placeCoach);
  window.removeEventListener("scroll", placeCoach, true);
  // Only a real acknowledgement is remembered. A resize or a route change
  // takes the tip off screen without the student having read it, and burning
  // the one showing on that would mean they never see it at all.
  if (remember){ try { localStorage.setItem(COACH_SEEN, "1"); } catch (e) {} }
}

/* Anchored by measurement rather than by a hard-coded offset: the chat is a
   sticky right column on a wide screen and a bottom sheet under 1024px, and a
   fixed position that is right for one is off-screen for the other. */
function placeCoach(){
  const el = $("coach"), chat = $("chatcol");
  if (!el || !chat) return;
  const r = chat.getBoundingClientRect();
  if (!r.width || !r.height){ dismissCoach(false); return; }
  const gap = 12;
  // Point RIGHT at the column when there is room beside it, DOWN at the sheet
  // handle when the chat is docked to the bottom of the screen.
  const beside = r.left > el.offsetWidth + gap * 2;
  el.dataset.dir = beside ? "right" : "down";
  if (beside){
    el.style.left = `${Math.round(r.left - el.offsetWidth - gap)}px`;
    el.style.top = `${Math.round(
      Math.min(Math.max(r.top + 34, gap), window.innerHeight - el.offsetHeight - gap))}px`;
  } else {
    el.style.left = `${Math.round(
      Math.max(gap, Math.min(r.right - el.offsetWidth - gap,
                             window.innerWidth - el.offsetWidth - gap)))}px`;
    el.style.top = `${Math.round(r.top - el.offsetHeight - gap)}px`;
  }
}

function showChatCoach(){
  dismissCoach(false);                       // never two at once
  try { if (localStorage.getItem(COACH_SEEN)) return; } catch (e) {}
  const el = document.createElement("div");
  el.className = "coach";
  el.id = "coach";
  el.setAttribute("role", "note");
  el.innerHTML =
    `<button type="button" class="x" aria-label="Got it">&times;</button>`
    + `<b>Design it here first.</b><br>Talk your approach through with the `
    + `tutor before you draw anything or write any code - that is what the `
    + `plan below is built from.<span class="caret" aria-hidden="true"></span>`;
  el.querySelector(".x").addEventListener("click", () => dismissCoach(true));
  document.body.appendChild(el);
  placeCoach();
  // ...and again once layout has settled: applyTutorGate() runs straight after
  // resetChat() and can show or hide the whole editor, which moves the column
  // this is anchored to.
  requestAnimationFrame(placeCoach);
  window.addEventListener("resize", placeCoach);
  window.addEventListener("scroll", placeCoach, true);
}

/* What the panels show while /history is in flight.

   Reopening a problem you have worked on before fetches the chat and the plan
   from the archive, and that takes a moment. Until it lands the page was
   showing a confident FRESH state - the tutor's opening greeting and "describe
   your approach and your plan will appear here" - which is the one thing that
   is definitely wrong for a student who has been here before. Then it would
   flip. So the greeting waits until we know there is nothing to restore, and
   the space it will occupy says it is looking.

   Skeletons rather than a spinner, matching everything else that fetches on
   this page (ui.css: "Every fetch shows one of these, never a blank page"),
   and sized to what is coming so the layout does not jump when it arrives. */
/* Everything a student must not touch while /history is in flight.

   The chat box stayed live during the restore, so a message typed then was
   sent against an empty chatLog and, a moment later, wiped by the transcript
   landing on top of it. The design panel is locked for the same reason: a
   design submitted before we know whether this problem was already unlocked is
   a round of review nobody needed. */
function setRestoring(on){
  for (const id of ["cinput", "designBtn", "designFile", "planSubmitBtn",
                    "submit"]){
    const el = $(id);
    if (!el) continue;
    if (on) disable(el, "Loading your earlier work on this problem…");
    else if (id === "submit") applyTutorGate();      // owns its own state
    else enable(el);
  }
  // The drop zone is a <div role="button">, and `disabled` means nothing on a
  // div - it would have stayed clickable while looking locked. A class that
  // kills pointer events is the only thing that actually stops it.
  const drop = $("designDrop");
  if (drop){
    drop.classList.toggle("inert", !!on);
    drop.setAttribute("aria-disabled", on ? "true" : "false");
    drop.tabIndex = on ? -1 : 0;
  }
  const inp = $("cinput");
  if (inp) inp.placeholder = on
    ? "Loading your earlier work…" : "Explain your thinking, or ask a question";
}

function historyLoading(){
  setRestoring(true);
  $("clog").innerHTML =
    `<div class="restoring" role="status">
       <span class="spin" aria-hidden="true"></span>
       <span>Looking for your earlier work on this problem&hellip;</span>
     </div>` + skeletonRows(3);
  chatLog = [];
  hideChips();
  $("planLive").innerHTML = skeletonRows(2);
}

function resetChat(title){
  $("clog").innerHTML = "";
  chatLog = [];
  bubble("bot", `We are on ${title}. Ask me to explain it, or tell me how you `
               + `are thinking of approaching it - I will not give you the `
               + `answer. When your plan is ready, submit a design (PNG, JPEG, `
               + `or PDF) to unlock the editor.`);
  showChips();
  showChatCoach();
}

async function sendToTutor(text){
  // They found the chat on their own, so the tip has done its job.
  dismissCoach(true);
  if (chatBusy) return;
  if (!openProblem){
    bubble("bot", "Open a problem on the left first - I can only help with the one you are working on.");
    return;
  }
  chatBusy = true;
  hideChips();
  // Whatever they type next IS their answer to the fork, so a stale pair of
  // buttons must not sit under it offering the choice a second time.
  hideFork();
  bubble("me", text);
  chatLog.push({role: "user", content: text});
  const thinking = bubble("bot", "");
  thinking.innerHTML = '<span class="think"><i></i><i></i><i></i></span>';
  try {
    const r = await fetch(`${API}/tutor_chat`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        slug: openProblem.slug,
        messages: chatLog,
        chunk_prompt: chunks[idx] ? chunks[idx].prompt : null,
        design_ok: tutorReleased,
        // So the tutor can put a release past the REAL gate before promising
        // anything. Without it the tutor says "sounds workable" and the gate
        // rejects the same plan thirty seconds later.
        plan: planGraph
      })
    });
    const data = await r.json();
    if (!r.ok){
      thinking.textContent = (data.detail && data.detail.message)
        || "The tutor is unavailable right now. Try again shortly.";
      chatLog.pop();                       // do not poison the next turn
      return;
    }
    renderLearningText(thinking, data.reply);
    chatLog.push({role: "assistant", content: data.reply});
    // Planning only. Once the design is approved they are implementing a plan
    // that has already been walked and passed, and the server pins this false.
    if (data.offtrack && !tutorReleased) showFork();
    // Not awaited: the graph grows in the background while the student reads
    // the reply. It is a picture, so it must never make the chat feel slower.
    refreshPlanGraph();
    // The tutor's own `ready` no longer opens the gate - only an APPROVED
    // DESIGN does. A student who talks a good game in chat but cannot draw the
    // plan has not shown they have one, and that is exactly the case this gate
    // exists to catch. `ready` now only tells them what to do next.
    if (data.ready && !tutorReleased){
      // Deliberately NOT "your plan is workable". The tutor reads a plan; the
      // review WALKS it through an example, and the two do not always agree -
      // a plan that counts nodes until the next one is None reads perfectly and
      // is off by one. Promising the plan is correct here and then rejecting it
      // thirty seconds later is worse than not promising.
      bubble("bot", "Sounds like you have a plan. Put it forward and I will "
                  + "check it properly - either submit the plan from this chat, "
                  + "or upload your own drawing as a PNG, JPEG, or PDF.");
    }
  } catch (e) {
    thinking.textContent = "Could not reach the tutor. Check your connection.";
    chatLog.pop();
  } finally {
    chatBusy = false;
    $("cinput").focus();
  }
}

/* Grow from two rows to about six, then scroll. Fixed at one row, a
   three-sentence explanation was written through a letterbox. */
function autogrow(el){
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 132) + "px";
}
function submitChat(){
  const el = $("cinput");
  const t = el.value.trim();
  if (!t) return;
  el.value = "";
  autogrow(el);
  sendToTutor(t);
}

/* ---------- design gate wiring ---------- */
const dDrop = $("designDrop"), dFile = $("designFile");
dDrop.onclick = () => dFile.click();
dDrop.onkeydown = e => {
  if (e.key === "Enter" || e.key === " "){ e.preventDefault(); dFile.click(); }
};
["dragenter","dragover"].forEach(ev => dDrop.addEventListener(ev, e => {
  e.preventDefault(); dDrop.classList.add("over");
}));
["dragleave","drop"].forEach(ev => dDrop.addEventListener(ev, e => {
  e.preventDefault(); dDrop.classList.remove("over");
}));
dDrop.addEventListener("drop", e => {
  if (e.dataTransfer.files.length) pickDesign(e.dataTransfer.files[0]);
});
dFile.addEventListener("change", e => pickDesign(e.target.files && e.target.files[0]));
$("designClear").onclick = () => { clearDesign(); dDrop.focus(); };
$("designBtn").addEventListener("click", uploadDesign);
$("planSubmitBtn").addEventListener("click", submitPlanGraph);
$("handbackBtn").addEventListener("click", downloadHandback);
$("restartBtn").addEventListener("click", openRestart);
$("restartNo").addEventListener("click", closeRestart);
$("restartYes").addEventListener("click", doRestart);
// Backdrop click or Escape dismisses it: a destructive dialog must be easy to
// get out of and hard to confirm by accident.
$("restartModal").addEventListener("click", e => {
  if (e.target === $("restartModal")) closeRestart();
});
addEventListener("keydown", e => {
  if (e.key === "Escape" && !$("restartModal").hidden) closeRestart();
});

$("cform").addEventListener("submit", e => { e.preventDefault(); submitChat(); });
$("cinput").addEventListener("input", e => autogrow(e.target));
$("cinput").addEventListener("keydown", e => {
  // Shift+Enter is a newline; plain Enter sends. Ignore Enter mid-IME so
  // composing in another language does not fire off a half-typed message.
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing){
    e.preventDefault();
    submitChat();
  }
});

/* Tutor drawer on smaller screens. Hidden content must also leave the tab
   order; a translated-offscreen form is still focusable without inert. */
const tutorDrawer = matchMedia("(max-width: 1439px)");
function setTutorOpen(open, focus = true){
  $("chatcol").classList.toggle("open", open);
  $("sheetTog").setAttribute("aria-expanded", String(open));
  $("sheetTog").setAttribute("aria-label", open ? "Close the tutor" : "Open the tutor");
  $("sheetTog").querySelector(".sheet-label").textContent = open ? "Close tutor" : "Open tutor";
  const closed = tutorDrawer.matches && !open;
  $("clog").inert = closed;
  $("cform").inert = closed;
  if (focus && open) $("cinput").focus();
}
$("sheetTog").onclick = () => setTutorOpen(!$("chatcol").classList.contains("open"));
tutorDrawer.addEventListener("change", () => setTutorOpen(false, false));
setTutorOpen(false, false);

function setWorkspaceFocus(on){
  document.body.classList.toggle("focus-workspace", on);
  const button = $("focusWork");
  button.setAttribute("aria-pressed", String(on));
  button.setAttribute("aria-label", on ? "Exit focus view" : "Focus editor");
  button.querySelector("span").textContent = on ? "Exit focus" : "Focus editor";
  if (editor) requestAnimationFrame(() => { editor.refresh(); matchGutter(); });
}
$("focusWork").onclick = () => setWorkspaceFocus(!document.body.classList.contains("focus-workspace"));
addEventListener("keydown", event => {
  if (event.key !== "Escape") return;
  if (tutorDrawer.matches && $("chatcol").classList.contains("open")){
    setTutorOpen(false, false);
    $("sheetTog").focus();
  } else if (document.body.classList.contains("focus-workspace")){
    setWorkspaceFocus(false);
    $("focusWork").focus();
  }
});

// The frozen gutter is measured off the live editor, so anything that can
// change the editor's metrics has to trigger a re-measure: the mono webfont
// arriving after first paint (the common case - the first render uses the
// fallback's digit width), and a resize or zoom.
addEventListener("resize", matchGutter);
if (document.fonts && document.fonts.ready) document.fonts.ready.then(matchGutter);

clearDesign();
clearLegacyTouch();
loadAssignments();
