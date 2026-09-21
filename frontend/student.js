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
// The wrong-direction fork's own memory. `lastOfftrackReason` is the tutor's
// OWN prior diagnosis (main/tutor.py offtrack_reason), echoed back only when
// the student explicitly asks to try something else - never shown on screen,
// only ever sent back to the same server that wrote it. `offtrackForkCount`
// is how many times THIS problem has forked so far, which is what lets the
// server narrow the question further each time instead of repeating itself.
// Both reset at the top of historyLoading(), the one place every problem
// open passes through before anything else happens.
let lastOfftrackReason = "", offtrackForkCount = 0;
// Turns sent to /tutor_chat. Comfortably inside the server's own ceiling
// (main/tutor.MAX_TURNS * 2), so a long conversation is trimmed on the way out
// rather than refused on arrival.
const TUTOR_WINDOW = 60;
// What one message may carry. main/tutor.MAX_MESSAGE_CHARS truncates past this
// SILENTLY - a student who pasted a long trace had the end of it quietly
// removed and was answered on the half that survived. Enforced here where it
// can be seen instead.
const MAX_MESSAGE_CHARS = 2000;
// A student may not write code until they have submitted a DESIGN and the
// reviewer has approved it. Designing before coding IS the pedagogy, so this
// gates the whole coding column, not just the submit button - leaving the
// editor typable invites them to start coding and back-fill a design after.
let tutorReleased = false;
// Non-null while the student is looking back at a step they already finished.
let reviewIdx = null;
let reviewDraft = null;
let stepPromptsPending = false;

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
    else if (stepPromptsPending) disable(btn, "Loading the instructions for your steps…");
    else if (btn.dataset.busy === "1") disable(btn, "Grading your last answer…");
    else enable(btn);
  }
  // CodeMirror ignores the underlying <textarea>'s `disabled` attribute, so
  // once fromTextArea() has run the lock MUST go through setOption("readOnly").
  // "nocursor" also blocks focus. Before the editor exists, fall back to the
  // textarea.
  if (editor){
    editor.setOption("readOnly", tutorReleased && !stepPromptsPending ? false : "nocursor");
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
  workspaceSync();
}

/* Approval is explicit, but the student chooses when to continue to Code. */
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
  workspaceSync();
}

/* Fetch the step prompts, which the server withholds until the design is
   accepted. Until then `chunks` carries empty prompts - the right count, the
   right indents, and nothing that gives the answer away. */
async function loadSteps(){
  if (!sessionId) return;
  const sid = sessionId;
  stepPromptsPending = true;
  applyTutorGate();
  try {
    const r = await fetch(`${API}/session_steps/${encodeURIComponent(sessionId)}`);
    if (sessionId !== sid) return;
    if (!r.ok) throw new Error("Steps unavailable");
    const d = await r.json();
    if (sessionId !== sid) return;
    if (d.chunks && d.chunks.length && d.chunks.every(c => c.prompt)){
      chunks = d.chunks;
      stepPromptsPending = false;
      $("msg").replaceChildren();
      render();
      applyTutorGate();
    } else throw new Error("Steps unavailable");
  } catch (e) {
    if (sessionId !== sid) return;
    show("warn", "Your plan is approved, but the coding instructions could not be loaded.");
    const retry = document.createElement("button");
    retry.type = "button"; retry.className = "ghost"; retry.textContent = "Retry loading steps";
    retry.onclick = loadSteps;
    $("msg").append(retry);
  }
}

/* A quiet orientation line replaces the repeated full-width unlock banner. */
function markUnlocked(text){
  $("codeOrientation").textContent = text;
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

/* Keep the preview current. New nodes can highlight when the student opens
   it, but no background update scrolls the page or changes the active stage. */
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

/* Announce the first available preview in the same place as later updates. */
function nudgeToPlan(){
  showPlanPing(); // Announce the preview without moving the student's page.
}

/* A quiet live status accompanies the collapsed preview. */
function showPlanPing(){
  $("planPreviewStatus").textContent = "Updated from your thinking";
}

/* Queue a redraw and hand back a promise for when the queue is empty.

   Callers on the chat path ignore the promise - the drawing must never delay
   the tutor's reply. submitPlanGraph() awaits it, because there the graph stops
   being a drawing and becomes the thing that gets judged. */
function refreshPlanGraph(){
  const requested=workspaceEpoch;
  const draw=()=>requested===workspaceEpoch?drawPlanGraph():undefined;
  planQueue=planQueue.then(draw,draw);return planQueue;
}

async function drawPlanGraph(){
  // A failure must leave the last good picture on screen.
  //
  // Once the design is approved the plan is FIXED. Re-extracting it from a
  // conversation that is now about debugging would quietly rewrite the plan the
  // student was approved on - and the later the conversation goes, the less it
  // is about the plan at all, so the redraw gets worse as it gets less welcome.
  if (planFrozen || !openProblem) return;
  const drawing=workspaceEpoch;planUpdating=true;workspaceSync();
  if(!planGraph?.nodes?.length)renderGraphLoading($("planLive"),"Building your working plan…");
  try {
    const r = await fetch(`${API}/plan_graph`, {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({slug: openProblem.slug,
                            messages: chatLog.concat(designLog),
                            current: planGraph})
    });
    if (!r.ok) throw Error("Plan unavailable");
    const g = await r.json();
    if(drawing!==workspaceEpoch||planFrozen)return;
    if (g && g.nodes && g.nodes.length){
      planGraph = g;
      paintPlan(g);
    }
  } catch(e){
    if(drawing===workspaceEpoch)$("planPreviewStatus").textContent="Could not update the plan. Your previous version is saved.";
  } finally {
    if(drawing===workspaceEpoch){
      planUpdating=false;syncPlanSubmit();
      if($("planPreviewStatus").textContent==="Updating your plan…")$("planPreviewStatus").textContent=planGraph?.nodes?.length?"Up to date":"Ready to build your plan";
    }
  }
}

async function showDualGraphs(sid){
  try {
    const r = await fetch(`${API}/graphs`, {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({session_id: sid, plan: planGraph})
    });
    if (sid !== sessionId) return;
    if (!r.ok) throw new Error("Comparison unavailable");
    const payload = await r.json();
    $("dualCard").hidden = false;
    renderDual($("dualWrap"), payload);
    // The original plan remains available in Plan and the reference panel.
    // The comparison belongs to Reflect, which the student opens themselves.
    $("planCard").hidden = false;
    const ping = $("planPing"); if (ping) ping.remove();
    comparisonReady();
  } catch (e) {
    if (sid === sessionId) comparisonUnavailable();
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

  // A verdict on THIS problem's design. Approval unlocks the coding stage, so
  // a review that lands after the student opened something else would open the
  // gate on a problem whose plan nobody has read - see sendToTutor for the
  // same guard on the chat path.
  const reviewing = workspaceEpoch;
  setBusy(btn, true, "Reviewing…");
  designMsg("info", "Reading your design. This takes a few seconds.");
  try {
    const r = await fetch(`${API}/design_review`, {method: "POST", body: fd});
    const data = await r.json();
    if (reviewing !== workspaceEpoch) return;
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
    if (reviewing !== workspaceEpoch) return;
    designMsg("bad", "Could not reach the reviewer. Check your connection and "
      + "try again - nothing was submitted.");
  } finally {
    if (reviewing === workspaceEpoch){
      setBusy(btn, false);
      if (!designFileRef) disable(btn, "Choose a PNG, JPEG or PDF first.");
    }
  }
}

/* Submit the plan the page drew from this student's chat, instead of a file.

   Shares everything with uploadDesign() below the network call: the same
   reviewer, the same rubric, the same transcript bookkeeping, the same unlock.
   What it skips is the screenshot round trip. */
async function submitPlanGraph(){
  const btn = $("planSubmitBtn");
  // Same guard, same reason as uploadDesign: this one ends in openGate(), and
  // an approval applied to whatever happens to be open is an unlocked editor
  // on a problem that was never planned.
  const reviewing = workspaceEpoch;
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
        + "the chat and it will appear in Your working plan.");
    }
    const r = await fetch(`${API}/design_review/plan`, {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({slug: openProblem.slug, graph: planGraph,
                            history: designLog, messages: chatLog})
    });
    const data = await r.json();
    if (reviewing !== workspaceEpoch) return;
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
    // APPROVED, and the reviewer also spotted the collect-then-transform shape:
    // offer the choice instead of unlocking straight through. openOptimizePopup
    // was written, styled and given both its handlers, and then nothing ever
    // called it - the reviewer has been returning `redundant` to a page that
    // dropped it on the floor, so the popup could not fire for anyone. Every
    // way out of the dialog still reaches openGate() except "Try something
    // simpler", which deliberately keeps them in planning for one more round.
    if (data.approved) data.redundant ? openOptimizePopup(data.redundant)
                                      : openGate();   // freezes the plan as it stands
  } catch (e) {
    if (reviewing !== workspaceEpoch) return;
    designMsg("bad", "Could not reach the reviewer. Check your connection and "
      + "try again - nothing was submitted.");
  } finally {
    if (reviewing === workspaceEpoch) setBusy(btn, false);
  }
}

/* Keep the next action discoverable, with a reason until a plan exists. */
function syncPlanSubmit(){
  const row=$("planSubmitRow");if(!row)return;
  // The ROW stays - it carries "Full file" and "Start over" too. Only the two
  // controls that submit a plan go once there is nothing left to submit; a
  // disabled "Submit plan for review" sitting there afterwards was just taking
  // up the slot beside the buttons that still do something.
  $("planSubmitBtn").hidden=tutorReleased;
  const ready=!!planGraph?.nodes?.length,busy=planLoading||planUpdating;
  $("planEmpty").hidden=ready||busy||historyUnavailable;
  if(resourceKind!=="plan")$("planCard").hidden=!(ready||busy||historyUnavailable);
  $("planPreview").setAttribute("aria-busy",String(busy));
  $("planLive").setAttribute("aria-busy",String(busy));
  $("planSubmitHint").textContent=planLoading?"Restoring your earlier work…":ready?"Review your working plan, then submit it.":"Talk through your approach with the tutor, or upload a plan.";
  $("openPlanUpload").hidden=tutorReleased;
  if(workspaceReadyState&&!planLoading&&!historyUnavailable)enable($("openPlanUpload"));
  else disable($("openPlanUpload"),"Wait for your earlier work to load.");
  const button=$("planSubmitBtn");
  if(button.dataset.busy!=="1"){
    if(workspaceReadyState&&ready&&!tutorReleased&&!busy&&!historyUnavailable)enable(button);
    else disable(button,busy?"Wait for your plan to finish loading.":"Build a plan with the tutor first, or upload your own plan.");
  }
}

/* ---------- starting a problem over ---------- */

let restartPrev = null;      // focus to restore when the dialog closes

/* ---------- the optimize-or-continue popup ----------

   Fires only when a plan is APPROVED and also has the redundant-structure
   shape (main/graphs.redundant_structure -> main/design_review._necessity_note).
   Never a rejection, never a warning - the plan already passed. This is a
   genuine choice, same spirit as the wrong-direction fork above: the student
   is told a leaner path may exist and decides for themselves whether it is
   worth the detour, rather than having it decided for them either way. */
let optimizePrev = null, optimizeRedundant = null;

function openOptimizePopup(redundant){
  optimizeRedundant = redundant;
  optimizePrev = document.activeElement;
  $("optimizeNote").textContent = redundant.note || "";
  $("optimizeModal").hidden = false;
  $("optimizeContinue").focus();
}

function closeOptimizePopup(){
  $("optimizeModal").hidden = true;
  optimizeRedundant = null;
  if (optimizePrev && optimizePrev.focus) optimizePrev.focus();
  optimizePrev = null;
}

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
  // The old wording here - "the rest is left exactly as it was handed out" -
  // described the handout and the code was rebuilding the UPLOAD, which is the
  // solved version. The file now matches the sentence instead of the sentence
  // being quietly wrong about the file.
  $("handbackSub").textContent = all
    ? "Every problem, with your completed answers filled in."
    : `Your completed answers for ${done} of ${total}, filled in at the right `
      + `place. The rest is left blank for you to finish.`;
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
let pQuery = "", pFilter = "all", pSort = "file";
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
/* "Solved with help" meant a step had used a revealed reference. Nothing is
   revealed to a student any more, so the split had stopped describing anything
   they could see - it read as a permanent demerit for a distinction the page no
   longer makes. The two sets stay apart underneath (the instructor's record
   still separates them); what a student is shown is one word. */
const STAT_LABEL = {solved: "Solved", helped: "Solved",
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
  if(!ASSIGNMENTS.length){paintStudentStats(null);$("assignCard").innerHTML=skeletonRows(2);}
  let d, solved, progress;
  try {
    [d, solved, progress] = await Promise.all([
      fetch(`${API}/assignments`).then(r => {if(!r.ok)throw Error("Assignments unavailable");return r.json();}),
      // A student with no solves yet is a 200 with an empty list; a failure
      // here must not cost them the assignment list, so it degrades to "none".
      fetch(`${API}/solved`).then(r => r.ok ? r.json() : {slugs: []})
                            .catch(() => ({slugs: []})),
      fetch(AcadiaCache.progressURL()).then(r=>r.ok?r.json():null).catch(()=>null)
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

  // Reuse the dashboard's assignment membership; descriptions load on demand.
  ASSIGNMENTS=list.map(a=>{
    const membership=progress?.problems?.filter(p=>String(p.assignment_id)===String(a.id));
    const slugs=(membership||[]).map(p=>p.slug);
    const failed=!membership||(a.ready>0&&!slugs.length);
    const solvedN=slugs.filter(isDone).length;
    const last=LAST_SLUG&&slugs.includes(LAST_SLUG)?1:0;
    return {...a,slugs,solvedN,last,failed};
  });

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
    rows.some(a => a.failed) ? "-" : rows.reduce((n,a) => n + (a.solvedN || 0), 0)] : ["-", "-", "-"];
  $("studentStats").querySelectorAll("[data-metric]").forEach((el,i) => el.textContent = values[i]);
}

function view(which){
  if (which !== "cSolve") {closeResource(false); workspaceEpoch++;}
  document.body.classList.toggle("in-studio", which === "cSolve");
  setWorkspaceFocus(false);
  setTutorOpen(false, false);
  ["cAssign","cProblem","cSolve"].forEach(id => $(id).hidden = id !== which);
  scrollTo({top: 0, behavior: "instant"});
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
  pQuery = ""; pFilter = "all"; pSort = "file";
  $("pfilter").value = "";
  $("psort").value = "file";
  $("pchips").querySelectorAll(".chip").forEach(c =>
    c.setAttribute("aria-pressed", String(c.dataset.f === "all")));
  $("assignName").textContent = a.name;
  document.title = `${a.name} · ACADIA`;
  // A breadcrumb in the header, so getting back one level does not mean going
  // all the way Home.
  setCrumbs([{label:"Assignments",go:goAssignments},{label:a.name}]);
  view("cProblem");

  if (a._problems){ PROBLEMS = a._problems; return renderProblems(); }

  // The list screen's own fetch for this assignment failed (or was never made).
  // Try again here, where there is room to say so and offer a retry.
  PROBLEMS = [];
  $("probSummary").textContent = "";
  $("problems").innerHTML = `<li style="grid-column:1/-1; display:block">${
    skeletonRows(3)}</li>`;
  // WHOSE LIST IS THIS. The heading, the crumb and the title are set above, at
  // once; the rows arrive later. Open A, then B before A answers, and A's rows
  // landed under B's name - the page said "Assignment B" over Assignment A's
  // problems, and clicking one opened a problem the heading never mentioned.
  // Every path out of the await re-checks that this is still the open one.
  const wanted = a.id;
  try {
    const r = await fetch(`${API}/assignments/${a.id}/problems`);
    if (!openAssign || openAssign.id !== wanted) return;   // they moved on
    if (!r.ok) throw new Error();
    a._problems = (await r.json()).problems || [];
    if (!openAssign || openAssign.id !== wanted) return;
  } catch {
    if (!openAssign || openAssign.id !== wanted) return;
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
  // ONE comparator, used for the top-level list and again inside each class
  // group, so the control means the same thing everywhere it is applied.
  const name = r => String(r.p.title || r.p.slug);
  /* THE ORDER THE TEACHER WROTE THEM IN, which is the default: group_order is
     the block's place in the uploaded .py and member_order its place within a
     class. Both already travel to the browser with every problem, and the
     server hands the list back in exactly this order - the alphabet was a
     choice the page was making on top of it, and it put "Employee Update"
     first in a file that ends with it. */
  const byFile = (a, b) => ((a.p.group_order ?? 0) - (b.p.group_order ?? 0))
                        || ((a.p.member_order ?? 0) - (b.p.member_order ?? 0));
  const sortRows = (a, b) => pSort === "status"
    ? (rank[a.state] - rank[b.state]) || name(a).localeCompare(name(b))
    : pSort === "alpha" ? name(a).localeCompare(name(b)) : byFile(a, b);

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
    .sort(sortRows);

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
  // THE CHOSEN ORDER APPLIES INSIDE A CLASS TOO. This re-sorted every group by
  // the teacher's file order unconditionally, throwing away whatever the
  // control said: picking "A - Z" left Stack sitting in isEmpty, __len__, push,
  // pop, peek and nothing on screen explained why. A sort control that visibly
  // does nothing is worse than no control. File order stays as the tiebreak, so
  // members the comparator calls equal still read in the order the teacher
  // grouped them.
  for (const g of groups)
    if (g.members)
      g.members.sort((a, b) => pSort === "status"
        // Under "status", the teacher's file order is the secondary key rather
        // than the alphabet: inside a class it carries real meaning - push
        // before pop before peek - and alphabetising it would scramble the
        // order the methods are meant to be learned in for no one's benefit.
        ? (rank[a.state] - rank[b.state]) || byFile(a, b)
        : sortRows(a, b));

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
    {label:"Assignments",go:goAssignments},
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
  $("editorWrap").hidden = false;         // ...and finish() hid the editor
  $("submit").style.display = ""; $("backP").style.display = "";
  $("msg").innerHTML = `<div class="banner info">Getting this problem ready…</div>`;
  $("attempts").textContent = "";
  accepted = []; idx = 0;
  chunks = []; sessionId = null; stepPromptsPending = false;
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
  resetWorkspace();
  const opening = workspaceEpoch;
  // The plan preview stays beside the conversation as the student's ideas grow.
  $("planCard").hidden = true;
  $("dualCard").hidden = true;
  // NOT resetChat() and NOT paintPlan(null) yet: both assert this is a fresh
  // start, and we do not know that until /history answers. See historyLoading.
  historyLoading();
  applyTutorGate();
  setReviewMode(false);

  // No solution is sent. The server reads the reference from the
  // database, so the browser never holds it.
  // Restore reads and session creation are independent; overlap the requests.
  const savedWork=fetch(`${API}/history/${encodeURIComponent(p.slug)}`).catch(()=>null);
  // OPENING CAN NOW TAKE A WHILE, AND THE SPINNER'S SENTENCE STOPS BEING TRUE.
  // "Looking for your earlier work" is accurate for about a second; after that
  // the wait is the server building this problem's steps, and when the student
  // planned a different approach from the teacher's it is rebuilding the whole
  // roadmap around theirs - a solution proposal, a decomposition, and every
  // gate (main/reroute.py). That is tens of seconds, and a stale sentence in
  // front of it reads as a hang.
  //
  // The page cannot know which of those is happening - the server decides, and
  // only says so in the response - so the wording says what is true either way
  // and stops claiming to be doing the thing it has finished.
  const waitNote = setTimeout(() => {
    const line = $("clog") && $("clog").querySelector(".restoring span:last-child");
    if (line) line.textContent =
      "Setting up the steps for this problem… this takes longer the first "
      + "time, and longer again if your approach differs from ours.";
  }, 4000);
  let r;
  try {
    r = await fetch(`${API}/decompose_chunks`, {
      method: "POST", headers: {"Content-Type": "application/json"},
      // No student_id: the server binds the session to whoever the cookie
      // says we are. Sending a name here used to let anyone claim anyone.
      body: JSON.stringify({slug: p.slug, title: p.title,
                            description: p.description})
    });
  } catch { clearTimeout(waitNote); if (opening === workspaceEpoch) return failStart("Could not reach the server."); else return; }
  clearTimeout(waitNote);
  if (opening !== workspaceEpoch) return;
  if (!r.ok) return failStart("This problem could not be started. Try another one.");

  const d = await r.json();
  if (opening !== workspaceEpoch) return;
  sessionId = d.session_id; chunks = d.chunks || []; header = d.header || "";
  // A RESUMED session hands back the student's own accepted prefix and the step
  // they had reached, so closing the tab no longer costs them their steps. It is
  // safe to put back precisely because it is the SAME session that graded it -
  // see main/sessions.find_resumable. A fresh session sends neither field and
  // these stay at the empty values set above.
  accepted = (d.accepted || []).map(a => ({code: a.code || "",
                                           how: a.how === "revealed" ? "revealed" : "own"}));
  idx = Math.min(Math.max(Number(d.index) || 0, 0), chunks.length);
  $("msg").innerHTML = "";
  ensureEditor();
  render();
  if (idx > 0) show("ok", idx === 1
    ? "Picking up where you left off - your first step is already in."
    : `Picking up where you left off - your first ${idx} steps are already in.`);
  applyTutorGate();
  // Nothing to put back, so this IS a fresh start - say hello and draw the
  // empty plan. Skipped when history was restored (it painted both already) and
  // when the student has already moved to another problem.
  if (!await restoreHistory(p,savedWork)){
    if (opening !== workspaceEpoch) return;
    resetChat(p.title || p.slug);
    paintPlan(null);
  }
  if (opening === workspaceEpoch) readyWorkspace();
}

/* Put back what this student already did on this problem.

   Everything below has been archived server-side all along (main/archive.py);
   nothing read it, so reopening a problem showed an empty chat, an empty plan
   graph and a locked editor - and looked exactly like the work had been
   deleted. It never was.

   The accepted code and the step position are NOT restored here, and that is a
   division of labour rather than a policy: they belong to a GRADING SESSION, so
   they come back with the session itself from /decompose_chunks, which now
   resumes the live one instead of issuing a fresh one every time. Replaying old
   answers into a NEW session would claim steps that session never graded, which
   is why this function still must not do it. What comes back here is the
   thinking around the code - the chat, the plan, the design verdict.

   Returns TRUE when it has painted the chat and the plan itself, so the caller
   knows not to overwrite them with a fresh greeting - and true as well when the
   student has moved on mid-fetch, because whatever they moved to owns those
   panels now. False means "nothing here, start clean". */
async function restoreHistory(p,request){
  const opening = workspaceEpoch;
  let h;
  try {
    const r=await (request||fetch(`${API}/history/${encodeURIComponent(p.slug)}`));
    if(opening!==workspaceEpoch)return true;
    if(!r||!r.ok)throw Error("History unavailable");
    h=await r.json();
  }catch{
    if(opening!==workspaceEpoch)return true;
    historyUnavailable=true;setRestoring(false);
    for(const id of ["cinput","designBtn","designFile","planSubmitBtn","submit","openPlanUpload"])disable($(id),"Retry loading your earlier work first.");
    $("cform").querySelector('button[type="submit"]').disabled=true;
    $("designDrop").classList.add("inert");$("designDrop").setAttribute("aria-disabled","true");$("designDrop").tabIndex=-1;
    $("clog").innerHTML='<div class="banner warn">Your earlier conversation could not be loaded. Retry to pick up your work safely.</div>';
    $("planLive").innerHTML='<div class="graph-load-error" role="alert"><strong>Your saved plan could not be loaded.</strong><p>Your work has not been reset.</p><button id="retryHistory" type="button">Retry loading</button></div>';
    $("planPreviewStatus").textContent="Could not load your saved plan";
    $("workspaceStatus").textContent="Retry loading your earlier work to continue.";syncPlanSubmit();
    $("retryHistory").onclick=async()=>{
      if(opening!==workspaceEpoch)return;
      historyUnavailable=false;historyLoading();
      if(!await restoreHistory(p)){
        if(opening!==workspaceEpoch)return;
        resetChat(p.title||p.slug);paintPlan(null);
      }
      if(opening===workspaceEpoch)readyWorkspace();
    };
    return true;
  }
  if (opening !== workspaceEpoch) return true;
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
    // Reopening an unlocked problem lands on the coding stage rather than on
    // the plan they already had accepted. Consumed once, when the stage
    // actually unlocks - see workspace.js.
    // ALREADY FINISHED lands on Reflect, not back in the working screen: see
    // solvedEarlier() in workspace.js. It also has to be marked complete
    // BEFORE the comparison block below, or comparisonReady() paints "a look
    // at your earlier work" over the congratulations.
    if (h.solved) solvedEarlier();
    resumeStage = h.solved ? "reflect" : "code";
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
    $("planCard").hidden = false;
    comparisonReady(true);
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
  failedWorkspace(msg);
}

/* UNSUBMITTED CODE, KEPT. render() reseeds the editor with a pad of spaces
   every time it runs, and start() calls render() - so opening another problem
   and coming back silently threw away whatever had been typed. Nothing warned,
   and there was nothing to warn about: it was gone before the page changed.

   Held per problem AND per step, because they are different drafts, and in
   sessionStorage so a reload keeps them too. Student code in the student's own
   tab: nothing here crosses to another person, and it is cleared the moment a
   step is accepted, so a finished answer never lingers as a draft.

   Every access is wrapped: a private window or blocked site data makes these
   throw, and losing a draft is a nuisance while losing the editor is an
   outage. */
const DRAFT_PREFIX = "acadia.draft.";

function draftKey(){
  return openProblem ? `${DRAFT_PREFIX}${openProblem.slug}:${idx}` : null;
}

function saveDraft(){
  const key = draftKey();
  if (!key || !editor) return;
  const value = editor.getValue();
  try {
    if (value.trim()) sessionStorage.setItem(key, value);
    else sessionStorage.removeItem(key);
  } catch (e) { /* no storage: the draft simply does not survive */ }
}

function takeDraft(){
  const key = draftKey();
  if (!key) return null;
  try { return sessionStorage.getItem(key); } catch (e) { return null; }
}

function clearDraft(){
  const key = draftKey();
  if (!key) return;
  try { sessionStorage.removeItem(key); } catch (e) { /* nothing to clear */ }
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
  // Saved as they type, so navigating away at any moment keeps the draft. The
  // write is one sessionStorage.setItem on a string a student typed by hand -
  // cheap enough not to need debouncing, and debouncing would reintroduce the
  // exact window this closes.
  editor.on("change", saveDraft);
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
  const how = accepted[i].how;
  const label = how === "revealed" ? "shown to you"
              : how === "covered" ? "covered by an earlier step" : "your answer";
  $("stepCount").innerHTML = `Step ${i + 1} of ${chunks.length}`
    + ` <span class="pill ${how === "revealed" ? "warn" : "ok"}">${label}</span>`;
  $("prompt").textContent = c.prompt || "";
  // A covered step has no code of its own - the student wrote it as part of an
  // earlier answer - so say that rather than showing them an empty box.
  $("reviewCode").textContent = how === "covered"
    ? "You already wrote this as part of an earlier step."
    : accepted[i].code;
  // REWORKING IS OFFERED FOR THEIR OWN ANSWERS ONLY. A revealed step is not
  // theirs to edit, and a covered one has no code of its own - reopening the
  // step that covered it is the same action, one pill to the left.
  const rework = $("reworkStep");
  if (rework) rework.hidden = how !== "own" || i >= idx;
  setReviewMode(true);
  renderStepper();
  $("msg").innerHTML = "";
  $("attempts").textContent = "";
  $("stepHistory").open = false;
  $("backToNow").textContent = idx < chunks.length ? "Back to the current step" : "Back to your completed code";
  $("backToNow").focus();
}

/* BACK TO A STEP THAT WAS ALREADY ACCEPTED.

   A student passed step 1, found a bug in it, and had nowhere to go: accepted
   code is frozen, and the only way to change it was Start over - the whole
   problem, plan and chat included. The freeze is right (the accepted prefix is
   what every later step was graded against), so this does not unfreeze
   anything; it asks the SERVER to put them back on that step, which drops the
   steps after it for the same reason they were frozen in the first place.

   The dropped answers come back with the response and are kept as drafts, so
   reworking step 1 never costs them the step 2 they had already written.

   REACHED FROM TWO PLACES, and the second one is the point. Review mode has the
   button, but getting there means knowing that a finished step pill is a
   button - so the row under the frozen listing offers the same thing where the
   student is already looking: at step 1's code, while writing step 2. */
async function reworkStep(i, btn){
  if (i === null || i === undefined || !sessionId) return;
  if (btn) setBusy(btn, true, "Opening that step\u2026");
  try {
    const r = await fetch(`${API}/reopen_step`, {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({session_id: sessionId, index: i})
    });
    if (!r.ok){
      let m = "That step could not be reopened. Reload and try again.";
      try { m = (await r.json()).detail.message || m; } catch (e) {}
      return toast(m, "bad");
    }
    const state = await r.json();
    // Their later answers, held where the editor will offer them back.
    for (const d of (state.dropped || [])){
      if (d.index === i || !d.code) continue;
      try { sessionStorage.setItem(`${DRAFT_PREFIX}${openProblem.slug}:${d.index}`,
                                   d.code); } catch (e) {}
    }
    const own = (accepted[i] || {}).code || "";
    accepted = accepted.slice(0, i);
    idx = Number(state.index);
    reviewIdx = null;
    reviewDraft = null;
    setReviewMode(false);
    render();
    // render() reseeds from the draft for this step; the answer they came back
    // to edit is what they expect to see, so it wins.
    if (editor && own){ editor.setValue(own); editor.focus(); }
    applyTutorGate();
    $("attempts").textContent = "";
    show("info", state.total_chunks - idx > 1
      ? `You are back on step ${idx + 1}. The steps after it are open again - `
        + `what you had written for them is kept as a draft.`
      : `You are back on step ${idx + 1}.`);
  } catch (e) {
    toast("Could not reach the server. Nothing was changed.", "bad");
  } finally {
    if (btn) setBusy(btn, false);
  }
}

/* The row under the frozen code: one button per step the student answered
   themselves, so changing an earlier answer is one click from the code it
   changes. Lives inside #codestack, which review mode hides wholesale - so
   there is never a second way in on a screen that has its own. */
function renderEditEarlier(){
  const row = $("editEarlier");
  if (!row) return;
  const mine = accepted
    .map((a, i) => ({a, i}))
    .filter(x => x.a && x.a.how === "own" && (x.a.code || "").trim());
  row.hidden = !mine.length;
  if (row.hidden){ row.innerHTML = ""; return; }
  row.innerHTML = '<span class="lab">Change an earlier step:</span>'
    + mine.map(x => `<button type="button" class="chip" data-edit="${x.i}"`
        + ` title="Reopen step ${x.i + 1}. Steps after it open again too - what`
        + ` you wrote for them is kept as a draft.">Step ${x.i + 1}</button>`).join("");
  row.querySelectorAll("[data-edit]").forEach(b =>
    b.onclick = () => reworkStep(+b.dataset.edit, b));
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

/* Keep a successful answer visible until the student elects to move on.
   The server has advanced; this is the existing read-only review view. */
function pauseAfterStep(completedIndex){
  render(); // Prepare an empty draft for the next step, then preserve it.
  openReview(completedIndex);
  $("backToNow").textContent = `Continue to step ${idx + 1} →`;
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
  renderEditEarlier();
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
    // A draft for THIS step wins over a fresh pad - see saveDraft above.
    const draft = takeDraft();
    editor.setValue(draft && draft.trim() ? draft : pad);
    editor.setCursor(draft && draft.trim()
      ? {line: editor.lineCount() - 1, ch: editor.getLine(editor.lineCount() - 1).length}
      : {line: 0, ch: pad.length});
    editor.refresh();
    // CodeMirror re-lays out its gutter after the option change, so the width
    // read during renderContext() above was the PREVIOUS one. Re-measure on the
    // next frame, when the new gutter is on screen.
    requestAnimationFrame(matchGutter);
    if (tutorReleased && workspaceStage === "code" && !resourceKind) editor.focus();
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

/* A short stable digest of a submission, so retrying the same answer reuses its
   submission id. Not a security boundary - the server owns the session and the
   verdict; this only has to be stable and collision-free enough that two
   DIFFERENT answers at one step do not share an id. */
function hash32(text){
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++){
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h.toString(36);
}

$("submit").onclick = async () => {
  const code = editor ? editor.getValue() : "";
  if (!code.trim()) return show("warn", "Write something first.");
  const btn = $("submit");
  // THE GRADE BELONGS TO THE PROBLEM THAT WAS SUBMITTED. Everything below
  // writes the answer into accepted[idx] and moves idx on, and both of those
  // are the OPEN problem's - so a verdict arriving after the student switched
  // filed one problem's code as another's accepted work and advanced its step.
  // sessionId travels in the request, so the server was never confused; only
  // the page was, which is exactly the part the student is reading.
  const grading = workspaceEpoch;
  setBusy(btn, true, "Grading…");
  // STABLE ACROSS RETRIES, which is the whole point of the id. It carried
  // Date.now(), so every press minted a fresh one: a submission that reached
  // the server and then lost the connection was told "your attempt was not
  // used", and pressing again asked the server to grade it a SECOND time under
  // an id it had never seen. The idempotency table could not match them.
  //
  // Keyed on the answer instead. The same code at the same step replays the
  // stored result; genuinely different code is a genuinely different attempt.
  const submissionId = `${sessionId}:${idx}:${hash32(code)}`;
  let r;
  try {
    r = await fetch(`${API}/grade_chunk`, {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({session_id: sessionId, submission_id: submissionId,
                            student_code: code, expected_index: idx})
    });
  } catch {
    if (grading !== workspaceEpoch) return;
    setBusy(btn, false);
    // NOT "your attempt was not used" - we do not know that. The request may
    // have arrived and been graded with the answer lost on the way back. What
    // IS true is that submitting the same code again cannot double-count it,
    // because the id above is derived from the code and the server replays a
    // stored result for an id it has already seen.
    return show("bad", "Could not reach the server. Press Submit again - the "
                     + "same answer will not be counted twice.");
  }
  if (grading !== workspaceEpoch) return;   // a different problem is open now
  setBusy(btn, false);

  if (!r.ok) {
    let m = "Something went wrong. Your attempt was not used.";
    try { const j = await r.json(); if (j.detail && j.detail.message) m = j.detail.message; } catch {}
    return show("warn", m);
  }

  const res = await r.json();
  if (grading !== workspaceEpoch) return;

  // Covers the OpenAI outage case. The server guarantees no attempt was spent.
  // An indeterminate verdict that came with a QUESTION is handled further down,
  // where the two ways forward are offered. Without this the bare reason - "we
  // could not confirm this step" - would win, which is the shrug the diagnosis
  // exists to replace.
  // ...and the cases ride along when the server sent any. The adapted tier
  // returns `indeterminate` WITH failing cases (grading.py's
  // adapted_evidence_only): showing the evidence without convicting on it is
  // the whole point of that tier, and this branch dropped it on the floor
  // while the reason sentence said "the case below is worth tracing by hand".
  // The promise and the thing promised are now made in the same place.
  if (res.verdict === "indeterminate" && !(res.needs_diagnosis && res.diagnosis))
    return show("warn", res.reason, failingCasesHTML(res));

  if (res.verdict === "correct") {
    // Store it the way the server did, not the way it was typed.
    accepted[idx] = {code: alignToStep(code, (chunks[idx] || {}).indent), how: "own"};
    clearDraft();                    // it is an answer now, not a draft
    const completedIndex = idx;
    // ONE ANSWER CAN SETTLE MORE THAN ONE STEP. A student who wrote the next
    // step's work inside this one has already done it, so the server advances
    // past both (main/bridge.find reports how far their code reached). Filling
    // the steps it covered keeps this array in step with the server's index -
    // left as holes, the progress bar read "1 / 3" while the server was on
    // step 3, and the stepper called a finished step "not started yet".
    // Their code is stored ONCE, against the step they typed it into: repeating
    // it here would draw their loop twice in the context pane.
    for (let i = completedIndex + 1; i < res.index; i++){
      accepted[i] = {code: "", how: "covered"};
    }
    idx = res.index;
    if (res.completed) return finish(res);
    pauseAfterStep(completedIndex);
    show("ok", res.reason);
    return;
  }

  if (typeof res.revealed_reference === "string") {
    accepted[idx] = {code: res.revealed_reference, how: "revealed"};
    const completedIndex = idx;
    idx = res.index;
    if (res.completed) return finish(res);
    pauseAfterStep(completedIndex);
    show("warn", res.reason + "\n\nReview the shown answer below, then continue when you’re ready.");
    return;
  }
  // COULD NOT CONFIRM -> PAUSE AND ASK, rather than let them carry on.
  //
  // This verdict costs no attempt and does not advance the step, so without
  // stopping here a student whose code IS wrong would resubmit into the same
  // wall indefinitely - "we could not confirm this" over and over with nothing
  // to act on. The question is built from a real input their own code was run
  // on (main/diagnose.py), so there is always something concrete to do next.
  //
  // The editor stays usable: this is a prompt to look again, not a lock-out,
  // and the two buttons are the two honest ways forward.
  if (res.needs_diagnosis && res.diagnosis){
    // The cases come FIRST, before the two buttons: a diagnosis that arrived
    // with evidence attached (the adapted tier can produce both) is asking
    // them to look at something, and the something has to be on screen.
    show("warn", res.diagnosis,
         failingCasesHTML(res)
         + '<div class="diagnosis-actions">'
         + '<button type="button" id="diagFix">'
         + 'I see it — let me fix my code</button>'
         + '<button type="button" id="diagExplain" class="ghost">'
         + 'My approach is different — let me explain</button></div>');
    const fix = $("diagFix"), explain = $("diagExplain");
    if (fix) fix.onclick = () => {
      $("msg").innerHTML = "";
      if (editor) editor.focus();
    };
    // Hands them to the tutor with the opening already typed, so they are not
    // made to restate the situation. The tutor is where a different approach
    // gets discussed; the grader has no opinion about approaches.
    if (explain) explain.onclick = () => {
      $("msg").innerHTML = "";
      setTutorOpen(true);
      const box = $("cinput");
      if (box){
        box.value = "My approach is different from the one you expected. "
                  + "Here is what I am doing and why: ";
        box.focus();
        box.setSelectionRange(box.value.length, box.value.length);
        box.dispatchEvent(new Event("input"));   // keep the counter honest
      }
    };
    return;
  }
  // A wrong answer now comes with the case that caught it, folded away. Closed
  // by default because the point of the step is for them to find it themselves;
  // one click away because "wrong on at least one case" and no case is a shrug,
  // and a student with no way forward stops rather than thinks.
  show("bad", res.reason, failingCasesHTML(res));
  // "Attempt 3 of 2" would be a lie now that there is no limit and the answer
  // is never shown. A plain count still tells them where they are without
  // implying a countdown to being given it.
  $("attempts").textContent = res.attempts
    ? `Attempt ${res.attempts}. Keep going - take as many as you need.` : "";
};

/* The disclosure holding the failing cases. Empty string when the server sent
   none - a crash or a timeout has no case to show.

   THE COUNT IS STATED, NOT COUNTED OFF THE LIST. The server shows at most
   grading.MAX_SHOWN_CASES of them, so three listings under a summary reading
   "3 cases" would quietly report a submission that failed seven as failing
   three - and a student who fixes those three and resubmits has been set up to
   be surprised. Where the two differ the summary says both numbers. */
function failingCasesHTML(res){
  const cases = (res.failing_cases || []).filter(c => String(c || "").trim());
  if (!cases.length) return "";
  const total = Number(res.failed_total) || cases.length;
  const hidden = total - cases.length;
  const noun = n => n === 1 ? "case" : "cases";
  const head = hidden > 0
    ? `Show ${cases.length} of the ${total} ${noun(total)} it failed`
    : `Show the ${total > 1 ? total + " " : ""}${noun(total)} it failed`;
  return `<details class="failCase">
      <summary>${esc(head)}</summary>
      ${cases.map(c => `<pre class="code">${esc(c)}</pre>`).join("")}
      ${hidden > 0 ? `<p class="failMore">${esc(
        `${hidden} more ${noun(hidden)} also failed. Fix these first - the same
         mistake is usually behind all of them.`.replace(/\s+/g, " "))}</p>` : ""}
    </details>`;
}

async function finish(res){
  const sid = sessionId, problem = openProblem;
  render();
  $("prompt").textContent = "";
  $("editorWrap").hidden = true;
  // With the editor gone the frozen listing IS the finished function, so let it
  // grow instead of scrolling inside a 230px window.
  $("ctx").style.maxHeight = "none";
  renderContext();
  $("submit").style.display = "none";
  show("ok", "Solved. Nice work.");
  completeWorkspace();
  try {
    await fetch(`${API}/mark_solved`, {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({session_id: sid})
    });
    // Record it locally too, so going back one screen shows it as done
    // without a round trip to re-read /solved. Which set matters: the server
    // is about to report the same split back, and disagreeing with it for one
    // screen is how a problem reads "Solved" until the next reload.
    if (problem){
      (res.solved_independently ? SOLVED : ASSISTED).add(problem.slug);
    }
  } catch {}
  // Both graphs, now that there is a finished function to draw the second one
  // from. Awaited last so a slow render never delays the "solved" message.
  if (sid === sessionId) await showDualGraphs(sid);
}

/* The next problem in this assignment, in the order the list is showing, and
   preferring one they have not finished - moving straight onto a problem they
   already solved is not "next". Nothing left after this one means the list
   itself is the next thing. */
function goNextProblem(){
  const i = openProblem
    ? PROBLEMS.findIndex(p => p.slug === openProblem.slug) : -1;
  const rest = i < 0 ? [] : PROBLEMS.slice(i + 1);
  const next = rest.find(p => !isDone(p.slug)) || rest[0];
  return next ? start(next) : backToProblems();
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
    <button type="button" class="chip" data-redirect="true" data-say="I would like to try a different
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
    //
    // Only "Try something else" carries the diagnosis back. "Keep going this
    // way" is a decision to NOT rethink - sending the hint along with it would
    // push the tutor toward the redirect they just declined.
    const extra = c.dataset.redirect
      ? {offtrack_hint: lastOfftrackReason, offtrack_count: offtrackForkCount}
      : {};
    sendToTutor(c.dataset.say.replace(/\s+/g, " ").trim(), extra);
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
  planLoading=!!on;
  $("planPreviewStatus").textContent=on?"Loading your saved plan…":planGraph?.nodes?.length?"Saved plan restored":"Ready to build your plan";
  $("cform").querySelector('button[type="submit"]').disabled=!!on;
  for (const id of ["cinput", "designBtn", "designFile", "planSubmitBtn",
                    "submit"]){
    const el = $(id);
    if (!el) continue;
    if (on) disable(el, "Loading your earlier work on this problem…");
    else if (id === "submit") applyTutorGate();      // owns its own state
    else if (id === "designBtn" && !designFileRef) disable(el, "Choose a PNG, JPEG or PDF first.");
    else if (id === "planSubmitBtn") syncPlanSubmit();
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
  lastOfftrackReason = ""; offtrackForkCount = 0;
  setRestoring(true);
  $("clog").innerHTML =
    `<div class="restoring" role="status">
       <span class="spin" aria-hidden="true"></span>
       <span>Looking for your earlier work on this problem&hellip;</span>
     </div>` + skeletonRows(3);
  chatLog = [];
  hideChips();
  $("planCard").hidden=false;$("planEmpty").hidden=true;renderGraphLoading($("planLive"));syncPlanSubmit();
}

function resetChat(title){
  $("clog").innerHTML = "";
  chatLog = [];
  bubble("bot", `Let’s think through ${title}. What should the function receive, `
               + `and what should it return? Tell me your approach in your own words. `
               + `When you’re ready, use Submit plan for review beside your working plan.`);
  showChips();
  // The stage introduction now provides the guidance once given by a popup.
}

async function sendToTutor(text, extra = {}){
  // They found the chat on their own, so the tip has done its job.
  dismissCoach(true);
  if (chatBusy || planLoading || historyUnavailable) return;
  if (!openProblem){
    bubble("bot", "Open a problem on the left first - I can only help with the one you are working on.");
    return;
  }
  // WHICH WORKSPACE ASKED. A tutor reply can arrive after the student has
  // moved to another problem, and everything below writes into whatever is
  // open NOW: the reply was appended to the new problem's chatLog, the
  // wrong-direction fork opened over it, and the busy flag it cleared was the
  // new request's. The answer to a question about problem A is not an answer
  // about problem B, so once the epoch moves this reply has nowhere to go.
  const asked = workspaceEpoch;
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
        // A WINDOW, not the whole log. The server rejects a conversation past
        // MAX_TURNS*2 with "start a fresh one", and the only recovery the page
        // offered was Start over - which wipes the plan, the design and every
        // accepted step to get past a message limit. The tutor already reads
        // only the last MAX_TURNS turns (main/tutor.reply), so sending more was
        // never doing anything except bringing that wall closer. The full log
        // stays on screen and in the archive; only what travels is trimmed.
        messages: chatLog.slice(-TUTOR_WINDOW),
        chunk_prompt: chunks[idx] ? chunks[idx].prompt : null,
        design_ok: tutorReleased,
        // So the tutor can put a release past the REAL gate before promising
        // anything. Without it the tutor says "sounds workable" and the gate
        // rejects the same plan thirty seconds later.
        plan: planGraph,
        ...extra
      })
    });
    const data = await r.json();
    if (asked !== workspaceEpoch) return;   // another problem is open now
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
    if (data.offtrack && !tutorReleased){
      lastOfftrackReason = data.offtrack_reason || "";
      offtrackForkCount++;
      showFork();
    }
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
    if (asked !== workspaceEpoch) return;
    thinking.textContent = "Could not reach the tutor. Check your connection.";
    chatLog.pop();
  } finally {
    // Cleanup is state too: clearing chatBusy here after the student moved on
    // would free a request belonging to the workspace now open, and let a
    // second message go out under it. resetWorkspace() clears the flag for the
    // new workspace, so nothing is left stuck either way.
    if (asked === workspaceEpoch){
      chatBusy = false;
      // Do not steal focus if the student started coding while the tutor replied.
      if (workspaceTutorVisible && $("cform").contains(document.activeElement)) $("cinput").focus({preventScroll:true});
    }
  }
}

/* Grow from two rows to about six, then scroll. Fixed at one row, a
   three-sentence explanation was written through a letterbox. */
function autogrow(el){
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 132) + "px";
}
function submitChat(){
  if(chatBusy||planLoading||historyUnavailable)return;
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
// ONE PER STAGE, not one on the page. Restart is a problem-level action, so it
// has to be reachable from wherever the student gives up - which is usually the
// planning stage, before any code exists. Two triggers rather than one shared
// header button because they are never on screen together (stage 1 owns the
// plan row, stage 2 the coding bar), and a control that follows you across
// stages reads as page chrome rather than as something that acts on THIS
// problem. Both open the same confirm dialog; nothing restarts on a click.
$("optimizeContinue").addEventListener("click", () => {
  closeOptimizePopup();
  openGate();
});
$("optimizeTry").addEventListener("click", () => {
  const r = optimizeRedundant;
  closeOptimizePopup();
  // Sent as the STUDENT'S OWN message, same convention the wrong-direction
  // fork above uses (sendToTutor) - it is a first-person request to reconsider,
  // never an instruction telling them what to remove. The tutor still owes
  // them a QUESTION back, not the fix - nothing about this route bypasses
  // that; it only decides to keep them in the planning stage for one more
  // round instead of unlocking code immediately.
  //
  // Never destructive either way - the plan already passed, so there is no
  // "undo" needed the way restart's confirm exists to prevent one.
  if (r) sendToTutor(
    // r.source / r.target are bare family names ("list", "dictionary"), so the
    // articles belong here - without them the student's own message went out
    // reading "collect everything into list first".
    `I'd like to see if I can simplify my plan - do I really need to collect `
    + `everything into a ${r.source} first, or can I build the ${r.target} `
    + `directly instead?`);
});
document.querySelectorAll("[data-restart]").forEach(b => b.addEventListener("click", openRestart));
$("reworkStep").addEventListener("click",
  () => reworkStep(reviewIdx, $("reworkStep")));
$("restartNo").addEventListener("click", closeRestart);
$("restartYes").addEventListener("click", doRestart);
// Backdrop click or Escape dismisses it: a destructive dialog must be easy to
// get out of and hard to confirm by accident.
$("restartModal").addEventListener("click", e => {
  if (e.target === $("restartModal")) closeRestart();
});
/* KEEP TAB INSIDE AN OPEN DIALOG.

   Both of these say aria-modal="true", and neither behaved like it: they are
   plain divs rather than <dialog>.showModal(), so the browser traps nothing and
   two presses of Tab walked a keyboard user out of the dialog and into the page
   behind it - which is still sitting there, still clickable, with a modal open
   over it and the buttons that dismiss it now unreachable in the tab order.
   Escape already worked; this is the other half. */
function openModal(){
  return [$("restartModal"), $("optimizeModal")].find(m => m && !m.hidden) || null;
}
addEventListener("keydown", e => {
  if (e.key !== "Tab") return;
  const modal = openModal();
  if (!modal) return;
  const stops = [...modal.querySelectorAll(
    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])')]
    .filter(el => !el.disabled);
  if (!stops.length) return;
  const first = stops[0], last = stops[stops.length - 1];
  const inside = modal.contains(document.activeElement);
  // Wrapping at the ends is the trap; pulling focus back in when it is already
  // outside covers the case where something else moved it while this was open.
  if (e.shiftKey ? (!inside || document.activeElement === first)
                 : (!inside || document.activeElement === last)){
    e.preventDefault();
    (e.shiftKey ? last : first).focus();
  }
});
addEventListener("keydown", e => {
  if (e.key === "Escape" && !$("restartModal").hidden) closeRestart();
  // Dismissing this one defaults to "continue" - the safe, non-destructive
  // side. Nothing is lost either way; an ambiguous dismiss should not be the
  // thing that decides to fork them into another chat message instead.
  if (e.key === "Escape" && !$("optimizeModal").hidden){
    closeOptimizePopup(); openGate();
  }
});
$("optimizeModal").addEventListener("click", e => {
  if (e.target === $("optimizeModal")){ closeOptimizePopup(); openGate(); }
});

$("cform").addEventListener("submit", e => { e.preventDefault(); submitChat(); });

/* THE LIMIT, SHOWN. main/tutor.MAX_MESSAGE_CHARS cuts a message at 2,000
   characters server-side and says nothing, so a student who pasted a long trace
   was answered on the half that survived and had no way to know the rest never
   arrived. `maxlength` on the field makes the cut impossible; this says how much
   room is left once it starts to matter, so hitting the wall is never a
   surprise either. */
$("cinput").addEventListener("input", () => {
  const left = MAX_MESSAGE_CHARS - $("cinput").value.length;
  const near = left <= 300;
  $("cinputCount").hidden = !near;
  if (near) $("cinputCount").textContent = left > 0
    ? `${left} characters left`
    : "That is the longest message the tutor can read. Send this, then carry on.";
});
$("cinput").addEventListener("input", e => autogrow(e.target));
$("cinput").addEventListener("keydown", e => {
  // Shift+Enter is a newline; plain Enter sends. Ignore Enter mid-IME so
  // composing in another language does not fire off a half-typed message.
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing){
    e.preventDefault();
    submitChat();
  }
});

/* The same conversation stays beside every learning stage. */
function setTutorOpen(open, focus = true){
  toggleWorkspaceTutor(open, focus);
}
$("sheetTog").onclick = () => setTutorOpen(!$("chatcol").classList.contains("open"));

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
  if (resourceKind){
    closeResource();
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
initWorkspace();
loadAssignments();
