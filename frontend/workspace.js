/* Presentation state only. Approval, grading, chat and records stay owned by
   student.js and the server. Changing stages never recreates an editor or chat. */
let workspaceStage = "read";
let workspaceReadyState = false, workspaceComplete = false, workspaceComparison = false;
let workspaceEpoch = 0;
let planLoading=false,planUpdating=false,historyUnavailable=false;
/* The stage to land on once it unlocks, set by the history replay and consumed
   once. A problem reopened with its plan already approved used to open on
   "Question & plan" every single time, so every return trip started with a
   click on "Code" that the page could have made itself. Only on RESUME: an
   approval earned in this sitting has its own moment ("Continue to Code →")
   and jumping the student past it would skip the one bit of feedback the gate
   exists to give. */
let resumeStage = null;
let resourceKind = null, resourceReturnFocus = null;
let workspaceTutorVisible = true;
// Reading and planning are ONE stage: the question has to stay on screen while
// the student talks the approach through, and the plan grows underneath it.
const stageNames = {read:"Read", code:"Code", reflect:"Reflect"};

function stageAvailable(stage){
  return stage === "read" || (workspaceReadyState &&
    ((stage === "code" && tutorReleased) || (stage === "reflect" && (workspaceComplete || workspaceComparison))));
}

/* One question card, shown by whichever stage is open. Coding starts it
   collapsed so the editor keeps the height; the student can reopen it. */
function placeQuestion(){
  $(workspaceStage === "code" ? "codeQuestion" : "readQuestion").append($("problemPaper"));
  $("problemDetails").open = workspaceStage !== "code";
}

function workspaceSync(){
  for (const button of document.querySelectorAll('.journey [data-stage]')){
    const stage = button.dataset.stage, available = stageAvailable(stage);
    const selected = workspaceStage === stage;
    button.setAttribute("aria-selected", String(selected));
    button.setAttribute("aria-disabled", String(!available));
    button.tabIndex = selected ? 0 : -1;
    const done = stage === "read" ? tutorReleased : stage === "code" ? workspaceComplete : false;
    button.classList.toggle("is-complete", done);
    button.querySelector('.journey-state').textContent = !available ? (stage === "code" ? "Requires an approved plan" : stage === "reflect" ? "Available after completion" : "Preparing this problem") : done ? "Completed" : "";
  }
  $("planApproved").hidden = !tutorReleased;
  // Only the SUBMISSION controls go once the plan is in - see syncPlanSubmit.
  // Hiding the whole block took "Full file" and "Start over" with it, which
  // are exactly the two a student wants on a problem they have already
  // unlocked and come back to.
  $("planUpload").hidden = tutorReleased;
  $("planStageHint").textContent = tutorReleased ? "Your approved approach stays here whenever you need to look back." : "Describe your approach to the tutor, or upload a plan you’ve made.";
  $("finishReview").hidden = !workspaceComplete;
  $("planPreviewStatus").textContent=historyUnavailable?"Could not load your saved plan":planLoading?"Loading your saved plan…":planUpdating?"Updating your plan…":planGraph?.nodes?.length?(tutorReleased?"Approved approach":"Up to date"):"Ready to build your plan";
  $("planEmpty").hidden=planLoading||planUpdating||historyUnavailable||!!planGraph?.nodes?.length;
  $("tutorContext").textContent = workspaceStage === "code" ? (workspaceComplete ? "Reviewing your solution" : "Working on your code") : {read:"Working through the question",reflect:"Reflecting on your solution"}[workspaceStage];
  $("backToWork").textContent = workspaceStage === "code" ? "Back to code ↑" : "Back to work ↑";
  syncPlanSubmit();
  placeWorkspaceChat();
  // Cleared before the call, so the chooseWorkspaceStage -> workspaceSync loop
  // runs exactly once.
  if (resumeStage && stageAvailable(resumeStage) && workspaceStage === "read"){
    const go = resumeStage; resumeStage = null;
    chooseWorkspaceStage(go, false);
  } else if (resumeStage && workspaceStage !== "read") resumeStage = null;
}

function placeWorkspaceChat(){
  // The chat never moves or replaces the work surface, including during coding.
  $("tutorDock").hidden = !workspaceTutorVisible;
  $("studyDivider").hidden = !workspaceTutorVisible;
  $("studyLayout").classList.toggle("tutor-hidden", !workspaceTutorVisible);
  $("clog").inert = !workspaceTutorVisible;
  $("cform").inert = !workspaceTutorVisible;
  $("chatcol").classList.toggle("open", workspaceTutorVisible);
  for (const id of ["sheetTog", "showTutor"]) $(id).setAttribute("aria-expanded", String(workspaceTutorVisible));
  $("showTutor").textContent = workspaceTutorVisible ? "Hide tutor" : "Show tutor";
}

function toggleWorkspaceTutor(open, focus = true){
  workspaceTutorVisible = open;
  placeWorkspaceChat();
  if (editor) requestAnimationFrame(() => {editor.refresh(); matchGutter();});
  if (focus && open){
    if (matchMedia("(max-width:1000px)").matches) $("tutorDock").scrollIntoView({block:"start", behavior:"instant"});
    $("cinput").focus({preventScroll:!matchMedia("(max-width:1000px)").matches});
  } else if (focus){
    const trigger = document.body.classList.contains("focus-workspace") ? document.querySelector('.code-resources [data-resource="tutor"]') : $("showTutor");
    trigger.focus();
  }
}

function chooseWorkspaceStage(stage, focus = true){
  if (!Object.hasOwn(stageNames, stage)) return;
  if (!stageAvailable(stage)){
    $("workspaceNotice").textContent = !workspaceReadyState ? "Your workspace is still getting ready. You can read the question while you wait." : stage === "code" ? "First, submit your plan for review. Coding unlocks when it is approved." : "Finish the coding steps to unlock your reflection.";
    return;
  }
  closeResource(false);

  workspaceStage = stage;
  $("workspaceNotice").textContent = "";
  for (const [key, name] of Object.entries(stageNames)) $("stage" + name).hidden = key !== stage;
  $("cSolve").dataset.stage = stage;
  placeQuestion();
  if (stage !== "code") setWorkspaceFocus(false);
  workspaceSync();
  if (focus){
    const panel = $("stage" + stageNames[stage]);
    panel.focus({preventScroll:true});
    $("cSolve").scrollIntoView({block:"start", behavior:"instant"});
  }
  if (stage === "code" && editor) requestAnimationFrame(() => {editor.refresh(); matchGutter();});
}

function returnWorkspaceResources(){
  $("planHome").append($("planCard"));
  $("resourceBody").replaceChildren();
  for (const button of document.querySelectorAll('[data-resource="plan"],[data-resource="file"]'))
    button.setAttribute("aria-expanded", "false");
}

/* THE WHOLE ASSIGNMENT FILE, as this student currently has it: their accepted
   answers sitting under their own defs at the right depth, every problem they
   have not finished left as a `# YOUR CODE STARTS HERE` stub, and everything
   they were GIVEN - helper classes, constructors, docstrings, the methods that
   are not exercises - exactly as the teacher wrote it.

   It exists because a step prompt is a keyhole. A student writing chunk 2 of
   `pop` cannot see what `__init__` called the field, what the class hands them,
   or where their own answer is going to land - and guessing at that is not the
   thing they are supposed to be learning.

   Fetched fresh on every open rather than cached: it changes the moment a step
   is accepted, and a stale copy of your own work is worse than a short wait.
   The server never puts a reference solution in it (api_server.assignment_file),
   so there is nothing here to withhold. */
async function renderFileResource(){
  const box = document.createElement("div");
  box.className = "fileview";
  $("resourceBody").append(box);
  const id = typeof openAssign !== "undefined" && openAssign ? openAssign.id : null;
  if (!id){
    box.innerHTML = '<p class="reference-empty">Open a problem from an assignment to see its file.</p>';
    return;
  }
  box.innerHTML = '<p class="reference-empty" role="status">Building your copy of the file…</p>';
  // The drawer can be closed, or the whole problem swapped, while this is in
  // flight. Either one makes the answer to this request the wrong thing to draw.
  const epoch = workspaceEpoch;
  const stale = () => epoch !== workspaceEpoch || resourceKind !== "file" || !box.isConnected;
  try {
    const response = await fetch(`${API}/assignments/${encodeURIComponent(id)}/file`, {cache: "reload"});
    if (!response.ok) throw new Error(String(response.status));
    const data = await response.json();
    if (stale()) return;
    // `going` is the half-finished problems - the route lists them separately
    // from `written` so a partly-answered method is not counted as done. It was
    // read below without ever being declared, which threw a ReferenceError on
    // EVERY render; the catch below could not tell that apart from the fetch
    // failing, so a file that had arrived intact still reported "could not be
    // loaded just now" and the Full file button never worked at all.
    const written = (data.written || []).length,
          going = (data.in_progress || []).length,
          left = (data.remaining || []).length;
    box.replaceChildren();
    const note = document.createElement("p");
    note.className = "fileview-note";
    note.innerHTML = esc(data.filename || "assignment.py") + " · <strong>"
      + esc(`${written} of ${written + going + left}`) + "</strong> finished"
      + (going ? esc(`, ${going} in progress`) : "") + "."
      + (left ? " The rest is marked <code>" + esc("# YOUR CODE STARTS HERE") + "</code>." : "");
    const pre = document.createElement("pre");
    pre.className = "code fileview-code";
    pre.tabIndex = 0;
    pre.textContent = data.text || "";
    box.append(note, pre);
  } catch {
    if (stale()) return;
    box.innerHTML = '<p class="reference-empty">Your file could not be loaded just now. Try opening it again in a moment.</p>';
  }
}

// The question travels with the stage, so the only thing still worth pulling
// into the work column is the plan - it lives on the other tab while coding.
function openResource(kind){
  if (!["plan", "tutor", "file"].includes(kind)) return;
  if (kind === "tutor"){
    toggleWorkspaceTutor(true);
    return;
  }
  if (resourceKind === kind){closeResource(); return;}
  if (!$("resourceDrawer").contains(document.activeElement)) resourceReturnFocus = document.activeElement;
  resourceKind = null;
  returnWorkspaceResources();
  resourceKind = kind;
  $("resourceDrawer").hidden = false;
  $("resourceBody").dataset.resource = kind;
  for (const button of document.querySelectorAll(`[data-resource="${kind}"]`)) button.setAttribute("aria-expanded", "true");
  $("resourceTitle").textContent = kind === "file" ? "The whole file" : "Your plan";
  exitResourceFullscreen();
  $("expandResource").hidden = kind !== "file";
  if (kind === "plan"){
    if (planGraph?.nodes?.length){$("planCard").hidden = false; $("resourceBody").append($("planCard"));}
    else {const empty = document.createElement("p"); empty.className = "reference-empty"; empty.textContent = "Describe your approach to the tutor to start building your plan."; $("resourceBody").append(empty);}
  }
  else if (kind === "file") renderFileResource();
  $("resourceBody").scrollTop = 0;
  $("closeResource").focus();
}

/* Fullscreen for the file view, on the pattern graphs.js already uses: ask for
   real fullscreen, and fall back to a fixed-position class when the browser
   refuses (Safari denies it outright inside some embedded contexts). A whole
   Python file in a 560px drawer is a lot of scrolling, and reading the shape of
   a class is the reason to open it at all. */
function toggleResourceFullscreen(){
  const drawer = $("resourceDrawer"), button = $("expandResource");
  const leave = () => {
    drawer.classList.remove("rfake");
    button.textContent = "Expand";
  };
  if (drawer.classList.contains("rfake")) { leave(); button.focus(); return; }
  if (document.fullscreenElement === drawer) { document.exitFullscreen?.(); return; }
  try {
    if (!drawer.requestFullscreen) throw new Error("unsupported");
    drawer.requestFullscreen().then(() => { button.textContent = "Close"; })
      .catch(() => { drawer.classList.add("rfake"); button.textContent = "Close"; });
  } catch {
    drawer.classList.add("rfake");
    button.textContent = "Close";
  }
}

function exitResourceFullscreen(){
  $("resourceDrawer").classList.remove("rfake");
  if (document.fullscreenElement === $("resourceDrawer")) document.exitFullscreen?.();
  $("expandResource").textContent = "Expand";
}

function closeResource(restoreFocus = true){
  if (!resourceKind) return;
  resourceKind = null;
  exitResourceFullscreen();
  returnWorkspaceResources();
  $("resourceDrawer").hidden = true;
  if (restoreFocus && resourceReturnFocus?.isConnected) resourceReturnFocus.focus();
  resourceReturnFocus = null;
}

function resetWorkspace(){
  planLoading=false;planUpdating=false;historyUnavailable=false;resumeStage=null;
  // A NEW workspace has nothing in flight, whatever the old one was waiting on.
  // The request handlers now decline to touch state after the epoch moves, so
  // without this the flag an abandoned request would have cleared stays true
  // and the new problem's chat is locked with no way back.
  chatBusy=false;
  $("planUpload").open=false;
  workspaceEpoch++;
  closeResource(false);
  workspaceReadyState = false;
  workspaceComplete = false;
  workspaceComparison = false;
  workspaceStage = "read";
  workspaceTutorVisible = true;
  $("stepHistory").open = false;
  $("reflectionCodeDetails").open = false;
  $("reflectionCodeDetails").hidden = false;
  $("workspaceStatus").textContent = "Preparing your workspace. You can read the question while it loads.";
  $("comparisonStatus").textContent = "";
  $("reflectionCode").textContent = "";
  $("readSignature").textContent = "";
  $("signatureBox").hidden = true;
  $("codeOrientation").textContent = "Your plan is ready. Let’s put it into practice.";
  chooseWorkspaceStage("read", false);
}

function readyWorkspace(){
  if(historyUnavailable)return;
  workspaceReadyState = true;
  $("readSignature").textContent = header;
  $("signatureBox").hidden = !header;
  $("workspaceStatus").textContent = "";
  workspaceSync();
}

function failedWorkspace(message){
  workspaceReadyState = false;
  $("workspaceStatus").replaceChildren();
  const text = document.createElement("span");text.textContent = message;
  const retry = document.createElement("button");retry.type = "button";retry.className = "ghost";retry.textContent = "Try again";
  retry.onclick = () => {if (openProblem) start(openProblem);};
  $("workspaceStatus").append(text, retry);
  workspaceSync();
}

function completeWorkspace(res){
  workspaceComplete = true;
  $("reflectTitle").textContent = res.solved_independently ? "You made it, one step at a time." : "You worked through every step.";
  const own = accepted.filter(step => step && step.how === "own").length;
  $("reflectSummary").textContent = `${own} of ${chunks.length} steps passed independently in this attempt. ${res.solved_independently ? "Take a moment to see how your thinking became code." : "Review the steps you needed help with, then practice them again when you’re ready."}`;
  $("reflectionCode").textContent = $("ctxCode").textContent;
  $("reflectionCodeDetails").hidden = false;
  $("reflectionGrades").href = "student-grades.html" + (openAssign ? `?assignment=${encodeURIComponent(openAssign.id)}` : "");
  $("comparisonStatus").textContent = "Preparing your plan and code comparison…";
  workspaceSync();
}

function comparisonReady(restored = false){
  workspaceComparison = true;
  $("comparisonStatus").textContent = "";
  $("reflectionGrades").href = "student-grades.html" + (openAssign ? `?assignment=${encodeURIComponent(openAssign.id)}` : "");
  if (restored && !workspaceComplete){
    $("reflectTitle").textContent = "A look at your earlier work.";
    $("reflectSummary").textContent = "This comparison is from your previous attempt. The Code stage begins a new set of steps; your recorded grades stay available in My grades.";
    $("reflectionCodeDetails").hidden = true;
  } else $("reflectionCodeDetails").hidden = false;
  workspaceSync();
}

function comparisonUnavailable(){
  const text = document.createElement("p");text.textContent = "Your problem is complete. The plan and code comparison could not be loaded.";
  const retry = document.createElement("button");retry.type = "button";retry.className = "ghost";retry.textContent = "Retry comparison";
  retry.onclick = () => {$("comparisonStatus").textContent = "Loading comparison…"; showDualGraphs(sessionId);};
  $("comparisonStatus").replaceChildren(text, retry);
}

function initWorkspace(){
  $("jumpToPlan").onclick=()=>$("planPreview").scrollIntoView({block:"start",behavior:"smooth"});
  $("openPlanUpload").onclick=()=>{
    if(planLoading||historyUnavailable||!workspaceReadyState)return;
    $("planUpload").open=true;$("designFile").click();
  };
  $("designFile").addEventListener("change",()=>{
    if($("designFile").files.length){$("planUpload").open=true;$("planUpload").scrollIntoView({block:"nearest",behavior:"smooth"});}
  });
  document.querySelectorAll('[data-stage]').forEach(button => button.addEventListener("click", () => chooseWorkspaceStage(button.dataset.stage)));
  document.querySelectorAll('[data-resource]').forEach(button => button.addEventListener("click", () => openResource(button.dataset.resource)));
  $("expandResource").addEventListener("click", toggleResourceFullscreen);
  // The real fullscreen has its own exit (Escape, the browser chrome); keep the
  // button's label honest when it is used.
  document.addEventListener("fullscreenchange", () => {
    if (document.fullscreenElement !== $("resourceDrawer"))
      $("expandResource").textContent = "Expand";
  });
  $("planContinue").onclick = () => chooseWorkspaceStage("code");
  $("finishReview").onclick = () => chooseWorkspaceStage("reflect");
  $("reflectionNext").onclick = backToProblems;
  $("closeResource").onclick = () => closeResource();
  $("showTutor").onclick = () => toggleWorkspaceTutor(!workspaceTutorVisible);
  $("backToWork").onclick = () => {
    const panel = $("stage" + stageNames[workspaceStage]);
    panel.scrollIntoView({block:"start", behavior:"instant"});
    if (workspaceStage === "code" && editor && tutorReleased && reviewIdx === null && !workspaceComplete) editor.focus();
    else panel.focus({preventScroll:true});
  };
  initTutorResize();
  document.querySelector('.journey [role="tablist"]').addEventListener("keydown", event => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const buttons = [...document.querySelectorAll('.journey [role="tab"]')];
    const index = buttons.indexOf(document.activeElement);
    const next = event.key === "Home" ? 0 : event.key === "End" ? buttons.length - 1 : (index + (event.key === "ArrowRight" ? 1 : -1) + buttons.length) % buttons.length;
    buttons.forEach((b, i) => {b.tabIndex = i === next ? 0 : -1;});
    buttons[next].focus();
  });
  $("resourceDrawer").addEventListener("keydown", event => {
    if (event.key !== "Escape") return;
    event.preventDefault(); event.stopPropagation();
    // Escape steps OUT of fullscreen before it closes the drawer. Collapsing
    // both at once means a student who expanded the file to read it loses the
    // file as well as the fullscreen, and has to find the button again.
    if ($("resourceDrawer").classList.contains("rfake")){
      exitResourceFullscreen(); $("expandResource").focus(); return;
    }
    closeResource();
  });
  resetWorkspace();
}

function initTutorResize(){
  const divider = $("studyDivider"), layout = $("studyLayout");
  let origin = null;
  const width = () => Math.round($("tutorDock").getBoundingClientRect().width);
  const maxWidth = () => Math.max(300, Math.min(480, layout.clientWidth - 508));
  const syncSize = () => {
    if (!workspaceTutorVisible || matchMedia("(max-width:1000px)").matches || !layout.clientWidth) return;
    if (width() > maxWidth()){resize(width()); return;}
    divider.setAttribute("aria-valuenow", String(width()));
    divider.setAttribute("aria-valuemax", String(maxWidth()));
  };
  const resize = next => {
    const size = Math.max(300, Math.min(maxWidth(), next));
    layout.style.setProperty("--tutor-width", `${size}px`);
    divider.setAttribute("aria-valuenow", String(size));
    divider.setAttribute("aria-valuemax", String(maxWidth()));
    if (editor) {editor.refresh(); matchGutter();}
  };
  divider.addEventListener("pointerdown", event => {
    if (event.button !== 0) return;
    origin = {x:event.clientX, width:width()};
    divider.setPointerCapture(event.pointerId);
    layout.classList.add("resizing");
    divider.focus();
  });
  divider.addEventListener("pointermove", event => {if (origin) resize(origin.width + origin.x - event.clientX);});
  const stop = () => {origin = null; layout.classList.remove("resizing");};
  divider.addEventListener("pointerup", stop);
  divider.addEventListener("pointercancel", stop);
  divider.addEventListener("lostpointercapture", stop);
  divider.addEventListener("keydown", event => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    resize(event.key === "Home" ? 300 : event.key === "End" ? maxWidth() : width() + (event.key === "ArrowLeft" ? 20 : -20));
  });
  addEventListener("resize", () => {
    if (workspaceTutorVisible && !matchMedia("(max-width:1000px)").matches && layout.style.getPropertyValue("--tutor-width")) resize(width());
    syncSize();
  });
  if (typeof ResizeObserver !== "undefined") new ResizeObserver(syncSize).observe(layout);
}
