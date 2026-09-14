/* Presentation state only. Approval, grading, chat and records stay owned by
   student.js and the server. Changing stages never recreates an editor or chat. */
let workspaceStage = "read", planMethod = "chat";
let workspaceReadyState = false, workspaceComplete = false, workspaceComparison = false;
let workspaceEpoch = 0;
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
  $("planMethods").hidden = tutorReleased;
  $("planStageHint").textContent = tutorReleased ? "Your approved approach stays here whenever you need to look back." : "Talk through your steps with the tutor, or upload a plan you’ve already made.";
  $("finishReview").hidden = !workspaceComplete;
  $("planPreviewStatus").textContent = planGraph?.nodes?.length ? (tutorReleased ? "Approved approach" : "Updated from your thinking") : "Your ideas will appear here";
  $("planEmpty").hidden = !!planGraph?.nodes?.length;
  $("tutorContext").textContent = workspaceStage === "code" ? (workspaceComplete ? "Reviewing your solution" : "Working on your code") : {read:"Working through the question",reflect:"Reflecting on your solution"}[workspaceStage];
  $("backToWork").textContent = workspaceStage === "code" ? "Back to code ↑" : "Back to work ↑";
  syncPlanSubmit();
  placeWorkspaceChat();
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
  $("workspaceOptions").open = false;
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

function choosePlanMethod(method){
  if (!["chat", "upload"].includes(method)) return;
  planMethod = method;
  for (const button of document.querySelectorAll('[data-plan-method]')) button.setAttribute("aria-pressed", String(button.dataset.planMethod === method));
  $("planChatPanel").hidden = method !== "chat";
  $("planUploadPanel").hidden = method !== "upload";
  placeWorkspaceChat();
}

function returnWorkspaceResources(){
  $("planHome").append($("planCard"));
  $("resourceBody").replaceChildren();
  for (const button of document.querySelectorAll('[data-resource="plan"]')) button.setAttribute("aria-expanded", "false");
}

// The question travels with the stage, so the only thing still worth pulling
// into the work column is the plan - it lives on the other tab while coding.
function openResource(kind){
  if (!["plan", "tutor"].includes(kind)) return;
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
  if (kind === "plan"){
    if (planGraph?.nodes?.length){$("planCard").hidden = false; $("resourceBody").append($("planCard"));}
    else {const empty = document.createElement("p"); empty.className = "reference-empty"; empty.textContent = "Describe your approach to the tutor to start building your plan."; $("resourceBody").append(empty);}
  }
  $("resourceBody").scrollTop = 0;
  $("closeResource").focus();
}

function closeResource(restoreFocus = true){
  if (!resourceKind) return;
  resourceKind = null;
  returnWorkspaceResources();
  $("resourceDrawer").hidden = true;
  if (restoreFocus && resourceReturnFocus?.isConnected) resourceReturnFocus.focus();
  resourceReturnFocus = null;
}

function resetWorkspace(){
  workspaceEpoch++;
  closeResource(false);
  workspaceReadyState = false;
  workspaceComplete = false;
  workspaceComparison = false;
  workspaceStage = "read";
  workspaceTutorVisible = true;
  choosePlanMethod("chat");
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
  document.querySelectorAll('[data-stage]').forEach(button => button.addEventListener("click", () => chooseWorkspaceStage(button.dataset.stage)));
  document.querySelectorAll('[data-resource]').forEach(button => button.addEventListener("click", () => openResource(button.dataset.resource)));
  document.querySelectorAll('[data-plan-method]').forEach(button => button.addEventListener("click", () => choosePlanMethod(button.dataset.planMethod)));
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
    if (event.key === "Escape") {event.preventDefault(); event.stopPropagation(); closeResource(); return;}
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
