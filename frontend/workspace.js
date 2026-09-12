/* Presentation state only. Approval, grading, chat and records stay owned by
   student.js and the server. Changing stages never recreates an editor or chat. */
let workspaceStage = "read", planMethod = "chat";
let workspaceReadyState = false, workspaceComplete = false, workspaceComparison = false;
let workspaceEpoch = 0;
let resourceKind = null, resourceReturnFocus = null, resourceInert = [];
const stageNames = {read:"Read", plan:"Plan", code:"Code", reflect:"Reflect"};

function stageAvailable(stage){
  return stage === "read" || (workspaceReadyState && (stage === "plan" ||
    (stage === "code" && tutorReleased) || (stage === "reflect" && (workspaceComplete || workspaceComparison))));
}

function workspaceSync(){
  for (const button of document.querySelectorAll('.journey [data-stage]')){
    const stage = button.dataset.stage, available = stageAvailable(stage);
    const selected = workspaceStage === stage;
    button.setAttribute("aria-selected", String(selected));
    button.setAttribute("aria-disabled", String(!available));
    button.tabIndex = selected ? 0 : -1;
    const done = stage === "read" ? workspaceStage !== "read" : stage === "plan" ? tutorReleased : stage === "code" ? workspaceComplete : false;
    button.classList.toggle("is-complete", done);
    button.querySelector('.journey-state').textContent = !available ? (stage === "code" ? "Requires an approved plan" : stage === "reflect" ? "Available after completion" : "Preparing this problem") : done ? "Completed" : "";
  }
  $("readContinue").disabled = !workspaceReadyState;
  $("readContinue").textContent = tutorReleased ? "Continue to Code →" : "Continue to Plan →";
  $("readNextTitle").textContent = tutorReleased ? "Your plan is already approved." : "A plan comes before the code.";
  $("readNextHint").textContent = tutorReleased ? "You can revisit your thinking or work through the coding steps." : "You’ll work through your approach next. No code needed yet.";
  $("planApproved").hidden = !tutorReleased;
  $("planMethods").hidden = tutorReleased;
  $("planStageHint").textContent = tutorReleased ? "Your approved approach stays here whenever you need to look back." : "Choose how you’d like to make your plan. You can switch methods at any time.";
  $("finishReview").hidden = !workspaceComplete;
  $("planPreviewStatus").textContent = planGraph?.nodes?.length ? (tutorReleased ? "Approved approach" : "Updated from your thinking") : "Your ideas will appear here";
  syncPlanSubmit();
  placeWorkspaceChat();
}

function placeWorkspaceChat(){
  const inline = workspaceStage === "plan" && planMethod === "chat" && !tutorReleased;
  const chat = $("chatcol");
  const target = resourceKind === "tutor" ? $("resourceBody") : inline ? $("planChatHome") : $("chatParking");
  if (chat.parentElement !== target) target.append(chat);
  const visible = resourceKind === "tutor" || inline;
  $("clog").inert = !visible;
  $("cform").inert = !visible;
  chat.classList.toggle("open", visible);
  $("sheetTog").setAttribute("aria-expanded", String(visible));
  $("sheetTog").setAttribute("aria-label", visible ? "Close the tutor" : "Open the tutor");
}

function chooseWorkspaceStage(stage, focus = true){
  if (!Object.hasOwn(stageNames, stage)) return;
  if (!stageAvailable(stage)){
    $("workspaceNotice").textContent = !workspaceReadyState ? "Your workspace is still getting ready. You can read the question while you wait." : stage === "code" ? "First, submit your plan in the Plan stage. Coding unlocks when it is approved." : "Finish the coding steps to unlock your reflection.";
    return;
  }
  closeResource(false);
  $("workspaceOptions").open = false;
  workspaceStage = stage;
  $("workspaceNotice").textContent = "";
  for (const [key, name] of Object.entries(stageNames)) $("stage" + name).hidden = key !== stage;
  $("cSolve").dataset.stage = stage;
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
  $("problemHome").append($("problemPaper"));
  $("planHome").append($("planCard"));
  placeWorkspaceChat();
}

function openResource(kind){
  if (!["problem", "plan", "tutor"].includes(kind)) return;
  if (!resourceKind && kind === "tutor" && workspaceStage === "plan" && planMethod === "chat" && !tutorReleased){
    $("cinput").focus();
    return;
  }
  if (!resourceKind){
    resourceReturnFocus = document.activeElement;
    resourceInert = [...document.body.children].filter(el => el !== $("resourceDrawer")).map(el => [el, el.inert]);
    resourceInert.forEach(([el]) => {el.inert = true;});
  }
  resourceKind = null;
  returnWorkspaceResources();
  resourceKind = kind;
  $("resourceDrawer").hidden = false;
  document.body.classList.add("resource-open");
  $("resourceTitle").textContent = {problem:"The question", plan:"Your plan", tutor:"Your tutor"}[kind];
  $("resourceBody").dataset.resource = kind;
  for (const b of document.querySelectorAll('.resource-tabs [data-resource]')) b.setAttribute("aria-pressed", String(b.dataset.resource === kind));
  if (kind === "problem") {$("problemDetails").open = true; $("resourceBody").append($("problemPaper"));}
  if (kind === "plan") {$("planCard").hidden = false; $("resourceBody").append($("planCard"));}
  placeWorkspaceChat();
  $("closeResource").focus();
}

function closeResource(restoreFocus = true){
  if (!resourceKind) return;
  resourceKind = null;
  returnWorkspaceResources();
  $("resourceDrawer").hidden = true;
  document.body.classList.remove("resource-open");
  resourceInert.forEach(([el, inert]) => {el.inert = inert;});
  resourceInert = [];
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
  choosePlanMethod("chat");
  $("stepHistory").open = false;
  $("planPreview").open = false;
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
  $("readContinue").onclick = () => chooseWorkspaceStage(tutorReleased ? "code" : "plan");
  $("planContinue").onclick = () => chooseWorkspaceStage("code");
  $("finishReview").onclick = () => chooseWorkspaceStage("reflect");
  $("reflectionNext").onclick = backToProblems;
  $("closeResource").onclick = () => closeResource();
  $("resourceDrawer").onclick = event => {if (event.target === $("resourceDrawer")) closeResource();};
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
    if (event.key !== "Tab") return;
    const elements = [...$("resourceDrawer").querySelectorAll('button, a[href], textarea, input, select, summary, [tabindex="0"]')].filter(el => !el.disabled && !el.closest('[hidden], [inert]') && getComputedStyle(el).display !== "none");
    const first = elements[0], last = elements[elements.length - 1];
    if (event.shiftKey && document.activeElement === first) {event.preventDefault();last?.focus();}
    else if (!event.shiftKey && document.activeElement === last) {event.preventDefault();first?.focus();}
  });
  resetWorkspace();
}
