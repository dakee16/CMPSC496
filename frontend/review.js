/* review.js - one problem waiting on review, said plainly, with ways to fix it.

   Everything shown comes from /teacher/review (main/review_help.py): the
   changes the tests missed and their lines are known for free; the plain words
   and the suggested examples are written once per problem version by one small
   model call, and the examples are checked against the teacher's own code
   before they are offered. Fixing goes through the same routes the inline fix
   panel uses - /teacher/problems/retry and /teacher/problems/accept. */
const S = requireSession("teacher");
const q = new URLSearchParams(location.search);
const A = q.get("a") || "", SLUG = q.get("slug") || "";
const $ = id => document.getElementById(id);
let view = null;
let notice = "";              // the last action's outcome, kept across redraws
// The teacher's own test case (15 Oct): what they typed and what it caught,
// kept across redraws until a new version is prepared.
let mine = {text: "", out: null};

mountHeader({variant: "instructor", active: "Assignments",
  crumbs: [{label: "Assignments", href: "teacher-assignments.html"},
           {label: "Review"}]});

async function call(path, method = "GET", body = null){
  const r = await fetch(`${API}${path}`, {method,
    headers: body ? {"Content-Type": "application/json"} : {},
    body: body ? JSON.stringify(body) : null});
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((d.detail && d.detail.message) || `The server said ${r.status}.`);
  return d;
}

const url = (slug = SLUG) =>
  `/teacher/review/${encodeURIComponent(A)}/${encodeURIComponent(slug)}`;

/* ---------- loading ---------- */

async function load(){
  if (!A || !SLUG){
    $("main").innerHTML = `<div class="banner bad">This link is missing which
      problem to review. Open it from the Assignments page.</div>`;
    return;
  }
  try { view = await call(url()); }
  catch (e) { return fail(e); }
  render();
  await explain();
}

async function explain(){
  // The plain words are written ONCE per problem version and remembered, so
  // this costs nothing on a second visit - the server answers from memory.
  if (reviewable(view) && view.checks.length && !view.help){
    $("helpNote").innerHTML = `<div class="banner info"><span class="spin"
      aria-hidden="true"></span> Writing a plain explanation and checking
      example fixes against your solution (about a cent, once)…</div>`;
    try { view = await call(url() + "/explain", "POST"); }
    catch (e) { view.help = {written: false, error: e.message}; }
    render();
  }
}

function fail(e){
  $("main").innerHTML = `<div class="banner bad">${esc(e.message ||
    "Could not load this review.")}</div>`;
}

/* ---------- drawing ---------- */

/* What this page is for, for this problem: tests to decide on (in review),
   wrong answers still passing (ready, with gaps), or roadmaps under the bar
   (14 Oct, main/roadmap_score) - or nothing. */
const roadmapsHere = v => !!(v.roadmaps && (v.roadmaps.review || v.roadmaps.below));
const reviewable = v => !!(v.in_review || v.gaps || roadmapsHere(v));
// The tests are the question only while a problem waits on THEM; once they
// passed (live, or stopped later at the roadmaps) nothing about them is to accept.
const testsInReview = v => !!(v.in_review && !(v.roadmaps && v.roadmaps.review));

function render(){
  const v = view, R = v.roadmaps || {};
  document.title = `Review: ${v.title} · ACADIA`;
  $("title").textContent = `Review: ${v.title}`;
  $("eyebrow").textContent = `${v.assignment.name || "Assignment"} · `
    + (v.in_review ? (R.review ? "Needs your review: roadmaps" : "Needs your review")
       : v.gaps ? "Ready, with gaps in its tests"
       : R.below ? "Ready, with roadmaps under the bar"
       : v.ready ? "Ready" : "Not ready");
  $("sub").textContent = v.in_review
    ? "Students can't see this problem until you decide. Nothing here changes "
      + "your solution unless you choose to."
    : v.gaps ? "Students can already work on this problem. The wrong answers below "
      + "would still be marked correct - adding the test cases closes that."
    : R.below ? "Students can already work on this problem, with the best roadmaps "
      + "below - none of which reaches the bar."
    : v.ready ? "Students can see this problem." : "";
  setCrumbs([{label: "Assignments", href: "teacher-assignments.html"},
             {label: v.assignment.name || "Assignment", href: "teacher-assignments.html"},
             {label: `Review: ${v.title}`}]);
  drawQueue();

  if (!reviewable(v)){
    $("main").innerHTML = v.ready
      ? `<div class="banner ok">${esc(v.title)} is ready and visible to students.
          Nothing is waiting on you here.</div>`
      : `<div class="banner warn">${esc(v.title)} is not waiting on review - it
          stopped for another reason: ${esc(v.error || "unknown")}. Fix it from
          the Assignments page.</div>`;
    return;
  }
  const showTests = v.checks.length > 0 || testsInReview(v);
  $("main").innerHTML = (roadmapsHere(v) ? roadmapCard(!showTests) : "")
    + (showTests ? testCards() : "")
    + (v.harmless && v.harmless.length ? harmlessCard() : "")
    + `<section class="card solution">
      <h2 style="margin:0 0 12px">Your solution</h2>
      <pre id="sol"></pre>
      <p class="note">${showTests ? "Highlighted: the lines the missed changes are on."
        : "The code every roadmap is cut from."}</p>
    </section>`;

  const lines = new Set(showTests ? v.lines || [] : []);
  $("sol").innerHTML = (v.source || "").split("\n").map((c, i) =>
    `<div class="srow${lines.has(i + 1) ? " hit" : ""}"><span class="n">${i + 1}</span><span
      class="c">${c ? esc(c) : " "}</span></div>`).join("");
  // Plain text, not hlPython: it highlights one line at a time here, and a
  // line inside a multi-line docstring is not valid Python on its own - it
  // broke the row it sat in.
  document.querySelectorAll("pre[data-test]").forEach(pre =>
    pre.innerHTML = hlPython(v.help.tests[+pre.dataset.test].doctest));
  if ($("trySrc")) $("trySrc").value = mine.text;
  if ($("tryEx") && mine.out && mine.out.doctest) $("tryEx").innerHTML = hlPython(mine.out.doctest);
  wire();
}

function testCards(){
  const v = view, h = v.help;
  const missed = v.checks.length, total = v.total || missed;
  const gaps = v.checks.filter(c => c.verdict === "gap").length;
  const maybe = v.checks.filter(c => c.verdict === "maybe_harmless").length;
  return `
    <section class="card">
      <h2 style="margin:0 0 6px">What we found</h2>
      <p class="hint" style="margin:0 0 16px">To check the tests are good enough
        to grade with, we break your solution on purpose - one small change at a
        time - and see whether a test notices.</p>
      <div class="score">
        <span class="num">${v.caught} of ${total}</span>
        <div class="meter" role="img" aria-label="${v.caught} caught, ${missed} missed">
          <i class="caught" style="width:${100 * v.caught / total}%"></i><i
             class="missed" style="width:${100 * missed / total}%"></i>
        </div>
      </div>
      <p class="lbl">mistakes were caught. <strong>${missed} slipped past</strong>:
        a student who made one of them would be marked correct${
        h && h.examples_ok ? (gaps === missed ? "."
          : gaps ? ` - we proved ${gaps} of them matter. The suggested test cases
              didn't catch the other ${missed - gaps}, which doesn't prove they're harmless.`
          : ". The suggested test cases didn't catch any of them, which doesn't prove they're harmless.") : "."}</p>
    </section>

    <section class="card">
      <h2 style="margin:0 0 14px">The ${missed === 1 ? "one" : missed} that slipped past</h2>
      <div id="helpNote">${h && !h.written ? `<div class="banner warn">The plain
        explanation could not be written right now (${esc(h.error || "unknown")}).
        What was tried and where is below; you can still decide.</div>` : ""}</div>
      ${v.checks.map(finding).join("")}
    </section>

    <section class="card">
      <h2 style="margin:0 0 14px">How to fix it</h2>
      ${options(gaps, maybe)}
      <div id="actMsg" aria-live="polite">${notice}</div>
    </section>`;
}

/* THE ROADMAPS, GRADED (14 Oct). Students are given the best three that
   reach the bar; this card is shown when none does. */
const PART_MAX = {"biggest step": 30, "balance": 25, "number of steps": 15,
                  "clean starts": 15, "values handed on": 15};

function roadmapCard(withMsg){
  const v = view, R = v.roadmaps;
  const live = !!v.ready;
  const list = (R.list || []).map((r, i) => `
    <li class="rmap">
      <div class="rmhead"><span class="rmpts">${r.points}<small>/100</small></span>
        <span class="rmshape">Roadmap ${i + 1}: ${r.sizes.length} ${r.sizes.length === 1 ? "step" : "steps"}
          of ${r.sizes.join(" + ")} lines</span></div>
      <ol class="rmsteps">${r.steps.map(q => `<li>${esc(q)}</li>`).join("")}</ol>
      <p class="rmparts">${Object.entries(r.parts).map(([k, n]) =>
        `${esc(k)} <b>${n}/${PART_MAX[k] || "?"}</b>`).join(" · ")}</p>
      ${r.faults.length ? `<p class="rmfault">Also: ${r.faults.map(esc).join("; ")}.</p>` : ""}
    </li>`).join("");
  const big = Math.max(0, ...(R.list || []).map(r => Math.max(...r.sizes)));
  return `
    <section class="card" id="rmCard">
      <h2 style="margin:0 0 6px">Roadmaps</h2>
      <p class="hint" style="margin:0 0 14px">A roadmap is your solution cut into
        steps for students. Each gets a grade out of 100 - small steps, even steps,
        the right number of steps, steps that start cleanly, and few values handed
        from one step to the next. Students are given the best three that score at
        least <strong>${R.cutoff}</strong>.</p>
      <div class="banner warn" style="margin:0 0 14px">None of this problem's roadmaps
        reaches ${R.cutoff} - the best scores ${R.best}. ${live
        ? "Students still get the best ones until you decide."
        : "Students can't see this problem until you decide."}</div>
      <ol class="rmaps">${list}</ol>
      <h3 style="margin:18px 0 10px; font-size:1rem">What you can do</h3>
      <div class="option${big > 12 ? " rec" : ""}">
        <h4>Split the method into smaller helper methods${big > 12
          ? ` <span class="pill ok">recommended</span>` : ""}</h4>
        <p>Your method is probably too long to cut into small steps${big > 12
          ? ` - its best roadmap still has a ${big}-line step` : ""}. Moving a part
          into its own helper method makes every step smaller. Edit it, then
          prepare it again.</p>
        <div class="acts"><button type="button" class="ghost" id="rmEdit">Open the editor</button></div>
        <div id="rmEditor"></div>
      </div>
      <div class="option${big > 12 ? "" : " rec"}">
        <h4>Build new roadmaps${big > 12 ? "" : ` <span class="pill ok">recommended</span>`}</h4>
        <p>Cuts your solution again from scratch. The new ones are used only if they
          score at least as well${live ? ", and students keep the current ones until then" : ""}.</p>
        <div class="acts"><button type="button" class="ghost" id="rmRebuild">Build new roadmaps</button>
          <span class="cost">A few cents - your tests are reused.</span></div>
      </div>
      <div class="option">
        <h4>${live ? "Keep these roadmaps" : "Publish with the best roadmap"}</h4>
        <p>${live ? "They stay as they are, and this stops being flagged."
          : "Students get the roadmaps above."} Your name is recorded against the decision.</p>
        <div class="acts"><button type="button" class="ghost" id="rmAccept">${live
          ? "Keep these roadmaps" : "Publish with the best roadmap"}</button>
          <span class="cost">Free.</span></div>
      </div>
      ${withMsg ? `<div id="actMsg" aria-live="polite">${notice}</div>` : ""}
    </section>`;
}

function harmlessCard(){
  return `<section class="card">
      <h2 style="margin:0 0 10px">Marked harmless</h2>
      <ul class="harmless">${view.harmless.map(c => `<li>${said(c.phrasing || c.label)}${
        c.line ? ` <span class="hl-ln">(line ${c.line})</span>` : ""} - marked by ${esc(c.by)}.
        Not counted as a gap.</li>`).join("")}</ul>
    </section>`;
}

/* `x` in the engine's wording is code: shown as code, never as backticks. */
const said = t => esc(t).replace(/`([^`]+)`/g, "<code>$1</code>");

function finding(c){
  const pill = {
    gap: `<span class="pill warn">real gap - we found a case where it matters</span>`,
    maybe_harmless: `<span class="pill">the suggested test cases missed it</span>`,
    unknown: ""}[c.verdict] || "";
  return `<article class="finding${c.verdict === "maybe_harmless" ? " maybe" : ""}">
    <div class="top">
      <h4>${c.title ? esc(c.title) : said(`We tried ${c.phrasing || c.label}`)}</h4>${pill}
    </div>
    ${c.line ? `<div class="where"><span class="ln">line ${c.line}</span><span
      class="src">${esc(c.code)}</span></div>` : ""}
    <dl>
      <dt>We tried</dt><dd>${said(c.phrasing || c.label)}. Every test still passed.</dd>
      ${c.missing ? `<dt>Missing test</dt><dd>${esc(c.missing)}.</dd>` : ""}
    </dl>
    ${view.ready ? `<div class="acts"><button type="button" class="ghost harmlessGo"
      data-label="${esc(c.label)}">Mark as harmless</button>
      <span class="cost">If no test could ever catch it. Your name is recorded.</span></div>` : ""}
  </article>`;
}

function options(gaps, maybe){
  const h = view.help;
  const tests = (h && h.tests) || [];
  const canAdd = !!(h && h.edited_source && tests.length);
  const n = tests.length;
  /* ONE CARD PER TEST CASE, each saying which wrong answer it would catch
     (13 Oct, Sanan): the fix for a problem in review is usually "add these
     tests", so the page shows exactly those tests and one button for all. */
  const add = canAdd ? `
    <div class="option rec">
      <h4>Add ${n === 1 ? "this test case" : `these ${n} test cases`}
        <span class="pill ok">recommended</span></h4>
      <p>Each one was run on your solution before it was shown here - every answer
        below is your code's - and on each wrong answer it is meant to catch. They
        go into the docstring, so students see them too.</p>
      <ol class="tcases">${tests.map((t, i) => `
        <li class="tcase">
          <div class="tchead"><span class="tcnum">Test case ${i + 1}</span>
            <span class="tcatch">Catches: ${t.catches.map(esc).join("; ")}</span></div>
          <pre class="code ex" data-test="${i}"></pre>
        </li>`).join("")}
      </ol>
      <ul class="checks">
        <li>Your solution passes all of them</li>
        <li>Together they catch ${h.catches} of the ${view.checks.length} wrong ${view.checks.length === 1 ? "answer" : "answers"} above</li>
      </ul>
      <div class="acts">
        <button type="button" id="addGo">Add ${n === 1 ? "it" : `all ${n}`} and prepare again</button>
        <button type="button" class="ghost" id="addEdit">Edit them first</button>
        <span class="cost">Prepares just this problem again - a minute or two.</span>
      </div>
    </div>` : "";
  if (!testsInReview(view))            // the tests passed already: nothing to accept
    return `${add}${yours(canAdd)}
    <div class="option">
      <h4>Leave it as it is</h4>
      <p>${canAdd ? (view.ready ? "Students keep working on it as it is now, and the wrong answers above keep passing."
          : "The tests stay as they are, and the wrong answers above keep passing.")
        : "Only if no test could ever tell these apart from your solution - for example, a line that sets something to what it already is. If you're not sure, write your own test case first."}</p>
    </div>
    <div class="option">
      <h4>Edit the problem yourself</h4>
      <p>Change the statement, the examples or the solution, then prepare this
        problem again.</p>
      <div class="acts"><button type="button" class="ghost" id="editGo">Open the editor</button></div>
      <div id="editor"></div>
    </div>`;
  const why = canAdd
    ? `Here, we found cases where they do differ, so we don't recommend it.`
    : `The suggested test cases didn't tell them apart, but that is not proof -
       write your own test case first if you're not sure.`;
  return `${add}${yours(canAdd)}
    <div class="option">
      <h4>These can't change the answer - publish anyway</h4>
      <p>Only if you've read the lines above and no student answer could differ
        because of them. ${why} Your name is recorded against the decision.</p>
      <div class="acts"><button type="button" class="ghost" id="acceptGo">Publish anyway</button></div>
    </div>
    <div class="option">
      <h4>Edit the problem yourself</h4>
      <p>Change the statement, the examples or the solution, then prepare this
        problem again.</p>
      <div class="acts"><button type="button" class="ghost" id="editGo">Open the editor</button></div>
      <div id="editor"></div>
    </div>`;
}

/* YOUR OWN TEST CASE (15 Oct). HW4 put: the suggested cases aimed at the
   wrong limit and none caught `>` -> `>=`. The teacher's own case is run on
   their solution and on every change above, free, and added like the
   suggested ones. */
function yours(canAdd){
  return `
    <div class="option${canAdd ? "" : " rec"}">
      <h4>Write your own test case${canAdd ? "" : ` <span class="pill ok">recommended</span>`}</h4>
      <p>Type it the way a docstring example looks - <code>&gt;&gt;&gt;</code> lines,
        or plain lines of Python. We run it on your solution and on each wrong
        answer above, and say which ones it catches. Your solution supplies the
        answers, so you don't need to type them.</p>
      <textarea class="rvEdit tryBox" id="trySrc" rows="4" spellcheck="false"
        aria-label="Your test case"
        placeholder="&gt;&gt;&gt; obj = YourClass(10)&#10;&gt;&gt;&gt; obj.method(10)"></textarea>
      <div class="acts"><button type="button" class="ghost" id="tryGo">Check it</button>
        <span class="cost">Free - nothing is changed until you add it.</span></div>
      <div id="tryOut" aria-live="polite">${tryResult()}</div>
    </div>`;
}

function tryResult(){
  const o = mine.out;
  if (!o) return "";
  if (!o.ok) return `<div class="banner bad">${esc(o.reason || "It could not be checked.")}</div>`;
  if (!o.catches.length)
    return `<div class="banner warn">It didn't catch any of them - your solution and
      every wrong answer above give the same results on it.</div>
      <pre class="code ex" id="tryEx"></pre>`;
  return `
    <div class="tcase">
      <div class="tchead"><span class="tcnum">Your test case</span>
        <span class="tcatch">Catches: ${o.catches.map(esc).join("; ")}</span></div>
      <pre class="code ex" id="tryEx"></pre>
    </div>
    <p class="note">${o.missed.length ? `Still not caught: ${o.missed.map(esc).join("; ")}.`
      : "It catches every wrong answer above."}</p>
    <div class="acts">
      <button type="button" id="tryAdd">Add it and prepare again</button>
      <button type="button" class="ghost" id="tryEdit">Edit it first</button>
      <span class="cost">Prepares just this problem again - a minute or two.</span>
    </div>
    <div id="tryEditor"></div>`;
}

async function tryCase(btn){
  mine.text = $("trySrc").value;
  if (!mine.text.trim()) return;
  setBusy(btn, true, "Checking…");
  try { mine.out = await call(url() + "/try", "POST", {text: mine.text}); }
  catch (e) { mine.out = {ok: false, reason: e.message}; }
  setBusy(btn, false);
  render();
}

function drawQueue(){
  const qd = view.queue || [];
  $("qsub").textContent = qd.length
    ? `${qd.length} problem${qd.length === 1 ? "" : "s"} in ${view.assignment.name || "this assignment"}`
    : "Nothing else is waiting.";
  $("queue").innerHTML = qd.map(p => `<li><a href="review.html?a=${encodeURIComponent(A)}&slug=${
    encodeURIComponent(p.slug)}" class="${p.slug === SLUG ? "on" : ""}"><span
    class="qn">${esc(p.title)}</span></a></li>`).join("");
}

/* ---------- acting ---------- */

function wire(){
  const add = $("addGo"), edit = $("addEdit"), acc = $("acceptGo"), ed = $("editGo");
  if (add) add.onclick = () => prepare(view.help.edited_source, add);
  if (edit) edit.onclick = () => openEditor(view.help.edited_source);
  if (ed) ed.onclick = () => openEditor(view.source);
  if (acc) acc.onclick = () => accept(acc);
  const rmEdit = $("rmEdit"), rmRebuild = $("rmRebuild"), rmAccept = $("rmAccept");
  if (rmEdit) rmEdit.onclick = () => openEditor(view.source, "rmEditor");
  if (rmRebuild) rmRebuild.onclick = () => rebuild(rmRebuild);
  if (rmAccept) rmAccept.onclick = () => keepRoadmaps(rmAccept);
  document.querySelectorAll("button.harmlessGo").forEach(b =>
    b.onclick = () => dismiss(b.dataset.label, b));
  const tryGo = $("tryGo"), tryAdd = $("tryAdd"), tryEdit = $("tryEdit");
  if (tryGo) tryGo.onclick = () => tryCase(tryGo);
  if (tryAdd) tryAdd.onclick = () => prepare(mine.out.edited_source, tryAdd);
  if (tryEdit) tryEdit.onclick = () => openEditor(mine.out.edited_source, "tryEditor");
}

function openEditor(text, host = "editor"){
  $(host).innerHTML = `<textarea class="rvEdit" id="edSrc" spellcheck="false"
    aria-label="Problem source"></textarea>
    <div class="acts"><button type="button" id="edGo">Prepare again</button>
      <span class="cost">Your edits stay here if anything goes wrong.</span></div>`;
  $("edSrc").value = text || "";
  $("edGo").onclick = () => prepare($("edSrc").value, $("edGo"));
  $("edSrc").focus();
  $(host).scrollIntoView({block: "nearest"});
}

function lock(on){
  document.querySelectorAll("#main button").forEach(b => { if (!b.dataset.busy) b.disabled = on; });
}

async function prepare(source, btn){
  if (!source || !source.trim()) return;
  notice = "";
  lock(true); setBusy(btn, true, "Preparing…");
  $("actMsg").innerHTML = `<div class="banner info">Preparing this problem again:
    writing tests, checking they are strong enough, and splitting it into steps.
    This takes a minute or two - you can leave this page open.</div>`;
  let d;
  try { d = await call("/teacher/problems/retry", "POST",
                       {assignment_id: A, slug: SLUG, source, keep_live: !!view.ready}); }
  catch (e) {
    setBusy(btn, false); lock(false);
    $("actMsg").innerHTML = `<div class="banner bad">${esc(e.message)} Nothing was
      changed.</div>`;
    return;
  }
  setBusy(btn, false);
  await after(d);
}

async function accept(btn){
  if (!confirm(`Publish "${view.title}" with ${view.checks.length} check(s)
    undecided? Your name is recorded against this.`)) return;
  lock(true); setBusy(btn, true, "Publishing…");
  $("actMsg").innerHTML = `<div class="banner info">Publishing: the tests are
    accepted, and the problem is now split into steps for students. A minute or
    two.</div>`;
  notice = "";
  let d;
  try { d = await call("/teacher/problems/accept", "POST", {assignment_id: A, slug: SLUG}); }
  catch (e) {
    setBusy(btn, false); lock(false);
    $("actMsg").innerHTML = `<div class="banner bad">${esc(e.message)}</div>`;
    return;
  }
  setBusy(btn, false);
  await after(d);
}

/* Build this problem's roadmaps again (14 Oct). Tests are reused. */
async function rebuild(btn){
  notice = "";
  lock(true); setBusy(btn, true, "Building…");
  $("actMsg").innerHTML = `<div class="banner info">Building new roadmaps: cutting
    your solution into steps again and grading each one. About a minute.</div>`;
  let d;
  try { d = await call("/teacher/problems/rebuild_roadmaps", "POST", {assignment_id: A, slug: SLUG}); }
  catch (e) {
    setBusy(btn, false); lock(false);
    $("actMsg").innerHTML = `<div class="banner bad">${esc(e.message)} Nothing was changed.</div>`;
    return;
  }
  setBusy(btn, false);
  await after(d, "rebuild");
}

async function keepRoadmaps(btn){
  if (!confirm(view.ready ? `Keep "${view.title}"'s roadmaps as they are? Your name is recorded against this.`
      : `Publish "${view.title}" with roadmaps under the bar? Your name is recorded against this.`)) return;
  notice = "";
  lock(true); setBusy(btn, true, view.ready ? "Keeping…" : "Publishing…");
  let d;
  try { d = await call("/teacher/problems/accept_roadmaps", "POST", {assignment_id: A, slug: SLUG}); }
  catch (e) {
    setBusy(btn, false); lock(false);
    $("actMsg").innerHTML = `<div class="banner bad">${esc(e.message)}</div>`;
    return;
  }
  setBusy(btn, false);
  await after(d, "keep");
}

async function dismiss(label, btn){
  notice = "";
  lock(true); setBusy(btn, true, "Marking…");
  let d;
  try { d = await call(url() + "/dismiss", "POST", {label}); }
  catch (e) {
    setBusy(btn, false); lock(false);
    notice = `<div class="banner bad">${esc(e.message)}</div>`;
    return render();
  }
  view = {...d, help: view.help};           // the help is per version, unchanged
  if (!reviewable(view)){
    render();
    $("main").innerHTML = `<div class="banner ok">Marked harmless. Nothing else is
      waiting on you for ${esc(view.title)}.</div>`;
    return;
  }
  notice = `<div class="banner ok">Marked harmless - it no longer counts as a gap.</div>`;
  render();
}

const DONE = {
  tests: "Prepared again: the tests now catch every wrong answer we tried, and students see the new version.",
  rebuild: "New roadmaps built: the best now reaches the bar, and students get them.",
  keep: "Kept: these roadmaps stay as they are, and this problem is no longer flagged."};

async function after(d, kind = "tests"){
  const wasLive = !!view.ready;
  mine.out = null;
  if (d.ready && wasLive){
    // Done on a live problem: say whether it worked, from the fresh view.
    try { view = await call(url()); } catch (e) { return fail(e); }
    if (!reviewable(view)){
      render();
      $("main").innerHTML = `<div class="banner ok">${DONE[kind]}</div>`;
      return;
    }
    const R = view.roadmaps || {};
    notice = `<div class="banner warn">${kind === "rebuild"
      ? `Built new roadmaps, but the best scores ${R.best} - still under ${R.cutoff}.`
      : view.gaps ? `Prepared again and still ready, but ${view.checks.length}
          wrong ${view.checks.length === 1 ? "answer still passes" : "answers still pass"} - shown above.`
      : "Done - but something else is still waiting, shown above."}</div>`;
    render();
    return explain();
  }
  if (d.ready){
    const next = (view.queue || []).find(p => p.slug !== SLUG);
    $("actMsg").innerHTML = `<div class="banner ok">${esc(view.title)} is ready and
      visible to students.${next ? ` <a href="review.html?a=${encodeURIComponent(A)}&slug=${
        encodeURIComponent(next.slug)}">Review ${esc(next.title)} next &rarr;</a>` : ""}</div>`;
    try { view.queue = (await call(url())).queue; drawQueue(); } catch {}
    return;
  }
  // Still not ready: show the fresh verdict - a new version may have new gaps.
  try { view = await call(url()); } catch (e) { return fail(e); }
  notice = `<div class="banner warn">${esc(d.error || "Still not ready.")}</div>`;
  render();
  return explain();
}

load();
