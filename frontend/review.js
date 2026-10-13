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
  // The plain words are written ONCE per problem version and remembered, so
  // this costs nothing on a second visit - the server answers from memory.
  if (view.in_review && view.checks.length && !view.help){
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

function render(){
  const v = view;
  document.title = `Review: ${v.title} · ACADIA`;
  $("title").textContent = `Review: ${v.title}`;
  $("eyebrow").textContent = `${v.assignment.name || "Assignment"} · `
    + (v.in_review ? "Needs your review" : v.ready ? "Ready" : "Not ready");
  $("sub").textContent = v.in_review
    ? "Students can't see this problem until you decide. Nothing here changes "
      + "your solution unless you choose to."
    : v.ready ? "Students can see this problem." : "";
  setCrumbs([{label: "Assignments", href: "teacher-assignments.html"},
             {label: v.assignment.name || "Assignment", href: "teacher-assignments.html"},
             {label: `Review: ${v.title}`}]);
  drawQueue();

  if (!v.in_review){
    $("main").innerHTML = v.ready
      ? `<div class="banner ok">${esc(v.title)} is ready and visible to students.
          Nothing is waiting on you here.</div>`
      : `<div class="banner warn">${esc(v.title)} is not waiting on review - it
          stopped for another reason: ${esc(v.error || "unknown")}. Fix it from
          the Assignments page.</div>`;
    return;
  }
  const missed = v.checks.length, total = v.total || missed;
  const h = v.help;
  const gaps = v.checks.filter(c => c.verdict === "gap").length;
  const maybe = v.checks.filter(c => c.verdict === "maybe_harmless").length;
  $("main").innerHTML = `
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
          : gaps ? ` - we proved ${gaps} of them matter, and found no case where
              the other ${missed - gaps} do.`
          : " - but we found no case where any of them changes an answer.") : "."}</p>
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
      <div id="actMsg" aria-live="polite"></div>
    </section>

    <section class="card solution">
      <h2 style="margin:0 0 12px">Your solution</h2>
      <pre id="sol"></pre>
      <p class="note">Highlighted: the lines the missed changes are on.</p>
    </section>`;

  const lines = new Set(v.lines || []);
  $("sol").innerHTML = (v.source || "").split("\n").map((c, i) =>
    `<div class="srow${lines.has(i + 1) ? " hit" : ""}"><span class="n">${i + 1}</span><span
      class="c">${c ? esc(c) : " "}</span></div>`).join("");
  // Plain text, not hlPython: it highlights one line at a time here, and a
  // line inside a multi-line docstring is not valid Python on its own - it
  // broke the row it sat in.
  if (h && h.doctest && $("ex")) $("ex").innerHTML = hlPython(h.doctest);
  wire();
}

/* `x` in the engine's wording is code: shown as code, never as backticks. */
const said = t => esc(t).replace(/`([^`]+)`/g, "<code>$1</code>");

function finding(c){
  const pill = {
    gap: `<span class="pill warn">real gap - we found a case where it matters</span>`,
    maybe_harmless: `<span class="pill">no case found - may be harmless</span>`,
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
  </article>`;
}

function options(gaps, maybe){
  const h = view.help;
  const canAdd = !!(h && h.edited_source);
  const add = canAdd ? `
    <div class="option rec">
      <h4>Add examples that cover ${gaps === view.checks.length ? "these" : `the ${gaps} real ${gaps === 1 ? "gap" : "gaps"}`}
        <span class="pill ok">recommended</span></h4>
      <p>We ran these against your solution before showing them to you - every
        answer below is your code's. They go into the docstring, so students see
        them too, and they become tests.</p>
      <pre class="code ex" id="ex"></pre>
      <ul class="checks">
        <li>Your solution runs every line</li>
        <li>${h.catches} of the ${view.checks.length} missed ${view.checks.length === 1 ? "change" : "changes"} now fail${h.catches === 1 ? "s" : ""} at least one line</li>
      </ul>
      <div class="acts">
        <button type="button" id="addGo">Add to the docstring and prepare again</button>
        <button type="button" class="ghost" id="addEdit">Edit them first</button>
        <span class="cost">Prepares just this problem again - a minute or two.</span>
      </div>
    </div>` : "";
  const why = canAdd
    ? `Here, we found cases where they do differ, so we don't recommend it.`
    : h && h.examples_ok
      ? `We tried examples and found no case where these changes give a
         different answer - often they touch something no test can see, like a
         node that has already been removed. That makes this a reasonable choice.`
      : `Only if you are sure.`;
  return `${add}
    <div class="option${!canAdd && h && h.examples_ok ? " rec" : ""}">
      <h4>These can't change the answer - publish anyway${!canAdd && h &&
        h.examples_ok ? ` <span class="pill ok">reasonable here</span>` : ""}</h4>
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
}

function openEditor(text){
  $("editor").innerHTML = `<textarea class="rvEdit" id="edSrc" spellcheck="false"
    aria-label="Problem source"></textarea>
    <div class="acts"><button type="button" id="edGo">Prepare again</button>
      <span class="cost">Your edits stay here if anything goes wrong.</span></div>`;
  $("edSrc").value = text || "";
  $("edGo").onclick = () => prepare($("edSrc").value, $("edGo"));
  $("edSrc").focus();
  $("editor").scrollIntoView({block: "nearest"});
}

function lock(on){
  document.querySelectorAll("#main button").forEach(b => { if (!b.dataset.busy) b.disabled = on; });
}

async function prepare(source, btn){
  if (!source || !source.trim()) return;
  lock(true); setBusy(btn, true, "Preparing…");
  $("actMsg").innerHTML = `<div class="banner info">Preparing this problem again:
    writing tests, checking they are strong enough, and splitting it into steps.
    This takes a minute or two - you can leave this page open.</div>`;
  let d;
  try { d = await call("/teacher/problems/retry", "POST",
                       {assignment_id: A, slug: SLUG, source}); }
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

async function after(d){
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
  render();
  $("actMsg").innerHTML = `<div class="banner warn">${esc(d.error ||
    "Still not ready.")}</div>`;
  if (view.in_review && !view.help) load();
}

load();
