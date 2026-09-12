const S = requireSession("teacher");
mountHeader({variant: "instructor", active: "Playground", wide: true});
const $ = id => document.getElementById(id);
const fmt = n => typeof n === "number"
  ? (Number.isInteger(n) ? String(n) : n.toFixed(2)) : String(n);
const j = v => { const s = JSON.stringify(v); return s.length > 120 ? s.slice(0, 119) + "…" : s; };

/* ================= problem list ================= */
let problems = [], selected = null;
const NO_ASSIGNMENT = "No assignment";

// Preparation stages, in pipeline order, as the teacher-facing fix panel names
// them. Mirrors main/publish.PREPARE_STAGES.
const STAGE_WORDS = {parses: "does not parse", runs: "solution fails to run",
                     tests: "no usable tests", strength: "oracle too weak",
                     steps: "could not decompose"};

/* Has live evidence overtaken the stored blocker?

   `prepare_error` is a HISTORICAL record of the last preparation run; the
   oracle and pool badges are read LIVE. A playground run rewrites the oracle
   cache and the decomposition pool but never touches the problems row, so a
   problem you just fixed still reads "oracle too weak" beside "oracle 100%" -
   two badges that look like a contradiction and are really a before and an
   after. Naming the state is the fix; the row itself only clears on a
   re-prepare. */
function staleBlocker(p){
  if (p.ready || !p.stage) return false;
  if (p.stage === "strength" && p.oracle === "strong") return true;
  if (p.stage === "steps" && p.pool_entries > 0) return true;
  return false;
}

async function loadProblems(keepSelection){
  let d;
  try { d = await (await fetch(`${API}/playground/problems`)).json(); }
  catch { $("pList").innerHTML = `<p class="hint">Could not reach the server.</p>`; return; }
  problems = d.problems || [];
  renderList($("pFilter").value.trim().toLowerCase());
  if (keepSelection && selected) {
    const btn = $("pList").querySelector(`[data-slug="${CSS.escape(selected)}"]`);
    if (btn) btn.classList.add("on");
  }
}

function renderList(filter){
  const rows = problems.filter(p => !filter
    || p.title.toLowerCase().includes(filter) || p.slug.includes(filter));
  if (!rows.length){
    $("pList").innerHTML = `<p class="hint" style="margin:8px 4px">Nothing matches.</p>`;
    return;
  }
  // Group by assignment; ungrouped problems land under "No assignment".
  const groups = new Map();
  rows.forEach(p => {
    const g = p.assignment || NO_ASSIGNMENT;
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(p);
  });
  // Real assignments first, alphabetically; the unassigned pile last. Insertion
  // order would otherwise put "No assignment" on top the moment one of the
  // hundred curated problems sorted first - burying the two assignments the
  // instructor actually uploaded under a wall of ones they never touched.
  const ordered = [...groups.entries()].sort(([a], [b]) =>
    (a === NO_ASSIGNMENT) - (b === NO_ASSIGNMENT) || a.localeCompare(b));

  const html = [];
  for (const [name, ps] of ordered){
    // "not ready", not "broken": some of these are fixed and merely waiting on
    // a re-prepare, which is not the same thing as a problem that still fails.
    const unready = ps.filter(p => !p.ready).length;
    html.push(`<div class="pgroup"><span>${esc(name)}</span>
      <span class="gcount${unready ? " bad" : ""}">${
        unready ? `${unready} of ${ps.length} not ready` : `${ps.length}`}</span></div>`);
    ps.forEach(p => {
      const stale = staleBlocker(p);
      const dot = p.ready ? "dot-ok"
                : stale ? "dot-stale"
                : (p.prepare_error ? "dot-bad" : "dot-none");
      const tags = [];
      if (stale)
        tags.push(`<span class="tag warn" title="The blocker below was cleared by a later run. The problems row still says it failed until you re-prepare.">fixed &mdash; re-prepare</span>`);
      else if (!p.ready && p.stage)
        tags.push(`<span class="tag bad">${esc(STAGE_WORDS[p.stage] || p.stage)}</span>`);
      if (p.oracle === "strong")
        tags.push(`<span class="tag ok">oracle ${Math.round((p.kill_rate_direct || 0) * 100)}%</span>`);
      else if (p.oracle === "review")
        tags.push(`<span class="tag warn" title="${
          p.undetermined || 0} mutant${p.undetermined === 1 ? "" : "s"} undetermined. Best case ${
          Math.round((p.kill_rate_upper || 0) * 100)}%, worst case ${
          Math.round((p.kill_rate_lower || 0) * 100)}%.">oracle ${
          Math.round((p.kill_rate_lower || 0) * 100)}-${
          Math.round((p.kill_rate_upper || 0) * 100)}%</span>`);
      else if (p.oracle === "weak")
        tags.push(`<span class="tag warn">oracle weak</span>`);
      else
        tags.push(`<span class="tag dim">no oracle yet</span>`);
      if (p.pool_entries)
        tags.push(`<span class="tag dim">${p.pool_entries} split${p.pool_entries === 1 ? "" : "s"}</span>`);
      if (!p.has_solution)
        tags.push(`<span class="tag bad">no ground truth</span>`);
      html.push(`<button class="prow ${p.slug === selected ? "on" : ""}"
          role="option" data-slug="${esc(p.slug)}">
        <span class="nm"><span class="dot ${dot}"></span>${esc(p.title)}</span>
        <span class="meta">${tags.join("")}</span>
      </button>`);
    });
  }
  $("pList").innerHTML = html.join("");
  $("pList").querySelectorAll(".prow").forEach(b =>
    b.onclick = () => openProblem(b.dataset.slug));
}

$("pFilter").addEventListener("input",
  e => renderList(e.target.value.trim().toLowerCase()));

/* ================= problem detail ================= */
async function openProblem(slug){
  selected = slug;
  $("pList").querySelectorAll(".prow").forEach(b =>
    b.classList.toggle("on", b.dataset.slug === slug));
  $("pEmpty").hidden = true;
  $("pDetail").hidden = false;
  $("runWrap").hidden = true;              // a transcript belongs to ONE problem
  $("runFeed").innerHTML = "";
  let d;
  try { d = await (await fetch(`${API}/playground/problem/${encodeURIComponent(slug)}`)).json(); }
  catch { return; }
  $("dTitle").textContent = d.title;
  $("dSlug").textContent = d.slug;
  $("dDiff").hidden = !d.difficulty;
  $("dDiff").textContent = d.difficulty || "";
  // `stage` is not on the detail payload; recover it from the list row, which
  // is the same data the badges were drawn from. The two endpoints spell the
  // oracle differently - the list sends a string, the detail an object - so
  // this normalises rather than spreading one over the other.
  const listed = problems.find(p => p.slug === slug) || {};
  const stale = staleBlocker({
    ready: d.ready,
    stage: listed.stage,
    oracle: d.oracle ? (d.oracle.strong ? "strong" : "weak") : null,
    pool_entries: d.pool_entries,
  });

  const r = $("dReady");
  r.className = "tag " + (d.ready ? "ok" : stale ? "warn"
                                  : (d.prepare_error ? "bad" : "dim"));
  r.textContent = d.ready ? "ready" : stale ? "fixed - re-prepare"
                          : (d.prepare_error ? "failed" : "not prepared");
  $("dError").hidden = !d.prepare_error || d.ready;
  $("dError").className = stale ? "banner warn failNote" : "banner bad failNote";
  $("dError").innerHTML = stale
    ? `<b>This blocker is out of date.</b> The row still records
       &ldquo;${esc(d.prepare_error)}&rdquo; from its last preparation, but a
       later run has since cleared it${d.oracle && d.oracle.strong
         ? ` &mdash; the oracle is now STRONG at
             ${Math.round(d.oracle.kill_rate_direct * 100)}%` : ""}.
       Re-prepare it from the <a href="teacher.html#list">Assignments</a> page
       to make it available to students.`
    : esc(d.prepare_error || "");
  const o = $("dOracle");
  o.hidden = !d.oracle;
  if (d.oracle) o.textContent = `oracle: ${d.oracle.strong ? "STRONG" : "weak"} · `
    + `${Math.round(d.oracle.kill_rate_direct * 100)}% · ${d.oracle.n_tests} tests`;
  $("dPool").hidden = !d.pool_entries;
  $("dPool").textContent = `${d.pool_entries} pooled split${d.pool_entries === 1 ? "" : "s"}`;
  $("dDesc").textContent = d.description || "(no description)";
  $("dSolution").textContent = d.solution || "(no ground truth stored)";
  $("runBtn").disabled = !d.solution;
}

/* ================= the knobs ================= */
let PARAMS = [];                 // registry from the server
const knobEls = new Map();       // key -> {input, val, wrap, spec}

async function loadParams(){
  let d;
  try { d = await (await fetch(`${API}/playground/params`)).json(); }
  catch { return; }
  PARAMS = d.params || [];
  const html = [];
  let lastGroup = null;
  PARAMS.forEach(p => {
    if (p.group !== lastGroup){
      html.push(`<div class="pgTitle">${esc(p.group)}</div>`);
      lastGroup = p.group;
    }
    html.push(`<div class="knob" data-key="${esc(p.key)}">
      <div class="krow">
        <label for="k-${esc(p.key)}" title="${esc(p.help)}">${esc(p.label)}</label>
        <span class="kval" id="kv-${esc(p.key)}">${fmt(p.default)}</span>
      </div>
      <input type="range" id="k-${esc(p.key)}" min="${p.min}" max="${p.max}"
             step="${p.step}" value="${p.default}"
             aria-describedby="kv-${esc(p.key)}">
    </div>`);
  });
  $("kGroups").innerHTML = html.join("");
  PARAMS.forEach(p => {
    const input = $(`k-${p.key}`), val = $(`kv-${p.key}`);
    const wrap = input.closest(".knob");
    knobEls.set(p.key, {input, val, wrap, spec: p});
    input.addEventListener("input", () => {
      val.textContent = fmt(Number(input.value));
      wrap.classList.toggle("changed", Number(input.value) !== p.default);
      paintChangedCount();
    });
  });
}

function overrides(){
  const out = {};
  knobEls.forEach(({input, spec}, key) => {
    const v = Number(input.value);
    if (v !== spec.default) out[key] = v;
  });
  return out;
}

function paintChangedCount(){
  const n = Object.keys(overrides()).length;
  $("kChanged").hidden = n === 0;
  $("kChanged").textContent = `${n} changed`;
}

$("kReset").onclick = () => {
  knobEls.forEach(({input, val, wrap, spec}) => {
    input.value = spec.default;
    val.textContent = fmt(spec.default);
    wrap.classList.remove("changed");
  });
  paintChangedCount();
};

/* ================= the live run ================= */
let running = false, aborter = null;

/* Stage cards. One card per pipeline stage; events land in the OPEN one. */
let curStage = null, curBody = null, curKonsole = null;
let mutRound = 0, mutRows = new Map(), mutBar = null;

function stageCard(label, name){
  if (curStage) settleStage("ok");
  const card = document.createElement("div");
  card.className = "stageCard";
  card.dataset.name = name;
  card.innerHTML = `<header><span class="sIcon run" aria-hidden="true"></span>
    <h3>${esc(label)}</h3></header><div class="sBody"></div>`;
  $("runFeed").appendChild(card);
  curStage = card;
  curBody = card.querySelector(".sBody");
  curKonsole = null;
  card.scrollIntoView({behavior: "smooth", block: "nearest"});
  return card;
}

function settleStage(state){
  if (!curStage) return;
  const i = curStage.querySelector(".sIcon");
  i.className = "sIcon " + state;
  i.textContent = state === "ok" ? "✓" : "!";
}

function line(html, cls){
  if (!curBody) return null;
  const el = document.createElement("div");
  el.className = "sline" + (cls ? " " + cls : "");
  el.innerHTML = html;
  curBody.appendChild(el);
  return el;
}

function konsole(text){
  if (!curBody) return;
  if (!curKonsole){
    curKonsole = document.createElement("pre");
    curKonsole.className = "konsole";
    curBody.appendChild(curKonsole);
  }
  curKonsole.textContent += text + "\n";
  curKonsole.scrollTop = curKonsole.scrollHeight;
}

/* One expandable row per mutant; repair events append into its log. */
function mutantRow(m){
  const el = document.createElement("details");
  el.className = "mut";
  el.innerHTML = `<summary><span class="mchip pend">…</span>
      <span class="mlabel">${esc(m.label)}</span></summary>
    <div class="mlog"></div>`;
  return el;
}

function setChip(el, status){
  const chip = el.querySelector(".mchip");
  // "undetermined" is the outcome A1 introduced and the one this screen exists
  // to help resolve, so it needs its own word and its own colour. It used to
  // fall through to the grey "pending" chip, which reads as "still running".
  // "proven_equivalent" is gone: nothing is excused by a failed search any
  // more, and the only mutants that ARE provably harmless are never generated.
  const map = {
    killed:            ["killed", "killed"],
    killed_on_retry:   ["retry", "killed on retry"],
    undetermined:      ["undet", "undetermined"],
    unresolved:        ["undet", "undetermined"],   // legacy name, same thing
    survived:          ["retry", "survived"],
  };
  const [cls, text] = map[status] || ["pend", status];
  chip.className = "mchip " + cls;
  chip.textContent = text;
}

function mutLog(index, html){
  const row = mutRows.get(index);
  if (!row) { line(html); return; }
  const el = document.createElement("div");
  el.innerHTML = html;
  row.querySelector(".mlog").appendChild(el);
}

/* ---- the event handlers, one per type the stream can carry ---- */
const HANDLERS = {
  stage(ev){
    if (ev.name === "start"){
      // The run header card; the title is already on screen, so keep it short.
      stageCard(ev.label, "start");
      return;
    }
    if (ev.name === "finished"){
      settleStage("ok");
      curStage = curBody = null;
      const done = document.createElement("div");
      done.className = "verdict strong";
      done.innerHTML = `<b>Run complete</b><span>Every stage finished. The verdict
        and any new tests were persisted, so this run counts as preparation.</span>`;
      $("runFeed").appendChild(done);
      return;
    }
    stageCard(ev.label, ev.name);
  },

  params(ev){
    const chips = Object.entries(ev.values).map(([k, v]) => {
      const spec = PARAMS.find(p => p.key === k);
      const changed = (ev.overridden || []).includes(k);
      return `<span class="tag ${changed ? "acc" : "dim"}"
        title="${esc(spec ? spec.help : "")}">${esc(spec ? spec.label : k)}: ${fmt(v)}</span>`;
    });
    $("runParams").innerHTML = chips.join("");
  },

  ground_truth(){ /* already shown in the problem panel */ },

  entry(ev){
    line(`Entry point: <b><code>${esc(ev.entry_name || "?")}(${
      (ev.params || []).map(esc).join(", ")})</code></b>`);
  },

  oracle_tests(ev){
    const t = ev.tests || [];
    line(`<b>${t.length}</b> usable test${t.length === 1 ? "" : "s"} kept after
      filtering ambiguous inputs. These are the oracles every later stage
      judges against:`);
    const rows = t.map((x, i) => `<tr><td>${i + 1}</td>
      <td>${esc(j(x.input))}</td><td>${esc(j(x.expected))}</td></tr>`).join("");
    const wrap = document.createElement("div");
    wrap.className = "otableWrap";
    wrap.innerHTML = `<table class="otable"><thead><tr><th>#</th>
      <th>input</th><th>expected (from ground truth)</th></tr></thead>
      <tbody>${rows}</tbody></table>`;
    curBody.appendChild(wrap);
  },

  round_start(ev){
    mutRound = ev.round;
    line(`<b>Round ${ev.round} of ${ev.max_rounds}</b> - scoring the suite as it
      stands (${ev.n_tests} tests).`);
  },

  mutants(ev){
    mutRows = new Map();
    line(`<b>${ev.total}</b> mutants - single mechanical edits of the ground
      truth. Each is run against every oracle test; a mutant no test can tell
      from the real solution is a hole in the suite.`);
    const bar = document.createElement("div");
    bar.className = "mutbar";
    bar.innerHTML = `<div class="bar"><i></i></div><span class="cnt">0 / ${ev.total}</span>`;
    curBody.appendChild(bar);
    mutBar = {el: bar, total: ev.total};
    const list = document.createElement("div");
    list.className = "mutlist";
    (ev.mutants || []).forEach(m => {
      const row = mutantRow(m);
      mutRows.set(m.index, row);
      list.appendChild(row);
    });
    curBody.appendChild(list);
  },

  mutant_result(ev){
    const row = mutRows.get(ev.index);
    if (!row) return;
    if (ev.killed){
      setChip(row, "killed");
      const by = (ev.per_test || []).filter(t => !t.pass).length;
      mutLog(ev.index, ev.crashed
        ? `Crashed or hung where the original ran - killed. <code>${esc(ev.error || "")}</code>`
        : `Killed in phase 1: disagreed with the expected output on <b>${by}</b> test${by === 1 ? "" : "s"}.`);
    } else {
      setChip(row, "survived");
      mutLog(ev.index, `Survived phase 1 - every oracle test agreed with this broken version.`);
    }
  },

  tally(ev){
    if (!mutBar) return;
    mutBar.el.querySelector("i").style.width = (100 * ev.processed / ev.total) + "%";
    mutBar.el.querySelector(".cnt").textContent =
      `${ev.processed} / ${ev.total} checked · ${ev.killed_so_far} killed`;
  },

  phase1_done(ev){
    line(`Suite as handed in killed <b>${ev.killed_direct} / ${ev.total}</b>.`
      + (ev.survivors.length
        ? ` <b>${ev.survivors.length}</b> survivor${ev.survivors.length === 1 ? "" : "s"} go to repair.`
        : ` No survivors.`), ev.survivors.length ? "" : "good");
  },

  retry_start(ev){ mutLog(ev.index, `- Repair attempt: ${esc(ev.detail || "")}`); },
  probes(ev){
    mutLog(ev.index, `Trying <b>${(ev.inputs || []).length}</b> free boundary
      probes (deterministic, no model cost).`);
  },
  // B2 - the free sweep now runs BEFORE any model call, so the transcript has
  // to show it in that position or the cost story reads backwards.
  sweep(ev){
    mutLog(ev.index, `Sweeping <b>${ev.n}</b> generated inputs - deterministic,
      seeded, and still free. The model is only asked once these come up empty.`);
  },
  probes_exhausted(ev){ mutLog(ev.index, esc(ev.detail || "No probe separated them - asking the model.")); },

  // A5 tier 2. Asking for the divergence POINT rather than for inputs is the
  // change that killed the reverse_integer overflow bug, so it is worth
  // narrating: the model supplies a target, the arithmetic happens locally.
  divergence_point(ev){
    mutLog(ev.index, `Tier 2 - asked where the two programs must diverge rather
      than for an input. Model says <b>${esc(String(ev.variable))}</b> must reach
      <b>${esc((ev.values || []).join(", "))}</b>.`);
  },
  inverted_candidates(ev){
    mutLog(ev.index, `Working backwards from ${esc(String(ev.variable))} =
      <b>${esc(String(ev.target))}</b> (and its neighbours) to inputs whose
      computation could land there:
      <code>${esc(JSON.stringify(ev.inputs || []))}</code>`);
  },
  llm_asking(ev){ mutLog(ev.index, esc(ev.detail || "Asking the model for distinguishing inputs…")); },
  llm_candidates(ev){
    mutLog(ev.index, `Model proposed: <code>${esc(j(ev.inputs || []))}</code>
      - inputs only; execution decides what they prove.`);
  },
  disagreement(ev){
    mutLog(ev.index, `Found a real disagreement (${esc(ev.source)} input):
      <code>${esc(j(ev.input))}</code> → ground truth says
      <code>${esc(j(ev.expected))}</code>, the mutant does not.`);
  },
  killed_by_earlier_counterexample(ev){ mutLog(ev.index, esc(ev.detail || "")); },
  oracle_grown(ev){
    mutLog(ev.index, `Added as a new oracle test - suite is now
      <b>${ev.n_tests}</b> tests.`);
  },
  search_empty(ev){ mutLog(ev.index, esc(ev.detail || "")); },
  search_error(ev){ mutLog(ev.index, `Search failed: <code>${esc(ev.error || "")}</code> - left undetermined.`); },
  // equivalence_sweep_start / equivalence_sweep_result were removed with A1.
  // Nothing emits them any more, and their text stated the rule that was
  // deleted - that total agreement across the sweep "excuses a survivor". The
  // sweep still runs, earlier and for free, but only as a SEARCH: see sweep()
  // above. Leaving dead handlers that document a repealed rule is how the old
  // rule comes back.
  // A2a. The backend was already emitting this and the page was dropping it on
  // the floor - which left "undetermined" with no explanation at all, on the
  // one screen whose job is to explain it. The three verdicts need opposite
  // responses, so the response is spelled out rather than left implied.
  probe_result(ev){
    const WHAT = {
      never_reached: ["NEVER REACHED",
        "No test ever runs this line, so the edit cannot matter. This is a gap "
        + "in the tests, not an equivalence question - a test that reaches here "
        + "would settle it."],
      no_infection: ["NO INFECTION",
        "The line runs, but the edit never changed the decision it makes. More "
        + "tests of the same kind cannot separate these two programs."],
      propagation: ["PROPAGATION",
        "The edit really does change the decision, but the difference is "
        + "swallowed before it reaches the output. Separating them needs a "
        + "structurally different test, not more of the same."],
      unknown: ["UNKNOWN",
        "The probe itself could not run, so nothing was learned either way."],
    };
    const [name, why] = WHAT[ev.verdict] || [String(ev.verdict || "?").toUpperCase(),
                                             ev.detail || ""];
    mutLog(ev.index, `<div class="probe"><b>${esc(name)}</b>
      <span class="pnum">· reached ${ev.reached ?? 0} ×, differed ${ev.differed ?? 0} ×</span>
      <div>${esc(why)}</div></div>`);
  },

  mutant_final(ev){
    const row = mutRows.get(ev.index);
    if (row) setChip(row, ev.status);
    if (ev.detail) mutLog(ev.index, esc(ev.detail));
  },

  round_summary(ev){
    const pct = x => Math.round(x * 100) + "%";
    line(`Round ${ev.round}: kill rate as handed in <b>${pct(ev.kill_rate_direct)}</b>
      (needs ${pct(ev.cutoff)}) · after repair ${pct(ev.kill_rate)} ·
      ${ev.n_tests} tests · <b>${esc(ev.status.toUpperCase())}</b>`,
      ev.status === "strong" ? "good" : "");
  },
  // B3. The loop stops on evidence about the mutants, not on a stalled score,
  // and saying which is the difference between "we gave up" and "more tests
  // provably cannot help here".
  stopping_early(ev){
    line(`<b>Stopping early.</b> ${esc(ev.detail || "")}`, "warn");
  },
  expanding_suite(ev){ line(esc(ev.detail || "Still weak - expanding the suite.")); },
  suite_expanded(ev){
    line(`Added <b>${ev.added}</b> fresh ground-truth-verified test${ev.added === 1 ? "" : "s"}
      (suite now ${ev.n_tests}) - rescoring.`);
  },
  expansion_empty(ev){ line(esc(ev.detail || "No new distinct tests could be generated."), "bad"); },
  base_run_failed(ev){ line(`Ground truth failed on the suite: <code>${esc(ev.error || "")}</code>`, "bad"); },
  evaluation_done(){ /* the verdict event right after carries the story */ },

  // A4. Three outcomes, not two. `strong` is a single bit and a needs_review
  // problem also has strong=false - rendering that as "ORACLE WEAK" tells the
  // instructor their problem is broken when it may need no change at all, on
  // the exact screen the upload page's "Review now" button sends them to.
  verdict(ev){
    const pct = x => Math.round(x * 100) + "%";
    const review = ev.needs_review || ev.status === "needs_review";
    const cls   = ev.strong ? "strong" : review ? "review" : "weak";
    const title = ev.strong ? "ORACLE STRONG"
                : review    ? "NEEDS YOUR REVIEW" : "ORACLE WEAK";

    // The range is the honest number whenever anything was left undetermined:
    // the low end assumes every one is a real bug, the high end assumes none is.
    const scored = (ev.kill_rate_lower !== undefined
                    && ev.kill_rate_upper !== undefined
                    && ev.kill_rate_lower !== ev.kill_rate_upper);
    const rate = scored
      ? `between <b>${pct(ev.kill_rate_lower)}</b> and <b>${pct(ev.kill_rate_upper)}</b>
         of mutants killed`
      : `${pct(ev.kill_rate_direct)} of mutants killed as handed in`;

    const why = review
      ? ` - ${ev.undetermined} mutant${ev.undetermined === 1 ? "" : "s"} could not
          be judged either way, and the verdict genuinely hangs on ${
          ev.undetermined === 1 ? "it" : "them"}. Open the mutants above: a
          <b>never reached</b> finding means a missing test, while
          <b>no infection</b> means more tests of the same kind cannot help.`
      : ev.strong && scored
        ? ` - the worst case already clears the cutoff, so the undetermined
            ${ev.undetermined === 1 ? "mutant" : "mutants"} cannot change the answer.`
        : "";

    const v = document.createElement("div");
    v.className = "verdict " + cls;
    v.innerHTML = `<b>${title}</b>
      <span>${rate} (cutoff ${pct(ev.cutoff)}) · ${pct(ev.kill_rate)} after repair ·
      ${ev.n_tests} tests over ${ev.rounds} round${ev.rounds === 1 ? "" : "s"}${
      ev.insufficient_mutants
        ? " · too few mutants to judge - never STRONG" : ""}.${why} Verdict persisted.</span>`;
    curBody.appendChild(v);
  },

  chunks(ev){
    line(`Decomposition accepted: <b>${ev.chunks.length}</b> chunks under
      <code>${esc(ev.header)}</code>. Each reference is the private answer to
      its step:`);
    (ev.chunks || []).forEach((c, i) => {
      const card = document.createElement("div");
      card.className = "chunkCard";
      card.innerHTML = `<p class="cp"><span class="tag acc">${esc(c.step_id || "Part " + (i + 1))}</span>
        ${esc(c.prompt)}</p><pre class="code">${esc(c.reference || "")}</pre>`;
      curBody.appendChild(card);
    });
  },

  necessity(ev){
    (ev.per_chunk || []).forEach(c => {
      line(`<span class="necline">${c.necessary
        ? `<span class="tag ok">load-bearing</span>`
        : `<span class="tag bad">not needed</span>`}
        <b>${esc(c.step_id)}</b> - knocked out, the assembly
        ${c.necessary ? "broke (" : "STILL PASSED ("}${esc(c.outcome || "")})</span>`);
    });
    line(ev.status === "pass"
      ? `Gate 1 <b>passed</b> - every chunk proved necessary.`
      : `Gate 1: <b>${esc(ev.status)}</b> - ${esc((ev.summary || "").split("\n")[0])}`,
      ev.status === "pass" ? "good" : "bad");
  },

  blocked(ev){
    settleStage("bad");
    const b = document.createElement("div");
    b.className = "blockedCard";
    b.innerHTML = `<b>Blocked at ${esc(ev.at)} - ${esc(ev.error_type)}</b>
      ${esc(ev.message)}`;
    (curBody || $("runFeed")).appendChild(b);
    curStage = curBody = null;
  },

  error(ev){
    settleStage("bad");
    const b = document.createElement("div");
    b.className = "blockedCard";
    b.innerHTML = `<b>Run error</b>${esc(ev.message || "")}`;
    (curBody || $("runFeed")).appendChild(b);
  },

  log(ev){ konsole(ev.text); },
  done(){ /* handled by the reader loop */ },
};

/* Reset the transcript panel for a new run. Shared by the two ways to watch
   one: starting a run here, and mirroring a run an upload is already doing. */
function beginRun(){
  running = true;
  aborter = new AbortController();
  $("runBtn").disabled = true;
  $("stopBtn").hidden = false;
  $("runWrap").hidden = false;
  $("runFeed").innerHTML = "";
  $("runParams").innerHTML = "";
  $("rawLog").textContent = "";
  $("runState").textContent = "running…";
  curStage = curBody = curKonsole = null;
  mutRows = new Map(); mutBar = null; mutRound = 0;
  $("runWrap").scrollIntoView({behavior: "smooth", block: "start"});
}

/* Read one NDJSON pipeline stream to its end and render it. The event
   vocabulary is main/live_playground.py's, and the upload path emits the same
   one (main/publish.py), so this renders either without knowing which it got.
   Blank lines are heartbeats from a mirrored run - skipped, same as any other
   empty line. Returns true if the run reported a failure. */
async function consumeRun(resp){
  const reader = resp.body.getReader();
  const dec = new TextDecoder();
  let buf = "", sawError = false;
  try {
    while (true){
      const {done, value} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0){
        const lineTxt = buf.slice(0, nl); buf = buf.slice(nl + 1);
        if (!lineTxt.trim()) continue;
        let ev;
        try { ev = JSON.parse(lineTxt); } catch { continue; }
        $("rawLog").textContent += lineTxt + "\n";
        const h = HANDLERS[ev.type];
        if (h) h(ev);
        else konsole(`[${ev.type}] ` + lineTxt);   // future event types stay visible
        if (ev.type === "blocked" || ev.type === "error") sawError = true;
      }
    }
  } catch (e) {
    if (e.name !== "AbortError") HANDLERS.error({message: String(e)});
    else konsole("- stopped watching; the run continues on the server -");
  }
  return sawError;
}

async function streamFailed(resp){
  let msg = `HTTP ${resp.status}`;
  try { const d = await resp.json(); msg = (d.detail && d.detail.message) || d.detail || msg; } catch {}
  HANDLERS.error({message: String(msg)});
  finishRun("failed");
}

async function runLive(){
  if (running || !selected) return;
  beginRun();

  let resp;
  try {
    resp = await fetch(`${API}/playground/live`, {
      method: "POST", headers: {"Content-Type": "application/json"},
      signal: aborter.signal,
      body: JSON.stringify({slug: selected, params: overrides()}),
    });
  } catch { finishRun("could not reach the server"); return; }
  if (!resp.ok) return streamFailed(resp);

  const sawError = await consumeRun(resp);
  finishRun(sawError ? "finished with a failure" : "finished");
  // The run wrote to the oracle cache and possibly the pool - refresh both
  // panels so the list's badges tell the new truth.
  loadProblems(true);
  if (selected) openProblemStatusOnly(selected);
}

/* MIRROR MODE. Attach to the preparation run an upload is already doing for
   one problem, rather than starting a second one - see main/prepare_bus.py.
   Nothing is started, nothing is paid for twice, and Stop only stops WATCHING.

   A run that has already finished replays its whole transcript and ends
   immediately, so a teacher can click a row after the fact and read back
   exactly what happened to that problem. */
async function watchPrepare(assignmentId, slug){
  if (running) return;
  beginRun();
  $("runBtn").hidden = true;              // there is nothing to start here
  konsole(`- mirroring the upload's preparation of "${slug}" -`);

  let resp;
  try {
    resp = await fetch(
      `${API}/teacher/prepare/live/${encodeURIComponent(assignmentId)}/${encodeURIComponent(slug)}`,
      {signal: aborter.signal});
  } catch { finishRun("could not reach the server"); return; }
  if (!resp.ok) return streamFailed(resp);

  const sawError = await consumeRun(resp);
  finishRun(sawError ? "finished with a failure" : "finished");
}

function finishRun(label){
  running = false;
  $("runBtn").disabled = false;
  $("stopBtn").hidden = true;
  $("runState").textContent = label;
  if (curStage) settleStage(label.includes("fail") ? "bad" : "ok");
}

/* Refresh only the status chips after a run, without clobbering the transcript
   the way a full openProblem() would. */
async function openProblemStatusOnly(slug){
  let d;
  try { d = await (await fetch(`${API}/playground/problem/${encodeURIComponent(slug)}`)).json(); }
  catch { return; }
  const o = $("dOracle");
  o.hidden = !d.oracle;
  if (d.oracle) o.textContent = `oracle: ${d.oracle.strong ? "STRONG" : "weak"} · `
    + `${Math.round(d.oracle.kill_rate_direct * 100)}% · ${d.oracle.n_tests} tests`;
  $("dPool").hidden = !d.pool_entries;
  $("dPool").textContent = `${d.pool_entries} pooled split${d.pool_entries === 1 ? "" : "s"}`;
}

$("runBtn").onclick = runLive;
$("stopBtn").onclick = () => { if (aborter) aborter.abort(); };
$("rawBtn").onclick = () => {
  const showing = $("rawLog").hidden;
  $("rawLog").hidden = !showing;
  $("rawBtn").setAttribute("aria-expanded", String(showing));
};

// ?slug=... opens straight onto one problem. This is what the upload page's
// "Review now" links to, so an instructor lands on the problem they were just
// told about instead of hunting for it in a list of twenty.
//
// ?watch=<assignment_id>&slug=... additionally MIRRORS the preparation run that
// upload is doing for that problem. The upload page links here while a problem
// is still working, so "why is this one taking so long" is one click away.
const qs = new URLSearchParams(location.search);
const watchId = qs.get("watch"), watchSlug = qs.get("slug");
if (watchId && watchSlug){
  // No problem list in mirror mode: the run being watched is the whole page,
  // and loading twenty problems only to ignore them is a slow way to show it.
  document.title = `Preparing ${watchSlug}`;
  watchPrepare(watchId, watchSlug);
} else {
  loadProblems().then(() => {
    if (watchSlug && problems.some(p => p.slug === watchSlug)) openProblem(watchSlug);
  });
}
loadParams();
