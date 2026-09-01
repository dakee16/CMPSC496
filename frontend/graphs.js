/* graphs.js — draws the dual-graph artifact.
 *
 * Consumes exactly what main/graphs.py emits: {nodes:[{id,kind,label,line}],
 * edges:[{src,dst,label}], meta:{source}}. One renderer for both graphs, which
 * is the whole reason the plan and the code share a schema — two renderers
 * would drift and the pictures would stop being comparable.
 *
 * Plain SVG, no library. A layered top-down layout is enough for straight-line
 * code with loops and branches, which is all a first-year problem produces.
 */

const G_W = 178, G_H = 46;        // node box
const G_GAPX = 26, G_GAPY = 40;   // space between boxes
const G_PAD = 18;

// Kind → colour role. Every value is a TOKEN, never a literal: the earlier
// version hardcoded #2c3358 / #4a5580 / #8892b8, which meant the flowchart drew
// the same dark greys on the light theme — edge labels came out at 3.07:1 and
// the arrows were nearly invisible. Tokens flip with the theme for free.
const G_STYLE = {
  start:  {fill: "var(--glass-bg-strong)", stroke: "var(--indigo-500)", text: "var(--indigo-400)"},
  end:    {fill: "var(--glass-bg-strong)", stroke: "var(--indigo-500)", text: "var(--indigo-400)"},
  step:   {fill: "var(--card-bg)", stroke: "var(--graph-node-line)", text: "var(--text-dim)"},
  branch: {fill: "var(--card-bg)", stroke: "var(--amber-text)",      text: "var(--amber-text)"},
  loop:   {fill: "var(--card-bg)", stroke: "var(--indigo-500)",      text: "var(--indigo-400)"},
  return: {fill: "var(--card-bg)", stroke: "var(--green-text)",      text: "var(--green-text)"}
};

// Node labels are code on one side and the student's own words on the other.
// Both read better monospaced here: they are short, scanned rather than read,
// and a fixed advance width stops similar labels jittering between rows.
const G_FONT = "'JetBrains Mono','SF Mono',Monaco,monospace";

function gEsc(s){
  return String(s == null ? "" : s).replace(/[&<>"']/g,
    c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}

/* Assign each node a row. Longest-path layering, with two guards against the
 * cycles a loop necessarily creates: edges labelled "repeat" are the back
 * edges and never deepen anything, and the relaxation is capped at one pass
 * per node so an unlabelled cycle terminates instead of hanging the page. */
function gLayer(graph){
  const depth = {};
  graph.nodes.forEach(n => { depth[n.id] = 0; });
  for (let i = 0; i < graph.nodes.length; i++){
    let changed = false;
    for (const e of graph.edges){
      if (e.label === "repeat") continue;
      if (!(e.src in depth) || !(e.dst in depth)) continue;
      if (depth[e.dst] < depth[e.src] + 1){ depth[e.dst] = depth[e.src] + 1; changed = true; }
    }
    if (!changed) break;
  }
  return depth;
}

function gPositions(graph){
  const depth = gLayer(graph), rows = {};
  graph.nodes.forEach(n => { (rows[depth[n.id]] = rows[depth[n.id]] || []).push(n.id); });
  const widest = Math.max(1, ...Object.values(rows).map(r => r.length));
  const pos = {};
  Object.keys(rows).forEach(d => {
    const row = rows[d];
    // Centre each row against the widest one so the drawing reads as a spine.
    const offset = ((widest - row.length) * (G_W + G_GAPX)) / 2;
    row.forEach((id, i) => {
      pos[id] = {x: G_PAD + offset + i * (G_W + G_GAPX),
                 y: G_PAD + Number(d) * (G_H + G_GAPY)};
    });
  });
  return {pos,
          w: G_PAD * 2 + widest * (G_W + G_GAPX) - G_GAPX,
          h: G_PAD * 2 + Object.keys(rows).length * (G_H + G_GAPY) - G_GAPY};
}

function gEdgePath(a, b, back){
  const x1 = a.x + G_W / 2, y1 = a.y + G_H, x2 = b.x + G_W / 2, y2 = b.y;
  if (back){
    // A back edge goes UP. Route it around the right so it never crosses the
    // body of the loop it belongs to.
    const side = Math.max(a.x, b.x) + G_W + 14;
    return `M ${a.x + G_W} ${a.y + G_H / 2} H ${side} V ${b.y + G_H / 2} H ${b.x + G_W}`;
  }
  const mid = (y1 + y2) / 2;
  return `M ${x1} ${y1} V ${mid} H ${x2} V ${y2}`;
}

/* Render one graph into `el`. An empty graph draws a hint rather than a blank
 * box, because "nothing captured yet" is itself information. */
function renderGraph(el, graph, emptyText){
  if (!el) return;
  if (!graph || !graph.nodes || !graph.nodes.length){
    el.innerHTML = `<p class="hint" style="padding:18px 4px">${
      gEsc(emptyText || "Nothing to draw yet.")}</p>`;
    el.removeAttribute("tabindex");
    return;
  }
  // A horizontally scrolling region must be reachable by keyboard, or a reader
  // who cannot use a mouse can never see the right-hand half of the graph.
  el.setAttribute("tabindex", "0");
  el.setAttribute("role", "group");
  el.setAttribute("aria-label", "Flowchart, scrolls horizontally");
  const {pos, w, h} = gPositions(graph);
  // Each graph needs its OWN marker id: two graphs on one page (plan and code)
  // both defining id="ah" makes the second definition win for both, and any
  // future recolouring of one would silently retint the other.
  const uid = "g" + Math.random().toString(36).slice(2, 8);
  const parts = [
    `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img" `,
    `aria-label="flowchart with ${graph.nodes.length} steps" `,
    `font-family="${G_FONT}">`,
    `<defs><marker id="${uid}" markerWidth="9" markerHeight="9" refX="8" refY="3" `,
    `orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="var(--graph-line)"/></marker></defs>`
  ];

  graph.edges.forEach(e => {
    const a = pos[e.src], b = pos[e.dst];
    if (!a || !b) return;                       // defensive: dangling edge
    const back = e.label === "repeat" || b.y <= a.y;
    parts.push(`<path d="${gEdgePath(a, b, back)}" fill="none" `
             + `stroke="var(--graph-line)" stroke-width="1.5" `
             + `stroke-linejoin="round" marker-end="url(#${uid})"${
                 back ? ' stroke-dasharray="4 3"' : ""}/>`);
    if (e.label){
      const lx = back ? Math.max(a.x, b.x) + G_W + 20
                      : (a.x + b.x) / 2 + G_W / 2 + 6;
      const ly = back ? (a.y + b.y) / 2 + G_H / 2 : (a.y + G_H + b.y) / 2;
      // paint-order lets the halo sit BEHIND the glyphs, so a label crossing an
      // edge stays readable without a box that would clutter the drawing.
      parts.push(`<text x="${lx}" y="${ly}" font-size="10.5" font-weight="500" `
               + `fill="var(--graph-label)" stroke="var(--card-bg)" `
               + `stroke-width="3.5" paint-order="stroke">${gEsc(e.label)}</text>`);
    }
  });

  graph.nodes.forEach(n => {
    const p = pos[n.id], s = G_STYLE[n.kind] || G_STYLE.step;
    const r = (n.kind === "start" || n.kind === "end") ? G_H / 2 : 10;
    parts.push(`<g class="gnode"><title>${gEsc(n.label)}</title>`);
    parts.push(`<rect x="${p.x}" y="${p.y}" width="${G_W}" height="${G_H}" `
             + `rx="${r}" fill="${s.fill}" stroke="${s.stroke}" stroke-width="1.5"/>`);
    // A 3px colour bar keeps the kind legible even for a reader who cannot
    // separate the amber and green outlines — colour is never the only cue.
    if (n.kind !== "start" && n.kind !== "end"){
      parts.push(`<rect x="${p.x + 1}" y="${p.y + 9}" width="3" `
               + `height="${G_H - 18}" rx="1.5" fill="${s.stroke}"/>`);
    }
    parts.push(`<text x="${p.x + 13}" y="${p.y + 18}" font-size="8.5" `
             + `font-weight="600" fill="${s.text}" letter-spacing="1.1">${
                 gEsc(n.kind.toUpperCase())}</text>`);
    // Labels are clipped to 60 chars server-side; this second trim is for the
    // pixel width of the box, which a character count does not predict. The
    // full text stays reachable as the <title> tooltip on the group.
    const label = n.label.length > 24 ? n.label.slice(0, 23) + "…" : n.label;
    parts.push(`<text x="${p.x + 13}" y="${p.y + 34}" font-size="11.5" `
             + `fill="var(--text-bright)">${gEsc(label)}</text></g>`);
  });

  parts.push("</svg>");
  el.innerHTML = parts.join("");
}

/* Render the finished artifact: both graphs plus what differs between them.
 * `payload` is exactly the body of POST /graphs. */
function gLegend(){
  // The node colours carry meaning, so they need a key. Each swatch repeats the
  // bar shape used inside the nodes rather than a plain dot, so the mapping is
  // literal instead of remembered.
  return `<ul class="gLegend">` + ["loop", "branch", "return", "step"].map(k =>
    `<li><span class="gSwatch" style="background:${
      (G_STYLE[k] || G_STYLE.step).stroke}"></span>${k}</li>`).join("") + `</ul>`;
}

function renderDual(el, payload){
  if (!el) return;
  const sim = payload.comparison && typeof payload.comparison.similarity === "number"
    ? Math.round(payload.comparison.similarity * 100) : null;
  el.innerHTML =
    `<div class="graphMeta">
       ${gLegend()}
       ${sim === null ? "" : `<span class="pill ${
          sim >= 70 ? "ok" : sim >= 40 ? "warn" : "bad"}">${sim}% structural match</span>`}
     </div>
     <div class="graphPair">
       <div class="graphBox">
         <h3><span class="gTag plan">Plan</span>What you said you would do</h3>
         <div class="graphScroll" id="gPlan"></div></div>
       <div class="graphBox">
         <h3><span class="gTag code">Code</span>What your code does</h3>
         <div class="graphScroll" id="gCode"></div></div>
     </div>
     <div class="graphDiff" id="gDiff"></div>`;

  renderGraph(document.getElementById("gPlan"), payload.plan,
              "No plan was captured from your chat.");
  renderGraph(document.getElementById("gCode"), payload.code,
              "No code submitted yet.");

  const c = payload.comparison || {};
  const rows = [];
  (c.notes || []).forEach(n => rows.push(`<li class="gNote">${gEsc(n)}</li>`));
  (c.plan_only || []).forEach(n => rows.push(
    `<li><span class="gTag plan">Plan only</span>`
    + `<code>${gEsc(n.label)}</code></li>`));
  (c.code_only || []).forEach(n => rows.push(
    `<li><span class="gTag code">Code only</span>`
    + `<code>${gEsc(n.label)}</code></li>`));

  document.getElementById("gDiff").innerHTML =
    `<h3>Where they differ</h3><ul>${rows.join("")}</ul>`
    + `<p class="hint">A difference is not automatically a mistake — you may `
    + `have simplified while writing. It is worth knowing which ones you chose `
    + `and which ones surprised you.</p>`;
}
