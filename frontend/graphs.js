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

// Kind → colour role. Matches the midnight-glass tokens in ui.css so the
// graphs look native rather than bolted on.
const G_STYLE = {
  start:  {fill: "var(--glass-bg-strong)", stroke: "var(--indigo-600)", text: "var(--indigo-600)"},
  end:    {fill: "var(--glass-bg-strong)", stroke: "var(--indigo-600)", text: "var(--indigo-600)"},
  step:   {fill: "var(--card-bg)",         stroke: "#2c3358",           text: "inherit"},
  branch: {fill: "var(--card-bg)",         stroke: "var(--amber-text)",  text: "var(--amber-text)"},
  loop:   {fill: "var(--card-bg)",         stroke: "var(--indigo-600)",  text: "var(--indigo-600)"},
  return: {fill: "var(--card-bg)",         stroke: "var(--green-text)",  text: "var(--green-text)"}
};

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
    return;
  }
  const {pos, w, h} = gPositions(graph);
  const parts = [
    `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img" `,
    `aria-label="flowchart with ${graph.nodes.length} steps">`,
    `<defs><marker id="ah" markerWidth="9" markerHeight="9" refX="8" refY="3" `,
    `orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#4a5580"/></marker></defs>`
  ];

  graph.edges.forEach(e => {
    const a = pos[e.src], b = pos[e.dst];
    if (!a || !b) return;                       // defensive: dangling edge
    const back = e.label === "repeat" || b.y <= a.y;
    parts.push(`<path d="${gEdgePath(a, b, back)}" fill="none" stroke="#4a5580" `
             + `stroke-width="1.5" marker-end="url(#ah)"${
                 back ? ' stroke-dasharray="4 3"' : ""}/>`);
    if (e.label){
      const lx = back ? Math.max(a.x, b.x) + G_W + 20
                      : (a.x + b.x) / 2 + G_W / 2 + 6;
      const ly = back ? (a.y + b.y) / 2 + G_H / 2 : (a.y + G_H + b.y) / 2;
      parts.push(`<text x="${lx}" y="${ly}" font-size="11" fill="#8892b8">${
        gEsc(e.label)}</text>`);
    }
  });

  graph.nodes.forEach(n => {
    const p = pos[n.id], s = G_STYLE[n.kind] || G_STYLE.step;
    const r = (n.kind === "start" || n.kind === "end") ? G_H / 2 : 9;
    parts.push(`<rect x="${p.x}" y="${p.y}" width="${G_W}" height="${G_H}" `
             + `rx="${r}" fill="${s.fill}" stroke="${s.stroke}" stroke-width="1.5"/>`);
    parts.push(`<text x="${p.x + 11}" y="${p.y + 18}" font-size="9.5" `
             + `fill="${s.text}" opacity=".75" letter-spacing=".6">${
                 gEsc(n.kind.toUpperCase())}</text>`);
    // Labels are clipped to 60 chars server-side; this second trim is for the
    // pixel width of the box, which a character count does not predict. The
    // full text stays reachable as a tooltip.
    const label = n.label.length > 26 ? n.label.slice(0, 25) + "…" : n.label;
    parts.push(`<text x="${p.x + 11}" y="${p.y + 34}" font-size="12.5" `
             + `fill="currentColor"><title>${gEsc(n.label)}</title>${
                 gEsc(label)}</text>`);
  });

  parts.push("</svg>");
  el.innerHTML = parts.join("");
}

/* Render the finished artifact: both graphs plus what differs between them.
 * `payload` is exactly the body of POST /graphs. */
function renderDual(el, payload){
  if (!el) return;
  el.innerHTML =
    `<div class="graphPair">
       <div class="graphBox"><h3>What you said you would do</h3>
         <div class="graphScroll" id="gPlan"></div></div>
       <div class="graphBox"><h3>What your code does</h3>
         <div class="graphScroll" id="gCode"></div></div>
     </div>
     <div class="graphDiff" id="gDiff"></div>`;

  renderGraph(document.getElementById("gPlan"), payload.plan,
              "No plan was captured from your chat.");
  renderGraph(document.getElementById("gCode"), payload.code,
              "No code submitted yet.");

  const c = payload.comparison || {};
  const rows = [];
  (c.notes || []).forEach(n => rows.push(`<li>${gEsc(n)}</li>`));
  (c.plan_only || []).forEach(n => rows.push(
    `<li><b>In your plan but not your code:</b> ${gEsc(n.label)} `
    + `<span class="pill">${gEsc(n.kind)}</span></li>`));
  (c.code_only || []).forEach(n => rows.push(
    `<li><b>In your code but not your plan:</b> ${gEsc(n.label)} `
    + `<span class="pill">${gEsc(n.kind)}</span></li>`));

  document.getElementById("gDiff").innerHTML =
    `<h3>Where they differ</h3><ul>${rows.join("")}</ul>`
    + `<p class="hint">A difference is not automatically a mistake — you may `
    + `have simplified while writing. It is worth knowing which ones you chose `
    + `and which ones surprised you.</p>`;
}
