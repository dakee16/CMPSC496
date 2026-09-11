/* graphs.js - draws the dual-graph artifact.
 *
 * Consumes exactly what main/graphs.py emits: {nodes:[{id,kind,label,line}],
 * edges:[{src,dst,label}], meta:{source}}. One renderer for both graphs, which
 * is the whole reason the plan and the code share a schema - two renderers
 * would drift and the pictures would stop being comparable.
 *
 * Plain SVG, no library. A layered top-down layout is enough for straight-line
 * code with loops and branches, which is all a first-year problem produces.
 *
 * Nodes SIZE TO THEIR CONTENT. The old renderer used a fixed 178px box and
 * hard-truncated the label to 23 characters, so "Take absolute value of n" was
 * clipped mid-word - the most visible defect on that screen. Labels now wrap to
 * two lines and the box widens to fit the longest line in the graph.
 */

const G_GAPX = 26, G_GAPY = 40;   // space between boxes
const G_PAD = 18;                 // margin inside the drawing
const G_LINE = 15;                // label line height
const G_CHAR = 7.25;              // JetBrains Mono advance width at 12px
const G_MAXCH = 24;               // characters per label line before wrapping
const G_MINW = 150, G_MAXW = 280; // node width bounds

// Kind → colour role. Every value is a TOKEN, never a literal: the earlier
// version hardcoded #2c3358 / #4a5580 / #8892b8, which meant the flowchart drew
// the same dark greys on the light theme - edge labels came out at 3.07:1 and
// the arrows were nearly invisible. Tokens flip with the theme for free.
const G_STYLE = {
  start:  {fill: "var(--glass-bg-strong)", stroke: "var(--accent-ink)", text: "var(--accent-ink)"},
  end:    {fill: "var(--glass-bg-strong)", stroke: "var(--accent-ink)", text: "var(--accent-ink)"},
  step:   {fill: "var(--surface)", stroke: "var(--graph-node-line)", text: "var(--text-subtle)"},
  branch: {fill: "var(--surface)", stroke: "var(--warning)",         text: "var(--warning)"},
  loop:   {fill: "var(--surface)", stroke: "var(--accent-ink)",    text: "var(--accent-ink)"},
  return: {fill: "var(--surface)", stroke: "var(--success)",         text: "var(--success)"}
};

// Node labels are code on one side and the student's own words on the other.
// Both read better monospaced here: they are short, scanned rather than read,
// and a fixed advance width stops similar labels jittering between rows - and
// it is what lets the character-count wrap below predict the pixel width.
const G_FONT = "'JetBrains Mono','SF Mono',Monaco,monospace";

function gEsc(s){
  return String(s == null ? "" : s).replace(/[&<>"']/g,
    c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}

/* Wrap a label to at most two lines of at most G_MAXCH characters.
 * A word longer than one line is hard-split rather than allowed to overflow.
 * Anything past two lines ends in an ellipsis; the full text stays reachable
 * through the <title> tooltip on the node. */
function gWrapLabel(label){
  const text = String(label || "").trim().replace(/\s+/g, " ");
  const words = text ? text.split(" ") : [];
  const lines = [];
  let cur = "", dropped = false;

  for (let i = 0; i < words.length; i++){
    let w = words[i];
    // A single word wider than the box: break it rather than let it overflow.
    while (w.length > G_MAXCH){
      if (cur){ lines.push(cur); cur = ""; }
      if (lines.length >= 2){ dropped = true; break; }
      lines.push(w.slice(0, G_MAXCH - 1) + "-");
      w = w.slice(G_MAXCH - 1);
    }
    if (dropped) break;
    const next = cur ? cur + " " + w : w;
    if (next.length <= G_MAXCH){ cur = next; continue; }
    lines.push(cur);
    if (lines.length >= 2){ dropped = true; cur = ""; break; }
    cur = w;
  }
  if (cur && lines.length < 2) lines.push(cur);
  else if (cur) dropped = true;
  if (!lines.length) lines.push("");

  if (dropped){
    const last = lines.length - 1;
    lines[last] = lines[last].slice(0, G_MAXCH - 1).replace(/[\s-]+$/, "") + "…";
  }
  return lines;
}

/* One box size for the whole graph, wide enough for its longest label line.
 * Uniform rather than per-node so the layered grid below stays simple and the
 * rows still line up. */
function gMetrics(graph){
  const lines = {};
  let widest = 0, tallest = 1;
  graph.nodes.forEach(n => {
    const ls = gWrapLabel(n.label);
    lines[n.id] = ls;
    tallest = Math.max(tallest, ls.length);
    ls.forEach(l => { widest = Math.max(widest, l.length); });
    // The kind eyebrow ("BRANCH") also has to fit, at 8.5px with wide tracking.
    widest = Math.max(widest, String(n.kind || "step").length * 0.95);
  });
  const W = Math.round(Math.min(G_MAXW, Math.max(G_MINW, widest * G_CHAR + 28)));
  const H = tallest > 1 ? 62 : 46;
  return {lines, W, H};
}

/* Assign each node a row. Longest-path layering, with two guards against the
 * cycles a loop necessarily creates: edges labelled "repeat" are the back
 * edges and never deepen anything, and the relaxation is capped at one pass
 * per node so an unlabelled cycle terminates instead of hanging the page. */
function gLayer(graph){
  const depth = {};
  graph.nodes.forEach(n => { depth[n.id] = 0; });

  /* BACK EDGES, found structurally rather than by their label. The extractor
     only writes "repeat" when it remembers to, and an unlabelled edge from a
     loop BODY back to its loop makes the loop deeper than the body it
     contains - so the loop is drawn below the thing it repeats, and the edge
     feeding it spans the whole drawing. A depth-first walk marks any edge
     pointing at a node already on the stack, which is exactly a back edge. */
  const out = {};
  graph.nodes.forEach(n => { out[n.id] = []; });
  graph.edges.forEach(e => { if (out[e.src]) out[e.src].push(e); });
  const back = new Set(), seen = new Set(), stack = new Set();
  const walk = id => {
    seen.add(id); stack.add(id);
    for (const e of out[id] || []){
      if (e.label === "repeat" || stack.has(e.dst)) back.add(e);
      else if (!seen.has(e.dst)) walk(e.dst);
    }
    stack.delete(id);
  };
  graph.nodes.forEach(n => { if (!seen.has(n.id)) walk(n.id); });
  for (let i = 0; i < graph.nodes.length; i++){
    let changed = false;
    for (const e of graph.edges){
      if (back.has(e)) continue;
      if (!(e.src in depth) || !(e.dst in depth)) continue;
      if (depth[e.dst] < depth[e.src] + 1){ depth[e.dst] = depth[e.src] + 1; changed = true; }
    }
    if (!changed) break;
  }
  return depth;
}

function gPositions(graph, W, H){
  const depth = gLayer(graph), rows = {};
  graph.nodes.forEach(n => { (rows[depth[n.id]] = rows[depth[n.id]] || []).push(n.id); });

  /* COMPACT THE ROWS. Depths come out of longest-path layering SPARSE - a node
     whose only path in is long sits at depth 5 while nothing occupies 3 or 4 -
     and this used the raw depth as the row index. That drew the gap as empty
     space with an edge running down through it: the "line that goes on for
     ages before anything appears".

     The height was computed from the number of OCCUPIED rows at the same time,
     so the canvas was also too short for what had just been laid out and the
     drawing ran past its own viewBox. One cause, both symptoms. */
  const used = Object.keys(rows).map(Number).sort((a, b) => a - b);
  const rowOf = {};
  used.forEach((d, i) => { rowOf[d] = i; });

  const widest = Math.max(1, ...Object.values(rows).map(r => r.length));
  const pos = {};
  used.forEach(d => {
    const row = rows[d];
    // Centre each row against the widest one so the drawing reads as a spine.
    const offset = ((widest - row.length) * (W + G_GAPX)) / 2;
    row.forEach((id, i) => {
      pos[id] = {x: G_PAD + offset + i * (W + G_GAPX),
                 y: G_PAD + rowOf[d] * (H + G_GAPY)};
    });
  });
  return {pos,
          w: G_PAD * 2 + widest * (W + G_GAPX) - G_GAPX,
          h: G_PAD * 2 + used.length * (H + G_GAPY) - G_GAPY};
}

function gEdgePath(a, b, back, W, H, side){
  const x1 = a.x + W / 2, y1 = a.y + H, x2 = b.x + W / 2, y2 = b.y;
  if (back){
    // A back edge goes UP, in a channel to the right of the WHOLE grid.
    //
    // It used to turn at max(a.x, b.x) + W + 14 - just right of its own two
    // endpoints - which is only clear of the drawing when those two happen to
    // sit in the rightmost column. On any branching plan it was a lane running
    // straight through the node beside them: the loop-back on a digit-sum plan
    // climbed through `return total`, and its "repeat" label landed on top of
    // that node's text. `side` now comes from the caller, which knows how wide
    // the grid is and gives each back edge its own lane.
    return `M ${a.x + W} ${a.y + H / 2} H ${side} V ${b.y + H / 2} H ${b.x + W}`;
  }
  const mid = (y1 + y2) / 2;
  return `M ${x1} ${y1} V ${mid} H ${x2} V ${y2}`;
}

/* A key for the node colours. Each swatch repeats the bar shape used inside
 * the nodes rather than a plain dot, so the mapping is literal instead of
 * remembered. */
function gLegend(){
  return `<ul class="gLegend">` + ["start", "loop", "branch", "return", "step"].map(k =>
    `<li><span class="gSwatch" style="background:${
      (G_STYLE[k] || G_STYLE.step).stroke}"></span>${k}</li>`).join("") + `</ul>`;
}

/* ------------------------------------------------------------------
 * Pan / zoom viewport
 * ------------------------------------------------------------------
 * The card is a fixed height and the drawing is whatever size it needs to be,
 * so one of the two has to give. Fit-to-view on render, then wheel to zoom,
 * drag to pan, three buttons for anyone who would rather not, and arrow keys
 * for anyone who cannot.
 *
 * Everything runs on a transform on one <g>. Nothing re-lays-out, so a long
 * plan stays smooth and the SVG keeps its own coordinates.
 */
function gAttachView(box, svg, layer, size){
  const state = {k: 1, tx: 0, ty: 0};
  const apply = () => layer.setAttribute("transform",
    `translate(${state.tx} ${state.ty}) scale(${state.k})`);

  const fit = () => {
    const bw = box.clientWidth, bh = box.clientHeight;
    if (!bw || !bh) return;
    // Never blow a small graph up past life size - a three-node plan filling a
    // 420px card looks like an error, not a diagram.
    state.k = Math.min(bw / size.w, bh / size.h, 1);
    state.tx = (bw - size.w * state.k) / 2;
    state.ty = (bh - size.h * state.k) / 2;
    apply();
  };

  // Zoom about a point, so whatever is under the cursor stays under it.
  const zoomAt = (factor, px, py) => {
    const k = Math.max(0.25, Math.min(3, state.k * factor));
    const r = k / state.k;
    state.tx = px - (px - state.tx) * r;
    state.ty = py - (py - state.ty) * r;
    state.k = k;
    apply();
  };
  const zoomCentre = f => zoomAt(f, box.clientWidth / 2, box.clientHeight / 2);

  /* A PLAIN wheel belongs to the page, never to this drawing.
   *
   * This used to preventDefault() every wheel event, and the plan card sits
   * directly below the workspace - so scrolling down past it froze the
   * document exactly there. The graph silently ate the gesture, the page
   * stopped moving, and the student could not get back up to the editor or the
   * submit button: the card read as a dead, static block.
   *
   * Ctrl/Cmd+wheel is the convention every map and diagram already uses, and
   * the +/- buttons, the fit button and the keyboard shortcuts all still zoom
   * with no modifier at all - so nothing is lost by giving the wheel back. */
  svg.addEventListener("wheel", e => {
    if (!e.ctrlKey && !e.metaKey) return;      // let the page scroll
    e.preventDefault();
    const r = box.getBoundingClientRect();
    zoomAt(e.deltaY < 0 ? 1.12 : 1 / 1.12, e.clientX - r.left, e.clientY - r.top);
  }, {passive: false});

  let drag = null;
  svg.addEventListener("pointerdown", e => {
    drag = {x: e.clientX, y: e.clientY, tx: state.tx, ty: state.ty};
    svg.setPointerCapture(e.pointerId);
    box.classList.add("panning");
  });
  svg.addEventListener("pointermove", e => {
    if (!drag) return;
    state.tx = drag.tx + (e.clientX - drag.x);
    state.ty = drag.ty + (e.clientY - drag.y);
    apply();
  });
  const endDrag = () => { drag = null; box.classList.remove("panning"); };
  svg.addEventListener("pointerup", endDrag);
  svg.addEventListener("pointercancel", endDrag);

  // A drawing that can only be reached with a mouse is a drawing half the
  // class cannot read.
  box.addEventListener("keydown", e => {
    const step = e.shiftKey ? 80 : 28;
    const moves = {ArrowLeft: [step, 0], ArrowRight: [-step, 0],
                   ArrowUp: [0, step], ArrowDown: [0, -step]};
    if (moves[e.key]){
      e.preventDefault();
      state.tx += moves[e.key][0]; state.ty += moves[e.key][1]; apply();
    } else if (e.key === "+" || e.key === "="){ e.preventDefault(); zoomCentre(1.15); }
    else if (e.key === "-"){ e.preventDefault(); zoomCentre(1 / 1.15); }
    else if (e.key === "0"){ e.preventDefault(); fit(); }
  });

  box.querySelector(".gzin").onclick  = () => zoomCentre(1.2);
  box.querySelector(".gzout").onclick = () => zoomCentre(1 / 1.2);
  box.querySelector(".gzfit").onclick = fit;

  /* REAL FULLSCREEN, which the expand icon has always promised and never done.
     It was wired to fit(), so pressing it on a graph too big to read just
     recentred the same small box. Fit stays - it is genuinely useful - and
     moves to its own control; this one now fills the screen, which is what a
     student pressing it wants. */
  const full = box.querySelector(".gzfull");
  if (full){
    full.onclick = () => {
      const doc = document;
      if (doc.fullscreenElement || doc.webkitFullscreenElement){
        (doc.exitFullscreen || doc.webkitExitFullscreen).call(doc);
      } else {
        const go = box.requestFullscreen || box.webkitRequestFullscreen;
        // No fullscreen API (older Safari in an iframe): fall back to filling
        // the window with CSS, so the control still does something.
        if (go) go.call(box).then(() => setTimeout(fit, 60)).catch(() => {
          box.classList.toggle("gfake"); fit();
        });
        else { box.classList.toggle("gfake"); setTimeout(fit, 60); }
      }
    };
    // Leaving fullscreen by Escape has to re-fit too, or the drawing comes
    // back sized for a screen it is no longer on.
    box.addEventListener("fullscreenchange", () => setTimeout(fit, 60));
  }

  // The card can be revealed while still hidden (display:none has no
  // clientWidth), so fit now, again next frame, and again on any resize.
  fit();
  requestAnimationFrame(fit);
  if (window.ResizeObserver) new ResizeObserver(fit).observe(box);
}

/* Render one graph into `el`.
 *
 * opts:
 *   height  px for the viewport (default 420)
 *   flash   ids of nodes to highlight as newly added
 *   legend  render the colour key above the drawing
 */
function renderGraph(el, graph, emptyText, opts = {}){
  if (!el) return;
  if (!graph || !graph.nodes || !graph.nodes.length){
    // "Nothing captured yet" is itself information, and a lone Start node
    // floating in a large empty box is not.
    el.innerHTML = `<div class="empty">
      <span class="eicon" aria-hidden="true">
        <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor"
             stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <rect x="4" y="3" width="16" height="5" rx="2"/>
          <rect x="7" y="16" width="10" height="5" rx="2"/>
          <path d="M12 8v8"/></svg></span>
      <p>${gEsc(emptyText || "Nothing to draw yet.")}</p></div>`;
    return;
  }

  const {lines, W, H} = gMetrics(graph);
  const {pos, w, h} = gPositions(graph, W, H);
  const flash = new Set(opts.flash || []);
  // Each graph needs its OWN marker id: two graphs on one page (plan and code)
  // both defining id="ah" makes the second definition win for both.
  const uid = "g" + Math.random().toString(36).slice(2, 8);
  // DIRECTION IS THE ONLY THING THAT TELLS THESE EDGES APART, so the head has
  // to survive fit-to-view shrinking it. markerUnits defaults to strokeWidth,
  // so these numbers are multiplied by the 1.5 below before they hit the
  // screen; the old 8x6 triangle read as a smudge on anything but a tiny graph.
  const parts = [
    `<defs><marker id="${uid}" markerWidth="11" markerHeight="8" refX="10" refY="4" `,
    `orient="auto"><path d="M0,0 L0,8 L10,4 z" fill="var(--graph-line)"/></marker></defs>`
  ];

  // Edge labels are collected and drawn AFTER the nodes. Painted in edge order
  // they went UNDER any node the routing passes behind, and the one that lost
  // most often was "repeat" on a loop-back - which, now that no edge is dashed,
  // is the only word naming the most important edge in the drawing.
  const elabels = [];
  // Lanes for the back edges, right of every node. A second one is 14px
  // further out so nested loops do not draw over each other.
  const CHAN = w - G_PAD + 14;
  let lane = 0;
  // How far right the routing actually reaches. gPositions() sizes the canvas
  // from the NODE grid alone, so a back edge - which runs in a channel outside
  // the rightmost column - and its label were laid out past the edge of the
  // drawing and clipped by fit-to-view.
  let reach = w;

  graph.edges.forEach(e => {
    const a = pos[e.src], b = pos[e.dst];
    if (!a || !b) return;                       // defensive: dangling edge
    const back = e.label === "repeat" || b.y <= a.y;
    const side = back ? CHAN + (lane++) * 14 : 0;
    if (back) reach = Math.max(reach, side + G_PAD);
    // EVERY edge is one solid line with one arrowhead. A back edge used to be
    // dashed, which students read as "optional", "maybe", or "not really part
    // of the plan" - none of which it is: it is the loop closing, the most
    // load-bearing edge in the drawing. It routes around the outside so it
    // never crosses the body of its own loop, and the arrow says which way it
    // goes, which is what the dashes were being asked to say and could not.
    parts.push(`<path d="${gEdgePath(a, b, back, W, H, side)}" fill="none" `
             + `stroke="var(--graph-line)" stroke-width="1.5" `
             + `stroke-linejoin="round" marker-end="url(#${uid})"/>`);
    if (e.label){
      const lx = back ? side + 6 : (a.x + b.x) / 2 + W / 2 + 6;
      const ly = back ? (a.y + b.y) / 2 + H / 2 : (a.y + H + b.y) / 2;
      if (back) reach = Math.max(reach, lx + e.label.length * 6.5 + G_PAD);
      // paint-order lets the halo sit BEHIND the glyphs, so a label crossing an
      // edge stays readable without a box that would clutter the drawing.
      elabels.push(`<text x="${lx}" y="${ly}" font-size="10.5" font-weight="500" `
                 + `fill="var(--graph-label)" stroke="var(--surface)" `
                 + `stroke-width="3.5" paint-order="stroke">${gEsc(e.label)}</text>`);
    }
  });

  graph.nodes.forEach(n => {
    const p = pos[n.id], s = G_STYLE[n.kind] || G_STYLE.step;
    const r = (n.kind === "start" || n.kind === "end") ? H / 2 : 10;
    const cx = p.x + W / 2;                      // node centre-line
    const ls = lines[n.id];
    parts.push(`<g class="gnode${flash.has(n.id) ? " gnew" : ""}">`
             + `<title>${gEsc(n.label)}</title>`);
    parts.push(`<rect x="${p.x}" y="${p.y}" width="${W}" height="${H}" `
             + `rx="${r}" fill="${s.fill}" stroke="${s.stroke}" stroke-width="1.5"/>`);
    // A colour bar along the TOP edge keeps the kind legible for a reader who
    // cannot separate the amber and green outlines - colour is never the only
    // cue. Centred on the top, it frames the label instead of fighting it.
    if (n.kind !== "start" && n.kind !== "end"){
      parts.push(`<rect x="${p.x + 14}" y="${p.y + 1}" width="${W - 28}" `
               + `height="3" rx="1.5" fill="${s.stroke}"/>`);
    }
    parts.push(`<text x="${cx}" y="${p.y + 16}" text-anchor="middle" font-size="8.5" `
             + `font-weight="700" fill="${s.text}" letter-spacing="1.4">${
                 gEsc(String(n.kind || "step").toUpperCase())}</text>`);
    // Label lines, vertically centred in whatever space the eyebrow leaves.
    const first = p.y + 22 + ((H - 26) - ls.length * G_LINE) / 2 + 11;
    ls.forEach((line, i) => {
      parts.push(`<text x="${cx}" y="${first + i * G_LINE}" text-anchor="middle" `
               + `font-size="12" fill="var(--text)">${gEsc(line)}</text>`);
    });
    parts.push(`</g>`);
  });
  parts.push(...elabels);

  const height = opts.height || 420;
  el.innerHTML =
    (opts.legend ? `<div class="graphMeta">${gLegend()}</div>` : "")
    + `<div class="gview" style="height:${height}px" tabindex="0" role="group"
            aria-label="Flowchart with ${graph.nodes.length} steps. Arrow keys pan, plus and minus zoom, 0 fits to view.">
         <svg width="100%" height="100%" font-family="${G_FONT}" role="img"
              aria-label="flowchart with ${graph.nodes.length} steps">
           <g class="gzoom">${parts.join("")}</g>
         </svg>
         <div class="gctl">
           <button type="button" class="gzout" aria-label="Zoom out" title="Zoom out">&minus;</button>
           <button type="button" class="gzin" aria-label="Zoom in" title="Zoom in">+</button>
           <button type="button" class="gzfit" aria-label="Fit to view" title="Fit to view">
             <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                  stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"
                  aria-hidden="true"><path d="M3 12h18M12 3v18"/></svg>
           </button>
           <button type="button" class="gzfull" aria-label="Fullscreen" title="Fullscreen">
             <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                  stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"
                  aria-hidden="true"><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/></svg>
           </button>
         </div>
       </div>`;

  const box = el.querySelector(".gview");
  gAttachView(box, box.querySelector("svg"), box.querySelector(".gzoom"),
              {w: reach, h});
}

/* Render the finished artifact: both graphs plus what differs between them.
 * `payload` is exactly the body of POST /graphs. */
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
         <h3><span class="gTag code">Code</span>What your code does</h3>
         <div id="gCode"></div></div>
       <div class="graphBox">
         <h3><span class="gTag plan">Plan</span>What you said you would do</h3>
         <div id="gPlan"></div></div>
     </div>
     <div class="graphDiff" id="gDiff"></div>`;

  // Code LEFT, plan RIGHT - the sides they were each built on. student.html's
  // .split puts the editor on the left and the tutor chat in the 372px column
  // on the right, so a plan extracted from that chat belongs on the right too.
  // Reversed, the reader had to cross the page to match each graph to where it
  // came from.
  renderGraph(document.getElementById("gCode"), payload.code,
              "No code submitted yet.", {height: 320});
  renderGraph(document.getElementById("gPlan"), payload.plan,
              "No plan was captured from your chat.", {height: 320});

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
    + `<p class="hint">A difference is not automatically a mistake - you may `
    + `have simplified while writing. It is worth knowing which ones you chose `
    + `and which ones surprised you.</p>`;
}
