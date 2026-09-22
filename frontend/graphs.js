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
  const lines = Object.create(null);
  let widest = 0, tallest = 1;
  graph.nodes.forEach(n => {
    const ls = gWrapLabel(typeof undash === "function" ? undash(n.label) : n.label);
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

/* The longest straight run of a polyline - the roomiest place to seat a label
   without it spilling around a corner. Returned as endpoints rather than a
   midpoint so a label that would collide with another can slide ALONG its own
   line instead of being pushed off it. */
function gLongestSegment(points){
  let seg=[points[0][0],points[0][1],points[0][0],points[0][1]],span=-1;
  for(let i=1;i<points.length;i++){
    const [ax,ay]=points[i-1],[bx,by]=points[i],d=Math.abs(bx-ax)+Math.abs(by-ay);
    if(d>span){span=d;seg=[ax,ay,bx,by];}
  }
  return seg;
}

/* Seat each label on its own segment, sliding along it when two would overlap.
   Labels are the only thing allowed to move here: an edge routed clear of the
   nodes must never be re-routed to make room for text. A label that cannot find
   a free spot keeps the last one tried - overlapping text on the right line
   still beats tidy text on the wrong one. */
function gSeatLabels(routes){
  const placed=[],hits=(a,b)=>a.x<b.x+b.w+4&&b.x<a.x+a.w+4&&a.y<b.y+b.h+3&&b.y<a.y+a.h+3;
  // Routing is orthogonal, so every segment is axis-aligned and a bounding-box
  // overlap IS the intersection test. A foreign line running under the label is
  // the whole defect being fixed: the box is opaque, so it breaks whatever it
  // covers, and a reader cannot tell which of the two broken lines it belongs to.
  const crosses=(l,pts)=>{
    for(let i=1;i<pts.length;i++){
      const [ax,ay]=pts[i-1],[bx,by]=pts[i];
      if(Math.min(ax,bx)<=l.x+l.w+3&&Math.max(ax,bx)>=l.x-3
       &&Math.min(ay,by)<=l.y+l.h+3&&Math.max(ay,by)>=l.y-3)return true;
    }
    return false;
  };
  for(const r of routes){
    if(!r.label)continue;
    const [ax,ay,bx,by]=r.label.seg,l=r.label;
    for(const t of [.5,.36,.64,.24,.76,.14,.86,.08,.92]){
      l.x=Math.max(2,ax+(bx-ax)*t-l.w/2);l.y=ay+(by-ay)*t-l.h/2;
      if(!placed.some(p=>hits(l,p))
       &&!routes.some(o=>o!==r&&crosses(l,o.points)))break;
    }
    placed.push(l);
  }
}

/* Cycle detection is structural: "repeat" can also label a forward edge
   into a loop body. Invisible rank constraints keep exits below loop bodies. */
function gStructure(graph){
  const ids=new Set(graph.nodes.map(n=>n.id));
  const edges=(graph.edges||[]).filter(e=>ids.has(e.src)&&ids.has(e.dst));
  const out=new Map(graph.nodes.map(n=>[n.id,[]]));
  const incoming=new Map(graph.nodes.map(n=>[n.id,0]));
  edges.forEach(e=>{out.get(e.src).push(e);incoming.set(e.dst,incoming.get(e.dst)+1);});
  const priority=e=>/^(no|false|done|exit|else)$/i.test(e.label||"")?1:0;
  out.forEach(list=>list.sort((a,b)=>priority(a)-priority(b)));
  const seen=new Set(),stack=new Set(),back=new Set();
  function visit(id){
    seen.add(id);stack.add(id);
    for(const e of out.get(id)){
      if(stack.has(e.dst))back.add(e);
      else if(!seen.has(e.dst))visit(e.dst);
    }
    stack.delete(id);
  }
  const roots=[...graph.nodes].sort((a,b)=>(a.kind==="start"?-2:incoming.get(a.id)===0?-1:0)-(b.kind==="start"?-2:incoming.get(b.id)===0?-1:0));
  roots.forEach(n=>{if(!seen.has(n.id))visit(n.id);});
  const forward=edges.filter(e=>!back.has(e)), ranking=[...forward];
  const predecessors=new Map(graph.nodes.map(n=>[n.id,[]]));
  edges.forEach(e=>predecessors.get(e.dst).push(e.src));
  for(const header of graph.nodes.filter(n=>n.kind==="loop")){
    const tails=edges.filter(e=>back.has(e)&&e.dst===header.id).map(e=>e.src);
    if(!tails.length)continue;
    const body=new Set([header.id]),todo=[...tails];
    while(todo.length){
      const id=todo.pop();if(body.has(id))continue;
      body.add(id);todo.push(...predecessors.get(id));
    }
    const exits=forward.filter(e=>e.src===header.id&&!body.has(e.dst));
    for(const exit of exits)for(const id of body){
      if(id!==header.id&&!ranking.some(e=>e.src===id&&e.dst===exit.dst))ranking.push({src:id,dst:exit.dst});
    }
  }
  const depth=Object.create(null),indegree=new Map(graph.nodes.map(n=>[n.id,0]));
  const rankedOut=new Map(graph.nodes.map(n=>[n.id,[]]));
  ranking.forEach(e=>{rankedOut.get(e.src).push(e);indegree.set(e.dst,indegree.get(e.dst)+1);});
  graph.nodes.forEach(n=>{depth[n.id]=0;});
  const queue=roots.filter(n=>!indegree.get(n.id)).map(n=>n.id);
  for(let i=0;i<queue.length;i++)for(const e of rankedOut.get(queue[i])){
    depth[e.dst]=Math.max(depth[e.dst],depth[e.src]+1);
    indegree.set(e.dst,indegree.get(e.dst)-1);if(!indegree.get(e.dst))queue.push(e.dst);
  }
  return {depth,back,edges};
}
function gLayer(graph){return gStructure(graph).depth;}
/* Distinct ports and gap tracks prevent shared line segments. Long forward
   branches travel on the left; loop returns travel on the right. Channels can
   be reused only by non-overlapping loops, keeping sequential loops compact. */
function gLayout(graph,W,H){
  const {depth,back,edges}=gStructure(graph),rows=[];
  graph.nodes.forEach(n=>{(rows[depth[n.id]]||=[]).push(n.id);});
  const rank=new Map();
  rows.forEach(row=>{
    const score=id=>{
      const parents=edges.filter(e=>e.dst===id&&!back.has(e)).map(e=>rank.get(e.src)).filter(Number.isFinite);
      return parents.length?parents.reduce((a,b)=>a+b,0)/parents.length:0;
    };
    row.sort((a,b)=>score(a)-score(b));row.forEach((id,i)=>rank.set(id,i));
  });
  const widest=Math.max(1,...rows.map(r=>r.length)),gridWidth=widest*(W+72)-72;
  const gapTracks=new Map();
  function track(gap){const n=gapTracks.get(gap)||0;gapTracks.set(gap,n+1);return {gap,n};}
  const outgoing=new Map(graph.nodes.map(n=>[n.id,[]])),incoming=new Map(graph.nodes.map(n=>[n.id,[]]));
  const routes=edges.map(edge=>{
    const reverse=back.has(edge),outer=reverse||depth[edge.dst]!==depth[edge.src]+1;
    const fromTrack=track(depth[edge.src]),toTrack=outer?track(depth[edge.dst]-1):fromTrack;
    const route={edge,back:reverse,outer,fromTrack,toTrack};
    outgoing.get(edge.src).push(route);incoming.get(edge.dst).push(route);return route;
  });
  const left=routes.filter(r=>r.outer&&!r.back),right=routes.filter(r=>r.back);
  const laneWidth=list=>Math.max(76,...list.map(r=>Math.min(22,String(r.edge.label||"").length)*6.5+28));
  const leftWidth=laneWidth(left),rightWidth=laneWidth(right);
  function allocate(list){
    const lanes=[];
    list.sort((a,b)=>Math.abs(depth[a.edge.src]-depth[a.edge.dst])-Math.abs(depth[b.edge.src]-depth[b.edge.dst]));
    for(const r of list){
      const lo=Math.min(depth[r.edge.src],depth[r.edge.dst])-.5,hi=Math.max(depth[r.edge.src],depth[r.edge.dst])+.5;
      let index=lanes.findIndex(lane=>lane.every(span=>hi<span[0]||lo>span[1]));
      if(index<0){index=lanes.length;lanes.push([]);}
      lanes[index].push([lo,hi]);r.laneIndex=index;
    }
    return lanes.length;
  }
  const leftCount=allocate(left),rightCount=allocate(right);
  const gridX=G_PAD+(leftCount?leftCount*leftWidth+24:0);
  const gapHeight=gap=>Math.max(68,(gapTracks.get(gap)||0)*24+28),y=[];
  y[0]=G_PAD+(gapTracks.has(-1)?gapHeight(-1):0);
  for(let row=1;row<rows.length;row++)y[row]=y[row-1]+H+gapHeight(row-1);
  const pos=Object.create(null);
  rows.forEach((row,d)=>row.forEach((id,i)=>{pos[id]={x:gridX+(gridWidth-row.length*(W+72)+72)/2+i*(W+72),y:y[d]};}));
  right.forEach(r=>{r.lane=gridX+gridWidth+32+r.laneIndex*rightWidth;});
  left.forEach(r=>{r.lane=gridX-32-r.laneIndex*leftWidth;});
  const trackY=t=>(t.gap===-1?G_PAD:y[t.gap]+H)+18+t.n*24;
  const portX=(id,list,r)=>pos[id].x+W*(list.indexOf(r)+1)/(list.length+1);
  routes.forEach(r=>{
    const {edge}=r,a=pos[edge.src],b=pos[edge.dst];
    const sx=portX(edge.src,outgoing.get(edge.src),r),dx=portX(edge.dst,incoming.get(edge.dst),r);
    const leave=trackY(r.fromTrack),enter=trackY(r.toTrack);
    r.points=[[sx,a.y+H],[sx,leave]];
    if(r.outer)r.points.push([r.lane,leave],[r.lane,enter],[dx,enter]);else r.points.push([dx,leave]);
    r.points.push([dx,b.y]);
    r.path=r.points.map(([x,y],i)=>(i?"L ":"M ")+x+" "+y).join(" ");
    if(edge.label){
      const label=typeof undash==="function"?undash(edge.label):String(edge.label),short=label.length>22?label.slice(0,21)+"…":label,width=short.length*6.5+14;
      // ON THE LINE IT NAMES, never merely near it. Placing the label from the
      // port/lane geometry put it in the GAP between the two edges leaving one
      // node: on the nested-loop fixture all three "repeat" labels landed
      // closer to the sibling "done" edge than to their own, and a label
      // floating between two arrows names neither of them. Anchored to the
      // middle of this route's longest straight run instead, so its opaque
      // background breaks its OWN polyline - the association is drawn rather
      // than left to the reader to infer. gLabelPlacement keeps it in frame and
      // off the other labels.
      r.label={text:short,full:label,w:width,h:20,
               seg:gLongestSegment(r.points),x:0,y:0};
    }
  });
  gSeatLabels(routes);
  const nodeReach=gridX+gridWidth+(rightCount?32+(rightCount-1)*rightWidth+rightWidth/2:0)+G_PAD;
  const w=Math.max(nodeReach,...routes.map(r=>r.label?r.label.x+r.label.w+G_PAD:0));
  const h=y[rows.length-1]+H+(gapTracks.has(rows.length-1)?gapHeight(rows.length-1):0)+G_PAD;
  return {pos,w,h,routes,depth,W,H};
}
function gPositions(graph,W,H){return gLayout(graph,W,H);}

/* A key for the node colours. Each swatch repeats the bar shape used inside
 * the nodes rather than a plain dot, so the mapping is literal instead of
 * remembered. */
function gLegend(){
  return `<ul class="gLegend">` + ["start", "loop", "branch", "return", "step"].map(k =>
    `<li><span class="gSwatch" style="background:${
      (G_STYLE[k] || G_STYLE.step).stroke}"></span>${k}</li>`).join("") + `</ul>`;
}

/* Native scrolling keeps diagrams legible; fit and zoom are explicit controls. */
function gAttachView(box,svg,layer,size){
  const frame=box.closest(".graph-frame");
  let scale=Math.max(.85,Math.min(1,(box.clientWidth||640)/size.w));
  function setScale(next,centre=true){
    const x=(box.scrollLeft+box.clientWidth/2)/scale,y=(box.scrollTop+box.clientHeight/2)/scale;
    scale=Math.max(.15,Math.min(2.5,next));
    svg.setAttribute("width",String(Math.ceil(size.w*scale)));svg.setAttribute("height",String(Math.ceil(size.h*scale)));
    frame.querySelector(".gzlevel").textContent=Math.round(scale*100)+"%";
    if(centre){box.scrollLeft=x*scale-box.clientWidth/2;box.scrollTop=y*scale-box.clientHeight/2;}
  }
  const fit=()=>{setScale(Math.min(1,box.clientWidth/size.w,box.clientHeight/size.h),false);box.scrollLeft=0;box.scrollTop=0;};
  svg.setAttribute("viewBox","0 0 "+size.w+" "+size.h);setScale(scale,false);
  const centreStart=()=>{if(box.clientWidth)box.scrollLeft=Math.max(0,(size.focusX||size.w/2)*scale-box.clientWidth/2);};
  centreStart();requestAnimationFrame(centreStart);
  frame.querySelector(".gzin").onclick=()=>setScale(scale*1.2);
  frame.querySelector(".gzout").onclick=()=>setScale(scale/1.2);
  frame.querySelector(".gzfit").onclick=fit;
  frame.querySelector(".gzlevel").onclick=()=>setScale(1);
  const full=frame.querySelector(".gzfull");
  full.onclick=async()=>{
    if(document.fullscreenElement===frame){await document.exitFullscreen();return;}
    if(frame.classList.contains("gfake")){frame.classList.remove("gfake");full.textContent="Expand";box.focus();return;}
    try{if(!frame.requestFullscreen)throw Error();await frame.requestFullscreen();}
    catch{frame.classList.add("gfake");full.textContent="Close";}
    box.focus();
  };
  frame.addEventListener("keydown",e=>{
    if(e.key==="Escape"&&frame.classList.contains("gfake")){e.stopPropagation();frame.classList.remove("gfake");full.textContent="Expand";full.focus();}
    else if(e.key==="+"||e.key==="="){e.preventDefault();setScale(scale*1.2);}
    else if(e.key==="-"){e.preventDefault();setScale(scale/1.2);}
    else if(e.key==="0"){e.preventDefault();fit();}
  });
  // Unmodified wheel and touch gestures use native scrolling, including at
  // viewport boundaries. No global listeners or observers survive a redraw.
  box.addEventListener("wheel",e=>{if(!e.ctrlKey&&!e.metaKey)return;e.preventDefault();setScale(scale*(e.deltaY<0?1.1:1/1.1));},{passive:false});
  let drag=null;
  svg.addEventListener("pointerdown",e=>{if(e.pointerType!=="mouse"||e.button!==0)return;drag={x:e.clientX,y:e.clientY,left:box.scrollLeft,top:box.scrollTop};svg.setPointerCapture(e.pointerId);box.classList.add("panning");});
  svg.addEventListener("pointermove",e=>{if(drag){box.scrollLeft=drag.left+drag.x-e.clientX;box.scrollTop=drag.top+drag.y-e.clientY;}});
  const stop=()=>{drag=null;box.classList.remove("panning");};
  svg.addEventListener("pointerup",stop);svg.addEventListener("pointercancel",stop);
}
function renderGraphLoading(el,text="Loading your saved plan…"){
  if(!el)return;
  el.setAttribute("aria-busy","true");
  el.innerHTML='<div class="graph-loading" role="status"><span class="spin" aria-hidden="true"></span><strong>'+gEsc(text)+'</strong><p>Your graph will appear here when it is ready.</p><div class="graph-loading-nodes" aria-hidden="true"><i class="skel"></i><span>↓</span><i class="skel"></i><span>↓</span><i class="skel"></i></div></div>';
}

function renderGraph(el, graph, emptyText, opts = {}){
  if (!el) return;
  el.setAttribute("aria-busy","false");
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
  const {pos, w, h, routes} = gLayout(graph, W, H);
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

  const elabels=[];
  routes.forEach(route=>{
    parts.push('<path d="'+route.path+'" fill="none" stroke="var(--code-bg)" stroke-width="6" stroke-linejoin="round"/>');
    parts.push('<path d="'+route.path+'" fill="none" stroke="var(--graph-line)" stroke-width="1.6" stroke-linejoin="round" marker-end="url(#'+uid+')"/>');
    if(route.label){
      const l=route.label;
      elabels.push('<g><title>'+gEsc(l.full)+'</title><rect x="'+l.x+'" y="'+l.y+'" width="'+l.w+'" height="'+l.h+'" rx="5" fill="var(--code-bg)" stroke="var(--border)"/><text x="'+(l.x+l.w/2)+'" y="'+(l.y+13.5)+'" text-anchor="middle" font-size="11" fill="var(--graph-label)">'+gEsc(l.text)+'</text></g>');
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

  const height=opts.height||420;
  el.innerHTML=(opts.legend?'<div class="graphMeta">'+gLegend()+'</div>':'')
    +'<div class="graph-frame"><div class="graph-toolbar"><span>Flowchart</span><div class="graph-tools">'
    +'<button type="button" class="gzout ghost" aria-label="Zoom out">−</button><button type="button" class="gzlevel ghost" aria-label="Reset to actual size">100%</button><button type="button" class="gzin ghost" aria-label="Zoom in">+</button><button type="button" class="gzfit ghost">Fit</button><button type="button" class="gzfull ghost" aria-label="Expand flowchart">Expand</button></div></div>'
    +'<div class="gview" style="height:'+height+'px" tabindex="0" role="region" aria-label="Scrollable flowchart. Arrow keys scroll; plus and minus zoom; 0 fits the graph."><svg font-family="'+G_FONT+'" role="img" aria-label="Flowchart with '+graph.nodes.length+' steps. A text version follows."><g class="gzoom">'+parts.join("")+'</g></svg></div><p class="graph-help">Scroll to explore · Fit shows the whole graph</p></div>'
    +'<details class="graph-outline"><summary>Read as steps</summary><ol>'
    +[...graph.nodes].sort((a,b)=>pos[a.id].y-pos[b.id].y||pos[a.id].x-pos[b.id].x).map(n=>'<li><strong>'+gEsc(n.label)+'</strong>'
      +(graph.edges||[]).filter(e=>e.src===n.id).map(e=>{const target=graph.nodes.find(n=>n.id===e.dst);return target?'<span>'+(e.label?gEsc(e.label)+': ':'Next: ')+gEsc(target.label)+'</span>':'';}).join("")+'</li>').join("")
    +'</ol></details>';

  const box = el.querySelector(".gview");
  gAttachView(box, box.querySelector("svg"), box.querySelector(".gzoom"),
              {w, h, focusX:pos[(graph.nodes.find(n=>n.kind==="start")||graph.nodes[0]).id].x+W/2});
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
