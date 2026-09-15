/* Node-only cache and geometry regressions. No backend or browser required. */
const assert=require("node:assert/strict"),fs=require("node:fs"),path=require("node:path"),vm=require("node:vm");
const read=name=>fs.readFileSync(path.resolve(__dirname,"../../frontend",name),"utf8");
const nested=require("./graph-fixture.cjs");
/* Distance from a point to an orthogonal polyline, for the label checks below. */
const segDist=(px,py,[ax,ay],[bx,by])=>{const dx=bx-ax,dy=by-ay,len=dx*dx+dy*dy;
  const t=len?Math.max(0,Math.min(1,((px-ax)*dx+(py-ay)*dy)/len)):0;
  return Math.hypot(px-(ax+dx*t),py-(ay+dy*t));};
const polyDist=(px,py,pts)=>{let m=Infinity;for(let i=1;i<pts.length;i++)m=Math.min(m,segDist(px,py,pts[i-1],pts[i]));return m;};
const graphContext=vm.createContext({});
vm.runInContext(read("graphs.js"),graphContext);
const diamond={nodes:["s","b","t","f","r"].map((id,i)=>({id,kind:["start","branch","step","step","return"][i],label:id})),edges:[{src:"s",dst:"b"},{src:"b",dst:"t",label:"yes"},{src:"b",dst:"f",label:"no"},{src:"t",dst:"r"},{src:"f",dst:"r"}]};
for(const graph of [nested,{nodes:[...nested.nodes].reverse(),edges:[...nested.edges].reverse()},diamond,
  {nodes:[{id:"s",kind:"start",label:"Start"},{id:"x",kind:"loop",label:"Repeat"},{id:"r",kind:"return",label:"End"}],edges:[{src:"s",dst:"x"},{src:"x",dst:"x",label:"retry"},{src:"x",dst:"r",label:"a long exit label that needs space"}]},
  {nodes:[{id:"__proto__",label:"Start",kind:"start"},{id:"constructor",label:"End",kind:"return"}],edges:[{src:"__proto__",dst:"constructor"}]}
]){
  const before=JSON.stringify(graph),{W,H}=graphContext.gMetrics(graph),layout=graphContext.gLayout(graph,W,H);
  assert.equal(JSON.stringify(graph),before,"Layout must not mutate a submitted graph");
  assert.equal(layout.routes.length,graph.edges.length);
  for(const route of layout.routes){
    for(const [x,y] of route.points)assert(Number.isFinite(x)&&Number.isFinite(y)&&x>=0&&y>=0&&x<=layout.w&&y<=layout.h,"Point outside drawing");
    if(route.label){
      assert(route.label.x>=0&&route.label.x+route.label.w<=layout.w,"Clipped edge label");
      // A label has to name ONE arrow. Placed from the port/lane geometry it
      // landed in the GAP between the two edges leaving a loop header, so all
      // three "repeat" labels on the nested fixture sat closer to the sibling
      // "done" edge than to their own - which names neither.
      const cx=route.label.x+route.label.w/2,cy=route.label.y+route.label.h/2;
      const own=polyDist(cx,cy,route.points);
      assert(own<1,`"${route.label.text}" floats ${own.toFixed(1)}px off its own edge`);
      for(const other of layout.routes)if(other!==route)
        assert(polyDist(cx,cy,other.points)>own,
          `"${route.label.text}" is no nearer its own edge than ${other.edge.src}\u2192${other.edge.dst}`);
    }
    for(let j=1;j<route.points.length;j++){
      const [x1,y1]=route.points[j-1],[x2,y2]=route.points[j];
      assert(x1===x2||y1===y2,"Orthogonal routing");
      for(const [id,p] of Object.entries(layout.pos)){
        const hit=x1===x2?x1>p.x+.01&&x1<p.x+W-.01&&Math.max(y1,y2)>p.y+.01&&Math.min(y1,y2)<p.y+H-.01
          :y1>p.y+.01&&y1<p.y+H-.01&&Math.max(x1,x2)>p.x+.01&&Math.min(x1,x2)<p.x+W-.01;
        assert(!hit,route.edge.src+"→"+route.edge.dst+" crosses "+id);
      }
    }
  }
  if(graph.nodes.length===14){
    assert(layout.depth.c>layout.depth.b,"Forward repeat enters loop body");
    assert(layout.depth.g>layout.depth.f,"Inner loop exit follows its body");
    assert(layout.depth.i>layout.depth.h,"Outer loop exit follows its body");
    assert(layout.depth.m>layout.depth.l,"Final return follows last loop");
    const returns=layout.routes.filter(r=>r.back);
    for(let i=0;i<returns.length;i++)for(let j=i+1;j<returns.length;j++){
      const a=returns[i],b=returns[j],ay=a.points.map(p=>p[1]),by=b.points.map(p=>p[1]);
      if(Math.max(...ay)>=Math.min(...by)&&Math.max(...by)>=Math.min(...ay))assert.notEqual(a.lane,b.lane,"Overlapping loop returns need separate lanes");
    }
  }
}
function storage(){
  const data={};
  return new Proxy(data,{get:(t,k)=>k==="getItem"?(key=>t[key]??null):k==="setItem"?((key,value)=>t[key]=String(value)):k==="removeItem"?(key=>delete t[key]):Reflect.get(t,k)});
}
(async()=>{
  const window=new EventTarget(),sessionStorage=storage(),localStorage=storage(),calls=[];
  let now=100000,version=1,fail=false,hold=null,status=200,refreshErrors=0;
  window.addEventListener("acadia:cache-error",()=>refreshErrors++);
  class Clock extends Date{static now(){return now;}}
  const Session={verified:{student_id:"a",role:"student"},verifiedAt:now,async check(){return this.verified;}};
  window.fetch=async(url,init={})=>{calls.push({url:String(url),method:init.method||"GET"});if(hold)await hold;if(fail)throw Error("offline");return new Response(JSON.stringify({version}),{status});};
  const context=vm.createContext({window,sessionStorage,localStorage,Session,API:"",Date:Clock,URL,Request,Response,CustomEvent,Intl,location:{href:"http://test/dashboard.html",origin:"http://test"}});
  vm.runInContext(read("cache.js"),context);
  const get=(init={})=>window.fetch("/student/progress?timezone=UTC",init).then(r=>r.json());
  const initial=await Promise.all([get(),get(),get()]);
  assert.equal(calls.length,1,"Coalesce concurrent GETs");assert.equal(initial[0].version,1);
  await get();assert.equal(calls.length,1,"Fresh cache");
  version=2;assert.equal((await get({cache:"reload"})).version,2);assert.equal(calls.length,2,"Explicit refresh bypasses cache");
  now+=46000;version=3;assert.equal((await get()).version,2,"Stale data remains available");
  await new Promise(resolve=>setImmediate(resolve));assert.equal((await get()).version,3,"Background refresh");
  Session.verified={student_id:"b",role:"student"};Session.verifiedAt=now;
  await get();assert.equal(calls.length,4,"Never reuse another student's cache");
  await window.fetch("/grade_chunk",{method:"POST"});
  await get();assert.equal(calls.length,6,"Successful writes invalidate results");
  await window.fetch("/history/problem");await window.fetch("/history/problem");
  assert.equal(calls.length,8,"History is never cached");
  window.AcadiaCache.invalidate();
  let release;hold=new Promise(r=>release=r);
  const inFlight=get();await Promise.resolve();window.AcadiaCache.invalidate();release();await inFlight;hold=null;
  const afterRace=calls.length;await get();assert.equal(calls.length,afterRace+1,"Old in-flight results cannot repopulate invalidated storage");
  window.AcadiaCache.clear();fail=true;await assert.rejects(get(),/offline/);fail=false;
  await get();assert(!Object.keys(sessionStorage).some(k=>k.includes("history")));
  now+=46000;status=503;
  assert.equal((await get()).version,3,"Keep the last successful result when background refresh fails");
  await new Promise(resolve=>setImmediate(resolve));assert.equal(refreshErrors,1,"HTTP background failures must be announced");
  status=200;fail=true;await get();await new Promise(resolve=>setImmediate(resolve));assert.equal(refreshErrors,2,"Transport background failures must be announced");fail=false;
  let liveChecks=0;Session.check=async()=>{liveChecks++;return {student_id:"c",role:"student"};};
  const accountChange=new Event("storage");accountChange.key="acadia.cache.generation";window.dispatchEvent(accountChange);
  assert.equal(Session.verified,null,"An account change in another tab invalidates the verified identity");
  await get();assert.equal(liveChecks,1,"Verify the live account before serving cached private results");
  window.AcadiaCache.clear();assert.equal(Object.keys(sessionStorage).length,0,"Logout clears private results");
  console.log("PASS: graph ordering, orthogonal routing, label bounds, labels on their own edge, loop lanes, prototype-safe IDs, cache TTL/coalescing/account isolation/refresh/invalidation/errors.");
})().catch(e=>{console.error(e);process.exitCode=1;});
