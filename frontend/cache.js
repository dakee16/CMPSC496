/* Short-lived private navigation cache. Auth, history, chat, streams and writes
   always go to the server. Data is scoped to a verified account and one tab. */
(function(){
  "use strict";
  const prefix="acadia.cache.v1:", generationKey="acadia.cache.generation";
  const network=window.fetch.bind(window), memory=new Map(), pending=new Map();
  const freshFor=45000, keepFor=300000;
  let epoch=0;
  const eligible=p=>p==="/student/progress"||p==="/assignments"||p==="/solved"
    ||/^\/assignments\/[^/]+\/problems$/.test(p)||/^\/teacher\/assignments\/[^/]+\/grades$/.test(p);
  function clear(broadcast=false){
    epoch++;memory.clear();pending.clear();
    try{Object.keys(sessionStorage).filter(k=>k.startsWith(prefix)).forEach(k=>sessionStorage.removeItem(k));}catch{}
    if(broadcast)try{localStorage.setItem(generationKey,Date.now()+":"+Math.random());}catch{}
  }
  function read(key){
    let entry=memory.get(key);
    if(!entry)try{entry=JSON.parse(sessionStorage.getItem(key)||"null");}catch{}
    if(!entry||typeof entry.body!=="string"||!Number.isFinite(entry.at)||Date.now()-entry.at>keepFor||Date.now()<entry.at)return null;
    memory.set(key,entry);return entry;
  }
  function write(key,entry){
    if(entry.body.length>750000)return;
    memory.set(key,entry);
    // Bound in-memory and persisted results; quota failures never break fetch.
    while(memory.size>24)memory.delete(memory.keys().next().value);
    try{
      sessionStorage.setItem(key,JSON.stringify(entry));
      const rows=Object.keys(sessionStorage).filter(k=>k.startsWith(prefix)).map(k=>{
        try{return {key:k,value:JSON.parse(sessionStorage.getItem(k))};}
        catch{return {key:k,value:{at:0,body:""}};}
      }).sort((a,b)=>b.value.at-a.value.at);
      let size=0;
      rows.forEach((r,i)=>{size+=(r.value.body||"").length;if(i>=24||size>1500000||Date.now()-r.value.at>keepFor){sessionStorage.removeItem(r.key);memory.delete(r.key);}});
    }catch{}
  }
  function cachedResponse(entry,state){
    return new Response(entry.body,{status:200,headers:{"Content-Type":"application/json","X-Acadia-Cache":state}});
  }
  async function refresh(key,url,init,previous){
    const requestEpoch=epoch, existing=pending.get(key);
    if(existing?.epoch===requestEpoch)return existing.promise;
    const promise=(async()=>{
      const r=await network(url,{...init,cache:"no-store"});
      const body=await r.text(), entry={body,at:Date.now()};
      if(r.status===200&&requestEpoch===epoch){
        try{
          JSON.parse(body);write(key,entry);
          if(previous&&previous.body!==body)window.dispatchEvent(new CustomEvent("acadia:cache-update",{detail:{url:new URL(url,location.href).pathname}}));
        }catch{}
      }
      return {body,status:r.status,statusText:r.statusText,headers:r.headers};
    })();
    pending.set(key,{epoch:requestEpoch,promise});
    try{return await promise;}
    finally{if(pending.get(key)?.promise===promise)pending.delete(key);}
  }
  window.AcadiaCache={clear,invalidate:()=>clear(true),progressURL(){
    let zone="UTC";try{zone=Intl.DateTimeFormat().resolvedOptions().timeZone||"UTC";}catch{}
    return API+"/student/progress?timezone="+encodeURIComponent(zone);
  }};
  window.addEventListener("storage",event=>{
    if(event.key===generationKey||event.key===null){clear();Session.verified=null;Session.verifiedAt=0;}
  });
  window.fetch=async(input,init={})=>{
    const url=new URL(typeof input==="string"||input instanceof URL?input:input.url,location.href);
    const method=String(init.method||(typeof input==="object"&&input.method)||"GET").toUpperCase();
    const sameOrigin=url.origin===location.origin;
    if(!sameOrigin||method!=="GET"||!eligible(url.pathname)||init.signal||(typeof Request!=="undefined"&&input instanceof Request)){
      const r=await network(input,init);
      if(sameOrigin&&r.ok&&!["GET","HEAD","OPTIONS"].includes(method))clear(true);
      return r;
    }
    const me=Session.verified&&Date.now()-Session.verifiedAt<30000?Session.verified:await Session.check();
    if(!me||me.student_id==null)return network(input,init);
    const key=prefix+encodeURIComponent(String(me.student_id)+":"+me.role)+":"+url.pathname+url.search;
    const entry=read(key),force=init.cache==="reload"||init.cache==="no-store";
    if(entry&&!force){
      const stale=Date.now()-entry.at>=freshFor;
      if(stale)refresh(key,url.pathname+url.search,init,entry).then(result=>{
        if(result.status<200||result.status>=300)throw new Error("Refresh unavailable");
        JSON.parse(result.body);
      }).catch(()=>window.dispatchEvent(new CustomEvent("acadia:cache-error",{detail:{url:url.pathname}})));
      return cachedResponse(entry,stale?"stale":"fresh");
    }
    const result=await refresh(key,url.pathname+url.search,init,entry);
    return new Response(result.body,result);
  };
})();
