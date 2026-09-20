/* Per-account, per-browser onboarding. Never writes course records. */
(function(){
  "use strict";
  let account=null;
  const key=me=>"acadia.tutorial.v1:"+encodeURIComponent(String(me.student_id)+":"+me.role);
  function remembered(me){try{return localStorage.getItem(key(me))||sessionStorage.getItem(key(me));}catch{return null;}}
  function remember(status){
    if(!account)return;
    const value=JSON.stringify({status,at:new Date().toISOString()});
    try{localStorage.setItem(key(account),value);}catch{try{sessionStorage.setItem(key(account),value);}catch{}}
  }
  function destination(value){
    const fallback=account?.role==="teacher"?"teacher.html":"dashboard.html";
    try{
      const url=new URL(value||fallback,location.href),page=url.pathname.split("/").pop();
      if(url.origin!==location.origin||!["dashboard.html","student.html","student-grades.html","teacher.html","teacher-assignments.html","grades.html","playground.html"].includes(page))return fallback;
      if(account?.role!=="teacher"&&["teacher.html","teacher-assignments.html","grades.html","playground.html"].includes(page))return "dashboard.html";
      return page+url.search+url.hash;
    }catch{return fallback;}
  }
  function start(){
    remember("started");
    location.assign("tutorial.html?return="+encodeURIComponent(destination(location.pathname+location.search+location.hash)));
  }
  function dialog(id,title,markup){
    document.getElementById(id)?.remove();
    const el=document.createElement("dialog"),previous=document.activeElement;
    el.id=id;el.className="acadia-dialog";el.innerHTML=markup;el.setAttribute("aria-labelledby",title);
    document.body.append(el);
    el.addEventListener("close",()=>{if(previous?.isConnected)previous.focus({preventScroll:true});});
    el.querySelectorAll("[data-close-dialog]").forEach(b=>b.onclick=()=>el.close());
    el.addEventListener("click",e=>{if(e.target!==el)return;const r=el.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)el.close();});
    el.showModal();return el;
  }
  /* Settings. The section headings say what each control is for, so the
     sentence under each one was restating its own title; only the note about
     what the tutorial remembers survives, because nothing else says it. */
  function settingsMarkup(me){
    const value = v => esc(v || "");
    return '<div class="dialog-heading"><h2 id="settingsTitle">Settings</h2>'
      + '<button type="button" class="ghost" data-close-dialog aria-label="Close settings">×</button></div>'
      + '<section class="settings-section"><h3>Your name</h3>'
      + '<form id="nameForm" class="settings-name" novalidate>'
      + '<label>First name<input id="firstName" name="first_name" type="text" maxlength="60" autocomplete="given-name" value="' + value(me && me.first_name) + '"></label>'
      + '<label>Last name<input id="lastName" name="last_name" type="text" maxlength="60" autocomplete="family-name" value="' + value(me && me.last_name) + '"></label>'
      + '<button id="saveName" type="submit">Save name</button></form>'
      + '<p class="settings-note" id="nameMsg" role="status" aria-live="polite">This is the name your instructor sees on your work.</p></section>'
      + '<section class="settings-section"><h3>Appearance</h3>'
      + '<div id="mtsTheme" class="settings-theme" role="group" aria-label="Color theme">'
      + '<button type="button" class="ghost" data-t="light">Light</button>'
      + '<button type="button" class="ghost" data-t="dark">Dark</button></div></section>'
      + '<section class="settings-section"><h3>Guided practice</h3>'
      + '<button id="retakeTutorial" type="button">Retake tutorial</button>'
      + '<p class="settings-note">Tutorial completion is remembered for your account on this browser. Practice never affects your course grades.</p></section>';
  }
  async function saveName(el){
    const msg=el.querySelector("#nameMsg"),save=el.querySelector("#saveName");
    const first=el.querySelector("#firstName").value.trim(),
          last=el.querySelector("#lastName").value.trim();
    if(!first||!last){msg.textContent="Enter your first and last name.";return;}
    setBusy(save,true,"Saving…");
    try{
      const r=await fetch(`${API}/auth/name`,{method:"POST",
        headers:{"Content-Type":"application/json"},credentials:"include",
        body:JSON.stringify({first_name:first,last_name:last})});
      const body=await r.json().catch(()=>({}));
      if(!r.ok)throw new Error(body?.detail?.message||"Could not save your name.");
      // Redraw rather than reload: the account chip and the greeting both read
      // the session copy, and leaving it stale is how a saved name looks like
      // it did not save.
      account=Session.set(body);Session.verified=body;
      if(typeof remountHeader==="function")remountHeader();
      document.dispatchEvent(new CustomEvent("acadia:name-changed",{detail:body}));
      msg.textContent="Saved.";
    }catch(e){msg.textContent=e.message;}
    finally{setBusy(save,false);}
  }
  async function openSettings(){
    const me=account||Session.get();
    const el=dialog("acadiaSettings","settingsTitle",settingsMarkup(me));
    el.querySelectorAll("[data-t]").forEach(b=>b.onclick=()=>Theme.set(b.dataset.t));
    syncThemeControls(Theme.get());el.querySelector("#retakeTutorial").onclick=start;
    el.querySelector("#nameForm").addEventListener("submit",e=>{e.preventDefault();saveName(el);});
    // The cached copy may predate a name set in another tab, so fill the fields
    // from the server once it answers - but never over something being typed.
    const fresh=Session.verified||await Session.check();
    if(!fresh||!el.isConnected)return;
    account=fresh;
    const f=el.querySelector("#firstName"),l=el.querySelector("#lastName");
    if(document.activeElement!==f&&!f.value)f.value=fresh.first_name||"";
    if(document.activeElement!==l&&!l.value)l.value=fresh.last_name||"";
  }
  async function init(){
    account=Session.verified||await Session.check();
    if(!account||account.student_id==null)return;
    const page=location.pathname.split("/").pop();
    if(!["dashboard.html","teacher.html"].includes(page)||remembered(account))return;
    remember("offered"); // Once on a landing page, never while solving.
    // THIS DESCRIBES THE PAGE IT OPENS, and for a while it did not: the
    // tutorial stopped being a factorial worksheet and became a tour of the
    // product, and this modal - the very first thing a new student sees - went
    // on advertising an exercise that no longer exists. Kept in the same words
    // as tutorial.html so the promise and the page agree.
    const el=dialog("acadiaWelcome","welcomeTitle",'<p class="eyebrow">WELCOME TO ACADIA</p><h2 id="welcomeTitle">How ACADIA works.</h2><p>'
      +(account.role==="teacher"?"See exactly what your students see: what this site is, how a problem is worked, and the rules it holds them to. Then return to your course workspace.":"A guided tour of what this site is, what you do here, and how each part of it works - worth five minutes before you open your first assignment.")
      +'</p><ol class="welcome-path"><li>What ACADIA is, and where everything lives</li><li>Finding your assignments and your problems</li><li>The three stages of a problem: plan, code, reflect</li><li>The rules - no answer reveal, no attempt limit, nothing to hand in</li></ol><div class="dialog-actions"><button class="ghost" id="skipTutorial" type="button">Maybe later</button><button id="beginTutorial" type="button">Take the tour →</button></div><p class="settings-note">About 5 minutes, and nothing in it is locked - jump to any chapter. You can always retake it in Settings.</p>');
    el.querySelector("#beginTutorial").onclick=start;
    el.querySelector("#skipTutorial").onclick=()=>{remember("skipped");el.close();};
  }
  window.AcadiaOnboarding={openSettings,start,destination,
    async ready(){account=Session.verified||await Session.check();return account;},
    finish(){remember("completed");},dismiss(){remember("skipped");}};
  document.addEventListener("click",event=>{
    if(event.target.closest("[data-open-settings]")){
      const menu=document.getElementById("whoMenu");if(menu)menu.hidden=true;
      document.getElementById("whoBtn")?.setAttribute("aria-expanded","false");openSettings();
    }
  });
  if(document.readyState==="loading")document.addEventListener("DOMContentLoaded",init,{once:true});else init();
})();
