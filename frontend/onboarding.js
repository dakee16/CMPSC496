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
      if(url.origin!==location.origin||!["dashboard.html","student.html","student-grades.html","teacher.html","grades.html","playground.html"].includes(page))return fallback;
      if(account?.role!=="teacher"&&["teacher.html","grades.html","playground.html"].includes(page))return "dashboard.html";
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
  function openSettings(){
    const el=dialog("acadiaSettings","settingsTitle",'<div class="dialog-heading"><h2 id="settingsTitle">Settings</h2><button type="button" class="ghost" data-close-dialog aria-label="Close settings">×</button></div><section class="settings-section"><h3>Appearance</h3><p>Choose the theme that feels comfortable to you.</p><div id="mtsTheme" class="settings-theme" role="group" aria-label="Color theme"><button type="button" class="ghost" data-t="light">Light</button><button type="button" class="ghost" data-t="dark">Dark</button></div></section><section class="settings-section"><h3>Guided practice</h3><p>Learn the question, plan, code and reflection workflow with a short factorial example.</p><button id="retakeTutorial" type="button">Retake tutorial</button><p class="settings-note">Tutorial completion is remembered for your account on this browser. Practice never affects your course grades.</p></section>');
    el.querySelectorAll("[data-t]").forEach(b=>b.onclick=()=>Theme.set(b.dataset.t));
    syncThemeControls(Theme.get());el.querySelector("#retakeTutorial").onclick=start;
  }
  async function init(){
    account=Session.verified||await Session.check();
    if(!account||account.student_id==null)return;
    const page=location.pathname.split("/").pop();
    if(!["dashboard.html","teacher.html"].includes(page)||remembered(account))return;
    remember("offered"); // Once on a landing page, never while solving.
    const el=dialog("acadiaWelcome","welcomeTitle",'<p class="eyebrow">WELCOME TO ACADIA</p><h2 id="welcomeTitle">Try one problem together.</h2><p>'
      +(account.role==="teacher"?"Explore the student experience with a short factorial exercise. Then return to your course workspace.":"Get comfortable with your tutor, working plan and coding steps in a short factorial exercise.")
      +'</p><ol class="welcome-path"><li>Understand the question</li><li>Build a working plan</li><li>Code and check your answer</li></ol><div class="dialog-actions"><button class="ghost" id="skipTutorial" type="button">Maybe later</button><button id="beginTutorial" type="button">Start guided practice →</button></div><p class="settings-note">About 3 minutes. You can always retake it in Settings.</p>');
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
