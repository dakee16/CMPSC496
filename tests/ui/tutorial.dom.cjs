const assert=require("node:assert/strict"),fs=require("node:fs"),path=require("node:path");
const {pathToFileURL}=require("node:url");
const read=name=>fs.readFileSync(path.resolve(__dirname,"../../frontend",name),"utf8");
const tick=()=>new Promise(r=>setTimeout(r,30));
(async()=>{
  const {Window}=await import(pathToFileURL(require.resolve("happy-dom")).href);
  const windows=[],calls=[];
  async function make(page,role="student",previous){
    const w=new Window({url:"http://localhost/"+page,width:1400,height:1000,settings:{enableJavaScriptEvaluation:true,disableJavaScriptFileLoading:true,disableCSSFileLoading:true,suppressInsecureJavaScriptEnvironmentWarning:true}});
    windows.push(w);w.scrollTo=()=>{};
    if(previous)for(const k of Object.keys(previous.localStorage))w.localStorage.setItem(k,previous.localStorage.getItem(k));
    w.document.write(read(page.split("?")[0]).replace(/<script\b[^>]*>[\s\S]*?<\/script>/g,""));
    const account={name:"Practice User",first_name:"Practice",last_name:"User",
                   student_id:"practice-"+role,role};
    w.fetch=async(url,init={})=>{
      calls.push({url:String(url),method:init.method||"GET"});
      // /auth/name answers with the renamed account, exactly as the route does.
      if(String(url).endsWith("/auth/name")){
        const sent=JSON.parse(init.body||"{}");
        Object.assign(account,{first_name:sent.first_name,last_name:sent.last_name,
                               name:`${sent.first_name} ${sent.last_name}`});
      }
      return new w.Response(JSON.stringify(account),{status:200});
    };
    w.eval(["ui.js","cache.js","onboarding.js",...(page.startsWith("tutorial")?["graphs.js","tutorial.js"]:[])].map(read).join("\n"));
    if(!page.startsWith("tutorial"))w.eval('requireSession();mountHeader();');
    await tick();return w;
  }
  try{
    const landing=await make("dashboard.html"),d=landing.document;
    assert(d.querySelector("#acadiaWelcome").open,"First visit offers tutorial");
    d.querySelector("#skipTutorial").click();
    const returning=await make("dashboard.html","student",landing);
    assert(!returning.document.querySelector("#acadiaWelcome"),"Remember first visit");
    returning.document.querySelector("#whoBtn").click();
    returning.document.querySelector('#whoMenu [data-open-settings]').click();
    const settings=returning.document.querySelector("#acadiaSettings");
    assert(settings.open);
    assert(settings.querySelector("#retakeTutorial"),"Retake remains available");
    // Each section is its own heading plus its control. The only prose left is
    // the note about what the tutorial remembers, which nothing else says.
    assert.equal(settings.querySelectorAll("p:not(.settings-note)").length,0,
      "settings sections carry no restated descriptions");

    // The name is editable, prefilled from the account, and what gets saved is
    // what the header redraws with - a saved name that leaves a stale chip
    // behind reads as a save that did not happen.
    assert.equal(settings.querySelector("#firstName").value,"Practice");
    assert.equal(settings.querySelector("#lastName").value,"User");
    settings.querySelector("#firstName").value="Ada";
    settings.querySelector("#lastName").value="Lovelace";
    settings.querySelector("#nameForm").dispatchEvent(
      new returning.Event("submit",{bubbles:true,cancelable:true}));
    await tick();
    assert(calls.some(c=>c.url.endsWith("/auth/name")&&c.method==="POST"),"name is saved server-side");
    assert.equal(returning.document.querySelector(".who .nm").textContent,"Ada Lovelace");
    assert.equal(JSON.parse(returning.sessionStorage.getItem("microtutor.session")).first,"Ada");
    assert.equal(settings.querySelector("#nameMsg").textContent,"Saved.");

    // Sign out: one item, and it actually calls the route and leaves the page.
    const menu=returning.document.querySelector("#whoMenu");
    assert.deepEqual([...menu.querySelectorAll("[role=menuitem]")].map(b=>b.textContent),
      ["Settings","Sign out"],"the account menu is the one place both live");
    returning.document.querySelector("#whoBtn").click();
    returning.document.querySelector("#miLogout").click();
    await tick();
    assert(calls.some(c=>c.url.endsWith("/logout")&&c.method==="POST"),"sign out calls /logout");
    assert.equal(returning.sessionStorage.getItem("microtutor.session"),null,"session copy is cleared");
    const w=await make("tutorial.html?return=student.html"),doc=w.document;
    const next=()=>doc.querySelector("#tutorialNext").click();
    const radio=(name,value)=>doc.querySelector('input[name="'+name+'"][value="'+value+'"]').click();
    // THE TOUR TEACHES THE PRODUCT, not factorial. Each step corrects one
    // belief a new student arrives with, and a wrong pick has to say which.
    next();assert(doc.querySelector("#tutorialFeedback").textContent.includes("Pick one"));
    radio("first","code");next();
    assert(doc.querySelector("#tutorialFeedback").textContent.includes("locked"),
      "the plan gate is the point of step 1");
    radio("first","plan");next();

    // Picking the workable plan draws the flowchart the page would have drawn.
    assert(!doc.querySelector("#practicePlanGraph svg"),"nothing drawn before a choice");
    radio("plan","thin");
    assert(!doc.querySelector("#practicePlanGraph svg"),"a thin plan draws nothing");
    next();assert(doc.querySelector("#tutorialFeedback").textContent.includes("counted"));
    radio("plan","full");assert(doc.querySelector("#practicePlanGraph svg"),"the plan is drawn");
    next();

    radio("wrong","reveal");next();
    assert(doc.querySelector("#tutorialFeedback").textContent.includes("never"),
      "nothing is ever revealed, and the tour has to say so");
    radio("wrong","retry");next();

    radio("saved","lost");next();
    assert(doc.querySelector("#tutorialFeedback").textContent.includes("saved"));
    radio("saved","kept");next();await tick();
    assert.equal(w.location.pathname,"/student.html","Finish closes sample");
    const marker=Object.keys(w.localStorage).find(k=>k.startsWith("acadia.tutorial"));
    assert.equal(JSON.parse(w.localStorage.getItem(marker)).status,"completed");
    assert.equal(w.AcadiaOnboarding.destination("https://evil.invalid"),"dashboard.html");
    assert.equal(w.AcadiaOnboarding.destination("teacher.html"),"dashboard.html");
    assert.equal(w.AcadiaOnboarding.destination("teacher-assignments.html"),"dashboard.html");
    const teacher=await make("tutorial.html","teacher");
    assert.equal(teacher.document.body.dataset.portal,"instructor");
    assert.equal(teacher.AcadiaOnboarding.destination("grades.html"),"grades.html");
    assert.equal(teacher.AcadiaOnboarding.destination("teacher-assignments.html"),"teacher-assignments.html");
    // The only writes in this whole file are the two ACCOUNT ones asserted
    // above. Anything else reaching the server would be the tutorial touching
    // a real session, submission or grade, which is the thing it must not do.
    const account=c=>c.url.endsWith("/auth/name")||c.url.endsWith("/logout");
    assert(calls.every(c=>account(c)||(c.method==="GET"&&c.url.includes("/auth/me"))),
      "Tutorial never creates a real session or grade");
    console.log("PASS: first-visit detection, account/browser memory, Settings replay, the four product-tour gates (plan lock, workable plan, never revealed, work is saved), automatic exit, role-aware navigation, and zero course writes.");
  }finally{for(const w of windows)await w.happyDOM.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
