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
    const rail=()=>[...doc.querySelectorAll("[data-tutorial-step]")];
    const at=()=>doc.querySelector("#tutorialPosition").textContent;
    // A choice's handler is onchange, and a synthetic click does not reliably
    // fire one, so the change is dispatched too. The handlers are idempotent,
    // so a DOM that fires both is not a different test.
    const pick=(name,value)=>{
      const el=doc.querySelector('input[name="'+name+'"][value="'+value+'"]');
      el.click();el.checked=true;
      el.dispatchEvent(new w.Event("change",{bubbles:true}));
      return el;
    };
    const whyFor=key=>doc.querySelector('[data-quiz="'+key+'"] .quiz-why');

    // NOTHING GATES, which is the rule the rewrite was built on. The previous
    // tour refused to advance until you picked the right radio, so the first
    // thing a brand-new student met was a quiz about a product they had not
    // seen yet. Continue continues with nothing picked.
    assert.equal(rail().length,6,"one rail tab per chapter");
    assert.equal(at(),"Chapter 1 of 6");
    assert(doc.querySelector("#tutorialBack").hidden,"nowhere to go back to yet");
    next();
    assert.equal(at(),"Chapter 2 of 6","Continue needs no answer");
    assert(!doc.querySelector("#tutorialBack").hidden);
    // ...and the rail is a rail, not a ladder: any chapter, at any time.
    rail()[5].click();
    assert.equal(at(),"Chapter 6 of 6","the rail jumps straight to the end");
    assert.equal(doc.querySelector("#tutorialNext").textContent,"Finish tour \u2192");
    assert.equal(rail()[5].getAttribute("aria-current"),"step");
    assert(rail()[0].classList.contains("complete"),"chapters behind you read as done");

    // THE PLAN CHAPTER DRAWS THE REAL THING - renderGraph, the same drawing
    // main/graphs.py feeds on the live page - and a thin plan draws nothing,
    // which is the lesson rather than a missing feature.
    rail()[2].click();
    assert(!doc.querySelector("#practicePlanGraph svg"),"nothing drawn before a choice");
    pick("planDemo","thin");
    assert(!doc.querySelector("#practicePlanGraph svg"),"a thin plan draws nothing");
    assert.equal(doc.querySelector("#planDemoWhy").className,"quiz-why no");
    assert(doc.querySelector("#planDemoWhy").textContent.includes("draws nothing"));
    pick("planDemo","full");
    assert(doc.querySelector("#practicePlanGraph svg"),"the workable plan is drawn");
    assert.equal(doc.querySelector("#planDemoWhy").className,"quiz-why ok");

    // THE CHECK COSTS NOTHING: a wrong pick explains itself and blocks nobody.
    // The four beliefs it corrects are the four the product rests on.
    rail()[5].click();
    assert.equal(whyFor("first").textContent,"","nothing is said before a pick");
    pick("first","code");
    assert(whyFor("first").textContent.includes("locked"),"the plan gate is the point of stage 1");
    assert.equal(whyFor("first").className,"quiz-why no");
    pick("first","plan");
    assert.equal(whyFor("first").className,"quiz-why ok");
    pick("plan","fail");
    assert(whyFor("plan").textContent.includes("rebuilt"),"a different approach gets its own steps");
    pick("wrong","reveal");
    assert(whyFor("wrong").textContent.startsWith("Never"),
      "nothing is ever revealed, and the tour has to say so");
    pick("saved","lost");
    assert(whyFor("saved").textContent.includes("saved"));
    // An answer survives leaving the chapter and coming back. Every move
    // re-renders the chapter from scratch, so without this a student who went
    // back to re-read one thing would find four questions blank again.
    rail()[0].click();rail()[5].click();
    assert(doc.querySelector('input[name="wrong"][value="reveal"]').checked);
    assert(whyFor("wrong").textContent.startsWith("Never"),"and its answer is still there");

    // THE ONE PROMISE THE TOUR MAY NOT BREAK. sessions.MAX_ATTEMPTS is None
    // and no tier reveals a reference, so a tour that hinted at a reveal or at
    // running out of tries would be teaching a product we do not ship.
    rail()[3].click();
    const codeChapter=doc.querySelector("#tutorialExercise").textContent;
    assert(codeChapter.includes("The answer is never shown to you"),codeChapter.slice(0,120));
    assert(/no attempt limit/i.test(codeChapter),"and no limit to run out of");

    // Mocks are scenery: hidden from assistive tech, and nothing inside them
    // is tabbable, so a keyboard user does not walk through screenshots.
    const mocks=[...doc.querySelectorAll(".mock")];
    assert(mocks.length&&mocks.every(m=>m.getAttribute("aria-hidden")==="true"),
      "every mock is aria-hidden");
    assert.equal(doc.querySelectorAll(".mock button,.mock a,.mock input,.mock [tabindex]").length,0,
      "and holds nothing focusable");

    rail()[5].click();next();await tick();
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
    console.log("PASS: first-visit detection, account/browser memory, Settings replay, six ungated chapters reachable from the rail, Continue without an answer, the real plan flowchart (thin draws nothing), the four self-check corrections with answers that persist across chapters, the never-revealed promise in the copy, inert mocks, automatic exit, role-aware navigation, and zero course writes.");
  }finally{for(const w of windows)await w.happyDOM.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
