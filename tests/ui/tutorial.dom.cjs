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
    w.fetch=async(url,init={})=>{calls.push({url:String(url),method:init.method||"GET"});return new w.Response(JSON.stringify({name:"Practice User",student_id:"practice-"+role,role}),{status:200});};
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
    returning.document.querySelector(".sidebar-settings").click();
    assert(returning.document.querySelector("#acadiaSettings").open);
    assert(returning.document.querySelector("#retakeTutorial"),"Retake remains available");
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
    const teacher=await make("tutorial.html","teacher");
    assert.equal(teacher.document.body.dataset.portal,"instructor");
    assert.equal(teacher.AcadiaOnboarding.destination("grades.html"),"grades.html");
    assert(calls.every(c=>c.method==="GET"&&c.url.includes("/auth/me")),"Tutorial never creates a real session or grade");
    console.log("PASS: first-visit detection, account/browser memory, Settings replay, the four product-tour gates (plan lock, workable plan, never revealed, work is saved), automatic exit, role-aware navigation, and zero course writes.");
  }finally{for(const w of windows)await w.happyDOM.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
