/* A TOUR OF ACADIA, not a programming exercise.
 *
 * This used to be a factorial worksheet: pick the base case, order four steps,
 * fill three dropdowns, say why `result = 1`. A student who finished it had
 * practised factorial and still did not know that the editor is locked until a
 * plan is reviewed, that the tutor will not answer a direct question, or that
 * they can retry a step for ever. Those are the things this app does
 * differently from every other site they have used, and they were in a
 * sidebar.
 *
 * So every question here is about USING ACADIA, and each one is aimed at a
 * belief a new student actually arrives with:
 *   1. "I'll just start typing"        -> the plan gate
 *   2. "I'll say I'll loop over it"    -> what a workable plan is
 *   3. "it'll show me after 2 tries"   -> nothing is ever revealed
 *   4. "I'd better not close the tab"  -> the session is saved
 *
 * No grading, tutor or session API is called from this page.
 */
(function(){
  "use strict";
  requireSession();
  const t=id=>document.getElementById(id);
  let step=0;
  const answers={first:"",plan:"",wrong:"",saved:""};

  // The same shape main/graphs.py emits, so the drawing on the plan step is the
  // real artifact a student will see rather than a picture of one.
  const graph={nodes:[{id:"s",kind:"start",label:"Take the dictionary"},
    {id:"a",kind:"step",label:"Count how often each value appears"},
    {id:"b",kind:"loop",label:"For each key and value"},
    {id:"c",kind:"branch",label:"Does this value appear once?"},
    {id:"d",kind:"step",label:"Add value → key to the result"},
    {id:"e",kind:"return",label:"Return the result"}],
    edges:[{src:"s",dst:"a"},{src:"a",dst:"b"},{src:"b",dst:"c",label:"next"},
      {src:"c",dst:"d",label:"yes"},{src:"d",dst:"b",label:"repeat"},
      {src:"c",dst:"b",label:"no"},{src:"b",dst:"e",label:"done"}]};

  const coaches=[
    '<p>Most sites let you type code straight away. This one does not, and that is the whole idea.</p>'
    +'<p>Every problem runs in three stages: <strong>Question &amp; plan</strong>, then <strong>Code</strong>, then <strong>Reflect</strong>. The editor stays locked until your plan has been looked at.</p>'
    +'<p>It is not a hoop. A plan you can explain is one you can debug.</p>',

    '<p>You build the plan by talking through your approach in the chat. As you explain it, the page draws it as a flowchart beside you.</p>'
    +'<p>The tutor will push back with questions. It is not being difficult - it has not been shown a solution and genuinely cannot hand you one.</p>'
    +'<p>Prefer paper? <strong>Upload a plan</strong> takes a photo or a PDF of a diagram instead.</p>',

    '<p>Once your plan is accepted the editor unlocks and the problem arrives in <strong>steps</strong>. You answer one at a time.</p>'
    +'<p>What you have already had accepted sits frozen above the editor, so you can always see the function taking shape.</p>'
    +'<p>Indentation is handled for you - you never have to guess how far in a step should sit.</p>',

    '<p>At the end you see your plan and your finished code side by side. The gap between them is usually where the interesting bug was.</p>'
    +'<p><strong>My grades</strong> shows your recorded step credit. <strong>Full file</strong>, in the coding bar, shows the whole assignment file with your answers in it.</p>'
    +'<p>That is everything. Open <strong>Assignments</strong> when you are ready.</p>'
  ];

  function message(text,kind="info"){
    t("tutorialFeedback").className="tutorial-feedback "+kind;
    t("tutorialFeedback").textContent=text;
  }

  /* One radio group. `name` doubles as the key in `answers`. */
  function choices(name,legend,options){
    return '<fieldset class="tutorial-choices"><legend>'+esc(legend)+'</legend>'
      +options.map(([value,label])=>'<label><input type="radio" name="'+name+'" value="'
        +esc(value)+'"'+(answers[name]===value?' checked':'')+'><span>'+label+'</span></label>').join("")
      +'</fieldset>';
  }

  function wireChoices(root,name){
    root.querySelectorAll('[name="'+name+'"]').forEach(el=>el.onchange=()=>{
      answers[name]=el.value;message("");
      // Seeing the flowchart IS the lesson on this step: a plan with enough in
      // it draws something, and a thin one does not. Drawn on selection, not on
      // Continue - Continue re-renders the page and would wipe it unseen.
      if(name==="plan"){
        const box=t("practicePlanGraph");
        if(el.value==="full"){
          renderGraph(box,graph,"",{height:300});
          message("That is the flowchart the page would draw while you typed that.","success");
        }else box.replaceChildren();
      }
    });
  }

  function render(){
    message("");
    t("tutorialBack").hidden=step===0;
    t("tutorialNext").textContent=step===3?"Finish tour →":"Continue →";
    t("tutorialPosition").textContent="Step "+(step+1)+" of 4";
    document.querySelectorAll("[data-tutorial-step]").forEach(el=>{
      if(Number(el.dataset.tutorialStep)===step)el.setAttribute("aria-current","step");
      else el.removeAttribute("aria-current");
      el.classList.toggle("complete",Number(el.dataset.tutorialStep)<step);
    });
    t("tutorialCoach").innerHTML=coaches[step];
    const root=t("tutorialExercise");

    if(step===0){
      root.innerHTML='<p class="eyebrow">1 · HOW A PROBLEM WORKS</p>'
        +'<h2 id="tutorialTitle">Plan first, then code.</h2>'
        +'<p>Opening a problem puts you on the first of three stages. You move through them in order, and the coding stage is locked until your plan has been reviewed.</p>'
        +'<pre class="code">1  Question &amp; plan   read it, talk your approach through, submit the plan\n'
        +'2  Code              the editor unlocks; answer the problem one step at a time\n'
        +'3  Reflect           your plan and your code, side by side</pre>'
        +choices("first","You have just opened a problem. What can you do first?",[
          ["code","Start typing code in the editor"],
          ["plan","Talk through how you would approach it"],
          ["answer","Look at the worked solution"]]);
      wireChoices(root,"first");

    }else if(step===1){
      root.innerHTML='<p class="eyebrow">2 · WHAT GETS YOUR PLAN ACCEPTED</p>'
        +'<h2 id="tutorialTitle">Say enough to be checked.</h2>'
        +'<p>A plan is accepted when it says four things, in your own words: what you keep track of, how you go through the input, how you produce the answer, and what happens in the awkward case. It does not have to be clever, or the approach anyone else would pick - it has to hold up.</p>'
        +'<p>Here are two messages about the same problem: <em>swap the keys and values of a dictionary, keeping only the values that appear exactly once</em>.</p>'
        +choices("plan","Which one gets you to the editor?",[
          ["thin","<strong>A.</strong> “I’ll loop through the dictionary and swap the keys and the values.”"],
          ["full","<strong>B.</strong> “I’ll count how many times each value shows up, keep the ones that appear exactly once, and build a new dictionary with the value as the key. An empty dictionary gives back an empty dictionary.”"]])
        +'<div id="practicePlanGraph"></div>';
      wireChoices(root,"plan");

    }else if(step===2){
      root.innerHTML='<p class="eyebrow">3 · ANSWERING A STEP</p>'
        +'<h2 id="tutorialTitle">Wrong is not the end of it.</h2>'
        +'<p>Each step is checked by actually running your code against hidden tests. If it does not pass you are told which cases failed, and you can try again as many times as you like.</p>'
        +'<p>The answer is <strong>never</strong> shown to you. There is no attempt limit to run out of, and no reveal waiting at the end of one - being stuck is where the learning is, so the site will not end it for you.</p>'
        +'<pre class="code">Your solution runs but gives the wrong answer on at least one case.\n'
        +'  ▸ Show 3 of the 7 cases it failed</pre>'
        +choices("wrong","Your step comes back wrong for the second time. What happens?",[
          ["reveal","The correct answer is filled in for you"],
          ["retry","You see the failing cases and can keep trying"],
          ["lock","The problem locks until your instructor reopens it"]]);
      wireChoices(root,"wrong");

    }else{
      root.innerHTML='<p class="eyebrow">4 · FINISHING, AND COMING BACK</p>'
        +'<h2 id="tutorialTitle">Your work waits for you.</h2>'
        +'<p>Nothing here needs saving. Every accepted step, every message and every version of your plan is recorded as you go.</p>'
        +'<p>Two buttons sit under the editor while you work: <strong>Full file</strong> shows the whole assignment file with your answers already in place, and <strong>Start over</strong> clears a problem back to the beginning if you want a clean run at it.</p>'
        +choices("saved","You close the tab half way through a problem. You come back tomorrow.",[
          ["lost","You start that problem again from step 1"],
          ["kept","You pick up on the step you had reached, with your accepted steps still there"]])
        +'<div class="practice-completion"><strong>That is the tour.</strong>'
        +'<p>This page closes when you finish it. You can take it again from <strong>Settings</strong> whenever you want.</p></div>';
      wireChoices(root,"saved");
    }
    if(step)root.focus({preventScroll:true});
  }

  function leave(completed){
    if(completed)AcadiaOnboarding.finish();else AcadiaOnboarding.dismiss();
    location.replace(AcadiaOnboarding.destination(
      new URLSearchParams(location.search).get("return")));
  }

  // A wrong answer is TEACHING, not a gate - each one names the belief it is
  // correcting rather than saying "try again".
  const gates=[
    {key:"first",right:"plan",
     wrong:{code:"Not yet - the editor is locked until your plan has been reviewed. Talking your approach through is what unlocks it.",
            answer:"There isn't one to look at. Nothing on this site will show you a solution, at any point."}},
    {key:"plan",right:"full",
     wrong:{thin:"A says what to do but not how it holds up: nothing about what is counted, or what happens when a value appears twice. The tutor would ask about both."}},
    {key:"wrong",right:"retry",
     wrong:{reveal:"No - the answer is never filled in, however many times you miss. You get the failing cases and another go.",
            lock:"Nothing locks. There is no attempt limit here at all."}},
    {key:"saved",right:"kept",
     wrong:{lost:"Not any more - your session is saved as you go, and reopening a problem puts you back on the step you had reached."}}
  ];

  t("leaveTutorial").onclick=()=>leave(false);
  t("tutorialBack").onclick=()=>{step--;render();};
  t("tutorialNext").onclick=()=>{
    const gate=gates[step],picked=answers[gate.key];
    if(!picked){message("Pick one to carry on.","warn");return;}
    if(picked!==gate.right){message(gate.wrong[picked]||"Not quite - have another look.","warn");return;}
    if(step===3){leave(true);return;}
    step++;render();
    t("tutorialExercise").scrollIntoView({block:"start",behavior:"instant"});
  };

  AcadiaOnboarding.ready().then(me=>{
    if(!me)return;
    mountHeader({variant:me.role==="teacher"?"instructor":"student",active:"Tutorial"});
    setCrumbs([{label:"How ACADIA works"}]);
    render();
  });
})();
