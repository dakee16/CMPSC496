/* A TOUR OF ACADIA, not a programming exercise.
 *
 * This page has one job: a student who has never seen the site should be able
 * to read it once and then work a real problem without guessing. So it is
 * written as an ANSWER TO THREE QUESTIONS, in this order:
 *
 *   what is this site        chapter 0
 *   where is my work         chapter 1
 *   how do I actually do it  chapters 2-4, one per stage of a problem
 *
 * and only then, once they have been told everything, a check they can get
 * wrong for free (chapter 5).
 *
 * Two rules it is built on, both learned from the version before it:
 *
 * NOTHING GATES. The old tour would not let you past a chapter until you
 * picked the right radio button, so the first thing a brand-new student met
 * was a quiz about a product they had not seen yet. Every chapter is reachable
 * from the rail at the top, at any time, and Continue always continues.
 *
 * SHOW THE ACTUAL SCREEN. Prose describing an interface is forgettable; the
 * interface is not. Each chapter carries a mock of the real thing - the stage
 * tabs, the problem list, the frozen-context editor, the pass/fail panel -
 * built from the same tokens as the real UI, with numbered notes pointing at
 * the parts that matter. The plan flowchart is not a mock at all: it is
 * renderGraph, the same drawing main/graphs.py feeds on the real page.
 *
 * No grading, tutor or session API is called from this page.
 */
(function(){
  "use strict";
  requireSession();
  const t=id=>document.getElementById(id);
  let step=0;

  /* ---------- small builders for the mock screenshots ---------- */

  // A framed "screenshot". Everything inside is inert markup; nothing in a
  // mock is ever focusable, so a keyboard user does not tab through scenery.
  const mock=(label,inner)=>'<figure class="mock" aria-hidden="true"><figcaption class="mock-bar">'
    +'<i></i><i></i><i></i><em>'+label+'</em></figcaption><div class="mock-body">'+inner+'</div></figure>';

  // Numbered notes under a mock. The numbers are the teaching - they are what
  // turns a picture into an explanation.
  const notes=items=>'<ol class="annots">'+items.map(([head,body])=>
    '<li><strong>'+head+'</strong><span>'+body+'</span></li>').join("")+'</ol>';

  const rules=items=>'<ul class="rulelist">'+items.map(([head,body])=>
    '<li><strong>'+head+'</strong><span>'+body+'</span></li>').join("")+'</ul>';

  /* ---------- the plan flowchart, drawn for real ---------- */

  // The same shape main/graphs.py emits, so the drawing on the plan chapter is
  // the real artifact a student will see rather than a picture of one.
  const graph={nodes:[{id:"s",kind:"start",label:"Take the dictionary"},
    {id:"a",kind:"step",label:"Count how often each value appears"},
    {id:"b",kind:"loop",label:"For each key and value"},
    {id:"c",kind:"branch",label:"Does this value appear once?"},
    {id:"d",kind:"step",label:"Add value → key to the result"},
    {id:"e",kind:"return",label:"Return the result"}],
    edges:[{src:"s",dst:"a"},{src:"a",dst:"b"},{src:"b",dst:"c",label:"next"},
      {src:"c",dst:"d",label:"yes"},{src:"d",dst:"b",label:"repeat"},
      {src:"c",dst:"b",label:"no"},{src:"b",dst:"e",label:"done"}]};

  /* ---------- the closing check ---------- */

  // Answering is optional and a wrong pick costs nothing: the point is to name
  // the belief a new student arrives with and replace it, not to score them.
  // Every one of these is answered somewhere above, on purpose.
  const quiz=[
    {key:"first",legend:"You have just opened a problem. What can you do first?",
     right:"plan",
     options:[["code","Start typing in the editor"],
       ["plan","Talk through how you would approach it"],
       ["answer","Look at the worked solution"]],
     why:{code:"The editor is locked until a plan has been reviewed - that lock is the whole idea of the site.",
       plan:"Right. Stage 1 is where you say what you intend to do; the editor unlocks when that plan holds up.",
       answer:"There isn't one to look at. Nothing here will show you a solution, at any point."}},
    {key:"plan",legend:"Your plan is approved, but it is not the approach your instructor used. What happens?",
     right:"reroute",
     options:[["fail","Your steps will ask for code you never intended to write"],
       ["reroute","The steps are rebuilt around your approach"],
       ["reject","It would not have been approved in the first place"]],
     why:{fail:"That was the old failure and it is exactly what rerouting fixes - the roadmap is rebuilt to match your plan.",
       reroute:"Right. A different-but-workable approach gets its own set of steps, checked against the same hidden tests.",
       reject:"No - your plan has to work, not to match. Recursion where the instructor looped is fine."}},
    {key:"wrong",legend:"Your step comes back wrong for the third time. What happens?",
     right:"retry",
     options:[["reveal","The correct answer is filled in for you"],
       ["retry","You see failing cases and can keep trying"],
       ["lock","The problem locks until your instructor reopens it"]],
     why:{reveal:"Never. The answer is not shown however many times you miss - being stuck is the part that teaches.",
       retry:"Right. Failing cases, an attempt count, and no limit on it.",
       lock:"Nothing locks, and there is no attempt limit here at all."}},
    {key:"saved",legend:"You close the tab half way through a problem and come back tomorrow.",
     right:"kept",
     options:[["lost","You start that problem again from step 1"],
       ["kept","You resume on the step you had reached"]],
     why:{lost:"Your session is saved as you go - accepted steps, chat and plan are all still there.",
       kept:"Right. Nothing here needs saving, and Start over is the only thing that clears a problem."}}];
  const picked={};

  /* ---------- the chapters ---------- */

  const chapters=[
  {
    tab:"What this is",
    eyebrow:"1 · WHAT ACADIA IS",
    title:"A place to practise Python by explaining yourself first.",
    coach:'<p>One sentence: <strong>ACADIA is where you do the Python your instructor sets, with a tutor beside you that will never hand you the answer.</strong></p>'
      +'<p>The thing that makes it different from every other site you have coded on is the order. You say what you are going to do, and only then do you get to type.</p>'
      +'<p>That is not a hoop. A plan you can explain is a plan you can debug.</p>',
    body:
      '<p>Your instructor publishes <strong>assignments</strong>. Each one holds a handful of <strong>problems</strong> - a function to write, a description of what it should do. You work through them here, in the browser, and what you finish is recorded for your course automatically.</p>'
      +'<p>Every problem runs through the same three stages, in order, and you can see where you are at all times:</p>'
      +mock("A problem, top of the page",
        '<div class="mock-stages">'
        +'<div class="mock-stage on"><b>1</b><span>Question &amp; plan<em>Read it, then work out your approach</em></span></div>'
        +'<div class="mock-stage lock"><b>2</b><span>Code<em>One step at a time</em></span><i>🔒</i></div>'
        +'<div class="mock-stage lock"><b>3</b><span>Reflect<em>See what you’ve learned</em></span><i>🔒</i></div></div>')
      +notes([
        ["Stage 2 is locked, and that is deliberate","You cannot type code until your plan has been looked at. Chapter 3 is all about getting through that gate."],
        ["You move forward, not around","Stages unlock in order. You can always go back to an earlier one to re-read your question or your plan."],
        ["A tutor sits on the right the whole time","It asks you questions about your thinking. It has never been shown the solution, so it could not give it to you even if you talked it into trying."]])
      +'<h3>Where everything lives</h3>'
      +'<p>Four places, and that is the whole site:</p>'
      +mock("The sidebar, on every page",
        '<div class="mock-nav"><div class="mock-brand"><i>◆</i><span>ACADIA<em>LEARNING STUDIO</em></span></div>'
        +'<div class="mock-navlabel">WORKSPACE</div>'
        +'<div class="mock-navitem on"><i>▦</i>Dashboard</div>'
        +'<div class="mock-navitem"><i>▤</i>My assignments</div>'
        +'<div class="mock-navitem"><i>▧</i>My grades</div>'
        +'<div class="mock-navfoot"><i>●</i><span>Your name<em>Account ▾</em></span></div></div>')
      +notes([
        ["Dashboard","Where you land. Your progress, and the problem to pick back up."],
        ["My assignments","Everything published to you, and the way in to any problem."],
        ["My grades","Step-by-step credit for what you have solved, as your instructor sees it."],
        ["Account ▾ → Settings","Your name, light or dark, and <strong>Retake tutorial</strong> - this page is always here."]])
  },
  {
    tab:"Finding work",
    eyebrow:"2 · FINDING YOUR WORK",
    title:"Assignment, then problem, then in.",
    coach:'<p>Two clicks from the sidebar to a problem: <strong>My assignments → an assignment → a problem</strong>.</p>'
      +'<p>Nothing is ever hidden from you. Every problem in a published assignment is open from the moment it appears - work them in any order you like.</p>'
      +'<p>If you just want to carry on where you left off, the Dashboard names the problem for you.</p>',
    body:
      '<p>Open <strong>My assignments</strong> and you get everything your instructor has published, with your progress on each. Open one and you get its problems:</p>'
      +mock("My assignments → Week 4 · Dictionaries",
        '<div class="mock-handback"><span><b>Your file so far</b>Your completed answers for 3 of 6, filled in at the right place.</span><i>Download .py</i></div>'
        +'<div class="mock-sumrow"><span>3 of 6 solved</span><div class="mock-progress"><i style="width:50%"></i></div></div>'
        +'<div class="mock-chips"><b class="on">All</b><b>Not started</b><b>In progress</b><b>Solved</b></div>'
        +'<div class="mock-rows">'
        +'<div class="mock-row"><span>count_words</span><b class="ok">Solved</b></div>'
        +'<div class="mock-row"><span>invert_dict</span><b class="mid">In progress</b></div>'
        +'<div class="mock-row"><span>merge_counts</span><b>Not started</b></div>'
        +'<div class="mock-row"><span>top_k_frequent</span><b>Not started</b></div></div>')
      +notes([
        ["Your file, from the first problem you finish","You do not have to wait for 100%. The download is a runnable Python file with your answers filled in under each function, and the ones you have not done left as they were handed out."],
        ["The status is the memory","Clicking an <strong>In progress</strong> row puts you back exactly where you stopped - same step, same code above it, same conversation."],
        ["Filter and sort","The chips narrow the list; the sort box beside them reorders it - by file, by name, or by status. Useful once an assignment has a dozen problems in it."],
        ["Order is yours to pick","Every problem in a published assignment is open from the moment you can see it. Nothing has to be solved before anything else."]])
      +'<p class="aside-note">Your instructor sees this same progress from their side, as it happens. There is no separate thing to submit or hand in.</p>'
  },
  {
    tab:"1 · Plan",
    eyebrow:"3 · STAGE ONE",
    title:"Question &amp; plan: say what you are going to do.",
    coach:'<p>You build a plan by <strong>talking through your approach in the chat</strong>, in ordinary words. No syntax, no code.</p>'
      +'<p>The tutor pushes back with questions. It is not being difficult - it has not been shown a solution and genuinely cannot hand you one, so questions are all it has.</p>'
      +'<p>Prefer paper? <strong>Upload a plan</strong> takes a photo or a PDF of a diagram instead.</p>',
    body:
      '<p>Stage 1 shows you the question, the exact function you have to write, and the tutor. You talk; it asks; the page draws.</p>'
      +mock("Stage 1 · invert_dict",
        '<div class="mock-split"><div class="mock-pane"><div class="mock-pane-title">The question</div>'
        +'<div class="mock-sig">def invert_dict(d: dict) -&gt; dict:</div>'
        +'<p class="mock-text">Swap the keys and values of a dictionary, keeping only the values that appear exactly once.</p></div>'
        +'<div class="mock-pane chat"><div class="mock-pane-title">✦ Your tutor</div>'
        +'<div class="mock-msg me">I’ll loop through the dictionary and swap the keys and the values.</div>'
        +'<div class="mock-msg">Two of the values are <code>3</code>. When you reach the second one, what does your new dictionary already have under the key <code>3</code>?</div>'
        +'<div class="mock-msg me">Oh - it would overwrite it. I need to count the values first.</div>'
        +'<div class="mock-input">Explain your thinking, or ask a question</div></div></div>')
      +notes([
        ["Type your reasoning, not code","“I’ll count each value, then keep the ones that appear once” is a plan. The tutor will not accept or write code here."],
        ["Being asked something is not being marked wrong","The questions are how a plan gets sharp. The exchange above is a plan working, not failing."],
        ["Ask it things too","“What is this question actually asking?” is a fair use of the tutor. Explaining the problem is part of its job; answering it is not."]])
      +'<h3>What makes a plan get through</h3>'
      +'<p>A plan is accepted when it says four things, in your own words:</p>'
      +rules([
        ["What you keep track of","the counts, the result dictionary, the running total - whatever you hold on to"],
        ["How you go through the input","once through the items, twice, recursively, in pairs"],
        ["How you produce the answer","what actually goes into the thing you return"],
        ["What happens in the awkward case","an empty input, a duplicate, a value that appears twice"]])
      +'<p>It does not have to be clever, and it does not have to be the approach anyone else picked. It has to hold up.</p>'
      +'<p>Same problem, two messages. Draw either one to see the difference:</p>'
      +'<fieldset class="tutorial-choices"><legend class="sr-only">Compare two plans</legend>'
      +'<label><input type="radio" name="planDemo" value="thin"><span><strong>A.</strong> “I’ll loop through the dictionary and swap the keys and the values.”</span></label>'
      +'<label><input type="radio" name="planDemo" value="full"><span><strong>B.</strong> “I’ll count how many times each value shows up, keep the ones that appear exactly once, and build a new dictionary with the value as the key. An empty dictionary gives back an empty dictionary.”</span></label>'
      +'</fieldset><p id="planDemoWhy" class="quiz-why" role="status" aria-live="polite"></p><div id="practicePlanGraph"></div>'
      +'<h3>Then: submit, and go</h3>'
      +notes([
        ["Submit plan for review","Your drawing goes for review. Approved, and a <strong>Continue to Code</strong> button appears - stage 2 is unlocked from then on."],
        ["“A simpler way exists”","Sometimes a box says your plan works but a leaner route exists. It is an offer, never a rejection: <strong>Continue with my plan</strong> is always there."],
        ["Your approach, your steps","If your plan is nothing like your instructor’s, the steps in stage 2 are rebuilt around <em>yours</em>. You are never asked for code you did not intend to write."]])
  },
  {
    tab:"2 · Code",
    eyebrow:"4 · STAGE TWO",
    title:"Code: one step at a time, and no reveal.",
    coach:'<p>The problem arrives in <strong>steps</strong>. You are asked for one piece of the function, you answer it, and the next one opens.</p>'
      +'<p>What you have already had accepted sits frozen above the editor, so you can always see the function taking shape.</p>'
      +'<p>Indentation is handled for you - you never have to guess how far in a step should sit.</p>',
    body:
      '<p>The editor is never a blank file. It is one step of your own plan, with everything you have already got right locked in above it:</p>'
      +mock("Stage 2 · step 3 of 5",
        '<div class="mock-prompt"><b>Step 3 of 5</b> · Add the value as a key and the original key as its value.</div>'
        +'<div class="mock-code frozen"><span class="ln">1</span><code>def invert_dict(d):</code>'
        +'<span class="ln">2</span><code>    counts = Counter(d.values())</code>'
        +'<span class="ln">3</span><code>    out = {}</code>'
        +'<span class="ln">4</span><code>    for key, value in d.items():</code>'
        +'<span class="ln">5</span><code>        if counts[value] == 1:</code></div>'
        +'<div class="mock-code live"><span class="ln">6</span><code class="cursor">out[value] = key</code></div>'
        +'<div class="mock-actions"><b>Submit step</b><span>Full file</span><span>Plan</span><span>Start over</span></div>')
      +notes([
        ["The grey lines are yours and they are settled","Accepted steps. You cannot edit them here - if you want to change your mind about one, Start over gives you a clean run."],
        ["You write one line’s worth of idea","Not the whole function. The step tells you what it wants; you write only that."],
        ["Full file / Plan","<strong>Full file</strong> shows the whole assignment file with your answers already in it. <strong>Plan</strong> re-opens your flowchart beside the editor. The tutor is still there for either."]])
      +'<h3>What happens when you submit</h3>'
      +'<p>Your code is actually run, against tests you do not get to see. Then one of three things:</p>'
      +mock("After Submit step",
        '<div class="mock-verdict ok">✓ That’s right. Step 4 of 5 is open.</div>'
        +'<div class="mock-verdict bad">Your solution runs but gives the wrong answer on at least one case.'
        +'<span class="mock-sub">▸ Show 3 of the 7 cases it failed</span>'
        +'<span class="mock-sub">Attempt 3. Keep going - take as many as you need.</span></div>'
        +'<div class="mock-verdict ask">✦ Your code ran. With <code>{‘a’: 1, ‘b’: 1}</code> it returns <code>{1: ‘b’}</code>. Walk me through what you expected there.</div>')
      +notes([
        ["Right","The step closes and the next one opens. That credit is recorded the moment it happens."],
        ["Wrong","You are told which cases failed and you go again. There is no attempt limit, and the attempt counter is there to tell you so."],
        ["Not sure","Sometimes it cannot decide, and rather than guess it asks you about something your own code actually did. Answering is not an attempt and cannot count against you."]])
      +'<p class="aside-note"><strong>The answer is never shown to you.</strong> Not after three tries, not after thirty, not at the end. There is no limit to run out of and no reveal waiting behind it - being stuck is where the learning is, so the site will not end it for you. What it will do is keep asking better questions.</p>'
  },
  {
    tab:"3 · Reflect",
    eyebrow:"5 · STAGE THREE",
    title:"Reflect, and where the credit goes.",
    coach:'<p>The last stage is not admin. It puts the plan you wrote in stage 1 next to the code you ended up with, and the gap between them is usually where the interesting bug was.</p>'
      +'<p>Nothing needs saving, ever. Every accepted step, every message and every version of your plan is recorded as you go.</p>',
    body:
      '<p>Finish the last step and stage 3 opens by itself:</p>'
      +mock("Stage 3 · Reflect",
        '<div class="mock-reflect"><b>✓ You worked through it.</b><span>5 steps, 9 attempts, one plan revision.</span></div>'
        +'<div class="mock-dual"><div><em>Your plan</em><div class="mock-flow"><i></i><i></i><i></i><i></i></div></div>'
        +'<div><em>Your code</em><div class="mock-flow code"><i></i><i></i><i></i><i></i></div></div></div>')
      +notes([
        ["Your plan vs. your code, side by side","Two flowcharts of the same solution: the one you described, and the one you wrote. Where they differ is what you learned on the way."],
        ["Your completed function","Expandable, in full, whenever you want it back."],
        ["Next problem →","Straight on to the next one in the assignment, without going back to the list."]])
      +'<h3>Your grades</h3>'
      +'<p><strong>My grades</strong> shows the credit recorded for you: per problem, per step, as it happened. Two things worth knowing about it:</p>'
      +rules([
        ["Credit is per step, not per problem","A problem you half-finished is not a zero. The steps you got are yours."],
        ["Attempts do not reduce it","Taking eleven tries on a step and getting it is worth what taking one try is worth."],
        ["Start over does not erase anything","It gives you a clean run at a problem. Your earlier attempt stays in your instructor’s record alongside the new one."]])
      +'<p class="aside-note">Close the tab mid-problem and nothing is lost. Reopening it puts you back on the step you had reached, with your accepted steps, your plan and your whole conversation still there.</p>'
  },
  {
    tab:"Check",
    eyebrow:"6 · THE SHORT VERSION",
    title:"Everything above, on one screen.",
    coach:'<p>If you remember nothing else from this tour, remember the six lines on the left.</p>'
      +'<p>The questions underneath are for you, not for a grade. Get them wrong as often as you like - each one says why.</p>'
      +'<p>You can reopen this page any time from <strong>Account ▾ → Settings → Retake tutorial</strong>.</p>',
    body:
      rules([
        ["Plan first, always","The editor is locked until your plan has been reviewed. Talk the approach through in the chat to unlock it."],
        ["The tutor cannot give you the answer","It was never shown one. Ask it to explain the question, to check your reasoning, to tell you what your code did - not to write anything."],
        ["Your approach is allowed to be yours","A workable plan that differs from your instructor’s gets its own steps, checked against the same tests."],
        ["Nothing is ever revealed","No attempt limit, no answer at the end of one. Wrong means failing cases and another go."],
        ["Everything is saved as it happens","No submit button for the assignment, nothing to hand in, nothing lost by closing the tab."],
        ["Credit is per step","Partial work counts, and the number of tries never costs you anything."]])
      +'<h3>Check yourself</h3>'
      +'<p>Four questions. Nothing is recorded and nothing is blocked - picking a wrong one just tells you why it is wrong.</p>'
      +'<div id="quizBox">'+quiz.map(q=>
        '<fieldset class="tutorial-choices quiz" data-quiz="'+q.key+'"><legend>'+esc(q.legend)+'</legend>'
        +q.options.map(([value,label])=>'<label><input type="radio" name="'+q.key+'" value="'+esc(value)+'"'
          +'><span>'+esc(label)+'</span></label>').join("")
        +'<p class="quiz-why" role="status" aria-live="polite"></p></fieldset>').join("")+'</div>'
      +'<div class="practice-completion"><strong>That is the tour.</strong>'
      +'<p>Open <strong>My assignments</strong> when you are ready. If something here does not match what you see, trust the real page - and ask your instructor.</p></div>'
  }];

  /* ---------- rendering ---------- */

  function buildRail(){
    t("tutorialProgress").innerHTML=chapters.map((c,i)=>
      '<button type="button" class="rail-tab" data-tutorial-step="'+i+'"><span>'+(i+1)+'</span>'+esc(c.tab)+'</button>').join("");
    t("tutorialProgress").querySelectorAll("[data-tutorial-step]").forEach(b=>
      b.onclick=()=>go(Number(b.dataset.tutorialStep)));
  }

  // Seeing the flowchart IS the lesson: a plan with enough in it draws
  // something, a thin one does not.
  function wirePlanDemo(root){
    root.querySelectorAll('[name="planDemo"]').forEach(el=>el.onchange=()=>{
      const box=t("practicePlanGraph"),why=t("planDemoWhy");
      if(el.value==="full"){
        renderGraph(box,graph,"",{height:300});
        why.className="quiz-why ok";
        why.textContent="That is the flowchart the page draws while you type plan B - every part of it came from a phrase in the message.";
      }else{
        box.replaceChildren();
        why.className="quiz-why no";
        why.textContent="Plan A draws nothing, because there is nothing in it to draw: no counting, and no rule for a value that appears twice. Those are the two things the tutor would ask you about.";
      }
    });
  }

  function wireQuiz(root){
    quiz.forEach(q=>{
      const field=root.querySelector('[data-quiz="'+q.key+'"]');
      if(!field)return;
      const why=field.querySelector(".quiz-why");
      const show=value=>{
        picked[q.key]=value;
        why.textContent=q.why[value]||"";
        why.className="quiz-why "+(value===q.right?"ok":"no");
      };
      // REPLAY THE PICK, not just its explanation. Every move re-renders the
      // chapter from one string that was built before anything was answered,
      // so the radio arrives unchecked however the question was answered - and
      // an explanation reading "Right." under four blank options is a page
      // that has lost track of what the student did. The `why` alone used to
      // come back; the dot has to come with it.
      if(picked[q.key]){
        const chosen=field.querySelector('input[value="'+picked[q.key]+'"]');
        if(chosen)chosen.checked=true;
        show(picked[q.key]);
      }
      field.querySelectorAll("input").forEach(el=>el.onchange=()=>show(el.value));
    });
  }

  function render(){
    const c=chapters[step],last=step===chapters.length-1;
    t("tutorialBack").hidden=step===0;
    t("tutorialNext").textContent=last?"Finish tour →":"Continue →";
    t("tutorialPosition").textContent="Chapter "+(step+1)+" of "+chapters.length;
    t("tutorialProgress").querySelectorAll("[data-tutorial-step]").forEach(el=>{
      const i=Number(el.dataset.tutorialStep);
      if(i===step)el.setAttribute("aria-current","step");else el.removeAttribute("aria-current");
      el.classList.toggle("complete",i<step);
    });
    t("tutorialCoach").innerHTML=c.coach;
    const root=t("tutorialExercise");
    root.innerHTML='<p class="eyebrow">'+c.eyebrow+'</p><h2 id="tutorialTitle">'+c.title+'</h2>'+c.body;
    wirePlanDemo(root);
    wireQuiz(root);
  }

  function go(next){
    if(next<0||next>=chapters.length)return;
    const moved=next!==step;
    step=next;render();
    if(moved){
      t("tutorialExercise").focus({preventScroll:true});
      t("tutorialExercise").scrollIntoView({block:"start",behavior:"instant"});
    }
  }

  function leave(completed){
    if(completed)AcadiaOnboarding.finish();else AcadiaOnboarding.dismiss();
    location.replace(AcadiaOnboarding.destination(
      new URLSearchParams(location.search).get("return")));
  }

  t("leaveTutorial").onclick=()=>leave(false);
  t("tutorialBack").onclick=()=>go(step-1);
  t("tutorialNext").onclick=()=>{
    if(step===chapters.length-1){leave(true);return;}
    go(step+1);
  };

  AcadiaOnboarding.ready().then(me=>{
    if(!me)return;
    mountHeader({variant:me.role==="teacher"?"instructor":"student",active:"Tutorial"});
    setCrumbs([{label:"How ACADIA works"}]);
    buildRail();render();
  });
})();
