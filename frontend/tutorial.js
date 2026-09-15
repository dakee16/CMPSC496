/* Temporary, interactive practice. No grading/tutor/session API calls. */
(function(){
  "use strict";
  requireSession();
  const t=id=>document.getElementById(id);
  let step=0,checked=false;
  const answers={base:"",initial:"",limit:"",operation:"",reflection:""};
  const plan=[{id:"return",label:"Return the result"},{id:"start",label:"Set result to 1"},{id:"loop",label:"For each number from 1 through n"},{id:"multiply",label:"Multiply result by that number"}];
  const expected=["start","loop","multiply","return"];
  const graph={nodes:[{id:"s",kind:"start",label:"Receive n"},{id:"a",kind:"step",label:"result = 1"},{id:"b",kind:"loop",label:"For number in 1 through n"},{id:"c",kind:"step",label:"result *= number"},{id:"d",kind:"return",label:"Return result"}],edges:[{src:"s",dst:"a"},{src:"a",dst:"b"},{src:"b",dst:"c",label:"next number"},{src:"c",dst:"b",label:"repeat"},{src:"b",dst:"d",label:"done"}]};
  const coaches=[
    '<p>Start with the inputs and outputs. Factorial multiplies every whole number from 1 through <code>n</code>.</p><p>Try a small example before writing code. What would your function return for <code>n = 0</code>?</p><p>In an assignment, your tutor asks questions like these to help you find your approach.</p>',
    '<p>A working plan explains your steps before you code. Put these four steps in order, then check the plan.</p><p>In assignments, describe your approach to the tutor or use <strong>Upload a plan</strong>. Submit it for review to unlock coding.</p><p>The loop arrow returns to its header. The <em>done</em> arrow leads to the return.</p>',
    '<p>Translate one step of your plan into code at a time.</p><p><code>range(1, n + 1)</code> includes <code>n</code> because Python excludes the upper limit.</p><p>Use <strong>Check practice code</strong> to try 0, 1 and 5. Real assignments grade the steps you submit and give feedback alongside your editor.</p>',
    '<p>You have read a question, built a plan and checked your code. Reflecting helps you carry what you learned into the next problem.</p><p>Open <strong>Assignments</strong> to start course work. <strong>My grades</strong> shows your recorded step credit.</p><p>Finish to remove this practice example. Retake it from <strong>Settings</strong> whenever you need it.</p>'
  ];
  function message(text,kind="info"){t("tutorialFeedback").className="tutorial-feedback "+kind;t("tutorialFeedback").textContent=text;}
  function choice(label,id,values,selected){
    return '<label class="tutorial-code-choice"><span>'+label+'</span><select id="'+id+'"><option value="">Choose…</option>'+values.map(v=>'<option value="'+esc(v)+'"'+(v===selected?' selected':'')+'>'+esc(v)+'</option>').join("")+'</select></label>';
  }
  function orderedPlan(){
    t("practicePlan").innerHTML=plan.map((item,index)=>'<li><span class="plan-order">'+(index+1)+'</span><strong>'+esc(item.label)+'</strong><div>'
      +'<button class="ghost" type="button" data-move="-1" data-index="'+index+'" aria-label="Move '+esc(item.label)+' up"'+(index===0?' disabled':'')+'>↑</button>'
      +'<button class="ghost" type="button" data-move="1" data-index="'+index+'" aria-label="Move '+esc(item.label)+' down"'+(index===plan.length-1?' disabled':'')+'>↓</button></div></li>').join("");
    t("practicePlan").querySelectorAll("[data-move]").forEach(button=>button.onclick=()=>{
      const index=Number(button.dataset.index),next=index+Number(button.dataset.move);
      [plan[index],plan[next]]=[plan[next],plan[index]];checked=false;t("practicePlanGraph").replaceChildren();
      message("Step moved. Check the plan when the order looks right.");orderedPlan();
      const controls=t("practicePlan").querySelectorAll('[data-index="'+next+'"]');
      (Array.from(controls).find(b=>!b.disabled)||t("checkPlan")).focus();
    });
  }
  function codePreview(){
    t("practiceCode").textContent="def factorial(n):\n    result = "+(answers.initial||"___")
      +"\n    for number in range(1, "+(answers.limit||"___")+"):\n        result "+(answers.operation||"___")+" number\n    return result";
  }
  function render(){
    checked=false;message("");
    t("tutorialBack").hidden=step===0;t("tutorialNext").textContent=step===3?"Finish tutorial →":"Continue →";
    t("tutorialPosition").textContent="Step "+(step+1)+" of 4";
    document.querySelectorAll("[data-tutorial-step]").forEach(el=>{
      if(Number(el.dataset.tutorialStep)===step)el.setAttribute("aria-current","step");else el.removeAttribute("aria-current");
      el.classList.toggle("complete",Number(el.dataset.tutorialStep)<step);
    });
    t("tutorialCoach").innerHTML=coaches[step];const root=t("tutorialExercise");
    if(step===0){
      root.innerHTML='<p class="eyebrow">1 · UNDERSTAND THE QUESTION</p><h2 id="tutorialTitle">Write a factorial function.</h2><p>Given a non-negative integer <code>n</code>, return the product of the integers from 1 through <code>n</code>. For zero, return 1.</p><pre class="code">factorial(4) → 1 × 2 × 3 × 4 → 24\nfactorial(1) → 1\nfactorial(0) → ?</pre><fieldset class="tutorial-choices"><legend>What should factorial(0) return?</legend>'
        +["0","1","An error"].map(v=>'<label><input type="radio" name="base" value="'+v+'"'+(answers.base===v?' checked':'')+'><span>'+v+'</span></label>').join("")+'</fieldset>';
      root.querySelectorAll('[name="base"]').forEach(el=>el.onchange=()=>{answers.base=el.value;message("");});
    }else if(step===1){
      root.innerHTML='<p class="eyebrow">2 · BUILD YOUR PLAN</p><h2 id="tutorialTitle">What happens first?</h2><p>Use the arrows to put these steps in order. Then check your plan to see its flowchart.</p><ol id="practicePlan" class="practice-plan"></ol><button id="checkPlan" class="ghost" type="button">Check plan</button><div id="practicePlanGraph"></div>';
      orderedPlan();t("checkPlan").onclick=()=>{
        checked=plan.every((item,index)=>item.id===expected[index]);
        if(checked){renderGraph(t("practicePlanGraph"),graph,"",{height:300});message("That works. Initialize the result, repeat the multiplication, then return it.","success");}
        else message("Start by setting result to 1. Multiplication belongs inside the loop, and the return comes last.","warn");
      };
    }else if(step===2){
      root.innerHTML='<p class="eyebrow">3 · PUT YOUR PLAN INTO CODE</p><h2 id="tutorialTitle">Fill in the three choices.</h2><div class="practice-code-options">'
        +choice("Starting result","practiceInitial",["0","1"],answers.initial)+choice("Range stops before","practiceLimit",["n","n + 1"],answers.limit)+choice("Update result with","practiceOperation",["+=","*="],answers.operation)
        +'</div><pre class="code" aria-label="Your practice Python code"><code id="practiceCode"></code></pre><button id="checkCode" type="button">Check practice code</button><div id="practiceResults"></div>';
      codePreview();
      for(const [id,key] of [["practiceInitial","initial"],["practiceLimit","limit"],["practiceOperation","operation"]])t(id).onchange=()=>{answers[key]=t(id).value;checked=false;codePreview();t("practiceResults").replaceChildren();message("");};
      t("checkCode").onclick=()=>{
        if(!answers.initial||!answers.limit||!answers.operation){message("Choose all three values first.","warn");return;}
        // Only evaluate the three displayed choices, never arbitrary code.
        const run=n=>{let r=Number(answers.initial),stop=answers.limit==="n + 1"?n+1:n;for(let j=1;j<stop;j++){if(answers.operation==="*=")r*=j;else r+=j;}return r;};
        const cases=[[0,1],[1,1],[5,120]].map(([n,expected])=>({n,expected,actual:run(n)}));
        checked=cases.every(test=>test.actual===test.expected);
        t("practiceResults").innerHTML='<table class="practice-results"><caption>Practice checks</caption><thead><tr><th>Input</th><th>Expected</th><th>Your result</th><th>Status</th></tr></thead><tbody>'+cases.map(test=>'<tr><td>factorial('+test.n+')</td><td>'+test.expected+'</td><td>'+test.actual+'</td><td>'+(test.actual===test.expected?'✓ Passed':'Try again')+'</td></tr>').join("")+'</tbody></table>';
        message(checked?"All three checks passed. Your function also handles zero.":answers.initial==="0"?"Multiplying by zero keeps the result at zero. Which starting value leaves multiplication unchanged?":answers.limit==="n"?"The upper limit is excluded. How can you include n in the loop?":"Factorial is a product. Which operation multiplies the running result?",checked?"success":"warn");
      };
    }else{
      root.innerHTML='<p class="eyebrow">4 · REFLECT</p><h2 id="tutorialTitle">One idea to take with you.</h2><p>Why does the function start with <code>result = 1</code>?</p><fieldset class="tutorial-choices"><legend class="sr-only">Choose the reason</legend>'
        +[["identity","Multiplying by 1 preserves a number, and an empty loop returns 1."],["count","It counts how many numbers are in the loop."]].map(([v,label])=>'<label><input type="radio" name="reflection" value="'+v+'"'+(answers.reflection===v?' checked':'')+'><span>'+label+'</span></label>').join("")
        +'</fieldset><div class="practice-completion"><strong>Ready for your own assignments</strong><p>Your practice example will close when you finish. Your course progress is unchanged.</p></div>';
      root.querySelectorAll('[name="reflection"]').forEach(el=>el.onchange=()=>{answers.reflection=el.value;message("");});
    }
    if(step)root.focus({preventScroll:true});
  }
  function leave(completed){
    if(completed)AcadiaOnboarding.finish();else AcadiaOnboarding.dismiss();
    location.replace(AcadiaOnboarding.destination(new URLSearchParams(location.search).get("return")));
  }
  t("leaveTutorial").onclick=()=>leave(false);t("tutorialBack").onclick=()=>{step--;render();};
  t("tutorialNext").onclick=()=>{
    if(step===0&&answers.base!=="1"){message("Factorial of zero is 1. Choose that base case to continue.","warn");return;}
    if(step===1&&!checked){message("Put the steps in order and check your plan first.","warn");t("checkPlan").focus();return;}
    if(step===2&&!checked){message("Check your practice code and get all three cases passing first.","warn");t("checkCode").focus();return;}
    if(step===3){if(answers.reflection!=="identity"){message("Think about the identity for multiplication, including when the loop runs zero times.","warn");return;}leave(true);return;}
    step++;render();t("tutorialExercise").scrollIntoView({block:"start",behavior:"instant"});
  };
  AcadiaOnboarding.ready().then(me=>{if(!me)return;mountHeader({variant:me.role==="teacher"?"instructor":"student",active:"Tutorial"});setCrumbs([{label:"Guided practice"}]);render();});
})();
