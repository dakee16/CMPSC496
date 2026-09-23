/* Shared learning workspace with the real UI scripts and synthetic responses.
   No live database, model, oracle or code execution. Happy DOM cannot test layout. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const root = path.resolve(__dirname, '../../frontend');
const read = name => fs.readFileSync(path.join(root, name), 'utf8');
const tick = () => new Promise(resolve => setTimeout(resolve, 30));
const problem = {slug:'employee',title:'Employee Update',description:'Carry employees forward. Keep `records` unchanged.\n\n>>> update(records)\n{"year": 2027}'};
const steps = [{prompt:'Read the previous records.',indent:0},{prompt:'Return the updated records.',indent:0}];
const graph = {nodes:[{id:'n0',kind:'start',label:'Read records'},{id:'n1',kind:'return',label:'Return updates'}],edges:[{src:'n0',dst:'n1'}]};

(async () => {
  const {Window} = await import(pathToFileURL(require.resolve('happy-dom')).href);
  const window = new Window({url:'http://localhost/student.html',width:1600,height:1000,settings:{enableJavaScriptEvaluation:true,disableJavaScriptFileLoading:true,disableCSSFileLoading:true,suppressInsecureJavaScriptEnvironmentWarning:true}});
  const doc = window.document;
  let value = '', selections = [], editorCreations = 0, focusCalls = 0;
  const options = {}, calls = [];
  let historyGate=null, historyFailure=false;
  // A resumed session's own shape, when a case needs one (accepted prefix,
  // where the student had got to, and however many steps that implies).
  let resumeSteps=null, resumeAccepted=null, resumeIndex=null;
  // A session-level refusal from /grade_chunk (403/404), as opposed to a verdict.
  let gradeRefusal=null;
  let reopenCopy=null;
  // What /replan answers when the plan is approved: the teacher's roadmap
  // stands (null), or one rebuilt around this student's own approach.
  let replanAnswer=null;
  let approved = false, stepFailure = false, comparisonFailure = false, openingFailure = false, verdict = 'incorrect', history = {found:false}, session = 0;
  try {
    doc.write(read('student.html').replace(/<script\b[^>]*>[\s\S]*?<\/script>/g,''));
    window.scrollTo = () => {};
    window.CodeMirror = {fromTextArea:() => {
      editorCreations++;
      const input = doc.createElement('textarea'); input.id='qaEditorInput'; doc.querySelector('#editorWrap').append(input);
      return {setOption:(k,v)=>options[k]=v,setSize:()=>{},on:()=>{},refresh:()=>{},focus:()=>{focusCalls++;input.focus();},setCursor:()=>{},getGutterElement:()=>({offsetWidth:32}),getValue:()=>value,setValue:v=>value=v,listSelections:()=>selections,setSelections:v=>selections=v,lineCount:()=>value.split('\n').length,getLine:n=>value.split('\n')[n]||''};
    }};
    window.fetch = async (url, init={}) => {
      const route = new URL(url,'http://localhost').pathname;
      calls.push({route,body:init.body});
      let data = {}, status = 200;
      if(route==='/auth/me') data={name:'Alex',role:'student',student_id:'synthetic'};
      else if(route==='/assignments') data={assignments:[]};
      else if(route==='/solved') data={slugs:[]};
      else if(route==='/student/progress') data={problems:[]};
      else if(route==='/decompose_chunks'){status=openingFailure?503:200;data={session_id:'s'+(++session),header:'def employee_update(records):',chunks:(resumeSteps||steps).map(s=>({...s,prompt:''}))};if(resumeAccepted)data={...data,resumed:true,accepted:resumeAccepted,index:resumeIndex,total_chunks:(resumeSteps||steps).length};}
      else if(route.startsWith('/history/')){if(historyGate)await historyGate;if(historyFailure)status=503;data=history;}
      else if(route.startsWith('/session_steps/')){status=stepFailure?503:200;data={chunks:resumeSteps||steps};}
      else if(route==='/tutor_chat') data={reply:'What changes from one year to the next?',ready:true,offtrack:true};
      else if(route==='/plan_graph') data=graph;
      else if(route==='/design_review/plan'||route==='/design_review') data={approved,reply:approved?'Your approach is approved.':'Explain how you will preserve the earlier records.',plan_graph:graph};
      else if(route==='/grade_chunk'){
        if(gradeRefusal){status=gradeRefusal.status;data={detail:gradeRefusal.detail};
          return new window.Response(JSON.stringify(data),{status,headers:{'Content-Type':'application/json'}});}
        const index=JSON.parse(init.body).expected_index;
        data=verdict==='correct'?{verdict:'correct',reason:'This step passes.',index:index+1,completed:index===1,solved_independently:true}:verdict==='indeterminate'?{verdict:'indeterminate',reason:'The grader is temporarily unavailable.'}:verdict==='diagnosis'?{verdict:'indeterminate',reason:'We could not confirm this step.',needs_diagnosis:true,diagnosis:"Try your code on this input: 'aab'. After your step, `seen` is {'a', 'b'}. "+'Does that give the next step everything it needs?'}:{verdict:'incorrect',reason:'Try the empty-input case.',failing_cases:['employee_update({})\n\nexpected: {}\nyou gave: None','employee_update({2019: {}})\n\nit raised: KeyError(2018)'],failed_total:5,attempts:1};
      }
      else if(route==='/replan') data=replanAnswer||{rerouted:false};
      else if(route==='/mark_solved') data={ok:true};
      else if(route==='/graphs'){status=comparisonFailure?503:200;data={plan:graph,code:graph,comparison:{similarity:1,notes:['The structure matches.']}};}
      else if(route==='/reopen_step') data={index:JSON.parse(init.body).index,attempts:0,completed:false,total_chunks:steps.length,dropped:[{index:0,code:'    draft = records.copy()'}],...(reopenCopy?{session_id:reopenCopy,copied_from:'finished-run'}:{})};
      else if(route.endsWith('/restart')) data={ok:true};
      else throw new Error('Unexpected request: '+route);
      return new window.Response(JSON.stringify(data),{status,headers:{'Content-Type':'application/json'}});
    };
    window.eval(['ui.js','cache.js','graphs.js','workspace.js','student.js'].map(read).join('\n')+'\nwindow.inspect = code => eval(code);');
    await tick();
    await window.start(problem);
    const visibleStage = () => [...doc.querySelectorAll('.journey-panel')].filter(el=>!el.hidden).map(el=>el.id);
    assert.deepEqual(visibleStage(),['stageRead']);
    assert(doc.querySelector('#statement pre').textContent.includes('update(records)'));
    assert.equal(doc.querySelector('#readSignature').textContent,'def employee_update(records):');
    assert.equal(doc.querySelector('#chatcol').parentElement.id,'tutorDock');
    assert.equal(doc.querySelector('#tutorDock').hidden,false,'Tutor is available from the question opening');
    assert.equal(doc.querySelector('#cform').inert,false);
    assert.equal(doc.querySelector('#tabCode').getAttribute('aria-disabled'),'true');
    doc.querySelector('#tabCode').click();
    assert.deepEqual(visibleStage(),['stageRead']);
    assert(doc.querySelector('#workspaceNotice').textContent.includes('submit your plan'));
    assert.equal(options.readOnly,'nocursor');
    doc.querySelector('#tabRead').focus();
    doc.querySelector('#tabRead').dispatchEvent(new window.KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true}));
    assert.equal(doc.activeElement.id,'tabCode');
    assert.deepEqual(visibleStage(),['stageRead'],'Reading and planning share one tab');
    assert.equal(doc.querySelector('#problemPaper').parentElement.id,'readQuestion');
    assert.equal(doc.querySelector('#chatcol').parentElement.id,'tutorDock');
    assert.equal(doc.querySelector('#planSubmitBtn').disabled,true);
    assert.equal(doc.querySelector('#planPreview').tagName,'SECTION','The plan graph has no collapse control');
    assert.equal(doc.querySelector('#planEmpty').hidden,false);
    doc.querySelector('#cinput').value='My unsent thinking';
    doc.querySelector('#planUpload').open=true;
    window.pickDesign(new window.File(['diagram'],'plan.pdf',{type:'application/pdf'}));
    doc.querySelector('#planUpload').open=false;
    assert.equal(doc.querySelector('#cinput').value,'My unsent thinking');
    doc.querySelector('#planUpload').open=true;
    assert.equal(doc.querySelector('#designName').textContent,'plan.pdf');
    doc.querySelector('#planUpload').open=false;
    await window.sendToTutor('I will copy the records and update the salaries.');
    await window.inspect('planQueue');
    assert(doc.querySelector('#fork'),'The tutor’s alternative-approach choices remain available');
    assert.equal(doc.querySelector('#planSubmitBtn').disabled,false);
    assert.equal(doc.querySelector('#planEmpty').hidden,true);
    await window.submitPlanGraph();
    assert.equal(doc.querySelector('#tabCode').getAttribute('aria-disabled'),'true');
    assert(doc.querySelector('#designMsg').textContent.includes('preserve'));
    approved=true;
    // WHOSE ROADMAP? is asked HERE, once, and nowhere else. It used to be
    // decided inside /decompose_chunks, which start(p) calls the moment a
    // student clicks into a problem - before any plan exists - so the teacher's
    // roadmap was taken and the session created, and nothing looked again.
    // Approval is the first moment the plan exists and the last moment no code
    // has been written.
    await window.submitPlanGraph(); await tick();
    assert(calls.some(c=>c.route==='/replan'),'the roadmap is chosen at approval');
    // Nothing was rebuilt, so the session and the steps stand as they were.
    assert(doc.querySelector('#codeOrientation').textContent.includes('unlocked'),
      doc.querySelector('#codeOrientation').textContent);
    assert.deepEqual(visibleStage(),['stageRead'],'Approval must not force a stage change');
    assert.equal(doc.querySelector('#planApproved').hidden,false);
    assert.equal(doc.querySelector('#tabCode').getAttribute('aria-disabled'),'false');
    doc.querySelector('#planContinue').click();
    assert.deepEqual(visibleStage(),['stageCode']);
    assert.equal(doc.querySelector('#stepCount').textContent,'Step 1 of 2');
    assert.equal(doc.querySelector('#tutorDock').hidden,false,'Coding keeps the tutor beside the editor');
    value='    draft = records.copy()'; selections=[{anchor:{line:0,ch:7},head:{line:0,ch:12}}];
    const conversation=doc.querySelector('#clog');
    assert.equal(doc.querySelector('#problemPaper').parentElement.id,'codeQuestion','Coding keeps the question on top');
    assert.equal(doc.querySelector('#problemDetails').open,false,'…minimised, and the student can reopen it');
    const trigger=doc.querySelector('.code-resources [data-resource="plan"]');
    trigger.focus();trigger.click();
    assert.equal(doc.querySelector('#resourceDrawer').hidden,false);
    assert.equal(doc.querySelector('#page').inert,false,'A reference never blocks code or chat');
    assert.equal(doc.querySelector('#stageCode').hidden,false);
    assert.equal(doc.querySelector('#cform').inert,false);
    assert.equal(doc.querySelector('#planCard').parentElement.id,'resourceBody');
    window.openResource('tutor');
    assert.equal(doc.querySelector('#clog'),conversation,'One shared conversation across all stages');
    assert.equal(doc.querySelector('#chatcol').parentElement.id,'tutorDock');
    assert.equal(doc.querySelector('#resourceDrawer').hidden,false,'Asking the tutor leaves the reference open');
    assert.equal(doc.querySelector('#cform').inert,false);
    doc.querySelector('#closeResource').dispatchEvent(new window.KeyboardEvent('keydown',{key:'Escape',bubbles:true}));
    assert.equal(doc.querySelector('#resourceDrawer').hidden,true);
    assert.equal(doc.querySelector('#page').inert,false);
    assert.equal(doc.activeElement,trigger);
    doc.querySelector('#showTutor').click();
    assert.equal(doc.querySelector('#tutorDock').hidden,true);
    assert.equal(doc.querySelector('#cform').inert,true);
    window.openResource('tutor');
    assert.equal(doc.querySelector('#tutorDock').hidden,false);
    assert.equal(doc.querySelector('#clog'),conversation);
    assert.equal(value,'    draft = records.copy()');
    doc.querySelector('#cinput').focus();
    const reply = window.sendToTutor('Why should I copy the record?');
    doc.querySelector('#qaEditorInput').focus();
    assert.equal(doc.activeElement.id,'qaEditorInput');
    await reply;
    assert.equal(doc.activeElement.id,'qaEditorInput','A tutor reply must not steal focus from coding');
    assert.equal(doc.querySelector('#stageCode').hidden,false);
    assert.equal(value,'    draft = records.copy()');
    const layout=doc.querySelector('#studyLayout'), divider=doc.querySelector('#studyDivider');
    Object.defineProperty(layout,'clientWidth',{configurable:true,value:1200});
    doc.querySelector('#tutorDock').getBoundingClientRect=()=>({width:parseInt(layout.style.getPropertyValue('--tutor-width'))||360});
    divider.focus();
    for(const [key,expected] of [['ArrowLeft','380'],['End','480'],['ArrowLeft','480'],['Home','300'],['ArrowRight','300']]){
      divider.dispatchEvent(new window.KeyboardEvent('keydown',{key,bubbles:true}));
      assert.equal(divider.getAttribute('aria-valuenow'),expected);
    }
    assert.equal(value,'    draft = records.copy()','Resizing preserves editor contents');
    window.chooseWorkspaceStage('read'); window.chooseWorkspaceStage('code');
    assert.equal(doc.querySelector('#problemPaper').parentElement.id,'codeQuestion','The question follows the open stage');
    assert.equal(value,'    draft = records.copy()');
    assert.equal(editorCreations,1);
    assert.equal(selections[0].head.ch,12);
    doc.querySelector('#focusWork').click();
    assert(doc.body.classList.contains('focus-workspace'));
    assert.equal(doc.querySelector('#tutorDock').hidden,false,'Focus mode still supports asking questions');
    window.dispatchEvent(new window.KeyboardEvent('keydown',{key:'Escape'}));
    assert(!doc.body.classList.contains('focus-workspace'));
    await doc.querySelector('#submit').onclick();
    assert(doc.querySelector('#msg .failCase'),'Failure details remain available');
    // Several cases now, and the summary states the REAL total - the sandbox
    // caps what it sends back, so counting the listings would under-report.
    assert.equal(doc.querySelectorAll('#msg .failCase pre').length,2,'Every case sent is shown');
    assert(doc.querySelector('#msg .failCase summary').textContent.includes('2 of the 5'),
      doc.querySelector('#msg .failCase summary').textContent);
    assert(doc.querySelector('#msg .failMore').textContent.includes('3 more'),'The hidden ones are counted');
    assert.equal(value,'    draft = records.copy()');
    verdict='indeterminate';await doc.querySelector('#submit').onclick();
    assert.equal(value,'    draft = records.copy()');

    // COULD NOT CONFIRM -> the page pauses and ASKS, instead of letting them
    // resubmit into the same wall. The verdict costs no attempt and does not
    // advance the step, so without this a student whose code IS wrong is left
    // with nothing to act on. See main/diagnose.py.
    value='    seen = set(records)';
    verdict='diagnosis';await doc.querySelector('#submit').onclick();
    assert(doc.querySelector('#msg .banner').textContent.includes("'aab'"),
      'The measured input is what they are asked to trace');
    assert(doc.querySelector('#msg .banner').textContent.includes('`seen`'),
      'Their OWN variable is shown back to them');
    assert(doc.querySelector('#diagFix'),'A way back to the editor');
    assert(doc.querySelector('#diagExplain'),'A way to say the approach differs');
    // A prompt, not a lock-out: the editor stays live and the draft survives.
    assert.notEqual(options.readOnly,'nocursor','The editor must stay usable');
    assert.equal(value,'    seen = set(records)','Their draft survives the prompt');
    doc.querySelector('#diagFix').click();
    assert.equal(doc.querySelector('#msg').innerHTML,'','Dismissing clears the prompt');
    value='    seen = set(records.keys())';
    verdict='diagnosis';await doc.querySelector('#submit').onclick();
    // The LABEL may not name an approach for theirs to differ from either.
    assert(!/different/i.test(doc.querySelector('#diagExplain').textContent),
      doc.querySelector('#diagExplain').textContent);
    doc.querySelector('#diagExplain').click();
    assert(doc.querySelector('#cinput').value.includes('explain my approach'),
      'Explaining opens the tutor with the opening already typed');
    // ...and it never names an approach for theirs to differ FROM. This drafted
    // "My approach is different from the one you expected" until an audit
    // caught it: the page putting words in the student's mouth that tell them a
    // preferred answer exists.
    assert(!/you expected|differs? from ours|our version/i.test(
      doc.querySelector('#cinput').value), doc.querySelector('#cinput').value);
    // THEY HAVE ALREADY TALKED TO THE TUTOR BY NOW, so the opener points at
    // that conversation instead of demanding it again. Reported from testing:
    // a student who had explained their whole approach in this chat, and been
    // answered on it, was handed an empty box and pasted the lot again - to a
    // tutor that is sent this very conversation (sendToTutor's TUTOR_WINDOW)
    // and already had every word of it.
    assert(doc.querySelector('#cinput').value.includes('already described it earlier'),
      doc.querySelector('#cinput').value);
    assert(!doc.querySelector('#cinput').value.includes('Here is what I am doing'),
      'it must not ask them to restate what the tutor is already holding');
    // With nothing said yet there is nothing to point at, so it does ask.
    // (Clicking Explain clears #msg, so the prompt has to be raised again.)
    const saidSoFar=window.inspect('chatLog.splice(0, chatLog.length)');
    const draftHeld=value;          // raising the prompt again must not cost the draft
    doc.querySelector('#cinput').value='';
    verdict='diagnosis';await doc.querySelector('#submit').onclick();
    doc.querySelector('#diagExplain').click();
    assert(doc.querySelector('#cinput').value.includes('Here is what I am doing'),
      doc.querySelector('#cinput').value);
    window.inspect('chatLog').push(...saidSoFar);
    value=draftHeld;
    doc.querySelector('#cinput').value='';
    // Put the draft back the way the rest of this file expects to find it.
    value='    draft = records.copy()';
    verdict='correct';await doc.querySelector('#submit').onclick();
    assert.equal(doc.querySelector('#reviewBox').hidden,false);
    assert(doc.querySelector('#backToNow').textContent.includes('Continue to step 2'));
    assert(doc.querySelector('#reviewCode').textContent.includes('records.copy()'));
    // BACK TO AN ACCEPTED STEP. Freezing accepted code is right - it is what
    // every later step was graded against - but having no way to reopen it
    // left a student who spotted a bug in step 1 with only Start over, which
    // gives up the whole problem. The server owns the state change; the page
    // leaves review mode and hands their own answer back to edit.
    assert.equal(doc.querySelector('#reworkStep').hidden,false,
      'Their own accepted step can be reopened');
    doc.querySelector('#reworkStep').click();await tick();
    assert(calls.some(c=>c.route==='/reopen_step'),'the server decides, not the page');
    assert.equal(doc.querySelector('#reviewBox').hidden,true,'review mode closes');
    assert(value.includes('records.copy()'),
      'with the answer they came back to edit already in the editor');
    assert(doc.querySelector('#msg').textContent.includes('back on step 1'));
    // Re-accepted, which is where the rest of this file expects to be.
    await doc.querySelector('#submit').onclick();
    assert.equal(doc.querySelector('#reviewBox').hidden,false);
    doc.querySelector('#backToNow').click();
    assert.equal(value,'    ','Continuing must start a new draft, not repeat the submitted answer');
    // AND WITHOUT THE REVIEW DETOUR. On step 2, with step 1 accepted, the way
    // back to step 1 sits under the code it would change - reaching it through
    // a finished step's pill means knowing that pill is a button.
    assert.equal(doc.querySelector('#editEarlier').hidden,false,
      'a finished step is reachable while still solving the next one');
    assert.deepEqual([...doc.querySelectorAll('#editEarlier [data-edit]')]
      .map(b=>b.textContent),['Step 1'],'their own answers, and only those');
    doc.querySelector('#editEarlier [data-edit]').click();await tick();
    assert(doc.querySelector('#msg').textContent.includes('back on step 1'));
    assert.equal(doc.querySelector('#editEarlier').hidden,true,
      'and with nothing accepted behind it, the row goes away');
    // Re-accepted, back to where the rest of this file expects to be.
    value='    draft = records.copy()';
    await doc.querySelector('#submit').onclick();
    doc.querySelector('#backToNow').click();
    assert.equal(value,'    ');
    value='    return updated';comparisonFailure=true;
    await doc.querySelector('#submit').onclick();
    assert.deepEqual(visibleStage(),['stageCode'],'Completion waits for the student to open Reflect');
    assert.equal(doc.querySelector('#finishReview').hidden,false);
    doc.querySelector('#finishReview').click();
    assert.deepEqual(visibleStage(),['stageReflect']);
    // No per-problem score: "2 of 2 steps passed" is a grade, and grades are
    // the course LMS's to report (the student nav in ui.js says why).
    assert(!/\d+ of \d+|%/.test(doc.querySelector('#reflectSummary').textContent),
      'a score reached the Reflect stage');
    assert(!doc.querySelector('#reflectionGrades'),'no link to a grades page');
    assert(doc.querySelector('#reflectionCode').textContent.includes('return updated'));
    assert(doc.querySelector('#comparisonStatus button'));
    comparisonFailure=false;doc.querySelector('#comparisonStatus button').click();await tick();
    assert.equal(doc.querySelector('#dualCard').hidden,false);
    assert.equal(doc.querySelector('#comparisonStatus').textContent,'');

    // Upload approval, unavailable step instructions, retry and a safe restart.
    approved=true;await window.start(problem);
    window.chooseWorkspaceStage('read');doc.querySelector('#planUpload').open=true;
    window.pickDesign(new window.File(['diagram'],'plan.pdf',{type:'application/pdf'}));
    stepFailure=true;await window.uploadDesign();await tick();
    assert.equal(doc.querySelector('#planApproved').hidden,false);
    window.chooseWorkspaceStage('code');
    assert.equal(doc.querySelector('#submit').disabled,true);
    assert.equal(options.readOnly,'nocursor');
    assert(doc.querySelector('#msg').textContent.includes('could not be loaded'));
    stepFailure=false;doc.querySelector('#msg button').click();await tick();
    assert.equal(doc.querySelector('#submit').disabled,false);
    window.openRestart();window.closeRestart();
    assert.equal(doc.querySelector('#tabCode').getAttribute('aria-disabled'),'false','Cancel restart keeps approval');
    await window.doRestart();
    assert.deepEqual(visibleStage(),['stageRead']);
    assert.equal(doc.querySelector('#tabCode').getAttribute('aria-disabled'),'true');
    assert.equal(doc.querySelector('#tabReflect').getAttribute('aria-disabled'),'true');
    assert.equal(doc.querySelector('#designPick').hidden,true);
    openingFailure=true;await window.start(problem);
    assert(doc.querySelector('#workspaceStatus button'),'Failed initialization has a visible retry');
    assert.equal(doc.querySelector('#tabCode').getAttribute('aria-disabled'),'true');
    openingFailure=false;doc.querySelector('#workspaceStatus button').click();await tick();
    assert.equal(doc.querySelector('#workspaceStatus').textContent,'');
    history={found:true,design_approved:true,plan:graph,code:graph,comparison:{similarity:1},messages:[{role:'user',content:'Earlier work'}]};
    await window.start(problem);await tick();
    // AN APPROVED SESSION REOPENS WHERE THE WORK IS. This asserted 'stageRead'
    // and was left behind when resume started restoring the coding stage: a
    // student who had already passed the gate was being put back on the
    // planning screen to re-find their own place. The gate is passed, so Code
    // is the honest landing - keeping the old expectation would mean changing
    // correct behaviour to satisfy a test rather than the other way round.
    assert.deepEqual(visibleStage(),['stageCode']);
    assert.equal(doc.querySelector('#tabCode').getAttribute('aria-disabled'),'false',
      'a restored approval must leave the editor open');
    assert.equal(doc.querySelector('#tabReflect').getAttribute('aria-disabled'),'false');
    assert(doc.querySelector('#reflectSummary').textContent.includes('previous attempt'));
    assert.equal(doc.querySelector('#reflectionCodeDetails').hidden,true,'Historical graphs cannot claim a new completed function');
    assert.equal(doc.querySelectorAll('#clog').length,1);

    // A COVERED STEP, AFTER A RELOAD. One submission can answer more than one
    // step, and main/sessions.commit_outcome records the code against the FIRST
    // of them and the rest as empty. public_session used to report those extra
    // steps as "own", and the page flattened anything non-revealed to "own" on
    // top of that - so a resumed covered step reviewed as a BLANK panel headed
    // "your answer" and offered Rework, which drops back to a step whose code is
    // already in the frozen prefix above it: nothing to write there, and a blank
    // submission grades incorrect. In-session this was always right; only the
    // resumed view lost it, so it took a reload to see.
    resumeSteps=[{prompt:'Read the previous records.',indent:0},
                 {prompt:'Copy them forward.',indent:0},
                 {prompt:'Return the updated records.',indent:0}];
    resumeAccepted=[{code:'    draft = records.copy()',how:'own'},
                    {code:'',how:'covered'}];
    resumeIndex=2;
    await window.start(problem);await tick();
    const coveredPill=doc.querySelectorAll('#stepper .steppill')[1];
    assert.equal(coveredPill.tagName.toLowerCase(),'button','a finished step is reachable');
    coveredPill.click();await tick();
    assert(doc.querySelector('#reviewCode').textContent.includes('already wrote this'),
      'a covered step explains itself instead of showing a blank answer');
    assert.equal(doc.querySelector('#reworkStep').hidden,true,
      'and offers no Rework, because there is no answer of their own to rework');
    // Their own step next to it still does offer it.
    doc.querySelectorAll('#stepper .steppill')[0].click();await tick();
    assert(doc.querySelector('#reviewCode').textContent.includes('records.copy()'));
    assert.equal(doc.querySelector('#reworkStep').hidden,false,
      'the step they did write is still reopenable');
    doc.querySelector('#backToNow').click();
    resumeSteps=resumeAccepted=resumeIndex=null;

    // A REBUILT ROADMAP IS ADOPTED WHOLE, session and all. The server retires
    // the old session when it seats a rebuilt one, so keeping its id here is
    // exactly what produces "That session belongs to someone else" two clicks
    // later. Nothing of theirs is lost: this only runs before any code exists,
    // and the server refuses to reroute a session that has accepted steps.
    replanAnswer={rerouted:true,session_id:'sess-rebuilt',total_chunks:2,
                  chunks:[{prompt:'Count each value.',indent:0},
                          {prompt:'Hand it back.',indent:0}]};
    approved=true;
    await window.start(problem);await tick();
    await window.submitPlanGraph();await tick();
    assert.equal(window.inspect('sessionId'),'sess-rebuilt',
      'the page must move to the session the rebuilt roadmap lives in');
    assert.equal(window.inspect('idx'),0);
    assert.equal(window.inspect('accepted.length'),0,'a rebuilt roadmap starts clean');
    // ...and it SAYS so, neutrally. It never offers a choice and never says a
    // different approach existed: a student told "ours differs from yours"
    // conforms out of caution, which is the opposite of the point.
    const said=doc.querySelector('#codeOrientation').textContent;
    assert(said.includes('follow the approach you described'),said);
    assert(!/teacher|expected|different|instead/i.test(said),said);
    replanAnswer=null;

    // THE ROUTER CAN ALSO FIRE FROM THE CODING STAGE, on what they WROTE.
    // A plan is prose and prose is lossy, so the plan-time check is a floor:
    // a student whose plan reads like the teacher's can still write something
    // genuinely different, which is what happened on `invert`. This asks only
    // after grading COULD NOT CONFIRM the step - never after an acceptance,
    // because a correct answer can be a different shape too.
    approved=true;
    await window.start(problem);await tick();
    await window.submitPlanGraph();await tick();
    doc.querySelector('#planContinue').click();
    value='    seen = [x for x in records]';
    replanAnswer={rerouted:true,session_id:'sess-from-code',total_chunks:2,
                  carried:'    draft = records.copy()',
                  chunks:[{prompt:'Gather them.',indent:0},
                          {prompt:'Hand it back.',indent:0}]};
    verdict='diagnosis';await doc.querySelector('#submit').onclick();
    await tick();await tick();await tick();   // replan, then render, then loadSteps
    const sent=calls.filter(c=>c.route==='/replan').pop();
    assert(sent&&JSON.parse(sent.body).code,'their code is what it asks about');
    assert.equal(window.inspect('sessionId'),'sess-from-code');
    assert(doc.querySelector('#msg').textContent.includes('follow the approach you described'),
      'msg was: '+JSON.stringify(doc.querySelector('#msg').textContent));
    // Work already accepted comes WITH them - it was graded against steps that
    // no longer exist, so it cannot stay accepted, but it is still their code.
    assert(value.includes('records.copy()'),'carried work is put back in the editor: '+value);
    assert(value.includes('seen = '),'...alongside what they were working on: '+value);
    assert.equal(window.inspect('accepted.length'),0);
    replanAnswer=null;verdict='incorrect';

    // A SESSION THAT IS NOT OURS IS NOT A DEAD END. Reported from testing:
    // "That session belongs to someone else." appeared over the editor with no
    // way out, because every !r.ok was shown and nothing more. The ownership
    // check is right - without it one student could submit against another's
    // session and spend their attempts - but the answer to "this id is not
    // yours" is to go and get the one that is, which reopening does.
    gradeRefusal={status:403,detail:{reason_code:'not_your_session',
                                     message:'That session belongs to someone else.'}};
    value='    draft = records.copy()';   // there has to be an answer to submit
    await doc.querySelector('#submit').onclick();await tick();
    assert(doc.querySelector('#msg .banner').textContent.includes('Reopen the problem'),
      doc.querySelector('#msg .banner').textContent);
    const reopenBtn=doc.querySelector('#sessionReopen');
    assert(reopenBtn,'a refused session must offer a way back');
    gradeRefusal=null;
    const callsBefore=calls.length;
    reopenBtn.click();await tick();
    assert(calls.slice(callsBefore).some(c=>c.route==='/decompose_chunks'),
      'and the way back actually reopens the problem');

    // A problem that is ALREADY SOLVED opens on the congratulations stage, not
    // back in the working screen with an unlocked editor and nothing to do.
    history={found:true,solved:true,design_approved:true,plan:graph,messages:[{role:'user',content:'Earlier work'}]};
    await window.start(problem);await tick();
    assert.deepEqual(visibleStage(),['stageReflect'],'a finished problem reopens on Reflect');
    assert(doc.querySelector('#reflectTitle').textContent.includes('Congratulations'),
      doc.querySelector('#reflectTitle').textContent);
    assert.equal(doc.querySelector('#reflectionNext').textContent.trim(),'Next problem →');
    assert(doc.querySelector('.reflection-actions [data-restart]'),'...and an offer to start it over');
    assert.equal(doc.querySelector('#reflectionCodeDetails').hidden,true,
      'no code was restored, so there is no completed function to disclose');
    // Saved chat and graph share an explicit loading/error boundary.
    let releaseHistory;
    historyGate=new Promise(resolve=>releaseHistory=resolve);
    const restoring=window.start(problem);await tick();
    assert.equal(doc.querySelector('#planPreview').getAttribute('aria-busy'),'true');
    assert.equal(doc.querySelector('#planEmpty').hidden,true);
    assert.equal(doc.querySelector('#planCard').hidden,false);
    assert(doc.querySelector('#planLive .graph-loading'));
    assert.equal(doc.querySelector('#cform button[type="submit"]').disabled,true);
    releaseHistory();historyGate=null;await restoring;
    assert.equal(doc.querySelector('#planPreview').getAttribute('aria-busy'),'false');
    assert(doc.querySelector('#planLive svg'));
    historyFailure=true;await window.start(problem);
    assert(doc.querySelector('#retryHistory'));
    assert.equal(doc.querySelector('#cinput').disabled,true);
    historyFailure=false;doc.querySelector('#retryHistory').click();await tick();
    assert.equal(doc.querySelector('#retryHistory'),null);
    assert.equal(doc.querySelector('#cinput').disabled,false);
    const ids=[...doc.querySelectorAll('[id]')].map(el=>el.id);
    assert.equal(new Set(ids).size,ids.length,'No duplicate IDs');
    assert(calls.some(c=>c.route==='/design_review')&&calls.some(c=>c.route==='/design_review/plan'));
    // A FINISHED PROBLEM REOPENS WITH ITS CODE. It used to open on a fresh
    // session, so the chat about this function came back beside an empty
    // editor on step 1. Its own finished session comes back now, and the code
    // stage shows what they wrote - read-only, with every step offered back.
    resumeAccepted=[{code:'draft = records.copy()',how:'own'},{code:'return updated',how:'own'}];
    resumeIndex=steps.length;
    const finishedCalls=calls.length;
    await window.start(problem);await tick();
    assert(doc.querySelector('#ctxCode').textContent.includes('records.copy()')&&
      doc.querySelector('#ctxCode').textContent.includes('return updated'),
      'their finished code is in the coding block');
    assert.equal(doc.querySelector('#editorWrap').hidden,true,'no step left to type into');
    assert.equal(doc.querySelector('#submit').style.display,'none');
    assert(doc.querySelector('#msg').textContent.includes('You finished this problem'));
    assert.deepEqual([...doc.querySelectorAll('#editEarlier [data-edit]')].map(b=>b.textContent),
      ['Step 1','Step 2'],'every step can be reopened from here');
    // Opening a finished problem is not finishing it again.
    assert(!calls.slice(finishedCalls).some(c=>c.route==='/mark_solved'),
      'reopening re-marked the problem solved');
    // Reworking a finished problem follows the COPY the server makes, and the
    // page becomes a working page again.
    reopenCopy='copy-of-finished';
    doc.querySelector('#editEarlier [data-edit]').click();await tick();
    assert.equal(window.inspect('sessionId'),'copy-of-finished','later grades go to the copy');
    assert.equal(doc.querySelector('#editorWrap').hidden,false);
    assert.notEqual(doc.querySelector('#submit').style.display,'none');
    assert(value.includes('records.copy()'),'with the step they chose back in the editor');
    resumeAccepted=resumeIndex=reopenCopy=null;
    console.log('PASS: Question+Plan → Code → Reflect; the question rides every stage; persistent tutor; inline references keep code/chat available; both planning methods; approval gates; draft/chat/file preservation; keyboard resizing; replies preserve editor focus; paced steps; grading errors; comparison retry; reopening an accepted step; a finished problem reopening with its code; restart; loading failures; restored work.');
    console.log('DOM behavior only. Browser layout and real CodeMirror still require Chromium.');
  } finally {await window.happyDOM.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
