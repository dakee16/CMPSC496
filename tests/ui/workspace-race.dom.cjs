/* A response that arrives after the student has moved on belongs to NOBODY.

   Every request here is started against problem A, held open, and released
   only once problem B is on screen. None of them may write into B: not the
   tutor's reply, not an approval that unlocks the editor, not graded code
   landing in B's accepted work, and not the busy flags their cleanup clears.

   Real ui.js / workspace.js / student.js against a synthetic server. No live
   database, model, oracle or code execution. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const root = path.resolve(__dirname, '../../frontend');
const read = name => fs.readFileSync(path.join(root, name), 'utf8');
const tick = () => new Promise(resolve => setTimeout(resolve, 30));

const A = {slug:'alpha', title:'Alpha', description:'Problem A.\n\n>>> alpha(1)\n1'};
const B = {slug:'beta',  title:'Beta',  description:'Problem B.\n\n>>> beta(2)\n2'};
const steps = [{prompt:'First step.',indent:0},{prompt:'Second step.',indent:0}];
const graph = {nodes:[{id:'n0',kind:'start',label:'Start'},{id:'n1',kind:'return',label:'Return'}],
               edges:[{src:'n0',dst:'n1'}]};

(async () => {
  const {Window} = await import(pathToFileURL(require.resolve('happy-dom')).href);
  const window = new Window({url:'http://localhost/student.html',width:1600,height:1000,
    settings:{enableJavaScriptEvaluation:true,disableJavaScriptFileLoading:true,
              disableCSSFileLoading:true,suppressInsecureJavaScriptEnvironmentWarning:true}});
  const doc = window.document;
  let value = '', session = 0;

  // Each of these parks the matching route until the test releases it, which is
  // how "still in flight while the student navigates away" is expressed.
  let holdTutor = null, holdReview = null, holdGrade = null;
  let reviewRedundant = null;      // what the reviewer reports alongside approval

  try {
    doc.write(read('student.html').replace(/<script\b[^>]*>[\s\S]*?<\/script>/g,''));
    window.scrollTo = () => {};
    window.CodeMirror = {fromTextArea:() => {
      const input = doc.createElement('textarea'); input.id='qaEditorInput';
      doc.querySelector('#editorWrap').append(input);
      return {setOption:()=>{},setSize:()=>{},on:()=>{},refresh:()=>{},focus:()=>{},
              setCursor:()=>{},getGutterElement:()=>({offsetWidth:32}),
              getValue:()=>value,setValue:v=>value=v,
              listSelections:()=>[],setSelections:()=>{}};
    }};

    window.fetch = async (url, init={}) => {
      const route = new URL(url,'http://localhost').pathname;
      let data = {}, status = 200;
      if(route==='/auth/me') data={name:'Alex',role:'student',student_id:'synthetic'};
      else if(route==='/assignments') data={assignments:[]};
      else if(route==='/solved') data={slugs:[]};
      else if(route==='/student/progress') data={problems:[]};
      else if(route==='/decompose_chunks')
        data={session_id:'s'+(++session),header:'def f():',
              chunks:steps.map(s=>({...s,prompt:''}))};
      else if(route.startsWith('/history/')) data={found:false};
      else if(route.startsWith('/session_steps/')) data={chunks:steps};
      else if(route==='/tutor_chat'){
        if(holdTutor) await holdTutor;
        data={reply:'REPLY FOR PROBLEM A',ready:false,offtrack:true,
              offtrack_reason:'a-reason'};
      }
      else if(route==='/plan_graph') data=graph;
      else if(route==='/design_review/plan'||route==='/design_review'){
        if(holdReview) await holdReview;
        data={approved:true,reply:'APPROVED PROBLEM A',plan_graph:graph,
              redundant:reviewRedundant};
      }
      else if(route==='/grade_chunk'){
        if(holdGrade) await holdGrade;
        data={verdict:'correct',reason:'A passes.',index:1,completed:false,
              solved_independently:true};
      }
      else if(route==='/mark_solved') data={ok:true};
      else if(route==='/graphs') data={plan:graph,code:graph,comparison:{similarity:1,notes:[]}};
      else if(route.endsWith('/restart')) data={ok:true};
      else throw new Error('Unexpected request: '+route);
      return new window.Response(JSON.stringify(data),{status,
        headers:{'Content-Type':'application/json'}});
    };

    window.eval(['ui.js','cache.js','graphs.js','workspace.js','student.js'].map(read).join('\n')
                + '\nwindow.inspect = code => eval(code);');
    await tick();

    /* ── 1. a tutor reply cannot land in another problem's chat ────────── */
    await window.start(A);
    let release;
    holdTutor = new Promise(r => { release = r; });
    const pending = window.sendToTutor('My plan for A');
    await tick();
    await window.start(B);                       // student moves to problem B
    release(); holdTutor = null;
    await pending; await tick();

    const chat = await window.inspect('chatLog.map(m => m.content).join("|")');
    assert(!chat.includes('REPLY FOR PROBLEM A'),
      'A tutor reply for one problem was appended to another problem\'s chat');
    assert(!doc.querySelector('#clog').textContent.includes('REPLY FOR PROBLEM A'),
      'A stale tutor reply was drawn into the new problem\'s transcript');
    assert(!doc.querySelector('#fork'),
      'The wrong-direction fork opened over a problem the tutor never read');
    // ...and the new workspace is still usable: a guarded cleanup must not
    // leave the chat locked behind a busy flag nobody will ever clear.
    assert.equal(await window.inspect('chatBusy'), false,
      'chatBusy stayed set after an abandoned request, locking the chat');

    /* ── 2. an approval cannot unlock a problem it never reviewed ──────── */
    await window.start(A);
    holdReview = new Promise(r => { release = r; });
    const review = window.submitPlanGraph();
    await tick();
    await window.start(B);
    release(); holdReview = null;
    await review; await tick();

    assert.equal(doc.querySelector('#tabCode').getAttribute('aria-disabled'),'true',
      'An approval for one problem unlocked the editor on another');
    assert(!doc.querySelector('#designMsg').textContent.includes('APPROVED PROBLEM A'),
      'An approval verdict for one problem was shown on another');
    assert.equal(await window.inspect('tutorReleased'), false,
      'A stale approval released the tutor gate on the wrong problem');

    /* ── 3. a grade cannot file its code as another problem's work ─────── */
    await window.start(A);
    await window.inspect('tutorReleased = true; openGate();');
    await tick();
    value = 'kept_from_A = 1';
    holdGrade = new Promise(r => { release = r; });
    doc.querySelector('#submit').click();
    await tick();
    await window.start(B);
    release(); holdGrade = null;
    await tick(); await tick();

    const acceptedB = await window.inspect('JSON.stringify(accepted)');
    assert(!acceptedB.includes('kept_from_A'),
      'Code graded for one problem was filed as another problem\'s accepted work');
    assert.equal(await window.inspect('idx'), 0,
      'A stale grade advanced the step index of the problem now open');

    /* ── 4. Tab cannot walk out of an open confirmation dialog ─────────── */
    // Both dialogs claim aria-modal="true" and neither is a real <dialog>, so
    // nothing trapped Tab and two presses put a keyboard user on the page
    // behind the modal - with the buttons that dismiss it now out of reach.
    // Focus is moved OUT explicitly here because a synthetic Tab does not move
    // it by itself; what is being tested is that the handler pulls it back.
    await window.start(A);
    await tick();
    window.openRestart();
    const dialog = doc.querySelector('#restartModal');
    assert.equal(dialog.hidden, false, 'the restart dialog did not open');
    doc.querySelector('#cinput').focus();
    assert(!dialog.contains(doc.activeElement), 'focus should start outside');
    doc.dispatchEvent(new window.KeyboardEvent('keydown',{key:'Tab',bubbles:true}));
    assert(dialog.contains(doc.activeElement),
      'Tab left focus outside an open dialog instead of returning it');
    doc.querySelector('#cinput').focus();
    doc.dispatchEvent(new window.KeyboardEvent('keydown',{key:'Tab',shiftKey:true,bubbles:true}));
    assert(dialog.contains(doc.activeElement),
      'Shift+Tab left focus outside an open dialog instead of returning it');
    // Wrapping at the end stays inside, too.
    const stops = [...dialog.querySelectorAll('button')];
    stops[stops.length - 1].focus();
    doc.dispatchEvent(new window.KeyboardEvent('keydown',{key:'Tab',bubbles:true}));
    assert.equal(doc.activeElement, stops[0],
      'Tab off the last control must wrap to the first, not leave');
    window.closeRestart();
    // With nothing open the key is not touched - ordinary tabbing still works.
    doc.querySelector('#cinput').focus();
    doc.dispatchEvent(new window.KeyboardEvent('keydown',{key:'Tab',bubbles:true}));
    assert.equal(doc.activeElement.id, 'cinput',
      'the trap must do nothing when no dialog is open');

    /* ── 5. an approved-but-redundant plan offers the choice ───────────── */
    // The reviewer has always returned `redundant`; nothing ever read it, so
    // the popup could not fire for anyone and the editor unlocked straight past
    // the teaching moment it exists for.
    await window.start(A);
    reviewRedundant = {source:'list', target:'dictionary',
                       note:'You could build the dictionary directly.'};
    await window.submitPlanGraph();
    await tick();
    assert.equal(doc.querySelector('#optimizeModal').hidden, false,
      'an approved-but-redundant plan did not offer the optimize choice');
    assert(doc.querySelector('#optimizeNote').textContent.includes('directly'),
      'the popup did not carry the reviewer\'s note');
    assert.equal(await window.inspect('tutorReleased'), false,
      'the editor unlocked before the student answered the choice');
    // Continuing is always available - an approved plan stays approved.
    doc.querySelector('#optimizeContinue').click();
    assert.equal(doc.querySelector('#optimizeModal').hidden, true);
    assert.equal(await window.inspect('tutorReleased'), true,
      'continuing from the popup must still unlock the editor');

    /* ── 6. ...and an ordinary approval is untouched by that path ───────── */
    await window.start(A);
    reviewRedundant = null;
    await window.submitPlanGraph();
    await tick();
    assert.equal(doc.querySelector('#optimizeModal').hidden, true,
      'a plan with nothing redundant in it must not be stopped');
    assert.equal(await window.inspect('tutorReleased'), true,
      'an ordinary approval must unlock the editor directly');

    await tick(); await tick();   // let pending work settle before teardown
    console.log('workspace-race.dom.cjs OK');
  } finally {
    await window.happyDOM?.close?.();
  }
})().catch(err => { console.error(err); process.exit(1); });
