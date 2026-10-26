/* Submitting a plan while the tutor is still answering.

   THE LEFTOVER (26 Oct). Submit stayed available while a tutor request was in
   flight. The plan was then judged WITHOUT the message the tutor was answering,
   and the tutor's reply landed afterwards - as a question under an approved
   plan. Now Submit stops the tutor request, sends its id and the message it was
   answering, and the reply never reaches the page.

   Real ui.js / workspace.js / student.js against a synthetic server. The
   "tutor" really is parked mid-request and honours its AbortSignal, the way a
   browser's fetch does. No live database, model, oracle or code execution. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const root = path.resolve(__dirname, '../../frontend');
const read = name => fs.readFileSync(path.join(root, name), 'utf8');
const tick = () => new Promise(resolve => setTimeout(resolve, 30));

const A = {slug:'alpha', title:'Alpha', description:'Problem A.\n\n>>> alpha(1)\n1'};
const steps = [{prompt:'First step.',indent:0},{prompt:'Second step.',indent:0}];
const graph = {nodes:[{id:'n0',kind:'start',label:'Start'},{id:'n1',kind:'return',label:'Return'}],
               edges:[{src:'n0',dst:'n1'}]};

(async () => {
  const {Window} = await import(pathToFileURL(require.resolve('happy-dom')).href);
  const window = new Window({url:'http://localhost/student.html',width:1600,height:1000,
    settings:{enableJavaScriptEvaluation:true,disableJavaScriptFileLoading:true,
              disableCSSFileLoading:true,suppressInsecureJavaScriptEnvironmentWarning:true}});
  const doc = window.document;
  let value = '', session = 0, replies = 0;
  const tutorBodies = [], planBodies = [], uploadBodies = [];
  let holdTutor = null, aborted = 0, tutorAnswer = null;

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
        tutorBodies.push(JSON.parse(init.body));
        if(holdTutor){
          // A browser's fetch rejects with AbortError the moment its signal aborts.
          await new Promise((resolve, reject) => {
            if(init.signal){
              const stop = () => { aborted++; reject(Object.assign(new Error('aborted'),{name:'AbortError'})); };
              if(init.signal.aborted) return stop();
              init.signal.addEventListener('abort', stop);
            }
            holdTutor.then(resolve);
          });
        }
        data = tutorAnswer || {reply:'TUTOR REPLY #'+(++replies),ready:false,offtrack:false};
      }
      else if(route==='/plan_graph') data=graph;
      else if(route==='/design_review/plan'){
        planBodies.push(JSON.parse(init.body));
        data={approved:true,reply:'APPROVED PLAN',redundant:null};
      }
      else if(route==='/design_review'){
        uploadBodies.push(init.body);
        data={approved:true,reply:'APPROVED DRAWING',plan_graph:graph};
      }
      else if(route==='/mark_solved') data={ok:true};
      else throw new Error('Unexpected request: '+route);
      return new window.Response(JSON.stringify(data),{status,
        headers:{'Content-Type':'application/json'}});
    };

    window.eval(['ui.js','cache.js','graphs.js','workspace.js','student.js'].map(read).join('\n')
                + '\nwindow.inspect = code => eval(code);');
    await tick();
    const clog = () => doc.querySelector('#clog').textContent;
    // A tutor row with no words in it: the dots were removed but the row stayed.
    const emptyTutorRows = () => [...doc.querySelectorAll('#clog .bub.bot')]
      .filter(row => !(row.querySelector('.btext') || {textContent:''}).textContent.trim()).length;

    /* ── 1. every tutor request carries its own id ─────────────────────── */
    await window.start(A);
    await window.sendToTutor('First message about my plan');
    await tick();
    await window.sendToTutor('A second message');
    await tick();
    const ids = tutorBodies.map(b => b.call_id);
    assert(ids.length === 2 && ids.every(i => typeof i === 'string' && i.length >= 8),
      'every tutor request must carry a call_id');
    assert.notEqual(ids[0], ids[1], 'two tutor requests must not share an id');

    /* ── 2. Submit while the tutor is answering ────────────────────────── */
    let release;
    holdTutor = new Promise(r => { release = r; });
    const pending = window.sendToTutor('And an empty list should give zero.');
    await tick();
    assert.equal(await window.inspect('chatBusy'), true, 'the tutor request is in flight');
    const parked = tutorBodies[tutorBodies.length - 1];
    const submit = window.submitPlanGraph();
    await tick();
    release(); holdTutor = null;                   // the server would still answer it
    await pending; await submit; await tick();

    assert.equal(aborted, 1, 'Submit did not stop the tutor request that was in flight');
    const sent = planBodies[planBodies.length - 1];
    assert.equal(sent.superseded, parked.call_id,
      'the submit did not say which tutor request it superseded');
    assert.equal(sent.latest, 'And an empty list should give zero.',
      'the submit did not carry the message the tutor was answering');
    assert(clog().includes('APPROVED PLAN'), 'the plan verdict was not shown');
    assert(!clog().includes('TUTOR REPLY #3'),
      'the tutor\'s reply to the message Submit superseded was shown anyway');
    assert(!doc.querySelector('#clog .think'),
      'the "tutor is thinking" dots were left behind');
    assert.equal(emptyTutorRows(), 0, 'an empty Tutor row was left in the conversation');
    assert(!clog().includes('Could not reach the tutor'),
      'stopping the tutor was reported to the student as a failure');
    assert.equal(await window.inspect('chatBusy'), false,
      'chatBusy stayed set after the tutor request was stopped');
    const log = JSON.parse(await window.inspect('JSON.stringify(chatLog)'));
    assert.equal(log[log.length - 1].content, 'And an empty list should give zero.',
      'the student\'s message is part of the conversation and must stay in it');
    assert(!log.some(m => m.role === 'assistant' && m.content === 'TUTOR REPLY #3'),
      'the stopped reply was written into the conversation');
    assert.equal(await window.inspect('tutorReleased'), true, 'an approval must still unlock');

    /* ── 3. nothing in flight: the submit sends neither field ──────────── */
    await window.start(A);
    await window.sendToTutor('A plan with nothing pending');
    await tick();
    await window.submitPlanGraph();
    await tick();
    const plain = planBodies[planBodies.length - 1];
    assert(!plain.superseded && !plain.latest,
      'a submit with no tutor request in flight must not claim to supersede one');

    /* ── 4. a reply the server dropped is not drawn either ─────────────── */
    await window.start(A);
    tutorAnswer = {reply:'', ready:false, offtrack:false, discarded:true};
    await window.sendToTutor('Said from another tab');
    await tick();
    tutorAnswer = null;
    assert(!doc.querySelector('#clog .think'), 'a dropped reply left the thinking dots on screen');
    assert.equal(emptyTutorRows(), 0, 'a dropped reply left an empty Tutor row');
    assert.equal(await window.inspect('chatBusy'), false);
    const log2 = JSON.parse(await window.inspect('JSON.stringify(chatLog)'));
    assert(!log2.some(m => m.role === 'assistant' && m.content === ''),
      'a dropped reply was written into the conversation as an empty turn');

    /* ── 5. stopping an old request never freezes the chat ─────────────── */
    await window.start(A);
    await window.sendToTutor('First');
    await tick();
    holdTutor = new Promise(r => { release = r; });
    const p1 = window.sendToTutor('Parked');
    await tick();
    const stopped = window.submitPlanGraph();
    await tick();
    release(); holdTutor = null;
    await p1; await stopped; await tick();
    // The editor is open now; the next message is an ordinary, answered one.
    await window.sendToTutor('Ordinary follow-up');
    await tick();
    assert.equal(await window.inspect('chatBusy'), false);
    assert(clog().includes('TUTOR REPLY'), 'the chat stopped working after a stopped request');

    /* ── 6. the same for a drawing ─────────────────────────────────────── */
    await window.start(A);
    await window.sendToTutor('My plan, drawn on paper');
    await tick();
    await window.inspect("designFileRef = new File(['png'], 'plan.png', {type:'image/png'}); 0");
    holdTutor = new Promise(r => { release = r; });
    const drawnPending = window.sendToTutor('One more thing about the empty list');
    await tick();
    const drawnParked = tutorBodies[tutorBodies.length - 1];
    const aboutToStop = aborted;
    const upload = window.uploadDesign();
    await tick();
    release(); holdTutor = null;
    await drawnPending; await upload; await tick();
    assert.equal(aborted, aboutToStop + 1, 'uploading a drawing did not stop the tutor request');
    const form = uploadBodies[uploadBodies.length - 1];
    assert.equal(form.get('superseded'), drawnParked.call_id,
      'the upload did not say which tutor request it superseded');
    assert.equal(form.get('latest'), 'One more thing about the empty list',
      'the upload did not carry the message the tutor was answering');
    assert.equal(emptyTutorRows(), 0, 'an empty Tutor row was left after an upload');

    console.log('PASS: every tutor request has an id; Submit stops the one in flight, sends its id '
      + 'and the message it was answering, and the late reply never reaches the chat (no thinking '
      + 'dots, no empty row, no failure message, message kept); an uploaded drawing does the same; '
      + 'nothing in flight sends neither field; a reply the server dropped draws nothing; and the '
      + 'chat keeps working afterwards.');
  } catch (err) {
    console.error(err);
    process.exitCode = 1;
  } finally {
    await window.happyDOM.close();
  }
})();
