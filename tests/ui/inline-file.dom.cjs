/* "Full file" opens beside the question in the coding stage, like the question
   itself, and draws the same bytes the drawer draws. */
const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path');
const {pathToFileURL} = require('node:url');
const root = path.resolve(__dirname, '../../frontend');
const read = n => fs.readFileSync(path.join(root, n), 'utf8');
const tick = () => new Promise(r => setTimeout(r, 40));
const problem = {slug:'frequency', title:'Frequency', description:'Count letters.'};
const steps = [{prompt:'Step one.',indent:0},{prompt:'Step two.',indent:0}];
const graph = {nodes:[{id:'n0',kind:'start',label:'Start'}],edges:[]};

(async () => {
  const {Window} = await import(pathToFileURL(require.resolve('happy-dom')).href);
  const window = new Window({url:'http://localhost/student.html',width:1400,height:900,
    settings:{enableJavaScriptEvaluation:true,disableJavaScriptFileLoading:true,
              disableCSSFileLoading:true,suppressInsecureJavaScriptEnvironmentWarning:true}});
  const doc = window.document;
  let fileCalls = 0;
  try {
    doc.write(read('student.html').replace(/<script\b[^>]*>[\s\S]*?<\/script>/g,''));
    window.scrollTo = () => {};
    window.CodeMirror = {fromTextArea:() => ({setOption:()=>{},setSize:()=>{},on:()=>{},
      refresh:()=>{},focus:()=>{},setCursor:()=>{},getGutterElement:()=>({offsetWidth:32}),
      getValue:()=>'',setValue:()=>{},listSelections:()=>[],setSelections:()=>{}})};
    window.fetch = async (url) => {
      const r = new URL(url,'http://localhost').pathname;
      let d = {};
      if(r==='/auth/me') d={name:'A',role:'student',student_id:'s'};
      else if(r==='/assignments') d={assignments:[]};
      else if(r==='/solved') d={slugs:[]};
      else if(r==='/student/progress') d={problems:[]};
      else if(r==='/decompose_chunks') d={session_id:'s1',header:'def frequency(txt):',chunks:steps.map(s=>({...s,prompt:''}))};
      else if(r.startsWith('/history/')) d={found:false};
      else if(r.startsWith('/session_steps/')) d={chunks:steps};
      else if(r==='/plan_graph') d=graph;
      else if(r.endsWith('/file')){ fileCalls++; d={filename:'LAB1.py',text:'def frequency(txt):\n    # YOUR CODE STARTS HERE\n    pass',written:['a'],in_progress:[],remaining:['b','c']}; }
      else if(r==='/design_review/plan') d={approved:true,reply:'ok',plan_graph:graph};
      return new window.Response(JSON.stringify(d),{status:200,headers:{'Content-Type':'application/json'}});
    };
    window.eval(['ui.js','cache.js','graphs.js','workspace.js','student.js'].map(read).join('\n')
                + '\nwindow.inspect = c => eval(c);');
    await tick();
    await window.start(problem);
    // The file belongs to the ASSIGNMENT, so the renderer needs one open.
    window.inspect('openAssign = {id: "a1", name: "LAB1"}');
    await tick();

    const card = doc.querySelector('#fileDetails');
    assert(card, 'the Full file disclosure is missing from the coding stage');
    assert.equal(doc.querySelector('#filePaper').closest('#stageCode') !== null, true,
      'it must sit in the coding stage, beside the question');
    assert.equal(doc.querySelector('#fileDetails summary h2').textContent.trim(), 'Full file');
    // Closed at first: it must not fetch anything nobody asked for.
    assert.equal(fileCalls, 0, 'closed disclosure must not fetch the file');

    // Setting .open fires `toggle` on its own - no synthetic event needed.
    card.open = true;
    await tick();
    assert.equal(fileCalls, 1, 'opening it fetches the file');
    assert(doc.querySelector('#fileInline .fileview-code'), 'the file body is drawn inline');
    assert(doc.querySelector('#fileInline .fileview-code').textContent.includes('YOUR CODE STARTS HERE'));
    assert(doc.querySelector('#fileInline .fileview-note').textContent.includes('1 of 3'));

    // Closing drops it, and reopening re-fetches: the file changes as steps
    // are accepted, so a cached copy of your own work would go stale.
    card.open = false;
    await tick();
    assert.equal(doc.querySelector('#fileInline').children.length, 0, 'closing clears it');
    card.open = true;
    await tick();
    assert.equal(fileCalls, 2, 'reopening re-fetches rather than showing a stale copy');

    console.log('inline-file OK');
  } finally { await window.happyDOM?.close?.(); }
})().catch(e => { console.error(e); process.exit(1); });
