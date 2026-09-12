/* Fast DOM checks: no network, browser binary, backend, or model calls.
   CodeMirror is stubbed here; real layout/editor checks live in the browser suite. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const root = path.resolve(__dirname, '../../frontend');
const read = name => fs.readFileSync(path.join(root, name), 'utf8');

(async () => {
  const {Window} = await import(pathToFileURL(require.resolve('happy-dom')).href);
  const window = new Window({url:'http://localhost/student.html',width:1600,height:1000,
    settings:{enableJavaScriptEvaluation:true,disableJavaScriptFileLoading:true,
      disableCSSFileLoading:true,suppressInsecureJavaScriptEnvironmentWarning:true}});
  const doc = window.document;
  try {
    doc.write(read('student.html').replace(/<script\b[^>]*>[\s\S]*?<\/script>/g,''));
    window.localStorage.setItem('mt.coach.tutor.v1','1');
    window.fetch = async url => new window.Response(JSON.stringify(
      String(url).includes('/auth/me') ? {name:'UI fixture',role:'student',student_id:'fixture'} :
      String(url).includes('/assignments') ? {assignments:[]} : {slugs:[]}),
      {status:200,headers:{'Content-Type':'application/json'}});
    window.scrollTo = () => {};
    let editorValue = '', selections = [{anchor:{line:0,ch:4},head:{line:0,ch:4}}];
    const options = {};
    window.CodeMirror = {fromTextArea:()=>({
      setOption:(key,value)=>options[key]=value,
      setSize:()=>{},on:()=>{},refresh:()=>{},focus:()=>{},setCursor:()=>{},
      getGutterElement:()=>({offsetWidth:32}),
      getValue:()=>editorValue,setValue:value=>editorValue=value,
      listSelections:()=>selections,setSelections:value=>selections=value
    })};
    // One evaluation matches classic browser scripts' shared lexical scope.
    window.eval(read('ui.js') + '\n' + read('graphs.js') + '\n' + read('student.js') +
      '\nwindow.acadiaEval = code => eval(code);');
    await new Promise(resolve=>setTimeout(resolve,20));
    assert.equal(doc.documentElement.dataset.theme,'light');
    assert.equal(doc.querySelector('.brand-name').firstChild.textContent,'ACADIA');
    doc.querySelector('[data-theme-toggle]').click();
    assert.equal(doc.documentElement.dataset.theme,'dark');
    assert.equal(window.localStorage.getItem('mt.theme'),'dark');
    doc.querySelector('[data-theme-toggle]').click();
    assert.equal(doc.documentElement.dataset.theme,'light');

    window.renderStatement('Read `value` and preserve the input.\nThis is one paragraph.\n\n>>> data = {"a": [1, 2]}\n>>> solve(data)\n{"a": [2, 3]}\n\nAfter the example.');
    assert.equal(doc.querySelectorAll('#statement pre').length,1);
    assert.equal(doc.querySelector('#statement pre').textContent,'>>> data = {"a": [1, 2]}\n>>> solve(data)\n{"a": [2, 3]}');
    assert.equal(doc.querySelector('#statement p').textContent,'Read value and preserve the input. This is one paragraph.');
    assert.equal(doc.querySelector('#statement p:last-child').textContent,'After the example.');
    window.renderStatement('```python\ndef f(n):\n    if n:\n        return "<img src=x onerror=alert(1)>"\n```');
    assert.equal(doc.querySelectorAll('#statement img').length,0);
    assert(doc.querySelector('#statement pre').textContent.includes('\n        return'));
    window.renderStatement('Example: f(1) -> 2; an explanation that must not disappear');
    assert(doc.querySelector('#statement').textContent.includes('an explanation that must not disappear'));
    window.renderStatement('- First thing\n- Second thing\n\n1. Plan\n2. Code');
    assert.equal(doc.querySelectorAll('#statement li').length,4);
    window.renderStatement('```python\n# unfinished fence\nx = 1');
    assert.equal(doc.querySelector('#statement pre').textContent,'# unfinished fence\nx = 1');
    window.renderStatement('');
    assert.equal(doc.querySelector('#statement').childElementCount,0);

    window.acadiaEval(`chunks = [{prompt:'First step',indent:0},{prompt:'Second step',indent:0},{prompt:'Third step',indent:0}]; header='def solve(n):'; ensureEditor(); render(); applyTutorGate();`);
    assert.equal(doc.querySelector('#workStep').hidden,true);
    assert.equal(doc.querySelector('#editorWrap').hidden,true);
    assert.equal(options.readOnly,'nocursor');
    assert.equal(doc.querySelector('#submit').disabled,true);
    window.acadiaEval('tutorReleased=true; applyTutorGate(); render();');
    assert.equal(doc.querySelector('#workStep').hidden,false);
    assert.equal(options.readOnly,false);
    assert.equal(doc.querySelector('#submit').disabled,false);
    assert.equal(doc.querySelector('.studio-brief #stepper'),null);
    assert.equal(doc.querySelector('#workStep #stepCount').textContent,'Step 1 of 3');
    window.markUnlocked('Design accepted');
    assert.equal(doc.querySelector('#designOK').nextElementSibling.id,'workStep');
    editorValue='    draft = n + 1';
    doc.querySelector('#focusWork').click();
    assert(doc.body.classList.contains('focus-workspace'));
    assert.equal(editorValue,'    draft = n + 1');
    doc.querySelector('#focusWork').click();
    assert(!doc.body.classList.contains('focus-workspace'));
    assert.equal(editorValue,'    draft = n + 1');
    window.acadiaEval(`accepted=[{code:'n = 1',how:'own'}];idx=1;renderStepper();`);
    doc.querySelector('#stepper button').click();
    assert.equal(doc.querySelector('#reviewBox').hidden,false);
    doc.querySelector('#backToNow').click();
    assert.equal(editorValue,'    draft = n + 1','Review must not discard the current draft');
    assert.equal(doc.querySelector('#reviewBox').hidden,true);
    assert.equal(doc.querySelector('#stepCount').textContent,'Step 2 of 3');
    window.acadiaEval('idx=chunks.length; applyTutorGate();');
    assert.equal(doc.querySelector('#editorWrap').hidden,true,'Completed sessions stay read-only');
    window.acadiaEval('idx=1; applyTutorGate();');
    const message=window.bubble('me','previous_year = year - 1\nnew_dict = {}');
    assert.equal(message.querySelector('pre').textContent,'previous_year = year - 1\nnew_dict = {}');
    window.happyDOM.setWindowSize({width:1024,height:800});
    window.setTutorOpen(false,false);
    assert.equal(doc.querySelector('#cform').inert,true);
    window.setTutorOpen(true,false);
    assert.equal(doc.querySelector('#cform').inert,false);
    assert.equal(doc.querySelector('#sheetTog').getAttribute('aria-expanded'),'true');
    window.setTutorOpen(false,false);
    assert.equal(doc.querySelector('#sheetTog').getAttribute('aria-expanded'),'false');
    const ids=[...doc.querySelectorAll('[id]')].map(el=>el.id);
    assert.equal(ids.length,new Set(ids).size,'Duplicate IDs');

    // Validate semantic foreground/background pairs in both token palettes.
    const tokens=read('tokens.css');
    const blocks=tokens.split(':root[data-theme="dark"]');
    const parse=text=>Object.fromEntries([...text.matchAll(/(--[\w-]+):\s*(#[0-9a-f]{6})/gi)].map(m=>[m[1],m[2]]));
    const light=parse(blocks[0]),dark={...light,...parse(blocks[1])};
    const luminance=hex=>{
      const c=hex.slice(1).match(/../g).map(v=>parseInt(v,16)/255).map(v=>v<=.04045?v/12.92:((v+.055)/1.055)**2.4);
      return c[0]*.2126+c[1]*.7152+c[2]*.0722;
    };
    const pairs=[['--text','--surface'],['--text-muted','--surface'],['--text-subtle','--bg'],['--on-accent','--accent'],['--accent-ink','--surface'],['--code-text','--code-bg'],['--code-dim','--code-bg'],['--code-green','--code-bg'],['--code-pink','--code-bg'],['--code-yellow','--code-bg'],['--success','--success-bg'],['--warning','--warning-bg'],['--danger','--danger-bg']];
    for(const [name,palette] of [['light',light],['dark',dark]]) for(const [fg,bg] of pairs){
      const a=luminance(palette[fg]),b=luminance(palette[bg]);
      const ratio=(Math.max(a,b)+.05)/(Math.min(a,b)+.05);
      assert(ratio>=4.5,`${name} ${fg}/${bg} contrast ${ratio.toFixed(2)} < 4.5`);
    }
    for(const name of fs.readdirSync(root).filter(name=>/\.(js|html)$/.test(name))){
      assert(!read(name).includes('MicroTutor'),name+' still contains the old UI name');
    }
    console.log('PASS: DOM behavior, source-safe formatting, themes, design gate, step review/draft preservation, focus, drawer accessibility, IDs, and 26 contrast pairs.');
    console.log('Layout, real CodeMirror, and screenshot checks require acadia.smoke.cjs in Chromium.');
  } finally {await window.happyDOM.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
