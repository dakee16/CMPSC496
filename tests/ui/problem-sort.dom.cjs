/* The sort control has to mean something INSIDE a class group too.

   A class's methods were re-sorted by the teacher's file order after the chosen
   comparator had already run, so picking "A - Z" left Stack sitting in
   isEmpty, __len__, push, pop, peek and nothing on screen said why. File order
   survives only as the tiebreak.

   Real ui.js / workspace.js / student.js against a synthetic server. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const root = path.resolve(__dirname, '../../frontend');
const read = name => fs.readFileSync(path.join(root, name), 'utf8');
const tick = () => new Promise(resolve => setTimeout(resolve, 30));

// Deliberately in the teacher's file order, which is NOT alphabetical.
const members = ['isEmpty', '__len__', 'push', 'pop', 'peek'];
const PROBLEMS = members.map((name, i) => ({
  slug: `stack-${name.replace(/_/g, '')}`.toLowerCase(),
  title: name, group_slug: 'stack', group_title: 'Stack',
  group_order: 0, member_order: i, ready: true,
}));

(async () => {
  const {Window} = await import(pathToFileURL(require.resolve('happy-dom')).href);
  const window = new Window({url:'http://localhost/student.html',width:1400,height:900,
    settings:{enableJavaScriptEvaluation:true,disableJavaScriptFileLoading:true,
              disableCSSFileLoading:true,suppressInsecureJavaScriptEnvironmentWarning:true}});
  const doc = window.document;
  try {
    doc.write(read('student.html').replace(/<script\b[^>]*>[\s\S]*?<\/script>/g,''));
    window.scrollTo = () => {};
    window.CodeMirror = {fromTextArea:() => ({setOption:()=>{},setSize:()=>{},on:()=>{},
      refresh:()=>{},focus:()=>{},setCursor:()=>{},getGutterElement:()=>({offsetWidth:32}),
      getValue:()=>'',setValue:()=>{},listSelections:()=>[],setSelections:()=>{}})};
    window.fetch = async (url) => {
      const route = new URL(url,'http://localhost').pathname;
      const data = route==='/auth/me' ? {name:'Alex',role:'student',student_id:'s'}
                 : route==='/assignments' ? {assignments:[]}
                 : route==='/solved' ? {slugs:[]}
                 : {};
      return new window.Response(JSON.stringify(data),{status:200,
        headers:{'Content-Type':'application/json'}});
    };
    window.eval(['ui.js','cache.js','graphs.js','workspace.js','student.js'].map(read).join('\n')
                + '\nwindow.inspect = code => eval(code);');
    await tick();

    const shown = () => [...doc.querySelectorAll('#problems .rowitem .rname')]
      .map(el => el.textContent.trim()).filter(Boolean);
    // Read before the test starts driving the control.
    const defaultSort = window.inspect('pSort');

    window.inspect(`PROBLEMS = ${JSON.stringify(PROBLEMS)}; pSort = 'alpha'; renderProblems();`);
    const alpha = shown();
    assert(alpha.length >= members.length, `nothing rendered: ${JSON.stringify(alpha)}`);
    const alphaMembers = alpha.filter(t => members.includes(t));
    assert.deepEqual(alphaMembers, [...members].sort((a,b)=>a.localeCompare(b)),
      'A - Z must alphabetize a class\'s methods, not leave them in file order');

    // Status order still applies inside the group, and equal states fall back
    // to the teacher's file order rather than to nothing.
    window.inspect(`pSort = 'status'; renderProblems();`);
    const byStatus = shown().filter(t => members.includes(t));
    assert.deepEqual(byStatus, members,
      'with every method in the same state, file order is the tiebreak');

    // ...and file order is a sort in its own right - the DEFAULT one, because
    // the teacher's file is the syllabus and the alphabet is not.
    window.inspect(`pSort = 'file'; renderProblems();`);
    assert.deepEqual(shown().filter(t => members.includes(t)), members,
      'File order must show the methods in the order the teacher wrote them');
    assert.equal(defaultSort, 'file',
      'file order is only useful as a default if it IS the default');

    console.log('problem-sort.dom.cjs OK');
  } finally {
    await window.happyDOM?.close?.();
  }
})().catch(err => { console.error(err); process.exit(1); });
