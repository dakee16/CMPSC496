const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {pathToFileURL}=require('node:url');
const fixture=require('./teacher-dashboard.fixture.cjs');
const read=name=>fs.readFileSync(path.resolve(__dirname,'../../frontend',name),'utf8');
const wait=()=>new Promise(r=>setTimeout(r,30));

(async()=>{
  const {Window}=await import(pathToFileURL(require.resolve('happy-dom')).href);
  const windows=[];
  const make=async(initial=fixture(),failed=false,file='teacher.html')=>{
    const w=new Window({url:'http://localhost/'+file+(file==='grades.html'?'?assignment=hw3':''),settings:{enableJavaScriptEvaluation:true,
      disableJavaScriptFileLoading:true,disableCSSFileLoading:true,suppressInsecureJavaScriptEnvironmentWarning:true}});
    windows.push(w);
    w.document.write(read(file).replace(/<script\b[^>]*>[\s\S]*?<\/script>/g,''));
    let fail=failed, payload=initial;
    const requests=[],paths=[];
    w.fetch=async(url,options)=>{
      const p=new URL(url,'http://localhost');
      paths.push(p.pathname);
      let data={};
      if(p.pathname==='/auth/me')data={student_id:'teacher-fixture',name:'Pat Teacher',role:'teacher'};
      if(p.pathname==='/assignments')data={assignments:initial.assignments};
      if(p.pathname==='/assignment_template')data={content:'def sample():\n    pass'};
      if(p.pathname==='/teacher/dashboard'){
        requests.push({url:p,options});
        data=p.searchParams.has('assignment_id')?fixture(p.searchParams.get('assignment_id')):payload;
        if(fail)return new w.Response('{}',{status:503});
      }
      return new w.Response(JSON.stringify(data),{status:200,headers:{'Content-Type':'application/json'}});
    };
    const script=file==='grades.html'?'grades.js':file==='teacher-assignments.html'?'teacher.js':'teacher-insights.js';
    w.eval(read('ui.js')+'\n'+read('cache.js')+'\n'+read(script));
    await wait();
    return {w,doc:w.document,requests,paths,setFail:v=>fail=v,setData:v=>payload=v};
  };
  try{
    const {w,doc,requests,paths,setFail,setData}=await make();
    assert.equal(doc.querySelector('#list'),null);
    assert.equal(doc.querySelector('#newAssignment'),null);
    assert.equal(doc.querySelector('#uploadDrawer'),null);
    assert.equal(paths.includes('/assignments'),false);
    assert.equal(paths.includes('/assignment_template'),false);
    assert.equal(doc.querySelector('[data-nav="Home"]').getAttribute('aria-current'),'page');
    assert.equal(doc.querySelector('[data-nav="Assignments"]').getAttribute('href'),'teacher-assignments.html');
    // ONE SQUARE TILE PER PROBLEM, each a LINK to that problem's own page.
    const tried=fixture().problems.filter(p=>p.attempted);
    const tiles=[...doc.querySelectorAll('.problem-tile')];
    assert.equal(tiles.length,tried.length,'every problem students tried has a tile');
    assert.equal(tiles[0].tagName,'A','a tile is a link, so it opens a page');
    const href=new URL(tiles[0].href);
    assert.equal(href.pathname,'/teacher-review.html');
    assert.equal(href.searchParams.get('slug'),tried[0].slug);
    assert.equal(href.searchParams.get('assignment'),tried[0].assignment_id);
    assert.match(tiles[0].textContent,new RegExp(tried[0].title));
    assert.match(tiles[0].querySelector('.tile-count').textContent,/8\s*students need help/);
    // Highlighted exactly while someone has an open issue - and the count says
    // so in text, so colour is never the only signal.
    tried.forEach((p,i)=>assert.equal(tiles[i].classList.contains('needs-attention'),p.needs_help>0,p.slug));
    assert(tiles.some(t=>!t.classList.contains('needs-attention')),'a problem with nothing open stays plain');
    // What was removed stays removed: the suggestion box, the bars, the
    // panel that unfolded under the list.
    for(const gone of ['.insight-takeaway','#reviewPriority','.insight-track','#insightFocus','#showAllInsights'])
      assert.equal(doc.querySelector(gone),null,gone+' is gone');
    assert.match(doc.querySelector('#chartExplanation').textContent,/haven’t corrected yet/);
    assert.equal(requests[0].options.cache,'no-store');

    doc.querySelector('#insightAssignment').value='hw3';
    doc.querySelector('#insightAssignment').dispatchEvent(new w.Event('change'));await wait();
    assert.equal(doc.querySelectorAll('.problem-tile').length,fixture('hw3').problems.filter(p=>p.attempted).length);
    assert.equal(requests.at(-1).url.searchParams.get('assignment_id'),'hw3');
    setFail(true);doc.querySelector('#refreshInsights').click();await wait();
    assert.match(doc.querySelector('#insightNotice').textContent,/last successful update/);
    assert(doc.querySelectorAll('.problem-tile').length>0,'the last good tiles stay up');
    setFail(false);doc.querySelector('#retryInsights').click();await wait();
    assert.equal(doc.querySelector('#insightNotice').textContent,'');

    const unsafe=fixture();unsafe.problems[0].title='<img src=x onerror=alert(1)>';
    setData(unsafe);doc.querySelector('#insightAssignment').value='';
    doc.querySelector('#insightAssignment').dispatchEvent(new w.Event('change'));await wait();
    assert.equal(doc.querySelectorAll('#classInsights img, #classInsights script').length,0);
    assert.match(doc.querySelector('.problem-tile').textContent,/<img/,'shown as text, not run');
    const empty=fixture();empty.problems=[];empty.assignments=[];
    empty.summary={students:0,active:0,needs_help:0,problems:0,attempted_problems:0,indeterminate:0};
    const blank=await make(empty);
    assert.match(blank.doc.querySelector('#classInsights').textContent,/Publish an assignment/);
    assert.equal(blank.doc.querySelector('#insightAssignment').disabled,true);
    const bad=await make(fixture(),true);
    assert.equal(bad.doc.querySelectorAll('.insight-board').length,0);
    assert.match(bad.doc.querySelector('#insightNotice').textContent,/temporarily unavailable/);
    bad.setFail(false);bad.doc.querySelector('#retryInsights').click();await wait();
    assert.equal(bad.doc.querySelectorAll('.insight-board').length,1);
    // Every issue seen: every tile back to normal.
    const clear=fixture();clear.summary.needs_help=0;
    clear.problems.forEach(p=>{p.needs_help=0;p.seen=p.follow_up.length;p.follow_up.forEach(s=>s.seen=true);});
    const caughtUp=await make(clear);
    assert.equal(caughtUp.doc.querySelectorAll('.needs-attention').length,0);
    assert.match(caughtUp.doc.querySelector('.problem-tile .tile-count').textContent,/0\s*students need help/);
    const grades=await make(fixture(),false,'grades.html');
    assert.equal(grades.doc.querySelector('#pick').value,'hw3','Gradebook opens the assignment from the dashboard link');
    const assignments=await make(fixture(),false,'teacher-assignments.html');
    assert.equal(assignments.doc.querySelectorAll('#listBody tbody tr').length,2);
    assert.equal(assignments.doc.querySelector('#classInsights'),null);
    assert.equal(assignments.paths.includes('/teacher/dashboard'),false);
    assert.equal(assignments.doc.querySelector('[data-nav="Assignments"]').getAttribute('aria-current'),'page');
    assert.equal(assignments.doc.querySelector('[data-nav="Home"]').getAttribute('aria-current'),null);
    assignments.doc.querySelector('#newAssignment').click();
    assert.equal(assignments.doc.querySelector('#uploadDrawer').open,true);
    assignments.doc.querySelector('#closeUpload').click();
    assert.equal(assignments.doc.querySelector('#uploadDrawer').open,false);
    console.log('PASS: teacher problem tiles linking to their review pages, attention highlighting and the seen count, filters, errors, separate assignment navigation, library and upload drawer.');
  }finally{for(const w of windows)await w.happyDOM.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
