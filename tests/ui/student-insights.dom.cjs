/* Dashboard, gradebook, and deep-link behavior with synthetic data. No network. */
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../../frontend');
const read=name=>fs.readFileSync(path.join(root,name),'utf8');
const wait=()=>new Promise(resolve=>setTimeout(resolve,30));
const fixture={
  generated_at:'2026-09-12T12:00:00Z',timezone:'UTC',
  summary:{problems:4,completed:1,independent:1,in_progress:1,not_started:2,total:8,earned:4,shown:0,remaining:4,percent:50,completion_percent:25,ungraded_problems:1,assignments:2,completed_assignments:0,active_days:2,weekly_submissions:5},
  assignments:[
    {id:'lab1',name:'Dictionaries <img src=x onerror=alert(1)>',problems:2,completed:1,independent:1,in_progress:1,total:6,earned:4,shown:0,remaining:2,percent:67,completion_percent:50,ungraded_problems:0},
    {id:'lab2',name:'Numbers',problems:2,completed:0,independent:0,in_progress:0,total:2,earned:0,shown:0,remaining:2,percent:0,completion_percent:0,ungraded_problems:1}
  ],
  problems:[
    {slug:'employee-update',title:'Employee Update',assignment_id:'lab1',status:'progress',solved:1,total:3,percent:33,last_activity:'2026-09-12T11:00:00Z',steps:[{number:1,status:'passed',attempts:2},{number:2,status:'needs_work',attempts:1},{number:3,status:'not_started',attempts:0}]},
    {slug:'word-count',title:'Word Count',assignment_id:'lab1',status:'solved',solved:3,total:3,percent:100,last_activity:'2026-09-11T11:00:00Z',steps:[{number:1,status:'passed',attempts:1},{number:2,status:'passed',attempts:1},{number:3,status:'passed',attempts:1}]},
    {slug:'digit-sum',title:'Digit Sum',assignment_id:'lab2',status:'todo',solved:0,total:2,percent:0,last_activity:null,steps:[{number:1,status:'not_started',attempts:0},{number:2,status:'not_started',attempts:0}]},
    {slug:'pending',title:'Pending',assignment_id:'lab2',status:'todo',solved:0,total:0,percent:null,last_activity:null,steps:[]}
  ],
  activity:Array.from({length:7},(_,i)=>({date:`2026-09-${String(6+i).padStart(2,'0')}`,submissions:i===5?2:i===6?3:0,passed:i===5?1:i===6?2:0,active:i>=5})),
  recent:[{slug:'employee-update',title:'Employee Update',assignment_id:'lab1',kind:'passed',step:1,at:'2026-09-12T11:00:00Z'}],next_up:['employee-update','digit-sum','pending']
};

(async()=>{
  const {Window}=await import(pathToFileURL(require.resolve('happy-dom')).href);
  const windows=[];
  let fail=false;
  const make=async(file,data=fixture,query='')=>{
    const window=new Window({url:'http://localhost/'+file+query,width:1600,height:1000,settings:{enableJavaScriptEvaluation:true,disableJavaScriptFileLoading:true,disableCSSFileLoading:true,suppressInsecureJavaScriptEnvironmentWarning:true}});
    windows.push(window);
    window.document.write(read(file).replace(/<script\b[^>]*>[\s\S]*?<\/script>/g,''));
    window.fetch=async url=>{
      const auth=String(url).includes('/auth/me');
      return new window.Response(JSON.stringify(auth?{name:'Alex Morgan',role:'student',student_id:'fixture'}:data),{status:!auth&&fail?503:200,headers:{'Content-Type':'application/json'}});
    };
    window.eval(read('ui.js')+'\n'+read('student-insights.js'));
    await wait();
    return window;
  };
  try{
    const dashboard=await make('dashboard.html');const doc=dashboard.document;
    assert.equal(doc.querySelector('[data-nav="Dashboard"]').getAttribute('aria-current'),'page');
    assert.equal(doc.querySelectorAll('.hnav a').length,3);
    assert.equal(doc.querySelectorAll('.activity-day').length,7);
    assert.equal(doc.querySelectorAll('.dash-metric').length,4);
    assert.equal(doc.querySelectorAll('.assignment-progress-item').length,2);
    assert.equal(doc.querySelectorAll('#insights img').length,0,'Titles are escaped');
    const continueLink=new URL(doc.querySelector('.learning-hero .action-link').href);
    assert.equal(continueLink.searchParams.get('assignment'),'lab1');
    assert.equal(continueLink.searchParams.get('problem'),'employee-update');
    assert.equal(doc.querySelector('.orbit-core strong').textContent,'25%');
    assert(doc.querySelector('#insightsTitle').textContent.includes('Alex'));
    doc.querySelector('[data-theme-toggle]').click();
    assert.equal(doc.documentElement.dataset.theme,'dark');
    fail=true;await dashboard.loadStudentProgress();
    assert(doc.querySelector('#progressNotice').textContent.includes('last successful update'));
    assert.equal(doc.querySelector('.orbit-core strong').textContent,'25%');
    fail=false;await dashboard.loadStudentProgress();
    assert.equal(doc.querySelector('#progressNotice').textContent,'');

    const grades=await make('student-grades.html');const g=grades.document;
    assert.equal(g.querySelector('[data-nav="Grades"]').getAttribute('aria-current'),'page');
    assert.equal(g.querySelector('.grade-total-number').textContent,'50%');
    assert.equal(g.querySelectorAll('[data-grade-detail]').length,4);
    const toggle=g.querySelector('[data-grade-detail]');toggle.click();
    assert.equal(toggle.getAttribute('aria-expanded'),'true');
    assert.equal(g.getElementById(toggle.getAttribute('aria-controls')).hidden,false);
    assert.equal(g.getElementById(toggle.getAttribute('aria-controls')).querySelectorAll('.step-result').length,3);
    toggle.click();assert.equal(toggle.getAttribute('aria-expanded'),'false');
    g.querySelector('#gradeAssignment').value='lab2';g.querySelector('#gradeAssignment').dispatchEvent(new grades.Event('change'));
    assert.equal(g.querySelectorAll('[data-grade-detail]').length,2);
    assert.equal(g.querySelector('.grade-total-number').textContent,'0%');
    assert([...g.querySelectorAll('.grade-score')].some(el=>el.textContent==='—'),'Unknown denominator remains ungraded');
    g.querySelector('#gradeSearch').value='missing';g.querySelector('#gradeSearch').dispatchEvent(new grades.Event('input'));
    assert(g.querySelector('#gradeResults').textContent.includes('No problems match'));
    g.querySelector('#clearGradeFilters').click();
    assert.equal(g.querySelectorAll('[data-grade-detail]').length,4);
    g.querySelector('#gradeStatus').value='progress';g.querySelector('#gradeStatus').dispatchEvent(new grades.Event('change'));
    assert.equal(g.querySelectorAll('[data-grade-detail]').length,1);
    assert(g.querySelector('#gradeResults').textContent.includes('Employee Update'));
    const scoped=await make('student-grades.html',fixture,'?assignment=lab1');
    assert.equal(scoped.document.querySelectorAll('[data-grade-detail]').length,2);

    const empty=structuredClone(fixture);
    empty.assignments=[];empty.problems=[];empty.recent=[];empty.next_up=[];
    empty.summary=Object.fromEntries(Object.keys(empty.summary).map(k=>[k,0]));empty.summary.percent=null;empty.summary.completion_percent=null;
    empty.activity=empty.activity.map(d=>({...d,submissions:0,passed:0,active:false}));
    const blank=await make('student-grades.html',empty);
    assert(blank.document.querySelector('#gradeResults').textContent.includes('No grades yet'));
    assert.equal(blank.document.querySelector('.grade-total-number').textContent,'—');
    fail=true;const unavailable=await make('dashboard.html');
    assert(unavailable.document.querySelector('#retryProgress'));
    assert.equal(unavailable.document.querySelector('.dash-metric'),null,'Unavailable is not zero progress');
    fail=false;unavailable.document.querySelector('#retryProgress').click();await wait();
    assert.equal(unavailable.document.querySelectorAll('.dash-metric').length,4);

    // Exercise a real dashboard deep link through student.js's loader.
    const study=new Window({url:'http://localhost/student.html?assignment=lab1&problem=employee-update',width:1600,height:1000,settings:{enableJavaScriptEvaluation:true,disableJavaScriptFileLoading:true,disableCSSFileLoading:true,suppressInsecureJavaScriptEnvironmentWarning:true}});
    windows.push(study);study.localStorage.setItem('mt.coach.tutor.v1','1');
    study.document.write(read('student.html').replace(/<script\b[^>]*>[\s\S]*?<\/script>/g,''));
    let opened=0;
    study.fetch=async url=>{
      const p=String(url);let body={};
      if(p.includes('/auth/me'))body={name:'Alex',role:'student',student_id:'fixture'};
      else if(p==='/assignments')body={assignments:[{id:'lab1',name:'Dictionaries',ready:1,total:1},{id:'hidden',name:'Unpublished draft',ready:1,total:1,published:false}]};
      else if(p==='/solved')body={slugs:[]};
      else if(p==='/assignments/lab1/problems')body={problems:[{slug:'employee-update',title:'Employee Update',description:'A problem',ready:true}]};
      else if(p==='/decompose_chunks'){opened++;body={session_id:'fixture',header:'def solve():',chunks:[{prompt:'',indent:0}]};}
      else if(p.startsWith('/history/'))body={found:false};
      return new study.Response(JSON.stringify(body),{status:200,headers:{'Content-Type':'application/json'}});
    };
    study.scrollTo=()=>{};
    study.CodeMirror={fromTextArea:()=>({setOption:()=>{},setSize:()=>{},on:()=>{},refresh:()=>{},focus:()=>{},setCursor:()=>{},getGutterElement:()=>({offsetWidth:32}),setValue:()=>{},getValue:()=>''})};
    study.eval(read('ui.js')+'\n'+read('graphs.js')+'\n'+read('student.js'));
    await wait();
    assert.equal(opened,1);
    assert.equal(study.document.querySelector('#probTitle').textContent,'Employee Update');
    assert.equal(study.document.querySelector('#cSolve').hidden,false);
    assert.equal(study.document.querySelector('#editorWrap').hidden,true,'Deep links preserve the design gate');
    assert.equal(study.document.querySelectorAll('#assignments .rowitem').length,1,'Unpublished assignments stay out of the student list');
    console.log('PASS: student navigation, dashboard metrics/activity, grade filters/details, theme toggle, safe titles, empty/error/retry states, and problem deep links with design gate.');
    console.log('DOM checks do not verify browser layout or screenshots.');
  }finally{for(const window of windows)await window.happyDOM.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
