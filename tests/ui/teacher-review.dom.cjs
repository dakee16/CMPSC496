/* The instructor's review page (teacher-review.html): a problem's students as
   a vertical list with "Issue seen", and one student's immersive view with
   their function up to the failing step and that attempt marked in red. */
const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path');
const {pathToFileURL}=require('node:url');
const fixture=require('./teacher-dashboard.fixture.cjs');
const read=name=>fs.readFileSync(path.resolve(__dirname,'../../frontend',name),'utf8');
const wait=()=>new Promise(r=>setTimeout(r,30));

const REVIEW={slug:'invert',title:'Invert a dictionary',assignment_id:'lab1',assignment:'LAB1 Dictionaries',
  student_id:'student-0',name:'Avery Chen',seen:false,header:'def invert(d):',total_steps:3,open_steps:[2,3],
  prefix:[{number:1,prompt:'Walk every pair',code:'out = {}\nfor k, v in d.items():'}],
  step:{number:2,prompt:'Swap each pair into the result',reason:'Your step runs but the answer is wrong.',
    code:'    out[k] = k',failing_cases:['invert({1: 2})\n\nexpected: {2: 1}\nyou gave: {1: 1}'],failed_total:4,
    attempts:[{at:'2026-09-16T11:00:00Z',verdict:'incorrect',reason:'First try.',code:'out[k] = v'},
              {at:'2026-09-16T12:00:00Z',verdict:'incorrect',reason:'Your step runs but the answer is wrong.',code:'out[k] = k'}],
    first_at:'2026-09-16T11:00:00Z',last_at:'2026-09-16T12:00:00Z'}};

(async()=>{
  const {Window}=await import(pathToFileURL(require.resolve('happy-dom')).href);
  const windows=[];
  const make=async(query,{dashboard=fixture(),review=REVIEW,reviewStatus=200}={})=>{
    const w=new Window({url:'http://localhost/teacher-review.html'+query,settings:{enableJavaScriptEvaluation:true,
      disableJavaScriptFileLoading:true,disableCSSFileLoading:true,suppressInsecureJavaScriptEnvironmentWarning:true}});
    windows.push(w);
    w.document.write(read('teacher-review.html').replace(/<script\b[^>]*>[\s\S]*?<\/script>/g,''));
    const posts=[];let data=dashboard;
    w.fetch=async(url,options={})=>{
      const p=new URL(url,'http://localhost');let body={},status=200;
      if(p.pathname==='/auth/me')body={student_id:'t',name:'Pat Teacher',role:'teacher'};
      else if(p.pathname==='/teacher/dashboard')body=data;
      else if(p.pathname==='/teacher/review'){body=review;status=reviewStatus;}
      else if(p.pathname==='/teacher/issues/seen'){
        const sent=JSON.parse(options.body);posts.push(sent);
        // The server's answer, reflected in the next dashboard read.
        data=structuredClone(data);
        for(const pr of data.problems)for(const s of pr.follow_up)
          if(pr.slug===sent.slug&&s.student_id===sent.student_id)s.seen=sent.seen;
        body={...sent};
      }
      return new w.Response(JSON.stringify(body),{status,headers:{'Content-Type':'application/json'}});
    };
    w.eval(read('ui.js')+'\n'+read('cache.js')+'\n'+read('teacher-review.js'));
    await wait();
    return {w,doc:w.document,posts};
  };
  try{
    // ── the list: one column, no folds ──
    const list=await make('?slug=invert&assignment=lab1');
    const p=fixture().problems.find(x=>x.slug==='invert');
    const rows=[...list.doc.querySelectorAll('.review-student')];
    assert.equal(rows.length,p.follow_up.length,'every student with an open issue is listed');
    assert.equal(list.doc.querySelectorAll('#reviewHost details').length,0,'no dropdowns to open one at a time');
    assert.match(rows[0].textContent,/Step 2: Return the value on top/,'the step, named by what it asked');
    const open=new URL(rows[0].querySelector('.review-student-link').href);
    assert.equal(open.searchParams.get('student'),p.follow_up[0].student_id);
    assert.equal(open.searchParams.get('slug'),'invert');
    assert.equal(list.doc.querySelector('.back-link').getAttribute('href'),'teacher.html');
    // "Issue seen" posts the mark, and the list redraws with the student seen.
    rows[0].querySelector('[data-seen]').click();await wait();
    assert.deepEqual(list.posts.at(-1),{slug:'invert',student_id:p.follow_up[0].student_id,seen:true});
    const after=[...list.doc.querySelectorAll('.review-student')];
    const moved=after.find(r=>r.textContent.includes(p.follow_up[0].name));
    assert(moved.classList.contains('is-seen'),'marked seen');
    assert.equal(after.at(-1),moved,'seen students move below the open ones');
    assert.match(moved.querySelector('[data-seen]').textContent,/undo/);
    moved.querySelector('[data-seen]').click();await wait();
    assert.equal(list.posts.at(-1).seen,false,'and it can be undone');

    // ── one student, immersive ──
    const one=await make('?slug=invert&assignment=lab1&student=student-0');
    const d=one.doc;
    assert.equal(d.querySelector('h1').textContent,'Avery Chen');
    assert.match(d.querySelector('.review-head .sub').textContent,/Stuck on step 2 of 3/);
    assert.match(d.querySelector('.review-head .sub').textContent,/2 attempts on this step, 2 wrong/);
    assert.match(d.querySelector('.review-prompt').textContent,/Swap each pair into the result/);
    assert.match(d.querySelector('.is-warn').textContent,/answer is wrong/);
    assert.match(d.body.textContent,/1 of the 4 cases it failed/);
    assert.match(d.querySelector('.review-card pre.code').textContent,/expected: \{2: 1\}/);
    // Their whole function up to the step: header, the accepted step, then the
    // failing attempt - and only the failing attempt in red.
    const cl=[...d.querySelectorAll('.cl-row')];
    assert.equal(cl.length,1+2+1);
    assert(cl[0].classList.contains('cl-head'));
    assert.equal(cl[1].querySelector('code').textContent,'    out = {}');
    assert.deepEqual(cl.map(r=>r.classList.contains('cl-fail')),[false,false,false,true]);
    assert.equal(cl[3].querySelector('code').textContent,'        out[k] = k');
    assert.match(cl[3].textContent,/failing attempt/,'marked in text, not only in colour');
    assert.match(d.querySelector('#listingNote').textContent,/Line 4 is the attempt at step 2 that failed/);
    assert.equal(d.querySelectorAll('.attempt').length,2);
    assert.match(d.querySelector('.step-switch').textContent,/Also stuck on: Step 3/);
    assert.equal(new URL(d.querySelector('.step-switch a').href).searchParams.get('step'),'3');
    assert.match(d.querySelector('a[download]').getAttribute('href'),/\/teacher\/assignments\/lab1\/transcript\/student-0$/);
    assert.equal(new URL(d.querySelector('.back-link').href).searchParams.get('student'),null,'back goes to the list');
    d.querySelector('[data-seen]').click();await wait();
    assert.deepEqual(one.posts.at(-1),{slug:'invert',student_id:'student-0',seen:true});

    // Student text is the most attacker-shaped thing on this page.
    const bad=structuredClone(REVIEW);bad.name='<img src=x onerror=alert(1)>';
    bad.step.code='<script>alert(2)</script>';bad.step.prompt='<img src=x onerror=alert(3)>';
    bad.step.failing_cases=['<img src=x onerror=alert(4)>'];bad.step.attempts[0].code='<script>alert(5)</script>';
    const x=await make('?slug=invert&student=student-0',{review:bad});
    assert.equal(x.doc.querySelectorAll('#reviewHost img, #reviewHost script').length,0);
    assert.match(x.doc.querySelector('.cl-fail code').textContent,/<script>/,'shown as text, not run');

    const missing=await make('?slug=invert&student=nobody',{reviewStatus:404,review:{detail:{}}});
    assert.match(missing.doc.querySelector('#reviewHost').textContent,/No saved work/);
    const none=await make('');
    assert.match(none.doc.querySelector('#reviewHost').textContent,/Choose a problem/);
    console.log('PASS: teacher review page: vertical student list, Issue seen and undo, immersive view with the function up to the failing step, failing lines marked in red and in text, failing cases, attempts, other open steps, escaping, missing work.');
  }finally{for(const w of windows)await w.happyDOM.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
