/* Browser regression checks. All application API responses are fixtures: this
   suite never starts the backend, grades real work, or invokes a model. */
const assert = require('node:assert/strict');
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');
const frontend = path.resolve(__dirname, '../../frontend');
const output = process.env.UI_ARTIFACTS;
const description = `Add a new year of employee records, carrying everyone forward with a raise.

d maps a year to a dictionary of employees, and each employee maps to a list of [title, employment status, salary]. Build the records for \`year\` by taking every employee from the year before it, keeping their title and status unchanged, and adding \`bonus\` to their salary. Store that under \`year\` in d and return d.

>>> records = {2020: {"John": ["Managing Director", "Full-time", 65000], "Sally": ["HR Director", "Full-time", 60000]}, 2021: {"John": ["Managing Director", "Full-time", 70000], "Sally": ["HR Director", "Full-time", 65000]}}
>>> employee_update(records, 7500, 2022)
{2022: {"John": ["Managing Director", "Full-time", 77500], "Sally": ["HR Director", "Full-time", 72500]}}

Remember: keep the earlier years unchanged.`;
const prompts = [
  'Extract the employee records from the previous year and prepare a structure to hold their updated records for the current year, without updating anything yet.',
  'For each employee, calculate their updated salary by adding the bonus while retaining their title and employment status.',
  "Add the new year's employee records to the main data and return it."
];
const chunks = prompts.map((prompt, i) => ({prompt, indent: i === 1 ? 0 : 0}));
const problems = [
  {slug:'employee-update',title:'Employee Update',description,ready:true},
  {slug:'word-count',title:'Word Count',description:'Count each word.',ready:true},
  {slug:'inventory',title:'Inventory',description:'Update the inventory.',ready:true}
];
const assignments = [{id:'lab1', name:'LAB1 – Dictionaries',ready:3,total:3,created_at:'2026-09-08T12:00:00Z'}];
const seen = [];
const unknown = [];
let approved = true;
let gradeIndex = 0;
let role = 'student';
let loggedIn = true;
const contentTypes = {'.html':'text/html','.css':'text/css','.js':'text/javascript','.svg':'image/svg+xml'};
const server = http.createServer((req,res) => {
  const pathname = new URL(req.url, 'http://localhost').pathname;
  const target = path.resolve(frontend, '.' + (pathname === '/' ? '/index.html' : pathname));
  if (!target.startsWith(frontend + path.sep)) {res.writeHead(403).end();return;}
  fs.readFile(target,(err,data)=>{if(err){res.writeHead(404).end();return;}res.setHeader('Content-Type',contentTypes[path.extname(target)]||'text/plain');res.end(data);});
});
(async () => {
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const base=`http://127.0.0.1:${server.address().port}`;
  const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
  const errors=[];
  try {
    const context=await browser.newContext({viewport:{width:1600,height:1000},reducedMotion:'reduce'});
    await context.addInitScript(()=>localStorage.setItem('mt.coach.tutor.v1','1'));
    await context.route(base+'/**', async route=>{
      const pathname=new URL(route.request().url()).pathname;
      if (/\.(html|css|js|svg)$/.test(pathname) || pathname==='/') return route.continue();
      seen.push(pathname);
      let json={}; let status=200;
      if(pathname==='/auth/me'){json={name:'Alex Morgan',student_id:'ui-fixture',role};if(!loggedIn)status=401;}
      else if(pathname==='/assignments') json={assignments};
      else if(pathname==='/solved') json={slugs:['word-count'],opened:['employee-update'],last_slug:'employee-update'};
      else if(pathname==='/assignments/lab1/problems') json={problems};
      else if(pathname==='/decompose_chunks'){gradeIndex=0;json={session_id:'ui-only',header:'def employee_update(d, bonus, year):',chunks:chunks.map(c=>({...c,prompt:approved?c.prompt:''}))};}
      else if(pathname.startsWith('/history/')) json=approved?{found:true,design_approved:true,messages:[{role:'user',content:'previous_year = year - 1\nprevious_year_records = d[previous_year]\nnew_dict = {}',at:'2026-09-08T12:00:00Z'},{role:'assistant',content:"What information needs to stay the same when you carry each employee's record forward?",at:'2026-09-08T12:01:00Z'}]}:{found:false};
      else if(pathname.startsWith('/session_steps/')) json={chunks};
      else if(pathname==='/grade_chunk'){gradeIndex++;json={verdict:'correct',index:gradeIndex,completed:gradeIndex===3,solved_independently:true,reason:'This step is correct.'};}
      else if(pathname==='/mark_solved') json={ok:true};
      else if(pathname==='/tutor_chat') json={reply:'Keep the original record intact.\n\n```python\n# Your next step\n```',ready:false};
      else if(pathname.includes('plan_graph')) json={nodes:[],edges:[]};
      else if(pathname==='/assignment_template') json={template:'# Assignment'};
      else if(pathname==='/teacher/assignments/lab1/grades') json={students:[],problems:3,total_steps:9};
      else if(pathname==='/playground/problems') json={problems:[]};
      else if(pathname==='/playground/params') json={};
      else {unknown.push(pathname);status=404;}
      await route.fulfill({status,contentType:'application/json',body:JSON.stringify(json)});
    });
    const page=await context.newPage();
    page.on('pageerror',e=>errors.push(e.message));
    const screenshot=async name=>{if(output){fs.mkdirSync(output,{recursive:true});await page.screenshot({path:path.join(output,name+'.png'),fullPage:true});}};
    const noOverflow=async(label)=>assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),label+' has horizontal page overflow');
    await page.goto(base+'/student.html');
    await page.locator('#assignments .rowitem').first().waitFor();
    assert.equal(await page.locator('html').getAttribute('data-theme'),'light');
    assert.match(await page.title(),/ACADIA/);
    await screenshot('acadia-assignments-light');
    await page.locator('#assignments .rowitem').first().click();
    await page.locator('#handbackRow').waitFor();
    const padding=await page.locator('#handbackRow').evaluate(el=>{const s=getComputedStyle(el);return [s.paddingTop,s.paddingBottom];});
    assert.deepEqual(padding,['20px','20px']);
    await screenshot('acadia-problems-light');
    await page.locator('[data-slug="employee-update"]').click();
    await page.waitForFunction(()=>!document.querySelector('#workStep').hidden);
    await page.waitForFunction(()=>document.querySelector('#prompt').textContent.length>20);
    assert.equal(await page.locator('#statement pre').count(),1);
    assert((await page.locator('#statement pre').textContent()).includes('\n>>> employee_update'));
    assert.equal(await page.locator('#statement code').count(),4); // 3 inline identifiers + example
    assert.equal(await page.locator('.studio-brief #stepper').count(),0);
    assert.equal(await page.locator('.studio-work #stepper').count(),1);
    assert(await page.locator('.studio-brief').evaluate(el=>el.offsetWidth>=800));
    assert.equal(await page.locator('#stageRead').isVisible(),true);
    assert.equal(await page.locator('#stageCode').isVisible(),false);
    assert.equal(await page.locator('#chatcol').isVisible(),false);
    await screenshot('acadia-question-read-light');
    await page.getByRole('button',{name:'Switch to dark mode',exact:true}).click();
    await screenshot('acadia-question-read-dark');
    await page.getByRole('button',{name:'Switch to light mode',exact:true}).click();
    await page.locator('#readContinue').click();
    await page.locator('#stageCode').waitFor();
    assert.equal(await page.locator('#clog .bub.me pre').count(),1);
    await noOverflow('Desktop light');
    await page.evaluate(()=>editor.setValue('    previous_year = year - 1\n    previous_year_records = d[previous_year]\n    new_dict = {}'));
    await screenshot('acadia-workspace-light');
    await page.getByRole('button',{name:'Switch to dark mode',exact:true}).click();
    await screenshot('acadia-workspace-dark');
    assert.equal(await page.locator('html').getAttribute('data-theme'),'dark');
    const colors=await page.locator('.CodeMirror').evaluate(el=>({bg:getComputedStyle(el).backgroundColor,color:getComputedStyle(el).color}));
    assert.notEqual(colors.bg,colors.color);
    await page.getByRole('button',{name:'Focus editor',exact:true}).click();
    assert.equal(await page.locator('.journey').isVisible(),false);
    assert((await page.evaluate(()=>editor.getValue())).includes('previous_year'));
    await page.keyboard.press('Escape');
    assert.equal(await page.locator('.journey').isVisible(),true);
    // Submission still uses the existing endpoint; step navigation is display-only.
    await page.locator('#submit').click();
    await page.locator('#reviewBox').waitFor();
    assert.match(await page.locator('#backToNow').textContent(),/Continue to step 2/);
    await page.locator('#backToNow').click();
    await page.waitForFunction(()=>document.querySelector('#stepCount').textContent==='Step 2 of 3');
    await page.evaluate(()=>editor.setValue('    draft = 1'));
    await page.locator('#stepHistory>summary').click();
    await page.locator('#stepper button').first().click();
    assert.equal(await page.locator('#reviewBox').isVisible(),true);
    await page.locator('#backToNow').click();
    assert.equal(await page.evaluate(()=>editor.getValue()),'    draft = 1');
    assert.match(await page.locator('#stepCount').textContent(),/Step 2 of 3/);
    await page.getByRole('button',{name:'Switch to light mode',exact:true}).click();
    for(const width of [1440,1366,1024,950,768,390]){
      await page.setViewportSize({width,height:900});
      await noOverflow('Width '+width);
      assert(await page.locator('#cform').evaluate(el=>el.inert));
      await page.locator('.code-resources [data-resource="tutor"]').click();
      assert.equal(await page.locator('#cform').evaluate(el=>el.inert),false);
      await noOverflow('Tutor panel at '+width);
      await page.keyboard.press('Escape');
      assert.equal(await page.locator('#resourceDrawer').isVisible(),false);
      await page.locator('.code-resources [data-resource="problem"]').click();
      assert.equal(await page.locator('#problemPaper').isVisible(),true);
      await noOverflow('Problem reference at '+width);
      await page.keyboard.press('Escape');
      assert.equal(await page.evaluate(()=>editor.getValue()),'    draft = 1');
      if(width===1366||width===390) await screenshot('acadia-workspace-'+width);
    }
    // Unsafe HTML stays literal, and fences, nested blocks and hard-wrapped
    // prose retain their intended structure without dropping any source.
    const rendering=await page.evaluate(()=>{
      const el=document.createElement('div');
      renderLearningText(el,'A wrapped\nsentence with `n`.\n\n```python\ndef f(n):\n    if n:\n        return "<img src=x onerror=alert(1)>"\n```\n\n>>> f(1)\n1\n\nAfter the example.');
      return {html:el.innerHTML,code:[...el.querySelectorAll('pre')].map(x=>x.textContent),paragraph:el.querySelector('p').textContent,imgs:el.querySelectorAll('img').length};
    });
    assert.equal(rendering.imgs,0);
    assert.equal(rendering.paragraph,'A wrapped sentence with n.');
    assert.equal(rendering.code.length,2);
    assert(rendering.code[0].includes('\n        return'));
    assert(rendering.html.includes('After the example.'));
    // A fresh problem remains gated; moving progress cannot disclose prompts.
    approved=false;
    await page.setViewportSize({width:1600,height:1000});
    await page.goto(base+'/student.html');
    assert.equal(await page.locator('html').getAttribute('data-theme'),'light');
    await page.locator('#assignments .rowitem').first().click();
    await page.locator('[data-slug="employee-update"]').click();
    await page.locator('#readContinue:not([disabled])').waitFor();
    assert.equal(await page.locator('#stageRead').isVisible(),true);
    await page.locator('#readContinue').click();
    await page.locator('#planChatHome #chatcol').waitFor();
    await screenshot('acadia-plan-chat-light');
    await page.locator('[data-plan-method="upload"]').click();
    await page.locator('#designPanel:not([hidden])').waitFor();
    assert.equal(await page.locator('#workStep').isVisible(),false);
    assert.equal(await page.locator('#editorWrap').isVisible(),false);
    assert(await page.locator('#submit').isDisabled());
    await screenshot('acadia-design-gate');
    for(const width of [768,390]){
      await page.setViewportSize({width,height:900});
      await noOverflow('Upload plan at '+width);
      await page.locator('[data-plan-method="chat"]').click();
      await noOverflow('Chat plan at '+width);
      await screenshot('acadia-plan-'+width);
      await page.locator('#tabRead').click();
      await noOverflow('Read at '+width);
      await screenshot('acadia-read-'+width);
      await page.locator('#readContinue').click();
    }
    await page.setViewportSize({width:1600,height:1000});
    role='teacher';
    for(const name of ['teacher','grades','playground']){
      await page.goto(base+'/'+name+'.html');
      await page.waitForFunction(()=>!document.querySelector('.skelRow'));
      assert.match(await page.title(),/ACADIA/);
      assert(!(await page.locator('body').innerText()).includes('MicroTutor'));
      await noOverflow(name);
      await screenshot('acadia-'+name+'-light');
      await page.getByRole('button',{name:'Switch to dark mode',exact:true}).click();
      await noOverflow(name+' dark');
      await page.getByRole('button',{name:'Switch to light mode',exact:true}).click();
    }
    loggedIn=false;
    await page.goto(base+'/login.html');
    await screenshot('acadia-login-light');
    await page.getByRole('button',{name:'Switch to dark mode',exact:true}).click();
    await page.reload();
    assert.equal(await page.locator('html').getAttribute('data-theme'),'dark');
    await screenshot('acadia-login-dark');
    assert.deepEqual(errors,[],'Browser JavaScript errors');
    assert.deepEqual(unknown,[],'Unexpected API routes');
    console.log('PASS: branding, fresh/saved themes, staged reading/planning/coding, description/code formatting, design gate, paced step submission/review, focus, reference drawer, seven widths and all UI pages.');
    console.log('Application requests were mocked; no backend/model/grading pipeline ran.');
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;}).finally(()=>server.close());
