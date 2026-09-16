/* Real browser layout and interactions; all API data is synthetic. */
const assert=require('node:assert/strict');
const fs=require('node:fs');
const http=require('node:http');
const path=require('node:path');
const {chromium}=require('playwright');
const fixture=require('./teacher-dashboard.fixture.cjs');
const root=path.resolve(__dirname,'../../frontend');
const server=http.createServer((req,res)=>{
  const url=new URL(req.url,'http://localhost');
  const file=path.resolve(root,'.'+url.pathname);
  if(!file.startsWith(root+path.sep))return res.writeHead(403).end();
  fs.readFile(file,(err,data)=>{
    if(err)return res.writeHead(404).end();
    res.setHeader('Content-Type',({'.html':'text/html','.css':'text/css','.js':'text/javascript','.svg':'image/svg+xml'})[path.extname(file)]||'text/plain');
    res.end(data);
  });
});
(async()=>{
  await new Promise(r=>server.listen(0,'127.0.0.1',r));
  let browser;
  try{
    browser=await chromium.launch({headless:true});
    const context=await browser.newContext({viewport:{width:1440,height:1100},reducedMotion:'reduce'});
    const base='http://127.0.0.1:'+server.address().port;
    let fail=false,empty=false;
    await context.addInitScript(()=>localStorage.setItem('acadia.tutorial.v1:'+encodeURIComponent('browser-teacher:teacher'),'{"status":"completed"}'));
    await context.route(base+'/**',async route=>{
      const url=new URL(route.request().url()),p=url.pathname;
      if(/\.(html|css|js|svg)$/.test(p))return route.continue();
      let json={},status=200;
      if(p==='/auth/me')json={student_id:'browser-teacher',name:'Pat Teacher',first_name:'Pat',role:'teacher'};
      else if(p==='/assignments')json={assignments:fixture().assignments.map(a=>({...a,total:4,ready:4,created_at:'2026-09-15T12:00:00Z'}))};
      else if(p==='/assignment_template')json={content:'def example():\n    pass'};
      else if(p==='/teacher/dashboard'){
        status=fail?503:200;json=fixture(url.searchParams.get('assignment_id'));
        if(empty){json.problems=[];json.summary={...json.summary,active:0,needs_help:0,attempted_problems:0};}
      }else status=404;
      await route.fulfill({status,json});
    });
    const page=await context.newPage(),errors=[];
    page.on('pageerror',e=>errors.push(e.message));
    await page.goto(base+'/teacher.html');
    await page.locator('.insight-bar-row').first().waitFor();
    assert.equal(await page.locator('#list, #newAssignment, #uploadDrawer').count(),0);
    assert.equal(await page.locator('.insight-bar-row').count(),6);
    await page.locator('.insight-bar-row').nth(1).focus();await page.keyboard.press('Enter');
    assert.match(await page.locator('#insightDetailTitle').innerText(),/postfix/);
    assert.equal(await page.locator('.insight-bar-row').nth(1).evaluate(el=>el===document.activeElement),true);
    await page.locator('#reviewPriority').click();
    assert.equal(await page.locator('#insightDetailTitle').evaluate(el=>el===document.activeElement),true);
    for(const theme of ['light','dark']){
      await page.evaluate(t=>Theme.set(t),theme);
      for(const width of [1440,1024,768,390,320]){
        await page.setViewportSize({width,height:1100});
        assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),`${theme} at ${width}: overflow`);
        for(const box of await page.locator('.insight-bar-row').all()){
          assert(await box.evaluate(el=>el.scrollWidth<=el.clientWidth+1),`${theme} at ${width}: bar overflow`);
        }
        if(process.env.UI_ARTIFACTS&&[1440,390].includes(width)){
          fs.mkdirSync(process.env.UI_ARTIFACTS,{recursive:true});
          await page.evaluate(()=>{document.activeElement.blur();window.scrollTo(0,0);});
          await page.screenshot({path:path.join(process.env.UI_ARTIFACTS,`teacher-${theme}-${width}.png`),fullPage:true});
        }
      }
    }
    await page.locator('#insightAssignment').selectOption('hw3');
    await page.waitForFunction(()=>document.querySelector('#insightDetailTitle')?.textContent.includes('postfix'));
    fail=true;await page.locator('#refreshInsights').click();
    await page.waitForFunction(()=>document.querySelector('#insightNotice').textContent.includes('last successful update'));
    assert.equal(await page.locator('.insight-bar-row').count(),6);
    fail=false;empty=true;await page.locator('#retryInsights').click();
    await page.waitForFunction(()=>document.querySelector('#classInsights').textContent.includes('first attempts'));
    assert.equal(await page.locator('.insight-bar-row').count(),0);
    await page.setViewportSize({width:1440,height:1100});
    await page.locator('[data-nav="Assignments"]').click();
    await page.locator('#listBody table').waitFor();
    assert.equal(new URL(page.url()).pathname,'/teacher-assignments.html');
    assert.equal(await page.locator('[data-nav="Assignments"]').getAttribute('aria-current'),'page');
    assert.equal(await page.locator('#classInsights').count(),0);
    await page.locator('#newAssignment').click();
    assert.equal(await page.locator('#uploadDrawer').isVisible(),true);
    await page.locator('#file').setInputFiles({name:'sample.py',mimeType:'text/x-python',buffer:Buffer.from('def sample():\n    return 1\n')});
    assert.equal(await page.locator('#fname').innerText(),'sample.py');
    assert.equal(await page.locator('#go').isEnabled(),true);
    await page.locator('#closeUpload').click();
    assert.equal(await page.locator('#uploadDrawer').isVisible(),false);
    await page.setViewportSize({width:390,height:1100});
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'Assignments mobile overflow');
    await page.locator('#navToggle').click();
    empty=false;
    await page.locator('[data-nav="Home"]').click();
    await page.locator('.insight-bar-row').first().waitFor();
    assert.equal(await page.locator('[data-nav="Home"]').getAttribute('aria-current'),'page');
    assert.equal(await page.locator('#list, #uploadDrawer').count(),0);
    assert.deepEqual(errors,[]);
    console.log('PASS: teacher dashboard in Chromium, both themes at 320–1440px, filters, errors, separate Assignments tab, upload selection and mobile navigation.');
  }finally{if(browser)await browser.close();await new Promise(r=>server.close(r));}
})().catch(e=>{console.error(e);process.exitCode=1;});
