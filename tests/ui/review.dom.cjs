/* review.dom.cjs - the review page, and the way to it from Assignments.

   THE REPORT (8 Oct, HW4). Three problems sat at "your review"; the button led
   to the Playground, which called them "failed" and "oracle too weak", and
   nothing anywhere said which checks were missed or what to do.

   AND (13 Oct, HW4): update was READY with a wrong answer still passing, and
   nothing said so. The page now also opens for a ready problem with gaps, and
   shows each test case to add with the wrong answers it catches.

   AND (14 Oct): roadmaps are graded (main/roadmap_score). A problem none of
   whose roadmaps reaches the bar waits here, or - live - is pointed out here;
   a missed change no test can catch can be marked harmless.

   AND (15 Oct, HW4 put): the model's suggested cases aimed at the wrong limit,
   none caught `>` -> `>=`, and the page called a real gap "may be harmless"
   with a Mark-as-harmless button beside it. It now says the suggestions
   missed, and the teacher can type their own case, checked for free.

   review.fixture.json is the REAL /teacher/review view of a small synthetic
   problem (Tank.fill), produced by main/review_help through the route - see
   test_review_page.py - so this checks the page against what the server
   actually sends. No model, no server, no teacher solution from a real
   assignment. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const read = name => fs.readFileSync(path.resolve(__dirname, '../../frontend', name), 'utf8');
const fixture = JSON.parse(fs.readFileSync(path.join(__dirname, 'review.fixture.json'), 'utf8'));
const wait = (ms = 40) => new Promise(r => setTimeout(r, ms));
const SLUG = fixture.after.slug;

// FAILED UNTIL PROVEN OTHERWISE (13 Oct): a broken page once left a window
// that never finished closing, and the suite exited 0 with no PASS and no
// error - which run-all counts as passing.
process.exitCode = 1;

(async () => {
  const {Window} = await import(pathToFileURL(require.resolve('happy-dom')).href);
  const windows = [];
  const make = async (file, query, answer) => {
    const w = new Window({url: `http://localhost/${file}${query}`, settings: {
      enableJavaScriptEvaluation: true, disableJavaScriptFileLoading: true,
      disableCSSFileLoading: true, suppressInsecureJavaScriptEnvironmentWarning: true}});
    windows.push(w);
    w.document.write(read(file).replace(/<script\b[^>]*>[\s\S]*?<\/script>/g, ''));
    const calls = [];
    w.fetch = async (url, options = {}) => {
      const p = new URL(url, 'http://localhost');
      const method = (options.method || 'GET').toUpperCase();
      calls.push({path: p.pathname, method, body: options.body ? JSON.parse(options.body) : null});
      const {__status = 200, ...data} = p.pathname === '/auth/me'
        ? {student_id: 'teacher-fixture', name: 'Pat Teacher', role: 'teacher'}
        : answer(p.pathname, method, options.body ? JSON.parse(options.body) : null) || {};
      return new w.Response(JSON.stringify(data), {status: __status, headers: {'Content-Type': 'application/json'}});
    };
    w.confirm = () => true;
    const script = file === 'review.html' ? 'review.js' : 'teacher.js';
    w.eval(read('ui.js') + '\n' + read('cache.js') + '\n' + read(script));
    await wait();
    return {w, doc: w.document, calls};
  };

  try {
    // ── the page, with a proven gap ───────────────────────────────────────
    const view = `/teacher/review/a-1/${SLUG}`;
    const page = await make('review.html', `?a=a-1&slug=${SLUG}`, (p, m) => {
      if (p === view) return fixture.before;
      if (p === view + '/explain' && m === 'POST') return fixture.after;
      if (p === '/teacher/problems/retry') return {ready: true, stage: null, chunks: 2, n_tests: 7};
    });
    await wait(80);
    const {doc, calls} = page;
    assert.equal(calls.filter(c => c.path === view + '/explain').length, 1,
      'the plain words are written once, on first open');
    const cards = [...doc.querySelectorAll('.finding')];
    assert.equal(cards.length, 3, 'one card per change the tests missed');
    assert.ok(cards.some(c => /skip the room check/.test(c.textContent)), 'plain words shown');
    assert.equal(doc.querySelectorAll('.finding .pill.warn').length, 2, 'two proven gaps');
    assert.match(doc.querySelector('.finding.maybe .pill').textContent, /suggested test cases missed it/,
      'an unproven one says exactly that - and never that it may be harmless');
    assert.ok(cards.every(c => c.querySelector('.where .src').textContent.trim()), 'each names its line');
    assert.ok(!/`/.test(doc.querySelector('.finding dd').innerHTML), 'code is code, not backticks');
    // One card per test case: the TEACHER's answers, and what each catches.
    const tcs = [...doc.querySelectorAll('.tcase')];
    assert.equal(tcs.length, fixture.after.help.tests.length);
    assert.match(tcs[0].querySelector('pre').textContent, /t\.fill\(7\)/);
    assert.match(tcs[0].querySelector('pre').textContent, /'FULL'/);
    assert.match(tcs[0].querySelector('.tcatch').textContent, /Catches: .*skip the room check/);
    assert.match(tcs[1].querySelector('.tcatch').textContent, /forget to raise the level/);
    assert.doesNotMatch(tcs[1].querySelector('.tcatch').textContent, /room check/);
    // The solution: every line, the missed ones marked, none in the site's
    // own wrapping `.row` (which broke the docstring line).
    const rows = doc.querySelectorAll('#sol .srow');
    assert.equal(rows.length, fixture.after.source.split('\n').length);
    assert.deepEqual([...doc.querySelectorAll('#sol .srow.hit .n')].map(n => +n.textContent),
      fixture.after.lines);
    assert.equal(doc.querySelectorAll('#sol .row').length, 0);
    assert.equal(doc.querySelector('#queue a.on').textContent.trim(), fixture.after.title);
    assert.match(doc.querySelector('#queue a').getAttribute('href'), /^review\.html\?a=a-1&slug=/);

    // The main button sends the checked edit to the one retry route.
    assert.equal(doc.querySelector('#addGo').textContent.trim(), 'Add all 2 and prepare again');
    assert.ok(doc.querySelector('#acceptGo'), 'not yet published: publishing anyway is a choice');
    doc.querySelector('#addGo').click();
    await wait(80);
    const retry = calls.find(c => c.path === '/teacher/problems/retry');
    assert.ok(retry, 'nothing was sent');
    assert.deepEqual(retry.body, {assignment_id: 'a-1', slug: SLUG, keep_live: false,
                                  source: fixture.after.help.edited_source});
    assert.match(doc.querySelector('#actMsg').textContent, /ready and\s+visible to students/);

    // ── the page, when nothing could be proven ────────────────────────────
    const harmless = JSON.parse(JSON.stringify(fixture.after));
    harmless.checks.forEach(c => { c.verdict = 'maybe_harmless'; });
    harmless.help.edited_source = null;
    harmless.help.catches = 0;
    const h = await make('review.html', `?a=a-1&slug=${SLUG}`, p =>
      p === view ? harmless : p === '/teacher/problems/accept' ? {ready: true} : null);
    await wait(60);
    assert.ok(!h.doc.querySelector('#addGo'), 'no fix offered that proves nothing');
    // The model's cases missing a change is no evidence it is harmless (15 Oct).
    assert.doesNotMatch(h.doc.querySelector('#main').textContent, /may be harmless|reasonable here/);
    assert.match(h.doc.querySelector('.lbl').textContent, /doesn't prove they're harmless/);
    assert.ok(h.doc.querySelector('.option.rec #tryGo'), 'writing your own case is what is recommended');
    h.doc.querySelector('#acceptGo').click();
    await wait(80);
    assert.deepEqual(h.calls.find(c => c.path === '/teacher/problems/accept').body,
      {assignment_id: 'a-1', slug: SLUG});

    // ── ready, with gaps: add the cases, never "publish anyway" ───────────
    const fixed = JSON.parse(JSON.stringify(fixture.gaps));
    fixed.gaps = false; fixed.checks = []; fixed.caught = fixed.total; fixed.help = null;
    let retried = false;
    const gp = await make('review.html', `?a=a-1&slug=${SLUG}`, p => {
      if (p === view) return retried ? fixed : fixture.gaps;
      if (p === '/teacher/problems/retry') { retried = true; return {ready: true}; }
    });
    await wait(60);
    assert.equal(gp.calls.filter(c => c.path === view + '/explain').length, 0,
      'the help is already written for this version: nothing to pay');
    assert.match(gp.doc.querySelector('#eyebrow').textContent, /Ready, with gaps/);
    assert.match(gp.doc.querySelector('#sub').textContent, /would still be marked correct/);
    assert.ok(!gp.doc.querySelector('#acceptGo'), 'already published: nothing to accept');
    assert.ok([...gp.doc.querySelectorAll('.option h4')].some(h => /Leave it as it is/.test(h.textContent)));
    assert.equal(gp.doc.querySelectorAll('.tcase').length, 2);
    gp.doc.querySelector('#addGo').click();
    await wait(100);
    assert.equal(gp.calls.find(c => c.path === '/teacher/problems/retry').body.keep_live, true,
      'a live problem is never hidden by a failed re-prepare');
    assert.match(gp.doc.querySelector('#main').textContent, /now\s+catch every wrong answer/);

    // ...and when the new version does not come out ready, the page says the
    // students' version is untouched - and keeps saying it after the redraw.
    const kept = await make('review.html', `?a=a-1&slug=${SLUG}`, p => {
      if (p === view) return fixture.gaps;
      if (p === '/teacher/problems/retry') return {ready: false,
        error: 'Nothing changed for students - they still see the version they had. The new version did not come out ready: x'};
    });
    await wait(60);
    kept.doc.querySelector('#addGo').click();
    await wait(100);
    assert.match(kept.doc.querySelector('#actMsg').textContent, /Nothing changed for students/);
    assert.ok(kept.doc.querySelector('#addGo') && !kept.doc.querySelector('#addGo').disabled,
      'the buttons come back for another try');

    // ── your own test case (15 Oct): what HW4 put needed ─────────────────
    const putLike = JSON.parse(JSON.stringify(fixture.gaps));
    putLike.checks.forEach(c => { c.verdict = 'maybe_harmless'; });
    Object.assign(putLike.help, {tests: [], edited_source: null, catches: 0});
    const answers = [fixture.try_none,
                     {__status: 400, detail: {message: 'Line 1 is not valid Python (never closed): t = Tank(10'}},
                     fixture.try_hit];
    const tp = await make('review.html', `?a=a-1&slug=${SLUG}`, (p, m) => {
      if (p === view) return putLike;
      if (p === view + '/try' && m === 'POST') return answers.shift();
      if (p === '/teacher/problems/retry') return {ready: true};
    });
    await wait(60);
    const td = tp.doc;
    assert.ok(td.querySelector('.option.rec #tryGo'), 'recommended when no suggestion caught anything');
    assert.match(td.querySelector('#tryGo').parentElement.textContent, /Free/);
    const tryIt = async text => {
      td.querySelector('#trySrc').value = text;
      td.querySelector('#tryGo').click();
      await wait(80);
    };
    await tryIt('t = Tank(10)\nt.fill(1)');
    assert.deepEqual(tp.calls.filter(c => c.path === view + '/try').at(-1).body, {text: 't = Tank(10)\nt.fill(1)'});
    assert.match(td.querySelector('#tryOut').textContent, /didn't catch any of the wrong answers/);
    // 18 Oct: still offered - a student's mistake need not be one of our slips.
    assert.match(td.querySelector('#tryOut').textContent, /You can still add it/);
    assert.ok(td.querySelector('#tryAdd'), 'a case that catches no slip can still be added');
    assert.equal(td.querySelector('#trySrc').value, 't = Tank(10)\nt.fill(1)', 'what was typed survives the redraw');
    await tryIt('t = Tank(10');
    assert.match(td.querySelector('#tryOut .banner.bad').textContent, /not valid Python/);
    await tryIt('>>> t = Tank(10)\n>>> t.fill(10)');
    assert.match(td.querySelector('#tryOut .tcatch').textContent, /Catches: .*>= instead of >/);
    assert.match(td.querySelector('#tryOut pre').textContent, /t\.fill\(10\)\s*'OK'/, "your solution's answer");
    assert.match(td.querySelector('#tryOut').textContent, /Still not caught: .*room check/);
    td.querySelector('#tryAdd').click();
    await wait(100);
    assert.deepEqual(tp.calls.find(c => c.path === '/teacher/problems/retry').body,
      {assignment_id: 'a-1', slug: SLUG, keep_live: true, source: fixture.try_hit.edited_source},
      'the checked edit, and a live problem is never hidden');

    // ── EDIT: a ready problem with nothing waiting (18 Oct, HW4 clear) ────
    // It had no way to be changed: the editor and the box lived only on a
    // flagged problem, and `clear` accepted a wrong loop the tests lacked.
    const cl = await make('review.html', `?a=a-1&slug=${SLUG}`, (p, m) => {
      if (p === view) return fixture.clean;
      if (p === view + '/try' && m === 'POST') return fixture.clean_try;
      if (p === '/teacher/problems/retry') return {ready: true};
    });
    await wait(60);
    assert.match(cl.doc.querySelector('#main .banner.ok').textContent, /Nothing is waiting on you/);
    assert.ok(cl.doc.querySelector('#editCard #trySrc'), 'the box is there');
    assert.ok(cl.doc.querySelector('#editCard #editGo'), 'and the editor');
    assert.match(cl.doc.querySelector('#editCard').textContent, /it becomes a test/);
    assert.equal(cl.calls.filter(c => c.path === view + '/explain').length, 0, 'nothing to explain, nothing paid');
    cl.doc.querySelector('#trySrc').value = 't = Tank(3)\nt.fill(3)\nt.fill(1)';
    cl.doc.querySelector('#tryGo').click(); await wait(80);
    assert.match(cl.doc.querySelector('#tryOut').textContent, /You can still add it/);
    cl.doc.querySelector('#tryAdd').click(); await wait(100);
    assert.deepEqual(cl.calls.find(c => c.path === '/teacher/problems/retry').body,
      {assignment_id: 'a-1', slug: SLUG, keep_live: true, source: fixture.clean_try.edited_source},
      'the case, added to the live problem without hiding it');

    // ── marked harmless: one button per missed change, on a live problem ──
    const marked = JSON.parse(JSON.stringify(fixture.gaps));
    const gone = marked.checks.shift();
    marked.harmless = [{label: gone.label, line: gone.line, phrasing: gone.phrasing, by: 'prof@psu.edu'}];
    const hm = await make('review.html', `?a=a-1&slug=${SLUG}`, (p, m) =>
      p === view ? fixture.gaps : p === view + '/dismiss' && m === 'POST' ? marked : null);
    await wait(60);
    const marks = [...hm.doc.querySelectorAll('button.harmlessGo')];
    assert.equal(marks.length, fixture.gaps.checks.length, 'one per missed change');
    marks[0].click();
    await wait(80);
    assert.deepEqual(hm.calls.find(c => c.path === view + '/dismiss').body, {label: gone.label});
    assert.equal(hm.doc.querySelectorAll('.finding').length, fixture.gaps.checks.length - 1);
    assert.match(hm.doc.querySelector('.harmless').textContent, /marked by prof@psu\.edu/);
    assert.ok(!(await make('review.html', `?a=a-1&slug=${SLUG}`, p => p === view ? fixture.after : null))
      .doc.querySelector('button.harmlessGo'), 'not on a problem still waiting on its tests');

    // ── roadmaps under the bar, waiting on review ─────────────────────────
    const RSLUG = fixture.roadmap_review.slug, rview = `/teacher/review/a-1/${RSLUG}`;
    const rr = await make('review.html', `?a=a-1&slug=${RSLUG}`, (p, m) => {
      if (p === rview) return fixture.roadmap_review;
      if (p === '/teacher/problems/accept_roadmaps') return {ready: true, accepted_by: 'prof@psu.edu'};
      if (p === '/teacher/problems/rebuild_roadmaps') return {ready: false, error: 'Waiting on your review: still'};
    });
    await wait(60);
    assert.match(rr.doc.querySelector('#eyebrow').textContent, /Needs your review: roadmaps/);
    assert.equal(rr.doc.querySelectorAll('.finding').length, 0, 'no test findings to show');
    const maps = [...rr.doc.querySelectorAll('.rmap')];
    assert.equal(maps.length, fixture.roadmap_review.roadmaps.list.length);
    assert.match(maps[0].querySelector('.rmpts').textContent, new RegExp(`^${fixture.roadmap_review.roadmaps.best}`));
    assert.match(maps[0].querySelector('.rmparts').textContent, /biggest step \d+\/30/);
    assert.match(rr.doc.querySelector('#rmCard .banner').textContent, /None of this problem's roadmaps\s+reaches 60/);
    assert.match(rr.doc.querySelector('#rmCard .option.rec').textContent, /helper methods/,
      'a long step: splitting the method is what helps');
    assert.equal(rr.doc.querySelector('#rmAccept').textContent.trim(), 'Publish with the best roadmap');
    rr.doc.querySelector('#rmRebuild').click();
    await wait(80);
    assert.deepEqual(rr.calls.find(c => c.path === '/teacher/problems/rebuild_roadmaps').body,
      {assignment_id: 'a-1', slug: RSLUG});
    assert.match(rr.doc.querySelector('#actMsg').textContent, /still/);
    rr.doc.querySelector('#rmAccept').click();
    await wait(80);
    assert.deepEqual(rr.calls.find(c => c.path === '/teacher/problems/accept_roadmaps').body,
      {assignment_id: 'a-1', slug: RSLUG});
    assert.match(rr.doc.querySelector('#actMsg').textContent, /ready and\s+visible to students/);

    // ── roadmaps under the bar on a live problem: kept, never hidden ──────
    let kept2 = false;
    const settled = {...fixture.roadmap_live, roadmaps: {...fixture.roadmap_live.roadmaps, below: false}};
    const rl = await make('review.html', `?a=a-1&slug=${RSLUG}`, p => {
      if (p === rview) return kept2 ? settled : fixture.roadmap_live;
      if (p === '/teacher/problems/accept_roadmaps') { kept2 = true; return {ready: true}; }
    });
    await wait(60);
    assert.match(rl.doc.querySelector('#eyebrow').textContent, /Ready, with roadmaps under the bar/);
    assert.match(rl.doc.querySelector('#rmCard .banner').textContent, /Students still get the best ones/);
    assert.equal(rl.doc.querySelector('#rmAccept').textContent.trim(), 'Keep these roadmaps');
    rl.doc.querySelector('#rmAccept').click();
    await wait(100);
    assert.match(rl.doc.querySelector('#main').textContent, /Kept: these roadmaps stay/);

    // ── the way in: Assignments sends a review to this page ───────────────
    const asg = await make('teacher-assignments.html', '', p => {
      if (p === '/assignments') return {assignments: [{id: 'a-1', name: 'Week 4 - Tanks',
        ready: 0, total: 2, published: true, created_at: '2026-10-09T12:00:00Z'}]};
      if (p === '/teacher/assignments/a-1/problems') return {failed: 1, problems: [
        {slug: SLUG, title: 'Tank Fill', ready: false,
         prepare_error: 'Waiting on your review: a few checks on the generated tests came back inconclusive.'},
        {slug: 'tank-drain', title: 'Tank Drain', ready: false,
         prepare_error: 'Could not split this problem into steps that hold together.'},
        {slug: 'tank-empty', title: 'Tank Empty', ready: true, gaps: 1},
        {slug: 'tank-size', title: 'Tank Size', ready: true, gaps: 0},
        {slug: 'tank-long', title: 'Tank Long', ready: true, gaps: 0, roadmaps_below: 46}]};
    });
    await wait(60);
    asg.doc.querySelector('[data-act="breakdown"]').click();
    await wait(80);
    const items = [...asg.doc.querySelectorAll('.bdList li')];
    const review = items.find(li => /Tank Fill/.test(li.textContent));
    const blocked = items.find(li => /Tank Drain/.test(li.textContent));
    assert.equal(review.querySelector('.pill').textContent.trim(), 'your review');
    assert.ok(review.querySelector('.pill').classList.contains('warn'), 'not red: nothing failed');
    const btn = review.querySelector('button.bdReview');
    assert.equal(btn.textContent.trim(), 'Review');
    assert.equal(btn.dataset.href, `review.html?a=a-1&slug=${SLUG}`);
    assert.equal(blocked.querySelector('.pill').textContent.trim(), 'blocked');
    assert.equal(blocked.querySelector('button.bdFix').textContent.trim(), 'Open and fix',
      'a problem that really failed keeps the in-place fix panel');
    const gapped = items.find(li => /Tank Empty/.test(li.textContent));
    assert.equal(gapped.querySelector('.pill').textContent.trim(), 'ready');
    assert.match(gapped.textContent, /1 deliberate mistake still passes/);
    assert.equal(gapped.querySelector('button.bdReview').dataset.href, 'review.html?a=a-1&slug=tank-empty');
    const long = items.find(li => /Tank Long/.test(li.textContent));
    assert.match(long.textContent, /no roadmap reaches the bar \(best 46\/100\)/);
    assert.equal(long.querySelector('button.bdReview').dataset.href, 'review.html?a=a-1&slug=tank-long');
    const clean = items.find(li => /Tank Size/.test(li.textContent));
    const edit = clean.querySelector('button.bdReview');
    assert.ok(edit && edit.textContent.trim() === 'Edit', 'a ready problem with nothing waiting can be edited');
    assert.equal(edit.dataset.href, 'review.html?a=a-1&slug=tank-size');

    process.exitCode = 0;
    console.log('PASS: review page - plain findings, proven gaps vs "may be harmless", one card per test case with what it catches, solution lines, one retry with the checked edit, publish anyway when nothing is proven, ready-with-gaps (no publish, never hidden), marking a change harmless, your own test case checked for free and added (even one that catches no slip), Edit for a problem with nothing waiting, roadmaps under the bar (waiting, and live), and the way in from Assignments.');
  } catch (e) {
    // The stack only, said BEFORE closing (which can hang): an assertion's
    // `actual` can be a page element, and printing one walks the whole page
    // until the process is killed - silently.
    console.error(String(e && e.stack || e));
    process.exitCode = 1;
  } finally { for (const w of windows) await w.happyDOM.close(); }
})().catch(e => { console.error(e); process.exitCode = 1; });
