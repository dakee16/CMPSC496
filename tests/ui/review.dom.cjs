/* review.dom.cjs - the review page, and the way to it from Assignments.

   THE REPORT (8 Oct, HW4). Three problems sat at "your review"; the button led
   to the Playground, which called them "failed" and "oracle too weak", and
   nothing anywhere said which checks were missed or what to do.

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
      const data = p.pathname === '/auth/me'
        ? {student_id: 'teacher-fixture', name: 'Pat Teacher', role: 'teacher'}
        : answer(p.pathname, method) || {};
      return new w.Response(JSON.stringify(data), {status: 200, headers: {'Content-Type': 'application/json'}});
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
    assert.match(doc.querySelector('.finding.maybe').textContent, /may be harmless/,
      'an unproven one says exactly that');
    assert.ok(cards.every(c => c.querySelector('.where .src').textContent.trim()), 'each names its line');
    assert.ok(!/`/.test(doc.querySelector('.finding dd').innerHTML), 'code is code, not backticks');
    // The examples carry the TEACHER's answers.
    const ex = doc.querySelector('#ex').textContent;
    assert.match(ex, /t\.fill\(7\)/);
    assert.match(ex, /'FULL'/);
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
    doc.querySelector('#addGo').click();
    await wait(80);
    const retry = calls.find(c => c.path === '/teacher/problems/retry');
    assert.ok(retry, 'nothing was sent');
    assert.deepEqual(retry.body, {assignment_id: 'a-1', slug: SLUG,
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
    assert.equal(h.doc.querySelector('#addGo'), null, 'no fix offered that proves nothing');
    assert.match(h.doc.querySelector('.option.rec').textContent, /publish anyway/i);
    assert.match(h.doc.querySelector('.option.rec').textContent, /reasonable here/);
    h.doc.querySelector('#acceptGo').click();
    await wait(80);
    assert.deepEqual(h.calls.find(c => c.path === '/teacher/problems/accept').body,
      {assignment_id: 'a-1', slug: SLUG});

    // ── the way in: Assignments sends a review to this page ───────────────
    const asg = await make('teacher-assignments.html', '', p => {
      if (p === '/assignments') return {assignments: [{id: 'a-1', name: 'Week 4 - Tanks',
        ready: 0, total: 2, published: true, created_at: '2026-10-09T12:00:00Z'}]};
      if (p === '/teacher/assignments/a-1/problems') return {failed: 1, problems: [
        {slug: SLUG, title: 'Tank Fill', ready: false,
         prepare_error: 'Waiting on your review: a few checks on the generated tests came back inconclusive.'},
        {slug: 'tank-drain', title: 'Tank Drain', ready: false,
         prepare_error: 'Could not split this problem into steps that hold together.'}]};
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

    console.log('PASS: review page - plain findings, proven gaps vs "may be harmless", the teacher\'s answers in the examples, solution lines, one retry with the checked edit, publish anyway when nothing is proven, and the way in from Assignments.');
  } finally { for (const w of windows) await w.happyDOM.close(); }
})().catch(e => { console.error(e); process.exitCode = 1; });
