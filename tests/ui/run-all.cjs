/* Run every DOM suite and report on ALL of them.

   WHY THIS REPLACED THE && CHAIN. `npm test` was one shell chain - suite A &&
   suite B && suite C - so the FIRST failure stopped everything behind it. One
   broken suite meant you learned nothing about the others, and a stale test
   (tutorial.dom.cjs sat failing for two days after the tour was rewritten)
   could hide a real regression further down the list. Every suite runs here,
   and the exit code is still non-zero if any of them failed, so nothing that
   watches this command gets a weaker signal than before.

   A suite is a file that CHECKS something and exits non-zero when it is wrong.
   The *.fixture.cjs / graph-fixture / preview-fixtures files are data and
   helpers that other suites import - they are deliberately not listed. The
   Chromium ones (acadia.smoke.cjs, teacher-insights.browser.cjs) need a real
   browser and stay on their own npm scripts. */
const {spawnSync} = require('node:child_process');

const SUITES = [
  'acadia.dom.cjs',
  'student-insights.dom.cjs',
  'teacher-insights.dom.cjs',
  'workspace.dom.cjs',
  'navigation-cache.cjs',
  'tutorial.dom.cjs',
  // These three existed for weeks and were in no run-list at all, so the file
  // drawer, problem ordering and the workspace's own race conditions were
  // covered on paper and unguarded in practice.
  'inline-file.dom.cjs',
  'problem-sort.dom.cjs',
  'workspace-race.dom.cjs',
];

const failed = [];
for (const suite of SUITES) {
  const r = spawnSync(process.execPath, [suite], {cwd: __dirname, stdio: 'inherit'});
  // A signal (no status) is a failure too - a crashed suite is not a pass.
  if (r.status !== 0) failed.push(`${suite}${r.status === null ? ' (crashed)' : ''}`);
}

console.log(`\n${SUITES.length - failed.length}/${SUITES.length} DOM suites passed.`);
if (failed.length) {
  console.error(`FAILED: ${failed.join(', ')}`);
  process.exitCode = 1;
}
