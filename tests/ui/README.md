# ACADIA UI checks

## Teacher dashboard

The teacher overview reads saved `mt_sessions` and `mt_submissions` records through
the teacher-only `/teacher/dashboard` endpoint. No migration, chart dependency,
model call, or new grade calculation is required. Insights cover ready problems
in published assignments and each registered student's latest session per problem.
Earlier sessions remain available in the work-history download.

Overview (`teacher.html`) contains class insights. The separate Assignments tab
(`teacher-assignments.html`) owns the library, upload drawer, preparation, and
publishing controls. The dashboard checks also cover navigation between these
pages and selecting a file in the upload drawer.

Each bar shows one measure: students with an incorrect answer they have not yet
corrected. A plain-language definition appears before the chart, and each row
spells out the student count and how many tried the problem. Repeated submissions
do not inflate these counts; indeterminate grading results are excluded. The
suggested review comes first. Selecting a problem opens its steps and students;
selecting a student reveals their feedback. Counting rules sit in a disclosure.

```sh
.venv/bin/python -m pytest tests/test_teacher_dashboard.py -q
cd tests/ui
node teacher-insights.dom.cjs
UI_ARTIFACTS=/tmp/acadia-teacher-dashboard npm run test:teacher-browser
```

The dashboard's DOM and Chromium checks pass, including light/dark themes and
320–1440px layouts. The broader DOM run has an existing failure at
`workspace.dom.cjs:213` (expected `stageRead`, got `stageCode`); that test and
all frontend files it loads are unchanged by the teacher-dashboard work.

These checks use synthetic API responses. They never start the Python backend,
query student records, invoke a model, or execute submitted Python code.

Requires Node 20.19+:

```sh
cd tests/ui
npm ci
npm test
npx playwright install chromium
UI_ARTIFACTS=./artifacts npm run test:browser
```

The browser suite serves the real frontend locally and intercepts application
API calls. CodeMirror and fonts load from the app's existing CDNs. Screenshots
are saved when `UI_ARTIFACTS` is set and require visual review.

## What is checked

- `acadia.dom.cjs`: ACADIA branding, fresh and saved themes, safe prose/code
  formatting, approval gating, step placement, completed-step review, editor
  draft preservation, focus mode, tutor visibility, unique IDs, and 26 semantic
  and code contrast pairs across both palettes.
- `student-insights.dom.cjs`: student navigation, dashboard metrics and activity,
  assignment links into the approval gate, grade filters and details, theme
  switching, safe titles, and empty/error/retry states.
- `workspace.dom.cjs`: the shared workspace, both planning methods, approval and
  rejection, unsent chat/code and selected-file preservation, a persistent tutor
  during coding, references that leave code and chat interactive, focus
  restoration, keyboard resizing limits, replies that do not steal editor
  focus, paced step review, grading failures, completion/comparison retries,
  restart, opening failures, restored work, graph loading while history is
  pending, and retry after history fails without discarding drafts.
- `navigation-cache.cjs`: account-scoped caching, concurrent request coalescing,
  expiry and background refresh, explicit refresh, write/logout invalidation,
  cross-tab account revalidation, stale HTTP/network errors, and in-flight
  invalidation races. Pure graph tests check nested-loop ordering, separate
  return lanes, orthogonal edges that avoid nodes, label bounds, reversed input
  order, self-loops, diamonds, and prototype-safe node IDs.
- `tutorial.dom.cjs`: first-visit invitation, dismissal, Settings replay, role
  handling, safe return destinations, the complete factorial exercise, and
  automatic exit. Asserts that tutorial requests are only GET `/auth/me`, with
  no course-record writes.
- `acadia.smoke.cjs`: the real CodeMirror editor in Chromium, both themes,
  reading/planning/coding, assignment handbacks, approval gating, step review,
  focus mode, tutor visibility, non-overlapping work/tutor panels, inline
  reference bounds, metric alignment, compact sign-in notes, and seven viewport
  widths across student/instructor/grades/playground/login screens.

Happy DOM cannot verify physical layout, real editor measurements, sticky
positioning, pointer dragging, or screenshots. The browser suite covers those
layout concerns where automated assertions are practical; inspect the images
for readability and visual balance.

The student progress endpoint also has focused Python tests:

```sh
python -m pytest tests/test_student_progress.py -q
```

These use a fake database and exercise real tally helpers and signed cookies:
student isolation, unpublished/unready exclusions, independent versus assisted
credit, duplicates, activity, and empty records. Live Supabase integration
requires a separate staging check.

## Shared question workspace

A student starts by reading the question, then makes a plan, codes, and reflects.
The same tutor stays alongside the main task throughout. On desktop, students
can resize the divider with a pointer or the arrow/Home/End keys, or hide the
tutor. On narrow screens the tutor follows the work in the same page, with
Ask tutor and Back to code shortcuts. It never overlays the editor.

| Stage | Main task | Supporting material |
| --- | --- | --- |
| Question & plan | Read the question and examples; discuss or upload an approach; submit it for review | The same tutor; live graph; upload and submit controls beside the plan |
| Code | Current step, editor, feedback, and deliberate continuation | Persistent tutor; inline question/plan; all steps; earlier answers; focus mode |
| Reflect | Completion and plan/code comparison | Tutor; completed function; grades; another problem |

Code still unlocks only after server approval. Reflect becomes available on
completion or when a historical comparison exists, labeled as earlier work.
Approval and completion do not move the student automatically. Passing a step
opens their submitted answer for review until they choose to continue.

Question and plan references expand in a bounded region above the work area.
They share the original DOM, support Escape and focus restoration, and leave
the rest of the page interactive. Chat never moves between containers. Changing
stages, hiding the tutor, resizing, and focus mode retain drafts and conversation.
Tutor responses do not steal focus from someone who has started coding.

The current step remains directly above the editor; the full step list is under
View all steps. Focus mode expands the work area while retaining tutor access.
Restart remains in the problem options menu with confirmation. Assignment
downloads remain in the problem list.

Overview metrics align labels and values on one left edge. Assignment rows,
next steps, and recent activity use compact layouts instead of nested oversized
cards. Both portals share the same density rules, Settings, and light/graphite
dark palettes.

## Navigation cache and guided practice

The frontend caches only an allowlist of GET results: progress, assignments,
assignment problems, solved status, and instructor grades. Entries are private
to a verified account and a browser tab, fresh for 45 seconds, and retained for
up to five minutes while a background refresh runs. Successful writes and
logout invalidate them; auth, chat, history, grading, and streaming responses
are never cached. Manual Refresh bypasses cached data. Assignments reuse
progress membership instead of requesting every problem list up front.

Tutorial completion is remembered per account on the current browser, not
across devices. The factorial exercise is isolated frontend practice; it never
creates an assignment, tutor session, submission, or grade. Finishing closes
the sample, and Settings always offers replay.

For manual testing with synthetic student/instructor data:

```sh
npm run preview:fixtures
```

Open `http://localhost:8123/dashboard.html` for the student portal or
`http://localhost:8124/teacher.html` for the instructor portal. These fixtures
are not a backend integration or production preview.

## Verification status

All five Node/DOM suites pass for this change, including 26 contrast pairs.
JavaScript syntax and patch whitespace were checked. Backend files are
unchanged; backend integration tests were not run for this frontend change.

The browser suite was updated but was not run for this revision: the available
browser rejected the local fixture URL with `net::ERR_BLOCKED_BY_CLIENT`.
Earlier browser results do not verify this revision. A real-browser pass is
still needed for both themes and portals, responsive layout, 200% zoom, long
content, keyboard navigation, pointer resizing, and graph zoom/fullscreen.
