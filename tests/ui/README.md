# ACADIA UI checks

These tests do not start the Python backend or use student records, credentials,
models, grading, oracle validation, or benchmark data. The browser suite serves
the actual frontend locally and intercepts every application API call with
synthetic fixtures. CodeMirror and fonts still load from the app's existing CDNs.

Requires Node 20.19+.

```sh
cd tests/ui
npm install
npm test
npx playwright install chromium
npm run test:browser
```

To capture the browser suite's screenshots:

```sh
UI_ARTIFACTS=./artifacts npm run test:browser
```

`acadia.dom.cjs` checks branding, fresh and saved themes, source-safe text/code
formatting, locked/unlocked states, progress placement, completed-step review,
draft preservation, focus mode, tutor drawer accessibility, duplicate IDs, and
26 semantic/code color contrast pairs. Its CodeMirror stub cannot validate real
editor behavior or browser layout.

`acadia.smoke.cjs` checks the real editor and DOM in Chromium, assignment and
handback views, the design gate, step submission/review using mocked responses,
focus mode, tutor drawer behavior, seven viewport widths, student/instructor/
grades/playground/login pages, and light/dark modes. It captures screenshots
when `UI_ARTIFACTS` is set. Screenshots require human visual review.

`student-insights.dom.cjs` checks the three student navigation sections,
dashboard metrics and activity, assignment links into the existing design gate,
grade filters and step details, safe rendering of titles, theme switching, and
empty/error/retry states. It uses synthetic records and does not verify layout.

The private progress endpoint has focused Python tests. With the backend's
dependencies and pytest installed, run from the repository root:

```sh
python -m pytest tests/test_student_progress.py -q
```

These tests exercise the real tally/percentage helpers and signed cookies with
a fake database: student isolation, unpublished/unready work, repeated attempts,
completion versus credit, missing step counts, local-day activity, paginated
history, authentication, no-store responses, and database failures. They do not
connect to Supabase or invoke grading, oracle execution, or model providers.

## Student dashboard and grades

- Dashboard is the student landing page. It shows completed problems, earned
  step credit, assignment progress, seven days of practice, recent activity,
  milestones, and links to the next unfinished problems.
- My assignments retains the existing practice flow. Links from the dashboard
  and grades open the chosen assignment/problem while preserving the design
  approval gate. Unpublished assignments are hidden in all three sections.
- My grades groups problems by assignment with search, status filters, sorting,
  step-level results, and links back to practice. Completion and earned credit
  are separate; unknown step counts display a dash, never a fabricated zero.
- Both pages use the shared light/dark tokens and responsive layouts. The new
  `/student/progress` read endpoint derives the account from the signed cookie,
  returns no reference code or peer records, and disables response caching.

Manual browser review for the new pages remains pending: check both themes at
375, 768, 1024, and 1600 pixels; keyboard navigation and step expansion; long
assignment/problem names; grade table scrolling on mobile; and dashboard links
into an assignment and a problem. Also confirm against a development database
that the student's recorded step credit matches the instructor's grade sheet.

## UI design decisions

- ACADIA uses a reusable vector A mark, teal actions, sea-glass light surfaces,
  and ink/mint dark surfaces. The editor follows the selected theme.
- Light is the default; an existing saved dark preference is respected.
  Legacy session/storage keys remain unchanged to avoid losing sign-in state.
- Solving collapses desktop navigation to a labeled, tooltip-backed icon rail.
  At 1440px and above, problem/editor/tutor share the screen. Below that, the
  tutor becomes a keyboard-accessible drawer, leaving more width for the work.
- The problem statement is collapsible and independently scrollable. Code and
  doctests retain whitespace and scroll horizontally instead of wrapping into
  illegible paragraphs. All source content is inserted as text, never raw HTML.
- Sticky progress and the current instruction sit directly above the editor.
  Completed steps remain reviewable; reviewing does not discard an unsent draft.
- Focus mode expands the work area without resetting editor contents. Escape
  exits focus mode or closes the tutor drawer.
- The handback/download area uses equal vertical padding and a single outlined
  surface instead of competing borders and inconsistent margins.

## Verification status for this change

DOM/contrast tests and JavaScript syntax checks were run in the editing
environment. Chromium could not be installed there: its download timed out,
and the system package manager was permission-blocked. Consequently, the
browser suite and screenshot/visual review remain pending and must be run
before merging. Both DOM suites and all eight focused progress endpoint tests
pass. Live Supabase integration was not exercised.
