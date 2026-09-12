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

`workspace.dom.cjs` exercises the staged question workflow using the actual
frontend scripts and synthetic API responses: reading first, locked stages,
keyboard navigation, both planning methods, rejection and approval, preserving
unsent chat/code and selected files, moving the same conversation into the
reference panel, focus restoration, step feedback and deliberate continuation,
grading failures, completion, comparison retry, restart, initialization failure,
and historical work. No browser layout or real editor measurement is asserted.

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

## Guided question workspace

The question opens in a reading stage with a wide statement and one next action.
Students can return to earlier stages without losing their current editor draft,
chat, selected file, or step review. Stage navigation does not make API calls.

| Stage | Main task | Available on demand |
| --- | --- | --- |
| Read | Understand the question and examples, then continue | Tutor explanation |
| Plan | Talk through an approach or upload a PNG/JPEG/PDF; submit for approval | Plan preview, question, tutor |
| Code | Read one instruction, write code, and inspect the result before continuing | All steps, earlier answers, question, approved plan, tutor, focus mode |
| Reflect | Review completion and the plan/code comparison | Completed function, recorded grades, another problem |

Code unlocks only after the existing server approval. Reflect opens after
completion or when a historical comparison exists; historical work is explicitly
labeled. Approval and completion offer a continuation action rather than moving
the student automatically. After a passing step, the existing read-only review
view displays their answer until they continue; the grading session has already
advanced, and the next editor draft remains separate.

The reference panel shares the original problem, plan, and chat DOM, preserving
their state. It supports Escape, focus trapping and restoration, and an inert
background. Plan updates and comparison responses no longer scroll the page.
Restart remains in the question options menu with its confirmation dialog.
Assignment downloads stay available in the problem list.

Check the new stages in a real browser at desktop, tablet, and phone widths,
including 200% zoom, long questions, both themes, the mobile keyboard, graph
zoom/fullscreen, and the reference panel. The browser suite now captures reading
and planning views and checks staged coding and the reference panel; its actual
execution and screenshot review remain pending in an environment with Chromium.

## Shared UI design decisions

- ACADIA uses a reusable vector A mark, teal actions, sea-glass light surfaces,
  and ink/mint dark surfaces. The editor follows the selected theme.
- Light is the default; an existing saved dark preference is respected.
  Legacy session/storage keys remain unchanged to avoid losing sign-in state.
- Solving collapses desktop navigation to a labeled, tooltip-backed icon rail.
  One learning stage occupies the main canvas, with help available on demand.
- The problem statement is collapsible and uses the page's natural scroll. Code and
  doctests retain whitespace and scroll horizontally instead of wrapping into
  illegible paragraphs. All source content is inserted as text, never raw HTML.
- The current step and instruction sit directly above the editor. The full
  step list is available through View all steps.
  Completed steps remain reviewable; reviewing does not discard an unsent draft.
- Focus mode hides the question heading and stage rail while expanding the work
  area without resetting editor contents. Escape exits focus or the reference panel.
- The handback/download area uses equal vertical padding and a single outlined
  surface instead of competing borders and inconsistent margins.

## Verification status for this change

DOM/contrast tests and JavaScript syntax checks were run in the editing
environment. Chromium could not be installed there: its download timed out,
and the system package manager was permission-blocked. Consequently, the
browser suite and screenshot/visual review remain pending and must be run
before merging. All three DOM suites pass. All eight focused progress endpoint
tests passed for the dashboard change; this workspace update does not change
backend files. Live Supabase integration was not exercised.
