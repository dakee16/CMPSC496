# ACADIA UI checks

These checks use synthetic API responses. They never start the Python backend,
query student records, invoke a model, or execute submitted Python code.

Requires Node 20.19+:

```sh
cd tests/ui
npm install
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
  restart, opening failures, and restored work.
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
| Read | Question, signature, and examples | Tutor explanation; one continuation action |
| Plan | Choose chat or PNG/JPEG/PDF upload; review and submit an approach | Live plan preview; inline question reference; the same tutor |
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

The sign-in account note now has its own scoped style and a simple top divider.
Overview metrics group each icon, label, and value together. Instructor heading,
metrics, and information sections use consistent 24px spacing.

## Verification status

All three DOM suites pass for this change, including 26 contrast pairs.
JavaScript syntax and patch whitespace were checked. Backend files are unchanged;
the eight focused progress tests passed for the preceding dashboard change.

`acadia.smoke.cjs` has since been run in Chromium and passes, with screenshots
captured. A visual pass covered the workspace and question screens in light and
dark at 1600px and the workspace at 390px; the frozen listing's gutter was
measured against CodeMirror's and both sets of digits right-align at the same
pixel, with the code columns flush.

Still reviewed by eye only when someone looks: tablet widths, 200% zoom,
long descriptions and code, narrow-screen keyboard use, pointer resizing, and
graph zoom/fullscreen.
