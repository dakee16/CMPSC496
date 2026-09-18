-- ACADIA — record the teacher's file order for problems uploaded before it
-- was stored.
--
-- WHY. main/assignments.py numbers every block as it reads down the uploaded
-- file, but only the CLASS path copied that number into a column. A flat
-- assignment therefore reached the database with group_order and member_order
-- both null, every reader fell back to sorting by slug, and LAB1 was served
-- alphabetically - so "Employee Update" was offered as a good place to start
-- when it is the LAST problem in the file. The code now stores the order for
-- new uploads; this fixes the rows that predate it.
--
-- HOW THE ORDER IS RECOVERED. Problems are prepared and inserted one at a time,
-- in file order (main/publish.prepare_assignment_stream is a plain sequential
-- loop), so created_at preserves the order the teacher wrote. Class problems
-- already have their real order and are left alone.


-- ─────────────────────────────────────────────────────────────────────────
-- STEP 0 — LOOK FIRST. Which rows have no order, and what will they become?
-- ─────────────────────────────────────────────────────────────────────────

SELECT a.name AS assignment, p.slug, p.created_at,
       row_number() OVER (PARTITION BY p.assignment_id ORDER BY p.created_at) - 1
         AS will_become_group_order
FROM problems p
JOIN assignments a ON a.id = p.assignment_id
WHERE p.group_order IS NULL
ORDER BY a.name, p.created_at;


-- ─────────────────────────────────────────────────────────────────────────
-- STEP 1 — BACKFILL. Only rows with no order at all are touched.
-- ─────────────────────────────────────────────────────────────────────────

BEGIN;

WITH ordered AS (
  SELECT id,
         row_number() OVER (PARTITION BY assignment_id ORDER BY created_at) - 1
           AS pos
  FROM problems
  WHERE group_order IS NULL
)
UPDATE problems p
SET group_order  = o.pos,
    -- A plain function is its own group of one, so it sits at member 0 and the
    -- single ordering "group_order, member_order, slug" works for both shapes.
    member_order = COALESCE(p.member_order, 0)
FROM ordered o
WHERE p.id = o.id;

COMMIT;


-- ─────────────────────────────────────────────────────────────────────────
-- STEP 2 — CONFIRM. This is the order students will now be served.
-- ─────────────────────────────────────────────────────────────────────────

SELECT a.name AS assignment, p.group_order, p.member_order, p.slug, p.title
FROM problems p
JOIN assignments a ON a.id = p.assignment_id
WHERE p.ready = true
ORDER BY a.name, p.group_order, p.member_order, p.slug;

-- LAB1 should read: frequency, invert, employee-update.
-- HW3 should keep its class order: Stack (isEmpty, __len__, push, pop, peek),
-- then Calculator, then AdvancedCalculator.
