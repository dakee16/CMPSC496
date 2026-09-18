-- ACADIA — wipe all student learning data, delete all accounts except three.
--
-- Written against the live schema on 2026-09-17. Run the STEP 0 block on its
-- own first and read the numbers; steps 1-2 are irreversible and there is no
-- undo in the Supabase SQL editor.
--
-- WHAT THIS DOES NOT COVER (SQL cannot reach them - see the notes at the end):
--   * the `designs` storage bucket, which holds uploaded plan images
--   * data/grading_sessions.sqlite3 on the server, which holds the live
--     grading sessions and every accepted answer


-- ─────────────────────────────────────────────────────────────────────────
-- STEP 0 — LOOK FIRST. Run this alone. Nothing is changed.
-- ─────────────────────────────────────────────────────────────────────────

SELECT 'students (total)'        AS what, count(*)::text AS n FROM students
UNION ALL SELECT 'students (to KEEP)',  count(*)::text FROM students
  WHERE username IN ('sqg6004@psu.edu','dzm6085@psu.edu','sumsaha@psu.edu')
UNION ALL SELECT 'students (to DELETE)', count(*)::text FROM students
  WHERE username NOT IN ('sqg6004@psu.edu','dzm6085@psu.edu','sumsaha@psu.edu')
UNION ALL SELECT 'mt_sessions',          count(*)::text FROM mt_sessions
UNION ALL SELECT 'mt_submissions',       count(*)::text FROM mt_submissions
UNION ALL SELECT 'mt_messages',          count(*)::text FROM mt_messages
UNION ALL SELECT 'mt_designs',           count(*)::text FROM mt_designs
UNION ALL SELECT 'mt_graphs',            count(*)::text FROM mt_graphs
UNION ALL SELECT 'solved',               count(*)::text FROM solved
UNION ALL SELECT 'student_interactions', count(*)::text FROM student_interactions;

-- And exactly which accounts go vs. stay:
SELECT username, role, first_name, last_name,
       CASE WHEN username IN ('sqg6004@psu.edu','dzm6085@psu.edu','sumsaha@psu.edu')
            THEN 'KEEP (data wiped)' ELSE 'DELETE ACCOUNT' END AS fate
FROM students ORDER BY fate, username;


-- ─────────────────────────────────────────────────────────────────────────
-- STEP 1 + 2 — THE WIPE. Run as one block.
--
-- Every learning record goes, for EVERYONE: the three kept accounts are kept
-- as accounts only, with no history behind them. So these deletes need no
-- WHERE clause - "wipe all student data" and "wipe everyone's data" are the
-- same statement, and an unconditional DELETE is easier to be sure about than
-- a filtered one.
-- ─────────────────────────────────────────────────────────────────────────

BEGIN;

-- 1. the learning record (children first; harmless even without FK cascades)
DELETE FROM mt_submissions;
DELETE FROM mt_messages;
DELETE FROM mt_designs;
DELETE FROM mt_graphs;
DELETE FROM mt_sessions;
DELETE FROM solved;
DELETE FROM student_interactions;

-- 2. the accounts, except the three that stay
DELETE FROM students
WHERE username NOT IN ('sqg6004@psu.edu','dzm6085@psu.edu','sumsaha@psu.edu');

COMMIT;


-- ─────────────────────────────────────────────────────────────────────────
-- STEP 3 — CONFIRM. Every count must be 0 except `students`, which must be
-- exactly the accounts you kept.
-- ─────────────────────────────────────────────────────────────────────────

SELECT 'mt_sessions' AS what, count(*)::text AS n FROM mt_sessions
UNION ALL SELECT 'mt_submissions',       count(*)::text FROM mt_submissions
UNION ALL SELECT 'mt_messages',          count(*)::text FROM mt_messages
UNION ALL SELECT 'mt_designs',           count(*)::text FROM mt_designs
UNION ALL SELECT 'mt_graphs',            count(*)::text FROM mt_graphs
UNION ALL SELECT 'solved',               count(*)::text FROM solved
UNION ALL SELECT 'student_interactions', count(*)::text FROM student_interactions;

SELECT username, role FROM students ORDER BY username;


-- ─────────────────────────────────────────────────────────────────────────
-- NOT DONE BY THIS FILE
--
-- A. UPLOADED PLAN IMAGES live in the `designs` storage bucket under
--    <student_id>/<slug>/round-N.<ext>. Deleting mt_designs removes the
--    RECORD of them, not the files. Empty that bucket from the Supabase
--    dashboard (Storage -> designs -> select all -> delete), or say the word
--    and I will give you a script that does it with the service key.
--
-- B. THE GRADING SESSION STORE is SQLite on the server's disk, not in
--    Postgres: data/grading_sessions.sqlite3 (DEPLOY.md - it is on a named
--    volume, so it survives rebuilds). It holds every live session and every
--    accepted answer, and it is what builds a student's working file. Leaving
--    it makes the site disagree with itself: the file would still show work
--    that the grades no longer know about. On the server:
--
--      sqlite3 data/grading_sessions.sqlite3 \
--        "DELETE FROM submissions; DELETE FROM sessions; VACUUM;"
--
--    Do the same on any dev machine that has been pointed at this database.
-- ─────────────────────────────────────────────────────────────────────────
