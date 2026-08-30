-- MicroTutor: teacher-uploaded assignments.
--
-- Safe to re-run. Adds the assignment concept and a per-problem readiness
-- flag, without touching the 100 existing curated problems (they simply have
-- assignment_id IS NULL and stay invisible to the assignment browser).

create table if not exists assignments (
  id           uuid primary key default gen_random_uuid(),
  name         text        not null,
  -- No teacher auth yet (demo role picker). PSU email/JWT replaces this later;
  -- keeping it a plain column now means that migration is a backfill, not a
  -- schema rewrite.
  teacher_name text        not null default 'demo-teacher',
  source_file  text,
  created_at   timestamptz not null default now()
);

alter table problems add column if not exists assignment_id uuid
  references assignments(id) on delete cascade;

-- ready = this problem has a STRONG mutation-validated oracle AND a gated
-- decomposition. ONLY ready problems may be shown to a student: an unready one
-- cannot be graded, so offering it would dead-end the student.
alter table problems add column if not exists ready boolean not null default false;

-- Why preparation failed, shown to the TEACHER only. Never sent to a student.
alter table problems add column if not exists prepare_error text;

create index if not exists problems_assignment_idx on problems(assignment_id);
create index if not exists problems_ready_idx      on problems(assignment_id, ready);

-- One slug per assignment, so re-uploading a file replaces rather than
-- duplicates. NOT partial: Postgres cannot infer a PARTIAL unique index for
-- ON CONFLICT unless the statement repeats the index predicate, which PostgREST
-- cannot send -- an earlier `where assignment_id is not null` here made every
-- upsert fail with 42P10. A plain index is safe for the pre-existing curated
-- problems because their assignment_id is NULL, and a key containing NULL never
-- conflicts with anything in a unique index.
create unique index if not exists problems_assignment_slug_uniq
  on problems(assignment_id, slug);
