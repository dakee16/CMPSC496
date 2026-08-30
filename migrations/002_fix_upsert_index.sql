-- FIX for migration 001: the unique index was created PARTIAL
-- (`where assignment_id is not null`). Postgres cannot use a partial index to
-- infer an ON CONFLICT target unless the statement repeats the predicate, which
-- PostgREST cannot send, so every problem upsert failed with:
--   42P10 there is no unique or exclusion constraint matching the ON CONFLICT
-- Preparation succeeded but nothing saved, leaving assignments at 0 / 0.
--
-- Safe on the existing curated problems: their assignment_id is NULL, and a
-- unique-index key containing NULL never conflicts with another row.

drop index if exists problems_assignment_slug_uniq;

create unique index if not exists problems_assignment_slug_uniq
  on problems(assignment_id, slug);
