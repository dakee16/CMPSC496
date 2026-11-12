# Scaling MicroTutor

A million concurrent students is **not reached by this code alone, and has not
been load-tested.** The application can now run as many stateless instances, but
the capacity still has to be bought, sharded and measured. The sections below
say what exists, what is left, and what it costs.

With none of these settings, the app runs exactly as `DEPLOY.md` describes.

## What the settings change

| Setting | Effect |
|---|---|
| `DATABASE_URL` | Grading sessions, oracle cache, chunk pool and transcripts move from local disk to Postgres |
| `REDIS_URL` | Login limiter, spend caps, model memo, teacher watch channels and the fleet rate limit are shared across processes. Trace lines go to stdout |
| `WEB_CONCURRENCY` | Uvicorn workers. `start.sh` refuses more than 1 without both URLs above |
| `MICROTUTOR_THREADS` | Requests each worker can have in flight (default 40) |
| `MICROTUTOR_RUNNER_URL`, `MICROTUTOR_RUNNER_TOKEN` | Student code runs on `main/runner_service.py`, scaled separately, with no app secrets and no outbound network |
| `MICROTUTOR_MAX_RUNS`, `MICROTUTOR_LLM_MAX_INFLIGHT` | Per-worker concurrency caps. Excess work fails fast as "could not check" and uses no attempt |
| `MICROTUTOR_LLM_RPM` | Fleet-wide model calls per minute. Set it below your OpenAI limit |
| `MICROTUTOR_STATIC_CACHE` | `Cache-Control` for static files, so a CDN can absorb page loads |

Point load balancers at `GET /ready`, which fails when a shared store is down.
`/health` only says the process is up.

## Cutting over from one box

1. Create a Postgres database, preferably behind a transaction-mode pooler.
2. Stop the app, then copy its data. This is safe to repeat:
   `DATABASE_URL=... python -m main.migrate_shared --sqlite /data/grading_sessions.sqlite3`
3. Add `DATABASE_URL` and `MICROTUTOR_RUNNER_TOKEN` to `.env`, then:
   `docker compose -f docker-compose.yml -f docker-compose.scale.yml up -d --build`
4. Install gVisor on the runner hosts and enable `runtime: runsc` in
   `docker-compose.scale.yml` before opening this to the public.

Keep total database connections in budget:
`replicas x workers x DB_POOL_MAX` must fit what the pooler accepts.

To read spend after scaling out, collect the `model_usage` lines from the
containers' stdout. `main/spend.py` reads a file, so point it at an export.

## What one million concurrent students requires

Assumption: each student makes one graded submission and sends one tutor message
every two minutes, or about **8,300 of each per second**. Scale linearly from
your real traffic.

| Tier | Requirement | Status |
|---|---|---|
| Model calls | About 500,000 per minute. With about 3,000 tokens in and 300 out, that is **about $315,000 per hour on gpt-4o** or about $19,000 on gpt-4o-mini, before grading calls. Roughly 1.6 billion tokens per minute | Needs a provider contract far above standard limits. The largest single constraint |
| Code execution | At least one run per submission. At the 53 ms measured on a dev laptop, 440+ cores. At the 5 s CPU limit, up to 41,000 | Scales with runner replicas. Needs autoscaling and gVisor |
| Sessions database | About 17,000 write transactions per second | Beyond a comfortable single primary. Needs a sharded Postgres such as Citus, keyed by `session_id` |
| Supabase tables | Every route reads or writes `students`, `problems`, `assignments` and `mt_*` through PostgREST | **Not addressed.** See below |
| Web tier | About 17,000+ requests per second | Add instances behind a real load balancer. Measure the per-worker rate first |
| Pages | Every static asset | Put a CDN in front and set `MICROTUTOR_STATIC_CACHE` |

## Work still to do in the code

Roughly in the order it will start to hurt:

1. **Move Supabase traffic off PostgREST.** Grading and chat write `mt_submissions`
   and `mt_messages` on every action, and `problems` is read on most routes. It
   needs read caching, batched writes through a queue, and the same sharding.
2. **Make model-waiting routes asynchronous.** Each call holds a thread for
   seconds. It works with large thread pools but is costly in memory.
3. **Move teacher uploads to a background job.** They hold one streaming request
   for minutes, so a deploy kills them. The live channel already spans workers.
4. **Replace `MICROTUTOR_ALLOWED_EMAILS`.** A roster in an environment variable
   does not hold a large population. It belongs in the database.
5. **Add email verification to registration.** Any address at an allowed domain
   can create an account today, and each one can spend model budget.
6. **Add per-student rate limits.** The fleet limit protects the provider
   account, but one student can still consume a large share of it.

## Before trusting any number

Load-test a staging copy with realistic traffic, and raise it stepwise while
watching `/ready`, database wait time, Redis latency and model 429s. Find each
tier's real ceiling. The figures above are estimates, not measurements.
