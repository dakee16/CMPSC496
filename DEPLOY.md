# Deploying MicroTutor

One `uvicorn` process serves the API **and** `frontend/`, so the browser is
same-origin and there is no CORS, no second service, and no separate frontend
deploy. `Dockerfile` builds it; `start.sh` runs it.

## Why not Vercel

Vercel (and any serverless host) breaks three things this app depends on:

| What | Where | Why serverless breaks it |
|---|---|---|
| Grading sessions are SQLite on local disk | `main/sessions.py:22,61` | The repo dir is read-only on Lambda, and `/tmp` is per-instance: the session one request creates, the next request cannot see. |
| Teacher upload runs for **minutes** | `frontend/api_server.py:956` | Function duration caps at 60s (Hobby) / 300s (Pro). A 10-problem file dies half-published. |
| Cross-request state is in process memory | `prepare_bus.py`, `auth.py:294`, `trace.py` | Per-instance. "Watch this problem" never connects; the login rate limit multiplies by instance count. |

Use a container host with a persistent volume: **Render, Railway, or Fly**.

## 1. A persistent volume is mandatory

Mount one and point all three paths at it. `start.sh` refuses to boot if any of
them is unset or lands inside the image, because that failure is otherwise
invisible until weeks later.

The oracle cache is the one that hurts. `main/grading.py:371` loads every
problem's tests from it, and regenerating a single problem's oracle costs
minutes of paid model work. It is also **tracked in git** — so without a volume,
every redeploy reverts it and every published problem becomes ungradeable at
once. On first boot against an empty volume, `start.sh` seeds it from the copy
in the image; after that the volume always wins.

Mount it at `/data`. Ownership needs no setup: the container starts as root,
`start.sh` chowns the volume, then drops to the non-root `appuser` before
uvicorn starts — so student code never runs as root, and the volume is still
writable.

## 2. Environment variables

Copy from `.env.example`, which lists every variable the running code reads
along with the file and line that reads it. The minimum:

```
MICROTUTOR_SESSION_SECRET   python -c "import secrets; print(secrets.token_hex(32))"
SUPABASE_URL                your project URL
SUPABASE_KEY                anon key
OPENAI_API_KEY              sk-...
MICROTUTOR_ENV              production        # anything but "dev" => Secure cookie
MICROTUTOR_ALLOWED_DOMAINS  psu.edu

MICROTUTOR_SESSION_DB       /data/grading_sessions.sqlite3
MICROTUTOR_ORACLE_CACHE     /data/oracles/tests_cache.json
MICROTUTOR_TRANSCRIPTS      /data/transcripts.json
```

Do **not** set `DATABASE_URL`, `MICROTUTOR_EXECUTION_BACKEND`, or the other
variables under "NOT WIRED UP" in `.env.example`. `main/config.py` is imported
by nothing, so they change no behaviour — setting them only creates a false
belief that its production checks are running.

`MICROTUTOR_ALLOWED_ORIGINS` is unused in a normal deploy: this app serves its
own pages, so the browser is same-origin.

## 3. One worker, deliberately

`start.sh` pins `--workers 1`. Three stores live in process memory and are wrong
the moment a second worker exists: the teacher's live watch channel
(`prepare_bus.py`), the per-account brute-force counter (`auth.py:294`), and the
trace ring (`trace.py`). Scale by making the instance bigger. Scaling *out*
requires moving those three to Postgres first.

## 4. Host notes

- **Render** — Docker service, add a Disk mounted at `/data`, health check path
  `/health`. `$PORT` is injected.
- **Railway** — detects the `Dockerfile`, add a Volume at `/data`, `$PORT` injected.
- **Fly** — `fly launch --no-deploy`, then `fly volumes create data`, mount it at
  `/data` in `fly.toml`, and set `internal_port` to match `$PORT` (default 8000).

Secrets go in the host's env/secrets UI. `.env` is excluded from the image by
`.dockerignore` — without that, `COPY . .` would bake your keys into a
published layer, where deleting the file later does not remove them.

## 5. After the first deploy

```
curl https://<host>/health            # {"status":"ok",...}
```

Then, in a browser: register an account, sign in, confirm the session cookie has
`Secure` set, open a problem, and submit one chunk. If grading returns
`oracle_unusable`, the volume did not get seeded — check the boot log for
`seeded oracle cache from the image`.

## Known limits

- **Student code runs in the hardened local harness** (`main/execution.py`), not
  a container-per-submission. AST policy, sanitized env, rlimits, isolated mode
  and a wall-clock timeout raise the cost of an attack; the module says plainly
  it is not a mathematically secure sandbox. The deployment container is the
  real boundary, which is why it runs as non-root.
- **Teacher upload holds one long request.** It streams NDJSON so it does not
  look hung, but put the host's request timeout above the longest upload you
  expect, or upload smaller files.
- **Sessions expire after 12h** (`sessions.py:24`) and are dropped on expiry.
  Permanent history lives in Supabase via `main/archive.py`.
