# Deploying MicroTutor on a single VM

Target: one Ubuntu VM with **1 vCPU and 2 GB RAM**. Nothing here is tied to
a cloud provider - any VM with Docker and a public IP works.

Two containers on one box: the app, and Caddy in front of it holding TLS and
the network fence. `docker-compose.yml` runs both.

The app serves the API **and** `frontend/`, so the browser is same-origin -
no CORS, no second service, no separate frontend deploy.

## Why not serverless

Vercel and friends break three things this app depends on: grading sessions are
SQLite on local disk (`main/sessions.py:22,61`), teacher upload runs for
**minutes** (`frontend/api_server.py:956`) against function caps of 60-300s, and
`prepare_bus.py` / `auth.py:294` / `trace.py` keep cross-request state in process
memory. All three want one long-lived process with a real disk.

## Sizing

Measured, not guessed:

```
app idle ..............  90 MB
caddy .................  ~30 MB
student subprocess ....  up to 256 MB each   (main/execution.py, RLIMIT_AS)
real data on disk .....  ~100 KB
```

The variable is concurrent students, and on 2 GB it has to be BOUNDED, not
hoped about. Grading runs on FastAPI's thread pool, so without a cap a burst of
submissions starts dozens of student processes at once and the OOM killer takes
the whole app down. `main/execution.py` therefore admits at most
`MICROTUTOR_EXEC_CONCURRENCY` (default **2**) student programs at a time; the
rest queue for up to `MICROTUTOR_EXEC_QUEUE_WAIT` seconds (default 30). A
submission that times out in the queue is reported as a server error, never as
a failed attempt, so the student just resubmits.

Worst case is therefore roughly 90 + 30 + 2 x 256 MB, well inside 2 GB. Each
run is capped at 5 CPU-seconds, so on one vCPU a queued student waits seconds,
not minutes. Raise the cap only after giving the VM more RAM and cores:

| Knob (in `.env`) | Default | What it bounds |
|---|---|---|
| `MICROTUTOR_EXEC_CONCURRENCY` | 2 | student programs running at once |
| `MICROTUTOR_EXEC_MEM_MB` | 256 | address space of ONE student program |
| `MICROTUTOR_EXEC_QUEUE_WAIT` | 30 | seconds a submission waits for a slot |

## 1. Prepare the VM

- **Ubuntu 22.04 or 24.04 LTS**, 1 vCPU / 2 GB.
- A **stable public IP** (a static/reserved address). If it changes, the DNS
  record goes stale and the certificate stops renewing.
- Inbound firewall (provider security group and/or `ufw`) open on **22, 80,
  443**.

**Leave 80 open to the whole internet.** The Let's Encrypt challenge comes from
their validation servers, not from campus. Restricting 80 lets the certificate
issue once and then silently fail to renew ~60 days later.

**Add swap.** 2 GB has no headroom for a `docker compose build` plus a running
app, and swap turns a brief memory spike into slowness instead of a killed
process:

```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
echo 'vm.swappiness=10' | sudo tee /etc/sysctl.d/99-swappiness.conf
sudo sysctl --system
free -h          # Swap: 2.0Gi
```

## 2. Install Docker

SSH in (the console's browser terminal is fine), then:

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
```

Log out and back in so the group takes effect. Verify with `docker ps`.

## 3. Point DNS at the box

Create an **A record** for your hostname → the static IP. Confirm it resolves
before step 5, or Caddy's first certificate request fails:

```bash
dig +short acadiapsu.com     # must print the static IP
```

## 4. Clone and configure

```bash
git clone https://github.com/dakee16/CMPSC496.git acadia
cd acadia
cp .env.example .env
nano .env
```

Fill in, at minimum:

```
MICROTUTOR_SESSION_SECRET   python3 -c "import secrets; print(secrets.token_hex(32))"
SUPABASE_URL                your project URL
SUPABASE_KEY                anon key
OPENAI_API_KEY              sk-...
MICROTUTOR_ENV              production
MICROTUTOR_ALLOWED_DOMAINS  psu.edu
MICROTUTOR_ALLOWED_EMAILS   the class roster, comma-separated (see step 7)
MICROTUTOR_INSTRUCTOR_EMAILS  instructors (also listed in ALLOWED_EMAILS)
MICROTUTOR_DOMAIN           acadiapsu.com
MICROTUTOR_ALLOWED_CIDRS    leave empty (see step 7)
```

Generate a **fresh** session secret; do not reuse the local one.

Do **not** set the three `/data` paths here - `docker-compose.yml:19-21` pins
them, so they cannot be mistyped on a new box.

Do **not** set `DATABASE_URL`, `MICROTUTOR_EXECUTION_BACKEND`, or anything under
"NOT WIRED UP" in `.env.example`. `main/config.py` is imported by nothing, so
those change no behaviour and only create a false belief that its production
checks run.

## 5. Start

```bash
docker compose up -d --build
docker compose logs -f
```

Watch for, in order:

```
seeded oracle cache from the image -> /data/oracles/tests_cache.json
LAUNCHED: uvicorn frontend.api_server:app --host 0.0.0.0 --port 8000 --workers 1
certificate obtained successfully          (from caddy)
```

No seed line means the volume did not mount - stop and fix before anyone signs
in. `main/grading.py:371` loads every problem's tests from that cache, and it is
minutes of paid model work per problem to regenerate.

## 6. Verify

```bash
curl https://acadiapsu.com/health      # {"status":"ok",...}
```

In a browser: register, sign in, check the cookie has **Secure** set (DevTools →
Application → Cookies), open a problem, submit one chunk.

Then check the fence sees **real** client IPs, not Docker's gateway:

```bash
docker compose logs caddy | tail -20
```

If every request logs `172.x.x.x`, Caddy is matching the Docker bridge and the
allowlist would be meaningless - fix that before narrowing `MICROTUTOR_ALLOWED_CIDRS`.

## 7. Restrict who can get in

Access is gated by the **roster**, not by network. `MICROTUTOR_ALLOWED_EMAILS`
(`main/auth.py`) is checked on registration, on sign-in, and on every request
carrying a cookie; anyone not on it gets the not-authorized page, and removing
an address cuts off an existing session on its next request. That does not care
where a student is sitting, which a VPN allowlist does.

**Set the roster before you hand out the URL.** Left empty, the old rule
applies: any address *ending with* `psu.edu` may register (`main/auth.py:122`),
with no confirmation email - so on a public URL anyone can create an account and
reach `main/execution.py`, which states plainly that it is a hardened harness,
not a secure sandbox.

```bash
nano .env      # MICROTUTOR_ALLOWED_EMAILS=abc123@psu.edu,def456@psu.edu,...
docker compose up -d
```

The Caddy network fence is still there and defaults to "allow everything". If
you later also want to narrow by network, set `MICROTUTOR_ALLOWED_CIDRS` to PSU
VPN **egress** ranges (from PSU IT) and check first that the Caddy logs show
real client IPs (step 6). It is optional, not a substitute for the roster.

## Shipping a change

Edit locally, never on the server - a stray edit there is either clobbered by
the next `git pull` or turns into a merge conflict at the worst moment.

```bash
# on your machine
git add -A && git commit -m "..." && git push

# on the server
cd ~/acadia && git pull && docker compose up -d --build
docker compose logs -f            # watch it come back, Ctrl-C to detach
```

`--build` is required for ANY code change, frontend included: `COPY . .` bakes
the files into the image, so nothing is live-mounted. It is not slow, though -
the pip layer is cached, so a typical rebuild is ~20s rather than the first
build's few minutes.

WHAT SURVIVES a rebuild: everything on the named volumes - grading sessions,
the oracle cache, transcripts, and Caddy's certificate. Containers are
disposable; the volumes are the state.

DOWNTIME is roughly 20-40s while the new container starts. Fine between labs,
worth avoiding mid-class.

### Only changed .env?

No rebuild needed - the image has not changed, only the environment handed to
it:

```bash
docker compose up -d
```

### Frontend edits and the browser cache

Nothing to do - just deploy. `RevalidatedFiles` (frontend/api_server.py) adds
`Cache-Control: no-cache` to every static file, so the browser revalidates on
each request and picks up a changed .js or .css immediately. "no-cache" still
caches: unchanged files answer 304 with no body, so this costs one small
conditional request per asset, not a re-download.

The old `?v=N` query strings are gone. They were manual cache-busting that the
header made redundant, and leaving them in only invited someone to keep bumping
a number that no longer does anything.

### Rolling back

```bash
git log --oneline -5                       # find the last good commit
git checkout <sha> && docker compose up -d --build
```

If that is not enough - a wrecked volume, a bad data migration - restore a VM
snapshot / disk backup instead (whatever your provider offers). Which is why the snapshot is taken BEFORE anything
risky, not after.

## Operational notes

- **Student programs are queued, not unbounded.** At most
  `MICROTUTOR_EXEC_CONCURRENCY` run at once (see Sizing). If students report
  "server busy" during a lab, the box is the bottleneck: give it more RAM/CPU
  and raise the cap, rather than raising the cap alone.
- **One worker, deliberately** (`start.sh`). `prepare_bus.py`, the per-account
  brute-force counter (`auth.py:294`) and the trace ring are process-local and
  wrong with a second worker. Scale up, not out; scaling out means moving those
  three to Postgres first.
- **The app is not published to the host** - `expose`, not `ports`. Publishing
  8000 would serve it on `http://<ip>:8000`, past both TLS and the fence.
- **Student code runs as non-root.** The container starts as root only to chown
  the volume, then `start.sh` drops to `appuser` before uvicorn starts.
- **Teacher upload holds one long request.** It streams NDJSON so it will not
  look hung, but prefer smaller files.
- **Sessions expire after 12h** (`sessions.py:24`). Permanent history lives in
  Supabase via `main/archive.py`.
