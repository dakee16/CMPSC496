# Deploying MicroTutor on AWS Lightsail

Two containers on one box: the app, and Caddy in front of it holding TLS and
the network fence. `docker-compose.yml` runs both.

The app serves the API **and** `frontend/`, so the browser is same-origin —
no CORS, no second service, no separate frontend deploy.

## Why not serverless

Vercel and friends break three things this app depends on: grading sessions are
SQLite on local disk (`main/sessions.py:22,61`), teacher upload runs for
**minutes** (`frontend/api_server.py:956`) against function caps of 60–300s, and
`prepare_bus.py` / `auth.py:294` / `trace.py` keep cross-request state in process
memory. All three want one long-lived process with a real disk.

## Sizing

Measured, not guessed:

```
app idle ..............  90 MB
student subprocess ....  up to 512 MB   (main/execution.py:54, RLIMIT_AS)
real data on disk .....  ~100 KB
```

Grading subprocesses run sequentially per request, so the variable is concurrent
students. **4 GB** covers a lab section comfortably. Disk and bandwidth are
irrelevant on any plan.

## 1. Create the instance

Lightsail console → **Create instance**:

- Platform **Linux/Unix**, blueprint **OS Only → Ubuntu 22.04 LTS** (or 24.04)
- Plan: the **4 GB** tier
- Create, then **Networking → attach a static IP**. Without this the address
  changes whenever the instance stops, and your DNS record goes stale.

Then **Networking → IPv4 Firewall**, add:

| Application | Port |
|---|---|
| SSH | 22 |
| HTTP | 80 |
| HTTPS | 443 |

**Leave 80 open to the whole internet.** The Let's Encrypt challenge comes from
their validation servers, not from campus. Restricting 80 to PSU ranges lets the
certificate issue once and then silently fail to renew ~60 days later. The
network fence belongs in `Caddyfile`, not in the firewall.

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
dig +short microtutor.example.edu     # must print the static IP
```

## 4. Clone and configure

```bash
git clone https://github.com/dakee16/CMPSC496.git microtutor
cd microtutor
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
MICROTUTOR_DOMAIN           microtutor.example.edu
MICROTUTOR_ALLOWED_CIDRS    (leave empty until PSU IT sends the ranges)
```

Generate a **fresh** session secret; do not reuse the local one.

Do **not** set the three `/data` paths here — `docker-compose.yml:19-21` pins
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

No seed line means the volume did not mount — stop and fix before anyone signs
in. `main/grading.py:371` loads every problem's tests from that cache, and it is
minutes of paid model work per problem to regenerate.

## 6. Verify

```bash
curl https://microtutor.example.edu/health      # {"status":"ok",...}
```

In a browser: register, sign in, check the cookie has **Secure** set (DevTools →
Application → Cookies), open a problem, submit one chunk.

Then check the fence sees **real** client IPs, not Docker's gateway:

```bash
docker compose logs caddy | tail -20
```

If every request logs `172.x.x.x`, Caddy is matching the Docker bridge and the
allowlist would be meaningless — fix that before narrowing `MICROTUTOR_ALLOWED_CIDRS`.

## 7. Lock it to the PSU network

Ask PSU IT for the VPN's **egress** ranges (what the outside world sees when a
student is on the VPN — not the campus wired blocks). Then:

```bash
nano .env      # MICROTUTOR_ALLOWED_CIDRS=128.118.0.0/16 146.186.0.0/16
docker compose up -d
```

Test from off-VPN: you should get the 403 naming the VPN, not a timeout.

This matters more than it looks. Registration checks only that an address *ends
with* `psu.edu` (`main/auth.py:122`) — there is no confirmation email — so on a
public URL anyone can create an account and reach `main/execution.py`, which
states plainly that it is a hardened harness, not a secure sandbox.

## Updating

```bash
cd microtutor && git pull && docker compose up -d --build
```

The named volume survives rebuilds. Take a Lightsail **snapshot** before
anything risky — it is the one-click way back.

## Operational notes

- **One worker, deliberately** (`start.sh`). `prepare_bus.py`, the per-account
  brute-force counter (`auth.py:294`) and the trace ring are process-local and
  wrong with a second worker. Scale up, not out; scaling out means moving those
  three to Postgres first.
- **The app is not published to the host** — `expose`, not `ports`. Publishing
  8000 would serve it on `http://<ip>:8000`, past both TLS and the fence.
- **Student code runs as non-root.** The container starts as root only to chown
  the volume, then `start.sh` drops to `appuser` before uvicorn starts.
- **Teacher upload holds one long request.** It streams NDJSON so it will not
  look hung, but prefer smaller files.
- **Sessions expire after 12h** (`sessions.py:24`). Permanent history lives in
  Supabase via `main/archive.py`.
