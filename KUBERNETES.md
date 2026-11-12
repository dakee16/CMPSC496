# Running MicroTutor on Kubernetes

This runs the scaled-out shape from `SCALING.md` on Kubernetes. The app and the
code runner each autoscale, and Postgres and Redis hold the shared state.
Nothing here changes how the Lightsail box in `DEPLOY.md` runs.

The test setup is one Ubuntu server running k3s. k3s is a full Kubernetes that
ships with the ingress (Traefik), the metrics the autoscaler reads, and local
volumes.

```
Traefik (TLS) ──> acadia-app  x2-4  ──> acadia-runner x2-4   (student code, no network out)
                     │    │
                     │    └──> acadia-redis      (limits, spend caps, live channels)
                     └──────> acadia-postgres   (sessions, oracles, roadmaps)
                     └──────> Supabase, OpenAI  (unchanged)
```

## What is in `k8s/`

| Path | What it is |
|---|---|
| `base/` | App, runner, Redis, ingress, autoscalers, disruption budgets, network policies |
| `components/in-cluster-postgres/` | A Postgres inside the cluster. The test overlay uses it |
| `overlays/test/` | One server: in-cluster Postgres, autoscaling capped at 4 pods each |
| `overlays/production/` | Managed Postgres from `DATABASE_URL`, runner under gVisor, higher caps |
| `cluster/` | Certificate issuers and Traefik's timeouts, applied once per cluster |
| `jobs/` | The seed job and the Lightsail import pod, run by `deploy.sh` |
| `scripts/` | `install-server.sh`, `build-image.sh`, `deploy.sh` |
| `secrets/` | Your `.env` and generated passwords. Gitignored, and never in the image |

## 1. The server

Ubuntu 22.04 or 24.04, with **4 vCPU and 8 GB** RAM. Two app pods, two runners,
Postgres, Redis and k3s itself need about 3 GB at rest. The rest is room for the
autoscaler to add pods. Open ports **22, 80 and 443**. Do not open 6443.

```bash
git clone https://github.com/dakee16/CMPSC496.git acadia && cd acadia
sudo bash k8s/scripts/install-server.sh
```

This installs Docker, k3s, Helm and cert-manager. It is safe to re-run. Log out
and back in afterwards, so the docker group applies.

## 2. Configure

```bash
mkdir -p k8s/secrets
cp /path/to/your/.env k8s/secrets/app.env        # the same file DEPLOY.md uses
cp k8s/deploy.env.example k8s/secrets/deploy.env
nano k8s/secrets/deploy.env                      # at least ACADIA_HOST
```

- **No DNS name yet:** set `ACADIA_HOST=<server-ip>.nip.io` and
  `TLS_ISSUER=selfsigned`. The browser warns once, then sign-in works.
- **Real DNS name:** point an A record at the server, set `ACME_EMAIL`, and
  start with `TLS_ISSUER=letsencrypt-staging`. Switch to `letsencrypt` once the
  staging certificate issues.

`app.env` needs `MICROTUTOR_ENV=production` (or any value except `dev`), or the
session cookie is not Secure. Leave out `DATABASE_URL`, `REDIS_URL` and the
`/data` paths: the manifests set them. `deploy.sh` generates the Postgres
password and the runner token, and stores them in `k8s/secrets/generated.env`.
Back that file up. If you lose it, the app can't log in to its own database.

## 3. Build and deploy

```bash
bash k8s/scripts/build-image.sh
bash k8s/scripts/deploy.sh
```

`deploy.sh` runs these steps in order, and stops before the app if any of them fails:

1. It applies the secrets.
2. It starts Postgres and Redis.
3. It runs the seed job, which copies the oracle tests that ship in the repo
   into Postgres. Without that, every published problem is ungradeable on a
   fresh database.
4. It rolls out the app and runner.

## 4. Verify

```bash
curl -k https://$ACADIA_HOST/ready         # {"status":"ok"}: Postgres and Redis reachable
kubectl -n acadia get pods,hpa
kubectl -n acadia logs deploy/acadia-app -f
```

Then repeat the browser checks from `DEPLOY.md`: register, sign in, open a
problem and submit one step. The submission runs on `acadia-runner`, which you
can confirm with `kubectl -n acadia logs deploy/acadia-runner`.

To see the autoscaler react, keep `kubectl -n acadia get hpa -w` open while
several people submit code at once. A quicker check: temporarily lower
`averageUtilization` in `base/runner.yaml` to 10 and redeploy. For a proper
load test, see "Before trusting any number" in `SCALING.md`.

## Moving data across from Lightsail

Do this on the first deploy only, so the real data lands before the seed fills
the gaps. Snapshot Lightsail first. Then, on the Lightsail box:

```bash
cd ~/acadia && docker compose stop app      # sessions must not change mid-copy
docker volume ls | grep microtutor-data     # usually acadia_microtutor-data
docker run --rm -v acadia_microtutor-data:/data -v "$PWD":/out alpine \
    tar czf /out/acadia-data.tgz -C /data .
docker compose start app
```

Copy `acadia-data.tgz` to the test server, then:

```bash
IMPORT_DATA=acadia-data.tgz bash k8s/scripts/deploy.sh
```

The import brings across grading sessions, oracle tests, roadmaps, transcripts
and review help. Spend caps (`account_spend.json`) start from zero, because
Redis now keeps them. Imported rows are never overwritten, so re-running is safe.

## Shipping a change

```bash
git pull
bash k8s/scripts/build-image.sh && bash k8s/scripts/deploy.sh
```

This is a rolling update with no downtime. Each old pod finishes its in-flight
requests (up to 15 minutes, for teacher uploads) before it stops. Editing only
`app.env` needs just `deploy.sh`, and the pods restart on their own.

Rolling back: `IMAGE=acadia:<older tag> bash k8s/scripts/deploy.sh`. The tags
are listed by `sudo k3s ctr images ls | grep acadia`.

## Troubleshooting

| Symptom | Look at |
|---|---|
| Pod `Pending` | `kubectl -n acadia describe pod <name>`. Usually not enough CPU or memory on the node |
| `ErrImageNeverPulled` / `ImagePullBackOff` | The image was imported on a different node. With more than one node, use `REGISTRY=` |
| App never `Ready` | `/ready` failed: `kubectl -n acadia logs deploy/acadia-app \| grep 'not ready'` |
| Seed job failed | `kubectl -n acadia logs job/acadia-migrate`. The app was not rolled out |
| Certificate not issued | `kubectl -n acadia describe certificate acadia-tls`. Let's Encrypt needs DNS and port 80 |
| Every submission "could not check" | Runner token mismatch or the runner is down: `kubectl -n acadia logs deploy/acadia-runner` |
| Allowlist blocks everyone | Traefik isn't seeing real client IPs. Check `kubectl -n kube-system get svc traefik` shows `externalTrafficPolicy: Local` |

## Before production

- **Database:** use managed Postgres with backups (RDS, or Supabase's own
  Postgres through its pooler), not the in-cluster one. Set `DATABASE_URL` in
  `app.env` and `OVERLAY=production`.
- **gVisor:** install `runsc` on every node and register it with containerd
  before using the production overlay. Its runner asks for `runtimeClassName:
  gvisor` and never starts without it.
- **More than one node:** push images to a registry (`REGISTRY=...`). Spread
  the pods and add a node autoscaler, such as Karpenter on EKS or the cloud's
  cluster autoscaler.
- **Ingress:** `base/ingress.yaml` is written for Traefik. On a cluster with a
  different ingress, replace the Middleware and its annotation, and carry over
  the 15-minute streaming timeouts from `cluster/traefik-config.yaml`.
- The unaddressed limits in `SCALING.md` still apply: PostgREST traffic to
  Supabase, the roster in an environment variable, and per-student rate limits.
