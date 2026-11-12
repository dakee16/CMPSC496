#!/usr/bin/env bash
# deploy.sh - apply Acadia to the cluster, in the order that cannot lose data:
# secrets, then Postgres and Redis, then the optional Lightsail import, then
# the seed job, and only then the app. Safe to re-run; it is also how a new
# image ships.
#
#     bash k8s/scripts/deploy.sh
#     IMPORT_DATA=acadia-data.tgz bash k8s/scripts/deploy.sh   (first time, KUBERNETES.md)
#     IMAGE=acadia:3f2a9c1 bash k8s/scripts/deploy.sh          (roll back)
#
# Reads k8s/secrets/app.env (your .env) and k8s/secrets/deploy.env.
set -euo pipefail
cd "$(dirname "$0")/../.."

NS=acadia
SECRETS=k8s/secrets
fail() { echo "FATAL: $*" >&2; exit 1; }
step() { echo; echo "==> $*"; }

# Explicit environment wins over deploy.env, so IMAGE=... on the command line
# rolls back without editing the file.
cli_image="${IMAGE:-}"
[ -f "$SECRETS/deploy.env" ] && . "$SECRETS/deploy.env"
IMAGE="${cli_image:-${IMAGE:-}}"
IMAGE="${IMAGE:-$(cat "$SECRETS/last-image" 2>/dev/null || true)}"
OVERLAY="${OVERLAY:-test}"
TLS_ISSUER="${TLS_ISSUER:-selfsigned}"

command -v kubectl >/dev/null || fail "kubectl not found - run k8s/scripts/install-server.sh"
[ -n "$IMAGE" ] || fail "no image - run k8s/scripts/build-image.sh first"
[ -n "${ACADIA_HOST:-}" ] || fail "ACADIA_HOST is unset - see k8s/deploy.env.example"
[ -f "$SECRETS/app.env" ] || fail "$SECRETS/app.env is missing - copy your .env there"
[ -d "k8s/overlays/$OVERLAY" ] || fail "no overlay k8s/overlays/$OVERLAY"
case "$TLS_ISSUER" in letsencrypt*) [ -n "${ACME_EMAIL:-}" ] || fail "TLS_ISSUER=$TLS_ISSUER needs ACME_EMAIL" ;; esac

# kubectl's --from-env-file takes every byte after '=' literally, so the
# inline comments .env.example carries ("gpt-4o   # :35 ...") would become part
# of the model name. Docker compose strips them; this does the same: drop
# comment lines, strip " # ..." from unquoted values, unwrap quoted ones.
clean_env() {
    awk '
        /^[[:space:]]*(#|$)/ { next }
        {
            line = $0
            sub(/^[[:space:]]*(export[[:space:]]+)?/, "", line)
            i = index(line, "=")
            if (i == 0) next
            key = substr(line, 1, i - 1)
            val = substr(line, i + 1)
            sub(/[[:space:]]+$/, "", key)
            if (key !~ /^[A-Za-z_][A-Za-z0-9_]*$/) next
            sub(/^[[:space:]]+/, "", val)
            q = substr(val, 1, 1)
            if (q == "\"" || q == "\047") {
                rest = substr(val, 2)
                j = index(rest, q)
                val = j ? substr(rest, 1, j - 1) : rest
            } else {
                sub(/[[:space:]]+#.*$/, "", val)
                sub(/[[:space:]]+$/, "", val)
            }
            print key "=" val
        }' "$1"
}
env_value() { grep -E "^$1=" "$2" 2>/dev/null | tail -1 | cut -d= -f2- || true; }
random_secret() { openssl rand -hex 32; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
clean_env "$SECRETS/app.env" > "$WORK/app.env"

# The same refusals start.sh and DEPLOY.md make, before anything is applied.
[ -n "$(env_value MICROTUTOR_SESSION_SECRET "$WORK/app.env")" ] \
    || fail "MICROTUTOR_SESSION_SECRET is empty in app.env - sign-in would answer 503"
[ "$(env_value MICROTUTOR_ENV "$WORK/app.env")" != dev ] \
    || echo "WARNING: MICROTUTOR_ENV=dev - the session cookie will not be Secure." >&2

# ── generated secrets: made once, kept in k8s/secrets/generated.env ─────────
# Regenerating the Postgres password would lock the app out of its own data,
# so an existing file is always reused.
GEN="$SECRETS/generated.env"
token="$(env_value MICROTUTOR_RUNNER_TOKEN "$GEN")"
pgpass="$(env_value POSTGRES_PASSWORD "$GEN")"
[ -n "$token" ] || token="$(random_secret)"
[ -n "$pgpass" ] || pgpass="$(random_secret)"
( umask 077; printf 'MICROTUTOR_RUNNER_TOKEN=%s\nPOSTGRES_PASSWORD=%s\n' "$token" "$pgpass" > "$GEN" )

printf 'MICROTUTOR_RUNNER_TOKEN=%s\n' "$token" > "$WORK/generated.env"
if [ "$OVERLAY" = test ]; then
    # Hex password, so it needs no escaping inside the URL.
    printf 'DATABASE_URL=postgresql://acadia:%s@acadia-postgres:5432/acadia\n' "$pgpass" >> "$WORK/generated.env"
else
    [ -n "$(env_value DATABASE_URL "$WORK/app.env")" ] \
        || fail "OVERLAY=$OVERLAY has no database in the cluster - set DATABASE_URL in app.env"
fi

CONFIG_SHA="$(cat "$WORK/app.env" "$WORK/generated.env" | sha256sum | cut -c1-16)"
render() {
    sed -e "s|__IMAGE__|$IMAGE|g" -e "s|__HOST__|$ACADIA_HOST|g" \
        -e "s|__TLS_ISSUER__|$TLS_ISSUER|g" -e "s|__CONFIG_SHA__|$CONFIG_SHA|g" \
        -e "s|__ACME_EMAIL__|${ACME_EMAIL:-}|g"
}
apply_secret() {
    kubectl -n "$NS" create secret generic "$1" --from-env-file="$2" \
        --dry-run=client -o yaml | kubectl apply -f - >/dev/null
}

echo "Deploying $IMAGE to https://$ACADIA_HOST (overlay: $OVERLAY, TLS: $TLS_ISSUER)"

step "namespace, certificate issuers, secrets"
kubectl apply -f k8s/base/namespace.yaml
if [ -n "${ACME_EMAIL:-}" ]; then
    render < k8s/cluster/issuers.yaml | kubectl apply -f -
else
    render < k8s/cluster/issuers.yaml | kubectl apply -f - -l '!acadia.io/acme'
fi
apply_secret acadia-app "$WORK/app.env"
apply_secret acadia-generated "$WORK/generated.env"
if [ "$OVERLAY" = test ]; then
    printf 'POSTGRES_PASSWORD=%s\n' "$pgpass" > "$WORK/postgres.env"
    apply_secret acadia-postgres "$WORK/postgres.env"
fi

kubectl kustomize "k8s/overlays/$OVERLAY" | render > "$WORK/all.yaml"

step "data tier (Redis$( [ "$OVERLAY" = test ] && echo ', Postgres'))"
kubectl apply -f "$WORK/all.yaml" -l acadia.io/tier=data
kubectl -n "$NS" rollout status statefulset/acadia-redis --timeout=300s
if [ "$OVERLAY" = test ]; then
    kubectl -n "$NS" rollout status statefulset/acadia-postgres --timeout=300s
fi

# ── optional: the Lightsail volume, before the seed so its newer rows win ───
if [ -n "${IMPORT_DATA:-}" ]; then
    [ -f "$IMPORT_DATA" ] || fail "IMPORT_DATA=$IMPORT_DATA does not exist"
    step "importing $IMPORT_DATA"
    kubectl -n "$NS" delete pod acadia-import --ignore-not-found --wait
    render < k8s/jobs/import-pod.yaml | kubectl -n "$NS" apply -f -
    kubectl -n "$NS" wait --for=condition=Ready pod/acadia-import --timeout=180s
    kubectl -n "$NS" cp "$IMPORT_DATA" acadia-import:/tmp/data.tgz
    kubectl -n "$NS" exec acadia-import -- sh -c '
        set -e
        mkdir -p /tmp/import && tar xzf /tmp/data.tgz -C /tmp/import
        MICROTUTOR_ORACLE_CACHE=/tmp/import/oracles/tests_cache.json \
        MICROTUTOR_CHUNK_POOL=/tmp/import/chunk_pool.json \
        MICROTUTOR_TRANSCRIPTS=/tmp/import/transcripts.json \
            python -m main.migrate_shared --sqlite /tmp/import/grading_sessions.sqlite3'
    kubectl -n "$NS" delete pod acadia-import --wait=false
fi

step "seeding Postgres from the image (fills gaps only)"
kubectl -n "$NS" delete job acadia-migrate --ignore-not-found --wait
render < k8s/jobs/migrate.yaml | kubectl -n "$NS" apply -f -
if ! kubectl -n "$NS" wait --for=condition=complete job/acadia-migrate --timeout=300s; then
    kubectl -n "$NS" logs job/acadia-migrate --tail=40 || true
    fail "the seed job did not finish - the app was NOT rolled out"
fi
kubectl -n "$NS" logs job/acadia-migrate

step "app, runner, ingress, autoscalers"
kubectl apply -f "$WORK/all.yaml"

# The optional network fence: Caddy's MICROTUTOR_ALLOWED_CIDRS, as a Traefik
# middleware. Unset means everyone, as on Lightsail.
cidrs="$(env_value MICROTUTOR_ALLOWED_CIDRS "$WORK/app.env" | tr -s ' ,' '\n' | sed '/^$/d')"
if [ -n "$cidrs" ]; then
    {
        printf 'apiVersion: traefik.io/v1alpha1\nkind: Middleware\nmetadata:\n  name: acadia-allowlist\nspec:\n  ipAllowList:\n    sourceRange:\n'
        printf '%s\n' "$cidrs" | sed 's/^/      - /'
    } | kubectl -n "$NS" apply -f -
    kubectl -n "$NS" annotate ingress acadia --overwrite \
        traefik.ingress.kubernetes.io/router.middlewares=acadia-acadia-https@kubernetescrd,acadia-acadia-allowlist@kubernetescrd
else
    kubectl -n "$NS" delete middleware acadia-allowlist --ignore-not-found >/dev/null
fi

kubectl -n "$NS" rollout status deploy/acadia-runner --timeout=600s
kubectl -n "$NS" rollout status deploy/acadia-app --timeout=900s

step "state"
kubectl -n "$NS" get pods,hpa,ingress
cat <<EOF

Live at https://$ACADIA_HOST - check it with:
  curl -k https://$ACADIA_HOST/ready        # {"status":"ok"} = Postgres and Redis reachable
EOF
