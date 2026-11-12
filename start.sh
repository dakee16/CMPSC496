#!/bin/sh
# start.sh - container entrypoint. Refuses to start into a configuration that
# would lose data quietly, then execs uvicorn.
#
# The checks below all guard the SAME failure, which stays invisible until a
# student hits Submit weeks later: anything written under /app lives in the
# image layer, so a redeploy silently reverts it. For the oracle cache that
# means every published problem becomes ungradeable at once
# (main/grading.py:371 loads its tests from there), and the only repair is
# re-running minutes of paid model work per problem.
set -e

fail() { echo "FATAL: $*" >&2; exit 1; }

# Each of these is an env knob the app already reads. No new configuration.
#   MICROTUTOR_SESSION_DB    main/sessions.py:62     - live grading sessions
#   MICROTUTOR_ORACLE_CACHE  main/oracle_store.py:23 - validated oracle tests
#   MICROTUTOR_TRANSCRIPTS   main/transcripts.py:39  - failed-prepare evidence
#   MICROTUTOR_CHUNK_POOL    main/run_phase1.py:55   - gated decompositions
#
# The pool was the one that got away: it had no env knob at all, so it was
# invisible to this loop and lived at /app/main/chunk_pool.json inside the
# image. Every redeploy silently emptied it and the model work to regenerate
# each decomposition was paid again.
for var in MICROTUTOR_SESSION_DB MICROTUTOR_ORACLE_CACHE MICROTUTOR_TRANSCRIPTS MICROTUTOR_CHUNK_POOL; do
    eval "path=\$$var"
    [ -n "$path" ] || fail "$var is unset. It must point at the mounted volume, not the image - see DEPLOY.md."
    case "$path" in
        /app/*|data/*|./data/*)
            fail "$var=$path is inside the image. A redeploy would discard it. Point it at the volume (e.g. /data/...)." ;;
    esac
    dir=$(dirname "$path")
    mkdir -p "$dir" || fail "cannot create $dir - is the volume mounted?"
    DIRS="$DIRS $dir"
done

# First boot on an empty volume: keep the oracles that shipped with the repo.
# Without this the volume starts empty and the already-published problems have
# no tests. Guarded by -e so a real cache is never overwritten by the stale
# image copy on a later deploy.
if [ ! -e "$MICROTUTOR_ORACLE_CACHE" ] && [ -e data/oracles/tests_cache.json ]; then
    cp data/oracles/tests_cache.json "$MICROTUTOR_ORACLE_CACHE"
    echo "seeded oracle cache from the image -> $MICROTUTOR_ORACLE_CACHE"
fi

# One worker unless the shared stores exist. Without them the session store,
# the login limiter and the teacher's live channel are each per-machine or
# per-process, and wrong the moment a second worker exists. See SCALING.md.
WORKERS="${WEB_CONCURRENCY:-1}"
if [ "$WORKERS" -gt 1 ] && { [ -z "$REDIS_URL" ] || [ -z "$DATABASE_URL" ]; }; then
    fail "WEB_CONCURRENCY=$WORKERS needs both REDIS_URL and DATABASE_URL - see SCALING.md."
fi
UVICORN="uvicorn frontend.api_server:app --host 0.0.0.0 --port ${PORT:-8000} --workers $WORKERS"

# Student code runs as a subprocess of this process (main/execution.py), and
# that module says plainly it is a hardened harness, not a secure sandbox. Not
# being root is the fence behind it, so we drop privileges before uvicorn ever
# starts.
#
# The container starts as root ON PURPOSE, rather than via a Dockerfile USER
# line: every host mounts its volume owned by root, so a non-root process could
# not write to /data and the service would refuse to boot on first deploy. Root
# is used for exactly two things - chowning the volume and stepping down.
if [ "$(id -u)" = "0" ]; then
    chown -R appuser:appuser $DIRS 2>/dev/null || true
    if command -v su >/dev/null 2>&1; then
        exec su appuser -s /bin/sh -c "exec $UVICORN"
    fi
    echo "WARNING: still root - 'su' is missing, so privileges were not dropped." >&2
fi

# Already non-root (some hosts do this themselves): the volume must be ours.
if [ "$(id -u)" != "0" ]; then
    for dir in $DIRS; do
        [ -w "$dir" ] || fail "$dir is not writable by $(id -un) (uid $(id -u)). Mount the volume writable, or let the container start as root so it can chown it."
    done
fi

exec $UVICORN
