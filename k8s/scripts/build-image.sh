#!/usr/bin/env bash
# build-image.sh - build the one image the app, runner and jobs all run, and
# hand it to the cluster. Tagged by commit, so every deploy names exactly
# what it runs and a rollback is IMAGE=<older tag>.
#
#     bash k8s/scripts/build-image.sh                     single-node k3s
#     REGISTRY=ghcr.io/you bash k8s/scripts/build-image.sh   any cluster
#
# Without REGISTRY the image is imported straight into k3s's containerd, which
# only that node can see. A second node needs a registry.
set -euo pipefail
cd "$(dirname "$0")/../.."

TAG="${TAG:-$(git rev-parse --short HEAD 2>/dev/null || echo dev)}"
# Uncommitted edits get their own tag, so the same sha never means two images.
if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    TAG="$TAG-dirty-$(date +%Y%m%d%H%M%S)"
fi
IMAGE="${REGISTRY:+$REGISTRY/}acadia:$TAG"

echo "==> docker build $IMAGE"
docker build -t "$IMAGE" .

if [ -n "${REGISTRY:-}" ]; then
    docker push "$IMAGE"
else
    echo "==> importing into k3s"
    docker save "$IMAGE" | sudo k3s ctr images import -
fi

mkdir -p k8s/secrets
echo "$IMAGE" > k8s/secrets/last-image
echo "==> built $IMAGE (deploy.sh uses it next)"
