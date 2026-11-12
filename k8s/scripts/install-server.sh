#!/usr/bin/env bash
# install-server.sh - everything a fresh Ubuntu 22.04/24.04 box needs to run
# Acadia on Kubernetes: Docker (to build the image), k3s (Kubernetes, with
# Traefik, metrics-server for autoscaling, and local volumes built in), Helm,
# and cert-manager. Safe to re-run.
#
#     sudo bash k8s/scripts/install-server.sh
#
# Versions float to each project's current stable release unless pinned:
#     K3S_VERSION=v1.33.5+k3s1 CERT_MANAGER_VERSION=v1.19.1 sudo -E bash ...
set -euo pipefail

[ "$(id -u)" = 0 ] || exec sudo -E bash "$0" "$@"
cd "$(dirname "$0")/../.."
TARGET_USER="${SUDO_USER:-root}"
TARGET_HOME="$(getent passwd "$TARGET_USER" | cut -d: -f6)"
step() { echo; echo "==> $*"; }

step "base packages"
apt-get update -qq
apt-get install -y -qq curl git openssl ca-certificates >/dev/null

step "Docker (builds the image)"
if ! command -v docker >/dev/null 2>&1; then
    curl -fsSL https://get.docker.com | sh
fi
if [ "$TARGET_USER" != root ]; then
    usermod -aG docker "$TARGET_USER"
fi

step "k3s"
if ! command -v k3s >/dev/null 2>&1; then
    # Pinned only when asked; the installer's default is the stable channel.
    [ -n "${K3S_VERSION:-}" ] && export INSTALL_K3S_VERSION="$K3S_VERSION"
    curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="server --write-kubeconfig-mode 0600" sh -
fi
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
# kubectl for the login user too, so deploy.sh runs without sudo.
install -d -m 0700 -o "$TARGET_USER" "$TARGET_HOME/.kube"
install -m 0600 -o "$TARGET_USER" /etc/rancher/k3s/k3s.yaml "$TARGET_HOME/.kube/config"
kubectl wait --for=condition=Ready node --all --timeout=300s

step "Traefik settings (long teacher uploads, real client IPs)"
kubectl apply -f k8s/cluster/traefik-config.yaml

step "Helm"
if ! command -v helm >/dev/null 2>&1; then
    curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
fi

step "cert-manager (TLS certificates)"
helm upgrade --install cert-manager cert-manager \
    --repo https://charts.jetstack.io \
    --namespace cert-manager --create-namespace \
    ${CERT_MANAGER_VERSION:+--version "$CERT_MANAGER_VERSION"} \
    --set crds.enabled=true \
    --wait --timeout 10m

step "checking what the autoscaler needs"
kubectl -n kube-system rollout status deploy/metrics-server --timeout=300s
kubectl -n kube-system rollout status deploy/traefik --timeout=300s 2>/dev/null \
    || echo "    traefik is still being redeployed with the new settings; it settles on its own"

cat <<EOF

Done. Next:
  1. Log out and back in (docker group), then from the repo root:
  2. mkdir -p k8s/secrets && cp .env k8s/secrets/app.env       (your real .env)
     cp k8s/deploy.env.example k8s/secrets/deploy.env          (then edit it)
  3. bash k8s/scripts/build-image.sh
  4. bash k8s/scripts/deploy.sh
Open ports 80 and 443 in the server firewall (Lightsail: Networking tab).
Keep 6443 closed to the internet.
EOF
