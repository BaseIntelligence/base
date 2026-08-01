#!/usr/bin/env bash
# remote-deploy.sh — rsync control-plane tree to a droplet and restart compose.
#
#   remote-deploy.sh --host root@IP --role master|validator [--gateway-endpoint URL]
#
# Does NOT copy secrets from the operator machine by default. Secrets must
# already exist on the host under deploy/env/*.env and deploy/secrets/* (age path).
# Optional: --bootstrap-secrets-from HOST copies secrets once from another host.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

HOST=""
ROLE=""
ENV="${BASE_DEPLOY_ENV:-staging}"
GATEWAY_ENDPOINT=""
BOOTSTRAP_FROM=""
BUILD_FROM="${BASE_DOCKER_BUILD_FROM:-source}"
REMOTE_DIR="${BASE_REMOTE_DIR:-/opt/base}"
SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)
if [[ -n "${BASE_SSH_IDENTITY:-}" ]]; then
  SSH_OPTS+=(-i "$BASE_SSH_IDENTITY")
fi

die() { echo "remote-deploy: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="${2:-}"; shift 2 ;;
    --role) ROLE="${2:-}"; shift 2 ;;
    --gateway-endpoint) GATEWAY_ENDPOINT="${2:-}"; shift 2 ;;
    --env) ENV="${2:-}"; shift 2 ;;
    --bootstrap-secrets-from) BOOTSTRAP_FROM="${2:-}"; shift 2 ;;
    --build-from) BUILD_FROM="${2:-}"; shift 2 ;;
    --remote-dir) REMOTE_DIR="${2:-}"; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) die "unknown arg: $1" ;;
  esac
done

[[ -n "$HOST" ]] || die "--host required"
case "$ROLE" in master|validator) ;; *) die "--role master|validator required" ;; esac
case "$ENV" in staging|prod) ;; *) die "--env staging|prod required" ;; esac

ssh_h() { ssh "${SSH_OPTS[@]}" "$HOST" "$@"; }
scp_h() { scp "${SSH_OPTS[@]}" "$@"; }

echo "remote-deploy: host=$HOST role=$ROLE env=$ENV remote=$REMOTE_DIR build_from=$BUILD_FROM"

ssh_h "mkdir -p '$REMOTE_DIR' && command -v docker >/dev/null"

if [[ -n "$BOOTSTRAP_FROM" ]]; then
  echo "remote-deploy: bootstrap secrets from $BOOTSTRAP_FROM"
  tmp="$(mktemp -d)"
  # shellcheck disable=SC2029
  ssh "${SSH_OPTS[@]}" "$BOOTSTRAP_FROM" \
    "tar -C /opt/gbase/deploy -cf - env secrets 2>/dev/null || tar -C /opt/base/deploy -cf - env secrets" \
    >"$tmp/secrets.tar"
  ssh_h "mkdir -p '$REMOTE_DIR/deploy' && tar -C '$REMOTE_DIR/deploy' -xf -" <"$tmp/secrets.tar"
  # Convert legacy GBASE_* keys to BASE_* if present.
  ssh_h "bash -s" <<'EOS'
set -euo pipefail
for f in /opt/base/deploy/env/*.env; do
  [[ -f "$f" ]] || continue
  if grep -q '^GBASE_' "$f" 2>/dev/null; then
    sed -i 's/^GBASE_/BASE_/g' "$f"
  fi
  chmod 600 "$f"
done
if [[ -d /opt/base/deploy/secrets ]]; then
  chown -R 65532:65532 /opt/base/deploy/secrets || true
  find /opt/base/deploy/secrets -type f -exec chmod 400 {} \;
fi
EOS
  rm -rf "$tmp"
fi

echo "remote-deploy: rsync tree"
if [[ "$BUILD_FROM" == "prebuilt" ]]; then
  for b in validator gateway updater agent-challenge hypertraining-challenge prism-challenge agent-runner; do
    [[ -x "$ROOT/target/release/$b" ]] || die "missing prebuilt binary target/release/$b — run: cargo build --release --features validator-bin/dcap -p validator-bin -p gateway-bin -p updater-bin -p agent-challenge-bin -p hypertraining-challenge-bin -p prism-challenge-bin -p agent-runner-bin"
  done
fi

RSYNC_SSH="ssh ${SSH_OPTS[*]}"
rsync -az --delete \
  -e "$RSYNC_SSH" \
  --exclude '.git/' \
  --exclude 'target/' \
  --exclude 'deploy/terraform/.terraform/' \
  --exclude 'deploy/terraform/terraform.tfstate*' \
  --exclude 'deploy/terraform/tfplan' \
  --exclude 'deploy/terraform/terraform.tfvars' \
  --exclude 'deploy/env/*.env' \
  --exclude 'deploy/secrets/' \
  --exclude 'miner-runtime/' \
  --exclude '.omo/' \
  "$ROOT/" "$HOST:$REMOTE_DIR/"

# Ensure secrets dirs exist (empty OK if not bootstrapped).
# deploy/secrets/lium is bind-mounted by prism-challenge, so it must be a real
# directory with real files: compose would otherwise create directories where
# the container expects files.
ssh_h "mkdir -p '$REMOTE_DIR/deploy/env' '$REMOTE_DIR/deploy/secrets/lium' '$REMOTE_DIR/deploy/secrets/wallets' \
  && chmod 700 '$REMOTE_DIR/deploy/secrets' '$REMOTE_DIR/deploy/secrets/lium' \
  && for f in api_key ssh_ed25519 ssh_ed25519.pub; do \
       [ -e '$REMOTE_DIR/deploy/secrets/lium/'\$f ] || : > '$REMOTE_DIR/deploy/secrets/lium/'\$f; \
     done \
  && chmod 400 '$REMOTE_DIR/deploy/secrets/lium/'* \
  && chown -R 65532:65532 '$REMOTE_DIR/deploy/secrets/lium' \
  && chmod -R a-w '$REMOTE_DIR/deploy/secrets/wallets' 2>/dev/null; \
  chown -R 65532:65532 '$REMOTE_DIR/deploy/secrets/wallets' 2>/dev/null; true"

# Materialize missing env from examples (dev-safe placeholders) if absent
ssh_h "bash -s" <<EOS
set -euo pipefail
cd '$REMOTE_DIR'
for ex in deploy/env/*.env.example; do
  base="\${ex%.example}"
  if [[ ! -f "\$base" ]]; then
    cp "\$ex" "\$base"
    # Prefer BASE_ keys from examples; strip any GBASE leftover
    sed -i 's/^GBASE_/BASE_/g' "\$base"
    chmod 600 "\$base"
    echo "created \$base from example"
  else
    sed -i 's/^GBASE_/BASE_/g' "\$base" || true
  fi
done
EOS

COMPOSE_FILES=(-f docker-compose.yml)
PROFILE_ARGS=()
case "$ROLE" in
  master)
    COMPOSE_FILES+=(-f deploy/compose/role-master.yml)
    PROFILE_ARGS=(--profile master)
    ;;
  validator)
    COMPOSE_FILES+=(-f deploy/compose/role-validator.yml)
    ;;
esac
case "$ENV" in
  staging) COMPOSE_FILES+=(-f deploy/compose/env-staging.yml) ;;
  prod)    COMPOSE_FILES+=(-f deploy/compose/env-prod.yml) ;;
esac

GE_EXPORT=""
if [[ -n "$GATEWAY_ENDPOINT" ]]; then
  GE_EXPORT="export BASE_GATEWAY_ENDPOINT='$GATEWAY_ENDPOINT';"
fi

if [[ "$BUILD_FROM" == "prebuilt" ]]; then
  echo "remote-deploy: sync release binaries"
  ssh_h "mkdir -p '$REMOTE_DIR/target/release'"
  rsync -az -e "$RSYNC_SSH"     "$ROOT/target/release/validator"     "$ROOT/target/release/gateway"     "$ROOT/target/release/updater"     "$ROOT/target/release/agent-challenge"     "$ROOT/target/release/hypertraining-challenge"     "$ROOT/target/release/prism-challenge"     "$ROOT/target/release/agent-runner"     "$HOST:$REMOTE_DIR/target/release/"
fi

echo "remote-deploy: build + up"
# shellcheck disable=SC2029
ssh_h "bash -s" <<EOS
set -euo pipefail
cd '$REMOTE_DIR'
$GE_EXPORT
export BASE_DOCKER_BUILD_FROM='$BUILD_FROM'
export COMPOSE_PROJECT_NAME=base
# Build service images from current tree (source) unless prebuilt binaries exist.
docker compose ${COMPOSE_FILES[*]} ${PROFILE_ARGS[*]} build
# The updater can only pull from a registry. Enable it only when the desired
# image is a registry reference (host/path@sha256:…), otherwise every tick 404s
# trying to pull a locally-built tag from Docker Hub.
UP_PROFILE=""
desired=\$(sed -n 's/^BASE_UPDATER_DESIRED_IMAGE=//p' deploy/env/updater.env 2>/dev/null | tail -1)
case "\$desired" in
  */*) UP_PROFILE="--profile auto-update"; echo "updater enabled (registry image)" ;;
  *)
    echo "updater disabled (desired image '\$desired' is not a registry reference)"
    # Deselecting a profile does not remove an already-running container.
    docker compose ${COMPOSE_FILES[*]} --profile auto-update rm -sf updater >/dev/null 2>&1 || true
    ;;
esac
docker compose ${COMPOSE_FILES[*]} ${PROFILE_ARGS[*]} \$UP_PROFILE up -d --remove-orphans
docker compose ${COMPOSE_FILES[*]} ${PROFILE_ARGS[*]} \$UP_PROFILE ps
# Local health probes via published tunnels if present, else container exec.
sleep 5
if curl -fsS -m 5 http://127.0.0.1:18080/healthz >/dev/null 2>&1; then
  echo "validator tunnel health: \$(curl -fsS -m 5 http://127.0.0.1:18080/healthz)"
elif docker compose ${COMPOSE_FILES[*]} ${PROFILE_ARGS[*]} exec -T validator curl -fsS -m 5 http://127.0.0.1:8080/healthz >/dev/null 2>&1; then
  echo "validator health: ok (in-container)"
else
  echo "validator health: probe deferred (container may still be starting)"
fi
if [[ '$ROLE' == 'master' ]]; then
  if docker compose ${COMPOSE_FILES[*]} ${PROFILE_ARGS[*]} exec -T gateway curl -fsS -m 5 http://127.0.0.1:8080/healthz >/dev/null 2>&1; then
    echo "gateway health: ok"
  else
    echo "gateway health: probe deferred"
  fi
fi
EOS

echo "remote-deploy: done ($ROLE @ $HOST)"
