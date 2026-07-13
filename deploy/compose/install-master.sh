#!/usr/bin/env bash
# One-command master Compose install for disposable / greenfield hosts.
# Creates protected secret files, a local master config, and runs
# `docker compose up -d --wait` for the exact target cardinality.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage: install-master.sh [--project-name NAME] [--port PORT] [--state-dir DIR]

Environment overrides (optional):
  BASE_MASTER_IMAGE_REPOSITORY / BASE_MASTER_IMAGE_DIGEST
  PRISM_IMAGE_REPOSITORY / PRISM_IMAGE_DIGEST
  POSTGRES_IMAGE_REPOSITORY / POSTGRES_IMAGE_DIGEST
  BASE_MASTER_HOST_PORT
EOF
}

PROJECT_NAME="${COMPOSE_PROJECT_NAME:-base-mission-master}"
HOST_PORT="${BASE_MASTER_HOST_PORT:-3180}"
STATE_DIR="${BASE_COMPOSE_STATE_DIR:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-name)
      PROJECT_NAME="$2"
      shift 2
      ;;
    --port)
      HOST_PORT="$2"
      shift 2
      ;;
    --state-dir)
      STATE_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${STATE_DIR}" ]]; then
  STATE_DIR="${XDG_STATE_HOME:-${HOME}/.local/state}/base-compose/${PROJECT_NAME}"
fi

SECRETS_DIR="${STATE_DIR}/secrets"
CONFIG_DIR="${STATE_DIR}/config"
mkdir -p "${SECRETS_DIR}" "${CONFIG_DIR}"
chmod 700 "${STATE_DIR}" "${SECRETS_DIR}" "${CONFIG_DIR}"

_random_token() {
  # 32-byte hex token suitable for file-backed secrets.
  if command -v openssl >/dev/null 2>&1; then
    openssl rand -hex 32
  else
    head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n'
  fi
}

ADMIN_TOKEN_FILE="${SECRETS_DIR}/admin_token"
POSTGRES_PASSWORD_FILE="${SECRETS_DIR}/postgres_password"
PRISM_TOKEN_FILE="${SECRETS_DIR}/prism_shared_token"
MASTER_CONFIG="${CONFIG_DIR}/master.yaml"

if [[ ! -f "${ADMIN_TOKEN_FILE}" ]]; then
  umask 077
  _random_token >"${ADMIN_TOKEN_FILE}"
fi
if [[ ! -f "${POSTGRES_PASSWORD_FILE}" ]]; then
  umask 077
  _random_token >"${POSTGRES_PASSWORD_FILE}"
fi
if [[ ! -f "${PRISM_TOKEN_FILE}" ]]; then
  umask 077
  _random_token >"${PRISM_TOKEN_FILE}"
fi
# Host parent dirs stay 0700. Application secrets (admin/prism) are owned by the
# non-root container uid (1000) with mode 0600. PostgreSQL's official image runs
# as its own user and only needs the password file mount, so that one file is
# mode 0640 (no world read) rather than 0600-as-uid-1000 which it cannot open.
chown 1000:1000 "${ADMIN_TOKEN_FILE}" "${PRISM_TOKEN_FILE}" 2>/dev/null || true
chmod 600 "${ADMIN_TOKEN_FILE}" "${PRISM_TOKEN_FILE}"
chmod 640 "${POSTGRES_PASSWORD_FILE}"

# Resolve local image digests when operator pins are not provided.
_resolve_local_digest() {
  local image_ref="$1"
  docker image inspect --format '{{.Id}}' "${image_ref}" 2>/dev/null \
    | sed -E 's/^sha256://'
}

_tag_with_digest() {
  local source_ref="$1"
  local target_repo="$2"
  local digest
  digest="$(_resolve_local_digest "${source_ref}")"
  if [[ -z "${digest}" ]]; then
    return 1
  fi
  docker tag "${source_ref}" "${target_repo}:compose"
  # Ensure a local named reference that compose can pin with @sha256.
  docker tag "${source_ref}" "${target_repo}@sha256:${digest}" 2>/dev/null || true
  printf '%s' "${digest}"
}

if [[ -z "${BASE_MASTER_IMAGE_REPOSITORY:-}" || -z "${BASE_MASTER_IMAGE_DIGEST:-}" ]]; then
  if digest="$(_tag_with_digest "${BASE_MASTER_LOCAL_IMAGE:-mission/base-master:local}" "mission/base-master")"; then
    BASE_MASTER_IMAGE_REPOSITORY="mission/base-master"
    BASE_MASTER_IMAGE_DIGEST="${digest}"
  elif digest="$(_tag_with_digest "ghcr.io/baseintelligence/base-master:m11-single-port" "mission/base-master")"; then
    BASE_MASTER_IMAGE_REPOSITORY="mission/base-master"
    BASE_MASTER_IMAGE_DIGEST="${digest}"
  else
    echo "BASE_MASTER_IMAGE_REPOSITORY/DIGEST unset and no local master image found." >&2
    echo "Build with: docker build -f docker/Dockerfile.master -t mission/base-master:local ${ROOT_DIR}" >&2
    exit 1
  fi
fi

if [[ -z "${PRISM_IMAGE_REPOSITORY:-}" || -z "${PRISM_IMAGE_DIGEST:-}" ]]; then
  if digest="$(_tag_with_digest "${PRISM_LOCAL_IMAGE:-prism-sdk-review-service:local}" "mission/prism")"; then
    PRISM_IMAGE_REPOSITORY="mission/prism"
    PRISM_IMAGE_DIGEST="${digest}"
  elif digest="$(_tag_with_digest "ghcr.io/baseintelligence/prism:m8-redeploy" "mission/prism")"; then
    PRISM_IMAGE_REPOSITORY="mission/prism"
    PRISM_IMAGE_DIGEST="${digest}"
  else
    echo "PRISM_IMAGE_REPOSITORY/DIGEST unset and no local prism image found." >&2
    exit 1
  fi
fi

POSTGRES_IMAGE_REPOSITORY="${POSTGRES_IMAGE_REPOSITORY:-postgres}"
if [[ -z "${POSTGRES_IMAGE_DIGEST:-}" ]]; then
  if ! docker image inspect postgres:16-alpine >/dev/null 2>&1; then
    docker pull postgres:16-alpine >/dev/null
  fi
  POSTGRES_IMAGE_DIGEST="$(_resolve_local_digest postgres:16-alpine)"
  if [[ -z "${POSTGRES_IMAGE_DIGEST}" ]]; then
    # Fallback to the known alpine digest present on mission hosts.
    POSTGRES_IMAGE_DIGEST="20edbde7749f822887a1a022ad526fde0a47d6b2be9a8364433605cf65099416"
  fi
  # Re-tag so compose can consume repository@sha256 form.
  docker tag "postgres:16-alpine" "postgres@sha256:${POSTGRES_IMAGE_DIGEST}" 2>/dev/null || true
fi

PG_PASSWORD="$(tr -d '\n' <"${POSTGRES_PASSWORD_FILE}")"
# Assemble operator-local master config with DB password (file mode 0600).
# The password never enters Compose YAML or container environment listings.
# Detect host docker.sock group so the non-root master can use compose.
DOCKER_GID="${BASE_DOCKER_GID:-}"
if [[ -z "${DOCKER_GID}" ]]; then
  if [[ -S /var/run/docker.sock ]]; then
    DOCKER_GID="$(stat -c '%g' /var/run/docker.sock 2>/dev/null || true)"
  fi
  DOCKER_GID="${DOCKER_GID:-987}"
fi
umask 077
cat >"${MASTER_CONFIG}" <<EOF
network:
  name: base
  netuid: 100
  chain_endpoint: null
  wallet_name: default
  wallet_hotkey: default
  wallet_path: null
  master_uid: 0
  # No-chain static metagraph: compose app/db networks are internal and do not
  # require live Subtensor connectivity for the control plane to become ready.
  mock_metagraph:
    - hotkey: 5FakeMasterHotkeyForDisposableComposeMission000001
      uid: 0
      validator_permit: true
      stake: 1000.0

master:
  # Public chain/registry control plane (not an operator master IP inventory).
  registry_url: https://chain.joinbase.ai
  proxy_host: 0.0.0.0
  proxy_port: 8081
  epoch_interval_seconds: 360
  metagraph_cache_ttl_seconds: 300
  registry_state_file: /var/lib/base/registry.json
  registry_reconcile_interval_seconds: 30
  challenge_image_update_interval_seconds: 0
  challenge_watcher_interval_seconds: 60
  challenge_watcher_state_path: /var/lib/base/challenge_watcher_state.json
  orchestration_interval_seconds: 30

validator:
  # Public network registry/weights default; Validators attach via --master-url.
  registry_url: https://chain.joinbase.ai
  registry_retry_seconds: 15
  weights_url: null
  weights_interval_seconds: 360
  weights_timeout_seconds: 15.0
  weights_retries: 3
  weights_freshness_seconds: 720
  submit_on_chain_enabled: false

database:
  url: postgresql+asyncpg://base:${PG_PASSWORD}@master-postgres:5432/base

docker:
  network_name: base_challenges
  secret_dir: /var/lib/base/secrets
  internal_network: true
  orchestration_backend: compose
  compose_project_name: ${PROJECT_NAME}
  compose_file: /run/base/compose/docker-compose.yml
  compose_override_dir: /var/lib/base/compose-overrides
  compose_env_file: /run/base/compose/.env
  broker_host: 0.0.0.0
  broker_port: 8082
  broker_url: http://127.0.0.1:9
  broker_workspace_dir: /tmp/base-docker-broker
  broker_allowed_images:
    - ghcr.io/baseintelligence/
  challenge_placement_constraint: null
  cpu_job_constraint: null
  gpu_job_constraint: null

security:
  admin_token_file: /run/secrets/admin_token

observability:
  log_json: true
  sentry_dsn: null
  otel_service_name: base-master
EOF
# Master config embeds the private-network DB URL; keep 0600 and uid 1000.
# Re-assert admin/prism modes after config write. Postgres password stays 0640.
chown 1000:1000 "${MASTER_CONFIG}" "${ADMIN_TOKEN_FILE}" \
  "${PRISM_TOKEN_FILE}" 2>/dev/null || true
chmod 600 "${MASTER_CONFIG}" "${ADMIN_TOKEN_FILE}" "${PRISM_TOKEN_FILE}"
chmod 640 "${POSTGRES_PASSWORD_FILE}"
unset PG_PASSWORD

export COMPOSE_PROJECT_NAME="${PROJECT_NAME}"
export BASE_MASTER_IMAGE_REPOSITORY BASE_MASTER_IMAGE_DIGEST
export PRISM_IMAGE_REPOSITORY PRISM_IMAGE_DIGEST
export POSTGRES_IMAGE_REPOSITORY POSTGRES_IMAGE_DIGEST
export BASE_MASTER_CONFIG="${MASTER_CONFIG}"
export BASE_ADMIN_TOKEN_FILE="${ADMIN_TOKEN_FILE}"
export BASE_POSTGRES_PASSWORD_FILE="${POSTGRES_PASSWORD_FILE}"
export PRISM_SHARED_TOKEN_FILE="${PRISM_TOKEN_FILE}"
export BASE_MASTER_HOST_PORT="${HOST_PORT}"
export BASE_DOCKER_GID="${DOCKER_GID}"
export BASE_COMPOSE_FILE="${COMPOSE_FILE}"

# Sealed compose env: host install pins required by docker-compose.yml
# interpolation. Mounted read-only into the master container so in-process
# ComposeChallengeOrchestrator can pass --env-file for dynamic compose up
# without re-exporting install vars into the host shell on every reconcile.
COMPOSE_ENV_FILE="${CONFIG_DIR}/compose.env"
umask 077
cat >"${COMPOSE_ENV_FILE}" <<EOF
# Generated by install-master.sh — do not hand-edit. Mode 0600.
COMPOSE_PROJECT_NAME=${PROJECT_NAME}
BASE_MASTER_IMAGE_REPOSITORY=${BASE_MASTER_IMAGE_REPOSITORY}
BASE_MASTER_IMAGE_DIGEST=${BASE_MASTER_IMAGE_DIGEST}
PRISM_IMAGE_REPOSITORY=${PRISM_IMAGE_REPOSITORY}
PRISM_IMAGE_DIGEST=${PRISM_IMAGE_DIGEST}
POSTGRES_IMAGE_REPOSITORY=${POSTGRES_IMAGE_REPOSITORY}
POSTGRES_IMAGE_DIGEST=${POSTGRES_IMAGE_DIGEST}
BASE_MASTER_CONFIG=${MASTER_CONFIG}
BASE_ADMIN_TOKEN_FILE=${ADMIN_TOKEN_FILE}
BASE_POSTGRES_PASSWORD_FILE=${POSTGRES_PASSWORD_FILE}
PRISM_SHARED_TOKEN_FILE=${PRISM_TOKEN_FILE}
BASE_MASTER_HOST_PORT=${HOST_PORT}
BASE_DOCKER_GID=${DOCKER_GID}
BASE_COMPOSE_FILE=${COMPOSE_FILE}
BASE_POSTGRES_DB=base
BASE_POSTGRES_USER=base
EOF
chown 1000:1000 "${COMPOSE_ENV_FILE}" 2>/dev/null || true
chmod 600 "${COMPOSE_ENV_FILE}"
export BASE_COMPOSE_ENV_FILE="${COMPOSE_ENV_FILE}"

echo "Installing master Compose project '${PROJECT_NAME}' (API 127.0.0.1:${HOST_PORT})"
docker compose --env-file "${COMPOSE_ENV_FILE}" -p "${PROJECT_NAME}" -f "${COMPOSE_FILE}" config --quiet
docker compose --env-file "${COMPOSE_ENV_FILE}" -p "${PROJECT_NAME}" -f "${COMPOSE_FILE}" up -d --wait

echo "Master Compose install complete."
docker compose --env-file "${COMPOSE_ENV_FILE}" -p "${PROJECT_NAME}" -f "${COMPOSE_FILE}" ps

# Seed the packaged prism challenge into the registry as ACTIVE so reconcile
# adopts the static container instead of treating it as foreign (VAL-COMPOSE-024).
ADMIN_TOKEN="$(tr -d '\n' <"${ADMIN_TOKEN_FILE}")"
PRISM_IMAGE_REF="${PRISM_IMAGE_REPOSITORY}@sha256:${PRISM_IMAGE_DIGEST}"
SEED_PAYLOAD="$(cat <<SEED
{
  "slug": "prism",
  "name": "PRISM",
  "image": "${PRISM_IMAGE_REF}",
  "version": "0.1.0",
  "emission_percent": "30.0000",
  "status": "active",
  "internal_base_url": "http://challenge-prism:8080",
  "required_capabilities": ["get_weights", "proxy_routes"],
  "resources": {},
  "volumes": {"sqlite": "challenge-prism-data"},
  "env": {
    "PRISM_COMBINED_MODE": "true",
    "PRISM_DOCKER_ENABLED": "false"
  },
  "secrets": [],
  "metadata": {"combined_mode_env": "PRISM_COMBINED_MODE"}
}
SEED
)"
if command -v curl >/dev/null 2>&1; then
  for _attempt in 1 2 3 4 5 6 7 8 9 10; do
    if curl -fsS -o /dev/null "http://127.0.0.1:${HOST_PORT}/health"; then
      break
    fi
    sleep 1
  done
  set +e
  create_code="$(
    curl -sS -o /tmp/base-compose-seed-prism.json -w '%{http_code}' \
      -X POST "http://127.0.0.1:${HOST_PORT}/v1/admin/challenges" \
      -H "Content-Type: application/json" \
      -H "X-Admin-Token: ${ADMIN_TOKEN}" \
      -d "${SEED_PAYLOAD}"
  )"
  if [[ "${create_code}" != "200" && "${create_code}" != "201" ]]; then
    # Already present: re-activate / refresh pin.
    curl -sS -o /tmp/base-compose-seed-prism-patch.json -w '%{http_code}' \
      -X PATCH "http://127.0.0.1:${HOST_PORT}/v1/admin/challenges/prism" \
      -H "Content-Type: application/json" \
      -H "X-Admin-Token: ${ADMIN_TOKEN}" \
      -d "{\"image\": \"${PRISM_IMAGE_REF}\", \"status\": \"active\"}" >/dev/null
    curl -sS -o /tmp/base-compose-seed-prism-activate.json -w '%{http_code}' \
      -X POST "http://127.0.0.1:${HOST_PORT}/v1/admin/challenges/prism/activate" \
      -H "X-Admin-Token: ${ADMIN_TOKEN}" >/dev/null || true
  fi
  set -e
  rm -f /tmp/base-compose-seed-prism.json /tmp/base-compose-seed-prism-patch.json \
    /tmp/base-compose-seed-prism-activate.json 2>/dev/null || true
fi
unset ADMIN_TOKEN SEED_PAYLOAD
