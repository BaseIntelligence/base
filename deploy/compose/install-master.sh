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
# Bind-mounted into non-root containers (uid 1000). Host parent dirs stay 0700.
# Files are therefore readable by the container user without embedding secrets in env.
chmod 644 "${ADMIN_TOKEN_FILE}" "${POSTGRES_PASSWORD_FILE}" "${PRISM_TOKEN_FILE}"

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
    - hotkey: 5ComposeMasterMockHotkey000000000000000000000000
      uid: 0
      validator_permit: true
      stake: 1000.0

master:
  registry_url: https://chain.joinbase.ai
  proxy_host: 0.0.0.0
  proxy_port: 8081
  epoch_interval_seconds: 360
  metagraph_cache_ttl_seconds: 300
  registry_state_file: /var/lib/base/registry.json
  registry_reconcile_interval_seconds: 0
  challenge_image_update_interval_seconds: 0
  orchestration_interval_seconds: 30

validator:
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
chmod 644 "${MASTER_CONFIG}"
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

echo "Installing master Compose project '${PROJECT_NAME}' (API 127.0.0.1:${HOST_PORT})"
docker compose -p "${PROJECT_NAME}" -f "${COMPOSE_FILE}" config --quiet
docker compose -p "${PROJECT_NAME}" -f "${COMPOSE_FILE}" up -d --wait

echo "Master Compose install complete."
docker compose -p "${PROJECT_NAME}" -f "${COMPOSE_FILE}" ps
