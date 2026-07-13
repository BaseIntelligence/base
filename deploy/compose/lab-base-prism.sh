#!/usr/bin/env bash
# Durable disposable Base master + postgres + Prism lab (Compose only).
# Unique project name, mission-band host port, digest-pinned images via install-master,
# always-teardown with sealed env. Never touches live Swarm or set_weights.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"
INSTALLER="${SCRIPT_DIR}/install-master.sh"
TEARDOWN="${SCRIPT_DIR}/teardown-master.sh"

usage() {
  cat <<'EOF'
Usage:
  lab-base-prism.sh up --project-name NAME --port PORT [--state-dir DIR]
  lab-base-prism.sh health --project-name NAME --port PORT [--state-dir DIR]
  lab-base-prism.sh down --project-name NAME [--state-dir DIR] [--keep-data]

Rules:
  - Project names should look like base-prism-lab-<suffix>
  - Port must NOT be 19080 or 443 (live production listeners)
  - Prefer free ports in 3181-3199 mission band
  - Always run "down" after lab work (destroys volumes by default)
EOF
}

cmd="${1:-}"
shift || true
if [[ -z "${cmd}" || "${cmd}" == "-h" || "${cmd}" == "--help" ]]; then
  usage
  exit 0
fi

PROJECT_NAME="${COMPOSE_PROJECT_NAME:-}"
HOST_PORT="${BASE_MASTER_HOST_PORT:-}"
STATE_DIR="${BASE_COMPOSE_STATE_DIR:-}"
KEEP_DATA=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-name) PROJECT_NAME="$2"; shift 2 ;;
    --port) HOST_PORT="$2"; shift 2 ;;
    --state-dir) STATE_DIR="$2"; shift 2 ;;
    --keep-data) KEEP_DATA=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${PROJECT_NAME}" ]]; then
  echo "--project-name is required" >&2
  usage >&2
  exit 2
fi

if [[ -z "${STATE_DIR}" ]]; then
  STATE_DIR="/tmp/${PROJECT_NAME}"
fi
ENV_FILE="${STATE_DIR}/config/compose.env"

case "${HOST_PORT}" in
  19080|443)
    echo "refusing live production port ${HOST_PORT}" >&2
    exit 2
    ;;
esac

case "${cmd}" in
  up)
    if [[ -z "${HOST_PORT}" ]]; then
      echo "--port is required for up" >&2
      exit 2
    fi
    bash "${INSTALLER}" --project-name "${PROJECT_NAME}" --port "${HOST_PORT}" --state-dir "${STATE_DIR}"
    ;;
  health)
    if [[ -z "${HOST_PORT}" ]]; then
      echo "--port is required for health" >&2
      exit 2
    fi
    curl -fsS "http://127.0.0.1:${HOST_PORT}/health"
    echo
    curl -fsS "http://127.0.0.1:${HOST_PORT}/ready"
    echo
    curl -fsS "http://127.0.0.1:${HOST_PORT}/version"
    echo
    docker compose --env-file "${ENV_FILE}" -p "${PROJECT_NAME}" -f "${COMPOSE_FILE}" \
      exec -T challenge-prism python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=5).read().decode())"
    docker compose --env-file "${ENV_FILE}" -p "${PROJECT_NAME}" -f "${COMPOSE_FILE}" \
      exec -T challenge-prism python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8080/version', timeout=5).read().decode())"
    docker compose --env-file "${ENV_FILE}" -p "${PROJECT_NAME}" -f "${COMPOSE_FILE}" ps
    ;;
  down)
    TEARDOWN_ARGS=(--project-name "${PROJECT_NAME}" --compose-file "${COMPOSE_FILE}" --env-file "${ENV_FILE}" --state-dir "${STATE_DIR}")
    if [[ "${KEEP_DATA}" -eq 0 ]]; then
      TEARDOWN_ARGS+=(--destroy-data)
    fi
    bash "${TEARDOWN}" "${TEARDOWN_ARGS[@]}"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
