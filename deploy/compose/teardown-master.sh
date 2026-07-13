#!/usr/bin/env bash
# Safe master teardown: containers/networks always removed.
# Persistent volumes retained unless --destroy-data is set (VAL-COMPOSE-060/061).
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: teardown-master.sh --project-name NAME [--destroy-data]
                          [--compose-file PATH] [--env-file PATH] [--state-dir DIR]

Without --destroy-data, PostgreSQL and challenge volumes remain.
With --destroy-data, only this project's declared volumes are removed.
Never touches live Swarm or unrelated resources.

Image pins and secret-file paths for docker-compose.yml are loaded from:
  1. --env-file PATH (install-master sealed compose.env preferred)
  2. BASE_COMPOSE_ENV_FILE when set
  3. --state-dir DIR/config/compose.env or XDG state base-compose/<project>/config/compose.env
EOF
}

PROJECT_NAME="${COMPOSE_PROJECT_NAME:-}"
DESTROY=0
COMPOSE_FILE=""
ENV_FILE="${BASE_COMPOSE_ENV_FILE:-}"
STATE_DIR="${BASE_COMPOSE_STATE_DIR:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-name) PROJECT_NAME="$2"; shift 2 ;;
    --destroy-data) DESTROY=1; shift ;;
    --compose-file) COMPOSE_FILE="$2"; shift 2 ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --state-dir) STATE_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${PROJECT_NAME}" ]]; then
  echo "--project-name is required" >&2
  exit 2
fi
COMPOSE_FILE="${COMPOSE_FILE:-${SCRIPT_DIR}/docker-compose.yml}"

if [[ -z "${ENV_FILE}" ]]; then
  if [[ -n "${STATE_DIR}" && -f "${STATE_DIR}/config/compose.env" ]]; then
    ENV_FILE="${STATE_DIR}/config/compose.env"
  else
    DEFAULT_STATE="${XDG_STATE_HOME:-${HOME}/.local/state}/base-compose/${PROJECT_NAME}/config/compose.env"
    if [[ -f "${DEFAULT_STATE}" ]]; then
      ENV_FILE="${DEFAULT_STATE}"
    fi
  fi
fi

# Compose-only: this script invokes only `docker compose` for the named project.
# Historical residual: a dead post-opts argv walk after the flag parser never
# executed meaningfully; it is intentionally absent (no security-theater loop).
# Prefer the install-master sealed compose.env so image digests and secret paths
# interpolate; fallback bare `docker compose -p` still scopes to PROJECT_NAME only.

if [[ -n "${ENV_FILE}" ]]; then
  if [[ ! -f "${ENV_FILE}" ]]; then
    echo "compose env file not found: ${ENV_FILE}" >&2
    exit 2
  fi
  echo "using sealed compose env: ${ENV_FILE}"
  if [[ "${DESTROY}" -eq 1 ]]; then
    echo "destructive teardown for project ${PROJECT_NAME}"
    docker compose --env-file "${ENV_FILE}" -p "${PROJECT_NAME}" -f "${COMPOSE_FILE}" \
      down --volumes --remove-orphans
  else
    echo "non-destructive teardown for project ${PROJECT_NAME} (volumes retained)"
    docker compose --env-file "${ENV_FILE}" -p "${PROJECT_NAME}" -f "${COMPOSE_FILE}" \
      down --remove-orphans
  fi
else
  if [[ "${DESTROY}" -eq 1 ]]; then
    echo "destructive teardown for project ${PROJECT_NAME}"
    docker compose -p "${PROJECT_NAME}" -f "${COMPOSE_FILE}" down --volumes --remove-orphans
  else
    echo "non-destructive teardown for project ${PROJECT_NAME} (volumes retained)"
    docker compose -p "${PROJECT_NAME}" -f "${COMPOSE_FILE}" down --remove-orphans
  fi
fi

echo "teardown complete"
