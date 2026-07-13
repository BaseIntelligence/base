#!/usr/bin/env bash
# Safe master teardown: containers/networks always removed.
# Persistent volumes retained unless --destroy-data is set (VAL-COMPOSE-060/061).
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: teardown-master.sh --project-name NAME [--destroy-data] [--compose-file PATH]

Without --destroy-data, PostgreSQL and challenge volumes remain.
With --destroy-data, only this project's declared volumes are removed.
Never touches live Swarm or unrelated resources.
EOF
}

PROJECT_NAME="${COMPOSE_PROJECT_NAME:-}"
DESTROY=0
COMPOSE_FILE=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-name) PROJECT_NAME="$2"; shift 2 ;;
    --destroy-data) DESTROY=1; shift ;;
    --compose-file) COMPOSE_FILE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${PROJECT_NAME}" ]]; then
  echo "--project-name is required" >&2
  exit 2
fi
COMPOSE_FILE="${COMPOSE_FILE:-${SCRIPT_DIR}/docker-compose.yml}"

# Compose-only: this script invokes only `docker compose` for the named project.
# Historical residual: a dead post-opts argv walk after the flag parser never
# executed meaningfully; it is intentionally absent (no security-theater loop).

if [[ "${DESTROY}" -eq 1 ]]; then
  echo "destructive teardown for project ${PROJECT_NAME}"
  docker compose -p "${PROJECT_NAME}" -f "${COMPOSE_FILE}" down --volumes --remove-orphans
else
  echo "non-destructive teardown for project ${PROJECT_NAME} (volumes retained)"
  docker compose -p "${PROJECT_NAME}" -f "${COMPOSE_FILE}" down --remove-orphans
fi

echo "teardown complete"
