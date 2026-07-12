#!/usr/bin/env bash
# Restore a control-plane backup into a disposable master project.
# Leaves mutators non-ready until migrations/integrity succeed (VAL-COMPOSE-068).
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: restore-master.sh --project-name NAME --backup-dir DIR [--compose-file PATH]

Requires an already running postgres service for the target project (install first
or `docker compose up -d master-postgres --wait`). Does not print credentials.
EOF
}

PROJECT_NAME="${COMPOSE_PROJECT_NAME:-}"
BACKUP_DIR=""
COMPOSE_FILE=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-name)
      PROJECT_NAME="$2"
      shift 2
      ;;
    --backup-dir)
      BACKUP_DIR="$2"
      shift 2
      ;;
    --compose-file)
      COMPOSE_FILE="$2"
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

if [[ -z "${PROJECT_NAME}" || -z "${BACKUP_DIR}" ]]; then
  echo "--project-name and --backup-dir are required" >&2
  exit 2
fi
COMPOSE_FILE="${COMPOSE_FILE:-${SCRIPT_DIR}/docker-compose.yml}"
DUMP="${BACKUP_DIR}/postgres/base.dump"
if [[ ! -f "${DUMP}" ]]; then
  echo "backup dump missing: ${DUMP}" >&2
  exit 1
fi

if [[ -f "${BACKUP_DIR}/manifest/postgres.dump.sha256" ]]; then
  if command -v sha256sum >/dev/null 2>&1; then
    (cd "${BACKUP_DIR}" && sha256sum -c manifest/postgres.dump.sha256)
  fi
fi

echo "restoring PostgreSQL dump into project ${PROJECT_NAME}"
# Terminate other sessions then recreate DB for clean restore.
docker compose -p "${PROJECT_NAME}" -f "${COMPOSE_FILE}" exec -T master-postgres \
  psql -U base -d postgres -v ON_ERROR_STOP=1 <<'SQL'
SELECT pg_terminate_backend(pid) FROM pg_stat_activity
  WHERE datname = 'base' AND pid <> pg_backend_pid();
DROP DATABASE IF EXISTS base;
CREATE DATABASE base OWNER base;
SQL

docker compose -p "${PROJECT_NAME}" -f "${COMPOSE_FILE}" exec -T master-postgres \
  pg_restore -U base -d base --no-owner --no-privileges --clean --if-exists \
  <"${DUMP}"

echo "restore complete; start base-master-validator and confirm /ready"
