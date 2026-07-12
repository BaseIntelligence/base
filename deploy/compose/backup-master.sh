#!/usr/bin/env bash
# Operator-safe control-plane backup for the master Compose project.
# Captures PostgreSQL schema/data plus watcher/registry state metadata needed to
# restore challenge registry, validators, assignments/results, raw snapshots,
# aggregation epochs/vectors, and watcher provenance (VAL-COMPOSE-065/CROSS-078).
#
# Never prints secret values. Backup artifacts are credential-free dumps.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: backup-master.sh --project-name NAME [--output-dir DIR] [--compose-file PATH]

Environment:
  COMPOSE_PROJECT_NAME   default project name when --project-name is omitted
  BASE_COMPOSE_STATE_DIR operator state root (for optional watcher/config copy)
EOF
}

PROJECT_NAME="${COMPOSE_PROJECT_NAME:-}"
OUTPUT_DIR=""
COMPOSE_FILE=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-name)
      PROJECT_NAME="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
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

if [[ -z "${PROJECT_NAME}" ]]; then
  echo "--project-name is required" >&2
  exit 2
fi
COMPOSE_FILE="${COMPOSE_FILE:-${SCRIPT_DIR}/docker-compose.yml}"
if [[ ! -f "${COMPOSE_FILE}" ]]; then
  echo "compose file missing: ${COMPOSE_FILE}" >&2
  exit 1
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${PWD}/base-backup-${PROJECT_NAME}-${STAMP}"
fi
umask 077
mkdir -p "${OUTPUT_DIR}/postgres" "${OUTPUT_DIR}/state" "${OUTPUT_DIR}/manifest"

# Prefer compose service exec; no password echoed.
if ! docker compose -p "${PROJECT_NAME}" -f "${COMPOSE_FILE}" ps --status running master-postgres >/dev/null 2>&1; then
  echo "master-postgres is not running in project ${PROJECT_NAME}" >&2
  exit 1
fi

PG_USER_DEFAULT="base"
PG_DB_DEFAULT="base"
DUMP_PATH="${OUTPUT_DIR}/postgres/base.dump"
echo "writing PostgreSQL custom dump to ${DUMP_PATH}"
docker compose -p "${PROJECT_NAME}" -f "${COMPOSE_FILE}" exec -T master-postgres \
  pg_dump -U "${PG_USER_DEFAULT}" -d "${PG_DB_DEFAULT}" -Fc --no-owner --no-privileges \
  >"${DUMP_PATH}"

# Optional host-side sealed config / watcher metadata (no secret values).
STATE_ROOT="${BASE_COMPOSE_STATE_DIR:-${XDG_STATE_HOME:-${HOME}/.local/state}/base-compose}/${PROJECT_NAME}}"
if [[ -d "${STATE_ROOT}/config" ]]; then
  # Copy only non-secret operator metadata names (paths, image pins). Skip secrets.
  if [[ -f "${STATE_ROOT}/config/compose.env" ]]; then
    # Substitute anything that looks like a path equality; values are paths only.
    grep -E '^(COMPOSE_PROJECT_NAME|BASE_MASTER_IMAGE_|PRISM_IMAGE_|POSTGRES_IMAGE_|BASE_MASTER_HOST_PORT|BASE_DOCKER_GID)=' \
      "${STATE_ROOT}/config/compose.env" >"${OUTPUT_DIR}/state/compose.pins.env" || true
  fi
fi

# Volume id inventory for recovery documentation (names only).
{
  echo "project=${PROJECT_NAME}"
  echo "timestamp=${STAMP}"
  docker volume ls --format '{{.Name}}' | grep -E "^${PROJECT_NAME}_" || true
} >"${OUTPUT_DIR}/manifest/volumes.txt"

# Digest of dump for restore verification (no credentials included).
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "${DUMP_PATH}" >"${OUTPUT_DIR}/manifest/postgres.dump.sha256"
else
  shasum -a 256 "${DUMP_PATH}" >"${OUTPUT_DIR}/manifest/postgres.dump.sha256"
fi

cat >"${OUTPUT_DIR}/manifest/backup.json" <<EOF
{
  "kind": "base-master-control-plane",
  "project": "${PROJECT_NAME}",
  "created_at": "${STAMP}",
  "postgres_dump": "postgres/base.dump",
  "format": "pg_dump -Fc",
  "includes": [
    "alembic_version",
    "challenges",
    "validators",
    "work_assignments",
    "work_results",
    "raw_weight_snapshots",
    "aggregation_epochs",
    "final_weight_vectors",
    "challenge_watcher_state"
  ],
  "excludes_secrets": true
}
EOF

echo "backup complete: ${OUTPUT_DIR}"
echo "manifest: ${OUTPUT_DIR}/manifest/backup.json"
