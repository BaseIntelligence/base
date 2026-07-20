#!/usr/bin/env bash
# Consistent backup of a challenge (Prism) volume /data SQLite database.
# Preserves submissions, scores, proofs, nonces, and raw-weight push cursor
# without master PostgreSQL credentials (VAL-COMPOSE-066).
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: backup-challenge.sh --project-name NAME [--service base-master-validator] [--output-dir DIR]

Default service is base-master-validator (embedded challenges live under
/var/lib/base/challenges/prism on the master volume). Historical
--service challenge-prism remains accepted for emergency dual-run stacks.
EOF
}

PROJECT_NAME="${COMPOSE_PROJECT_NAME:-}"
# Embedded topology: challenge SQLite is under the master container volume.
SERVICE="base-master-validator"
OUTPUT_DIR=""
COMPOSE_FILE=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-name) PROJECT_NAME="$2"; shift 2 ;;
    --service) SERVICE="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
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
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="${OUTPUT_DIR:-${PWD}/challenge-backup-${PROJECT_NAME}-${SERVICE}-${STAMP}}"
umask 077
mkdir -p "${OUTPUT_DIR}/data" "${OUTPUT_DIR}/manifest"

# Online consistent SQLite backup inside the challenge (or master embed) container.
docker compose -p "${PROJECT_NAME}" -f "${COMPOSE_FILE}" exec -T "${SERVICE}" \
  python - <<'PY'
from pathlib import Path
import sqlite3
import sys

candidates: list[Path] = []
for path in (
    Path("/var/lib/base/challenges/prism/prism.sqlite3"),
    Path("/var/lib/base/challenges/prism/challenge.sqlite3"),
    Path("/data/prism.sqlite3"),
    Path("/data/challenge.sqlite3"),
):
    if path.is_file():
        candidates.append(path)
if not candidates:
    for root in (Path("/var/lib/base/challenges/prism"), Path("/data")):
        if root.is_dir():
            candidates.extend(sorted(root.glob("*.sqlite3")))
if not candidates:
    print(
        "no sqlite database under embed path or /data",
        file=sys.stderr,
    )
    sys.exit(1)
src = candidates[0]
dst = Path("/tmp/challenge-backup.sqlite3")
if dst.exists():
    dst.unlink()
with sqlite3.connect(f"file:{src}?mode=ro", uri=True) as conn:
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    with sqlite3.connect(dst) as out:
        conn.backup(out)
print(dst)
PY

docker compose -p "${PROJECT_NAME}" -f "${COMPOSE_FILE}" cp \
  "${SERVICE}:/tmp/challenge-backup.sqlite3" \
  "${OUTPUT_DIR}/data/challenge.sqlite3"

docker compose -p "${PROJECT_NAME}" -f "${COMPOSE_FILE}" exec -T "${SERVICE}" \
  rm -f /tmp/challenge-backup.sqlite3 || true

if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "${OUTPUT_DIR}/data/challenge.sqlite3" \
    >"${OUTPUT_DIR}/manifest/challenge.sqlite3.sha256"
fi

cat >"${OUTPUT_DIR}/manifest/backup.json" <<EOF
{
  "kind": "challenge-volume",
  "project": "${PROJECT_NAME}",
  "service": "${SERVICE}",
  "created_at": "${STAMP}",
  "sqlite": "data/challenge.sqlite3",
  "excludes_master_credentials": true
}
EOF

echo "challenge backup complete: ${OUTPUT_DIR}"
