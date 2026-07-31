#!/usr/bin/env bash
# Real restore drill: load dump into scratch DB; exact row-count match vs meta.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib-promote.sh
source "${SCRIPT_DIR}/lib-promote.sh"

require_cmd psql
require_cmd gunzip
require_cmd python3

DUMP=""
META=""
S3_URI=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dump) DUMP="${2:-}"; shift 2 ;;
    --meta) META="${2:-}"; shift 2 ;;
    --s3-uri) S3_URI="${2:-}"; shift 2 ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) die "unknown arg: $1" ;;
  esac
done

: "${PGHOST:?PGHOST required}"
: "${PGPORT:=5432}"
: "${PGUSER:?PGUSER required}"
: "${PGPASSWORD:?PGPASSWORD required}"
export BASE_RESTORE_DB="${BASE_RESTORE_DB:-base_restore_drill}"
SCRATCH_DB="$BASE_RESTORE_DB"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/base-restore-drill.XXXXXX")"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

if [[ -n "$S3_URI" ]]; then
  require_cmd aws
  : "${BASE_BACKUP_ENDPOINT:?BASE_BACKUP_ENDPOINT required for --s3-uri}"
  : "${AWS_ACCESS_KEY_ID:?required}"
  : "${AWS_SECRET_ACCESS_KEY:?required}"
  DUMP="${WORK}/dump.sql.gz"
  META="${WORK}/dump.meta.json"
  s3_cp "$S3_URI" "$DUMP"
  s3_cp "${S3_URI%.sql.gz}.meta.json" "$META"
fi

[[ -n "$DUMP" && -f "$DUMP" ]] || die "--dump file required (or --s3-uri)"
[[ -n "$META" && -f "$META" ]] || die "--meta file required"

SQL="${WORK}/dump.sql"
gunzip -c "$DUMP" >"$SQL"

log "recreate scratch database ${SCRATCH_DB}"
PGPASSWORD="$PGPASSWORD" psql \
  --host="$PGHOST" --port="$PGPORT" --username="$PGUSER" --dbname=postgres \
  -v ON_ERROR_STOP=1 -q \
  -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '${SCRATCH_DB}' AND pid <> pg_backend_pid();" \
  >/dev/null 2>&1 || true

PGPASSWORD="$PGPASSWORD" psql \
  --host="$PGHOST" --port="$PGPORT" --username="$PGUSER" --dbname=postgres \
  -v ON_ERROR_STOP=1 -q \
  -c "DROP DATABASE IF EXISTS ${SCRATCH_DB};" \
  >/dev/null 2>&1

PGPASSWORD="$PGPASSWORD" psql \
  --host="$PGHOST" --port="$PGPORT" --username="$PGUSER" --dbname=postgres \
  -v ON_ERROR_STOP=1 -q \
  -c "CREATE DATABASE ${SCRATCH_DB};" \
  >/dev/null 2>&1

log "restore into ${SCRATCH_DB}"
PGPASSWORD="$PGPASSWORD" psql \
  --host="$PGHOST" --port="$PGPORT" --username="$PGUSER" --dbname="$SCRATCH_DB" \
  -v ON_ERROR_STOP=1 -q \
  -f "$SQL" >/dev/null 2>"$WORK/restore.psql.err"

export PGHOST PGPORT PGUSER PGPASSWORD
python3 - "$META" <<'PY'
import json, os, subprocess, sys

meta_path = sys.argv[1]
with open(meta_path, encoding="utf-8") as f:
    meta = json.load(f)
expected = meta.get("row_counts_exact") or meta.get("row_counts") or {}

env = {**os.environ, "PGPASSWORD": os.environ["PGPASSWORD"]}
db = os.environ.get("BASE_RESTORE_DB", "base_restore_drill")
base = [
    "psql",
    "--host", os.environ["PGHOST"],
    "--port", os.environ.get("PGPORT", "5432"),
    "--username", os.environ["PGUSER"],
    "--dbname", db,
    "-At", "-q",
]
tables = subprocess.check_output(
    base + ["-c", "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY 1;"],
    env=env, text=True,
).splitlines()
actual = {}
for t in tables:
    t = t.strip()
    if not t:
        continue
    if not t.replace("_", "").isalnum():
        raise SystemExit(f"unsafe table name: {t!r}")
    n = subprocess.check_output(
        base + ["-c", f'SELECT count(*) FROM public."{t}";'],
        env=env, text=True,
    ).strip()
    actual[t] = int(n)

missing = sorted(set(expected) - set(actual))
if missing:
    print(f"RESTORE_DRILL_FAIL missing_tables={missing}", file=sys.stderr)
    sys.exit(2)

bad = []
for t, exp in expected.items():
    got = actual.get(t)
    if got != int(exp):
        bad.append({"table": t, "expected": int(exp), "got": got})
if bad:
    print(f"RESTORE_DRILL_FAIL mismatches={bad}", file=sys.stderr)
    sys.exit(3)

out = {
    "ok": True,
    "scratch_db": db,
    "tables": actual,
    "expected": {k: int(v) for k, v in expected.items()},
    "total_rows_matched": sum(int(v) for v in expected.values()) if expected else sum(actual.values()),
}
print(json.dumps(out, indent=2, sort_keys=True))
print("RESTORE_DRILL_OK", file=sys.stderr)
PY
