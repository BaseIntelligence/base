#!/usr/bin/env bash
# pg_dump → S3-compatible object store (DO Spaces or local MinIO).
# Credentials via env only — never committed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib-promote.sh
source "${SCRIPT_DIR}/lib-promote.sh"

require_cmd pg_dump
require_cmd psql
require_cmd aws
require_cmd gzip
require_cmd sha256sum
require_cmd python3

: "${PGHOST:?PGHOST required}"
: "${PGPORT:=5432}"
: "${PGUSER:?PGUSER required}"
: "${PGPASSWORD:?PGPASSWORD required}"
: "${PGDATABASE:?PGDATABASE required}"
: "${GBASE_BACKUP_ENDPOINT:?GBASE_BACKUP_ENDPOINT required (Spaces or MinIO URL)}"
: "${AWS_ACCESS_KEY_ID:?AWS_ACCESS_KEY_ID required}"
: "${AWS_SECRET_ACCESS_KEY:?AWS_SECRET_ACCESS_KEY required}"

ENV_SEG="${GBASE_BACKUP_ENV:-local}"
STAMP="$(utc_stamp)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/gbase-pg-backup.XXXXXX")"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

DUMP_SQL="${WORK}/gbase-${ENV_SEG}-${STAMP}.sql"
DUMP_GZ="${DUMP_SQL}.gz"
META="${WORK}/gbase-${ENV_SEG}-${STAMP}.meta.json"

log "pg_dump ${PGUSER}@${PGHOST}:${PGPORT}/${PGDATABASE}"
PGPASSWORD="$PGPASSWORD" pg_dump \
  --host="$PGHOST" \
  --port="$PGPORT" \
  --username="$PGUSER" \
  --dbname="$PGDATABASE" \
  --format=plain \
  --no-owner \
  --no-acl \
  --file="$DUMP_SQL"

ROW_JSON="$(
  export PGHOST PGPORT PGUSER PGPASSWORD PGDATABASE
  python3 - <<'PY'
import json, os, subprocess
env = {**os.environ, "PGPASSWORD": os.environ["PGPASSWORD"]}
base = [
    "psql",
    "--host", os.environ["PGHOST"],
    "--port", os.environ.get("PGPORT", "5432"),
    "--username", os.environ["PGUSER"],
    "--dbname", os.environ["PGDATABASE"],
    "-At",
]
tables = subprocess.check_output(
    base + ["-c", "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY 1;"],
    env=env, text=True,
).splitlines()
counts = {}
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
    counts[t] = int(n)
print(json.dumps(counts, separators=(",", ":")))
PY
)"

gzip -n -c "$DUMP_SQL" >"$DUMP_GZ"
SUM="$(sha256_file "$DUMP_GZ")"
BYTES="$(wc -c <"$DUMP_GZ" | tr -d ' ')"

python3 - "$META" "$ENV_SEG" "$PGDATABASE" "$PGHOST" "$SUM" "$BYTES" "$ROW_JSON" <<'PY'
import json, sys
from datetime import datetime, timezone
path, env_seg, db, host, digest, nbytes, rows = sys.argv[1:8]
meta = {
    "env": env_seg,
    "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "database": db,
    "host": host,
    "dump_sha256": digest,
    "dump_bytes": int(nbytes),
    "format": "plain-sql-gz",
    "row_counts": json.loads(rows),
    "row_counts_exact": json.loads(rows),
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, sort_keys=True)
    f.write("\n")
PY

s3_mb_if_needed
BUCKET="$(s3_bucket)"
PREFIX="$(s3_prefix)"
KEY_BASE="s3://${BUCKET}/${PREFIX}/${ENV_SEG}/${STAMP}"
s3_cp "$DUMP_GZ" "${KEY_BASE}.sql.gz"
s3_cp "$META" "${KEY_BASE}.meta.json"

if [[ -n "${GBASE_BACKUP_OUT:-}" ]]; then
  mkdir -p "$(dirname "$GBASE_BACKUP_OUT")"
  cp -f "$DUMP_GZ" "$GBASE_BACKUP_OUT"
  cp -f "$META" "${GBASE_BACKUP_OUT%.sql.gz}.meta.json"
fi

cat <<EOF
{
  "ok": true,
  "s3_uri": "${KEY_BASE}.sql.gz",
  "meta_uri": "${KEY_BASE}.meta.json",
  "dump_sha256": "$SUM",
  "dump_bytes": $BYTES,
  "created_at": "$(utc_now)",
  "env": "$ENV_SEG",
  "row_counts": $ROW_JSON
}
EOF
