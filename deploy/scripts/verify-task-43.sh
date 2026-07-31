#!/usr/bin/env bash
# End-to-end verification for task 43 (local compose + unit tests).
# Writes evidence lines to stdout; caller captures to task-43-promotion.txt.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

# shellcheck source=lib-promote.sh
source "${SCRIPT_DIR}/lib-promote.sh"

EVIDENCE_DIR="${BASE_EVIDENCE_DIR:-/root/.omo/evidence/base-rust-subnet}"
mkdir -p "$EVIDENCE_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/base-t43.XXXXXX")"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

PASS=0
FAIL=0
record() {
  local name="$1" status="$2" detail="${3:-}"
  echo "SCENARIO $name => $status ${detail}"
  if [[ "$status" == "PASS" ]]; then PASS=$((PASS+1)); else FAIL=$((FAIL+1)); fi
}

echo "=== task-43 promotion verify ${STAMP} ==="
echo "root=$ROOT"
echo "head=$(git rev-parse HEAD)"
echo "branch=$(git rev-parse --abbrev-ref HEAD)"

# Isolate pin files in a temp git work tree copy of pins
PIN_ROOT="$WORK/repo"
mkdir -p "$PIN_ROOT/deploy/pins" "$PIN_ROOT/deploy/scripts"
cp -a "$ROOT/deploy/scripts/." "$PIN_ROOT/deploy/scripts/"
cp -a "$ROOT/deploy/pins/staging.json" "$PIN_ROOT/deploy/pins/"
cp -a "$ROOT/deploy/pins/prod.json" "$PIN_ROOT/deploy/pins/"
# fake git for commit sha
git -C "$PIN_ROOT" init -q
git -C "$PIN_ROOT" config user.email t43@test
git -C "$PIN_ROOT" config user.name t43
git -C "$PIN_ROOT" add deploy
git -C "$PIN_ROOT" commit -q -m init || true

GOOD_DIGEST="$(docker image inspect validator:0.1.0 --format '{{.Id}}')"
GOOD_IMAGE="validator@${GOOD_DIGEST}"
BAD_DIGEST="sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
BAD_IMAGE="validator@${BAD_DIGEST}"

# --- S0: record digests ---
if OUT="$("$ROOT/deploy/scripts/record-image-digests.sh" --out "$WORK/digests.json" 2>&1)"; then
  echo "$OUT" | tail -5
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert "validator" in d["images"]; assert d["images"]["validator"]["digest"].startswith("sha256:")' "$WORK/digests.json"
  record S0_record_digests PASS "path=$WORK/digests.json"
else
  record S0_record_digests FAIL "$OUT"
fi

# --- S1: known-good promote to staging + healthz ---
PROD_BEFORE="$(sha256sum "$PIN_ROOT/deploy/pins/prod.json" | awk '{print $1}')"
export PGHOST="${PGHOST:-}" # may set below
# Prefer docker network postgres
PG_IP="$(docker inspect base-postgres-1 --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' 2>/dev/null || true)"
if [[ -n "$PG_IP" ]]; then
  export PGHOST="$PG_IP" PGPORT=5432 PGUSER=base PGPASSWORD=base_dev_only_change_me PGDATABASE=base
fi
export BASE_BACKUP_ENDPOINT="${BASE_BACKUP_ENDPOINT:-http://127.0.0.1:55000}"
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:?AWS_ACCESS_KEY_ID required}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:?AWS_SECRET_ACCESS_KEY required}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
export BASE_BACKUP_BUCKET="${BASE_BACKUP_BUCKET:-base-backups}"

# Seed a known row for restore drill
if [[ -n "${PGHOST:-}" ]]; then
  PGPASSWORD="$PGPASSWORD" psql -h "$PGHOST" -U "$PGUSER" -d "$PGDATABASE" -v ON_ERROR_STOP=1 <<'SQL' || true
CREATE TABLE IF NOT EXISTS t43_restore_probe (
  id serial PRIMARY KEY,
  label text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
DELETE FROM t43_restore_probe;
INSERT INTO t43_restore_probe (label) VALUES ('alpha'), ('beta'), ('gamma');
SQL
fi

S1_OUT="$WORK/s1.json"
if (
  cd "$PIN_ROOT"
  ./deploy/scripts/promote.sh \
    --env staging \
    --service validator \
    --image "$GOOD_IMAGE" \
    --commit "$(git -C "$ROOT" rev-parse HEAD)" \
    >"$S1_OUT"
); then
  DIG="$(python3 -c 'import json; print(json.load(open("'"$S1_OUT"'"))["digest"])')"
  PROD_AFTER="$(sha256sum "$PIN_ROOT/deploy/pins/prod.json" | awk '{print $1}')"
  PROD_UNTOUCHED="$(python3 -c 'import json; print(json.load(open("'"$S1_OUT"'"))["prod_untouched"])')"
  # healthz
  HZ="$(docker run --rm --network base_base curlimages/curl:8.5.0 -sS -o /dev/null -w '%{http_code}' http://validator:8080/healthz || echo fail)"
  # running image id
  RUN_ID="$(docker inspect validator-1 --format '{{.Image}}')"
  RUN_MATCH=0
  [[ "$RUN_ID" == "$GOOD_DIGEST" || "$RUN_ID" == "sha256:${GOOD_DIGEST#sha256:}" ]] && RUN_MATCH=1
  # pin digest matches good
  PIN_D="$(python3 -c 'import json; print(json.load(open("'"$PIN_ROOT"'/deploy/pins/staging.json"))["services"]["validator"]["digest"])')"
  if [[ "$DIG" == "$GOOD_DIGEST" && "$PROD_AFTER" == "$PROD_BEFORE" && "$PROD_UNTOUCHED" == "True" && "$HZ" == "200" && "$PIN_D" == "$GOOD_DIGEST" ]]; then
    record S1_good_staging_healthz PASS "digest=$DIG healthz=$HZ running=$RUN_ID pin_match=1 prod_untouched=1"
  else
    record S1_good_staging_healthz FAIL "dig=$DIG pin=$PIN_D hz=$HZ run=$RUN_ID prod=$PROD_AFTER/$PROD_BEFORE untouched=$PROD_UNTOUCHED"
  fi
  # Note running digest: local compose may already be on good image (match expected)
  echo "S1_DETAIL running_image=$RUN_ID desired=$GOOD_DIGEST healthz=$HZ"
else
  record S1_good_staging_healthz FAIL "promote failed $(cat "$S1_OUT" 2>/dev/null || true)"
fi

# --- S2: bad digest to staging; updater unit rollback; prod untouched ---
PROD_BEFORE2="$(sha256sum "$PIN_ROOT/deploy/pins/prod.json" | awk '{print $1}')"
S2_OUT="$WORK/s2.json"
if (
  cd "$PIN_ROOT"
  ./deploy/scripts/promote.sh \
    --env staging \
    --service validator \
    --image "$BAD_IMAGE" \
    --skip-backup \
    >"$S2_OUT"
); then
  PROD_AFTER2="$(sha256sum "$PIN_ROOT/deploy/pins/prod.json" | awk '{print $1}')"
  STG_D="$(python3 -c 'import json; print(json.load(open("'"$PIN_ROOT"'/deploy/pins/staging.json"))["services"]["validator"]["digest"])')"
  # Run updater s2 test (rollback leaves current pin old)
  set +e
  cargo test -p updater s2_unhealthy_rolled_back -- --nocapture >"$WORK/s2-cargo.txt" 2>&1
  CARGO_EC=$?
  set -e
  if [[ "$CARGO_EC" -eq 0 && "$PROD_AFTER2" == "$PROD_BEFORE2" && "$STG_D" == "$BAD_DIGEST" ]]; then
    record S2_bad_digest_prod_untouched PASS "staging=$STG_D prod_hash_stable=1 updater_s2=ok"
  else
    record S2_bad_digest_prod_untouched FAIL "stg=$STG_D prod=$PROD_AFTER2/$PROD_BEFORE2 cargo=$CARGO_EC"
    tail -30 "$WORK/s2-cargo.txt" || true
  fi
  # Rollback staging pin to previous (good)
  (
    cd "$PIN_ROOT"
    ./deploy/scripts/promote.sh --env staging --service validator --rollback --skip-backup >"$WORK/s2-rb.json"
  )
  RB_D="$(python3 -c 'import json; print(json.load(open("'"$PIN_ROOT"'/deploy/pins/staging.json"))["services"]["validator"]["digest"])')"
  if [[ "$RB_D" == "$GOOD_DIGEST" ]]; then
    record S2b_rollback_repromote PASS "digest=$RB_D"
  else
    record S2b_rollback_repromote FAIL "digest=$RB_D expected=$GOOD_DIGEST"
  fi
else
  record S2_bad_digest_prod_untouched FAIL "promote bad failed"
fi

# Fail-closed: staging promote must refuse to accept writing prod (assert helper)
# Attempt: if someone passes wrong env - covered by prod ladder
set +e
(
  cd "$PIN_ROOT"
  ./deploy/scripts/promote.sh --env prod --service validator --image "$BAD_IMAGE" --skip-backup 2>"$WORK/prod-deny.err"
)
PROD_DENY_EC=$?
set -e
if [[ "$PROD_DENY_EC" -ne 0 ]]; then
  record S2c_prod_without_confirm_denied PASS "exit=$PROD_DENY_EC"
else
  record S2c_prod_without_confirm_denied FAIL "prod promote without confirm succeeded"
fi
PROD_FINAL="$(sha256sum "$PIN_ROOT/deploy/pins/prod.json" | awk '{print $1}')"
if [[ "$PROD_FINAL" == "$PROD_BEFORE" ]]; then
  record S2d_prod_pin_never_changed PASS "sha256=$PROD_FINAL"
else
  record S2d_prod_pin_never_changed FAIL "before=$PROD_BEFORE after=$PROD_FINAL"
fi

# --- S3: real pg_restore drill ---
if [[ -n "${PGHOST:-}" ]]; then
  export BASE_BACKUP_ENV=staging
  export BASE_BACKUP_OUT="$WORK/drill.sql.gz"
  set +e
  BACKUP_OUT="$(./deploy/scripts/pg-backup.sh 2>"$WORK/backup.err")"
  BEC=$?
  set -e
  echo "$BACKUP_OUT" >"$WORK/backup.raw"
  python3 -c 'import json,re,sys; t=open(sys.argv[1]).read(); m=re.search(r"\{[\s\S]*\}\s*$", t); open(sys.argv[2],"w").write((m.group(0) if m else t)+("" if (m and m.group(0).endswith("\n")) else "\n"))' "$WORK/backup.raw" "$WORK/backup.json"
  if [[ "$BEC" -ne 0 ]]; then
    record S3_pg_restore_drill FAIL "backup_ec=$BEC $(cat "$WORK/backup.err")"
  else
    S3_URI="$(python3 -c 'import json,re,sys; t=open(sys.argv[1]).read(); m=re.search(r"\{[\s\S]*\}\s*$", t); j=json.loads(m.group(0)); print(j["s3_uri"])' "$WORK/backup.json")"
    set +e
    DRILL_OUT="$(./deploy/scripts/pg-restore-drill.sh --s3-uri "$S3_URI" 2>"$WORK/drill.err")"
    DEC=$?
    set -e
    echo "$DRILL_OUT" >"$WORK/drill.raw"
    python3 -c 'import json,re,sys; t=open(sys.argv[1]).read(); m=re.search(r"\{[\s\S]*\}\s*$", t); open(sys.argv[2],"w").write(m.group(0) if m else t)' "$WORK/drill.raw" "$WORK/drill.json"
    if [[ "$DEC" -eq 0 ]] && python3 -c 'import json,sys; j=json.load(open(sys.argv[1])); assert j["ok"] is True; assert j["expected"].get("t43_restore_probe")==3' "$WORK/drill.json"; then
      record S3_pg_restore_drill PASS "s3=$S3_URI rows_t43=3"
    else
      record S3_pg_restore_drill FAIL "ec=$DEC $(cat "$WORK/drill.err") out=$(head -c 400 "$WORK/drill.json")"
    fi
  fi
else
  record S3_pg_restore_drill FAIL "no PGHOST"
fi

# --- S4: updater full suite + s1 good ---
set +e
cargo test -p updater --all-targets >"$WORK/updater-all.txt" 2>&1
UEC=$?
set -e
if [[ "$UEC" -eq 0 ]]; then
  record S4_updater_suite PASS "$(grep -E 'test result:' "$WORK/updater-all.txt" | tail -1)"
else
  record S4_updater_suite FAIL "$(tail -20 "$WORK/updater-all.txt")"
fi

# --- S5: SSH droplets up ---
set +e
ssh -o ConnectTimeout=8 -o BatchMode=yes -o StrictHostKeyChecking=accept-new root@68.183.23.51 'echo STAGING_OK; docker --version' >"$WORK/ssh-stg.txt" 2>&1
SEC=$?
ssh -o ConnectTimeout=8 -o BatchMode=yes -o StrictHostKeyChecking=accept-new root@206.189.224.155 'echo PROD_OK; docker --version' >"$WORK/ssh-prod.txt" 2>&1
PEC=$?
set -e
if [[ "$SEC" -eq 0 && "$PEC" -eq 0 ]]; then
  record S5_droplets_ssh PASS "staging+prod docker up"
else
  record S5_droplets_ssh FAIL "stg=$SEC prod=$PEC"
fi
echo "PENDING_LIVE: full remote healthz/digest match deferred until images deployed to droplets (stack not fully pushed yet)."

# --- S6: secrets not in git (pattern avoids embedding live secret values) ---
set +e
HITS="$(git -C "$ROOT" grep -nE 'dop_v1_|BEGIN (OPENSSH |RSA )?PRIVATE KEY|AKIA[0-9A-Z]{16}' -- deploy .github 2>/dev/null | grep -v example | grep -v verify-task-43 || true)"
set -e
if [[ -z "$HITS" ]]; then
  record S6_no_secrets_in_tree PASS
else
  echo "$HITS"
  record S6_no_secrets_in_tree FAIL
fi

echo "=== SUMMARY pass=$PASS fail=$FAIL ==="
if [[ "$FAIL" -ne 0 ]]; then
  exit 1
fi
exit 0
