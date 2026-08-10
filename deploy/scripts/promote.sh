#!/usr/bin/env bash
# promote — digest pin promotion with mandatory backup (task 43).
#
#   promote.sh --env staging|prod --service validator --image REPO@sha256:HEX
#   promote.sh --env staging --service validator --digest sha256:HEX --repository validator
#   promote.sh --env prod --service validator --image ... --confirm-prod
#   promote.sh --rollback --env staging|prod [--service validator]
#
# Fail-closed:
#   - Digests only.
#   - --env staging NEVER writes deploy/pins/prod.json.
#   - --env prod requires --confirm-prod AND staging already has same digest
#     (unless --force-prod).
#   - Backup MUST succeed before pin write (unless --skip-backup for unit tests).
#   - Rollback = restore previous snapshot for that env.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=lib-promote.sh
source "${SCRIPT_DIR}/lib-promote.sh"

require_cmd python3
require_cmd sha256sum

ENV_NAME=""
SERVICE="validator"
IMAGE_REF=""
DIGEST_ONLY=""
REPOSITORY=""
CONFIRM_PROD=0
FORCE_PROD=0
SKIP_BACKUP=0
ROLLBACK=0
COMMIT_SHA="${BASE_PROMOTE_COMMIT:-}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="${2:-}"; shift 2 ;;
    --service) SERVICE="${2:-}"; shift 2 ;;
    --image) IMAGE_REF="${2:-}"; shift 2 ;;
    --digest) DIGEST_ONLY="${2:-}"; shift 2 ;;
    --repository) REPOSITORY="${2:-}"; shift 2 ;;
    --confirm-prod) CONFIRM_PROD=1; shift ;;
    --force-prod) FORCE_PROD=1; shift ;;
    --skip-backup) SKIP_BACKUP=1; shift ;;
    --rollback) ROLLBACK=1; shift ;;
    --commit) COMMIT_SHA="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) die "unknown arg: $1" ;;
  esac
done

[[ -n "$ENV_NAME" ]] || die "--env staging|prod required"
case "$ENV_NAME" in staging|prod) ;; *) die "env must be staging|prod" ;; esac
case "$SERVICE" in validator|gateway|updater|prism-challenge|design-challenge|bounty-challenge) ;; *) die "service must be validator|gateway|updater|prism-challenge|design-challenge|bounty-challenge" ;; esac

PIN_PATH="$(pin_path_for_env "$ROOT" "$ENV_NAME")"
PROD_PATH="$(prod_pin_path "$ROOT")"
[[ -f "$PIN_PATH" ]] || die "missing pin file: $PIN_PATH"
[[ -f "$PROD_PATH" ]] || die "missing prod pin file: $PROD_PATH"

PROD_HASH_BEFORE="$(sha256_file "$PROD_PATH")"
PIN_HASH_BEFORE="$(sha256_file "$PIN_PATH")"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/base-promote.XXXXXX")"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT
NEW_JSON_PATH="${WORK}/new-pin.json"
BACKUP_JSON_PATH="${WORK}/backup.json"
echo '{"ok":false,"skipped":true}' >"$BACKUP_JSON_PATH"

if [[ "$ROLLBACK" -eq 1 ]]; then
  python3 - "$PIN_PATH" "$SERVICE" "$NEW_JSON_PATH" <<'PY'
import json, sys
from datetime import datetime, timezone
path, service, out = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path, encoding="utf-8") as f:
    pin = json.load(f)
prev = pin.get("previous")
if not prev or "services" not in prev:
    raise SystemExit("no previous snapshot to roll back to")
old_services = pin.get("services", {})
pin["services"] = prev["services"]
pin["commit_sha"] = prev.get("commit_sha", pin.get("commit_sha"))
pin["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
pin["previous"] = {
    "services": old_services,
    "commit_sha": pin.get("commit_sha"),
    "updated_at": pin.get("updated_at"),
}
if service not in pin["services"]:
    raise SystemExit(f"service {service} missing in previous snapshot")
with open(out, "w", encoding="utf-8") as f:
    json.dump(pin, f, indent=2, sort_keys=True)
    f.write("\n")
PY
  DIGEST="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["services"][sys.argv[2]]["digest"])' "$NEW_JSON_PATH" "$SERVICE")"
  IMAGE_REF="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["services"][sys.argv[2]]["image"])' "$NEW_JSON_PATH" "$SERVICE")"
  log "rollback $ENV_NAME/$SERVICE → $DIGEST"
else
  if [[ -n "$IMAGE_REF" ]]; then
    is_digest_pinned_image "$IMAGE_REF" || die "image must be repo@sha256:<64-hex>"
    DIGEST="$(normalize_digest "$IMAGE_REF")"
    REPOSITORY="$(repo_from_image "$IMAGE_REF" "$REPOSITORY")"
  elif [[ -n "$DIGEST_ONLY" ]]; then
    DIGEST="$(normalize_digest "$DIGEST_ONLY")"
    [[ -n "$REPOSITORY" ]] || REPOSITORY="base-${SERVICE}"
    IMAGE_REF="${REPOSITORY}@${DIGEST}"
  else
    die "provide --image REPO@sha256:… or --digest sha256:… --repository REPO"
  fi

  if [[ "$ENV_NAME" == "prod" ]]; then
    [[ "$CONFIRM_PROD" -eq 1 || "$FORCE_PROD" -eq 1 ]] || \
      die "prod promote requires --confirm-prod (fail-closed)"
    if [[ "$FORCE_PROD" -eq 0 ]]; then
      STAGING_PATH="$(pin_path_for_env "$ROOT" staging)"
      STAGING_DIGEST="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["services"][sys.argv[2]]["digest"])' "$STAGING_PATH" "$SERVICE")"
      [[ "$STAGING_DIGEST" == "$DIGEST" ]] || \
        die "prod ladder: staging $SERVICE digest is $STAGING_DIGEST, not $DIGEST (promote staging first or --force-prod)"
    fi
  fi

  if [[ -z "$COMMIT_SHA" ]]; then
    COMMIT_SHA="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  fi

  DIGEST="$DIGEST" IMAGE_REF="$IMAGE_REF" REPOSITORY="$REPOSITORY" \
  SERVICE="$SERVICE" COMMIT_SHA="$COMMIT_SHA" ENV_NAME="$ENV_NAME" \
  python3 - "$PIN_PATH" "$NEW_JSON_PATH" <<'PY'
import json, os, sys
from datetime import datetime, timezone
path, out = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8") as f:
    pin = json.load(f)
service = os.environ["SERVICE"]
digest = os.environ["DIGEST"]
image = os.environ["IMAGE_REF"]
repo = os.environ["REPOSITORY"]
commit = os.environ["COMMIT_SHA"]
env_name = os.environ["ENV_NAME"]
pin["previous"] = {
    "services": json.loads(json.dumps(pin.get("services") or {})),
    "commit_sha": pin.get("commit_sha"),
    "updated_at": pin.get("updated_at"),
}
svcs = pin.setdefault("services", {})
svcs[service] = {"repository": repo, "image": image, "digest": digest}
pin["env"] = env_name
pin["commit_sha"] = commit
pin["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
with open(out, "w", encoding="utf-8") as f:
    json.dump(pin, f, indent=2, sort_keys=True)
    f.write("\n")
PY
fi

if [[ "$SKIP_BACKUP" -eq 0 ]]; then
  require_cmd pg_dump
  : "${PGHOST:?PGHOST required for backup (or --skip-backup)}"
  : "${PGUSER:?PGUSER required}"
  : "${PGPASSWORD:?PGPASSWORD required}"
  : "${PGDATABASE:?PGDATABASE required}"
  : "${BASE_BACKUP_ENDPOINT:?BASE_BACKUP_ENDPOINT required}"
  : "${AWS_ACCESS_KEY_ID:?required}"
  : "${AWS_SECRET_ACCESS_KEY:?required}"
  export BASE_BACKUP_ENV="$ENV_NAME"
  log "backup before promote ($ENV_NAME)"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo '{"ok":true,"dry_run":true}' >"$BACKUP_JSON_PATH"
  else
    "${SCRIPT_DIR}/pg-backup.sh" >"$BACKUP_JSON_PATH"
    python3 -c 'import json,sys; j=json.load(open(sys.argv[1])); assert j.get("ok") is True' "$BACKUP_JSON_PATH" \
      || die "backup failed — pin NOT updated (fail-closed)"
  fi
else
  log "WARNING: --skip-backup (tests only)"
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  log "dry-run: would write $PIN_PATH"
  cat "$NEW_JSON_PATH"
  if [[ "$ENV_NAME" == "staging" ]]; then
    [[ "$(sha256_file "$PROD_PATH")" == "$PROD_HASH_BEFORE" ]] || die "prod pin mutated during staging dry-run"
  fi
  exit 0
fi

atomic_write "$PIN_PATH" "$(cat "$NEW_JSON_PATH")"

DESIRED_ENV="${ROOT}/deploy/pins/${ENV_NAME}.desired.env"
DESIRED_IMAGE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["services"][sys.argv[2]]["image"])' "$NEW_JSON_PATH" "$SERVICE")"
atomic_write "$DESIRED_ENV" "BASE_UPDATER_DESIRED_IMAGE=${DESIRED_IMAGE}"

if [[ "$ENV_NAME" == "staging" ]]; then
  PROD_HASH_AFTER="$(sha256_file "$PROD_PATH")"
  [[ "$PROD_HASH_AFTER" == "$PROD_HASH_BEFORE" ]] || die "FAIL-CLOSED: prod pin changed during staging promote"
fi

if [[ "$ENV_NAME" == "prod" && "$ROLLBACK" -eq 0 ]]; then
  [[ "$(sha256_file "$PIN_PATH")" != "$PIN_HASH_BEFORE" ]] || die "prod pin unchanged after promote"
fi

PROD_HASH_AFTER="$(sha256_file "$PROD_PATH")"
python3 - "$NEW_JSON_PATH" "$BACKUP_JSON_PATH" "$ENV_NAME" "$SERVICE" "$PIN_PATH" "$DESIRED_ENV" \
  "$PROD_HASH_BEFORE" "$PROD_HASH_AFTER" "$ROLLBACK" <<'PY'
import json, sys
pin_path, backup_path, env_name, service, pin_file, desired_env, prod_b, prod_a, rb = sys.argv[1:10]
pin = json.load(open(pin_path, encoding="utf-8"))
backup = json.load(open(backup_path, encoding="utf-8"))
svc = pin["services"][service]
out = {
    "ok": True,
    "env": env_name,
    "service": service,
    "image": svc["image"],
    "digest": svc["digest"],
    "pin_path": pin_file,
    "desired_env": desired_env,
    "prod_pin_sha256_before": prod_b,
    "prod_pin_sha256_after": prod_a,
    "prod_untouched": prod_b == prod_a,
    "backup": backup,
    "rollback": rb == "1",
}
print(json.dumps(out, indent=2, sort_keys=True))
PY
log "promoted $ENV_NAME/$SERVICE → $DESIRED_IMAGE"
