#!/usr/bin/env bash
# Shared helpers for base promotion / backup (task 43).
# shellcheck disable=SC2034
set -euo pipefail

DIGEST_RE='^sha256:[0-9a-f]{64}$'

die() {
  echo "promote: ERROR: $*" >&2
  exit 1
}

log() {
  echo "promote: $*" >&2
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

# Normalize image ref → digest (sha256:…). Accepts repo@sha256:… or bare sha256:….
normalize_digest() {
  local ref="$1"
  local d
  if [[ "$ref" == *@sha256:* ]]; then
    d="${ref##*@}"
  elif [[ "$ref" == sha256:* ]]; then
    d="$ref"
  else
    die "not a digest-pinned ref (need repo@sha256:<64-hex> or sha256:<64-hex>): $ref"
  fi
  d="$(printf '%s' "$d" | tr '[:upper:]' '[:lower:]')"
  [[ "$d" =~ $DIGEST_RE ]] || die "invalid digest: $d"
  printf '%s' "$d"
}

# Extract repository from repo@sha256:… (or default service name).
repo_from_image() {
  local ref="$1"
  local default_repo="${2:-}"
  if [[ "$ref" == *@sha256:* ]]; then
    printf '%s' "${ref%@*}"
  elif [[ -n "$default_repo" ]]; then
    printf '%s' "$default_repo"
  else
    die "cannot derive repository from bare digest without --repository"
  fi
}

is_digest_pinned_image() {
  local ref="$1"
  [[ "$ref" == *@sha256:* ]] || return 1
  local d
  d="$(normalize_digest "$ref")"
  [[ "$d" =~ $DIGEST_RE ]]
}

# Atomic JSON write via temp + rename.
atomic_write() {
  local dest="$1"
  local content="$2"
  local dir tmp
  dir="$(dirname "$dest")"
  mkdir -p "$dir"
  tmp="$(mktemp "${dir}/.pin.XXXXXX")"
  printf '%s\n' "$content" >"$tmp"
  chmod 0644 "$tmp"
  mv -f "$tmp" "$dest"
}

pin_path_for_env() {
  local root="$1"
  local env="$2"
  case "$env" in
    staging|prod) printf '%s/deploy/pins/%s.json' "$root" "$env" ;;
    *) die "env must be staging|prod (got: $env)" ;;
  esac
}

# Fail-closed: path that must never be written when env=staging.
prod_pin_path() {
  local root="$1"
  printf '%s/deploy/pins/prod.json' "$root"
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

# S3/Spaces endpoint helpers. Prefer real DO Spaces when configured;
# fall back to local S3-compatible (MinIO) via BASE_BACKUP_*.
s3_endpoint() {
  printf '%s' "${BASE_BACKUP_ENDPOINT:-${AWS_ENDPOINT_URL:-}}"
}

s3_bucket() {
  printf '%s' "${BASE_BACKUP_BUCKET:-base-backups}"
}

s3_prefix() {
  printf '%s' "${BASE_BACKUP_PREFIX:-pg}"
}

# aws s3 cp wrapper with optional custom endpoint.
s3_cp() {
  local src="$1"
  local dst="$2"
  local ep
  ep="$(s3_endpoint)"
  if [[ -n "$ep" ]]; then
    aws --endpoint-url "$ep" s3 cp "$src" "$dst" --only-show-errors
  else
    aws s3 cp "$src" "$dst" --only-show-errors
  fi
}

s3_ls() {
  local uri="$1"
  local ep
  ep="$(s3_endpoint)"
  if [[ -n "$ep" ]]; then
    aws --endpoint-url "$ep" s3 ls "$uri"
  else
    aws s3 ls "$uri"
  fi
}

s3_mb_if_needed() {
  local bucket
  bucket="$(s3_bucket)"
  local ep
  ep="$(s3_endpoint)"
  if [[ -n "$ep" ]]; then
    aws --endpoint-url "$ep" s3 mb "s3://${bucket}" 2>/dev/null || true
  else
    aws s3 mb "s3://${bucket}" 2>/dev/null || true
  fi
}

utc_now() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

utc_stamp() {
  date -u +%Y%m%dT%H%M%SZ
}
