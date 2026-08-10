#!/usr/bin/env bash
# Encrypt compose env files with age for droplet delivery (R11).
# Never prints or embeds private keys. Ciphertext only on stdout paths.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: age-encrypt-env.sh --recipient AGE1... --src-dir DIR [--out-dir DIR] [--services LIST]

Encrypts <svc>.env (or falls back to <svc>.env.example) to <out>/<svc>.env.age
for postgres,validator,gateway,updater,design-challenge,prism-challenge,
design-egress-proxy,bounty-challenge by default.

Does not read or write age private keys.
EOF
}

RECIPIENT=""
SRC_DIR=""
OUT_DIR=""
SERVICES="postgres validator gateway updater design-challenge prism-challenge design-egress-proxy bounty-challenge"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --recipient) RECIPIENT="${2:-}"; shift 2 ;;
    --src-dir) SRC_DIR="${2:-}"; shift 2 ;;
    --out-dir) OUT_DIR="${2:-}"; shift 2 ;;
    --services) SERVICES="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$RECIPIENT" || -z "$SRC_DIR" ]]; then
  usage >&2
  exit 2
fi
if [[ ! "$RECIPIENT" =~ ^age1 ]]; then
  echo "recipient must be an age1... public key" >&2
  exit 2
fi
if [[ ! -d "$SRC_DIR" ]]; then
  echo "src-dir not a directory: $SRC_DIR" >&2
  exit 1
fi

OUT_DIR="${OUT_DIR:-$SRC_DIR}"
mkdir -p "$OUT_DIR"
umask 077

if ! command -v age >/dev/null 2>&1; then
  echo "age binary required" >&2
  exit 1
fi

for svc in $SERVICES; do
  plain=""
  if [[ -f "$SRC_DIR/${svc}.env" ]]; then
    plain="$SRC_DIR/${svc}.env"
  elif [[ -f "$SRC_DIR/${svc}.env.example" ]]; then
    plain="$SRC_DIR/${svc}.env.example"
  else
    echo "missing ${svc}.env or ${svc}.env.example under $SRC_DIR" >&2
    exit 1
  fi
  out="$OUT_DIR/${svc}.env.age"
  age -r "$RECIPIENT" -o "$out" "$plain"
  chmod 600 "$out"
  echo "encrypted $plain -> $out"
done

echo "OK: ciphertext under $OUT_DIR"