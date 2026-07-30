#!/usr/bin/env bash
# Push age-encrypted env files to a droplet and optionally materialize.
# Identity must already be on the box at AGE_IDENTITY (OOB). No private keys
# are read by this script unless --identity is passed for a local dry-run.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: age-push-env.sh --host user@ip --age-dir DIR [--remote-dir PATH] [--materialize]

Copies *.env.age to the droplet under remote-dir (default /opt/gbase/deploy/env).
With --materialize, runs materialize-env.sh on the remote host using
AGE_IDENTITY=/etc/gbase/age-identity.txt (must already exist, mode 600).
EOF
}

HOST=""
AGE_DIR=""
REMOTE_DIR="/opt/gbase/deploy/env"
MATERIALIZE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="${2:-}"; shift 2 ;;
    --age-dir) AGE_DIR="${2:-}"; shift 2 ;;
    --remote-dir) REMOTE_DIR="${2:-}"; shift 2 ;;
    --materialize) MATERIALIZE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$HOST" || -z "$AGE_DIR" ]]; then
  usage >&2
  exit 2
fi
if [[ ! -d "$AGE_DIR" ]]; then
  echo "age-dir not a directory: $AGE_DIR" >&2
  exit 1
fi

shopt -s nullglob
files=("$AGE_DIR"/*.env.age)
if [[ ${#files[@]} -eq 0 ]]; then
  echo "no *.env.age in $AGE_DIR" >&2
  exit 1
fi

ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$HOST" \
  "mkdir -p '$REMOTE_DIR' && chmod 700 '$REMOTE_DIR'"

scp -o BatchMode=yes "${files[@]}" "${HOST}:${REMOTE_DIR}/"
echo "copied ${#files[@]} age file(s) -> ${HOST}:${REMOTE_DIR}/"

if [[ "$MATERIALIZE" -eq 1 ]]; then
  ssh -o BatchMode=yes "$HOST" bash -s <<'REMOTE'
set -euo pipefail
export AGE_IDENTITY="${AGE_IDENTITY:-/etc/gbase/age-identity.txt}"
if [[ ! -f "$AGE_IDENTITY" ]]; then
  echo "missing identity on host: $AGE_IDENTITY (deliver OOB first)" >&2
  exit 1
fi
if [[ -x /opt/gbase/deploy/scripts/materialize-env.sh ]]; then
  GBASE_SECRETS_DIR=/opt/gbase/deploy/env \
    /opt/gbase/deploy/scripts/materialize-env.sh
elif [[ -x ./deploy/scripts/materialize-env.sh ]]; then
  ./deploy/scripts/materialize-env.sh
else
  echo "materialize-env.sh not found on host; ciphertext is in place" >&2
  exit 1
fi
REMOTE
fi

echo "OK: push complete"