#!/usr/bin/env bash
# Materialize compose env files at mode 0600.
#
# Modes:
#   1) From committed *.env.example (local/dev verify) — default
#   2) From age-encrypted files when AGE_IDENTITY + *.env.age are present
#
# Secrets never enter images or cloud-init; operators decrypt on the box (R11).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_DIR="${BASE_SECRETS_DIR:-$ROOT/deploy/env}"
AGE_IDENTITY="${AGE_IDENTITY:-${BASE_AGE_IDENTITY:-}}"

mkdir -p "$ENV_DIR"
umask 077

materialize_one() {
  local name="$1"
  local example="$ENV_DIR/${name}.env.example"
  local out="$ENV_DIR/${name}.env"
  local age_file="$ENV_DIR/${name}.env.age"

  if [[ -n "$AGE_IDENTITY" && -f "$age_file" && -f "$AGE_IDENTITY" ]]; then
    age -d -i "$AGE_IDENTITY" -o "$out" "$age_file"
    chmod 0600 "$out"
    echo "decrypted $age_file -> $out (0600)"
    return
  fi

  if [[ ! -f "$example" ]]; then
    echo "missing example: $example" >&2
    exit 1
  fi
  cp "$example" "$out"
  chmod 0600 "$out"
  echo "materialized $out from example (0600)"
}

# Challenge env files carry BASE_DATABASE_URL (must match postgres.env).
# Without them, compose refuses to start design/prism (no silent memory store).
for svc in postgres validator gateway updater design-challenge prism-challenge design-egress-proxy; do
  materialize_one "$svc"
done

echo "OK: env files ready under $ENV_DIR"
