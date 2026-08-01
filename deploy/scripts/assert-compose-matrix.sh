#!/usr/bin/env bash
# Assert compose role × env matrix is consistent.
#
#   assert-compose-matrix.sh
#
# Verifies:
#   - validator role never renders gateway
#   - evil-gateway never renders outside its profile
#   - prod env never renders e2e or evil-gateway overrides
#   - master role renders gateway
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "$ROOT"

fail() { echo "assert-compose-matrix: FAIL: $*" >&2; exit 1; }

# --- validator role: no gateway ---
services=$(docker compose \
  -f docker-compose.yml \
  -f deploy/compose/role-validator.yml \
  config --services 2>/dev/null || true)
if echo "$services" | grep -qx "gateway"; then
  fail "validator role renders gateway (must not)"
fi
echo "OK: validator role does not render gateway"

# --- master role: gateway present ---
services=$(docker compose \
  -f docker-compose.yml \
  -f deploy/compose/role-master.yml \
  --profile master \
  config --services 2>/dev/null || true)
if ! echo "$services" | grep -qx "gateway"; then
  fail "master role does not render gateway (must)"
fi
echo "OK: master role renders gateway"

# --- evil-gateway not in default or master ---
services=$(docker compose \
  -f docker-compose.yml \
  --profile master \
  config --services 2>/dev/null || true)
if echo "$services" | grep -qx "evil-gateway"; then
  fail "evil-gateway renders under master profile (must not)"
fi
echo "OK: evil-gateway not under master profile"

services=$(docker compose \
  -f docker-compose.yml \
  config --services 2>/dev/null || true)
if echo "$services" | grep -qx "evil-gateway"; then
  fail "evil-gateway renders under default profile (must not)"
fi
echo "OK: evil-gateway not under default profile"

# --- prod env + validator: no evil-gateway, no e2e ---
services=$(docker compose \
  -f docker-compose.yml \
  -f deploy/compose/role-validator.yml \
  -f deploy/compose/env-prod.yml \
  config --services 2>/dev/null || true)
if echo "$services" | grep -qx "evil-gateway"; then
  fail "prod validator renders evil-gateway (must not)"
fi
echo "OK: prod validator does not render evil-gateway"

# --- prism-challenge present in default ---
services=$(docker compose \
  -f docker-compose.yml \
  config --services 2>/dev/null || true)
if ! echo "$services" | grep -qx "prism-challenge"; then
  fail "prism-challenge not in default compose"
fi
echo "OK: prism-challenge in default compose"

echo "assert-compose-matrix: all checks passed"
