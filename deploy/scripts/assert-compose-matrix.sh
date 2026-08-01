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

# --- no fake chain backend survives anywhere in the matrix ---
for env_file in deploy/compose/env-staging.yml deploy/compose/env-prod.yml; do
  for role_file in deploy/compose/role-master.yml deploy/compose/role-validator.yml; do
    rendered=$(docker compose \
      -f docker-compose.yml \
      -f "$role_file" \
      -f "$env_file" \
      --profile master \
      config 2>/dev/null || true)
    if echo "$rendered" | grep -qi "fake_owner\|BASE_CHAIN_BACKEND"; then
      fail "$role_file + $env_file still references a fake chain backend"
    fi
    if ! echo "$rendered" | grep -q "BASE_CHAIN_ENDPOINT"; then
      fail "$role_file + $env_file does not set BASE_CHAIN_ENDPOINT"
    fi
  done
done
echo "OK: no fake chain backend in any role x env combination"

# --- each env pins its own netuid and endpoint ---
staging=$(docker compose -f docker-compose.yml -f deploy/compose/role-master.yml \
  -f deploy/compose/env-staging.yml --profile master config 2>/dev/null || true)
echo "$staging" | grep -q "test.finney.opentensor.ai" \
  || fail "staging does not point at the testnet endpoint"
echo "$staging" | grep -q "BASE_NETUID: \"541\"" \
  || fail "staging netuid is not 541"

prod=$(docker compose -f docker-compose.yml -f deploy/compose/role-master.yml \
  -f deploy/compose/env-prod.yml --profile master config 2>/dev/null || true)
echo "$prod" | grep -q "entrypoint-finney.opentensor.ai" \
  || fail "prod does not point at the mainnet endpoint"
echo "$prod" | grep -q "BASE_NETUID: \"100\"" \
  || fail "prod netuid is not 100"
echo "$prod" | grep -q "test.finney" \
  && fail "prod references the testnet endpoint"
echo "OK: staging pins testnet/541 and prod pins mainnet/100"

echo "assert-compose-matrix: all checks passed"
