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

# deploy/env/*.env holds real config and is gitignored, so it is absent on a
# fresh checkout. `docker compose config` validates `env_file: required: true`,
# so without these the renders below fail and every grep passes vacuously.
# Materialise the examples for the duration of the run.
CREATED=()
cleanup() { for f in ${CREATED+"${CREATED[@]}"}; do rm -f "$f"; done; }
trap cleanup EXIT
for ex in deploy/env/*.env.example; do
  [ -e "$ex" ] || continue
  real="${ex%.example}"
  if [ ! -e "$real" ]; then
    cp "$ex" "$real"
    CREATED+=("$real")
  fi
done

# Render a compose combination, failing loudly instead of yielding "" on error.
render() {
  local out
  if ! out=$(docker compose "$@" 2>&1); then
    echo "$out" >&2
    fail "docker compose $* failed to render"
  fi
  printf '%s\n' "$out"
}

# --- validator role: no gateway ---
services=$(render \
  -f docker-compose.yml \
  -f deploy/compose/role-validator.yml \
  config --services)
if echo "$services" | grep -qx "gateway"; then
  fail "validator role renders gateway (must not)"
fi
echo "OK: validator role does not render gateway"

# --- master role: gateway present ---
services=$(render \
  -f docker-compose.yml \
  -f deploy/compose/role-master.yml \
  --profile master \
  config --services)
if ! echo "$services" | grep -qx "gateway"; then
  fail "master role does not render gateway (must)"
fi
echo "OK: master role renders gateway"

# --- evil-gateway not in default or master ---
services=$(render \
  -f docker-compose.yml \
  --profile master \
  config --services)
if echo "$services" | grep -qx "evil-gateway"; then
  fail "evil-gateway renders under master profile (must not)"
fi
echo "OK: evil-gateway not under master profile"

services=$(render \
  -f docker-compose.yml \
  config --services)
if echo "$services" | grep -qx "evil-gateway"; then
  fail "evil-gateway renders under default profile (must not)"
fi
echo "OK: evil-gateway not under default profile"

# --- prod env + validator: no evil-gateway, no e2e ---
services=$(render \
  -f docker-compose.yml \
  -f deploy/compose/role-validator.yml \
  -f deploy/compose/env-prod.yml \
  config --services)
if echo "$services" | grep -qx "evil-gateway"; then
  fail "prod validator renders evil-gateway (must not)"
fi
echo "OK: prod validator does not render evil-gateway"

# --- prism-challenge present in default ---
services=$(render \
  -f docker-compose.yml \
  config --services)
if ! echo "$services" | grep -qx "prism-challenge"; then
  fail "prism-challenge not in default compose"
fi
echo "OK: prism-challenge in default compose"

# --- no fake chain backend survives anywhere in the matrix ---
for env_file in deploy/compose/env-staging.yml deploy/compose/env-prod.yml; do
  for role_file in deploy/compose/role-master.yml deploy/compose/role-validator.yml; do
    rendered=$(render \
      -f docker-compose.yml \
      -f "$role_file" \
      -f "$env_file" \
      --profile master \
      config)
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
staging=$(render -f docker-compose.yml -f deploy/compose/role-master.yml \
  -f deploy/compose/env-staging.yml --profile master config)
echo "$staging" | grep -q "test.finney.opentensor.ai" \
  || fail "staging does not point at the testnet endpoint"
echo "$staging" | grep -q "BASE_NETUID: \"541\"" \
  || fail "staging netuid is not 541"

prod=$(render -f docker-compose.yml -f deploy/compose/role-master.yml \
  -f deploy/compose/env-prod.yml --profile master config)
echo "$prod" | grep -q "entrypoint-finney.opentensor.ai" \
  || fail "prod does not point at the mainnet endpoint"
echo "$prod" | grep -q "BASE_NETUID: \"100\"" \
  || fail "prod netuid is not 100"
echo "$prod" | grep -q "test.finney" \
  && fail "prod references the testnet endpoint"
echo "OK: staging pins testnet/541 and prod pins mainnet/100"

echo "assert-compose-matrix: all checks passed"
