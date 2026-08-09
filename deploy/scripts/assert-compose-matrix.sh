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

# --- validator role: no gateway, no challenge execution surface ---
services=$(render \
  -f docker-compose.yml \
  -f deploy/compose/role-validator.yml \
  config --services)
if echo "$services" | grep -qx "gateway"; then
  fail "validator role renders gateway (must not)"
fi
for banned in design-challenge design-egress-proxy prism-challenge socket-proxy; do
  if echo "$services" | grep -qx "$banned"; then
    fail "validator role renders $banned (master-only; must not)"
  fi
done
echo "OK: validator role does not render gateway or challenge services"

# --- master role: gateway + challenge services present; no on-chain validator ---
services=$(render \
  -f docker-compose.yml \
  -f deploy/compose/role-master.yml \
  --profile master \
  config --services)
if ! echo "$services" | grep -qx "gateway"; then
  fail "master role does not render gateway (must)"
fi
for required in design-challenge design-egress-proxy prism-challenge socket-proxy; do
  if ! echo "$services" | grep -qx "$required"; then
    fail "master role does not render $required (must)"
  fi
done
if echo "$services" | grep -qx "validator"; then
  fail "master role renders validator (dual submitter; must not — use validator host)"
fi
echo "OK: master role renders gateway and challenge services (no validator)"

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

# --- prod env + validator: no evil-gateway, no e2e, no challenges ---
services=$(render \
  -f docker-compose.yml \
  -f deploy/compose/role-validator.yml \
  -f deploy/compose/env-prod.yml \
  config --services)
if echo "$services" | grep -qx "evil-gateway"; then
  fail "prod validator renders evil-gateway (must not)"
fi
for banned in design-challenge design-egress-proxy prism-challenge socket-proxy; do
  if echo "$services" | grep -qx "$banned"; then
    fail "prod validator renders $banned (master-only; must not)"
  fi
done
echo "OK: prod validator does not render evil-gateway or challenge services"

# --- staging/prod never enable host SimSandbox ---
for env_file in deploy/compose/env-staging.yml deploy/compose/env-prod.yml; do
  rendered=$(render \
    -f docker-compose.yml \
    -f deploy/compose/role-master.yml \
    -f "$env_file" \
    --profile master \
    config)
  if echo "$rendered" | grep -qE 'BASE_ALLOW_HOST_SIM:[[:space:]]*["'\'']?(1|true|TRUE|yes)["'\'']?'; then
    fail "$env_file enables BASE_ALLOW_HOST_SIM (host Sim forbidden on droplets)"
  fi
  if echo "$rendered" | grep -qE 'DESIGN_FORCE_SIM:[[:space:]]*["'\'']?(1|true|TRUE|yes)["'\'']?'; then
    fail "$env_file enables DESIGN_FORCE_SIM (Docker-only on droplets)"
  fi
  # Screenshot Chromium isolation: must force egress proxy (not empty / direct).
  if ! echo "$rendered" | grep -qE 'DESIGN_SCREENSHOT_PROXY:[[:space:]]*http://design-egress-proxy:8094'; then
    fail "$env_file master render missing DESIGN_SCREENSHOT_PROXY=http://design-egress-proxy:8094"
  fi
done
echo "OK: staging/prod do not enable host SimSandbox"
echo "OK: staging/prod force screenshot Chromium through design-egress-proxy"

# --- prism-challenge + design-challenge present in default ---
default_services=$(render \
  -f docker-compose.yml \
  config --services)
if ! echo "$default_services" | grep -qx "prism-challenge"; then
  fail "prism-challenge not in default compose"
fi
if ! echo "$default_services" | grep -qx "design-challenge"; then
  fail "design-challenge not in default compose"
fi
if ! echo "$default_services" | grep -qx "design-egress-proxy"; then
  fail "design-egress-proxy not in default compose"
fi
echo "OK: prism-challenge + design-challenge + design-egress-proxy in default compose"

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

# --- local overlay: still testnet, no fake backend ---
local_rendered=$(render \
  -f docker-compose.yml \
  -f deploy/compose/role-master.yml \
  -f deploy/compose/env-staging.yml \
  -f deploy/compose/env-local.yml \
  --profile master \
  config)
if echo "$local_rendered" | grep -qi "fake_owner\|BASE_CHAIN_BACKEND"; then
  fail "env-local introduces a fake chain backend"
fi
echo "$local_rendered" | grep -q "test.finney.opentensor.ai" \
  || fail "env-local stack lost the testnet endpoint"
echo "$local_rendered" | grep -q "BASE_NETUID: \"541\"" \
  || fail "env-local stack lost netuid 541"
local_services=$(render \
  -f docker-compose.yml \
  -f deploy/compose/role-master.yml \
  -f deploy/compose/env-staging.yml \
  -f deploy/compose/env-local.yml \
  --profile master \
  config --services)
echo "$local_services" | grep -qx "gateway" \
  || fail "env-local master stack does not render gateway"
echo "$local_services" | grep -qx "validator" \
  || fail "env-local master stack does not render co-located validator"
echo "$local_services" | grep -qx "prism-challenge" \
  || fail "env-local master stack does not render prism-challenge"
echo "$local_services" | grep -qx "design-challenge" \
  || fail "env-local master stack does not render design-challenge"
echo "$local_services" | grep -qx "design-egress-proxy" \
  || fail "env-local master stack does not render design-egress-proxy"
echo "$local_rendered" | grep -qE 'published: "?28093"?' \
  || fail "env-local does not publish design-challenge on 28093"
for banned in agent-challenge hypertraining-challenge miner-agent miner-socket-proxy base-agent; do
  if echo "$local_services" | grep -qx "$banned"; then
    fail "removed service still rendered: $banned"
  fi
done
# Default compose must also forbid removed services.
for banned in agent-challenge hypertraining-challenge miner-agent miner-socket-proxy base-agent; do
  if echo "$default_services" | grep -qx "$banned"; then
    fail "removed service still in default compose: $banned"
  fi
done
echo "OK: env-local preserves testnet/541; design on 28093; removed agent/hypertraining/miner services"

echo "assert-compose-matrix: all checks passed"
