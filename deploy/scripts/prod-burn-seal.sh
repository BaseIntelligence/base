#!/usr/bin/env bash
# Interim prod burn-seal: keep a fresh sealed bundle on the master gateway so
# the Rust validator can Match + CRV4 submit every rate-limit window.
#
# Why: the validator verifies the bundle metagraph at the seal's pinned block.
# The public Finney RPC prunes state after ~256 blocks (~51 min), so a seal
# older than that can never Match and weights freeze on-chain. This script
# re-seals at the chain tip; run it on a systemd timer
# (deploy/systemd/base-burn-seal.timer, 21 min — just above the 100-block
# WeightsSetRateLimit, well inside the pruning window).
#
# Runs on the prod master only. No secrets are stored in this file; the
# challenge mini-secrets stay at $BASE_CHALLENGE_SK_FILE / $BASE_DESIGN_SK_FILE
# (mode 0400).
#
# D24 (exact-E): only challenges with emission_share_bps > 0 must have a
# complete leaf set. Extra leaves for a 0-bps challenge (design today) are
# IncompleteParticipantSet. Trust root is prism = 10000 / design = 0, so this
# script emits prism NoScore only. All-NoScore still aggregates to uid-0 burn.
set -euo pipefail

BASE_HOME="${BASE_HOME:-/opt/base}"
GATEWAY="${BASE_GATEWAY_ENDPOINT:-http://127.0.0.1:8080}"
NETUID="${BASE_NETUID:-100}"
# Ordered failover list wins; weights-smoke passes it straight to
# chain-live, which cools a rate-limited endpoint and tries the next.
CHAIN="${BASE_CHAIN_ENDPOINTS:-${BASE_CHAIN_ENDPOINT:-wss://entrypoint-finney.opentensor.ai:443}}"
SK="${BASE_CHALLENGE_SK_FILE:-${BASE_HOME}/deploy/secrets/prism_sk}"
DESIGN_SK="${BASE_DESIGN_SK_FILE:-${BASE_HOME}/deploy/secrets/design_sk}"
BIN="${WEIGHTS_SMOKE_BIN:-${BASE_HOME}/bin/weights-smoke}"
LOG="${BURN_SEAL_LOG:-/var/log/base-burn-seal.log}"
LOCK="${BURN_SEAL_LOCK:-/run/base-burn-seal.lock}"
# Admin bearer for /v1/admin/seal (required once gateway enforces it).
if [[ -z "${BASE_GATEWAY_ADMIN_TOKEN:-}" && -z "${BASE_GATEWAY_ADMIN_TOKEN_FILE:-}" \
  && -f "${BASE_HOME}/deploy/secrets/gateway_admin_token" ]]; then
  export BASE_GATEWAY_ADMIN_TOKEN_FILE="${BASE_HOME}/deploy/secrets/gateway_admin_token"
fi

exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "$(date -Is) skip: another run holds ${LOCK}" >>"${LOG}"
  exit 0
fi

{
  echo "$(date -Is) seal start gateway=${GATEWAY} netuid=${NETUID} challenge=prism"
  if out="$("${BIN}" --gateway "${GATEWAY}" --burn --netuid "${NETUID}" \
      --chain-endpoint "${CHAIN}" --challenge-id prism --challenge-sk "${SK}" 2>&1)"; then
    echo "${out}" | grep -E 'seal ok|latest OK' || echo "${out}" | tail -3
    echo "$(date -Is) seal ok"
  else
    rc=$?
    echo "${out}" | tail -8
    echo "$(date -Is) seal FAILED rc=${rc}"
    exit "${rc}"
  fi
} >>"${LOG}" 2>&1
