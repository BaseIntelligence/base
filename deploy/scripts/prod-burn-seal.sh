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
# challenge mini-secret stays at $BASE_CHALLENGE_SK_FILE (mode 0400).
set -euo pipefail

BASE_HOME="${BASE_HOME:-/opt/base}"
GATEWAY="${BASE_GATEWAY_ENDPOINT:-http://127.0.0.1:8080}"
NETUID="${BASE_NETUID:-100}"
CHAIN="${BASE_CHAIN_ENDPOINT:-wss://entrypoint-finney.opentensor.ai:443}"
SK="${BASE_CHALLENGE_SK_FILE:-${BASE_HOME}/deploy/secrets/prism_sk}"
BIN="${WEIGHTS_SMOKE_BIN:-${BASE_HOME}/bin/weights-smoke}"
LOG="${BURN_SEAL_LOG:-/var/log/base-burn-seal.log}"
LOCK="${BURN_SEAL_LOCK:-/run/base-burn-seal.lock}"

exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "$(date -Is) skip: another run holds ${LOCK}" >>"${LOG}"
  exit 0
fi

{
  echo "$(date -Is) seal start gateway=${GATEWAY} netuid=${NETUID}"
  if out="$("${BIN}" --gateway "${GATEWAY}" --burn --netuid "${NETUID}" \
      --chain-endpoint "${CHAIN}" --challenge-sk "${SK}" 2>&1)"; then
    echo "${out}" | grep -E 'seal ok|latest OK' || echo "${out}" | tail -3
    echo "$(date -Is) seal ok"
  else
    rc=$?
    echo "${out}" | tail -5
    echo "$(date -Is) seal FAILED rc=${rc}"
    exit "${rc}"
  fi
} >>"${LOG}" 2>&1
