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
# D24 (exact-E): since the 2000/8000 trust-root activation, sealing requires a
# complete leaf set from EVERY >0-bps challenge at the seal epoch. The design
# pass therefore emits NoScore leaves first; its own seal attempt 409s until
# the prism pass lands (tolerated), and the prism pass then seals the complete
# set. All-NoScore still aggregates to the uid-0 burn vector.
set -euo pipefail

BASE_HOME="${BASE_HOME:-/opt/base}"
GATEWAY="${BASE_GATEWAY_ENDPOINT:-http://127.0.0.1:8080}"
NETUID="${BASE_NETUID:-100}"
CHAIN="${BASE_CHAIN_ENDPOINT:-wss://entrypoint-finney.opentensor.ai:443}"
SK="${BASE_CHALLENGE_SK_FILE:-${BASE_HOME}/deploy/secrets/prism_sk}"
DESIGN_SK="${BASE_DESIGN_SK_FILE:-${BASE_HOME}/deploy/secrets/design_sk}"
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
  # Design NoScore leaves (D24 participant). A 409 "incomplete participant
  # set" on its seal attempt is expected until the prism pass seals.
  dout="$("${BIN}" --gateway "${GATEWAY}" --burn --netuid "${NETUID}" \
    --chain-endpoint "${CHAIN}" --challenge-id design --challenge-sk "${DESIGN_SK}" 2>&1 || true)"
  echo "${dout}" | grep -E 'submitted|seal ok|latest OK|incomplete' | tail -2
  if ! echo "${dout}" | grep -qE 'submitted|seal ok'; then
    echo "$(date -Is) design leaves FAILED (no submission)"
    echo "${dout}" | tail -5
    exit 1
  fi
  # Prism pass: seals the now-complete D24 set at the tip.
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
