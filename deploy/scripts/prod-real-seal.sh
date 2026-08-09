#!/usr/bin/env bash
# Prod real-epoch sealer: seal the CURRENT chain epoch on the master gateway
# as soon as both >0-bps challenges (design + prism) have posted their leaf
# sets for it. This replaces the interim block-scale burn-seal as the source
# of served weights once real scores flow.
#
# Why a blind retry loop: the seal endpoint is fail-safe — it 409s with
# `IncompleteParticipantSet` until both same-epoch sets exist, and re-sealing
# an already-sealed epoch is a no-op conflict. So we simply attempt the
# current epoch every 10 min and log the outcome.
#
# block_b pins the bundle metagraph. Both challenges pin their expected set
# at the epoch's start block (`last_epoch_block`), so block_b = the current
# LastEpochBlock gives an exact D24 participant match by construction —
# metagraph churn *inside* the epoch can no longer break the seal.
#
# Chain reads use plain HTTPS JSON-RPC state_getStorage with baked Substrate
# storage keys (twox128("SubtensorModule") ++ twox128(item) ++ netuid LE;
# Identity hasher on the key, matching chain-live). No secrets in this file.
set -euo pipefail

BASE_HOME="${BASE_HOME:-/opt/base}"
GATEWAY="${BASE_GATEWAY_ENDPOINT:-http://127.0.0.1:8080}"
NETUID="${BASE_NETUID:-100}"
LOG="${REAL_SEAL_LOG:-/var/log/base-real-seal.log}"
LOCK="${REAL_SEAL_LOCK:-/run/base-real-seal.lock}"
# Ordered failover; first reachable endpoint wins per call.
CHAIN_ENDPOINTS="${BASE_CHAIN_ENDPOINTS:-https://bittensor-finney.api.onfinality.io/public-ws,https://entrypoint-finney.opentensor.ai:443}"

# twox128("SubtensorModule") ++ twox128(item) prefixes (verified on finney).
K_SUBNET_EPOCH_INDEX="658faa385070e074c85bf6b568cf05554f101d7a30ae31c7ab3099206c5ae12b"
K_LAST_EPOCH_BLOCK="658faa385070e074c85bf6b568cf055590010c37124c14146041452f9ffba0df"

# Substrate Identity hasher on u16 netuid = little-endian bytes (not printf %04x).
netuid_le_hex() {
  python3 -c 'import sys; print(int(sys.argv[1]).to_bytes(2, "little").hex())' "$1"
}

rpc_storage() {
  # $1 = 0x-prefixed storage key; prints little-endian integer or nothing.
  local key="$1" ep out raw
  local -a eps
  IFS=',' read -r -a eps <<<"${CHAIN_ENDPOINTS}"
  for ep in "${eps[@]}"; do
    out="$(curl -fsS -m 15 -H 'content-type: application/json' \
      -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"state_getStorage\",\"params\":[\"${key}\"]}" \
      "${ep}" 2>/dev/null)" || continue
    raw="$(printf '%s' "${out}" | jq -r '.result // empty')"
    if [[ -n "${raw}" && "${raw}" != "null" ]]; then
      python3 -c 'import sys; print(int.from_bytes(bytes.fromhex(sys.argv[1][2:]), "little"))' "${raw}"
      return 0
    fi
  done
  return 1
}

exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "$(date -Is) skip: another run holds ${LOCK}" >>"${LOG}"
  exit 0
fi

{
  netuid_hex="$(netuid_le_hex "${NETUID}")"
  epoch="$(rpc_storage "0x${K_SUBNET_EPOCH_INDEX}${netuid_hex}")" || {
    echo "$(date -Is) chain read failed (epoch) key=0x${K_SUBNET_EPOCH_INDEX}${netuid_hex}"
    exit 1
  }
  leb="$(rpc_storage "0x${K_LAST_EPOCH_BLOCK}${netuid_hex}")" || {
    echo "$(date -Is) chain read failed (last_epoch_block) key=0x${K_LAST_EPOCH_BLOCK}${netuid_hex}"
    exit 1
  }
  auth_args=()
  if [[ -n "${BASE_GATEWAY_ADMIN_TOKEN:-}" ]]; then
    auth_args=(-H "Authorization: Bearer ${BASE_GATEWAY_ADMIN_TOKEN}")
  elif [[ -n "${BASE_GATEWAY_ADMIN_TOKEN_FILE:-}" && -f "${BASE_GATEWAY_ADMIN_TOKEN_FILE}" ]]; then
    auth_args=(-H "Authorization: Bearer $(tr -d '[:space:]' <"${BASE_GATEWAY_ADMIN_TOKEN_FILE}")")
  elif [[ -f "${BASE_HOME}/deploy/secrets/gateway_admin_token" ]]; then
    auth_args=(-H "Authorization: Bearer $(tr -d '[:space:]' <"${BASE_HOME}/deploy/secrets/gateway_admin_token")")
  fi
  resp="$(curl -fsS -m 60 -X POST -H 'content-type: application/json' \
    ${auth_args[@]+"${auth_args[@]}"} \
    -d "{\"epoch\":${epoch},\"netuid\":${NETUID},\"block_b\":${leb}}" \
    "${GATEWAY}/v1/admin/seal" 2>&1)" && rc=0 || rc=$?
  if [[ ${rc} -eq 0 ]]; then
    echo "$(date -Is) seal ok epoch=${epoch} block_b=${leb}: ${resp}"
  else
    # 409 (sets incomplete / already sealed) is the expected steady state
    # while waiting on a challenge emission; anything else needs a look.
    echo "$(date -Is) seal pending/failed rc=${rc} epoch=${epoch} block_b=${leb}: ${resp}"
  fi
} >>"${LOG}" 2>&1
