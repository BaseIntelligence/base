#!/usr/bin/env bash
# Prod real-epoch sealer: seal (or tip-reseal) the newest chain epoch on the
# master gateway that has a complete D24 participant set from every >0-bps
# challenge.
#
# Tip reseal: `POST /v1/admin/seal` rebuilds from current leaves. When the tip
# is already sealed but leaves tip-superseded (design/prism re-emit), a new
# `epoch_bundle.revision` is appended. Identical merkle/vector → no-op 200.
#
# Why a walk-back: design historically emitted only in a tight late-tempo
# window and could skip alternate epochs (end-of-epoch relabel race). Waiting
# solely on *current* epoch then 409s forever while `/v1/weights/latest` stays
# pinned on an older real seal (burn seals cannot outrank it). Trying current,
# then current-1 … recovers the newest sealable epoch. Walk-back remains for
# incomplete D24; tip reseal success on current stops the walk (expected).
#
# block_b pins the bundle metagraph to that epoch's start block
# (LastEpochBlock − k×tempo) so D24 participant matching holds.
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
# How many prior epochs to try when current is incomplete (≈12h at tempo 360).
WALK_BACK="${REAL_SEAL_WALK_BACK:-16}"
# Ordered failover; first reachable endpoint wins per call.
CHAIN_ENDPOINTS="${BASE_CHAIN_ENDPOINTS:-https://bittensor-finney.api.onfinality.io/public-ws,https://entrypoint-finney.opentensor.ai:443}"

# twox128("SubtensorModule") ++ twox128(item) prefixes (verified on finney).
K_SUBNET_EPOCH_INDEX="658faa385070e074c85bf6b568cf05554f101d7a30ae31c7ab3099206c5ae12b"
K_LAST_EPOCH_BLOCK="658faa385070e074c85bf6b568cf055590010c37124c14146041452f9ffba0df"
# twox128(SubtensorModule) ++ twox128(Tempo)
K_TEMPO="658faa385070e074c85bf6b568cf05557641384bb339f3758acddfd7053d3317"

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

# Tempo is Option<u16> on chain (0x01 + LE u16) or bare u16 depending on codec;
# accept both shapes.
rpc_tempo() {
  local key="$1" ep out raw
  local -a eps
  IFS=',' read -r -a eps <<<"${CHAIN_ENDPOINTS}"
  for ep in "${eps[@]}"; do
    out="$(curl -fsS -m 15 -H 'content-type: application/json' \
      -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"state_getStorage\",\"params\":[\"${key}\"]}" \
      "${ep}" 2>/dev/null)" || continue
    raw="$(printf '%s' "${out}" | jq -r '.result // empty')"
    if [[ -n "${raw}" && "${raw}" != "null" ]]; then
      python3 -c '
import sys
b=bytes.fromhex(sys.argv[1][2:])
# Plain u16 (ValueQuery) or Option<u16> (0x01 + LE).
if len(b)==3 and b[0]==1:
    print(int.from_bytes(b[1:], "little"))
elif len(b)>=2:
    print(int.from_bytes(b[:2], "little"))
else:
    sys.exit(1)
' "${raw}" && return 0
    fi
  done
  return 1
}

attempt_seal() {
  local epoch="$1" block_b="$2"
  local resp rc=0
  resp="$(curl -sS -m 60 -X POST -H 'content-type: application/json' \
    -w '\n%{http_code}' \
    ${auth_args[@]+"${auth_args[@]}"} \
    -d "{\"epoch\":${epoch},\"netuid\":${NETUID},\"block_b\":${block_b}}" \
    "${GATEWAY}/v1/admin/seal" 2>&1)" || rc=$?
  local http body
  http="$(printf '%s' "${resp}" | tail -1)"
  body="$(printf '%s' "${resp}" | sed '$d')"
  if [[ ${rc} -eq 0 && "${http}" == "200" ]]; then
    echo "$(date -Is) seal ok epoch=${epoch} block_b=${block_b}: ${body}"
    return 0
  fi
  echo "$(date -Is) seal pending/failed epoch=${epoch} block_b=${block_b} http=${http:-?} rc=${rc}: ${body}"
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
  # Tempo storage key: twox128(SubtensorModule)++twox128(Tempo)++netuid LE.
  # Fallback 360 (finney default) if the read fails.
  tempo="$(rpc_tempo "0x${K_TEMPO}${netuid_hex}" 2>/dev/null || true)"
  if [[ -z "${tempo}" || "${tempo}" -le 0 ]]; then
    tempo=360
  fi
  auth_args=()
  if [[ -n "${BASE_GATEWAY_ADMIN_TOKEN:-}" ]]; then
    auth_args=(-H "Authorization: Bearer ${BASE_GATEWAY_ADMIN_TOKEN}")
  elif [[ -n "${BASE_GATEWAY_ADMIN_TOKEN_FILE:-}" && -f "${BASE_GATEWAY_ADMIN_TOKEN_FILE}" ]]; then
    auth_args=(-H "Authorization: Bearer $(tr -d '[:space:]' <"${BASE_GATEWAY_ADMIN_TOKEN_FILE}")")
  elif [[ -f "${BASE_HOME}/deploy/secrets/gateway_admin_token" ]]; then
    auth_args=(-H "Authorization: Bearer $(tr -d '[:space:]' <"${BASE_HOME}/deploy/secrets/gateway_admin_token")")
  fi

  echo "$(date -Is) seal walk start chain_epoch=${epoch} last_epoch_block=${leb} tempo=${tempo} walk_back=${WALK_BACK}"

  sealed=0
  for ((k=0; k<=WALK_BACK; k++)); do
    try_epoch=$((epoch - k))
    if (( try_epoch <= 0 )); then
      break
    fi
    try_block=$((leb - k * tempo))
    if (( try_block < 0 )); then
      break
    fi
    if attempt_seal "${try_epoch}" "${try_block}"; then
      sealed=1
      break
    fi
  done

  if [[ "${sealed}" -ne 1 ]]; then
    echo "$(date -Is) seal walk exhausted: no complete D24 set in last ${WALK_BACK} epochs (latest remains stale until design+prism emit)"
    exit 1
  fi
} >>"${LOG}" 2>&1
