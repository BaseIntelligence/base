#!/usr/bin/env bash
# Agent Challenge local staging - real Phala CVMs, real TDX quotes, real gates.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MONOREPO_ROOT="$(cd "${PKG_DIR}/../../.." && pwd)"
COMPOSE_FILE="${PKG_DIR}/docker-compose.staging.yml"
CONFIG_DIR="${SCRIPT_DIR}/config"
WORK_DIR="${SCRIPT_DIR}/work"
KR_DIR="${CONFIG_DIR}/kr"
EVIDENCE_DIR="${AC_STAGING_EVIDENCE_DIR:-/var/lib/base/e2e/ac-staging}"
HOST_PORT="${AC_STAGING_PORT:-18082}"
LOOPBACK_BASE="http://127.0.0.1:${HOST_PORT}"
MINER_ZIP="${PKG_DIR}/scripts/miner_agent/dist/miner_agent.zip"
EXPECTED_AGENT_HASH="61cca9bc06c52644182a4de98b89207742369589859d84a00ac6494327413f68"
COMPOSE="docker compose -f ${COMPOSE_FILE} --project-directory ${PKG_DIR}"

ONLY_REVIEW=0; ONLY_EVAL=0; DOWN_ONLY=0; SKIP_BUILD=0; KEEP_UP=0
DRY_RUN_TEARDOWN=0; ACCOUNT_SWEEP=0
MONEY_CAP="${AC_STAGING_MONEY_CAP:-8}"
RUNTIME_H="${AC_STAGING_RUNTIME_HOURS:-1}"
SUBMISSION_ID=""

usage(){ cat <<'EOF'
Usage: run_staging.sh [--review-only|--eval-only|--down|--skip-build|--keep-up]
                      [--submission-id N] [--money-cap USD] [--runtime-hours H]
                      [--dry-run-teardown] [--account-sweep]

CVM teardown is owned-only: only ids recorded in this run's track file
(and work/owned_cvms.txt) are deleted. Account-wide sweeps are NEVER default.
  --dry-run-teardown   Plan deletes (JSON) and exit without deleting anything.
  --account-sweep      LOUD opt-in leftover; still refuses foreign ids (owned-only).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --review-only) ONLY_REVIEW=1; shift ;;
    --eval-only) ONLY_EVAL=1; shift ;;
    --down) DOWN_ONLY=1; shift ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --keep-up) KEEP_UP=1; shift ;;
    --dry-run-teardown) DRY_RUN_TEARDOWN=1; shift ;;
    --account-sweep) ACCOUNT_SWEEP=1; shift ;;
    --submission-id) SUBMISSION_ID="${2:-}"; shift 2 ;;
    --money-cap) MONEY_CAP="${2:-}"; shift 2 ;;
    --runtime-hours) RUNTIME_H="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

mkdir -p "${WORK_DIR}" "${EVIDENCE_DIR}" "${KR_DIR}"
RUN_ID="run-$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${EVIDENCE_DIR}/${RUN_ID}"
mkdir -p "${RUN_DIR}"
CVM_TRACK="${RUN_DIR}/cvms.txt"; : >"${CVM_TRACK}"
# Durable owned-id list for --down across invocations (same work dir).
OWNED_CVMS_FILE="${WORK_DIR}/owned_cvms.txt"
touch "${OWNED_CVMS_FILE}"
LOG="${RUN_DIR}/staging.log"
# Line-buffer tee so progress is visible under nohup/pipe (block-buffering hides stalls).
if command -v stdbuf >/dev/null 2>&1; then
  exec > >(stdbuf -oL -eL tee -a "${LOG}") 2>&1
else
  exec > >(tee -a "${LOG}") 2>&1
fi

log(){ printf '[staging] %s\n' "$*"; }
die(){ log "FAIL: $*"; exit 1; }
uvrun(){ env UV_CACHE_DIR=/var/tmp/uv-cache uv run --package agent-challenge "$@"; }

load_phala_key(){
  if [[ -n "${PHALA_CLOUD_API_KEY:-}" ]]; then return 0; fi
  local cfg="${HOME}/.phala/config.json"
  [[ -f "${cfg}" ]] || die "missing ${cfg} and PHALA_CLOUD_API_KEY"
  PHALA_CLOUD_API_KEY="$(python3 -c "import json;d=json.load(open('${cfg}'));print(d['profiles']['echobts-projects']['token'])")"
  export PHALA_CLOUD_API_KEY
  [[ -n "${PHALA_CLOUD_API_KEY}" ]] || die "empty Phala token"
}
load_openrouter_key(){
  if [[ -n "${OPENROUTER_API_KEY:-}" ]]; then return 0; fi
  local cfg="${HOME}/.local/share/opencode/auth.json"
  [[ -f "${cfg}" ]] || die "missing ${cfg} and OPENROUTER_API_KEY"
  OPENROUTER_API_KEY="$(python3 -c "import json;d=json.load(open('${cfg}'));print(d['openrouter']['key'])")"
  export OPENROUTER_API_KEY
  [[ -n "${OPENROUTER_API_KEY}" ]] || die "empty OpenRouter key"
}

phala_get_cvms(){
  python3 - <<'PY'
import json,os,urllib.request
req=urllib.request.Request("https://cloud-api.phala.com/api/v1/cvms",headers={"X-API-Key":os.environ["PHALA_CLOUD_API_KEY"],"User-Agent":"phala-cloud-cli/1.1.19","Accept":"application/json"})
with urllib.request.urlopen(req,timeout=60) as r: data=json.loads(r.read())
items=data if isinstance(data,list) else (data.get("items") or data.get("data") or data.get("cvms") or [])
slim=[]
ids=[]
for i in items:
  if not isinstance(i,dict):
    continue
  api_id=str(i.get("id") or i.get("cvm_id") or "")
  if api_id:
    ids.append(api_id)
  slim.append({
    "id": api_id,
    "cvm_id": str(i.get("cvm_id") or "") or None,
    "vm_uuid": str(i.get("vm_uuid") or "") or None,
    "name": i.get("name"),
    "app_id": i.get("app_id"),
    "status": i.get("status"),
  })
print(json.dumps({"count":len(ids),"ids":ids,"items":slim}))
PY
}
phala_delete_cvm(){
  local id="$1"; [[ -z "$id" ]] && return 0
  # Hard guard: refuse any id not owned (track may hold vm_uuid; resolve via listing).
  local listing
  listing="$(phala_get_cvms || echo '{"count":0,"ids":[],"items":[]}')"
  if ! python3 "${SCRIPT_DIR}/cvm_teardown_policy.py" \
      --owned-file "${CVM_TRACK}" --owned-file "${OWNED_CVMS_FILE}" \
      --account-ids-json "${listing}" \
      --check-id "${id}" >/dev/null; then
    log "REFUSED delete of non-owned CVM id=${id}"
    return 2
  fi
  python3 - <<PY
import os,urllib.request,urllib.error
cid="${id}"
req=urllib.request.Request(f"https://cloud-api.phala.com/api/v1/cvms/{cid}",method="DELETE",headers={"X-API-Key":os.environ["PHALA_CLOUD_API_KEY"],"User-Agent":"phala-cloud-cli/1.1.19","Accept":"application/json"})
try:
  with urllib.request.urlopen(req,timeout=60) as r: print(f"delete {cid} -> {r.status}")
except urllib.error.HTTPError as e:
  print(f"delete {cid} -> HTTP {e.code}")
  if e.code not in (200,204,404): raise
PY
}
track_cvm(){
  local id="$1"; [[ -n "$id" ]] || return 0
  grep -qxF "$id" "${CVM_TRACK}" 2>/dev/null || echo "$id" >>"${CVM_TRACK}"
  grep -qxF "$id" "${OWNED_CVMS_FILE}" 2>/dev/null || echo "$id" >>"${OWNED_CVMS_FILE}"
}

extract_json_field(){
  # usage: extract_json_field FILE field_name
  local file="$1" field="$2"
  python3 - <<PY
import json,re
from pathlib import Path
text=Path("${file}").read_text(errors="replace")
val=""
def walk(x):
  global val
  if isinstance(x,dict):
    if "${field}" in x and isinstance(x["${field}"],str) and x["${field}"]:
      return x["${field}"]
    for v in x.values():
      r=walk(v)
      if r: return r
  elif isinstance(x,list):
    for i in x:
      r=walk(i)
      if r: return r
  return None
for blob in [text]+text.splitlines():
  s=blob.strip()
  if not s.startswith("{"): continue
  try: o=json.loads(s)
  except Exception: continue
  val=walk(o) or val
if not val and "${field}"=="cvm_id":
  m=re.search(r"cvm_[A-Za-z0-9]+", text)
  val=m.group(0) if m else ""
print(val)
PY
}
extract_phase(){
  local file="$1" kind="$2"
  python3 - <<PY
import json,re
from pathlib import Path
text=Path("${file}").read_text(errors="replace")
prefix="${kind}_"
phase=""
def find(x,d=0):
  if d>12: return None
  if isinstance(x,dict):
    for k in ("phase","review_phase","status"):
      v=x.get(k)
      if isinstance(v,str) and v.startswith(prefix): return v
    for v in x.values():
      r=find(v,d+1)
      if r: return r
  elif isinstance(x,list):
    for i in x:
      r=find(i,d+1)
      if r: return r
  return None
for blob in [text]+text.splitlines():
  s=blob.strip()
  if not s.startswith("{"): continue
  try: o=json.loads(s)
  except Exception: continue
  phase=find(o) or phase
if not phase:
  m=re.search(rf"{prefix}[a-z_]+", text)
  phase=m.group(0) if m else ""
print(phase)
PY
}
extract_guest_proof(){
  local file="$1"
  python3 - <<PY
import json,sys
from pathlib import Path
text=Path("${file}").read_text(errors="replace")
proof=None; score=None
def walk(x,d=0):
  global proof,score
  if d>14: return
  if isinstance(x,dict):
    if isinstance(x.get("guest_artifact_proof"),dict): proof=x["guest_artifact_proof"]
    if x.get("schema_version")==1 and {"expected_hash","download_hash","executed_hash"}<=set(x): proof=x
    if "score" in x and isinstance(x["score"],(int,float)): score=x["score"]
    for v in x.values(): walk(v,d+1)
  elif isinstance(x,list):
    for i in x: walk(i,d+1)
for blob in [text]+text.splitlines():
  s=blob.strip()
  if s.startswith("{"):
    try: walk(json.loads(s))
    except Exception: pass
expected="${EXPECTED_AGENT_HASH}"
print("SCORE", score)
print("PROOF", json.dumps(proof) if proof else None)
if not proof: sys.exit(3)
eh=str(proof.get("expected_hash") or ""); dh=str(proof.get("download_hash") or ""); xh=str(proof.get("executed_hash") or "")
ok = eh==dh==xh==expected and proof.get("match", True) is not False
print("PROOF_OK", ok); print("expected", eh); print("download", dh); print("executed", xh)
sys.exit(0 if ok else 2)
PY
}

plan_owned_teardown(){
  local account_json="${1:-}"
  local args=(python3 "${SCRIPT_DIR}/cvm_teardown_policy.py"
    --owned-file "${CVM_TRACK}" --owned-file "${OWNED_CVMS_FILE}" --dry-run)
  if [[ -n "${account_json}" ]]; then
    args+=(--account-ids-json "${account_json}")
  fi
  if [[ "${ACCOUNT_SWEEP}" == "1" ]]; then
    args+=(--account-sweep)
  fi
  "${args[@]}"
}

teardown_cvms(){
  # SAFETY: delete ONLY CVMs this staging run/work dir owns. Never account-sweep.
  load_phala_key
  local listing account_json plan will_delete id
  listing="$(phala_get_cvms || echo '{"count":-1,"ids":[]}')"
  echo "${listing}" | tee "${RUN_DIR}/cvms-before-teardown.json" >/dev/null
  account_json="${listing}"
  plan="$(plan_owned_teardown "${account_json}")"
  echo "${plan}" | tee "${RUN_DIR}/cvm-teardown-plan.json" >/dev/null
  will_delete="$(python3 -c "import json,sys; print(' '.join(json.load(sys.stdin).get('will_delete') or []))" <<<"${plan}")"
  log "teardown plan (owned-only): will_delete=[${will_delete}]"
  log "teardown plan JSON: ${RUN_DIR}/cvm-teardown-plan.json"
  if [[ "${ACCOUNT_SWEEP}" == "1" ]]; then
    log "WARNING: --account-sweep set but foreign CVMs are still NEVER deleted"
  fi
  if [[ "${DRY_RUN_TEARDOWN}" == "1" ]]; then
    log "dry-run-teardown: not deleting any CVM"
    echo "${plan}"
    return 0
  fi
  if [[ -z "${will_delete// /}" ]]; then
    log "teardown: no owned CVMs to delete (foreign account CVMs left untouched)"
  else
    log "teardown: deleting owned ids only: ${will_delete}"
    for id in ${will_delete}; do
      [[ -n "$id" ]] || continue
      uvrun python -m agent_challenge.selfdeploy teardown --cvm-id "$id" >/dev/null 2>&1 \
        || phala_delete_cvm "$id" || true
    done
  fi
  # Drop successfully targeted ids from durable owned list (best-effort).
  if [[ -f "${OWNED_CVMS_FILE}" && -n "${will_delete// /}" ]]; then
    local tmp_owned keep d
    tmp_owned="$(mktemp)"
    while read -r id; do
      [[ -n "$id" ]] || continue
      keep=1
      for d in ${will_delete}; do [[ "$id" == "$d" ]] && keep=0 && break; done
      [[ "$keep" == "1" ]] && echo "$id"
    done <"${OWNED_CVMS_FILE}" >"${tmp_owned}" || true
    mv -f "${tmp_owned}" "${OWNED_CVMS_FILE}"
  fi
  listing="$(phala_get_cvms || echo '{"count":-1,"ids":[]}')"
  echo "${listing}" | tee "${RUN_DIR}/cvms-final.json"
  local owned_left cnt
  cnt="$(python3 -c "import json;print(json.load(open('${RUN_DIR}/cvms-final.json')).get('count',-1))")"
  owned_left="$(python3 -c "
import json
from pathlib import Path
final=set(json.load(open('${RUN_DIR}/cvms-final.json')).get('ids') or [])
owned=set()
for p in ('${CVM_TRACK}','${OWNED_CVMS_FILE}'):
  path=Path(p)
  if path.is_file():
    owned |= {ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.strip().startswith('#')}
print(' '.join(sorted(owned & final)))
")"
  if [[ -n "${owned_left// /}" ]]; then
    log "WARNING: owned CVMs still present after teardown: ${owned_left} (account count=${cnt})"
    return 1
  fi
  log "teardown OK: all owned CVMs gone (account GET /cvms count=${cnt}; foreign left untouched)"
  return 0
}

ensure_kr_materials(){
  [[ -f "${KR_DIR}/server.crt" && -f "${KR_DIR}/server.key" && -f "${KR_DIR}/ca.crt" ]] || die "missing KR TLS under ${KR_DIR}"
  [[ -f "${KR_DIR}/golden.key" ]] || { openssl rand -out "${KR_DIR}/golden.key" 32; chmod 600 "${KR_DIR}/golden.key"; }
  [[ -f "${KR_DIR}/eval-allowlist.json" ]] || cp "${CONFIG_DIR}/eval_allowlist.json" "${KR_DIR}/eval-allowlist.json"
  python3 - <<PY
import json
from pathlib import Path
p=Path("${KR_DIR}/eval-allowlist.json")
d=json.loads(p.read_text())
ents=d.get("entries", d if isinstance(d,list) else [])
for e in ents: e.setdefault("key_provider","phala")
p.write_text(json.dumps({"entries": ents}, indent=2)+"\n")
PY
  cp -f "${KR_DIR}/ca.crt" "${CONFIG_DIR}/kr-server-ca.crt"
  [[ -f "${KR_DIR}/client-trust.crt" ]] || cp -f "${KR_DIR}/ca.crt" "${KR_DIR}/client-trust.crt"
}

start_kr(){
  ensure_kr_materials
  if ss -lntp 2>/dev/null | grep -q ':8701'; then log "KR already listening on :8701"; return 0; fi
  log "starting staging key-release RA-TLS on 0.0.0.0:8701"
  local kr_log="${WORK_DIR}/kr.log" kr_pid="${WORK_DIR}/kr.pid"
  (
    cd "${MONOREPO_ROOT}"
    export KEY_RELEASE_HOST=127.0.0.1 KEY_RELEASE_PORT=8700
    export KEY_RELEASE_RA_TLS_HOST=0.0.0.0 KEY_RELEASE_RA_TLS_PORT=8701
    export KEY_RELEASE_RA_TLS_CERT_FILE="${KR_DIR}/server.crt"
    export KEY_RELEASE_RA_TLS_KEY_FILE="${KR_DIR}/server.key"
    export KEY_RELEASE_RA_TLS_CA_FILE="${KR_DIR}/client-trust.crt"
    export CHALLENGE_KEY_RELEASE_ALLOWLIST_FILE="${KR_DIR}/eval-allowlist.json"
    export CHALLENGE_GOLDEN_KEY_FILE="${KR_DIR}/golden.key"
    export CHALLENGE_DATABASE_URL="sqlite+aiosqlite:///${WORK_DIR}/kr.sqlite3"
    export CHALLENGE_KEY_RELEASE_ACCEPTABLE_TCB=UpToDate
    export CHALLENGE_KEY_RELEASE_NONCE_TTL_SECONDS=300
    exec env PYTHONUNBUFFERED=1 UV_CACHE_DIR=/var/tmp/uv-cache uv run --package agent-challenge python -u -m agent_challenge.keyrelease.server
  ) >"${kr_log}" 2>&1 &
  echo $! >"${kr_pid}"
  for i in $(seq 1 60); do
    if grep -q 'production raw RA-TLS listening' "${kr_log}" 2>/dev/null; then log "KR up (pid=$(cat "${kr_pid}"))"; return 0; fi
    if ss -lntp 2>/dev/null | grep -q ':8701'; then log "KR up via :8701 (pid=$(cat "${kr_pid}"))"; return 0; fi
    if ! kill -0 "$(cat "${kr_pid}")" 2>/dev/null; then tail -50 "${kr_log}" || true; die "KR process exited"; fi
    sleep 1
  done
  tail -50 "${kr_log}" || true
  die "KR did not become ready"
}
stop_kr(){ if [[ -f "${WORK_DIR}/kr.pid" ]]; then kill "$(cat "${WORK_DIR}/kr.pid")" 2>/dev/null || true; rm -f "${WORK_DIR}/kr.pid"; fi; }

teardown_local(){
  log "teardown local compose + tunnel + staging KR"
  ${COMPOSE} down -v --remove-orphans 2>/dev/null || ${COMPOSE} down --remove-orphans || true
  if [[ -f "${WORK_DIR}/cloudflared.pid" ]]; then kill "$(cat "${WORK_DIR}/cloudflared.pid")" 2>/dev/null || true; rm -f "${WORK_DIR}/cloudflared.pid"; fi
  stop_kr
}

cleanup_all(){
  local ec=$?; set +e
  log "cleanup trap (exit=${ec})"
  teardown_cvms || true
  if [[ "${KEEP_UP}" != "1" || "${ec}" != "0" ]]; then teardown_local || true; fi
  exit "${ec}"
}
trap cleanup_all EXIT INT TERM

if [[ "${DRY_RUN_TEARDOWN}" == "1" ]]; then
  load_phala_key
  teardown_cvms
  trap - EXIT INT TERM
  log "PASS --dry-run-teardown complete"; exit 0
fi

if [[ "${DOWN_ONLY}" == "1" ]]; then
  load_phala_key; teardown_cvms; teardown_local
  trap - EXIT INT TERM
  log "PASS --down complete"; exit 0
fi

load_phala_key; load_openrouter_key
pre="$(phala_get_cvms)"
echo "${pre}" | tee "${RUN_DIR}/cvms-before.json" >/dev/null
pre_cnt="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("count",0))' "${RUN_DIR}/cvms-before.json")"
if [[ "${pre_cnt}" != "0" ]]; then
  log "WARNING: ${pre_cnt} CVMs already live on account — NOT sweeping (owned-only policy)."
  log "WARNING: foreign/prod CVMs will be left alone. Use --down after a prior staging run to clear owned ids in work/owned_cvms.txt."
  if [[ "${ACCOUNT_SWEEP}" == "1" ]]; then
    log "WARNING: --account-sweep does not expand deletes beyond owned track (safety)."
  fi
fi

if [[ ! -f "${MINER_ZIP}" ]]; then log "building miner_agent.zip"; (cd "${PKG_DIR}/scripts/miner_agent" && python3 build_zip.py); fi
zip_hash="$(sha256sum "${MINER_ZIP}" | awk '{print $1}')"
[[ "${zip_hash}" == "${EXPECTED_AGENT_HASH}" ]] || die "miner zip hash ${zip_hash} != ${EXPECTED_AGENT_HASH}"
log "miner zip ok hash=${zip_hash}"

if [[ "${SKIP_BUILD}" != "1" ]]; then log "building runtime image"; ${COMPOSE} build agent-challenge; fi
chmod a+r "${CONFIG_DIR}/challenge_token" "${CONFIG_DIR}/review_evidence_encryption_key" 2>/dev/null || true
log "compose up"; ${COMPOSE} up -d agent-challenge

log "waiting /health on ${LOOPBACK_BASE}"
for i in $(seq 1 90); do
  if curl -sf "${LOOPBACK_BASE}/health" >/dev/null 2>&1; then
    curl -sf "${LOOPBACK_BASE}/health" | tee "${RUN_DIR}/health.json"; log "health OK"; break
  fi
  sleep 2
  if [[ "$i" == "90" ]]; then ${COMPOSE} logs --no-color --tail=100 agent-challenge || true; die "health never became ready"; fi
done

PUBLIC_BASE=""; CF=""
if command -v cloudflared >/dev/null 2>&1; then CF="$(command -v cloudflared)"
elif [[ -x /tmp/cloudflared ]]; then CF=/tmp/cloudflared; fi
[[ -n "${CF}" ]] || die "cloudflared not found"
mkdir -p "${WORK_DIR}/cf-config"
printf "%s\n" "# quick-tunnel only" >"${WORK_DIR}/cf-config/config.yml"
start_cloudflared(){
  # Kill prior quick-tunnel only (never touch system /etc/cloudflared tunnels).
  if [[ -f "${WORK_DIR}/cloudflared.pid" ]]; then
    kill "$(cat "${WORK_DIR}/cloudflared.pid")" 2>/dev/null || true
    rm -f "${WORK_DIR}/cloudflared.pid"
  fi
  pkill -f "${WORK_DIR}/cf-config/config.yml" 2>/dev/null || true
  rm -f "${WORK_DIR}/cloudflared.log"
  log "starting cloudflared tunnel → ${LOOPBACK_BASE}"
  # setsid: survive parent tool timeouts that signal the whole process group
  setsid "${CF}" --config "${WORK_DIR}/cf-config/config.yml" tunnel --url "${LOOPBACK_BASE}" --no-autoupdate \
    >"${WORK_DIR}/cloudflared.log" 2>&1 < /dev/null &
  echo $! >"${WORK_DIR}/cloudflared.pid"
  local url="" i
  for i in $(seq 1 60); do
    if ! kill -0 "$(cat "${WORK_DIR}/cloudflared.pid")" 2>/dev/null; then
      log "cloudflared exited early; log tail:"; tail -30 "${WORK_DIR}/cloudflared.log" || true
      return 1
    fi
    url="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "${WORK_DIR}/cloudflared.log" 2>/dev/null | head -1 || true)"
    if [[ -n "${url}" ]]; then
      PUBLIC_BASE="${url}"
      return 0
    fi
    sleep 1
  done
  log "cloudflared URL not observed; log tail:"; tail -40 "${WORK_DIR}/cloudflared.log" || true
  return 1
}
PUBLIC_BASE=""
for cf_try in 1 2 3; do
  if start_cloudflared; then break; fi
  log "cloudflared attempt ${cf_try} failed; retrying"
  sleep 2
done
[[ -n "${PUBLIC_BASE}" ]] || die "could not establish public HTTPS tunnel"
log "public base=${PUBLIC_BASE}"; echo "${PUBLIC_BASE}" >"${RUN_DIR}/public_base.txt"
pub_ok=0
PUB_HOST="${PUBLIC_BASE#https://}"; PUB_HOST="${PUB_HOST%%/*}"
for i in $(seq 1 30); do
  if ! kill -0 "$(cat "${WORK_DIR}/cloudflared.pid")" 2>/dev/null; then
    log "cloudflared died during public health wait"; break
  fi
  PUB_IP="$(dig +short @1.1.1.1 "${PUB_HOST}" A 2>/dev/null | head -1 | tr -d "[:space:]")"
  if [[ -n "${PUB_IP}" ]] && curl -sf --max-time 10 --resolve "${PUB_HOST}:443:${PUB_IP}" "${PUBLIC_BASE}/health" >"${RUN_DIR}/health-public.json" 2>/dev/null; then
    cat "${RUN_DIR}/health-public.json"; log "public health OK (via ${PUB_IP})"; pub_ok=1; break
  fi
  sleep 2
done
[[ "${pub_ok}" == "1" ]] || die "public tunnel health never ready"

start_kr

export SELFDEPLOY_ALLOW_INSECURE_LOOPBACK=1 CHALLENGE_ALLOW_DEV_URLS=1
export PHALA_CLOUD_API_KEY OPENROUTER_API_KEY
export LLM_COST_LIMIT="${LLM_COST_LIMIT:-5}"
export CHALLENGE_SHARED_TOKEN_FILE="${CONFIG_DIR}/challenge_token"
export CHALLENGE_PHALA_RA_TLS_SERVER_CA_FILE="${CONFIG_DIR}/kr-server-ca.crt"
export KEY_RELEASE_SERVER_CA_FILE="${CONFIG_DIR}/kr-server-ca.crt"

HOTKEY_JSON="${WORK_DIR}/hotkey.json"
if [[ ! -f "${HOTKEY_JSON}" ]]; then
  python3 - <<'PY' >"${HOTKEY_JSON}"
from bittensor_wallet import Keypair
import json
m=Keypair.generate_mnemonic(); kp=Keypair.create_from_mnemonic(m)
print(json.dumps({"ss58": kp.ss58_address, "mnemonic": m}))
PY
  chmod 600 "${HOTKEY_JSON}"
fi
HOTKEY="$(python3 -c "import json;print(json.load(open('${HOTKEY_JSON}'))['ss58'])")"
export MINER_HOTKEY_MNEMONIC="$(python3 -c "import json;print(json.load(open('${HOTKEY_JSON}'))['mnemonic'])")"
log "hotkey=${HOTKEY}"

cd "${MONOREPO_ROOT}"

if [[ "${ONLY_EVAL}" != "1" ]]; then
  log "submitting miner agent zip"
  set +e
  uvrun python "${PKG_DIR}/scripts/submit_agent.py" submit \
    --api-base "${LOOPBACK_BASE}" --zip "${MINER_ZIP}" --name "staging-miner" \
    --hotkey-mnemonic "${MINER_HOTKEY_MNEMONIC}" --confirm-empty \
    >"${RUN_DIR}/submit.txt" 2>&1
  sub_ec=$?; set -e
  cat "${RUN_DIR}/submit.txt"
  [[ "${sub_ec}" == "0" ]] || die "submit failed ec=${sub_ec}"
  SUBMISSION_ID="$(RUN_DIR="${RUN_DIR}" python3 - <<'PY'
import os, re
text=open(os.environ["RUN_DIR"]+"/submit.txt").read()
for pat in [r'"submission_id"\s*:\s*(\d+)', r'submission_id=(\d+)']:
  m=re.search(pat,text,re.I)
  if m: print(m.group(1)); break
else: raise SystemExit('no submission id')
PY
)"
  log "submission_id=${SUBMISSION_ID}"; echo "${SUBMISSION_ID}" >"${RUN_DIR}/submission_id.txt"
fi
[[ -n "${SUBMISSION_ID}" ]] || die "submission id required"

if [[ "${ONLY_EVAL}" != "1" ]]; then
  log "review deploy (real Phala CVM tdx.small ${RUNTIME_H}h cap \$${MONEY_CAP})"
  set +e
  uvrun python -m agent_challenge.selfdeploy review deploy \
    --base-url "${PUBLIC_BASE}" --submission-id "${SUBMISSION_ID}" --hotkey "${HOTKEY}" --auto-sign \
    --openrouter-key-env OPENROUTER_API_KEY \
    --review-runtime-hours "${RUNTIME_H}" --eval-runtime-hours "${RUNTIME_H}" --money-cap-usd "${MONEY_CAP}" \
    >"${RUN_DIR}/review-deploy.json" 2>"${RUN_DIR}/review-deploy.err"
  rev_ec=$?; set -e
  cat "${RUN_DIR}/review-deploy.json" || true; cat "${RUN_DIR}/review-deploy.err" || true
  REV_CVM="$(extract_json_field "${RUN_DIR}/review-deploy.json" cvm_id)"
  [[ -n "${REV_CVM}" ]] || REV_CVM="$(extract_json_field "${RUN_DIR}/review-deploy.err" cvm_id)"
  if [[ -n "${REV_CVM}" ]]; then track_cvm "${REV_CVM}"; log "review_cvm_id=${REV_CVM}"; echo "${REV_CVM}" >"${RUN_DIR}/review_cvm_id.txt"; fi
  [[ "${rev_ec}" == "0" ]] || die "review deploy failed ec=${rev_ec}"

  log "polling review result → review_allowed"
  allowed=0
  for i in $(seq 1 90); do
    set +e
    uvrun python -m agent_challenge.selfdeploy review result \
      --base-url "${LOOPBACK_BASE}" --submission-id "${SUBMISSION_ID}" --hotkey "${HOTKEY}" --auto-sign \
      >"${RUN_DIR}/review-result-${i}.json" 2>/dev/null
    set -e
    phase="$(extract_phase "${RUN_DIR}/review-result-${i}.json" review)"
    log "review poll ${i}: phase=${phase:-unknown}"
    case "${phase}" in
      review_allowed) allowed=1; cp -f "${RUN_DIR}/review-result-${i}.json" "${RUN_DIR}/review-allowed.json"; break ;;
      review_rejected|review_escalated|review_error|review_expired|review_cancelled) die "review terminal phase=${phase}" ;;
    esac
    sleep 20
  done
  [[ "${allowed}" == "1" ]] || die "review_allowed not reached"
  log "review_allowed OK"
  if [[ -n "${REV_CVM}" ]]; then
    log "teardown review CVM ${REV_CVM}"
    uvrun python -m agent_challenge.selfdeploy review teardown --cvm-id "${REV_CVM}" \
      >"${RUN_DIR}/review-teardown.json" 2>&1 || phala_delete_cvm "${REV_CVM}" || true
    grep -vxF "${REV_CVM}" "${CVM_TRACK}" >"${CVM_TRACK}.tmp" 2>/dev/null || true
    mv "${CVM_TRACK}.tmp" "${CVM_TRACK}" 2>/dev/null || true
    grep -vxF "${REV_CVM}" "${OWNED_CVMS_FILE}" >"${OWNED_CVMS_FILE}.tmp" 2>/dev/null || true
    mv "${OWNED_CVMS_FILE}.tmp" "${OWNED_CVMS_FILE}" 2>/dev/null || true
  fi
fi

if [[ "${ONLY_REVIEW}" == "1" ]]; then
  teardown_cvms; trap - EXIT INT TERM
  [[ "${KEEP_UP}" == "1" ]] || teardown_local
  log "PASS review-only"; log "evidence: ${RUN_DIR}"; exit 0
fi

log "eval deploy (real Phala CVM tdx.small ${RUNTIME_H}h)"
TOKEN_FILE="${WORK_DIR}/eval-run-token"; rm -f "${TOKEN_FILE}"
set +e
uvrun python -m agent_challenge.selfdeploy eval deploy \
  --base-url "${PUBLIC_BASE}" --submission-id "${SUBMISSION_ID}" --hotkey "${HOTKEY}" --auto-sign \
  --token-output "${TOKEN_FILE}" \
  --eval-runtime-hours "${RUNTIME_H}" --review-runtime-hours "${RUNTIME_H}" --money-cap-usd "${MONEY_CAP}" \
  >"${RUN_DIR}/eval-deploy.json" 2>"${RUN_DIR}/eval-deploy.err"
eval_ec=$?; set -e
cat "${RUN_DIR}/eval-deploy.json" || true; cat "${RUN_DIR}/eval-deploy.err" || true
EVAL_CVM="$(extract_json_field "${RUN_DIR}/eval-deploy.json" cvm_id)"
[[ -n "${EVAL_CVM}" ]] || EVAL_CVM="$(extract_json_field "${RUN_DIR}/eval-deploy.err" cvm_id)"
EVAL_RUN_ID="$(extract_json_field "${RUN_DIR}/eval-deploy.json" eval_run_id)"
[[ -n "${EVAL_RUN_ID}" ]] || EVAL_RUN_ID="$(extract_json_field "${RUN_DIR}/eval-deploy.err" eval_run_id)"
if [[ -n "${EVAL_CVM}" ]]; then track_cvm "${EVAL_CVM}"; log "eval_cvm_id=${EVAL_CVM}"; echo "${EVAL_CVM}" >"${RUN_DIR}/eval_cvm_id.txt"; fi
if [[ -n "${EVAL_RUN_ID}" ]]; then echo "${EVAL_RUN_ID}" >"${RUN_DIR}/eval_run_id.txt"; log "eval_run_id=${EVAL_RUN_ID}"; fi
[[ "${eval_ec}" == "0" ]] || die "eval deploy failed ec=${eval_ec}"
[[ -f "${TOKEN_FILE}" ]] || die "missing eval run token file"
chmod 600 "${TOKEN_FILE}"

log "polling eval status → eval_accepted + guest_artifact_proof"
got=0
for i in $(seq 1 120); do
  set +e
  uvrun python -m agent_challenge.selfdeploy eval status \
    --base-url "${LOOPBACK_BASE}" --submission-id "${SUBMISSION_ID}" --hotkey "${HOTKEY}" --auto-sign \
    >"${RUN_DIR}/eval-status-${i}.json" 2>/dev/null
  set -e
  phase="$(extract_phase "${RUN_DIR}/eval-status-${i}.json" eval)"
  log "eval poll ${i}: phase=${phase:-unknown}"
  set +e; curl -sf "${LOOPBACK_BASE}/submissions/${SUBMISSION_ID}/status" >"${RUN_DIR}/submission-status-${i}.json" 2>/dev/null; set -e
  if extract_guest_proof "${RUN_DIR}/eval-status-${i}.json" >"${RUN_DIR}/proof-try.txt" 2>/dev/null; then
    cp -f "${RUN_DIR}/eval-status-${i}.json" "${RUN_DIR}/result-envelope.json"
    cp -f "${RUN_DIR}/proof-try.txt" "${RUN_DIR}/proof-summary.txt"; got=1; break
  fi
  if [[ -f "${RUN_DIR}/submission-status-${i}.json" ]] && extract_guest_proof "${RUN_DIR}/submission-status-${i}.json" >"${RUN_DIR}/proof-try.txt" 2>/dev/null; then
    cp -f "${RUN_DIR}/submission-status-${i}.json" "${RUN_DIR}/result-envelope.json"
    cp -f "${RUN_DIR}/proof-try.txt" "${RUN_DIR}/proof-summary.txt"; got=1; break
  fi
  case "${phase}" in
    eval_accepted)
      cp -f "${RUN_DIR}/eval-status-${i}.json" "${RUN_DIR}/result-envelope.json"
      if extract_guest_proof "${RUN_DIR}/result-envelope.json" >"${RUN_DIR}/proof-summary.txt" 2>/dev/null; then got=1; fi
      break ;;
    eval_rejected|eval_error|eval_expired|eval_cancelled)
      cp -f "${RUN_DIR}/eval-status-${i}.json" "${RUN_DIR}/result-envelope.json"; break ;;
  esac
  sleep 30
done

if [[ -n "${EVAL_CVM:-}" ]]; then
  log "teardown eval CVM ${EVAL_CVM}"
  uvrun python -m agent_challenge.selfdeploy eval teardown --cvm-id "${EVAL_CVM}" \
    >"${RUN_DIR}/eval-teardown.json" 2>&1 || phala_delete_cvm "${EVAL_CVM}" || true
fi
teardown_cvms

if [[ "${got}" != "1" ]]; then
  if [[ -f "${RUN_DIR}/result-envelope.json" ]]; then
    extract_guest_proof "${RUN_DIR}/result-envelope.json" | tee "${RUN_DIR}/proof-summary.txt" \
      || die "guest_artifact_proof missing or hash mismatch"
  else
    die "eval did not produce accepted result with guest_artifact_proof"
  fi
fi

log "PASS full staging loop"
log "evidence: ${RUN_DIR}"
cat "${RUN_DIR}/proof-summary.txt"
trap - EXIT INT TERM
if [[ "${KEEP_UP}" != "1" ]]; then teardown_local; else log "keeping AC up on ${LOOPBACK_BASE}"; fi
exit 0
