#!/usr/bin/env bash
# Host-side digest reconciler for independent Compose validators (Option A).
# Tracks BASE_VALIDATOR_TRACK_IMAGE (default base-validator-runtime:latest),
# always applies repository@sha256:<digest> pins, never bare :latest runtime.
# This script always runs on the host only. The agent may also mount docker.sock
# for later challenges-on-validator prep; image auto-update remains host-side.
set -euo pipefail

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }

PROJECT_NAME="${COMPOSE_PROJECT_NAME:-}"
ARTIFACTS_DIR="${BASE_VALIDATOR_ARTIFACTS_DIR:-}"
COMPOSE_FILE="${BASE_VALIDATOR_COMPOSE_FILE:-}"
ENV_FILE="${BASE_VALIDATOR_ENV_FILE:-}"
STATE_FILE="${BASE_VALIDATOR_IMAGE_UPDATE_STATE:-}"
TRACK_IMAGE="${BASE_VALIDATOR_TRACK_IMAGE:-ghcr.io/baseintelligence/base-validator-runtime:latest}"
HOLD="${BASE_VALIDATOR_IMAGE_UPDATE_HOLD:-0}"
DRY_RUN="${BASE_VALIDATOR_IMAGE_UPDATE_DRY_RUN:-0}"
SERVICE_NAME="${BASE_VALIDATOR_SERVICE_NAME:-validator}"
DOCKER_BIN="${DOCKER_BIN:-docker}"
COMMAND_TIMEOUT="${BASE_VALIDATOR_IMAGE_UPDATE_TIMEOUT:-300}"
MAX_ATTEMPTS="${BASE_VALIDATOR_IMAGE_UPDATE_MAX_ATTEMPTS:-5}"
BASE_DELAY="${BASE_VALIDATOR_IMAGE_UPDATE_BASE_DELAY:-60}"
MAX_DELAY="${BASE_VALIDATOR_IMAGE_UPDATE_MAX_DELAY:-1800}"

usage() {
  cat <<'EOF'
Usage: validator-image-updater.sh [once]

Environment (required):
  COMPOSE_PROJECT_NAME
  BASE_VALIDATOR_ARTIFACTS_DIR   # or explicit compose/env/state paths

Optional:
  BASE_VALIDATOR_COMPOSE_FILE
  BASE_VALIDATOR_ENV_FILE
  BASE_VALIDATOR_IMAGE_UPDATE_STATE
  BASE_VALIDATOR_TRACK_IMAGE     (default: ghcr.io/baseintelligence/base-validator-runtime:latest)
  BASE_VALIDATOR_IMAGE_UPDATE_HOLD=0|1
  BASE_VALIDATOR_IMAGE_UPDATE_DRY_RUN=0|1
EOF
}

if [[ "${1:-once}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -z "${PROJECT_NAME}" ]]; then
  log "error: COMPOSE_PROJECT_NAME is required"
  exit 2
fi

if [[ -n "${ARTIFACTS_DIR}" ]]; then
  COMPOSE_FILE="${COMPOSE_FILE:-${ARTIFACTS_DIR}/docker-compose.validator.yml}"
  ENV_FILE="${ENV_FILE:-${ARTIFACTS_DIR}/.env}"
  STATE_FILE="${STATE_FILE:-${ARTIFACTS_DIR}/image_update_state.json}"
fi

if [[ -z "${COMPOSE_FILE}" || -z "${ENV_FILE}" ]]; then
  log "error: set BASE_VALIDATOR_ARTIFACTS_DIR or compose/env file paths"
  exit 2
fi

STATE_FILE="${STATE_FILE:-$(dirname "${ENV_FILE}")/image_update_state.json}"

# Prefer in-image Python kernel when the package is available on host; otherwise
# a pure bash/curl path below. Source-free hosts can still use docker-run of the
# pin with host docker.sock for the short-lived helper only (never the agent).
_run_python_kernel() {
  local py
  for py in python3 python; do
    if command -v "${py}" >/dev/null 2>&1 \
      && "${py}" -c 'import base.supervisor.validator_image_updater' >/dev/null 2>&1; then
      exec "${py}" -m base.supervisor.validator_image_updater once
    fi
  done
  return 1
}

export COMPOSE_PROJECT_NAME PROJECT_NAME
export BASE_VALIDATOR_ARTIFACTS_DIR="${ARTIFACTS_DIR:-}"
export BASE_VALIDATOR_COMPOSE_FILE="${COMPOSE_FILE}"
export BASE_VALIDATOR_ENV_FILE="${ENV_FILE}"
export BASE_VALIDATOR_IMAGE_UPDATE_STATE="${STATE_FILE}"
export BASE_VALIDATOR_TRACK_IMAGE="${TRACK_IMAGE}"
export BASE_VALIDATOR_IMAGE_UPDATE_HOLD="${HOLD}"
export BASE_VALIDATOR_IMAGE_UPDATE_DRY_RUN="${DRY_RUN}"

if _run_python_kernel; then
  exit 0
fi

# ---------- pure bash fallback (docker + curl/jq or docker manifest) ----------

_hex64='[0-9a-f]{64}'
_digest_re="sha256:${_hex64}"

normalize_digest() {
  local raw="${1:-}"
  raw="$(printf '%s' "${raw}" | tr 'A-F' 'a-f' | tr -d '[:space:]')"
  if [[ "${raw}" =~ ^sha256:${_hex64}$ ]]; then
    printf '%s' "${raw}"
    return 0
  fi
  if [[ "${raw}" =~ ^${_hex64}$ ]]; then
    printf 'sha256:%s' "${raw}"
    return 0
  fi
  if [[ "${raw}" =~ (sha256:${_hex64}) ]]; then
    printf '%s' "${BASH_REMATCH[1]}"
    return 0
  fi
  return 1
}

read_env_key() {
  local key="$1"
  local file="$2"
  [[ -f "${file}" ]] || return 0
  # shellcheck disable=SC2162
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line#"${line%%[![:space:]]*}"}"
    [[ -z "${line}" || "${line}" == \#* ]] && continue
    [[ "${line}" == export\ * ]] && line="${line#export }"
    case "${line}" in
      "${key}="*)
        local val="${line#*=}"
        val="${val%\"}"
        val="${val#\"}"
        val="${val%\'}"
        val="${val#\'}"
        printf '%s' "${val}"
        return 0
        ;;
    esac
  done <"${file}"
}

write_env_atomic() {
  local repo="$1"
  local dig_hex="$2"
  local tmp
  tmp="$(mktemp "${ENV_FILE}.tmp.XXXXXX")"
  umask 077
  {
    printf 'COMPOSE_PROJECT_NAME=%s\n' "${PROJECT_NAME}"
    printf 'BASE_VALIDATOR_IMAGE_REPOSITORY=%s\n' "${repo}"
    printf 'BASE_VALIDATOR_IMAGE_DIGEST=%s\n' "${dig_hex}"
    printf 'BASE_VALIDATOR_TRACK_IMAGE=%s\n' "${TRACK_IMAGE}"
    # Preserve known operator paths if present.
    for key in BASE_VALIDATOR_CONFIG BASE_VALIDATOR_PROTOCOL_IDENTITY BASE_VALIDATOR_BROKER_TOKEN BASE_VALIDATOR_IMAGE_UPDATE_HOLD; do
      val="$(read_env_key "${key}" "${ENV_FILE}" || true)"
      if [[ -n "${val}" ]]; then
        printf '%s=%s\n' "${key}" "${val}"
      fi
    done
  } >"${tmp}"
  chmod 600 "${tmp}"
  mv -f "${tmp}" "${ENV_FILE}"
  chmod 600 "${ENV_FILE}"
}

load_state_json() {
  if [[ -f "${STATE_FILE}" ]] && command -v jq >/dev/null 2>&1; then
    cat "${STATE_FILE}"
  else
    printf '%s\n' '{"desired_digest":null,"current_digest":null,"rollback_digest":null,"phase":"idle","attempts":0,"next_eligible_at":null,"hold":false,"last_error":null,"alerted":false}'
  fi
}

save_state_json() {
  local json="$1"
  local tmp
  tmp="$(mktemp "${STATE_FILE}.tmp.XXXXXX")"
  umask 077
  printf '%s\n' "${json}" >"${tmp}"
  chmod 600 "${tmp}"
  mv -f "${tmp}" "${STATE_FILE}"
  chmod 600 "${STATE_FILE}"
}

now_unix() { date +%s; }

resolve_remote_digest() {
  local image="$1"
  # Prefer docker manifest inspect (uses docker config auth when present).
  local out dig
  if out="$("${DOCKER_BIN}" manifest inspect --verbose "${image}" 2>/dev/null)"; then
    if dig="$(printf '%s' "${out}" | tr -d '\r' | grep -Eo "sha256:${_hex64}" | head -n1)"; then
      normalize_digest "${dig}" && return 0
    fi
  fi
  # Registry HEAD via curl: parse tag.
  local name tag registry repo
  name="${image%@*}"
  if [[ "${name}" == *:* ]]; then
    tag="${name##*:}"
    name="${name%:*}"
  else
    tag="latest"
  fi
  if [[ "${name}" == */*/* ]] || [[ "${name}" == *.*/* ]]; then
    registry="${name%%/*}"
    repo="${name#*/}"
  else
    registry="docker.io"
    repo="${name}"
  fi
  local accept='application/vnd.docker.distribution.manifest.list.v2+json,application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json'
  local url="https://${registry}/v2/${repo}/manifests/${tag}"
  local headers
  headers="$(curl -sSIL -H "Accept: ${accept}" "${url}" 2>/dev/null || true)"
  if printf '%s' "${headers}" | grep -qi 'Docker-Content-Digest:'; then
    dig="$(printf '%s' "${headers}" | grep -i 'Docker-Content-Digest:' | head -n1 | awk '{print $2}' | tr -d '\r')"
    normalize_digest "${dig}" && return 0
  fi
  # Anonymous 401 challenge — public GHCR packages often still return digest after token.
  local realm service scope token
  if printf '%s' "${headers}" | grep -qi 'www-authenticate:'; then
    local auth
    auth="$(printf '%s' "${headers}" | grep -i 'www-authenticate:' | head -n1)"
    realm="$(printf '%s' "${auth}" | sed -n 's/.*realm="\([^"]*\)".*/\1/p')"
    service="$(printf '%s' "${auth}" | sed -n 's/.*service="\([^"]*\)".*/\1/p')"
    scope="$(printf '%s' "${auth}" | sed -n 's/.*scope="\([^"]*\)".*/\1/p')"
    if [[ -n "${realm}" ]]; then
      token="$(curl -fsS "${realm}?service=${service}&scope=${scope}" 2>/dev/null \
        | sed -n 's/.*"token"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
        | head -n1 || true)"
      if [[ -z "${token}" ]]; then
        token="$(curl -fsS "${realm}?service=${service}&scope=${scope}" 2>/dev/null \
          | sed -n 's/.*"access_token"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
          | head -n1 || true)"
      fi
      if [[ -n "${token}" ]]; then
        headers="$(curl -sSIL -H "Accept: ${accept}" -H "Authorization: Bearer ${token}" "${url}" 2>/dev/null || true)"
        dig="$(printf '%s' "${headers}" | grep -i 'Docker-Content-Digest:' | head -n1 | awk '{print $2}' | tr -d '\r' || true)"
        if dig="$(normalize_digest "${dig}")"; then
          printf '%s' "${dig}"
          return 0
        fi
      fi
    fi
  fi
  return 1
}

compose_cmd() {
  # Drop process-env IMAGE pins so --env-file is authoritative (compose
  # prefers existing process env over the env-file, which would freeze
  # auto-update on the pre-tick digest).
  env -u BASE_VALIDATOR_IMAGE_REPOSITORY -u BASE_VALIDATOR_IMAGE_DIGEST \
    -u COMPOSE_FILE -u COMPOSE_PATH \
    timeout "${COMMAND_TIMEOUT}" "${DOCKER_BIN}" compose -p "${PROJECT_NAME}" \
    -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" "$@"
}


inspect_running_digest() {
  local out dig name
  # Only Config.Image: some engines reject {{json .RepoDigests}} with a
  # template error that empties the whole inspect output.
  for name in "${PROJECT_NAME}-${SERVICE_NAME}-1" "${PROJECT_NAME}_${SERVICE_NAME}_1"; do
    out="$("${DOCKER_BIN}" inspect --format '{{.Config.Image}}' "${name}" 2>/dev/null || true)"
    [[ -z "${out}" ]] && continue
    if dig="$(normalize_digest "${out}" 2>/dev/null)"; then
      printf '%s' "${dig}"
      return 0
    fi
  done
  return 1
}

current_digest_from_env() {
  local raw
  raw="$(read_env_key BASE_VALIDATOR_IMAGE_DIGEST "${ENV_FILE}" || true)"
  normalize_digest "${raw}" || true
}

current_repo_from_env() {
  read_env_key BASE_VALIDATOR_IMAGE_REPOSITORY "${ENV_FILE}" || true
}

hold_from_env() {
  local flag
  flag="$(read_env_key BASE_VALIDATOR_IMAGE_UPDATE_HOLD "${ENV_FILE}" || true)"
  case "${flag,,}" in
    1|true|yes|on) return 0 ;;
  esac
  return 1
}

# Demand explicit tag on track image.
if [[ "${TRACK_IMAGE}" != *:* ]]; then
  log "error: track image must include an explicit tag (got ${TRACK_IMAGE})"
  exit 2
fi

case "${HOLD,,}" in
  1|true|yes|on)
    log "info: skipped-held project=${PROJECT_NAME}"
    exit 0
    ;;
esac
if hold_from_env; then
  log "info: skipped-held (env) project=${PROJECT_NAME}"
  exit 0
fi

state_json="$(load_state_json)"
attempts=0
next_eligible_at=0
desired_digest=""
rollback_digest=""
alerted=false
if command -v jq >/dev/null 2>&1; then
  attempts="$(printf '%s' "${state_json}" | jq -r '.attempts // 0')"
  next_eligible_at="$(printf '%s' "${state_json}" | jq -r '.next_eligible_at // 0')"
  desired_digest="$(printf '%s' "${state_json}" | jq -r '.desired_digest // empty')"
  rollback_digest="$(printf '%s' "${state_json}" | jq -r '.rollback_digest // empty')"
  alerted="$(printf '%s' "${state_json}" | jq -r '.alerted // false')"
fi
now="$(now_unix)"
if [[ -n "${next_eligible_at}" && "${next_eligible_at}" != "null" && "${next_eligible_at}" != "0" ]]; then
  if (( now < ${next_eligible_at%.*} )); then
    log "info: backoff project=${PROJECT_NAME} until ${next_eligible_at}"
    exit 0
  fi
fi

if ! remote="$(resolve_remote_digest "${TRACK_IMAGE}")"; then
  log "warn: digest resolution failed for ${TRACK_IMAGE}"
  exit 0
fi
remote="$(normalize_digest "${remote}")"
current_env="$(current_digest_from_env || true)"
current_run="$(inspect_running_digest || true)"
# Prefer running container as truth when present so a rewritten .env with a
# failed/stale recreate does not permanently no-op.
current="${current_run:-${current_env}}"

if [[ "${desired_digest}" != "${remote}" ]]; then
  attempts=0
  alerted=false
  desired_digest="${remote}"
fi

if [[ -n "${current_env}" && "${current_env}" == "${remote}" \
  && -n "${current_run}" && "${current_run}" == "${remote}" ]]; then
  log "info: no-op project=${PROJECT_NAME} already at ${remote}"
  if command -v jq >/dev/null 2>&1; then
    save_state_json "$(jq -n \
      --arg d "${remote}" \
      --arg c "${current_run}" \
      --arg t "${TRACK_IMAGE}" \
      '{desired_digest:$d,current_digest:$c,rollback_digest:null,phase:"idle",attempts:0,next_eligible_at:null,hold:false,last_error:null,alerted:false,track_image:$t}')"
  fi
  exit 0
fi

# Env already at desired but container lags: force recreate without clamoring as new resolve.
if [[ -n "${current_env}" && "${current_env}" == "${remote}" \
  && ( -z "${current_run}" || "${current_run}" != "${remote}" ) ]]; then
  log "info: env pin current but container lags (run=${current_run:-none}); forcing recreate"
fi

if (( attempts >= MAX_ATTEMPTS )); then
  log "warn: exhausted for project=${PROJECT_NAME} digest=${remote}; skip until new digest"
  exit 0
fi

repo="$(current_repo_from_env)"
if [[ -z "${repo}" ]]; then
  # strip tag from track image
  repo="${TRACK_IMAGE%:*}"
  repo="${repo%@*}"
fi
dig_hex="${remote#sha256:}"

# Pin policy: never compose-run bare :latest
if [[ -z "${repo}" || -z "${dig_hex}" || ! "${remote}" =~ ^sha256:${_hex64}$ ]]; then
  log "error: pin policy reject repo=${repo} digest=${remote}"
  exit 1
fi

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN,,}" == "true" ]]; then
  log "info: dry-run project=${PROJECT_NAME} current=${current:-none} desired=${remote} pin=${repo}@${remote}"
  exit 0
fi

rollback_digest="${current:-}"
log "info: updating project=${PROJECT_NAME} to ${repo}@${remote}"
write_env_atomic "${repo}" "${dig_hex}"

if ! compose_cmd pull "${SERVICE_NAME}"; then
  log "error: compose pull failed"
  if [[ -n "${rollback_digest}" ]]; then
    write_env_atomic "${repo}" "${rollback_digest#sha256:}"
    compose_cmd up -d --force-recreate --no-deps "${SERVICE_NAME}" || true
  fi
  attempts=$((attempts + 1))
  delay=$(( BASE_DELAY * (2 ** (attempts - 1)) ))
  if (( delay > MAX_DELAY )); then delay="${MAX_DELAY}"; fi
  next=$(( now + delay ))
  if command -v jq >/dev/null 2>&1; then
    save_state_json "$(jq -n \
      --arg d "${remote}" \
      --arg r "${rollback_digest}" \
      --arg t "${TRACK_IMAGE}" \
      --argjson a "${attempts}" \
      --argjson n "${next}" \
      '{desired_digest:$d,current_digest:null,rollback_digest:$r,phase:"backoff",attempts:$a,next_eligible_at:$n,hold:false,last_error:"pull failed",alerted:false,track_image:$t}')"
  fi
  exit 1
fi

if ! compose_cmd up -d --force-recreate --no-deps "${SERVICE_NAME}"; then
  log "error: compose recreate failed; rolling back"
  if [[ -n "${rollback_digest}" ]]; then
    write_env_atomic "${repo}" "${rollback_digest#sha256:}"
    compose_cmd up -d --force-recreate --no-deps "${SERVICE_NAME}" || true
  fi
  attempts=$((attempts + 1))
  delay=$(( BASE_DELAY * (2 ** (attempts - 1)) ))
  if (( delay > MAX_DELAY )); then delay="${MAX_DELAY}"; fi
  next=$(( now + delay ))
  if command -v jq >/dev/null 2>&1; then
    save_state_json "$(jq -n \
      --arg d "${remote}" \
      --arg r "${rollback_digest}" \
      --arg t "${TRACK_IMAGE}" \
      --argjson a "${attempts}" \
      --argjson n "${next}" \
      '{desired_digest:$d,current_digest:null,rollback_digest:$r,phase:"backoff",attempts:$a,next_eligible_at:$n,hold:false,last_error:"recreate failed",alerted:false,track_image:$t}')"
  fi
  exit 1
fi

# Verify running container actually carries the desired digest (brief wait;
# container name / image metadata can lag a second after compose up).
verify_run=""
for _try in 1 2 3 4 5 6 7 8 9 10; do
  verify_run="$(inspect_running_digest || true)"
  if [[ -n "${verify_run}" && "${verify_run}" == "${remote}" ]]; then
    break
  fi
  sleep 1
done
if [[ -z "${verify_run}" || "${verify_run}" != "${remote}" ]]; then
  log "error: post-recreate verify failed (run=${verify_run:-none} desired=${remote}); rolling back"
  if [[ -n "${rollback_digest}" ]]; then
    write_env_atomic "${repo}" "${rollback_digest#sha256:}"
    compose_cmd up -d --force-recreate --no-deps "${SERVICE_NAME}" || true
  fi
  attempts=$((attempts + 1))
  delay=$(( BASE_DELAY * (2 ** (attempts - 1)) ))
  if (( delay > MAX_DELAY )); then delay="${MAX_DELAY}"; fi
  next=$(( now + delay ))
  if command -v jq >/dev/null 2>&1; then
    save_state_json "$(jq -n \
      --arg d "${remote}" \
      --arg r "${rollback_digest}" \
      --arg t "${TRACK_IMAGE}" \
      --argjson a "${attempts}" \
      --argjson n "${next}" \
      '{desired_digest:$d,current_digest:null,rollback_digest:$r,phase:"backoff",attempts:$a,next_eligible_at:$n,hold:false,last_error:"verify failed",alerted:false,track_image:$t}')"
  fi
  exit 1
fi

log "info: updated project=${PROJECT_NAME} to ${repo}@${remote}"
if command -v jq >/dev/null 2>&1; then
  save_state_json "$(jq -n \
    --arg d "${remote}" \
    --arg c "${remote}" \
    --arg t "${TRACK_IMAGE}" \
    '{desired_digest:$d,current_digest:$c,rollback_digest:null,phase:"idle",attempts:0,next_eligible_at:null,hold:false,last_error:null,alerted:false,track_image:$t}')"
fi
exit 0
