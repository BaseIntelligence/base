#!/usr/bin/env bash
# Master container supervisor: public proxy + optional localhost challenge ASGI.
#
# Topology (VAL-MEMB-001/002):
#   base master proxy  :8081          (public path; CMD / args)
#   prism              127.0.0.1:18080
#   agent-challenge    127.0.0.1:18081
#
# Dual-run safe: set BASE_MASTER_EMBED_CHALLENGES=0 to run proxy-only while a
# separate challenge-* Compose service still owns ASGI. Default is embed ON.
#
# Data paths (under master volume /var/lib/base):
#   /var/lib/base/challenges/prism
#   /var/lib/base/challenges/agent-challenge
#
# Shared tokens (file paths; never inline secrets):
#   PRISM_SHARED_TOKEN_FILE (default /run/secrets/prism_shared_token)
#   CHALLENGE_SHARED_TOKEN_FILE (default /run/secrets/agent_challenge_shared_token,
#     falls back to the prism token file when that path is absent and prism exists)
set -euo pipefail

log() {
  printf '%s [master-entrypoint] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2
}

EMBED_ENABLED="${BASE_MASTER_EMBED_CHALLENGES:-1}"
PRISM_HOST="${BASE_MASTER_PRISM_HOST:-127.0.0.1}"
PRISM_PORT="${BASE_MASTER_PRISM_PORT:-18080}"
AC_HOST="${BASE_MASTER_AC_HOST:-127.0.0.1}"
AC_PORT="${BASE_MASTER_AC_PORT:-18081}"

PRISM_DATA_DIR="${BASE_MASTER_PRISM_DATA_DIR:-/var/lib/base/challenges/prism}"
AC_DATA_DIR="${BASE_MASTER_AC_DATA_DIR:-/var/lib/base/challenges/agent-challenge}"

# Operator-supplied overrides merged into the isolated child environments.
# Defaults live beside each challenge's data so they survive image rebuilds.
PRISM_ENV_FILE="${BASE_MASTER_PRISM_ENV_FILE:-${PRISM_DATA_DIR}/embed.env}"
AC_ENV_FILE="${BASE_MASTER_AC_ENV_FILE:-${AC_DATA_DIR}/embed.env}"

# Child PIDs for cleanup (proxy is usually the last foreground wait target).
CHILD_PIDS=()

cleanup() {
  local pid
  for pid in "${CHILD_PIDS[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${CHILD_PIDS[@]:-}"; do
    wait "${pid}" 2>/dev/null || true
  done
}

trap cleanup EXIT INT TERM

embed_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

# Merge operator-supplied KEY=VALUE lines into an isolated child environment.
#
# Embedded challenges start under `env -i` so Prism never sees CHALLENGE_* and
# agent-challenge never sees PRISM_*. That isolation is deliberate, but it also
# dropped every operator setting -- including the Phala attestation switches
# (CHALLENGE_PHALA_ATTESTATION_ENABLED / CHALLENGE_ATTESTED_REVIEW_ENABLED) and
# the eval/review app identities -- because the built-in list was hardcoded with
# no extension point. This is that extension point: allowlisted keys from the
# file are appended AFTER the defaults, so the file wins.
#
# Values are never logged (secrets hygiene); only key names are.
load_challenge_env_file() {
  local -n _target_env="$1"
  local env_file="$2"
  shift 2
  local -a allowed_prefixes=("$@")

  if [[ -z "${env_file}" ]]; then
    return 0
  fi
  if [[ ! -e "${env_file}" ]]; then
    log "no env file at ${env_file}; using built-in defaults only"
    return 0
  fi
  if [[ ! -r "${env_file}" ]]; then
    log "ERROR: env file ${env_file} exists but is not readable"
    return 1
  fi

  local line key value prefix allowed
  local -a accepted=()
  while IFS= read -r line || [[ -n "${line}" ]]; do
    # Trim surrounding whitespace.
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    if [[ -z "${line}" || "${line}" == '#'* ]]; then
      continue
    fi
    line="${line#export }"
    if [[ "${line}" != *=* ]]; then
      continue
    fi
    key="${line%%=*}"
    value="${line#*=}"
    if [[ ! "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
      continue
    fi
    # Strip one layer of matching quotes around the value.
    if [[ "${value}" == \"*\" && "${#value}" -ge 2 ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "${value}" == \'*\' && "${#value}" -ge 2 ]]; then
      value="${value:1:${#value}-2}"
    fi
    allowed=0
    for prefix in "${allowed_prefixes[@]}"; do
      if [[ "${key}" == "${prefix}"* ]]; then
        allowed=1
        break
      fi
    done
    if (( ! allowed )); then
      continue
    fi
    _target_env+=("${key}=${value}")
    accepted+=("${key}")
  done < "${env_file}"

  if (( ${#accepted[@]} )); then
    log "loaded ${#accepted[@]} override(s) from ${env_file}: ${accepted[*]}"
  else
    log "no applicable overrides in ${env_file}"
  fi
}

prepare_challenge_dirs() {
  mkdir -p \
    "${PRISM_DATA_DIR}/tmp" \
    "${AC_DATA_DIR}/agents" \
    "${AC_DATA_DIR}/tmp"
  # Writable for uid 1000 inside the image; ignore when volume already owned.
  chmod 700 "${PRISM_DATA_DIR}" "${AC_DATA_DIR}" 2>/dev/null || true
  chmod 700 "${PRISM_DATA_DIR}/tmp" "${AC_DATA_DIR}/tmp" 2>/dev/null || true
}

resolve_token_file() {
  # $1 preferred path, $2 optional fallback path
  local preferred="${1:-}"
  local fallback="${2:-}"
  if [[ -n "${preferred}" && -f "${preferred}" ]]; then
    printf '%s\n' "${preferred}"
    return 0
  fi
  if [[ -n "${fallback}" && -f "${fallback}" ]]; then
    printf '%s\n' "${fallback}"
    return 0
  fi
  if [[ -n "${preferred}" ]]; then
    printf '%s\n' "${preferred}"
    return 0
  fi
  printf '%s\n' "${fallback}"
}

start_embedded_challenges() {
  prepare_challenge_dirs

  local prism_token_default="/run/secrets/prism_shared_token"
  local ac_token_default="/run/secrets/agent_challenge_shared_token"
  # Also accept shared file under secrets dir used by some compose installs.
  local shared_fallback="/run/secrets/base/challenge_token"

  local prism_token
  prism_token="$(resolve_token_file \
    "${PRISM_SHARED_TOKEN_FILE:-${prism_token_default}}" \
    "${shared_fallback}")"
  local ac_token
  ac_token="$(resolve_token_file \
    "${CHALLENGE_SHARED_TOKEN_FILE:-${ac_token_default}}" \
    "${prism_token}")"

  # Shared non-challenge env for both children (do NOT export CHALLENGE_* or
  # PRISM_* into this shell: base.challenge_sdk rejects foreign prefixes, and
  # CHALLENGE_* leaked into Prism Settings previously raised
  # "Unknown challenge configuration key: CHALLENGE_ARTIFACT_ROOT").
  local tmpdir="${TMPDIR:-${PRISM_DATA_DIR}/tmp}"
  local log_level="${BASE_MASTER_CHALLENGE_LOG_LEVEL:-info}"
  local path_env="${PATH:-/usr/local/bin:/usr/bin:/bin}"
  local home_env="${HOME:-/var/lib/base}"
  local py_path="${PYTHONPATH:-}"

  local -a prism_env=(
    "PATH=${path_env}"
    "HOME=${home_env}"
    "PYTHONDONTWRITEBYTECODE=1"
    "PYTHONUNBUFFERED=1"
    "TMPDIR=${tmpdir}"
    "TEMP=${tmpdir}"
    "TMP=${tmpdir}"
    "PRISM_COMBINED_MODE=${PRISM_COMBINED_MODE:-true}"
    "PRISM_SLUG=${PRISM_SLUG:-prism}"
    "PRISM_DATABASE_URL=${PRISM_DATABASE_URL:-sqlite+aiosqlite:////var/lib/base/challenges/prism/prism.sqlite3}"
    "PRISM_SHARED_TOKEN_FILE=${prism_token}"
    "PRISM_MASTER_BASE_URL=${PRISM_MASTER_BASE_URL:-http://127.0.0.1:8081}"
    "PRISM_RAW_WEIGHT_PUSH_ENABLED=${PRISM_RAW_WEIGHT_PUSH_ENABLED:-true}"
    "PRISM_DOCKER_ENABLED=${PRISM_DOCKER_ENABLED:-false}"
    "PRISM_WORKER_PLANE__ENABLED=${PRISM_WORKER_PLANE__ENABLED:-false}"
    "PRISM_DOCKER_BACKEND=${PRISM_DOCKER_BACKEND:-cli}"
    # HOTPATCH allowlist: OpenRouter plagiarism + worker plane.
# PROD POLICY: Prism eval never runs on master — CPU_REEXEC must stay false;
# miners supply Lium pods; admission_requires_worker default false (unattested/host path).
    "PRISM_WORKER_PLANE__CPU_REEXEC_TEST_MODE=${PRISM_WORKER_PLANE__CPU_REEXEC_TEST_MODE:-false}"
    "PRISM_WORKER_PLANE__ADMISSION_REQUIRES_WORKER=${PRISM_WORKER_PLANE__ADMISSION_REQUIRES_WORKER:-false}"
    "PRISM_WORKER_PLANE__MASTER_BASE_URL=${PRISM_WORKER_PLANE__MASTER_BASE_URL:-http://127.0.0.1:8081}"
    "PRISM_PLAGIARISM_LLM_ENABLED=${PRISM_PLAGIARISM_LLM_ENABLED:-false}"
    "PRISM_PLAGIARISM_LLM_REQUIRED=${PRISM_PLAGIARISM_LLM_REQUIRED:-false}"
    "PRISM_OPENROUTER_API_KEY_FILE=${PRISM_OPENROUTER_API_KEY_FILE:-/run/secrets/openrouter_api_key}"
    "PRISM_OPENROUTER_BASE_URL=${PRISM_OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}"
    "PRISM_OPENROUTER_MODEL=${PRISM_OPENROUTER_MODEL:-x-ai/grok-4.5}"
    "PRISM_ALLOW_INSECURE_SIGNATURES=${PRISM_ALLOW_INSECURE_SIGNATURES:-false}"
    "PRISM_CONSTATION_BASE_URL=${PRISM_CONSTATION_BASE_URL:-http://127.0.0.1:8081}"
  )
  # Optional constation token only when parent set it (avoid empty unknown noise)
  if [[ -n "${PRISM_CONSTATION_INTERNAL_TOKEN:-}" ]]; then
    prism_env+=("PRISM_CONSTATION_INTERNAL_TOKEN=${PRISM_CONSTATION_INTERNAL_TOKEN}")
  fi
  if [[ -n "${py_path}" ]]; then
    prism_env+=("PYTHONPATH=${py_path}")
  fi

  local -a ac_env=(
    "PATH=${path_env}"
    "HOME=${home_env}"
    "PYTHONDONTWRITEBYTECODE=1"
    "PYTHONUNBUFFERED=1"
    "TMPDIR=${AC_DATA_DIR}/tmp"
    "TEMP=${AC_DATA_DIR}/tmp"
    "TMP=${AC_DATA_DIR}/tmp"
    "CHALLENGE_COMBINED_WORKER=${CHALLENGE_COMBINED_WORKER:-true}"
    "CHALLENGE_DATABASE_URL=${CHALLENGE_DATABASE_URL:-sqlite+aiosqlite:////var/lib/base/challenges/agent-challenge/agent-challenge.sqlite3}"
    "CHALLENGE_DATA_DIR=${CHALLENGE_DATA_DIR:-${AC_DATA_DIR}}"
    "CHALLENGE_ARTIFACT_ROOT=${CHALLENGE_ARTIFACT_ROOT:-${AC_DATA_DIR}/agents}"
    "CHALLENGE_SHARED_TOKEN_FILE=${ac_token}"
    "CHALLENGE_MASTER_BASE_URL=${CHALLENGE_MASTER_BASE_URL:-http://127.0.0.1:8081}"
    "CHALLENGE_DOCKER_ENABLED=${CHALLENGE_DOCKER_ENABLED:-false}"
    "CHALLENGE_DOCKER_BACKEND=${CHALLENGE_DOCKER_BACKEND:-cli}"
    "CHALLENGE_SUBMISSION_ENV_ENCRYPTION_KEY_FILE=${CHALLENGE_SUBMISSION_ENV_ENCRYPTION_KEY_FILE:-/run/secrets/base/submission_env_encryption_key}"
  )
  if [[ -n "${py_path}" ]]; then
    ac_env+=("PYTHONPATH=${py_path}")
  fi

  # Operator overrides win over the defaults above. Prefixes stay disjoint per
  # challenge so the env -i isolation is preserved.
  load_challenge_env_file prism_env "${PRISM_ENV_FILE}" \
    PRISM_ PHALA_ DSTACK_ OPENROUTER_API_KEY \
    CHALLENGE_UNATTESTED_EXECUTION CHALLENGE_NO_PHALA NO_PHALA
  load_challenge_env_file ac_env "${AC_ENV_FILE}" \
    CHALLENGE_ BASE_CHALLENGE_ PHALA_ DSTACK_ OPENROUTER_API_KEY

  log "starting embedded prism on ${PRISM_HOST}:${PRISM_PORT}"
  # env -i: isolate prefixes so Prism never sees CHALLENGE_* and AC never sees
  # unrelated PRISM_* secrets as accidental Settings keys.
  env -i "${prism_env[@]}" \
    uvicorn prism_challenge.app:app \
      --host "${PRISM_HOST}" \
      --port "${PRISM_PORT}" \
      --log-level "${log_level}" &
  CHILD_PIDS+=("$!")

  log "starting embedded agent-challenge on ${AC_HOST}:${AC_PORT}"
  env -i "${ac_env[@]}" \
    uvicorn agent_challenge.app:app \
      --host "${AC_HOST}" \
      --port "${AC_PORT}" \
      --log-level "${log_level}" &
  CHILD_PIDS+=("$!")
}

# --- main --------------------------------------------------------------------

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
base-master-entrypoint — supervise master proxy + optional localhost challenges

Default (BASE_MASTER_EMBED_CHALLENGES=1):
  • uvicorn prism_challenge.app:app     127.0.0.1:18080
  • uvicorn agent_challenge.app:app     127.0.0.1:18081
  • then exec remaining args as master (default: base master proxy)

Proxy-only / dual-run with external challenge-* services:
  BASE_MASTER_EMBED_CHALLENGES=0

Ports (override with BASE_MASTER_PRISM_PORT / BASE_MASTER_AC_PORT):
  public proxy   8081
  prism          127.0.0.1:18080
  agent-challenge 127.0.0.1:18081

Data dirs:
  /var/lib/base/challenges/prism
  /var/lib/base/challenges/agent-challenge
EOF
  exit 0
fi

if embed_truthy "${EMBED_ENABLED}"; then
  if ! command -v uvicorn >/dev/null 2>&1; then
    log "ERROR: uvicorn not found; master image must install prism-challenge + agent-challenge"
    exit 127
  fi
  if ! python -c "import prism_challenge.app, agent_challenge.app" 2>/dev/null; then
    log "ERROR: challenge packages not importable; rebuild master image with monorepo packages"
    exit 127
  fi
  start_embedded_challenges
else
  log "BASE_MASTER_EMBED_CHALLENGES=${EMBED_ENABLED}: skipping embedded challenge ASGI"
fi

if [[ "$#" -eq 0 ]]; then
  set -- base master proxy --config config/master.example.yaml
fi

log "starting master process: $*"
# Run master in background so trap cleanup can stop challenges if master exits.
"$@" &
CHILD_PIDS+=("$!")
MASTER_PID="${CHILD_PIDS[-1]}"

# Wait for the master process specifically (not challenge children first).
set +e
wait "${MASTER_PID}"
status=$?
set -e
log "master process exited status=${status}"
exit "${status}"
