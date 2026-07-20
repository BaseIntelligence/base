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

  export PRISM_COMBINED_MODE="${PRISM_COMBINED_MODE:-true}"
  export PRISM_SLUG="${PRISM_SLUG:-prism}"
  export PRISM_DATABASE_URL="${PRISM_DATABASE_URL:-sqlite+aiosqlite:////var/lib/base/challenges/prism/prism.sqlite3}"
  export PRISM_SHARED_TOKEN_FILE="${prism_token}"
  export PRISM_MASTER_BASE_URL="${PRISM_MASTER_BASE_URL:-http://127.0.0.1:8081}"
  export PRISM_RAW_WEIGHT_PUSH_ENABLED="${PRISM_RAW_WEIGHT_PUSH_ENABLED:-true}"
  export PRISM_DOCKER_ENABLED="${PRISM_DOCKER_ENABLED:-false}"
  export PRISM_WORKER_PLANE__ENABLED="${PRISM_WORKER_PLANE__ENABLED:-false}"
  export PRISM_DOCKER_BACKEND="${PRISM_DOCKER_BACKEND:-cli}"
  # Eval/static gates need writable non-noexec temp under the data volume.
  export TMPDIR="${TMPDIR:-${PRISM_DATA_DIR}/tmp}"
  export TEMP="${TEMP:-${TMPDIR}}"
  export TMP="${TMP:-${TMPDIR}}"

  export CHALLENGE_COMBINED_WORKER="${CHALLENGE_COMBINED_WORKER:-true}"
  export CHALLENGE_DATABASE_URL="${CHALLENGE_DATABASE_URL:-sqlite+aiosqlite:////var/lib/base/challenges/agent-challenge/agent-challenge.sqlite3}"
  export CHALLENGE_DATA_DIR="${CHALLENGE_DATA_DIR:-${AC_DATA_DIR}}"
  export CHALLENGE_ARTIFACT_ROOT="${CHALLENGE_ARTIFACT_ROOT:-${AC_DATA_DIR}/agents}"
  export CHALLENGE_SHARED_TOKEN_FILE="${ac_token}"
  export CHALLENGE_MASTER_BASE_URL="${CHALLENGE_MASTER_BASE_URL:-http://127.0.0.1:8081}"
  export CHALLENGE_DOCKER_ENABLED="${CHALLENGE_DOCKER_ENABLED:-false}"
  export CHALLENGE_DOCKER_BACKEND="${CHALLENGE_DOCKER_BACKEND:-cli}"

  log "starting embedded prism on ${PRISM_HOST}:${PRISM_PORT}"
  uvicorn prism_challenge.app:app \
    --host "${PRISM_HOST}" \
    --port "${PRISM_PORT}" \
    --log-level "${BASE_MASTER_CHALLENGE_LOG_LEVEL:-info}" &
  CHILD_PIDS+=("$!")

  log "starting embedded agent-challenge on ${AC_HOST}:${AC_PORT}"
  uvicorn agent_challenge.app:app \
    --host "${AC_HOST}" \
    --port "${AC_PORT}" \
    --log-level "${BASE_MASTER_CHALLENGE_LOG_LEVEL:-info}" &
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
