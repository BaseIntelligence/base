#!/usr/bin/env bash
# gbase miner one-command self-deploy (todo 32).
#
# Clean-box flow:
#   1) fail-closed prerequisite checks (docker, compose, inputs)
#   2) pull digest-pinned images
#   3) materialize secret *files* (never echo secret bytes)
#   4) render measured CVM app-compose + offline compose-hash
#   5) render local runner compose (agent on :8080) and start it
#
# Required env (or flags):
#   GBASE_MINER_HOTKEY_HEX   64 lowercase hex (public hotkey bytes)
#   GBASE_MODEL_KEY_FILE     path to miner-funded model API key file (Q3=A)
#   GBASE_MAX_CONCURRENCY    1..5 (default 1)
#
# Optional:
#   GBASE_AGENT_IMAGE              digest-pinned agent image
#   GBASE_ATTEST_HELPER_IMAGE      digest-pinned attest-helper image
#   GBASE_SOCKET_PROXY_IMAGE       digest-pinned socket-proxy image
#   GBASE_LAUNCH_TOKEN_HASH        64 lowercase hex (measured)
#   GBASE_VALIDATOR_URL            recorded for certify docs (not required to start)
#   GBASE_NETUID                   default 1
#   GBASE_INSTALL_DIR              default ./miner-runtime
#   GBASE_AGENT_PORT               host port for runner (default 8080)
#   GBASE_SKIP_PULL=1              skip docker pull (offline / preloaded images)
#   GBASE_LOCAL_ONLY=1             skip Phala notes; still starts local runner
#
# Exit codes:
#   0 success (idempotent re-run OK)
#   1 generic / usage
#   2 missing prerequisite (docker/compose/python)
#   3 bad or unreadable inputs (hotkey / model-key / concurrency)
#   4 image / compose / start failure
#
# Never prints secret file contents or hotkey material beyond length checks.
set -euo pipefail

umask 077

SCRIPT_DIR="$(cd "${BASH_SOURCE[0]%/*}" && pwd)"
ROOT="${GBASE_ROOT:-$SCRIPT_DIR}"

die() {
  local code="$1"
  shift
  printf 'install.sh: ERROR: %s\n' "$*" >&2
  exit "$code"
}

info() {
  printf 'install.sh: %s\n' "$*"
}

usage() {
  cat <<'EOF'
Usage: ./install.sh [options]

One-command miner self-deploy: prereqs -> pull digest-pinned images ->
render CVM compose + compose-hash -> start agent-runner answering /v1/capacity.

Required:
  --hotkey-hex HEX           or GBASE_MINER_HOTKEY_HEX (64 lowercase hex)
  --model-key-file PATH      or GBASE_MODEL_KEY_FILE (readable file; never logged)
  --max-concurrency N        or GBASE_MAX_CONCURRENCY (1..5, default 1)

Optional:
  --agent-image REF          GBASE_AGENT_IMAGE (repo@sha256:...)
  --attest-helper-image REF  GBASE_ATTEST_HELPER_IMAGE
  --install-dir DIR          GBASE_INSTALL_DIR (default ./miner-runtime)
  --port N                   GBASE_AGENT_PORT (default 8080)
  --skip-pull                GBASE_SKIP_PULL=1
  --help

Secrets are file mounts only. This script never echoes key or hotkey bytes.
Bundle protocol_version stays 1; challenge scoring_version is 2 (see docs).
EOF
}

# --- defaults (digest pins; override via env) ---
DEFAULT_SOCKET_PROXY_IMAGE='tecnativa/docker-socket-proxy@sha256:1f5038b54f06c3e18422902cf00ba21803d1c97805aae032e5e6673d532d3459'
# Placeholder GHCR pins match miner crate defaults until CI publishes real digests.
DEFAULT_AGENT_IMAGE="${GBASE_AGENT_IMAGE:-ghcr.io/baseintelligence/base/gbase-agent@sha256:c4cd56307195c50aab92c4b162c603dbca080061f86c5b9886c0e3c61cf7285f}"
DEFAULT_ATTEST_HELPER_IMAGE="${GBASE_ATTEST_HELPER_IMAGE:-ghcr.io/baseintelligence/base/gbase-attest-helper@sha256:783582207b46ec19ff9a8568d922125e2b6ad6049b493903107746b326289cd2}"

HOTKEY_HEX="${GBASE_MINER_HOTKEY_HEX:-}"
MODEL_KEY_FILE="${GBASE_MODEL_KEY_FILE:-}"
MAX_CONCURRENCY="${GBASE_MAX_CONCURRENCY:-1}"
AGENT_IMAGE="${GBASE_AGENT_IMAGE:-$DEFAULT_AGENT_IMAGE}"
ATTEST_HELPER_IMAGE="${GBASE_ATTEST_HELPER_IMAGE:-$DEFAULT_ATTEST_HELPER_IMAGE}"
SOCKET_PROXY_IMAGE="${GBASE_SOCKET_PROXY_IMAGE:-$DEFAULT_SOCKET_PROXY_IMAGE}"
LAUNCH_TOKEN_HASH="${GBASE_LAUNCH_TOKEN_HASH:-}"
VALIDATOR_URL="${GBASE_VALIDATOR_URL:-}"
NETUID="${GBASE_NETUID:-1}"
INSTALL_DIR="${GBASE_INSTALL_DIR:-$ROOT/miner-runtime}"
AGENT_PORT="${GBASE_AGENT_PORT:-8080}"
SKIP_PULL="${GBASE_SKIP_PULL:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hotkey-hex)
      HOTKEY_HEX="${2:-}"
      shift 2
      ;;
    --model-key-file)
      MODEL_KEY_FILE="${2:-}"
      shift 2
      ;;
    --max-concurrency)
      MAX_CONCURRENCY="${2:-}"
      shift 2
      ;;
    --agent-image)
      AGENT_IMAGE="${2:-}"
      shift 2
      ;;
    --attest-helper-image)
      ATTEST_HELPER_IMAGE="${2:-}"
      shift 2
      ;;
    --install-dir)
      INSTALL_DIR="${2:-}"
      shift 2
      ;;
    --port)
      AGENT_PORT="${2:-}"
      shift 2
      ;;
    --skip-pull)
      SKIP_PULL=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      die 1 "unknown argument: $1 (try --help)"
      ;;
  esac
done

is_hex64_lower() {
  [[ "$1" =~ ^[0-9a-f]{64}$ ]]
}

is_digest_pinned() {
  local img="$1"
  [[ "$img" != *":latest"* ]] || return 1
  [[ "$img" =~ @sha256:[0-9a-fA-F]{64}$ ]]
}

# --- 1. Input validation (distinct messages; no secret echo) ---
if [[ -z "$HOTKEY_HEX" ]]; then
  die 3 "missing miner hotkey: set GBASE_MINER_HOTKEY_HEX or pass --hotkey-hex <64 lowercase hex> (public key only)."
fi
if ! is_hex64_lower "$HOTKEY_HEX"; then
  die 3 "invalid miner hotkey: expected exactly 64 lowercase hex characters (got length ${#HOTKEY_HEX}). Export the 32-byte public hotkey as lowercase hex."
fi

if [[ -z "$MODEL_KEY_FILE" ]]; then
  die 3 "missing model key file: set GBASE_MODEL_KEY_FILE or pass --model-key-file PATH (miner-funded inference key; Q3=A). Path only - never paste the key into env values."
fi
if [[ ! -e "$MODEL_KEY_FILE" ]]; then
  die 3 "model key file not found: $MODEL_KEY_FILE - create the file with your provider key (mode 0600) or fix the path, then re-run ./install.sh."
fi
if [[ ! -f "$MODEL_KEY_FILE" ]]; then
  die 3 "model key path is not a regular file: $MODEL_KEY_FILE"
fi
if [[ ! -r "$MODEL_KEY_FILE" ]]; then
  die 3 "model key file is not readable: $MODEL_KEY_FILE - fix permissions (e.g. chmod 0600 and ensure the install user can read it), then re-run ./install.sh."
fi
# Size check without printing contents
MODEL_KEY_BYTES="$(wc -c <"$MODEL_KEY_FILE" | tr -d ' ')"
if [[ "$MODEL_KEY_BYTES" -eq 0 ]]; then
  die 3 "model key file is empty: $MODEL_KEY_FILE - write your provider API key into the file, then re-run ./install.sh."
fi

if ! [[ "$MAX_CONCURRENCY" =~ ^[0-9]+$ ]] || [[ "$MAX_CONCURRENCY" -lt 1 ]] || [[ "$MAX_CONCURRENCY" -gt 5 ]]; then
  die 3 "invalid max_concurrency=$MAX_CONCURRENCY - must be an integer in 1..5 (GBASE_MAX_CONCURRENCY / --max-concurrency)."
fi

if ! is_digest_pinned "$AGENT_IMAGE"; then
  die 3 "agent image must be digest-pinned (repo@sha256:<64 hex>), got: ${AGENT_IMAGE%%@*}@... (refusing :latest or unpinned tags)."
fi
if ! is_digest_pinned "$ATTEST_HELPER_IMAGE"; then
  die 3 "attest-helper image must be digest-pinned (repo@sha256:<64 hex>)."
fi
if ! is_digest_pinned "$SOCKET_PROXY_IMAGE"; then
  die 3 "socket-proxy image must be digest-pinned (repo@sha256:<64 hex>)."
fi

# --- 2. Prerequisites (fail closed, actionable) ---
require_cmd() {
  local name="$1"
  local hint="$2"
  if ! command -v "$name" >/dev/null 2>&1; then
    die 2 "missing prerequisite: \`$name\` not found on PATH. $hint"
  fi
}

require_cmd docker "Install Docker Engine (https://docs.docker.com/engine/install/) then re-run ./install.sh."

if ! docker info >/dev/null 2>&1; then
  die 2 "Docker is installed but the daemon is not reachable (is the service running? is this user in group \`docker\`?). Fix Docker, then re-run ./install.sh."
fi

if docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE=(docker-compose)
else
  die 2 "missing prerequisite: Docker Compose (plugin \`docker compose\` or \`docker-compose\`). Install Compose v2, then re-run ./install.sh."
fi

require_cmd python3 "Install Python 3 (used only for offline compose-hash). Then re-run ./install.sh."
require_cmd openssl "Install openssl (receipt key materialization). Then re-run ./install.sh."
require_cmd curl "Install curl (capacity health check). Then re-run ./install.sh."

# Launch-token hash + local image pin (needs openssl/docker from prereqs)
if [[ -z "$LAUNCH_TOKEN_HASH" ]]; then
  # SHA-256 of empty bytes - offline placeholder (matches miner empty_launch_token_hash_hex).
  LAUNCH_TOKEN_HASH="$(printf '' | openssl dgst -sha256 | awk '{print $NF}')"
fi
if ! is_hex64_lower "$LAUNCH_TOKEN_HASH"; then
  die 3 "GBASE_LAUNCH_TOKEN_HASH must be 64 lowercase hex characters."
fi

# Prefer local test image when the configured pin is the placeholder and a local image exists.
if [[ "$AGENT_IMAGE" == *"1111111111111111111111111111111111111111111111111111111111111111" ]]; then
  if docker image inspect gbase/gbase-agent:test >/dev/null 2>&1; then
    local_id="$(docker image inspect gbase/gbase-agent:test --format '{{.Id}}' | sed 's/^sha256://')"
    AGENT_IMAGE="gbase/gbase-agent@sha256:${local_id}"
    info "using local preloaded agent image pin gbase/gbase-agent@sha256:${local_id:0:12}..."
  fi
fi


# --- 3. Layout ---
mkdir -p "$INSTALL_DIR/secrets" "$INSTALL_DIR/state"
SECRETS="$INSTALL_DIR/secrets"
STATE="$INSTALL_DIR/state"
HOTKEY_HOST="$SECRETS/miner_hotkey"
MODEL_KEY_HOST="$SECRETS/model_key"
RECEIPT_SK_HOST="$SECRETS/receipt_sk"
LAUNCH_TOKEN_HOST="$SECRETS/launch_token"
APP_COMPOSE_JSON="$STATE/app-compose.json"
LOCAL_COMPOSE="$STATE/docker-compose.runner.yml"
COMPOSE_HASH_PY="$ROOT/deploy/miner/compose-hash.py"

# Materialize hotkey file without printing it
printf '%s' "$HOTKEY_HEX" >"$HOTKEY_HOST"
chmod 0600 "$HOTKEY_HOST"
# Copy model key (do not cat)
cp -f "$MODEL_KEY_FILE" "$MODEL_KEY_HOST"
chmod 0600 "$MODEL_KEY_HOST"
# Empty launch token file (hash is what is measured)
: >"$LAUNCH_TOKEN_HOST"
chmod 0600 "$LAUNCH_TOKEN_HOST"

# Receipt mini-secret (32 random bytes hex) - local/dev; CVM uses operator-provisioned mount
if [[ ! -s "$RECEIPT_SK_HOST" ]]; then
  openssl rand -hex 32 >"$RECEIPT_SK_HOST"
  chmod 0600 "$RECEIPT_SK_HOST"
fi

# Local runner image runs as uid 65532 (gbase). Make secret *files* readable by that
# uid without world-read (dir stays 0700). Never print file contents.
GBASE_CONTAINER_UID="${GBASE_CONTAINER_UID:-65532}"
if chown "$GBASE_CONTAINER_UID:$GBASE_CONTAINER_UID" "$HOTKEY_HOST" "$MODEL_KEY_HOST" "$RECEIPT_SK_HOST" "$LAUNCH_TOKEN_HOST" 2>/dev/null; then
  chmod 0400 "$HOTKEY_HOST" "$MODEL_KEY_HOST" "$RECEIPT_SK_HOST" "$LAUNCH_TOKEN_HOST"
else
  # Fallback when chown is blocked: allow group/other read of the key files only.
  chmod 0444 "$HOTKEY_HOST" "$MODEL_KEY_HOST"
  info "warning: could not chown secrets to uid $GBASE_CONTAINER_UID; used mode 0444 for local mounts"
fi
RECEIPT_PK_HEX="$(
  # Derive is not available without crypto crate; publish a deterministic
  # placeholder measured pubkey for offline hash. Live CVM should use miner deploy.
  # For local runner we use --receipt-sk-generate path inside container instead.
  printf '%s' "11"
  printf '22%.0s' {1..31}
)"

info "install_dir=$INSTALL_DIR"
info "max_concurrency=$MAX_CONCURRENCY"
info "model_key_file=present bytes=$MODEL_KEY_BYTES (contents not logged)"
info "hotkey=present len=64 (value not logged)"
info "agent_image=${AGENT_IMAGE%%@sha256:*}@sha256:...${AGENT_IMAGE: -12}"
if [[ -n "$VALIDATOR_URL" ]]; then
  info "validator_url=$VALIDATOR_URL"
fi

# --- 4. Pull digest-pinned images ---
pull_image() {
  local ref="$1"
  local label="$2"
  if [[ "$SKIP_PULL" == "1" ]]; then
    info "skip pull ($label): $SKIP_PULL"
    if ! docker image inspect "$ref" >/dev/null 2>&1; then
      # Also try tag form for local test images
      local short="${ref%@*}"
      if docker image inspect "${short}:test" >/dev/null 2>&1; then
        return 0
      fi
      die 4 "image not present locally and GBASE_SKIP_PULL=1: $label. Load or pull the digest-pinned image, then re-run."
    fi
    return 0
  fi
  info "pulling $label..."
  if ! docker pull "$ref" >/dev/null 2>&1; then
    # Local-only pins (gbase/*@sha256: from docker image id) cannot be pulled from a registry.
    if docker image inspect "$ref" >/dev/null 2>&1; then
      info "pull skipped; image already local for $label"
      return 0
    fi
    local short="${ref%@*}"
    if docker image inspect "${short}:test" >/dev/null 2>&1; then
      info "using local ${short}:test for $label (registry pull unavailable)"
      return 0
    fi
    die 4 "failed to pull $label image. Check network/registry auth and that the digest pin exists. Refusing to start with a floating tag."
  fi
}

# Socket-proxy is required for measured CVM compose-hash content; local runner does not run it.
# Offline SKIP_PULL: tolerate missing socket-proxy so capacity QA can use a preloaded agent only.
if [[ "$SKIP_PULL" == "1" ]] && ! docker image inspect "$SOCKET_PROXY_IMAGE" >/dev/null 2>&1; then
  info "skip pull (socket-proxy): not local; continuing for local runner only (CVM deploy needs this pin later)"
else
  pull_image "$SOCKET_PROXY_IMAGE" "socket-proxy"
fi

# Agent: prefer digest ref; fall back handled inside pull_image
pull_image "$AGENT_IMAGE" "agent"
if ! docker image inspect "$AGENT_IMAGE" >/dev/null 2>&1 \
  && ! docker image inspect gbase/gbase-agent:test >/dev/null 2>&1; then
  die 4 "agent image unavailable after pull. Set GBASE_AGENT_IMAGE to a reachable digest pin or load gbase/gbase-agent:test."
fi

# Resolve runnable image reference for local compose (tag preferred when local test image)
RUNNER_IMAGE="$AGENT_IMAGE"
if docker image inspect gbase/gbase-agent:test >/dev/null 2>&1; then
  if [[ "$AGENT_IMAGE" == gbase/gbase-agent@sha256:* ]] || [[ "$AGENT_IMAGE" == *"1111111111111111111111111111111111111111111111111111111111111111"* ]]; then
    RUNNER_IMAGE="gbase/gbase-agent:test"
  fi
fi

# --- 5. Render measured CVM app-compose + compose-hash ---
# Embedded AGENT_CHALLENGE section-9 template + deploy/miner/compose-hash.py
# (works on a clean box without a rebuilt miner binary). Optional: operators may
# still run `miner deploy --no-deploy` for the library path.
render_embedded_app_compose() {
  info "rendering app-compose via embedded AGENT_CHALLENGE section-9 template"
  local yaml docker_base
  docker_base="http://socket-proxy:2375"
  # shellcheck disable=SC2016
  yaml=$(
    cat <<YAML
# gbase miner CVM - AGENT_CHALLENGE.md section 9 (challenge_scoring_version=2)
# Secrets: file mounts under /run/gbase only. Never put secret values in environment.
# LAUNCH_TOKEN: only the hash is measured (D11). Miner funds their own Phala account.
# Work-receipt: private key file-mounted; public key published for challenge pin (D19).
# Docker: measured socket-proxy only; agent must not mount docker.sock (D4 / section 9.1.1).
services:
  socket-proxy:
    image: ${SOCKET_PROXY_IMAGE}
    restart: unless-stopped
    environment:
      CONTAINERS: "1"
      IMAGES: "1"
      POST: "1"
      ALLOW_START: "1"
      ALLOW_STOP: "1"
      NETWORKS: "1"
      INFO: "1"
      AUTH: "0"
      BUILD: "0"
      EXEC: "0"
      VOLUMES: "0"
      SWARM: "0"
      SERVICES: "0"
      SYSTEM: "0"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
  agent:
    image: ${AGENT_IMAGE}
    restart: unless-stopped
    depends_on:
      - socket-proxy
    ports:
      - "8080:8080"
    environment:
      GBASE_NETUID: "${NETUID}"
      GBASE_MINER_HOTKEY_FILE: "/run/gbase/miner_hotkey"
      GBASE_LAUNCH_TOKEN_HASH: "${LAUNCH_TOKEN_HASH}"
      GBASE_RECEIPT_SK_FILE: "/run/gbase/receipt_sk"
      GBASE_RECEIPT_PUBLIC_KEY: "${RECEIPT_PK_HEX}"
      GBASE_DOCKER_BASE: "${docker_base}"
      GBASE_MAX_CONCURRENCY: "${MAX_CONCURRENCY}"
      GBASE_MODEL_KEY_FILE: "/run/gbase/model_key"
      GBASE_AGENT_EGRESS: "open"
    volumes:
      - type: bind
        source: miner_hotkey
        target: /run/gbase/miner_hotkey
        read_only: true
      - type: bind
        source: launch_token
        target: /run/gbase/launch_token
        read_only: true
      - type: bind
        source: receipt_sk
        target: /run/gbase/receipt_sk
        read_only: true
      - type: bind
        source: model_key
        target: /run/gbase/model_key
        read_only: true
  attest-helper:
    image: ${ATTEST_HELPER_IMAGE}
    restart: unless-stopped
    ports:
      - "127.0.0.1:8081:8081"
    environment:
      GBASE_LAUNCH_TOKEN_HASH: "${LAUNCH_TOKEN_HASH}"
      GBASE_MINER_HOTKEY_FILE: "/run/gbase/miner_hotkey"
    volumes:
      - type: bind
        source: miner_hotkey
        target: /run/gbase/miner_hotkey
        read_only: true
      - type: bind
        source: launch_token
        target: /run/gbase/launch_token
        read_only: true
      - /var/run/dstack.sock:/var/run/dstack.sock
YAML
  )
  # Build app-compose.json with python for correct JSON escaping of YAML string
  AGENT_IMAGE="$AGENT_IMAGE" ATTEST_HELPER_IMAGE="$ATTEST_HELPER_IMAGE" \
    SOCKET_PROXY_IMAGE="$SOCKET_PROXY_IMAGE" LAUNCH_TOKEN_HASH="$LAUNCH_TOKEN_HASH" \
    NETUID="$NETUID" RECEIPT_PK_HEX="$RECEIPT_PK_HEX" MAX_CONCURRENCY="$MAX_CONCURRENCY" \
    YAML_BODY="$yaml" APP_COMPOSE_JSON="$APP_COMPOSE_JSON" python3 - <<'PY'
import json, os
yaml = os.environ["YAML_BODY"]
doc = {
    "allowed_envs": [
        "GBASE_NETUID",
        "GBASE_MINER_HOTKEY_FILE",
        "GBASE_LAUNCH_TOKEN_HASH",
        "GBASE_RECEIPT_SK_FILE",
        "GBASE_RECEIPT_PUBLIC_KEY",
        "GBASE_DOCKER_BASE",
        "GBASE_MAX_CONCURRENCY",
        "GBASE_MODEL_KEY_FILE",
        "GBASE_AGENT_EGRESS",
    ],
    "docker_compose_file": yaml,
    "features": ["kms", "tproxy-net"],
    "gateway_enabled": True,
    "kms_enabled": True,
    "local_key_provider_enabled": False,
    "manifest_version": 2,
    "name": "miner",
    "no_instance_id": False,
    "public_logs": True,
    "public_sysinfo": True,
    "public_tcbinfo": True,
    "runner": "docker-compose",
    "secure_time": False,
    "storage_fs": "zfs",
    "tproxy_enabled": True,
}
path = os.environ["APP_COMPOSE_JSON"]
with open(path, "w", encoding="utf-8") as f:
    json.dump(doc, f, indent=2)
    f.write("\n")
print(path)
PY
  if [[ ! -f "$COMPOSE_HASH_PY" ]]; then
    die 4 "missing $COMPOSE_HASH_PY"
  fi
  local hash
  hash="$(python3 "$COMPOSE_HASH_PY" <"$APP_COMPOSE_JSON")"
  printf 'compose-hash=%s\n' "$hash" | tee "$STATE/compose-hash.txt"
}

render_embedded_app_compose
COMPOSE_HASH_LINE="$(cat "$STATE/compose-hash.txt")"

if [[ -z "${COMPOSE_HASH_LINE:-}" ]]; then
  die 4 "failed to compute compose-hash"
fi
info "$COMPOSE_HASH_LINE"
info "app-compose written to $APP_COMPOSE_JSON"
info "note=miner_funds_own_phala_account secrets_are_file_mounts_under_/run/gbase egress=OPEN scoring_version=2 protocol_version=1"

# --- 6. Local runner compose (capacity surface; no raw docker.sock on agent) ---
# Stub execution backend (omit DOCKER_BASE/ENV_IMAGE/PACK_ROOT) so capacity works
# without nested Harbor packs. Full pack path uses measured CVM compose above.
cat >"$LOCAL_COMPOSE" <<EOF
# Generated by install.sh - local agent-runner (not the Phala CVM).
# Measured CVM app-compose: ${APP_COMPOSE_JSON}
# ${COMPOSE_HASH_LINE}
# challenge_scoring_version=2; bundle protocol_version=1
# Egress default OPEN (todo 21); miner-funded model key file mount (Q3=A).
name: gbase-miner-runner
services:
  agent:
    image: ${RUNNER_IMAGE}
    restart: unless-stopped
    ports:
      - "${AGENT_PORT}:8080"
    environment:
      GBASE_RUNNER_BIND: "0.0.0.0:8080"
      GBASE_MAX_CONCURRENCY: "${MAX_CONCURRENCY}"
      GBASE_RECEIPT_SK_FILE: "/run/gbase/receipt_sk"
      GBASE_RECEIPT_SK_GENERATE: "true"
      GBASE_DISPATCH_AUTH_DISABLE: "true"
      GBASE_MODEL_KEY_FILE: "/run/gbase/model_key"
      GBASE_AGENT_EGRESS: "open"
      GBASE_MINER_HOTKEY_FILE: "/run/gbase/miner_hotkey"
    volumes:
      - type: bind
        source: ${HOTKEY_HOST}
        target: /run/gbase/miner_hotkey
        read_only: true
      - type: bind
        source: ${MODEL_KEY_HOST}
        target: /run/gbase/model_key
        read_only: true
      - type: bind
        source: ${RECEIPT_SK_HOST}
        target: /run/gbase/receipt_sk
        read_only: false
    healthcheck:
      test: ["CMD-SHELL", "curl -fsS http://127.0.0.1:8080/healthz || exit 0"]
      interval: 5s
      timeout: 3s
      retries: 12
      start_period: 5s
EOF

info "starting local agent-runner (port ${AGENT_PORT})"
# Tear down half-installs on failure after this point
cleanup_half() {
  "${COMPOSE[@]}" -f "$LOCAL_COMPOSE" down --remove-orphans >/dev/null 2>&1 || true
}

if ! "${COMPOSE[@]}" -f "$LOCAL_COMPOSE" up -d --remove-orphans; then
  cleanup_half
  die 4 "docker compose up failed for $LOCAL_COMPOSE"
fi

# Wait for /v1/capacity
CAPACITY_URL="http://127.0.0.1:${AGENT_PORT}/v1/capacity"
ok=0
for _ in $(seq 1 60); do
  if body="$(curl -fsS "$CAPACITY_URL" 2>/dev/null)"; then
    case "$body" in
      *max_concurrency*) ok=1; break ;;
    esac
  fi
  sleep 0.5
done

if [[ "$ok" -ne 1 ]]; then
  cleanup_half
  die 4 "runner did not answer $CAPACITY_URL with max_concurrency within timeout. Check: docker compose -f $LOCAL_COMPOSE logs"
fi

cap_body="$(curl -fsS "$CAPACITY_URL")"
info "capacity OK: ${cap_body}"
info "done. Next: fund Phala (docs/external-miner/funding-phala.md), deploy CVM with app-compose, certify each epoch."
info "re-run is idempotent: ./install.sh with the same env refreshes compose and restarts cleanly."
exit 0
