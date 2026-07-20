#!/usr/bin/env bash
# One-command independent validator Compose install (supported shipping path).
# Creates protected config + protocol identity, then runs
# `docker compose up -d` from validator-only artifacts (no master source,
# master PostgreSQL, or challenge services). Default profile is weight-only:
# GET https://chain.joinbase.ai/v1/weights/latest + set_weights (when gated);
# challenge execution adapters are OFF (master is sole writer).
# Host docker.sock is mounted into the agent container (optional migration
# prep only — not for challenge control-plane). Docker Compose is the only
# required runtime for new installs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.validator.yml"

usage() {
  cat <<'EOF'
Usage: install-validator.sh --master-url URL [options]

Agent-only install: validators NEVER run master, PostgreSQL control plane, or
challenge services. Shipping default is weight-only against the public master:
  GET {master}/v1/weights/latest  then  set_weights (own wallet, when gated on)
Challenge execution adapters default OFF (no submissions/leaderboard writer on
the validator). Optional future audit re-exec is non-write only and must be
enabled explicitly in validator.yaml. docker.sock is optional migration prep,
not challenge control-plane.

Required:
  --master-url URL           Absolute Base master coordination API URL (http/https).
                             This is master_url (register/heartbeat + weights).
                             Public network Base master API (shipping default):
                               https://chain.joinbase.ai
                             Local smoke only:
                               http://127.0.0.1:3180
                             Verify /health returns role=master / base-master
                             before using any public hostname.

Options:
  --project-name NAME        Unique Compose project (default: base-mission-validator)
  --state-dir DIR            Operator state directory (default: XDG state path)
  --wallet-name NAME         Protocol identity wallet name (default: validator)
  --wallet-hotkey NAME       Protocol identity hotkey name (default: default)
  --display-name TEXT        Optional public display name
  --capabilities CSV         Capability list (default: cpu)
  --image-repository REPO    Validator image repository pin
  --image-digest DIGEST      Validator image sha256 digest (64 hex)
  --submit-on-chain          Enable on-chain submission (requires wallet mount)
  --copy-artifacts DIR       Also copy validator Compose artifacts into DIR
                             (source-free host directory for re-install)
  --auto-update              Enable host-side image auto-update (default: ON)
  --no-auto-update           Opt out of host-side digest-tracked auto-update
  --track-image REPO:TAG     Mutable track tag resolved to digest each tick
                             (default: ghcr.io/baseintelligence/base-validator-runtime:latest)
  --image-update-interval N  systemd timer seconds (default: 90; range 60-120)
  -h, --help

Environment overrides:
  VALIDATOR_MASTER_URL
  BASE_VALIDATOR_IMAGE_REPOSITORY / BASE_VALIDATOR_IMAGE_DIGEST
  BASE_VALIDATOR_LOCAL_IMAGE
  BASE_VALIDATOR_AUTO_UPDATE          (0/false to default off; default ON)
  BASE_VALIDATOR_TRACK_IMAGE
  COMPOSE_PROJECT_NAME
EOF
}

PROJECT_NAME="${COMPOSE_PROJECT_NAME:-base-mission-validator}"
MASTER_URL="${VALIDATOR_MASTER_URL:-}"
STATE_DIR="${BASE_VALIDATOR_STATE_DIR:-}"
WALLET_NAME="validator"
WALLET_HOTKEY="default"
DISPLAY_NAME=""
CAPABILITIES="cpu"
SUBMIT_ON_CHAIN=0
COPY_ARTIFACTS=""
IMAGE_REPO="${BASE_VALIDATOR_IMAGE_REPOSITORY:-}"
IMAGE_DIGEST="${BASE_VALIDATOR_IMAGE_DIGEST:-}"
# Auto-update ON by default (host-side timer; agent mounts docker.sock separately
# for later challenges-on-validator migration prep — auto-update still host-side).
ENABLE_AUTO_UPDATE=1
case "${BASE_VALIDATOR_AUTO_UPDATE:-1}" in
  0|false|FALSE|no|NO|off|OFF) ENABLE_AUTO_UPDATE=0 ;;
esac
TRACK_IMAGE="${BASE_VALIDATOR_TRACK_IMAGE:-ghcr.io/baseintelligence/base-validator-runtime:latest}"
IMAGE_UPDATE_INTERVAL="${BASE_VALIDATOR_IMAGE_UPDATE_INTERVAL:-90}"
# Detect host docker.sock group so the non-root agent (uid 1000) can open the
# mounted socket (group_add), same pattern as the master installer.
DOCKER_GID="${BASE_DOCKER_GID:-}"
if [[ -z "${DOCKER_GID}" ]]; then
  if [[ -S /var/run/docker.sock ]]; then
    DOCKER_GID="$(stat -c '%g' /var/run/docker.sock 2>/dev/null || true)"
  fi
  DOCKER_GID="${DOCKER_GID:-987}"
fi
DOCKER_SOCKET="${BASE_DOCKER_SOCKET:-/var/run/docker.sock}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --master-url)
      MASTER_URL="$2"
      shift 2
      ;;
    --project-name)
      PROJECT_NAME="$2"
      shift 2
      ;;
    --state-dir)
      STATE_DIR="$2"
      shift 2
      ;;
    --wallet-name)
      WALLET_NAME="$2"
      shift 2
      ;;
    --wallet-hotkey)
      WALLET_HOTKEY="$2"
      shift 2
      ;;
    --display-name)
      DISPLAY_NAME="$2"
      shift 2
      ;;
    --capabilities)
      CAPABILITIES="$2"
      shift 2
      ;;
    --image-repository)
      IMAGE_REPO="$2"
      shift 2
      ;;
    --image-digest)
      IMAGE_DIGEST="$2"
      shift 2
      ;;
    --submit-on-chain)
      SUBMIT_ON_CHAIN=1
      shift
      ;;
    --copy-artifacts)
      COPY_ARTIFACTS="$2"
      shift 2
      ;;
    --auto-update)
      ENABLE_AUTO_UPDATE=1
      shift
      ;;
    --no-auto-update)
      ENABLE_AUTO_UPDATE=0
      shift
      ;;
    --track-image)
      TRACK_IMAGE="$2"
      shift 2
      ;;
    --image-update-interval)
      IMAGE_UPDATE_INTERVAL="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${MASTER_URL}" ]]; then
  echo "validator install requires --master-url (absolute http/https Base master URL)" >&2
  echo "Validators never run master. Point --master-url at the Base master/coordination API" >&2
  echo "this operator actually uses. Public network Base master API:" >&2
  echo "  https://chain.joinbase.ai  (verify GET /health role=master)" >&2
  echo "Local disposable smoke only: http://127.0.0.1:<port>" >&2
  exit 2
fi

# Reject empty or clearly invalid master URLs early (VAL-SDK-086).
# master_url is the Base master coordination root only (register/heartbeat/pull/
# result + weights when the master hosts both). It is never defaulted to a public
# IP inventory. Public Settings defaults for registry/weights recommend the
# public network Base master API: https://chain.joinbase.ai
case "${MASTER_URL}" in
  http://*|https://*) ;;
  *)
    echo "validator.agent.master_url must be an absolute http(s) URL, got: ${MASTER_URL}" >&2
    exit 2
    ;;
esac
if [[ "${MASTER_URL}" == "http://localhost"* || "${MASTER_URL}" == "http://127.0.0.1"* ]]; then
  # Loopback is allowed for disposable local smoke masters only. Empty/self
  # defaults are never invented: master URL must be explicit.
  :
fi

if [[ -z "${STATE_DIR}" ]]; then
  STATE_DIR="${XDG_STATE_HOME:-${HOME}/.local/state}/base-compose/${PROJECT_NAME}"
fi

SECRETS_DIR="${STATE_DIR}/secrets"
CONFIG_DIR="${STATE_DIR}/config"
IDENTITY_DIR="${STATE_DIR}/identity"
ARTIFACTS_DIR="${STATE_DIR}/artifacts"
mkdir -p "${SECRETS_DIR}" "${CONFIG_DIR}" "${IDENTITY_DIR}" "${ARTIFACTS_DIR}"
chmod 700 "${STATE_DIR}" "${SECRETS_DIR}" "${CONFIG_DIR}" "${IDENTITY_DIR}" "${ARTIFACTS_DIR}"

_random_token() {
  if command -v openssl >/dev/null 2>&1; then
    openssl rand -hex 32
  else
    head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n'
  fi
}

BROKER_TOKEN_FILE="${SECRETS_DIR}/broker_token"
CONFIG_FILE="${CONFIG_DIR}/validator.yaml"
HOTKEY_PUB_FILE="${CONFIG_DIR}/hotkey.ss58"

if [[ ! -f "${BROKER_TOKEN_FILE}" ]]; then
  umask 077
  _random_token >"${BROKER_TOKEN_FILE}"
fi
# Container runs as uid 1000; keep host parent 0700 and make mounted files readable.
chmod 644 "${BROKER_TOKEN_FILE}"

_resolve_local_digest() {
  local image_ref="$1"
  docker image inspect --format '{{.Id}}' "${image_ref}" 2>/dev/null \
    | sed -E 's/^sha256://'
}

_tag_with_digest() {
  local source_ref="$1"
  local target_repo="$2"
  local digest
  digest="$(_resolve_local_digest "${source_ref}")"
  if [[ -z "${digest}" ]]; then
    return 1
  fi
  docker tag "${source_ref}" "${target_repo}:compose" 2>/dev/null || true
  docker tag "${source_ref}" "${target_repo}@sha256:${digest}" 2>/dev/null || true
  printf '%s' "${digest}"
}

if [[ -z "${IMAGE_REPO}" || -z "${IMAGE_DIGEST}" ]]; then
  if digest="$(_tag_with_digest "${BASE_VALIDATOR_LOCAL_IMAGE:-base-sdk-review-validator-runtime:local}" "mission/base-validator-runtime")"; then
    IMAGE_REPO="mission/base-validator-runtime"
    IMAGE_DIGEST="${digest}"
  elif digest="$(_tag_with_digest "platform-validator:ci-test" "mission/base-validator-runtime")"; then
    IMAGE_REPO="mission/base-validator-runtime"
    IMAGE_DIGEST="${digest}"
  else
    echo "BASE_VALIDATOR_IMAGE_REPOSITORY/DIGEST unset and no local validator image found." >&2
    echo "Build the validator runtime image or set BASE_VALIDATOR_IMAGE_* pins." >&2
    exit 1
  fi
fi

# Protocol identity must be a real directory tree (not a host symlink bounce).
# Under uid 1000 + read-only identity mount, a symlink whose parent is mode
# 0700 (or not traversable by the container user) fails bittensor wallet load.
# Prefer a real directory with parents at least mode 0755 for container reads.
if [[ -L "${IDENTITY_DIR}" ]]; then
  echo "warning: BASE_VALIDATOR protocol identity path is a symlink (${IDENTITY_DIR});" >&2
  echo "  bind a real directory readable by uid 1000 (parent dirs typically mode 755)." >&2
fi

# Create a disposable protocol-identity wallet when none exists.
_hotkey_exists() {
  [[ -f "${IDENTITY_DIR}/${WALLET_NAME}/hotkeys/${WALLET_HOTKEY}" ]]
}

_gen_wallet_py() {
  # $1 = wallet root path inside the Python process
  cat <<'PY'
import sys
from pathlib import Path
import bittensor as bt

wallet_path = Path(sys.argv[1])
wallet_name = sys.argv[2]
wallet_hotkey = sys.argv[3]
wallet_path.mkdir(parents=True, exist_ok=True)
wallet = bt.Wallet(name=wallet_name, hotkey=wallet_hotkey, path=str(wallet_path))
wallet.create_if_non_existent(coldkey_use_password=False, hotkey_use_password=False)
print(wallet.hotkey.ss58_address)
# Ensure non-root container uid can read protocol identity mounts.
for path in [wallet_path, *wallet_path.rglob("*")]:
    mode = 0o755 if path.is_dir() else 0o644
    try:
        path.chmod(mode)
    except OSError:
        pass
PY
}

if ! _hotkey_exists; then
  GEN_OK=0
  # Prefer host Python with bittensor when available.
  if command -v python3 >/dev/null 2>&1 \
    && python3 -c 'import bittensor' >/dev/null 2>&1; then
    if _gen_wallet_py | python3 - "${IDENTITY_DIR}" "${WALLET_NAME}" "${WALLET_HOTKEY}" >/dev/null; then
      GEN_OK=1
    fi
  fi
  if [[ "${GEN_OK}" -ne 1 ]]; then
    # Generate wallet inside the validator image (has bittensor).
    GEN_SCRIPT="$(mktemp)"
    _gen_wallet_py >"${GEN_SCRIPT}"
    if docker run --rm \
      -u 0:0 \
      -v "${IDENTITY_DIR}:/wallets" \
      -v "${GEN_SCRIPT}:/tmp/gen_wallet.py:ro" \
      --entrypoint python \
      "${IMAGE_REPO}@sha256:${IMAGE_DIGEST}" \
      /tmp/gen_wallet.py /wallets "${WALLET_NAME}" "${WALLET_HOTKEY}" >/dev/null; then
      GEN_OK=1
    fi
    rm -f "${GEN_SCRIPT}"
  fi
  if [[ "${GEN_OK}" -ne 1 ]]; then
    echo "failed to create protocol identity wallet under ${IDENTITY_DIR}" >&2
    exit 1
  fi
  # Host-side chmod in case the generator ran as container root.
  chmod -R a+rX "${IDENTITY_DIR}" || true
fi

# Capture public hotkey without printing private material.
HOTKEY_SS58=""
if command -v python3 >/dev/null 2>&1 && python3 -c 'import bittensor' >/dev/null 2>&1; then
  HOTKEY_SS58="$(
    python3 - <<PY
import bittensor as bt
w = bt.Wallet(name="${WALLET_NAME}", hotkey="${WALLET_HOTKEY}", path="${IDENTITY_DIR}")
print(w.hotkey.ss58_address)
PY
  )"
fi
if [[ -z "${HOTKEY_SS58}" ]]; then
  HOTKEY_SS58="$(
    docker run --rm \
      -v "${IDENTITY_DIR}:/wallets:ro" \
      --entrypoint python \
      "${IMAGE_REPO}@sha256:${IMAGE_DIGEST}" \
      -c "import bittensor as bt; w=bt.Wallet(name='${WALLET_NAME}', hotkey='${WALLET_HOTKEY}', path='/wallets'); print(w.hotkey.ss58_address)"
  )"
fi
if [[ -z "${HOTKEY_SS58}" ]]; then
  echo "failed to resolve public protocol hotkey for ${WALLET_NAME}/${WALLET_HOTKEY}" >&2
  exit 1
fi
printf '%s\n' "${HOTKEY_SS58}" >"${HOTKEY_PUB_FILE}"
chmod 644 "${HOTKEY_PUB_FILE}"

# Render capability list as YAML array.
_cap_yaml="["
IFS=',' read -r -a _caps <<<"${CAPABILITIES}"
_first=1
for cap in "${_caps[@]}"; do
  cap_trimmed="$(echo "${cap}" | tr -d '[:space:]')"
  [[ -z "${cap_trimmed}" ]] && continue
  if [[ ${_first} -eq 1 ]]; then
    _cap_yaml+="\"${cap_trimmed}\""
    _first=0
  else
    _cap_yaml+=", \"${cap_trimmed}\""
  fi
done
_cap_yaml+="]"

SUBMIT_FLAG="false"
if [[ "${SUBMIT_ON_CHAIN}" -eq 1 ]]; then
  SUBMIT_FLAG="true"
fi

# Render validator config:
# - agent.master_url ALWAYS equals --master-url (coordination API).
# - registry_url / weights_url follow the same master when that master hosts
#   both (Compose master default). Operators on a multi-front production
#   network may later edit registry_url/weights_url independently, but the
#   generated install must never invent a non-master public hostname here.
umask 077
cat >"${CONFIG_FILE}" <<EOF
network:
  name: base
  netuid: 100
  chain_endpoint: null
  wallet_name: ${WALLET_NAME}
  wallet_hotkey: ${WALLET_HOTKEY}
  wallet_path: /var/lib/base/identity
  master_uid: 0

validator:
  # When the master hosts registry + weights, keep these equal to master_url.
  # Public shipping example: https://chain.joinbase.ai
  registry_url: ${MASTER_URL}
  registry_retry_seconds: 15
  weights_url: ${MASTER_URL}
  weights_interval_seconds: 360
  weights_timeout_seconds: 15.0
  weights_retries: 3
  weights_freshness_seconds: 720
  submit_on_chain_enabled: ${SUBMIT_FLAG}
  submission_state_dir: /var/lib/base/state
  agent:
    # Coordination API pointer only (never a non-master challenge front).
    master_url: ${MASTER_URL}
    capabilities: ${_cap_yaml}
    poll_interval_seconds: 5.0
    request_timeout_seconds: 15.0
    # Weight-only default: no Prism/AC challenge adapters, no assignment execute.
    # Master is sole writer (submissions/leaderboard). Do not flip true unless you
    # intentionally run an experimental executor profile (still never challenge DB).
    challenge_execution_enabled: false
    # No local broker/challenge-control-plane in the independent agent profile.
    # Host docker.sock is composed-in separately (migration prep only); broker stubbed.
    broker_url: http://127.0.0.1:9
    broker_token_file: /run/secrets/base_broker_token
EOF

if [[ -n "${DISPLAY_NAME}" ]]; then
  cat >>"${CONFIG_FILE}" <<EOF
    display_name: "${DISPLAY_NAME}"
EOF
fi

cat >>"${CONFIG_FILE}" <<'EOF'

docker:
  network_name: base_validator_local
  secret_dir: /var/lib/base/secrets
  internal_network: true
  broker_url: http://127.0.0.1:9
  broker_allowed_images:
    - ghcr.io/baseintelligence/

observability:
  log_json: true
  sentry_dsn: null
  otel_service_name: base-validator
EOF
chmod 644 "${CONFIG_FILE}"

# Stage validator-only deployment artifacts for source-free reinstalls.
cp -f "${COMPOSE_FILE}" "${ARTIFACTS_DIR}/docker-compose.validator.yml"
# Live .env used for compose --env-file and host-side image auto-update.
umask 077
cat >"${ARTIFACTS_DIR}/.env" <<EOF
COMPOSE_PROJECT_NAME=${PROJECT_NAME}
BASE_VALIDATOR_IMAGE_REPOSITORY=${IMAGE_REPO}
BASE_VALIDATOR_IMAGE_DIGEST=${IMAGE_DIGEST}
BASE_VALIDATOR_CONFIG=${CONFIG_FILE}
BASE_VALIDATOR_PROTOCOL_IDENTITY=${IDENTITY_DIR}
BASE_VALIDATOR_BROKER_TOKEN=${BROKER_TOKEN_FILE}
BASE_VALIDATOR_TRACK_IMAGE=${TRACK_IMAGE}
BASE_VALIDATOR_IMAGE_UPDATE_HOLD=0
BASE_DOCKER_GID=${DOCKER_GID}
BASE_DOCKER_SOCKET=${DOCKER_SOCKET}
EOF
chmod 600 "${ARTIFACTS_DIR}/.env"
cp -f "${ARTIFACTS_DIR}/.env" "${ARTIFACTS_DIR}/.env.example"
chmod 600 "${ARTIFACTS_DIR}/.env.example"

# Stage host-side image updater (auto-update remains host-side; agent sock is separate).
UPDATER_SRC="${SCRIPT_DIR}/validator-image-updater.sh"
UPDATER_UNIT_SRC="${SCRIPT_DIR}/systemd/base-validator-image-updater@.service"
UPDATER_TIMER_SRC="${SCRIPT_DIR}/systemd/base-validator-image-updater@.timer"
if [[ -f "${UPDATER_SRC}" ]]; then
  cp -f "${UPDATER_SRC}" "${ARTIFACTS_DIR}/validator-image-updater.sh"
  chmod 755 "${ARTIFACTS_DIR}/validator-image-updater.sh"
fi
if [[ -f "${UPDATER_UNIT_SRC}" ]]; then
  cp -f "${UPDATER_UNIT_SRC}" "${ARTIFACTS_DIR}/base-validator-image-updater@.service"
fi
if [[ -f "${UPDATER_TIMER_SRC}" ]]; then
  cp -f "${UPDATER_TIMER_SRC}" "${ARTIFACTS_DIR}/base-validator-image-updater@.timer"
fi
# Durable state next to artifacts (mode 0600 once first tick writes it).
: >"${ARTIFACTS_DIR}/image_update_state.json" 2>/dev/null || true
chmod 600 "${ARTIFACTS_DIR}/image_update_state.json" 2>/dev/null || true

if [[ -n "${COPY_ARTIFACTS}" ]]; then
  mkdir -p "${COPY_ARTIFACTS}"
  cp -f "${ARTIFACTS_DIR}/docker-compose.validator.yml" "${COPY_ARTIFACTS}/"
  cp -f "${ARTIFACTS_DIR}/.env" "${COPY_ARTIFACTS}/.env"
  if [[ -f "${ARTIFACTS_DIR}/validator-image-updater.sh" ]]; then
    cp -f "${ARTIFACTS_DIR}/validator-image-updater.sh" "${COPY_ARTIFACTS}/"
  fi
  for unit in base-validator-image-updater@.service base-validator-image-updater@.timer; do
    if [[ -f "${ARTIFACTS_DIR}/${unit}" ]]; then
      cp -f "${ARTIFACTS_DIR}/${unit}" "${COPY_ARTIFACTS}/"
    fi
  done
  # Do not copy identity secrets unless the operator state dir is the target.
  echo "Copied validator Artifacts to ${COPY_ARTIFACTS}"
fi

export COMPOSE_PROJECT_NAME="${PROJECT_NAME}"
export BASE_VALIDATOR_IMAGE_REPOSITORY="${IMAGE_REPO}"
export BASE_VALIDATOR_IMAGE_DIGEST="${IMAGE_DIGEST}"
export BASE_VALIDATOR_CONFIG="${CONFIG_FILE}"
export BASE_VALIDATOR_PROTOCOL_IDENTITY="${IDENTITY_DIR}"
export BASE_VALIDATOR_BROKER_TOKEN="${BROKER_TOKEN_FILE}"
export BASE_DOCKER_GID="${DOCKER_GID}"
export BASE_DOCKER_SOCKET="${DOCKER_SOCKET}"

echo "Installing agent-only validator Compose project '${PROJECT_NAME}'"
echo "  master_url=${MASTER_URL}  (coordination API; registry/weights follow when master hosts both)"
echo "  protocol_hotkey=${HOTKEY_SS58}"
echo "  state_dir=${STATE_DIR}"
echo "  submit_on_chain=${SUBMIT_FLAG}"
echo "  auto_update=${ENABLE_AUTO_UPDATE} (host-side digest tracker)"
echo "  docker_socket=${DOCKER_SOCKET} (gid=${DOCKER_GID}; mounted for later challenges-on-validator prep)"
echo "  profile: agent-only (no master, postgres, or challenge control-plane on this host)"
echo "  note: container HOME=/var/lib/base/state (writable under read_only rootfs for bittensor)"

docker compose -p "${PROJECT_NAME}" -f "${COMPOSE_FILE}" --env-file "${ARTIFACTS_DIR}/.env" config --quiet
docker compose -p "${PROJECT_NAME}" -f "${COMPOSE_FILE}" --env-file "${ARTIFACTS_DIR}/.env" up -d

# Host-side auto-update: systemd timer@ + service@ installed by default.
if [[ "${ENABLE_AUTO_UPDATE}" -eq 1 ]]; then
  if [[ ! -f "${ARTIFACTS_DIR}/validator-image-updater.sh" ]]; then
    echo "warning: auto-update requested but validator-image-updater.sh missing under artifacts" >&2
  elif ! command -v systemctl >/dev/null 2>&1; then
    echo "warning: systemctl unavailable; staged updater at ${ARTIFACTS_DIR}/validator-image-updater.sh for cron" >&2
  else
    install -d -m 0755 /usr/local/lib/base
    install -m 0755 "${ARTIFACTS_DIR}/validator-image-updater.sh" /usr/local/lib/base/validator-image-updater.sh
    if [[ -f "${ARTIFACTS_DIR}/base-validator-image-updater@.service" ]]; then
      install -m 0644 "${ARTIFACTS_DIR}/base-validator-image-updater@.service" \
        /etc/systemd/system/base-validator-image-updater@.service
    fi
    if [[ -f "${ARTIFACTS_DIR}/base-validator-image-updater@.timer" ]]; then
      install -m 0644 "${ARTIFACTS_DIR}/base-validator-image-updater@.timer" \
        /etc/systemd/system/base-validator-image-updater@.timer
      if [[ "${IMAGE_UPDATE_INTERVAL}" != "90" ]]; then
        install -d -m 0755 "/etc/systemd/system/base-validator-image-updater@${PROJECT_NAME}.timer.d"
        cat >"/etc/systemd/system/base-validator-image-updater@${PROJECT_NAME}.timer.d/interval.conf" <<EOF
[Timer]
OnUnitActiveSec=${IMAGE_UPDATE_INTERVAL}
EOF
      fi
    fi
    install -d -m 0755 /etc/base/validator-image-updater
    cat >"/etc/base/validator-image-updater/${PROJECT_NAME}.env" <<EOF
COMPOSE_PROJECT_NAME=${PROJECT_NAME}
BASE_VALIDATOR_ARTIFACTS_DIR=${ARTIFACTS_DIR}
BASE_VALIDATOR_COMPOSE_FILE=${ARTIFACTS_DIR}/docker-compose.validator.yml
BASE_VALIDATOR_ENV_FILE=${ARTIFACTS_DIR}/.env
BASE_VALIDATOR_IMAGE_UPDATE_STATE=${ARTIFACTS_DIR}/image_update_state.json
BASE_VALIDATOR_TRACK_IMAGE=${TRACK_IMAGE}
BASE_VALIDATOR_IMAGE_UPDATE_HOLD=0
EOF
    chmod 600 "/etc/base/validator-image-updater/${PROJECT_NAME}.env"
    systemctl daemon-reload
    systemctl enable --now "base-validator-image-updater@${PROJECT_NAME}.timer"
    echo "Enabled host auto-update timer: base-validator-image-updater@${PROJECT_NAME}.timer"
    echo "  track_image=${TRACK_IMAGE} (always applied as repository@sha256:<digest>)"
    echo "  hold: set BASE_VALIDATOR_IMAGE_UPDATE_HOLD=1 in ${ARTIFACTS_DIR}/.env"
    echo "  state: ${ARTIFACTS_DIR}/image_update_state.json"
  fi
else
  echo "Auto-update disabled (--no-auto-update). Image pins stay operator-driven."
fi

echo "Validator Compose install complete."
docker compose -p "${PROJECT_NAME}" -f "${COMPOSE_FILE}" --env-file "${ARTIFACTS_DIR}/.env" ps
echo "Hotkey public identity (also written to ${HOTKEY_PUB_FILE}): ${HOTKEY_SS58}"
echo "Register this hotkey in the master mock_metagraph (validator_permit: true) for coordination tests."
echo "Operator note: keep protocol identity as a real directory readable by uid 1000 (avoid host symlinks with restrictive parents)."
echo "Operator note: validators never run master. Confirm --master-url /health is Base master (role=master)."
echo "Operator note: validator runtime images auto-update by default via host timer (digest pins only)."
echo "Operator note: agent mounts host docker.sock (prod prep); still agent-only Compose (no master stack)."
