#!/usr/bin/env bash
# Local full-subnet control-plane test against Bittensor testnet (netuid 541).
#
# Brings up master profile + env-staging + env-local, waits for healthz,
# optionally starts a cloudflared quick tunnel to gateway :8080, and writes
# deploy/env/local-tunnel.env (gitignored) with the ephemeral public URL.
#
# Usage:
#   ./deploy/scripts/local-e2e.sh --help
#   ./deploy/scripts/local-e2e.sh --dry-run
#   ./deploy/scripts/local-e2e.sh --smoke          # healthz + weights seal smoke (no gateway wallet)
#   ./deploy/scripts/local-e2e.sh --live           # require owner wallet + REQUIRE_OWNER=1
#   ./deploy/scripts/local-e2e.sh --smoke --no-tunnel
#   ./deploy/scripts/local-e2e.sh --smoke --no-weights-smoke
#   ./deploy/scripts/local-e2e.sh --down
#
# See docs/runbooks/local-testnet-e2e.md
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

MODE="smoke"
DO_TUNNEL=1
DO_BUILD=1
DO_UP=1
DO_WEIGHTS_SMOKE=1
DRY_RUN=0
WAIT_SECS="${LOCAL_E2E_WAIT_SECS:-180}"
STATE_DIR="${BASE_STATE_DIR:-$ROOT/.local/base-state}"
TUNNEL_ENV="$ROOT/deploy/env/local-tunnel.env"
TUNNEL_LOG="$ROOT/.local/cloudflared-gateway.log"
TUNNEL_PID_FILE="$ROOT/.local/cloudflared-gateway.pid"
# Dedicated config so a host /etc/cloudflared/config.yml (named tunnel +
# catch-all http_status:404) cannot shadow the quick tunnel.
TUNNEL_CONFIG="$ROOT/.local/cloudflared-quick.yml"
COMPOSE_PROJECT="${COMPOSE_PROJECT_NAME:-base}"
GATEWAY_HOST_PORT="${LOCAL_GATEWAY_HOST_PORT:-8080}"
VALIDATOR_HOST_PORT="${LOCAL_VALIDATOR_HOST_PORT:-28080}"
PRISM_HOST_PORT="${LOCAL_PRISM_HOST_PORT:-28092}"
DESIGN_HOST_PORT="${LOCAL_DESIGN_HOST_PORT:-28093}"
BASE_SECRETS_DIR="${BASE_SECRETS_DIR:-${HOME}/.base-secrets}"

# Default public-only hotkey for smoke (same placeholder as gateway.env.example usage).
# Not the on-chain owner — advisory mode allows mismatch.
SMOKE_GATEWAY_HOTKEY_DEFAULT="1ab7145525140560cb64e1e89fae8258e813ba12d9c20faaeabc17f95ba5fe7e"

usage() {
  cat <<'EOF'
local-e2e.sh — local master+gateway+validator stack on testnet 541

Usage:
  local-e2e.sh [options]

Modes (pick one; default --smoke):
  --smoke           Bring stack up + healthz + weights seal smoke. Advisory
                    owner check. No gateway owner wallet and no validator
                    chain-submit wallet required for /v1/weights/latest.
  --live            Fail-closed owner check (BASE_GATEWAY_REQUIRE_OWNER=1).
                    Requires deploy/secrets/wallets/base-owner (+ validator
                    wallet for on-chain weight submit).

Actions:
  --down            Tear down the local compose project and stop the tunnel
  --dry-run         Print compose files, env exports, and checks; do not start
  --help            Show this help

Flags:
  --no-tunnel       Skip cloudflared quick tunnel
  --no-build        Skip docker compose build
  --no-weights-smoke
                    Skip leaf→seal→/v1/weights/latest probe (healthz only)
  --wait SECS       Health wait budget (default 180)

Prerequisites:
  - Docker + Compose v2
  - cloudflared (unless --no-tunnel)
  - deploy/env/*.env via materialize-env.sh (examples OK for smoke)
  - For --live: base-owner wallet under deploy/secrets/wallets/ (btcli layout)
  - For --live on-chain weight submit: base-validator wallet
  - Secret files: gateway_sk (seal), prism_sk/design_sk (challenge leaf sigs).
    Smoke prefers ~/.base-secrets/challenge-*.sk when pubs match trust root;
    otherwise mints and rebuilds the local trust root. Gateway wallet is NOT
    required to serve sealed weights.

Environment knobs (optional):
  BASE_STATE_DIR              Host state root (default: .local/base-state)
  BASE_SECRETS_DIR            Challenge/owner age/sk sources (default: ~/.base-secrets)
  BASE_DOCKER_BUILD_FROM      prebuilt|source (default: prebuilt)
  LOCAL_GATEWAY_HOTKEY        Override smoke public hotkey (64 hex)
  LOCAL_PRISM_FORCE_SIM       default true
  LOCAL_DESIGN_FORCE_SIM      default true
  LOCAL_DESIGN_EGRESS_SIM     default true
  LOCAL_DESIGN_SIM_STAGE_DELAY_MS  ms pause after each design stage (default 0)
  LOCAL_PRISM_SIM_STAGE_DELAY_MS   ms pause after each prism stage (default 0)
  LOCAL_ATTEST_VERIFIER       default mock_ok

Wallet roles:
  - Gateway owner wallet / REQUIRE_OWNER: master-only identity check (live).
    Not required for POST /v1/weights/raw, admin seal, or GET /v1/weights/latest.
  - gateway_sk: mini-secret for bundle seal signatures (required for seal).
  - prism_sk / design_sk: challenge leaf signatures (must match trust root pubs).
  - Validator wallet: on-chain weight submit only (not weights/latest serving).

EOF
}

log() { printf 'local-e2e: %s\n' "$*"; }
err() { printf 'local-e2e: ERROR: %s\n' "$*" >&2; }
die() { err "$*"; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --smoke) MODE=smoke; shift ;;
    --live) MODE=live; shift ;;
    --down) MODE=down; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --no-tunnel) DO_TUNNEL=0; shift ;;
    --no-build) DO_BUILD=0; shift ;;
    --no-weights-smoke) DO_WEIGHTS_SMOKE=0; shift ;;
    --wait)
      WAIT_SECS="${2:?--wait requires seconds}"
      shift 2
      ;;
    *) die "unknown arg: $1 (try --help)" ;;
  esac
done

COMPOSE_FILES=(
  -f docker-compose.yml
  -f deploy/compose/role-master.yml
  -f deploy/compose/env-staging.yml
  -f deploy/compose/env-local.yml
)
PROFILE_ARGS=(--profile master)

compose() {
  COMPOSE_PROJECT_NAME="$COMPOSE_PROJECT" docker compose "${COMPOSE_FILES[@]}" "${PROFILE_ARGS[@]}" "$@"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

stop_tunnel() {
  if [[ -f "$TUNNEL_PID_FILE" ]]; then
    local pid
    pid="$(cat "$TUNNEL_PID_FILE" 2>/dev/null || true)"
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      log "stopping cloudflared pid=$pid"
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
    rm -f "$TUNNEL_PID_FILE"
  fi
  # Best-effort: orphan quick tunnels from prior runs targeting this host port.
  pkill -f "cloudflared tunnel --url http://127.0.0.1:${GATEWAY_HOST_PORT}" 2>/dev/null || true
}

write_tunnel_env() {
  local url="$1"
  local host
  host="${url#https://}"
  host="${host#http://}"
  host="${host%%/*}"
  umask 077
  mkdir -p "$(dirname "$TUNNEL_ENV")"
  cat >"$TUNNEL_ENV" <<EOF
# Generated by local-e2e.sh — do not commit.
# Ephemeral cloudflared quick tunnel for this run.
BASE_GATEWAY_PUBLIC_URL=${url}
BASE_DOMAIN=${host}
# External validators/miners (outside this compose network) should use:
#   export BASE_GATEWAY_ENDPOINT=${url}
# Co-located validator keeps http://gateway:8080 via env-local.yml.
EOF
  chmod 0600 "$TUNNEL_ENV"
  log "wrote $TUNNEL_ENV"
}

start_tunnel() {
  require_cmd cloudflared
  mkdir -p "$(dirname "$TUNNEL_LOG")"
  stop_tunnel
  # Non-empty placeholder; --url still requests a trycloudflare.com quick tunnel.
  # Avoid loading /etc/cloudflared/config.yml (often a named tunnel with 404 default).
  printf '# local-e2e quick tunnel — do not point at named-tunnel credentials\n' >"$TUNNEL_CONFIG"
  local target="http://127.0.0.1:${GATEWAY_HOST_PORT}"
  log "starting cloudflared quick tunnel → $target"
  nohup env -u TUNNEL_TOKEN cloudflared tunnel --no-autoupdate \
    --config "$TUNNEL_CONFIG" --url "$target" \
    >"$TUNNEL_LOG" 2>&1 &
  echo $! >"$TUNNEL_PID_FILE"
  local url="" i registered=0
  for i in $(seq 1 90); do
    if grep -q 'Registered tunnel connection' "$TUNNEL_LOG" 2>/dev/null; then
      registered=1
    fi
    url="$(grep -oE 'https://[a-zA-Z0-9.-]+\.trycloudflare\.com' "$TUNNEL_LOG" | head -1 || true)"
    if [[ -n "$url" && "$registered" -eq 1 ]]; then
      # Quick tunnels can take a few seconds for public DNS; retry healthz.
      if curl -fsS -m 12 "$url/healthz" >/dev/null 2>&1; then
        write_tunnel_env "$url"
        log "tunnel URL: $url (healthz ok)"
        return 0
      fi
    fi
    sleep 1
  done
  if [[ -n "$url" ]]; then
    write_tunnel_env "$url"
    log "tunnel URL: $url (edge healthz not confirmed yet — try curl \$BASE_GATEWAY_PUBLIC_URL/healthz)"
    return 0
  fi
  die "cloudflared did not publish a trycloudflare.com URL within 90s (see $TUNNEL_LOG)"
}

ensure_env_files() {
  # Challenge env files are required by compose (BASE_DATABASE_URL → Postgres).
  if [[ ! -f deploy/env/postgres.env || ! -f deploy/env/gateway.env \
    || ! -f deploy/env/validator.env || ! -f deploy/env/design-challenge.env \
    || ! -f deploy/env/prism-challenge.env ]]; then
    log "materializing deploy/env/*.env from examples"
    ./deploy/scripts/materialize-env.sh
  fi
  # Keep app URLs in lockstep with postgres.env (avoids stale example passwords
  # like base:base or user:pass after a fresh materialize).
  local url
  url="$(database_url_from_postgres_env)"
  for f in deploy/env/gateway.env deploy/env/validator.env \
    deploy/env/design-challenge.env deploy/env/prism-challenge.env; do
    if grep -q '^BASE_DATABASE_URL=' "$f" 2>/dev/null; then
      sed -i "s|^BASE_DATABASE_URL=.*|BASE_DATABASE_URL=${url}|" "$f"
    else
      printf '\nBASE_DATABASE_URL=%s\n' "$url" >>"$f"
    fi
    chmod 0600 "$f" 2>/dev/null || true
  done
}

ensure_state_dirs() {
  mkdir -p "$STATE_DIR/design/staging"
  # Design sandbox binds must be writable by uid 65532 inside containers.
  chmod 777 "$STATE_DIR/design/staging" 2>/dev/null || true
}

# Create a 32-byte secret file if missing. live mode refuses to invent wallets.
# Derive 32-byte hex public key from a mini-secret file (raw 32 or hex text).
pubkey_hex_from_sk_file() {
  local path="$1"
  python3 - "$path" <<'PY'
import sys
from pathlib import Path
from substrateinterface.keypair import Keypair
raw = Path(sys.argv[1]).read_bytes()
if len(raw) == 32:
    seed = raw
else:
    text = raw.decode().strip()
    if text.startswith(("0x", "0X")):
        text = text[2:]
    seed = bytes.fromhex(text)
kp = Keypair.create_from_seed(seed.hex())
pk = kp.public_key
print(pk.hex() if isinstance(pk, (bytes, bytearray)) else bytes(pk).hex())
PY
}

install_secret_bytes() {
  local path="$1"
  local src="$2"
  mkdir -p "$(dirname "$path")"
  cp -f "$src" "$path"
  chown 65532:65532 "$path" 2>/dev/null || true
  chmod 0400 "$path"
}

ensure_secret_file() {
  local path="$1"
  local label="$2"
  if [[ -f "$path" && ! -d "$path" ]]; then
    return 0
  fi
  if [[ -d "$path" ]]; then
    die "$path is a directory (Docker created it for a missing bind). Remove it and re-run."
  fi
  if [[ "$MODE" == "live" ]]; then
    die "missing required secret for --live: $path ($label)"
  fi
  log "minting smoke dummy secret: $path"
  mkdir -p "$(dirname "$path")"
  dd if=/dev/urandom bs=32 count=1 status=none of="$path"
  chown 65532:65532 "$path" 2>/dev/null || true
  chmod 0400 "$path"
}

# Prefer real challenge sk from BASE_SECRETS_DIR when it matches committed config pubs.
# Random smoke mints that diverge from the local trust root break leaf verify (401).
ensure_challenge_sk_aligned() {
  local path="$1"
  local challenge_id="$2"
  local fallback_sk="$3"
  local expected_pub=""
  if [[ -f "$ROOT/config/challenges.toml" ]]; then
    expected_pub="$(python3 - "$ROOT/config/challenges.toml" "$challenge_id" <<'PY'
import sys, tomllib
from pathlib import Path
doc = tomllib.loads(Path(sys.argv[1]).read_text())
cid = sys.argv[2]
for c in doc.get("challenges", []):
    if c.get("id") == cid:
        print(c.get("public_key", "").strip().lower())
        break
PY
)"
  fi
  if [[ -f "$path" && -n "$expected_pub" ]]; then
    local got
    got="$(pubkey_hex_from_sk_file "$path" 2>/dev/null || true)"
    if [[ "${got,,}" == "$expected_pub" ]]; then
      return 0
    fi
    log "warning: $path pub ${got:-?} ≠ config $challenge_id pub $expected_pub"
  fi
  if [[ -n "$expected_pub" && -f "$fallback_sk" ]]; then
    local fb_pub
    fb_pub="$(pubkey_hex_from_sk_file "$fallback_sk" 2>/dev/null || true)"
    if [[ "${fb_pub,,}" == "$expected_pub" ]]; then
      log "installing aligned $challenge_id sk from $fallback_sk → $path"
      install_secret_bytes "$path" "$fallback_sk"
      return 0
    fi
  fi
  ensure_secret_file "$path" "$challenge_id mini-secret"
  # Force trust-root rebuild so minted pubs are what the gateway verifies.
  rm -f "$ROOT/.local/trust-root/challenges.toml" \
        "$ROOT/.local/trust-root/challenges.toml.sig"
}

ensure_secrets() {
  ensure_secret_file deploy/secrets/gateway_sk "gateway seal mini-secret"
  ensure_challenge_sk_aligned \
    deploy/secrets/prism_sk prism \
    "${BASE_SECRETS_DIR}/challenge-prism.sk"
  ensure_challenge_sk_aligned \
    deploy/secrets/design_sk design \
    "${BASE_SECRETS_DIR}/challenge-design.sk"
  mkdir -p deploy/secrets/lium deploy/secrets/openrouter deploy/secrets/design
  # Touch placeholders so compose bind-mounts stay files/dirs of the right kind.
  [[ -e deploy/secrets/lium/api_key ]] || : >deploy/secrets/lium/api_key
  [[ -e deploy/secrets/lium/ssh_ed25519 ]] || : >deploy/secrets/lium/ssh_ed25519
  [[ -e deploy/secrets/lium/ssh_ed25519.pub ]] || : >deploy/secrets/lium/ssh_ed25519.pub
  [[ -e deploy/secrets/openrouter/api_key ]] || : >deploy/secrets/openrouter/api_key
  [[ -e deploy/secrets/design/annotator_tokens ]] || : >deploy/secrets/design/annotator_tokens
  chown 65532:65532 deploy/secrets/design/annotator_tokens 2>/dev/null || true
  chmod 0400 deploy/secrets/design/annotator_tokens 2>/dev/null || true
}

# Ephemeral owner-signed trust root for local stacks (prod owner key is not required).
# Writes under .local/trust-root/; env-local.yml bind-mounts that dir to
# /etc/base/config inside gateway/validator. BASE_TRUST_ROOT_DIR must be the
# *in-container* path — a host absolute path is invisible to containers.
#
# Challenge public_keys ALWAYS come from deploy/secrets/{prism,design}_sk so
# leaf signatures verify. A stale trust root with mismatched pubs is rebuilt.
ensure_local_trust_root() {
  local dir="$ROOT/.local/trust-root"
  mkdir -p "$dir"
  export BASE_TRUST_ROOT_DIR=/etc/base/config

  local prism_pk design_pk
  prism_pk="$(pubkey_hex_from_sk_file "$ROOT/deploy/secrets/prism_sk")"
  design_pk="$(pubkey_hex_from_sk_file "$ROOT/deploy/secrets/design_sk")"

  local need_rebuild=0
  if [[ ! -f "$dir/challenges.toml" || ! -f "$dir/challenges.toml.sig" || ! -f "$dir/owner.pubkey" ]]; then
    need_rebuild=1
  else
    python3 - "$dir/challenges.toml" "$prism_pk" "$design_pk" <<'PY' || need_rebuild=1
import sys, tomllib
from pathlib import Path
doc = tomllib.loads(Path(sys.argv[1]).read_text())
rows = {c["id"]: c.get("public_key", "").lower() for c in doc.get("challenges", [])}
want = {"prism": sys.argv[2].lower(), "design": sys.argv[3].lower()}
for cid, pk in want.items():
    if rows.get(cid) != pk:
        sys.exit(1)
sys.exit(0)
PY
  fi

  if [[ "$need_rebuild" -eq 0 ]]; then
    log "using existing local trust root: $dir (pubs match challenge sk files)"
    return 0
  fi

  log "generating ephemeral owner-signed local trust root in $dir (pubs from challenge sk)"
  local age_id="${AGE_IDENTITY:-/root/.base-secrets/age-identity.txt}"
  [[ -f "$age_id" ]] || age_id="${BASE_SECRETS_DIR}/age-identity.txt"
  local recip=""
  if [[ -f "$age_id" ]]; then
    recip="$(grep 'public key:' "$age_id" 2>/dev/null | awk '{print $4}' || true)"
  fi
  if [[ -z "$recip" ]]; then
    die "need age identity to sign local trust root ($age_id); or place matching owner-signed challenges under $dir"
  fi
  if [[ ! -f "$dir/owner.pubkey" || ! -f "$dir/owner.age" ]]; then
    cargo run -q -p trustroot-bin -- keygen \
      --out-pub "$dir/owner.pubkey" \
      --out-secret "$dir/owner.age" \
      --age-recipient "$recip"
  fi
  python3 - "$dir/challenges.toml" "$prism_pk" "$design_pk" <<'PY2'
import pathlib, sys
prism_pk, design_pk = sys.argv[2], sys.argv[3]
text = (
    "version = 1\nintroduced_epoch = 0\n\n"
    f'[[challenges]]\nid = "prism"\npublic_key = "{prism_pk}"\n'
    "emission_share_bps = 10000\npolicy = \"all_metagraph_hotkeys\"\n\n"
    f'[[challenges]]\nid = "design"\npublic_key = "{design_pk}"\n'
    "emission_share_bps = 0\npolicy = \"all_metagraph_hotkeys\"\n\n"
)
pathlib.Path(sys.argv[1]).write_text(text)
PY2
  # measurements: empty allowlist (base-agent CVM path removed)
  cat > "$dir/measurements.toml" <<'EOF'
version = 1
introduced_epoch = 0
measurements = []
EOF
  cargo run -q -p trustroot-bin -- sign \
    --key "$dir/owner.age" \
    --age-identity "$age_id" \
    --input "$dir/challenges.toml" --kind challenges
  cargo run -q -p trustroot-bin -- sign \
    --key "$dir/owner.age" \
    --age-identity "$age_id" \
    --input "$dir/measurements.toml" --kind measurements
  log "local trust root ready (ephemeral owner); gateway_sk seals bundles — no gateway wallet required for weights/latest"
}

wallet_hotkey_present() {
  local name="$1"
  local root="deploy/secrets/wallets/$name"
  [[ -f "$root/hotkeys/default" ]] || [[ -f "$root/hotkey" ]] || [[ -d "$root" && -n "$(find "$root" -type f 2>/dev/null | head -1)" ]]
}

check_live_prereqs() {
  if ! wallet_hotkey_present "base-owner"; then
    die "--live requires deploy/secrets/wallets/base-owner (btcli wallet; owns testnet netuid 541)"
  fi
  if ! wallet_hotkey_present "base-validator"; then
    err "warning: deploy/secrets/wallets/base-validator missing — stack can healthz but weight submit fails closed"
  fi
}

database_url_from_postgres_env() {
  local pg="$ROOT/deploy/env/postgres.env"
  [[ -f "$pg" ]] || die "missing $pg (run materialize-env.sh)"
  local user pass db
  user="$(sed -n 's/^POSTGRES_USER=//p' "$pg" | tail -1)"
  pass="$(sed -n 's/^POSTGRES_PASSWORD=//p' "$pg" | tail -1)"
  db="$(sed -n 's/^POSTGRES_DB=//p' "$pg" | tail -1)"
  [[ -n "$user" && -n "$pass" && -n "$db" ]] || die "postgres.env missing POSTGRES_USER/PASSWORD/DB"
  printf 'postgres://%s:%s@postgres:5432/%s' "$user" "$pass" "$db"
}

port_in_use() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -tln | grep -qE ":${port}\\s" && return 0
  fi
  return 1
}

check_host_ports() {
  local p
  for p in "$GATEWAY_HOST_PORT" "$VALIDATOR_HOST_PORT" "$PRISM_HOST_PORT" "$DESIGN_HOST_PORT"; do
    if port_in_use "$p"; then
      # Allow re-bind when this compose project already publishes the port.
      if docker ps --format '{{.Names}} {{.Ports}}' \
        | grep -E "^${COMPOSE_PROJECT}-" \
        | grep -qE "(:|::)${p}->|0\\.0\\.0\\.0:${p}->|127\\.0\\.0\\.1:${p}->"; then
        continue
      fi
      die "host port ${p} is already in use (staging SSH tunnel on 18080?). Set LOCAL_*_HOST_PORT or free it"
    fi
  done
}

export_mode_env() {
  export BASE_STATE_DIR="$STATE_DIR"
  export BASE_DOCKER_BUILD_FROM="${BASE_DOCKER_BUILD_FROM:-prebuilt}"
  export BASE_GATEWAY_ENDPOINT="${BASE_GATEWAY_ENDPOINT:-http://gateway:8080}"
  export LOCAL_PRISM_FORCE_SIM="${LOCAL_PRISM_FORCE_SIM:-true}"
  export LOCAL_DESIGN_FORCE_SIM="${LOCAL_DESIGN_FORCE_SIM:-true}"
  # Host SimSandbox opt-in for local overlay only (bin refuse on prod/mainnet).
  export BASE_ALLOW_HOST_SIM="${BASE_ALLOW_HOST_SIM:-1}"
  export LOCAL_DESIGN_EGRESS_SIM="${LOCAL_DESIGN_EGRESS_SIM:-true}"
  export LOCAL_ATTEST_VERIFIER="${LOCAL_ATTEST_VERIFIER:-mock_ok}"
  export LOCAL_GATEWAY_HOST_PORT="$GATEWAY_HOST_PORT"
  export LOCAL_VALIDATOR_HOST_PORT="$VALIDATOR_HOST_PORT"
  export LOCAL_PRISM_HOST_PORT="$PRISM_HOST_PORT"
  export LOCAL_DESIGN_HOST_PORT="$DESIGN_HOST_PORT"
  # Align app DATABASE_URL with whatever postgres.env will create (avoids
  # stale gateway.env pointing at a different database name).
  if [[ -z "${LOCAL_DATABASE_URL:-}" && -f "$ROOT/deploy/env/postgres.env" ]]; then
    LOCAL_DATABASE_URL="$(database_url_from_postgres_env)"
  fi
  export LOCAL_DATABASE_URL="${LOCAL_DATABASE_URL:-}"

  if [[ "$MODE" == "live" ]]; then
    export LOCAL_REQUIRE_OWNER="${LOCAL_REQUIRE_OWNER:-1}"
    export LOCAL_GATEWAY_WALLET="${LOCAL_GATEWAY_WALLET:-base-owner}"
    export LOCAL_GATEWAY_WALLET_HOTKEY="${LOCAL_GATEWAY_WALLET_HOTKEY:-default}"
    export LOCAL_VALIDATOR_WALLET="${LOCAL_VALIDATOR_WALLET:-base-validator}"
    export LOCAL_VALIDATOR_WALLET_HOTKEY="${LOCAL_VALIDATOR_WALLET_HOTKEY:-default}"
    # Clear public-only hotkey so wallet resolution wins.
    export LOCAL_GATEWAY_HOTKEY="${LOCAL_GATEWAY_HOTKEY:-}"
  else
    export LOCAL_REQUIRE_OWNER="${LOCAL_REQUIRE_OWNER:-0}"
    export LOCAL_GATEWAY_WALLET="${LOCAL_GATEWAY_WALLET:-}"
    export LOCAL_VALIDATOR_WALLET="${LOCAL_VALIDATOR_WALLET:-}"
    if [[ -z "${LOCAL_GATEWAY_HOTKEY:-}" ]]; then
      if [[ -f deploy/env/gateway.env ]] && grep -q '^BASE_GATEWAY_HOTKEY=' deploy/env/gateway.env; then
        LOCAL_GATEWAY_HOTKEY="$(sed -n 's/^BASE_GATEWAY_HOTKEY=//p' deploy/env/gateway.env | tail -1)"
      fi
      export LOCAL_GATEWAY_HOTKEY="${LOCAL_GATEWAY_HOTKEY:-$SMOKE_GATEWAY_HOTKEY_DEFAULT}"
    else
      export LOCAL_GATEWAY_HOTKEY
    fi
  fi
}

print_plan() {
  cat <<EOF
local-e2e plan
  mode:            $MODE
  dry_run:         $DRY_RUN
  tunnel:          $DO_TUNNEL
  build:           $DO_BUILD
  weights_smoke:   $DO_WEIGHTS_SMOKE
  project:         $COMPOSE_PROJECT
  state_dir:       $STATE_DIR
  wait_secs:       $WAIT_SECS
  compose files:   ${COMPOSE_FILES[*]}
  profiles:        ${PROFILE_ARGS[*]}
  LOCAL_REQUIRE_OWNER=$LOCAL_REQUIRE_OWNER
  LOCAL_GATEWAY_WALLET=${LOCAL_GATEWAY_WALLET:-<empty>}
  LOCAL_VALIDATOR_WALLET=${LOCAL_VALIDATOR_WALLET:-<empty>}
  LOCAL_GATEWAY_HOTKEY=${LOCAL_GATEWAY_HOTKEY:-<empty>}
  LOCAL_DATABASE_URL=$LOCAL_DATABASE_URL
  BASE_GATEWAY_ENDPOINT=$BASE_GATEWAY_ENDPOINT
  BASE_DOCKER_BUILD_FROM=$BASE_DOCKER_BUILD_FROM
EOF
}

wait_health() {
  local name="$1" url="$2"
  local i
  log "waiting for $name at $url (up to ${WAIT_SECS}s)"
  for i in $(seq 1 "$WAIT_SECS"); do
    if curl -fsS -m 3 "$url" >/dev/null 2>&1; then
      log "$name healthy: $(curl -fsS -m 3 "$url")"
      return 0
    fi
    sleep 1
  done
  err "$name not healthy after ${WAIT_SECS}s"
  compose ps || true
  compose logs --tail=80 "$name" 2>/dev/null || true
  return 1
}

wait_all_health() {
  # Core control plane — fail the run if these do not come up.
  wait_health gateway "http://127.0.0.1:${GATEWAY_HOST_PORT}/healthz"
  wait_health validator "http://127.0.0.1:${VALIDATOR_HOST_PORT}/healthz"
  # Challenges are best-effort for smoke (glibc prebuilt mismatch is common).
  local soft_wait=30
  local saved="$WAIT_SECS"
  WAIT_SECS="$soft_wait"
  wait_health prism-challenge "http://127.0.0.1:${PRISM_HOST_PORT}/health" || \
    log "warning: prism not healthy (continuing; try BASE_DOCKER_BUILD_FROM=source)"
  wait_health design-challenge "http://127.0.0.1:${DESIGN_HOST_PORT}/health" || \
    log "warning: design not healthy (continuing; try BASE_DOCKER_BUILD_FROM=source)"
  WAIT_SECS="$saved"
}

# Prove seal→serve without a gateway owner wallet: signed leaves + admin seal +
# GET /v1/weights/latest must be 200. Uses prism_sk + gateway_sk only.
probe_weights_latest() {
  if [[ "$DO_WEIGHTS_SMOKE" -ne 1 ]]; then
    log "skipping weights seal smoke (--no-weights-smoke)"
    return 0
  fi
  local gw="http://127.0.0.1:${GATEWAY_HOST_PORT}"
  local pre
  pre="$(curl -sS -m 5 -o /tmp/local-e2e-weights-pre.json -w '%{http_code}' \
    "${gw}/v1/weights/latest" || true)"
  log "pre-seal GET /v1/weights/latest → HTTP ${pre:-?} (200 burn sealed=false expected before smoke)"

  log "building weights-smoke helper"
  cargo build -q -p weights-smoke --release

  local netuid endpoint
  netuid="$(sed -n 's/^BASE_NETUID=//p' "$ROOT/deploy/env/gateway.env" 2>/dev/null | tail -1)"
  netuid="${netuid:-541}"
  endpoint="$(sed -n 's/^BASE_CHAIN_ENDPOINT=//p' "$ROOT/deploy/env/gateway.env" 2>/dev/null | tail -1)"
  endpoint="${endpoint:-wss://test.finney.opentensor.ai:443}"

  log "running weights-smoke (leaf submit → admin/seal → weights/latest)"
  ./target/release/weights-smoke \
    --gateway "$gw" \
    --challenge-sk "$ROOT/deploy/secrets/prism_sk" \
    --challenge-id prism \
    --netuid "$netuid" \
    --chain-endpoint "$endpoint" \
    | tee /tmp/local-e2e-weights-latest.json \
    >/dev/null

  local code
  code="$(curl -sS -m 5 -o /tmp/local-e2e-weights-post.json -w '%{http_code}' \
    "${gw}/v1/weights/latest")"
  [[ "$code" == "200" ]] || die "weights/latest still HTTP $code after seal smoke (not a gateway-wallet issue)"
  log "weights seal smoke OK: GET /v1/weights/latest → 200"
}

print_summary() {
  local pub=""
  if [[ -f "$TUNNEL_ENV" ]]; then
    pub="$(sed -n 's/^BASE_GATEWAY_PUBLIC_URL=//p' "$TUNNEL_ENV" | tail -1)"
  fi
  cat <<EOF

========== local-e2e ready ($MODE) ==========
Internal (compose network):
  gateway:            http://gateway:8080
  validator probe:    http://127.0.0.1:${VALIDATOR_HOST_PORT}/healthz
  prism:              http://127.0.0.1:${PRISM_HOST_PORT}/health
  design:             http://127.0.0.1:${DESIGN_HOST_PORT}/health

EOF
  if [[ -n "$pub" ]]; then
    cat <<EOF
Public gateway (ephemeral tunnel):
  $pub
  External validators/miners:
    export BASE_GATEWAY_ENDPOINT=$pub
  Tunnel env file: $TUNNEL_ENV
  Tunnel log:      $TUNNEL_LOG

EOF
  else
    cat <<EOF
No tunnel URL (use --no-tunnel or tunnel failed). Gateway is on host :${GATEWAY_HOST_PORT}.

EOF
  fi
  cat <<EOF
Teardown:
  ./deploy/scripts/local-e2e.sh --down

Notes:
  - Co-located validator keeps BASE_GATEWAY_ENDPOINT=http://gateway:8080
  - Restarting services after tunnel start is optional; tunnel is L7 ingress only
  - /v1/weights/latest needs sealed leaves (challenge_sk + gateway_sk), not a gateway wallet
  - --live still needs human-held wallets under deploy/secrets/wallets/ for owner check + chain submit
  - Re-run seal smoke alone:
      cargo run -q --release -p weights-smoke -- --gateway http://127.0.0.1:${GATEWAY_HOST_PORT}
=============================================
EOF
}

# --- actions -----------------------------------------------------------------

require_cmd docker
require_cmd curl

if [[ "$MODE" == "down" ]]; then
  # DATABASE_URL not required for teardown; provide a placeholder for compose.
  export LOCAL_DATABASE_URL="${LOCAL_DATABASE_URL:-postgres://base:base@postgres:5432/base}"
  export_mode_env
  print_plan
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "dry-run: would compose down + stop tunnel"
    exit 0
  fi
  stop_tunnel
  compose down --remove-orphans || true
  rm -f "$TUNNEL_ENV"
  log "down complete"
  exit 0
fi

ensure_env_files
export_mode_env
[[ -n "$LOCAL_DATABASE_URL" ]] || die "LOCAL_DATABASE_URL unset (need deploy/env/postgres.env)"
print_plan

if [[ "$DRY_RUN" -eq 1 ]]; then
  log "dry-run: rendering compose config"
  compose config --services
  log "dry-run: OK (no containers started)"
  exit 0
fi

docker compose version >/dev/null

if [[ "$DO_TUNNEL" -eq 1 ]]; then
  require_cmd cloudflared
fi

ensure_state_dirs
ensure_secrets
ensure_local_trust_root
if [[ "$MODE" == "live" ]]; then
  check_live_prereqs
fi
check_host_ports

if [[ "$DO_BUILD" -eq 1 ]]; then
  log "building images (BASE_DOCKER_BUILD_FROM=$BASE_DOCKER_BUILD_FROM)"
  compose build
fi

log "starting stack"
compose up -d --remove-orphans
compose ps

wait_all_health || die "health checks failed — see logs above"

# Seal path is independent of tunnel / owner wallet; run before tunnel so a
# tunnel flake cannot mask a weights regression.
probe_weights_latest || die "weights seal smoke failed"

if [[ "$DO_TUNNEL" -eq 1 ]]; then
  start_tunnel
else
  log "skipping tunnel (--no-tunnel)"
fi

print_summary
