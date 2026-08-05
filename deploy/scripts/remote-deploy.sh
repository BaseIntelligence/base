#!/usr/bin/env bash
# remote-deploy.sh — rsync control-plane tree to a droplet and restart compose.
#
#   remote-deploy.sh --host root@IP --role master|validator [--gateway-endpoint URL]
#                     [--build-from source|prebuilt|registry]
#
# --build-from:
#   source   — docker compose build on the droplet (Rust compile in Docker)
#   prebuilt — rsync target/release binaries, compose build with BUILD_FROM=prebuilt
#   registry — pull GHCR digest pins from deploy/pins/<env>.json, retag to local
#              Compose tags (validator:0.1.0, …), compose up --no-build
#
# Does NOT copy secrets from the operator machine by default. Secrets must
# already exist on the host under deploy/env/*.env and deploy/secrets/* (age path).
# Optional: --bootstrap-secrets-from HOST copies secrets once from another host.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=lib-promote.sh
source "${SCRIPT_DIR}/lib-promote.sh"

HOST=""
ROLE=""
ENV="${BASE_DEPLOY_ENV:-staging}"
GATEWAY_ENDPOINT=""
BOOTSTRAP_FROM=""
BUILD_FROM="${BASE_DOCKER_BUILD_FROM:-source}"
REMOTE_DIR="${BASE_REMOTE_DIR:-/opt/base}"
# Host-side staging root for held-out verifier binds; must match the compose
# bind source and the container's BASE_VERIFY_WORK_ROOT byte-for-byte.
STATE_ROOT="${BASE_STATE_DIR:-/var/lib/base}"
GHCR_PREFIX="${BASE_GHCR_PREFIX:-ghcr.io/baseintelligence/base}"
PIN_SERVICES=(validator gateway updater prism-challenge design-challenge)
SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)
if [[ -n "${BASE_SSH_IDENTITY:-}" ]]; then
  SSH_OPTS+=(-i "$BASE_SSH_IDENTITY")
fi

# lib-promote.sh defines die() with a promote: prefix — restore ours.
die() { echo "remote-deploy: $*" >&2; exit 1; }

# Validate deploy/pins/<env>.json for registry mode (fail-closed).
validate_registry_pins() {
  local pin_path="$1"
  local env_name="$2"
  local digests_path=""
  [[ -f "$pin_path" ]] || die "missing pin file: $pin_path"
  digests_path="$(python3 - "$pin_path" "$ROOT" <<'PY'
import json, sys
from pathlib import Path
pin = json.load(open(sys.argv[1], encoding="utf-8"))
root = Path(sys.argv[2])
commit = (pin.get("commit_sha") or "").strip()
print(root / "deploy" / "digests" / f"{commit}.json" if commit else "")
PY
)"
  python3 - "$pin_path" "$env_name" "$digests_path" "${PIN_SERVICES[@]}" <<'PY'
import json, re, sys
from pathlib import Path
path, env_name, digests_path, *services = sys.argv[1:]
with open(path, encoding="utf-8") as f:
    pin = json.load(f)
digest_re = re.compile(r"^sha256:[0-9a-f]{64}$")
def placeholder(d: str) -> bool:
    h = d.removeprefix("sha256:").lower()
    return bool(re.fullmatch(r"0+", h) or re.fullmatch(r"0+1", h))
commit = (pin.get("commit_sha") or "").strip().lower()
if not commit or re.fullmatch(r"0+", commit):
    raise SystemExit(f"registry mode: pin commit_sha missing or placeholder in {path}")
svcs = pin.get("services") or {}
for svc in services:
    if svc not in svcs:
        raise SystemExit(f"registry mode: missing service {svc!r} in {path}")
    d = (svcs[svc].get("digest") or "").strip().lower()
    if not digest_re.match(d):
        raise SystemExit(f"registry mode: invalid digest for {svc}: {d!r}")
    if placeholder(d):
        raise SystemExit(
            f"registry mode: placeholder digest for {svc} in {path} "
            f"(promote real GHCR digests before --build-from registry)"
        )
# prism is a default compose service. Prod registry deploys require
# deploy/digests/<sha>.json (committed by images.yml). Staging may omit it when
# still using --build-from source.
dp = Path(digests_path) if digests_path else None
if dp is not None and not dp.is_file():
    if env_name == "prod":
        raise SystemExit(
            f"registry mode: prod requires digest manifest {dp} "
            "(images.yml must commit deploy/digests/<sha>.json)"
        )
    print(
        f"registry mode: WARNING: missing {dp} — "
        "prism/base-attest-helper will not be pulled",
        file=sys.stderr,
    )
print(f"registry pins ok env={env_name} commit={commit} services={','.join(services)}")
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="${2:-}"; shift 2 ;;
    --role) ROLE="${2:-}"; shift 2 ;;
    --gateway-endpoint) GATEWAY_ENDPOINT="${2:-}"; shift 2 ;;
    --env) ENV="${2:-}"; shift 2 ;;
    --bootstrap-secrets-from) BOOTSTRAP_FROM="${2:-}"; shift 2 ;;
    --build-from) BUILD_FROM="${2:-}"; shift 2 ;;
    --remote-dir) REMOTE_DIR="${2:-}"; shift 2 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) die "unknown arg: $1" ;;
  esac
done

[[ -n "$HOST" ]] || die "--host required"
case "$ROLE" in master|validator) ;; *) die "--role master|validator required" ;; esac
case "$ENV" in staging|prod) ;; *) die "--env staging|prod required" ;; esac
case "$BUILD_FROM" in source|prebuilt|registry) ;; *)
  die "--build-from must be source|prebuilt|registry (got: $BUILD_FROM)"
esac

ssh_h() { ssh "${SSH_OPTS[@]}" "$HOST" "$@"; }
scp_h() { scp "${SSH_OPTS[@]}" "$@"; }

echo "remote-deploy: host=$HOST role=$ROLE env=$ENV remote=$REMOTE_DIR build_from=$BUILD_FROM"

if [[ "$BUILD_FROM" == "registry" ]]; then
  PIN_PATH="$(pin_path_for_env "$ROOT" "$ENV")"
  validate_registry_pins "$PIN_PATH" "$ENV"
fi

ssh_h "mkdir -p '$REMOTE_DIR' && command -v docker >/dev/null"

if [[ -n "$BOOTSTRAP_FROM" ]]; then
  echo "remote-deploy: bootstrap secrets from $BOOTSTRAP_FROM"
  tmp="$(mktemp -d)"
  # shellcheck disable=SC2029
  ssh "${SSH_OPTS[@]}" "$BOOTSTRAP_FROM" \
    "tar -C /opt/gbase/deploy -cf - env secrets 2>/dev/null || tar -C /opt/base/deploy -cf - env secrets" \
    >"$tmp/secrets.tar"
  ssh_h "mkdir -p '$REMOTE_DIR/deploy' && tar -C '$REMOTE_DIR/deploy' -xf -" <"$tmp/secrets.tar"
  # Convert legacy GBASE_* keys to BASE_* if present.
  ssh_h "bash -s" <<'EOS'
set -euo pipefail
for f in /opt/base/deploy/env/*.env; do
  [[ -f "$f" ]] || continue
  if grep -q '^GBASE_' "$f" 2>/dev/null; then
    sed -i 's/^GBASE_/BASE_/g' "$f"
  fi
  chmod 600 "$f"
done
if [[ -d /opt/base/deploy/secrets ]]; then
  chown -R 65532:65532 /opt/base/deploy/secrets || true
  find /opt/base/deploy/secrets -type f -exec chmod 400 {} \;
fi
EOS
  rm -rf "$tmp"
fi

echo "remote-deploy: rsync tree"
if [[ "$BUILD_FROM" == "prebuilt" ]]; then
  for b in validator gateway updater prism-challenge design-challenge design-egress-proxy challenge-review; do
    [[ -x "$ROOT/target/release/$b" ]] || die "missing prebuilt binary target/release/$b — run: cargo build --release --features validator-bin/dcap -p validator-bin -p gateway-bin -p updater-bin -p prism-challenge-bin -p design-challenge-bin -p design-egress-proxy-bin && cargo build --release -p challenge-review-bin"
  done
fi

RSYNC_SSH="ssh ${SSH_OPTS[*]}"
rsync -az --delete \
  -e "$RSYNC_SSH" \
  --exclude '.git/' \
  --exclude 'target/' \
  --exclude 'deploy/terraform/.terraform/' \
  --exclude 'deploy/terraform/terraform.tfstate*' \
  --exclude 'deploy/terraform/tfplan' \
  --exclude 'deploy/terraform/terraform.tfvars' \
  --exclude 'deploy/env/*.env' \
  --exclude 'deploy/secrets/' \
  --exclude 'miner-runtime/' \
  --exclude '.omo/' \
  "$ROOT/" "$HOST:$REMOTE_DIR/"

# Ensure secrets dirs exist (empty OK if not bootstrapped).
# deploy/secrets/lium is bind-mounted by prism-challenge, so it must be a real
# directory with real files: compose would otherwise create directories where
# the container expects files.
# Same footgun for file mounts: if prism_sk / design_sk are missing, Docker
# creates *directories* at those paths and the challenge bins fail with
# "Is a directory" / "secret file missing". Materialize empty files when
# absent; if a directory already poisoned the path, replace it with a file.
ssh_h "mkdir -p '$REMOTE_DIR/deploy/env' '$REMOTE_DIR/deploy/secrets/lium' \
  '$REMOTE_DIR/deploy/secrets/openrouter' '$REMOTE_DIR/deploy/secrets/design' \
  '$REMOTE_DIR/deploy/secrets/wallets' \
  && chmod 700 '$REMOTE_DIR/deploy/secrets' '$REMOTE_DIR/deploy/secrets/lium' \
  && for f in api_key ssh_ed25519 ssh_ed25519.pub; do \
       [ -e '$REMOTE_DIR/deploy/secrets/lium/'\$f ] || : > '$REMOTE_DIR/deploy/secrets/lium/'\$f; \
     done \
  && [ -e '$REMOTE_DIR/deploy/secrets/openrouter/api_key' ] || : > '$REMOTE_DIR/deploy/secrets/openrouter/api_key' \
  && [ -e '$REMOTE_DIR/deploy/secrets/design/annotator_tokens' ] || : > '$REMOTE_DIR/deploy/secrets/design/annotator_tokens' \
  && for sk in prism_sk design_sk; do \
       p='$REMOTE_DIR/deploy/secrets/'\$sk; \
       if [ -d \"\$p\" ]; then rm -rf \"\$p\"; fi; \
       [ -e \"\$p\" ] || : > \"\$p\"; \
       chmod 400 \"\$p\"; chown 65532:65532 \"\$p\"; \
     done \
  && chmod 400 '$REMOTE_DIR/deploy/secrets/lium/'* \
       '$REMOTE_DIR/deploy/secrets/openrouter/api_key' \
       '$REMOTE_DIR/deploy/secrets/design/annotator_tokens' \
  && chown -R 65532:65532 '$REMOTE_DIR/deploy/secrets/lium' \
       '$REMOTE_DIR/deploy/secrets/openrouter' \
       '$REMOTE_DIR/deploy/secrets/design' \
  && chmod -R a-w '$REMOTE_DIR/deploy/secrets/wallets' 2>/dev/null; \
  chown -R 65532:65532 '$REMOTE_DIR/deploy/secrets/wallets' 2>/dev/null; true"

# Design sandbox staging root. The challenge container stages bind sources
# here and hands them to the host's Docker daemon, which resolves bind sources
# on the host filesystem, so the path must exist on the host with the container
# uid as owner, or 'docker compose' would auto-create it root-owned and every
# run would fail on staging I/O.
ssh_h "install -d -m 0775 -o 65532 -g 65532 '$STATE_ROOT/design/staging'"


# Materialize missing env from examples (dev-safe placeholders) if absent
ssh_h "bash -s" <<EOS
set -euo pipefail
cd '$REMOTE_DIR'
for ex in deploy/env/*.env.example; do
  base="\${ex%.example}"
  if [[ ! -f "\$base" ]]; then
    cp "\$ex" "\$base"
    # Prefer BASE_ keys from examples; strip any GBASE leftover
    sed -i 's/^GBASE_/BASE_/g' "\$base"
    chmod 600 "\$base"
    echo "created \$base from example"
  else
    sed -i 's/^GBASE_/BASE_/g' "\$base" || true
  fi
done
EOS

COMPOSE_FILES=(-f docker-compose.yml)
PROFILE_ARGS=()
case "$ROLE" in
  master)
    COMPOSE_FILES+=(-f deploy/compose/role-master.yml)
    PROFILE_ARGS=(--profile master)
    ;;
  validator)
    COMPOSE_FILES+=(-f deploy/compose/role-validator.yml)
    ;;
esac
case "$ENV" in
  staging) COMPOSE_FILES+=(-f deploy/compose/env-staging.yml) ;;
  prod)    COMPOSE_FILES+=(-f deploy/compose/env-prod.yml) ;;
esac

GE_EXPORT=""
if [[ -n "$GATEWAY_ENDPOINT" ]]; then
  GE_EXPORT="export BASE_GATEWAY_ENDPOINT='$GATEWAY_ENDPOINT';"
fi

if [[ "$BUILD_FROM" == "prebuilt" ]]; then
  echo "remote-deploy: sync release binaries"
  ssh_h "mkdir -p '$REMOTE_DIR/target/release'"
  rsync -az -e "$RSYNC_SSH" \
    "$ROOT/target/release/validator" \
    "$ROOT/target/release/gateway" \
    "$ROOT/target/release/updater" \
    "$ROOT/target/release/prism-challenge" \
    "$ROOT/target/release/design-challenge" \
    "$ROOT/target/release/design-egress-proxy" \
    "$HOST:$REMOTE_DIR/target/release/"
fi

echo "remote-deploy: build + up"
# shellcheck disable=SC2029
ssh_h "bash -s" <<EOS
set -euo pipefail
cd '$REMOTE_DIR'
$GE_EXPORT
export BASE_DOCKER_BUILD_FROM='$BUILD_FROM'
export COMPOSE_PROJECT_NAME=base
GHCR_PREFIX='${GHCR_PREFIX}'
BUILD_FROM='$BUILD_FROM'
ENV_NAME='$ENV'
PIN_PATH="deploy/pins/\${ENV_NAME}.json"

pull_retag() {
  local pull_ref="\$1"
  local local_tag="\$2"
  echo "remote-deploy: docker pull \$pull_ref"
  docker pull "\$pull_ref"
  docker tag "\$pull_ref" "\$local_tag"
  echo "remote-deploy: retagged → \$local_tag"
}

ghcr_pull_ref() {
  # Prefer full registry image from pin; else GHCR prefix + service@digest.
  local service="\$1"
  local image="\$2"
  local digest="\$3"
  if [[ "\$image" == */*@sha256:* ]]; then
    printf '%s' "\$image"
  else
    printf '%s/%s@%s' "\$GHCR_PREFIX" "\$service" "\$digest"
  fi
}

if [[ "\$BUILD_FROM" == "registry" ]]; then
  echo "remote-deploy: registry mode — pull GHCR digests (no compose build)"
  [[ -f "\$PIN_PATH" ]] || { echo "missing \$PIN_PATH" >&2; exit 1; }
  COMMIT_SHA=\$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("commit_sha",""))' "\$PIN_PATH")
  # Required pin services → local Compose tags (<svc>:0.1.0).
  python3 - "\$PIN_PATH" <<'PY' | while IFS=\$'\t' read -r service image digest; do
import json, sys
pin = json.load(open(sys.argv[1], encoding="utf-8"))
for svc, meta in sorted((pin.get("services") or {}).items()):
    print(f"{svc}\t{meta.get('image','')}\t{meta.get('digest','')}")
PY
    [[ -n "\$service" ]] || continue
    ref=\$(ghcr_pull_ref "\$service" "\$image" "\$digest")
    pull_retag "\$ref" "\${service}:0.1.0"
  done
  # Optional challenge / agent images from deploy/digests/<sha>.json when present.
  DIGESTS_FILE="deploy/digests/\${COMMIT_SHA}.json"
  if [[ -f "\$DIGESTS_FILE" ]]; then
    echo "remote-deploy: optional pulls from \$DIGESTS_FILE"
    python3 - "\$DIGESTS_FILE" <<'PY' | while IFS=\$'\t' read -r service image digest tag; do
import json, sys
optional = {
    "prism-challenge",
    "design-challenge",
    "design-egress-proxy",
    "design-runtime",
    "design-review",
    "base-attest-helper",
}
data = json.load(open(sys.argv[1], encoding="utf-8"))
images = data.get("images") or {}
for svc in sorted(optional):
    meta = images.get(svc) or {}
    digest = (meta.get("digest") or "").strip()
    if not digest.startswith("sha256:"):
        continue
    image = (meta.get("image") or "").strip()
    tag = (meta.get("tag") or f"{svc}:0.1.0").strip()
    # Prefer local compose tag name (last path segment) when tag is a registry ref.
    if "/" in tag and ":" in tag:
        tag = tag.rsplit("/", 1)[-1]
    if ":" not in tag:
        tag = f"{svc}:0.1.0"
    print(f"{svc}\t{image}\t{digest}\t{tag}")
PY
      [[ -n "\$service" ]] || continue
      ref=\$(ghcr_pull_ref "\$service" "\$image" "\$digest")
      pull_retag "\$ref" "\$tag"
    done
  else
    echo "remote-deploy: no \$DIGESTS_FILE — skipping optional design/prism/attest-helper pulls"
    echo "remote-deploy: (those images are only required for staging --build-from source)"
  fi
  # design-runtime is not a compose service; retag for sandbox pulls when present.
else
  # Build service images from current tree (source) or prebuilt binaries.
  docker compose ${COMPOSE_FILES[*]} ${PROFILE_ARGS[*]} build
  # Sandbox runtime image (not a long-running compose service).
  docker build -f deploy/Dockerfile --target design-runtime -t design-runtime:0.1.0 .
  # Anti-cheat review image (one-shot containers spawned by design-challenge).
  docker build -f deploy/Dockerfile --target design-review -t design-review:0.1.0 .
fi

# The updater can only pull from a registry. Enable it only when the desired
# image is a registry reference (host/path@sha256:…), otherwise every tick 404s
# trying to pull a locally-built tag from Docker Hub.
UP_PROFILE=""
desired=\$(sed -n 's/^BASE_UPDATER_DESIRED_IMAGE=//p' deploy/env/updater.env 2>/dev/null | tail -1)
if [[ -z "\$desired" && -f "deploy/pins/\${ENV_NAME}.desired.env" ]]; then
  desired=\$(sed -n 's/^BASE_UPDATER_DESIRED_IMAGE=//p' "deploy/pins/\${ENV_NAME}.desired.env" | tail -1)
fi
case "\$desired" in
  */*) UP_PROFILE="--profile auto-update"; echo "updater enabled (registry image)" ;;
  *)
    echo "updater disabled (desired image '\$desired' is not a registry reference)"
    # Deselecting a profile does not remove an already-running container.
    docker compose ${COMPOSE_FILES[*]} --profile auto-update rm -sf updater >/dev/null 2>&1 || true
    ;;
esac
UP_ARGS=(up -d --remove-orphans)
if [[ "\$BUILD_FROM" == "registry" ]]; then
  UP_ARGS+=(--no-build)
fi
docker compose ${COMPOSE_FILES[*]} ${PROFILE_ARGS[*]} \$UP_PROFILE "\${UP_ARGS[@]}"
# Profile-disabled services are not started, but an older compose project may
# still be running them. On validator, force-remove master-only challenge
# surfaces so smoke health does not see stale unhealthy containers.
if [[ '$ROLE' == 'validator' ]]; then
  docker compose ${COMPOSE_FILES[*]} rm -sf \
    prism-challenge design-challenge design-egress-proxy socket-proxy \
    >/dev/null 2>&1 || true
fi
docker compose ${COMPOSE_FILES[*]} ${PROFILE_ARGS[*]} \$UP_PROFILE ps
# Local health probes via published tunnels if present, else container exec.
sleep 5
if curl -fsS -m 5 http://127.0.0.1:18080/healthz >/dev/null 2>&1; then
  echo "validator tunnel health: \$(curl -fsS -m 5 http://127.0.0.1:18080/healthz)"
elif docker compose ${COMPOSE_FILES[*]} ${PROFILE_ARGS[*]} exec -T validator curl -fsS -m 5 http://127.0.0.1:8080/healthz >/dev/null 2>&1; then
  echo "validator health: ok (in-container)"
else
  echo "validator health: probe deferred (container may still be starting)"
fi
if [[ '$ROLE' == 'master' ]]; then
  if docker compose ${COMPOSE_FILES[*]} ${PROFILE_ARGS[*]} exec -T gateway curl -fsS -m 5 http://127.0.0.1:8080/healthz >/dev/null 2>&1; then
    echo "gateway health: ok"
  else
    echo "gateway health: probe deferred"
  fi
  # Registry is in-memory — re-seed challenge backends after every redeploy.
  echo "remote-deploy: registering challenge backends"
  python3 - <<'PY'
import json, urllib.request, urllib.error
backends = [
    ("prism", "http://prism-challenge:8092"),
    ("design", "http://design-challenge:8093"),
]
for cid, url in backends:
    payload = json.dumps({"challenge_id": cid, "base_url": url, "weight": 1}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8080/v1/admin/backends",
        data=payload,
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"backend {cid} → {url} (HTTP {resp.status})")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        if e.code == 409:
            print(f"backend {cid} → {url} (HTTP 409 already present)")
        else:
            print(f"backend {cid} register failed HTTP {e.code}: {body}", flush=True)
    except Exception as e:
        print(f"backend {cid} register failed: {e}", flush=True)
PY
  curl -fsS -m 5 http://127.0.0.1:8080/challenge/prism/health >/dev/null \
    && curl -fsS -m 5 http://127.0.0.1:8080/challenge/design/health >/dev/null \
    && echo "challenge proxy health: ok" \
    || echo "challenge proxy health: deferred (backends may still be starting)"
fi
EOS

echo "remote-deploy: done ($ROLE @ $HOST)"
