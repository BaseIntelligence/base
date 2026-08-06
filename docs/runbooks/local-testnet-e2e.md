# Local testnet E2E runbook

Run the full BASE control-plane stack (master + gateway + validator + challenges)
on a developer machine against **Bittensor Finney testnet netuid 541**, with an
ephemeral public gateway URL via **cloudflared quick tunnel**.

Normative pointers: [`../../AGENTS.md`](../../AGENTS.md), [`../../deploy/AGENTS.md`](../../deploy/AGENTS.md).  
Staging droplet procedure (different path): [`staging-testnet-e2e.md`](staging-testnet-e2e.md).

## Feasibility

**Yes**, with caveats:

| Piece | Local? | Notes |
|-------|--------|-------|
| Compose master + staging overlays | yes | `role-master` + `env-staging` + `env-local` |
| Live chain connect (testnet 541) | yes | Needs egress to `wss://test.finney.opentensor.ai:443` |
| Gateway `/healthz` | yes | Gateway always uses live chain (no fake backend) |
| Sealed `/v1/weights/latest` | yes | **`--smoke`** runs `weights-smoke` (leaves + admin seal). Needs `prism_sk` + `gateway_sk`, **not** a gateway owner wallet |
| Ephemeral public URL | yes | cloudflared quick tunnel → host `:8080` |
| Owner fail-closed (`REQUIRE_OWNER=1`) | yes | **`--live`** needs `base-owner` wallet files |
| On-chain weight submit / epoch dispatch | yes | Needs **validator** wallet + real `challenge_sk` matching trust root |
| DigitalOcean / Terraform | out of scope | Do not use this path for droplet deploy |

## Prerequisites

- Docker Engine + Compose v2
- `cloudflared` on `PATH` (unless `--no-tunnel`)
- Repo checkout with ability to build images (`BASE_DOCKER_BUILD_FROM=prebuilt` after `cargo build --release` of service bins, or `source` for in-Docker rustc)
- Disk for images + `.local/base-state/`

### Secrets

| Mode | Required |
|------|----------|
| `--smoke` | `gateway_sk` (seal) + `prism_sk` / `design_sk` with pubs matching the local trust root. Prefers `~/.base-secrets/challenge-*.sk` when they match `config/challenges.toml`; otherwise mints and rebuilds `.local/trust-root`. Public-only `BASE_GATEWAY_HOTKEY` is enough (advisory owner check). **No gateway owner wallet.** |
| `--live` | `deploy/secrets/wallets/base-owner` (btcli layout; must be on-chain owner of netuid 541). Prefer real seal/challenge secrets matching `config/`. Validator wallet for **on-chain** weight submit. |

Never commit wallets or `deploy/env/*.env` / `deploy/env/local-tunnel.env`.

**Roles:** gateway owner wallet / `REQUIRE_OWNER` = master identity check only. Serving sealed weights uses `gateway_sk` + challenge leaf sigs. Validators fetch `/v1/weights/latest`; they do not need a gateway wallet.

```bash
# Env examples → mode 0600 files
./deploy/scripts/materialize-env.sh

# Host layout for wallets (live)
# deploy/secrets/wallets/base-owner/hotkeys/default
# deploy/secrets/wallets/base-validator/hotkeys/default
chown -R 65532:65532 deploy/secrets/wallets
chmod -R u=rX,go= deploy/secrets/wallets
```

## Commands

```bash
# Plan only (no containers)
./deploy/scripts/local-e2e.sh --dry-run

# Smoke: stack + healthz + weights seal smoke + tunnel (default)
./deploy/scripts/local-e2e.sh --smoke

# Live: owner fail-closed + challenge dispatch on
./deploy/scripts/local-e2e.sh --live

# Healthz only (skip leaf→seal→weights/latest)
./deploy/scripts/local-e2e.sh --smoke --no-weights-smoke

# No public URL
./deploy/scripts/local-e2e.sh --smoke --no-tunnel

# Re-run seal smoke against an already-up gateway
cargo run -q --release -p weights-smoke -- \
  --gateway http://127.0.0.1:8080 \
  --challenge-sk deploy/secrets/prism_sk

# Teardown
./deploy/scripts/local-e2e.sh --down
```

Challenge verification must **simulate a submission** (harness/intake) and probe failures (bad harness, sanitize, quota, routes) in addition to the weights seal smoke above — see root [`AGENTS.md`](../../AGENTS.md).

Compose matrix equivalent (what the script runs):

```bash
export BASE_STATE_DIR="$PWD/.local/base-state"
export LOCAL_REQUIRE_OWNER=0          # 1 for live
export LOCAL_GATEWAY_HOTKEY=<64 hex>  # smoke without wallet
docker compose \
  -f docker-compose.yml \
  -f deploy/compose/role-master.yml \
  -f deploy/compose/env-staging.yml \
  -f deploy/compose/env-local.yml \
  --profile master up -d
```

## Health checks

Local overlay uses **2808x** loopback ports so they do not collide with common
staging SSH tunnels on `18080`/`18090`. Override with `LOCAL_*_HOST_PORT`.

| Service | URL |
|---------|-----|
| gateway | `http://127.0.0.1:8080/healthz` |
| sealed weights | `http://127.0.0.1:8080/v1/weights/latest` (200 burn `sealed:false` before smoke; `sealed:true` after) |
| validator | `http://127.0.0.1:28080/healthz` |
| prism | `http://127.0.0.1:28092/health` |
| design | `http://127.0.0.1:28093/health` |

Look for validator coordination (`Match epoch=`) and, when a signing key is loaded,
`Match → submit_intent` / `submit_timelocked ok` (or `set_weights ok` if CR is off):

```bash
docker compose \
  -f docker-compose.yml \
  -f deploy/compose/role-master.yml \
  -f deploy/compose/env-staging.yml \
  -f deploy/compose/env-local.yml \
  --profile master logs -f validator
```

## Tunnel URL

`local-e2e.sh` starts an isolated quick tunnel (dedicated `--config` under
`.local/` so a host `/etc/cloudflared/config.yml` named-tunnel catch-all cannot
404 the URL):

```bash
cloudflared tunnel --config .local/cloudflared-quick.yml --url http://127.0.0.1:8080
```

and writes gitignored `deploy/env/local-tunnel.env`:

```bash
BASE_GATEWAY_PUBLIC_URL=https://<random>.trycloudflare.com
BASE_DOMAIN=<random>.trycloudflare.com
```

- **Inside compose**: validator keeps `BASE_GATEWAY_ENDPOINT=http://gateway:8080`.
- **Outside compose** (remote validator / miner):  
  `export BASE_GATEWAY_ENDPOINT=$BASE_GATEWAY_PUBLIC_URL`

Quick tunnels are ephemeral (new hostname each run; no uptime SLA). Do not use for prod.

## Teardown

```bash
./deploy/scripts/local-e2e.sh --down
# or:
docker compose -f docker-compose.yml -f deploy/compose/role-master.yml \
  -f deploy/compose/env-staging.yml -f deploy/compose/env-local.yml \
  --profile master down --remove-orphans
pkill -f 'cloudflared tunnel --url http://127.0.0.1:8080' || true
rm -f deploy/env/local-tunnel.env
```

## Compose matrix caveats

- `env-local.yml` is **local-only**. `remote-deploy.sh` never selects it.
- `assert-compose-matrix.sh` still validates staging/prod role×env; local overlay must not introduce `fake_owner` / `BASE_CHAIN_BACKEND`.
- `docker-compose.e2e.yml` (`fake_owner`) is legacy cleartext harness — not the testnet path.
- Agent-v1 / Phala CVM miner overlays are removed; miners submit over HTTP (see [`../external-miner/`](../external-miner/)).
- `env-staging` sets `BASE_GATEWAY_REQUIRE_OWNER=1`; `env-local` overrides via `LOCAL_REQUIRE_OWNER` (smoke defaults to `0`).
- Host probe ports default to `2808x` (not role-master `1808x`) to avoid staging SSH tunnels.
- `BASE_DOCKER_BUILD_FROM=prebuilt` copies host `target/release/*` into a Debian bookworm image. Binaries built on a newer glibc host (e.g. needing `GLIBC_2.39`) will crash in-container — use `BASE_DOCKER_BUILD_FROM=source` or rebuild on bookworm. `local-e2e.sh` may treat prism/design health as soft-fail so gateway+validator smoke can still complete while deploy-wiring finishes.
- Rebuild gateway/validator from **current** `main` before `--live`: stale `target/release/gateway` may still contain the removed `fake_owner` path and will not exercise real testnet owner checks.
- If the host runs a system named tunnel (`/etc/cloudflared/config.yml`), `local-e2e.sh` passes a dedicated `--config` under `.local/` so the named-tunnel catch-all cannot 404 the quick URL.
