# AGENTS.md — deploy (DigitalOcean + Compose)

Operator/agent contract for staging and prod. Full procedures live in [`README.md`](README.md); do not duplicate them here.

## Topology (4 droplets, NYC1)

| Host | Role | Gateway |
|------|------|---------|
| `base-staging` | staging master | yes (`role-master` + `env-staging`) |
| `base-staging-validator` | staging validator | no — VPC → staging master `:8080` |
| `base-prod` | prod master | yes (`role-master` + `env-prod`) |
| `base-prod-validator` | prod validator | no — VPC → prod master `:8080` |

Terraform: [`terraform/`](terraform/). Firewall: SSH from operator IP; CI uses ephemeral `/32` via `.github/actions/do-firewall` (always tear down). Spaces for Postgres backups (promote/restore).

## Compose matrix

`remote-deploy.sh --env staging|prod --role master|validator` stacks:

| File | Purpose |
|------|---------|
| `compose/role-master.yml` | gateway profile, VPC publish |
| `compose/role-validator.yml` | no gateway; external gateway endpoint |
| `compose/env-staging.yml` | testnet 541, faster coordination |
| `compose/env-prod.yml` | mainnet, conservative intervals |
| `compose/env-local.yml` | **local only** — ports/smoke knobs/tunnel env; always on top of `env-staging` |

Verify: `./deploy/scripts/assert-compose-matrix.sh`.  
Root `docker-compose.staging-*.yml` overrides are **obsolete** — use `deploy/compose/` only.  
`remote-deploy.sh` never selects `env-local*.yml`.

## Postgres vs ephemeral state

Compose always runs a digest-pinned `postgres` service (`base-pgdata` volume, healthcheck, `deploy/env/postgres.env`). App `BASE_DATABASE_URL` must match that file (materialize via `./deploy/scripts/materialize-env.sh`; local-e2e also injects `LOCAL_DATABASE_URL` from it).

| Data | Store |
|------|--------|
| Design harnesses / runs / stages / artifacts metadata / admin rounds | **Postgres** (`design_*`) |
| Prism submissions / stage events | **Postgres** (`prism_*`) |
| Gateway raw weight leaves + sealed bundles | **Postgres** (`raw_weight_snapshot`, `epoch_bundle`, …) |
| Validator attestations (when DB configured) | **Postgres** |
| Design sandbox staging files | volume `${BASE_STATE_DIR}/design/staging` + `design-artifacts` |
| Gateway challenge **backend registry** | **in-memory** — re-seed after gateway restart (`remote-deploy.sh` does this on master) |
| site-api (`GET /v1/site/*`) | no DB — proxies challenge upstreams via gateway |
| Unit/integration tests | may construct `Memory*Store` directly; omit `BASE_DATABASE_URL` only there |

Migrations (`crates/db/migrations`) run on boot in gateway / design-challenge / prism-challenge when `BASE_DATABASE_URL` is set. Compose requires `deploy/env/{design,prism}-challenge.env` so challenges cannot silently boot on memory.

Verify rows (local master stack):

```bash
docker compose -f docker-compose.yml exec -T postgres \
  sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c \
  "SELECT COUNT(*) FROM design_harness; SELECT COUNT(*) FROM prism_submission;"'
```

## Local testnet E2E

Full procedure: [`docs/runbooks/local-testnet-e2e.md`](../docs/runbooks/local-testnet-e2e.md).

```bash
./deploy/scripts/materialize-env.sh
./deploy/scripts/local-e2e.sh --dry-run          # plan + compose render
./deploy/scripts/local-e2e.sh --smoke            # healthz + weights seal smoke + tunnel
./deploy/scripts/local-e2e.sh --live             # owner wallet + REQUIRE_OWNER=1
./deploy/scripts/local-e2e.sh --down
```

| Prereq | smoke | live |
|--------|-------|------|
| Docker, Compose v2 | yes | yes |
| `cloudflared` (or `--no-tunnel`) | yes | yes |
| `deploy/env/*.env` (examples OK) | yes | yes |
| `gateway_sk` (seal) + `prism_sk` / `design_sk` (leaf sigs; pubs ↔ trust root) | yes (prefer `~/.base-secrets/challenge-*.sk`) | real preferred |
| `deploy/secrets/wallets/base-owner` | **no** (not needed for `/v1/weights/latest`) | **yes** (netuid 541 owner) |
| `base-validator` wallet | **no** (fetch-only) | for on-chain weight submit |
| Fresh `target/release/{gateway,validator,…}` (or `BASE_DOCKER_BUILD_FROM=source`) | recommended | **required** for real chain |

**Weights seal smoke (default on `--smoke`):** after healthz, `local-e2e.sh` runs `weights-smoke` — signed prism leaves for the live metagraph → `POST /v1/admin/seal` → assert `GET /v1/weights/latest` is **200** with **`sealed: true`**. Skip with `--no-weights-smoke`. Pre-seal, latest is **200 burn** (`sealed: false`, uid 0 = 100%) — never 404; that is unrelated to a missing gateway owner wallet.

**Challenge verification:** on **master** only (validator has **no challenge exec**). Simulate submissions end-to-end — submit **baseline** + submit **cheat**, poll `/v1/runs/{id}` + `/events` + `/logs`, probe edges (bad harness, sanitize, quota, routes), then **admin winners** (`GET/POST /v1/admin/rounds/{id}/…` with bearer from `deploy/secrets/design/annotator_tokens`) and confirm leaf → seal → `GET /v1/weights/latest` **`sealed: true`**. **Never host Sim in staging/prod** (`BASE_ALLOW_HOST_SIM` / host `SimSandbox` are CI/local only). Healthz alone is insufficient.

Tunnel writes gitignored `deploy/env/local-tunnel.env` (`BASE_GATEWAY_PUBLIC_URL`). Co-located validator stays on `http://gateway:8080`; external clients use the tunnel URL. Host probe ports default to `2808x` (avoid staging SSH on `1808x`).

## CI: staging vs prod

| Lane | Trigger | Build stance |
|------|---------|--------------|
| Staging | CI green on `dev` (`deploy-staging.yml`) | `--build-from source` on droplet OK for iteration |
| Images | Push to `dev` (`images.yml`) | Build/push GHCR digests; promote + **commit** `deploy/pins/staging.json` + `deploy/digests/<sha>.json` |
| Prod | Tag `v*.*.*` (`deploy-prod.yml`) | **`--build-from registry` only** — promote staging→prod pins, pull GHCR digests; no Rust source build on prod hosts |

Ladder: CI → GHCR digests → `deploy/pins/staging.json` (committed by `images.yml`) → tag → preflight (CI + staging pins match tag SHA) → `promote.sh` → `remote-deploy.sh --build-from registry`. Details: [`README.md`](README.md) § Auto CI deploy and § Promotion pipeline.

## Secrets / age

- Identity OOB on host: `/etc/base/age-identity.txt` (or `AGE_IDENTITY`) — never in Terraform/cloud-init.
- Materialize: `./deploy/scripts/materialize-env.sh` → `deploy/env/*.env` mode **0600**.
- Runtime secret files (wallets, keys): mode **0400**, owner **uid 65532**.
- Helpers: `age-encrypt-env.sh`, `age-push-env.sh`. Checklist: [`docs/OPERATOR_SECURITY.md`](../docs/OPERATOR_SECURITY.md).

## First prod tag checklist

1. Staging healthy on the exact commit you will tag; `deploy/pins/staging.json` `commit_sha` matches that SHA.
2. Digests recorded / promoted for services you will ship (`promote.sh`, `verify-task-43.sh` locally if needed).
3. Age identity + env ages present on both prod hosts; wallets hotkeys under `deploy/secrets/wallets/` (0400 / 65532).
4. Mainnet owner wallet placed; set `BASE_GATEWAY_REQUIRE_OWNER=1` when ready (ops gap until then — see [`docs/COMPLETENESS.md`](../docs/COMPLETENESS.md)).
5. Cut `vX.Y.Z` on `dev`, push tag; pass `deploy-prod` preflight + `environment: production` reviewers.
6. Smoke `/healthz` on both prod hosts; confirm `evil-gateway` absent.

## Out of scope for agents (ops)

- DO Spaces credentials in GitHub (`BASE_BACKUP_ENDPOINT`, `SPACES_*`) for fail-closed prod backup
- Mainnet owner wallet + `BASE_GATEWAY_REQUIRE_OWNER=1`
- GitHub `production` environment required reviewers / `dev` branch protection
- TLS ACME termination (ports 80/443 open; ACME not fully shipped)
- Terraform remote state backend (recommended, not blocking app deploy)
- Bootstrap of age/secrets on brand-new droplets (OOB)
