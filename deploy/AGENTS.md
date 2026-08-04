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

Verify: `./deploy/scripts/assert-compose-matrix.sh`.  
Root `docker-compose.staging-*.yml` overrides are **obsolete** — use `deploy/compose/` only.

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
