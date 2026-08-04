# AGENTS.md — BASE control plane

Short contract for agents and operators. Prefer linking over restating runbooks.

## Monorepo map

| Path | Role |
|------|------|
| `bins/` | Runnable processes (validator, gateway, updater, challenges, …) |
| `crates/` | Libraries shared by binaries |
| `xtask/` | Repo gates and maintenance tasks |
| `deploy/` | Compose matrix, Terraform, pins, secrets helpers, remote deploy |
| `docs/` | Architecture, frozen specs, runbooks, completeness |
| `config/` | Shared non-secret configuration |

Working branch: **`dev`**. Prod ships from annotated tags `v*.*.*` cut on `dev`.

## Non-negotiables

- **Digest-only images** in deploy paths — no floating tags in prod pins/compose.
- **Secrets** via age + files under `deploy/env/` / `deploy/secrets/` — never baked into images or cloud-init.
- **Gateway runs on master only** (`--profile master` / `role-master.yml`). Validators point at the master gateway over VPC.
- **`evil-gateway` is test-only** — never enable on prod hosts; assert with `deploy/scripts/assert-evil-gateway-not-default.sh`.
- Platform is **DigitalOcean Droplets + Docker Compose**, not App Platform / DOKS.

## Required gates (before merge)

Match CI (`.github/workflows/ci.yml`):

- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- tests + `cargo deny`
- `cargo run -p xtask -- loc-cap`
- `cargo run -p xtask -- consensus-lint`
- `cargo run -p xtask -- spec-check`
- plus challenge/docs xtasks wired in CI (`agent-challenge-check`, `hypertraining-check`, `external-docs-check`)

## Where to read what

| Need | Start here |
|------|------------|
| System map / process topology | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| Deploy / Compose / DO topology | [`deploy/README.md`](deploy/README.md) + [`deploy/AGENTS.md`](deploy/AGENTS.md) |
| Doc authority vs evidence | [`docs/AGENTS.md`](docs/AGENTS.md) |
| Component status | [`docs/COMPLETENESS.md`](docs/COMPLETENESS.md) |
| Frozen contracts | [`docs/BUNDLE_SPEC.md`](docs/BUNDLE_SPEC.md), [`docs/AGENT_CHALLENGE.md`](docs/AGENT_CHALLENGE.md) |
| Threat / operator checklist | [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md), [`docs/OPERATOR_SECURITY.md`](docs/OPERATOR_SECURITY.md) |

## Do not commit

- `deploy/env/*.env` (materialized secrets)
- `deploy/secrets/**` (except documented `README.md` placeholders)
- `deploy/terraform/*.tfstate*` / `terraform.tfvars` / local `.terraform/`
- Age identities, wallets, `receipt_sk`, `*.pem` / `*.key` / `*.age`
- Treating `docs/evidence/` or `docs/spikes/` as product code or normative spec
