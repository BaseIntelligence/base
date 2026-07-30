# gbase

Rust Bittensor subnet workspace for **BaseIntelligence**.

Greenfield successor path for Base Intelligence subnet work: validators, miners, and shared crates live here (workspace layout lands in a follow-on commit). This repository is intentionally separate from [`BaseIntelligence/base`](https://github.com/BaseIntelligence/base) (Python Metis stack); settings and history do **not** inherit from `base`.

## Branch

- Default branch: **`reborn`** (only long-lived branch at bootstrap).
- All PRs target `reborn`.

## Toolchain

- Host / workspace pin: **Rust 1.96.0** via [`rust-toolchain.toml`](./rust-toolchain.toml).
- Bittensor SDK pin used by consumers: `bittensor` / related crates may still require **1.89** via a directory-level `rust-toolchain.toml` override (see SDK pin `e4ffa2e1325c6c7db618dbceaf396310a170990c` from the gbase plan task 4). Dual-toolchain is expected until upstream catches up.

## Layout

- Cargo workspace (`resolver = "3"`): `crates/*`, `bins/*`, `xtask`
- Stub member: `crates/workspace-smoke` (keeps `cargo metadata` green)
- Gates: `cargo fmt`, `clippy -D warnings`, `test`, `cargo deny`, `xtask loc-cap`, `xtask consensus-lint`, `xtask spec-check`, `xtask agent-challenge-check`, `xtask external-docs-check`
- CI: [`.github/workflows/ci.yml`](./.github/workflows/ci.yml) on push/PR to `reborn`
- Docs: [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md), frozen [`docs/BUNDLE_SPEC.md`](./docs/BUNDLE_SPEC.md) + [`docs/AGENT_CHALLENGE.md`](./docs/AGENT_CHALLENGE.md), [`docs/THREAT_MODEL.md`](./docs/THREAT_MODEL.md), runbooks under `docs/runbooks/`, miner-facing [`docs/external-miner/`](./docs/external-miner/)

## Gateway (master-only)

`gateway` is the subnet-owner process (D3). On startup it resolves on-chain
`SubnetOwnerHotkey` via `ChainClient` and compares it to `GBASE_GATEWAY_HOTKEY`
(32-byte hex). On mismatch it emits a structured fatal log and **exits 2 before
binding any listener**. On match it serves:

- Telemetry: `/healthz`, `/readyz`, `/metrics`
- Backend registry (routing only — D18): `/v1/admin/backends` CRUD (no signing keys)
- Challenge reverse proxy: `/challenge/{id}/*` with round-robin, passive ejection
  after N failures, re-admission after cooldown
- Sole TLS owner (D20): cleartext by default; `rustls-acme` DNS-01 is task 42

Env knobs: `GBASE_GATEWAY_LISTEN`, `GBASE_GATEWAY_HOTKEY`,
`GBASE_GATEWAY_FAIL_THRESHOLD` (default 3), `GBASE_GATEWAY_COOLDOWN_SECS` (default 30),
`GBASE_GATEWAY_TLS*` (stub until task 42).

Compose: see [`docker-compose.yml`](./docker-compose.yml) and [`deploy/README.md`](./deploy/README.md).
The gateway service MUST use Docker Compose profile **`master`**
so a default `docker compose up` does **not** start the gateway; operators run
`docker compose --profile master up` on the owner host only.

## Compose stack

Host control plane (task 40):

| Service | Profile | Notes |
|---------|---------|--------|
| postgres | default | Postgres 16, volume + healthcheck, digest-pinned |
| validator | default | multi-stage `deploy/Dockerfile` target `validator` |
| updater | default | digest-pinned rollouts via socket-proxy |
| socket-proxy | default | **only** mount of `/var/run/docker.sock` |
| gateway | **`master`** | owner host only |

```bash
./deploy/scripts/materialize-env.sh   # age-decrypt or copy examples → deploy/env/*.env (0600)
docker compose up -d                  # 4 services, no gateway
docker compose --profile master up -d # 5 services including gateway
```

Secrets are age-decrypted env files at mode 0600 — never baked into images or cloud-init.

## License

Apache License 2.0 — see [LICENSE](./LICENSE).
