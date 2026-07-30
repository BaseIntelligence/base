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
- Stub member: `crates/gbase-workspace-smoke` (keeps `cargo metadata` green)
- Gates: `cargo fmt`, `clippy -D warnings`, `test`, `cargo deny`, `xtask loc-cap`, `xtask consensus-lint`
- CI: [`.github/workflows/ci.yml`](./.github/workflows/ci.yml) on push/PR to `reborn`

## Gateway (master-only)

`gbase-gateway` is the subnet-owner process (D3). On startup it resolves on-chain
`SubnetOwnerHotkey` via `ChainClient` and compares it to `GBASE_GATEWAY_HOTKEY`
(32-byte hex). On mismatch it emits a structured fatal log and **exits 2 before
binding any listener**. On match it serves the telemetry router (`/healthz`,
`/readyz`, `/metrics`).

Compose (task 40): the gateway service MUST use Docker Compose profile **`master`**
so a default `docker compose up` does **not** start the gateway; operators run
`docker compose --profile master up` on the owner host only. Full stack wiring
(postgres, validator, updater, socket-proxy digests) lands in task 40 — this
skeleton only documents the profile contract.

## License

Apache License 2.0 — see [LICENSE](./LICENSE).
