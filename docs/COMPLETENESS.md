# Base completeness matrix

Honest per-component status as of `dev` HEAD. Updated as phases land.

## Legend

| Tag | Meaning |
|-----|---------|
| **done** | Implemented, tested, wired into a running binary. |
| **fake** | Code exists and passes tests, but the running binary uses `FakeChain` / `fake_owner` — not live data. |
| **lib-only** | Library crate is complete; no binary drives it in production. |
| **stub** | Trait method returns `NotImplemented`; placeholder for future wiring. |
| **missing** | No code, no compose service, no CI image. |

## Chain layer

| Component | Status | Notes |
|-----------|--------|-------|
| `ChainClient` trait | done | 14 methods, full trait surface. |
| `FakeChain` | done | Deterministic in-memory, used by all binaries today. |
| `NotImplementedChain` | stub | Every method returns `Err(NotImplemented)`. |
| `LiveRpcChain` (feature `live`) | stub | `current_block` + `block_hash` only; `metagraph_at`, `set_weights`, `submit_timelocked_weights` all `NotImplemented`. |
| `chain-live` crate | **done** | New crate: full JSON-RPC reads + sr25519 signed `set_weights` / `commit_timelocked_mechanism_weights`. Wired into `bins/validator` and `bins/gateway` via `BASE_CHAIN_BACKEND=live`. |
| `BASE_CHAIN_ENDPOINT` | done | Read by `config::Config`; consumed by `chain-live::LiveChainClient::connect`. |
| CRV4 tlock encryption | **deferred** | Drand BLS12-381 IBE not in the dependency graph. Extrinsic path ships; encryption is a spike issue. |

## Validator

| Component | Status | Notes |
|-----------|--------|-------|
| Health endpoints (`/healthz`, `/readyz`, `/metrics`) | done | |
| Attestation (`/v1/attest/*`) | done | `AttestState::with_ok_verifier` / `with_pcs_timeout`. |
| Bundle fetch + `compare_bundle` | done | Continuous coordination loop. |
| `set_weights` / `submit_timelocked_weights` | fake | `FakeChain` accepts; no live extrinsic submission. |
| Live chain backend switch | done | `BASE_CHAIN_BACKEND=live` in `bins/validator` uses `chain_live::LiveChainClient`; `fake` stays default. |

## Gateway

| Component | Status | Notes |
|-----------|--------|-------|
| Master check (`SubnetOwnerHotkey`) | fake | `BASE_CHAIN_BACKEND=fake_owner` — owner hotkey == configured hotkey. |
| Registry + proxy | done | |
| Bundle seal (`POST /v1/weights/raw` → `GET /v1/weights/latest`) | done | |
| Live chain backend switch | done | `BASE_CHAIN_BACKEND=live` in `bins/gateway` uses `chain_live::LiveChainClient`; `fake_owner` stays default. |

## agent-challenge

| Component | Status | Notes |
|-----------|--------|-------|
| Health + pack routes | done | `/healthz`, `/readyz`, pack catalog. |
| `run_epoch_dispatch` | lib-only | Full dispatch logic in `crates/agent-challenge/src/epoch_loop.rs`; called only from tests. |
| `submit_signed_leaf_set` | lib-only | Gateway client in `crates/agent-challenge/src/submit.rs`; called only from tests. |
| Daemon epoch driver | done | Background task in `bins/agent-challenge` drives `run_epoch_dispatch` → `submit_signed_leaf_set`. Enabled via `BASE_CHALLENGE_DISPATCH=1`. |

## hypertraining-challenge

| Component | Status | Notes |
|-----------|--------|-------|
| Crate (`crates/hypertraining-challenge`) | done | Full sim E2E pipeline. |
| Binary (`bins/hypertraining-challenge`) | done | Health + submit on `:8091`. |
| Compose service | done | `docker-compose.yml` service on `:8091`. |
| Dockerfile target | done | `deploy/Dockerfile` target `hypertraining-challenge`. |
| GHCR image | done | Added to `images.yml` matrix and `ghcr-public.yml`. |
| `ci.yml` xtask gate | done | `hypertraining-check` added to `ci.yml`. |

## prism-challenge

| Component | Status | Notes |
|-----------|--------|-------|
| Crate (`crates/prism-challenge`) | done | Lium client + sim backend + pipeline. |
| Binary (`bins/prism-challenge`) | done | Health + submit on `:8092`. |
| Compose service | done | Added to `docker-compose.yml` on `:8092`. |
| Dockerfile target | done | `deploy/Dockerfile` target `prism-challenge`. |
| GHCR image | done | Added to `images.yml` matrix and `ghcr-public.yml`. |

## Infrastructure

| Component | Status | Notes |
|-----------|--------|-------|
| Terraform droplets | done | 4 of 4: staging master, staging validator, prod master, prod validator. |
| Staging master | done | Migrated to `/opt/base` CI-managed; old `/opt/gbase` stack torn down. |
| Staging validator | done | Redeployed from same commit; `bundle gateway signature invalid` resolved. |
| Prod master | pending | `/opt/base` exists, awaiting first tag-based deploy. |
| `deploy-staging.yml` | done | Auto on CI green; passes `--env staging`, fail-closed health gate. |
| `deploy-prod.yml` | done | Tag-based (`v*.*.*`); preflight checks CI + staging pins; `environment: production`. |
| GitHub secrets | done | All 8 secrets set (staging + prod master + prod validator + gateway URLs). |
