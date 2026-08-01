# Base completeness matrix

Honest per-component status as of `dev` HEAD. Updated as phases land.

## Legend

| Tag | Meaning |
|-----|---------|
| **done** | Implemented, tested, wired into a running binary. |
| **sim** | Code exists and passes tests, but the running binary uses a simulated backend, not live data. |
| **lib-only** | Library crate is complete; no binary drives it in production. |
| **stub** | Trait method returns `NotImplemented`; placeholder for future wiring. |
| **test-only** | Compiled and exercised by tests; deliberately unreachable from any shipped binary. |
| **missing** | No code, no compose service, no CI image. |

## Chain layer

| Component | Status | Notes |
|-----------|--------|-------|
| `ChainClient` trait | done | 14 methods, full trait surface. |
| `FakeChain` | test-only | Deterministic in-memory. No longer reachable from any binary; used by unit and adversarial tests. |
| `NotImplementedChain` | stub | Every method returns `Err(NotImplemented)`. |
| `LiveRpcChain` (feature `live`) | stub | `current_block` + `block_hash` only; `metagraph_at`, `set_weights`, `submit_timelocked_weights` all `NotImplemented`. |
| `chain-live` crate | **done** | Full JSON-RPC reads (`Identity` hasher, `Keys` double-map enumeration, `ValueQuery` defaults) + sr25519 signed `set_weights` / `commit_timelocked_mechanism_weights`. The **only** backend in `bins/validator` and `bins/gateway`; both fail fast if the chain is unreachable. Four `#[ignore]` tests read live testnet 541. |
| `BASE_CHAIN_ENDPOINT` | done | Read by `config::Config`; consumed by `chain-live::LiveChainClient::connect`. |
| CRV4 tlock encryption | **deferred** | Drand BLS12-381 IBE not in the dependency graph. Extrinsic path ships; encryption is a spike issue. |

## Validator

| Component | Status | Notes |
|-----------|--------|-------|
| Health endpoints (`/healthz`, `/readyz`, `/metrics`) | done | |
| Attestation (`/v1/attest/*`) | done | Real Intel DCAP via `dcap-qvl` when built `--features dcap` (the container default). Verified against live Intel PCS; a tampered quote yields `CryptoInvalid`. Mock verifiers remain for tests only. |
| Bundle fetch + `compare_bundle` | done | Continuous coordination loop. |
| `set_weights` / `submit_timelocked_weights` | done | Live extrinsics. The signing key is derived from the Bittensor wallet mnemonic via `keystore` and installed with `LiveChainClient::set_signing_key`; without it submission fails closed. |
| Chain backend | done | Live only. `FakeChain` was removed from `bins/validator`; there is no switch left to misconfigure. |

## Gateway

| Component | Status | Notes |
|-----------|--------|-------|
| Master check (`SubnetOwnerHotkey`) | done | Read from the live chain. Advisory by default; `BASE_GATEWAY_REQUIRE_OWNER=1` makes it fail-closed (set in staging). |
| Registry + proxy | done | |
| Bundle seal (`POST /v1/weights/raw` → `GET /v1/weights/latest`) | done | |
| Chain backend | done | Live only. `fake_owner` was removed from `bins/gateway`. |

## agent-challenge

| Component | Status | Notes |
|-----------|--------|-------|
| Health + pack routes | done | `/healthz`, `/v1/catalog`, `/v1/packs/{id}`. |
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
| Prod master | pending | Droplet up, awaiting the mainnet owner wallet and the first `v*.*.*` tag. |
| `deploy-staging.yml` | done | Auto on CI green; passes `--env staging`, fail-closed health gate. |
| `deploy-prod.yml` | done | Tag-based (`v*.*.*`); preflight checks CI + staging pins; `environment: production`. |
| GitHub secrets | done | All 8 secrets set (staging + prod master + prod validator + gateway URLs). |

## Keys and identity

| Component | Status | Notes |
|-----------|--------|-------|
| `keystore` crate | done | BIP39 (pinned 2048-word English list) → Substrate `PBKDF2-HMAC-SHA512(entropy, "mnemonic"+password, 2048)` → sr25519. Cross-checked against `substrateinterface` and against all six local wallets. |
| Bittensor wallet reader | done | Reads `~/.bittensor/wallets/<name>/hotkeys/<hotkey>`; re-derives the key and rejects the file if the derived public key disagrees with the stored one. |
| Hotkey resolution | done | `keystore::resolve_*_from_env`: wallet → mnemonic file → secret-key file → public-only hex/SS58. A mnemonic is never read from a plain env var. |
| Gateway / validator hotkeys | done | Both resolve from `BASE_*_WALLET`. Staging uses `base-owner` (gateway) and `base-validator`. |
| Wallets on hosts | done | Only the hotkey file is shipped, mode 0400, owned by uid 65532, under `deploy/secrets/wallets/`. |

## Challenge backends

| Component | Status | Notes |
|-----------|--------|-------|
| deepagent task packs | done | Real pinned HuggingFace download of `BaseIntelligence/deepagent@5fe4e783` `tasks/` (9 packs). Path-traversal rejection, exec-bit preservation, `Link` pagination, LFS sha256 checks. Persisted in the `agent-pack-source` volume. |
| prism Lium backend | done | `PRISM_FORCE_SIM=false` in staging; the binary logs `eval_backend=lium`. API key is mounted from a file so it never appears in `docker inspect`. |
| Phala deploy | done | Invokes the real `phala` CLI. |

## Known gaps

| Gap | Impact |
|-----|--------|
| CRV4 tlock encryption | Drand BLS12-381 IBE is still absent; the extrinsic path ships without it. |
| DCAP verify holds the attest mutex | A cold Intel PCS fetch (up to 20 s) serialises attestation submissions. |
| DCAP error classification | Matches on `anyhow` message text; re-run `cargo test -p attest-policy --features dcap` after any `dcap-qvl` bump. |
| `agent-pack` LOC | 1496 of a 1500 cap; the next addition needs a crate split. |
| Mainnet (netuid 100) | Owner wallet not yet on this machine, so prod runs with `BASE_GATEWAY_REQUIRE_OWNER=0`. |
