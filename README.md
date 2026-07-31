<div align="center">

# BASE

**Multi-challenge Bittensor subnet control plane (Rust).**

[![CI](https://github.com/BaseIntelligence/base/actions/workflows/ci.yml/badge.svg)](https://github.com/BaseIntelligence/base/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/BaseIntelligence/base)](https://github.com/BaseIntelligence/base/blob/dev/LICENSE)
[![Bittensor](https://img.shields.io/badge/Bittensor-subnet-black.svg)](https://bittensor.com/)

</div>

## What it is

BASE is the Bittensor subnet control plane for BaseIntelligence. This branch (`dev`)
is the **Rust greenfield** workspace: gateway, validator, agent-challenge, miner
CVM templates, and shared crates. It coordinates agent challenges (native pack
executor — no Harbor product CLI), seals a final weight vector, and serves it to
validators. The gateway is the sole TLS / public edge process.

- **Agent packs**: Harbor-format task workspaces from the pinned catalog
  ([BaseIntelligence/deepagent](https://huggingface.co/datasets/BaseIntelligence/deepagent) /
  git pin), graded by the in-tree native executor + Docker socket-proxy.
- **Miner CVM**: Phala / dstack measured compose — `socket-proxy` + `agent` +
  `attest-helper` (digest-pinned images on GHCR).
- **Emission**: validators call on-chain `set_weights` from sealed
  `GET /v1/weights/latest` — never the gateway.

## Branch

| Branch | Role |
|--------|------|
| **`dev`** | Active Rust control plane (this tree). PRs target `dev`. |
| `main` | Legacy / prior stack — do not mix secret material across histories. |

## Miners

Day-1: [docs/external-miner/](docs/external-miner/)

1. Deploy a measured CVM (`miner deploy`) with digest-pinned `base-agent` +
   `base-attest-helper` + socket-proxy.
2. Fund your own Phala account; hotkey + launch token + receipt sk are **files**
   under `/run/base/` (never env secret values).
3. Certify each epoch (`miner certify`) via loopback attest-helper
   `GET /v1/quote` → validator attest API.

## Validators

Weight-only path after seal:

```bash
curl -fsS "$GATEWAY/v1/weights/latest"
```

Then `set_weights` with your wallet. Operator compose:

```bash
./deploy/scripts/materialize-env.sh
docker compose up -d                  # postgres, validator, updater, socket-proxy
docker compose --profile master up -d # + gateway (subnet owner host only)
```

Details: [deploy/README.md](deploy/README.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Images (GHCR)

CI workflow [`.github/workflows/images.yml`](.github/workflows/images.yml) builds
and pushes digest-pinned service images to `ghcr.io/baseintelligence/base/*`.
Never `:latest` in measured compose.

| Target | Image suffix |
|--------|----------------|
| validator | `validator` |
| gateway | `gateway` |
| updater | `updater` |
| agent-challenge | `agent-challenge` |
| base-agent | `base-agent` (miner runner) |
| base-attest-helper | `base-attest-helper` (quote helper) |

## Toolchain

- Rust **1.96.0** (`rust-toolchain.toml`)
- Workspace: `crates/*`, `bins/*`, `xtask`
- Gates: `fmt`, `clippy -D warnings`, `test`, `cargo deny`, `xtask loc-cap`,
  `xtask consensus-lint`, `xtask spec-check`, `xtask agent-challenge-check`

## Docs

| Doc | Content |
|-----|---------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System map |
| [docs/AGENT_CHALLENGE.md](docs/AGENT_CHALLENGE.md) | Pack grade + miner CVM contract |
| [docs/BUNDLE_SPEC.md](docs/BUNDLE_SPEC.md) | Sealed weight bundle |
| [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) | Security claims |
| [docs/runbooks/](docs/runbooks/) | Ops cutovers (incl. measurement re-pin) |

## License

Apache License 2.0 — see [LICENSE](./LICENSE).
