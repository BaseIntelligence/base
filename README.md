<div align="center">

# BASE

**Multi-challenge Bittensor subnet platform with master/validator orchestration.**

<a href="docs/miner/README.md">Miners</a> ·
<a href="docs/validator/README.md">Validators</a> ·
<a href="docs/master/README.md">Master</a> ·
<a href="docs/architecture.md">Architecture</a> ·
<a href="docs/challenges.md">Challenges</a> ·
<a href="docs/security.md">Security</a> ·
<a href="https://joinbase.ai">Website</a>

[![CI](https://github.com/BaseIntelligence/base/actions/workflows/ci.yml/badge.svg)](https://github.com/BaseIntelligence/base/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/BaseIntelligence/base)](https://github.com/BaseIntelligence/base/blob/main/LICENSE)
[![Bittensor](https://img.shields.io/badge/Bittensor-subnet-black.svg)](https://bittensor.com/)

![BASE Banner](assets/banner.jpg)

![Live challenge board](assets/challenges.svg)

</div>

---

## Overview

BASE is a **multi-challenge Bittensor subnet platform**: independent challenge subnets run under one
validator network. BASE routes miner traffic to the right challenge, collects each challenge's raw
weights, normalizes emissions, maps hotkeys to Bittensor UIDs, and publishes the final vector for
validators to submit on-chain. Each challenge lives in its own repository and owns its submissions,
scoring, state, and public miner experience; BASE is the orchestration layer that runs them as one
subnet.

It runs as a single **Docker Swarm**. A **master** (manager node) hosts the public proxy, the
validator coordination plane, the LLM gateway, the broker, and the challenge API services — it
coordinates and aggregates but **never executes** evaluation. Online **validators** are the
decentralized executors: each registers with the master, pulls assignments, and runs evaluation on
its own broker. There is no Kubernetes; the only backend is Swarm.

## Architecture

```mermaid
flowchart LR
    U[Miners] --> P[Public proxy]
    P --> CH[Challenge APIs]
    CH --> AG[Weight aggregator]

    subgraph MASTER [Master - manager node]
        P
        CP[Coordination plane]
        GW[LLM gateway]
        AG
        PG[(Control-plane Postgres)]
    end

    V[Validators - executors] -- register / pull / result --> CP
    V -- provider calls --> GW
    V --> VB[Own broker + Docker eval]
    AG --> W[GET /v1/weights/latest]
    V -- fetch weights --> W
    V -- set_weights own hotkey --> BT[Bittensor]
```

## How It Works

1. The master tracks active challenges and their emission shares, and **auto-deploys** their services from the registry (a newly-registered ACTIVE challenge propagates with no manual step).
2. Challenge services run isolated from the control plane and each other, each on its own `/data` volume.
3. Miners reach a challenge through BASE's public proxy.
4. Validators register, pull assignments from the coordination plane, execute evaluation on their own broker, and post results.
5. Each challenge computes raw hotkey weights from those validator-reported results.
6. BASE normalizes challenge outputs, applies emission shares, and maps hotkeys to UIDs.
7. Each validator fetches the final vector from the weights API and submits it on-chain under its own hotkey; the master aggregates but **never submits on-chain**.

If a challenge fails, BASE isolates that challenge's contribution without taking down the subnet.

## Roles

| Role | Responsibility |
|------|----------------|
| **Master** | Coordinates + aggregates; runs the proxy, coordination plane, LLM gateway, broker, and challenge services. Never executes evals or submits on-chain. |
| **Validators** | Decentralized executors: register + heartbeat, pull assignments, run on their own broker, route LLM calls through the gateway, and submit their own weights. |
| **Challenge owners** | Own an independent repo, image, scoring logic, and state; expose the standard internal weight contract. |
| **Workers** | Miner-funded GPU executors for PRISM (Lium/Targon or local), carrying an `ExecutionProof`. |

## Miner-Funded GPU Worker Plane

Optional, gated behind `compute.worker_plane_enabled` (env `BASE_COMPUTE__WORKER_PLANE_ENABLED`,
default **off** ⇒ byte-for-byte legacy behavior). It moves PRISM's heavy GPU evaluation onto
**worker agents in GPU instances the miners fund** (rented on **Lium** or **Targon**, or local),
deployed with the `base worker` CLI. Validators keep only light plausibility checks, probabilistic
replay audits, and weight submission.

- **Signed enrollment** — the miner signs a hotkey↔worker binding; provider keys (`LIUM_API_KEY` / `TARGON_API_KEY`) stay in the miner's environment and never reach the master.
- **Anti-collusion** — a worker never evaluates its owner's submission; each unit replicates across **R=2 distinct-owner** workers and is reconciled by `ExecutionProof.manifest_sha256`.
- **Proof tiers** — tier 0 (manifest hash + sr25519 signature), tier 1 (pinned image digest), tier 2 (in-guest attestation, gated off on Targon). Audit sampling is tier-modulated.
- **Admission rule** — when enforced, a miner needs ≥1 active bound worker to submit to PRISM, else `403 NO_ACTIVE_WORKER`.

See the <a href="docs/miner/worker-plane.md">miner worker deployment guide</a>.

## Documentation

| Audience | Guide | Contents |
|----------|-------|----------|
| Miners | <a href="docs/miner/README.md">Miner guide</a> | Choose a challenge, submit through the proxy, track leaderboards |
| Miners | <a href="docs/miner/worker-plane.md">Worker deployment</a> | Deploy a miner-funded GPU worker on Lium/Targon |
| Validators | <a href="docs/validator/README.md">Validator guide</a> | Install the submit-only on-chain weight submitter |
| Validators | <a href="docs/operations/validator.md">Validator operations</a> | Submitter plus manager-service runbook |
| Operators | <a href="docs/deploy.md">Deploy from scratch</a> | End-to-end Swarm bring-up quickstart |
| Operators | <a href="docs/master/README.md">Foundation master guide</a> | Cortex Foundation master bring-up |
| Developers | <a href="docs/architecture.md">Architecture</a> | Control-plane vs worker topology, the broker contract |
| Developers | <a href="docs/challenges.md">Challenges</a> | The challenge model |
| Developers | <a href="docs/challenge-integration.md">Challenge integration</a> | The API contract a challenge must expose |
| Developers | <a href="docs/security.md">Security model</a> | Trust boundaries and secret handling |
| Developers | <a href="docs/versioning.md">Versioning</a> | SemVer, Git tag, and GHCR tag policy |
| Developers | <a href="docs/reward-semantics.md">Reward semantics</a> | Terminal-Bench scorer reward mapping |

## Deploy

`deploy/swarm/install-swarm.sh` is the canonical, Swarm-only entry point: **dry-run by default**,
mutates only with `--apply`, every destructive step behind its own flag. Production is pre-mainnet
hardened (image-pin, TLS, external-Postgres, and broker-allowlist policy guards fire at config load).

```bash
./deploy/swarm/install-swarm.sh          # dry-run: prints the planned docker swarm commands
./deploy/swarm/install-swarm.sh --apply  # apply on a disposable / owned host
```

The manager (`node.role==manager`) runs the control plane and the challenge services; CPU/GPU
workers are enrolled with the `base master worker` CLI and scheduled by label
(`node.labels.base.workload==cpu` / `node.labels.base.workload==gpu`).

Full walkthrough (images, volumes, worker enrollment, on-chain submission, public edge) in
<a href="docs/deploy.md">Deploy from scratch</a> and <a href="deploy/swarm/README.md">deploy/swarm/README.md</a>.

## Validation Quick Reference

Run from the repository root; live Swarm checks require Docker.

```bash
uv sync --extra dev --extra master
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest -m "not postgres" --cov=base --cov-report=term-missing --cov-fail-under=80
```

Evidence for local validation should live in a local, gitignored directory and must never contain
tokens, credentialed database URLs, registry credentials, or private keys.

## Repository Layout

```text
platform/
  src/base/    # CLI, APIs, orchestration, Bittensor wrappers
  alembic/     # PostgreSQL migrations
  config/      # YAML example configs
  docker/      # Dockerfiles and OCI image assets
  deploy/      # Swarm installer, supervisor unit, submitter, daemon.json templates
  docs/        # Project, miner, validator, and challenge docs
  tests/       # Unit / runtime validation tests
```

## License

Apache-2.0
