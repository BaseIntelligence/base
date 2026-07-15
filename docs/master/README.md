# Master Installation Guide

Operator guide for the supported **Docker Compose** master control plane. Compose is
the only shipping runtime for new installs. Do not use Swarm scripts for greenfield
bring-up: historical material lives under `deploy/swarm/` and is **unsupported**.

This guide covers installing the master application, PostgreSQL control plane,
long-lived challenge services, and the in-process digest-aware challenge watcher. It
does not configure on-chain submission (validators own wallets and `set_weights`).

## Topology

| Piece | Details |
| --- | --- |
| Installer | `deploy/compose/install-master.sh` |
| Compose file | `deploy/compose/docker-compose.yml` |
| App | `base-master-validator` (proxy, coordination, aggregation, watcher) |
| Database | `master-postgres` (private `db` network only) |
| Challenges | one long-lived `challenge-<slug>` per ACTIVE registry entry |
| Secrets | host files under `${XDG_STATE_HOME:-~/.local/state}/base-compose/<project>/secrets` |
| Live production | Compose project **`base-master-prod`**; public API **`https://chain.joinbase.ai` only** (Swarm inactive after cutover). `chain.platform.network` secondary/non-authoritative. |

There is **no LLM gateway** container or Swarm broker overlay in this path. The master
coordinates and aggregates; it **never** submits on-chain weights and **never** launches
evaluator containers.

## Install

```bash
./deploy/compose/install-master.sh --project-name base-mission-master --port 3180
```

Optional immutable image pins (repository + sha256 digest):

- `BASE_MASTER_IMAGE_REPOSITORY` / `BASE_MASTER_IMAGE_DIGEST`
- `PRISM_IMAGE_REPOSITORY` / `PRISM_IMAGE_DIGEST`
- `POSTGRES_IMAGE_REPOSITORY` / `POSTGRES_IMAGE_DIGEST`

When pins are unset, the installer may resolve local mission image digests for disposable
bring-up. Production should pin published digests.

What the installer does:

1. Creates state/config dirs (`0700`) and secret files (`0600`).
2. Writes a local master config suitable for the Compose networks.
3. Runs `docker compose up -d --wait` for the master project cardinality.

Detailed networking, secrets, and evaluation boundary rules live in
[docs/compose.md](../compose.md) and [docs/deploy.md](../deploy.md).

## Auto-update (challenge watcher)

Challenge image rollouts are owned by the **master-resident watcher**, not by Swarm
`docker service update` of mutable tags:

| Property | Behavior |
| --- | --- |
| Pin model | Immutable `repository@sha256:<digest>` only |
| Scope | Project-bound Compose services only |
| Steps | resolve → controlled pull → targeted recreate → health/version verify → commit |
| Failure | Rollback to previous digest; bounded exponential backoff |
| Durability | Intent and phase survive master restart (filesystem state) |
| Forbidden | Evaluator container creation; Swarm/stack APIs |

Operators freeze or recover by keeping freeze/desired digests stable and consulting
watcher health after pull failures. Configurable knobs (when present in master settings)
include interval and state path defaults under the master application data volume.

## Runtime checks

```bash
docker compose -p base-mission-master -f deploy/compose/docker-compose.yml ps
curl -fsS http://127.0.0.1:3180/health
curl -fsS http://127.0.0.1:3180/version
curl -fsS http://127.0.0.1:3180/v1/registry
docker compose -p base-mission-master -f deploy/compose/docker-compose.yml logs -f base-master-validator
```

Backup / restore / teardown scripts (Compose-only):

```bash
./deploy/compose/backup-master.sh --project-name base-mission-master --output-dir ./backup-master
./deploy/compose/restore-master.sh --project-name base-mission-master --backup-dir ./backup-master
./deploy/compose/teardown-master.sh --project-name base-mission-master
```

## Explicit non-goals

- No on-chain `set_weights` from the master.
- No Swarm init, secrets, overlays, or `docker service` create paths for new installs.
- No reintroduction of LLM gateway secrets or routes.
- No application-owned short-lived evaluator jobs.

## Historical Swarm note

`deploy/swarm/install-swarm.sh` and related supervisor/unit files are **historical /
unsupported** for new installs. They must not be documented as the required master path
and must not be run against live production Swarm fabric from this guide. See
[deploy/swarm/README.md](../../deploy/swarm/README.md) for the explicit non-target banner.

## Agent Challenge attested proxy flag

The master proxy setting `master.agent_challenge_attested_routes_enabled` (default
**false**) selects the public agent-challenge topology:

- **Off:** legacy signed submission / env / launch passthrough (byte-identical).
- **On:** fail-closed allowlist for review/eval miner flows; private and result
  routes never fall through the public edge.

Full attested evaluation ownership and score policy stay in the agent-challenge
service. See [Architecture](../architecture.md#agent-challenge-phala-intel-tdx-path)
and [Challenges](../challenges.md). Foundation install diffs for challenge-side
CVM credentials are out of scope here and must not appear in this guide.
