# Compose-only deployment

The supported Base deployment path is **Docker Compose** on a single host.

## Master project

Entrypoint: `deploy/compose/docker-compose.yml`

One-command install (creates local secret files and starts services):

```bash
./deploy/compose/install-master.sh --project-name base-mission-master --port 3180
```

Exact cardinality after install:

| Service | Role |
| --- | --- |
| `base-master-validator` | Master API, coordination, aggregation, health/version, digest-aware challenge watcher |
| `master-postgres` | PostgreSQL 16 control plane (private) |
| `challenge-prism` | One long-lived combined Prism challenge service |

Cardinality is exactly one application container, one PostgreSQL container, and one long-lived container per active challenge. There is no gateway, broker sidecar, challenge PostgreSQL, evaluator, or worker sidecar service in this topology.

### Challenge auto-update (watcher)

The watcher runs **inside** `base-master-validator` and is the only supported challenge
auto-update path:

1. Resolve approved immutable image pin (`repository@sha256:<digest>`).
2. Persist desired/current digest and rollout phase (durable intent).
3. Controlled pull of the desired image.
4. Targeted recreate of only the affected Compose service (project boundary).
5. Health + version verification before commit.
6. On failure: restore previous digest, bounded backoff, resume after restart.

It never mutates live Swarm fabric, never runs evaluator containers, and only acts
inside the configured Compose project.

### Networking

- `db` network: internal bridge. Master + PostgreSQL only. No host publication of `5432`.
- `app` network: internal bridge. Master + challenge services.
- Only the master public API is published, bound to loopback in the test range `3100-3199` (default `3180`).

### Secrets

Secrets are host files (mode `0600`, parent dirs `0700`), bind-mounted read-only.
Compose manifests never embed secret values. Production failed-closed when the
required `*_FILE` paths are missing, empty, world/group-readable, or replaced by
inline environment secrets. Operator-local state for greenfield install lives
under `${XDG_STATE_HOME:-~/.local/state}/base-compose/<project>/`.

### Images

Every deployable first-party reference is immutable: `repository@sha256:<64 hex>`. Mutable `latest` tags are rejected by the contract tests.

### Evaluation boundary

Base and Prism do **not** launch evaluator containers. Prism runs in `PRISM_COMBINED_MODE=true` and verifies/ingests external results. External long-lived TEE runtimes are never lifecycle-managed by this Compose project.

## Validator project

Entrypoint: `deploy/compose/docker-compose.validator.yml`

One-command install for an independent validator (no master source tree required once artifacts and image pins exist):

```bash
./deploy/compose/install-validator.sh \
  --project-name base-mission-validator-a \
  --master-url http://127.0.0.1:3180
```

Or, with only the validator deployment artifacts on a clean host directory:

```bash
docker compose -p base-mission-validator-a \
  -f docker-compose.validator.yml --env-file .env up -d
```

Each validator is an independent Compose project with its own network, volume, protocol identity, optional submission wallet, and credentials. Required inputs:

| Input | Role |
| --- | --- |
| `COMPOSE_PROJECT_NAME` | Unique project name (distinct network + state volume) |
| `BASE_VALIDATOR_IMAGE_*` | Immutable image repository + sha256 digest |
| `BASE_VALIDATOR_CONFIG` | Host path to `validator.yaml` (`validator.agent.master_url` required) |
| `BASE_VALIDATOR_PROTOCOL_IDENTITY` | Host directory for the protocol signing wallet |
| `BASE_VALIDATOR_BROKER_TOKEN` | Host secret file mounted read-only |

Validators never receive master PostgreSQL credentials, challenge volumes, Docker socket access, aggregation controls, or challenge lifecycle operators. Teardown of one validator project does not affect another validator or the master:

```bash
docker compose -p base-mission-validator-a -f docker-compose.validator.yml down
# preserve identity/state for reinstall, or add -v to drop disposable state
```

## Backup, restore, and teardown

Operator scripts live next to the manifests (mode-aware, Compose-only, no Swarm):

```bash
# Control-plane PostgreSQL + provenance metadata (no secret values in artifacts)
./deploy/compose/backup-master.sh --project-name base-mission-master --output-dir ./backup-master
./deploy/compose/restore-master.sh --project-name base-mission-master --backup-dir ./backup-master

# Challenge-owned SQLite volume (Prism long-lived data)
./deploy/compose/backup-challenge.sh --project-name base-mission-master --service challenge-prism

# Non-destructive teardown retains postgres/challenge volumes
./deploy/compose/teardown-master.sh --project-name base-mission-master
# Explicit data destruction of owned volumes only
./deploy/compose/teardown-master.sh --project-name base-mission-master --destroy-data
```

Production rejects inline or missing file-backed secrets (`admin_token_file` mode
`0600` / parent `0700`) and never embeds secret values in Compose YAML or `/metrics`.

Operational metrics for the weight plane are exposed on `GET /metrics`
(low-cardinality counters: accepted/rejected/replay pushes, aggregation outcomes,
fetch failures, submit outcomes) without tokens or DSN material.

## Unsupported (removed from target path)

- Docker Swarm installers, overlays, secrets, replicated jobs, placement constraints
- LLM gateway services, tokens, routes, and provider clients
- Application-launched `docker run` / `docker compose run` evaluator jobs

Historical `deploy/swarm/` material is not a supported operator path for new installs.
