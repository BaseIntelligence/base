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
| `base-master-validator` | Master API, coordination, aggregation, health/version |
| `master-postgres` | PostgreSQL 16 control plane (private) |
| `challenge-prism` | One long-lived combined Prism challenge service |

Cardinality is exactly one application container, one PostgreSQL container, and one long-lived container per active challenge. There is no gateway, broker sidecar, challenge PostgreSQL, evaluator, or worker sidecar service in this topology.

### Networking

- `db` network: internal bridge. Master + PostgreSQL only. No host publication of `5432`.
- `app` network: internal bridge. Master + challenge services.
- Only the master public API is published, bound to loopback in the test range `3100-3199` (default `3180`).

### Secrets

Secrets are host files (mode `0600`), bind-mounted read-only. Compose manifests never embed secret values. Operator-local state for greenfield install lives under `${XDG_STATE_HOME:-~/.local/state}/base-compose/<project>/`.

### Images

Every deployable first-party reference is immutable: `repository@sha256:<64 hex>`. Mutable `latest` tags are rejected by the contract tests.

### Evaluation boundary

Base and Prism do **not** launch evaluator containers. Prism runs in `PRISM_COMBINED_MODE=true` and verifies/ingests external results. External long-lived TEE runtimes are never lifecycle-managed by this Compose project.

## Validator project

Entrypoint: `deploy/compose/docker-compose.validator.yml`

Each validator is an independent Compose project with its own network, volume, identity, and wallet material. Validators never receive master PostgreSQL credentials and never orchestrate challenges.

## Unsupported (removed from target path)

- Docker Swarm installers, overlays, secrets, replicated jobs, placement constraints
- LLM gateway services, tokens, routes, and provider clients
- Application-launched `docker run` / `docker compose run` evaluator jobs

Historical `deploy/swarm/` material is not a supported operator path for new installs.
