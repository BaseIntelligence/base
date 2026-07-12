# Deploy From Scratch

Compose-only path for a fresh host: one master project (control plane + long-lived
challenge services) and optional independent validator projects. Swarm is **not** a
supported install destination.

## Quick start

```bash
# 1. Master control plane + packaged challenge-prism
./deploy/compose/install-master.sh --project-name base-mission-master --port 3180

# 2. Health / version
curl -fsS http://127.0.0.1:3180/health
curl -fsS http://127.0.0.1:3180/version

# 3. Independent validator (own project, identity, and wallet)
./deploy/compose/install-validator.sh \
  --project-name base-mission-validator-a \
  --master-url http://127.0.0.1:3180
```

Immutable image pins may be supplied via environment (`BASE_MASTER_IMAGE_*`,
`PRISM_IMAGE_*`, `POSTGRES_IMAGE_*`, `BASE_VALIDATOR_IMAGE_*`). When unset, the
install helpers resolve local mission image digests.

## Topology

| Compose project | Services |
|-----------------|----------|
| Master (`install-master.sh`) | `base-master-validator`, `master-postgres`, one long-lived `challenge-<slug>` per active challenge |
| Validator (`install-validator.sh`) | one `validator` runtime with own identity/wallet |

Networks (master project):

- `db` (internal): master + PostgreSQL only; no host `5432`.
- `app` (internal): master + challenge services.
- `public` (non-internal): master host port only (default `127.0.0.1:3180`).

Secrets are host files mode `0600` bind-mounted read-only. Compose manifests never
embed secret values. Operator-local state lives under
`${XDG_STATE_HOME:-~/.local/state}/base-compose/<project>/`.

## Operator surfaces

| Goal | Command / surface |
|------|-------------------|
| Master install | `./deploy/compose/install-master.sh` |
| Validator install | `./deploy/compose/install-validator.sh` |
| Health | `GET /health`, `GET /ready`, `GET /version` on the master port |
| Registry (public read) | `GET /v1/registry` |
| Challenge activate | `POST /v1/admin/challenges/{slug}/activate` with `X-Admin-Token` |
| Challenge deactivate | `POST /v1/admin/challenges/{slug}/deactivate` |
| Raw-weight / vector status | Master admin and unpublished weight routes (see API docs) |
| Update / watcher | Master-resident challenge watcher (digest pin pull + recreate) |

For deeper service cardinality, networking, and no-evaluator rules see
[Compose-only deployment](compose.md).

## Runtime behavior

- The master reconcile loop adopts healthy challenge containers after restart
  and installs services for newly ACTIVE registry challenges (project-scoped
  Compose only; never `docker service` / Swarm).
- Inactive, draft, and disabled challenges never start.
- Deactivation stops and removes the managed long-lived container while keeping
  the named state volume for reactivation.
- Prism runs in combined mode (`PRISM_COMBINED_MODE=true`). Base and Prism never
  launch evaluator containers.

## Unsupported / historical

- `deploy/swarm/` (including `install-swarm.sh`, overlays, Swarm secrets,
  replicated jobs, placement constraints, and the host supervisor) is a frozen
  historical artifact, **not** a supported operator path for new installs.
- LLM gateway services, tokens, routes, and provider clients have been removed
  from the target path.

If a future multi-host footprint is required, it is out of scope for this
release and must not reintroduce Swarm as a silent fallback from Compose
installers or README navigation.

## Verify install

```bash
docker compose -p base-mission-master -f deploy/compose/docker-compose.yml ps
curl -fsS http://127.0.0.1:3180/health
curl -fsS http://127.0.0.1:3180/v1/registry
```

Teardown (mission-owned project only):

```bash
docker compose -p base-mission-master -f deploy/compose/docker-compose.yml down -v
```
