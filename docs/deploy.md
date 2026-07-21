# Deploy From Scratch

Compose-only path for a fresh host: one master project (control plane + **embedded**
challenge ASGI inside the master image) and optional independent **weight-only**
validator projects. Swarm is **not** a supported install destination.

## Quick start

```bash
# 1. Master control plane (embeds Prism + agent-challenge on localhost)
./deploy/compose/install-master.sh --project-name base-mission-master --port 3180

# 2. Health / version / public challenge paths (unchanged prefixes)
curl -fsS http://127.0.0.1:3180/health
curl -fsS http://127.0.0.1:3180/version
curl -fsS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:3180/challenges/prism/openapi.json
curl -fsS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:3180/challenges/agent-challenge/openapi.json

# 3. Independent weight-only validator (own project/identity; never runs master)
# Network shipping example:
./deploy/compose/install-validator.sh \
  --project-name base-validator-live \
  --master-url https://chain.joinbase.ai
# Local disposable master only (secondary smoke):
#   --master-url http://127.0.0.1:3180 --project-name base-mission-validator-a
```

Immutable image pins may be supplied via environment (`BASE_MASTER_IMAGE_*`,
`POSTGRES_IMAGE_*`, `BASE_VALIDATOR_IMAGE_*`). When unset, the install helpers resolve
local mission image digests. **`PRISM_IMAGE_*` is not required** for master topology
(challenges ship inside `base-master`); historical GHCR challenge names remain for
emergency dual-run / rollback only.

`--master-url` is the Base master coordination API pointer only. Validators do not
host master, control-plane PostgreSQL, or challenge services. Shipping default is
weight-only (`challenge_execution_enabled: false`): `GET /v1/weights/latest` + own-wallet
`set_weights` when gated. Host `docker.sock` on the agent is optional migration prep,
not challenge control-plane.

## Topology

| Compose project | Services |
|-----------------|----------|
| Master (`install-master.sh`) | `base-master-validator` (proxy + embedded Prism `:18080` + AC `:18081`), `master-postgres` |
| Validator (`install-validator.sh`) | one `validator` runtime (agent-only, weight-only) with own identity/wallet |

There is **no** separate `challenge-prism` / `challenge-agent-challenge` Compose service
in the shipping master project. Public path prefixes stay `/challenges/prism` and
`/challenges/agent-challenge` on the proxy httpx path.

Networks (master project):

- `db` (internal): master + PostgreSQL only; no host `5432`.
- `app` (internal): available for attachments; challenges bind loopback inside master.
- `public` (non-internal): master host port only (default `127.0.0.1:3180`).

Secrets are host files mode `0600` bind-mounted read-only. Compose manifests never
embed secret values. Operator-local state lives under
`${XDG_STATE_HOME:-~/.local/state}/base-compose/<project>/`.

## Operator surfaces

| Goal | Command / surface |
|------|-------------------|
| Master install | `./deploy/compose/install-master.sh` |
| Validator install | `./deploy/compose/install-validator.sh` (default `--master-url https://chain.joinbase.ai`) |
| Health | `GET /health`, `GET /ready`, `GET /version` on the master port |
| Registry (public read) | `GET /v1/registry` |
| Challenge public OpenAPI | `GET /challenges/prism/openapi.json`, `GET /challenges/agent-challenge/openapi.json` |
| Weights | `GET /v1/weights/latest` (validators; master never `set_weights`) |
| Challenge activate | `POST /v1/admin/challenges/{slug}/activate` with `X-Admin-Token` |
| Challenge deactivate | `POST /v1/admin/challenges/{slug}/deactivate` |
| Raw-weight / vector status | Master admin and unpublished weight routes (see API docs) |
| Challenge roll | Rebuild/repin **master** image (embedded packages); watcher interval shipping default **0** |

For deeper service cardinality, networking, and no-evaluator rules see
[Compose-only deployment](compose.md) and [Master install](master/README.md).

## Runtime behavior

- Master entrypoint supervises proxy + localhost challenge uvicons; registry
  `internal_base_url` seeds are `http://127.0.0.1:18080` / `:18081`.
- Challenge watcher + registry reconcile shipping intervals are **0** (safe without
  `challenge-*` containers). Emergency dual-run may raise intervals and restore an
  external challenge service — not the default install path.
- Master is sole writer (submissions/leaderboard/aggregation). Validators never write
  those surfaces.
- Prism runs in combined mode (`PRISM_COMBINED_MODE=true`) inside the master container.
  Base and Prism never launch evaluator containers.

## Monorepo local image builds

First-party challenge + base images are built from this monorepo
(`BaseIntelligence/base`). Public **GHCR image names are unchanged**; only the
build context moved under `packages/challenges/*`.

| Image (name never renames) | Monorepo local build |
|----------------------------|----------------------|
| `ghcr.io/baseintelligence/base-master` | `docker build -f docker/Dockerfile.master -t ghcr.io/baseintelligence/base-master:local .` |
| `ghcr.io/baseintelligence/base-validator-runtime` | `docker build -f docker/Dockerfile.validator-runtime -t ghcr.io/baseintelligence/base-validator-runtime:local .` |
| `ghcr.io/baseintelligence/prism` | `docker buildx build -f packages/challenges/prism/Dockerfile --build-context monorepo=. --target service -t ghcr.io/baseintelligence/prism:local packages/challenges/prism` |
| `ghcr.io/baseintelligence/prism-evaluator` | same Dockerfile, `--target evaluator` |
| `ghcr.io/baseintelligence/agent-challenge` | `docker buildx build -f packages/challenges/agent-challenge/Dockerfile --build-context monorepo=. --target runtime -t ghcr.io/baseintelligence/agent-challenge:local packages/challenges/agent-challenge` |
| `ghcr.io/baseintelligence/agent-challenge-terminal-bench-runner` | same Dockerfile, `--target terminal-bench-runner` |

Compose installers accept operator pins via `BASE_MASTER_IMAGE_*`,
`PRISM_IMAGE_*`, `BASE_VALIDATOR_IMAGE_*` (repository + digest). For disposable
local smoke you can set `PRISM_LOCAL_IMAGE=ghcr.io/baseintelligence/prism:local`
(and the matching master local tag) so `install-master.sh` resolves a digest from
the monorepo-built image. Production still rolls digest-only pins of the same
public names.

Public challenge API slugs stay `/challenges/prism` and
`/challenges/agent-challenge` (proxy + registry); monorepo packaging does not
rename routes. Layout ADR: [monorepo.md](monorepo.md). Challenge image CI:
`.github/workflows/challenge-images.yml`.

## Unsupported / historical

- `deploy/swarm/` (including `install-swarm.sh`, overlays, Swarm secrets,
  replicated jobs, placement constraints, and the host supervisor) is a frozen
  historical artifact, **not** a supported operator path for new installs.
  Historical notes there still reference the same public GHCR names; greenfield
  challenge image builds now use monorepo paths above, not standalone clones.
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
