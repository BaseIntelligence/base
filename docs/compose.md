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
| `base-master-validator` | Master API, coordination, aggregation; **embeds** Prism + agent-challenge ASGI on loopback (see below) |
| `master-postgres` | PostgreSQL 16 control plane (private) |

Cardinality is exactly one application container and one PostgreSQL container.
There is **no** `challenge-prism` / `challenge-*` Compose service, no gateway,
broker sidecar, challenge PostgreSQL, evaluator, or worker sidecar.

### Embedded challenges inside master

The `base-master` image (`docker/Dockerfile.master`) installs monorepo packages
`prism-challenge` and `agent-challenge` and runs them under
`docker/master-entrypoint.sh` (supervisor) on **localhost only**:

| Process | Bind | Notes |
| --- | --- | --- |
| `base master proxy` | `0.0.0.0:8081` (published host port) | Public API + `/challenges/*` reverse proxy (httpx unchanged) |
| Prism ASGI | `127.0.0.1:18080` | `uvicorn prism_challenge.app:app` |
| Agent-challenge ASGI | `127.0.0.1:18081` | `uvicorn agent_challenge.app:app` |

Data paths on the master volume:

- `/var/lib/base/challenges/prism` (SQLite + TMPDIR)
- `/var/lib/base/challenges/agent-challenge` (SQLite + artifacts)

Shared tokens stay file-backed (`PRISM_SHARED_TOKEN_FILE`,
`CHALLENGE_SHARED_TOKEN_FILE`). Registry seed and
`default_internal_base_url()` use:

- Prism: `http://127.0.0.1:18080`
- agent-challenge: `http://127.0.0.1:18081`

Public path prefixes `/challenges/prism` and `/challenges/agent-challenge` are
unchanged. **No `PRISM_IMAGE_*` pin is required** for master topology; optional
historical pins are ignored by `install-master.sh` for Compose interpolation.

Emergency dual-run (proxy-only master + separate challenge container) is
operator-only: set `BASE_MASTER_EMBED_CHALLENGES=0` **and** override registry
`internal_base_url` + restore a challenge service file; not the shipping path.

### Challenge auto-update (watcher)

With embedded challenges there is no separate challenge Compose service to pull
or recreate. Shipping defaults set:

- `BASE_MASTER_CHALLENGE_WATCHER_INTERVAL_SECONDS=0` (watcher lifespan off)
- `BASE_MASTER_REGISTRY_RECONCILE_INTERVAL_SECONDS=0` (no dynamic challenge start)

Master health therefore stays green without `challenge-*` services
(VAL-MEMB-005). When re-enabled for emergency dual-run, the watcher still only
acts inside the configured Compose project (resolve → pull → targeted recreate →
health/version → commit/rollback) and never mutates live Swarm fabric.

### Independent validator image auto-update

Each validator Compose install enables a **host-side** timer by default
(`base-validator-image-updater@<project>.timer`) that tracks
`ghcr.io/baseintelligence/base-validator-runtime:latest` by digest, rewrites
project `.env` pins atomically, and recreates only the agent service. Runtime
is always `repository@sha256:<digest>` (never bare `:latest`). Image auto-update
remains host-side even though shipping Compose also mounts host `docker.sock`
into the agent (prod prep for a later challenges-on-validator path). Opt out
with `install-validator.sh --no-auto-update`.

### Networking

- `db` network: internal bridge. Master + PostgreSQL only. No host publication of `5432`.
- `app` network: internal bridge (available for future attachments; challenges bind loopback inside master).
- Only the master public API is published, bound to loopback in the test range `3100-3199` (default `3180`).

### Secrets

Secrets are host files (mode `0600`, parent dirs `0700`), bind-mounted read-only.
Compose manifests never embed secret values. Production failed-closed when the
required `*_FILE` paths are missing, empty, world/group-readable, or replaced by
inline environment secrets. Operator-local state for greenfield install lives
under `${XDG_STATE_HOME:-~/.local/state}/base-compose/<project>/`.

### Images

Every deployable first-party reference is immutable: `repository@sha256:<64 hex>`. Mutable `latest` tags are rejected by the contract tests.

Public image **names** (never renamed by the monorepo residual):

| Role | GHCR name |
| --- | --- |
| Master | `ghcr.io/baseintelligence/base-master` |
| Validator runtime | `ghcr.io/baseintelligence/base-validator-runtime` |
| Prism challenge | `ghcr.io/baseintelligence/prism` |
| Prism evaluator | `ghcr.io/baseintelligence/prism-evaluator` |
| Agent Challenge | `ghcr.io/baseintelligence/agent-challenge` |
| AC terminal-bench runner | `ghcr.io/baseintelligence/agent-challenge-terminal-bench-runner` |

#### Monorepo local builds (Compose lab / operator pin)

Build from the Base monorepo root (BuildKit required for challenge images). Package
sources live under `packages/challenges/{prism,agent-challenge}`; workspace `base`
is supplied via named context `monorepo=.`:

```bash
# Prism long-lived service image (public name unchanged)
docker buildx build \
  -f packages/challenges/prism/Dockerfile \
  --build-context monorepo=. \
  --target service \
  -t ghcr.io/baseintelligence/prism:local \
  packages/challenges/prism

# Agent Challenge runtime (public name unchanged)
docker buildx build \
  -f packages/challenges/agent-challenge/Dockerfile \
  --build-context monorepo=. \
  --target runtime \
  -t ghcr.io/baseintelligence/agent-challenge:local \
  packages/challenges/agent-challenge

# Master + validator-runtime (repo-root Dockerfiles)
docker build -f docker/Dockerfile.master \
  -t ghcr.io/baseintelligence/base-master:local .
docker build -f docker/Dockerfile.validator-runtime \
  -t ghcr.io/baseintelligence/base-validator-runtime:local .
```

Wire a local pin into Compose install without renaming GHCR:

```bash
PRISM_LOCAL_IMAGE=ghcr.io/baseintelligence/prism:local \
BASE_MASTER_LOCAL_IMAGE=ghcr.io/baseintelligence/base-master:local \
  ./deploy/compose/install-master.sh --project-name base-mission-master --port 3180
```

Or export `PRISM_IMAGE_REPOSITORY` + `PRISM_IMAGE_DIGEST` (and master counterparts)
explicitly. See [deploy.md](deploy.md#monorepo-local-image-builds) and
[monorepo.md](monorepo.md).

### Evaluation boundary

Base and Prism do **not** launch evaluator containers. Prism runs in `PRISM_COMBINED_MODE=true` and verifies/ingests external results. External long-lived TEE runtimes are never lifecycle-managed by this Compose project.

## Public Base master API vs registry aliases

Validators **never run master**. An independent validator install is agent-only
(no master app, PostgreSQL control plane, or challenge services). Shipping
Compose mounts host `docker.sock` into the agent for later challenges-on-validator
migration prep; it does not add a master/postgres/challenges stack.
`--master-url` / `validator.agent.master_url` is a **client pointer** to the
Base master / coordination API the operator actually runs.

Do not conflate these roles:

| Concept | Role |
| --- | --- |
| `master_url` (`--master-url`) | Base master coordination API (register/heartbeat/pull/result). Required and explicit. |
| `registry_url` / `weights_url` | Public registry / weights aliases. When the master hosts both, installers set them equal to `master_url`. Product docs may also document network defaults separately. |

### Hostnames

| Hostname | Role | Operator guidance |
| --- | --- | --- |
| `https://chain.joinbase.ai` | Authoritative public Base master API for this network (`role=master`). Settings defaults, installer samples, and public weights examples use this host. Live master Compose project: `base-master-prod`. | Network validators: `--master-url https://chain.joinbase.ai`. Verify `GET /health` returns `role=master` / ready. |
| Other historical public hostnames | Non-authoritative secondary only (may return 502). | Do not ship as master URL or installer default. |
| `http://127.0.0.1:<port>` or private operator master | Disposable local smoke / private operator control plane | Allowed when the operator explicitly passes that URL; never invented as a silent default. |

Docker Compose installers must not invent alternate public master hostname or IP
defaults. Historical Swarm advertise addresses under `deploy/swarm/` are unsupported
for greenfield Compose installs.

## Validator project

Entrypoint: `deploy/compose/docker-compose.validator.yml`

One-command install for an independent validator (no master source tree required once artifacts and image pins exist):

```bash
# Public network shipping example
./deploy/compose/install-validator.sh \
  --project-name base-validator-live \
  --master-url https://chain.joinbase.ai

# Local disposable master only (secondary smoke/dev)
# ./deploy/compose/install-validator.sh \
#   --project-name base-mission-validator-a \
#   --master-url http://127.0.0.1:3180
```

Or, with only the validator deployment artifacts on a clean host directory:

```bash
docker compose -p base-validator-live \
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
| `BASE_DOCKER_GID` / `BASE_DOCKER_SOCKET` | Host docker group + socket path (`group_add` + bind) |

Validators never receive master PostgreSQL credentials, challenge volumes, aggregation controls, or master challenge lifecycle operators. Host `docker.sock` is mounted into the agent (uid `1000` + `group_add` docker GID) for production prep; the project remains agent-only. Teardown of one validator project does not affect another validator or the master:

```bash
docker compose -p base-mission-validator-a -f docker-compose.validator.yml down
# preserve identity/state for reinstall, or add -v to drop disposable state
```

### Validator HOME and identity under `read_only`

The validator service runs `read_only: true` with `user: "1000:1000"`. Writable surfaces are:

| Path | Purpose |
| --- | --- |
| `/var/lib/base/state` (named volume) | Runtime state + **`HOME`** (defaults to this path so bittensor can write `$HOME/.bittensor`) |
| `/tmp` (tmpfs) | Temporary files |

Do **not** set `HOME` to `/var/lib/base`: that path is not itself a volume, so bittensor membership/wallet caches fail with read-only filesystem errors. `install-validator.sh` and `docker-compose.validator.yml` default `HOME=/var/lib/base/state`.

Protocol identity (`BASE_VALIDATOR_PROTOCOL_IDENTITY` → `/var/lib/base/identity`) is mounted **read-only**. Bind a **real directory** tree with parents traversable by uid 1000 (typically parent mode `0755`). A host symlink whose parent is mode `0700` prevents wallet load inside the container even when the leaf wallet files look correct.

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
Swarm historical notes still document the same public GHCR names; new challenge
image builds use monorepo paths under `packages/challenges/*` (see Images above),
not standalone prism/agent-challenge clones.
