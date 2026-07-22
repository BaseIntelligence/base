# Compose install (master embed)

Supported shipping path: **Docker Compose** with **master + PostgreSQL only**.
Prism and Agent Challenge run as **localhost ASGI** inside the master container
(supervisor + reverse proxy). There are **no** required `challenge-*` Compose app
containers. Swarm is **not** a supported install destination (historical
`deploy/swarm/` only).

## Topology

```text
Master container (base-master-validator)
  ├─ master proxy / API          :8081 (public)
  ├─ continuous weights sealer
  ├─ Prism ASGI                  127.0.0.1:18080
  └─ Agent Challenge ASGI        127.0.0.1:18081
PostgreSQL (master-postgres)
```

Public path prefixes (unchanged): `/challenges/prism`, `/challenges/agent-challenge`.

Persistent challenge data: `/var/lib/base/challenges/{prism,agent-challenge}`.

`PRISM_IMAGE_*` is **not required** (images are embedded in the master). Shared tokens
still mount via files (for example `PRISM_SHARED_TOKEN_FILE`).
`challenge_watcher_interval_seconds` / watcher defaults stay off so missing
challenge containers never break master health.

## Exact cardinality

Services in `deploy/compose/docker-compose.yml`:

1. `master-postgres`
2. `base-master-validator`

No `challenge-prism` / `challenge-agent-challenge` services.

## Install master

```bash
./deploy/compose/install-master.sh --project-name base-master --port 8081
```

Smoke:

```bash
curl -fsS http://127.0.0.1:8081/health
curl -fsS http://127.0.0.1:8081/v1/registry
curl -fsS http://127.0.0.1:8081/v1/weights/latest
```

Public network master: **https://chain.joinbase.ai** (`role=master`).

## Install weight-only validator

```bash
./deploy/compose/install-validator.sh \
  --project-name base-validator \
  --master-url https://chain.joinbase.ai
```

`master_url` (installer `--master-url`) is the validator coordination root against the
Base master. Public `registry_url` / weights default to `https://chain.joinbase.ai`.
Independent validator projects never run master/postgres/challenge writers. They mount
host `docker.sock`, keep `HOME` writable under `/var/lib/base/state` (including
`.bittensor` wallet cache) even when the rootfs is `read_only`, run as uid **1000**,
and use `docker-compose.validator.yml`. Details: [validator.md](validator.md).

## Monorepo local image builds (optional)

Challenge and master Dockerfiles live in this repo:

- `packages/challenges/prism` · GHCR `ghcr.io/baseintelligence/prism` (+ `prism-evaluator`)
- `packages/challenges/agent-challenge` · GHCR `ghcr.io/baseintelligence/agent-challenge`
  (+ `agent-challenge-terminal-bench-runner`)
- Master / validator-runtime: `ghcr.io/baseintelligence/base-master`,
  `ghcr.io/baseintelligence/base-validator-runtime`

BuildKit named context:

```bash
docker buildx build \
  --build-context monorepo=. \
  -f packages/challenges/prism/Dockerfile \
  packages/challenges/prism
```

Workflow: `.github/workflows/challenge-images.yml` (`challenge-images` / mono-ci-images).
GHCR **names** never rename. Layout notes: root `AGENTS.md` and workspace
`pyproject.toml` (`packages/challenges/*` members).

## Safety

- Master **never** `set_weights`
- No multi-writer SQLite across challenge services (sole writer = master-embed ASGI)
- Secrets stay file-backed (`*_FILE`, mode `0600`), never embedded in compose YAML
- LLM gateway services are removed from the target path

## API truth

OpenAPI in code:

- `/openapi.json`
- `/challenges/prism/openapi.json`
- `/challenges/agent-challenge/openapi.json`
