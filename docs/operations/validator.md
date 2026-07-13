# Validator Operations

Run these from the repository root. This runbook covers the supported **Docker
Compose** validator path and the on-chain weights relationship to the master.

Compose is the only supported operator path for new installs. Historical Swarm
units under `deploy/swarm/` are unsupported for greenfield bring-up.

## Install, update, stop

### Preferred: agent-only independent Compose validator

Validators **never run master**. `install-validator.sh` starts only the agent
container and requires an explicit Base master coordination URL.

```bash
# Local disposable master (mission smoke)
./deploy/compose/install-validator.sh \
  --project-name base-mission-validator-a \
  --master-url http://127.0.0.1:3180

# Public network Base master API
./deploy/compose/install-validator.sh \
  --project-name base-validator-live \
  --master-url https://chain.joinbase.ai

docker compose -p base-mission-validator-a \
  -f deploy/compose/docker-compose.validator.yml ps

# update: pull/recreate with a new digest pin, then
# re-run install-validator.sh or docker compose up -d with updated env

# stop
docker compose -p base-mission-validator-a \
  -f deploy/compose/docker-compose.validator.yml down
```

Each validator is an independent Compose project with its own identity, network,
volume, and secrets. See [Validator guide](../validator/README.md) and
[Compose deployment](../compose.md).

**Read-only rootfs notes:** container `HOME` must be the writable state volume
(`/var/lib/base/state`) so bittensor can create `$HOME/.bittensor`. Bind
protocol identity as a real directory readable by uid 1000 (avoid host
symlinks with restrictive parent modes). See `docs/compose.md` for the table.

### Weight model reminder

Challenges push raw weights to the master; the master aggregates; validators fetch
`GET /v1/weights/latest` and call `set_weights` with their own wallets. There is
**no LLM gateway** assignment token loop in the target path.

## Runtime commands

```bash
# validator project
docker compose -p base-mission-validator-a \
  -f deploy/compose/docker-compose.validator.yml ps
docker compose -p base-mission-validator-a \
  -f deploy/compose/docker-compose.validator.yml logs -f validator

# master project (if co-located)
docker compose -p base-mission-master -f deploy/compose/docker-compose.yml ps
curl -fsS http://127.0.0.1:3180/health
curl -fsS http://127.0.0.1:3180/version
curl -fsS http://127.0.0.1:3180/v1/registry
curl -fsS http://127.0.0.1:3180/v1/weights/latest
```

## Secret handling

Validators need protocol identity and, if submitting on-chain, submit wallet
material. Never place coldkey material on the node, and never store mnemonics or
hotkeys in `.env`, shell history, support threads, screenshots, or evidence logs.

Secrets are host files mode `0600` (parent dirs `0700`) bind-mounted read-only.
Compose manifests must never embed secret values. Master Postgres credentials are
never mounted into validator projects.

## Validator agent (decentralized executor)

The validator agent remains the long-running process that can register, heartbeat,
pull assignments, execute or verify work, and report results:

```bash
base validator agent --config /path/to/validator.yaml
```

On the Compose path the process is packaged as the `validator` service from
`docker-compose.validator.yml`. Relevant config keys:

```yaml
validator:
  # When master hosts registry + weights, keep these equal to master_url.
  # Public network Settings default:
  registry_url: https://chain.joinbase.ai
  weights_url: null
  agent:
    # Base master coordination API only (--master-url). Required; never invented.
    # Local smoke:
    master_url: http://127.0.0.1:3180
    # Public network sample: https://chain.joinbase.ai
    capabilities: ["cpu"]
    version: "0.1.0"
    heartbeat_interval_seconds: 60
    poll_interval_seconds: 5.0
```

`master_url` is the Base master coordination root. `registry_url` / `weights_url`
are public/registry aliases that may share that host; installers copy
`--master-url` into all three when the master hosts both. Never default operators
to a bare manager IP or invent localhost for production.

There is no master LLM gateway route and no per-assignment `BASE_LLM_GATEWAY_URL`
/ `BASE_GATEWAY_TOKEN` contract in the shipping target path.

### Validator agent image (digest pin)

Pin the validator runtime by immutable digest:

```text
repository@sha256:<64-hex>
```

Mutable `latest` alone is not a production selector. Master challenge auto-update
is the master-resident digest-aware watcher; validators do not mutate master
challenge services.

## Agent Challenge evaluation notes

Agent Challenge Terminal-Bench evaluation, when used, runs through challenge-local
and broker/session contracts owned by the agent-challenge repository. Base public
proxy only exposes challenge public routes and must block `/internal/*`,
`POST /internal/v1/submissions/{submission_id}/launch`, and generic benchmark
execution-shaped routes. **Base/Prism application code must not create short-lived
evaluator containers.**

Inspect long-lived challenge container logs on the master Compose project:

```bash
docker compose -p base-mission-master -f deploy/compose/docker-compose.yml \
  logs --since=30m challenge-prism
```

Use placeholder service names and never print token values.

## Validation

Prefer targeted suite entries from `services.yaml` (for example
`base_targeted_validator_compose`). Full tree format/lint/coverage is reserved for
release gates, not ordinary operator-doc edits.

```bash
uv run pytest tests/unit/test_validator_compose_artifact.py \
  tests/unit/test_validator_agent_cli_docs.py -q
```

If Docker Compose or a Python tool is unavailable, record the missing tool as a
blocker instead of marking that surface as tested.

## Historical Swarm material

`deploy/swarm/submitter/`, `install-swarm.sh`, `docker service` checks, and
Swarm image-updater loops are **historical / unsupported** for new installs.
They must not be required steps in operator runbooks. See
[deploy/swarm/README.md](../../deploy/swarm/README.md).
