# Validator Operations

Run these from the repository root. This runbook covers the supported **Docker
Compose** validator path and the on-chain weights relationship to the master.

Compose is the only supported operator path for new installs. Historical Swarm
units under `deploy/swarm/` are unsupported for greenfield bring-up.

## Install, update, stop

### Preferred: weight-only independent Compose validator

Validators **never run master**. `install-validator.sh` starts only the agent
container (weight-only default) and requires an explicit Base master coordination
URL. Challenge execution adapters default **off**; the project never includes
master, postgres, or `challenge-*` services.

```bash
# Public network Base master API (shipping primary) — weight-only
./deploy/compose/install-validator.sh \
  --project-name base-validator-live \
  --master-url https://chain.joinbase.ai

# Local disposable master (mission smoke only)
./deploy/compose/install-validator.sh \
  --project-name base-mission-validator-a \
  --master-url http://127.0.0.1:3180

docker compose -p base-validator-live \
  -f deploy/compose/docker-compose.validator.yml ps

# Image auto-update is ON by default (host timer tracks :latest digests).
# Opt out: add --no-auto-update. Freeze later:
#   BASE_VALIDATOR_IMAGE_UPDATE_HOLD=1 in the project .env
# Manual pin still works:
#   edit BASE_VALIDATOR_IMAGE_DIGEST then compose up -d --force-recreate

# stop
docker compose -p base-validator-live \
  -f deploy/compose/docker-compose.validator.yml down
# Also stop auto-update when tearing down permanently:
#   systemctl disable --now base-validator-image-updater@base-validator-live.timer
```

Each validator is an independent Compose project with its own identity, network,
volume, and secrets. Images auto-update by default via a **host-side** digest
reconciler. Shipping Compose may mount host `docker.sock` into the agent
(uid `1000` + docker group) as **optional** migration prep only — not for
challenge control-plane. The project remains agent-only (no master/postgres/
challenge stack). See [Validator guide](../validator/README.md) and
[Compose deployment](../compose.md).

**Read-only rootfs notes:** container `HOME` must be the writable state volume
(`/var/lib/base/state`) so bittensor can create `$HOME/.bittensor`. Bind
protocol identity as a real directory readable by uid 1000 (avoid host
symlinks with restrictive parent modes). See `docs/compose.md` for the table.

### Weight-only model (shipping)

1. Master (sole writer) embeds challenges + aggregates raw weights.
2. Validator fetches `GET https://chain.joinbase.ai/v1/weights/latest`.
3. Validator calls `set_weights` with its own wallet when
   `submit_on_chain_enabled` is true (default false).
4. `validator.agent.challenge_execution_enabled` defaults **false** — no Prism/AC
   adapters, no submissions/leaderboard writer on the validator.
5. Optional future audit re-exec is non-write only (not the default path).

There is **no LLM gateway** assignment token loop in the target path.

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
  weights_url: https://chain.joinbase.ai
  submit_on_chain_enabled: false
  agent:
    # Base master coordination API only (--master-url). Required; never invented.
    # Public network sample:
    master_url: https://chain.joinbase.ai
    # Weight-only default (no challenge adapters / no writer role):
    challenge_execution_enabled: false
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
/ `BASE_GATEWAY_TOKEN` contract in the shipping target path. Challenge execution
adapters stay off unless an operator explicitly sets
`challenge_execution_enabled: true` (still never a challenge DB / leaderboard
writer).

### Validator agent image (digest pin)

Pin the validator runtime by immutable digest:

```text
repository@sha256:<64-hex>
```

Mutable `latest` alone is not a production runtime selector. Validator runtime
images auto-update by default through a host-side digest reconciler that always
applies `repository@sha256:<digest>`. Master-embedded challenge processes live
inside the master container; validators do not mutate master challenge services.
Shipping Compose may mount host docker.sock into the agent as optional migration
prep only (still agent-only weight-only: no master/postgres/challenge stack).

## Challenge evaluation notes (master-owned)

Challenge scoring / submissions / leaderboard are **master-owned** (embedded ASGI
in the master container). Normal validators do not run Terminal-Bench or Prism
re-exec by default. Base public proxy exposes challenge public routes and must
block `/internal/*` and benchmark launch-shaped routes.

Inspect embedded challenge logs on the **master** project (service name is the
master app container, not a separate `challenge-*` service):

```bash
docker compose -p base-mission-master -f deploy/compose/docker-compose.yml \
  logs --since=30m base-master-validator
```

Never print token values.

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
