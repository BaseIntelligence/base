# Validator Guide (Docker Compose)

This guide covers running a BASE validator as an **independent Docker Compose
project**. Compose is the only supported shipping backend for new installs. There is
no Kubernetes path, and Docker Swarm is **not** required.

## Weight-only default (shipping)

Normal validators are **weight-only**:

1. Point `--master-url` at the public Base master (`https://chain.joinbase.ai`).
2. Register / heartbeat with that master.
3. Fetch `GET https://chain.joinbase.ai/v1/weights/latest`.
4. Call `set_weights` with the **validator's own wallet** when
   `submit_on_chain_enabled` is turned on (default off / human-gated).

**Validators never run master**, never host PostgreSQL control-plane, never run
`challenge-prism` / `challenge-agent-challenge` services, and never write miner
submissions or leaderboards. Challenge ASGI and scoring run **inside the master
container** on localhost; public paths stay on `chain.joinbase.ai`.

Challenge execution adapters (`prism` / `agent-challenge` assignment cycles)
default **off** via `validator.agent.challenge_execution_enabled: false` (installer
and Settings default). Optional future **audit re-exec** may re-run challenge work
for inspection only; it must stay **non-write** (no submissions/leaderboard
mutation). Do not enable challenge execution on a shipping network validator unless
you intentionally run an experimental executor profile.

Host `docker.sock` on the Compose project is optional migration prep only; it is
**not** a challenge control-plane and does not start challenge services.

See [Compute Requirements](#compute-requirements).

### `master_url` vs registry / weights aliases

| Setting | Role |
| --- | --- |
| `validator.agent.master_url` (`--master-url`) | Required Base master coordination API (register/heartbeat/pull/result). Never invented. |
| `validator.registry_url` / `validator.weights_url` | Registry / published-weights aliases. Installer sets them equal to `--master-url` when that master hosts both. |

### Public Base master URL

| URL | Role |
| --- | --- |
| `https://chain.joinbase.ai` | Authoritative public Base master / coordination / weights API (`role=master`). Settings defaults, installer samples, and public weights examples recommend this host. Live Compose master: `base-master-prod`. |
| Other historical public hostnames | Non-authoritative secondary only (may return 502). Do not use as shipping `--master-url`. |
| `http://127.0.0.1:3180` (or private operator master) | Local disposable or private master for smoke / self-hosted control planes only (explicit `--master-url`). |

Default public weights example:

```text
https://chain.joinbase.ai/v1/weights/latest
```

Network operators should point `--master-url` (and therefore generated registry/weights
when the master hosts both) at `https://chain.joinbase.ai`, or at their own operator
master when self-hosting.

## Compute Requirements

| Validator profile | Compute |
|-------------------|---------|
| **Weight-only (shipping default)** | 2 vCPU, 4 GB RAM |
| Experimental executor profile (`challenge_execution_enabled: true`) | 8 vCPU, 32 GB RAM+ |
| PRISM on master (not on validator) | Challenge scoring lives on master; validators do not need GPU for weights |

Shipping network validators use the weight-only row. Master embeds challenge ASGI;
miners talk to `https://chain.joinbase.ai`. Optional experimental executor profiles
are not the default and still must never host challenge writer DBs.

## Supported install (one command)

Installers and manifests:

| Artifact | Path |
| --- | --- |
| Installer | `deploy/compose/install-validator.sh` |
| Compose file | `deploy/compose/docker-compose.validator.yml` |
| Example env | `deploy/compose/.env.validator.example` |

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

Required inputs the installer stages (or accepts via env / flags):

| Input | Role |
| --- | --- |
| `--master-url` / `VALIDATOR_MASTER_URL` | Absolute master coordination URL |
| `BASE_VALIDATOR_IMAGE_*` | Immutable image repository + sha256 digest |
| Protocol identity | Host directory for the protocol signing wallet |
| Broker / capability config | Written into `validator.yaml` under the operator state dir |
| `BASE_DOCKER_GID` / socket | Host docker group + `/var/run/docker.sock` bind (default on) |

Each validator runs as **its own Compose project** with distinct network, volume,
identity, and secrets. It never receives master PostgreSQL credentials, challenge
`/data` volumes, or aggregation operators. Shipping Compose mounts host
`docker.sock` into the agent for later challenges-on-validator prep.

Source-free reinstall pattern (after `install-validator.sh --copy-artifacts DIR`):

```bash
docker compose -p base-mission-validator-a \
  -f docker-compose.validator.yml --env-file .env up -d
```

Teardown of one validator only:

```bash
docker compose -p base-mission-validator-a -f docker-compose.validator.yml down
# add -v only when disposable state should be destroyed
```

### Image pins and updates

Production validators use **digest-pinned** images
(`repository@sha256:<64 hex>`). Mutable `latest` without a digest is never a
runtime selector. Runtime compose always runs the pin form above.

By default, `install-validator.sh` enables a **host-side** systemd timer
(`base-validator-image-updater@<project>.timer`, every 60–120s) that:

1. Resolves `ghcr.io/baseintelligence/base-validator-runtime:latest` to a
   `sha256:<digest>`.
2. Compares it to `BASE_VALIDATOR_IMAGE_DIGEST` in the project `.env`.
3. On change: rewrites `.env` to a REPO+DIGEST pin only, then
   `docker compose pull && up -d --force-recreate --no-deps validator`.
4. On failure: restores last-known-good, applies exponential backoff, and after
   exhaustion skips until a **new** remote digest appears.

Image auto-update remains **host-side** (the agent also mounts `docker.sock` for
later challenges-on-validator prep, but the digest reconciler is the host timer).
Opt out with `--no-auto-update`, or freeze later with
`BASE_VALIDATOR_IMAGE_UPDATE_HOLD=1`. State lives next to artifacts as
`image_update_state.json`.

Challenge auto-update on the **master** side remains the master-resident
digest-aware watcher (see [master guide](../master/README.md) and
[compose.md](../compose.md)); validators never orchestrate master challenge rollouts.

## Weight submission model

1. Challenges (embedded on master) push authenticated raw weights to the master.
2. Master persists snapshots and aggregates a final vector (sole writer).
3. Validator fetches `GET /v1/weights/latest` (default host
   `https://chain.joinbase.ai`).
4. Validator submits with its own hotkey / wallet (`set_weights` path) when gated on.
5. Master never submits on-chain. Validators never write submissions/leaderboard.

Installer-generated `validator.yaml` sets:

```yaml
validator:
  registry_url: https://chain.joinbase.ai   # or your --master-url
  weights_url: https://chain.joinbase.ai
  submit_on_chain_enabled: false              # flip with --submit-on-chain
  agent:
    master_url: https://chain.joinbase.ai
    challenge_execution_enabled: false        # weight-only default
```

There is **no LLM gateway** in the current target path: agents do not obtain scoped
gateway tokens for a master `/llm/v1` route.

## Secret rule

The validator needs its protocol identity (and, if submitting on-chain, the submit
wallet material). Never place coldkey material on disposable nodes, in shell history,
logs, screenshots, support channels, or evidence files. Prefer file-backed secrets
mode `0600` under the installer's state directory.

Typical on-chain hotkey layout when using a host wallet mount:

```text
/var/lib/base/wallets/<wallet>/hotkeys/<hotkey>
```

## Operator FAQ

**Is Kubernetes or Swarm required?** No. New installs use Docker Compose only.

**Do I need to run the challenges on the validator host?** No. Challenges are
embedded inside the **master** container (localhost ASGI). Validators do not start
`challenge-*` Compose services and do not write submissions/leaderboard. Shipping
default leaves challenge execution adapters **off**.

**Do I need the master database?** No. Validators never open the control-plane
PostgreSQL; they talk HTTP to the master and (optionally) to the chain.

**What are the minimum requirements?** See [Compute Requirements](#compute-requirements).
A weight-only node needs a network path to master and chain plus identity material.

**What if the requirements are too high?** Use the Bittensor CHK / stake weight check
flow to give validator power to a recommended BASE validator hotkey instead of running
submission yourself:

```text
5GziQCcRpN8NCJktX343brnfuVe3w6gUYieeStXPD1Dag2At
```

## Runtime checks

```bash
docker compose -p base-mission-validator-a -f deploy/compose/docker-compose.validator.yml ps
docker compose -p base-mission-validator-a -f deploy/compose/docker-compose.validator.yml logs -f validator
# Against the master:
curl -fsS http://127.0.0.1:3180/v1/weights/latest
```

## Historical Swarm note

`deploy/swarm/install-swarm.sh`, Swarm validator services, and `deploy/swarm/submitter/`
systemd units describe a **historical / unsupported** path. They are not required for
new validators. Prefer `install-validator.sh` and the validator Compose file above. See
[deploy/swarm/README.md](../../deploy/swarm/README.md).

## Validation commands

Targeted local checks before changing validator install docs (repository root):

```bash
uv run ruff check src/base/validator tests/unit/test_validator_compose_artifact.py
uv run pytest tests/unit/test_validator_compose_artifact.py tests/unit/test_validator_agent_cli_docs.py -q
```

Start a live submit path only when wallet material on the host is intentionally
authorized for chain use. CI publishes Docker images to GHCR only from trusted events.

Agent Challenge Phala full-attested mode is also **not** a multi-replica validator re-exec path: evaluation is one miner-funded external eval (R=1) and BASE creates zero agent-challenge multi-replica work rows for that submission. Validators may still run sampled labelled replay audits and the legacy R=1 `own_runner` path when attestation flags are off. See [Architecture: Agent Challenge Phala path](../architecture.md#agent-challenge-phala-intel-tdx-path).
