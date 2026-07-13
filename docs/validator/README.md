# Validator Guide (Docker Compose)

This guide covers running a BASE validator as an **independent Docker Compose
project**. Compose is the only supported shipping backend for new installs. There is
no Kubernetes path, and Docker Swarm is **not** required.

**Validators never run master.** The install is agent-only: one validator container
points at an external Base master/coordination API. Challenge control-plane state,
aggregation, PostgreSQL, and the challenge watcher stay on the master host. On-chain
submission always uses the **validator's own wallet**. See
[Compute Requirements](#compute-requirements).

### `master_url` vs registry / weights aliases

| Setting | Role |
| --- | --- |
| `validator.agent.master_url` (`--master-url`) | Required Base master coordination API (register/heartbeat/pull/result). Never invented. |
| `validator.registry_url` / `validator.weights_url` | Registry / published-weights aliases. Installer sets them equal to `--master-url` when that master hosts both. |

### Public hostnames (preferred product vs live known-good)

| URL | Status as of 2026-07-13 |
| --- | --- |
| `https://chain.platform.network` | Preferred **product** Base master hostname once cutover completes. Live `/health` currently serves **agent-challenge**, not Base master. Do not force this as an operator default without re-verify. |
| `https://chain.joinbase.ai` | Live known-good Base master front (`role=master`). Settings defaults and public weights examples use this until cutover. |
| `http://127.0.0.1:3180` | Local disposable master for smoke only. |

Default public weights example (live known-good):

```text
https://chain.joinbase.ai/v1/weights/latest
```

Network operators may point `--master-url` (and therefore generated registry/weights)
at `https://chain.joinbase.ai` today, or at their own operator master. After
`chain.platform.network` fronts Base master end-to-end, product messaging prefers that
hostname; until then document both.

## Compute Requirements

| Validator profile | Compute |
|-------------------|---------|
| Submit-only / simple validator (no challenge execution) | 2 vCPU, 4 GB RAM |
| Validator running base (agent-challenge) evaluation | 8 vCPU, 32 GB RAM |
| PRISM challenge | No additional local GPU required for standard Verify path |

PRISM heavy GPU evaluation is delegated to miner-funded workers (worker plane) or
external long-lived TEE paths; the validator performs light verification and
probabilistic audit work. Operator sizing should still leave headroom for Docker
and chain clients.

## Supported install (one command)

Installers and manifests:

| Artifact | Path |
| --- | --- |
| Installer | `deploy/compose/install-validator.sh` |
| Compose file | `deploy/compose/docker-compose.validator.yml` |
| Example env | `deploy/compose/.env.validator.example` |

```bash
./deploy/compose/install-validator.sh \
  --project-name base-mission-validator-a \
  --master-url http://127.0.0.1:3180
```

Required inputs the installer stages (or accepts via env / flags):

| Input | Role |
| --- | --- |
| `--master-url` / `VALIDATOR_MASTER_URL` | Absolute master coordination URL |
| `BASE_VALIDATOR_IMAGE_*` | Immutable image repository + sha256 digest |
| Protocol identity | Host directory for the protocol signing wallet |
| Broker / capability config | Written into `validator.yaml` under the operator state dir |

Each validator runs as **its own Compose project** with distinct network, volume,
identity, and secrets. It never receives master PostgreSQL credentials, challenge
`/data` volumes, Docker socket access, or aggregation operators.

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
(`repository@sha256:<64 hex>`). Mutable `latest` without a digest is not a production
selector. Validator image updates are operator-driven Compose recreate (or a future
validator-side pin process), not Swarm `docker service update` of a mutable tag.

Challenge auto-update on the **master** side is the master-resident digest-aware
watcher (see [master guide](../master/README.md) and [compose.md](../compose.md));
validators do not orchestrate master challenge rollouts.

## Weight submission model

1. Challenges push authenticated raw weights to the master.
2. Master persists snapshots and aggregates a final vector.
3. Validator fetches `GET /v1/weights/latest` (or the configured weights URL).
4. Validator submits with its own hotkey / wallet (`set_weights` path).
5. Master never submits on-chain.

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

**Do I need to run the challenges on the validator host?** No. Long-lived challenge
services run in the master Compose project. The validator coordinates with the master
and runs evaluation only for work assigned to that validator.

**Do I need the master database?** No. Validators never open the control-plane
PostgreSQL; they talk HTTP to the master and (optionally) to the chain.

**What are the minimum requirements?** See [Compute Requirements](#compute-requirements).
A weights-only node needs a network path to master and chain plus identity material.

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
