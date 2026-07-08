# Validator Guide (Docker Swarm)

This guide covers running a BASE validator on Docker Swarm. There is no
Kubernetes anywhere in BASE: the only backend is Docker Swarm.

Validators run in a range of profiles, from a submit-only on-chain weight
submitter to a full challenge-evaluating validator node. The simplest profile is
the submit-only submitter, a systemd service that fetches the master weight
vector from the public BASE endpoint and submits it on-chain. It runs
no challenge orchestration: all challenge services run on the BASE master
(manager) node. See [Compute Requirements](#compute-requirements) for sizing.

The default weights endpoint is:

```text
https://chain.joinbase.ai/v1/weights/latest
```

## Compute Requirements

Compute depends on which evaluation work the validator performs. These numbers
are authoritative.

| Validator profile | Compute |
|-------------------|---------|
| Submit-only / simple validator (no challenge execution) | 2 vCPU, 4 GB RAM |
| Validator running the base (agent-challenge) evaluation | 8 vCPU, 32 GB RAM |
| PRISM challenge | No additional compute required |

PRISM adds no validator compute: its heavy GPU evaluation is **delegated to
miner-funded worker agents** (the worker plane), and the validator only performs
light verification plus probabilistic replay audits. A validator never needs a
local GPU for PRISM.

## Automatic Install (One Command)

`deploy/swarm/install-swarm.sh` is the one-command install path. On a blank host
it installs everything it needs: Docker Engine (`ensure_docker`), the `uv`
runtime (`ensure_uv`, when the supervisor is installed), and the Docker Swarm
(`swarm init`).

### Dry-run by default

The installer is **dry-run by default**: with no flags it prints every planned
command and changes nothing. Pass `--apply` to execute; every destructive step
stays behind its own explicit flag.

```bash
bash deploy/swarm/install-swarm.sh --help   # list flags + required env
bash deploy/swarm/install-swarm.sh          # dry-run: prints the plan, changes nothing
```

### Auto-update

`--validator-node` brings up `base validator agent` as an **auto-updatable**
Swarm service (`base-validator-agent`) plus a node-local base-supervisor whose
image-updater digest-pins that service on every new
`base-validator-runtime:latest` digest. That image-updater **is** the
auto-update: the validator's `base` code rolls forward automatically, with no
manual `docker service update`.

`--install-supervisor` enables the `base-supervisor.service` systemd unit (the
control-plane auto-update unit). Its image-updater runs on a 60s interval, and
optional base self-update is wired only when `SUPERVISOR_SELF_UPDATE_MANIFEST_URL`
is set (otherwise self-update is explicitly disabled, never left inert).

### Quick start: validator node

Required environment for `--validator-node`:

- `VALIDATOR_MASTER_URL`: the master coordination/gateway root (for example
  `http://<master-host>:19080`). There is no default: an unset value fails fast
  so a validator never points at its own advertise address.
- `VALIDATOR_BROKER_TOKEN`: the validator's own broker token (mounted at
  `/run/secrets/base_broker_token`).
- the validator hotkey wallet, staged under `VALIDATOR_WALLET_PATH` (default
  `/var/lib/base/wallets`), wallet name `VALIDATOR_WALLET_NAME`.

`VALIDATOR_CAPABILITIES` selects the evaluation work: `["cpu"]` (default) runs
the base agent-challenge (Terminal-Bench) CPU evaluation. PRISM GPU evaluation is
**delegated** to the worker plane, so no GPU capability is needed for PRISM
(`["gpu","cpu"]` remains available for the legacy path where a validator runs
PRISM GPU re-execution at concurrency 1).

```bash
export VALIDATOR_MASTER_URL="http://<master-host>:19080"
export VALIDATOR_BROKER_TOKEN="<validator-broker-token>"
export VALIDATOR_CAPABILITIES='["cpu"]'   # base agent-challenge; PRISM is delegated
# stage the validator hotkey wallet under /var/lib/base/wallets first

# 1) DRY-RUN (default): prints the planned docker swarm commands, changes nothing
bash deploy/swarm/install-swarm.sh --validator-node

# 2) APPLY: execute, and enable the node-local auto-update supervisor unit
bash deploy/swarm/install-swarm.sh --validator-node --apply --install-supervisor
```

The dry-run renders the node-local supervisor config
(`validator_agent_target_enabled: true`, watching
`base-validator-runtime:latest`) and the per-validator `validator.yaml`, and
prints the `docker service create base-validator-agent ...` it would run. Review
the plan before you pass `--apply`.

## Secret Rule

The submitter needs exactly one secret: the validator hotkey. Never place coldkey
material on the node, in shell history, logs, screenshots, support channels, or
evidence files. Generate the hotkey files on a trusted machine and copy only the
hotkey (not the coldkey) into:

```text
/var/lib/base/wallets/base-validator/hotkeys/validator
```

## Install The Submitter

The submitter ships in `deploy/swarm/submitter/`:

| File | Destination | Purpose |
|------|-------------|---------|
| `run_submitter.py` | `/var/lib/base/submitter/run_submitter.py` | Submit-only process. |
| `submitter.yaml` | `/etc/base/submitter.yaml` | Credential-free config (netuid, wallet identity, master `weights_url`). |
| `base-submitter.service` | `/etc/systemd/system/base-submitter.service` | systemd unit. |

Install and start it:

```bash
cp deploy/swarm/submitter/run_submitter.py   /var/lib/base/submitter/
cp deploy/swarm/submitter/submitter.yaml     /etc/base/submitter.yaml
cp deploy/swarm/submitter/base-submitter.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now base-submitter.service
```

The unit runs under `Restart=always` with `HOME=/var/lib/base` so Bittensor
resolves `~/.bittensor` consistently. It opens no control-plane database
connection: it only talks to the master over HTTP and to the chain.

## How The Submitter Works

The submitter polls `validator.weights_url` (default `https://chain.joinbase.ai`)
at `validator.weights_interval_seconds`, reads `/v1/weights/latest`, and submits
the fetched vector on-chain for the configured `network.netuid`. It retries on
transient master or chain failures and skips submission when the master vector is
stale beyond `validator.weights_freshness_seconds`. The relevant `submitter.yaml`
keys:

```yaml
network:
  netuid: 100
  wallet_name: base-validator
  wallet_hotkey: validator
  wallet_path: /var/lib/base/wallets
  master_uid: 0
validator:
  weights_url: https://chain.joinbase.ai
  weights_interval_seconds: 360
  weights_timeout_seconds: 15.0
  weights_retries: 3
  weights_freshness_seconds: 720
```

## Operator FAQ

**Is Kubernetes required?** No. There is no Kubernetes anywhere in BASE. The
submitter is a single systemd-managed Python process; it deploys no orchestrator
and runs no challenge workloads.

**Do I need to run the challenges?** No. Challenge services run on the BASE master
(manager) node as Docker Swarm services pinned to `node.role==manager`. The
submitter only reads the master weight vector and submits it on-chain.

**Do I need a database?** No. The submit path never opens the control-plane
database; the shared PostgreSQL is used by the master/manager only.

**What are the minimum requirements?** See
[Compute Requirements](#compute-requirements). In short: a submit-only node needs
very little (a Python runtime, network access to the master and chain, and the
validator hotkey file) and fits in 2 vCPU / 4 GB RAM.

**What if the requirements are too high?** Use the Bittensor CHK / stake weight
check flow to give validator power to the recommended BASE validator hotkey
instead of running the submitter yourself:

```text
5GziQCcRpN8NCJktX343brnfuVe3w6gUYieeStXPD1Dag2At
```

## Manual Install

To run without the unit file, run the same process under any supervisor:

```text
python /var/lib/base/submitter/run_submitter.py --config /etc/base/submitter.yaml
```

Ensure the hotkey file exists at
`/var/lib/base/wallets/base-validator/hotkeys/validator` and that `submitter.yaml`
points `validator.weights_url` at the master endpoint.

## Runtime Checks

```bash
systemctl status base-submitter.service
journalctl -u base-submitter.service -f
```

## Validation Commands

Before changing the submitter or docs, run:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest
```

Start the submitter only when the hotkey material on the node is safe to use for
on-chain submission. CI publishes Docker images to GHCR only from trusted events:
PRs build with `push: false`, while `main`, `v*.*.*` tags, and confirmed manual
runs publish `base` and `base-master` images to GHCR.
