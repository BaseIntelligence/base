# Validator Submitter Guide (Docker Swarm)

This guide is only for normal validators. It installs the submit-only on-chain
weight submitter as a systemd service. The submitter fetches the master weight
vector from the public BASE endpoint and submits it on-chain. It runs
no challenge orchestration: all challenge services run on the BASE
master (manager) node.

The default weights endpoint is:

```text
https://chain.joinbase.ai/v1/weights/latest
```

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

The submitter polls `validator.weights_url` (default
`https://chain.joinbase.ai`) at `validator.weights_interval_seconds`, reads
`/v1/weights/latest`, and submits the fetched vector on-chain for the configured
`network.netuid`. It retries on transient master or chain failures and skips
submission when the master vector is stale beyond
`validator.weights_freshness_seconds`. The relevant `submitter.yaml` keys:

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

### Is Kubernetes required?

No. There is no Kubernetes anywhere in BASE. The submitter is a single
systemd-managed Python process. It does not deploy an orchestrator and does not
run challenge workloads.

### Do I need to run the challenges?

No. Challenge services run on the BASE master (manager) node, scheduled as
Docker Swarm services with the placement constraint `node.role==manager`. The
submitter only reads the master weight vector and submits it on-chain.

### Do I need a database?

No. The submit path never opens the control-plane database. The shared
control-plane PostgreSQL is used by the master/manager only and is not a submitter
dependency.

### What are the minimum requirements?

A submit-only node needs very little: a Python runtime, network access to the
master endpoint and the chain, and the validator hotkey file. A node that also
acts as the Swarm manager should have at least 2 vCPUs and 8 GB RAM.

### What if the requirements are too high?

Use the Bittensor CHK / stake weight check flow to give validator power to the
recommended BASE validator hotkey instead of running the submitter yourself:

```text
5GziQCcRpN8NCJktX343brnfuVe3w6gUYieeStXPD1Dag2At
```

## Manual Install

If you do not use the unit file, run the same process under any supervisor:

```text
python /var/lib/base/submitter/run_submitter.py --config /etc/base/submitter.yaml
```

Ensure the hotkey file exists at
`/var/lib/base/wallets/base-validator/hotkeys/validator` and that
`submitter.yaml` points `validator.weights_url` at the master endpoint.

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
