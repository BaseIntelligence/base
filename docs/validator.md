# Validator Quick Start

This page is only for the normal validator on-chain submitter. The submitter is a
minimal systemd service that fetches the master weight vector from
`https://chain.platform.network/v1/weights/latest` and submits it on-chain. It runs
no challenge orchestration; all challenge services run on the Platform master
(manager) node.

## Install the submitter

The submitter ships in `deploy/swarm/submitter/`:

- `run_submitter.py` is the submit-only process.
- `submitter.yaml` is the credential-free config (netuid, wallet identity, and the master `weights_url`).
- `platform-submitter.service` is the systemd unit.

Install it on the validator node:

```bash
cp deploy/swarm/submitter/run_submitter.py   /var/lib/platform/submitter/
cp deploy/swarm/submitter/submitter.yaml     /etc/platform/submitter.yaml
cp deploy/swarm/submitter/platform-submitter.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now platform-submitter.service
```

The unit runs the submitter under `Restart=always`, reading the validator hotkey
from `/var/lib/platform/wallets/platform-validator/hotkeys/validator`. It opens no
control-plane database connection: it only talks to the master over HTTP and to the
chain. Never place coldkey material on the node.

Follow the submitter:

```bash
journalctl -u platform-submitter.service -f
```

Stop the submitter:

```bash
systemctl disable --now platform-submitter.service
```

## Requirements

The submitter is a single Python process plus the validator hotkey. It does not
need a Kubernetes cluster, an orchestrator, or a local database. A validator node
that also acts as the Swarm manager should have at least 2 vCPUs and 8 GB RAM; a
submit-only node needs far less.

If running even the submitter is too much, use the Bittensor CHK / stake weight
check flow to give validator power to the recommended Platform validator hotkey
instead of running it yourself:
`5GziQCcRpN8NCJktX343brnfuVe3w6gUYieeStXPD1Dag2At`.

## Manual Install

If you do not use the unit file, run the same process under any supervisor:

```text
python deploy/swarm/submitter/run_submitter.py --config /etc/platform/submitter.yaml
```

Set `validator.weights_url` to the master endpoint and `network.netuid`,
`network.wallet_name`, `network.wallet_hotkey`, and `network.wallet_path` to your
validator identity. The submitter polls `/v1/weights/latest` on
`validator.weights_interval_seconds` and retries on transient master or chain
failures.

## Safety

- The submitter never needs coldkey material.
- The submitter only reads the master weights vector and submits it on-chain.
- The default weights source is `https://chain.platform.network`.
- The submitter runs no challenge services; challenges run on the master node.
- Keep the hotkey file readable only by the submitter service account.
