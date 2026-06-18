# Cortex Foundation Master Installation Guide

Foundation-only installer for Cortex Foundation master infrastructure. Do not run this for validators or third-party operators.

This guide covers the committed Docker Swarm bring-up for the master control plane. It installs the Platform master admin API, proxy, broker, the challenge services, and the systemd supervisor on the manager node. It does not configure the on-chain submitter, chain submission, or any key material.

## Manager node

The master runs as a single-node Docker Swarm manager. The manager hosts the admin API, proxy, broker, supervisor, and the challenge service containers themselves. Challenge code runs on the manager with the placement constraint `node.role==manager`; only short-lived broker jobs are dispatched to worker nodes.

Default service ports on the manager:

```text
admin  : 8000
proxy  : 8080
broker : 8082
```

The control-plane database URL is supplied through a Docker secret, not on the command line. Use a disposable host when validating the full bring-up.

## Automatic Install

Bring up the manager from the repository root with the Swarm installer:

```bash
./deploy/swarm/install-swarm.sh
```

`install-swarm.sh` is **dry-run by default**: with no flags it prints every planned mutating command and changes nothing. It performs work only with `--apply`, and every destructive step is behind its own explicit opt-in flag so the operator advances one step at a time:

```bash
./deploy/swarm/install-swarm.sh --apply
./deploy/swarm/install-swarm.sh --apply --restart-dockerd        # write /etc/docker/daemon.json + restart dockerd
./deploy/swarm/install-swarm.sh --apply --single-node-placement  # non-default placement override
./deploy/swarm/install-swarm.sh --apply --static-challenges      # create challenge services directly
```

The installer initializes the Swarm, creates the encrypted overlay networks (`platform_challenges` and the internal `platform_jobs_internal`, MTU 1450), creates the value-bearing Docker secrets via stdin (never argv), and creates the master admin, proxy, broker, and challenge services. No secret value is ever printed; plan output shows only the environment variable name. No docker-compose or stack YAML is produced or consumed; the installer is imperative `docker swarm` / `docker service create` / `docker secret` / `docker network` only.

## Supervisor

The control-plane supervisor replaces the old Kubernetes CronJobs with a single watchdog-supervised systemd service. Install the unit from `deploy/swarm/platform-supervisor.service`:

```bash
cp deploy/swarm/platform-supervisor.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now platform-supervisor.service
systemctl status platform-supervisor.service
```

The unit is `Type=notify` with a 30s watchdog and runs:

```text
platform master supervisor --config /etc/platform/master.yaml
```

The supervisor loops run on the manager only: broker-health, timeout-reaper, image-updater, challenge-image-updater, config-sync, and self-update. The image updaters resolve the public GHCR tag digest and roll the Swarm services to `tag@sha256:<digest>` only when a mutable tag moves; no GHCR pull secret is required for public packages.

## Worker enrollment

Workers run short-lived CPU/GPU broker jobs and are added manually with a Swarm join token (no SSH). The `platform master worker` CLI group manages them from the manager:

- `platform master worker token [--cpu|--gpu]` prints the join command for a CPU or GPU worker:

  ```text
  docker swarm join --token <TOKEN> <MANAGER_IP>:2377
  ```

- `platform master worker list` lists enrolled nodes and their workload labels.
- `platform master worker label <node> --workload cpu|gpu` sets the workload label the broker schedules against (`node.labels.platform.workload`).
- `platform master worker drain <node>` drains a node before maintenance.
- `platform master worker rm <node>` removes a node from the Swarm.
- `platform master worker inspect <node>` shows node detail.

Enrollment flow:

1. On the manager, run `platform master worker token --cpu` (or `--gpu`) and copy the printed `docker swarm join` command.
2. On the worker, install the matching `daemon.json` (`deploy/swarm/daemon.worker.json` for GPU workers, which advertises `node-generic-resources: ["NVIDIA-GPU=GPU-<uuid>"]` and registers the NVIDIA runtime), then run the join command.
3. On the manager, label the new node: `platform master worker label <node> --workload cpu` or `--workload gpu`.

The broker then schedules CPU jobs onto `node.labels.platform.workload==cpu` and GPU jobs onto `node.labels.platform.workload==gpu` with `--generic-resource NVIDIA-GPU=<N>`.

## Explicit Non Goals

- It does not create the on-chain submitter.
- It does not run the master weights CLI command.
- It does not create a master on-chain submission unit.
- It does not ask for, print, or store key material.
- It does not produce or consume docker-compose / stack files.

## Runtime Checks

```bash
docker service ls
docker service ps platform-master-admin platform-master-proxy platform-master-broker
docker service logs -f platform-master-admin
journalctl -u platform-supervisor.service -f
docker node ls
```

## Validation Commands

Before changing the installer or docs, run:

```bash
bash -n deploy/swarm/install-swarm.sh
uv run ruff check .
uv run mypy src tests
uv run pytest
```

Run the full installer only when the current host is owned by Cortex Foundation master infrastructure.
