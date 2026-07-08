# Cortex Foundation Master Installation Guide

Foundation-only installer for Cortex Foundation master infrastructure. Do not run this for validators or third-party operators.

This guide covers the committed Docker Swarm bring-up for the master control plane: the BASE master proxy, broker, challenge services, and the systemd supervisor on the manager node. It does not configure the on-chain submitter, chain submission, or any key material.

## Manager node

The master runs as a single-node Docker Swarm manager hosting the platform API (a single proxy that also serves the `/v1/registry` and `/v1/weights/latest` reads plus the token-gated admin routes), the broker, the supervisor, and the challenge service containers. Challenge code runs on the manager pinned to `node.role==manager`; only short-lived broker jobs are dispatched to worker nodes. Default manager ports are proxy `8080` and broker `8082`. The control-plane database URL is supplied through a Docker secret, never on the command line. Use a disposable host when validating a full bring-up.

## Automatic Install

`install-swarm.sh` is **dry-run by default**: with no flags it prints every planned mutating command and changes nothing. It performs work only with `--apply`, and every destructive step is behind its own explicit opt-in flag:

```bash
./deploy/swarm/install-swarm.sh                                  # dry-run: prints the plan
./deploy/swarm/install-swarm.sh --apply
./deploy/swarm/install-swarm.sh --apply --restart-dockerd        # write /etc/docker/daemon.json + restart dockerd
./deploy/swarm/install-swarm.sh --apply --single-node-placement  # non-default placement override
./deploy/swarm/install-swarm.sh --apply --static-challenges      # create challenge services directly
```

It initializes the Swarm, creates the encrypted overlay networks (`base_challenges` and the internal `base_jobs_internal`, MTU 1450), creates value-bearing Docker secrets via stdin (never argv), and creates the master proxy, broker, and challenge services. No secret value is ever printed; plan output shows only the environment variable name. It produces no docker-compose or stack YAML: the installer is imperative `docker swarm` / `docker service create` / `docker secret` / `docker network` only.

## Supervisor

The control-plane supervisor replaces the old Kubernetes CronJobs with a single watchdog-supervised systemd service. Install the unit from `deploy/swarm/base-supervisor.service`:

```bash
cp deploy/swarm/base-supervisor.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now base-supervisor.service
```

The unit is `Type=notify` with a 30s watchdog and runs `base master supervisor --config /etc/base/master.yaml`. Its manager-only loops are broker-health, timeout-reaper, image-updater, challenge-image-updater, config-sync, and self-update. The image updaters resolve the public GHCR tag digest and roll the Swarm services to `tag@sha256:<digest>` only when a mutable tag moves; no GHCR pull secret is required for public packages.

## Worker enrollment

Workers run short-lived CPU/GPU broker jobs and are added manually with a Swarm join token (no SSH). Manage them from the manager with the `base master worker` CLI group: `list`, `token [--cpu|--gpu]`, `label <node> --workload cpu|gpu`, `drain <node>`, `inspect <node>`, `rm <node>`.

Enrollment flow:

1. On the manager, run `base master worker token --cpu` (or `--gpu`) and copy the printed join command:

   ```text
   docker swarm join --token <TOKEN> <MANAGER_IP>:2377
   ```

2. On the worker, install the matching `daemon.json` (`deploy/swarm/daemon.worker.json` for GPU workers, which advertises `node-generic-resources: ["NVIDIA-GPU=GPU-<uuid>"]` and registers the NVIDIA runtime), then run the join command.
3. On the manager, label the node with `base master worker label <node> --workload cpu` or `--workload gpu` (setting `node.labels.base.workload`).

The broker then schedules CPU jobs onto `node.labels.base.workload==cpu` and GPU jobs onto `node.labels.base.workload==gpu` with `--generic-resource NVIDIA-GPU=<N>`.

## Explicit Non Goals

- No on-chain submitter, master weights CLI, or master on-chain submission unit.
- Never asks for, prints, or stores key material.
- Produces or consumes no docker-compose / stack files.

## Runtime Checks

```bash
docker service ls
docker service ps base-master-proxy base-docker-broker
docker service logs -f base-master-proxy
journalctl -u base-supervisor.service -f
docker node ls
```

## Validation Commands

Before changing the installer or docs, run `bash -n deploy/swarm/install-swarm.sh`, `uv run ruff check .`, `uv run mypy src tests`, and `uv run pytest`. Run the full installer only when the current host is owned by Cortex Foundation master infrastructure.
