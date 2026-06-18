# Validator Operations

Run these commands from the repository root. This runbook covers the normal
validator on-chain submitter and the Docker Swarm services it depends on.

## Install Or Update

The submitter is a systemd service installed from `deploy/swarm/submitter/`. It is
a single process plus the validator hotkey; it needs no orchestrator and no local
database.

```bash
cp deploy/swarm/submitter/run_submitter.py   /var/lib/platform/submitter/
cp deploy/swarm/submitter/submitter.yaml     /etc/platform/submitter.yaml
cp deploy/swarm/submitter/platform-submitter.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now platform-submitter.service
```

To update, replace the files and restart the unit:

```bash
systemctl restart platform-submitter.service
```

Stop the submitter:

```bash
systemctl disable --now platform-submitter.service
```

## Submitter Runtime Commands

```bash
systemctl status platform-submitter.service
journalctl -u platform-submitter.service -f
journalctl -u platform-submitter.service --since=30m
```

## Manager Service Commands

Challenge services and the master control plane run as Docker Swarm services on the
manager node. Inspect them with the Swarm CLI:

```bash
docker service ls
docker service ps platform-master-admin platform-master-proxy platform-master-broker
docker service logs -f platform-master-admin
docker service logs --tail 200 platform-master-broker
docker node ls
```

## Supervisor

The manager-only supervisor runs the broker-health, timeout-reaper, image-updater,
challenge-image-updater, config-sync, and self-update loops:

```bash
systemctl status platform-supervisor.service
journalctl -u platform-supervisor.service -f
```

## Secret Handling

The only secret the submitter needs is the validator hotkey. Never place coldkey
material on the node. Do not store mnemonics or hotkeys in `.env`, shell history,
support threads, screenshots, or evidence logs.

The submitter reads the hotkey from:

```text
/var/lib/platform/wallets/platform-validator/hotkeys/validator
```

Keep that file readable only by the submitter service account. Control-plane and
challenge secrets on the manager are Docker secrets mounted at
`/run/secrets/platform/<name>`; never print their values.

## Worker Nodes

Worker nodes run short-lived broker jobs and are managed from the manager with the
`platform master worker` CLI group:

```bash
platform master worker list
platform master worker token --cpu
platform master worker token --gpu
platform master worker label <node> --workload cpu
platform master worker label <node> --workload gpu
platform master worker drain <node>
platform master worker inspect <node>
platform master worker rm <node>
```

The broker schedules CPU jobs onto `node.labels.platform.workload==cpu` and GPU
jobs onto `node.labels.platform.workload==gpu` with `--generic-resource
NVIDIA-GPU=<N>`.

## Agent Challenge Platform SDK Execution Checks

Agent Challenge production Terminal-Bench rollout uses `platform_sdk` through the
generic Platform broker. The public proxy must still expose only challenge public
routes and must block `/internal/*`, `POST
/internal/v1/submissions/{submission_id}/launch`, and generic benchmark
execution-shaped routes such as `/benchmark-executions`; the broker is an internal
execution substrate, not a public miner API.

Use placeholder service names only and avoid printing token values:

```bash
docker service ps <agent-challenge-service>
docker service logs <agent-challenge-service> --since=30m | rg 'terminal_bench|platform_sdk|tb_running'
docker service logs platform-master-broker --since=30m | rg 'run request|created job|agent-challenge-terminal-bench-runner'
docker service logs <agent-challenge-service> --since=30m | rg -- '--environment-import-path agent_challenge_runner.platform_environment:PlatformEnvironment'
! docker service logs <agent-challenge-service> --since=30m | rg --fixed-strings -- '--env daytona'
! docker service logs <agent-challenge-service> --since=30m | rg --fixed-strings -- '--env platform'
curl -sS '<api-base-url>/submissions/<submission-id>/status' | rg '"status":"evaluating"|"phase":"evaluation"|"status":"valid"|"status":"error"'
```

Safe Agent Challenge knobs are `CHALLENGE_TERMINAL_BENCH_EXECUTION_BACKEND=platform_sdk`, broker URL plus token file, `CHALLENGE_PLATFORM_SDK_RUNNER_IMAGE=ghcr.io/platformnetwork/agent-challenge-terminal-bench-runner:latest`, `CHALLENGE_PLATFORM_SDK_ENVIRONMENT_IMPORT_PATH=agent_challenge_runner.platform_environment:PlatformEnvironment`, and a scoped allowed-image policy. Platform SDK Harbor commands use `--environment-import-path`, not `--env platform`, and production does not require Daytona credentials. Roll back to `harbor` only for non-production testing or for an explicitly credentialed legacy Harbor environment; production remains `platform_sdk` after rollout.

## Validation

```bash
bash -n deploy/swarm/install-swarm.sh
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest --cov=platform_network --cov-report=term-missing --cov-fail-under=80
```

If Docker, the Swarm, or a Python tool is unavailable, record the missing tool as a
blocker instead of marking that surface as tested.
