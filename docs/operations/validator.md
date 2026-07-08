# Validator Operations

Run these from the repository root. This runbook covers the on-chain submitter and the Docker Swarm services it depends on.

## Install, Update, Stop

The submitter is a systemd service installed from `deploy/swarm/submitter/`: one process plus the validator hotkey, with no orchestrator and no local database.

```bash
cp deploy/swarm/submitter/run_submitter.py   /var/lib/base/submitter/
cp deploy/swarm/submitter/submitter.yaml     /etc/base/submitter.yaml
cp deploy/swarm/submitter/base-submitter.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now base-submitter.service
```

To update, replace the files and `systemctl restart base-submitter.service`; to stop, `systemctl disable --now base-submitter.service`.

## Runtime Commands

```bash
# submitter
systemctl status base-submitter.service
journalctl -u base-submitter.service -f
# manager control plane + challenge services (Swarm CLI)
docker service ls
docker service ps base-master-proxy base-docker-broker
docker service logs -f base-master-proxy
docker node ls
# supervisor (broker-health, timeout-reaper, image-updater, challenge-image-updater, config-sync, self-update)
systemctl status base-supervisor.service
```

## Secret Handling

The only secret the submitter needs is the validator hotkey. Never place coldkey material on the node, and never store mnemonics or hotkeys in `.env`, shell history, support threads, screenshots, or evidence logs. The submitter reads the hotkey from:

```text
/var/lib/base/wallets/base-validator/hotkeys/validator
```

Keep that file readable only by the submitter service account. Manager control-plane and challenge secrets are Docker secrets mounted at `/run/secrets/base/<name>`; never print their values.

## Worker Nodes

Worker nodes run short-lived broker jobs and are managed from the manager with the `base master worker` CLI group:

```bash
base master worker list
base master worker token --cpu
base master worker token --gpu
base master worker label <node> --workload cpu
base master worker label <node> --workload gpu
base master worker drain <node>
base master worker inspect <node>
base master worker rm <node>
```

The broker schedules CPU jobs onto `node.labels.base.workload==cpu` and GPU jobs onto `node.labels.base.workload==gpu` with `--generic-resource NVIDIA-GPU=<N>`.

## Validator Agent (Decentralized Executor)

The validator agent is the decentralized executor that performs evaluation work. It extends the `base validator` CLI and runs as a long-running loop:

```bash
base validator agent --config /etc/base/validator.yaml
```

The agent:

- hotkey-registers and heartbeats with the master coordination plane on a configurable interval (registration is an idempotent server-side upsert and all assignment state lives on the master, so it recovers across restarts);
- pulls its assigned work units, executes each on its own Docker broker, and posts results back to the master;
- obtains a scoped gateway token per assignment and routes every LLM call through the master gateway (agents receive `BASE_LLM_GATEWAY_URL` → the gateway `/llm/v1` route + a scoped `BASE_GATEWAY_TOKEN`). The validator holds no provider key.

Run `base master broker` on the validator node so the agent has its own broker; it never executes work on the master. Relevant `validator.yaml` keys:

```yaml
validator:
  agent:
    master_url: https://chain.joinbase.ai
    gateway_url: https://chain.joinbase.ai
    capabilities: ["cpu"]
    version: "0.1.0"
    heartbeat_interval_seconds: 60
    poll_interval_seconds: 5.0
    broker_url: http://127.0.0.1:8082
    broker_token_file: /run/secrets/base_broker_token
```

Leave `heartbeat_interval_seconds` unset to use the interval the master returns from registration. A GPU validator advertises `capabilities: ["cpu", "gpu"]`.

### Validator Agent Image (GHCR digest-pin)

The validator agent ships in the `ghcr.io/baseintelligence/base` image, built and pushed by the CI `docker-build`/`docker-publish` matrix from `docker/Dockerfile.validator`. It carries the same GHCR tag policy as the master/broker images (`latest` only from `main`, plus semver and `sha-<sha>` tags). Pin it by immutable digest on the validator node, exactly as the manager pins `ghcr.io/baseintelligence/base-master`:

```bash
docker pull ghcr.io/baseintelligence/base:latest
# resolve the digest, then run/auto-update the agent against the pinned ref:
#   ghcr.io/baseintelligence/base@sha256:<64-hex>
```

The manager proxy/broker services stay current via the supervisor's GHCR digest-pin loop (`SwarmImageUpdater`, which refuses any non-`@sha256:` ref); the validator agent follows the same policy. Check it with `journalctl -u base-validator-agent.service -f`.

## Agent Challenge Execution Backend Checks

Agent Challenge Terminal-Bench evaluation runs through the `own_runner` backend: the agent-challenge worker sidecar dispatches a non-privileged Docker-out-of-Docker job to the BASE broker, which runs the eval inside the `ghcr.io/baseintelligence/agent-challenge-terminal-bench-runner` image. `own_runner` is the only supported backend (no Daytona or `base_sdk` path in production). The public proxy exposes only challenge public routes and must block `/internal/*`, `POST /internal/v1/submissions/{submission_id}/launch`, and generic benchmark execution-shaped routes such as `/benchmark-executions`; the broker is an internal execution substrate, not a public miner API.

Use placeholder service names and avoid printing token values:

```bash
docker service ps <agent-challenge-service>
docker service logs <agent-challenge-service> --since=30m | rg 'terminal_bench|own_runner|tb_running'
docker service logs base-docker-broker --since=30m | rg 'run request|created job|agent-challenge-terminal-bench-runner'
curl -sS '<api-base-url>/submissions/<submission-id>/status' | rg '"status":"evaluating"|"status":"valid"|"status":"error"'
```

The relevant knobs are `CHALLENGE_TERMINAL_BENCH_EXECUTION_BACKEND=own_runner`, `CHALLENGE_DOCKER_BACKEND=broker`, the broker URL plus token file, `CHALLENGE_HARBOR_RUNNER_IMAGE` (the deployed `agent-challenge-terminal-bench-runner` tag), and a scoped `CHALLENGE_DOCKER_ALLOWED_IMAGES` covering that runner. They are set on both the agent-challenge API and worker sidecar; see the agent-challenge repository docs for the full reference.

## Validation

```bash
bash -n deploy/swarm/install-swarm.sh
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest --cov=base --cov-report=term-missing --cov-fail-under=80
```

If Docker, the Swarm, or a Python tool is unavailable, record the missing tool as a blocker instead of marking that surface as tested.
