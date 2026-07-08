# Deploy From Scratch

End-to-end path to stand up the full subnet on a fresh Docker Swarm: a manager node (control plane
plus the long-lived challenge services) and one or more CPU/GPU workers (short-lived broker eval
jobs). Weights are computed **dry-run** by default; on-chain submission is per-validator and gated
by `validator.submit_on_chain_enabled`. Run `install-swarm.sh` **dry-run first** (no flags) and only
`--apply` on a host you own.

The three backend repositories are sibling checkouts under a common parent (`platform/`,
`agent-challenge/`, `prism/`); the frontend deploys separately.

## Topology and Ports

| Node | Swarm role | Runs |
|------|------------|------|
| Manager (also validator / hotkey node) | `node.role==manager` | Control plane (proxy / broker / supervisor) **and** the challenge services |
| CPU worker | `node.labels.base.workload==cpu` | Short-lived CPU broker jobs |
| GPU worker | `node.labels.base.workload==gpu` | Short-lived GPU broker jobs; advertises `NVIDIA-GPU` as a Swarm generic resource |

Manager control-plane services are published on fixed host ports by `install-swarm.sh --apply`
(overridable via `MASTER_PROXY_PORT` / `MASTER_BROKER_PORT`):

| Manager service (host-published) | Host port |
|---------|-----------|
| base-master-proxy (single public API; serves `/v1/registry`, `/v1/weights/latest`, `/health`, routes `/challenges/*`) | 19080 |
| base-docker-broker | 8082 |

Challenge services and the Postgres stores are **overlay-internal** (no host publish): clients reach
challenges **through the proxy** over the `base_challenges` overlay
(e.g. `http://127.0.0.1:19080/challenges/prism/...`), and the master reaches Postgres by service name.

| Overlay-internal service | Container port |
|---------|-----------|
| challenge-agent-challenge (plus worker sidecar) | 8000 |
| challenge-prism (SQLite-backed) | 8080 |
| base-master-postgres / challenge-*-postgres | 5432 |

GPU eval jobs are dispatched by the broker to a GPU worker via `node.labels.base.workload==gpu` plus
`--generic-resource NVIDIA-GPU=<N>`.

## Step 1 — Build the images

`<tag>` is your release tag (a SemVer such as `3.0.0`, or `latest` for the mutable channel).

```bash
# base-master (this repo): proxy + broker + supervisor
docker build -f docker/Dockerfile.master -t ghcr.io/baseintelligence/base-master:<tag> .

# prism API + GPU evaluator (from ../prism)
docker build --target service   -t ghcr.io/baseintelligence/prism:<tag> ../prism
docker build --target evaluator -t ghcr.io/baseintelligence/prism-evaluator:<tag> ../prism

# agent-challenge API + eval-job image (from ../agent-challenge)
docker build --target runtime               -t ghcr.io/baseintelligence/agent-challenge:<tag> ../agent-challenge
docker build --target terminal-bench-runner -t ghcr.io/baseintelligence/agent-challenge-terminal-bench-runner:<tag> ../agent-challenge
```

> **Build-order coupling:** prism pins its `base` dependency by git (public HEAD), so a fresh `prism`
> build bundles whatever is on the **pushed** platform HEAD. Push the platform commits the
> prism/broker images depend on **before** building `prism` / `prism-evaluator`.

## Step 2 — Publish or stage the images

- **GHCR publish (preferred):** `docker push` each tag to `ghcr.io/baseintelligence/*`. Public
  packages need no pull secret; the supervisor image-updaters then track digests automatically.
- **Local-only staging:** build each image on the node that runs it and deploy with
  `docker service update --no-resolve-image` so a non-registry tag resolves to the node-local image.

## Step 3 — Provision named volumes and secrets

On the **GPU worker**, stage the locked PRISM data and reference tokenizers as read-only volumes:

- `prism_fineweb_edu_train` → `/data/fineweb-edu/train` (miner-visible, read-only)
- `prism_fineweb_edu_val`, `prism_fineweb_edu_test` → secret held-out, scorer-only (never mounted in the `network=none` eval container)
- `prism_reference_tokenizers` → `/opt/prism/reference-tokenizers`

On the **manager**, provision the agent-challenge read-only task cache and golden volumes:

```bash
deploy/swarm/acquire-agent-challenge-cache.sh
```

Verify each volume is both present **and populated** (a Docker named volume is auto-created empty on
first mount). The central LLM gates reach the provider only through the master gateway using the
scoped `base_gateway_token` (`source=llm_review`); no raw provider key is mounted on any challenge or
eval container. The single provider key is held only by the master gateway.

## Step 4 — Bring up the manager

`deploy/swarm/install-swarm.sh` is the canonical entry point: **dry-run by default**, mutates only
with `--apply`, and keeps every destructive step behind its own flag.

```bash
export IMAGE_MASTER=ghcr.io/baseintelligence/base-master:<tag>
export IMAGE_PRISM=ghcr.io/baseintelligence/prism:<tag>
export IMAGE_PRISM_EVALUATOR=ghcr.io/baseintelligence/prism-evaluator:<tag>
export IMAGE_AGENT_CHALLENGE=ghcr.io/baseintelligence/agent-challenge:<tag>
export AGENT_CHALLENGE_RUNNER_IMAGE=ghcr.io/baseintelligence/agent-challenge-terminal-bench-runner:<tag>

./deploy/swarm/install-swarm.sh                            # dry-run: prints the planned docker commands
./deploy/swarm/install-swarm.sh --apply                    # apply on a disposable / owned host
./deploy/swarm/install-swarm.sh --apply --restart-dockerd  # also write daemon.json + restart dockerd
```

The installer initializes the Swarm, creates the encrypted overlay networks (`base_challenges`,
`base_jobs_internal`, MTU 1450), creates the value-bearing Docker secrets via stdin, and creates the
master proxy/broker. Challenge services are then deployed automatically by the proxy's registry
reconcile loop (one combined-mode service per ACTIVE challenge); `--static-challenges` instead
creates them directly. It also wires the coordination plane and the LLM gateway (mandatory
`GATEWAY_TOKEN`, advertised `GATEWAY_PUBLIC_BASE_URL`, server-side provider keys, optional `HF_TOKEN`).

## Step 5 — Enroll worker nodes

Workers are added manually with a Swarm join token (no SSH). From the manager:

```bash
base master worker token --cpu      # or --gpu — prints the docker swarm join command
```

On the worker, install the matching `daemon.json` and join, then label the node back on the manager:

```bash
JOIN_TOKEN=<TOKEN> scripts/install-worker.sh --manager-addr <MANAGER_IP>:2377 --workload cpu --restart-dockerd --apply
base master worker label <node> --workload cpu      # or gpu
```

See [`deploy/swarm/README.md`](../deploy/swarm/README.md) for `daemon.json` details, networking
ports, and the prune policy.

## Step 6 — On-chain weight submission

On-chain submission is per-validator (the validator agent submits its own weights when
`validator.submit_on_chain_enabled` is on). A dedicated submit-only host is also supported:

```bash
cp deploy/swarm/submitter/run_submitter.py       /var/lib/base/submitter/
cp deploy/swarm/submitter/submitter.yaml         /etc/base/submitter.yaml
cp deploy/swarm/submitter/base-submitter.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now base-submitter.service
```

Decentralized evaluation runs on validator nodes (hotkey must hold a metagraph validator permit):

```bash
base validator agent --config config/validator.example.yaml
```

Full submitter configuration is in the [Validator guide](validator/README.md); manager runbooks are
in [Validator operations](operations/validator.md).

## Step 7 — Verify

```bash
docker service ls
curl -sf http://127.0.0.1:19080/health                                 # proxy
curl -sf http://127.0.0.1:8082/health                                  # broker
curl -sf http://127.0.0.1:19080/v1/registry                            # registry (via proxy)
curl -sf http://127.0.0.1:19080/v1/weights/latest                      # weights (via proxy)
curl -sf http://127.0.0.1:19080/challenges/prism/leaderboard           # prism, via proxy
curl -sf http://127.0.0.1:19080/challenges/agent-challenge/leaderboard  # agent-challenge, via proxy
```

## Step 8 — Public edge (Cloudflare)

The single platform API listens on `127.0.0.1:19080`. To expose it publicly as
`https://chain.joinbase.ai`, front it with a Cloudflare tunnel using **one catch-all ingress rule**
(`chain.joinbase.ai -> http://127.0.0.1:19080`) with **no `/v1` path-split**: the one port already
serves `/health`, `/v1/registry`, `/v1/weights/latest`, `/challenges/*`, and the token-gated
control-plane routes. Public read routes return `200`; admin-write/control-plane routes stay private
(`401`/`405`), and `/internal/*` returns `404` at the edge.
