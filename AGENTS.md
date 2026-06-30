# Platform agent/developer notes

Operational invariants that are easy to regress on a manual edit and are
enforced by tests. Keep this in sync with the code it references.

## Eval job network isolation (base_jobs_internal)

The agent-challenge runs miner eval jobs as short-lived Docker Swarm
replicated-jobs dispatched by the broker (`base-docker-broker`). The miner's
**untrusted agent code runs INSIDE that job container**, so whatever network the
job joins, the miner code can reach. The job legitimately needs to reach exactly
TWO swarm services by name:

- `challenge-agent-challenge:8000` — the agent-challenge API, for real-time trial
  log streaming (`CHALLENGE_TERMINAL_BENCH_LOG_STREAM_URL` /
  `AGENT_CHALLENGE_INTERNAL_BASE_URL`).
- `base-master-proxy` — the master LLM gateway, for the agent's gated LLM calls
  (`CHALLENGE_LLM_GATEWAY_BASE_URL`).

### The topology (baked into code; never needs a manual `docker service update`)

- A dedicated overlay **`base_jobs_internal`** is created `--internal` (NO
  internet egress) and `--attachable`
  (`deploy/swarm/install-swarm.sh` `create_networks` /
  `swarm_backend.DEFAULT_JOB_NETWORK`).
- The eval **JOB** runs on `base_jobs_internal`
  (`CHALLENGE_DOCKER_BROKER_NETWORK=base_jobs_internal`, set by
  `cli_app/main.py::AGENT_CHALLENGE_JOB_NETWORK`, which reuses
  `swarm_backend.DEFAULT_JOB_NETWORK` as the single source of truth).
- The agent-challenge **API + worker** AND the **master proxy** are ATTACHED to
  `base_jobs_internal` in ADDITION to `base_challenges`, so the job can resolve /
  reach ONLY those by name.

### Why (security)

- The job reaches the API (logs) + proxy (LLM gateway), but **NOT**
  `base-master-postgres` (postgres lives on `base_challenges`, which the job is
  NOT on), and has **no direct internet** (the overlay is `--internal`). The
  agent's LLM traffic therefore goes only through the master gateway.
- Putting the job on `base_challenges` would work for DNS but would also expose
  postgres:5432 to miner code — NOT acceptable. Putting it on the default bridge
  fails DNS for swarm service names (the live breakage this fixes).

### Where it is wired

| Concern | Code |
|---------|------|
| Job network constant | `src/base/cli_app/main.py::AGENT_CHALLENGE_JOB_NETWORK` (= `swarm_backend.DEFAULT_JOB_NETWORK` = `base_jobs_internal`) |
| Multi-network service plan | `swarm_backend.SwarmServicePlan.extra_networks` → one `--network` per network in `build_service_create_argv` |
| API/worker multi-home (dynamic) | `SwarmChallengeOrchestrator(job_network_slugs={"agent-challenge"})`; `_challenge_plan` sets `extra_networks` and `start_challenge` ensures the internal overlay exists |
| API/worker multi-home (static) | `install-swarm.sh` `CHALLENGE_EXTRA_NETWORKS=("${NET_JOBS_INTERNAL}")` on the agent-challenge api + worker |
| Proxy multi-home | `install-swarm.sh` `_deploy_master_service` adds a second `--network "${NET_JOBS_INTERNAL}"` for the proxy only |
| Network creation | `install-swarm.sh` `create_networks` / `_create_overlay "${NET_JOBS_INTERNAL}" true` (internal) |

### Do NOT change

- The broker (`base-docker-broker`) is **not** on `base_jobs_internal` — only the
  proxy serves the gateway. Adding the broker would be unnecessary surface.
- **terminal-bench TASK containers** (where `git clone` / installs happen) are
  launched separately on the host docker daemon with per-task `allow_internet`
  (default-bridge public egress). Their networking MUST stay unrestricted public
  egress — this isolation is about the JOB orchestrator container ONLY.
- prism services are **not** multi-homed onto `base_jobs_internal`: the prism eval
  job is egress-locked by the broker pinning the JOB to the internal overlay
  (`broker_egress_locked_slugs`), not by multi-homing the long-lived prism
  service.

Tests: `tests/unit/test_swarm_backend.py` (multi-network argv + orchestrator
multi-homing), `tests/unit/test_seed_docker_backend.py` (job network constant +
LOG_STREAM host == service name), `tests/unit/test_client_service_cli_config.py`,
`tests/unit/test_install_swarm_decentralized_deploy.py` (proxy + api/worker
attach).
