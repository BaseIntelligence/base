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

## Master registry-driven challenge deploy (reconciler)

The master (`base master proxy`) runs a background **registry reconcile loop**
that turns the challenge registry into running challenge services. This is what
makes installing `base` (master) auto-deploy every ACTIVE challenge, and makes a
newly-registered ACTIVE challenge propagate automatically with NO static
per-challenge `docker service create` step. (Historically the master had no such
loop: `SwarmChallengeOrchestrator` only did per-spec start/stop/restart, admin
create just wrote a DB row, and the only reconcile was the legacy validator-side
`NormalValidatorRunner.run_once`. The `install-swarm.sh` `deploy_challenges`
default-path comment now reflects this real behavior.)

### Behavior (idempotent, reconcile-to-registry)

- Each pass reads `registry.list(active_only=True)` and calls
  `orchestrator.start_challenge(spec)` for every ACTIVE challenge. Start is
  invoked **exactly once per challenge** (the reconciler tracks what it has
  deployed), and `start_challenge` is itself idempotent (it reuses an existing
  service), so a fresh master that inherits already-running services converges
  harmlessly.
- A challenge whose status is no longer ACTIVE (DRAFT / INACTIVE / DISABLED, or
  removed from the registry) has its service **stopped** via
  `orchestrator.stop_challenge(slug)` on the next pass.
- DRAFT / INACTIVE / DISABLED challenges are **never** started (belt-and-suspenders:
  the reconciler also re-filters to ACTIVE even if a registry ignores
  `active_only`).
- A start/stop that raises is logged and retried next pass; one failure never
  aborts the whole pass or stops the loop.

### Where it is wired

| Concern | Code |
|---------|------|
| Reconciler + loop + lifespan | `src/base/master/orchestration.py::MasterChallengeReconciler` / `run_registry_reconcile_loop` / `build_master_registry_reconcile_lifespan` |
| Shared spec builder (same shape as the legacy runner) | `src/base/master/docker_orchestrator.py::challenge_spec_from_registry` (also used by `validator/normal_runner.py`); emits `workload_class="service"` |
| Cadence / opt-out | `MasterSettings.registry_reconcile_interval_seconds` (default `60.0`; `<=0` disables — default-on for the master) |
| Wire-up | `master/app_proxy.py::create_proxy_app` (`registry_reconciler` + `registry_reconcile_interval_seconds`, composed via `_combine_lifespans`); constructed in `cli_app/main.py::master_proxy` from the same `orchestrator` the runtime controller uses |

Tests: `tests/unit/test_master_registry_reconciler.py` (faked registry +
orchestrator: start/idempotent/add/deactivate/remove/reactivate, non-ACTIVE never
started, spec parity, start-failure retry, async registry, loop + lifespan).
