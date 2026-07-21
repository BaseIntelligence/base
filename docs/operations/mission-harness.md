# Local mission harness (advanced / non-day-1)

> **Not miner day-1.** Miners should start at [docs/miner/README.md](../miner/README.md) and
> [docs/challenges.md](../challenges.md) against **https://chain.joinbase.ai**. This page is an
> optional **local test/mission harness** for cross-area drills. Production shipping topology is
> master-embed Compose + weight-only validators — not this multi-process mission launcher.

A thin, config-driven launcher that stands up the **whole compute plane locally on loopback ports
3100-3199 with no GPU** and runs six operator-observable drills (HTTP/CLI only). It is a
test/mission harness, **not a production path**: it uses a static mock metagraph and Prism's
CPU re-exec seam, whereas production uses `base master proxy` with a real subtensor
metagraph and provider-backed paths as configured.

## What it stands up

| Component | Script | Repo |
| --- | --- | --- |
| base master (worker plane ON, validator plane ON, orchestration loop) | `base/scripts/mission/mission_master.py` | base |
| prism service (worker plane + admission gate + CPU re-exec test mode) | `prism/scripts/mission/mission_prism.py` | prism |
| worker agents (base worker plane; executor = prism CPU re-exec) | `prism/scripts/mission/mission_worker.py` | prism |
| stub gpu validator (audits disputes by deterministic replay) | `prism/scripts/mission/mission_validator.py` | prism |
| orchestrator + drills + PID teardown | `prism/scripts/mission/launch.py` | prism |

* **Mock metagraph**: five owner (miner) hotkeys (`//MissionAlice…//MissionErin`, no permit) plus a
  validator hotkey (`//MissionValidator1`, with a permit), seeded statically so signed
  worker/validator requests authenticate with no chain.
* **CPU re-exec seam as explicit config**: prism runs with
  `worker_plane.cpu_reexec_test_mode: true`, installing
  `prism_challenge.evaluator.mock_reexec.cpu_reexec_run` over `DockerExecutor.run` with a tiny
  deterministic `PrismContext` (vocab 64 / seq 16 / step budget 24). All drill observations are
  black-box.
* **Two+ workers on DISTINCT owner hotkeys**: each worker's executor runs the real CPU re-exec,
  normalizes the volatile `compute` timing fields, and signs a tier-0 ExecutionProof, so honest
  replicas of one submission converge on a single `manifest_sha256`.

## Virtualenv (important)

The current base source and the current prism source must share ONE interpreter that also has
`torch`. The prism virtualenv has torch + the current prism source, but a *stale* installed `base`;
run everything through the prism venv with the current base source shadowed onto `PYTHONPATH`:

```bash
cd <repo-root>            # the dir containing both base/ and prism/
export PYTHONPATH="$PWD/base/src:$PWD/prism/src"
prism/.venv/bin/python prism/scripts/mission/launch.py            # all drills
prism/.venv/bin/python prism/scripts/mission/launch.py --only 1,4 # a subset
```

The launcher prints the PID of every spawned process and, in a `finally`, **kills them all by
PID** so nothing is left listening.

## Drills (observable via API/CLI only)

1. **VAL-CROSS-001** full pipeline: a submission from a hotkey with an active worker → exactly one
   gpu work unit on prism `/internal/v1/work_units` → assigned to 2 distinct-owner workers → both
   post ExecutionProofs → reconciliation finds equal `manifest_sha256`, accepts, forwards one
   result → prism records a score (`GET /v1/submissions/{id}`).
2. **VAL-CROSS-002** self-eval exclusion under scarcity: with exactly 2 active workers where one is
   owned by the submitter H, H's unit is never assigned to H's own worker (it degrades to the other
   worker or holds pending).
3. **VAL-CROSS-003** divergence: one worker corrupts its manifest → reconciliation disputes the
   unit → a `<uid>:audit` unit (`required_capability=gpu`, executor kind `validator`) is created and
   assigned to the stub validator → the validator's deterministic replay yields the authoritative
   hash → the diverging worker gets a `worker_faults` row visible in `GET /v1/workers`, and the
   submission carries no finalized score.
4. **VAL-CROSS-004** admission gate: `POST /v1/submissions` from a hotkey with no active worker →
   403 `NO_ACTIVE_WORKER`; after a worker for that hotkey registers + heartbeats
   (`GET /v1/workers/active?hotkey=` returns ≥1) the resubmission (fresh nonce) is accepted.
6. **VAL-CROSS-009** fleet agreement: `GET /v1/workers` and `base worker status` (pointed at the
   same master) report the same worker ids, owners, providers, statuses, and fault counts.
11. **VAL-CROSS-011** dispute lifecycle discoverable via APIs alone: after the divergence drill, the
   whole dispute → audit → invalidation → fault chain is reconstructed from operator surfaces with
   **zero DB/file reads**:
   * **(a) disputed state + (b) audit unit** come from the signed `GET /v1/workers/units` (the
     master unit-status read surface): the primary unit reads `status: "disputed"` with its replicas
     (worker id, owner hotkey, posted `manifest_sha256`, proof presence), and its `audit` block
     names the linked audit unit id, `executor_kind: "validator"`, and terminal `outcome`
     (`pending` / `passed` / `mismatch-resolved`).
   * **(c) invalidation** comes from prism `GET /v1/submissions/{id}` (never a live-ranked
     `completed` score).
   * **(d) fault** is visible on both `GET /v1/workers` and `base worker status`.

The dispute-lifecycle reconstruction (VAL-CROSS-011) runs at the end of drill 3 (it needs a live
dispute), so selecting drill `3` also exercises it.

`GET /v1/workers/units` is a signed read-only surface authenticated exactly like `GET /v1/workers`
(a registered worker OR an eligible validator signed request — NOT the admission internal bridge
bearer). With `compute.worker_plane_enabled` off its router is unmounted (404).

## Ports

* master: `127.0.0.1:3110`
* prism: `127.0.0.1:3120`

Both are inside the mission-allowed range 3100-3199. No component listens outside that range.

## Legacy regression (flags OFF) — exact verification procedure (VAL-CROSS-006)

This reproducible procedure proves that with ALL new flags OFF the compute plane behaves
byte-for-byte as pre-mission. It has two parts: (a) both default suites green fully offline, and
(b) a legacy validator-flow smoke.

Prerequisites: the test PostgreSQL is up on 15433 (`services.yaml` `test-postgres`) and both venvs
are synced (`services.yaml` `install`). No credentials are needed; the procedure UNSETS
`LIUM_API_KEY` / `TARGON_API_KEY` / `BASE_LIVE_PROVIDER_TESTS` and leaves
`BASE_COMPUTE__WORKER_PLANE_ENABLED` unset (flags OFF).

### Offline egress guard

`base/scripts/mission/no_external_egress.py` is a pytest plugin (`-p no_external_egress`) that blocks
any DNS resolution or socket connect to a non-loopback host (raising an `OSError` subclass) while
leaving loopback (test PostgreSQL on 127.0.0.1:15433, in-process stubs) and AF_UNIX sockets working.
It installs from `pytest_configure` (before collection), so a module that probes the network at
import time is guarded and simply SKIPS — "zero real egress to lium.io / api.targon.com". Unit-tested
by `base/tests/unit/test_no_external_egress.py`.

### (a) Both default suites, offline, flags OFF

base (with the test PostgreSQL on 15433):

```bash
cd base
env -u LIUM_API_KEY -u TARGON_API_KEY -u BASE_LIVE_PROVIDER_TESTS -u BASE_COMPUTE__WORKER_PLANE_ENABLED \
  PYTHONPATH="$PWD/scripts/mission" \
  BASE_TEST_DATABASE_URL=postgresql+asyncpg://base:base@localhost:15433/base_test \
  uv run pytest -p no_external_egress --cov=base --cov-report=term-missing --cov-fail-under=80 \
  -q -p no:cacheprovider
```

Expected: green, e.g. `1543 passed`, `Total coverage: 88.55%` (≥ 80% gate). No base test needs the
network; the live Lium/Targon E2E is a standalone gated script (`base/scripts/live_lium_e2e.py`,
`BASE_LIVE_PROVIDER_TESTS=1`) not collected by the default suite.

prism (the same egress guard, from the base scripts dir; `distributed_gloo` deselected as non-gating):

```bash
cd prism
env -u LIUM_API_KEY -u TARGON_API_KEY -u BASE_LIVE_PROVIDER_TESTS -u BASE_COMPUTE__WORKER_PLANE_ENABLED \
  PYTHONPATH="/root/prism-compute-plane/base/scripts/mission" \
  uv run pytest -p no_external_egress -m "not distributed_gloo" --cov=prism_challenge \
  --cov-report=term-missing --cov-fail-under=80 -q -p no:cacheprovider
```

Expected: green, e.g. `892 passed, 9 skipped, 7 deselected`, `Total coverage: 83.07%` (≥ 80% gate).
The `7 deselected` are the non-gating `distributed_gloo` tests; the `9 skipped` are harness/dataset
tests whose prep step needs to stage a tokenizer/dataset over the network, which the egress guard
disables — exactly the offline behaviour.

### (b) Legacy validator-flow smoke

Stands up the SAME local deployment with flags OFF (base master `worker_plane_enabled: false`, prism
`worker_plane_enabled: false`) and drives one legacy submission end-to-end. Scripts:
`prism/scripts/mission/legacy_smoke.py` (orchestrator) + `mission_legacy_validator.py` (a real
`validator_dispatch` executor) + the flags-OFF configs of `mission_master.py` / `mission_prism.py`.
It prints every PID and kills them all by PID in a `finally`.

```bash
cd /root/prism-compute-plane           # dir containing base/ and prism/
export PYTHONPATH="$PWD/base/src:$PWD/prism/src"
prism/.venv/bin/python prism/scripts/mission/legacy_smoke.py
```

Expected: `OVERALL: PASS`. The smoke asserts, via HTTP + a read of the master/prism databases:

1. the submission is ACCEPTED with no `NO_ACTIVE_WORKER` 403 (admission gate off, flags OFF);
2. prism exposes exactly ONE gpu work unit for it (`GET /internal/v1/work_units`);
3. the base master assigns that unit to the VALIDATOR
   (`work_assignments.assigned_validator_hotkey` == the validator hotkey, `required_capability=gpu`)
   and NEVER to a worker: `worker_registrations` and `worker_assignments` are empty and
   `GET /v1/workers` shows zero workers — no worker-plane rows/side effects;
4. the validator executes it via the real `validator_dispatch` path
   (`mission_legacy_validator.py` logs `pulled + executing prism unit … via validator_dispatch`,
   `pulled=1 executed=1`) and prism records a score
   (`GET /v1/submissions/{id}` → `completed` with a `final_score`).

`legacy_smoke.py` uses ports 3112 (master) / 3122 (prism), inside 3100-3199, distinct from the
worker-plane drills' 3110/3120 so the two harnesses never collide.
