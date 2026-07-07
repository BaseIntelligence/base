# Local mission harness (cross-repo end-to-end)

A thin, config-driven launcher that stands up the **whole compute plane locally on loopback ports
3100-3199 with no GPU** and runs six operator-observable drills (all via HTTP/CLI only). It is a
test/mission harness, **not a production path**: it uses a static mock metagraph and the prism
repo's own CPU re-exec seam. Production still uses `base master proxy` with a real subtensor
metagraph and the real GPU broker.

## What it stands up

| Component | Script | Repo |
| --- | --- | --- |
| base master (worker plane ON, validator plane ON, orchestration loop) | `base/scripts/mission/mission_master.py` | base |
| prism service (worker plane + admission gate + CPU re-exec test mode) | `prism/scripts/mission/mission_prism.py` | prism |
| worker agents (base worker plane; executor = prism CPU re-exec) | `prism/scripts/mission/mission_worker.py` | prism |
| stub gpu validator (audits disputes by deterministic replay) | `prism/scripts/mission/mission_validator.py` | prism |
| orchestrator + drills + PID teardown | `prism/scripts/mission/launch.py` | prism |

* **Mock metagraph**: five owner (miner) hotkeys (`//MissionAlice…//MissionErin`, no permit) plus a
  validator hotkey (`//MissionValidator1`, with a validator permit), seeded statically so signed
  worker/validator requests authenticate with no chain.
* **CPU re-exec seam as explicit config**: prism runs with
  `worker_plane.cpu_reexec_test_mode: true`, which installs
  `prism_challenge.evaluator.mock_reexec.cpu_reexec_run` over `DockerExecutor.run` and uses a tiny
  deterministic `PrismContext` (vocab 64 / seq 16 / step budget 24). The launcher never touches any
  HTTP surface; all drill observations are black-box.
* **Two+ workers on DISTINCT owner hotkeys**: each worker's executor runs the real CPU re-exec,
  normalizes the volatile `compute` timing fields, and signs a tier-0 ExecutionProof, so honest
  replicas of the same submission converge on one `manifest_sha256`.

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

The dispute lifecycle (VAL-CROSS-011) is reconstructable after the fact from the same surfaces:
the fault on `GET /v1/workers` (and `base worker status`) references the disputed unit id, and the
prism submission status shows it never finalized to a live-ranked score.

## Ports

* master: `127.0.0.1:3110`
* prism: `127.0.0.1:3120`

Both are inside the mission-allowed range 3100-3199. No component listens outside that range.
