# PRISM → Lium dispatch (T14)

Master-owned path that admits Prism GPU training/eval work onto **our** Lium
machines. This is **compute only** — not TEE, not constation elevation.

## Trigger (ops)

```bash
# Master (preferred for "our machines")
export BASE_LIUM_TRAINING__ENABLED=true
export BASE_LIUM_TRAINING__API_KEY_FILE=/run/secrets/lium_api_key   # never log
export BASE_LIUM_TRAINING__SSH_PUBLIC_KEY_FILE=/run/secrets/lium_ssh.pub
# Optional spend / concurrency guards (defaults in LiumTrainingSettings):
# BASE_LIUM_TRAINING__MAX_PRICE_PER_HOUR=1.50
# BASE_LIUM_TRAINING__CONCURRENCY_CAP=3
# BASE_LIUM_TRAINING__DAILY_SPEND_CEILING_USD=50

# Optional worker label (default stays base_gpu)
export PRISM_EXECUTION_BACKEND=lium
```

Fail-closed:

- `lium_training.enabled=false` (default) → `try_build_lium_capacity_scheduler` returns `None`; no rentals.
- `enabled=true` without `api_key` / `api_key_file` → hard `build_*` raises `ValueError`; soft `try_build_*` returns `None` so master still boots.

## Code path

1. `cli_app/main.py` — `try_build_lium_capacity_scheduler(settings)` when master starts.
2. `MasterOrchestrationDriver.bridge_pending_work` — Prism GPU units → `lium_scheduler.enqueue(submission_id, job_id)`.
3. `run_once` — `lium_scheduler.tick()` admits FIFO when 1-GPU Blackwell inventory is free.
4. `LiumCapacityScheduler` — provisions via training-locked `LiumClient.for_prism_training` (`lium_capacity.py` + `lium_training_wiring.py`).
5. Worker construct — `PRISM_EXECUTION_BACKEND=lium` / `execution_backend=lium` is always allowed at `PrismWorker` construction (**no** constation bundle). Bundle remains optional API-compat only.

Empty inventory → lease stays `queued` with `reason=capacity_wait` (never a terminal fail for capacity).

## Config keys (`LiumTrainingSettings`)

| Field | Env | Default | Notes |
|-------|-----|---------|-------|
| `enabled` | `BASE_LIUM_TRAINING__ENABLED` | `false` | Master plane off until ops flips |
| `api_key` | `BASE_LIUM_TRAINING__API_KEY` | `null` | Prefer file; never log |
| `api_key_file` | `BASE_LIUM_TRAINING__API_KEY_FILE` | `null` | File contents read lazily |
| `ssh_public_key_file` | `BASE_LIUM_TRAINING__SSH_PUBLIC_KEY_FILE` | `null` | Injected into pods |
| `max_price_per_hour` | `BASE_LIUM_TRAINING__MAX_PRICE_PER_HOUR` | `1.50` | Per-GPU hourly ceiling |
| `max_lifetime_hours` | `BASE_LIUM_TRAINING__MAX_LIFETIME_HOURS` | `4.0` | Pod lifetime guard |
| `concurrency_cap` | `BASE_LIUM_TRAINING__CONCURRENCY_CAP` | `3` | Max concurrent training pods |
| `daily_spend_ceiling_usd` | `BASE_LIUM_TRAINING__DAILY_SPEND_CEILING_USD` | `50.0` | Blocks **new** admissions only |
| `pod_name_prefix` | `BASE_LIUM_TRAINING__POD_NAME_PREFIX` | `prism-train-` | Recover matches this prefix |

Prism worker label: `execution_backend` / `PRISM_EXECUTION_BACKEND` (`base_gpu` default; `lium` allowed bare).

## What this is not

- **Not TEE.** No attestation claim, no constation_ok elevation from selecting `lium`.
- **Not live billable by default.** Tests mock the client; live rentals require explicit ops env + real key.
- Constation modules stay in-tree for the separate ingestion elevation path (todos 19–22).

## T15 handoff (live 1M e2e)

1. Pack `examples/tiny-1m` (or equivalent tiny recipe) as the Prism work unit.
2. Set `BASE_LIVE_PROVIDER_TESTS=1` **and** a real key via `BASE_LIUM_TRAINING__API_KEY_FILE` only in a controlled env.
3. Run master with `BASE_LIUM_TRAINING__ENABLED=true`; confirm bridge enqueue + tick provisions a pod; tear down pods after.
4. Do **not** claim TEE / constation from a successful rental.
5. Capture spend + pod IDs (not API key material) in evidence.

## Tests

```bash
UV_CACHE_DIR=/var/tmp/uv-cache uv run pytest \
  packages/challenges/prism/tests/test_execution_backend_constation_gate.py \
  tests/unit/test_lium_dispatch_unattested.py \
  tests/unit/test_orchestration_lium.py \
  tests/unit/test_lium_training_wiring.py \
  -q
```
