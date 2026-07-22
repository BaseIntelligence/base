> **API truth is OpenAPI** (`https://chain.joinbase.ai/challenges/prism/openapi.json`, `/docs`).
> Day-1 miners: repo-root [`docs/miner/getting-started.md`](../../../../docs/miner/getting-started.md).
> This page is a short product pin note, not a route dump.

# Validator Guide

## Purpose

PRISM lets validators operate an "ability to learn" challenge for BASE: accept signed miner
submissions, run the static sandbox and **deterministic admission**, re-execute the miner's training
loop under a forced random init on locked data (or ingest an `ExternalResultEnvelope` when the worker
plane is on), compute the prequential bits-per-byte score, push **raw weights** to the BASE master, and
let validators submit the master-aggregated vector with their own wallets.

## Responsibilities

- accept only signed submissions and enforce replay protection and size limits;
- keep evaluation isolated from the host (broker-backed containers, `network=none`);
- keep the locked `val`/`test` splits secret and never expose them to a miner script;
- force the seed and deterministic flags so runs reproduce;
- compute the score from the challenge-owned capture, never trusting miner-reported numbers;
- protect shared BASE and broker tokens (there is **no** LLM gateway token);
- monitor scoring, rejections, failures, and raw-weight push / exported inventory weights.

## Evaluation Lifecycle

1. A miner submits a signed two-script bundle; PRISM validates the hotkey, timestamp, nonce, and size.
2. The bundle is resolved into the two-script contract and inspected by the AST sandbox.
3. The forced-seed `build_model` instantiation enforces the dual param ladder (124M explore / 350M promote).
4. The multi-GPU static contract and single-node bound are checked.
5. **Deterministic admission** runs (source similarity + anti-cheat). Exact duplicates and quarantine-band
   similarity **reject** terminally; there is no held-for-review path and no LLM hard gate.
6. The challenge re-executes `training.py` under a forced random init on the locked train split and
   captures the single-pass online loss itself (broker/container or worker-plane path).

PRISM lets validators operate an "ability to learn" challenge for BASE: accept signed miner
submissions, run the static sandbox and **deterministic admission**, re-execute the miner's training
loop under a forced random init on locked data (or ingest an `ExternalResultEnvelope` when the worker
plane is on), compute the prequential bits-per-byte score, push **raw weights** to the BASE master, and
let validators submit the master-aggregated vector with their own wallets.

- accept only signed submissions and enforce replay protection and size limits;
- keep evaluation isolated from the host (broker-backed containers, `network=none`);
- keep the locked `val`/`test` splits secret and never expose them to a miner script;
- force the seed and deterministic flags so runs reproduce;
- compute the score from the challenge-owned capture, never trusting miner-reported numbers;
- protect shared BASE and broker tokens (there is **no** LLM gateway token);
- monitor scoring, rejections, failures, and raw-weight push / exported inventory weights.

1. A miner submits a signed two-script bundle; PRISM validates the hotkey, timestamp, nonce, and size.
2. The bundle is resolved into the two-script contract and inspected by the AST sandbox.
3. The forced-seed `build_model` instantiation enforces the dual param ladder (124M explore / 350M promote).
4. The multi-GPU static contract and single-node bound are checked.
5. **Deterministic admission** runs (source similarity + anti-cheat). Exact duplicates and quarantine-band
   similarity **reject** terminally; there is no held-for-review path and no LLM hard gate.
6. The challenge re-executes `training.py` under a forced random init on the locked train split and
   captures the single-pass online loss itself (broker/container or worker-plane path).
7. PRISM computes the prequential bits-per-byte score, the held-out delta tie-breaker, and the
   anti-memorization gap, and writes `prism_run_manifest.v2.json`.
8. Scores persist; the leaderboard ranks by `final_score`; PRISM raw-weight push feeds master
   aggregation; on-chain `set_weights` remains validator-owned.

Settings use the `PRISM_` prefix (and compatible `CHALLENGE_` values where declared) to bootstrap the
process and provide fallback defaults. Runtime policy is SQL-first, and official runtime config fails
closed when an active SQL value is invalid:

```text
SQL active value → env/Pydantic default → schema default
```

| Key | Policy area |
| --- | --- |
| `reward_pools` | Weight-split row (compat); raw emission shares and inventory weights normalize per hotkey from `final_score`. |
| `score_weights` | Score-weight row (compat); live primary score is prequential bits-per-byte. |
| `benchmark_weights` | Benchmark-mix row (compat); not part of the live bits-per-byte score. |
| `duplicate_thresholds` | Source/graph/quarantine/static-reject thresholds (quarantine → terminal reject). |
| `gpu_policy` | Max GPU count, fixed GPU count, GPU type, fixed-profile flag. |
| `dataset_configs` | Locked FineWeb-Edu sample count, frozen revision, split names. |
| `execution_mode_targets` | The `gpu_proxy_eval` and `full_scale_eval` token/GPU targets. |
| `artifact_limits` | Code/artifact size limits plus the required `prism_run_manifest.v2.json` name. |
| `sandbox_limits` | Docker, CPU, memory, PID, timeout, network, read-only limits. |
| `diagnostics_thresholds` | Activation, gradient, attention, representation health thresholds. |
| `loss_comparability_policy` | Comparable-loss requirements and byte-normalized fallback. |

The former `llm_review_policy` SQL key and LLM hard-gate settings are **removed**. Residual
gateway/review configuration fails closed at load.

Rows are audited (`config_key`, `value_json`, `schema_version`, `updated_by`, `updated_at`,
`effective_from`, `enabled`); the active row per key is the newest enabled row whose `effective_from`
has arrived.

| Setting | Purpose |
| --- | --- |
| `PRISM_DATABASE_URL` | Persistent SQLite storage location. |
| `PRISM_SHARED_TOKEN` / `PRISM_SHARED_TOKEN_FILE` | BASE internal-call token (prefer file delivery). |
| `PRISM_PUBLIC_SUBMISSIONS_ENABLED` | Enables the direct public submission route. |
| `PRISM_SIGNATURE_TTL_SECONDS` | Replay-protection timestamp window. |
| `PRISM_EPOCH_SECONDS` | Scoring epoch length. |
| `PRISM_MAX_CODE_BYTES` | Maximum submission size. |
| `PRISM_MAX_PARAMETERS` | Explore-stage hard parameter cap (default 124M); promote pin uses 350M. |
| `PRISM_BASE_EVAL_IMAGE` | Digest-pinned `prism-evaluator` image (sentencepiece + offline tiktoken). |
| `PRISM_BASE_EVAL_DATA_DIR` | Read-only locked FineWeb-Edu **train** mount. |
| `PRISM_BASE_EVAL_VAL_DATA_DIR` | Secret held-out **val** split (scorer-only; never mounted into eval). |
| `PRISM_BASE_EVAL_MAX_GPU_COUNT` | Max GPU count (default and hard max 8). |
| `PRISM_BASE_EVAL_GPU_COUNT` | Scored GPU count (default 1; the `nproc=1` path). |
| `PRISM_DISTRIBUTED_CONTRACT_POLICY` | `reject` / `flag` / `off` for the multi-GPU contract. |
| `PRISM_COMBINED_MODE` | API + in-process queue drain in one long-lived Compose service. |

**Removed** (do not configure; residual names fail closed):

- `PRISM_LLM_REVIEW_ENABLED`
- `PRISM_LLM_GATEWAY_URL` / `BASE_LLM_GATEWAY_URL`
- `PRISM_GATEWAY_TOKEN_FILE` / `BASE_GATEWAY_TOKEN_FILE`

Use secret files for the shared and broker tokens only.

## FineWeb-Edu And Execution Modes

| Mode | Operator use | Dataset target |
| --- | --- | --- |
| `gpu_proxy_eval` | Default official scored re-execution. | FineWeb-Edu `sample-10BT` locked shards. |
| `full_scale_eval` | Larger official scored re-execution. | FineWeb-Edu `sample-10BT` then `sample-100BT` phases. |

Both modes are score-eligible and run on the locked, read-only FineWeb-Edu data with `network=none`. The
retired local-CPU smoke mode is gone (except explicit CPU re-exec test mode). Official scoring uses a
fixed `gpu_policy` profile: the max is 8 GPUs and the scored run uses 1 GPU
(`torchrun --standalone --nnodes=1 --nproc-per-node=1`); PRISM is single-node only.

```http
POST /v1/submissions
GET  /v1/submissions/{submission_id}
GET  /v1/leaderboard
GET  /v1/architectures
GET  /v1/architectures/{architecture_id}/variants
GET  /v1/epochs/current
GET  /health
GET  /version
GET  /internal/v1/get_weights      # Authorization: Bearer <shared-token>, X-Base-Challenge-Slug: prism
POST /internal/v1/work_units/result  # ExternalResultEnvelope only when worker plane enabled
```

`get_weights` returns one normalized weight per hotkey (best submission per hotkey) for inventory.
Production emissions flow as authenticated **raw-weight push** → master aggregation → validator
`set_weights`. The architecture auto-report route is removed.

PRISM uses the static AST sandbox, the forced-seed parameter cap, the multi-GPU static contract, and
**deterministic** similarity/anti-cheat checks. A reject from any static gate or the similarity band is
terminal before any GPU work. A borderline duplicate is folded into a terminal rejection at ingress:
there is no operator hold-resolution surface and no LLM hard gate.

## Provider Trust And IMAGE_PIN

- Production scoring does **not** require Prism TEE evidence; operators trust **Lium/Targon** providers.
- Pin the evaluator/worker image via `worker_plane.pinned_image_digest` (**IMAGE_PIN**, max tier **1**).
- Do not treat inventory probes or opaque quote fields as elevated-tier proof.
- **REAL-PROVIDER TEE** is retired as a Prism product goal (historical lab labels may still say BLOCKED).

PRISM installs Base **v3.1.2** from the public wheel pin in `pyproject.toml`. Keep installs locked to
that immutable artifact; do not vendor a parallel SDK.

**Setup:** persistent SQLite storage; shared-token delivery via files or a secret manager; the broker,
digest-pinned evaluator image, and read-only locked-data mounts; submission size and parameter limits;
combined-mode Compose service when deploying under the master; then run targeted config/scoring suites
and submit a known-safe bundle to confirm the leaderboard and raw-weight / `get_weights` inventory.

**Operation and security:** require hotkey signatures and short replay windows; keep the `val`/`test`
splits secret and out of any miner-visible path, fixture, or log; keep the eval container on
`network=none` with the rootfs read-only except `artifacts_dir`; keep broker and BASE tokens out of
logs; never reintroduce gateway tokens; monitor rejected, failed, and completed submissions; and
confirm chain submission remains validator-owned only.
