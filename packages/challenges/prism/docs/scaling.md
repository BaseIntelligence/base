# Scaling Evaluation

PRISM scores a single-GPU forced-init re-execution, but the contract is built to scale: the miner owns
multi-GPU execution, and the challenge keeps the score compute-normalized so hardware does not change
the ranking. This covers the execution modes, the single-node multi-GPU contract, the compute budget,
and how multi-GPU correctness is validated on one physical GPU.

## Execution Modes

Both official modes use the challenge-authored `prism_run_manifest.v2.json` contract:

| Mode | Purpose | Dataset target |
| --- | --- | --- |
| `gpu_proxy_eval` | Default official scored re-execution. | FineWeb-Edu `sample-10BT` locked shards. |
| `full_scale_eval` | Larger official scored re-execution. | FineWeb-Edu `sample-10BT` then `sample-100BT` phases. |

Both are score-eligible and run on the locked FineWeb-Edu data, mounted read-only, with `network=none`,
`HF_HUB_OFFLINE=1`, and `HF_DATASETS_OFFLINE=1`, so there is no network during training. The retired
local-CPU smoke mode no longer exists.

## Single-Node Multi-GPU Contract

The miner's `training.py` owns multi-GPU scaling. The harness launches
`torchrun --standalone --nnodes=1 --nproc-per-node=<gpu_count>` and exposes `WORLD_SIZE`, `RANK`, and
`LOCAL_RANK`. PRISM is **single-node** only: runs use 1-8 GPUs on one node, and the official scored run
uses `torchrun --standalone --nnodes=1 --nproc-per-node=1` (the `nproc=1` path, since one physical GPU
exists). Requests above 8 GPUs or for multiple nodes are rejected.

A correct `training.py` calls `init_process_group` (nccl on GPU) and `set_device(local_rank)`, wraps the
model with DDP or FSDP and shards data per-rank, does rank-0-only writes, all-reduces metrics, tears the
process group down on exit, and also works at `world_size=1`.

## Validating Multi-GPU On One GPU

True 8-GPU scaling is an accepted, unverifiable limitation on a one-GPU node. Correctness is validated
three ways:

1. **Static contract** — the AST check verifies the distributed primitives and a rank-0 write guard and
   enforces the single-node bound.
2. **gloo multi-rank test** — a CPU **gloo** run at world size 2 and 4 asserts the loss decreases and
   parameters stay byte-identical across ranks (DDP gradient-sync correctness).
3. **Advisory NCCL `nproc=2`** — an indicative run time-sharing the single GPU, advisory only.

## Compute Budget, Not Wall-Clock

The score is compute-normalized by tokens (and optionally FLOPs), never wall-clock. Wall-clock is only a
safety cap, enforced in layers:

1. a graceful budget at which the runner stops the single-pass loop and scores the partial stream;
2. a hard watchdog that terminates a loop hanging outside the instrumented iterator;
3. an outer docker/broker timeout strictly above the graceful budget plus the watchdog grace.

A faster or larger GPU configuration does not change the ranking; it only changes how much of the budget
the run can use.

## Compute Block In The Manifest

The challenge records a typed, observability-only compute block in `prism_run_manifest.v2.json`: the
leased `gpu_count` (1 for the scored `nproc=1` path), the launch shape (`world_size`, `nproc_per_node`,
`device`), and the realized parameter count of the trained model. It never affects scoring: the
bits-per-byte `final_score` never reads `gpu_count`, so there is no GPU-count reward and no multi-GPU
scaling bonus.

## Reference Studies

- **Loss vs compute** — Kaplan et al., 2020: compare comparable loss trajectories, not one checkpoint.
- **Compute-optimal scaling** — Hoffmann et al., 2022: normalize by tokens/compute.
- **Large-batch dynamics** — McCandlish et al., 2018: scaling the batch across ranks must preserve a
  stable, descending loss.
- **Dataset provenance** — Penedo et al., 2024 (*The FineWeb Datasets*): freeze the revision and shards.

## Scale-eval ladder (research multi-seed + densify)

Architecture fair-eval at larger K / longer context uses the **Official Comparison** heldout-primary
rank and host **Complete View** densify. Product helpers live in
`prism_challenge.evaluator.scale_eval` (see [Official Comparison §11.3](official-comparison.md)):

| Phase | Pin sketch | Densify |
| --- | --- | --- |
| P0 | explore, seq=128, token_budget=500k, seeds K≥3 `(1337,2027,4242)` | host long_ctx + sample_eff on trained_state |
| P1 | `scale_p1_protocol_pin()` — seq **≥256** (target **512**), token_budget **≥1M** (to 2M) | long_ctx T up to 1024 when feasible |
| P2 | promote 350M confirm/revoke (`scale_p2_protocol_pin`) | same multi-axis densify |
| P3 | `full_scale_eval` / sample-100BT readiness (`scale_p3_protocol_pin` + probe) | public K≥3 lock + research annex non-emission |

**Scale ladder (explore → promote → full_scale):** public Official posture requires multi-seed
**K≥3** (default seeds `(1337, 2027, 4242)`). Prior **K=1** cups are **provisional only** (no
architecture-superiority language). Machine map: `scale_ladder_document()` in
`prism_challenge.evaluator.scale_eval`.

**P1 product knobs (VAL-SCALE-006):** `ProtocolPin.seq_len` / `token_budget` pass through
`explore_protocol_pin`, official compare harness, and worker-plane
`PrismSettings.sequence_length` + optional `token_budget` (`PRISM_TOKEN_BUDGET`) into
`PrismContext` via `prism_context_kwargs` / `prism_context_from_protocol_pin`. There is no
hardcoded seq=128-only trap on those paths; Official short-ctx default remains 128/500k when
knobs are unset. Tests: `tests/test_scale_pin_passthrough.py`.

### P3 full_scale / sample-100BT readiness (VAL-SCALE-015)

`full_scale_eval` is already a first-class `ExecutionMode`. Runtime
`execution_mode_targets.full_scale_eval` wires phase-1 FineWeb-Edu **sample-10BT** then
phase-2 **sample-100BT** (100BT token target) without changing the emission rank key.

| Surface | Role |
| --- | --- |
| `probe_full_scale_readiness(...)` | Dry-run mount/manifest readiness for train + val + sample-100BT paths |
| `scale_p3_protocol_pin()` | Matched promote pin (K≥3, seq/budget floors) for full_scale posture |
| `runtime_policy_defaults()["execution_mode_targets"]["full_scale_eval"]` | Dataset subset + token targets |

**Honest dry-run:** the probe does **not** require 100BT spend. When mounts or manifests are
missing it returns **`BLOCKED_with_reason`** (never invents READY scores). When locked mounts
are present it may report **READY** for mode wiring only (`full_scale_train_executed=false`,
`requires_100bt_spend=false`, `emission_changed=false`).

Default path knobs (operator config / env):

* train mount: `base_eval_train_data_dir` / `PRISM_BASE_EVAL_TRAIN_DATA_DIR` (default `/data/fineweb-edu/train`)
* secret val (host scorer only): `base_eval_val_data_dir` / `PRISM_BASE_EVAL_VAL_DATA_DIR`
* phase-2 root: sibling `.../fineweb-edu/sample-100BT` (or pass `phase2_data_dir=` to the probe)

```bash
export UV_CACHE_DIR=/var/tmp/uv-cache PRISM_DOCKER_BACKEND=cli
uv run python -c "from prism_challenge.evaluator.scale_eval import probe_full_scale_readiness; print(probe_full_scale_readiness(dry_run=True).as_dict())"
uv run pytest tests/test_scale_fullscale_readiness.py -q
```

### Research protocol annex (non-emission, VAL-SCALE-017)

Additive machine object `research_protocol_annex()` (`schema=prism_research_protocol_annex.v1`)
binds Complete View / multimetric identities and sets **`non_emission=true`**,
**`emission_weight_crown=false`**, **`silent_emission_rewrite=false`**. Multimetric and Complete
View stay **published research grade**. A future protocol v2 emission fold needs an explicit
product feature; this annex never silently rewrites the emission crown.

Emission leaderboard remains heldout primary + bpb secondary. Complete View / multimetric are
**published research grade**, not silent emission crowns. Wall-clock never ranks. Prefer host
densify before new GPU trains. Prism ships **no** tee package (provider trust + IMAGE_PIN only).
