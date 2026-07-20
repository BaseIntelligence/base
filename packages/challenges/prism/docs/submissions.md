# Submission Format

PRISM is a **research lab**: the **norm** is to try **new architectures**; the **goal** is to find
**more performant** ones under fair challenge-owned re-exec. PRISM accepts a **two-script** bundle: a
`.zip` archive (or directory snapshot) containing a model `architecture.py` and a training
`training.py`. PRISM fixes the FineWeb-Edu dataset and the evaluation protocol, not the model search
space beyond the Python contract, the AST sandbox, the **dual param ladder** (124M explore /
350M promote), and the resource limits.

The miner owns the model and the training loop. The challenge owns the dataset and the scoring: it
re-executes `training.py` under a forced random init and a fixed seed, records the online loss stream
itself, authors the run manifest, and ignores any value the miner reports.
A single combined module no longer satisfies the contract.
See [Architecture](architecture.md) and [Scoring](scoring.md) for the re-execution and scoring detail.
For the architecture-agnostic **Official Comparison Protocol v1** (scientific multi-axis grade —
held-out primary ranking, honest hooks table, GPU deferred without NVIDIA), multimetric scorecard
annex **v1.1** (`scorecard_id=multimetric.v1.1`), Complete View, and challenge-owned train series
**`prism_train_series.v1`**, see [Official Comparison](official-comparison.md). Multimetric /
Complete View are **published research grade** and do **not** silently replace the emission crown
(held-out primary + bpb secondary).
## The Two-Script Contract

A bundle must contain two **distinct** scripts.

`architecture.py` exposes a factory that returns a `torch.nn.Module`:

```python
def build_model(ctx):
    return MyModel(ctx.vocab_size)
```

It may use any PyTorch structure inside the AST sandbox, the dual param ladder (default
**explore ≤ 124M**; promote pin ≤ **350M**), and the resource limits. Novel architectures are
**expected** (Transformer, looped depth, pure-torch SSM, LightDeepLoop-class modules, …). It must
not read data, open files, touch the network, or reference the dataset.

`training.py` exposes the miner-owned training loop:

```python
def train(ctx):
    model = ctx.build_model()
    # optimizer/schedule, read the locked train split from ctx.data_dir, tokenize,
    # run the loop, handle multi-GPU, write only under ctx.artifacts_dir.
    ...
```

`train(ctx)` owns the optimizer, schedule, dataloading from the read-only locked train split,
tokenization, the multi-GPU strategy, and the loop. It reports progress only through the
challenge-provided logging handle, never as the basis of the score.

### Honest training hooks (challenge-owned score)

| Hook | Required? | Authoritative for score? |
| --- | --- | --- |
| `build_model(ctx) → nn.Module` | Yes | Indirectly (must construct under forced seed / dual param ladder) |
| `train(ctx)` consuming `ctx.iter_train_batches(...)` | Yes for honest online capture | Capture path is challenge-owned; return value ignored |
| Challenge-owned **`prism_train_series.v1`** (online CE/bpb, tokens, wall, **grad_norm**, **clip_events**) | Required when Official pin sets `require_train_series` | **Yes for series residual / fail-closed Official pin** — challenge capture only |
| Optional logs / free-form diagnostics under `artifacts_dir` | Optional | **No** — non-authoritative diagnostics only |
| Miner dashboards / miner-written loss or grad series | Optional cosmetics | **Never** — ignored for grade; does not unblock missing challenge series |
| Miner self-reported bpb / `final_score` / home-rolled manifest | Forbidden as trust root | **Never** — Prism recomputes official metrics |

**Rating honesty (scientific vs emission):** multi-axis Official Comparison / Complete View is the
**scientific miner architecture grade** (published research; multimetric does **not** silently replace
emission). The **emission crown** is Official-like: **held-out / generalization primary**,
prequential bpb **secondary**. Train series densify sample-eff / stability residual and operator
time-flow visibility, and **never sole primary rank** over held-out or recomputed bpb. Official grade
can **fail-closed** if the pin requires series and challenge capture is missing/empty/corrupt.

Wall-clock and miner-timing claims never rank. Full Official Comparison honesty checklist:
[Official Comparison Protocol v1](official-comparison.md#7-training-script-honest-hooks-contract)
and [§17 train series telemetry](official-comparison.md#17-challenge-owned-train-series-telemetry-prism_train_seriesv1).

An optional `prism.yaml` declares the entrypoints and tokenizer:

```yaml
architecture:
  entrypoint: architecture.py
training:
  entrypoint: training.py
tokenizer: gpt2
```

Absent `prism.yaml`, PRISM uses the default entrypoints (`architecture.py`, `training.py`) and symbols
(`build_model`, `train`); when present, declared entrypoints are honored exactly, with no silent
fallback. The two entrypoints must be distinct files: **the single-module re-export idiom no longer
satisfies the contract**.

## PrismContext

Both scripts receive a `PrismContext`:

| Field / method | Meaning |
| --- | --- |
| `vocab_size`, `max_seq_len` | Token-id geometry |
| `max_params` | Hard parameter cap for the stage (**124M explore** provisional; **350M promote**) |
| `seed` | The forced seed (challenge-controlled; the miner cannot change it) |
| `data_dir` | Read-only path to the locked FineWeb-Edu **train** split |
| `artifacts_dir` | The only writable path (rank-0 writes) |
| `device`, `world_size`, `rank`, `local_rank` | Distributed launch geometry |
| `token_budget`, `step_budget` | Compute budget for the run |
| `build_model()` | Builds the model from `architecture.py` |
| `reference_tokenizer(name)` | Loads a pre-staged offline tokenizer (`"gpt2"` or `"llama"`); never touches the network |

The miner does **not** control the dataset content or splits, the seed/init, the scoring, or the
held-out evaluation.

## Locked FineWeb-Edu Data Plane

The dataset is a pinned FineWeb-Edu subset in fixed, disjoint parts:

- `train/` raw text shards, exposed **read-only** at `ctx.data_dir`;
- `val/` and `test/` held-out raw text, **secret** and **never exposed** to the miner script, read only
  by the challenge scorer.

Delivery is a read-only bind mount on the GPU node. The eval container runs with `network=none`,
`HF_HUB_OFFLINE=1`, and `HF_DATASETS_OFFLINE=1`, so there is **no network** during training. The miner
tokenizes raw text from `ctx.data_dir` with its own or a pre-staged reference tokenizer, failing closed
if the locked data is missing rather than fabricating data.

## Multi-GPU Contract

The miner's `training.py` owns multi-GPU scaling. The harness launches
`torchrun --standalone --nnodes=1 --nproc-per-node=<gpu_count>`, exposing `WORLD_SIZE`, `RANK`, and
`LOCAL_RANK`. PRISM is **single-node** only: runs use 1-8 GPUs on one node, and the official scored run
uses `torchrun --standalone --nnodes=1 --nproc-per-node=1` (the `nproc=1` path, since one physical GPU
exists). Requests above 8 GPUs or for multiple nodes are rejected.

A correct `training.py` calls `init_process_group` (nccl on GPU) and `set_device(local_rank)`, wraps the
model with DDP or FSDP and shards data per-rank, does rank-0-only writes, all-reduces reported metrics,
tears down the process group on exit, and also works at `world_size=1`. Correctness is validated off the
single physical GPU with a static contract check and a **gloo** multi-rank test (world size 2 and 4 on
CPU) asserting the loss decreases and parameters stay byte-identical across ranks. True 8-GPU scaling is
an accepted, unverifiable limitation on a one-GPU node.

## Artifact Manifest

The runner writes a challenge-authored `prism_run_manifest.v2.json` from the captured loss stream: the
prequential bits-per-byte score block, the held-out delta and anti-memorization gap, the compute block
(leased `gpu_count`, world size, device, realized parameter count), the run provenance, and the byte
coverage. Any miner-written manifest or reported metric is discarded.

## Minimal Example (default exploration under 124M)

```text
project.zip
  architecture.py   # exposes build_model(ctx)
  training.py       # exposes train(ctx)
  prism.yaml        # optional
```

The container resolves `architecture.py::build_model` and `training.py::train`, forces the seed,
launches torchrun, and captures the online loss itself. **Default exploration shapes under the
124M explore cap** (start here before any 350M promote):

- Transformer: [tiny ~1M-parameter example](../examples/tiny-1m/README.md) (family
  `transformer-tiny-1m`)
- Mamba / pure-torch SSM: [mamba-tiny example](../examples/mamba-tiny/README.md) (family
  `mamba-tiny-1m`; no blocked native `mamba_ssm` for the static lab path)

Package either or both with `scripts/pack_seed_family.py` (shared outer two-script zip shape). Lab
family knobs (param counting shape, batch/LR step flu, pure-torch SSM caveats, multi-GPU primitives)
are documented on each seed README and in the [miner hub](miner/README.md#lab-seed-families).
Novel architectures beyond these seeds are first-class under the AST + ladder caps.

**Day-1 network submit** (production): pack a seed, sign with your hotkey, and
`POST https://chain.joinbase.ai/v1/challenges/prism/submissions` with signature headers. Walkthrough:
[Miner getting started](miner/getting-started.md).

## ZIP Safety Rules

ZIP submissions are extracted defensively: no path traversal, no symlinks, limited file count, limited
total bytes, only approved text or code suffixes. Unsupported or unsafe archives are rejected before
evaluation.
