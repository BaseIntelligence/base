# dense-1b

Reference AutoModel patch for Prism recipe 2.1: a **dense ~975M** transformer
(GQA + RMSNorm + SwiGLU + RoPE + QK-norm) on **1× NVIDIA B200** (~180–192 GB,
TE on, mb≥8, ckpt off, DDP world=1). Explicit env fallbacks: **2×/8× RTX PRO
6000 Blackwell** (~96 GB, TE on, mb≥4) or **4× RTX 5090** (32 GB, TE off,
mb=1).

Fine-grained **MoE at ~1B is a bad default** (tiny expert GEMMs, irregular
routing, no NVFP4 wgrad). MoE remains a miner experiment if you want it —
it is **not** this reference. The retired LoopMoE example is invalid as the
pack you copy.

`min_params` 850M / `max_params` 1B, total unique params, tied embeddings
once. The old **215M** LoopMoE width is also invalid.

This is **miner documentation + example code**. It is not a control-plane
binary, not a scored organizer baseline, and not a live `:28092` flip.

## Submit

| File | Role |
|------|------|
| `automodel.base` | Pin id — must match `GET /v1/recipe` (`automodel@v0.5.0`) |
| `automodel.patch` | Unified diff vs that pin (entry, model, kernels, DDP worker) |
| `prism.toml` | Optional entry pointer |
| `requirements.txt` | Comment-only pin so AutoModel `pyproject.toml` is not installed (debian blinker). TE defaults on 96/180 GB-class. |

Pack the four files at the ZIP root and `POST /v1/submissions` with your
hotkey + `X-Lium-Api-Key`. See [`../../prism.md`](../../prism.md).

Unpacked modules (`entry.py`, `model.py`, `kernels.py`, `ddp_worker.py`)
are the same tree the patch applies under
`nemo_automodel/components/models/dense1b/` — useful for local reading.

`python3 count_params.py` prints `n_params` (must land in 850e6–1e9).

## Contract this example honors

- `build_model(ctx)` returns an `nn.Module`; train consumes
  `ctx["train_stream"]` only (G6 / dual-cap accounting).
- Multi-GPU: rank 0 owns the harness stream and scatters each global batch.
  Rank 0 also writes `dense_1b_ddp/telemetry.json` (loss series + probe
  curve) so the parent `prism_telemetry` / G6 ingest sees DDP/ZeRO workers.
- Default `DENSE1B_PARALLEL=zero1` (`ZeroRedundancyOptimizer`). `fsdp`
  selects FSDP2 (`fully_shard`) when available. `ddp` remains a fallback
  (debug only). Single-GPU B200 stays `world=1`.
- 180 GB-class (`B200` in `gpu_type`, or ≥170 GiB): `DENSE1B_TE=1` default,
  `DENSE1B_MICRO_BATCH≥8`, activation ckpt **off**. 96 GB-class (`gpu_count==2`
  and name contains 6000, or ≥90 GiB): TE on, mb≥4, ckpt off. 32 GB 5090:
  TE off, mb=1, ckpt on.
- `ctx["gpu_count"]` / `ctx["gpu_type"]` / TE `NVFP4BlockScaling` when the
  class exists (consumer Blackwell: `disable_rht=True`,
  `disable_stochastic_rounding=True`).
- Optional env (miner-side, not organizer knobs):
  `DENSE1B_MICRO_BATCH`; `DENSE1B_PARALLEL=zero1|fsdp|ddp`;
  `DENSE1B_TE=0|1`; `DENSE1B_CKPT=0|1`.
- The harness skips `FlopCounterMode` on the full 850M–1B graph (analytic
  6N) so GPU0 is free before `mp.spawn`.

Do not point this example at live `:28092` to flip scoring. Live defaults
stay `PRISM_SCORING_MODE=benchmarks` and `PRISM_ANCHOR_VERSION=0`.
