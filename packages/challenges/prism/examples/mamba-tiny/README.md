# Tiny ~1M-Parameter Two-Script Example (Mamba / SSM family)

A minimal, valid PRISM v2 submission and the **default Mamba/SSM exploration shape under the 124M
explore ladder**: a weight-tied ~1M-parameter pure-PyTorch selective state-space (Mamba-style) LM
under the two-script contract. Registered packaging family id: **`mamba-tiny-1m`**.

This seed is intentionally free of the blocked native `mamba_ssm` C++/CUDA stack. Static lab
gates and scavenged re-exec paths only require torch + allowlisted imports.

## Layout

```text
examples/mamba-tiny/
  prism.yaml         # declares the architecture + training entrypoints and the tokenizer
  architecture.py    # exposes build_model(ctx); pure-torch SelectiveSSM stack only
  training.py        # exposes train(ctx); the miner-owned loop
```

- `architecture.py` exposes `build_model(ctx)` and is pure: it never reads data, opens files, or
  touches the network, and it never imports `mamba_ssm` / `torch.utils.cpp_extension`.
- `training.py` exposes `train(ctx)`: same multi-GPU static surface as the Transformer seed
  (`init_process_group`, DDP, `DistributedSampler` marker, rank-0 save, teardown).

## Family knobs vs transformer-tiny-1m (lab operators)

| Knob | Mamba tiny (`mamba-tiny-1m`) | Transformer tiny (`transformer-tiny-1m`) |
| --- | --- | --- |
| Architecture family | Pure-torch selective SSM (S6-style scan + depthwise conv + gated residual) | Decoder transformer (RMSNorm + causal MHA + SwiGLU) |
| Parameter geometry | `dim=128`, `layers=2`, `d_state=16`, `d_conv=4`, `expand=2`, `dt_rank=8`; count is **realized** `nn.Module` params under forced-seed `build_model` (weight tying) | `dim=128`, `heads=4`, `layers=2`, `mlp_ratio=4`; same counting surface |
| Cap | ≤ explore **124_000_000** (promote 350M out of scope for ~1M seed) | ≤ explore **124_000_000** |
| Native deps (static lab path) | **None** — pure PyTorch scan; do not require `mamba_ssm` | Pure torch only |
| Tokenizer | `prism.yaml` → `gpt2` (offline) | `gpt2` |
| Step throughput | `LOCAL_BATCH=4`, AdamW `lr=0.003` (slightly lower for sequential scan stability), grad clip `1.0`; ranking is compute-normalized | `LOCAL_BATCH=4`, AdamW `lr=0.005` |
| Multi-GPU | Single-node ≤8: same static primitives as Transformer; works at `world_size=1` | Same |
| Stability caveats | Sequential purge scan is slower/token than fused CUDA kernels; prefer shorter `seq_len` / modest LR if loss spikes; stay under explore 124M | Prefer small LR/batch if loss spikes |
| Packaging | Shared harness `scripts/pack_seed_family.py --family mamba-tiny-1m` | `--family transformer-tiny-1m` |

Pack a submit-shaped zip with the shared harness (same outer two-script shape as Transformer):

```bash
uv run python scripts/pack_seed_family.py --family mamba-tiny-1m --output-dir dist/seed-packages
# dual-family wired shape for lab harness:
uv run python scripts/pack_seed_family.py --output-dir dist/seed-packages --json
```

## How It Is Scored

Same as Transformer: challenge re-executes `train(ctx)` under forced random init on locked
FineWeb-Edu, captures single-pass online loss, ranks **emission** held-out primary + prequential bpb
secondary. Miner return values and private manifests are ignored.

## Submit

Submit as a `.zip` bundle of this directory through the public route (when enabled) or the BASE
proxy. See [docs/submissions.md](../../docs/submissions.md) and the
[miner guide](../../docs/miner/README.md#lab-seed-families).
