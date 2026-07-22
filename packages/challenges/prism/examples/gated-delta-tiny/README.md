# Tiny ~1-3M-Parameter Two-Script Example (Gated Delta family)

A minimal, valid PRISM v2 submission implementing a **gated delta-rule linear recurrence**
under the 124M explore ladder (DeltaNet arXiv 2406.06484 class; pure sequential scan, no
fused linear-attention kernels). Registered packaging family id: **`gated-delta-tiny-1m`**.

## Layout

```text
examples/gated-delta-tiny/
  prism.yaml
  architecture.py
  training.py
```

## Family knobs (lab operators)

| Knob | Gated-delta tiny (`gated-delta-tiny-1m`) |
| --- | --- |
| Architecture family | Sequential gated delta-rule recurrence + depthwise conv + SwiGLU |
| Parameter geometry | `dim=128`, `heads=4`, `layers=2`, `d_state=32`, `d_conv=4`, `mlp_ratio=4`; weight-tied emb/lm_head |
| Cap | ≤ explore **124_000_000** (~1.5–3M thrash target) |
| Native deps | Pure torch only (no flash_attn / flash_linear_attn / mamba_ssm) |
| Tokenizer | `gpt2` |
| Step throughput | `LOCAL_BATCH=4`, AdamW `lr=0.003`, grad clip `1.0`; sequential scan is slower/token (wall-clock never ranks) |
| Multi-GPU | Single-node ≤8 Imp-compatible static contract |
| Imp contrast | Vs `mamba-tiny-1m`: delta/gated-delta write rule vs S6 diagonal SSM |

Pack:

```bash
uv run python scripts/pack_seed_family.py --family gated-delta-tiny-1m --output-dir dist/seed-packages
```

## Submit

See [docs/submissions.md](../../docs/api.md) and the [miner guide](../../../../docs/miner/getting-started.md).
