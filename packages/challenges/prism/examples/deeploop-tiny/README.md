# Tiny ~1M-Parameter Two-Script Example (DeepLoop-class family)

A minimal, valid PRISM v2 submission implementing a **DeepLoop-class** looped residual
decoder under the 124M explore ladder (arXiv 2607.13491 lineage; shared-weight physical
blocks unrolled with residual loop scales). Registered packaging family id:
**`deeploop-tiny-1m`**.

## Layout

```text
examples/deeploop-tiny/
  prism.yaml         # architecture + training entrypoints and tokenizer
  architecture.py    # build_model(ctx); DeepLoopLM only
  training.py        # train(ctx); miner-owned loop
```

## Family knobs (lab operators)

| Knob | DeepLoop tiny (`deeploop-tiny-1m`) |
| --- | --- |
| Architecture family | Shared-weight residual blocks looped L times (DeepLoop-class) over causal MHA + SwiGLU |
| Parameter geometry | `dim=128`, `heads=4`, physical blocks=1, `loops=4`, `mlp_ratio=4`; weight-tied emb/lm_head |
| Cap | ≤ explore **124_000_000** (~1–1.5M realized thrash target) |
| Tokenizer | `prism.yaml` → `gpt2` |
| Step throughput | `LOCAL_BATCH=4`, AdamW `lr=0.004`, grad clip `1.0`; scores compute-normalized |
| Multi-GPU | Single-node ≤8: same static primitives as Imp seeds; works at `world_size=1` |
| Stability | Loop residual scales init small (0.1); prefer modest LR if loss spikes |
| Imp contrast | Vs `transformer-tiny-1m`: same microkernel with **shared params + loop depth** |

Pack:

```bash
uv run python scripts/pack_seed_family.py --family deeploop-tiny-1m --output-dir dist/seed-packages
```

## Submit

See [docs/submissions.md](../../docs/api.md) and the [miner guide](../../../../docs/miner/getting-started.md).
