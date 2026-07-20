# Tiny ~4-12M-Parameter Two-Script Example (DeepSeekMoE-class)

A minimal, valid PRISM v2 submission implementing **fine-grained MoE FFN** with
shared-expert isolation and top-k routing under the 124M explore ladder
(DeepSeekMoE arXiv 2401.06066; DeepSeek-V2 MoE section).

**Honesty:** this is a **mechanism downscale** of DeepSeek-class MoE, **not** full
DeepSeek-V3/V4 / Kimi-K2 multi-hundred-B or T-scale MoE checkpoints. No expert
parallelism or DualPipe is required.

Registered packaging family id: **`ds-moe-tiny-1m`**.

## Layout

```text
examples/ds-moe-tiny/
  prism.yaml
  architecture.py
  training.py
```

## Family knobs (lab operators)

| Knob | DS-MoE tiny (`ds-moe-tiny-1m`) |
| --- | --- |
| Architecture family | Dense causal MHA + fine-grained MoE FFN (1 shared + 8 routed, top_k=2) |
| Parameter geometry | `dim=128`, `heads=4`, `layers=2`, `n_routed=8`, `top_k=2`, `expert_mult=2`; weight-tied emb/lm_head |
| Cap | ≤ explore **124_000_000** (~4–12M **total** params thrash target; activated ≪ total) |
| Native deps | Pure torch only (no EP infra, no flash_attn) |
| Tokenizer | `gpt2` |
| Step throughput | `LOCAL_BATCH=4`, AdamW `lr=0.0035`, grad clip `1.0`; scores compute-normalized |
| Multi-GPU | Single-node ≤8 Imp-compatible static contract |
| Imp contrast | Vs `transformer-tiny-1m`: sparse expert FFN capacity vs dense SwiGLU |

Pack:

```bash
uv run python scripts/pack_seed_family.py --family ds-moe-tiny-1m --output-dir dist/seed-packages
```

## Submit

See [docs/submissions.md](../../docs/submissions.md) and the [miner guide](../../docs/miner/README.md).
