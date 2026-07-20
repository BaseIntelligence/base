# Tiny ~3-8M-Parameter Two-Script Example (MLA / DeepSeek-class)

A minimal, valid PRISM v2 submission implementing **Multi-Head Latent Attention (MLA)**
under the 124M explore ladder (DeepSeek-V2 arXiv 2405.04434 lineage; pure torch only).

**Honesty:** this is a **mechanism downscale** of DeepSeek-class MLA, **not** full
DeepSeek-V3/V4 frontier weights or multi-hundred-B checkpoints.

Registered packaging family id: **`mla-tiny-1m`**.

## Layout

```text
examples/mla-tiny/
  prism.yaml
  architecture.py
  training.py
```

## Family knobs (lab operators)

| Knob | MLA tiny (`mla-tiny-1m`) |
| --- | --- |
| Architecture family | Joint low-rank KV latent + multi-head up-proj + decoupled RoPE + SwiGLU |
| Parameter geometry | `dim=128`, `heads=4`, `layers=3`, `kv_lora_rank=32`, `rope_dim=16`, `mlp_ratio=4`; weight-tied emb/lm_head |
| Cap | ≤ explore **124_000_000** (~3–8M thrash target) |
| Native deps | Pure torch only (no flash_attn / FlashMLA / cpp_extension) |
| Tokenizer | `gpt2` |
| Step throughput | `LOCAL_BATCH=4`, AdamW `lr=0.004`, grad clip `1.0`; scores compute-normalized |
| Multi-GPU | Single-node ≤8 Imp-compatible static contract |
| Imp contrast | Vs `transformer-tiny-1m`: same residual outer, latent KV path instead of full K/V |

Pack:

```bash
uv run python scripts/pack_seed_family.py --family mla-tiny-1m --output-dir dist/seed-packages
```

## Submit

See [docs/submissions.md](../../docs/submissions.md) and the [miner guide](../../docs/miner/README.md).
