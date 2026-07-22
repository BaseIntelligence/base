# Tiny ~2-4M-Parameter Two-Script Example (Hybrid Attn × SSM family)

A minimal, valid PRISM v2 submission implementing a **hybrid tiny causal attention ×
pure-torch selective SSM** stack under the 124M explore ladder (Hymba/Jamba/Zamba-mini
spirit; arXiv 2411.13676 / 2403.19887 / 2405.16712). Registered packaging family id:
**`hybrid-attn-ssm-tiny-1m`**.

## Layout

```text
examples/hybrid-attn-ssm-tiny/
  prism.yaml
  architecture.py
  training.py
```

## Family knobs (lab operators)

| Knob | Hybrid tiny (`hybrid-attn-ssm-tiny-1m`) |
| --- | --- |
| Architecture family | Pure-torch SelectiveSSM residual + sparse causal MHA every k layers + SwiGLU |
| Parameter geometry | `dim=128`, `layers=3`, `heads=4`, `d_state=16`, `d_conv=4`, `expand=2`, `dt_rank=8`, `attn_every=2`; weight-tied emb/lm_head |
| Cap | ≤ explore **124_000_000** (~2–4M thrash target) |
| Native deps | Pure torch only (no `mamba_ssm` / `flash_attn`) |
| Tokenizer | `gpt2` |
| Step throughput | `LOCAL_BATCH=4`, AdamW `lr=0.0035`, grad clip `1.0` |
| Multi-GPU | Single-node ≤8 Imp-compatible static contract |
| Imp contrast | Interpolates `transformer-tiny-1m` and `mamba-tiny-1m` microkernels |

Pack:

```bash
uv run python scripts/pack_seed_family.py --family hybrid-attn-ssm-tiny-1m --output-dir dist/seed-packages
```

## Submit

See [docs/submissions.md](../../docs/api.md) and the [miner guide](../../../../docs/miner/getting-started.md).
