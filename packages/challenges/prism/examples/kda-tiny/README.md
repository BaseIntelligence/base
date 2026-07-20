# Tiny ~3-10M-Parameter Two-Script Example (Kimi Delta Attention / KDA)

A minimal, valid PRISM v2 submission implementing **Kimi Delta Attention (KDA)**
under the 124M explore ladder (Kimi Linear arXiv 2510.26692; pure sequential scan).

**Honesty:** this is a **mechanism downscale** of Kimi Linear / KDA-class ideas,
**not** full Kimi Linear-48B, Kimi K2, or any production "K3" multi-hundred-B model.
No flash-linear-attention / FLA kernel is required for correctness.

Registered packaging family id: **`kda-tiny-1m`**.

## Layout

```text
examples/kda-tiny/
  prism.yaml
  architecture.py
  training.py
```

## Family knobs (lab operators)

| Knob | KDA tiny (`kda-tiny-1m`) |
| --- | --- |
| Architecture family | Sequential Kimi Delta Attention with **channel-wise** forget gates + SwiGLU |
| Parameter geometry | `dim=128`, `heads=4`, `layers=3`, `d_state=48`, `d_conv=4`, `mlp_ratio=4`; weight-tied emb/lm_head |
| Cap | ≤ explore **124_000_000** (~3–10M thrash target) |
| Native deps | Pure torch only (no FLA / flash_linear_attn / mamba_ssm) |
| Tokenizer | `gpt2` |
| Step throughput | `LOCAL_BATCH=4`, AdamW `lr=0.003`, grad clip `1.0`; sequential scan is slower/token (wall-clock never ranks) |
| Multi-GPU | Single-node ≤8 Imp-compatible static contract |
| vs gated-delta-tiny | Channel-wise forget + learned key→state map + explicit Kimi citation path (not a rename) |
| Imp contrast | Vs `mamba-tiny-1m`: delta+channel gate not S6 diagonal A |

Pack:

```bash
uv run python scripts/pack_seed_family.py --family kda-tiny-1m --output-dir dist/seed-packages
```

## Submit

See [docs/submissions.md](../../docs/submissions.md) and the [miner guide](../../docs/miner/README.md).
