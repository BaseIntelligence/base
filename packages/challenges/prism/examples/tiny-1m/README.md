# Tiny ~1M-Parameter Two-Script Example (Transformer family)

A minimal, valid PRISM v2 submission and the **default Transformer exploration shape under the
124M explore ladder**: a weight-tied ~1.05M-parameter decoder transformer split into the two-script
contract. Registered packaging family id: **`transformer-tiny-1m`**.

## Layout

```text
examples/tiny-1m/
  prism.yaml         # declares the architecture + training entrypoints and the tokenizer
  architecture.py    # exposes build_model(ctx); defines the model only
  training.py        # exposes train(ctx); the miner-owned loop
```

- `architecture.py` exposes `build_model(ctx)` and is pure: it never reads data, opens files, or
  touches the network.
- `training.py` exposes `train(ctx)`: it forces the seed, builds the model via `architecture.py`,
  reads the read-only locked train split from `ctx.data_dir`, tokenizes with the pre-staged gpt2
  reference tokenizer (offline), runs a single-node multi-GPU-safe loop, and writes only under
  `ctx.artifacts_dir`.

## Family knobs (lab operators)

| Knob | Transformer tiny-1m value |
| --- | --- |
| Architecture family | Decoder transformer (RMSNorm + causal MHA + SwiGLU), weight-tied emb/lm_head |
| Parameter geometry | `dim=128`, `heads=4`, `layers=2`, `mlp_ratio=4`; count is **realized** `nn.Module` params under forced-seed `build_model` (weight tying avoids double-counting head/emb) |
| Cap | Must stay ≤ explore **124_000_000** (default lab ladder; promote pin 350M is out of scope for this ~1M seed) |
| Tokenizer | `prism.yaml` → `gpt2` (offline pre-staged; no network) |
| Step throughput | `LOCAL_BATCH=4`, AdamW `lr=0.005`, grad clip `1.0`; ranking is compute-normalized (tokens), not wall-clock |
| Multi-GPU | Single-node ≤8 GPUs: `init_process_group`, `set_device(local_rank)`, DDP wrap, `DistributedSampler` marker, rank-0 `torch.save`, `destroy_process_group`; also works at `world_size=1` (scored nproc=1) |
| Stability | Prefer small LR/batch if loss spikes; do not smuggle pretrained weights (step-0 anomaly zeros the score) |

Pack a submit-shaped zip with the shared harness:

```bash
uv run python scripts/pack_seed_family.py --family transformer-tiny-1m --output-dir dist/seed-packages
```

## How It Is Scored

The challenge re-executes `train(ctx)` under a forced random initialization on the locked FineWeb-Edu
train split, captures the single-pass online (predict-then-train) loss itself, and ranks
**emission** with held-out / generalization **primary** and prequential bits-per-byte **secondary**.
Any value this submission reports and any manifest it writes are ignored; the challenge authors
`prism_run_manifest.v2.json`.

## Submit

Day-1 production path: pack this family, sign with your hotkey, and POST the raw zip to the
joinbase bridge:

```text
POST https://chain.joinbase.ai/v1/challenges/prism/submissions
```

See [miner getting started](../../docs/miner/getting-started.md),
[docs/submissions.md](../../docs/submissions.md), and the [miner hub](../../docs/miner/README.md).
