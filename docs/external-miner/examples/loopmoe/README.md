# loopmoe

Reference AutoModel patch for Prism recipe 2.1: LoopMoE with rank-0
harness-stream DDP and optional Transformer Engine NVFP4.

This is **miner documentation + example code**. It is not a control-plane
binary, not a scored organizer baseline, and not a live `:28092` flip.

## Submit

| File | Role |
|------|------|
| `automodel.base` | Pin id — must match `GET /v1/recipe` (`automodel@v0.5.0`) |
| `automodel.patch` | Unified diff vs that pin (entry, model, kernels, DDP worker) |
| `prism.toml` | Optional entry pointer |
| `requirements.txt` | Optional TE / FLA wheels (installed network-on, then train goes offline) |

Pack the four files at the ZIP root and `POST /v1/submissions` with your
hotkey + `X-Lium-Api-Key`. See [`../../prism.md`](../../prism.md).

Unpacked modules (`entry.py`, `model.py`, `kernels.py`, `ddp_worker.py`)
are the same tree the patch applies under
`nemo_automodel/components/models/loopmoe/` — useful for local reading.

## Contract this example honors

- `build_model(ctx)` returns an `nn.Module`; train consumes
  `ctx["train_stream"]` only (G6 / dual-cap accounting).
- Multi-GPU: rank 0 owns the harness stream and scatters each global batch.
  Rank 0 also writes `loopmoe_ddp/telemetry.json` (loss series + probe
  curve) so the parent `prism_telemetry` / G6 ingest sees DDP workers.
- `ctx["gpu_count"]` / TE `NVFP4BlockScaling` when the class exists
  (consumer Blackwell: `disable_rht=True`, `disable_stochastic_rounding=True`).
- Optional env (miner-side, not organizer knobs):
  `LOOPMOE_DELTA_KERNEL=chunk_wy` (default) or `kda`;
  `LOOPMOE_MICRO_BATCH` (default 8).

Do not point this example at live `:28092` to flip scoring. Live defaults
stay `PRISM_SCORING_MODE=benchmarks` and `PRISM_ANCHOR_VERSION=0`.
