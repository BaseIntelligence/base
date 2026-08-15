# Transformer++ reference baseline (Prism v3, E6)

Modern GPT at the 350M cap. One of the two reference architectures the v3
anchors are measured on (the other is `baselines/hybrid_delta/`); miners must
beat these, not a 2017 vanilla Transformer.

## Anchor config (default — what the harness builds with `build_model(ctx)`)

| Knob | Value |
|------|-------|
| vocab | 50257 (GPT-2, tied embeddings) |
| d_model | 1024 |
| n_layer | 24 |
| n_head | 16 (head_dim 64) |
| MLP | SwiGLU, hidden 2560 (= 2.5·d) |
| Norm | pre-norm RMSNorm (eps 1e-6) |
| Positions | RoPE, theta 50000, NeoX half-split |
| Biases | none anywhere |

Exact parameter math (verified by `count_params.py`, static formula ==
`torch` build):

| Component | Params |
|-----------|--------|
| tok_emb (tied head) | 50257 × 1024 = 51,463,168 |
| attn qkvo / block | 4 × 1024² = 4,194,304 |
| SwiGLU / block | 3 × 1024 × 2560 = 7,864,320 |
| 2 RMSNorm / block | 2,048 |
| block total × 24 | 12,060,672 × 24 = 289,456,128 |
| final RMSNorm | 1,024 |
| **TOTAL** | **340,920,320** |

Cap check: 340,920,320 ≤ 350,000,000 — headroom 9,079,680 (2.59%).
SwiGLU hidden is 2.5·d rather than the canonical 8/3·d precisely so 24
layers fit under the cap (8/3·d at L=24 lands at 353.45M — over).

## Eval-context contract (G5 evaluates up to 32k)

No learned position table: RoPE only, cos/sin cached per attention module
with amortized 2× growth, computed at whatever length the eval requests.
Attention is `F.scaled_dot_product_attention(is_causal=True)` — no O(t²)
mask materialization, memory-efficient backend kicks in for long contexts.
Forward works at any t (verified at t = 1, 777, 2048 on tiny, and t = 1536
at the default config on CPU).

## Training recipe (`training.py`)

- AdamW, betas (0.9, 0.95), eps 1e-8, wd 0.1; **no decay** on norms,
  embeddings (incl. the tied head), or any <2D parameter.
- Peak LR **6e-4**, linear warmup 2% of `max_train_steps`, cosine to 10%.
- Grad clip 1.0; bf16 autocast on CUDA, fp32 on CPU; CE computed in fp32.
- Data: `ctx["train_stream"]` (harness-owned, seeded; `.tokens_seen` is the
  authoritative token counter). Legacy fallback: direct `dataset_path`
  tokenization with the same window semantics when no stream is in ctx
  (pre-1.3.0 harness).
- Stop conditions are checked **before** pulling the next batch so the
  authoritative token counter never counts an unconsumed batch.
- `guard()` every step; wall clock self-monitored against
  `ctx["train_hours_cap"]` minus `WALL_MARGIN_S` (180 s) — see contract
  notes. `prism_telemetry.report(loss, step, grad_norm)` every 10 steps;
  `finish_evaluation()` at the end.
- Deterministic under `ctx["seed"]` (init + legacy data order; the harness
  owns the stream's seed).

Batch is the harness's `batch_size × seq_len` = 8 × 512 = 4096 tokens/step;
20k steps ≈ 82M tokens max. No grad accumulation (step cap is the binding
constraint; accumulation would trade steps for tokens 1:1).

## Harness contract notes (discovered while writing this — E1 harness)

1. **`guard()` tripping fails the whole run.** `prismlib.miner_entry` only
   catches `FinishEvaluation` around `train()`; the harness's internal
   `_CapExceeded` propagates to the generic handler → `status: fail`. And
   the guard clock starts at **build_model**, not at train start. So miner
   code must self-monitor wall time and call `finish_evaluation()` before
   the cap; we stop at cap − 180 s and additionally catch `Exception` from
   `guard()` as a last-resort graceful stop. (Integrator: consider catching
   `_CapExceeded` in `miner_entry` like `FinishEvaluation` →
   `finish_reason: "cap_exceeded"`.)
2. **Scoring reads** `out.logits if hasattr(out, "logits") else out`
   (`prismlib/scoring.py`, `prismlib/probes.py`). We return a tiny
   `ModelOutput` with `.logits` AND set `self.logits` (both patterns).
3. **`val_ce_bpb` feeds untruncated val texts** and aligns targets to the
   last `logits.shape[1]` positions — self-truncation is allowed but we
   handle full length instead.
4. `prism_telemetry.report` **requires** `loss=` and `step=` (ValueError
   otherwise); every report can fire a G6 probe (default every 25 reports),
   so we report every 10 steps, not every step.
5. The harness moves the model to `ctx["device"]` itself after
   `build_model` — `build_model` must return a CPU module (do not
   `.to(device)` inside).
6. Param cap is `sum(p.numel() for p in model.parameters())` ≤
   `ctx["max_params"]` (default 350,000,000; `PRISM_TEST_MAX_PARAMS`
   shrinks it for staging/e2e — tiny overrides via ctx top-level keys or an
   `arch` dict: `vocab_size, d_model, n_layer, n_head, mlp_hidden,
   rope_theta, init_std`).
7. G8 µP LR-transfer honors `ctx["prism_width_multiplier"]`: scales
   `d_model` / `mlp_hidden` (and `n_head` to keep head_dim) so a 4×
   build exceeds 1.5× base params. Multiplier `1.0` (default / absent)
   leaves the anchor config unchanged (still ≤350M). The harness µP
   sweep overlays a fixed small probe geometry (`d_model=128`,
   `n_layer=4`, …) before applying the multiplier — not the scored
   ≤350M config — so 4× stays on-GPU; honor top-level / `arch`
   width-depth overrides as well.

## lib.rs registration snippet (for the integrator — do NOT apply here)

```rust
/// Reference baseline (E6): Transformer++ `architecture.py`.
pub const BASELINE_TRANSFORMER_PP_ARCHITECTURE_PY: &str =
    include_str!("../baselines/transformer_pp/architecture.py");
/// Reference baseline (E6): Transformer++ `training.py`.
pub const BASELINE_TRANSFORMER_PP_TRAINING_PY: &str =
    include_str!("../baselines/transformer_pp/training.py");
/// Reference baseline (E6): 3:1 gated delta-net/attention hybrid `architecture.py`.
pub const BASELINE_HYBRID_DELTA_ARCHITECTURE_PY: &str =
    include_str!("../baselines/hybrid_delta/architecture.py");
/// Reference baseline (E6): 3:1 gated delta-net/attention hybrid `training.py`.
pub const BASELINE_HYBRID_DELTA_TRAINING_PY: &str =
    include_str!("../baselines/hybrid_delta/training.py");
```

plus a contract test mirroring `baseline_satisfies_contract`:

```rust
#[test]
fn v3_baselines_satisfy_contract() {
    check_contract(BASELINE_TRANSFORMER_PP_ARCHITECTURE_PY, BASELINE_TRANSFORMER_PP_TRAINING_PY)
        .expect("transformer_pp");
    check_contract(BASELINE_HYBRID_DELTA_ARCHITECTURE_PY, BASELINE_HYBRID_DELTA_TRAINING_PY)
        .expect("hybrid_delta");
}
```

Corpus ids must start with `baseline` (`BASELINE_CORPUS_PREFIX`) to stay
exempt from the copy gate — e.g. `baseline-transformer-pp`,
`baseline-hybrid-delta`.

## Test evidence (2026-08-06, CPU, torch 2.x cpu wheel)

- `count_params.py`: static 340,920,320 == torch build 340,920,320; cap OK.
- Tiny config (d=128, L=4): deterministic init under seed; forward at
  t ∈ {1, 777, 2048}; `.logits` on return object and `self.logits`.
- Default config CPU forward at t=64 and t=1536 (> train seq_len): finite
  logits.
- `train()` 2 steps via fake `train_stream`: exactly 2 batches consumed
  (tokens_seen 128 = 2 × 2 × 32), real loss/grad_norm reported; legacy
  `dataset_path` fallback (pyarrow + injected tokenizer) also trains.
