# Hybrid gated delta-net / attention reference baseline (Prism v3, E6)

3:1 hybrid in the Kimi Linear / Qwen3-Next mold (see
`docs/spikes/prism-v3/research/03-beyond-transformers.md` §2, §3d): blocks
3, 7, 11, 15, 19, 23 (0-indexed) are sliding-window softmax attention; the
other 18 are gated delta-rule linear-attention blocks. This is the strong
reference miners must beat — per the survey, "beating 2017 attention is no
longer informative".

## Anchor config (default — what `build_model(ctx)` builds)

| Knob | Value |
|------|-------|
| vocab | 50257 (GPT-2, tied embeddings) |
| d_model | 1024 |
| n_layer | 24 (18 delta + 6 attention, every 4th attends) |
| attention | 16 heads × 64, exact sliding window **512**, RoPE θ=50000 |
| delta heads | 8, d_k = d_v = 128 (state 128×128 per head) |
| delta extras | causal depthwise conv (k=4, SiLU) on q/k/v; per-head output RMSNorm; sigmoid output gate |
| MLP | SwiGLU, hidden 2048 (= 2·d) |
| chunk | 64 |
| decay init | α ≈ e^−0.02 ≈ 0.98 per token (`softplus(a_bias) = 0.02`) |
| grad_checkpoint | True — activation-checkpoint every block when grads are enabled (see E11 fix below); no params, no numerics change |

Exact parameter math (verified by `count_params.py`, static == torch build):

| Component | Params |
|-----------|--------|
| tok_emb (tied head) | 51,463,168 |
| delta mixer projections / block | wq,wk,wa d→1024 + wv,wgate 1024→d… = 3·d·H·dk + 3·d·H·dv = 6,291,456 |
| delta misc / block | wβ 8,192 + a_bias 1,024 + convs 12,288 + head norm 128 + 2 block norms 2,048 = 23,680 |
| SwiGLU / block | 3 × 1024 × 2048 = 6,291,456 |
| delta block total × 18 | 12,606,592 × 18 = 226,918,656 |
| attn block (qkvo 4,194,304 + mlp 6,291,456 + norms 2,048) × 6 | 10,487,808 × 6 = 62,926,848 |
| final RMSNorm | 1,024 |
| **TOTAL** | **341,309,696** |

Cap check: 341,309,696 ≤ 350,000,000 — headroom 8,690,304 (2.48%).
MLP hidden is 2·d (vs 2.5·d in transformer_pp) because the delta mixer
carries 6 square projections vs attention's 4; both baselines land at
~341M for a clean iso-param comparison.

G8 µP LR-transfer honors `ctx["prism_width_multiplier"]`: scales
`d_model` / `mlp_hidden` / `delta_{key,value}_dim` (and `attn_heads` to
keep head_dim) so a 4× build exceeds 1.5× base params. Multiplier `1.0`
(default / absent) leaves the anchor unchanged (still ≤350M). The harness
µP sweep starts from a fixed small probe geometry (not the scored ≤350M
config) before applying the multiplier; honor top-level / `arch`
width-depth overrides as well.

## The delta block (canonical formulation, appendix 03 §2)

Per head, state S ∈ R^{dv×dk}, l2-normalized keys, per-channel forget gate
α_t ∈ (0,1)^dk (KDA-style channel-wise decay), per-head write rate
β_t ∈ (0,1):

    S_t = S_{t-1} Diag(α_t) (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ
    o_t = RMSNorm(S_t q_t) ⊙ σ(W_g x)  →  W_o

### Chunked evaluation in pure torch (v0 tradeoff)

Training/eval use the chunked WY-representation algorithm (chunk 64):
within a chunk all rank-1 updates compose in parallel via one unit
lower-triangular solve `(I + A) U = diag(β)(V − (k e^L) Sᵀ)`; the recurrent
state is carried across chunks only (t/64 sequential steps). With
L = cumsum(log α) and the masked pairwise decay tensor
`Dec[t,i] = e^{L_t − L_i}` (i ≤ t):

    A  = tril₋₁( β · Σ_c k_t k_i Dec )        B = Σ_c q_t k_i Dec
    O  = (q e^L) Sᵀ + B U                     S ← S Diag(e^{L_c}) + Uᵀ (k e^{L_c − L})

Correctness: matches the naive sequential recurrence to ≤ 1.2e-6 (fp32)
across single/multi-chunk, ragged-tail, and decay/no-decay configs (see
test evidence). The pairwise tensor is materialized per chunk instead of
factoring `e^{L_t}·e^{−L_i}` (which overflows fp32 for strong decay);
entries are ≤ 1 by construction, masked via −inf before `exp`. Per-step
log-decay is clamped at −2 (α ≥ 0.135/token — beyond that is instant
forgetting anyway).

**Cost vs fused kernels (expected slowdown):** peak working memory is
O(b·h·chunk²·d_k) per chunk — 134 MB transient per chunk at the anchor
train shape (8×8×64²×128 fp32) vs O(b·h·chunk·d_k) for FLA's fused Triton
kernels, and the einsum-heavy inner loop is memory-bound where fused
kernels use tensor-core matmuls plus recomputation. Expect roughly a
**2–4× training slowdown** vs an FLA `chunk_gated_delta_rule`
implementation at equal config (worse at long eval contexts), plus the
activation-checkpointing recompute below. This is the documented v0
tradeoff: the recipe forbids non-torch deps in the two-file submission
contract, and a fused/Triton kernel is an explicitly anticipated
miner-side optimization (E5 source-tree submissions make it practical).
The recurrence core runs in fp32 (state stability); projections run under
bf16 autocast on CUDA.

### Activation checkpointing (E11 memory fix, 2026-08-06)

**Bug:** the v0 shipped without checkpointing and OOMed deterministically
on the first training forward on the real Lium pod (RTX 5090, 31.36 GiB:
`CUDA out of memory`, ~30.55 GiB requested). Root cause: the "transient
per chunk" analysis above ignored autograd. The fp32 pairwise decay tensor
`dec` (b,h,c,c,dk) = (8,8,64,64,128) = 128 MiB, its einsum bmm operands
(262144×128 fp32, 128 MiB each), and the broadcast mask are **saved for
backward** for every chunk of every delta block: measured 5.81 GiB saved
per delta block at the production train shape (batch 8 × seq 512, the
harness defaults `PRISM_TRAIN_BATCH_SIZE=8` / `PRISM_SEQ_LEN=512`), i.e.
18 × 5.81 ≈ 104.5 GiB total (measured full-model forward peak RSS on
CPU: 79.3 GiB — the pod allocator gives up at 30.55 GiB).

**Fix:** every block runs under `torch.utils.checkpoint.checkpoint`
(`use_reentrant=False`) whenever grads are enabled (`grad_checkpoint`
config flag, default True; ctx-overridable like every DEFAULTS key). Only
block inputs persist across the forward; chunk internals are recomputed
block-by-block in backward. Eval/probes/scoring run under `no_grad` and
bypass checkpointing entirely (no recompute cost, no behavior change;
G8's `enable_grad` micro-steps get the memory-safe path automatically).

**Numerics:** unchanged. Checkpointed vs plain gradients are bitwise
identical (max |Δgrad| = 0.0, same seed, multi-chunk + window-crossing
shapes); the recompute runs the same kernels under the same autocast
state. Params unchanged: 341,309,696 ≤ 350M.

**Memory after fix** (measured, CPU RSS, default config, batch 8 × seq
512): 5.5 GiB after forward, 9.3 GiB after forward+backward (was 79.3 /
80.2 GiB). Estimated pod peak (RTX 5090, bf16 autocast): params fp32
1.28 + grads 1.28 + AdamW m/v 2.56 + block inputs bf16 0.19 + one delta
block's recompute graph 5.81 + logits/CE chain ~2.0 + CUDA context &
fragmentation ~1 ≈ **~14 GiB peak**, vs the 31.36 GiB card — comfortable
margin (target was ≤ ~24 GiB).

**Throughput impact:** one extra forward per block per step → step time
× ~4/3 at equal FLOPs; expect **~25–35% fewer tokens** inside the 6h wall
cap vs the (non-fitting) uncheckpointed run. The wall cap is unchanged and
self-monitored (`WALL_MARGIN_S`); fewer steps just means fewer tokens_seen,
not a cap violation.

### Sliding-window attention (exact)

t ≤ 512: single `is_causal` SDPA. t > 512: query-blocked exact SWA — each
≤512-query block attends its key span `[qs − 511, qe)` under a small banded
bool mask (≤ 512×1023), verified equal to naive masked attention at 2e-9.
No O(t²) mask at 32k eval contexts.

## Training recipe (`training.py`)

Identical structure to `baselines/transformer_pp/training.py` (see its
NOTES.md for the full contract notes: guard/wall-cap quirk, telemetry
cadence, `ModelOutput` + `self.logits`, stream-before-stop token
accounting). **Only difference: peak LR 3e-4 (vs 6e-4)** — gated
delta-rule stacks (per-channel decay + β write rates + output gates) are
spike-prone at this scale (cf. xLSTM/GDN reports in appendix 03 §2), so the
hybrid runs a more conservative peak with the same 2% warmup + cosine-to-10%
shape, AdamW (0.9, 0.95), wd 0.1 (no decay on norms, embeddings, or <2D
params — i.e. `a_bias` is exempt; all projections incl. `wbeta` and the
(c,1,k) conv taps are decayed), clip 1.0.

## Eval-context contract

No learned positions anywhere (RoPE only in attention blocks); the chunk
loop and windowed attention are length-agnostic. Verified at t ∈ {1, 16,
64, 65, 200, 1029} on tiny (crosses chunk and window boundaries) and
t = 1536 at the default config on CPU.

## lib.rs registration snippet

See `baselines/transformer_pp/NOTES.md` (single snippet covers both
baselines). Corpus ids: `baseline-hybrid-delta` (prefix `baseline` keeps
the copy gate exempt).

## Test evidence (2026-08-06, CPU, torch 2.x cpu wheel)

- `count_params.py`: static 341,309,696 == torch build 341,309,696; cap OK.
- Chunked delta rule == naive sequential recurrence, max abs err ≤ 1.2e-6
  over T ∈ {1, 4, 16, 17, 20, 70} × {decay, no decay}.
- Windowed SWA == exact masked attention, max abs err 1.9e-9 at t=150,
  window 64.
- 3:1 block pattern asserted (`DeltaBlock ×3 + AttnBlock` per 4).
- Deterministic init under seed; grad flows to **every** parameter
  (incl. `wa`, `a_bias`, `wbeta`, conv taps) — checked non-None, non-zero.
- Default config CPU forward at t=64 and t=1536: finite logits.
- `train()` 2 steps via fake `train_stream` (exact token accounting) and
  via the legacy `dataset_path` fallback.

### E11 checkpoint fix evidence (2026-08-06, CPU, torch 2.13.0+cpu)

- Saved-for-backward diagnostic (autograd hooks, production shape 8×512):
  one `GatedDeltaMixer` saves 5.81 GiB (top tensors: (8,8,64,64,128) fp32
  `dec` + (262144,128,1)/(262144,1,128) fp32 einsum operands, 128 MiB
  each); ×18 delta blocks ≈ 104.5 GiB — the OOM root cause.
- Tiny config (d=64, L=4, chunk 16, window 64): forward+backward, finite
  loss, grads flow to every parameter with checkpointing engaged.
- Checkpointed vs plain (same seed, t=100 multi-chunk + window crossing):
  identical loss, max |Δgrad| = 0.0 (bitwise).
- Production batch/seq shape (b=8, t=512) on tiny config: forward+backward
  executes (chunked + checkpointed paths shape-correct).
- no_grad eval at t=1030 (crosses window 64 and chunk 16): finite logits,
  checkpoint bypassed.
- `train()` 2 steps via fake `train_stream`: exact token accounting
  (tokens_seen = 2·b·t), finite loss.
- Default config, production shape 8×512, CPU RSS: 5.5 GiB after forward,
  9.3 GiB after forward+backward (pre-fix: 79.3 / 80.2 GiB).
- `grad_checkpoint=False` escape hatch still builds and runs.
