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

**Cost vs fused kernels (expected slowdown):** peak extra memory is
O(b·h·chunk²·d_k) per layer — 134 MB transient per chunk at the anchor
train shape (8×8×64²×128 fp32) vs O(b·h·chunk·d_k) for FLA's fused Triton
kernels, and the einsum-heavy inner loop is memory-bound where fused
kernels use tensor-core matmuls plus recomputation. Expect roughly a
**2–4× training slowdown** vs an FLA `chunk_gated_delta_rule`
implementation at equal config (worse at long eval contexts). This is the
documented v0 tradeoff: the recipe forbids non-torch deps in the two-file
submission contract, and a fused/Triton kernel is an explicitly anticipated
miner-side optimization (E5 source-tree submissions make it practical).
The recurrence core runs in fp32 (state stability); projections run under
bf16 autocast on CUDA.

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
