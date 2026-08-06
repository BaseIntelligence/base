# Appendix 07 — Measuring Inference/Serving Efficiency
> Research appendix for the Prism v3 evaluation proposal (`docs/spikes/prism-v3/`). Produced 2026-08-06 via arXiv/web research. Non-normative spike document.

*Survey for an open architecture competition (100M–3B, arbitrary architectures). All arXiv IDs verified as of 2026-08-06.*

---

## 1. Core metrics and measurement harnesses

### 1.1 Metric definitions (get these exactly right — tools disagree)

| Metric | Definition | Watch out for |
|---|---|---|
| **TTFT** (time to first token) | Request sent → first response token received. Includes queueing + prefill + network. | Client-side TTFT includes network/queueing; server-side isolates prefill. Disclose which. |
| **TPOT / ITL** (time per output token / inter-token latency) | Mean time between consecutive decode tokens. genai-perf/AIPerf: `(e2e_latency − TTFT)/(output_tokens − 1)` — TTFT *excluded*. LLMPerf *includes* TTFT. | The two conventions differ materially at short outputs. Fix one (recommend genai-perf convention). |
| **E2EL** | Request sent → final token. | Dominated by output length; only meaningful at fixed OSL. |
| **Output throughput (tok/s)** | Total output tokens / wall time from first request to last response. | Inflated by forcing long outputs (`--ignore-eos`); deflated by early EOS. Fix OSL *and* run a natural-EOS cross-check. |
| **Per-user throughput** | Output tokens per request / that request's generation time → asymptotically `1/TPOT`. | Collapses as concurrency rises; report alongside system throughput. |
| **Goodput** | Requests/s meeting SLOs, e.g. `ttft:2000ms tpot:50ms` (vLLM `--goodput`, from DistServe, arXiv:2401.09670, Jan 2024). | The single most gaming-resistant throughput metric. |
| **Prefill throughput (input tok/s)** | ISL / prefill time, single request, swept over ISL. | Prefix caching must be off; use unique prefixes. |
| **Peak VRAM** | Peak allocated (weights + activations + state pool + fragmentation). | Use allocator stats *and* NVML; they differ by fragmentation/reserved blocks. |
| **Memory-bandwidth utilization** | `(bytes read per decode step × tok/s) / peak HBM BW`. At batch 1, decode is BW-bound: bytes ≈ weights + state read per token. | H100 SXM peak = 3.35 TB/s. A 3B BF16 model has a ~1.8 ms/token floor (~560 tok/s) — utilization % exposes kernel quality vs. architecture quality. |
| **Energy per token (J/token)** | `∫P(t)dt / tokens`, phase-aligned to prefill/decode. | NVML samples at ~10–100 Hz and misses µs bursts; integrate over whole runs, subtract idle baseline. TokenPowerBench (Niu et al., AAAI 2026) and ML.ENERGY (arXiv:2405.06320, May 2024) are the reference methodologies. Reference points: Llama-3-70B FP8 on 8×H100 ≈ 0.39 J/token; LLaMA-65B on A100-era ≈ 3–4 J/token; J/token drops ~25% from batch 32→256 then plateaus. |

### 1.2 Harnesses

- **vLLM `vllm bench serve`** (formerly `benchmarks/benchmark_serving.py`; vLLM itself: PagedAttention, arXiv:2309.06180, Sep 2023). Reports mean/median/std/p50–p99 of TTFT, TPOT, ITL, E2EL; `--goodput ttft:X tpot:Y`; datasets `random` (fixed `--random-input-len/--random-output-len`, `--random-range-ratio`, `--random-prefix-len`), `sharegpt`, `sonnet`; `--max-concurrency`, `--request-rate`, `--ignore-eos`. Also `vllm bench latency` (batch latency) and `vllm bench throughput` (offline). **Recommendation: this is the right primary harness** — it is the de facto standard, and Kimi Linear (arXiv:2510.26692, Oct 2025) showed novel-architecture vendors now ship vLLM integrations.
- **NVIDIA genai-perf → AIPerf** (Triton `perf_analyzer` lineage). The cleanest *client-side* tool: TTFT, time-to-second-token, ITL (TTFT-excluded), output token throughput, per-user throughput, request throughput, all with avg/min/max/p75/p90/p99; concurrency and request-rate sweeps against any OpenAI-compatible endpoint. Best used as the **load generator** pointed at whatever server a submission provides.
- **LMDeploy** (`profile_generation.py`, `profile_restful_api`): TTFT/token-latency/throughput for TurboMind; useful as a second engine but its kernel coverage for exotic architectures is thinner than vLLM's plugin path.
- **MLPerf Inference** (MLCommons): the gold standard for *audited* comparison, not a harness you'd reuse directly. LLM benchmarks (GPT-J 6B; Llama2-70B v4.0; Llama3.1-405B v4.1; Llama3.1-8B and DeepSeek-R1 v5.0, Apr 2025) run under LoadGen scenarios (Offline, Server, and an **Interactive** category with stricter bounds) with 99th-percentile TTFT/TPOT constraints — e.g. Llama3.1-8B: 2000/100 ms conversational, 500/30 ms interactive; Llama3.1-405B: 6000/175 ms server. Closed division enforces reference-model equivalence, accuracy ≥99% (99.9% high-accuracy) of the FP reference, generation length ≥90% of reference, mandatory TEST06 compliance, and audits (§7). Steal the *rules*, not the infrastructure.

**Practical note:** for 100M–3B models, harness overhead is non-negligible. Calibrate the measurement stack itself (time a trivially small model end-to-end; disclose overhead), prefer server-side timestamps, and standardize token counting with one reference tokenizer (Artificial Analysis uses tiktoken `o200k_base` for exactly this reason — different tokenizers shift "tok/s" by several percent).

---

## 2. Architecture-specific state costs

### 2.1 Analytical state accounting (per sequence, BF16)

**Attention KV cache** (bytes/token) = `2 · L_attn · n_kv_heads · d_head · bytes`:
- **MHA** → `2·L·H·d_h`; **MQA** (Shazeer, arXiv:1911.02150, Nov 2019) → `2·L·d_h`; **GQA** (Ainslie et al., arXiv:2305.13245, May 2023; uptrainable from MHA at ~5% of pretraining compute) → `2·L·G·d_h`. Llama-3-8B (GQA-8): 512 KiB/token; Llama-3-70B: 320 KiB/token.
- **MLA** (DeepSeek-V2, arXiv:2405.04434, May 2024): cache one latent per token, `L · (d_c + d_rope)` elements = DeepSeek-V2: 60 × (512+64) × 2 B ≈ **67.5 KiB/token** (paper claims 93.3% KV reduction vs. its 67B MHA predecessor and 5.76× max generation throughput). Quality reportedly ≥ MHA — i.e., cache compression without the MQA quality hit.

**Recurrent state (SSM / linear attention)** — *constant per sequence, zero per token*:
- Mamba (arXiv:2312.00752, Dec 2023): per layer `d_inner · d_state` SSM state + `d_inner · (d_conv−1)` conv state.
- Mamba-2 (arXiv:2405.21060, May 2024): `n_heads · d_head · d_state = d_inner · d_state` per layer. Mamba-2-2.7B (64 layers, d_inner 5120, d_state 128): **~80 MiB fixed**, flat in context. Compare: full-attention 8B at 32k context = 16 GiB.
- Linear attention / DeltaNet family (GLA, Gated DeltaNet, KDA): `n_heads · d_k · d_v` per layer — typically 0.25–1 MiB/layer, also flat.

**Hybrids** pay both, additively by layer: Jamba (arXiv:2403.19887, Mar 2024) 1:7 attention:Mamba; Nemotron-H (arXiv:2504.03624, Apr 2025) mostly Mamba-2 with a handful of attention layers; MiniMax-M1 (arXiv:2506.13585, Jun 2025; base MiniMax-Text-01 arXiv:2501.08313) 7:1 lightning:softmax; Kimi Linear 3:1 KDA:MLA → 75% KV reduction, 6.3× decode throughput at 1M context. **Diffusion LMs** are a special case: naive bidirectional decoding recomputes everything each step (no KV reuse); block-diffusion (LLaDA2.x) caches committed blocks, so their "state" looks attention-like per block with a step-count multiplier on compute.

### 2.2 Fair reporting: the "state card"

Never report a single "KV cache size." Require three numbers plus a curve:

1. **Incremental state bytes/token** (slope): attention layers only; 0 for pure recurrent.
2. **Fixed state bytes/sequence** (intercept): recurrent + conv state; 0 for pure attention.
3. **Effective bytes/token at max context** = `(fixed + slope · ctx_max) / ctx_max` — the single scalar that fairly amortizes recurrent state (e.g., Mamba-2-2.7B at 32k ≈ 2.6 KiB/token effective vs. ~150–500 KiB/token for GQA transformers).
4. **Memory-vs-context curve**: peak VRAM at batch 32 over context {1k, 4k, 16k, 32k, max}. Linear fit separates the paradigms by construction — flat (recurrent), linear (attention), piecewise/kinked (hybrid) — and empirically *validates* the analytical claim (require agreement within 10% of the allocator-measured state pool).
5. **Capacity frontier**: max concurrent sequences at 8k context in 80 GB. This is where small state converts into serving economics (Kimi Linear's 6× decode-at-1M argument is really a capacity argument), and it can't be gamed by kernel tricks.

---

## 3. Quality–efficiency Pareto reporting

**How the credible papers do it:**
- **Kimi Linear** (arXiv:2510.26692, Oct 2025): the current best practice — *identical training recipe* for the linear-attention and full-MLA models, quality compared at iso-recipe, then efficiency reported as TPOT-vs-context and prefill-vs-context curves out to 1M (6.3× decode, ~3× prefill at 1M), plus open-sourced vLLM integration so numbers are reproducible.
- **Nemotron-H** (arXiv:2504.03624, Apr 2025): accuracy parity vs. Qwen2.5/Llama-3.1 at same scale, throughput measured on H100 at a *disclosed* operating point (65,536 in / 1,024 out) → "up to 3×." Honest, but note the speedup is context-dependent — at short context the same model is far less than 3× faster.
- **MiniMax-M1** (arXiv:2506.13585, Jun 2025): FLOP-based claim — 25% of DeepSeek-R1's FLOPs at 100k generation length. FLOPs-at-generation-length is hardware-independent but says nothing about wall-clock (lightning-attention kernels are less mature than attention kernels).
- **Mercury** (arXiv:2506.17298, Jun 2025): the gold standard for credibility — throughput (1,109 tok/s Mini, 737 tok/s Small on H100) measured by **Artificial Analysis independently**, not by the vendor.

**Independent verification practice (Artificial Analysis methodology):** TTFT; output speed = tokens/s after first token; E2E response time; "Time to First *Answer* Token" for reasoning models (excludes thinking); token counts normalized via tiktoken `o200k_base`; workloads at ~100/1k/10k input tokens; 8 measurements/day, median over 14 days, P5–P95 published. Their stance is explicit: results represent *experienced* performance, not theoretical maximum.

**Pitfalls of vendor-reported throughput (each observed in the wild):**
- "Up to N×" quoted at 1M context when the competition/context of interest is 4–32k.
- Batch-size cherry-picking: batch-256 throughput for a "fast model" claim; batch-1 latency for a "responsive model" claim — never both.
- Tokenizer drift: more tokens/s because the tokenizer emits more tokens per character.
- Output-length gaming: throughput with `--ignore-eos` at OSL=2048; or latency on prompts where the model answers in 5 tokens.
- dtype asymmetry (FP8 for the new model, BF16 for the baseline), prefix caching left on for "prefill" numbers, warmup shapes exactly matching benchmark shapes, and omitted percentiles (p99 TPOT is where hybrid/recurrent irregularities show up).
- Quality measured at a different operating point than speed (acute for diffusion LMs, §5).

**Rule for the competition:** quality and efficiency must be measured on the *same checkpoint, same decoding configuration, same machine*, and every speed claim must name (batch, ISL, OSL, dtype, engine, percentile).

---

## 4. Quantization robustness as an architecture property

**Evidence base:**
- **Evaluating Quantized LLMs** (arXiv:2402.18158, Feb 2024; 11 families, 125M–180B, incl. Mamba): W8A8 is nearly lossless across almost all families; W4A4 collapses; weight-only W4 and KV4 are within ~2% for most ≥7B models; recommendations are size-dependent (for <13B: W8/W8A8/KV8). Notably includes a non-transformer family — quantization behavior *is* architecture-dependent.
- **Scaling Laws for Precision** (Kumar et al., arXiv:2411.04330, Nov 2024; ICLR 2025; 465 pretraining runs): PTQ degradation **grows with tokens trained** — past a critical data size, more pretraining data *actively hurts* post-quantization quality; models trained in lower precision are more PTQ-robust; weight-precision gains saturate ~6–7 bits. Implication for a competition: heavily-trained 3B submissions are *expected* to quantize worse — that's a measurable architecture/training property, not noise.
- **"Give Me BF16 or Give Me Death"** (arXiv:2411.02355, Nov 2024; 500k+ evals on Llama-3.1): FP8 W8A8 effectively lossless; well-tuned INT8 W8A8 only 1–3% degradation; W4A16 (GPTQ/AWQ, g128) competitive with 8-bit; deployment-wise W4A16 wins synchronous/low-batch, W8A8 wins continuous batching.
- **Mechanism:** outlier activations drive quantization error — emergent outliers (LLM.int8(), arXiv:2208.07339, Aug 2022), massive activations/attention sinks (arXiv:2402.17762, Feb 2024; arXiv:2309.17453), and architectural mitigations (Quantizable Transformers, arXiv:2306.12929, Jun 2023: clipped softmax, gated attention, no weight-norm pathologies). GLU variants and unbounded norms quantize worse; some novel blocks (SSM selective scans, large dynamic-range gates) have *underexplored* quantization behavior — a genuine differentiator for a novelty competition. KV-cache-specific: KIVI (arXiv:2402.02750, Feb 2024) shows KV4 is near-lossless for standard attention; whether recurrent states tolerate 4-bit is an open question submissions should answer.

**Quantization probe for the eval (fixed, organizer-run, ~half a day per model):**
1. Fixed calibration set (512 sequences from a held-out competition corpus), fixed recipe: **W4A16 GPTQ g128** and **W8A8 dynamic** (SmoothQuant-style, arXiv:2211.10438); for attention-containing models add **KV-cache INT8**; for recurrent models add **state INT8** where the engine supports it.
2. Measure Δ on a small fixed suite (e.g., 200-sample GSM8K, 5-shot MMLU subset, perplexity on a fixed held-out slice) vs. the model's own BF16.
3. Report `Δquality` and `Δthroughput` at batch 1 and 32. Scoring: no penalty within a 1% quality gate (MLPerf-style); beyond that, deduct from the quality score rather than the efficiency score — quantization fragility is a quality-risk property. Publish the delta table regardless; it's high-signal architecture information.

---

## 5. Speculative / parallel decoding interactions

**The landscape:**
- AR speculative decoding (Leviathan et al., arXiv:2211.17192, Nov 2022; Chen et al., arXiv:2302.01318, Feb 2023; Medusa arXiv:2401.10774, EAGLE arXiv:2401.15077, Jan 2024) is an *orthogonal serving technique* for AR models — 2–3× on top of whatever the architecture gives.
- Diffusion LMs decode *natively in parallel*: Mercury (arXiv:2506.17298) >1,000 tok/s on H100; LLaDA (arXiv:2502.09992, Feb 2025); **LLaDA2.1** (arXiv:2602.08676, Feb 2026) adds token-to-token editing with explicit **Speedy/Quality modes** — 892 tok/s on HumanEval+ at 100B — making the speed–quality knob a first-class, vendor-admitted object. Fast-dLLM (arXiv:2505.22615) adds caching/confidence-thresholded parallel decode for open dLLMs.
- Diffusion↔AR interop is now real: dLLMs as *drafters* for AR verifiers — DiffuSpec (arXiv:2510.02358, Oct 2025, up to 3×), SpecDiff-2 (arXiv:2511.00606, Nov 2025, MLSys 2026, up to 5.5×), DEER (arXiv:2512.15176, Dec 2025, 32-token acceptance lengths), Speculative Diffusion Decoding (Christopher et al., NAACL 2025).
- Recurrent-depth (Huginn, arXiv:2502.05171, Feb 2025): per-token latency scales with the iteration knob `r` (quality improves to ~64 steps); natively supports self-speculative decoding and KV-cache sharing.

**How to score different generation paradigms on one latency axis:**

1. **Iso-quality is the only fair axis.** Each paradigm has a compute knob (AR: none/drafter; diffusion: steps × confidence threshold; recurrent-depth: iterations). Score every submission at the knob setting where its quality (on the competition suite) matches its own best quality within a tolerance (e.g., ≤0.5% relative drop). LLaDA2.1's S/Q modes are the vendor-endorsed version of exactly this.
2. **Require the knob curve anyway:** quality vs. tok/s at ≥3 knob settings. It's cheap and exposes models that only hit parity at unusable settings.
3. **No external drafters in scored runs** — speculative decoding with a separate model measures the drafter, not the architecture. Built-in/self-speculative (Medusa-style heads, recurrent-depth self-speculation, dLLM editing) is allowed but must be disclosed and quality-gated. Optionally report an informative "with ecosystem acceleration" number.
4. **TTFT needs redefinition for non-AR models:** diffusion LMs emit no meaningful "first token." Use *time to first committed/finalized token* (or first answer token, Artificial Analysis-style) and E2EL at fixed OSL as the primary latency metrics for all paradigms — E2EL at fixed (ISL, OSL) is the one metric that is paradigm-neutral.
5. **Normalize tokens** with the reference tokenizer (§1.2) — non-negotiable when a diffusion submission uses a different tokenizer than an AR one.

---

## 6. Edge/CPU deployment (secondary)

LFM2 (Liquid AI, technical report arXiv:2511.23404, Nov 2025) is the template: hybrid convolution+attention models benchmarked with **llama.cpp Q4_0, batch 1**, reporting prefill and decode tok/s at 1k/4k prefixes on a Snapdragon 8 Elite (Galaxy S25) and a Ryzen AI 9 HX 370 — claiming ~2× decode/prefill over Qwen3 on CPU and Pareto-dominance vs. Qwen3/Llama-3.2/Gemma-3/SmolLM3/Granite-4 at each size. The methodology lesson: edge numbers are reported at *fixed quantization, fixed runtime, batch 1, two prefix lengths* — because variance across devices/toolchains dwarfs everything else.

**Is it worth a track?** Qualified yes, as a *secondary/informative* track only: (a) 100M–3B is exactly the edge-relevant scale, and CPU decode exposes memory-bandwidth-friendliness that GPU batch-256 hides (convolutions and small recurrent states shine); (b) but results are toolchain-hostage — llama.cpp/ExecuTorch kernel coverage for a novel architecture will lag by months, so you'd benchmark the port, not the architecture. Recommendation: one reference CPU (e.g., Ryzen HX 370 or a fixed cloud ARM instance), one runtime (llama.cpp), Q4_0, batch 1, prefill/decode at 1k/4k prefixes, **no points** — publish as a dashboard column. Revisit once the winning architectures have real ports.

---

## 7. Gaming risks and MLPerf-style mitigations

| Gaming vector | Instance | MLPerf-derived mitigation |
|---|---|---|
| Kernels specialized to benchmark shapes | Shape-specialized kernels are allowed by MLPerf only if *general*; audit reviews source | Hidden-shape audit: after the deadline, rerun on withheld (batch, ISL, OSL) points incl. odd sizes (17, 33, ISL±1); score = hidden subset, not the public grid |
| Batch-size cherry-picking | "Up to" numbers at the model's best batch | Fixed concurrency sweep {1, 32, 256}; all points published; score uses all |
| Warmup/caching tricks | Prefix caching on shared prefixes; CUDA-graph capture of only benchmark shapes; JIT inside timed region | Prefix caching **disabled**; unique random prefixes (`--random-prefix-len 0`); fixed warmup budget (1 full discarded run); CUDA graphs allowed but must cover the declared shape range; cold-start run reported separately |
| dtype tricks | FP8 submission vs BF16 baselines; quality shaved below the gate | Closed division: BF16/FP16 weights+activations mandatory; FP8 only in an Open division; MLPerf-style accuracy gate: ≥99% of the submission's *own* BF16 quality |
| Output-length gaming | `--ignore-eos` inflation; early-EOS deflation | MLPerf TEST06 analog: fixed-OSL runs for throughput **plus** natural-EOS run whose mean generation length must be within 90–110% of the reference model's |
| Tokenizer gaming | More tokens/char → higher "tok/s" | Reference-tokenizer normalization (tiktoken o200k) + chars/s cross-check |
| Result caching across queries | Prohibited outright in MLPerf (KV reuse within a sequence only) | Fresh prompts per run, fixed seeds, LoadGen-style open-loop request generation |
| Selective hardware reporting | Boost clocks, cold GPU | Locked clocks (`nvidia-smi -lgc`), logged temperature/power, driver/CUDA pinned |
| Cherry-picked runs | Best-of-N silently | Fixed protocol: 1 warmup + 3 timed runs, median reported, CV <5% or rerun; raw logs published |

MLPerf's structural answers worth copying wholesale: **closed division** (organizer-controlled harness and configuration; submitters provide weights + engine plugin, not the benchmark), **mandatory compliance tests** (TEST01 accuracy, TEST06 LLM generation-length), **source-code inspection sufficient to reproduce**, and **post-hoc audits of a random subset** with disqualification as the penalty. For a small competition, auditing 2 random + 1 nominated submission per round is affordable and changes incentives completely.

---

## 8. Concrete protocol: single reference GPU, 100M–3B

### 8.1 Environment lock
- **GPU:** 1× H100 80GB SXM (community reference; MLPerf and Artificial Analysis both standardize on H100). Locked clocks, ECC on, no MIG, pinned driver/CUDA. Log temperature and power throughout.
- **Engine:** vLLM (fixed V1 version, e.g. pin at competition start). Submissions unsupported upstream provide a vLLM plugin *or* any server exposing OpenAI-compatible streaming; all servers pass a harness-compliance test (identical client behavior). Exotic fallbacks (pure HF loop) permitted with a disclosed, calibrated overhead constant.
- **Client:** `vllm bench serve` / AIPerf (genai-perf), fixed seeds, streaming on, prefix caching off, `--ignore-eos` for fixed-OSL runs only.
- **Precision:** BF16 end-to-end (closed division). Token counting: tiktoken `o200k_base` for all reported tok/s.

### 8.2 Metric set (all reported; starred = scored)
1. **TTFT\*** p50/p99, single request, ISL ∈ {128, 1024, 8192, 32768}, OSL=8. (Non-AR: time-to-first-committed-token.)
2. **TPOT\*** (decode latency) p50/p99 at batch 1, 256 tokens generated from prefixes {0, 4096, 16384, 32768} → the *decode-latency-vs-context curve* (this is where SSM/hybrid advantages and attention degradation actually show).
3. **Throughput\***: max output tok/s at concurrency {1, 32, 256}, ISL=1024, OSL=256, open-loop; plus **goodput** under SLO `TTFT≤2s, TPOT≤50ms`.
4. **Prefill throughput**: input tok/s, single request, ISL ∈ {512, 2048, 8192, 32768}.
5. **Peak VRAM\*** at (batch 32, 8k) and at max feasible configuration; plus **capacity frontier\***: max concurrent sequences at 8k in 80 GB.
6. **Memory-bandwidth utilization** at batch-1 decode: `(bytes read/token × tok/s)/3.35 TB/s`; bytes read = weights + state (from the state card, validated against allocator).
7. **Energy**: NVML ≥10 Hz, idle-subtracted, phase-aligned **J/token\*** for prefill and decode at (batch 32, ISL 1024, OSL 256) — TokenPowerBench methodology.
8. **State card\***: slope (bytes/token), intercept (bytes/sequence), effective bytes/token @ 32k, memory-vs-context curve {1k–32k} @ batch 32; analytical vs. measured must agree within 10%.
9. **Quantization probe** (§4): W4A16-g128 GPTQ + W8A8 dynamic + KV/state INT8; Δquality on the fixed mini-suite, Δthroughput at batch 1/32.
10. **Paradigm knobs** (§5): knob curve ≥3 points; scored at iso-quality setting; no external drafters.

### 8.3 Sweep grid and run policy
- **Grid:** concurrency {1, 32, 256} × ISL {128, 1024, 8192, 32768} for latency; throughput at ISL 1024/OSL 256; decode-vs-context at batch 1. Feasibility caveat: batch 256 × 32k prefill is impossible for full attention on one GPU (~1.2 TB KV for a 3B GQA model) — record infeasible cells as such; the capacity frontier metric (8.2.5) captures this fairly instead of letting attention models no-show the cell.
- **Runs:** 1 full warmup run (discarded, fixed duration) + **3 timed runs**; ≥1000 decode steps per batch-1 run; report median and p99; **CV >5% → rerun**; all raw logs and configs published.
- **Duration discipline:** each throughput point ≥60 s steady-state; request arrivals open-loop (Poisson) at fixed rate or saturated concurrency — disclose which.
- **Anti-gaming:** hidden post-deadline shape subset; natural-EOS cross-check (90–110% of reference generation length); 99%-of-own-BF16 quality gate; 2 random + 1 nominated source audits.

### 8.4 Folding into the competition score
**Primary: Pareto axis (recommended).** Leaderboard = 2D Pareto of **quality score** (competition quality eval) vs. **serving cost** = GPU-seconds per 1M output tokens, derived from the *saturated-throughput* run (best feasible concurrency, disclosed) — the economically meaningful scalar. Non-dominated sorting for ranks; publish the frontier plot (quality vs. cost, points annotated with TPOT@batch-1). This is the Kimi Linear/Nemotron-H framing done rigorously, and it is immune to weight-tuning disputes.

**Secondary: z-scored efficiency group (for the "efficiency prize" and dashboards).** Within each size class (100M/500M/1B/3B): z-score TPOT@batch1, TTFT@8k, tok/s@batch32, tok/s@batch256, effective state bytes/token@32k, J/token@batch32, and quant-probe Δ; winsorize at ±3σ; average with pre-published weights (suggested: latency 30%, throughput 30%, memory/state 20%, energy 10%, quant robustness 10%). Do **not** use the z-score group as the primary axis — arbitrary weights hide dominance relationships and invite single-metric extremization.

**Hard gates before any scoring:** quality floor (must beat a fixed weak baseline), 99%-of-own-BF16 quant gate for closed division, state-card validation, compliance/audit pass. A submission that games and fails audit is disqualified, not re-scored — the MLPerf lesson is that the *credible threat* of audit is what keeps the honest majority honest.

---

**Key sources (all verified 2026-08-06):** GQA 2305.13245 (May 2023) · MLA/DeepSeek-V2 2405.04434 (May 2024) · Mamba 2312.00752 (Dec 2023) · Mamba-2 2405.21060 (May 2024) · Jamba 2403.19887 (Mar 2024) · vLLM 2309.06180 (Sep 2023) · DistServe/goodput 2401.09670 (Jan 2024) · Evaluating Quantized LLMs 2402.18158 (Feb 2024) · KIVI 2402.02750 (Feb 2024) · Massive Activations 2402.17762 (Feb 2024) · Scaling Laws for Precision 2411.04330 (Nov 2024) · "Give Me BF16 or Give Me Death" 2411.02355 (Nov 2024) · ML.ENERGY 2405.06320 (May 2024) · Nemotron-H 2504.03624 (Apr 2025) · MiniMax-M1 2506.13585 (Jun 2025) · Mercury 2506.17298 (Jun 2025) · Kimi Linear 2510.26692 (Oct 2025) · LFM2 2511.23404 (Nov 2025) · LLaDA 2502.09992 (Feb 2025) · LLaDA2.1 2602.08676 (Feb 2026) · Huginn recurrent-depth 2502.05171 (Feb 2025) · DiffuSpec 2510.02358 (Oct 2025) · SpecDiff-2 2511.00606 (Nov 2025, MLSys 2026) · DEER 2512.15176 (Dec 2025) · TokenPowerBench (AAAI 2026) · MLPerf Inference v5.0 rules (Apr 2025) · Artificial Analysis methodology (artificialanalysis.ai/methodology).
