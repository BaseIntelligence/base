# Appendix 06 — Measuring Training Cost Fairly
> Research appendix for the Prism v3 evaluation proposal (`docs/spikes/prism-v3/`). Produced 2026-08-06 via arXiv/web research. Non-normative spike document.

# Measuring and Comparing Training Cost of Novel ML Architectures Fairly

*Survey compiled 2026-08-06. arXiv IDs and dates inline throughout.*

---

## 1. Cost-metrics taxonomy — and which ones lie

### 1.1 The metrics

| Metric | What it measures | How it's computed | Main failure mode for novel architectures |
|---|---|---|---|
| **FLOPs (total training)** | Arithmetic work | Analytical: \(C \approx 6ND\) for dense transformers (Kaplan et al., 2001.08361, Jan 2020); or profiler-counted | Architecture- and implementation-dependent; ignores memory movement and parallelism |
| **Wall-clock** | Elapsed time to target/budget | Organizer-side timestamps | Hardware/implementation coupling (which is sometimes the point) |
| **GPU-hours** | Wall-clock × GPUs | Trivial on single-GPU pods | Identical to wall-clock at 1 GPU |
| **Dollars** | GPU-hours × spot/reserved price | Price sheet | Prices fluctuate; SKU-dependent |
| **Energy (kWh) / carbon (kgCO₂e)** | Physical energy; emissions | Integrate power draw over time; × PUE; × grid carbon intensity | Grid/PUE unknowns; GPU-only vs node-level ambiguity |
| **Tokens/sec throughput** | Data consumption rate | Tokens ÷ elapsed | Says nothing about quality per token |
| **MFU** | Achieved FLOPs/s ÷ peak dense FLOPs of the SKU | PaLM (2204.02311, Apr 2022); Korthikanti et al. (2205.05198, May 2022) also define HFU | Requires a FLOPs count — inherits all FLOPs problems |
| **Memory footprint (peak)** | Max VRAM | `torch.cuda.max_memory_allocated`, NVML | A constraint more than a cost; determines feasible batch/model per SKU |

### 1.2 Why FLOPs lie for novel architectures

The canonical critiques:

- **The Efficiency Misnomer** (Dehghani et al., **2110.12894**, Oct 2021; ICLR 2022): FLOPs, parameter count, and speed *contradict each other*. FLOPs ignore degree of parallelism (depth, recurrence) and memory-access cost; parameter-matching for "fair" comparison is equally misleading (shared parameters can be few yet slow).
- **The Hardware Lottery** (Hooker, **2009.06489**, Sep 2020; CACM 2021): ideas win because they fit the hardware/software stack, not because they're superior. GPUs are over-optimized for dense matmul; anything else (capsule routing then, SSM scans now) pays an artificial tax.
- **Roofline model** (Williams, Waterman, Patterson, CACM 2009) and Horace He's *Making Deep Learning Go Brrrr From First Principles* (2023): kernels are compute-bound, **bandwidth-bound**, or overhead-bound. Two architectures with identical FLOPs can differ by an order of magnitude in wall-clock if one is memory-bound.

Concretely, for the architecture families your competition will attract:

- **SSMs / linear attention** (Mamba, **2312.00752**, Dec 2023; Kimi Linear, **2510.26692**, Oct 2025): FLOPs depend on the *chunk size* \(C\) — an implementation knob, not an architecture property. Kimi Linear's own accounting gives per-head FLOPs of \(6Td_h^2 + 3TCd_h + TC^2\) for KDA at \(C{=}64\) vs \(2T^2d_h\) for full attention. Change \(C\) and the "cost" changes without any quality change. The \(6ND\) approximation silently breaks.
- **MoE**: total params vs active params vs active FLOPs are three different numbers. The NanoGPT speedrun handles this by capping *active* parameters per token (≤124M), explicitly allowing MoE.
- **Recurrent-depth / weight-shared** (Universal Transformer-style): parameters and FLOPs fully decouple; iso-parameter comparison is meaningless.
- **Custom kernels**: a novel architecture with a fused Triton kernel vs a naïve PyTorch reference can differ 5–10× in wall-clock at identical FLOPs. Mamba's headline "5× higher throughput than Transformers" is a kernel statement as much as an architecture statement.

**Gameability ranking** (most → least gameable in your setting): FLOPs (choose the counting convention that favors you; chunk-size laundering) > params (MoE/sharing games) > throughput (batch/seq-len games) > energy (mostly honest but noisy) > dollars (price noise) > **wall-clock on pinned hardware, measured organizer-side** (hardest to fake — you can't lie about elapsed time on someone else's machine).

---

## 2. Speedrun methodology: time-to-target competitions

### 2.1 The NanoGPT speedrun (Keller Jordan et al., May 2024 → present)

Task: **train a 124M-active-parameter model to ≤ 3.28 validation loss on FineWeb, fastest wall-clock on 8×H100 wins** ([KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)). The 3.28 target matches Karpathy's llm.c GPT-2 reproduction (45 min, 05/28/24).

**Record evolution** (from the repo's record table and record-holder write-ups):

| Date | Record | Key innovation |
|---|---|---|
| 05/28/24 | 45.0 min | llm.c baseline |
| 06/06/24 | 31.4 min | Tuned LR + rotary embeddings |
| 10/04/24 | 24.9 min | Muon optimizer |
| 11/19/24 | 5.03 min | 64K-ctx FlexAttention |
| 01/26/25 | 2.93 min | Reduced batch size |
| 02/01/25 | — | **Timing-rule change** (see below) |
| 07/13/25 | 2.86 min | BOS-aligned batches, cooldown tuning |
| 09/29/25 | 2.48 min | Polar Express (replaces Newton-Schulz) |
| 12/22/25 | 1.99 min | Multi-token prediction |
| 01/19/26 | 1.66 min | Bigram hash embedding |
| 02/02/26 | 1.53 min (#69) | Kernel tuning |
| ~05/26 | **≈80 s (#82)** | Gated Exclusive Self-Attention (XSA), per-(layer,head) tanh gates funding a 30-step schedule cut; 79.85 s at val loss 3.27865 ± 0.00145, n=10, Welch \(p \approx 0.0014\) |

That's **~34× in two years**, with roughly equal contributions from architecture, optimizer, data-ordering, and systems work — which is exactly why it's the best existing model for your competition.

**Anti-gaming rules that make it work:**

1. **Fixed data pipelines** — you may change batch size, sequence length, attention structure, but *not the underlying token stream order*.
2. **≤124M active parameters per token** (MoE allowed; untied embedding counts only hidden_dim).
3. **Statistical significance**: submissions must supply enough run logs to show mean val loss ≤ 3.28 at **p < 0.01** (one-sample t-test; example code in-repo). Pure systems changes that don't touch the ML are waived.
4. **Same-hardware rebaseline**: a record must beat the prior record *when both are timed on the same box* (records are validated on Prime Intellect 8×H100; the WR #82 write-up notes a −0.554 s delta on 8×H200 vs −0.685 s on RunPod 8×H100 — hardware-setup variance is real and explicitly re-measured).
5. **Timing integrity after record #21** (02/01/25): the 10-step untimed "grace period" was replaced by an explicit **untimed warmup on dummy data**, and `torch._inductor.config.coordinate_descent_tuning` was **banned** because it shifted ~25 min of work into untimed compilation. Lesson: *anything untimed will be weaponized*.
6. **The target is defined mathematically, not procedurally**: "a probability model assigning ≥ exp(−3.28 × 10,485,760) to the first 10,485,760 FineWeb val tokens" — evaluation at any sequence length is allowed if it's a valid probability model. This deliberately legitimized a **test-time-training exploit** (PR #205: TTT during final validation, −30 steps) — an instructive precedent: if you don't want eval-time compute spent on your metric, you must cap it explicitly, because *someone will do it*.
7. Discretionary quality gates: the record is kept 0.001–0.002 nats below target as a variance buffer; PRs that consume the buffer must beat a naive step-count decrease at equivalent loss.

There are also **Track 2** (GPT-2 Medium, 2.92 target) and **Track 3** (optimization: minimize *steps* at fixed arch/data/batch with unlimited wall-clock — record 2,690 steps as of 06/2026), the latter being the sample-efficiency mirror of the main track.

### 2.2 The lineage of time-to-target

- **DAWNBench** (Coleman et al., NeurIPS MLSys Workshop 2017 / SysML 2019): the original **time-to-accuracy** (TTA) competition — time and cloud *dollars* to 93% top-5 on ImageNet. The follow-up analysis (Coleman et al., SIGOPS OSR 2019) showed TTA has a **low coefficient of variation** — i.e., time-to-target is a statistically well-behaved metric. Retired 03/27/2020 into MLPerf.
- **MLPerf Training** (MLCommons, 2018–): industrialized TTA with **closed division** (same model, optimizer, preprocessing, hyperparameters, and quality target as the reference — apples-to-apples hardware/framework comparison) and **open division** (any method that reaches target). Scored as absolute time-to-train per system; *no cross-hardware normalization* (see §5).
- **MLCommons AlgoPerf: Training Algorithms** (Dahl et al., **2306.07179**, Jun 2023; results Aug 2024): time-to-target across 8 fixed workloads on fixed hardware, with the crucial innovation of **two tuning rulesets** (see §4). Winners: Distributed Shampoo (external tuning, −28% vs baseline), Schedule-Free AdamW (self-tuning, −8%).
- **CIFAR-10 speedruns** (tysam's hlb-CIFAR10, 2022; Keller Jordan's cifar10-speedrun, 2023–24): 94% accuracy on a single A100 in seconds — the single-GPU ancestor of your setting.
- **MLE-bench** (OpenAI, **2410.07095**, Oct 2024; ICLR 2025): 75 Kaggle competitions as agentic ML-engineering tasks; best setup (o1-preview + AIDE) medaled in 16.9%. Notable for explicitly studying **resource scaling** (more time/compute → more medals) and pretraining contamination.
- **RE-Bench** (METR, **2411.15114**, Nov 2024): 7 open-ended ML R&D environments scored against 71 human-expert 8-hour attempts; agents win 4× at 2-hour budgets, humans win 2× at 32-hour budgets. Key methodological import: **score as a function of total compute-time budget**, with best-of-k explicitly modeled — directly relevant to how you set and interpret your 6h cap.

---

## 3. Iso-compute comparison protocols in architecture papers

### 3.1 What the cited papers actually did

| Paper | arXiv / date | Control protocol |
|---|---|---|
| RetNet | 2307.08621, Jul 2023 | Matched model sizes (1.3B–13B) vs Transformer, same data/config — **iso-parameter, iso-recipe** |
| RWKV | 2305.13048, May 2023 | Matched params/tokens on The Pile |
| Mamba | 2312.00752, Dec 2023 | Chinchilla-style protocol: 125M–1.3B, **300B tokens**, vs a strong "Transformer++" baseline — **iso-token, iso-size** |
| Griffin | 2402.19427, Feb 2024 | Hawk/Griffin vs MQA-Transformer baseline, **300B tokens**, matched FLOPs at several scales |
| xLSTM | 2405.04517, May 2024 | 15B-token ablations, then **300B SlimPajama tokens** (deliberately same as Mamba/Griffin) at 125M/350M/760M/1.3B; log-log scaling curves — **iso-token + scaling-curve offset** |
| xLSTM Scaling Laws | 2510.02228, Oct 2025 (ICLR 2026) | The modern gold standard: **IsoFLOP profiles + parametric fit**, 80M–7B params, 2B–2T tokens, covering compute-optimal *and* over-training regimes; shows xLSTM Pareto-dominates Transformers in loss-at-iso-FLOP |
| Kimi Linear | 2510.26692, Oct 2025 | **Iso-parameter** vs a full-attention MLA baseline, **1.4T tokens** for both, plus closed-form theoretical FLOPs per head and systems numbers (75% KV-cache reduction, 6× decoding throughput at 1M ctx) |

The field's de-facto standard has converged on: *same tokens, same data, same recipe, multiple model sizes, fit a scaling curve, compare offsets* — with IsoFLOP profiles when budget allows.

### 3.2 Iso-FLOP vs iso-token vs iso-wallclock pitfalls

- **Iso-FLOP**: the Chinchilla frame (Hoffmann et al., **2203.15556**, Mar 2022; NeurIPS 2022): 400+ models, three estimation approaches (fixed-model loss curves; IsoFLOP profiles giving \(N_{opt} \propto C^{0.49}, D_{opt} \propto C^{0.51}\); parametric fit \(L = E + A/N^\alpha + B/D^\beta\)), yielding ~20 tokens/param and the 70B/1.4T Chinchilla. A 2024 replication (Besiroglu et al., **2404.10102**, Apr 2024) found the Approach-3 parametric fit unreliable but confirmed ~20 tok/param from the other two. *Pitfall for novel architectures*: FLOPs are ill-defined (chunk size, active params, recurrence depth — §1.2), and iso-FLOP silently picks a tokens/params allocation that may favor one architecture.
- **Iso-token**: fixes data, lets compute float. Measures **sample efficiency**, not compute efficiency — a heavier-FLOPs architecture wins trivially. Fine as one axis, fatal as the only one.
- **Iso-wallclock**: the most decision-relevant (it's what practitioners actually buy) but couples architecture to kernel maturity — the hardware lottery (2009.06489). Mitigation, which your competition already embodies: **allow custom kernels** so implementation quality is part of the submission, and report MFU/throughput alongside to expose *why* something is fast.
- **Cross-scale instability**: Tay et al., *Scaling Laws vs Model Architectures* (**2207.10551**, Jul 2022; Findings of EMNLP 2023) compared 10 architectures and found **the best architecture flips across compute scales**; the vanilla Transformer had the highest scaling exponent \(\alpha\) even where it lost at small scale. A single-budget comparison (your 6h) is a *slice* — a win means "wins at this budget," not "wins."

### 3.3 The overtrained-small-model confound

- **Data-constrained scaling** (Muennighoff et al., **2305.16264**, May 2023; JMLR 2025): up to **4 epochs of repeated data is nearly free**; value of further repetition decays to zero (\(R_D^* \approx 15\), \(R_N^* \approx 5.3\)). So "train smaller, longer, with repetition" is a legitimate lever, not a cheat.
- **Inference-aware scaling** (Sardana, Portes, Doubov, Frankle, **2401.00448**, Jan 2024; ICML 2024): 47 models at 10–10,000 tokens/param show **no saturation** — quality keeps improving far past Chinchilla-optimal; with any significant inference demand you *should* overtrain a small model.
- **Consequence**: at fixed wall-clock or FLOPs, "architecture A beats B" can be an artifact of A sitting at a better (higher tokens/param) point on the allocation curve. In a competition this is fine — allocation *is part of the entry* — but in the write-up/leaderboard you should report (params, tokens, tokens/param) so observers can tell architecture gains from allocation gains.

---

## 4. Hyperparameter fairness: "architecture wins" vs "tuning wins"

The uncomfortable evidence:

- **Melis et al., 1707.05589** (Jul 2017; ICLR 2018): tuned LSTMs beat most then-novel architectures; many "architecture wins" were "we tuned ours, they didn't tune the baseline."
- **Narang et al., 2102.11972** (Feb 2021; EMNLP 2021): re-implemented ~20 Transformer modifications in one codebase — **most did not meaningfully improve**; the ones that did were minor changes, param increases, or invented in the same codebase. Recommendations: fixed HPs or measured HP robustness, multiple trials with mean/std.
- **Dodge et al., 1909.03004** (Sep 2019; EMNLP 2019): report **expected validation performance as a function of tuning budget** — the honest way to compare when tuning effort differs.
- **Tay et al. 2207.10551** (§3.2): architecture rankings are scale-dependent, compounding the problem.

The principled fix — hyperparameter transfer:

- **μP/μTransfer** (Yang et al., **2203.03466**, Mar 2022; NeurIPS 2022): in Maximal Update Parametrization, optimal HPs (esp. LR) are stable across width; tune on a 13M–40M proxy, transfer zero-shot. Tuned GPT-3 6.7B at **7% of one pretraining run's cost**.
- **A Spectral Condition for Feature Learning** (Yang, Simon, Bernstein, **2310.17813**, Oct 2023): μP reduced to one rule — scale spectral norms of weights and updates like \(\sqrt{\text{fan-out}/\text{fan-in}}\) — that extends to **any architecture and any optimizer**. This is the key enabler for a fair open-architecture competition: participants *can* parametrize novel layers so that small-proxy sweeps transfer. (Empirical caveats in *An Empirical Study of μP Learning Rate Transfer*, 2404.05728, Apr 2024; extensions to Muon/Shampoo/Sophia-class optimizers exist as of ICLR 2026 submissions.)

**How competitions neutralize tuning** — the AlgoPerf model (**2306.07179**) is the best template:

1. **External-tuning ruleset**: submitter defines a search space (same for all workloads), the benchmark runs a fixed parallel budget of trials, and only the *fastest* trial counts — tuning is real but capped and identical across entrants. Score = median over 3 studies of best-trial time-to-target.
2. **Self-tuning ruleset**: no exposed HPs; all adaptation happens **on the clock** inside a single run (with a 1.5× time allowance). Measures total practical cost.

For your competition: require either (a) a declared, capped HP budget with **sweep cost reported in GPU-hours and added to a total-cost ledger** (Dodge-style), or (b) on-the-clock tuning inside the 6h. Publishing the organizer's own baseline sweep on a reference transformer calibrates what "reasonable tuning" buys.

---

## 5. Hardware normalization across GPU SKUs

Peak dense BF16 and memory specs for your pod classes:

| SKU | BF16 dense TFLOPS | Memory | Bandwidth | TDP |
|---|---|---|---|---|
| L40S | 362 | 48 GB GDDR6 | 864 GB/s | 350 W |
| A100 80GB SXM | 312 | 80 GB HBM2e | 2,039 GB/s | 400 W |
| H100 SXM | 989 | 80 GB HBM3 | 3.35 TB/s | 700 W |
| H200 SXM | 989 | 141 GB HBM3e | 4.8 TB/s | 700 W |

**Why TFLOPs-ratio normalization fails**: compute ratio H100/L40S ≈ 2.7×, but *bandwidth* ratio ≈ 3.9×. A bandwidth-bound architecture (SSM scans, small matmuls, custom elementwise kernels) scales with the 3.9× number; a matmul-bound transformer with the 2.7× number. Any single normalization factor systematically (dis)advantages architecture classes — a hardware-lottery amplifier. MFU has the same disease: 40% MFU is excellent for a novel custom-kernel architecture and mediocre for a transformer.

**The MLPerf answer**: don't normalize. MLPerf publishes **absolute** time-to-train per system and partitions results into **divisions** (closed: reference model/optimizer/HPs; open: anything that hits target) and categories (available/preview/research). Comparability comes from the task and rules, not from arithmetic on specs.

**Recommendation: pin one SKU per track.** One leaderboard per GPU (e.g., an L40S track and an H100 track). If you can only afford one, pick the SKU your participants can also rent cheaply (L40S-class) for iteration parity, or H100 for headline relevance. Note that pinning the SKU also pins memory (48 vs 80 GB), which is itself an architecture-relevant constraint — report it as such.

---

## 6. Energy and carbon measurement

- **Tools**: CodeCarbon (mlco2/codecarbon; CPU+GPU+RAM power via RAPL/NVML × regional grid intensity), experiment-impact-tracker (Henderson et al., **2002.05651**, Feb 2020; JMLR 2020), CarbonTracker (Anthony et al., 2007.03051, Jul 2020), ML CO₂ Impact Calculator (Lacoste et al., 1910.09700, Oct 2019).
- **Canonical accounting papers**: Strubell et al. (**1906.02243**, Jun 2019; ACL 2019 — the wake-up call), Patterson et al. (**2104.10350**, Apr 2021 — the "measure the full lifecycle, location matters 5–10×" rebuttal/refinement), Green AI (Schwartz et al., **1907.10597**, Jul 2019).
- **MLPerf Power** (**2410.12032**, Oct 2024; HPCA 2025): the rigorous methodology — power logged **≥1 Hz** at PSU/node level (IPMI/Redfish out-of-band preferred), measurement window aligned to the performance log's run-start/run-stop timestamps, **energy-to-train = ∫ power dt** over that window, Olympic scoring matched to the performance runs.

**Is it worth scoring?** Measure it (it's nearly free: `nvidia-smi --query-gpu=power.draw,utilization.gpu,memory.used,clocks.sm --format=csv -lms 100` or pynvml at 10–100 Hz, integrated to kWh), **report it, but don't score it**. At a fixed SKU and fixed 6h cap, energy correlates strongly with wall-clock; the residual variance (utilization quality) is noisy at nvidia-smi's sampling accuracy, and carbon adds two unknowns you don't control (PUE, grid mix). Put kWh and estimated CO₂e on the metric card, not in the ranking.

---

## 7. Sample efficiency

- **Loss-vs-tokens curves**: log val loss at fixed token intervals; compare curves, not endpoints. This is what xLSTM (2405.04517) and Mamba (2312.00752) actually plot.
- **Tokens-to-threshold**: tokens (or steps) needed to cross a fixed loss — the speedrun's Track 3 (min steps to 3.28; record 2,690 as of 06/2026) is exactly this. Statistically delicate near the threshold (variance inflates), so average over seeds.
- **Intermediate-checkpoint scoring — BabyLM 2025** (Findings, ACL 2025, 2025.babylm-main.28; CFP **2502.10645**, Feb 2025; 2026 CFP 2602.20092): the productionized version. Required checkpoints **every 1M words to 10M, every 10M to 100M, every 100M to 1B**; intermediate checkpoints evaluated on a **"fast" 20% subset** of zero-shot tasks to bound eval cost; full eval (incl. fine-tuning) only on the final model. 2025 also added explicit compute/epoch limits after the 2024 finding that compute, not sample efficiency, was driving scores. Data caps are enforced as **words seen** (counting repetitions), not epochs — the right unit.
- Complementary design: **DataComp-LM** (**2406.11794**, Jun 2024) fixes the training code and compute and varies only data — proof that "fix everything except one axis" competitions produce clean, comparable signal (its 7B baseline matched Llama-3-8B-class quality with 6.6× less compute).

---

## 8. Concrete recommendation for the single-GPU-pod competition (6h cap)

### 8.1 Constraints

1. **Pin one SKU per track** (§5). No cross-SKU normalization — publish separate leaderboards.
2. **Primary constraint: 6h wall-clock, organizer-measured** (orchestrator timestamps, first optimizer step → last). Drop or subordinate the 20k-step cap: **steps are gameable via batch size** (tokens/step is free), so a step cap without a tokens/step cap is not a constraint. Replace with an **organizer-counted token cap** if you want a data axis (BabyLM-style "words seen"), or drop it and let allocation be part of the entry (speedrun-style).
3. **Untimed surface minimized**: data download and image build untimed; **explicit untimed warmup on dummy data**; ban exotic compile flags that shift work into untimed compilation (the speedrun's post-record-21 rule, 02/01/25). Everything else on the clock.
4. **Eval protocol fixed and time-boxed**: organizer-held eval set, fixed sequence length/eval code, capped eval time — otherwise you will get a legitimate-but-unwanted test-time-training exploit (modded-nanogpt PR #205).

### 8.2 What to measure (all organizer-side, streamed to organizer storage)

| Quantity | Mechanism |
|---|---|
| Wall-clock | Orchestrator timestamps; CUDA-event timing inside the run as secondary |
| Tokens consumed | Organizer-side dataloader, fixed seed/order, per-step token counter (this also kills "silently trained on less/more data" ambiguity — note training on *less* data is a legitimate strategy when quality is the metric; the counter makes it *visible*, not forbidden) |
| Power/energy | pynvml or `nvidia-smi -lms 100` sampling of `power.draw`, clocks, utilization; integrate → GPU kWh; RAPL for CPU if available; MLPerf-Power-style alignment to the timed window (≥1 Hz minimum) |
| Peak memory | `torch.cuda.max_memory_allocated` + NVML `memory.used` peak (constraint compliance on 48/80 GB) |
| Throughput | tokens/s from the two counters above |
| FLOPs / MFU | torch.profiler (`with_flops=True`) or kernel-level accounting; where FLOPs are ill-defined (SSM chunk size, MoE, recurrence), report "effective MFU vs reference transformer at same wall-clock" instead of pretending to a number |
| Params | Organizer-side audit: total params, **active params per token** (speedrun rule: MoE fine, cap the active count) |
| Loss trajectory | Per-N-step val loss on organizer data → loss-vs-tokens curve, tokens-to-threshold |
| Intermediate checkpoints | Every ~30 min or ~2B tokens, evaluated on a fast task subset (BabyLM-2025 protocol) |

### 8.3 What to score

- **Primary score: final quality at the fixed budget** — hidden held-out loss (plus a small fixed downstream suite) after 6h on the pinned SKU. This is iso-wallclock: the least gameable, most decision-relevant metric (§1.2, §3.2). The speedrun alternative (fixed target, minimize time) is equally sound but wastes the pod when entries differ in ceiling; quality-at-fixed-budget uses all 6h of signal from every entry.
- **Statistics**: n ≥ 3 seeds for finalists; report mean ± std; Welch t-test for close calls (speedrun's p < 0.01 bar; DAWNBench's analysis showed time/loss-to-target metrics have low CV, so n=3–5 suffices).
- **Cost axis: Pareto, not penalty.** Publish the (quality, wall-clock), (quality, tokens), (quality, kWh) scatter and crown the Pareto frontier; use hypervolume if you need a single "efficiency prize" number. A penalty term (score = quality − λ·log cost) requires choosing λ, which arbitrarily decides how much cost is "worth" — defensible only if you publish λ and its sensitivity. Keep FLOPs **out** of the score entirely (§1.2); it belongs on the metric card as context.
- **Tuning fairness**: adopt AlgoPerf's two rulesets as two divisions — *external-tuning* (declared search space, capped tuning budget, sweep GPU-hours reported and added to a total-cost ledger per Dodge 1909.03004) and *self-tuning* (everything on the clock). Strongly encourage μP/spectral parametrization (2203.03466, 2310.17813) in the participant guide so proxy-scale sweeps transfer.

### 8.4 Anti-gaming checklist

1. Organizer-controlled, seeded, counted data stream; no network in the sandbox.
2. Hidden held-out eval + decontamination + canary sequences to detect eval memorization; public val set for iteration.
3. Fixed, time-boxed eval procedure (blocks eval-time-compute exploits).
4. Untimed-warmup and compile-flag rules (blocks compile-time laundering).
5. Active-parameter and per-token-FLOPs audit (blocks MoE/parameter laundry).
6. Full telemetry (loss curves, grad norms, tokens/step, power) streamed off-pod — anomalies (e.g., loss plateaus consistent with skipped steps, clock throttling games) are detectable post hoc.
7. **Organizer re-runs the top-k submissions** from the submitted container before awarding — the single most effective deterrent (Patterson et al. 2104.10350: requiring release of enough information to recreate results was benchmarking's effective deterrent to fudging).
8. Significance testing before ranking close entries (p < 0.01).

**Bottom line**: pin the hardware, fix the budget in wall-clock and tokens measured on your side of the API, score quality-at-budget with statistical rigor, present cost as a Pareto axis (wall-clock / tokens / kWh / peak memory) rather than a FLOPs penalty, force tuning into the open with AlgoPerf-style rulesets, and re-run the winners. FLOPs, params, and MFU are context for the write-up — never the score.
