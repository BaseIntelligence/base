# Appendix 13 — From Small-Scale Winner to Frontier Relevance
> Research appendix for the Prism v3 evaluation proposal (`docs/spikes/prism-v3/`). Produced 2026-08-06 via arXiv/web research. Non-normative spike document.

# What Small-Scale (100M–3B) Architecture Research Must Demonstrate to Matter at the Frontier

*An honest assessment of the transfer path, for a competition whose ambition is that winners could eventually challenge Opus-class / GPT-5-class models. All arXiv IDs and dates cited inline; key 2025–2026 claims verified against primary sources on 2026-08-06.*

---

## 1. Evidence of small→large transfer: what actually made it

The historical record is real but heavily filtered. For every small-scale result that transferred, dozens did not. The successes share a pattern I return to throughout: **they were validated at ≥1B with matched recipes, shipped with kernels, and solved an efficiency problem the frontier labs were already feeling.**

**Confirmed transfers (small → production frontier):**

| Innovation | Discovery scale | Frontier adoption | arXiv / date |
|---|---|---|---|
| Gated DeltaNet | ~1.3B-class linear-attn research line (DeltaNet lineage; Gated DeltaNet 2412.06464, Dec 2024) | **Qwen3-Next-80B-A3B** (2505.09388 line; released Sep 2025): 3:1 GDN:Gated-Attention hybrid, 80B total / 3B active, 15T tokens; carried forward into Qwen3.5 (~397B-A17B, 2026) | 2412.06464 (Dec 2024) |
| Kimi Delta Attention (KDA) | 1.4T-token iso-recipe comparisons at ~3B-A3B | **Kimi Linear 48B-A3B** (2510.26692, Oct 30, 2025): 3:1 KDA:MLA hybrid, 5.7T tokens, −75% KV cache, 6.3× decode throughput at 1M ctx, claims **first linear-attn model to beat full attention under fair comparison** | 2510.26692 (Oct 2025) |
| Mamba-2 / SSD layers | 130M–2.7B (Mamba 2312.00752, Dec 2023; Mamba-2 2405.21060, May 2024) | Production hybrids: NVIDIA Nemotron-H family, Falcon-Mamba, Codestral-Mamba; Zamba2; IBM Granite 4.0 hybrids. Notably **not** in the largest frontier flagships — it survives mostly *as the minority component of hybrids* | 2312.00752 / 2405.21060 |
| MoE fine-graining (fine-grained experts + shared-expert isolation) | DeepSeekMoE at 2B–16B (2401.06066, Jan 2024) | DeepSeek-V2 (2405.04434) → V3 (2412.19437, 671B-A37B) → **Kimi K2** (2507.20534, Jul 2025, 1T-A32B): fine-grained experts + shared expert is now the default open-frontier MoE recipe | 2401.06066 (Jan 2024) |
| MLA (multi-head latent attention) | DeepSeek-V2 (2405.04434, May 2024), validated at 236B-A21B directly | DeepSeek-V3/R1, **Kimi K2/K2.5**, GLM-5 — MLA is now the dominant KV-cache-efficient attention at the open frontier; it is also the *global-attention component* inside Kimi Linear | 2405.04434 (May 2024) |
| μP (maximal update parametrization) | Tensor Programs V (2203.03466, Mar 2022), toy-to-small scale | Hyperparameter transfer used in GPT-4-era training (publicly described by Greg Yang / Microsoft); foundational to Moonshot's **muon-clip / Muon at 1T scale** in Kimi K2 | 2203.03466 (Mar 2022) |
| Squared-ReLU (ReLU²) | Primer search at ~1.9B-equiv TPU-days (2109.08668, Sep 2021) | Standard in the NanoGPT speedrun; appears in modern frontier MLP variants (e.g., used widely in fast-training recipes); a genuine small-scale-search win | 2109.08668 (Sep 2021) |
| Muon optimizer | 124M NanoGPT speedrun (Keller Jordan, late 2024) | **Kimi K2 (1T-A32B)** and **GLM-4.5** trained with Muon — the single clearest "speedrun → frontier" artifact, per METR's 2026-04-21 analysis of the speedrun record history | kellerjordan.github.io/posts/muon (2024) |

**The counter-examples (small-scale wins that died, or stalled, at scale):**

- **Pure Mamba/SSM as full-attention replacement.** Mamba matched transformers at ≤2.7B on perplexity, but recall-heavy tasks (MQAR, needle) degrade; "The Illusion of State in State-Space Models" (2404.08819, Apr 2024) formalized why finite-state models can't do state tracking that attention does trivially. Every serious production deployment ended up **hybrid**, keeping 25% full attention. The pure-SSM dream at frontier scale is dead as of 2026.
- **Early linear attention (Performer, Linear Transformer, cosFormer lineage).** Wins at 100M–1B on loss/FLOP, but TTT-E2E (2512.23675, Dec 2025) shows plainly that Mamba-2 and Gated DeltaNet **fail to scale with context length in loss** the way full attention does — the gap *opens* at 64K–128K. Pure linear attention's "matched at 1B" claims were artifacts of 2K–4K eval contexts.
- **Hyena (2302.10866)**, long-conv family: competitive at small scale, no frontier adoption; subsumed by delta-rule linear attention.
- **Most NAS-discovered architectures** (Evolved Transformer 1901.11117 onward): wins within the search distribution evaporate when data/recipes change. Architecture search overfits to its eval harness — a direct warning for competition design.
- **RetNet (2307.08621)**: Microsoft's "retention" promised transformer parity; small-scale numbers held, but no second lab reproduced at scale and it was quietly abandoned.
- **Primer itself**: only ReLU² survived contact with the field; the searched full architecture did not.

**Reading of the record:** transfer happens for *mechanism-level* innovations (a better attention variant, a better optimizer update rule, a better MoE routing structure, a parametrization that makes HPs transfer) — almost never for *wholesale architecture replacements*. And it happens when the innovation attacks a **cost axis the frontier cares about** (KV cache, long-context throughput, HP-tuning cost, optimizer steps) more often than a pure loss-at-fixed-compute axis.

---

## 2. What frontier labs actually adopt

**The "validated at 1B, adopted at 100B" pattern is real but has a specific shape.** Frontier labs run internal ladders roughly like: 10M–100M for mechanism sanity → 1B–3B × ~1T tokens for iso-recipe comparisons → 10B–30B pilot → flagship. Kimi Linear is the public exemplar: 1.4T-token matched-recipe runs at 3B-A3B against full MLA, *then* 48B-A3B at 5.7T tokens (2510.26692, Oct 2025). Qwen3-Next ran "systematic experiments" across SWA/Mamba2/GDN at small scale before committing the 80B flagship to GDN 3:1 (Sep 2025). The bar was never "won at 350M"; it was "won at 3B under our recipe, with the gap stable or growing."

**Why frontier training is conservative — risk asymmetry.** A 1T-parameter run costs tens of millions of dollars and weeks of cluster time; a failed run is a quarter-delay event. This produces rational risk aversion:

- An architecture change must clear not "is it better" but **"is it better with ≥95% confidence of not blowing up at 10× my validation scale"** — loss spikes, grad explosions, MoE router collapse, and instability bugs scale superlinearly with model size. Qwen3-Next's tech notes explicitly list "training-stability-friendly optimizations" (zero-centered weight-decayed layernorm etc.) as a *first-class feature* of the architecture, not an afterthought.
- **Ecosystem switching costs are real**: a novel block with no fast kernel is a 2–5× wall-clock tax on every experiment forever. Adoption follows kernels.
- **Reputational asymmetry**: nobody gets fired for shipping a well-tuned dense-MoE transformer; someone gets fired for shipping an exotic architecture that underperforms at 500B.

**The kernel/ecosystem moat — arguably the dominant adoption variable.** Gated DeltaNet and KDA are production-viable *because* of **FLA** (fla-org/flash-linear-attention) — Songlin Yang's chunkwise Triton kernels are literally what both Qwen3-Next and Kimi Linear trained with (Kimi Linear's open-sourced KDA kernel lives in FLA; verified 2026-08-06). Mamba spread via **mamba-ssm** CUDA kernels (Tri Dao). MLA spread because DeepSeek open-sourced both weights and inference code, and vLLM/SGLang absorbed it. Counter-proof: RetNet had Microsoft backing but no community kernel ecosystem, and died. **For a competition: a winner without a fused, correct, benchmarked kernel is not an architecture — it's a suggestion.** This is a hard requirement, not a nice-to-have.

**Also note the quiet truth about what "adoption" means in 2026:** the frontier has *converged on heterogeneity*. Qwen3.5 = GDN hybrid; GLM-5 = MLA + DeepSeek Sparse Attention; Kimi = KDA:MLA hybrid; MiniMax-M2.5 = full attention "for reliability." There is no single winner — labs adopt **modular mechanisms they can mix at chosen ratios**. Competition winners should be designed as *drop-in blocks*, not whole-model bets.

---

## 3. Scaling-exponent evidence: does the gap grow or shrink?

This is the scientific core of the credibility question, and the literature is more nuanced than either hype or dismissal:

**"Same exponent" results (architecture choice doesn't change the loss-vs-compute slope):**

- *Scaling Laws for Linear Complexity Language Models* (EMNLP 2024, aclanthology 2024.emnlp-main.916): TNL/HGRN2/cosFormer2 vs LLaMA, 70M→7B, 300B tokens, 1,376 checkpoints — linear-complexity models show **similar scaling capability** to transformers on loss and downstream tasks.
- *Solvable Attention for Neural Scaling Laws* (ICLR 2025) and follow-on theory: with fixed context length, architecture choice moves the **constant (offset)**, not the exponent, of the compute scaling law.
- Multimodal-side replication (2604.10064): fitted exponents for linear vs softmax attention within noise of each other.

**"Different exponent / diverging gap" results (the counterweight):**

- **TTT-E2E (2512.23675, Dec 2025, NVIDIA)** is the most important recent datapoint: at 3B × 164B tokens, full attention's loss keeps improving with context length; **Mamba-2 and Gated DeltaNet flatten** (their loss-vs-full-attention gap grows from 8K→128K); only TTT-E2E maintains the same loss slope as full attention while keeping O(1) decode latency (2.7× faster at 128K, ~35× at 2M per NVIDIA's technical blog). I.e., **for the context-length axis, the exponent differs** — and this is exactly the axis frontier labs now compete on.
- *Scaling Context Requires Rethinking Attention* (2507.04239, Jul 2025): state size, not parameter count, controls in-context learning scaling; RWKV gets "almost no benefit" from added context beyond its state capacity. Pure recurrent models hit a state-capacity wall.
- *Fundamental Limitations on Subquadratic Alternatives* (2410.04271, Oct 2024): theory-side impossibility results for subquadratic architectures matching attention's expressivity on core reasoning tasks.

**Kimi Linear's answer to this (2510.26692) is the current state of the art in "credible scaling evidence":** not pure linear attention (they *concede* the pure form loses), but a 3:1 hybrid, compared **iso-recipe** (same data, same tokens, same HPs) against full MLA at 3B-A3B × 1.4T tokens, across short-context, long-context, *and* RL-scaling regimes, with the efficiency advantage shown *growing* with sequence length (2.3× at 512K, 2.9× at 1M prefill; 6× decode at 1M). That is what "the gap grows with compute, in our favor, on the axis that matters" looks like when done properly.

**What would make a frontier lab take a competition winner seriously (empirically grounded):**

1. Iso-recipe, iso-compute comparison against a *strong contemporary baseline* (not GPT-2-era transformer; today that means MLA + fine-grained MoE + modern norm/position stack), at **minimum two scales separated by ≥10×** (e.g., 400M and 3B) with the gap constant or widening.
2. A **loss-vs-context-length curve** out to ≥64K, because that's where small-scale winners most often silently die (TTT-E2E Fig. 1 is the template).
3. Fit actual scaling exponents (loss vs C over ≥3 compute points) and show the challenger exponent ≥ baseline exponent, not just a better constant.
4. Evidence on a **recall/state-tracking probe suite** (MQAR, RULER, needle) — the known graveyard axis of subquadratic models (2404.08819, 2507.04239).
5. Seeds: ≥3, with variance reported. Single-seed small-scale architecture wins are usually noise; the NanoGPT speedrun community now demands Welch-test significance for sub-millinit differences (see WR #82, May 2026).

---

## 4. The capability gap, honestly

What frontier models have that no 350M model can exhibit — and whose property it is:

| Capability | Architecture property? | Verdict |
|---|---|---|
| Long-horizon agency (multi-hour task chains) | Mostly **post-training (RL) + scale** | Not testable at 350M. Architecture contributes only via long-context stability. |
| RL post-training gains (o1/K2-style reasoning RL) | Partially architecture-adjacent | Kimi Linear notably evaluated **RL scaling** iso-recipe (2510.26692) — this is now the bar. A competition can't run meaningful RLVR at 350M, but can require the winner to survive RL fine-tuning at 3B validation without collapse. |
| Instruction following / chat | **Post-training + data** | Nothing architectural; untestable in competition. |
| Tool use / function calling | Post-training + long-context | Untestable except as long-context probe. |
| Multimodality | Architecture (encoders) + scale | Out of scope for a text-architecture competition. |
| Long-context retrieval & reasoning | **Genuinely architectural** (state size, cache) | Testable — this is the one frontier-relevant axis a small competition *can* legitimately claim. |
| Loss/compute efficiency (pretraining) | **Genuinely architectural** | Testable — the classic and honest claim. |
| Training stability at scale | Architectural (norms, residuals, parametrization) | Partially testable: require μP-style HP transfer 100M→3B with zero retuning, plus spike-free loss curves. |
| Knowledge capacity | Scale (parameter count) | Not architectural; a 350M winner has ~zero world knowledge, by design. |

**What the competition can honestly claim:** "we search for architectures with better loss/compute constants, better loss-vs-context scaling, better HP transfer, and better throughput — the properties that are (a) measurable at 100M–3B and (b) historically the ones that transferred (§1)."

**What it cannot claim:** "our winner beats Claude Opus / GPT-5." Frontier capability in 2026 is **>70% data, scale, and post-training** (RLVR, agentic RL, tool-use training, multimodal alignment) and <30% architecture — arguably less. Kimi Linear 48B-A3B, the best-validated architecture innovation of 2025, is *not* a frontier-capability model; it's an efficiency play that Moonshot adopted because it preserves their capability curve at 6× lower decode cost. Architecture innovation at this point **buys efficiency and headroom, not capability miracles**. Any miner-facing messaging that implies otherwise is hype, and sophisticated miners/labs will discount the competition accordingly.

---

## 5. From winning architecture to frontier model: the pipeline

The credible path from a 350M competition winner to something a frontier lab would pilot:

**Stage 0 — Competition win (350M):** beat baseline under fixed data/compute/HP budget; ≥3 seeds; recall probes; open weights + code.

**Stage 1 — Paper-quality evidence package (3B validation):**
- Iso-recipe runs at ≥2–3 scales (400M / 1.5B / 3B), ≥100B–500B tokens at the top tier, gap constant-or-widening (§3).
- Loss-vs-context curves to 64K+; MQAR/RULER probes.
- Stability: loss-spike count vs baseline; μP HP-transfer 100M→3B with zero retuning; grad-norm telemetry.
- **Open fused kernel** (Triton/CUDA) with correctness tests and a vLLM/FLA-style integration — the FLA/mamba-ssm precedent (§2) makes this non-negotiable.
- Ablations isolating each novel component (single-component wins are adoptable; entangled 7-change architectures are not).

**Stage 2 — Distillation/conversion short-circuit (the underappiated path):** a new block doesn't need a from-scratch flagship to prove itself. **MOHAWK** (2408.10189, Aug 2024) distilled Phi-1.5 into Mamba-2 with 3B tokens; **Mamba-in-the-Llama** (2408.15237, Aug 2024) linearized Llama-3-8B retaining most capability at a fraction of training cost; **Jet-Nemotron/PostNAS** (2508.15884, Aug 2025; NeurIPS 2025) is the industrial version: freeze a pretrained transformer's MLPs, learn optimal full-attention placement, swap in a novel linear block (JetBlock), hardware-aware search — 2B model matching Qwen3-1.7B accuracy with **53.6× decode throughput at 256K**. PostNAS proves a competition winner can be *grafted into an existing frontier-adjacent model* for ~1/100th the cost of pretraining — this should be a required Stage-2 deliverable.

**Stage 3 — 30B pilot:** 30B-A3B-class MoE, ~1T tokens, iso-recipe vs the lab's current recipe. This is the Qwen3-Next/Kimi-Linear gate.

**Stage 4 — Frontier flagship:** someone else's $50M+.

**Compute estimates (2026 prices, H100-class at ~$2–3/GPU-hr, MFU ~40–50%):**

| Tier | Scale | Tokens | ≈ H100-hours | ≈ Cost |
|---|---|---|---|---|
| Competition | 350M × 20B tok | — | ~400–800 | ~$1–2K |
| 3B validation | 3B × 300B tok (+ 2 smaller tiers, 3 seeds) | — | ~30–60K total | **~$80–150K** |
| Kernel + PostNAS graft | 2–8B graft onto open model | ~50–100B tok | ~10–25K | ~$30–60K |
| 30B pilot | 30B-A3B × 1T tok | — | ~300–500K | **~$1–2M** |

**Who funds it:** realistically — (a) the subnet treasury / emissions-funded compute grants (Bittensor precedent: subnets like Templar/pretraining subnets pool miner compute, though quality-control is hard); (b) foundation-model grant programs (Google TRC, NVIDIA academic grants, Lambda/Together research credits, OpenRouter/a16z-style ecosystem funds); (c) a frontier or near-frontier lab that adopts the winner as a *research collaboration* — which is exactly how GDN (MIT/NVIDIA lineage) got into Qwen. The honest model: the competition produces the **0→1 evidence**; the treasury or a sponsor produces the **3B replication**; a lab produces 30B+. No crypto incentive scheme has yet funded a credible $1M+ 30B pilot — treat that stage as "partnership-gated," not "treasury-gated."

---

## 6. Competition positioning precedents

| Competition | Positioning vs frontier | What it achieved | Lesson |
|---|---|---|---|
| **BabyLM** (2023–, CoNLL/shared task) | Explicitly *not* frontier — cognitively-plausible data-limited LM (≤100M words) | Genuine research lineage (LTG-BERT, curiosity-driven curricula), NeurIPS/ACL papers, academic prestige; zero frontier adoption | Framing honesty built durable credibility. It never claimed frontier relevance, so it never lost it. |
| **NanoGPT speedrun** (modded-nanogpt, May 2024–) | "Fastest GPT-2 training," explicitly engineering-scale | The strongest precedent for us: **Muon → Kimi K2 (1T) and GLM-4.5** (METR 2026-04-21 analysis); 77+ records, 45min→<90s, strict reproducibility protocol, per-record significance testing; spawned value embeddings, Paired Head Attention, Polar Express | Works because: fixed target, fixed hardware, open PRs, brutal reproducibility. Transfer happened for *mechanisms*, and the winners were people labs then hired/collaborated with. |
| **ARC Prize** (ARC-AGI-1, $1M+, 2024–2025) | Deliberately orthogonal to frontier — abstraction/reasoning benchmark | Enormous prestige; drove test-time-compute discourse (o3's 2024 result came *through* ARC-AGI). But: leaderboard-topping solutions did not transfer as products; frontier labs consumed the *benchmark*, not the winners | A competition can shape frontier *thinking* without its artifacts shipping. Prestige ≠ adoption. |
| **MLCommons (MLPerf Training)** | Industry benchmarking consortium, frontier labs participate | Became the neutral referee for *hardware/training-speed*, not architecture discovery | Standardization prestige comes from neutrality and rigor, not from prizes. |
| **Bittensor pretraining subnets** (e.g., pretraining/PTN-style) | Crypto-incentivized loss competitions at ~1B | Real decentralized training progress but no architecture transfer to frontier; incentive-gaming (data leakage, eval overfitting) is a persistent problem | Crypto incentives amplify both participation *and* gaming. Mechanism design (held-out evals, reproducibility gates) decides which dominates. |

**Net lessons:** (1) the only precedent with a verified small→frontier artifact (speedrun/Muon) had **extreme reproducibility discipline and open code**; (2) prestige tracks honesty of framing; (3) crypto-incentivized versions need anti-gaming as a first-class design constraint.

---

## 7. Concrete recommendation

### A. Required evidence package from winners (make these prize-release gates)

1. **Multi-tier scaling runs**, not one 350M number: winner must submit runs at **3 compute tiers ≥10× apart** (e.g., 100M × 5B tok, 350M × 20B tok, 1.5B × 100B tok), iso-recipe vs the competition baseline (which itself must be a *2026-strength* reference: MLA-style attention, fine-grained MoE or strong dense, RoPE/QK-norm/modern norm stack — refresh the baseline yearly or results are meaningless).
2. **Scaling-exponent fit**: loss-vs-compute exponent reported over the 3 tiers; claim validity requires challenger exponent ≥ baseline exponent (gap constant or growing). A win that's a shrinking constant gets a smaller prize tier — encode this in the reward function.
3. **Loss-vs-context curve** at the top tier (8K→64K minimum): this is where small-scale winners silently die (TTT-E2E 2512.23675 Fig. 1; 2507.04239). Include MQAR/RULER-style recall probes.
4. **Stability report**: ≥3 seeds with variance; zero loss spikes unexplained; **μP HP transfer demonstrated from 100M to 1.5B with zero retuning** — the single cheapest proxy for "won't blow up at 100× scale."
5. **Open kernel**: fused Triton/CUDA implementation, unit-tested against a reference, benchmarked on H100-class hardware, submitted to FLA or equivalent. No kernel, no top prize. (§2: this is the actual adoption bottleneck.)
6. **Ablations**: each novel component isolated. Frontier labs adopt single mechanisms (KDA gate, MLA compression, Muon update), never entangled stacks.
7. **PostNAS-style graft demo** (2408.15237 / 2508.15884 lineage): distill/graft the winning block into an open 2–8B model and show capability retention. This is the cheapest possible "this actually works in a real model" proof and massively de-risks lab adoption.

### B. Milestone ladder (with go/no-go gates)

- **M0 — Competition (350M):** as now, plus anti-gaming: held-out eval set, sealed data, reproducibility re-runs of top-3 submissions by organizers. *Cost: existing.*
- **M1 — 1.5B replication (organizer/treasury-funded, ~$15–30K):** rerun winner iso-recipe at 1.5B × 100B tokens. **Gate:** gap holds within noise → publish as a preprint with the miner as lead author. *This preprint is the real prize* — it's what a lab reads.
- **M2 — 3B validation + kernel ecosystem (~$80–150K):** 3B × 300B tokens, context-scaling curves, FLA merge, vLLM integration. **Gate:** publishable scaling story + merged kernel → actively shop to labs. *Funding: treasury grant + cloud research credits (TRC/NVIDIA/Lambda-style); this is the right ask size for those programs.*
- **M3 — 30B pilot (~$1–2M):** only via **lab partnership or ecosystem-fund sponsorship**; do not pretend the treasury funds this. Structure it as: the competition provides evidence + miner; partner provides compute; co-authored tech report (the DeepSeekMoE/Kimi-Linear model).
- **M4 — Frontier:** out of the competition's hands, by definition.

### C. Honest framing for miners (put this in the docs verbatim)

> Architecture is roughly the efficiency-and-headroom layer of frontier models, not the capability layer. No 350M model demonstrates agency, instruction following, or reasoning — those come from scale, data, and post-training. What a winning architecture *can* do is what Gated DeltaNet did in Qwen3-Next (Sep 2025), KDA did in Kimi Linear (2510.26692), MLA did across the DeepSeek lineage (2405.04434), and Muon did in Kimi K2 (2507.20534): get adopted as a component inside frontier models because it demonstrably scales better or cheaper under matched recipes. "Challenging Opus" means your block ends up inside an Opus-class model — via our evidence ladder and a lab partnership — not that a 350M winner beats anything. Competitions that promised the latter (most NAS, pure-SSM, RetNet-era claims) produced nothing; competitions that promised the former (NanoGPT speedrun → Muon) changed what frontier labs train with.

**The brutal summary:** the transfer path is real but narrow — mechanism-level, efficiency-angled, kernel-backed, iso-recipe-validated at ≥1B–3B, and stability-proven. Build the competition's reward function and prize gates around exactly those five properties, budget ~$100–150K/season for the 3B validation tier, and position M3+ as partnership-gated. That is the maximum a small-scale architecture competition can credibly be — and it is, per the 2024–2026 record, genuinely enough to matter.
