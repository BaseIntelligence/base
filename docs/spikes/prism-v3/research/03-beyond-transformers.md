# Appendix 03 — Beyond the Transformer: Architecture Survey 2023–2026
> Research appendix for the Prism v3 evaluation proposal (`docs/spikes/prism-v3/`). Produced 2026-08-06 via arXiv/web research. Non-normative spike document.

# Beyond the Transformer: A Technical Survey of Alternative Architectures (2023–2026)

**Purpose:** input to the design of an open competition where participants invent better sequence-modeling architectures. Focus: arXiv work 2023–2026, verified via web search on 2026-08-06.

**Evidence labels used throughout:**
- **[Replicated]** — independent labs/teams have reproduced or deployed the result.
- **[Single-lab]** — strong paper, but results come from one group; open code/weights may exist.
- **[Vendor claim]** — numbers from a model release/blog; treat with caution.
- **[Hype-flagged]** — significant public attention with credible critiques or failed replications.

**One-paragraph state of the field (Aug 2026):** No pure non-attention architecture has displaced the Transformer at the frontier, but the *hybrid* thesis has decisively won in 2025–2026: Qwen3-Next (Gated DeltaNet + gated attention), Kimi Linear (KDA + MLA), IBM Granite 4.0 (Mamba-2 + attention, 9:1), NVIDIA Nemotron-H, TII Falcon-H1, MiniMax-M1 (lightning attention), and DeepSeek-V3.2 (learned sparse attention) all ship non-standard attention in production-scale models. The newest credible results: **Mamba-3** (Mar 2026), **TTT-E2E** (Dec 2025, first method to match full-attention loss-vs-context scaling with constant latency), **Nested Learning/HOPE** (NeurIPS 2025), and **Kimi Linear** (Oct 2025, first linear-attention model to beat full attention in fair iso-recipe comparisons). Automated architecture discovery (ASI-Arch, AlphaEvolve, PostNAS) is now real and directly relevant to competition design.

---

## 1. State-Space Models (Mamba lineage)

**Key papers**
| Work | arXiv | Date |
|---|---|---|
| S4 (Structured State Spaces) | 2111.00396 | Oct 2021 (foundational) |
| S4D (diagonal S4) | 2206.12037 | Jun 2022 |
| S5 (simplified, parallel scan) | 2208.04933 | Aug 2022 |
| H3 (Hungry Hungry Hippos) | 2212.14052 | Dec 2022 |
| Mamba (selective SSM, "S6") | 2312.00752 | Dec 2023 |
| Mamba-2 (SSD, "Transformers are SSMs") | 2405.21060 | May 2024 |
| **Mamba-3** (trapezoidal discretization, complex states, MIMO) | 2603.15569 | Mar 2026 |
| HGRN / HGRN2 | 2304.09887 / 2404.07904 | Apr 2023 / Apr 2024 |

**Core mechanism.** A linear recurrence \(h_t = A_t h_{t-1} + B_t x_t\), \(y_t = C_t h_t\) with input-dependent (selective) parameters; the recurrence is computed as a parallel associative scan during training (sequence-parallel) and as an O(1)-state RNN during decoding. Mamba-2's State-Space Duality (SSD) shows SSMs and (masked) linear attention are the same object viewed as recurrence vs. quadratic form, enabling chunkwise hardware-efficient kernels. Mamba-3 adds a second-order "exponential-trapezoidal" discretization (removing the short conv), complex-valued state tracking via a data-dependent RoPE trick (restoring state-tracking ability lost when Mamba-1 went real-valued), and a multi-input multi-output (MIMO) formulation that improves quality at equal decode latency.

**Efficiency vs Transformer (iso-compute).** Mamba-2 2.7B matched/beat Transformer++ 2.7B trained on the same tokens [Replicated]; ~5× higher decode throughput than a same-size Transformer, constant memory (no KV cache). Mamba-3 at 1.5B: +0.6 pts avg downstream over Gated DeltaNet, +1.8 pts total with MIMO; matches Mamba-2 perplexity with half the state size [Single-lab, but from the Mamba authors with open kernels in `state-spaces/mamba`]. Training throughput is on par with attention thanks to chunkwise kernels.

**Known weaknesses.** Exact associative recall / copying is provably capacity-limited by state size: "Repeat After Me" (2402.01032, copying), "RNNs are not Transformers (Yet)" (2402.18510, in-context retrieval bottleneck), "The Illusion of State" (2411.12512, state-tracking hardness for SSMs/linear RNNs) [all Replicated]. Empirically weak on MQAR-style recall (Zoology, 2312.04927) and long-context retrieval vs. attention. Length generalization beyond training context is unreliable. Mitigations: hybrid attention layers, delta-rule updates (§2), larger/complex states (Mamba-3).

**Maturity: High.** Production deployments: AI21 Jamba (2403.19887; Jamba-1.5 2408.12570), Codestral Mamba, Falcon-Mamba, NVIDIA Nemotron-H (2504.03624), IBM Granite 4.0 (9:1 Mamba-2:attention, Oct 2025), Tencent Hunyuan-T1/TurboS. Distillation from Transformers works (MOHAWK 2408.10189; "The Mamba in the Llama" 2408.15237).

**Competition substrate: Excellent.** Trainable at 100M–3B by small teams (mature Triton kernels, FLA library, nanoGPT-scale recipes); clear headroom (state size, discretization, gating); Mamba-3 shows the design space is still moving in 2026.

---

## 2. Linear RNNs / Gated Recurrence (delta-rule family)

**Key papers**
| Work | arXiv | Date |
|---|---|---|
| RetNet (retention mechanism) | 2307.08621 | Jul 2023 |
| RWKV-4 | 2305.13048 | May 2023 |
| RWKV-5/6 ("Eagle/Finch") | 2404.05892 | Apr 2024 |
| **RWKV-7 "Goose"** (generalized delta rule) | 2503.14456 | Mar 2025 |
| RWKV-8 "Heron" (DeepEmbed, DeepEmbedAttention, ROSA suffix automaton) | no arXiv; GitHub/wiki previews | May 2025– , experimental |
| xLSTM | 2405.04517 | May 2024 |
| **xLSTM 7B** (2.3T tokens, DCLM) | 2503.13427 | Mar 2025 (ICML 2025) |
| Griffin / Hawk | 2402.19427 | Feb 2024 |
| RecurrentGemma | 2404.07839 | Apr 2024 |
| DeltaNet | 2102.11174 | Feb 2021 (foundational) |
| GLA (gated linear attention) | 2312.06635 | Dec 2023 |
| **Gated DeltaNet** | 2412.06464 | Dec 2024 |
| DeltaProduct | 2502.10297 | Feb 2025 |
| GateLoop | 2311.01926 | Nov 2023 |
| Longhorn (online-learning SSM) | 2407.14235 | Jul 2024 |
| Log-Linear Attention | 2506.04761 | Jun 2025 |
| **Kimi Linear / KDA** | 2510.26692 | Oct 2025 |

**Core mechanism.** These are linear attention / gated RNNs where the state update implements online regression — the delta rule ("erase then write": \(S_t = S_{t-1}(I - \beta k k^\top) + \beta v k^\top\)) — giving a finite-state memory with much better recall than additive linear attention. Gated DeltaNet adds a per-channel forget gate; RWKV-7 generalizes this with vector-valued gating and "in-context learning rates"; Kimi Delta Attention (KDA) adds finer-grained channel-wise gating plus a specialized diagonal-plus-low-rank chunkwise kernel. xLSTM instead modernizes the LSTM cell (exponential gating, matrix memory, parallelizable "mLSTM"); Griffin combines a real-gated linear RNN (RG-LRU) with local sliding-window attention.

**Efficiency vs Transformer (iso-compute).**
- Griffin-3B beat a Transformer baseline at 1T tokens; Griffin-14B ≈ Llama-2-13B at ~6× fewer tokens [Single-lab, Google].
- xLSTM 7B: competitive with Llama/Mistral-class 7–8B models on downstream tasks at only 2.3T tokens, with the highest prefill/decode throughput and lowest memory of any 7B tested [Single-lab, open weights].
- RWKV-7 2.9B: claims parity with Qwen2.5-3B-class models, especially multilingual [Single-lab/community].
- **Kimi Linear (48B total / 3B active, 3:1 KDA:MLA): the first linear-attention model to outperform full attention under identical training recipes** — short context, long context, and RL post-training — with 75% KV-cache reduction and up to 6× decoding throughput at 1M context [Single-lab but open kernels + checkpoints; corroborated by Qwen3-Next's independent adoption of the same Gated DeltaNet hybrid recipe].
- Qwen3-Next-80B-A3B (3:1 GDN:gated-attention): ≈ Qwen3-32B dense at <10% of training GPU-hours, >10× throughput beyond 32K context [Vendor claim, open weights].

**Known weaknesses.** Same theory-level recall/state-tracking ceiling as SSMs (2402.18510, 2411.12512), though delta-rule models are the strongest finite-state family on recall. xLSTM needed gate soft-capping and re-init to avoid loss spikes at 7B. RWKV lags on English reasoning vs. top Transformers at equal size; RWKV-8 remains experimental (no paper, toy-scale ROSA demos: 1M-param 40-digit arithmetic). Pure versions still lose to hybrids on retrieval-heavy evals — every serious 2025–26 deployment keeps some full-attention layers.

**Maturity: High and rising** — this family (not vanilla Mamba) is what frontier-adjacent labs actually adopted in late 2025 (Qwen3-Next, Kimi Linear). RetNet itself is largely superseded by delta-rule descendants.

**Competition substrate: Excellent — arguably the best.** The FLA (flash-linear-attention) ecosystem gives small teams Triton kernels for GLA/DeltaNet/GDN/KDA variants; the gating/update-rule design space is wide, formally grounded (online optimization view; see MIRAS in §4), and proven to yield wins at 100M–3B.

---

## 3. Sub-quadratic Attention and Hybrid Stacks

### 3a. Long-convolution / implicit filters
- **Hyena** (2302.10866, Feb 2023): implicit long convolutions parameterized by an MLP + elementwise gating; sub-quadratic via FFT. **HyenaDNA** (2306.15794) scaled to 1M context at single-nucleotide resolution; **StripedHyena-7B** (2311.09446, Together AI) was the first credible open non-attention 7B. **Evo / Evo 2** (Arc Institute; bioRxiv 2024/2025) use StripedHyena-class backbones for genomics.
- **Status:** [Replicated in bio], but for general LM it lost the efficiency war to SSMs/delta-nets (FFT convs are awkward on GPU vs. chunkwise scans). **Maturity: medium-low for LM; niche in genomics.** Competition substrate: moderate.

### 3b. Sliding-window and static sparse attention
- Mistral 7B (2310.06825) popularized SWA; Gemma 2 (2408.00118) and Gemma 3 (2503.19786) alternate local:global layers (1:1 and 5:1); GPT-OSS (Aug 2025) alternates banded/full. Longformer (2004.05150) / BigBird (2007.14062) are the classical static-sparse lineage; LongNet (2307.02486) dilated attention.
- **Status:** [Replicated everywhere] — boring but unbeaten cost/benefit; SWA is a component of nearly every hybrid (Samba, TTT-E2E). Weakness: hard context limit, no cross-window retrieval without global layers.

### 3c. Learned sparse attention (2025's big attention story)
| Work | arXiv | Date | Idea |
|---|---|---|---|
| NSA (Native Sparse Attention, DeepSeek) | 2502.11089 | Feb 2025 | Hierarchical: compressed + selected + sliding branches, hardware-aligned, trained end-to-end |
| MoBA (Kimi) | 2502.13189 | Feb 2025 | MoE-style block routing of attention; validated up to ~7B and 10M-context extrapolation |
| **DSA (DeepSeek Sparse Attention)** | 2512.02556 (V3.2 report); V3.2-Exp release Sep 2025 | Sep–Dec 2025 | Per-token "lightning indexer" scores all past tokens, attends to top-2048; instantiated under MLA |

DSA is the landmark: DeepSeek-V3.2-Exp continued-training a 685B model with DSA at **parity with the dense V3.1-Terminus baseline** across benchmarks, then shipped it as DeepSeek-V3.2 (GPT-5-class reasoning, IMO/IOI gold) with >50% API price cut [Vendor claim, but open weights + kernels in TileLang/CUDA]. Inference-time sparsity (MInference 2407.02490, Quest 2406.10774, DuoAttention 2410.10819, SeerAttention 2410.13276, XAttention 2503.16428) is adjacent but not architectural. Attention-variant micro-innovations worth knowing: Differential Transformer (2410.05258), Selective Attention (2410.02703), Multi-Token Attention (2504.00927), Forgetting Transformer (2503.02130), TPA (2501.06425), MLA (DeepSeek-V2, 2405.04434), GQA (2305.13245).

**Maturity: High (DSA is production);** weaknesses: indexer training complexity, kernel engineering burden, top-k selection can miss rare retrieval.

### 3d. Hybrid SSM/linear-RNN + attention stacks
| Work | arXiv | Date | Recipe |
|---|---|---|---|
| Jamba (AI21) | 2403.19887 | Mar 2024 | 1:7 attention:Mamba + MoE, 52B-A12B, 256K ctx |
| Samba (Microsoft) | 2406.07522 | Jun 2024 | Mamba + SWA interleaved, 3.8B, near-perfect passkey to 256K |
| Zamba / Zamba2 (Zyphra) | 2405.16712 / 2411.15242 | May/Nov 2024 | Mamba(+2) backbone + shared attention blocks |
| Hymba (NVIDIA) | 2411.13676 | Nov 2024 | Attention + Mamba heads in parallel per layer |
| Bamba (IBM+Princeton+CMU+UIUC) | 2412.15255 | Dec 2024 | Mamba-2 hybrid, 9B, distillation-friendly |
| MiniMax-01 / **MiniMax-M1** | 2501.08313 / 2506.13585 | Jan/Jun 2025 | 7:1 lightning-attention (TransNormerLLM lineage, 2307.14995; Lightning Attention-2 2401.04658) : softmax; 456B-A45.9B, 1M ctx, RL at 25% of R1's FLOPs @100K gen |
| Nemotron-H (NVIDIA) | 2504.03624 | Apr 2025 | Mamba-2 + few attention layers, 8B/56B, ≥ Llama-3.1-8B/70B accuracy at up to 3× throughput |
| Falcon-H1 (TII) | 2507.22448 | Jul 2025 | Parallel attention‖Mamba-2 heads per block, tunable ratio, 0.5B–34B; 34B ≈ 70B-class |
| **Qwen3-Next** | blog Sep 2025 (Qwen3 report: 2505.09388) | Sep 2025 | 3:1 GDN:gated-attention, 80B-A3B, 512 experts |
| **Kimi Linear** | 2510.26692 | Oct 2025 | 3:1 KDA:MLA, beats full attention iso-recipe |
| **Granite 4.0 (IBM)** | blog Oct 2025 | Oct 2025 | 9:1 Mamba-2:attention, no positional encoding, 350M–32B-A9B, −70% memory, 2× inference |
| Jet-Nemotron (NVIDIA) | 2508.15884 | Aug 2025 | PostNAS-searched hybrid; JetBlock linear attention; 2B beats Qwen3/Qwen2.5/Gemma3/Llama3.2 at 53.6× decode throughput (256K ctx) |

**Verdict:** hybrids are the **de facto winner of 2023–2026** [Replicated across ≥7 independent orgs]. Weaknesses: residual retrieval gap vs. full attention at extreme context (Kimi Linear claims to have closed it at their scale); two heterogeneous memory systems complicate serving; ratio/placement choices are still empirical folk knowledge (PostNAS is a first systematic answer).

**Competition substrate: Excellent** — the ratio/placement/mixer design space is huge, cheap to explore at 100M–3B, and directly transferable to frontier practice.

---

## 4. Neural Memory and Test-Time Compute

| Work | arXiv | Date | Note |
|---|---|---|---|
| Memorizing Transformers | 2203.08913 | Mar 2022 | kNN lookup over past activations (foundational) |
| RETRO | 2112.04426 | Dec 2021 | retrieval-chunk cross-attention (foundational) |
| RMT | 2207.06881 / 2304.11062 | 2022–23 | recurrent memory tokens |
| LongMem | 2306.07174 | Jun 2023 | decoupled memory network sidecar |
| Infini-attention | 2404.07143 | Apr 2024 | compressive memory + local attention |
| MemoryLLM | 2402.04624 | Feb 2024 | self-updatable latent memory pool |
| Larimar (IBM) | 2403.11901 | Mar 2024 | episodic memory as Kanerva-style SDM; fast fact editing |
| **Memory Layers at Scale (Meta)** | 2412.09764 | Dec 2024 | product-key memory layers replace FFNs; up to 128B memory params, 1T tokens; >2×-compute dense models and iso-param MoE beaten on factual QA |
| **TTT layers** | 2407.04620 | Jul 2024 | hidden state = weights of an inner model updated by gradient descent at test time; TTT-Linear/MLP beat Mamba & Transformer++ at 125M–1.3B |
| **Titans** | 2501.00663 | Jan 2025 | deep neural long-term memory with surprise (gradient)-gated writes + forgetting; >2M context; beats Mamba-2/GDN/Transformer++ at ≤760M; BABILong SOTA |
| **MIRAS** (Moneta, Yaad, Memora) | 2504.13173 | Apr 2025 | unifying framework: sequence model = associative memory + attentional-bias objective + retention gate + update rule; new variants beat linear RNNs on recall tasks |
| **ATLAS** | 2505.23735 | May 2025 | higher-capacity memory, sliding-window (non-online) updates, Muon-optimized memory; +80% over Titans at 10M-context BABILong; "DeepTransformers" generalization |
| MesaNet | 2506.05233 | Jun 2025 | locally optimal test-time training layer |
| **SEAL (MIT)** | 2506.10943 | Jun 2025 | RL-trained self-edits → LoRA weight updates; knowledge incorporation 33.5%→47.0% on SQuAD-no-context |
| **Nested Learning / HOPE (Google)** | 2512.24695 | Dec 2025 (NeurIPS 2025) | model = nested multi-level optimization problems; continuum memory system; self-modifying Titans variant; better LM + continual learning + long-context than Titans at ~1.3B |
| **TTT-E2E (Stanford/NVIDIA)** | 2512.23675 | Dec 2025 | plain SWA Transformer that keeps doing next-token SGD on the context at test time, with meta-learned init; **first method to match full-attention loss-vs-context scaling while keeping RNN-like constant latency** (2.7× faster than full attention at 128K, 35× at 2M); Mamba-2/GDN do *not* match that scaling |

**Core mechanism.** Instead of a fixed-size state updated by a hand-written rule, the memory is itself a small network (or a weight subset) updated by gradient descent on an internal objective at inference; "surprise" (gradient norm) gates writes; meta-learning at train time prepares the model for this inner loop. MIRAS shows Transformers, linear RNNs, and Titans are all instances of one associative-memory schema — a genuinely useful unifying lens for architecture invention.

**Weaknesses.** [Single-lab] almost across the board (Google for Titans/MIRAS/ATLAS/NL; NVIDIA/Stanford for TTT-E2E) — independent replications at scale are still thin. Gradient-at-inference complicates serving; meta-training needs grad-of-grad (TTT-E2E: 3.4× slower training at 8K due to FlashAttention lacking second-order support); online-update stability and catastrophic forgetting are active problems (NL/HOPE directly target them); benchmark contamination risk when models learn at test time.

**Maturity: Low-medium but the fastest-moving frontier** (five major papers in 2025 alone). **Competition substrate: Very strong for a research-y track** — huge, formally grounded design space (objective × retention × update rule × memory architecture), demonstrated wins over Transformer++ at exactly the 100M–3B competition scale, but higher engineering risk than §1–3.

---

## 5. Mixture-of-Experts Advances

| Work | arXiv | Date | Contribution |
|---|---|---|---|
| Switch / GLaM / ST-MoE | 2101.03961 / 2112.06905 / 2202.08906 | 2021–22 | foundations, stability, routing z-loss |
| Expert-Choice routing | 2211.15841 | Nov 2022 | experts pick tokens; perfect load balance, no token dropping |
| Soft MoE | 2308.00951 | Aug 2023 | fully differentiable soft merging |
| Sparse Upcycling | 2312.07526 | Dec 2023 | dense→MoE checkpoint conversion |
| Mixtral 8×7B | 2401.04088 | Jan 2024 | open-weight MoE landmark |
| **DeepSeekMoE** | 2401.06066 | Jan 2024 | **fine-grained experts + shared experts** — the two ideas everyone now copies |
| JetMoE | 2404.07413 | Apr 2024 | MoE everywhere (attention too), $0.1M training |
| PEER | 2407.04152 | Jul 2024 | product-key retrieval over ~1M tiny experts |
| Aux-loss-free balancing | 2408.15664 | Aug 2024 | bias-term balancing without auxiliary loss |
| OLMoE | 2409.02060 | Sep 2024 | fully open MoE recipe (1B-A7B) |
| DeepSeek-V3 | 2412.19437 | Dec 2024 | 671B-A37B, 256 routed + 1 shared, aux-loss-free + seq-aux, MTP |
| Qwen3 | 2505.09388 | May 2025 | 235B-A22B, 128 experts, no shared expert (interesting reversal) |
| **Kimi K2** | 2507.20534 | Jul 2025 | 1T-A32B, **384 fine-grained experts + 1 shared, sparsity 48**; scaling-law case for higher sparsity; MuonClip (QK-clip) for stability at 15.5T tokens, zero loss spikes |
| GLM-4.5 | 2508.06471 | Aug 2025 | 355B-A32B fine-grained MoE |
| Granite 4.0 Tiny/Small | (IBM, Oct 2025) | Oct 2025 | fine-grained MoE + shared experts inside a Mamba hybrid |
| Qwen3-Next | (Sep 2025) | Sep 2025 | 512 experts, 10+1 active — extreme sparsity (3.7%) |

**Consensus advances [all heavily Replicated in frontier training]:** fine-grained expert segmentation, 1–N always-on shared experts, aux-loss-free (bias) balancing, upcycling, and sparsity scaling laws (K2: sparsity 48 cuts FLOPs ~1.7× vs sparsity 8 at equal loss). Expert-choice routing remains academically interesting but lost to token-choice + bias balancing in practice.

**Weaknesses:** total-parameter memory tax (all experts resident), all-to-all comms, routing pathology at small scale, kernel maturity for fine-grained experts. **Iso-compute:** strictly better loss/FLOP than dense — this is settled.

**Maturity: Highest of any family here** (it *is* the frontier). **Competition substrate: Good but with caveats** — small teams can train 100M–3B MoEs (OLMoE/JetMoE recipes), and combining MoE with hybrid attention/memory is underexplored; but pure-MoE innovation is incremental and big-lab-dominated. Related efficiency axes: Mixture-of-Depths (2404.02258), MatFormer (2310.07707), BitNet-style 1.58-bit (2402.17764).

---

## 6. Diffusion LMs and Non-Autoregressive Paradigms

| Work | arXiv | Date | Note |
|---|---|---|---|
| SEDD | 2310.16834 | Oct 2023 | score-entropy discrete diffusion |
| MDLM | 2406.07524 | Jun 2024 | simple masked-diffusion recipe |
| DiffuLLaMA | 2410.17891 | Oct 2024 | AR→diffusion conversion |
| **LLaDA** | 2502.09992 | Feb 2025 | 8B masked diffusion trained from scratch ≈ LLaMA3-8B on several benchmarks; no reversal curse |
| Block Diffusion | 2503.09573 | Mar 2025 | AR-between-blocks, diffusion-within-block |
| d1 | 2504.12216 | Apr 2025 | RL post-training for dLLMs |
| Fast-dLLM / v2 | 2505.22618 / 2509.26328 | May/Sep 2025 | training-free ~10× decode speedup; block-wise |
| MMaDA / LaViDa | 2505.15809 / 2505.16839 | May 2025 | multimodal dLLMs |
| **Mercury (Inception Labs)** | 2506.17298 | Jun 2025 | commercial diffusion coder; 737–1109 tok/s on H100, ~5–10× faster than speed-optimized frontier AR at comparable quality [Vendor claim, Artificial Analysis-verified throughput] |
| Dream | 2508.15487 | Aug 2025 | 7B diffusion, AR-initialized |
| Seed Diffusion (ByteDance) | 2508.02193 | Aug 2025 | code dLLM, 2146 tok/s on H20 |
| DiffuCoder | 2506.20624 | Jun 2025 | code dLLM + coupled GRPO |
| LLaDA-MoE | 2509.24389 | Sep 2025 | sparse MoE dLLM |
| **LLaDA2.0** | 2512.15745 | Dec 2025 | **100B MoE dLLM via 3-phase AR→block-diffusion conversion** (WSD schedule), confidence-prediction auxiliary loss for parallel decoding; 16B-mini and 100B-flash, open |
| **LLaDA2.1** | 2602.08676 | Feb 2026 | token-to-token editing + mask-to-token joint decoding; first large-scale RL framework for dLLMs; 663–892 tok/s on coding benchmarks at 100B |
| SDAR | ACL 2026 Findings (Cheng et al.) | 2025–26 | systematic AR→block-diffusion conversion 1.7B–30B; AR backbones beat masked-diffusion backbones for hybrids; 2.3× wall-clock on H200 |
| Mercury 2 | blog, Feb 2026 | Feb 2026 | first reasoning dLLM; ~1009 tok/s on Blackwell; AIME'25 91.1 [Vendor claim] |
| Gemini Diffusion | blog, May 2025 | — | Google consumer dLLM experiment |

**Core mechanism.** Train a denoiser over masked/noised tokens; at inference, iteratively unmask many tokens in parallel (optionally block-wise left-to-right). Speed comes from parallel commits; quality scales with denoising steps, giving a tunable speed/quality knob AR lacks.

**Weaknesses [Replicated]:** quality still trails top AR at equal size (math/reasoning especially, though LLaDA2.x closed much of the gap); no KV cache in the AR sense (each denoise step reprocesses); variable-length generation is awkward; RL post-training is immature (LLaDA2.1 is the first large-scale attempt); throughput claims depend heavily on batch and acceptance thresholds.

**Maturity: Medium and rising fast** — now commercial (Mercury 2, Gemini Diffusion) and at 100B scale (LLaDA2.x). **Competition substrate: Strong** — MDLM/SEDD/LLaDA codebases are open and small-scale-trainable; the speed/quality Pareto gives an objective second axis to compete on; many unsolved problems (cache, length, RL, editing).

---

## 7. Exotic Lines — Dead Ends vs. Promising

| Line | Key refs | Status & verdict |
|---|---|---|
| **KAN (Kolmogorov-Arnold Networks)** | 2404.19756 (Apr 2024); KAN 2.0 2408.10205 | **[Hype-flagged → mostly dead end for LM].** Fair iso-param/iso-FLOP comparisons (2407.16674) show KAN ≈ a special MLP with learnable spline activations; wins only on symbolic-formula tasks; *worse* catastrophic forgetting than MLPs; vulnerable to noise (2408.07906). Survives as a component in science/SLU niches, not as an LM backbone. |
| **Liquid networks** | LTC 2006.04439; CfC 2208.08647; Liquid-S4 2209.12951; **LFM2 tech report 2511.23404** (Nov 2025) | **Pivot tells the story:** Liquid AI's production LFM2 (350M–8.3B) dropped ODE dynamics for hardware-searched **gated short convolutions + GQA** — i.e., converged to the hybrid mainstream. ODE-based LTC: dead for LM; CfC survives in robotics/edge. LFM2 itself (2× faster CPU prefill/decode, 10–12T tokens) is a credible edge model [Vendor claim, open weights]. |
| **Hypernetworks / weight generation** | 1609.09106; HyperFormer++ 2110.01591; GHN-2 2110.13100; GHN-3 2410.18155; G.pt 2410.11082; "Neural Network Diffusion" 2402.13144 | **Research-curious, not an LM substrate.** GHN-3/G.pt predict usable parameters for unseen architectures; useful for NAS-in-weights and competition tooling (fast surrogate evaluation), but no evidence of beating trained Transformers. |
| **Energy-based approaches** | **EBT 2507.02092** (Jul 2025); energy-based diffusion LM 2502.10256 | **[Single-lab, speculative].** EBTs claim up to 35% faster pretraining scaling than Transformer++ and System-2-style gains from extra inference optimization, modality-agnostic. Bold claims, one paper, iterative inference cost; worth watching, not yet a substrate. LeCun's JEPA line (I-JEPA 2301.08243, V-JEPA 2404.08471/2506.09985) is a representation-learning alternative, not an LM. |
| **Bayesian / probabilistic** | Bayesian Flow Networks 2308.07037 | Niche; BFN influenced discrete-diffusion thinking but no LM-scale wins. |
| **Evolved / searched architectures** | Evolved Transformer 1901.11117; Primer 2109.08668; AutoML-Zero 2003.03384 | Historical proof that search finds real gains (Primer's squared ReLU stuck). See §8 for the 2025–26 revival. |
| **Sparse Distributed Memory / Hopfield** | Modern Hopfield Networks 2008.02217; SDM-continual 2211.02373; Larimar 2403.11901 | **Intellectually important** (Hopfield≡attention equivalence; SDM's read/write locality) and Larimar shows fast editable episodic memory, but no competitive LM at scale. Promising as a *component* (editable memory), not a backbone. |
| **Capsule networks** | 1710.09829 and follow-ups | **Dead end** for sequence modeling — routing cost, no scaling evidence, community inactive. |
| **Forward-Forward / non-backprop** | 2212.13345; predictive-coding training work | **Dead end at LM scale** so far; no credible path to 100M+ LM quality. |
| **Neural ODEs / continuous depth** | 1806.07366 | Dead for LM (solver cost); ideas absorbed into diffusion/flow. |
| **Spiking / neuromorphic** | various | Niche hardware play; no LM-relevant results. |
| **Recurrent-depth / latent reasoning** | Universal Transformer 1807.03819; ACT 1603.08983; Looped Transformers 2301.13196; Relaxed Recursive 2410.16672; **Huginn-3.5B** 2502.05171; **MoR** 2507.10524 (NeurIPS'25); **HRM** 2506.21734; **TRM** 2510.04871; CTM 2505.05522; COCONUT 2412.06749; Quiet-STaR 2403.09629; CODI 2502.21074 | **Genuinely promising, with one hype caution.** MoR: iso-FLOP Pareto wins at 135M–1.7B (router-assigned per-token recursion depth + recursion-wise KV cache, 2× throughput) [Single-lab, open code]. Huginn: recurrent-depth 3.5B scales with test-time loops. **HRM [Hype-flagged]:** ARC Prize replication found the hierarchy contributed little; gains came from the outer refinement loop + task augmentation, largely transductive memorization; TRM (7M params, one 2-layer net) then beat it (ARC-AGI-1 45%, ARC-AGI-2 8%) — real but puzzle-domain, not LM. CTM (Sakana, 2505.05522): beautiful neuron-level temporal dynamics, not competitive. |
| **Tokenizer-free / concept-level** | BLT 2412.09871; LCM 2412.08821; MambaByte 2401.13660 | BLT (byte-level, entropy-patch) is credible and replicated-ish at 8B; LCM (sentence-latent) is [Single-lab, Meta]. Orthogonal axis, combinable with any backbone. |

---

## 8. 2025–2026 Frontier: Automated Architecture Discovery

This is the most competition-relevant development: **architecture invention itself is being automated and scaled.**

| Work | arXiv | Date | Result |
|---|---|---|---|
| ADAS (Automated Design of Agentic Systems) | 2408.08435 | Aug 2024 | meta-agent programs new agents |
| AI Scientist v1/v2 (Sakana) | 2408.06292 / 2504.08066 | Aug 2024 / Apr 2025 | end-to-end automated research; v2 got a workshop-accepted paper |
| EvoPrompting / LLMatic | 2309.08532 / 2406.16130 | 2023–24 | LLM-driven NAS |
| Darwin-Gödel Machine | 2505.22954 | May 2025 | self-modifying coding agents |
| **AlphaEvolve (DeepMind)** | 2506.13131 | Jun 2025 | evolutionary LLM coding agent; 48-mult 4×4 complex matmul (first improvement over Strassen in that setting in 56 yrs), data-center scheduling, TPU circuit simplification, **sped up training of the LLM underpinning itself**; matched best-known on 75% of ~50 open math problems, beat 20% |
| **ASI-Arch ("AlphaGo Moment for Model Architecture Discovery")** | 2507.18074 | Jul 2025 | **Fully autonomous multi-agent system that ran 1,773 experiments / 20,000 GPU-hours and discovered 106 novel SOTA linear-attention architectures**, with emergent design principles surpassing human-designed baselines; claims first empirical *scaling law for scientific discovery itself*. (This is almost certainly the "ASHA-discovery" line referenced in the request.) [Single-lab, open-sourced framework + architectures] |
| **Jet-Nemotron / PostNAS** | 2508.15884 | Aug 2025 | architecture search *starting from a pretrained Transformer* (freeze MLPs, learn attention-layer placement, select/design linear blocks, hardware-aware HP search) — cheap enough for small teams; produced JetBlock |
| ArchAgent | 2602.22425 | Feb 2026 | AlphaEvolve-style discovery for CPU cache policies; found simulator loopholes ("simulator escapes") — a warning for automated competition evaluation |
| **Mamba-3** | 2603.15569 | Mar 2026 | human-designed, but shows the SSM design space still yields structured gains |
| Survey: "The End of Transformers?" | 2510.05364 | Oct 2025 | critical survey of sub-quadratic challengers; conclusion: no pure challenger yet, hybrids winning |

**Theory corner (matters for competition evaluation):** finite-state models (SSM/linear RNN) are provably weaker at state tracking and in-context retrieval (2411.12512, 2402.18510); copying needs Ω(state) capacity (2402.01032); CoT formally expands Transformer power (2405.17313); log-precision Transformers sit in uniform TC⁰ (2402.09268). Implication: **a pure finite-state model cannot fully replace attention — hybrid or unbounded-memory designs are the theoretically sound competition target.**

---

## 9. Credible Candidates for an Architecture-Invention Competition — Ranked

Ranking criteria: (a) demonstrated ability to beat a strong Transformer baseline at 100M–3B iso-compute; (b) trainability by small teams (open kernels/datasets/recipes); (c) size of unexplored design space; (d) evaluation robustness; (e) frontier relevance.

| Rank | Substrate | Why | Risk |
|---|---|---|---|
| **1** | **Hybrid linear-RNN/SSM + attention stacks** (delta-rule family: GDN/KDA/Mamba-3 blocks + sparse/SWA/full attention) | The empirically winning paradigm of 2025–26 (Kimi Linear, Qwen3-Next, Granite 4.0, Nemotron-H, Falcon-H1, Mamba-3). FLA/mamba-ssm kernels, open recipes, huge space (update rule × gating × ratio × placement × head mixing). ASI-Arch proved even automated search finds SOTA here at ~20K GPU-hours — perfect competition scale. | Low risk, but crowded — bar for novelty must be set high (e.g., must beat a Kimi-Linear-style 3:1 hybrid reference, not a vanilla Transformer). |
| **2** | **Neural memory / test-time-training layers** (Titans/MIRAS/ATLAS/HOPE/TTT-E2E design space) | Largest *formally grounded* open space (memory arch × attentional bias × retention × update rule); repeated small-scale wins over Transformer++ and Mamba-2; TTT-E2E's constant-latency + full-attention-scaling result (Dec 2025) suggests a real breakthrough direction. | Medium-high: grad-at-inference engineering, stability, fewer open kernels, contamination-aware eval needed. Best as the "high-risk/high-reward" track. |
| **3** | **Masked / block diffusion LMs** | Open codebases (MDLM, LLaDA, Fast-dLLM), trainable at 100M–3B, second optimization axis (parallel decode speed) makes for a richer competition objective; many unsolved problems (cache, editing, RL, length). | Medium: quality gap vs AR persists; evaluation must include speed, not just perplexity. |
| **4** | **Recurrent-depth / adaptive-compute Transformers** (MoR, Huginn, TRM-style) | Cheap to train, iso-FLOP wins at exactly competition scale (135M–1.7B), parameter-efficiency story is strong; TRM shows tiny nets + recursion + deep supervision is underexplored. | Medium: sequential depth hurts batched inference; HRM episode shows evaluation must punish transductive/augmentation gaming. |
| **5** | **Learned sparse attention** (NSA/MoBA/DSA-style) | Production-validated at the highest level (DeepSeek-V3.2); clear compute savings with parity. | Medium-high for small teams: kernel engineering is the real cost; reference PyTorch implementations are slow, biasing against good ideas. |
| **6** | **Fine-grained MoE + memory layers** (PEER, Memory-Layers-at-Scale, shared experts) | Reliable iso-FLOP gains, easy to train, combinable with ranks 1–2. | Low ceiling on "beyond-Transformer" novelty; better as a composition axis than a standalone track. |
| **7 (wildcard track)** | **Exotics:** EBT (2507.02092), SDM/Larimar-style editable memory, hypernetwork weight-gen, KAN hybrids, liquid/CfC, tokenizer-free (BLT) | Where a genuinely new idea could come from; EBT and editable-memory are the only ones with plausible LM upside. | High: KAN/capsule/FF/Neural-ODE are effectively dead ends for LM; expect most entries to fail — price accordingly. |

**Competition design recommendations (evidence-backed):**
1. **Make the baseline a hybrid, not a vanilla Transformer** — beating 2017 attention is no longer informative; use a strong Transformer++ *and* a 3:1 delta-net/attention hybrid reference at 100M/1B/3B iso-FLOP tiers.
2. **Two-axis scoring:** quality (perplexity + downstream + recall suite: MQAR/Zoology, RULER, BABILong, copying/state-tracking probes per 2402.01032/2411.12512) × efficiency (decode throughput + memory at long context, measured wall-clock, not FLOPs only).
3. **Mandate theory-aware evals:** pure finite-state models have provable ceilings; require retrieval/state-tracking probes so entries can't win by silently dropping capabilities.
4. **Guard against gaming:** ArchAgent's "simulator escapes" (2602.22425) and the HRM/ARC episode (transductive memorization, augmentation-driven gains) are direct warnings — use held-out tasks, forbid test-time training on eval data unless that's the declared mechanism, and audit outer-loop tricks.
5. **Allow a PostNAS/conversion track** (Jet-Nemotron, MOHAWK, Mamba-in-the-Llama, SDAR): starting from a pretrained checkpoint is the cheapest legitimate path and mirrors how the field actually moves.
6. **Feasibility is proven:** ASI-Arch's 106 SOTA linear-attention discoveries at 20K GPU-hours, nanoGPT-speedrun culture, and the FLA ecosystem show 100M–3B architecture invention is squarely within small-team budgets.

**Bottom line:** as of Aug 2026 the credible "beyond Transformer" story is not a single replacement but a convergence — gated delta-rule/SSM state + a few attention layers + fine-grained MoE + (increasingly) learned test-time memory — with diffusion LMs as the leading non-AR paradigm and automated discovery (ASI-Arch, AlphaEvolve, PostNAS) as the meta-trend most likely to produce whatever comes next.
