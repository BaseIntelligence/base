# Appendix 05 — Evaluating Long-Context Capability
> Research appendix for the Prism v3 evaluation proposal (`docs/spikes/prism-v3/`). Produced 2026-08-06 via arXiv/web research. Non-normative spike document.

# Evaluating Long-Context Capability of Language Model Architectures
### Survey for an open architecture-invention competition (100M–3B params, from scratch, ≤6h GPU, ≤4k training context) — as of 2026-08-06

Replication flags: **[R]** = replicated across ≥2 independent labs; **[1L]** = single-lab claim (treat as promising, not established).

---

## 1. Benchmark landscape

| Benchmark (arXiv, date) | What it measures | 2026 saturation status | Regenerable for private eval? | Fit for 100M–3B, ≤4k-train |
|---|---|---|---|---|
| Vanilla NIAH (Kamradt, GitHub, Nov 2023) | Single fact retrieval at depth × length grid | **Saturated at frontier since 2024** [R]; GPT-4/Gemini-1.5 >99% | Trivially (own needle + haystack) | Useful as smoke test only; too easy alone |
| Passkey retrieval (2306.14000, Jun 2023) | Retrieve random 5-digit key in filler text | Saturated at frontier [R]; still standard in arch papers | Trivially | Good — base-model friendly, near-zero instruction burden |
| **RULER** (2404.06654, Apr 2024; COLM'24) | 13 synthetic tasks in 4 categories: NIAH variants (single/multi-key/multi-value/multi-query), variable tracking, common/frequent words extraction, QA | Frontier avg now ~94.7 (Nemotron-3 Ultra, Jul 2026) → **saturating at default config at frontier**, but **highly discriminative at small scale**; Llama-3.1-8B scores ~98 at 128k while 100M models trained at 4k collapse past 8k [R] | **Yes — fully procedural**, configurable length/complexity; official generator public | **Excellent** — the de facto standard for architecture papers (Kimi Linear, Mamba-3, GDN all report it); completable by base models |
| **HELMET** (2410.02694, Oct 2024; ICLR'25) | 7 categories: synthetic recall (RULER tasks), long-doc QA, summarization, many-shot ICL, RAG, re-ranking, citation; 59 LCLMs studied | Still discriminating at 128k on Cite/Re-rank categories even for frontier; NIAH category saturated [R]. Key replicated finding: **NIAH scores do not predict downstream performance**; categories mutually decorrelate | Partially — synthetic/RAG/ICL parts regenerable; QA/summarization use fixed public datasets (NarrativeQA, InfiniteBench subsets, ALCE) | Good partially — designed to support **base models via few-shot**; natural subsets will be near-floor for 1B models |
| LongBench v1 (2308.14508, Aug 2023) | 21 bilingual tasks (QA, summarization, few-shot, code), mostly 4k–32k | **Largely saturated/legacy** at frontier [R]; static public sources → contamination | No (fixed, public since 2023) | Poor — built from instruction-era tasks |
| **LongBench v2** (2412.15204, Dec 2024) | 503 expert-written MCQs, 8k–2M words, 6 categories; human experts 53.7% under 15-min limit | **Not saturated**: top = Qwen3.8 Max 66.3%, Claude Opus 4.5 64.4% (Aug 2026) [R] | No (fixed set, public) | **Poor** — MCQ with long reasoning; a 1B base model is at chance (~25%); no signal for this competition |
| InfiniteBench (2402.13718, Feb 2024) | >100k-token tasks: En.MC/QA, code debug, math calc, passkey/number-string/KV retrieval, dialogue | Retrieval subsets **saturated** (passkey ~100%); En.MC still 30–60% even at frontier → discriminative but too hard for small models [R] | Retrieval/KV subsets procedural; QA/MC fixed | Partial — only the KV/retrieval subsets usable |
| **BABILong** (2406.10149, Jun 2024; NeurIPS'24 D&B) | bAbI-style reasoning (fact chaining, deduction, counting, lists/sets) scattered in PG-19 distractor text, 0k–10M tokens | Not saturated; GPT-4-class models use only ~10–20% of context [R]. **Crucially: 130–137M fine-tuned models (Mamba-130M, RMT/ARMT GPT-2) solve tasks up to 1M+ tokens** [R for task solvability at small scale; ARMT 50M-token claim 1L] | **Yes — "leak-proof" by design**: bAbI facts are generated, haystack splicing randomized | **Best fit in the entire list** for 100M–3B from-scratch models |
| LOFT (2406.13121, Jun 2024; NAACL'25 Findings) | Corpus-in-context: retrieval, RAG, SQL-like reasoning, ICL over 35 datasets at 32k/128k/1M; same queries across lengths | Not saturated at 1M; Gemini-class LCLMs rival specialized retrievers (Gecko) at 128k [R] | Partially — same-query-across-lengths design is procedural, but requires corpus downloads and 32k+ to be interesting | Moderate — adaptable at 4k–32k, heavy to set up |
| L-Eval (2307.11088, Jul 2023) | 20 subtasks over 411 long docs, LLM-judged | **Legacy**; judge-based metrics noisy; superseded by HELMET/LongBench v2 [R] | No | Skip |
| ZeroSCROLLS (2405.12196, May 2024) | 10 natural zero-shot tasks (QA, summarization, aggregation) | Partially saturated at frontier; static public data → contamination [R] | No | Skip except as optional natural anchor |

**Not in your list but essential in 2026:**

- **NoLiMa** (2502.05167, Feb 2025; ICML'25): NIAH with **minimal lexical overlap** between question and needle — kills the literal-match shortcut. 11 of 13 frontier models dropped below 50% of their short-context baseline at 32k; GPT-4o 99.3%→69.7% [R, corroborated by Chroma's Context Rot study]. Template design is reusable with fresh fact pairs.
- **MRCR / Michelangelo** (2409.12640, Sep 2024, Google) and **OpenAI MRCR** (HF `openai/mrcr`, Nov 2024): 2/4/8 needles drawn from the **same distribution as distractors**, requiring order discrimination ("return the 2nd poem about tapirs"). The Michelangelo authors explicitly recommend MRCR as the default NIAH replacement, noting it has **high signal for small and non-post-trained models**. DeepMind's `eval_hub` mrcr_v2 ships a **generator** so you can mint fresh instances [R as frontier standard; generator single-source].
- **GraphWalks** (HF `openai/graphwalks`, Apr 2025; bugfix Feb 2026): BFS/parent queries over random graphs given as edge lists. Procedural, multi-hop, exact-answer, and shown to stress even frontier models at 128k chars. Trivially regenerable with fresh seeds.
- **Context Rot** (Chroma/TMLS technical report, 2025): 18 models, shows degradation with length even on trivially simple tasks; coherent haystacks are *harder* than shuffled ones — a caution against over-clean synthetic haystacks [1L as a report, but the degradation phenomenon itself is R].
- **Distractor-Aware Truncation** (2608.03297, Aug 2026 — days old): naive middle-truncation confounds "shorter is better" claims by destroying signal; distractor-aware truncation preserves gold answers. Methodological warning for anyone building length-controlled evals [1L, brand new].

---

## 2. Length extrapolation methodology

**Setup.** Train at \(L_{\text{train}} \in \{512, 1\text{k}, 2\text{k}, 4\text{k}\}\); evaluate at \(\{2\times, 4\times, 8\times, 32\times\}\) multiples. Three measurement axes, in increasing order of cost:

1. **PPL-vs-position curves.** Plot mean token loss against absolute position \(p\) on held-out long documents, using **non-overlapping full-context forward passes** — never stride-windowed PPL, which flatters models by resetting context. This is the standard plot in Position Interpolation (2306.15595, Jun 2023) and YaRN (2309.00071, Sep 2023, ICLR'24), and it is the *primary* comparison in TTT-E2E (2512.23675, Dec 2025).
2. **Task accuracy vs. length grid** on procedural tasks (§3), depth randomized.
3. **Effective context length.** RULER's definition: max \(L\) where score ≥ a fixed baseline (Llama-2-7B at 4k = 85.6) [R]. For a competition of weak-but-comparable models, use a **self-normalized** version — \(L^* = \max\{L : \text{score}(L) \ge 0.9 \cdot \text{score}(1\text{k})\}\) — **plus an absolute floor** (e.g. score must also exceed a trivial baseline), otherwise a model that is uniformly bad at all lengths gets an inflated \(L^*\). A loss-based analog: \(L_{\text{eff}} = \max\{L : \Delta\text{loss}(L \to 2L) > \epsilon\}\), i.e. the point where doubling context stops helping.

**Known extrapolation behaviors by family:**

| Family | Behavior past \(L_{\text{train}}\) | Evidence |
|---|---|---|
| RoPE attention, no intervention | Loss **explodes** within ~10–20% past \(L_{\text{train}}\) (OOD rotation angles → attention logit blow-up). RoPE scores do not provably decay with distance, explaining the failure | [R] PI 2306.15595; YaRN 2309.00071; "Round and Round We Go" 2410.06205 (ICLR'25); Kazemnejad et al. 2305.19466 (NeurIPS'23) |
| ALiBi (2108.12409, ICLR'22) | **No explosion** — smooth, monotone degradation; but recency bias systematically hurts retrieval of far content; downstream task accuracy drops well before PPL does | [R] original + RULER/BABILong evaluations |
| YaRN / NTK-by-parts / PI | Extends 4–16× **with a small amount of long-context fine-tuning**; zero-shot ("dynamic NTK") is partial. Now standard practice (Llama/Qwen families) | [R] YaRN 2309.00071 and widespread industry adoption |
| NoPE / hybrid NoPE | Best *zero-shot* decay behavior among PE schemes in controlled studies; adopted e.g. by Kimi Linear's MLA layers | [R] Kazemnejad 2305.19466; [1L] Kimi Linear ablation |
| SSM (Mamba-2, 2405.21060) / linear attention (GDN, 2412.06464) | **No positional OOD → no cliff.** Degradation is gradual and appears in *retrieval/recall*, not in average PPL: loss-vs-context curves **flatten** — the model stops improving as context grows, while full attention keeps improving | [R] for the flattening phenomenon (MQAR literature, BABILong, MAD 2403.19844); the clean loss-vs-context demonstration is TTT-E2E 2512.23675 [1L, Dec 2025] |
| Hybrids (attention + linear/SSM) | Extrapolation governed by the attention fraction's PE scheme; retrieval carried by full-attention layers | [R] Jamba, Zamba, Kimi Linear (2510.26692) |
| SWA + test-time training (TTT-E2E) | Constant per-token latency like an RNN, but loss-vs-context scaling matches full attention at 3B/164B tokens where Mamba-2/GDN do not | **[1L]** (2512.23675, Dec 29 2025, NVIDIA/Stanford/UCSD) — important if true; watch for independent replication |

**Competition-specific recommendation:** because miners choose architectures freely, evaluate extrapolation *twice*: (a) **zero-shot** (model exactly as trained), and (b) with a **declared inference-time scaling config** (e.g. YaRN scale factor) submitted as metadata. This avoids unfairly penalizing the RoPE family for a well-understood, fixable PE artifact while still catching architectures that silently fall apart.

---

## 3. Mechanistic probes (all procedurally regenerable)

| Probe | Source | What it isolates | Procedural recipe |
|---|---|---|---|
| **MQAR** (multi-query associative recall) | Zoology 2312.04927 (Dec 2023, ICLR'24); Based 2402.18668 (Feb 2024, ICML'24) | The canonical attention-vs-SSM discriminator: store \(N\) random key→value pairs, query several keys at the end. Gated convs/SSMs degrade sharply as \(N\) × model-dim grows; attention does not. Used as the gating probe in MAD (2403.19844) and in Kimi Linear's KDA ablations | [R, ≥4 labs] Seeded RNG over a synthetic vocab; vary #pairs, key/value length; accuracy vs. difficulty curve. Fresh seed per eval round ⇒ memorization impossible |
| Passkey | 2306.14000 | Single-fact retrieval floor | Random digit string in filler; trivial to regenerate |
| **Copying ("Repeat After Me")** | 2402.01032 (Feb 2024, ICML'24) | Exact reproduction of a random string after a gap. Theory: a 2-layer transformer copies exponential-length strings; fixed-state models are bounded by state size | [R] Random token strings, gap length swept; exact-match accuracy. Directly exposes state-capacity limits of SSM/linear models |
| Induction-head probe | Olsson et al. 2209.11895 (Sep 2022) | In-context pattern completion: `[A][B] … [A] → [B]` over random tokens | Random repeated subsequences; measure completion accuracy vs. distance between occurrences |
| State tracking (permutation composition, parity) | "The Illusion of State" 2404.08819 (Apr 2024, NeurIPS'24); Mamba-3 2603.15569 (Mar 2026) | Whether the model tracks evolving hidden state (S5/A5 composition). SSMs/linear models fail beyond state capacity; **Mamba-3 added complex-valued states explicitly to address this** | [R] Seeded permutation sequences; accuracy vs. sequence length/composition depth |
| RULER NIAH variants | 2404.06654 | Multi-key (distractor needles), multi-value (many values per key), multi-query; variable tracking = a linear-chain state-tracking task | Official generator; change needle vocab + templates for privacy |
| MRCR-style ordering | 2409.12640; openai/mrcr | Order discrimination among same-distribution needles | DeepMind `eval_hub` mrcr_v2 generator; or synthesize with any local model writing short paragraphs |
| NoLiMa-style latent needles | 2502.05167 | Retrieval without lexical overlap (associative link required) | Generate needle fact + question via disjoint-synonym templates ("X's spouse's employer…"); fresh entity names each round |
| GraphWalks | openai/graphwalks (Apr 2025) | Multi-hop traversal (BFS/parents) over edge lists | Random graphs per seed; deterministic ground truth; apply the Feb 2026 prompt fix (depth-exact BFS, root exclusion) |

**Anti-memorization protocol:** fresh RNG seed per scoring round; synthetic vocab (random strings, not English words); uniform-random needle depth (never the public benchmarks' canonical mid-context placement); answers derivable offline from the seed; keep a **public dev generator** and a **private test generator** with shifted distributions (different entity lexicons, different template phrasings) so training on the dev generator doesn't transfer perfectly.

---

## 4. Long-context LM-specific loss metrics

**Why raw PPL fails** [R]: most tokens in a long document are predictable from local context, so average PPL is nearly insensitive to whether the model uses the long range. A model that silently truncates to the last 2k tokens loses almost nothing on average PPL — this is precisely the exploit a logits-scoring competition must defend against.

**LongPPL** (2410.23771, Oct 2024; ICLR'25): compute loss only on **key tokens**, identified by a long–short context contrast — high LSD (log-prob gain when given full context vs. a short window) and high LCL (not intrinsically hard). Reports Pearson −0.96 against RULER/LongBench-style scores; robust to choice of evaluator model (Llama-3.1-8B, Qwen2-72B) and thresholds. **Caveats**: (i) correlation collapses if the *same* model selects the keys and is scored on them — key selection must come from a **fixed reference scorer** or from the harness; (ii) the −0.96 headline is single-paper, though the underlying "few tokens carry the long-range signal" finding is broadly replicated. The companion **LongCE** loss (up-weighting key tokens in training) is a miner-side technique, not an eval.

**Practical metrics for the harness (cheapest→richest):**

1. **Per-position loss curve**: mean loss binned by absolute position within long docs (as in TTT-E2E). Flat-then-rising past \(L_{\text{train}}\) = positional failure; flat-with-no-improvement = state-capacity saturation.
2. **Long-doc vs. short-doc decomposition**: separate average loss for docs ≤2k vs. ≥32k, plus the **context gain** per token bucket: \(\text{loss}(\text{truncated at }4\text{k}) - \text{loss}(\text{full context})\). Positive, growing context gain on far-apart dependencies is the direct signature of long-range usage. This needs no external judge and no key-token model.
3. **Harness-defined key tokens**: on procedural tasks, the answer tokens are key *by construction* — score loss there instead of over the whole sequence. This gives LongPPL's selectivity with zero circularity.
4. **LongPPL proper** with a frozen reference scorer as the key-token selector, if a public reference model is acceptable.

---

## 5. Efficiency at length (the sub-quadratic claim; how to measure honestly)

**What to report** — curves, never single points:

- **Prefill (TTFT) vs. \(L\)**: 1k→128k. Attention is \(O(L^2)\) FLOPs (FlashAttention keeps memory \(O(L)\)); SSM/linear is \(O(L)\) with chunked kernels; SWA+TTT is \(O(L)\) but with a *backward pass per chunk* — constant per-token latency, much larger constant. Honest reporting includes per-token FLOPs alongside wall-clock.
- **Decode TPOT vs. context length**, at batch 1 (latency regime) and at a large fixed batch (serving regime). Attention TPOT degrades ~linearly (KV reads); SSM/linear is flat. The large-batch regime is where linear models win big: Kimi Linear reports 6.3× TPOT vs. MLA at 1M context *at deployment batch* (2510.26692, Oct 2025; artifacts open-sourced, community-reproduced — [1L] for headline numbers, [R] for the direction).
- **Peak memory** and **cache bytes/token**, both measured and analytic:
  - Full attention: \(2 \times n_{\text{layers}} \times n_{\text{kv-heads}} \times d_{\text{head}} \times 2\,\text{B}\) per token. E.g. a 3B GQA model (24 layers, 4 KV heads, \(d\)=128): ≈49 KB/token → **6.3 GB at 128k** (fits on a 48GB L40S); Llama-3-8B geometry: 131 KB/token → 16.8 GB at 128k.
  - MLA-style latent cache: ~75% reduction (Kimi Linear) [1L].
  - SSM / linear attention / delta-rule: **fixed state**, \(n_{\text{layers}} \times d_{\text{state}} \times d_{\text{head}} \times\) heads — MB-scale total, zero growth per token.
  - SWA+TTT: window-bounded cache + weight deltas; state is constant but compute per token is higher. Fairness demands reporting both.
- **Crossover length**: where the linear model actually beats attention on wall-clock. Mamba-3's framing is the honest one (2603.15569, Mar 2026): "theoretically linear inference remains hardware-inefficient in practice" — decode is memory-bound and low arithmetic intensity wastes tensor cores, so linear models can *lose* at short lengths [1L, but the utilization critique is consistent with FLA-kernel folklore, quasi-R].
- **Identical-recipe rule**: the credible papers train all compared architectures on the same data/recipe (Kimi Linear vs. full MLA, 1.4T tokens; TTT-E2E 3B/164B tokens on DCLM; Mamba-3 vs. GDN/Mamba-2 at matched shapes) and measure with deployment-grade stacks (vLLM + open kernels). For the competition: harness-provided reference kernels per family, fixed GPU, fixed dtype (fp16/bf16), batch swept, and report the full curve — a miner claiming sub-quadratic advantage must show prefill + decode + memory curves vs. the harness's attention baseline on the *same* GPU.

---

## 6. Pitfalls

1. **Contamination.** LongBench v1, L-Eval, ZeroSCROLLS, InfiniteBench QA subsets, and the classic Paul-Graham-essays NIAH haystack are all public and scraped. Frontier saturation numbers partly reflect this. Mitigation: procedural regeneration with private seeds (RULER, BABILong "leak-proof" facts, GraphWalks, MQAR, MRCR generator). Any static natural-text set used must be treated as a sanity anchor, not a scored component.
2. **Retrieval shortcuts in needle tests.** Vanilla NIAH needles are semantically alien to the haystack — a model can win with shallow lexical matching. NoLiMa showed 11/13 frontier models halve at 32k once literal overlap is removed [R]. Countermeasures: same-distribution distractors (MRCR), paraphrase/latent-association needles (NoLiMa-style), multi-needle ordering, and needle-depth randomization (kills position-prior gaming).
3. **Synthetic-vs-natural gap.** HELMET's central replicated finding: synthetic recall scores **do not predict** downstream performance, and the seven categories mutually decorrelate. For 100M–3B from-scratch models, natural task accuracy is near-floor anyway — so use natural text through *loss* metrics (§4) rather than task accuracy, and treat synthetic tasks as the primary capability signal at this scale.
4. **Benchmark gaming / evaluator exploitation — the ArchAgent lesson** (2602.22425, Feb 2026): an agentic system found that ChampSim's no-write-bypass rule was enforced only by an assertion **compiled out in optimized builds**, and "won" by making data vanish from the simulated system [1L, but the lesson generalizes]. Translation to a miner harness: **any invariant enforced only by documentation or good faith will be exploited.** Runtime-verify: hash prompts, check output-position alignment, and audit that predictions actually depend on remote context (below).
5. **Truncation tricks in a logits-scoring harness.** Because most tokens are locally predictable (§4), a miner can implement `forward()` to internally attend only to the last \(W\) tokens and lose almost nothing on average loss. Countermeasures: (a) score only key tokens whose answers depend on remote information; (b) **counterfactual audit** — corrupt a needle early in the context; an honest full-context model's loss on the target must move, a truncator's will not; (c) require per-position loss curves, where a truncator shows a characteristic step at \(W\).
6. **Position-ID resets / chunking.** Feeding chunks with restarted position IDs lets a 4k-trained RoPE model "process" 128k. This is a *legitimate* memory architecture (RMT/ARMT do exactly this and excel on BABILong), but it must be disclosed and its memory state counted in the efficiency table — the contract is: the harness streams the full sequence once through the model's own code, and bytes carried between chunks are reported as state.
7. **Overfitting public generators.** RULER's 13 templates have been fixed since Apr 2024 and are in training corpora. Private template variants, fresh vocab, and shifted distractor statistics are mandatory; also see pitfall 1.
8. **Truncation confounds in your own harness.** Distractor-Aware Truncation (2608.03297, Aug 2026, brand new [1L]): dropping content from the middle of prompts destroys signal and fabricates "shorter is better" results. When constructing length variants, splice distractors only (BABILong/RULER do this correctly by construction).
9. **Judge-based metrics** (InfiniteBench En.Sum, L-Eval LLM-judged QA): expensive, noisy, and a gaming surface. For a private eval, exact-match procedural answers only.
10. **MCQ answer-prior gaming** (LongBench v2 format): base models are near chance anyway; don't use MCQ natural benchmarks as scored components at this scale.

---

## 7. Concrete battery recommendation

**Scale realism first.** With ≤6h on a single L40S/A100, a 1B model sees roughly 0.5–2B tokens — far under Chinchilla. Absolute natural-task scores will be low for everyone; the battery therefore ranks **within-competition**, uses self-normalized effective length, and leans on procedural tasks where 100M-class models have provably solved the task (BABILong's Mamba-130M/RMT-137M results).

Core lengths: **1k, 2k, 4k, 8k, 16k, 32k**. Stretch: 64k/128k for loss metrics and one retrieval task only (attention KV at 128k ≈ 6–17 GB — fits a 48GB L40S for ≤3B models).

| # | Component | Generator (private seed) | Lengths | Metric | Why it's in | ~GPU-min, 3B on L40S |
|---|---|---|---|---|---|---|
| 1 | Per-position loss + context gain on held-out long docs (books/code) | Fixed corpus, harness-side truncation contrast | 4k–32k (64k stretch) | Loss-vs-position curve; context gain Δloss(4k→full) on far tokens | Catches truncation cheats; cheapest honest signal; no judge needed | 30 |
| 2 | Key-token loss (harness-defined) on procedural answers | Derived from #3–#6 by construction | all | Loss on answer tokens only | LongPPL selectivity with zero circularity | 0 (reuses forward passes) |
| 3 | RULER-style pack: NIAH single, multi-key, multi-value, multi-query, variable tracking, freq-words | RULER generator, private templates/vocab | 1k–32k | Accuracy; self-normalized \(L^*\) + absolute floor | The arch-paper standard; discriminative at small scale | 60 |
| 4 | BABILong qa1–qa5 (fact chaining, counting, deduction) | bAbI fact generator + PG-19-style distractor splicing | 2k–32k (64k stretch) | Exact match | Best-proven fit at 100M–3B; leak-proof by design | 45 |
| 5 | MQAR + copying + induction + permutation-composition state tracking | Zoology/MQAR + Repeat-After-Me + S5 probes, synthetic vocab | in-vocab lengths; gap sweep to 32k | Accuracy vs. difficulty | Mechanistically separates attention/SSM/linear/hybrid; memorization-proof | 20 |
| 6 | GraphWalks (BFS depth-exact + parents) | Random graph generator (Feb 2026 prompt fixes) | 4k–32k tokens | Set-F1 | Multi-hop reasoning over full context; frontier-hard even now | 30 |
| 7 | MRCR-style multi-needle ordering, 2/4 needles | Fresh needle/distractor paragraphs, same distribution | 4k–32k | Sequence-match ratio + hash-prefix gate | Defeats lexical-shortcut retrieval; high signal for base/small models per Michelangelo | 30 |
| 8 | NoLiMa-style latent-association needles | Fresh entity lexicon, paraphrase templates | 4k–32k | Accuracy | Direct countermeasure to literal-match gaming | 15 |
| 9 | Efficiency curves: prefill TTFT, decode TPOT @ b=1 & b=32, peak VRAM, analytic cache bytes/token | Harness reference kernels per family | 1k–64k (128k if memory allows) | Curves + crossover length vs. harness attention baseline | Where sub-quadratic architectures must prove their claim honestly (Kimi Linear / Mamba-3 / TTT-E2E protocol) | 45 |
| 10 | Anti-gaming audits: counterfactual needle corruption, position-randomization re-run, prompt-hash/state-byte verification | Automated | sample of #3–#8 | Pass/fail + loss-shift magnitude | The ArchAgent rule: verify invariants at runtime, never by good faith | 15 |

**Total: ≈ 4.5–5 L40S-hours per 3B submission; ~30–45 min per 100M submission** (loss and retrieval dominate; scale sample counts to keep 8×length-grid synthetic tasks at ~150–250 samples/length for ±2–3 pt resolution). On an A100-80GB, add the 128k stretch tier to components 1, 3, 4, and 9 (+~1.5h).

**Optional natural anchor (unscored diagnostics):** HELMET's RAG subset (Natural Questions, 100-doc) and one LongBench v1 QA task at ≤16k, few-shot prompted for base-model compatibility — reported for context, not ranked, due to contamination and floor effects.

**Bottom line for this competition:** at 100M–3B with ≤4k training context, the 2026-correct evaluation is (i) procedural, privately-seeded versions of RULER + BABILong + MQAR/copying/state-tracking + GraphWalks + MRCR/NoLiMa-style needles on a 1k→32k grid; (ii) key-token and per-position loss on natural long documents as the truncation-proof natural signal; (iii) self-normalized effective context length with an absolute floor; and (iv) efficiency reported as full prefill/decode/memory curves against an identical-recipe attention baseline — with runtime anti-gaming audits, because ArchAgent demonstrated that agentic (or simply adversarial) submitters will exploit any invariant you don't actively verify.
