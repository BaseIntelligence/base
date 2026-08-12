# Appendix 10 — Evaluating Reasoning at 100M–3B Scale
> Research appendix for the Prism v3 evaluation proposal (`docs/spikes/prism-v3/`). Produced 2026-08-06 via arXiv/web research. Non-normative spike document.

# Evaluating Reasoning in Small Language Models (100M–3B) Trained From Scratch

**Survey date: 2026-08-06. All arXiv IDs below were verified against the live arXiv record during this survey.**

**Citation corrections to the brief (load-bearing, so flagged up front):**
- "**CoT expands power, 2405.17313**" — that ID is Larson–Vakil–Vogt, *The interpolation problem* (algebraic geometry). The intended results are **Li et al., 2402.12875** (Feb 2024, ICLR'24, *Chain of Thought Empowers Transformers to Solve Inherently Serial Problems*) and **Merrill & Sabharwal, 2310.07923** (Oct 2023, ICLR'24, *The Expressive Power of Transformers with Chain of Thought*).
- "**State-tracking, 2411.12512**" — that ID is Bangachev et al. on sparse LWE. The intended paper is **Merrill, Petty & Sabharwal, 2404.08819** (Apr 2024, ICML'24, *The Illusion of State in State-Space Models*).
- "**TC^0 bounds, 2402.09268**" — that ID is Sanford–Hsu–Telgarsky, *Transformers, Parallel Computation, and Logarithmic Depth* (Feb 2024, ICML'24), a transformer↔MPC simulation result, not the TC^0 bound. The TC^0 upper bound is **Merrill & Sabharwal, 2207.00729** (Jul 2022, TACL'23, *The Parallelism Tradeoff*).
- Two other IDs I initially expected from memory were also wrong and are corrected below: ProofWriter is **2012.13048** (not 2104.07871), ProntoQA is **2210.01240** (not 2210.01217), *The Parallelism Tradeoff* is **2207.00729** (not 2205.13536), TinyGSM is **2312.09241** (not 2312.06741).

---

## 1. What reasoning is measurable at 100M–3B

**The hard fact about GSM8K at this scale.** For from-scratch models trained on generic web corpora, GSM8K (Cobbe et al., 2110.14168, Oct 2021) 8-shot is statistically indistinguishable from noise below ~7B:

| Model | Recipe | GSM8K (flex/strict) |
|---|---|---|
| Pythia-2.8B (2304.01373, Apr 2023) | Pile, 0.3T tok | 1.59 / 2.12 |
| Pythia-6.9B | Pile, 0.3T tok | 2.05 / 2.43 |
| Pythia-12B | Pile, 0.3T tok | 3.49 / 4.62 |
| OLMo-1B (2402.00838, Feb 2024) | Dolma, 3T tok | 1.82 / 2.27 |
| OLMo-7B (first release) | Dolma, 2.5T tok | 4.02 / 4.09 |
| OLMo-7B-0724 (math-heavy mix) | + math data | 28.66 / 28.73 |
| OLMo-2-7B | + math data | 66.72 / 66.79 |

MMLU sits at the 25% chance floor across the entire Pythia suite (25.05→25.74 from 2.8B→12B). Meanwhile the *same* Pythia/OLMo checkpoints show clean, monotone scale separation on likelihood-scored multiple choice: ARC-E (1803.05457), HellaSwag (1905.07830), PIQA (1911.11641), SciQ (1707.06209), WinoGrande (1907.10641), LAMBADA (1606.06031). This is the first design lesson: **below ~3B, generative exact-match on famous benchmarks reads zero; likelihood-scored discrimination is alive and well.**

**The GSM8K "1–3B discrimination range" is real but confounded.** Modern small models *do* separate on GSM8K: Qwen2.5 (2412.15115, Dec 2024) 0.5B/1.5B/3B-Instruct score 49.6 / 73.2 / 86.7 (4-shot); SmolLM2-1.7B-Instruct ≈ 45; Qwen2.5 base BBH (2210.09261, Oct 2022) goes 20.3 → 45.1 → 56.3 across 0.5B→3B. But these are 18T-token recipes saturated with synthetic math/code — the spread measures **data-mix quality**, not architecture. TinyGSM (2312.09241, Dec 2023) proves the point brutally: a 1.3B model + 1.3B verifier hits 81.5% on GSM8K after fine-tuning on 12.3M GPT-3.5-generated problems — a 1.3B "beating" its own teacher. Orca-Math (2402.14830, Feb 2024) does the same at 7B (86.8%). For an architecture competition, GSM8K-style scores are only interpretable if the training corpus is held fixed across submissions — and even then, prefer a *private generator* (§2, §6).

**What discriminates below 1B.** Empirically, at 100M–1B the following separate architectures and data recipes: single- and two-step arithmetic word problems (SVAMP, 2103.07191; ASDiv, 2106.15772; MultiArith-style sets), boolean-expression evaluation, tracking-shuffled-objects with 1–2 swaps, Dyck-1/Dyck-2 next-token prediction, modular addition under grokking dynamics (Power et al., 2201.02177, Jan 2022), copying/induction-head tasks (Olsson et al., 2209.11895, Sep 2022), bAbI tasks 1–3 (Weston et al., 1502.05698, Feb 2015), and depth ≤2 deductive closure (RuleTaker, 2002.05867; ProofWriter, 2012.13048). Note the common structure: **short chains, verifiable answers, likelihood-scoreable**. Razeghi et al. (2202.07206, Feb 2022) add a caveat: few-shot numerical reasoning correlates with pretraining *term frequency of the specific numbers* — so even the numbers must be rotated, not just the templates.

---

## 2. Procedural / synthetic reasoning generators

Generators are the core technology for this competition: fresh instances each round make memorization useless, and difficulty knobs let you keep every scale band in the discriminative 20–80% accuracy window.

**GSM-Symbolic** (Mirzadeh et al., Apple, 2410.05229, Oct 2024). 100 symbolic templates built from GSM8K test items, 50 instantiations each. Findings that transfer directly to small-model eval design:
- All 20+ SOTA models drop when *only numbers change* (0.3–9.2%), and accuracy varies up to 15% across instantiations of the *same* template — single-instance evals of a template are meaningless; report the distribution.
- Accuracy degrades monotonically as clauses are added (their GSM-M1 → Symbolic → P1 → P2 tiers), with variance *increasing* with difficulty — clause count is a clean difficulty knob.
- **GSM-NoOp** (one irrelevant but plausible-looking clause) causes 17.5–65.7% drops. Distractor clauses are the sharpest existing probe of "pattern matching vs. reasoning," and they cost nothing to generate.

**Templated arithmetic, older and cheaper.** The DeepMind *Mathematics Dataset* (Saxton et al., 1904.01557, Apr 2019) is the original MathEval-style procedural generator: ~20 school-math modules (arithmetic, algebra, calculus), millions of fresh items, and — critically — **extrapolation splits** (train on small operands, test on large). Operand-range holdout is the cheapest memorization-proofing available and should be standard in every arithmetic tier.

**BIG-Bench-style task generators.** BIG-bench (2206.04615, Jun 2022) ships ~100 programmatic tasks with generators — `dyck_languages`, `arithmetic`, `boolean_expressions`, `tracking_shuffled_objects`, `web_of_lies`, `logical_deduction` — and BBH (2210.09261) is the subset where CoT beats direct answering at 7B+. Below 3B most BBH *generative* tasks are at chance, but the generators remain usable with difficulty dialed down (fewer swaps, shorter expressions), and every one is likelihood-scoreable as MC.

**Logic puzzles with real generators.**
- **Knights & Knaves**: Xie et al., 2410.23123 (Oct 2024) — a dynamic K&K generator, N = 2…8 people (~10^24 unique 8-person puzzles), plus a six-way perturbation suite (statement/leaf/name/reorder/role flips) and a per-sample memorization metric. Headline finding: fine-tuned LLMs reach near-perfect train accuracy yet break under slight perturbation — direct evidence that static puzzle sets measure memorization while perturbed generators measure reasoning. Ships train/test splits, ideal for a competition.
- **Zebra puzzles**: ZebraLogic (Lin et al., 2502.01100, Feb 2025) — 1,000 CSP logic-grid puzzles, 2x2 to 6x6, with *quantified* complexity (search-space size, Z3 conflict count). "Curse of complexity": collapse past ~10^7 search space; 7–10B models solve <1% of hard puzzles. For 100M–3B, the 2x2–3x3 tiers with cell-wise partial credit are the usable band.

**Graph algorithms as text.** NLGraph (Wang et al., 2305.10037, May 2023, NeurIPS'23): 29,370 problems over 8 tasks (connectivity, cycle, shortest path, topological sort, max flow, bipartite matching, Hamilton path, GNN simulation). Two findings that matter at small scale: models lean on spurious correlations (e.g., node-mention frequency for connectivity — your generator must balance these), and prompting gains vanish on the complex tasks. CLRS-Text (Markeeva et al., DeepMind, 2406.04229, Jun 2024) textualizes all 30 CLRS algorithm traces with ID/OOD splits — the standardized version of "algorithms as text," with trace-matching giving you process metrics for free.

**Dyck languages / formal-language probes.** Delétang et al., 2207.02098 (Jul 2022, ICLR'23, *Neural Networks and the Chomsky Hierarchy*): 20,910 models × 15 tasks; transformers and RNNs fail to generalize on non-regular tasks, LSTMs handle regular + counter languages, only stack/tape-augmented networks generalize on context-free/context-sensitive tasks. Hahn (2009.03364, Sep 2020) gives the theoretical transformer limit on Dyck/parity. Liu et al., 2210.10749 (Oct 2022, ICLR'23, *Transformers Learn Shortcuts to Automata*) is the essential warning for generator-based eval: transformers learn O(log T)-depth *shortcut* solutions to automata simulation that fit the training distribution and break OOD — **so every formal-language tier needs a length-generalization split**, or you'll measure shortcut-fitting, not architecture.

**Scratchpad / CoT format effects.** Nye et al., 2112.00114 (Dec 2021): intermediate-state format dominates multi-step execution performance. Wei et al., 2201.11903 (Jan 2022): CoT prompting. Practical issue: from-scratch ≤3B base models often can't follow CoT format from few-shot alone. Two mitigations: (a) score both direct-answer and scratchpad-conditioned variants; (b) CoT-decoding (Wang & Zhou, 2402.10200, Feb 2024, NeurIPS'24) — branch on top-k tokens at the first decoding step to elicit latent CoT paths from base models with no prompt engineering, using the top-token probability gap as a confidence selector.

**ARC-AGI-style grids as text.** ARC (Chollet, 1911.01547, Nov 2019) serialized as text is at-chance below 3B and nearly so for frontier models pre-o1; treat it as a negative control / headroom indicator only. If you want a small-scale analog, use 1D cellular automata or tiny (≤5x5, 2-color) grid transforms; ConceptARC (2305.07141, May 2023) is the easier graded variant.

---

## 3. Chain-of-thought evaluation methodology

**Outcome vs. process metrics.** Uesato et al., 2211.14275 (Nov 2022) and Lightman et al., 2305.20050 (May 2023, *Let's Verify Step by Step*) established that process supervision beats outcome supervision for reliability at scale. At 100M–3B you cannot afford a learned process reward model — but **synthetic generators give you process supervision for free**: ProntoQA (Saparov & He, 2210.01240, Oct 2022, ICLR'23) generates first-order-logic proof structures that let you mechanically parse model CoT into formal proofs and score *every step*; ProofWriter (Tafjord et al., 2012.13048, Dec 2020, ACL'21) scores proof-tree accuracy, not just answers; CLRS-Text scores trace tokens. This is the right small-scale substitute for PRMs: **verifiable traces, not learned judges.**

**The measurement-artifact warning (design rule #1).** Schaeffer, Miranda & Koyejo, 2304.15004 (Apr 2023, NeurIPS'23 outstanding paper): "emergent" jumps are largely metric artifacts. Exact-match on a length-L answer is a (per-token accuracy)^L deformation of a smoothly improving quantity; switching to token edit distance or Brier score turns cliffs into smooth curves. At 100M–3B this is not a philosophical point — it is the difference between "all submissions score ~0" and a usable leaderboard. **Every exact-match metric in the battery ships with a likelihood-based companion** (answer-token logprob, Brier on options, edit distance on traces).

**pass@k vs. maj@k at small scale.** Self-consistency (Wang et al., 2203.11171, Mar 2022) and *Large Language Monkeys* (Brown et al., 2407.21787, Jul 2024): coverage (pass@k with an oracle verifier) scales log-linearly over four orders of magnitude of k (Llama-3 on GSM8K exceeds 95% at k=10,000), while majority voting and reward-model selection **plateau beyond ~100 samples**. The pass@k–maj@k gap therefore measures *selection* headroom, not capability. At small scale, report pass@1 (greedy), maj@8, pass@8 (unbiased estimator per Chen et al., 2107.03374); the gap tells you whether a submission's weakness is reasoning or answer selection — cheap and diagnostic. Snell et al., 2408.03314 (Aug 2024) show small model + search can beat a 14x larger model on easy tiers, so fix the sampling budget across submissions.

**Answer extraction and format sensitivity.** Report strict and flexible match side by side (the OLMo eval tables show these differ by ~0.5–1 point even at 2% absolute — proportionally huge). Sclar et al., 2310.11324 (Oct 2023) document accuracy swings of tens of points from prompt formatting alone; OLMES (Gu et al., 2406.08446, Jun 2024) is the standard to adopt: cloze (CF) vs. multiple-choice (MCF) formulations, per-task normalization choices (PMI where answer vocabulary is unusual — Holtzman et al., 2104.08315; character-length normalization otherwise), both formulations run and documented. For a competition: freeze the prompt, freeze the extraction regex, and score format-failure separately from reasoning-failure so a model that can't emit "#### 42" isn't confused with one that can't add.

---

## 4. Formal-language / theory-of-computation probes as architecture X-rays

The theory stack, correctly cited, and the cheap empirical probe each result licenses:

| Theory result | Claim | Cheap probe it licenses |
|---|---|---|
| Merrill & Sabharwal, 2207.00729 (Jul 2022, TACL'23) | Log-precision constant-depth transformers ⊆ logspace-uniform TC^0; can't solve linear equalities (P-complete), CFG membership, automaton simulation (NC^1-complete) unless collapses | Problems *outside* TC^0 as no-CoT failure tests |
| Li et al., 2402.12875 (Feb 2024, ICLR'24) | Constant-precision transformers w/o CoT ⊆ AC^0; T CoT steps ⇒ any size-T circuit; linear CoT ⇒ all regular languages incl. **S5 word problem** (NC^1-complete); poly CoT ⇒ P/poly. Empirics: S5 composition, iterated squaring, circuit value — CoT rescues low-depth models | **S5 permutation composition**, ±scratchpad budget — the single best architecture probe |
| Merrill & Sabharwal, 2310.07923 (Oct 2023, ICLR'24) | CoT length as compute resource: log steps ≤ L; linear steps ⇒ regular languages; poly steps ⇒ exactly P | Difficulty tiers indexed by *required* chain length |
| Merrill & Sabharwal, 2404.09255 (Apr 2024) | Circuit characterizations of transformers-as-transducers on regular/Dyck languages | Dyck-k with/without CoT |
| Merrill, Petty & Sabharwal, 2404.08819 (Apr 2024, ICML'24) | Linear & Mamba-style SSMs ∈ TC^0: provably cannot compose permutations, track chess moves, evaluate code, track entities; experiments confirm Mamba fails like transformers; RNNs succeed | Permutation composition + entity tracking as **SSM blind-spot** tests |
| Chen et al., 2412.06148 (Dec 2024, ICML'25) | Selective SSM/Mamba ∈ DLOGTIME-uniform TC^0 — same class as transformers | Same probes; predicts SSMs gain nothing on state tracking |
| Jelassi et al., 2402.01032 (Feb 2024, ICML'24) | 2-layer transformer copies exponential-length strings via n-gram hashing (induction heads); GSSMs bounded by fixed state; prefix-key variant favors GSSMs | Copying + n-gram lookup, suffix-key vs prefix-key — **transformer strength / SSM weakness**, cleanly signed |
| Sanford, Hsu & Telgarsky, 2402.09268 (Feb 2024, ICML'24) | Transformers ↔ MPC: depth O(log D) solves connectivity of diameter-D components, conditionally optimal | Graph connectivity with diameter as the knob — a **depth-resource** probe |
| Liu et al., 2210.10749 (Oct 2022, ICLR'23) | Transformers learn brittle O(log T)-depth shortcuts to automata; constant-depth shortcuts exist only for solvable-group automata unless TC^0=NC^1 | Length-generalization splits on every automaton/Dyck task |
| Delétang et al., 2207.02098 (Jul 2022, ICLR'23) | Empirical Chomsky map: transformers/RNNs fail non-regular generalization; LSTMs do regular+counter; stack/tape needed for CF/CS | The cross-architecture expectation table itself |

Also relevant: Kim & Schuster, 2305.02369 (May 2023) — entity-tracking probe with a mechanistic account of how transformers do it; *Little Depth Goes a Long Way* (ICLR 2025, OpenReview zDze7VtB5C) — log-depth scaling suffices for connectivity/regular-language recognition and beats width or short-CoT scaling, which tells you how to set depth tiers.

**The probe set this implies (all generatable in <100 lines each):** (a) **S5 word problem** — compose k permutations, predict product or identity-check; tiers k ∈ {5, 10, 20, 50}; run with and without scratchpad tokens. Expectation: every constant-depth architecture fails without CoT; architectures with real recurrence or depth scaling pull ahead. This is the money probe for an architecture-invention contest. (b) **Dyck-k**, k ∈ {1, 2, 4}, next-token closing prediction, train ≤40 tokens / test 41–80. (c) **Copying suite**: exact copy 32→256 tokens (length generalization), suffix-key n-gram lookup (transformer-favored), prefix-key (SSM-favored). (d) **Graph connectivity/shortest path** as text with diameter/edge-count knobs (NLGraph format; balance node-mention frequency). (e) **Modular arithmetic** mod 97/113 with held-out operand ranges — grokking-style ID-vs-OOD separation (Power et al., 2201.02177; progress measures in Nanda et al., 2301.05217). (f) **Parity at long length** as a stress test (in TC^0 but empirically hard for transformers to length-generalize).

---

## 5. Reasoning-through-length

Small from-scratch models live at 1k–8k context, so "long context" here means *retrieval × reasoning composites at 512–8192 tokens*, not million-token haystacks.

- **bAbI** (Weston et al., 1502.05698, Feb 2015): the original generator — 20 tasks (single/two/three supporting facts, counting, path finding, deduction). Small models solve it *when fine-tuned*; few-shot it discriminates at 1–3B. The task-1→3 hop ladder is exactly the right difficulty axis.
- **BABILong** (Kuratov et al., 2406.10149, Jun 2024, NeurIPS'24 D&B): bAbI facts scattered through PG19 noise, 0k–10M tokens. Findings: LLMs effectively use only 10–20% of context; RAG plateaus at ~60% on single-fact QA regardless of length; and — most relevant — **fine-tuned small models (RMT-137M, Mamba-130M) solve the tasks**, proving they're within small-model reach. The paper explicitly notes generated benchmarks are contamination-immune. Caveat for adaptation: BABILong's scalability relies on task sentences being distributionally distinct from background; for a fair small-model probe, generate distractors from the *same* grammar so the task is reasoning, not distribution-shift detection.
- **RULER** (Hsieh et al., 2404.06654, Apr 2024): 13 synthetic tasks in 4 categories with complexity knobs. The two that matter here: **variable tracking (VT)** — X1=V, X2=X1, X3=X2… chains scattered in noise, return all names bound to V; complexity = chains × hops; a minimal multi-hop coreference probe — and **common/frequent words extraction** (aggregation). Models near-perfect on vanilla NIAH collapse on VT at 32k; at 1–4k with 1–3 chains × 2–8 hops it discriminates small architectures cleanly (this is where SSM/hybrid state-handling differences show up).
- **Lost in the Middle** (Liu et al., 2307.03172, Jul 2023): U-shaped positional bias — randomize needle positions and report position-stratified accuracy, or you'll conflate position robustness with reasoning.
- **MuSR** (Sprague et al., 2310.16049, Oct 2023, ICLR'24): neurosymbolic synthetic-to-natural generator (murder mysteries, object placement, team allocation; ~1000-word narratives over formal logic trees). Small models are near chance on the full version, but the *generator* scales down — shorten narratives and prune fact trees for a soft-reasoning tier with natural-language surface form.

---

## 6. Contamination resistance

**Why static reasoning sets die within months.** The leakage path is no longer just verbatim test sets in crawls — it's synthetic data. TinyGSM (2312.09241) and Orca-Math (2402.14830) show benchmark-shaped synthetic data is now standard practice; Phi (Gunasekar et al., 2306.11644, Jun 2023) showed "textbook-quality" synthetic data dominates at 1.3B. Any public static set is one synthetic-data generation away from uselessness, and the GSM8K numbers in §1 (OLMo-7B: 4% → 29% → 67% across data revisions, same architecture) show exactly this confound in the wild.

**The GSM1k mirror-gap method** (Zhang et al., 2405.00332, May 2024, NeurIPS'24 D&B): commission 1,205 human-written problems matched to GSM8K on human solve rate, solution steps, answer magnitude. The GSM8K−GSM1k accuracy gap upper-bounds contamination+overfitting: drops up to 8% (final version; v1 reported up to 13%), with some families (Phi, Mistral) systematically overfit across sizes and frontier models clean. The killer analysis: Spearman r² = 0.36 between a model's per-character log-likelihood of generating GSM8K and its GSM8K−GSM1k gap — partial memorization, measurable with open weights. **For the competition: build a private mirror of your public arithmetic tier and compute a mirror-gap per submission.** Complementary detectors: Min-K% Prob (Shi et al., 2310.16789, Oct 2023) for membership inference if training data is auditable; rephrased-sample degradation (Yang et al., 2311.04850, Nov 2023); n-gram task contamination (Li & Flanigan, 2312.16337, Dec 2023).

**Rotating generator seeds as the permanent fix.** The dynamic-benchmark line has converged on the answer: DyVal (Zhu et al., 2309.17167, Sep 2023, ICLR'24) — DAG-structured dynamic generation with controllable complexity; NPHardEval (Fan et al., 2312.14890, Dec 2023, ACL'24) — 900 algorithmic questions across P/NP-complete/NP-hard with monthly refresh; LiveBench (White et al., 2406.19314, Jun 2024, ICLR'25) — monthly questions from recent sources, objective ground truth, full refresh every 6 months; the K&K generator (2410.23123) with its perturbation suite. The competition rule that falls out: **publish the generators, rotate and hash the eval seed each round, keep a private seed family (and a private template family) for final scoring.** Generators make this free; static sets make it impossible.

---

## 7. The concrete battery: 100M–3B from-scratch reasoning eval

Design constraints honored: fresh instances per round (seeded generators, private final-seed family); every exact-match metric paired with a likelihood companion (Schaeffer); difficulty tiers placing each scale band in the 20–80% discriminative window; architecture blind spots signed in advance (§4); no learned judges; total budget ≈ **1.5–2.5 A100-40G-hours per submission per seed, ×3 seeds ≈ 4–7 A100-hours** (pessimistic; bf16, continuous batching).

**Tier 0 — sanity gate (~5 min).** Exact copy (32 tokens), Dyck-1 next-token, 1-step arithmetic, format compliance. A submission failing Tier 0 gets Tier 1–4 likelihood scores only (its EM numbers are uninterpretable).

**Tier 1 — symbolic core (~30 min).**
1. *Templated arithmetic* (GSM-Symbolic-style templates + Saxton-style modules): tiers by clause count {1,2,3,4+}, plus a **NoOp distractor** variant and an **operand-range extrapolation** split. Metrics: EM (strict+flexible), answer-token logprob, per-digit accuracy. Expected range: 100M ≈ 20–60% on 1-step; 3B ≈ 5–30% on 3-step-with-distractor.
2. *Deductive closure* (ProofWriter/RuleTaker-style generator): depths {0,1,2,3,5}, closed- and open-world; fictional-ontology variant (ProntoQA-style). Metrics: answer EM + **proof-step accuracy** (process metric, free from the generator). Depth 0–1 discriminates 100M–500M; depth 2–3 discriminates 1–3B.
3. *Boolean expressions + tracking-shuffled-objects* (BIG-bench generators), likelihood over options; swaps ∈ {1,2,3}.

**Tier 2 — formal-language architecture probes (~30 min).**
4. *Dyck-k*, k ∈ {1,2,4}: closing-tag prediction; length split train ≤40 / test 41–80. Metrics: token acc, sequence EM, closing-token logprob.
5. *S5 permutation composition*: k ∈ {5,10,20,50}, identity-check + full-product; each instance scored **with and without a scratchpad budget**. The no-CoT/CoT delta is the architecture signature (2402.12875, 2404.08819).
6. *Copying suite*: exact copy 32→256; suffix-key n-gram lookup; prefix-key variant; selective copy. Signed expectations: transformers strong on suffix-key, SSMs collapse beyond state size, prefix-key reverses it (2402.01032).
7. *Modular arithmetic* mod 97/113, held-out operand range; ID-vs-OOD gap as the algorithmic-generalization measure.

**Tier 3 — reasoning-through-length (~45 min).**
8. *bAbI-mini generator*: tasks 1–3 + counting + path-finding at context {512, 1k, 2k}, distractors from the same grammar, density knob.
9. *Variable tracking* (RULER VT): {1,2,3} chains × {2,4,8} hops at {1k, 2k, 4k}; metrics: set-F1 + exact-set EM; needle positions randomized and position-stratified (2307.03172).
10. *Retrieval×compute composite*: scattered facts + one arithmetic step over retrieved values ("A has 3 apples … B has twice A's …"), distance-to-evidence as the knob.

**Tier 4 — logic-puzzle ceiling (~15 min).**
11. *Knights & Knaves* (2410.23123-style generator): N ∈ {2,3,4}; full perturbation suite as a built-in memorization check; metrics: puzzle EM + per-statement truth-value accuracy (soft companion).
12. *Zebra-mini* (2502.01100-style): 2x2, 2x3, 3x3; cell-wise accuracy (soft) + puzzle EM (hard). Expect ≈0% puzzle-EM at 3x3 for ≤3B — that's headroom indication, not a bug.

**Scoring protocol (fixed across submissions).**
- **Dual metrics everywhere**: EM + answer-token logprob; Brier score on all MC (2304.15004); token edit distance on traces. OLMES conventions for MC: CF with character-length normalization, PMI where answer vocab is unusual, both CF and MCF documented (2406.08446, 2104.08315).
- **Sampling**: pass@1 greedy, maj@8, pass@8 at temp 0.7, unbiased estimator; report the pass@8−maj@8 gap as selection headroom (2407.21787, 2203.11171). Base models that can't follow CoT format get the CoT-decoding variant (2402.10200) or fixed 8-shot exemplars — same for everyone.
- **Extraction**: frozen prompt, frozen regex, last-number fallback; format-failure reported separately from reasoning-failure.
- **Anti-contamination**: generators public, per-round seeds rotated + hash-committed before the round opens; final scoring on a private seed family + the K&K-style perturbation suite + a GSM1k-style **private mirror of Tier 1** with a per-submission mirror-gap (2405.00332); Min-K% spot check (2310.16789) where training data is auditable. Optional: an "alignment split" from the same generators with disjoint seeds, so submissions may adapt format without ever seeing eval-distribution instances.
- **Calibration target**: each scale band {100M, 300M, 1B, 3B} must have ≥3 subtasks where the current best submission sits in 20–80%; re-tier the knobs (clauses, hops, k, chains) between rounds as the field improves — the generator knobs make this a one-line change.

**What this battery measures that MATH/GPQA/BBH cannot at this scale:** a smooth, likelihood-anchored capability curve (not at-chance noise); signed architecture blind spots (state tracking, copying, depth, stack); reasoning-vs-memorization separation by construction (rotating seeds, perturbation suites, mirror-gap); and process quality via verifiable traces — all inside a single-digit A100-hour budget per submission.
