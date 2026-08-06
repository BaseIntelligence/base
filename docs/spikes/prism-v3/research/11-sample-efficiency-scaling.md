# Appendix 11 — Sample Efficiency and Scaling Prediction
> Research appendix for the Prism v3 evaluation proposal (`docs/spikes/prism-v3/`). Produced 2026-08-06 via arXiv/web research. Non-normative spike document.

# Evaluating Sample Efficiency and Predicting Large-Scale Performance from Small Runs

*Survey for an architecture-invention competition (≤6h, 1 GPU, 100M–3B models). All arXiv IDs verified as of 2026-08-06.*

---

## 1. Scaling-law fitting methodology

### 1.1 The Chinchilla template and its fragility

The canonical fit is Hoffmann et al., **"Training Compute-Optimal Large Language Models" (2203.15556, Mar 2022)**: over 400 runs, three estimation approaches, converging on \(L(N,D) = E + A/N^{\alpha} + B/D^{\beta}\) and the ~20 tokens/parameter rule. Approach 3 (a single parametric fit via L-BFGS on Huber loss, \(\delta=10^{-3}\), over a grid of initializations) is the one everyone reuses — and the one that broke.

**Besiroglu et al., "Chinchilla Scaling: A replication attempt" (2404.10102, Apr 2024, Epoch AI)** attempted to replicate Approach 3 from plot-extracted data and found:

- The reported Approach-3 estimates are **inconsistent with Hoffmann's own Approaches 1 & 2** and fit the reconstructed data poorly.
- The reported confidence intervals are **implausibly narrow** — intervals that tight would require >600,000 experiments; they ran <500.
- Two concrete causes: (a) a poor loss scale caused the optimizer to **terminate before convergence**; (b) the parameters printed in the paper body were **rounded**, introducing substantial bias in downstream predictions (the TeX source had more digits).
- Their re-derivation (higher data exponent \(\beta\) and irreducible loss \(E\)) restores consistency with the 20-tokens/param rule.

**Lessons for a competition:** (i) optimizer convergence criteria and loss scaling change fitted exponents materially; (ii) never report \(\alpha\) without a bootstrap/Huber-fit CI; (iii) rounding fitted constants silently corrupts extrapolation — carry full precision; (iv) cross-validate any parametric fit against a non-parametric method (IsoFLOP interpolation) before trusting the exponent.

### 1.2 Data-constrained scaling

**Muennighoff et al., "Scaling Data-Constrained Language Models" (2305.16264, May 2023, NeurIPS 2023)**: 400+ runs, 10M–9B params, up to 1500 epochs. Key results: up to **4 epochs of repeated data is nearly free** (negligible loss change vs. unique data); the value of repetition has a **half-life at ~16 epochs** (\(R_D^* \approx 15\)), after which returns decay rapidly toward zero; they extend Chinchilla with an "effective data" term \(D' = U_D + U_D R_D^*(1 - e^{-R_D/R_D^*})\). Adding code data buys ~2× effective scaling. Directly relevant: a 6h/1-GPU competition is *always* data-constrained relative to model size, so submissions will live in the multi-epoch regime — epoch counts must be logged and capped, because 4–16 epochs quietly converts a "sample efficiency" contest into a "memorization tolerance" contest.

### 1.3 How many runs, at what sizes, give a reliable α?

Empirical practice from the papers that actually do per-architecture comparisons:

| Paper | Protocol | Scale span |
|---|---|---|
| Gadre et al. (2403.08540) | 104 models, 3 data distributions | 0.011B–6.9B, multiple token/param ratios |
| Kimi Linear (2510.26692, Oct 2025) | 5 model sizes, per-size grid-searched HPs, Chinchilla methodology | small ladder → 3B active / 48B total |
| xLSTM scaling laws (2510.02228, Oct 2025, ICLR 2026) | IsoFLOP **and** parametric \(L(N,D)\) fits, cross-checked | 80M–7B, 2B–2T tokens |
| Model ladders (2412.04403, Dec 2024, OLMo team) | 4 sizes × multiple durations, 1% of target compute | 190M–1.3B ladder → predict 7B/13B |

Convergent guidance: **≥4–5 points spanning ≥1.5 orders of magnitude in compute** is the floor for an identifiable \((E, A, \alpha)\); the model-ladder paper explicitly finds that "using less compute to train fewer ladder models tends to deteriorate predictions." Each point must be **near its own compute-optimal frontier** — a badly-tuned small run biases \(\alpha\) far more than adding another run fixes. Two schedule caveats matter operationally: cosine LR only achieves its optimum when the cycle length equals the training duration, so **Hägele et al., "Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations" (2405.18392, May 2024)** recommend constant-LR + cooldown, which matches cosine and lets you **cool down from intermediate checkpoints** — i.e., one long run yields many valid (C, L) points. This is the single most useful trick for a time-capped competition.

### 1.4 Per-architecture exponent comparisons (the precedent for your scoring)

- **Mamba (2312.00752, Dec 2023)**: Chinchilla-protocol IsoFLOP curves, 125M–1.3B on the Pile; first attention-free model to match a strong "Transformer++" (RoPE/SwiGLU/RMSNorm) recipe, with the gap favoring Mamba as sequence length grows. RWKV/RetNet baselines OOM'd at 8k context — an early warning that *implementation maturity* contaminates architecture comparisons.
- **RWKV (2305.13048, May 2023)**: RNN with linear attention; scaling claims rest on smaller ladders and are weaker evidence than Mamba's.
- **xLSTM (2405.04517, May 2024)** and especially the dedicated follow-up **"xLSTM Scaling Laws" (2510.02228)**: xLSTM is **Pareto-dominant** over Transformers in loss-vs-compute across 80M–7B; power-law exponents stay constant in the over-training regime (mirroring Gadre et al.'s Transformer result); compute-optimal xLSTM is *larger* than compute-optimal Transformer at equal budget — i.e., architectures differ in **both** exponent and optimal allocation, so a single-tier competition can crown the wrong winner.
- **Kimi Linear (2510.26692, Oct 2025)**: hybrid KDA:MLA 3:1; with identical recipe, ~**1.16× compute efficiency** over full-attention MLA under Chinchilla-style fitting, validated at 3B-active/48B-total and 1.4T tokens. Note the pattern: the field now considers "trained 5 sizes + grid-searched HPs + held recipe constant" the minimum evidentiary bar for an architecture claim.

**Takeaway:** the literature's standard is per-architecture *frontier* comparison (IsoFLOP or parametric), not single-point deltas. Your competition should score the fitted frontier, not the single 6h number.

---

## 2. Loss → downstream mapping: when can you honestly claim transfer?

- **Gadre et al., "Language Models Scale Reliably with Over-Training and on Downstream Tasks" (2403.08540, Mar 2024)**: the optimistic anchor. 104 models (0.011B–6.9B) on C4/RedPajama/RefinedWeb. Two results: (a) loss extrapolates across over-training (predicted a 1.4B/900B-token run — 32× over-trained — from 300× less compute); (b) **average top-1 error across a task suite** follows a smooth map from perplexity, \(\mathrm{Err}(L) = \epsilon - k\,e^{-\gamma L}\) (a sigmoid in accuracy space), predicted within ~1pp for a 6.9B model from 20× less compute. Crucial qualifiers: the mapping works for **aggregate** error, among models **trained on the same data**, and is far less reliable per-task.
- **Lourie, Hu & Cho, "Scaling Laws Are Unreliable for Downstream Tasks: A Reality Check" (2507.00885, Jul 2025; Findings of EMNLP 2025)**: the necessary counterweight. Meta-analysis of published downstream scaling data: smooth, predictable (post-transformation linear) scaling holds in only **39% of cases** (18 of Gadre's 46 tasks). The rest are inverse, nonmonotonic, noisy, trendless, or "breakthrough." Seemingly benign changes — pretraining corpus, validation set, task formatting — can **flip the sign** of a trend. Their phrase: "perplexity is not all you need."
- **Ruan, Maddison & Hashimoto, "Observational Scaling Laws" (2405.10938, May 2024; NeurIPS 2024 spotlight)**: bypass training entirely — build scaling laws from ~100 public models in a low-dimensional **capability space**; families differ only in compute→capability efficiency. Emergent phenomena become smooth sigmoids in this space and their transition points are forecastable from models only slightly above chance; agentic performance (GPT-4-level) is predictable from non-agentic benchmarks; effects of CoT/self-consistency are predictable too. For you: a *shared capability axis* across submissions (rather than raw benchmark averages) is the statistically sound way to compare architectures with different efficiency profiles.
- **Ge et al., "Capability Salience Vector" (2506.13216, Jun 2025; ACL 2025)**: raw validation loss treats all tokens as equal; CSV learns per-token importance weights \(W=\{w_{s,i}\}\) aligned to "meta-capabilities," fit jointly with a sigmoidal loss→accuracy law (Levenberg–Marquardt). It substantially improves downstream predictability **across data distributions** — exactly the regime where Gadre's mapping breaks.

**Honest-claim rule for the competition:** you may claim "transfers" only for (a) validation loss itself (reliably extrapolable in-regime), and (b) *aggregate* downstream error on tasks you have **empirically verified to scale smoothly within your own ladder** (the 39% paper's core demand). Per-task downstream claims, and any claim about capabilities not yet on the smooth part of their sigmoid at your scale, are not honestly scoreable.

---

## 3. μP/μTransfer as the fairness substrate

**Yang & Hu, "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer" (2203.03466, Mar 2022; NeurIPS 2022)**. Maximal Update Parametrization (μP) makes per-layer updates width-invariant, so optimal LR/init/multipliers are stable across width; empirically also across **depth (pre-LN only), batch size, sequence length, and training time**, given minimums (~width 256, depth 4, batch 32, 5k steps). Explicitly **not** transferable: regularization HPs (dropout, weight decay — they scale with data, not width), init scale across depth, and anything in post-LN Transformers across depth.

Why this is the fairness substrate for your competition:

1. **Without μP, the small-scale winner is partly a hyperparameter-tuning lottery.** An architecture whose optimum happens to sit near the default HPs at 100M wins at tier 1 and loses at 3B. μP (or organizer-mandated per-tier HP budgets) converts "who got lucky at small scale" into "whose architecture actually scales."
2. **Transfer across architectures is not free.** μP's guarantees are derived per-parametrization-class; novel layers (new gating, new mixers, new normalization) require re-deriving the abc-parametrization (the Tensor Programs framework, extended to general architectures in TP VI, 2310.02244), and new optimizers (Muon-style orthogonalized updates, schedule-free) change which HPs transfer. Practical rule: require submissions to either (a) implement the standard μP multipliers for their custom layers and demonstrate LR-stability across a 4× width sweep at tiny scale, or (b) accept a fixed organizer HP sweep budget per tier. Option (a) is itself a meaningful, checkable engineering gate.
3. **Known breakers to police:** embedding/softmax parametrization, depth transfer (keep depth fixed within a tier or use depth-aware variants), regularization HPs (fix them centrally), and sub-threshold model sizes (below width ~256, μP's stable regime hasn't kicked in — don't let tier 1 be smaller than that).

---

## 4. Intermediate-checkpoint scoring

Final loss after 6h conflates "fast learner" with "good asymptote." The fix is to score the **loss-vs-tokens curve** as the object of interest:

- **BabyLM 2025 made this mandatory** (CFP 2502.10645; findings 2025.babylm-main.28): submissions must upload checkpoints at **every 1M words up to 10M, every 10M up to 100M, every 100M up to 1B**, evaluated on a **fast** pipeline (20% subsample of tasks, zero-shot only, no finetuning); the **full** pipeline (incl. finetuning on subsampled ≤10k-example GLUE tasks, with highly correlated tasks like CoLA/SST2/QNLI removed) runs only on the final checkpoint. They also added age-of-acquisition-style trajectory evals (`eval_aoa`).
- **Metrics that capture sample efficiency:** (a) **tokens-to-threshold** — tokens needed to cross a fixed loss/accuracy level (directly the "sample efficiency" quantity; robust because it's a horizontal, not vertical, comparison); (b) **area-under-the-loss-curve** over log-tokens — rewards early learning without ignoring the endpoint; (c) per-checkpoint downstream scores on the fast eval, giving a trajectory rather than a point.
- **Why better than final loss:** final loss is one noisy draw from the end of an arbitrary budget; the curve exposes crossing points (architecture A wins at 100M tokens, B at 1B — exactly the question your competition exists to ask), and curve integrals are far less sensitive to the LR-decay tail. Pair with **constant-LR + cooldown** (2405.18392): evaluate at constant-LR checkpoints (comparable mid-training states) plus one cooled-down endpoint per tier (comparable "finished" states).

---

## 5. BabyLM deep dive (2023–2026)

**Setup.** Data-limited pretraining: **Strict = 100M words, Strict-Small = 10M words** (roughly what a human child hears by age ~12). Evaluation: BLiMP-style zero-shot grammatical minimal pairs, (Super)GLUE finetuning, MSGS, plus later multimodal, interaction, human-likeness, and AoA tasks; from 2025, mandatory intermediate checkpoints + fast/full eval split; from 2026, server-side scoring against held-out targets (`babylm-org/babylm-eval`).

**What won, year by year:**

- **2023** (findings: 2023.conll-babylm.1): **ELC-BERT** (Charpentier & Samuel) won *both* strict tracks — an LTG-BERT backbone (NormFormer-style extra normalization, GEGLU, DeBERTa-style disentangled attention, scaled init) plus learned per-layer mixing of all previous layers' outputs. Trained **>450 epochs (Strict) / >2000 epochs (Strict-Small)** — extreme multi-epoch, consistent with 2305.16264's finding that repetition retains value far past 4 epochs at tiny data scales. Beat Llama-2 and RoBERTa skylines trained on ~100× more data. **Architecture + training regime, not curriculum, won.**
- **2024** (findings: 2412.05149 / 2024.conll-babylm.1): **GPT-BERT** (same authors) — a hybrid MLM+CLM objective in one stack — won both strict tracks. The organizers' mixed-effects regression over all submissions: significant positive effects for **training-objective innovation (β=4.5, p<0.001), dataset construction (β=4.8, p<0.05), architectural innovation (β=3.5, p<0.05)**; **curriculum learning was popular and ineffective (β=−3.6, p=0.055)**; "linguistic bias" approaches actively hurt (β=−7.3).
- **2025** (findings: 2025.babylm-main.28; CFP 2502.10645): added an Interaction track (teacher feedback) and compute caps. Strict-track NLP winner: **masked diffusion LM with frequency-informed training** (Georgiou et al., 2025.babylm-main.38) — LTG-BERT backbone + AdaLN timestep conditioning, cosine/bimodal noise schedules, rare-token-prioritized masking, 126.6M params, 10 epochs. Headline finding: **"We do not observe a complete correlation between training FLOPs and performance"** — objectives and architectures, not raw compute, drove the leaderboard.
- **2026** (CFP **2602.20092**, Feb 2026, "BabyLM Turns 4 and Goes Multilingual"): new **Multilingual track** on BabyBabelLM (2510.10159) — English/Dutch/Chinese, custom mixtures within a 100M-token budget adjusted per-language by a **Byte Premium**; Multimodal and Interaction folded into Strict/Strict-Small as allowed techniques rather than tracks; detoxified data re-release; EMNLP 2026 workshop.

**Lessons for your design:** (1) fixed-data + fixed-eval + open-method is a proven, fair format; (2) winners came from **objective/architecture innovation every single year**, while curriculum learning — the most popular approach — never won on average: don't overweight fashionable-but-weak levers in scoring; (3) FLOPs ≠ outcome at fixed data, so cap *both* time and tokens and log epochs; (4) fast/full eval split and checkpoint submission are cheap and worth copying; (5) hidden/server-side targets (2026) prevent eval overfitting; (6) the Byte-Premium idea generalizes: if you ever allow tokenizer changes, normalize budgets in bytes, not tokens.

---

## 6. Data-efficiency literature and the attribution problem

- **DoReMi (2305.10429, May 2023, NeurIPS 2023)**: Group-DRO domain reweighting on a 280M proxy transfers to an 8B model (30× larger): +6.5pp average few-shot accuracy, 2.6× fewer steps to parity. Proof that **mixture weights found at small scale transfer up** — and therefore that data mixing is a confound you must control.
- **DCLM (2406.11794, Jun 2024)**: a controlled testbed (240T-token pool, fixed recipes, 412M–7B, 53 evals). Finding: **model-based filtering is the dominant lever**; DCLM-Baseline trains 7B to 64% MMLU with 6.6× less compute than Llama-3-8B's comparable average. Data curation alone produced gains larger than most architecture papers' — the attribution problem in one number.
- **FineWeb / FineWeb-Edu (2406.17557, Jun 2024)**: 15T tokens from 96 Common Crawl snapshots with fully ablated dedup/filter choices; FineWeb-Edu = Llama-3-70B educational-quality classifier (0–5, keep ≥3 — discards 92%) → 1.3T tokens, large MMLU/ARC/OpenBookQA gains.
- **Pruning:** Sorscher et al., **"Beyond neural scaling laws" (2206.14486, Jun 2022)** — with a good pruning metric, error can beat power-law scaling (theoretically exponential) in dataset size; **Marion et al., "When Less is More" (2309.04564, Sep 2023)** — at LLM pretraining scale, simple **perplexity-band filtering** (keep mid-perplexity samples, scored by a *small* reference model) beats EL2N/memorization-style metrics. Dedup (Lee et al., 2107.06499) is table stakes.

**Can miners choose data curation within a fixed token budget, fairly?** Yes, but only if you make it a *declared, separately scored dimension*. Data curation changes both \(E\) (irreducible loss on your eval distribution) and the effective \(\alpha\); DCLM shows its effect size rivals architecture. Recommended: (a) **default fixed dataset** (organizer-released, deduped, documented) for the main architecture track — this is what makes α comparisons interpretable; (b) an optional **open-data track** with a fixed *unique-token budget from a fixed pool* (curation allowed, no external data, budget in bytes), scored separately; (c) require the training mix + any proxy-model scoring pipeline to be logged so data vs. architecture attribution is at least post-hoc analyzable. DoReMi-style proxy training must count against the 6h budget or it imports unbounded hidden compute.

---

## 7. Emergent abilities and small-scale blindness

- **Schaeffer, Miranda & Koyejo, "Are Emergent Abilities of Large Language Models a Mirage?" (2304.15004, Apr 2023; NeurIPS 2023 outstanding paper)**: >92% of claimed BIG-bench emergent abilities appear only under **nonlinear or discontinuous metrics** (exact-match accuracy, multiple-choice grade). Switch to continuous metrics (token edit distance, Brier score) and the same outputs improve smoothly. Two failure modes named: metric-induced sharpness, and **too few test samples to measure small models above floor**.
- **Wu & Lo, "U-shaped and Inverted-U Scaling behind Emergent Abilities" (2410.01692, Oct 2024)**: slice benchmarks by difficulty — hard questions show **U-shaped** scaling (get *worse* before better), easy questions inverted-U then steady improvement; the two cancel, faking stagnation, then the easy-slice reversal produces the "emergence" jump. Their Slice-and-Sandwich pipeline predicts emergence thresholds from sub-threshold models using a continuous Target-Conditioned Brier score.

**Consequences for scoring below ~3B:** (i) multi-step reasoning, long-context retrieval, robust instruction following, and agentic behavior are at or near floor at 100M–3B — **do not score them**; a near-zero, noise-dominated metric rewards luck and selects against nothing; (ii) where you do use downstream tasks, use **continuous metrics** (Brier, per-token log-prob on the correct choice, edit distance) with enough items to lift small models off the floor; (iii) treat any apparent "breakthrough" on your leaderboard as a metric artifact until proven otherwise; (iv) difficulty-slice your eval tasks so a submission that improves the *hard* slice (U-shaped: transiently worse) isn't penalized for genuine progress.

---

## 8. Concrete recommendation: the protocol

**Tiers and budget.** Three compute tiers, not one: **T1 ≈ 100M params, T2 ≈ 400M, T3 ≈ 1.5B** (all within 6h/1 GPU at fixed tokens-per-param ≈ 20–40, i.e. mildly over-trained; keep depth/width ratios within μP's validated regime, width ≥ 256). Three tiers is the minimum that separates *exponent* from *offset*; two tiers can only rank, three tiers can fit.

**Fairness substrate.** Mandate μP-style parametrization with organizer-published base HPs per tier; submissions with novel layers must pass an automated **LR-stability gate** (optimal LR within ±2× across a 4× width sweep at ≤50M scale) or accept a fixed organizer HP-sweep budget (same GPU-minutes for everyone). Fix regularization HPs, schedule shape, tokenizer, and eval data centrally. Schedule: **constant LR + cooldown** (2405.18392), with checkpoints at fixed token counts (log-spaced, BabyLM-style) and a cooldown branch from the final checkpoint of each tier.

**Scored objects (per submission).**
1. **Curve score (40%)**: area under the validation-loss-vs-log-tokens curve at T2, plus tokens-to-threshold for 2–3 pre-registered loss levels. This is the sample-efficiency term.
2. **Frontier score (30%)**: fit \(L(C) = E + A\,C^{-\alpha}\) per submission on the three tier endpoints (Huber loss, L-BFGS grid, full precision, **bootstrap CI over 1k resamples of checkpoints/seeds**). Score predicted \(L\) at a reference budget ~10× above T3. Gate: the fit must achieve held-out \(R^2 \geq 0.98\) on a T2.5 probe tier the organizers run (or a withheld cooldown point); failed fits fall back to frontier-less ranking. Report \(\alpha\) with CIs; **never** score raw \(\alpha\) — Besiroglu showed how easily it is mis-estimated even with 400 runs.
3. **Downstream score (20%)**: aggregate error over a *small, pre-validated* task battery — only tasks whose loss→metric mapping you have empirically confirmed smooth on your own baseline ladder (expect ~40% of candidate tasks to qualify, per 2507.00885), continuous metrics only, difficulty-sliced reporting. Map loss→error with the Gadre sigmoid or a CSV-style reweighting (2506.13216) fit on organizer baselines, not per-submission.
4. **Trajectory/downstream-checkpoint bonus (10%)**: fast-eval (BabyLM-2025-style, 20% subsample, zero-shot) on intermediate checkpoints; rewards architectures that *become capable sooner*, the trajectory analogue of sample efficiency.

**Scaling-exponent bonus, honestly bounded.** Award the "scaling bonus" on the **frontier score's slope with its CI**: bonus ∝ improvement in predicted \(L\) at the 10× reference budget, shrunk by CI width (wide CI → bonus → 0). Explicitly excluded from claims: any capability whose sigmoid transition sits above T3 scale (§7), per-task downstream superiority (39% problem), and any statement beyond "transfers to ~10–30× the largest tier" — the literature supports loss extrapolation over ~2 orders of magnitude in compute (Gadre: 300× for loss, 20× for aggregate downstream), so a 100M–1.5B ladder honestly licenses claims about ~10B–30B, and **only directional** claims at 100B.

**Data policy.** Main track: fixed organizer dataset (deduped, documented, byte-normalized budget), epoch cap ≈ 4–8 with mandatory logging (2305.16264). Optional open-data track with fixed pool + fixed unique-token budget, scored separately, curation compute counted in the 6h.

**Anti-gaming.** Hidden held-out eval targets with server-side scoring (BabyLM 2026 model); eval-loss computed on organizer data only; checkpoint hashes logged at emission time; μP gate and epoch caps enforced by the harness, not the honor system.

**What this buys you:** the winner is the architecture with the best *fitted frontier and learning trajectory* under identical data, schedule, and HP-transfer rules — i.e., the design most likely to keep winning at 10B+, with the competition's claims explicitly bounded to what the scaling literature says is knowable from 3–5 small runs.
