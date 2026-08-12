# Appendix 12 — Aggregating Metrics into a Reward Scalar
> Research appendix for the Prism v3 evaluation proposal (`docs/spikes/prism-v3/`). Produced 2026-08-06 via arXiv/web research. Non-normative spike document.

# Aggregating Heterogeneous Metrics into One Reward Scalar: A Survey for a Token-Incentivized Architecture Competition

**Citation corrections up front** (verified 2026-08-06): "The Benchmark Lottery" is **arXiv 2107.07002** (Dehghani et al., Jul 2021) — arXiv 2407.08702 is an unrelated dipolar Bose-gas physics paper. GSM1k is **arXiv 2405.00332** (Zhang et al., May 2024). All other IDs cited check out (2403.04132 = Chatbot Arena, 2211.09110 = HELM, 2504.20879 = Leaderboard Illusion, 2411.00640 = clustered-SE eval statistics).

---

## 1. Taxonomy of aggregation methods

Every aggregation pipeline is **(a) normalize per metric → (b) combine → (c) map to payout**. The exploitable surface lives in all three stages.

### 1a. Normalization choices (the deepest decision)

| Method | Formula | Failure mode |
|---|---|---|
| Min-max (field-relative) | \((x - \min)/(\max - \min)\) | **Field-dependent**: adding a weak submission rescales everyone (an IIA violation, §3). Outlier-fragile. |
| Z-score | \((x-\mu)/\sigma\) | Field-dependent; implicitly upweights noisy/high-variance metrics (large \(\sigma\) compresses); assumes symmetric noise. |
| Chance-correction / fixed anchors | \(s = (x - c)/(h - c)\), clipped \([0,1]\) | Requires choosing anchors \(c\) (chance floor) and \(h\) (excellence ceiling), but is **field-independent and pre-registrable**. Precedents: BIG-bench normalized preferred metric, \(100(x-\text{low})/(\text{high}-\text{low})\), arXiv 2206.04615 (Jun 2022); Open LLM Leaderboard v2, \(100(x-\text{chance})/(1-\text{chance})\), floors at 0 (Jun 2024). Cohen's \(\kappa\) and MCC are the classic chance-corrected agreement metrics. |
| Rank / quantile transform | order statistics | Field-dependent by construction; see 1c. |

**Verdict**: only fixed-anchor normalization survives economic adversaries, because \(\min, \max, \mu, \sigma\) become attack targets the moment miners can influence the comparison set (Sybil weak entries).

### 1b. Combination operators

- **Arithmetic mean** (weighted sum): fully compensatory — a 0 on recall is silently offset by bpb gains. For *ratios* it is provably inconsistent: Fleming & Wallace (CACM 29(3), 1986) proved the geometric mean is the unique average of normalized ratios with the multiplicative property (reference-independent rankings); this is why SPEC CPU uses geomean.
- **Weighted geometric mean** \(\exp(\sum_k w_k \ln g_k)\): penalizes weak axes multiplicatively; compensation cost of a zero axis is total. Collapses at \(g_k \approx 0\) — pair with gates/floors rather than \(\epsilon\)-hacks.
- **Power means** \(M_p = (\sum_k w_k g_k^p)^{1/p}\) with \(p<0\): tunable weak-axis emphasis; \(p \to -\infty\) converges to \(\min\) (pure worst-axis). Kaggle's Jigsaw Unintended Bias competition (2019) used exactly this: \(0.25\cdot\text{AUC}_{\text{overall}} + \sum 0.25 \cdot M_{-5}(\text{subgroup AUCs})\) with \(p=-5\) to force worst-subgroup improvement.
- **Rank aggregation** (Borda, mean rank): Borda ≡ mean win rate up to a monotone linear map (Hardt & Recht, *The Emerging Science of Machine Learning Benchmarks*, ch. 12, 2025, mlbenchmarks.org). Immune to monotone metric distortions, but inherits Arrow's IIA failure (1950): adding irrelevant weak models can flip top-2 order; Condorcet cycles; Kemeny optimal aggregation is NP-hard. HELM's mean win rate is the canonical instance — and its abandonment is the canonical lesson (§2).
- **Bradley-Terry / Elo from per-instance outcomes** (arXiv 2403.04132, Mar 2024): fit \(P(i \succ j) = e^{\beta_i}/(e^{\beta_i}+e^{\beta_j})\) by MLE over item-level pairwise outcomes; sandwich (Huber) SEs, pivot bootstrap, multiplicity-corrected simultaneous CIs, e-values for anytime monitoring, active pair-sampling. Strengths: uses instance-level information, honest CIs. Weaknesses for us: purely relative scale (no absolute quality), assumes transitivity, **field-dependent**, and catastrophically vulnerable to selective submission (§3).
- **IRT-based aggregation**: model item difficulty \(b_i\) and discrimination jointly with model ability \(\theta_m\) (tinyBenchmarks, arXiv 2402.14992, Feb 2024 — ~100 curated items estimate MMLU within ~2%; "Efficient Benchmarking", arXiv 2308.11696, Aug 2023 — anchor points, and a finding that HELM's mean win rate is "unreliable and gameable"). Multidimensional IRT gives one \(\theta\) per axis — principled uncertainty per axis — but still requires a scalarization step for rewards, and adds misspecification risk. Best used to *shrink eval cost* and *quantify per-axis uncertainty*, not as the final combiner.
- **Pareto fronts (no scalarization)**: dominance, hypervolume contribution, NSGA-II (Deb et al., IEEE TEC 2002); performance profiles (Dolan & Moré, Math. Prog. 2002) as the solver-benchmark analog of a dashboard. Cannot directly emit validator weights — see §5.
- **MCDA methods**: TOPSIS (Hwang & Yoon 1981; distance to ideal/anti-ideal — sensitive to normalization choice), weighted product model (≡ weighted geometric mean), weighted sum (≡ arithmetic mean). The MCDA literature's most useful export is **weight stability-interval analysis**: report the weight neighborhood within which the ranking is unchanged.
- **Lexicographic gates + score hybrid**: hard per-axis thresholds first, scalar score only among survivors. Precedents: MLPerf Inference closed division accuracy gates (99%/99.9% of reference model, per-benchmark bands; verified in mlcommons inference rules) and ARC Prize's 85% grand-prize threshold + \$/task efficiency cap (§2). Gates are the single most effective anti-degenerate-win device; they convert "must-have" axes from compensable score terms into constraints.

### 1c. A theoretical boundary

Hardt & Recht (ch. 12) formalize an empirical Arrow analog for benchmarks: any aggregation that is far from single-task is sensitive to *irrelevant changes* — either (i) addition of weak models (rank-based methods fail) or (ii) monotone rescaling \(s \mapsto as+b\) of a task metric (score-based methods with field-relative normalization fail). You cannot have both invariances. Consequence: **choose which invariance to buy, and fix the other dimension by governance (pre-registration)**, not by clever math.

---

## 2. What real leaderboards use, and why they changed

| Leaderboard | Aggregation | Documented failure → change |
|---|---|---|
| **HELM** (arXiv 2211.09110, Nov 2022) | Mean win rate across scenarios (Borda variant) | MWR depends on the comparison set and flips on small score perturbations; arXiv 2308.11696 showed it unreliable/gameable. **HELM Capabilities (2025) switched to mean normalized score**, stating exactly these two reasons. |
| **Open LLM Leaderboard v1** (HF, 2023) | Average of 6 raw-ish scores (ARC, HellaSwag, MMLU, TruthfulQA, WinoGrande, GSM8K) | Saturation near ceiling; contamination (fine-tunes on MMLU auxiliary-train/test-adjacent data); TruthfulQA-style metric stuffing. **v2 (26 Jun 2024)**: harder suite (MMLU-Pro 2406.01574, GPQA 2311.12022, MuSR 2310.16049, MATH-L5 2103.03874, IFEval 2311.07911, BBH 2210.09261) + **chance-baseline normalization with subtask-level normalization before averaging**. Remaining weakness: still arithmetic mean (compensatory), still public test data. |
| **BIG-bench** (arXiv 2206.04615, Jun 2022) | Task-authored anchors: \(100(x-\text{low})/(\text{high}-\text{low})\), arithmetic mean over 204 tasks | Anchor calibration is rough (human rater mean ≈ 45, best rater ≈ 80 NPM); authors themselves warn the aggregate masks per-task regressions and "breakthrough vs. gradual" heterogeneity. Lesson: anchor quality bounds composite quality. |
| **MLPerf** (MLCommons) | **No scalar aggregate** — per-benchmark results; **closed division** = fixed model + accuracy gates (99%/99.9% bands), all scenarios mandatory; **open division** = arbitrary models, must report accuracy | Gate + division design is the durable export: it solved the recipe-vs-hardware conflation that made cross-submission comparison meaningless. Marketing still cherry-picks per-benchmark wins — the cost of refusing to scalarize. |
| **Chatbot Arena** (arXiv 2403.04132, Mar 2024) | BT MLE on pairwise human votes; sandwich CIs, multiplicity-corrected simultaneous intervals, e-values, anomaly detection | **The Leaderboard Illusion** (arXiv 2504.20879, Apr 2025): selective disclosure — 27 private Meta variants before Llama-4, best-of-n publication bias; data asymmetry (two providers ≈19–20% of battles each vs 29.7% for 83 open models combined); up to 112% relative Arena-score gain from arena-distribution access. Fixes proposed: publish all variant scores, no retraction, uniform private-testing limits. Also style bias (verbosity) → later style-control BT. |
| **ARC Prize** (2024–2025; tech report arXiv 2601.10904, Jan 2026; benchmark lineage: Chollet's *Measure of Intelligence*, arXiv 1911.01547, Nov 2019) | **Single metric**: pass@2 accuracy on 120 private tasks; 85% gate for grand prize; efficiency as a *constraint* (~\$/task cap), not an axis; semi-private set for live leaderboard, **private set scored once at the end** | Deliberate rationale: a scalarization of ambiguous dimensions is more gameable than one sharply defined target; pass@2 absorbs task ambiguity; one-shot private scoring kills submission-feedback overfitting. Cost: grand prize unclaimed (gates can stall payouts — in an emissions context this means burn, not redistribution). |
| **Kaggle** | Public LB (dev subset) / **private LB (final, only scored round)**, ≤2 final submissions; composite metrics where needed, e.g. Jigsaw 2019's power-mean composite | Public-LB overfitting shakeups are folklore-documented; the 2-submission cap is best-of-n defense. Composite gaming happens at the metric's seams (Jigsaw teams optimized the \(p=-5\) subgroup structure directly — the intended behavior, but proof that competitors optimize *the exact functional form you publish*). |
| **Dynabench** (arXiv 2104.14337, Apr 2021) | No fixed aggregate; dynamic human-and-model-in-the-loop rounds | Argues static average-case benchmarks saturate; worst-case/adversarial rounds are the anti-saturation mechanism. Relevant to probe metrics (recall/copying), which saturate fastest. |

---

## 3. Gaming aggregation — and what resists it

Attack surface, mapped to defenses:

1. **Metric stuffing** (max the easiest, weakest-weighted axis; v1-Leaderboard TruthfulQA pattern). Defense hierarchy: gates (threshold the axis) > geometric/power mean with \(p<0\) (weak axes dominate marginal utility: elasticity of \(g_k\) is \(w_k/g_k\), so a *low* axis — not an easy one — has the highest return to effort; this is the property you want) > arithmetic (compensates silently). Cap all normalized scores at 1 so saturated axes yield zero marginal reward.
2. **Variance exploitation / best-of-n**: submit many variants, disclose the max. Documented at scale in 2504.20879 (27 private variants). Defenses: one scored submission per miner per round; **rank by lower confidence bound** (LCB) so small-sample luck is priced, not rewarded; minimum per-metric item counts.
3. **Selective submission / retraction**: 2504.20879's core finding. Defense: every evaluated submission's scores are published permanently (their own recommendation); no retraction; private final eval is the *only* scored round (ARC Prize / Kaggle private LB).
4. **Collusion**: pairwise/preference aggregation (BT on miner-vs-miner battles) is Sybil- and collusion-susceptible — miners can farm wins against their own weak entries. Rank aggregation is Sybil-sensitive too (add weak models, flip top-2; the IIA example from ch. 12). Defense: **absolute scoring against fixed, pre-registered anchors**; reference evals re-run by the harness (never trust miner-reported numbers); cap submissions per hotkey.
5. **Field-composition manipulation**: z-scores, min-max, ranks, BT ratings all move when weak/strong entries arrive. Only fixed-anchor absolute scores are invariant.
6. **Threshold saturation**: any axis that saturates stops differentiating and shifts all leverage to other axes (GSM8K → GSM1k, arXiv 2405.00332, found up to 13% drops for some model families; GSM-Symbolic, arXiv 2410.05229, Oct 2024, showed large variance under mere re-instantiation). Defense: pre-registered **saturation tripwires** (when top-quartile spread on an axis < \(u\), the axis is regenerated harder or its weight redistributes by a pre-declared rule), plus template-regenerated probes (GSM-Symbolic style) for recall/copying/reasoning.

**Manipulation-resistance ordering** (most → least): **fixed anchors + lexicographic gates + power/geometric mean** > fixed-anchor arithmetic mean > IRT/BT latent scales > rank/Borda > field-relative min-max/z-score. The last two are unacceptable under economic adversaries.

---

## 4. Statistical robustness

- **CIs on composite scores**: arXiv 2411.00640 (Evan Miller, Nov 2024) is the reference: report CLT SEs per metric; **cluster standard errors at the dependency unit** (passage/domain/template family — clustered SEs run up to 3× naive on popular evals); use **paired-difference analysis** because frontier models' per-question outcomes correlate 0.3–0.7 — a free variance reduction when comparing submissions on the same harness. For the composite itself: **clustered bootstrap of the full pipeline** — resample clusters, recompute normalization → group means → composite, take percentile CIs (B ≥ 1000).
- **Sensitivity analysis over weights**: Benchmark Lottery (2107.07002) showed task-subset choice alone produced 6 distinct top-1 orderings across 70 subsets of one benchmark; ~60/70 top-3 disagreements. Mandatory practice: publish ranking under leave-one-metric-out and under ±10–20% weight jitter; choose weights inside a stable region (Kendall \(\tau \geq 0.9\) within the neighborhood) — and say so in the spec. arXiv 2402.01781 (Feb 2024) adds that even trivial perturbations (choice order, answer symbols, scoring rule) shift MCQ ranks up to 8 positions → use hybrid scoring and symbol/position randomization in the harness.
- **Pre-registration**: weights, anchors, gates, eval-suite version/hash, item-generation seeds, CI method, and the emission mapping published (hash-committed) *before* the round opens; changes only via governance with advance notice. This substitutes for the invariance no aggregation can have (§1c).
- **Private final eval as the only scored round**: ARC Prize (semi-private for feedback, private for the single final scoring) and Kaggle private LBs are the working precedent. Combine with template regeneration so "private" means "freshly instantiated," not merely "withheld."
- **Multiplicity**: when publishing many pairwise comparisons, use simultaneous CIs (Arena's chi-square CLT confidence set, 2403.04132) or e-values for anytime-valid monitoring across rounds.

---

## 5. Multi-objective reward distribution (quality vs. efficiency)

When submissions are Pareto-incomparable (better loss, worse cost), options ranked for our setting:

1. **Divisions with separate pre-registered pools** (MLPerf closed/open analog — the strongest precedent). **Fixed-recipe division**: compute budget, data order, tokenizer, and training steps pinned; only the architecture varies → quality axes become commensurate; efficiency axes become *eligibility constraints* (finish within budget) rather than scored dimensions. **Open division**: anything goes; score quality-per-cost via the efficiency group in the composite. Split the emission pool by a fixed, pre-registered ratio (e.g., 75/25). This removes most Pareto ambiguity *by construction* — it never arises.
2. **Explicit pre-registered scalarization** (what §7 does): honest about trade-off prices; requires sensitivity analysis (§4).
3. **Pareto-front reward splitting** (pay all non-dominated submissions, share ∝ marginal hypervolume contribution): elegant but hypervolume is field-dependent (IIA-style sensitivity again) and hard to explain to miners; not recommended as primary.
4. **Tournaments**: BT/Swiss pairwise formats only make sense for subjective/preference axes; for numeric axes they add collusion surface with no information gain. If used anywhere, use fixed reference opponents, never miner-vs-miner.

Gate interaction: follow ARC Prize/MLPerf — efficiency works best as a **constraint** (caps: tokens-to-train, \$/epoch, latency ceiling) in the fixed-recipe division, and as a **scored axis** only in the open division.

---

## 6. Composite vs. dashboard

**For the composite**: validators must emit a weight vector; a scalar is not optional (token emissions are a ranking-to-reward map). Governance also needs an objective, non-negotiable rule — a published formula is a Schelling point that miners can't litigate.

**For the dashboard**: every scalarization hides axis regressions (BIG-bench's own caveat), invites Goodharting of the exact published form (Jigsaw), and destroys information humans need (HELM's multi-metric stance; Model Cards, arXiv 1810.03993, Oct 2018).

**Hybrid (recommended, and what the best-run systems converged on)**: composite → emissions; full dashboard → humans, containing per-metric raw + normalized scores, per-axis 95% CIs, gate pass/fail, the weight-sensitivity report, and **all submissions' scores, non-retractable** (the Leaderboard Illusion fix). HELM pairs a headline number with exactly this transparency layer; ARC Prize pairs a single number with full task-level disclosure after scoring closes.

---

## 7. Concrete recommendation for the architecture competition

**Design: fixed anchors + two-level means + gated weighted geometric mean + LCB payout + pre-registration + private final round.**

### Step 0 — Pre-registration (T−14 days before round \(r\))

Hash-commit and publish: metric list and item generators (seeded), anchors \((c_m, h_m)\), gates, group weights \(w_k\), CI parameters, emission mapping parameters, harness version. Changes only via governance, effective next round.

### Step 1 — Per-metric normalization (fixed anchors, clipped to \([0,1]\))

- **Accuracy-like metrics** (downstream tasks, reasoning, recall/copying probes, long-context buckets):

\[ \tilde{x} = \mathrm{clip}_{[0,1]}\frac{x - c}{1 - c}, \quad c = \text{chance level (e.g., } 1/\text{\#choices; } 0 \text{ for generative)} \]

- **bpb per domain \(d\)** (lower-better, multiplicative scale → log-ratio between fixed anchors):

\[ \tilde{b}_d = \mathrm{clip}_{[0,1]}\frac{\ln b_{\text{chance}} - \ln b_d}{\ln b_{\text{chance}} - \ln b_{\text{ref}}}, \quad b_{\text{chance}} = \log_2 |\mathcal{V}|,\ b_{\text{ref}} = \text{pre-registered reference architecture's bpb} \]

- **Efficiency metrics** (open division only): \(\tilde{e} = \mathrm{clip}_{[0,1]} \ln(e_{\text{cap}}/e_{\text{ref}})^{-1} \ln(e/e_{\text{ref}})\)-style log-ratio to the reference recipe, capped; in the fixed-recipe division these are **pass/fail budget constraints** instead.
- **Stability**: bounded construction, e.g. fraction of seeds completing without divergence minus a loss-spike penalty, mapped to \([0,1]\).

All anchors are constants of the spec — never functions of the current submission field.

### Step 2 — Two-level group means (prevents large families dominating)

Within each group \(k\), average normalized scores **per sub-metric first, then across sub-metrics** (the MuSR/BBH subtask-normalization precedent):

\[ g_k = \tfrac{1}{|S_k|}\sum_{s \in S_k} \tilde{x}_s \]

Groups: **G1** intrinsic fit (mean of \(\tilde b_d\) over domains), **G2** downstream, **G3** procedural probes (recall/copying), **G4** reasoning, **G5** long-context, **G6** train-cost efficiency (open division), **G7** inference efficiency (open division), **G8** training stability.

### Step 3 — Gates (lexicographic, before any scoring)

A submission is **ineligible (share = 0, emission burns)** unless all hold:

- \(g_3 \geq 0.25\) — kills "great loss, zero recall" degenerate wins;
- \(g_8 \geq 0.5\) — stability floor;
- fixed-recipe division: all budget constraints met (MLPerf accuracy-band analog);
- per-metric item counts sufficient that each axis's clustered 95% CI half-width \(\leq \delta\) (pre-registered, e.g. \(\delta = 0.05\)).

### Step 4 — Composite (weighted geometric mean over group scores)

\[ C = \prod_{k} g_k^{\,w_k}, \qquad \sum_k w_k = 1 \]

Suggested starting weights (pre-register, then sensitivity-test): fixed-recipe division \(w = (\text{G1 }0.30,\ \text{G2 }0.20,\ \text{G3 }0.10,\ \text{G4 }0.15,\ \text{G5 }0.15,\ \text{G8 }0.10)\); open division adds G6/G7 at 0.10 each, renormalized. Geometric mean is chosen because: (i) weak axes have the highest marginal elasticity \(w_k/g_k\) — stuffing a saturated axis yields ~nothing, lifting a weak axis yields the most, which is exactly the incentive alignment we want; (ii) Fleming–Wallace consistency for ratio-like normalized scores; (iii) gates absorb the zero-collapse risk. (If governance prefers a knob, generalize to \(M_p\), \(p \in [-2, 0]\); \(p=-5\) is Jigsaw's proven worst-axis setting.)

### Step 5 — CIs and payout ranking

- Clustered bootstrap (clusters = domain/passage/template family), \(B \geq 1000\), recomputing the *entire* pipeline per resample → 95% CI on \(C\) per miner; paired resampling across miners (same eval items) for difference CIs exploiting the 0.3–0.7 per-question correlation (2411.00640).
- **Payout ranking by lower confidence bound**: \(\mathrm{LCB}_i = C_i - 1.645\,\mathrm{SE}(C_i)\) (one-sided 95%). Prices small-sample luck at zero; variance exploitation stops paying.
- Publish: composite, per-axis \(g_k\), CIs, gate status, and the sensitivity annex (leave-one-metric-out rankings; \(\pm 15\%\) weight-jitter ranking; require \(\tau \geq 0.9\) within the jitter neighborhood or the round is flagged for governance review).

### Step 6 — Composite → emissions

Among eligible miners in each division, with pre-registered floor \(F\) (e.g., \(F = 0.10\)) and temperature \(\tau = 2\):

\[ \text{share}_i = \frac{\max(0,\ \mathrm{LCB}_i - F)^{\tau}}{\sum_j \max(0,\ \mathrm{LCB}_j - F)^{\tau}} \times P_{\text{division}} \]

- \(P_{\text{fixed}} : P_{\text{open}} = 0.75 : 0.25\) of the round's miner emission pool (pre-registered); a submission competes in exactly one division (miner's declared choice, no double-dipping).
- No eligible miners (or all below floor) → shares burn, consistent with the fail-closed burn-vector semantics already in the gateway (`GET /v1/weights/latest` burn fallback).
- One submission per hotkey per round; all submissions and scores published permanently; the **private, freshly template-regenerated final eval is the only scored round** (ARC/Kaggle precedent; GSM-Symbolic-style regeneration for G3/G4/G5 items), with a semi-private dev set available during the round for feedback only.
- Leaf emission → `POST /v1/weights/raw` → seal → verify `sealed: true` per the mandatory verification path in `AGENTS.md`.

### Why not the alternatives, in one line each

Rank/Borda (field-composition Sybil attacks, HELM's documented retreat); BT/Elo (collusion + selective-disclosure surface, 2504.20879); z-score/min-max (field-relative, rescaled by weak entries); pure Pareto (no scalar for validators; hypervolume is field-dependent); arithmetic mean of normalized scores (acceptable fallback — it's what Open LLM Leaderboard v2 ships — but compensatory, so it *requires* gates to be safe, and at that point geometric dominates it).

### Reference list (verified)

HELM 2211.09110 (Nov 2022) · Chatbot Arena 2403.04132 (Mar 2024) · Leaderboard Illusion 2504.20879 (Apr 2025) · Clustered SEs for evals 2411.00640 (Nov 2024) · Benchmark Lottery 2107.07002 (Jul 2021) · BIG-bench 2206.04615 (Jun 2022) · tinyBenchmarks (IRT) 2402.14992 (Feb 2024) · Efficient Benchmarking 2308.11696 (Aug 2023) · When Benchmarks are Targets 2402.01781 (Feb 2024) · GSM-Symbolic 2410.05229 (Oct 2024) · GSM1k 2405.00332 (May 2024) · Dynabench 2104.14337 (Apr 2021) · MMLU 2009.03300 (Sep 2020) · MMLU-Pro 2406.01574 (Jun 2024) · GPQA 2311.12022 (Nov 2023) · IFEval 2311.07911 (Nov 2023) · MuSR 2310.16049 (Oct 2023) · BBH 2210.09261 (Oct 2022) · MATH 2103.03874 (Mar 2021) · CheckList 2005.04118 (May 2020) · Measure of Intelligence (ARC) 1911.01547 (Nov 2019) · ARC Prize 2025 Technical Report 2601.10904 (Jan 2026) · Model Cards 1810.03993 (Oct 2018) · Hardt & Recht, *The Emerging Science of Machine Learning Benchmarks*, ch. 12 (2025, mlbenchmarks.org) · Fleming & Wallace, CACM 29(3) (1986) · Bradley & Terry (1952) · Arrow (1950) · Gibbard–Satterthwaite (1973/75) · Dolan & Moré, Math. Prog. (2002) · Deb et al., NSGA-II, IEEE TEC (2002) · Hwang & Yoon, TOPSIS (1981) · MLCommons Inference rules & submission guidelines (closed-division 99%/99.9% accuracy gates) · Open LLM Leaderboard v2 normalization docs (Jun 2024) · Kaggle Jigsaw Unintended Bias evaluation spec (2019, \(p=-5\) power-mean composite) · arcprize.org/competitions/2025 rules.

**Bottom-line formula**: fixed-anchor chance-corrected normalization per metric → two-level arithmetic group means → lexicographic gates (probes/stability/budgets) → weighted geometric mean composite → clustered-bootstrap LCB payout ranking → floor-and-temperature emission shares per division, all parameters hash-committed pre-round and scored once on a private regenerated eval, with the full per-axis dashboard published for humans.
