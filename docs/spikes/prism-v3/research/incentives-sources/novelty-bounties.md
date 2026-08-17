# Rewarding Genuine Novelty That Pays Off — Evidence Review

Research date: **2026-08-16**. Prepared for incentive design of a decentralized neural-architecture-research competition (miners submit LM architectures + training code; operator trains/evals each on ~6 GPU-hours; emissions reward better architectures). Central threat: **copy the leader, tweak slightly**.

Every claim is tagged:
- **[E]** = EVIDENCED (specific numbers/results in a citable source)
- **[R]** = REPORTED (documented in a primary/credible secondary source, but qualitative)
- **[A]** = ANECDOTAL / inference (my reading, not a measured result)

---

## 0. Executive verdict (read this first)

1. **Do not pay for a novelty score computed on submitted code.** Code-similarity novelty is the single most adversarially fragile signal in this entire review. Automated, semantics-preserving obfuscation reduces MOSS similarity for a *known plagiarised pair* to a **median 7.5%**, below the **4.5% median of genuinely unrelated submissions** — i.e. detection becomes literally impossible, not merely degraded **[E]** (Sağlam et al., ICSE-SEET 2024). Mossad defeats MOSS, JPlag and Sherlock in minutes, producing dozens of variants each rated "no more suspicious than legitimate assignments" **[E]** (OOPSLA 2020).
2. **Pay for marginal contribution to a portfolio, not for raw score and not for raw difference.** This is exactly the path Numerai walked over 7 years: originality/leaderboard bonus (exploited, removed) → MMC → TC → back to MMC. Contribution metrics are *self-certifying*: a copy of the leader has near-zero orthogonal component, so it earns ~nothing automatically, with no plagiarism detector in the loop **[E/R]**.
3. **Quality-Diversity (MAP-Elites) is the right archive shape, not pure novelty search.** Novelty search alone provably fails once behaviour space is large and novelty decouples from utility **[E]** (Cuccu & Gomez 2011).
4. **Spread the GPU budget; don't concentrate it.** The strongest citable evidence in metascience is that research impact is a *decelerating* function of funding, so impact-per-dollar falls with grant size **[E]** (Fortin & Currie 2013; Mongeon et al. 2016).
5. **Retro/lineage funding schemes fail in predictable, documented ways**: tea.xyz produced **>150,000** farmed npm packages **[E]**; Optimism RetroPGF 3 became a popularity contest with badgeholders receiving **>15 DMs each** **[E]**.

---

# A. Measuring novelty computationally, and its gameability

## A.1 Code similarity / plagiarism detection: the measured evasion numbers

### Baseline tool families

| Tool | Method | Clone types caught |
|---|---|---|
| MOSS | Winnowing — local fingerprinting of k-gram token hashes (Schleimer, Wilkerson & Aiken, SIGMOD 2003) | Type-1/2 well; Type-3 poorly |
| JPlag | Tokenizes, then **Greedy String Tiling** over token/AST sequences | Type-1/2/3; robust after 2024 normalization work |
| Dolos | Token-based, tree-sitter fronted | Type-1/2 |
| copydetect | Open-source winnowing reimplementation | Type-1/2 |
| CodeBERT / GraphCodeBERT / UniXcoder / code2vec | Learned embeddings, cosine similarity or fine-tuned classifier | Claims Type-4 (semantic) |

### Measured evasion — this is the critical evidence

**Mossad (Devore-McDonald & Berger, OOPSLA 2020)** **[E]**
- Fully automatic program transformation using genetic-programming-inspired repeated insertion of semantics-preserving dead statements.
- Defeats **four** detectors including **MOSS, JPlag, and Sherlock**.
- Runs "in minutes"; from a *single* program generates **dozens of variants**, each classified as **no more suspicious than legitimate assignments**.
- User study: graduate TAs rated Mossad-generated code **as readable as authentic student code** — so a human reviewer screen does not obviously catch it.
- URL: https://doi.org/10.1145/3428206 (2020)

**JPlag obfuscation-resilience study (Sağlam, Hahner, Wittler, Kotthoff — ICSE-SEET 2024)** **[E]** — the single most useful table in this literature:

| Detector | Median similarity, *obfuscated plagiarism* | Median similarity, *unrelated originals* | Median delta | Verdict |
|---|---|---|---|---|
| **JPlag** (with 2024 normalization) | **100%** | 5.8% | **94.2 pp** | separable |
| **Dolos** | 26.7% | 14.3% | **12.4 pp** | "impaired" |
| **MOSS** | **7.5%** | 4.5% | **3.0 pp** | "**impossible**" |

Direct quote: *"reliably differentiating plagiarism from the original is impaired for Dolos and impossible for MOSS."* Cause stated: Dolos and MOSS provide **no defense against automated obfuscation attacks**; reordering and inserting statements "is a viable attack for Dolos and MOSS but no longer for JPlag."
- URL: https://doi.org/10.1145/3639478.3643074 (2024)

**Follow-up (Sağlam et al., arXiv 2505.20158, May 2025)** **[E]**
- Extends to AI-driven obfuscation (LLM rewrites). Reports defense mechanisms recovering detection, with a **notable 8.92 pp** improvement figure and cost of **up to 19 pp** on some configurations.
- Confirms Devore-McDonald & Berger's attack "effectively deceives both JPlag and MOSS, **reducing the calculated similarity between a plagiarism instance and its source below the average similarity of unrelated student solutions**."
- Notes LLMs "significantly exacerbate this problem" and make automated obfuscation "easier than ever."
- URL: https://arxiv.org/pdf/2505.20158 (2025)

**Graded obfuscation ladder (SCITEPRESS 2026 comparative study)** **[E]** — useful because it isolates *which* transformation kills *which* tool:

| Attack level | Transformation | MOSS | JPlag | Dolos | copydetect |
|---|---|---|---|---|---|
| L0 | comments + whitespace only | 94% | 100% | 94% | 100% |
| L1 | identifier renaming + function reordering | **73%** | — | 86% | — |
| L2 | `for` → `while` conversion | **44%** (described as "critical failure") | — | — | robust |

Finding: MOSS's winnowing "is highly sensitive to the specific textual value of tokens, not just their type," and "treats loop constructs as distinct semantic markers rather than equivalent control structures."
- URL: https://www.scitepress.org/Papers/2026/148365/148365.pdf (2026)

### ML-based code similarity is *also* evadable — measured attack success rates

**ALERT (Yang, Shi, He, Lo — ICSE 2022), "Naturalness Aware Attack"** **[E]** — black-box, naturalness-aware identifier substitution:

| Victim model | Vulnerability prediction | **Clone detection** | Authorship attribution |
|---|---|---|---|
| CodeBERT | 53.62% ASR | **27.79% ASR** | 35.78% ASR |
| GraphCodeBERT | 76.95% ASR | **7.96% ASR** | 61.47% ASR |

Adversarial fine-tuning raised robust accuracy by 87.59% / 92.32% respectively — i.e. hardening works but is an arms race.
- URL: https://arxiv.org/pdf/2201.08698 (2022)

**GraphCodeAttack (arXiv 2308.11161)** **[E]** — mines discriminative AST patterns and inserts them as dead code:
- **+30% ASR** over CARROT, **+33%** over ALERT on average.
- CodeBERT clone detection: **ASR 0.40** vs ALERT 0.27 vs CARROT 0.10 (~42% better than ALERT, 4× CARROT).
- Authorship attribution on GraphCodeBERT: **ASR 0.841** vs 0.598 / 0.615.
- Honest caveat from the authors: "The success rates on GraphCodeBERT [clone detection] are low."
- URL: https://arxiv.org/html/2308.11161 (2023, v2 2024)

**Extensive study (arXiv 2311.07553)** **[E/R]** — five SOTA attacks × three PTMCs (CodeBERT, CodeGPT, PLBART) × three tasks. Findings: (1) PTMCs "can be easily attacked under all three tasks," less robust on generation than understanding; (2) **effectiveness/efficiency trade-off** — highest-ASR attack queries the model the most times; (3) adversarial-example quality depends heavily on identifier-substitution strategy (context-aware > cosine-similarity > random).
- URL: https://doi.org/10.48550/arxiv.2311.07553 (2023)

**Semantic Robustness of Models of Source Code (Ramakrishnan et al., arXiv 2002.03043)** **[E]** — code2seq's prediction changes merely by **adding logging print statements**. Adversarial training with k=1 improves robustness against a k=5 adversary.
- URL: https://arxiv.org/pdf/2002.03043 (2020)

### A.1 design implications

- **[A]** Any novelty gate implemented as "similarity to existing submissions < threshold" is an *objective function handed to the attacker*. It is differentiable-in-practice: the attacker has your detector, can query it, and dead-code insertion alone moves MOSS by 50+ points.
- **[A]** If you must use similarity, use it **only as a fraud tripwire on the high-similarity side** (catching lazy copies), never as a *reward* on the low-similarity side. Rewarding low similarity pays for obfuscation.
- **[A]** Prefer **JPlag-class normalized token/AST matching** over MOSS/Dolos/embeddings, per the 94.2 pp vs 3.0 pp separation. But note this is a 2024 result in an active arms race.
- **[A]** The most robust "did you copy" signal in your setting is not textual at all: it is **behavioural** — training-curve fingerprints, per-token loss profiles, activation statistics, learned-weight spectra. These are far more expensive to fake than source text because faking them requires actually changing the model's computation.

---

## A.2 Novelty search in evolutionary computation

### The founding result

**Lehman & Stanley, "Abandoning Objectives: Evolution Through the Search for Novelty Alone," Evolutionary Computation 19(2), 2011** **[E]**
- Core claim: objective functions are **deceptive** — they "may actively misdirect search toward dead ends." Novelty search *ignores* the objective and rewards behavioural novelty relative to an archive.
- In **maze navigation** and **3-D biped walking**, novelty search **significantly outperforms** objective-based search.
- Key mechanism: "because many points in the search space collapse to a single behavior, the search for novelty is often feasible"; and "because there are only so many simple behaviors, the search for novelty leads to increasing complexity."
- **Dimensionality robustness [E]:** sampling navigator position k times gives a 2k-dim behaviour characterization. Even at **200 samples (400 dimensions)**, "the performance of novelty search is largely unaffected." Each point = average of 40 runs.
- URL: https://doi.org/10.1162/evco_a_00025 (2011); dissertation https://joellehman.com/lehman-dissertation.pdf

### The documented failure mode — this is the critique you asked for

**Cuccu & Gomez, "When Novelty Is Not Enough," EvoStar 2011** **[E]** — the direct rebuttal:
- Verbatim: *"we show that **novelty search alone does not scale to large search spaces**, but, when combined with fitness-based selection, it can be a useful diversity sustaining mechanism."*
- Domain: a **deceptive Tartarus (block-packer)** problem — deliberately chosen because its solution space is much larger than the maze.
- Result: "selecting for novelty alone **does not offer an advantage over fitness-based selection**"; novelty "encourages behavioral diversity, but does not necessarily lead to high average fitness."
- **The diagnosis, which is exactly your "novelty for its own sake" concern:** novelty search worked in the maze *only because* "the novelty of an individual is correlated with an intuitive measure of utility: its final position in the maze." *"The problem arises as soon as novelty and utility are decoupled."*
- **The adversarial-design statement [E]:** *"one can always design a fitness function such that the solutions discovered by novelty alone perform arbitrarily badly."*
- URL: https://people.idsia.ch/~tino/papers/cuccu.evostar11.pdf (2011)

**Preliminary Analysis of Simple Novelty Search (Evolutionary Computation 32(3), 2024)** **[E/R]**
- Formal analysis of archive dynamics. Identifies the **archive-saturation scaling pathology**: "large archive sizes make the algorithm increasingly unlikely to select points close enough to the surface of the archive that mutation has a reasonable probability to generate a point that will meet the sparseness criterion." I.e. exploration stalls not because the space is full but because *selection dilutes* as the archive grows.
- Nuance worth keeping: unbounded space is not per se fatal; there exist parameterizations (σ=0.1, ρ_min=0.2, d=5) where the algorithm converges in every sense.
- URL: https://direct.mit.edu/evco/article/32/3/249/116787/ (2024)

**Novelty Search with Local Competition (Lehman & Stanley, GECCO 2011)** **[R]** — documented limitations: archive grows unboundedly (compute overhead in novelty scoring); performance sensitive to novelty threshold and k; behaviour characterization + distance metric are problem-specific and hard to design.

### Quality-Diversity as the portfolio mechanism

**MAP-Elites (Mouret & Clune, arXiv 1504.04909, 2015)** **[E]**
- Discretizes a user-chosen feature space into cells; each cell keeps the single fittest individual ("elite") ever mapped there. "Illuminates" the space rather than returning one point.
- **Measured:** MAP-Elites scores significantly higher (**p < 1×10⁻⁷**) than **all three** controls — traditional EA, **novelty search with local competition**, and random sampling — on **all four** criteria: global performance, global reliability, precision, coverage. Domains: modular neural networks, simulated soft robots, real soft robots.
- Notable: because it explores more, it *also* finds a better single best solution than the objective-driven baseline.
- URL: https://doi.org/10.48550/arxiv.1504.04909 (2015)

**Quality Diversity: A New Frontier for EC (Pugh, Soros, Stanley — Frontiers in Robotics and AI, 2016)** **[R]** — QD framing; standard metrics are **coverage** (fraction of cells filled), **QD-score** (sum of elite fitnesses), **global best**. Key structural distinction: in MAP-Elites niches are *explicitly defined* rather than passively emergent from local competition.
- URL: https://doi.org/10.3389/frobt.2016.00040 (2016)

**Cully et al., "Robots that can adapt like animals," Nature 521:503, 2015** **[R]** — MAP-Elites archive used as a **repertoire of skills** that a damaged hexapod draws on to recover. This is the clearest existence proof that a QD archive functions as a *portfolio asset*, not just a diversity bookkeeping device.

**Dominated Novelty Search (arXiv 2502.00593, 2025)** **[E/R]** — modern critique of both MAP-Elites and threshold-based local competition: MAP-Elites "requires a pre-defined and fixed grid" and "cannot be applied" when descriptors are **learned/unbounded/changing**; Threshold-Elites needs a complex container-size control mechanism. DNS instead rewards solutions that "either outperform their neighbors or find unique behaviors compared to better-performing solutions" — an emergent competition structure needing no bounds or distance thresholds.
- URL: https://arxiv.org/html/2502.00593v1 (2025)

### A.2 design implications

- **[A]** The Cuccu & Gomez result is the theoretical core of your problem statement. Novelty is only safe as a *reward* where novelty and utility are **coupled by construction**. In your setting they are decoupled by default: an architecture can be wildly novel in code-space and useless in loss-space.
- **[A]** MAP-Elites gives you the shape of a defensible mechanism: define a **small number of interpretable architecture-descriptor axes** (e.g. attention mechanism family, parameter-sharing scheme, sequence-mixing operator class, depth/width ratio, positional-encoding family), bin them, and pay the **elite of each cell**. Novelty is then never rewarded directly — it is only rewarded *as the price of admission to an underpopulated cell where you must still win on loss*. A leader-copy lands in an occupied cell and must beat the incumbent elite outright.
- **[A]** Crucially, MAP-Elites cell descriptors are chosen by *you*, the operator. That converts "novelty" from an unbounded gameable metric into a **bounded, enumerable, operator-controlled taxonomy**. The attacker cannot invent new cells to farm; they can only fill cells you have declared interesting. This is the single most important structural transfer from this literature.
- **[A]** But heed DNS: fixed grids age badly. Expect to revise the descriptor taxonomy each season, and expect miners to cluster at the descriptor boundaries you publish.

---

## A.3 Numerai: the full originality → MMC → TC → MMC history

This is the strongest real-money precedent for "pay for marginal contribution, not raw score." Your recollection is correct in substance. The precise sequence:

### Phase 1 — explicit originality/uniqueness marketing (2019)

Numerai blog, **"A New Data Science Competition Where Being Different Pays"** **[R]**
- Verbatim: *"participants will be paid not only for performance — they will also be paid for **originality and uniqueness**."*
- Rationale stated: "Users are already using modeling approaches on our data that we have no idea how to recreate… These types of users are extremely valuable to the meta model, but are **not being proportionally rewarded yet**."
- Mechanism introduced to deliver it: **Meta Model Contribution (MMC)** — "residualize (or subtract out) the meta model predictions from the submission. Whatever is left over after being residualized to the meta model is what we score versus the true stock market results."
- URL: https://blog.numer.ai/a-new-data-science-competition-where-being-different-pays/

### Phase 2 — MMC1 → MMC2 (2019–2020)

**MMC2 Announcement** (forum) **[R]**
- **MMC1** = classic leave-one-out: "takes the stake-weighted metamodel, and then tries removing each user from it and seeing how much it hurts the metamodel." **This is literally a leave-one-out marginal-contribution estimator.**
- **MMC2** = "the **residual MMC** method" — uniform-transform the SWMM, neutralize each model against it, take covariance with target, divide by 0.29² to rescale into correlation space.
- Why they switched: MMC1's LOO framing was replaced by residualization for stability and interpretability; the divide-by-0.29² makes MMC comparable in magnitude to main-tournament CORR.
- URL: https://forum.numer.ai/t/mmc2-announcement/93

### Phase 3 — the exploited bonus and its removal (2020) — the key cautionary tale

**"Leaderboard Bonus Exploit Uncovered"** (forum) **[E]**
- The leaderboard bonus (a *reputation/consistency* bonus, distinct from MMC) was **provably exploited**.
- Specific numbers: accounts `Madmax`, `Madmin`, `The_Guy` "began with **40 NMR each in October 2019**. To date, the models have a combined **222 NMR**. 17 for Madmax, 38 for The_Guy, and **167 for Madmin**." Characterized as "a clear exploitation of the payout system for **over 100 NMR and 85% returns in less than 6 months**."
- Mechanism: an **asymmetry** the attacker could straddle across multiple accounts to guarantee high payout with minimal risk (a hedged-multi-account attack).
- The design philosophy statement, which is the most quotable line in this whole review: *"**We don't want users to have to think about potential punishments. They should be able to play the game given the incentives we've given them.** That's why leaderboard bonus is going away. **We see it as our responsibility to make not-exploitable payout systems.**"*
- Punishment was deliberately mild: ineligible for leaderboard bonus for its remaining 100 days, **no ban, keeps all NMR earned**.
- URL: https://forum.numer.ai/t/leaderboard-bonus-exploit-uncovered/200 (2020)

**"MMC - Payout Details and Analysis"** (forum) **[E/R]**
- *"A primary reason [for removal] is simply that it is **susceptible to some certain attacks in which a bad-actor could game the tournament and guarantee high payouts for himself with minimal risk**."*
- *"MMC better allocates the payout pool to the users who are contributing to the metamodel the most, **without needing a leaderboard bonus add-on**."*
- *"As MMC and the primary tournament have **attack resistance as a core design goal**, we can safely increase this limit without the same fear of exploitation."*
- Secondary reason **[E]**: the bonus had hit its **250 NMR/day cap**, so the effective "3% a day" was already 2.7% and shrinking as the tournament grew — the bonus **did not scale with TVL**.
- Dates **[E]**: MMC payouts began **May 31**; **Leaderboard Bonus discontinued Wednesday, September 9, 2020**.
- URL: https://forum.numer.ai/t/mmc-payout-details-and-analysis/220 (2020)

### Phase 4 — True Contribution (TC), 2022–2023

**"True Contribution Details"** (forum) + docs **[E/R]**
- Motivation: "users are evaluated only at the signal level," but "Numerai's portfolio is created by running our custom optimizer on the Meta Model signal… This can create divergence between what appears to be a good model at the signal level and a model that is truly helping the fund create better portfolios."
- Mechanism **[E]**: (1) stake-weighted meta model (SWMM); (2) SWMM + **hundreds of risk constraints** (market, country, sector neutralization) → optimizer → hypothetical portfolio; (3) **compute the gradient of the optimized portfolio return with respect to the stake**; TC = magnitude of that gradient. Implemented with **cvxpylayers** (cvxpy convex optimization as a differentiable PyTorch layer).
- Explicit analogy in their own docs: *"This is akin to Neural Network architecture and the idea of gradient descent."*
- Rollout **[E]**: from **April 9** [2022], staking options became (0x or 1x CORR) + (0x, 1x, or 2x TC); **MMC staking discontinued that date**; TC staking **opt-in only**, no automatic conversion of MMC stakes.
- URL: https://forum.numer.ai/t/true-contribution-details/5128 ; https://docs.numer.ai/numerai-tournament/scoring/true-contribution-tc

### Phase 5 — TC retired, back to MMC (Jan 2, 2024) — read the stated reasons carefully

**"Changing Scoring & Payouts Again To MMC Only"** (forum) **[E/R]** — the postmortem on TC:
- *"TC has some weaknesses in that it is **blackbox** and also **tied to certain optimizer settings**. The optimizer settings for Numerai One and Supreme are different from each other and **change from time to time**. To have TC stay 'True' the whole time would require **constant alterations to it** — even **the size of the funds influences TC**. So TC is **challenging to maintain without becoming even more blackbox and mysterious**."*
- Why MMC came back: a new target (**Teager**) "does almost all the important transformations that the optimizer does **within the target**, making MMC on this target a great measurement of contribution." Also, **Numerai now publishes the Meta Model signal**, so *"MMC is not a blackbox any more and can be computed locally."*
- Why they made it **mandatory** rather than opt-in **[E/R]**: *"Many users would simply stake CORR with a large stake and not stake MMC or TC at all. The problem is many large stakers this year have had **persistently negative TC & far worse TC than benchmark models**… if they weren't staking TC this year, they would hardly burn at all if their CORR was okay."* Conclusion: users persistently hurting the Meta Model "should earn a **strongly and persistently negative return** on their stake."
- URL: https://forum.numer.ai/t/changing-scoring-payouts-again-to-mmc-only/6794 (late 2023)

**"MMC staking starts Jan 2, 2024"** **[E]**
- From rounds on/after **January 2, 2024**: fixed multipliers **0.5 × CORR + 2 × MMC**. The 2024 Grandmasters season decided on CORR and MMC.
- Forum thread also shows a later revision discussed as **0.5 CORR + 3× MMC**, and records that **MMC displayed on the website was corrected from an erroneous version** mid-week — i.e. even the operator shipped a scoring bug in the metric.
- Substantive user objection worth recording **[R]**: *"A model's CORR is not how much it contributed to the MM… However MM doesn't come for free, so you should also reward the models that created the MM in the first place and the risk the users take in staking those models. Currently this part is rewarded 0.5×CORR, which is **too low**."* And: *"Using the stake, and hence payout, as a mechanism to optimize the MM is **in conflict with how the users see the stake and the payout**."*
- URL: https://forum.numer.ai/t/mmc-staking-starts-jan-2-2024/6827

### Phase 6 — current state (2026) **[E]**

**Numerai blog, "Payout Updates for the 2026 Season"** — changes effective **January 1, 2026**:

| Tournament | Change |
|---|---|
| **Numerai** | New scoring/payout/leaderboard target **Ender20** (first payout-target change since 2023). **CORR multiplier 0.5 → 0.75**; **MMC multiplier 2.0 → 2.25** |
| **Signals** | Payout clip **1.7% → 3.5%**; MPC switched from **L1 → L2 norm** |
| **Crypto** | CORR multiplier **0 → 0.05**; MMC multiplier **1 → 0.5** |

- December [2025] payouts: **$254,137** worth of NMR.
- URL: https://blog.numer.ai/numerai-monthly-numercon-speakers-new-dataset-target-2026-payout-updates/

Current docs **[E]**: payout scores are **CORR and MMC only**. Informational-but-unpaid: FNC, CWMM, BMC. Legacy continuous-staking formula: `score = corr20*corr_mult + mmc20*mmc_mult; payout = stake * clip(payout_factor * score, -0.05, 0.05)`, with `payout_factor = min(1, stake_threshold / total_at_risk)` and stake thresholds Numerai **72000**, Signals **36000**, Crypto **10000**.
- URL: https://docs.numer.ai/numerai-tournament/scoring ; https://docs.numer.ai/numerai-tournament/staking

**MMC definition as currently documented [E]:** "the covariance of a model with the target, **after its predictions have been neutralized to the Meta Model**." Procedure: normalize submission → normalize Meta Model → neutralize submission to Meta Model → covariance of neutralized submission with target. BMC is the same against stake-weighted **Benchmark** Models.
- URL: https://docs.numer.ai/numerai-tournament/scoring/meta-model-contribution-mmc

### A.3 — the lessons, stated plainly

1. **[E]** Numerai **never paid for raw originality/dissimilarity**. They marketed "being different pays" but *implemented it as marginal contribution to an ensemble*. Difference only paid **when it was orthogonal AND predictive**. That distinction is the whole ballgame.
2. **[E]** The thing that got exploited was **not** the contribution metric — it was the **leaderboard/reputation bonus**, a score-shaped side-payment. Contribution metrics survived 7 years; the bonus lasted months.
3. **[E]** TC failed for **operational**, not conceptual, reasons: blackbox, coupled to a changing optimizer, not locally computable by participants, and needing constant recalibration. **Legibility and local verifiability are load-bearing incentive properties**, not nice-to-haves.
4. **[E]** They deliberately made the contribution metric **mandatory** because opt-in let large stakers collect on the easy metric while damaging the portfolio.
5. **[R]** Live user objection you should pre-empt: pure marginal-contribution payment **under-rewards the participants who built the baseline in the first place**. Numerai's answer is the residual CORR term (now 0.75×). A pure-marginal design has a **free-rider-on-the-frontier problem**: the miner who found the good architecture earns little once everyone converges to it.
6. **[A]** The direct transfer to your setting: define a **stake-weighted or emission-weighted "meta-architecture" reference** (e.g. the current best-known architecture, or an ensemble/frontier of accepted ones) and pay each submission on the **residual improvement after neutralizing against that reference**. A leader-copy has ~zero residual by construction. **No plagiarism detector required** — and that is precisely why it survives adversarial pressure where A.1 methods do not.

---

## A.4 Marginal-contribution / Shapley-value reward: mechanisms and cost

### The cost wall — the numbers that matter for a 6-GPU-hour-per-eval budget

**Data Shapley (Ghorbani & Zou, ICML 2019)** **[E]**
- Exact Shapley requires evaluating **2^(N−1)** coalitions; each evaluation **retrains a model**.
- **TMC-Shapley** (Truncated Monte Carlo): sample random permutations; truncate a permutation once V(S) is "within the performance tolerance of V(D)," setting remaining marginals to zero. Justification: as |S| grows, the marginal of one more point shrinks; and V(S) is itself only known to within test-set noise (quantifiable via bootstrap).
- Authors' own limitation statement: "in settings where the number of data points is large or the predictive model requires high computational power (**e.g. deep neural networks**), applying TMC-Shapley can be **quite expensive**."
- URL: https://proceedings.mlr.press/v97/ghorbani19c/ghorbani19c.pdf (2019)

**Sample-complexity ladder [E]** (arXiv 2302.11431 and Jia et al. 2019):

| Estimator | Utility evaluations |
|---|---|
| Exact Shapley | 2^(N−1) |
| Permutation sampling (Castro et al. 2009) | Ω(N² log N) |
| Group-testing SV estimator (Jia et al. 2019) | Ω̃(N (log N)²), or Ω̃(N/ε²) |
| **Leave-one-out (LOO)** | **N retrainings** |
| KNN-Shapley proxy | O(N log N) |

- URL: https://ar5iv.labs.arxiv.org/html/2302.11431 (2023)

**Empirical scalability ceiling [E]** (openreview SygBIxSFDS): "the approaches in Ghorbani & Zou (2019) can manage data size **up to one thousand** for simple models such as logistic regression and shallow neural networks, while **failing** to estimate the Shapley value for larger data sizes and deep nets in a reasonable amount of time." Also: influence functions (Koh & Liang 2017) need O(N·p² + p³) for the Hessian inverse, or O(Np) approximated — "still expensive for large networks."
- URL: https://openreview.net/pdf?id=SygBIxSFDS

### Real deployments

| System | Mechanism | Status |
|---|---|---|
| **Numerai MMC1** | Literal leave-one-out on the stake-weighted meta model **[E]** | deployed, then replaced by residual form |
| **Numerai MMC2 / current MMC** | Residualization/neutralization against Meta Model — **an O(1) closed-form proxy for LOO marginal contribution** **[E]** | **live, primary payout metric, 2024–2026** |
| **Numerai TC** | Gradient of optimized portfolio return w.r.t. stake, via cvxpylayers **[E]** | deployed Apr 2022, **retired Jan 2, 2024** |
| **Numerai BMC** | Same, neutralized against stake-weighted benchmark models **[E]** | live but **informational, not paid** |
| Federated-learning contribution schemes | Shapley/LOO variants for client valuation **[R]** | research-stage; I did not find a citable large production deployment in this pass |
| Ocean Protocol / data markets | Data valuation **[A]** | not verified in this pass — treat as unresearched |

### A.4 design implications

- **[A]** **The single most important practical finding in this section:** Numerai's production answer to "Shapley is too expensive" was **not** a cleverer Shapley approximation. It was to replace the coalition game entirely with a **one-shot orthogonalization** — normalize, neutralize against the reference, take covariance with the target. That is **O(1) per submission**, exactly computable, and locally verifiable by the participant. For a 6-GPU-hour eval budget, this is the only tractable family.
- **[A]** Concretely for you: your "target" is validation loss/downstream metric; your "meta model" is the reference architecture or frontier. Compute the submission's **improvement residual after regressing out the reference's per-example performance profile**. Copies of the leader correlate ~1.0 with the reference profile → residual ≈ 0 → payout ≈ 0. Genuinely different architectures that win on *different examples* have a large residual even at equal aggregate loss. **This directly rewards "novelty that pays off" and ignores novelty that doesn't.**
- **[A]** Cost check: if you ever want true LOO over an ensemble of k accepted architectures, that is **k retrainings at 6 GPU-hours each**. At k=20 that is 120 GPU-hours per round. Feasible occasionally, not per-submission. Reserve LOO for a periodic (e.g. seasonal) recalibration of the reference, and use the O(1) residual for per-submission payout.
- **[E→A]** Heed the TMC truncation insight: **V(S) is only meaningful to within evaluation noise**. With 6 GPU-hours and presumably a single seed, your per-submission metric has real variance. Any contribution metric should be compared against a **bootstrap/seed-noise floor**, or you will pay for noise. Numerai's `clip(±0.05)` and Kaggle's public/private split are both structural acknowledgements of the same problem.

---

# B. Funding / rewarding open research — precedents

## B.1 Prize contests

### DARPA Grand Challenge (2004, 2005) — the "set the bar too high and pay nothing" precedent **[E]**

| | 2004 | 2005 |
|---|---|---|
| Date | March 13, 2004 | October 8, 2005 |
| Course | 142 miles, Barstow CA → Primm NV, 10 h limit | 132 miles, southern Nevada |
| Registered / raced | 107 registered, **15 raced** | **195 registered**, 23 raced |
| Finishers | **0** | **5** |
| Best result | **7.5 miles** (~5% of course) | Stanley, 6 h 53 min 58 s |
| Prize | **$1M unclaimed** | **$2M** to Stanford Racing Team |
| DARPA spend | ~**$13M** estimated (IEEE Spectrum) | — |

- Sandstorm (2004 favourite) covered **12 km** before getting stuck at Dagget Ridge, revving until "much of the rubber on its front tires had burned off and both half-shafts broke."
- DARPA announced the second challenge **one day after** the first ended, 18 months out.
- **[R]** DARPA's own retrospective frames the zero-payout year as successful: "a most important first step" that "helped to create a mindset and research community."
- URLs: https://www.darpa.mil/about/innovation-timeline/grand-challenge ; https://www.darpa.mil/news/2014/grand-challenge-ten-years-later ; https://spectrum.ieee.org/dusted-no-winners-in-darpas-1-million-robotic-race-across-the-mojavedesert (2004) ; https://robots.stanford.edu/papers/thrun.stanley05.pdf
- **[A]** Lesson: a **hard absolute threshold** with a rollover produced a 20× improvement in 18 months and cost nothing in year one. But it required the sponsor to tolerate a **zero-payout round** — impossible for a token-emission system with continuous emissions. Your analogue must be a *relative* frontier, not an absolute bar.

### Netflix Prize (2006–2009) — the ensemble/deployment trap **[E]**

- Target: **10% RMSE improvement** over Cinematch; 0.9525 → **0.8572 or less**.
- **Progress Prize (year 1):** Korbell team, **8.43%** improvement, **>2,000 hours of work**, an ensemble of **107 algorithms**. They handed over source code.
- Netflix took the **two best components only**: Matrix Factorization ("SVD") alone → **0.8914 RMSE**; RBM alone → **0.8990**; **linear blend of the two → 0.88**. Both went into production and (as of the 2012 post) "are still used as part of our recommendation engine."
- Engineering reality **[E]**: components "were built to handle **100 million ratings, instead of the more than 5 billion that we have**," and "were not built to adapt as members added more ratings."
- **Grand Prize (Sept 2009):** BellKor's Pragmatic Chaos, **10.06%**, blending 100+ models. Netflix's verbatim verdict: *"We evaluated some of the new methods offline but the **additional accuracy gains that we measured did not seem to justify the engineering effort** needed to bring them into a production environment. Also, our focus on improving Netflix personalization **had shifted** to the next level by then."*
- Also documented **[E]**: the metric itself was a proxy — "we had to come up with a **proxy question that was easier to evaluate and quantify**: the root mean squared error of the predicted rating."
- Also **[E]**: in Nov 2006 "Simon Funk" **publicly posted** his matrix-factorization approach, which "seeded the entire competition with latent factor methods and demonstrated that the 10% target was achievable."
- URLs: https://netflixtechblog.com/netflix-recommendations-beyond-the-5-stars-part-1-55838468f429 (2012) ; https://amatria.in/pubs/recsys12-tutorial.pdf ; https://queirozf.com/entries/the-netflix-prize-changing-requirements-and-cost-effectiveness
- **[A]** Three transferable lessons, all directly on point for you:
  1. **A single scalar leaderboard metric drives competitors into blend-space.** Once the frontier is reached by blending, marginal wins come from *combinatorial stacking of near-duplicates*, not from new ideas — the structural cousin of your copy-the-leader attack. Blending was rational precisely because the metric rewarded it.
  2. **The valuable artifact was the year-1 8.43% solution's two components, not the 10.06% winner.** Paying maximally for the last 1.6% bought nothing deployable. **Diminishing-returns-to-metric is real and steep.**
  3. **Simon Funk's public post was arguably the highest-leverage contribution of the entire competition and won no prize.** Any scoring system that pays only terminal performance systematically fails to pay the person who unlocked the paradigm. This is the strongest argument in this whole review for a **lineage/credit term** in your design.

### Kaggle — two-stage structure as anti-gaming machinery **[E]**

- **Two-stage competitions**: stage 1 = train + submit to temporary leaderboard; stage 2 = same models predict on a **previously unavailable test set**.
- Stated purpose **[E]**: *"to prevent **hand labeling and leaderboard probing** of the test data."*
- Enforcement **[E]**: must upload source code with parameters by the stage-1 deadline; may submit an **encrypted archive with a checksum**, key revealed only on winning; **checksum mismatch → disqualification**.
- Constraint on stage 2 **[E]**: "You are allowed to **re-train** your model (including the stage one data), but **your code should not change**. You should **not** be doing any hyperparameter tuning in the second stage. Parameter tuning is permitted **as long as it is fully automated**."
- Medals/points use the **stage-1 participant count**, since stage-2 participation drops.
- Public/private split **[E]**: public leaderboard on a sample, private on the remainder, and Kaggle explicitly warns "a high public score doesn't guarantee a high private score. **Avoid 'chasing' the public leaderboard**." Leakage remedies: relaunch the competition or generate a new test set.
- Empirical study of Kaggle contest design (Marshall, "Dynamic Tournament Design: Evidence from Kaggle") **[E]**: sample of contests averaging **$30,489** prizes, ≥1,000 submissions, average **894 teams** per contest; e.g. Heritage Health Prize split test data **30% public / 70% private**; includes a **randomized controlled trial run on Kaggle**.
- URLs: https://www.kaggle.com/two-stage-frequently-asked-questions ; https://www.kaggle.com/docs/competitions ; https://g-marshall.github.io/kaggle1.pdf
- **[A]** The code-escrow-with-checksum pattern is directly implementable for you and cheap: require the architecture + training code hash **committed before** the eval set/seed is revealed. It converts "tweak until the eval likes it" from a strategy into a detectable protocol violation.

### XPRIZE / other bounties **[A — under-researched in this pass]**

I did not obtain primary XPRIZE outcome data (Ansari XPRIZE $10M, SpaceShipOne 2004) in this session beyond general knowledge, so I am **not** presenting figures. Same for AI-safety bounties, Erdős/Clay problems, AI Grant, and Focused Research Organizations. **Treat these as open items.** Flagging rather than fabricating.

---

## B.2 Prediction markets for replication — a cheap forecast layer that measurably works

### Dreber, Pfeiffer, Almenberg, Isaksson, Wilson, Chen, Nosek, Johannesson, PNAS 2015 **[E]**

Setup: real-money prediction markets on whether **44** psychology studies in the Reproducibility Project: Psychology would replicate; **41** replications completed in time.

| Quantity | Value |
|---|---|
| Total transactions | **2,496** |
| Transactions per market | 28–108 (**mean 56.7**) |
| Active traders per market | 18–40 (**mean 26.7**) |
| Mean final market price | **55%** (range 13–88%) |
| Actual replication rate | **16/41 = 39%** replicated; 25/41 = 61% did not |
| **Market binary accuracy** | **71% (29/41)**, p = 0.012 vs 50% |
| Expected accuracy if prices were calibrated | **69%** — very close to observed 71% |
| First market set vs second | **87% (20/23)** vs **50% (9/18)**, p = 0.016 (Fisher's exact) |
| vs survey of the same forecasters | markets **outperformed** the survey |

- No detectable long/short bias; trading was not dominated by a small subset of traders.
- URL: https://doi.org/10.1073/pnas.1516179112 (2015); PDF https://www.stat.berkeley.edu/~aldous/157/Papers/prediction_replication.pdf

### Pooled analysis across four forecasting projects (PLOS ONE 2021) **[E]**

| | Accuracy | Correlation with outcome |
|---|---|---|
| **Prediction markets** | **73% (75/103)** | **0.581** (p < .001) |
| Surveys | **66% (68/103)** | **0.564** (p < .001) |

- Difference between markets and surveys **not statistically significant** (χ²(1) = 1.12, p = 0.29).
- Per-project: RPP **71% vs 58%**; ML2 **75% vs 67%**.
- Dataset released as R package `pooledmaRket`.
- URL: https://doi.org/10.1371/journal.pone.0248780 (2021)

### DARPA SCORE (2019 → results reported 2026) **[E]** — the largest test, and the ML result is the important one

- Scope: claims sampled from **3,900 papers**, published **2009–2018**, in **62 journals** across criminology, economics, education, finance, health, management, marketing, organizational behavior, psychology, political science, public administration, sociology. **865 researchers** involved. Published as a **Nature collection** (3 papers + 6 preprints); data/code on OSF.
- **Human expert methods: 76–78% success** at predicting replication (repliCATS structured elicitation, Replication Markets).
- **Machine learning methods: the three tested — Synthetic Markets, MACROSCORE, and A+ — were "not consistently effective" at predicting whether claims would replicate.** This is a negative result and it matters.
- Reproducibility (same data, same analysis): **~half** of claims precisely reproduced, **~three-quarters** at least approximately reproduced. Higher in political science and economics, in more recent publications, and in journals requiring data sharing.
- Replicability (new data): **~half** replicated with significance and same pattern, but effect sizes shrank — **median >50% reduction in effect size**, **>80% reduction in explained variance**.
- repliCATS pilot (SIPS 2019, independent of SCORE): 5 groups × 5 participants on 25 claims → **84% classification accuracy, AUC 0.94**.
- URLs: https://www.cos.io/score ; https://www.darpa.mil/research/programs/systematizing-confidence-in-open-research-and-evidence ; https://www.cos.io/about/news/large-scale-collaboration-releases-new-findings-on-research-credibility ; https://doi.org/10.31222/osf.io/2pczv

### B.2 design implications

- **[E→A]** A forecast/market layer buys you **~71–78% accuracy at predicting whether an expensive evaluation will confirm a claim**, at near-zero compute cost. That is a genuinely useful **triage** signal for allocating 6-GPU-hour slots: let stakers/forecasters price "will this architecture beat the frontier?" and spend GPU on the highest-uncertainty-times-impact submissions.
- **[E]** But calibrate expectations honestly: **71–78% is a triage signal, not an adjudicator.** Note also the market accuracy dropped to **50%** in the second RPP market set — market performance is **not stable across batches**. Never let the market replace the eval; use it only to order the queue.
- **[E]** SCORE's negative ML result is a warning directly applicable to you: automated credibility scoring **underperformed humans** on 3,900 papers with DARPA-scale funding. Do not assume an automated "is this architecture promising?" classifier will work; the best-funded attempt to date did not consistently beat expert forecasts.
- **[A]** Markets also give you a **skin-in-the-game anti-copy mechanism**: if forecasters can short a submission, copy-the-leader submissions get priced to zero expected marginal gain before you spend a single GPU-hour on them.

---

## B.3 Concentrated vs distributed research funding — the core evidence

**This is the section most directly analogous to "one long GPU run vs many short runs," and the evidence points one way.**

### Fortin & Currie, "Big Science vs. Little Science," PLOS ONE 8(6):e65263, 2013 **[E]**

- Data: individual university researchers in **three disciplines** funded by NSERC (Canada). Four impact indices over a **four-year** window: publications, citations, most-cited article, count of highly-cited articles.
- Formal framing: model impact **I = a·F^b**. If **b < 1**, impact is *decelerating* in funding, impact-per-dollar **falls** with grant size, and "**many small grants should yield greater total impact than a few big grants**."
- Findings, verbatim:
  - "Impact is **positively, but only weakly**, related to funding."
  - "Impact was generally a **decelerating function of funding**. **Impact per dollar was therefore lower for large grant-holders.** This is **inconsistent with the hypothesis that larger grants lead to larger discoveries**."
  - "the impact of researchers who **received increases in funding did not predictably increase**."
  - Researchers who also got CIHR funding on top of NSERC "were **not more productive**."
- Conclusion: "scientific impact (as reflected by publications) is **only weakly limited by funding**. We suggest that funding strategies that target **diversity, rather than 'excellence'**, are likely to prove to be more productive."
- URL: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0065263 (2013)

### Mongeon, Brodeur, Beaudry, Larivière, "Concentration of research funding leads to decreasing marginal returns," Research Evaluation, 2016 **[E]**

- Data: **12,720 researchers** in Québec, funding **1998–2012**, output/impact **2000–2013**.
- Findings: "both in terms of the quantity of papers produced and of their scientific impact, the concentration of research funding in the hands of a so-called '**elite**' of researchers generally produces **diminishing marginal returns**." Further: "**the most funded researchers do not stand out** in terms of output and scientific impact."
- Policy statement: "this concentration… brings **no clear collective advantages**… **such funding policies are not efficient**… one may wonder what could justify awarding millions of dollars to a few researchers while many others receive nothing." Supports "a higher number of **smaller grants** to a higher number of researchers."
- URL: https://doi.org/10.1093/reseval/rvw007 (2016)

### Aagaard, Kladakis, Nielsen, "Concentration or dispersal of research funding?", Quantitative Science Studies 1(1):117, 2020 **[E]** — the systematic review

- Method: screened **3,567 articles**, examined **92** in depth, juxtaposed **20 years** of empirical research on grant size vs performance.
- Headline: "the review demonstrates a **strong inclination toward arguments in favor of increased dispersal**."
- Mechanisms found: concentration "may in fact lead to **diseconomies of scale**"; "the **majority of extant empirical research finds little or no convincing evidence** to justify funding policies aimed at concentrating resources to achieve economic efficiency"; concentration "on average, leads to **decreasing marginal returns**… **above a certain threshold**."
- **Quantified optimum [E]:** "numerous empirical studies suggest that research productivity can be increased by spreading out funding on many small and medium-sized research teams, averaging from around **five to eight group members**."
- URL: https://direct.mit.edu/qss/article/1/1/117/15557/ (2020)

### B.3 design implications

- **[E]** Three independent bodies of evidence — two large single-country bibliometric studies plus a 92-paper systematic review — converge: **impact is decelerating in resource concentration, with a threshold above which marginal returns fall.**
- **[A]** Direct translation: **many 6-GPU-hour runs beat few 60-GPU-hour runs, in expectation, for discovery.** Your instinct to use a short standardized budget per submission is supported by the metascience literature, not just by cost.
- **[A]** Important caveat I want to flag rather than paper over: this literature measures *publications and citations*, not *frontier capability gains*, and the deep-learning scaling literature points the opposite way for **exploitation** (bigger runs → better models, reliably). The reconciliation: **dispersal wins for search/discovery; concentration wins for exploitation of an already-identified direction.** Your competition is a discovery mechanism, so dispersal is right — but a *separate, concentrated* "scale up the winner" stage is complementary, not contradictory. Note this is my inference **[A]**, not a measured finding.
- **[A]** The "five to eight group members" optimum has no clean analogue, but it does argue against a winner-take-all emission curve: pay a **band** of top submissions, not a single leader.

---

## B.4 Staged / tiered evaluation, peer-review reliability, and lotteries

### Can peer review distinguish top proposals? — largely no **[E]**

**Fang, Bowen & Casadevall, eLife 2016, "NIH peer review percentile scores are poorly predictive of grant productivity"** **[E]** — the strongest numbers:

| Metric | Value |
|---|---|
| Sample | **102,740** funded NIH grants with percentile scores ≤20 |
| Slope, publications per percentile increment | **−0.132 ± 0.005** |
| Slope, citations per percentile increment | **−9.6 ± 0.337** |
| **r²** | **0.0078** |
| Variance in productivity explained by percentile (random forest) | **~1%** |
| **ROC AUC for discriminating productivity** | **0.54** |
| Grants with percentile score of **zero** that produced **no citations** | **17% (334/1,987)** |

- Verbatim: "all of the effort currently spent in peer review has a **minimal impact in stratifying meritorious applications relative to what would be expected from a random ranking**."
- URL: https://elifesciences.org/articles/13323 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC4769156/ (2016)

**Danthi, Wu, Shi, Lauer, Circulation Research 2014 (NHLBI)** **[E]**
- **1,492** cardiovascular R01 grants, initial funding 2001–2008; **7,654 grant-years**, **$3,004 million** awarded, **16,793 publications** (2001–2012), **2,224,255 citations**.
- "**No clear association** between percentile ranking and outcomes… as percentile ranking decreased (a 'better score') we did **not** observe a corresponding monotonic increase in publications produced or citations received **per million dollars spent**." Persisted after adjusting for confounders, human vs non-human research, and high-volume study sections.
- Strongest predictor of citations-per-million-dollars was **average number of grants acknowledged per paper** (inverse-V shape, peaking at **3–4 grants**), not percentile.
- URL: https://doi.org/10.1161/circresaha.114.302656 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC3959724/

**Lauer et al., NIMH cohort** **[E]** — **1,755** de novo R01s funded ≥2 years, 2000–2009: **no association** between percentile ranking and subsequent productivity/citation impact, even after accounting for subject category, publication years, duration and amount of funding, and investigator-specific measures. Consistent with NIGMS (Berg 2013) and an NSF unit. **Prior investigator funding and prior productivity were moderately strong predictors** of citation impact.
- URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC5526589/

### Lottery + review hybrids — deployed, with outcome data **[E]**

**Taxonomy (Research Evaluation, rvae025, 2024)** **[E]** — six types, with real adopters:

| Type | Description | Adopters |
|---|---|---|
| 0 | Traditional peer review | most funders |
| 1 | Tie-breaking partial lottery | **Swiss National Science Foundation** |
| 2 | Partial lottery with bypass | **Volkswagen Foundation**, **Austrian Science Fund (FWF)**, **Novo Nordisk Foundation** |
| 3 | Partial lottery of fundable proposals | **Health Research Council of New Zealand**, Science for Technological Innovation (NZ), **New Frontiers in Research Fund (Canada)**, **The British Academy** |
| 4 | Full lottery | none |

- URL: https://doi.org/10.1093/reseval/rvae025 (2024)

**Volkswagen Foundation "Experiment!" (2013 → 2020/21)** **[E]** — the closest thing to an RCT on this question:
- **>5,000 applications for 183 approved grants.** Three-step review; a **supervised physical lottery** drew from proposals the jury rated "top" or "entry in lottery," creating **two cohorts from the same applicant pool**.
- **Result: "a comparison of the jury-selected and lottery-selected cohorts revealed very similar research outputs and outcomes, including publications, patents, and funding/career effects. The Foundation's assessment of outputs suggested no significant difference in quality between the two groups."**
- Interim findings also reported: randomization **increased diversity** of funded projects and **encouraged risk-taking proposals**; welcomed by most of the research community; relieved reviewer burden on "numerous high-quality proposals that were **rationally indistinguishable**."
- URLs: https://sfdora.org/2025/03/27/insights-on-partial-randomization-in-research-funding-learnings-from-the-volkswagen-foundation/ (2025-03-27) ; https://www.embo.org/blog/focal-randomisation-in-grant-allocation-a-new-wave-of-innovation-in-grant-funding/

**HRC New Zealand Explorer Grants (since 2013)** **[E]** — **NZ$150,000**, up to 24 months, **anonymised** application, merit-then-random: "**all applications that meet the criteria are equally eligible to receive funding**"; eligible applications are **randomly ordered** and funded down that order until budget is exhausted. Explicit rationale: evidence that "conventional ranked peer review struggles to reliably distinguish between proposals clustered near a funding cutoff, and that **fine-grained ranking near the margin can carry more noise than signal**."
- URLs: https://casrai.org/guides/hrc-explorer-grant ; Avin, "Mavericks and lotteries," Stud. Hist. Phil. Sci. 2019, https://www.sciencedirect.com/science/article/pii/S0039368118300190

### Two-stage contest theory **[E]**

- **Screening in Multistage Contests (M&SOM, 2023)** — models screening imperfection via **true-positive rate (sensitivity)** and **true-negative rate (specificity)**. Two mechanisms by which screening raises effort: the **encouragement effect** and the **competitive contest effect**. Findings: filtering out true negatives is optimal with exogenous fit; with **endogenous** fit and **less up-front complexity**, **coarse (imperfect) screening is beneficial** to manage competition and stimulate effort — otherwise use more accurate screening.
  - URL: https://doi.org/10.1287/msom.2021.0378
- **Optimal two-stage elimination contests for crowdsourcing (Trans. Res. Part E 145, 2021)** — (i) **optimal to have exactly two participants in the final stage**; (ii) participants exert more effort when fewer remain; (iii) **sequential-elimination beats sub-elimination and no-elimination**, and is also preferred by high-ability participants.
  - URL: https://ideas.repec.org/a/eee/transe/v145y2021ics1366554520308012.html
- **Fu & Lu, "The optimal multi-stage contest," Economic Theory 51(2):351–382, 2012** **[E]** — when contest technology is **sufficiently noisy**, multi-stage beats single-stage. For concave and moderately convex impact functions, allocate the **entire purse to a single final prize**. Additional stages **always increase total effort**. Optimal contest: **eliminate one contestant per stage until a single winner takes the whole purse**.
  - URL: https://ideas.repec.org/a/spr/joecth/v51y2012i2p351-382.html
- **Shortlist size (arXiv 2502.09014 / 2602.11914)** **[E]** — designer picks shortlist size m (2≤m≤n) and prize vector under budget B; cut-off score disclosed. To maximize **highest individual performance**, optimum is typically **two finalists**; to maximize **total effort**, optimal m depends on the ability distribution F. Result: optimal number of **non-zero** prizes is shortlist size **minus one** — one zero-prize is necessary to incentivize effort, more than one is unnecessary.

### B.4 design implications

- **[E→A]** Tension to resolve deliberately: contest theory says **concentrate the purse and narrow to ~2 finalists** (maximizes peak effort); metascience (B.3) says **disperse** (maximizes discovery per dollar). They are not actually contradictory — contest theory maximizes *effort*, metascience measures *discovery per dollar*. For architecture search you want discovery, and effort is cheap relative to GPU. **Favor dispersal, but note Fu & Lu's condition: multi-stage wins specifically when the contest technology is noisy — and your 6-GPU-hour single-seed evaluation is exactly a noisy contest technology.** So: multi-stage yes, winner-take-all no.
- **[E]** The peer-review evidence (AUC **0.54**, ~**1%** of variance explained, **17%** of perfect-score grants producing zero citations) is a direct argument that **fine-grained ranking near the top is noise**. If your score differences between the top submissions are within eval noise, ranking them precisely is theater — and worse, it is *exactly the margin the copy-the-leader attacker exploits*, since they only need to beat the leader by epsilon.
- **[A]** Strong, concrete recommendation supported by Volkswagen's two-cohort result: **quantize the top band and randomize within it.** Define a merit threshold, then allocate emissions by lottery (or equal split) among all submissions that clear it and occupy distinct QD cells. This removes the epsilon-improvement attack surface entirely: beating the leader by 0.1% moves you into the same band, not above it. Volkswagen's evidence says you lose **no measurable quality** by doing this.
- **[A]** Use coarse screening early (M&SOM: coarse screening is *beneficial* when up-front complexity is low), i.e. a cheap static/1-GPU-minute smoke check before granting a 6-GPU-hour slot.

---

## B.5 Registered reports / preregistration as an anti-Goodhart device

Your recollection of the numbers is close but conflates two studies. Both are real; here are the exact figures.

### Scheel, Schijen & Lakens, AMPPS 4(2), 2021, "An Excess of Positive Results" **[E]**

| Sample | Positive-result rate (first hypothesis) |
|---|---|
| Standard reports (random sample, N=152) | **96.05%** |
| **Registered Reports** (full population as of Nov 2018, N=71) | **43.66%** |
| Excluding close replications | **95.95%** vs **50.00%** |

- So **~56% null** in RRs vs ~4% null in standard reports.
- Statistics: χ²(1) = 77.96, p < .001; z = 7.61.
- Contextualization: their SR rate (96.05%) is slightly but **non-significantly** higher than Fanelli (2010)'s **91.5%**; their RR rate (43.66%) is "comparable with the **39.5%** reported by Allen and Mehler (2019) despite some differences in method."
- Their calibration argument **[E]**: "Assuming no publication bias and no QRPs, authors of [SRs] would need to test **almost exclusively true hypotheses (>90%) with more than 90% power**" — implausible, therefore bias.
- URLs: https://journals.sagepub.com/doi/10.1177/25152459211007467 (2021-04) ; preprint https://doi.org/10.31234/osf.io/p6e9c (2020-02-05)

### Allen & Mehler (PsyArXiv, posted 17 Oct 2018; reported in Nature Index) **[E]** — this is the **61%** figure

- Sample: **113 registered reports** across biomedical and psychological sciences; **296 discrete hypotheses** identified.
- **61% of hypotheses were not supported** by the published results.
- Breakdown: **66% null** for replication-oriented studies; **55% null** for original research.
- Comparison baseline given by the authors: null results in the general literature estimated at **5–20%**.
- Caveat recorded in the same article (Scheel): the study may **underestimate** the true null rate, since other research estimates **>90%** of psychology hypotheses tested are in fact false; and authors may "use this format strategically."
- URL: https://www.nature.com/nature-index/news/first-analysis-of-pre-registered-studies-shows-sharp-rise-in-null-findings

### B.5 design implications

- **[E]** So the honest headline is: **~56–61% null in registered reports vs ~4–20% in the standard literature** — a 3–15× difference depending on which baseline you take. Your "~60% vs 5–20%" recollection matches Allen & Mehler exactly.
- **[A]** The mechanism is the transferable part: **the accept/reject decision is made before the outcome is known**, so the outcome cannot be optimized against. Your analogue is a **preregistered hypothesis with a committed eval protocol**: the miner declares (and hashes) *what* their architectural change is and *what* they predict it will improve, before the eval seed and held-out set are revealed. Combine with Kaggle-style code escrow (B.1).
- **[A]** This buys something that no similarity metric can: it makes a copy-the-leader submission **rhetorically as well as numerically empty** — the miner must state a mechanistic hypothesis, and "I changed the leader's LR schedule" is a visible, cheap-to-adjudicate non-hypothesis. It moves adjudication from "is this code different?" (gameable, per A.1) to "is this claim distinct and did it hold?" (much harder to fake).
- **[A]** Corollary you should design for explicitly: **if you adopt registered-report structure, expect ~60% of your submissions to be nulls, and you must pay something for a well-specified null.** Otherwise you have merely reintroduced publication bias, and miners will only ever submit safe epsilon-tweaks — which is the original attack. Paying for informative nulls is the mechanism that makes genuine exploration rational.

---

## B.6 Negative results / reproducibility publishing — does it raise field productivity?

**Honest answer: I did not find direct causal evidence that publishing negative results increases field-level productivity.** **[A]** What I did find, and did not find:

- **[E]** SCORE establishes the *magnitude of the problem* in social/behavioral science: ~half of claims replicate; effect sizes shrink by a **median >50%**, explained variance by **>80%**. Reproducibility was **higher in journals that required data sharing** — this is the closest thing to an institutional-intervention effect in the data I gathered, and it is **correlational, not causal**.
- **[E]** Registered reports demonstrably change *what gets published* (B.5). That nulls reach the literature at 56–61% is established. That this **increases downstream productivity** is not something I found measured.
- **[A]** I am flagging this as a **genuine evidence gap**, not summarizing weak evidence as strong. If this matters to your design, it needs a dedicated search (candidate leads: the "file drawer"/publication-bias meta-analytic literature; Journal of Negative Results in BioMedicine's impact; the ML reproducibility-challenge series).

---

# C. Lineage / citation-credit mechanisms and their documented failures

## C.1 tea.xyz — the largest documented lineage-farming attack **[E]**

Mechanism being gamed: tea.xyz rewarded maintainers in TEA tokens based on **package usage / number of dependents** — i.e. reward routed up a dependency graph. Led by Homebrew creator Max Howell.

**Timeline with numbers:**

| Date | Event | Scale |
|---|---|---|
| **Feb–Mar 2024** | Spam **PRs** flooded GitHub projects; maintainers had to clean up. Howell called it "disgusting and counter productive," said he was "furious about it" | — |
| **Apr 2024** | Sonatype reports npm flooded with spam packages carrying `tea.yaml` | **~15,000** packages (Sonatype headline: 15K; blog slug says 10,000) |
| **Mar–Apr 2024** | Phylum isolates auto-generated dependency campaign | **14,000** packages registered with tea across all ecosystems, **npm hardest hit** |
| **2024** | Socket finds tea spam also on **PyPI** and **RubyGems**, all same author on PyPI | — |
| **Later 2024** | Renewed wave; Socket: "slow downs in our **infrastructure**" traced to spam packages with **thousands of transitive dependencies** and auto-generated names | thousands more |
| **Oct 24 – Nov 12, 2025** | Amazon Inspector detection campaign; coordinated with OpenSSF, each package getting a MAL-ID within 30 min | **>150,000 packages** — "one of the largest package flooding incidents in open source registry history" |
| **2025** | Endor Labs "IndonesianFoods worm" | **>43,000** spam packages, **≥11 accounts**, **~2 years**, **>1% of the entire npm ecosystem**; some packages have **thousands of weekly downloads** |

**The exact attack mechanic [E]:** "By embedding `tea.yaml` files across thousands of spam packages and **interlinking them through circular dependencies**, the attackers inflated their 'impact scores' and claimed TEA token rewards for artificial ecosystem value." Socket: "spammers are then incentivized to create massive dependency trees with thousands of garbage packages that are transitive dependencies of those that contain the `tea.xyz` yaml file… creating a **spider web effect of artificially inflated numbers of dependents**." One package README **boasted about the earnings**. Tooling used: `ReguideWIKI/teaSimple-vCore`.

**Aggravating detail [E]:** Endor Labs notes that although the attack was described in April 2024, "**most of the offending packages were kept on npm until today**" (2025) — i.e. remediation lagged by ~18 months, and the dormant packages with real download counts constitute a **latent supply-chain risk**.

- URLs: https://www.sonatype.com/blog/devs-flood-npm-with-10000-packages-to-reward-themselves-with-tea-tokens ; https://socket.dev/blog/tea-xyz-spam-plagues-npm-and-rubygems-package-registries ; https://socket.dev/blog/massive-automated-spam-campaign-floods-npm-registry-with-thousands-of-garbage-tea-xyz-packages ; https://aws.amazon.com/blogs/security/amazon-inspector-detects-over-150000-malicious-packages-linked-to-token-farming-campaign/ (2025) ; https://www.endorlabs.com/learn/the-great-indonesian-tea-theft-analyzing-a-npm-spam-campaign

**[A] The lesson, stated as sharply as the evidence permits:** tea.xyz is the definitive case study that **any reward routed over a graph whose edges the rewarded party can create is farmed to destruction.** The attacker does not need to fake *quality*; they only need to fake *topology*. 150,000 packages is not a bug in tea's implementation — it is the equilibrium of the mechanism. **If your lineage graph lets miners declare their own ancestry/dependency edges, you have built tea.xyz.** Edges must be either (a) operator-attested, (b) derived from evidence the miner cannot cheaply manufacture (e.g. measured weight/activation similarity from the actual trained artifact), or (c) economically costly to create.

## C.2 Optimism RetroPGF — popularity bias, quorum gaming, expert conflict **[E]**

**Scale [R]:** RetroPGF has distributed **>60M OP** to hundreds of projects; **850M OP (20% of total supply)** reserved for public goods.

**Round 3 documented failures — from Optimism's own retrospective [E]:**

1. **Quorum → popularity contest.** "Applicants were in fear of not meeting the quorum and reached out to badgeholders **en masse**… A small unrepresentative poll showed that **each badgeholder received more than 15 DMs** from applicants. This created a dynamic which was perceived **more like a popularity contest** than an exercise in objectively evaluating past contributions."
2. **Self-selection → selection bias.** "Badgeholders could **self-select** which applications to vote on. This resulted in selection bias, in which some badgeholders selected applications which they were **already familiar with, and thought positively of**, while not voting on applications which they were not familiar with or thought negatively of." Consequence: "making it **less likely for an unknown applicant to meet quorum**."
3. **No comparable metrics.** "The absence of [structured], and comparable impact metrics, and the reliance on **individual subjective review criteria**, made it difficult to objectively measure impact."
4. **The expert-conflict dilemma.** "badgeholders who are **experts within a particular field usually also work in this field and are associated with one or multiple applications**… we should expect more badgeholders [voting] on projects that are [their own]. To ensure an unbiased process, **collusion and bribery resistance is crucial**."

**From community feedback threads [E/R]:**
- Volume: **600+ projects**, up **6×** on the prior round despite prior-round feedback that volume was already overwhelming. "It's impossible to expect every badgeholder to go through 600+ projects."
- Fairness inversion: "**Last round was more fair for smaller or lesser known projects as there was no minimum vote requirement.** The result would probably be only the most well known projects are funded."
- **Lists** (curation aid) judged net-harmful: "There's **no level of expertise required to make a list**, so we're seeing a large amount of lists that include low impact projects which then highlights them to other badgeholders… **lists are more harmful than helpful**." "Lists create **recency bias**."
- Enumerated problems in the voting-algorithms thread: **1. Popularity Bias & Nepotism**, 2. Lack of feedback to projects, 3. Quantifying impact as an OP amount, 4. Considering previous grants/income, 5. Badgeholder overwhelm, **6. Over-rewarding subpar projects, under-rewarding top-tier projects**, 7. Inequitable recognition of badgeholder effort.
- **Perverse quorum incentive, precisely stated [E]:** quorum "**incentivizes projects concerned about making quorum to promote to all badgeholders** to get any RetroPGF (maximize 'likes')" while it "**disincentivizes projects confident of making quorum from creating awareness** outside badgeholders already aware of their impact, as this could **reduce** RetroPGF ('love' vs 'like')." — i.e. the mechanism punishes the confidently-excellent for being visible.
- Reflexivity warning **[R]**: using "previous RetroPGF success" as an impact metric risks "getting projects **trapped in reinforcing loops** of misled voting."
- Original quorum intent **[E]**: "**Minimum threshold of badgeholder votes a project must receive to qualify — this is to prevent against a small number of badgeholders colluding to dictate the allocation of OP.**" So the anti-collusion device *became* the popularity-contest device.

**Mitigations adopted/proposed [R]:** metrics-informed evaluation with **Open Source Observer**; category/domain-scoped voting so badgeholders "only need to vote on areas of expertise"; **random assignment** of badgeholders to projects with **reviewer identities hidden until after the process** to reduce collusion; guest voters.

- URLs: https://optimism.io/blog/retropgf-3-learnings-reflections ; https://gov.optimism.io/t/retropgf-round-3-feedback-thread/6177 ; https://gov.optimism.io/t/retropgf-3-round-design/6802/1 ; https://gov.optimism.io/t/retropgf-experimentation-voting-algorithms/7216 ; https://gitcoin.co/apps/optimism-retropgf

**[A] Lesson:** retroactive funding replaces "predict what will work" with "recognize what worked" — but **recognition is bounded by evaluator attention**, and attention is the scarce, gameable resource. At 600 applications with self-selected review, the mechanism degenerates into a measure of *marketing reach*. Two directly-applicable fixes are already in Optimism's own remediation list: **random reviewer assignment** and **blind review until completion**.

## C.3 Gitcoin quadratic funding — sybil/collusion and the pairwise mitigation **[E]**

**Mechanism [E]:** Buterin, Hitzig & Weyl, "A Flexible Design for Funding Public Goods" (Management Science, 2019; earlier as "Liberal Radicalism," 2018). Matching = **square of the sum of square roots** of individual contributions. Compresses large donations, amplifies broad support. Gitcoin used the CQF mechanism to allocate **$25,000** in **February 2019**.

**Scale [R]:** Gitcoin Grants has distributed **>$60 million** to **>3,700 projects**.

**Documented attack surface [E]:**
- **Sybil:** "grants which receive many small contributions result in a larger 'top-off' value from the benefactor, **incentivizing an attack vector to create multiple dummy accounts**." Concretely: "an actor could decide to create a grant, donate to himself, and collect [matching] as 'interest'."
- **Collusion:** "malicious real users **secretly coordinate** among themselves to game the system."
- **The hard part, stated honestly [E]:** "The difficulty of addressing these attack vectors is compounded by the **overlap with transaction patterns resulting from legitimate, organic contributions**." — i.e. sybil signatures and genuine grassroots support look alike.

**Mitigations and their measured effect [E]:**

| Mitigation | Mechanism | Evidence |
|---|---|---|
| **Pairwise coordination subsidies** (Buterin, ethresear.ch #5553) | "the amount of funds a specific pair puts toward the same grant is **evidence of how coordinated they are**, and so the more grants both of them donate to, the more **constricted** the CLR match for that pair" | Deployed by Gitcoin **[E]**; graph in Gitcoin's repo shows number of contributions dominates contribution amount regardless of mechanism, and it penalizes grants dominated by large amounts |
| **Gitcoin Passport** (verifiable-credential "stamps" → passport score gating matching weight) | Identity layering: BrightID, Idena, POAP, 3box; later ML-based model score, onchain passport, **stamp rotation to counter farming** | **"~60% reduction in suspicious activity in GG23"**; "Gitcoin Passport reduced flagged addresses by ~60%" **[E]** |
| **Sybil-score ML detection** | Assigns users a sybil score | "ML-based detection caught patterns humans missed" **[R]** |
| **COCM** (Connection-Oriented Cluster Matching) | Cluster-aware matching adjustment | deployed in post-round review **[R]** |
| **MACI** (clr.fund) | ZK: "voters can't prove how they voted (prevents bribery)"; votes encrypted | "**Doesn't directly prevent sybils** but removes incentive" **[E]** |
| Post-round manual review | Operators adjust/reallocate final allocations | standard practice **[R]** |

**Gitcoin's own honest conclusions [E]:** "**No single solution is sufficient**"; "**Some sophisticated attacks still succeed**"; "Accept tradeoffs: some accessibility loss for integrity"; "Attackers evolve, defenses must too."

- URLs: https://doi.org/10.1287/mnsc.2019.3337 ; https://gitcoin.co/mechanisms/quadratic-funding ; https://github.com/gitcoinco/quadratic-funding ; https://blog.block.science/how-to-attack-and-defend-quadratic-funding/ ; https://gitcoin.co/research/quadratic-funding-sybil-resistance ; https://ethresear.ch/t/pairwise-coordination-subsidies-a-new-quadratic-funding-design/5553

**[A] The transferable idea is pairwise-bounded matching.** It is the only mechanism in this entire review that *directly penalizes correlated behaviour* — and correlated behaviour is exactly the signature of copy-the-leader. Generalized to your setting: **discount rewards between pairs of submissions in proportion to their measured co-movement** (correlated per-example errors, correlated training dynamics, shared descriptor cells). Two miners submitting near-identical architectures mutually constrict each other's payout. This composes cleanly with the Numerai residualization approach and needs no plagiarism detector. Note the honest caveat that survived 7 years of Gitcoin operation: **~60% reduction, not elimination.**

## C.4 Dependency funding — thanks.dev **[E/R]**

**Mechanism [E]:** "(1) walk your repositories; (2) grab the manifest files; (3) collate your dependency tree **up to 3 levels deep**; (4) **trickle your donation breadth first** across said tree." Weighting is by **frequency of being depended upon**, not stars or popularity. Donors can boost/reduce weight **at the programming-language level and GitHub-org level**. thanks.dev takes a **5%** commission. Maintainers **must register** to receive funds.

**Real deployment numbers [E]:** Canonical committed **US$120,000 over 12 months** at **$10,000/month**, reaching **over 350 GitHub users and orgs**. Adopters also include Sentry, Cash App, Sourcegraph. Sentry's per-project amounts illustrate the granularity problem: **$7.17** to actix, **$8.82** to axios, **$5.43** to blakeembrey, **$28.74** to brianc. Distribution goes to **up to eight decimal places** on the dollar. Context **[E]**: "just one npm package has on average **79 dependencies**."

**Documented criticisms [R]:**
- Micro-donation futility: amounts are disproportionately small relative to maintenance effort.
- **Explicit gaming vectors named in community discussion [R]:** "things like **breaking your project up into many small packages**" and "incentive to **juice download numbers**." *(Note: this is precisely the tea.xyz mechanic, predicted independently.)*
- Coverage gap: projects not in a large codebase's dependency tree are "outside the algorithm's scope"; end-user-facing projects get nothing.
- Registration gap: "the algorithm might scan and find you, but if you haven't registered, the money won't flow."
- Demand-side gap **[R]**: most organizations relying on OSS "are not tech companies… (e.g. airports and hospitals) and are unlikely to ever fund their own software supply chains."

- URLs: https://canonical.com/blog/canonical-thanks-dev-giving-back-to-open-source-developers ; https://www.theregister.com/software/2023/04/07/its-time-to-pay-up-for-open-source-with-direct-donations/656979 (2023-04-07) ; https://news.ycombinator.com/item?id=42312469 ; https://shenxianpeng.github.io/en/posts/2026/thanks-dev/

**[A] Contrast with tea.xyz that is worth internalizing:** thanks.dev routes over the **same kind of graph** but has **not** produced a 150,000-package farming incident. The difference is not a cleverer algorithm — it is that **the payout is a fixed philanthropic budget, not a token with speculative upside, and the graph edges come from the *donor's* repositories, not the recipient's declarations.** Attack profitability, not graph design, is what determined the outcome. **For an emissions-funded system you are structurally in tea.xyz's position, not thanks.dev's** — so you need the operator-attested-edge discipline from C.1.

## C.5 Academic credit-assignment algorithms **[E]**

**Shen & Barabási, "Collective credit allocation in science," PNAS 111(34):12325–12330, 2014** **[E]**

Algorithm, precisely:
1. Target paper p₀ with m coauthors. Identify all papers citing p₀ → set D = {d₁…d_l}.
2. Identify all **co-cited** papers P = {p₀, p₁…p_n} — the complete set of papers cited by papers in D.
3. **Co-citation strength** s_j = number of times p₀ and p_j are cited together by papers in D. (Worked example: s₁=1 because only d₁ cites p₀ and p₁ together; s₂=4 because d₁,d₂,d₃,d₅ cite p₀ and p₂ together.) p₀ counts as a co-cited paper of itself with strength = its own citation count — so **highly-cited papers are less perturbable by other co-cited papers**.
4. Build credit allocation matrix A, A_ij = credit author a_i gets from co-cited paper p_j, using a **fractional** allocation that **does not depend on author order**.
5. Total credit c_i = weighted sum of local credit over all co-cited papers.

Rationale **[E]**: "Co-citation strength captures the intuition that **papers by an author that are perceived to be very relevant to paper p₀ should increase the author's perceived contribution to p₀**." Aim is to reproduce "the informal collective credit allocation of science" — the community's perception, not a self-declared contribution statement.

Validation **[E]**: identifies the authors of **Nobel-winning papers** credited for the discovery **independent of their position in the author list**; can also compare researchers in the same field who never published together.

- URLs: https://doi.org/10.1073/pnas.1401992111 ; https://www.pnas.org/doi/10.1073/pnas.1401992111 ; PDF https://www.barabasi.com/media/pub_imports/files/492.pdf

**[A] Why this one is unusually relevant to you.** The Shen–Barabási design has exactly the property tea.xyz lacked: **credit is inferred from third-party behaviour (who cites what together), not declared by the beneficiary.** An author cannot inflate their credit share by asserting relevance; they can only do so if *other people's* citing behaviour treats their other work as relevant to the target. That is a **costly-to-forge, externally-generated edge**. Your analogue: derive lineage weight from **which prior architectures the operator's own measurements show a submission to be derived from** (weight/activation/curve similarity, ablation overlap), and from **which architectures other successful submissions build on** — never from a miner-supplied "I built on X" field. Note the flip side **[A]**: the algorithm is *inherently popularity-coupled* (it is built on citation counts), so it inherits RetroPGF's popularity-bias problem. It allocates *shares* well; it does not tell you the *size* of the pie.

**[A] Not researched in this pass, flagged as gaps:** OpenAlex-based funding proposals; "impact certificates" as a distinct instrument from RetroPGF; PageRank-style citation-credit variants beyond Shen–Barabási (e.g. CiteRank, SARA); Protocol Labs dependency-funding programs beyond thanks.dev/tea.xyz.

---

# D. Synthesis: what survives adversarial pressure

## D.1 Ranking of novelty-measurement approaches by adversarial robustness

| Rank | Approach | Robustness | Evidence |
|---|---|---|---|
| **1** | **Marginal contribution to a reference/portfolio** (Numerai MMC: neutralize against reference, covariance with target) | **Highest.** Self-certifying — a copy has ~zero orthogonal component by construction. O(1) cost. Locally verifiable. | 7 years live at real money **[E]** |
| **2** | **Operator-defined QD cells** (MAP-Elites-style bounded descriptor taxonomy; pay cell elites) | **High.** Attacker cannot invent cells; must beat an incumbent within a cell you defined. | MAP-Elites beats novelty-search-LC on all 4 criteria, p<1e-7 **[E]** |
| **3** | **Pairwise-bounded / correlation-discounted rewards** (Gitcoin pairwise subsidies) | **High-ish.** Directly penalizes correlated behaviour = the copy signature. But: ~60% reduction, not elimination. | Deployed; ~60% suspicious-activity reduction **[E]** |
| **4** | **Preregistration + code escrow** (registered reports + Kaggle two-stage checksum) | **High for a different threat** — stops post-hoc metric-chasing and eval probing rather than copying. | RR nulls 56–61% vs 4–20% **[E]**; Kaggle DQ-on-mismatch **[E]** |
| **5** | **Behavioural/artifact fingerprints** (training curves, per-example loss profiles, weight spectra) | **Medium-high [A]** — faking requires changing actual computation. Not directly evidenced in this review. | inference from A.1 |
| **6** | **JPlag-class normalized AST/token matching** | **Medium.** 94.2 pp separation post-2024-normalization, but an active arms race. | **[E]** |
| **7** | **Forecast/market triage layer** | **Medium as triage only.** 71–78% accuracy; unstable across batches (dropped to 50% in one set). | **[E]** |
| **8** | **Learned code embeddings** (CodeBERT/GraphCodeBERT/UniXcoder) | **Low.** Clone-detection ASR 27.79% (ALERT) → 0.40 (GraphCodeAttack) on CodeBERT. | **[E]** |
| **9** | **MOSS / Dolos / winnowing similarity** | **Broken for this purpose.** Obfuscated plagiarism median 7.5% vs unrelated 4.5% → 3.0 pp delta → "impossible." | **[E]** |
| **10** | **Raw novelty / dissimilarity score as a reward term** | **Actively harmful.** Pays for obfuscation; "one can always design a fitness function such that solutions discovered by novelty alone perform arbitrarily badly." | **[E]** |
| **11** | **Self-declared lineage/dependency edges** | **Catastrophic.** 150,000 farmed packages. | **[E]** |

## D.2 The composite mechanism the evidence supports

**[A]** All of the following is my synthesis, grounded in the cited evidence but not itself a measured result:

1. **Never score novelty directly.** Score **residual improvement after neutralizing against the current frontier/reference** (Numerai MMC). Copy-the-leader → residual ≈ 0 → payout ≈ 0, automatically, with no detector to evade.
2. **Bound the novelty space yourself.** Publish a small operator-controlled descriptor taxonomy (MAP-Elites cells over architecture families). Pay cell elites. This converts unbounded gameable novelty into an enumerable, operator-owned set.
3. **Discount correlated submissions pairwise** (Gitcoin). Two miners whose per-example error profiles co-move mutually constrict each other's payout.
4. **Quantize the top band and randomize/split within it** (Volkswagen two-cohort result: no measurable quality loss; NIH: AUC 0.54 near the top). Kills the epsilon-improvement attack: beating the leader by 0.1% lands you in the same band, not above it.
5. **Two stages, coarse then expensive** (M&SOM; Fu & Lu — multi-stage wins precisely when contest technology is noisy, and 6-GPU-hours/single-seed *is* noisy). Cheap screen → 6-GPU-hour slot.
6. **Preregister + escrow.** Hash architecture, code, and predicted-improvement claim before eval seed/held-out set is revealed. Checksum mismatch = disqualification (Kaggle).
7. **Pay for well-specified nulls.** RR evidence says ~60% of honest exploration returns nulls. If nulls pay zero, rational miners submit only safe tweaks — reproducing the original attack.
8. **Lineage edges must be operator-attested or measurement-derived, never self-declared** (tea.xyz; Shen–Barabási's third-party co-citation principle).
9. **Add a small explicit frontier-founder term.** Netflix's Simon Funk unlocked the paradigm and won nothing; Numerai users argued the 0.5×CORR residual under-rewards those who built the meta model. Pure-marginal payment has a free-rider-on-the-frontier problem.
10. **Budget dispersal over concentration** (Fortin & Currie; Mongeon; Aagaard review), while accepting a separate concentrated scale-up stage for exploitation.

## D.3 Residual risks in the above **[A]**

- **Descriptor-boundary clustering.** Publishing cells tells miners exactly where the underpopulated cells are; expect gaming *toward* empty cells with minimum-effort architectures that technically qualify. Mitigation: cells must still require beating a quality floor (QD, not novelty).
- **Reference-gaming.** If the reference/meta-architecture is stake-weighted, a large staker can shift the reference to make their own submission look orthogonal. Numerai lives with this; note their fix was making MMC mandatory and locally computable.
- **Fixed grids age badly** (Dominated Novelty Search critique). Plan seasonal taxonomy revision.
- **Noise floor.** With single-seed 6-GPU-hour evals, a meaningful fraction of "improvements" is seed noise. Bootstrap the noise floor and clip payouts (Numerai clips ±5%) or you will pay for variance.
- **Legibility is load-bearing.** TC died of being blackbox and non-locally-computable despite being conceptually better than MMC. Whatever you ship, miners must be able to compute it themselves.

---

# Sources

All URLs accessed **2026-08-16**. Dates are publication dates where known.

## A.1 Code similarity and evasion
1. Devore-McDonald & Berger, "Mossad: defeating software plagiarism detection," OOPSLA 2020 — https://doi.org/10.1145/3428206
2. Sağlam, Hahner, Wittler, Kotthoff, "Obfuscation-Resilient Software Plagiarism Detection with JPlag," ICSE-SEET 2024 — https://doi.org/10.1145/3639478.3643074
3. Sağlam et al., "Evaluating Software Plagiarism Detection in the Age of AI," arXiv:2505.20158, May 2025 — https://arxiv.org/pdf/2505.20158
4. "Comparative Analysis of Non-Commercial Plagiarism Detectors for Computer Science Education," SCITEPRESS 2026 — https://www.scitepress.org/Papers/2026/148365/148365.pdf
5. Yang, Shi, He, Lo, "Natural Attack for Pre-trained Models of Code" (ALERT), ICSE 2022 — https://arxiv.org/pdf/2201.08698
6. "Adversarial Attacks on Code Models with Discriminative Graph Patterns" (GraphCodeAttack), arXiv:2308.11161, 2023/2024 — https://arxiv.org/html/2308.11161
7. "An Extensive Study on Adversarial Attack against Pre-trained Models of Code," arXiv:2311.07553, 2023 — https://doi.org/10.48550/arxiv.2311.07553
8. Ramakrishnan et al., "Semantic Robustness of Models of Source Code," arXiv:2002.03043, 2020 — https://arxiv.org/pdf/2002.03043
9. Schleimer, Wilkerson, Aiken, "Winnowing: Local Algorithms for Document Fingerprinting," SIGMOD 2003 (MOSS foundation; cited via [4])

## A.2 Novelty search and Quality-Diversity
10. Lehman & Stanley, "Abandoning Objectives: Evolution Through the Search for Novelty Alone," Evolutionary Computation 19(2), 2011 — https://doi.org/10.1162/evco_a_00025 ; UCF copy https://stars.library.ucf.edu/cgi/viewcontent.cgi?article=2529&context=facultybib2010
11. Lehman dissertation, "Evolution Through the Search for Novelty" — https://joellehman.com/lehman-dissertation.pdf ; https://stars.library.ucf.edu/cgi/viewcontent.cgi?article=3213&context=etd
12. **Cuccu & Gomez, "When Novelty Is Not Enough," EvoStar 2011** — https://people.idsia.ch/~tino/papers/cuccu.evostar11.pdf
13. Mouret & Clune, "Illuminating search spaces by mapping elites," arXiv:1504.04909, 2015 — https://doi.org/10.48550/arxiv.1504.04909 ; https://ar5iv.labs.arxiv.org/html/1504.04909
14. Pugh, Soros, Stanley, "Quality Diversity: A New Frontier for Evolutionary Computation," Frontiers in Robotics and AI, 2016 — https://doi.org/10.3389/frobt.2016.00040
15. "Preliminary Analysis of Simple Novelty Search," Evolutionary Computation 32(3):249, 2024 — https://direct.mit.edu/evco/article/32/3/249/116787/
16. "Dominated Novelty Search: Rethinking Local Competition in Quality-Diversity," arXiv:2502.00593, 2025 — https://arxiv.org/html/2502.00593v1
17. "Multi-objective Analysis of MAP-Elites Performance," arXiv:1803.05174 — https://arxiv.org/pdf/1803.05174
18. Cully, Clune, Tarapore, Mouret, "Robots that can adapt like animals," Nature 521:503, 2015 (cited via [14])

## A.3 Numerai
19. "A New Data Science Competition Where Being Different Pays," Numerai blog — https://blog.numer.ai/a-new-data-science-competition-where-being-different-pays/
20. "MMC2 Announcement," Numerai forum — https://forum.numer.ai/t/mmc2-announcement/93
21. **"Leaderboard Bonus Exploit Uncovered," Numerai forum, 2020** — https://forum.numer.ai/t/leaderboard-bonus-exploit-uncovered/200
22. **"MMC — Payout Details and Analysis," Numerai forum, 2020** — https://forum.numer.ai/t/mmc-payout-details-and-analysis/220
23. "True Contribution Details," Numerai forum, 2022 — https://forum.numer.ai/t/true-contribution-details/5128
24. "True Contribution (TC)," Numerai docs — https://docs.numer.ai/numerai-tournament/scoring/true-contribution-tc
25. **"Changing Scoring & Payouts Again To MMC Only," Numerai forum, late 2023** — https://forum.numer.ai/t/changing-scoring-payouts-again-to-mmc-only/6794
26. "MMC staking starts Jan 2, 2024," Numerai forum — https://forum.numer.ai/t/mmc-staking-starts-jan-2-2024/6827
27. "Meta Model Contribution (MMC)," Numerai docs — https://docs.numer.ai/numerai-tournament/scoring/meta-model-contribution-mmc
28. "Scoring," Numerai docs — https://docs.numer.ai/numerai-tournament/scoring
29. "Staking," Numerai docs — https://docs.numer.ai/numerai-tournament/staking
30. **"Numerai Monthly: … 2026 Payout Updates," Numerai blog, Jan 2026** — https://blog.numer.ai/numerai-monthly-numercon-speakers-new-dataset-target-2026-payout-updates/
31. "Alpha," Numerai Signals docs — https://docs.numer.ai/numerai-signals/scoring/alpha
32. "Atomic Blockchain Staking," Numerai docs — https://docs.numer.ai/numerai-tournament/atomic-blockchain-staking

## A.4 Shapley / marginal contribution
33. Ghorbani & Zou, "Data Shapley: Equitable Valuation of Data for Machine Learning," ICML 2019 — https://proceedings.mlr.press/v97/ghorbani19c/ghorbani19c.pdf
34. "A Note on 'Towards Efficient Data Valuation Based on the Shapley Value'," arXiv:2302.11431, 2023 — https://ar5iv.labs.arxiv.org/html/2302.11431
35. "An Empirical and Comparative Analysis of Data Valuation with Scalable Algorithms" (KNN-Shapley) — https://openreview.net/pdf?id=SygBIxSFDS
36. "Fast-DataShapley: Neural Modeling for Training Data Valuation," arXiv:2506.05281, 2025 — https://arxiv.org/html/2506.05281v3
37. Koh & Liang, "Understanding Black-box Predictions via Influence Functions," 2017 (cited via [35])

## B.1 Prize contests
38. DARPA, "Grand Challenge" innovation timeline — https://www.darpa.mil/about/innovation-timeline/grand-challenge
39. DARPA, "The DARPA Grand Challenge: Ten Years Later," 2014-03-13 — https://www.darpa.mil/news/2014/grand-challenge-ten-years-later
40. IEEE Spectrum, "Dusted: No winners in DARPA's $1 million robotic race," 2004-03-18 — https://spectrum.ieee.org/dusted-no-winners-in-darpas-1-million-robotic-race-across-the-mojavedesert
41. Thrun et al., "Stanley: The robot that won the DARPA Grand Challenge" — https://robots.stanford.edu/papers/thrun.stanley05.pdf
42. **Amatriain & Basilico, "Netflix Recommendations: Beyond the 5 stars (Part 1)," Netflix Tech Blog, 2012** — https://netflixtechblog.com/netflix-recommendations-beyond-the-5-stars-part-1-55838468f429
43. Amatriain, "Building Industrial-scale Real-world Recommender Systems," RecSys 2012 tutorial — https://amatria.in/pubs/recsys12-tutorial.pdf
44. "Lessons from the Netflix Prize: Changing Requirements and Cost-Effectiveness" — https://queirozf.com/entries/the-netflix-prize-changing-requirements-and-cost-effectiveness
45. Kaggle, "Two-Stage Frequently Asked Questions" — https://www.kaggle.com/two-stage-frequently-asked-questions
46. Kaggle, "Getting Started on Kaggle / competitions docs" — https://www.kaggle.com/docs/competitions
47. Marshall, "Dynamic Tournament Design: Evidence from Kaggle" — https://g-marshall.github.io/kaggle1.pdf

## B.2 Prediction markets for replication
48. **Dreber, Pfeiffer, Almenberg, Isaksson, Wilson, Chen, Nosek, Johannesson, "Using prediction markets to estimate the reproducibility of scientific research," PNAS 2015** — https://doi.org/10.1073/pnas.1516179112 ; PDF https://www.stat.berkeley.edu/~aldous/157/Papers/prediction_replication.pdf
49. "Predicting replicability — Analysis of survey and prediction market data from large-scale forecasting projects," PLOS ONE 2021 — https://doi.org/10.1371/journal.pone.0248780
50. **Center for Open Science, "SCORE"** — https://www.cos.io/score
51. DARPA, "SCORE: Systematizing Confidence in Open Research and Evidence" — https://www.darpa.mil/research/programs/systematizing-confidence-in-open-research-and-evidence
52. COS, "Large-Scale Collaboration Releases New Findings on Research Credibility," 2026 — https://www.cos.io/about/news/large-scale-collaboration-releases-new-findings-on-research-credibility
53. "Predicting reliability through structured expert elicitation with repliCATS" — https://doi.org/10.31222/osf.io/2pczv
54. Open Science Collaboration, "Estimating the reproducibility of psychological science," Science 2015 (RPP; cited via [48])
55. Camerer et al., Social Sciences Replication Project (cited via [49])

## B.3 Funding concentration vs dispersal
56. **Fortin & Currie, "Big Science vs. Little Science: How Scientific Impact Scales with Funding," PLOS ONE 8(6):e65263, 2013** — https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0065263
57. **Mongeon, Brodeur, Beaudry, Larivière, "Concentration of research funding leads to decreasing marginal returns," Research Evaluation, 2016** — https://doi.org/10.1093/reseval/rvw007
58. **Aagaard, Kladakis, Nielsen, "Concentration or dispersal of research funding?", Quantitative Science Studies 1(1):117, 2020** — https://direct.mit.edu/qss/article/1/1/117/15557/

## B.4 Peer review, lotteries, contest theory
59. **Fang, Bowen, Casadevall, "NIH peer review percentile scores are poorly predictive of grant productivity," eLife 2016** — https://elifesciences.org/articles/13323 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC4769156/
60. Danthi, Wu, Shi, Lauer, "Percentile Ranking and Citation Impact of a Large Cohort of NHLBI-funded Cardiovascular R01 Grants," Circulation Research 2014 — https://doi.org/10.1161/circresaha.114.302656 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC3959724/
61. "Association of percentile ranking with citation impact and productivity in a large cohort of de novo NIMH-funded R01 grants" — https://pmc.ncbi.nlm.nih.gov/articles/PMC5526589/
62. "Funding lotteries for research grant allocation: An extended taxonomy and evaluation of their fairness," Research Evaluation rvae025, 2024 — https://doi.org/10.1093/reseval/rvae025
63. **DORA, "Insights on Partial Randomization in Research Funding: Learnings from the Volkswagen Foundation," 2025-03-27** — https://sfdora.org/2025/03/27/insights-on-partial-randomization-in-research-funding-learnings-from-the-volkswagen-foundation/
64. EMBO, "Focal randomisation in grant allocation" — https://www.embo.org/blog/focal-randomisation-in-grant-allocation-a-new-wave-of-innovation-in-grant-funding/
65. HRC New Zealand Explorer Grant guide — https://casrai.org/guides/hrc-explorer-grant
66. Avin, "Mavericks and lotteries," Studies in History and Philosophy of Science, 2019 — https://www.sciencedirect.com/science/article/pii/S0039368118300190
67. "Screening in Multistage Contests," M&SOM — https://doi.org/10.1287/msom.2021.0378
68. "Optimal two-stage elimination contests for crowdsourcing," Transportation Research Part E 145, 2021 — https://ideas.repec.org/a/eee/transe/v145y2021ics1366554520308012.html
69. Fu & Lu, "The optimal multi-stage contest," Economic Theory 51(2):351–382, 2012 — https://ideas.repec.org/a/spr/joecth/v51y2012i2p351-382.html
70. "Optimal Contest Design with Entry Restriction," arXiv:2502.09014 — https://ar5iv.labs.arxiv.org/html/2502.09014
71. "Incentive Effects of a Cut-Off Score: Optimal Contest Design with Transparent Pre-Selection," arXiv:2602.11914 — https://arxiv.org/pdf/2602.11914

## B.5 Registered reports
72. **Scheel, Schijen, Lakens, "An Excess of Positive Results: Comparing the Standard Psychology Literature With Registered Reports," AMPPS 4(2), Apr 2021** — https://journals.sagepub.com/doi/10.1177/25152459211007467 ; preprint 2020-02-05 https://doi.org/10.31234/osf.io/p6e9c
73. **Nature Index, "First analysis of 'pre-registered' studies shows sharp rise in null findings" (Allen & Mehler, PsyArXiv 2018-10-17)** — https://www.nature.com/nature-index/news/first-analysis-of-pre-registered-studies-shows-sharp-rise-in-null-findings
74. Fanelli (2010), 91.5% positive-result baseline (cited via [72])
75. Szucs & Ioannidis (2017) (cited via [72])

## C. Lineage / retro-funding
76. **Sonatype, "Devs Flood npm with 15K Packages to Receive Tea tokens," 2024** — https://www.sonatype.com/blog/devs-flood-npm-with-10000-packages-to-reward-themselves-with-tea-tokens
77. **Socket, "tea.xyz spam plagues npm and RubyGems package registries," 2024** — https://socket.dev/blog/tea-xyz-spam-plagues-npm-and-rubygems-package-registries
78. Socket, "Massive Automated Spam Campaign Abuses GitHub to Flood npm Registry," 2024 — https://socket.dev/blog/massive-automated-spam-campaign-floods-npm-registry-with-thousands-of-garbage-tea-xyz-packages
79. **AWS Security Blog, "Amazon Inspector detects over 150,000 malicious packages linked to token farming campaign," Nov 2025** — https://aws.amazon.com/blogs/security/amazon-inspector-detects-over-150000-malicious-packages-linked-to-token-farming-campaign/
80. **Endor Labs, "The Great Indonesian TEA Theft: Analyzing a NPM Spam Campaign," 2025** — https://www.endorlabs.com/learn/the-great-indonesian-tea-theft-analyzing-a-npm-spam-campaign
81. **Optimism, "RetroPGF 3: Learnings & Reflections"** — https://optimism.io/blog/retropgf-3-learnings-reflections
82. Optimism Collective, "RetroPGF Round 3 Feedback Thread" — https://gov.optimism.io/t/retropgf-round-3-feedback-thread/6177
83. Optimism Collective, "RetroPGF 3: Round Design" — https://gov.optimism.io/t/retropgf-3-round-design/6802/1
84. Optimism Collective, "RetroPGF Experimentation: Voting Algorithms" — https://gov.optimism.io/t/retropgf-experimentation-voting-algorithms/7216
85. Gitcoin, "Optimism RetroPGF" overview — https://gitcoin.co/apps/optimism-retropgf
86. **Buterin, Hitzig, Weyl, "A Flexible Design for Funding Public Goods," Management Science, 2019** — https://doi.org/10.1287/mnsc.2019.3337
87. Gitcoin, "Quadratic Funding" mechanism page — https://gitcoin.co/mechanisms/quadratic-funding
88. gitcoinco/quadratic-funding (implementation + pairwise notes) — https://github.com/gitcoinco/quadratic-funding
89. **BlockScience, "How to Attack and Defend Quadratic Funding"** — https://blog.block.science/how-to-attack-and-defend-quadratic-funding/
90. Gitcoin, "Sybil Resistance in Quadratic Funding: 2024 Approaches" — https://gitcoin.co/research/quadratic-funding-sybil-resistance
91. Buterin, "Pairwise coordination subsidies: a new quadratic funding design," ethresear.ch — https://ethresear.ch/t/pairwise-coordination-subsidies-a-new-quadratic-funding-design/5553
92. Canonical, "Canonical + thanks.dev = giving back to open source developers" — https://canonical.com/blog/canonical-thanks-dev-giving-back-to-open-source-developers
93. The Register, "It's time to pay up for open source, with direct donations," 2023-04-07 — https://www.theregister.com/software/2023/04/07/its-time-to-pay-up-for-open-source-with-direct-donations/656979
94. Hacker News, "I algorithmically donated $5000 to Open Source" (thanks.dev gaming-vector discussion) — https://news.ycombinator.com/item?id=42312469
95. Shen (2026), "How to Claim the 'Lottery Ticket' of Open Source — thanks.dev's Operational Mechanism" — https://shenxianpeng.github.io/en/posts/2026/thanks-dev/
96. **Shen & Barabási, "Collective credit allocation in science," PNAS 111(34):12325–12330, 2014** — https://doi.org/10.1073/pnas.1401992111 ; https://www.pnas.org/doi/10.1073/pnas.1401992111 ; PDF https://www.barabasi.com/media/pub_imports/files/492.pdf

## Other
97. eLife, "Point of View: How should novelty be valued in science?" — https://elifesciences.org/articles/28699

---

# Explicit evidence gaps (not researched or not found in this pass)

Flagging rather than fabricating:
- **XPRIZE** outcome data and incentive-prize effectiveness studies — no primary figures obtained.
- **AI Grant, AI-safety/alignment bounties, Erdős/Clay problems, Focused Research Organizations** — not researched.
- **Negative-results publishing → field productivity**: no direct causal evidence found (see B.6).
- **OpenAlex-based funding proposals; "impact certificates"** as an instrument distinct from RetroPGF — not researched.
- **PageRank-style citation-credit algorithms** beyond Shen–Barabási (CiteRank, SARA, etc.) — not researched.
- **Ocean Protocol / data-market** Shapley deployments; **federated-learning contribution schemes** in production — not verified.
- **NSF triage / NIH triage** mechanics specifically (as distinct from percentile-reliability studies) — not researched.
- **Type-1/2/3/4 clone-detection baseline accuracy** and **tree-edit-distance** methods — covered only indirectly via the evasion literature; no clean baseline table obtained.
- **RetroPGF Rounds 4–6 / Retro Funding Missions** outcomes — only Round 3 obtained in depth.

