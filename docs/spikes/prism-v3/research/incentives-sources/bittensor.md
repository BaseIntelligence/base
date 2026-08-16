# Rewarding Model/Architecture Competition Under Noisy, Gameable Evaluation

**Survey of Bittensor subnets and comparable decentralized-AI / noisy-signal markets**
Compiled 2026-08-16. Online sources only. Parameters quoted are from primary sources (repo code, whitepapers, official docs, dated commits) wherever possible; anything not in a source is marked **not documented**.

## How to read this report

Each mechanism is described as: **what it is mechanically** → **what problem it solves** → **evidence it worked/failed** → **source + date**.

Three reliability tiers are used throughout:

- **[CODE]** — read directly from a source repository or on-chain reference. Highest confidence; these are live parameters.
- **[DOC]** — official whitepaper, docs site, or team blog post. Confident about intent, but may lag the code.
- **[3P]** — third-party analyst, wiki, or community post. Directional only; flagged where it conflicts with [CODE]/[DOC].

A recurring caution: several widely-cited subnet "explainers" (subnetalpha.ai, taopedia.org, simplytao.ai, bittensor123.com, subnetradar.com) appear to be **LLM-generated aggregators**. They are useful for orientation and occasionally surface details absent elsewhere, but at least one numeric claim from them directly contradicts the primary whitepaper (see §1.3). I have not treated any of them as authoritative for parameters.

---

# Part 1 — Subnet 9: pretraining (the original model-competition design)

The single most relevant precedent for "pay for the best model under noisy evaluation." Important framing: **SN9 ran a model competition from Oct 2023 to ~June 2025, then abandoned it.** The competition design and the reasons for its abandonment are both documented, which makes it unusually valuable evidence.

## 1.1 Scoring: sampled loss → pairwise wins → win rate

Miners train models offline, upload to Hugging Face, and commit metadata (HF repo + a hash of the model) to the Bittensor chain. Validators download each model and evaluate it against **randomly sampled pages** of a web-scale corpus. [DOC]

> "Miners within this subnet are evaluated based on the number of times the model they have hosted has a lower loss than another model on the network **when randomly sampling from the near infinite Falcon Refined Web pretraining dataset**. To perform well, miners must attain the lowest loss on the largest number of random batches. Finding the best model and delta at the earliest block ensures the most incentive."
> — [macrocosm-os/pretraining README](https://github.com/macrocosm-os/pretraining)

The scoring pipeline is explicitly **pairwise and rank-based, not absolute-score-based**:

1. Compute per-batch loss `L_ij` for model `i` on batch `j` (cross-entropy; perplexity is its exponentiation).
2. For every pair `(a, b)` and every batch `k`, determine a **pairwise win** via `iswin(...)` (below).
3. Sum wins per model → **win rate**.
4. Convert win rate to weights via a softmax with a very low temperature.

This is the key structural choice: **the noisy scalar (loss on a random sample) is never used directly.** It is reduced to a win/loss indicator per batch, then aggregated across many batches and many opponents. Rank aggregation over many noisy comparisons is far more robust than thresholding a single noisy mean — a pattern that recurs in Templar's OpenSkill (§5) and Gradients' paired bootstrap (§4).

## 1.2 The epsilon rule (incumbent advantage) — the mechanism you asked about

Your recollection is correct. The rule is real, and it is implemented as a **timestamp-priority tiebreak with a margin requirement**, where the margin **decays over time**.

The exact function, from the SN9/SN37 validator docs: [CODE]

```python
def iswin(loss_a, loss_b, block_a, block_b, epsilon_func, curr_block):
    loss_a = (1 - epsilon_func(curr_block, block_a)) * loss_a if block_a < block_b else loss_a
    loss_b = (1 - epsilon_func(curr_block, block_b)) * loss_b if block_b < block_a else loss_b
    return loss_a < loss_b
```

The **earlier-submitted** model gets its loss multiplied by `(1 - epsilon)`, i.e. artificially improved. A newer model wins only if it beats the incumbent's *discounted* loss.

The documented rationale is explicitly anti-copying: [DOC]

> "The behaviour of `iswin(...)` intentionally skews the win function to reward models which have been hosted earlier such that newer models are only better than others iff their loss is `epsilon` percent lower... This undermines the obvious optimal strategy for miners to copy the publicly available models from other miners. They **can** and should copy other miners, but they will always obtain fewer wins compared to them until they also decrease their loss by `epsilon`."
> — [finetuning/docs/validator.md](https://github.com/macrocosm-os/finetuning/blob/main/docs/validator.md)

**This is the single most transferable idea in the survey.** Note precisely what it achieves: it does not *detect* copying at all. It makes copying **unprofitable by construction** — a bit-identical copy always ties on loss and always loses on timestamp. There is no classifier to fool, no similarity threshold to evade. Copying is permitted, and rendered pointless unless accompanied by genuine improvement.

### Epsilon implementations [CODE]

From [`taoverse/model/competition/epsilon.py`](https://github.com/macrocosm-os/taoverse/blob/main/src/taoverse/model/competition/epsilon.py) — two classes:

- `FixedEpsilon(epsilon)` — constant.
- `LinearDecay(start_epsilon, end_epsilon, decay_blocks)`:

```python
block_difference   = max(current_block - model_block, 0)
block_adjustment   = min(block_difference / self.decay_blocks, 1)
epsilon_adjustment = block_adjustment * (self.start_epsilon - self.end_epsilon)
return self.start_epsilon - epsilon_adjustment
```

Epsilon is a function of **the incumbent's age**, decaying linearly from `start_epsilon` to `end_epsilon` over `decay_blocks`, then flat. A fresh leader is well protected; a stale leader's protection erodes to near zero. This is exactly the "decaying over time" design you remembered, and it directly targets the hoarding failure mode in §1.5.

### Live parameters

**SN9 pretraining** — [`pretraining/constants/__init__.py`](https://github.com/macrocosm-os/pretraining/blob/main/constants/__init__.py) [CODE]

| Competition | Epsilon function | Start | End | Decay window |
|---|---|---|---|---|
| `B3_MODEL` (3.2–3.4B) | `LinearDecay` | 0.005 (0.5%) | 0.0001 (0.01%) | `7200 * 2` blocks (~2 days) |
| `B14_MODEL` (13.7–13.9B) | `LinearDecay` | 0.005 (0.5%) | 0.0001 (0.01%) | `7200 * 2` blocks (~2 days) |
| `TTS_V0` (350–400M) | `LinearDecay` | 0.005 (0.5%) | 0.0005 (0.05%) | `7200 * 10` blocks (~10 days) |

An earlier `_TMP` constant block in the same file shows the previous schedule (`0.005 → 0.0005` over `7200*7` and `7200*10` blocks), i.e. **the decay was tightened over time** — from ~7–10 days down to ~2 days, and the floor lowered 5×. Directionally, the team reduced incumbent protection substantially.

**SN37 finetuning** — [`finetuning/constants/__init__.py`](https://github.com/macrocosm-os/finetuning/blob/main/constants/__init__.py) [CODE]

| Competition | Epsilon function | Start | End | Decay window |
|---|---|---|---|---|
| `B7_MULTI_CHOICE` | `LinearDecay` | 0.05 (5%) | 0.01 (1%) | `7200 * 5` (~5 days) |
| `INSTRUCT_8B` | `LinearDecay` | 0.05 (5%) | 0.01 (1%) | `7200 * 1` (~1 day) |
| `DISTILLED_REASONING_3B` | `LinearDecay` | 0.05 (5%) | 0.01 (1%) | `7200 * 3` (~3 days) |

**SN37's epsilon is 10× SN9's** (5%→1% vs 0.5%→0.01%). Not explained in any source I found, but the plausible reading is that SN37's eval — multiple-choice accuracy and reference loss on small synthetic samples (120–150 rows/page) — is far noisier per-sample than SN9's perplexity over 15–30 pages of web text, so a wider margin is needed to keep the leaderboard from churning on noise. **The epsilon is being used to absorb evaluation variance, not just to deter copying.** That dual role is worth noting: a single parameter is doing two jobs, and the right value for deterring copying is not necessarily the right value for suppressing noise-driven churn.

## 1.3 Epsilon values: documented history, and one conflict

- **ε = 0.5%** as of the pretraining whitepaper. Rationale given verbatim: [DOC]
  > "the threshold has to be high enough to ensure stability on the subnet — we don't have constant changes to the top-performing model being driven by minuscule changes in model performance. 0.5% is a high enough threshold... The second criteria is that the improvement threshold has to be low enough that miners are incentivised to train many increasingly powerful models."
- **ε = 0.1%** for an experimental parallel competition named **7B\***, introduced **8 August 2024**, "identical to the 7B competition except for a lower epsilon threshold," explicitly framed as an A/B test to find the optimal ε by head-to-head comparison. [DOC]
- **"Dynamic epsilon"** (decay over time) was announced for release "before the end of August 2024," motivated by model hoarding. [DOC] It shipped — `LinearDecay` is the live implementation.

**Conflict flagged:** a third-party page claims the standard 7B competition used **ε = 3% initially** ([subnetalpha.ai](https://subnetalpha.ai/subnet/iota/)) [3P]. The whitepaper says 0.5% at time of writing, and no repo constant I read shows 3%. **I could not verify 3%; treat it as unsupported.** This is a good example of why the aggregator sites should not be trusted for numbers.

Source: [LLM pretraining: The Use-Case Blockchain Has Been Waiting For?](https://www.macrocosmos.ai/research/pretraining_whitepaper.pdf), Macrocosmos (§2.3.3, §3.3.3, §3.4.2, footnote 2). Undated in-document; content places it ~Aug–Sep 2024.

## 1.4 Emission collapse: winner-take-all, stated as a deliberate choice

> "compensation for miners is distributed on a winner-takes-all basis. **The top performing model receives most (95%+) of the miners' emissions** on the subnet, while every other model receives almost nothing." [DOC]

Mechanically this is a low-temperature softmax over win rate — [`pretraining/constants`](https://github.com/macrocosm-os/pretraining/blob/main/constants/__init__.py) [CODE]:

```python
alpha = 0.5        # validator weight moving average term
temperature = 0.01 # "0.01 gives ~96% to best model with only ~3 receiving any weights"
```

The code comment confirms the 95%+ figure. SN37 instead uses a hard cutoff — `ALPHA = 0.90` (EMA) and `MIN_WEIGHT_THRESHOLD = 0.18`, with comments explaining the intent: [CODE]

```python
# Any miners with a combined competition weight below this threshold will instead receive 0 weight.
# This is intended to help vtrust in conjunction with a low alpha by handling the tail ends.
# At 1 eval per 270 blocks, newly winning models will start recieving weight after ~540 blocks.
# Previously winning models will phase out after ~2970 blocks, at which point only the new winner will have weight.
```

Two distinct devices worth separating:

- **A slow EMA on weights** (α = 0.5 SN9 / 0.90 SN37) — smooths handover so a single lucky eval cannot flip the leader instantly. Emission moves over hours/days, not per-epoch.
- **A floor that zeroes the tail** — keeps validators in tight agreement (protecting `vtrust`) rather than each expressing a slightly different noisy ranking of also-rans.

The stated justification for WTA: demand for AI models is concentrated in the leading model, so "intense competition is more valuable than diversification," and WTA "encourages greater professionalism among miners." It was also claimed as an anti-collusion device: [DOC]

> "There is no economy of scale for individual actors who run multiple miners, as only a single model can win at once."

That last claim is the anti-sybil argument for WTA, and it is genuinely sound as far as it goes: farming many hotkeys buys nothing when only rank 1 pays. But note it is an argument against *sybil farming specifically* — and the same concentration that defeats sybils is what produced the hoarding failure below.

## 1.5 Documented failure modes — the most valuable evidence in this report

The SN9 team documented its own exploits, which is rare and worth taking seriously.

### (a) Model hoarding — acknowledged design flaw

> "It can sometimes be the case that the top miner already has an even better pretrained model, but the subnet's current incentive does not encourage them to publish it. **This misalignment between the miner's optimal strategy and the performance of the subnet as a whole reveals a design flaw**, which the rebasing strategy fails to correct by itself. We are implementing a decaying version of the epsilon threshold, which will directly address this issue." [DOC] (whitepaper §3.4.2)

The optimal strategy is stated plainly in §3.3.3: hold **one** model at the top of the leaderboard plus "multiple unsubmitted models at progressively higher performance levels," releasing them only as needed. Under WTA + fixed epsilon, publishing your best model is strictly dominated. The network is starved of exactly the improvements it is paying for.

**This is the central lesson.** WTA plus a static incumbent advantage produces rational withholding. Decaying epsilon is the documented fix — as the leader's protection erodes, the challenger's threshold drops, forcing the leader to publish improvements to stay ahead. The eventual, more thorough answer was to abandon WTA entirely (§2).

### (b) Model inflation ("monsters") and copying ("vampires") — measured, not theorized

The team built a weight-visualization tool ("X-ray") and published what they found in the 14B and 3B competitions: [DOC]

> "After x-raying several models, we found that **many of them were engaging in model-copying**... In some instances, you cannot even see the differences between them... In some scenarios there has been **direct copying with practically zero amendments**. This occurs, in part, on the 14B competition, but you can especially see it in the 3B competition."

And on inflation — padding a smaller model with dead parameters to satisfy a size floor:

> "what's fascinating is some miners appear to be **copying models that are parameter padding, and then further padding their variation**."

Two findings deserve emphasis:

1. **Copying happened at scale despite the epsilon rule.** Epsilon makes copying unprofitable *for the top slot*, but copies still populated the leaderboard. The rule bounds the damage; it does not prevent the behavior. Compare Templar, which needed a *separate* explicit anti-copy mechanism (§5.3).
2. **Parameter-count constraints were gamed via padding.** A min/max parameter band is a trivially satisfiable proxy for "model of size N." SN9's current constants show tight bands (`min_model_parameter_size=13_700_000_000`, `max=13_900_000_000`) [CODE] — necessary, but padding satisfies a band without delivering a real 14B model. **Any structural constraint on a submission will be met in the cheapest way that passes the check.**

Source: [Monsters, vampires, and X-rays: subnet 9's Halloween deep dive](https://macrocosmosai.substack.com/p/monsters-vampires-and-x-rays-subnet), Macrocosmos, ~Oct 2024.

### (c) Anti-collaboration weight obfuscation

SN9 encouraged "rebasing" — building on the current best public model. Miners countered with tricks that preserve inference quality while sabotaging *further training* by others: [DOC]

> "**Vanishing gradients**: Weights that precede normalization layers may be rescaled so that they are orders of magnitude smaller..."

The whitepaper names two such techniques and says monitoring and red-teaming are used to detect them. **No automated detection is documented.** An open-weights requirement does not guarantee a *usable* artifact; a submission can be functional for evaluation yet deliberately hostile to reuse.

### (d) Benchmark contamination — claimed handled, weakly evidenced

The whitepaper argues awareness of Goodhart's Law, avoids known benchmarks (HellaSwag, MMLU-Pro) as reward signals, and uses perplexity on large web corpora instead. On overfitting: [DOC]

> "miners know the dataset from which the evaluation subset comes from. To any machine learning expert, this would be a major red-flag for overfitting. However, our benchmarking shows that models perform well on common benchmarks that the miners are not training on."

The defense is **corpus size** — the eval sample is drawn from a corpus too large to memorize — plus **held-out-benchmark correlation** as the audit. The evidence offered is indirect (transfer to unrewarded benchmarks), and the whitepaper offers no contamination-rate measurement. Reasonable, but not a strong result.

## 1.6 Copy prevention / ownership: what actually exists

| Device | Present? | Detail |
|---|---|---|
| Model hash pre-registration | **Yes** | Miner commits HF repo + model hash to chain; validators verify the download matches. [DOC] |
| Timestamp priority | **Yes** | Block number of the metadata commit; drives `iswin`. [CODE] |
| Commit-reveal of the *model* | **Partial (SN37 only)** | See below. |
| Cryptographic proof of training | **No** | Nothing resembling proof-of-learning. Contrast Gensyn (§8) / Prime Intellect (§7). |

SN37 added two protections absent in SN9: [DOC]

> "The hash for the model is **encrypted with the hotkey of the uploader** to ensure that attackers can't copy commits directly from the chain. Models are also **uploaded to a private repository by default** to ensure that attackers can't monitor Hugging Face repositories for updates."
> — [Fine-tuning, finely tuned: How SN37...](https://macrocosmosai.substack.com/p/fine-tuning-finely-tuned-how-sn37)

This closes a real hole: with a plaintext hash on-chain and a public HF repo, a competitor can watch for a new commit and race a copy. Encrypting the hash to the uploader's hotkey means a copier cannot even confirm what was submitted; the private repo removes the update-monitoring channel. Chain commits are rate-limited to ~20 min/hotkey by Bittensor itself.

Note the tension with §1.5(a): private-by-default protects the miner but weakens the "rebasing"/open-weights ideal that SN9 relied on for compounding progress. **Copy-resistance and open collaboration are in direct conflict**, and SN9/SN37 resolved it in opposite directions.

Rate limits and delays that shape the timing game: [CODE]

- SN9: `SYNC_BLOCK_CADENCE = 150` (~30 min), `EVAL_BLOCK_DELAY = 250`, `model_retry_cadence = 300` (~1 h), `chain_update_cadence = 20 min`.
- SN37: `SYNC_BLOCK_CADENCE = 360`, `eval_block_delay` 460–1600 blocks (~1.5–5 h), `model_retry_cadence = 1200` (~4 h).

`eval_block_delay` is a deliberate quarantine: a model is not evaluated until N blocks after submission, so validators converge on the same view before weights move. It costs latency and buys agreement.

## 1.7 SN9 evaluation detail (for calibrating sample sizes) [CODE]

Current `B14_MODEL` eval mix — six datasets, weighted, `MAX_BATCHES_PER_DATASET = 50`, `BATCH_SIZE = 1`:

| Task | Method | Weight | Pages/eval |
|---|---|---|---|
| FINEWEB | `TEXT_LOSS` | 0.30 | 15 |
| FINEWEB_EDU2 | `TEXT_LOSS` | 0.25 | 15 |
| STACKV2_DEDUP | `TEXT_LOSS` | 0.35 | 30 |
| PES2OX | `TEXT_LOSS` | 0.05 | 2 |
| FINEMATH_3P | `TEXT_LOSS` | 0.03 | 6 |
| INFIWEBMATH_3P | `TEXT_LOSS` | 0.02 | 6 |

Multi-dataset weighting is itself an anti-gaming device: overfitting one corpus is punished by the others. Competitions also carry `NormValidationConstraints` (`norm_eps_soft=200`, `norm_eps_soft_percent_threshold=0.15`, `norm_eps_hard=1000`) [CODE] — a structural sanity check on weight norms, plausibly a countermeasure to the rescaling tricks in §1.5(c), though no source explicitly says so. **My inference, not documented.**

---

# Part 2 — Subnet 9 today: IOTA (WTA explicitly abandoned)

The most direct verdict on the model-competition design comes from the team that ran it. IOTA went to mainnet **2 June 2025**.

> "In August 2024, Bittensor's Subnet 9 (SN9) demonstrated that a distributed network of incentivized, permissionless actors could each pre-train LLMs ranging from 700 million to 14 billion parameters, while surpassing established baselines. While that work validated blockchain-based decentralized pretraining as viable, **it contained core issues: (i) every miner had to fit an entire model locally, and (ii) "winner-takes-all" rewards encouraged model hoarding.**"
> — [IOTA: A Technical Primer for Release](https://arxiv.org/abs/2507.17766), arXiv 2507.17766, July 2025 [DOC]

**The two named failures of the model-competition design are per-miner capital cost and hoarding.** The first is a scaling limit — WTA on whole models means every entrant must fund a full training run, which capped SN9 at ~6 miners per an external review [3P]. The second is the incentive flaw of §1.5(a). Both are consequences of paying only for a finished artifact.

## 2.1 IOTA's mechanism

A single model is split across miners **pipeline-parallel**; activations stream through. Rewards are for *work done*, not for winning:

> "**Granular, continuous incentives** – Validators continually measure each miner's contribution; token emissions are proportional to the work done by each node, rather than the previously utilized winner-takes-all incentive landscape in SN9."

Mechanically:

- **Reward base**: number of **backward passes successfully processed**, `S_m^n` (miner `m`, validation epoch `n`). The paper calls this a "simple linear reward structure... eliminating incentives for throughput manipulation or other gaming strategies during non-validation periods." [DOC]
- **Verification by recomputation + surprise**: "a portion of the miner's training is completely rerun on the validator side. Forward and backwards passes are checked against the submitted miner activations using a **cosine similarity**." Critically: "**miners are not aware of when they are being monitored, preventing them from selectively behaving correctly only during observed intervals.**" [DOC]
- **Temporal decay** `γ`: score is a step function — fixed score for period `γ`, then 0. Stability/agility tradeoff is quantified as `N_scores = γ / T_s`; simulations "predict that we will likely need to synchronise multiple times per hour to ensure that we can have γ < 10h." Higher `γ` = stabler weights but longer immunity needed and slower removal of bad miners. [DOC]
- **Butterfly All-Reduce** as collusion detector: redundant shard assignment means "every miner 'shares' one shard with every single other miner... making it trivial to detect cheating miners." Miners don't know the global split, which "prevents collusion between cabalistic miners." An agreement matrix for 50 miners shows malicious actors out of consensus. [DOC]

## 2.2 CLASP (Shapley-value attribution) — designed, **not shipped**

> "**CLASP is not included in the initial release**; it remains an active area of research and is intended for integration into the incentive mechanism once the system stabilizes post-launch." [DOC]

Worth understanding as a design, since it targets credit assignment under interdependence. Samples are routed through random pathways; the orchestrator records per-sample losses and the pathway taken; validators use loss-and-pathway records to estimate each miner's marginal contribution — a sampling approximation of a Shapley value. Bad actors show "abnormally high per-sample losses" and are flagged by z-score-like normalization.

Reported evidence is a **toy simulation only**: 5 layers × 5 miners, loss ~N(4.5, 0.2), malicious miners assumed to raise loss and stdev by 10%. Under those assumptions bad actors are outliers. An interesting side effect is noted: honest miners in the same layer as bad actors see reduced attribution ("intrinsic balancing"), which sharpens outlier sensitivity but also means **collateral damage to innocent neighbors**.

Stated assumptions, both load-bearing: per-sample loss measurements are accurate and tamper-proof; and miners cannot coordinate across layers. The mitigation proposed for the first is top-`k` compressed logits so validators can recompute exact losses.

**No production results for CLASP were found.** Any claim that SN9 currently pays via Shapley values is unsupported — several [3P] aggregators state or imply this and are wrong as of the primary source.

## 2.3 Status

IOTA is live; SN9 alpha ~$30M market cap ([CoinGecko](https://www.coingecko.com/en/categories/bittensor-subnets), 2026-08). A [3P] search summary mentioned a 16B-parameter distributed run tested July 2026 — **I could not verify this from a primary source and do not rely on it.**

---

# Part 3 — Subnet 37 (Taoverse/Macrocosmos finetuning): the competition framework

Same codebase lineage as SN9 (both depend on `taoverse`), so `iswin`/epsilon/WTA all carry over. What SN37 adds is the **competition framework** and **multi-task evaluation**.

## 3.1 Competition schedule as a first-class object [CODE]

`COMPETITION_SCHEDULE_BY_BLOCK` is a list of `(block, [Competition])`. Each `Competition` carries a `reward_percentage` (asserted to sum to 1.0) and a list of `EvalTask`s (weights also asserted to sum to 1.0). Constraints are pinned per competition: `max/min_model_parameter_size`, `sequence_length`, `allowed_architectures` (explicit list of HF classes), `tokenizer`, `max_bytes`, `eval_block_delay`, `epsilon_func`.

Notable: **competitions have scheduled sunsets written into code**, e.g. `SUNSET_B7_BLOCK = 4_675_163` and `SUNSET_INSTRUCT_8B_BLOCK = 5_158_632` (annotated "23:59 GMT+0 on Tuesday, March 18, 2025"). Emission splits shift at a future block, announced in advance. This is a clean way to retire a saturated or contaminated task without an emergency intervention — the deprecation is pre-committed and publicly auditable.

Also notable: `tokenizer="Xenova/gpt-4"` is **forced** in most competitions, while `INSTRUCT_8B` sets `tokenizer=None` with the comment "Any tokenizer can be used." Forcing a shared tokenizer makes cross-model loss comparisons apples-to-apples — otherwise loss is not comparable across submissions at all. **A necessary condition for loss-based ranking that is easy to overlook.**

## 3.2 Multi-task evaluation with normalization [CODE]

The `DISTILLED_REASONING_3B` competition (current as of the constants file):

| Task | Method | Dataset | Weight | Sample config |
|---|---|---|---|---|
| SYNTHETIC_1_SFT | `REFERENCE_LOSS` | `SYNTHETIC_1_SFT` | 0.5 | 1 page, 150 rows, `target_size: 120` |
| CODEFORCES_COTS | `REFERENCE_LOSS` | `CODEFORCES_COTS` | 0.5 | 1 page, 120 rows |

The earlier `B7_MULTI_CHOICE` mix shows the range of eval methods: `MULTIPLE_CHOICE` on `SYNTHETIC_MMLU` (0.75), `REFERENCE_LOSS` on `WORD_SORTING` with `INVERSE_EXPONENTIAL` normalization (`ceiling: 40.0`), `TEXT_LOSS` on `FINEWEB` (`ceiling: 20.0`), and `IF_EVAL` (instruction-following) at 0.1.

`NormalizationId.INVERSE_EXPONENTIAL` with a `ceiling` is a mechanism worth flagging: it maps an unbounded loss onto a bounded score, so a catastrophic score on one sub-task cannot dominate the weighted sum. Without it, weights across heterogeneous metrics are meaningless. **Note the sample sizes: 120–150 examples.** That is small, and explains why SN37's epsilon is 10× SN9's.

## 3.3 Evaluation-data integrity — the attack surface SN37 focused on

SN37 sources eval data from other subnets (originally SN18 synthetic, now SN1/prompting), which creates an obvious injection vector: if a miner can influence the eval data, they win without a better model.

> "The criteria above **protect against all known vulnerabilities that miners could exploit to inject favorable/malicious data into the fine-tuning validator's validation dataset.** The above criteria are our contributions to the design of the fine-tuning subnet and are improvements to both its security and overall quality compared to the previous implementation of fine-tuning in Bittensor." [DOC]

The last clause refers to the **prior Bittensor finetuning subnet (SN6, Nous Research)** — so the eval-integrity design was a response to weaknesses in an earlier attempt. Concrete devices visible in code/docs: [CODE]

- `SAMPLE_VALI_MIN_STAKE = 10_000` — only sufficiently-staked validators can serve sample data.
- `PROMPTING_MAX_AGE = timedelta(hours=4)` — reject stale eval data.
- `MIN_ALLOWED_SAMPLES = 50` — refuse to score a task on too few samples.
- `NUM_CONFIGS_TO_SAMPLE = 10`.
- **"Hash-based synchronization ensures secure and fair evaluation"** (README) — validators agree on which data to use via hashes, so a miner cannot get a favorable draw from a single validator.
- `WEIGHT_SYNC_VALI_MIN_STAKE = 100_000`, `WEIGHT_SYNC_MINER_MIN_PERCENT = 0.01`, `scan_top_model_cadence = 30 min` — validators cross-check which models other large validators are weighting, a peer-audit against a single validator being fooled.

**The generalizable point: when eval data comes from anywhere a competitor can influence, data provenance becomes the primary attack surface** — ahead of the scoring rule itself.

## 3.4 Model retention [CODE]

```python
sample_min = 3   # SN37   (SN9: 5)
updated_models_limit = sample_min * len(MODEL_CONSTRAINTS_BY_COMPETITION_ID) + 7
```

A bounded number of models are held for evaluation at once; `sample_min` per competition are retained into the next loop. This is a *compute-budget* constraint on the validator (you cannot re-evaluate every historical model forever), and it interacts with fairness: a model that isn't held isn't scored.

## 3.5 Status and the WTA verdict, again

An independent analysis [3P] states SN37 momentum slowed, attributing it partly to the mechanism:

> "the incentive mechanism remains based on a 'winner-takes-all' design, **which is less effective than the collaborative systems recently adopted** (notably on IOTA, Subnet 9)."
> — [Macrocosmos: Bittensor's decentralized OpenAI?](https://oakresearch.io/en/analyses/fundamentals/macrocosmos-bittensor-decentralized-open-ai), OAK Research

Third-party judgment, but consistent with the primary-source reasoning in §2. SN37 is now labeled "Aurelius" on-chain and remains active (~$3–4.6M cap). I found **no postmortem of a specific "model theft incident" on SN37**; the copying evidence is SN9's X-ray post (§1.5b). If a documented SN37 theft incident exists, I did not find it — the closest documented exploits in the ecosystem are on other subnets (§6.4).

## 3.6 Nous Research / SN6 — could not substantiate

SN6 was the earlier Nous finetuning subnet ([NousResearch/finetuning-subnet](https://github.com/NousResearch/finetuning-subnet)), mechanically similar (train → HF → validators score → Yuma), using SN18 data [3P]. SN37's docs imply its eval-integrity design fixed vulnerabilities in that implementation, but **no specific SN6 exploit writeup was found.** Nous's current decentralized-training work is **Psyche** — coordinated on **Solana**, not Bittensor — using DisTrO/DeMo, with clients that train, verify (recompute), and witness. Psyche is a *training coordination* system, not a model competition, so it is out of scope for incumbent/challenger design. ([nousresearch.com/nous-psyche](https://nousresearch.com/nous-psyche/), May 2025.) Note: **netuid 6 today is a forecasting subnet, not finetuning** [3P] — subnet IDs get recycled, so ID-based references age badly.

---

# Part 4 — Gradients / G.O.D (SN56, Rayon Labs): the most sophisticated design found

**This is the highest-value section of the report.** SN56 has, as of mid-2026, the most carefully engineered answer to "how does a challenger displace an incumbent when the measurement is noisy" that I found anywhere — including outside crypto. Parameters below are read from live code with dated commits.

## 4.1 The fundamental design shift: submit *code*, not *models*

In July 2025 SN56 moved from "miners train on their own hardware and return a model" to **miners submit a training repository + exact commit SHA; validators execute it**. [DOC]

> "Miners register a FastAPI endpoint and submit links to their training repositories (with exact commit SHAs) for each tournament type; Validators run a tournament lifecycle...; Trainers execute the validated code in isolated Docker containers with assigned GPUs, mounted cache volumes, and standardized runtime arguments, then upload results to Hugging Face for evaluation."
> — [bittensor.ai/subnets/56](https://bittensor.ai/subnets/56)

The subnet owner's stated motivation was **client trust** (customers didn't want data going to anonymous miners), but the incentive-design consequence is larger:

- **Compute is equalized.** All entrants run on identical validator-provided GPUs, so you are ranking *methods*, not budgets. This directly addresses SN9's capital-cost problem (§2).
- **The artifact is reproducible.** A pinned commit re-run on known hardware is auditable in a way an uploaded weight file never is.
- **Containers have no internet access** [DOC] — closing data-exfiltration and eval-lookup channels.
- **The winning script is published open-source** each tournament, "resetting the baseline for everyone" [3P].

The tradeoff, noted by an external reviewer: publishing the winning recipe "hand[s] rivals and centralized incumbents the latest best recipe for free" [3P] ([SubnetRadar](https://subnetradar.com/research/subnets/56)). Same open-vs-copy-resistance tension as §1.6, resolved toward openness — but here it is safe *because* epsilon-like champion protection exists (§4.3), and because everyone re-runs on equal hardware.

## 4.2 Tournament structure [CODE]

From [`validator/tournament/constants.py`](https://github.com/gradients-ai/G.O.D/blob/main/validator/tournament/constants.py):

- `TOURNAMENT_INTERVAL_HOURS = 120` (5 days); three independent tracks (text, image, environment) on a weekly schedule.
- Field-size-dependent brackets: `SMALL_TOURNAMENT_MIN_PARTICIPANTS = 3`, `SMALL_TOURNAMENT_MAX_PARTICIPANTS = 14`, `SMALL_TOURNAMENT_GROUP_TASKS = 3`, `SMALL_TOURNAMENT_ADVANCE = 2`; `MAX_NUMBER_OF_MINERS_FOR_KNOCKOUT_ROUND = 8`; `EXPECTED_GROUP_SIZE = 32`.
- Environment (RL) tournaments use PvP: `MAX_ENVIRONMENT_GROUP_SIZE = 5` with the comment "With 5 members, a group evaluates at most C(5, 2) = 10 PvP pairs."
- **Paid entry**: `TOURNAMENT_TEXT_PARTICIPATION_FEE_RAO = 350_000_000` (0.35 TAO), environment 0.3 TAO, image 0.2 TAO. A [dated commit](https://github.com/gradients-ai/G.O.D/commit/c83bc13a6087121e176c600ee666e90d33706a71) (2026-06-19) raised text/env 0.20→0.25 and image 0.15→0.20; current code shows higher still. **Entry fees are a direct anti-sybil / anti-spam device** — each additional hotkey costs real TAO, so submission farming has a per-attempt price.
- `MAX_TRAINING_ATTEMPTS = 2`; `PERCENTAGE_OF_TASKS_SHOULD_BE_SUCCESS = 0.5` (round sanity check).

## 4.3 The incumbent/challenger rule — a statistically principled epsilon

The **boss round**: the bracket winner does not become champion; they must beat the *defending champion*. The evolution of this rule is documented in commits and is instructive.

**Earlier design** ([commit a0da6a2](https://github.com/gradients-ai/G.O.D/commit/a0da6a2687d6bc5407ee9a0c23104bd4bf357827)): per-task-type win requirements — text by majority (2/3 or 3/3), **image requires a perfect 3/3 sweep** — plus "progressive threshold system with exponential decay based on consecutive wins," where "the advantage required to dethrone a champion decreases with each successful defense using an exponential decay formula." That is structurally the same idea as SN9's decaying epsilon: protection erodes with tenure.

**Commit c83bc13** (2026-06-19) then changed direction:

> - Base track weights -> text 0.20, image 0.15, env 0.15 (sum 0.50), so the base pool is always 50% of subnet emission.
> - Burn bar (EMISSION_MULTIPLIER_THRESHOLD) 0.05 -> 0.10: **a champion must beat the boss by 10%+ to earn above the base floor and reduce burn.**
> - **Champion time decay halved: 0.33%/day -> 0.165%/day.**
> - **Progressive per-task dethrone thresholds disabled** (EXPONENTIAL_*_THRESHOLD -> 0.0); boss-round tasks are now straight score comparisons.
> - Text/image dethrone rule: simple majority -> **comprehensive victory, i.e. lose at most one boss-round task (5/6)**.

**Current live rule** — this is the part worth studying closely. [CODE]

```python
BOSS_ROUND_WIN_MARGIN = 0.01              # relative margin (GRPO/env only)
BOSS_ROUND_TIE_DEADZONE_NATS = 0.01
BOSS_ROUND_MIN_WIN_RATE = 0.55
BOSS_ROUND_MIN_MEAN_GAP_NATS = 0.01
BOSS_ROUND_BOOTSTRAP_CONFIDENCE = 0.99
BOSS_ROUND_BOOTSTRAP_RESAMPLES = 10_000
BOSS_ROUND_BOOTSTRAP_SEED = 20260808      # "Fixed so two validators scoring the same boss round always agree."
PAIRED_BOSS_ROUND_TASK_TYPES = (INSTRUCTTEXTTASK, DPOTASK, CHATTASK)
```

The in-code comments are the clearest statement of the noisy-evaluation problem I found in any source, and they explain *why a relative margin is wrong*:

> "A relative margin is the wrong scale for a log-likelihood loss: a difference of D nats is D nats of evidence whether the loss is 0.02 or 2.0, so `abs(boss) * 1%` collapses to nothing exactly where the task is most saturated. Worse, **a scalar mean carries no information about its own uncertainty, so no threshold on it can separate a real win from held-out sampling noise.**
> Instead both models are scored on the identical held-out set and **compared example by example. Example difficulty dominates the variance in both losses and cancels in the pairing.**"

So the mechanism is a **paired per-example comparison with a bootstrap confidence bound**:

1. Score champion and challenger on the *identical* held-out set.
2. Compare **per example**. Pairing cancels example-difficulty variance — the dominant noise term. (This is a paired test, not an unpaired one, and that choice is the crux.)
3. Ignore differences below a **dead zone** of 0.01 nats: "0.01 nats ~= 1% more probability assigned to it — trivial. **Without a dead zone the win count is decided at the 5th decimal.**"
4. Require the challenger to win **≥55% of *decided* examples**, evaluated at a **one-sided bootstrap lower bound** (99%, 10k resamples) — "not a noise threshold: at ~1000 examples the standard error on a win rate is ~1.6%, and the bootstrap below is what handles noise. This is the policy statement of how much more *consistent* a challenger has to be."
5. **And** require mean gap ≥ 0.01 nats, "so it cannot win on a majority of hairline examples while being materially worse where it loses."
6. Require a minimum count of decided examples; saturated tasks fail this naturally. "at exactly this many decided examples the standard error is ~5%, so a bare 65% will fail the bound and a much larger observed margin is needed — **the gate is automatically stricter when it has less to go on.**"

Two design notes with unusually good reasoning attached:

- **Why 55% and not higher** — "A genuinely better model with a wide per-example spread can sit near 55% and still be the better model; **demanding much more of it selects for low-variance submissions rather than good ones.**" A high win-rate bar is not a free safety margin; it biases toward a different objective.
- **Empirical calibration** — "Calibrated against **62 boss-round text tasks over 120 days, where the median separation between the two competitors was 0.0094 nats**: at 0.02 only a quarter of matchups were close enough to even be winnable, which compounded over 4-of-5 tasks makes the crown near-permanent."

That last sentence is the key documented failure-and-fix in this whole report: **a threshold set too high, compounded across multiple required task wins, makes the incumbent effectively permanent.** They measured the real separation distribution, found the bar was above the median, and lowered it. They also record the observed win rate on every task, won or lost, so the threshold can be recalibrated against data rather than intuition (`round_results.py`, per comments) — the parameter is explicitly marked "Provisional."

A separate comment documents a concrete false-negative that motivated the mean-gap floor being set equal to the dead zone: a second larger threshold "is what made a model better by 0.011 nats on ALL 800 samples - 100% win rate - lose the task."

**Champion decay**: `0.165%/day` [CODE commit] — the incumbent's advantage erodes with tenure, same principle as SN9's `LinearDecay` but applied to emission rather than to the comparison threshold.

**Emission shape**: base pool floored at 50% of subnet emission across tracks; a champion must beat the boss by ≥10% to earn *above* the floor and reduce burn. So there are two distinct bars — one to *hold the crown*, a higher one to *earn premium emission*. Excess is burned rather than paid out. `RUNNER_UP_EMISSION_DAYS = 7`.

## 4.4 Anti-gaming devices in SN56 [CODE]

- `MAX_NEAR_DUPLICATE_RATE = 0.20` — "Reject a task whose dataset near-duplicate rate (from baseline_stats) is at or above this fraction." **Direct, quantified contamination control on the eval set itself.** The most concrete anti-contamination parameter found in this survey.
- `OBFUSCATION_DETECTION_PATH = "./validator/tournament/obfuscation_detection/anti_obfuscation"` — a dedicated module for detecting obfuscated submitted code (recall SN9's weight-obfuscation problem, §1.5c).
- **LLM-based submission dedup**: `TOURN_DEDUP_ENABLED = True`, `TOURN_DEDUP_CLAUDE_MODEL = "claude-opus-4-8"`, budget `$15`, `TOURN_DEDUP_CONCURRENCY = 8`; plus `CLAUDE_REPO_DIFF_MODEL = "claude-sonnet-4-5"` for repo-diff review. **They pay an LLM to detect copy-and-tweak submissions** — a live, deployed answer to the copy-detection problem that epsilon rules only partially solve.
- `trust_remote_code` **off** during eval, which constrains seed models to standard architectures — a sandboxing constraint that also limits architecture novelty.
- Boss-round tasks drawn from a **historical window** (`BOSS_ROUND_HISTORICAL_START_DATE = 2025-06-01` … `END = 2025-08-01`, `MIN_SUCCESSFUL_SCORES_FOR_HISTORICAL_TASK = 2`) — held-out tasks with known-good baselines.
- **Forced models** remove hardware/model-choice confounds: `PRE_BOSS_MODEL = "Qwen/Qwen3-32B"`, `ENV_TARGET_TOURN_MODEL = "Qwen/Qwen2.5-7B-Instruct"`, `CONTINUOUS_SFT_LINEAGES = {"qwen": "Qwen/Qwen3-8B-Base"}`, `FORCED_BOSS_ENVIRONMENT = SWE_INFINITE`.
- `BOSS_ROUND_BOOTSTRAP_SEED = 20260808` with "**Never vary this**" — determinism so independent validators reach identical verdicts. A subtle but essential detail: any randomized scoring procedure in a multi-validator system must be seeded, or validators disagree and consensus punishes honest ones.
- **Continuous-SFT lineages** — each round trains the next data chunk from the *previous winner's* checkpoint, carrying a lineage forward across tournaments, with the strictest gate ("challenger must win EVERY continuous-SFT task"). This is SN9's "rebasing" idea done properly: cumulative progress on a shared artifact, with the champion's work as the mandatory starting point.

## 4.5 Reported performance and structural risk

Rayon claims 180 controlled experiments vs HuggingFace AutoTrain, TogetherAI, Google Vertex AI, Databricks, with win rates reported ~82–100% depending on competitor, measured as loss on held-out test data [3P] ([SimplyTao](https://simplytao.ai/blog/your-simple-guide-to-gradients-sn56)). **Vendor-originated claim, not independently verified.**

Structural weakness, per external review [3P]: "A single main validator is the design's weak point, partly offset by independent auditors who can recompute seven days of weights." Equalized compute requires a trusted compute provider — the fairness gain is paid for with centralization. Auditor recomputation is the mitigation.

---

# Part 5 — Templar (SN3): Gauntlet — rank ratings + explicit anti-copy proof

Templar is a *collaborative* training subnet (miners submit pseudo-gradients, not models), but its incentive design solves the noisy-scoring problem in a way that transfers directly.

Source: **[Incentivizing Permissionless Distributed Learning of LLMs](https://arxiv.org/abs/2505.21684)**, arXiv 2505.21684, May 2025 [DOC]; docs at [docs.tplr.ai/incentive-design](https://docs.tplr.ai/incentive-design/).

## 5.1 LossScore → OpenSkill rating: the explicit noise fix

The primary score is the change in loss from applying one peer's pseudo-gradient: `s_i = L_before - L_after`.

The paper is unusually candid that **this raw score is too noisy to pay on**:

> "A significant issue with loss-based scores is that they are **not consistent over time**; indeed, even adjacent iterates can lead to very different scores for the same peer running the same strategy. This problem is exacerbated by the fact that practically, **the validator cannot evaluate all peers' contributions at each communication round**. On the other hand, we observed that at any given round, **ranking based on LossScores correlated well with high quality contributions**... We thus utilize a rank-based rating system **OpenSkill**, which is well suited to estimating relative peer ranks under sparse evaluation."

**The pattern: the noisy absolute measurement is discarded; only the ranking is kept; ranks accumulate into a skill rating over time.** Each round a random subset `S` of peers is ranked by LossScore and their OpenSkill (PlackettLuce) ratings updated. This is the same insight as SN9's pairwise wins and SN56's paired comparison, but with an explicit Bayesian skill model that handles **sparse, partial** evaluation — you never need to evaluate everyone in the same round.

Reported evidence (Figure 2): simulating three peers (one processing more data, one desynchronized, one baseline), "the loss score is highly variable from step to step, however relative performance is consistent and the loss rating can quickly differentiate between peers." [DOC]

Also note `β_t = c * α_t` with `c < 1` — the evaluation step size is deliberately smaller than the learning rate, because "stepping with too large a step size is more likely to lead to negative loss scores, and in our empirical observations inconsistent rankings between peers." **Measurement hyperparameters were tuned to reduce ranking noise.**

## 5.2 Superlinear reward to deter sybils [DOC]

Normalized incentives use exponent `c = 2`:

> "we use `c=2`, with the goal to increase competition amongst peers. Indeed, the **non-linear incentive is designed to encourage participants to register fewer high-performing peers versus many weaker peers.** For example if a user has access to 10 GPUs it is preferred they take care of optimizing their configuration to produce a single high quality pseudo-gradient with all 10 GPU as opposed to registering 10 individual peers."

**A convex reward curve is an anti-sybil device** — it makes concentration strictly better than splitting. Same objective as SN9's WTA anti-sybil argument (§1.4), but tunable via one exponent rather than an all-or-nothing collapse. Aggregation uses top-`G` peers with weight `1/G` (`G = 15` in the live run).

## 5.3 Proof of Computation — explicit anti-copy, named as a threat [DOC]

Templar names peer copying as a distinct attack:

> "**Peer Copying** - A peer attempts to copy a valid pseudo-gradient uploaded by another peer and post it before the communication period is completed."

Countermeasures include a **put window** (submissions outside a short publication window are ignored — a timing constraint that denies the copier a window to observe-then-submit), a **Sync Score** heuristic measuring how many update steps a peer's model diverges from the validator's (**threshold = 3** in practice), and a penalty that "rapidly degrade[s] the score when a peer repeatedly fails the fast evaluation," removing them from aggregation quickly.

Two-phase evaluation is itself a cost-management pattern: (a) an expensive primary evaluation on a few peers per round; (b) a cheap fast check (uptime, sync, basic validity) on many peers per round. **Cheap checks catch obvious cheating at scale; expensive checks establish quality on a sample.**

## 5.4 Byzantine tolerance — honest about limits [DOC]

The paper concedes residual vulnerability: "(a) peers whose malicious behavior is not detected by the incentive mechanism, and (b) **a single bad value sent before the peer can be downweighted**" — e.g. one enormous-magnitude pseudo-gradient that disrupts aggregation if included even once. Mitigations: sign-based aggregation, and normalizing contributions in the DCT-encoded domain so each peer contributes equally. Reported to "significantly reduce the impact of byzantine peers while having no impact on convergence in the fully cooperative (simulated) setting."

**A detection-and-downweight loop always has a one-shot window before it reacts.** If a single submission can cause irreversible damage, detection speed is not a sufficient defense; the aggregation rule must be bounded a priori.

## 5.5 Live result [DOC]

1.2B model, **20K communication rounds**, permissionless, FineWebEdu, `G = 15` aggregated per round, **5 peers evaluated per validator per round**, ~400K tokens/peer/iteration target. Downstream metrics reported "competitive" vs an AdamW baseline and vs the DeMo paper. Claimed as "the first truly permissionless pre-training LLM run." Templar received ~4% of daily emissions as of early Sept 2025 ([Galaxy Research](https://www.galaxy.com/insights/research/decentralized-ai-training)).

---

# Part 6 — Yuma consensus, weight copying, and why WTA makes it worse

This is the layer *beneath* every subnet mechanism, and it has the best-documented exploit in the entire ecosystem.

## 6.1 The problem [DOC]

Source: **[Weight Copying in Bittensor](https://blog.bittensor.com/weight-copying-in-bittensor-422585ab8fa5)**, Opentensor Foundation, **29 May 2024** (accompanying Bittensor 7.0.1 / testchain 1.1.0 and a formal technical paper).

Yuma Consensus rewards validator *agreement*. Weights are public on-chain. So a validator can skip all evaluation work and submit the previous epoch's consensus:

> "Because of the consensus algorithm's emphasis on rewarding agreement among the subnet validators, this leads to the perverse outcome that **subnet validators who copy the average or median of the weights will receive higher rewards than weight-originators. We refer to this as copier advantage.**"

Two harms are named: resources diverted from real validation, and the copier "occupying a slot that could otherwise go to a weight-originating subnet validator." The article grounds this in the wisdom-of-crowds literature (Galton's ox, 1906) — aggregation only works if participants are **independent**, and copying destroys independence.

**Interaction with winner-take-all — the point you asked about.** WTA makes weight-copying dramatically easier, and the mechanism is worth stating precisely: under WTA the weight vector is nearly one-hot and *low-entropy*. A copier needs only to identify the current leader — one integer — rather than reproduce a full ranking. The signal is trivially inferable from public emission data, and it changes rarely (that stability is the *point* of epsilon/EMA smoothing). So the very devices that make a WTA leaderboard stable against noise also make it cheap to copy: **stability of the target is what the copier needs.** Conversely, the Opentensor analysis notes commit-reveal is most effective where there is high miner turnover and frequent rank changes — precisely what WTA + epsilon + weight EMA suppress. I did not find a source stating this tension explicitly, so **this synthesis is my inference**, but it follows directly from both documented mechanisms.

Confirming the practical severity, from the SN2 case study (§6.4): "Despite leveraging commit reveal, weight copiers had amassed a stake weight of **1-kappa**, which means consensus was being formed by non-productive validators."

## 6.2 Mitigation 1 — Commit-Reveal (CR3/CRV3) [DOC/CODE]

**Original (2024)**: validators submit an encrypted hash instead of plaintext weights; automatic decryption after a configurable number of blocks. Copiers get only stale weights.

**CRV3 (current)** uses **timelock encryption (TLE) targeting a future drand randomness round**; the runtime auto-decrypts when the drand pulse arrives. Nobody — *including the submitting validator* — can reveal early. Enabled per-subnet via `CommitRevealWeightsEnabled`; extrinsics `commit_crv3_mechanism_weights` / reveal. Introduced in runtime `v360`.
Sources: [subtensor.com/reference/mechanisms/commit-reveal](https://subtensor.com/reference/mechanisms/commit-reveal), [learnbittensor.org/concepts/scoring-rewards/commit-reveal).

**Measured effectiveness (2024 analysis, Figures 1–3):** relative dividend rate `G` of a copier vs the median validator, as reveal delay increases. Documented findings:

- Across **30 subnets**, **20 could be pushed below `G = 1`** by a long enough interval; **10 could not.**
- `G` **converges at a commit-reveal interval of ~1800 blocks (5 tempos)**; "A longer commit reveal weights interval value yields little benefit, but slows down the evaluation of miners."
- Copying was "very profitable" on subnet 30; only reached `G ≈ 0.95` on subnets 10, 16, 24, 31, 32.
- **`G` is not always monotonically decreasing** in the interval — "There is no strict forward explanation for such an event."

The design target is explicitly economic, not absolute: reduce copier reward below the **18% validator take** on delegated stake, so copying becomes worse than simply nominating. Also recommended: **increase the immunity period by the same number of blocks as the reveal delay**, since new miners cannot be scored while weights are concealed.

**A third of subnets could not be protected by commit-reveal at any delay.** That is the headline number, and it means commit-reveal alone is insufficient by the designers' own measurement.

## 6.3 Mitigation 2 — Liquid Alpha [DOC/CODE]

Bonds between validators and miners update by EMA: `B(t) = alpha * ΔB + (1 - alpha) * B(t-1)`, where by default `alpha = 1 - bonds_moving_average/1e6` (default 900,000 → **alpha = 0.1**). Validators who bond to good miners *early* earn more when consensus catches up: `d_i = Σ_j B_ij × I_j`.

Liquid alpha makes `alpha` **dynamic per validator-miner pair**, via a sigmoid on distance from consensus:

```
diff_buy      = clamp(weight[i][j] - consensus[j], 0, 1)
diff_sell     = clamp(bond[i][j] - weight[i][j], 0, 1)
combined_diff = diff_buy if weight >= bond else diff_sell
sigmoid_value = 1 / (1 + exp(steepness / -100 * (combined_diff - 0.5)))
alpha[i][j]   = alpha_low + sigmoid_value * (alpha_high - alpha_low)
```

Documented parameters: `alpha_low` default **0.7**, `alpha_high` default **0.9** (on-chain constants `alphaLow = 45875`, `alphaHigh = 58982` normalized against u16 max), `AlphaSigmoidSteepness` default **1000**. Requires **`yuma3_enabled`** — "Liquid alpha only takes effect when `yuma3_enabled` is also on; the classic bond path ignores the toggle entirely." Consensus is a stake-weighted median with `kappa` default `32767/65535 ≈ 0.5`; weights above consensus are clipped down.
Sources: [bittensor.com/docs/concepts/emissions](https://www.bittensor.com/docs/concepts/emissions), [subtensor.com/reference/mechanisms/liquid-alpha](https://subtensor.com/reference/mechanisms/liquid-alpha).

Documented worked example: consensus 0.5 for miner M; validator V sets 0.5 → distance 0 → alpha ≈ 0.90; validator W sets 0.9 → distance 0.4 → alpha ≈ 0.70; "V builds bonds ~1.3× faster than W." Since copiers start from zero bonds and lag by a reveal interval, slow bond accumulation compounds their disadvantage.

## 6.4 Documented incident: CR3 was cryptographically broken (Aug 2025)

**The best-documented gaming incident in the ecosystem**, and it is a validator-side one.

Source: **[Battle-testing Yuma3](https://inference-labs.medium.com/battle-testing-yuma3-a8136c797d1f)**, Inference Labs (SN2) with Rhef (SN12), **14 August 2025**.

Timeline and findings:

- Motivation: on SN2, "despite leveraging commit reveal, weight copiers had amassed a stake weight of 1-kappa."
- **9 June 2025** — SN2 v8.2.0 deployed Yuma3 + a scheme splitting miners into 8 groups, with 1/8 triggering **bonds reset** each epoch so honest validators would always be "one step ahead" in discovering re-scored miners. The bonds-reset flag turned out not to be enableable; they PR'd it as a hyperparameter.
- **7 July 2025** — bonds reset shipped. Two copiers dropped below honest validators; **the top copier still gained ground.**
- They switched from `uid % 8` to **pseudorandom block-hash-based grouping**, to defeat a copier suspected of using an LSTM to predict weights. The top copier *still* submitted near-perfect weights.
- Root cause, found by timeline-instrumenting commits, reveals and drand pulses: **the encryption key protecting weight payloads was not secure**, so "all weights could be decrypted not just before the epoch, but **even before the commit transaction was included in a block**." Two additional bugs: `bittensor-drand` selected the reveal round too early, and weights could be revealed in the same epoch they were committed.
- Verified by writing a decryption script against fresh commits (succeeded), then patching `bittensor-drand` with proper randomization (`AES-GCM decryption failed` — confirming the fix).
- After the patch, "the leading copier was forced to fall back to... conventional copying methods... and lagged behind in terms of dividends as a result."

**Fixes shipped in Bittensor 9.9.0 / Subtensor 3.2.4** — four changes:

1. Randomization of commit encryption keys, eliminating standard (non-timelock) decryption.
2. Epoch-calculation fix so the earliest possible reveal is within the target reveal epoch.
3. **Commit messages now include the submitting validator's hotkey, cross-checked by subtensor against the extrinsic caller** — "to ensure actors cannot blindly copy their peers' commits and submit them to the chain as their own."
4. Reveal-buffer logic adjusted.

Lessons, stated plainly:

- **A commit-reveal scheme is only as strong as its key management.** For ~14 months the documented mitigation was cryptographically bypassable, and the ecosystem believed it was protected. Fix #3 also shows the *naive* attack (replaying someone else's ciphertext as your own) was possible.
- Months of mechanism-level countermeasures (bonds reset, pseudorandom grouping, weight variation) **failed to fix what was actually a cryptographic bug.** When a mechanism underperforms, the implementation is a candidate cause before the design is.
- Inference Labs has proposed **BIT-0002** and **BIT-0008** — validators run zero-knowledge circuits attesting to proper validation, i.e. *proving* work was done rather than inferring it from agreement.

## 6.5 Other documented gaming incidents

Quality varies; these are [3P] community/analyst reports, not vendor postmortems. I include them because they are the only documented incidents I could find, but they should be read as allegations with supporting detail rather than confirmed findings.

**Subnet 33 — validator-mimicry and key monopoly.** ([Bittensor_player, Medium](https://medium.com/@bittensor_player/bittensor-subnet-33-from-innovation-to-exploitation-4aae796a5f6c)) The dominant miner, "leveraging their large number of keys, can sequentially collect all the material processed by the validator and generate an optimal response by **mimicking the validator's results — without actually performing the intended task**." An open pre-processing database allowed full-conversation access; "The proof? Identical results across all conversation windows." Reported 19 Dec 2024; author claims 3 months of denial before the database was closed. Also describes a **key-buying race** (attempting to purchase 80 keys during the immunity window) — concrete multi-hotkey farming, and evidence that *immunity period length* is itself a contested lever.

**Subnet 44 — exploit history and centralized challenge control.** ([Zane Merritt, Medium](https://zanemerritt.medium.com/bittensor-subnet-44-sports-ai-goldmine-or-insiders-playground-a-deep-investigation-d855e3fd07d7)) Lists patched exploits: "Static keypoints trick where you submit the same keypoints every time. Class-swap exploit for mislabeling objects to game scoring. Scoreboard spam where miners detect UI elements as players. **Prefabricated responses where you don't actually process videos.**" Validators fetch challenges from the subnet owner's own backend (`/api/tasks/next/v2`) — "centralized and unauditable," with an "Unknown challenge pool where reuse and memorization advantages are unclear." Also notes `score^12` exponentiation amplifying small advantages into total dominance — **a caution about extreme reward convexity.**

**Subnet 34 (BitMind) — benchmark control and record alteration.** ([Zane Merritt, May 2026](https://zanemerritt.medium.com/subnet-34-i-connected-the-dots-on-bitmind-and-heres-what-bittensor-can-t-afford-to-ignore-a7576135c10d)) Alleges scoring runs on the team's own cloud against private holdout datasets, scored by team-controlled code behind a team API; an ambiguous winner rule (round 1 combined MCC vs later per-modality MCC, with a UID winning overall despite another having higher individual MCC); and altered records characterized as bugs only after screenshots surfaced. The author's proposed remedies are a good checklist regardless of the allegations' status: **"Independent validators should evaluate models locally against committed datasets in reproducible Docker environments, with image hashes, dataset manifest hashes, and scoring code hashes published before each round opens"**, plus escrowed hashes of private winning models so an independent auditor can verify provenance without public disclosure.

**Not found / unverified.** Despite targeted searching I found **no** documented case of: (a) a successful "submit a copy of the leader with small noise added" attack *winning* emission on SN9/SN37 — epsilon appears to have prevented the payoff even though copying was widespread (§1.5b); (b) a formal benchmark-contamination measurement on a model-competition subnet; (c) a named SN37 "model theft" postmortem. **Absence of evidence here is weak evidence — these subnets have small communities and no obligation to publish postmortems.**

---

# Part 7 — Is any Bittensor subnet doing genuine NAS?

**Short answer: two tried; both are gone.** As of 2026-08 I found **no live Bittensor subnet running genuine neural architecture search.**

## 7.1 SN31 — NASChain (dormant)

Genuine NAS by design: a **genetic algorithm** where each architecture is a **binary-encoded "genome"**, inspired by **NSGA-Net**, optimizing the multi-objective tradeoff of accuracy vs parameter count vs FLOPs. A central "Genomaster" assigned genomes to miners for parallel training (one job per miner initially). Because NAS is multi-objective, "there will be more than one optimal solution" — the output is a **Pareto front**, not a single winner.
Sources: [NASChain repo](https://github.com/mutexlocker/NASChain/tree/main); [Tensorplex Labs review](https://medium.com/@tensorplexlabs/bittensor-subnet-review-sn31-neural-architecture-search-dd90b5d84f0b) (reports models found at the CIFAR-10 Pareto frontier).

**Status: dormant.** Netuid 31 was **re-registered** and now hosts an unrelated project ("rec4ll", decentralized RAG) [3P] ([SubnetRadar](https://subnetradar.com/research/subnets/31)): "It is a re-registration of the slot that until early 2025 hosted NASChain, a decentralised neural-architecture-search project that has since gone dormant."

**Mechanism note of real interest:** NAS is intrinsically multi-objective, so WTA is a poor fit — a Pareto front has many valid winners. NASChain's reward had to span a frontier rather than collapse to one leader. Tensorplex reports the subnet later moved "from a PoW-based neural architecture search with NSGA-Net" toward challenging miners with "their best neural architecture search **or other AutoML strategies** to expand the Pareto frontier even further" — i.e. from *running a prescribed search* to *competing on search methods*, the same shift Gradients made (§4.1).

## 7.2 SN49 — Hivetrain AutoML (deregistered)

More ambitious than NAS: searching for **novel loss functions, activation functions, and potentially whole algorithms**, explicitly inspired by **AutoML Zero**, using **genetic programming** with evolutionary + gradient-based optimization.
Sources: [Hivetrain/DistributedAutoML](https://github.com/Hivetrain/DistributedAutoML/); [bittensor123.com/subnets/sn49](https://bittensor123.com/subnets/sn49/).

**Status: deregistered in late 2025** [3P]. Under the mechanism restored in Sept 2025, the 128-subnet cap means a new registration deregisters the **lowest-price non-immune subnet**; immunity is **4 months**; deregistered subnets' alpha is liquidated to TAO for holders. ([taostats.io/docs/subnet-registration](https://docs.taostats.io/docs/subnet-registration))

**This is the most important structural finding in this section.** Both NAS/AutoML subnets died, and the deregistration rule explains why the death was likely structural rather than technical: **subnet survival depends on alpha token price, not on research output.** Architecture search is long-horizon, capital-intensive, and produces results that are hard to price — exactly the profile that loses a market-cap ranking contest against subnets with near-term revenue. A research-oriented mechanism must survive a market-based selection process that does not measure research quality.

The nearest live analogues are **method-competition** subnets rather than architecture search: Gradients SN56 (AutoML tournaments over *training code*, §4) and, loosely, SN37 (fixed architecture list, competing on weights). Note SN37/SN9 explicitly **whitelist architectures** (`allowed_architectures = [MistralForCausalLM, LlamaForCausalLM, ...]`) [CODE] — the opposite of architecture search. SN9's whitepaper says relaxing that is aspirational: "we will investigate how we can safely relax key constraints on the competitions such as architectures and tokenizers" [DOC] — the blocker being that a free architecture choice breaks loss comparability and opens gaming surface (cf. §3.1 on forced tokenizers).

## 7.3 Other subnets checked

- **Chutes (SN64, Rayon Labs)** — serverless inference, not model competition. Relevant only for **GraVal**, a hardware-attestation protocol: proof-of-verifiable-work challenges with **seeded matrix multiplications and GPU-specific decryption**, timing-bounded, to prevent fake/virtual GPUs; plus symmetric-key exchange derived from GPU properties, dummy-socket port validation, and periodic filesystem-hash challenges. In [`api/graval_worker.py`](https://github.com/chutesai/chutes-api/blob/main/api/graval_worker.py) [CODE] the verifier checks both correct plaintext **and elapsed time against an expected estimate**. Scoring reported [3P] as 7-day windows: 55% compute units, 25% successful invocations, 15% chute diversity, 5% bounties. **Takeaway: if you pay for compute, you must attest the hardware, and timing is part of the proof.**
- **BitMind (SN34)** — now **GAS (Generative Adversarial Subnet)**: two tracks, discriminative miners submitting deepfake detectors and generative miners producing synthetic media. Scoring: `sn34_score` = **geometric mean of MCC and Brier score** ("measuring both accuracy and calibration"); generative miners get "base reward for valid content × multiplier for fooling discriminators." ([BitMind-AI/bitmind-subnet](http://github.com/BitMind-AI/bitmind-subnet)) **A genuinely adversarial, co-evolving benchmark — the eval set is regenerated by the adversary, so it cannot be memorized.** That is a structural answer to contamination, at the cost of a non-stationary target. See §6.5 for allegations about its scoring governance.

---

# Part 8 — Non-Bittensor precedents

## 8.1 Numerai — the closest real precedent for paying for noisy predictive signal

**Read this section first if the goal is paying for noisy signal.** Numerai has run this problem for ~9 years with real capital, and its documented *sequence of failures* is more informative than any single formula.

### 8.1.1 Payout formula (legacy continuous staking) [DOC]

```
score  = corr20 * corr_multiplier + mmc20 * mmc_multiplier
payout = stake * clip(payout_factor * score, -0.05, 0.05)
```

- **Max ±5% of stake per round.** Positive → NMR minted; negative → **NMR burned** (sent to a null address, "disappearing, not simply being sent to another user").
- Multipliers for the main tournament: **0.5 × CORR + 2 × MMC**.
- Signals used **1 × FNCv4 + 2 × MMC**, then from **2 Sept 2025**: **0.3 × Alpha + 0.8 × MPC**.
- Scoring horizon: `corr20`/`mmc20` are 20-day scores, so ~1 month to fully score a submission.

**Payout factor — the burn/emission governor:**

```
payout_factor = min(1, stake_threshold / total_at_risk)
```

Stake thresholds: **Numerai 72,000 NMR; Signals 36,000; Crypto 10,000.** Payouts scale *down* as total stake grows. **This is a supply-side control on reward inflation that does not require changing the scoring rule** — as more capital chases the signal, per-unit payout falls automatically.

Sources: [docs.numer.ai/numerai-tournament/staking](https://docs.numer.ai/numerai-tournament/staking), [Payouts and Performance (DeepWiki)](https://deepwiki.com/numerai/docs/7.2-payouts-and-performance).

### 8.1.2 Why staking exists — the mechanism-design argument [DOC]

Numerai's own framing is the sharpest articulation of the noisy-signal trust problem I found anywhere:

> "Numerai trades real capital based on the Meta Model, so it needs to know which predictions it can trust before relying on them. **Trust is not the same as being right** — any submission can be accidentally right, and **random or adversarial submissions will sometimes score well by pure luck**. Staking is how Numerai separates good-faith predictions from noise."

Two functions are stated: (1) "skin in the game" lets Numerai trust staked predictions; (2) "**Payouts and burns continuously improve the weights of the Meta Model.**"

**This is the key structural idea and it is different from everything in Bittensor.** The stake *is* the model weight in the ensemble. Payouts and burns are not merely compensation — they are the **gradient update on the ensemble weights**. Good models accumulate stake and therefore influence; bad models burn away and lose influence. The economic layer and the aggregation layer are the same mechanism. **A submitter with no capital at risk cannot be distinguished from a lucky guesser, so requiring capital at risk is what makes a noisy signal payable at all.**

### 8.1.3 The originality problem — MMC and the metric graveyard

Numerai's central difficulty is **not** measuring accuracy; it is measuring **marginal contribution** — paying for signal it does not already have. The history is a sequence of attempts:

**MMC (Meta Model Contribution)** — orthogonalize a user's predictions against the Meta Model, then measure the residual's value: "does a small weight on your signal when added to the Stake Weighted Meta Model improve or hurt its correlation with the target?" [DOC]

**TC (True Contribution)**, ~2022 — went further, measuring the **gradient of portfolio returns with respect to the user's stake**. The design notes are excellent on why a gradient beats leave-one-out: [DOC]

> "we realized that the leave-one-user-out method is really just approximating a gradient calculation... A true gradient calculation would also have the nice properties that 1) it can be computed for all users simultaneously from a single portfolio optimization rather than computing a separate optimization for each user held out and 2) **it will assign the same values to identical signals with different stakes** and 3) it will assign proper values to 0 stakes."

Property (2) is precisely the anti-duplicate property: **two identical signals get identical scores regardless of stake size**, so you cannot win by splitting or by staking bigger. Compare SN9's epsilon, which achieves something similar by timestamp instead.
([True Contribution Details](https://forum.numer.ai/t/true-contribution-details/5128))

**TC was then abandoned.** Reverted to MMC for rounds from **2 Jan 2024** (0.5×CORR + 2×MMC). Stated reasons: [DOC]

> "In the past, MMC had some weaknesses. It used to be computed against old targets without feature penalization or without liquidity adjustments... Because MMC on old targets had these weaknesses, we developed TC. However, we are now almost ready with a new target (called Teager) which does almost all the important transformations that the optimizer does **within the target**, making MMC on this target a great measurement of contribution."

**A better target made a simpler metric sufficient.** Rather than building an ever-more-elaborate contribution metric on top of a poor target, they improved the target and reverted to the simpler metric. Second stated reason — **local computability**:

> "Calculating MMC in a bagged/LOO formulation makes the calculation more opaque to data scientists because you all don't have access to other models' raw predictions, thus you can't optimize for it locally. **Local optimization is a key characteristic.**" ... "Numerai now gives out the Meta Model signal so MMC is not a blackbox any more and can be computed locally."

**If participants cannot compute the objective locally, they cannot optimize it, and the mechanism fails** — not from gaming but from unimprovability. A metric that is robust but opaque may be worse than a slightly gameable metric that participants can actually target. **This is a genuine tension with commit-reveal and hidden-eval-set designs**, which deliberately reduce what participants can compute.

**Why optional staking on contribution metrics failed** — a clean documented incentive failure: [DOC]

> "In the past MMC and TC were optional to stake on. Many users would simply stake CORR with a large stake and not stake MMC or TC at all. The problem is many large stakers this year have had persistently negative TC & far worse TC than benchmark models. This would be fine if these models were being burned away but if they weren't staking TC this year, **they would hardly burn at all if their CORR was okay or flat.** The point of payouts is to get feedback into the stakes so that the Stake Weighted Meta Model can improve. **Users persistently hurting the Meta Model but doing okay on CORR shouldn't be able to earn a positive return on their stake.**"

**If the metric that measures marginal value is opt-in, nobody opts in.** Participants select the metric they score well on, and the ensemble stops improving. Contribution-based payment must be mandatory to work.
([Changing Scoring & Payouts Again To MMC Only](https://forum.numer.ai/t/changing-scoring-payouts-again-to-mmc-only/6794), [MMC staking starts Jan 2, 2024](https://forum.numer.ai/t/mmc-staking-starts-jan-2-2024/6827))

Metric churn is itself documented: TC "quietly disappeared after round 713"; MMC is "the most-revised metric on the platform, with six numbered versions across rounds 168–1255"; FNC went through two retirements before FNCv4 [3P] ([nmrdash.com cheatsheet](https://nmrdash.com/articles/numerai-metrics-cheatsheet)). **Expect to revise the scoring metric repeatedly; design for versioned, dated metric changes rather than a permanent formula.** (SN37's block-scheduled competition sunsets, §3.1, are the Bittensor analogue.)

### 8.1.4 Churn / turnover thresholds — a hard usability gate [DOC]

A mechanism with no Bittensor analogue, and directly relevant to "noisy signal that must be *usable*". From **20 Sept 2024**, Signals enforces:

```
max_churn    = max([churn(t, t-1), churn(t, t-2), ..., churn(t, t-5)])
max_turnover = max([turnover(t, t-1), ..., turnover(t, t-5)])
if max_churn >= 15% or max_turnover >= 25%:  stake = 0
```

Plus: "Any model that has not submitted in the previous week will have its stake set to 0."

Rationale — a signal that thrashes cannot be traded:

> "If a Signals submission has high churn, then Numerai can't trade the signal easily... Most Signals models have > 20% week-over-week churn... the average individual churn of Signals models is nearly 70% correlated with the Signals Meta Model Churn."

And the feasibility argument, which is the right way to justify a hard threshold: "**Is 15% too low? No. Our v43.cyrus_plus_teager model has never breached 15% churn**, so we know this is an achievable level." They demonstrated the constraint was satisfiable with their own reference model before imposing it, and open-sourced the calculation ([numerai-tools](https://github.com/numerai/numerai-tools)) so participants can check locally.

Note the asymmetry: this applies to Signals, **not** the main tournament, because "Numerai models... cannot control their churn level due to the obfuscation of the dataset. Instead, we have crafted a dataset that naturally results in lower-churn models." **Where you cannot expose a constraint to participants, engineer the data so the constraint is satisfied by default.**
([Signals Churn Threshold](https://forum.numer.ai/t/signals-churn-threshold/7648), [Signals scoring docs](https://docs.numer.ai/numerai-signals/scoring))

### 8.1.5 Current state — Atomic Blockchain Staking (2026) [DOC]

Announced at **NumerCon 2026**; replaces continuous staking. Documented changes:

- **Clip → ±1** (±100%/round), **payout_factor → 1**; `payoutFactor`, `stakeThreshold`, `stakeCap` are **deprecated/`null`** for upgraded contracts — "read payout policy from the tournament round configuration."
- Signals multipliers planned **4 × Alpha + 8 × MPC**. The docs explicitly warn this was *not* stated to apply to Numerai Classic and "must be read from the applicable round rather than inferred across tournaments."
- Removes **overlap leverage**. The quantitative explanation is a nice illustration of an unintended-leverage bug: Numerai/Crypto had 24 overlapping rounds but the 5% clip capped effective leverage at ~1.2× and real payouts of 1–2% meant experienced leverage was **below 1**; Signals had 64 overlapping rounds with a 3.5% clip → **2.24×** effective leverage, and "hits the clip value far more often, meaning Signals actually does experience this higher leverage."
- Per-round settlement via **Merkle root** with model-scoped claims (`totalStaked`, `remainingPayout`, `remainingBurn`).
- Rollout: Crypto **16 June 2026** (24 business days); Signals TBD (64); Numerai TBD (24).

**Overlapping scoring windows silently create leverage.** If round N's stake is still at risk while rounds N+1…N+k are also live, effective exposure is a multiple of nominal stake, and the multiple depends on window length and clip. Numerai shipped this for years before quantifying and removing it.
([What is Atomic Blockchain Staking?](https://forum.numer.ai/t/what-is-atomic-blockchain-staking/8302), [docs](https://docs.numer.ai/numerai-tournament/atomic-blockchain-staking))

### 8.1.6 The originality thesis [DOC]

> "Numerai Signals is all about creating signals with **predictive orthogonal components** — the original part of the signals that we don't already have. The incentives are therefore around creating signals from unusual data sources or unusual modeling techniques... **Numerai Signals rewards people how the market should reward them: for the marginal predictive value of the non-redundant component of their signal.**"

And the participant-side consequence, from the forum: "if you are un-original your signal or model most likely won't do well **because they will neutralize it**... So good and different is what you're looking for."

**Duplicates are not punished; they are neutralized to zero value.** This is the same philosophy as SN9's epsilon (copying permitted, made unprofitable) but implemented via orthogonalization rather than timestamps — and it generalizes better, because it handles *near*-duplicates and independent rediscovery, not just copies.
([Building the Last Hedge Fund: Introducing Numerai Signals](https://medium.com/numerai/building-the-last-hedge-fund-introducing-numerai-signals-12de26dfa69c))

## 8.2 Prime Intellect — INTELLECT-2 (verification, not competition) [DOC]

32B reasoning model via globally distributed async RL. **No incumbent/challenger competition** — the relevant contribution is **verification of untrusted work**.

- **TOPLOC** — "a locality-sensitive hashing scheme for efficient verifiable inference. It detects tampering or precision changes in model inference and **works reliably across nondeterministic GPU hardware**."
- **Asymmetric verification cost**: "The Inference Provider performs batched inferences and generates commits for the computations performed, while **the Verifier audits these commits up to 100× faster than the time it takes the inference provider to generate the responses.**"
- **Random spot-checking with unpredictability**: "Further speedup can be obtained for the Verifier by not checking every batch but instead sampling randomly. **Since the Inference Provider does not know which generations will be checked by the Verifier, they are incentivized to be honest on all**."
- Enforcement: "accepted files feed the trainer, while invalid ones **slash and remove the submitting node from the pool**."
- Validators run "computation, sampling and data sanity check[s]".
- Scale: 285k verifiable math/coding tasks; binary task reward + length reward; two-sided GRPO clipping; SHARDCAST for weight broadcast.

**The 100× verification asymmetry plus unpredictable sampling is the general shape of cheap honest-work enforcement** — the same pattern as IOTA's surprise recomputation (§2.1) and Templar's two-phase evaluation (§5.3). You do not need to verify everything; you need the *probability* of being checked to be unpredictable and the penalty to exceed the gain.
([arXiv 2505.07291](https://arxiv.org/abs/2505.07291), May 2025; [blog](https://www.primeintellect.ai/blog/intellect-2-release))

## 8.3 Gensyn — Verde / RepOps / Judge (verification via refereed delegation) [DOC]

The most rigorous verification work found, and explicitly **not** proof-of-learning.

- **Refereed delegation**: delegate to ≥2 untrusted providers; "with a guarantee of obtaining the correct result **if at least one of them is honest**." On disagreement, a **two-level bisection game** pinpoints first the disagreeing training *iteration*, then the disagreeing *operator* in the compute graph; "the referee[] only needs to compute a single operator." Providers commit to each iteration via a **Merkle tree of (inputs, operator, output) tuples**, combined into a higher-level Merkle tree; providers "are required to only store and hash intermediate training checkpoints."
- **RepOps (Reproducible Operators)** — bitwise-reproducible ML operators that eliminate hardware nondeterminism "by enforcing a fixed execution order of floating point operations," so honest providers on different hardware produce **bitwise identical** outputs. **Without bitwise determinism, disagreement is uninformative** — you cannot tell cheating from float-ordering differences. This is the enabling primitive.
- **Cost comparison [DOC]**: refereed delegation costs "at least a factor 2 total overhead" and "less than an order of magnitude" with RepOps — "**dramatically more efficient than cryptographic proofs (4 orders of magnitude)**." Concretely: SNARKs ~10,000× vs ~2–10× for refereed delegation. **This is the number that should decide any verification-design debate.**
- **Acknowledged open problems** (unusually candid): needs "a robust ecosystem of trainers... unlikely to collude or suffer related faults (e.g. by running the same third party data center)"; "**incentives are needed to compensate trainers both for running the original computation and for interacting with the referee**"; and EVM "was not designed with ML" in mind. **Verification protocols need their own incentive layer** — someone must be paid to dispute, or nobody disputes.
- **Judge** applies Verde to *evaluation*, making "every judgment... independently checked," with provenance tracking from ONNX nodes through graph transformations. A described RL-Swarm mechanism has models bet on correct answers with **progressive information revelation and early-correct-bets-pay-more** — a market-scoring-rule flavor, interesting as a way to reward *early* correct judgment.

Sources: [arXiv 2502.19405](https://arxiv.org/abs/2502.19405) (Feb 2025); [Verde blog](https://blog.gensyn.ai/verde-a-verification-system-for-machine-learning-over-untrusted-nodes/); [Verde in production](https://blog.gensyn.ai/verde-verification-system-in-production/); [Introducing Judge](https://blog.gensyn.ai/introducing-judge/).

## 8.4 Flock.io — stake-weighted voting with slashing [DOC]

Federated-learning-oriented; the mechanism is **role-randomized proposer/voter with reward-and-slash**:

- Participants stake `$FLOCK`; each round they are **randomly assigned on-chain** as **proposer** or **voter**.
- Proposers train locally and share updates; voters aggregate and validate against **their own local held-out data**, producing a validation score. Each participant's local dataset is "randomly partitioned into a training set and a test set, which will not be shared."
- **Commit/reveal voting**, stake-weighted, determines the winning model. "Only a model confirmed to be better is aggregated and adopted as the new base model."
- Payoff rule [DOC]: if the aggregated vote is non-negative, all proposers are rewarded (proportional to stake) and voters who voted non-negative are rewarded, others slashed; if negative, proposers are slashed and negative-voters rewarded. **Slashed tokens are redistributed to honest participants.**
- Contract parameters exist (`_totalNumberOfRounds`, `_minStakeThreshold`, `_initialRewardPoolSize`) but **specific values are not documented** in what I read.

**This is a truth-by-majority mechanism, not a measurement mechanism** — you are rewarded for *agreeing with the stake-weighted majority*, which is structurally the same incentive that produces Bittensor's weight-copying problem (§6.1). Voting with your own private held-out data is the independence-preserving feature; if voters could see each other's votes before committing, it would collapse. Note the whitepaper's own listed defense against model poisoning is "majority voting minimizes the impact of single malicious participants" — sound against *independent* faults, weak against *correlated* ones.
([docs.flock.io](https://docs.flock.io/flock-products/fl-alliance/task-lifecycle-deep-dive/1.-staking-and-role-assignment), [whitepaper](https://www.flock.io/whitepaper))

## 8.5 Sentient — OML fingerprinting [3P/DOC]

**OML 1.0** ("Open, Monetizable and Loyal AI") embeds secret **(key, response) fingerprint pairs** into a model via fine-tuning, so a model's provenance can be proven later; provers can detect protocol violations and **slash the violator's collateral**. Only lightly verified in this survey — I did not reach Sentient's primary whitepaper.

**Relevance to model competition is real though:** fingerprinting is the one mechanism found that could *positively identify* a submitted model as a derivative of a specific prior model, rather than inferring copying from behavioral similarity (SN9's X-ray) or defeating it economically (epsilon). Worth deeper investigation for any design needing provenance rather than deterrence.

---

# Part 9 — Cross-cutting synthesis

## 9.1 The five families of solution to "noisy scalar → payment decision"

Every system surveyed does at least one of these. Listed roughly in increasing statistical sophistication.

| # | Pattern | Mechanism | Where |
|---|---|---|---|
| 1 | **Discard magnitude, keep order** | Reduce noisy loss to pairwise win/loss, aggregate many comparisons | SN9 `iswin` + win rate; Templar OpenSkill |
| 2 | **Pair the comparison** | Score both models on the *identical* held-out set, compare per-example so item difficulty cancels | Gradients paired boss round (§4.3) |
| 3 | **Dead zone + confidence bound** | Ignore differences below a meaningful floor; require the margin to hold at a bootstrap lower bound | Gradients (0.01 nats, 99%/10k) |
| 4 | **Smooth the payment, not the score** | EMA on weights, temporal decay, minimum-weight floors | SN9 α=0.5; SN37 α=0.90 + floor 0.18; IOTA γ; SN56 0.165%/day |
| 5 | **Require skin in the game** | Stake at risk; burn on bad performance; stake *is* the ensemble weight | Numerai; Flock; SN56 entry fees |

**The strongest single design in the survey is #2 + #3 together** (Gradients). Pairing removes the dominant variance term; the dead zone prevents decisions on meaningless differences; the bootstrap bound scales strictness to available evidence automatically. And it is calibrated against measured data (62 tasks, median separation 0.0094 nats) rather than chosen a priori.

## 9.2 Anti-copying: four strategies, in order of robustness

1. **Make copying unprofitable, not impossible** — epsilon/timestamp priority (SN9/SN37). No classifier to evade; a copy always ties on quality and loses on time. **Most robust because there is nothing to fool.** But bounds the payoff without preventing the behavior (§1.5b).
2. **Neutralize redundancy** — orthogonalize against the existing aggregate; pay only the residual (Numerai MMC/TC). Handles near-duplicates and independent rediscovery, which timestamp priority does not. TC's "identical signals get identical scores regardless of stake" is the property to aim for.
3. **Deny the information needed to copy** — encrypt the hash to the uploader's hotkey, private repos (SN37); commit-reveal (Yuma); put windows (Templar); unpredictable check timing (IOTA/Prime Intellect). Effective but **brittle**: §6.4 shows what one key-management bug costs.
4. **Detect it** — X-ray weight visualization (SN9), LLM repo-diff/dedup (SN56), sync-score divergence (Templar), fingerprinting (Sentient). Necessary as a backstop; inherently an arms race.

**Copy-resistance trades directly against open collaboration.** SN9 wanted "rebasing" on public weights and got copying; SN37 went private-by-default and lost the compounding. Gradients resolves it best: publish everything, but equalize compute and protect the champion statistically, so a copy of the public recipe wins nothing.

## 9.3 What the evidence says about winner-take-all

Documented consequences, all from primary sources:

- **Hoarding** — "winner-takes-all rewards encouraged model hoarding" (IOTA paper); a self-described "design flaw" (SN9 whitepaper §3.4.2). Rational strategy is to hold improvements back.
- **High barrier / thin participation** — every miner must fund a full training run; SN9 reportedly reached ~6 miners [3P].
- **Easier weight-copying** — a near-one-hot weight vector is cheap to infer, and epsilon/EMA deliberately keep it stable (my inference, §6.1).
- **Near-permanent incumbency if thresholds are set too high** — measured by Gradients: a bar above the median separation, compounded over 4-of-5 required tasks, "makes the crown near-permanent."

Documented benefits: concentrates effort on the frontier where demand is; "encourages greater professionalism"; and a genuine **anti-sybil property** — multiple hotkeys gain nothing when only rank 1 pays.

**Both teams that ran WTA at scale moved away from it.** SN9 → proportional-to-work (IOTA); SN56 → 50% base floor + champion premium + burn. Templar's superlinear `c=2` is the middle path: keep the anti-sybil convexity without the all-or-nothing cliff. **If sybil-resistance is the reason for WTA, a convex reward curve or an entry fee achieves it at far lower cost.**

## 9.4 Evaluation integrity checklist (assembled from all sources)

- **Sample from a corpus too large to memorize**, and audit by transfer to unrewarded benchmarks (SN9 §1.5d — reasonable but weakly evidenced).
- **Quantify contamination**: `MAX_NEAR_DUPLICATE_RATE = 0.20` (SN56) — the only hard number found.
- **Multi-task, weighted, normalized** so overfitting one task is punished by others; bound each sub-score (`INVERSE_EXPONENTIAL` + ceiling) so one blowup cannot dominate.
- **Force the tokenizer** (and other comparability constraints) or cross-model losses are not comparable at all.
- **Control eval-data provenance** — stake gates, freshness limits (4h), minimum sample counts, hash-based validator synchronization (SN37 §3.3). Where competitors can influence eval data, this is the primary attack surface.
- **Delay evaluation** (`eval_block_delay`) so validators converge before weights move.
- **Seed all randomness deterministically** — `BOSS_ROUND_BOOTSTRAP_SEED`, "Never vary this." Unseeded randomized scoring makes honest validators disagree, and consensus then punishes them.
- **Unpredictable spot-checks with recomputation**, exploiting verification asymmetry (100× TOPLOC; cosine-similarity recompute in IOTA; two-phase Gauntlet).
- **Expect structural constraints to be met minimally** — parameter bands invite padding (§1.5b).
- **Adversarial/co-evolving eval sets** (BitMind GAS) structurally resist contamination, at the cost of a non-stationary target.
- **Pre-commit hashes** of datasets, scoring code, and images before a round opens; allow independent auditor re-runs (§6.5 remedy list).

## 9.5 Numerai-specific lessons that generalize

- **Contribution-based payment must be mandatory.** Optional contribution metrics → nobody stakes them → ensemble stops improving (§8.1.3).
- **Participants must be able to compute the objective locally**, or they cannot optimize it. Directly tensions with hidden eval sets and commit-reveal.
- **Improve the target before elaborating the metric.** A better target (Teager) let them retire TC and revert to simpler MMC.
- **Expect to revise scoring repeatedly** — design for versioned, dated metric changes (cf. SN37's block-scheduled sunsets).
- **A payout factor that scales inversely with total stake** governs reward inflation without touching the scoring rule.
- **Burn is a first-class mechanism**, not an accident — irreversible destruction is what makes negative scores meaningful. SN56 likewise burns rather than paying out below its threshold.
- **Watch for accidental leverage from overlapping scoring windows** (§8.1.5).
- **Usability constraints may need separate hard gates** (churn 15% / turnover 25%), justified by demonstrating a reference model satisfies them, with the calculation open-sourced.

---

# Part 10 — What I could NOT verify

Stated explicitly, as requested.

**Parameters I could not find:**

- **IOTA's live `γ` (temporal decay) value.** The paper gives the formalism and says simulations suggest γ < 10h is desirable; the deployed number is **not documented** in what I read.
- **IOTA's spot-check rate, cosine-similarity threshold, and penalty magnitudes** — not documented.
- **CR3 reveal intervals actually configured per subnet today.** The 2024 analysis says `G` converges at ~1800 blocks (5 tempos); current per-subnet settings were not retrieved.
- **Flock.io's concrete values** for `_minStakeThreshold`, slash percentages, round counts, reward pool sizes — contracts documented structurally only.
- **SN56's `EMISSION_MULTIPLIER_THRESHOLD` and champion-decay values in the current `validator/core/constants.py`** — I read these from the 2026-06-19 commit message (0.10 burn bar; 0.165%/day) and from `validator/tournament/constants.py`, but did not fetch `core/constants.py` itself to confirm they are unchanged since.
- **`BOSS_ROUND` minimum decided-example count** — comments reference it ("Below this many decided examples...") but the constant's value was not visible in the section I read.
- **Numerai Classic's current live multipliers under v3.** Docs explicitly warn the 4×Alpha + 8×MPC figures were stated for Signals and must not be inferred for Numerai. Treat Classic's v3 multipliers as **not documented**.
- **Sentient OML** — only lightly verified; no primary whitepaper read.

**Claims I could not substantiate:**

- **SN9 ε = 3%** for the standard 7B competition [3P] — contradicted by the whitepaper's 0.5%. **Do not use.**
- **A 16B IOTA training run in July 2026** — [3P] search summary only.
- **Gradients' 82–100% win rates vs AutoTrain/TogetherAI/Vertex/Databricks** — vendor claim, not independently verified.
- **A specific SN37 "model theft" incident/postmortem** — searched, not found.
- **A documented successful "copy the leader + noise" attack winning emission** on SN9/SN37 — copying is documented as widespread (X-ray), but no source shows it capturing the top slot. Epsilon appears to have worked as designed here.
- **Any formal benchmark-contamination measurement** on a model-competition subnet, beyond SN56's near-duplicate rate cap and SN9's indirect transfer argument.
- **SN6 (Nous) specific exploits.** SN37 docs imply prior vulnerabilities existed; no writeup found. Note netuid 6 is now a different project.
- **SN33/SN44/SN34 allegations (§6.5)** — single-author community investigations, not confirmed by the subnet teams. Treated as allegations.
- **Exact date of the SN9 → IOTA cutover.** The paper says mainnet graduation 2 June 2025; whether the old competition ran in parallel afterward is unclear.
- **CLASP in production** — the paper explicitly says it was *not* in the initial release; I found no evidence it shipped since. Several [3P] sources wrongly imply it is live.

**Structural caveat on sources:** several subnet-explainer sites appear LLM-generated and one contradicts a primary whitepaper on a numeric parameter. I have marked all such uses [3P] and relied on repo code and papers for every parameter presented as live.

---

# Sources

## Bittensor SN9 / SN37 / Taoverse
- [macrocosm-os/pretraining](https://github.com/macrocosm-os/pretraining) — SN9 repo/README
- [pretraining/constants/__init__.py](https://github.com/macrocosm-os/pretraining/blob/main/constants/__init__.py) — live SN9 epsilon, temperature, eval mix **[CODE]**
- [macrocosm-os/finetuning](https://github.com/macrocosm-os/finetuning) — SN37 repo/README
- [finetuning/constants/__init__.py](https://github.com/macrocosm-os/finetuning/blob/main/constants/__init__.py) — live SN37 epsilon, ALPHA, MIN_WEIGHT_THRESHOLD, competition schedule **[CODE]**
- [finetuning/docs/validator.md](https://github.com/macrocosm-os/finetuning/blob/main/docs/validator.md) — `iswin` implementation + anti-copy rationale **[CODE/DOC]**
- [finetuning/docs/miner.md](https://github.com/macrocosm-os/finetuning/blob/main/docs/miner.md) — `--load_best`, upload rate limits
- [macrocosm-os/taoverse](https://github.com/macrocosm-os/taoverse) — shared library
- [taoverse epsilon.py](https://github.com/macrocosm-os/taoverse/blob/main/src/taoverse/model/competition/epsilon.py) — `FixedEpsilon`, `LinearDecay` **[CODE]**
- [LLM pretraining: The Use-Case Blockchain Has Been Waiting For? (PDF)](https://www.macrocosmos.ai/research/pretraining_whitepaper.pdf) — ε=0.5%, WTA 95%+, hoarding, exploits (~Aug–Sep 2024)
- [Monsters, vampires, and X-rays: subnet 9's Halloween deep dive](https://macrocosmosai.substack.com/p/monsters-vampires-and-x-rays-subnet) — measured copying/inflation (~Oct 2024)
- [Fine-tuning, finely tuned: How SN37 is delivering SOTA fine-tuning](https://macrocosmosai.substack.com/p/fine-tuning-finely-tuned-how-sn37) — encrypted hash, private repos, eval integrity
- [IOTA: A Technical Primer for Release, arXiv 2507.17766](https://arxiv.org/abs/2507.17766) — WTA abandonment, γ decay, CLASP, Butterfly All-Reduce (Jul 2025)
- [docs.macrocosmos.ai/subnets/subnet-9-iota](https://docs.macrocosmos.ai/subnets/subnet-9-iota)
- [Macrocosmos: Bittensor's decentralized OpenAI?](https://oakresearch.io/en/analyses/fundamentals/macrocosmos-bittensor-decentralized-open-ai) — OAK Research **[3P]**

## Gradients / SN56
- [G.O.D validator/tournament/constants.py](https://github.com/gradients-ai/G.O.D/blob/main/validator/tournament/constants.py) — paired bootstrap gate, dead zone, win rate, seed, dedup, fees **[CODE]**
- [commit c83bc13 (2026-06-19)](https://github.com/gradients-ai/G.O.D/commit/c83bc13a6087121e176c600ee666e90d33706a71) — 50% base floor, 0.10 burn bar, 5/6 dethrone, 0.165%/day decay
- [commit a0da6a2](https://github.com/gradients-ai/G.O.D/commit/a0da6a2687d6bc5407ee9a0c23104bd4bf357827) — earlier progressive/exponential-decay dethrone thresholds
- [bittensor.ai/subnets/56](https://bittensor.ai/subnets/56) — tournament lifecycle, boss-round gates
- [Your Simple Guide to Gradients (SN56)](https://simplytao.ai/blog/your-simple-guide-to-gradients-sn56) **[3P]**
- [SubnetRadar SN56](https://subnetradar.com/research/subnets/56) **[3P]**
- [IQ.wiki: Gradients](https://iq.wiki/wiki/gradients) **[3P]**

## Templar / SN3
- [Incentivizing Permissionless Distributed Learning of LLMs, arXiv 2505.21684](https://arxiv.org/abs/2505.21684) — Gauntlet, OpenSkill, `c=2`, sync threshold 3, peer copying (May 2025)
- [docs.tplr.ai/incentive-design](https://docs.tplr.ai/incentive-design/)

## Yuma consensus / weight copying
- [Weight Copying in Bittensor](https://blog.bittensor.com/weight-copying-in-bittensor-422585ab8fa5) — Opentensor, 29 May 2024; copier advantage, G analysis, 1800-block convergence, 20/30 subnets
- [Battle-testing Yuma3](https://inference-labs.medium.com/battle-testing-yuma3-a8136c797d1f) — Inference Labs, 14 Aug 2025; **CR3 encryption break + 4 fixes in Bittensor 9.9.0 / Subtensor 3.2.4**
- [subtensor.com — Commit-Reveal (CRV3)](https://subtensor.com/reference/mechanisms/commit-reveal)
- [subtensor.com — Liquid Alpha](https://subtensor.com/reference/mechanisms/liquid-alpha) — sigmoid formula, alpha_low/high
- [subtensor.com — Weight Management](https://subtensor.com/reference/mechanisms/weight-management)
- [bittensor.com/docs/concepts/emissions](https://www.bittensor.com/docs/concepts/emissions) — kappa, bonds EMA, Yuma3, liquid alpha defaults
- [learnbittensor.org — Weight Copying](https://learnbittensor.org/concepts/scoring-rewards/weight-copying) · [Commit Reveal](https://learnbittensor.org/concepts/scoring-rewards/commit-reveal) · [Liquid Alpha](https://learnbittensor.org/concepts/scoring-rewards/liquid-alpha)
- [taostats — Subnet Registration/Deregistration](https://docs.taostats.io/docs/subnet-registration) — 128 cap, 4-month immunity, lowest-price dereg

## Gaming incidents **[3P]**
- [Bittensor Subnet 33: From Innovation to Exploitation](https://medium.com/@bittensor_player/bittensor-subnet-33-from-innovation-to-exploitation-4aae796a5f6c) — validator mimicry, key monopoly (Dec 2024)
- [Bittensor Subnet 44: Sports AI Goldmine or Insider's Playground?](https://zanemerritt.medium.com/bittensor-subnet-44-sports-ai-goldmine-or-insiders-playground-a-deep-investigation-d855e3fd07d7) — exploit list, centralized challenge API, score^12
- [Subnet 34: I Connected the Dots on BitMind](https://zanemerritt.medium.com/subnet-34-i-connected-the-dots-on-bitmind-and-heres-what-bittensor-can-t-afford-to-ignore-a7576135c10d) — May 2026; benchmark control, remediation checklist

## NAS / AutoML subnets
- [NASChain (SN31)](https://github.com/mutexlocker/NASChain/tree/main) — NSGA-Net genetic NAS, binary genomes, Pareto front
- [Bittensor Subnet Review — SN31 NAS](https://medium.com/@tensorplexlabs/bittensor-subnet-review-sn31-neural-architecture-search-dd90b5d84f0b) — Tensorplex **[3P]**
- [SubnetRadar SN31](https://subnetradar.com/research/subnets/31) — NASChain dormant, slot re-registered **[3P]**
- [Hivetrain/DistributedAutoML (SN49)](https://github.com/Hivetrain/DistributedAutoML/) — AutoML Zero-style loss/activation search
- [bittensor123.com/subnets/sn49](https://bittensor123.com/subnets/sn49/) **[3P]**

## Other subnets
- [chutes-api graval_worker.py](https://github.com/chutesai/chutes-api/blob/main/api/graval_worker.py) — PoVW challenge + timing check **[CODE]**
- [GraVal Protocol (DeepWiki)](https://deepwiki.com/chutesai/chutes/9.4-graval-protocol) · [GraVal Middleware](https://deepwiki.com/chutesai/chutes/5.2-graval-middleware)
- [BitMind-AI/bitmind-subnet (SN34)](http://github.com/BitMind-AI/bitmind-subnet) — GAS, MCC × Brier geometric mean
- [Why Chutes? An Institutional Review](https://taodaily.io/why-chutes-an-institutional-review/) — scoring breakdown **[3P]**

## Numerai
- [Staking | Numerai Docs](https://docs.numer.ai/numerai-tournament/staking) — payout formula, ±5% clip, payout_factor, stake thresholds
- [Atomic Blockchain Staking | Docs](https://docs.numer.ai/numerai-tournament/atomic-blockchain-staking) — v3, deprecated fields, Merkle settlement
- [What is Atomic Blockchain Staking?](https://forum.numer.ai/t/what-is-atomic-blockchain-staking/8302) — NumerCon 2026, leverage quantification, new multipliers
- [MMC staking starts Jan 2, 2024](https://forum.numer.ai/t/mmc-staking-starts-jan-2-2024/6827) — TC→MMC, 0.5×CORR + 2×MMC, local computability
- [Changing Scoring & Payouts Again To MMC Only](https://forum.numer.ai/t/changing-scoring-payouts-again-to-mmc-only/6794) — Teager target; opt-in-metric failure
- [True Contribution Details](https://forum.numer.ai/t/true-contribution-details/5128) — gradient-of-returns; identical-signal property
- [Signals Churn Threshold](https://forum.numer.ai/t/signals-churn-threshold/7648) — 15% churn, 20 Sep 2024
- [Scoring | Numerai Signals](https://docs.numer.ai/numerai-signals/scoring) — churn/turnover formulas, 25% turnover
- [Signals Alpha, MPC, and Turnover](https://forum.numer.ai/t/signals-alpha-mpc-and-turnover/8162) — 0.3×Alpha + 0.8×MPC from 2 Sep 2025
- [Numerai 60D Scores](https://forum.numer.ai/t/numerai-60d-scores/8037) — May 2025, CORR20/60 Sharpe 1.86 vs 2.97
- [Building the Last Hedge Fund: Introducing Numerai Signals](https://medium.com/numerai/building-the-last-hedge-fund-introducing-numerai-signals-12de26dfa69c) — originality thesis
- [numerai/numerai-tools](https://github.com/numerai/numerai-tools) — open-sourced scoring/churn code
- [Payouts and Performance (DeepWiki)](https://deepwiki.com/numerai/docs/7.2-payouts-and-performance) **[3P]**
- [Numerai metrics cheatsheet](https://nmrdash.com/articles/numerai-metrics-cheatsheet) — metric version history **[3P]**

## Non-Bittensor decentralized training
- [INTELLECT-2, arXiv 2505.07291](https://arxiv.org/abs/2505.07291) — TOPLOC, 100× verification asymmetry, random spot-checks, slashing (May 2025)
- [INTELLECT-2 Release blog](https://www.primeintellect.ai/blog/intellect-2-release)
- [Verde, arXiv 2502.19405](https://arxiv.org/abs/2502.19405) — refereed delegation, RepOps, cost comparison vs SNARKs (Feb 2025)
- [Verde blog](https://blog.gensyn.ai/verde-a-verification-system-for-machine-learning-over-untrusted-nodes/) · [Verde in production](https://blog.gensyn.ai/verde-verification-system-in-production/) · [Introducing Judge](https://blog.gensyn.ai/introducing-judge/)
- [Psyche Network Architecture](https://nousresearch.com/nous-psyche/) — Nous, Solana-coordinated (May 2025)
- [FLock docs: staking & role assignment](https://docs.flock.io/flock-products/fl-alliance/task-lifecycle-deep-dive/1.-staking-and-role-assignment) · [rewards](https://docs.flock.io/flock-products/fl-alliance/task-lifecycle-deep-dive/4.-rewards) · [smart contracts](https://docs.flock.io/flock-products/fl-alliance/smart-contracts-deep-dive) · [whitepaper](https://www.flock.io/whitepaper)
- [Decentralized AI Training: How Crypto Can Power Open AI](https://www.galaxy.com/insights/research/decentralized-ai-training) — Galaxy Research; emission splits, Templar ~4%
- [Bittensor Decentralized Training (PDF)](https://cruciblelabs.com/wp-content/uploads/2024/12/Bittensor-Decentralized-Training.pdf) — Crucible Labs, Dec 2024; subnet taxonomy, "winner-take-nearly-all" critique **[3P]**

