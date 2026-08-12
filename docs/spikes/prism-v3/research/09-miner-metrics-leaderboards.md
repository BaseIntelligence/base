# Appendix 09 — Miner-Reported Metrics and Leaderboard Design
> Research appendix for the Prism v3 evaluation proposal (`docs/spikes/prism-v3/`). Produced 2026-08-06 via arXiv/web research. Non-normative spike document.

# Self-Reported Metrics in ML Competitions and Transparent Leaderboard Design

*Survey date: 2026-08-06. All arXiv IDs and dates verified against primary sources this session. One citation correction up front:* **arXiv 2407.08702 is not the "benchmark lottery" paper** — it is a physics paper on dipolar Bose gases. The Benchmark Lottery is Dehghani et al., **arXiv 2107.07002** (Jul 2021, NeurIPS 2021). The leaderboard-sensitivity paper you may have had in mind is *When Benchmarks are Targets* (Alzahrani et al.), **arXiv 2402.01781** (Feb 2024). I cite the correct IDs below.

Throughout, I ground the final design in your existing stack: the PRISM architecture competition already runs miner `architecture.py` + `training.py` on operator-rented Lium pods, computes bpb harness-side, and requires telemetry hooks (`docs/PRISM_RECIPE.md` lines 26–44):

> The harness registers a `prism_telemetry` module before miner code loads
> (also at `ctx["telemetry"]`). `training.py` MUST:
>
> ```python
> import prism_telemetry
>
> prism_telemetry.report(loss=..., step=..., grad_norm=..., layer_stats=...)  # every N steps
> prism_telemetry.finish_evaluation()  # optional early stop: score the model as-is
> ```
>
> The harness captures the series into `METRICS_JSON.telemetry.loss_series`
> (persisted master-side in `prism_telemetry` and surfaced on the site).

That hook is exactly where participant metrics enter; the question is how to label, validate, and display them so they inform without corrupting the scored lattice.

---

## 1. Trust models for self-reported metrics

Five production systems span the full trust spectrum. The useful way to compare them is along two axes: **(a) who executes the measurement** (participant vs organizer vs auditor), and **(b) who sees the result before publication** (selective-disclosure channel).

**Kaggle — organizer-computed with a two-split honesty device.** Participants submit predictions (or, in code competitions, notebooks that Kaggle re-runs after the deadline). The server evaluates the *whole* test set internally but displays only the public-split score; final standings come from the disjoint private split. Rank churn between the two ("shake-up") is a measured, well-documented phenomenon — public 3rd → private 22nd is unremarkable — and Roelofs et al., *A Meta-Analysis of Overfitting in Machine Learning* (NeurIPS 2019; MetaKaggle data), quantified adaptive overfitting to the public split across thousands of competitions. Two lessons: (i) never let participants compute the scored number on their own hardware, and (ii) even organizer-computed scores need a held-out channel, because any visible score becomes a gradient signal. A subtler failure: public/private splits are not always random — at least one competition scored the *first 50% of rows chronologically*, turning tuning from interpolation into extrapolation with no visible warning. Publish how the split is drawn.

**MLPerf — self-measured, but checked, peer-reviewed, and sampled-audited.** Submitters run benchmarks on their own hardware, so the raw numbers *are* self-reported. Trust is manufactured procedurally: a mandatory submission checker validates completeness/correctness; the closed division forces reference-equivalent models, fixed pre/post-processing, no retraining, and accuracy within 99%/99.9% of reference; an encrypted tarball + verifier UI pins what was submitted; a confidential peer-review window lets other submitters file objections that must be resolved; and each round up to two submissions — one at random, one committee-selected from nominations ("new HW/SW, unusual features, performance outside expectations") — receive an independent third-party audit whose report recommends compliance or not. Power submissions require certified measurement gear. Lesson: **self-measurement is acceptable if (1) the artifact is pinned and inspectable, (2) peers can object, and (3) audits are sampled with teeth** (results can be pulled post-publication).

**Papers with Code — pure self-report, no verification, now dead.** PwC aggregated `<task, dataset, metric>` SOTA tables from papers and community submissions with essentially no verification; provenance was a link to the PDF. It created enormous citation-grade gravity for numbers nobody checked. Meta sunsetted it **July 24–25, 2025**; the domain redirects to Hugging Face and the data survives only as static archives (`paperswithcode/paperswithcode-data`; `pwc-archive` on HF: 9,327 benchmark leaderboards, 79,817 paper-code links, frozen). Lesson: an unlabeled self-reported number, once displayed next to verified ones, launderers itself into "truth." If you display self-reported metrics at all, the label must be structural, not a footnote.

**Open LLM Leaderboard — organizer-reproduced, still gamed.** Hugging Face re-ran the EleutherAI harness itself (v1, May 2023), so scores were organizer-reproduced — the strongest practical tier — and *still* suffered: fine-tunes on test sets, contaminated merges climbing the board, and ConStat (Dekoninck et al., NeurIPS 2024, **arXiv 2405.16281**, May 25 2024) detecting contamination in the **top-3 leaderboard models** at submission time, with estimated inflation of 3–8 points on affected benchmarks. Dekoninck et al., *Evading Data Contamination Detection for Language Models is (too) Easy* (**arXiv 2402.02823**, Feb 5 2024) showed Evasive Augmentation Learning — training on rephrased benchmark items — inflates scores while defeating overlap-based detectors. v2 (Jun 26, 2024) responded with harder benchmarks and normalized aggregation (§4), and HF **retired the leaderboard in March 2025**, citing gaming and misleadingness. Lesson: organizer-reproduction fixes *measurement* trust but not *benchmark* trust; contamination and saturation are orthogonal failure modes needing their own audits (§5).

**Chatbot Arena — organizer-collected per-instance votes, undermined by selective disclosure.** The methodology paper (Chiang et al., **arXiv 2403.04132**, Mar 6 2024) is the statistical gold standard: Bradley–Terry coefficients fit by MLE over the full vote matrix, 95% CIs via Huber–White sandwich robust standard errors (chosen over pivot bootstrap after simulation), an active sampling rule with up to 54% sample-efficiency gain, and anomalous-user detection via sequential p-value/e-value combination. But *The Leaderboard Illusion* (Longpre et al., **arXiv 2504.20879**, Apr 29 2025; NeurIPS 2025 D&B track) documented the trust hole: undisclosed private testing let preferred providers evaluate variants privately and retract all but the best — **27 private variants tested by Meta before Llama-4's release**, with simulation and live experiments showing **~+100 Arena points from testing just 10 variants**, because best-of-N retraction violates the BT model's unbiased-sampling assumption. Compounded by data-access asymmetry: the top two providers received an estimated 19.2% and 20.4% of all battle data while 83 open-weight models shared 29.7%, and Arena-distribution access yielded up to +112% relative on ArenaHard with no out-of-distribution gain. Lesson for your design: **who controls which results become visible is as important as how numbers are measured.** Any channel where a participant can try N times and publish the max is a bias injector — including telemetry, where best-of-N becomes "submit the prettiest curve."

**Verification tiers** (use these as first-class labels):

| Tier | Name | Definition | Exemplar |
|---|---|---|---|
| T0 | Self-reported, unverified | Participant computes and reports; no artifact check | Papers with Code, arXiv tables |
| T1 | Self-reported, artifact-backed | Participant reports; pinned code+logs public; community/organizer *can* reproduce, with removal on failed challenge | modded-nanogpt speedrun |
| T2 | Organizer-reproduced | Organizer re-executes measurement on its own infra | Open LLM Leaderboard, Kaggle code rerun |
| T3 | Audited | Independent third party re-executes/inspects with authority to revoke | MLPerf closed-division audits, ACM "Results Reproduced" |

The **modded-nanogpt speedrun** (KellerJordan/modded-nanogpt) deserves special mention because it is the closest existing analogue to your competition — an open, competitive, architecture/optimization-invention leaderboard where records are self-reported wall-clock times. Its trust machinery: logfiles must embed the *complete* training code (third-party optimizer code inlined) so anyone can reproduce by extracting everything before the `===` marker; submissions must ship enough runs for **p < 0.01 statistical significance** that mean val loss meets target (systems-only speedups exempted); official records are **re-timed on organizer-controlled 8×H100 nodes** (Tier 2 promotion of the one number that matters); and failed community reproduction is explicit grounds for record removal — exercised in practice (e.g. PR #151's verification thread: N=26 runs, mean/std, p≈1.1e-05, baseline re-runs to exclude PyTorch-version effects). That is the two-zone pattern in the wild: self-reported for admission, organizer-reproduced for canonization.

---

## 2. Telemetry validation techniques

Ordered from cheapest and hardest to game to most expensive.

**2.1 Cross-checks against organizer-observed ground truth.** Your harness already owns the strongest anchors: it times the run itself, pins the dataset (SHA-256 verified fineweb-edu shard), and computes bpb. Every self-reported series can be checked against organizer-side facts:

- *Wall-clock consistency*: Σ(reported step times) ≤ measured wall-clock × (1+ε). Miner reports 40k steps in a 6h window whose implied step time contradicts the pod's measured duration → flagged.
- *Token accounting*: tokens_seen must equal steps × batch × seq_len as configured, and must be consistent with harness-side dataloader counts (the harness, not miner code, should count tokens pulled from the pinned stream).
- *Loss-curve anchoring*: the harness evaluates final bpb itself; the miner-reported loss series must terminate in a value compatible with that independent eval within a tolerance band (architectures with different objectives need a declared, per-recipe mapping; an unmapped divergence is a flag). Intermediate organizer probes (eval at fixed token checkpoints) strengthen this from endpoint-anchored to curve-anchored — cheap on a pod you already own.
- *Receipt cross-check*: you already have this as the `inconsistent_metrics` cheat code ("bpb impossible vs tokens/wall_clock/receipt"). Generalize it from pass/fail on the scored metric to a validation lattice over all telemetry.

**2.2 Physics / thermodynamic sanity.** Compute implied efficiency from self-reported throughput and reject the impossible. The canonical metric is **MFU — model FLOPs utilization** — from the PaLM paper (Chowdhery et al., **arXiv 2204.02311**, Apr 2022): observed tokens/s divided by theoretical max tokens/s at peak FLOPs, where FLOPs/token = 6N (+ attention term) counts only required forward+backward work, excluding rematerialization. Reference points: GPT-3 21.3%, Gopher 32.5%, Megatron-Turing 30.2%, PaLM 46.2% on TPU v4; well-tuned H100 runs land ~35–55%. Per-GPU-class ceilings give you a hard gate: MFU > ceiling(gpu_type) ⇒ fabricated, mismeasured, or a FLOP-count exploit — and since this is an *architecture* competition, compute MFU from harness-measured wall-clock and organizer-computed FLOPs/token (from the submitted `architecture.py`), never from miner-reported FLOPs. Analogous ceilings: memory bandwidth bounds for claimed activation sizes, power caps for claimed efficiency.

**2.3 Internal consistency of series.** Monotonicity where declared (cumulative counters); step-index contiguity; no duplicate timestamps; gradient-norm series that are bit-constant or perfectly smooth are synthetic; loss/grad-norm correlation structure (a loss plateau with exploding grad norms and no LR change is incoherent); hash-chaining telemetry reports (each report embeds hash of previous) makes post-hoc editing of a series tamper-evident; sandbox clock checks (monotonic vs wall clock) catch replayed logs.

**2.4 Cross-submission statistical outlier detection.** Per metric, across the cohort: robust z-scores (median ± k·MAD) on terminal values; Mahalanobis/Isolation-Forest on feature vectors (final loss, tokens/s, MFU, grad-norm distribution moments); duplicate/near-duplicate series detection across miners (copied curves — you already do AST-copy detection on code; do the same on telemetry). **Critical caveat for an architecture competition: outliers are the product.** A genuinely novel architecture *should* sit off the cohort distribution. Outlier detection must therefore feed a review queue and a display badge ("atypical — under review"), never an automatic zero. Reserve penalties for *provable* dishonesty: physics violations (2.2), contradictions of organizer ground truth (2.1), tampered hash chains, copied series. Those map to your existing agentic cheat path (`inconsistent_metrics` → `Score(0)`); implausible-but-not-impossible telemetry gets quarantined labeling instead (§7).

**2.5 Penalty posture.** MLPerf's sampled audits + public objection window is the right incentive shape: the *possibility* of re-measurement plus the *certainty* of consequence for proven fabrication. Publish the penalty taxonomy in advance (pre-registered, like scoring weights): proven fabrication → terminal rejection; buggy-but-honest telemetry (e.g., off-by-1000 units) → quarantine + fix window; no penalty for weird-but-consistent numbers.

---

## 3. Metric schema design for arbitrary participant metrics

Precedents converge on the same design invariants:

- **OpenTelemetry semantic conventions**: dot-namespaced lowercase names (`gen_ai.client.token.usage`), units in UCUM (`{token}`, `s`), instrument types (counter/gauge/histogram), and a hard cardinality discipline — per-request or per-conversation IDs must *never* be metric dimensions because each unique attribute set materializes a new time series. The GenAI conv's own design debate is instructive: cache/reasoning token splits became *partitioned attributes* (`gen_ai.token.type=input|output` with orthogonal cache/reasoning dimensions) precisely so totals stay additive.
- **MLflow LogBatch limits**: ≤1000 metrics + 100 params + 100 tags per request, ≤1000 entities total, 1 MB request cap, keys ≤250 chars, values typed as f64 with NaN/Inf handled explicitly.
- **W&B operating limits**: ≤100k distinct metric keys per project, ≤500k steps per run, ≤1000 log calls/min, ≤100k values/min; nested dict keys flatten to dot-separated names, which silently multiplies cardinality — "too many distinct metrics, not too many steps" is their #1 performance failure mode.

Schema proposal pattern (concretized for your stack in §7):

- **Namespacing**: `org.*` reserved for organizer-measured (reject at ingest if miner-emitted); `miner.<group>.<name>` for participant metrics. Registry table enforces per-submission key uniqueness and caps.
- **Typed values**: `scalar` (f64), `series` (per-step f64 arrays, server-side downsampled for display), `histogram` (exponential bucket boundaries, OTel-style), plus optional `unit` (UCUM subset) and display-only `direction` (`higher|lower|neutral`). Strings/enums are tags, not metrics — they don't chart.
- **Cardinality limits**: per submission, e.g. ≤64 distinct miner scalar keys, ≤16 series, ≤10k points/series ingested (1k displayed), ≤1 MB per report call, rate-limited calls. Sizing is one order of magnitude under W&B's pain threshold, which is far beyond what a 6h pod run legitimately needs.
- **Versioning**: `schema_version` in every envelope; metric *semantics* versioned by name (a unit change is a new metric name, not a reinterpretation — OTel's rule); the whole contract pinned to your existing `recipe_version` (currently 1.2.0).
- **Storage**: at competition scale (thousands of runs × ≤10k points), Postgres is sufficient and you already run it: registry + scalars in typed columns, series as JSONB arrays (or a hypertable if you later want cross-run time-series analytics — TimescaleDB if you stay in Postgres, ClickHouse only if analytics become the product). Keep the **raw signed payload immutable** (content-addressed) alongside any downsampled display copy; the audit trail is what makes T1-tier trust possible.

---

## 4. Leaderboard design that resists Goodhart

**Aggregation is where gaming enters.** The Open LLM Leaderboard v1 summed raw accuracies; v2 (Jun 2024) switched to normalizing each benchmark between its random-guess baseline (0) and max (100) *before* averaging, explicitly because raw averaging let easy benchmarks dominate and near-random hard benchmarks vanish. The reshuffle was violent (Qwen1.5-32B-Chat: rank 57 → top 10) — proof that the aggregation function, not the models, was producing part of the ranking. *The Emerging Science of Machine Learning Benchmarks* (mlbenchmarks.org, ch. 12) and *How Not to Lie with a Benchmark* (which reorders GLUE/SuperGLUE by swapping arithmetic for geometric/harmonic means, demoting outlier-driven leaders) both document the same sensitivity. Dehghani et al.'s **Benchmark Lottery (arXiv 2107.07002, Jul 2021)** generalizes it: relative performance flips under different task subsets ("task selection bias"), so any single fixed suite is a lottery ticket; mitigations are breadth, robust statistics, and outlier-resistant aggregation. Alzahrani et al., *When Benchmarks are Targets* (**arXiv 2402.01781**, Feb 2024), shows LLM leaderboard rankings flip under minor benchmark perturbations — and that the fix is reporting that fragility rather than hiding it.

**Design responses, with citations:**

- **Multi-metric display over composite collapse.** HELM (Liang et al., **arXiv 2211.09110**, Nov 17 2022) is the canonical precedent: 16 core scenarios × 7 metric categories (accuracy, calibration, robustness, fairness, bias, toxicity, efficiency) measured densely (98/112 pairs) so that trade-offs are *shown* instead of averaged away. If you must publish a single scored number for the chain (your SCORE_MAX lattice requires it), keep it organizer-computed and pair it on the site with the full panel — the composite is for scoring, the panel is for truth.
- **Pareto-front visualization** for quality-vs-efficiency (bpb vs wall-clock/GPU-hours/tokens): render the frontier, don't collapse it; a submission 2% worse on bpb at 40% less compute is a different winner, and the display should say so.
- **Confidence intervals, always.** Miller, *Adding Error Bars to Evals* (**arXiv 2411.00640**, Nov 1 2024): report SEM-based CIs per score; use **clustered** intervals when eval items share passages (naive CIs are anti-conservative); use **paired-difference analysis** for model-vs-model comparisons — per-question score correlation of 0.3–0.7 between frontier models makes pairing a free variance reduction; run a power analysis so the eval is sized to detect the differences you intend to rank on. Chatbot Arena's practice of shipping CIs next to every rating is what made "rank 4 vs rank 5 is noise" sayable in public.
- **Arena-style inference from per-instance data.** BT-from-votes (**arXiv 2403.04132**) shows the value of retaining instance-level records: CIs, anomalous-voter detection, drift analysis all require the unaggregated data. For your admin-judged tracks (design winners) this argues for storing per-judgment records, not just outcomes. And the Leaderboard Illusion (**arXiv 2504.20879**) supplies the counter-principle: BT/Elo validity rests on unbiased sampling — so ban private retraction by construction (every scored run is public, no best-of-N publication channels) and publish the per-instance data so third parties can re-estimate rankings and audit you.
- **Pre-registration of scoring.** Publish scoring function, weights, normalization constants, and penalty taxonomy before the round opens; hash-pin them (your frozen-spec discipline — `BUNDLE_SPEC.md`, `scoring_version`, recipe pins — is already the right mechanism); changes require a version bump with an announcement window. This is the competition analogue of clinical-trial pre-registration and directly defuses "the organizers moved the goalposts" and "the organizers picked the aggregate that favored X."

---

## 5. What to display for an architecture competition

For each published submission (and each registered architecture), in display order:

1. **Training curves** — loss-vs-tokens *and* loss-vs-wall-clock, with the x-axes from organizer counters. Overlay organizer eval probes (independent bpb at checkpoints) as markers on the miner-reported curve; agreement is itself a trust signal, so show it.
2. **Eval suite breakdown** — per-benchmark table with Miller-style CIs (clustered where items share context), not just the headline bpb. Drill-down per benchmark; no hidden subscores feeding the composite.
3. **Efficiency panel** — wall-clock, tokens/s, GPU-hours, organizer-computed **MFU (arXiv 2204.02311)** with the per-GPU ceiling drawn as a line; cost per run (you already track pod costs).
4. **Contamination audit results** — n-gram overlap of training data against eval sets, plus a ConStat-style (**arXiv 2405.16281**) reference-benchmark comparison where feasible; display the *method and its known evasion limits* (**arXiv 2402.02823**) next to the result, because "passed contamination check" overclaims.
5. **Reproducibility badges** — borrow ACM Artifact Review & Badging's scoped, independent badges: *Artifacts Available*, *Artifacts Evaluated–Functional/Reusable*, *Results Reproduced* (independent team, author artifacts), *Results Replicated* (independent team, no author artifacts). The crucial design property (per ACM/CASRAI) is that each badge certifies exactly one scoped claim — a badge is not a general quality seal. NeurIPS's reproducibility program (checklist mandatory since 2019, desk-reject if missing; program evaluated in Pineau et al., *Improving Reproducibility in ML Research*, JMLR 2021) shows the checklist alone moves community behavior. For you: "organizer re-run matched claimed bpb within ε" is a badge you can issue the moment you re-run anything.
6. **Architecture cards** — model cards (Mitchell et al., **arXiv 1810.03993**, Oct 2018, FAT* 2019) for the artifact; datasheets (Gebru et al., **arXiv 1803.09010**, Mar 2018) / data statements (Bender & Friedman, TACL 2018) for training data. Minimum card: params, FLOPs/token, context length, dataset pin + SHA-256, container digest, code hashes, receipt, license. Your top-model publish to `BaseIntelligence/prism` already ships code + METRICS + README — the card formalizes the README.

---

## 6. Abuse vectors in displayed metrics, and display-side mitigations

| Abuse | Mechanism | Mitigation |
|---|---|---|
| Vanity metrics | Miner invents "coherence index: 99.9%" with no definition; number is unverifiable by design | Zone B requires declared unit + direction + schema; no definition → rejected at ingest. Label "self-reported, unverified" is structural (banner + color), not a tooltip |
| Metric flooding | Thousands of keys to crowd the UI, bury bad news, DoS dashboards | Hard cardinality caps at ingest (§3); Zone B renders as a bounded, paginated panel; default view shows organizer-pinned keys only |
| Misleading units/scales | nats vs bits, per-token vs per-batch "loss," truncated y-axes, log scales | UCUM units in schema; display normalizes to declared canonical units; fixed axes on organizer panels; Zone B plots carry auto-generated unit captions |
| Cherry-picked windows | Best 100-step window presented as the run | Series must start at step 0 and be contiguous; display shows full range by default |
| Smoothing/fabricated curves | Averaged-away instabilities; fully synthetic series; copied from another miner | Consistency checks (§2.3–2.4): curve anchored to organizer bpb eval, grad-norm/loss coherence, hash-chained reports, cross-miner duplicate-series detection |
| Best-of-N telemetry | Retry until the curve looks good; publish only that | All runs public (no retraction channel — the Leaderboard Illusion fix); per-hotkey history visible; your one-accepted-submission gating already limits this, keep the failed/retried rows visible |
| Scored-metric laundering | Getting an impressive self-reported number treated as scored | Architectural: Zone B is unreachable from the scoring path (§7); the composite renders only `org.*` inputs |

The general display principle: **organizer-defined fixed panels first (identical layout for every submission), a quarantined "participant metrics" section second (clearly labeled, consistency badges attached), and no pathway by which a Zone B number can be sorted on, aggregated, or averaged into anything.**

---

## 7. Concrete recommendation: the two-zone metric system for PRISM

**Principle.** Every metric exists in exactly one of two zones, distinguished at the schema level, the storage level, and the pixel level. Zone membership determines trust label, validation intensity, and — most importantly — reachability of the scoring path.

**Zone A — organizer-measured, verified, scored.** Computed by the harness on the pod or derived from organizer-side observations. Fixed, closed set, versioned with the recipe:

| Key | Source | Scored? |
|---|---|---|
| `org.eval.bpb` | Harness final eval on pinned shard | Yes (sole score input, SCORE_MAX lattice, unchanged) |
| `org.run.wall_clock_s` | Harness monotonic clock | No (display + validation anchor) |
| `org.run.tokens_seen` | Harness dataloader counter | No (anchor) |
| `org.run.steps` | Harness | No (anchor) |
| `org.run.gpu_type`, `org.run.container_digest`, `org.run.dataset_sha256` | Pod/receipt | No (provenance) |
| `org.eff.mfu` | Derived: harness wall-clock × organizer-computed FLOPs/token from `architecture.py` | No (efficiency panel; ceiling-checked) |
| `org.eval.bpb_at_tokens[k]` | Optional intermediate probes at fixed token checkpoints | No (curve anchors) |
| `org.audit.contamination_*` | Post-run overlap/reference-benchmark scan | No (audit panel; flags feed agentic review) |

**Zone B — participant-reported, displayed-but-labeled, validated, never scored.** Everything arriving through `prism_telemetry.report(...)` beyond reserved keys. This is a strict generalization of the hook you already mandate — `loss`, `step`, `grad_norm`, `layer_stats` become declared Zone B metrics under a schema instead of ad-hoc kwargs.

*Envelope (schema_version pinned to recipe_version):*

```json
{
  "schema_version": "1.3.0",
  "submission_id": "…",
  "seq": 1240,
  "prev_hash": "sha256:…",
  "metrics": [
    {"name": "miner.train.loss", "type": "series", "unit": "{bit}/token",
     "direction": "lower", "step": [0, 50, 100], "value": [9.2, 7.1, 6.4]},
    {"name": "miner.probe.grad_norm", "type": "series", "unit": "1",
     "direction": "neutral", "step": [0, 50], "value": [0.8, 0.31]},
    {"name": "miner.sys.tokens_per_s", "type": "scalar", "unit": "{token}/s",
     "direction": "higher", "value": 41000.0}
  ]
}
```

*Ingest rules:* names must match `miner\.[a-z0-9_]+\.[a-z0-9_.]+`; `org.*` from miner code → reject; types ∈ {scalar, series, histogram}; units from a UCUM subset; per-submission caps (64 scalars, 16 series, 10k points/series, 1 MB/report, rate limit); NaN/Inf flagged; `prev_hash` chain verified against stored head (tamper-evidence).

*Storage (Postgres, extending the existing `prism_telemetry` table rather than adding a TSDB):*

```sql
CREATE TABLE prism_metric_def (
  submission_id   text NOT NULL,
  name            text NOT NULL,          -- miner.* only
  type            text NOT NULL,          -- scalar | series | histogram
  unit            text,
  direction       text,                   -- display hint only
  schema_version  text NOT NULL,
  PRIMARY KEY (submission_id, name)
);

CREATE TABLE prism_metric_value (
  submission_id   text NOT NULL,
  name            text NOT NULL,
  zone            text NOT NULL,          -- 'A' | 'B'  (A rows are harness-written)
  payload         jsonb NOT NULL,         -- series arrays / scalar / histogram
  raw_hash        text NOT NULL,          -- content-addressed, immutable
  validation_status   text NOT NULL DEFAULT 'pending',  -- ok | flagged | quarantined
  validation_findings jsonb,
  created_at      timestamptz NOT NULL DEFAULT now()
);
```

*Validation pipeline (async, post-ingest, before public display):* schema check → consistency lattice → status.

1. `tokens ≤ steps × batch × seq_len`; Σ step times ≤ `org.run.wall_clock_s` × 1.05.
2. `org.eff.mfu` ≤ ceiling(`org.run.gpu_type`); flag any miner throughput series implying MFU above ceiling.
3. Terminal `miner.train.loss` compatible with `org.eval.bpb` under the recipe-declared mapping, tolerance-banded; intermediate probes anchor the curve if enabled.
4. Monotonicity/contiguity for declared counters; no duplicate steps; hash chain intact.
5. Cross-miner duplicate-series detection; robust z-score (median ± 6·MAD) per metric across the cohort.
6. Verdicts: violations of (1)–(3) or a broken hash chain → `quarantined` + routed to the agentic anti-cheat path as evidence (`inconsistent_metrics`-class); (5) outliers → `flagged` ("atypical — under review"), *never* auto-zero; clean → `ok`.

*Promotion rule.* A Zone B metric graduates to Zone A only when the organizer re-measures it: the harness computes the quantity itself in a new recipe version, the metric is renamed `org.*`, and only then may it enter scoring. Example path: miners keep reporting `miner.sys.tokens_per_s` → organizer adds harness-side throughput instrumentation in recipe 1.4 → `org.sys.tokens_per_s` exists → the Zone B key is retired from the default display. **Zone B values never appear in leaves, never enter the SCORE_MAX lattice, and are excluded from any sortable leaderboard column.** Scoring stays exactly what it is today: pure organizer bpb with the existing cheat/copy hard-zeros.

*Display principles (site):*

1. Submission page opens on Zone A panels with identical layout for every row: score + CI, curves with organizer probe overlays, efficiency panel (MFU with ceiling line), provenance (hashes, digest, receipt), contamination audit, badges.
2. Zone B renders below a permanent banner — "Participant-reported · unverified · not used for scoring" — with per-metric status badges (`consistent with organizer measurements` / `atypical` / `quarantined: failed wall-clock cross-check`).
3. Leaderboard sorts and aggregates expose Zone A only; Pareto view (bpb vs wall-clock) uses organizer axes exclusively.
4. No retraction: every scored run stays public with its telemetry and validation history (Leaderboard-Illusion closure), and failed/quarantined rows remain visible to keep the penalty taxonomy credible.

The net effect: miners get a rich, public telemetry channel — genuinely useful for an architecture community that wants to inspect each other's training dynamics — while every incentive to fabricate it is removed, because fabrication is checkable against harness ground truth, physics ceilings, and cohort statistics, and because no self-reported number can ever move a score.

---

*Key sources: Leaderboard Illusion (arXiv 2504.20879, Apr 2025) · Chatbot Arena (2403.04132, Mar 2024) · Miller, error bars for evals (2411.00640, Nov 2024) · Benchmark Lottery (2107.07002, Jul 2021 — correcting the 2407.08702 citation, which is a physics paper) · When Benchmarks are Targets (2402.01781, Feb 2024) · HELM (2211.09110, Nov 2022) · PaLM/MFU (2204.02311, Apr 2022) · ConStat (2405.16281, May 2024) · Evading contamination detection (2402.02823, Feb 2024) · Model cards (1810.03993, Oct 2018) · Datasheets (1803.09010, Mar 2018) · MLPerf inference submission/audit rules (mlcommons GitHub) · Kaggle overfitting meta-analysis (Roelofs et al., NeurIPS 2019) · Open LLM Leaderboard v2 launch (Jun 2024) and retirement (Mar 2025) · Papers with Code shutdown (Jul 2025) · modded-nanogpt records policy · OTel GenAI semantic conventions · MLflow/W&B logging limits · NeurIPS reproducibility program (JMLR 2021) · ACM Artifact Review & Badging.*
