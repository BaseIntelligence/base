# Prism v3 — Scoring Redesign Proposal (Synthesis)

> Status: spike / proposal — NOT normative. Normative contract remains docs/PRISM.md.
> Branch: prism-better. Date: 2026-08-06.
> Research base: 13 appendices in research/ (12 external-research + 1 codebase state).

This document is the single synthesis of the thirteen research appendices under
`research/`. It recommends **one** concrete scoring mechanism for Prism v3. It
changes nothing by itself: `docs/PRISM.md` (scoring_version 2, pure bpb) remains
the normative contract until governance adopts a v3 spec.

---

## TL;DR

Replace bpb-only scoring with a **pre-registered, gated, weighted-geometric
composite over eight capability/efficiency groups**, measured entirely on
organizer infrastructure against **private, procedurally regenerated evaluation
assets**, ranked by **lower confidence bound (LCB)** of the composite, and mapped
onto the existing `[0, SCORE_MAX]` lattice. The eight groups: intrinsic fit
(private bpb, 0.25), downstream core (0.15), recall/state probes (0.10, gated),
reasoning (0.15), long-context (0.15), training/sample efficiency (0.075),
inference efficiency (0.075), stability (0.05, gated). Hard anti-overfit tier:
private fresh-crawl stream, difficulty-matched private mirrors with a
mirror-gap penalty, canary/contamination audits on top submissions. Miners
submit full source trees with custom CUDA/Triton kernels, executed under a
gVisor-hardened, network-deny, fresh-pod sandbox with popcorn-style harness
discipline. Miner-reported metrics live in a schema'd **Zone B** that is
displayed, validated, and structurally barred from scoring. Frontier framing is
honest: a 350M winner is an evidence package (M0) feeding an organizer-funded
M1–M4 milestone ladder, not an Opus-class claim.

---

## 1. Why current scoring fails

Today's pipeline is real and worth keeping: miners submit `architecture.py` +
`training.py` (or training-only against a published `arch_id`), the operator
trains on a rented Lium pod, and the harness — not the miner — computes the
score `[research/01-current-state-implementation.md §2–3]`. The failure is in
what gets measured and on what data:

1. **The metric is one average over a public cut.** The score is pure
   bits-per-byte on 256 texts at indices `[2048, 2304)` of a published shard,
   mapped by `score = SCORE_MAX / (1 + bpb)` `[research/01-current-state-implementation.md §3]`.
   Cross-entropy averages every token equally, while capability lives in a
   small set of key tokens; standard perplexity is near-uncorrelated with
   long-context accuracy, and loss→downstream mappings are predictably fit-able
   in only ~39% of cases `[research/04-evaluation-beyond-loss.md §1]`. Loss is a
   screening metric, not a verdict.
2. **The validation data is gameable by construction.** There is no held-out
   private set: the val cut is part of the published pin, and miners receive
   the full parquet path — anti-overfit today relies on the agentic anti-cheat
   plus seed lattice, explicitly "weak by design"
   `[research/01-current-state-implementation.md §4, §6]`. A model that
   memorizes benchmark-adjacent text scores lower loss with zero capability
   gain `[research/04-evaluation-beyond-loss.md §1(d), §3]`.
3. **bpb is blind to exactly the axes that distinguish architectures.** A model
   that silently truncates its context loses almost nothing on average loss,
   because most tokens are locally predictable
   `[research/05-long-context.md §4, §6.5]`. Finite-state models (SSMs, linear
   RNNs) have provable recall/state-tracking ceilings that average PPL cannot
   see `[research/03-beyond-transformers.md §8 theory corner]`. The current
   score therefore gives **no architecture-diversity signal**: a hybrid with
   genuine long-range capability and a truncated Transformer are nearly
   indistinguishable.
4. **No cost or efficiency axis exists.** Training wall-clock is capped but not
   scored; inference cost is not measured at all; `tokens_seen` in METRICS_JSON
   is hardcoded to 2048, undermining even metrics-consistency checks
   `[research/01-current-state-implementation.md §9]`. Yet frontier adoption of
   new architectures is overwhelmingly efficiency-angled
   `[research/13-frontier-transfer.md §1–2]`.
5. **Single-draw statistics.** One seed, one 6h run, one number: seed/checkpoint
   variance at this scale is substantial, and differences smaller than seed
   noise are not wins `[research/04-evaluation-beyond-loss.md §5]`.

What must survive any redesign: the operator-executed harness, the copy gate /
AST similarity / agentic anti-cheat stack, the telemetry hook, master-only
evaluation, the competition credit (`max(own, owner)`), and the epoch-close
D24 leaf emission path `[research/01-current-state-implementation.md §3, §6–7;
research/02-current-state-docs-contracts.md §1]`.

---

## 2. The recommended scoring mechanism

### 2.0 Design shape

One open division (Prism's ethos is open-recipe), one composite. The pipeline
follows appendix 12's recommended design verbatim — fixed anchors + two-level
means + gated weighted geometric mean + LCB payout + pre-registration +
private final scoring `[research/12-score-aggregation.md §7]`:

```
Step 0  Pre-register (hash-commit) metrics, generators, anchors, gates,
        weights, CI parameters, harness version — before each recipe version.
Step 1  Normalize every metric against FIXED anchors, clipped to [0,1].
Step 2  Two-level group means: average within sub-metric, then across.
Step 3  Lexicographic gates (probes, stability, budgets, CI sufficiency).
Step 4  Composite C = prod_k g_k^(w_k), sum w_k = 1.
Step 5  Clustered bootstrap (B >= 1000) -> SE(C); rank by LCB = C - 1.645*SE.
Step 6  lattice_i = round(SCORE_MAX * LCB_i); ineligible -> Score(0);
        existing max(own, owner) credit + epoch-close emission unchanged.
```

Step 6 deliberately deviates from appendix 12's floor-and-temperature pool
split: the lattice mapping `SCORE_MAX × LCB` is **absolute** (fixed-anchor),
so adding weak or strong submissions to an epoch batch cannot rescale anyone
else — the field-composition attack that rules out min-max, z-score, ranks,
and Elo under economic adversaries `[research/12-score-aggregation.md §1a, §3]`.
It also preserves all existing emit machinery (D24-complete sets,
first-write-wins, `Score(0)` never setting arch best)
`[research/02-current-state-docs-contracts.md §1, §5]`.

### 2.1 Evaluation data tier (anti-overfit foundation)

Three tiers, on the ARC Prize / Kaggle pattern `[research/04-evaluation-beyond-loss.md §7]`:

- **Public dev tier.** The current published shard and published probe
  generators (rotating seeds). Miners iterate here; it is explicitly assumed
  contaminated and is never scored.
- **Semi-private feedback tier.** A rotating fresh-crawl text stream plus
  generator instances from the *public* template family with rotated seeds;
  used for any scores shown before finalization.
- **Private decisive tier.** Fully private assets: the fresh-crawl stream's
  held-out portion, a **private template family** (shifted vocab, lexicons,
  distractor statistics) for all procedural generators, and difficulty-matched
  **mirrors** of the public anchors. All scored numbers come from this tier,
  computed once by the harness `[research/12-score-aggregation.md §7 step 6;
  research/05-long-context.md §3 anti-memorization protocol;
  research/10-reasoning-small-scale.md §6]`.

Because every submission is already organizer-executed, "private final eval is
the only scored round" costs Prism nothing — miners never computed their own
score in the first place. Private eval assets are streamed harness-side and
never written to a miner-readable path (§3)
`[research/08-custom-kernels-sandbox.md §7.2]`.

### 2.2 The benchmark suite (groups G1–G8)

Every choice below traces to the appendices; the scale reality (≤350M params,
≤6h, single GPU, GPT-2 tokenizer kept) rules out MMLU/MMLU-Pro/GPQA/MATH/BBH —
at-chance noise at this scale `[research/04-evaluation-beyond-loss.md §2;
research/10-reasoning-small-scale.md §1]`.

| Group | Content | Source |
|---|---|---|
| **G1 Intrinsic fit** | bpb on a multi-domain held-out suite + the private fresh-crawl stream; key-token-weighted variant (harness-defined key tokens, zero circularity); per-position loss decomposition | `[research/04-evaluation-beyond-loss.md §6.1, §8 T0]`, `[research/05-long-context.md §4]` |
| **G2 Downstream core** | LAMBADA, HellaSwag, PIQA, ARC-E/C, WinoGrande, BoolQ, OBQA — loglikelihood `acc_norm`, 0-shot, frozen prompts (OLMES conventions); paired private mirrors | `[research/04-evaluation-beyond-loss.md §2, §8 T1]` |
| **G3 Recall / state probes** | MQAR sweep, copying (Repeat-After-Me) with gap sweep, induction-head probe, S5 permutation composition, passkey — synthetic vocab, fresh seed per round | `[research/05-long-context.md §3]`, `[research/10-reasoning-small-scale.md §4]`, `[research/04-evaluation-beyond-loss.md §4, §6.3]` |
| **G4 Reasoning** | Templated arithmetic (GSM-Symbolic-style, clause-count tiers, NoOp distractors, operand-range extrapolation); deductive closure (ProofWriter-style, depth 0–3, proof-step accuracy); boolean expressions; Dyck-k with length split; modular arithmetic with held-out operand ranges; small-N Knights & Knaves with perturbation suite. Every exact-match metric paired with a likelihood companion | `[research/10-reasoning-small-scale.md §2, §4, §7]`, `[research/11-sample-efficiency-scaling.md §7]` |
| **G5 Long-context** | Private-template RULER pack (NIAH variants, variable tracking, freq-words), BABILong qa1–qa5 with same-grammar distractors, GraphWalks, MRCR-style ordering, NoLiMa-style latent needles, on a 1k–32k grid; per-position loss + context-gain on natural long docs; self-normalized effective length \(L^*\) with an absolute floor | `[research/05-long-context.md §2–4, §7]` |
| **G6 Training / sample efficiency** | Loss-vs-tokens curve from organizer intermediate probes: area-under-curve over log-tokens + tokens-to-threshold at 2–3 pre-registered levels; organizer-side token counter (fixed, seeded stream) | `[research/11-sample-efficiency-scaling.md §4, §8]`, `[research/06-training-cost.md §7–8]` |
| **G7 Inference efficiency** | Organizer-run on the same pod after training: TTFT and TPOT-vs-context (batch 1, to 32k), throughput/goodput at concurrency {1, 32}, peak VRAM, state card (slope, intercept, effective bytes/token @32k, measured-vs-analytic within 10%), J/token; W4A16/W8A8 quant deltas on a mini-suite | `[research/07-inference-efficiency.md §2, §4, §8]` |
| **G8 Stability** | Fraction of seeds completing without divergence; loss-spike count; µP LR-stability (optimal LR within ±2× across a 4× width sweep at tiny scale, automated) | `[research/11-sample-efficiency-scaling.md §3, §8]`, `[research/13-frontier-transfer.md §7.A4]` |

Reported but **never scored** (metric card, not lattice): FLOPs, MFU, kWh/CO₂e,
total vs active params — FLOPs are ill-defined for SSM/MoE/recurrent
architectures and belong in context, not in the reward
`[research/06-training-cost.md §1.2, §8.3; research/07-inference-efficiency.md §8.2]`.

### 2.3 Normalization, gates, composite, weights

**Per-metric normalization (fixed anchors, clipped \([0,1]\))**
`[research/12-score-aggregation.md §7 step 1]`:

- Accuracy-like:
  \[ \tilde{x} = \mathrm{clip}_{[0,1]} \frac{x - c}{1 - c} \]
  with \(c\) the chance floor (\(1/\#\text{choices}\); 0 for generative).
- bpb per domain \(d\) (log-ratio between fixed anchors):
  \[ \tilde{b}_d = \mathrm{clip}_{[0,1]} \frac{\ln b_{\text{chance}} - \ln b_d}{\ln b_{\text{chance}} - \ln b_{\text{ref}}} \]
  with \(b_{\text{chance}} = \log_2 |\mathcal{V}|\) and \(b_{\text{ref}}\) the
  pre-registered reference architecture's bpb.
- Efficiency: capped log-ratio to the reference recipe; stability: bounded
  construction as in appendix 12.

Anchors are constants of the spec, measured on **two reference architectures
the operator trains itself** under the identical 6h budget: a strong
Transformer++ *and* a 3:1 delta-net/attention hybrid — beating a 2017 vanilla
Transformer is no longer informative `[research/03-beyond-transformers.md §9
rec 1]`, and the reference must be refreshed yearly to stay 2026-strength
`[research/13-frontier-transfer.md §7.A1]` (the current ~12M TinyGPT baseline
`[research/01-current-state-implementation.md §5]` is retired as a scoring
anchor).

**Gates (lexicographic, before any scoring)** `[research/12-score-aggregation.md §7 step 3]`:
ineligible (share zero) unless \(g_3 \ge 0.25\) (kills "great loss, zero
recall" degenerate wins), \(g_8 \ge 0.5\) (stability floor), all budget
constraints met (≤350M params, ≤6h wall, organizer token cap), and every
axis's clustered 95% CI half-width ≤ \(\delta = 0.05\). Existing hard-zeros
(agentic cheat/suspicious, copy gate, similarity) are unchanged and applied
first `[research/01-current-state-implementation.md §6]`.

**Composite** `[research/12-score-aggregation.md §7 step 4]`:
\[ C = \prod_{k=1}^{8} g_k^{\,w_k}, \qquad \sum_k w_k = 1 \]

| Group | G1 | G2 | G3 | G4 | G5 | G6 | G7 | G8 |
|---|---|---|---|---|---|---|---|---|
| Weight | 0.25 | 0.15 | 0.10 | 0.15 | 0.15 | 0.075 | 0.075 | 0.05 |

Justification: appendix 12's suggested fixed-division weights are
(0.30, 0.20, 0.10, 0.15, 0.15, 0.10) over G1–G5+G8, with G6/G7 added at 0.10
each and renormalized in the open division `[research/12-score-aggregation.md
§7 step 4]`; the table above is that shape, rounded, with quality axes at 0.80.
G1 keeps the largest share because private multi-domain bpb remains the most
reliable single signal at this scale — but is cut from 1.00 to 0.25 because
loss is screening, not verdict `[research/04-evaluation-beyond-loss.md §1]`.
G4/G5 at 0.15 each because reasoning-through-length and long-context retrieval
are the axes where architectures genuinely differ and the one frontier-relevant
claim a small competition can honestly make
`[research/13-frontier-transfer.md §4]`. G6/G7 are scored, not just reported,
because frontier adoption is efficiency-angled
`[research/13-frontier-transfer.md §1–2]` — but small, because cost axes must
not dominate a quality competition `[research/06-training-cost.md §8.3]`. The
geometric mean is chosen over arithmetic because weak axes carry the highest
marginal elasticity \(w_k / g_k\): stuffing a saturated axis yields ~nothing,
lifting a weak axis yields the most — the exact incentive alignment wanted —
and gates absorb the zero-collapse risk `[research/12-score-aggregation.md
§1b, §3, §7 step 4]`.

**Statistics** `[research/12-score-aggregation.md §7 step 5;
research/04-evaluation-beyond-loss.md §5]`: clustered bootstrap (clusters =
domain/passage/template family, \(B \ge 1000\)) recomputing the entire
pipeline per resample → SE(C); payout ranking by
\[ \mathrm{LCB}_i = C_i - 1.645\,\mathrm{SE}(C_i) \]
so small-sample luck is priced at zero. Item-level clustering uses Miller's
protocol (naive SEs are up to 3× too small; paired per-question comparisons
exploit the 0.3–0.7 cross-model correlation). One training seed per submission
by default (pod economics); the organizer re-runs top-of-epoch submissions on
2 fresh seeds and reports mean ± std before they are cited as winners
`[research/04-evaluation-beyond-loss.md §5; research/08-custom-kernels-sandbox.md §7.4.7]`.

**Sensitivity governance**: publish the ranking under leave-one-metric-out and
±15% weight jitter; weights are chosen inside a Kendall \(\tau \ge 0.9\)
stable neighborhood, and the jitter report ships with every round — the
Benchmark Lottery lesson `[research/12-score-aggregation.md §4]`.

### 2.4 Anti-overfitting tier (scored, not advisory)

- **Mirror-gap penalty.** Private difficulty-matched mirrors of the G2 anchors
  and the G4 arithmetic tier (GSM1k pattern — the single most deployable
  contamination-adjusted scoring idea). The per-submission gap
  \(\max(0,\ (x_{\text{public}} - x_{\text{mirror}}) - \tau_m)\) is deducted
  from G2/G4 group scores before the composite, with \(\tau_m\)
  pre-registered `[research/04-evaluation-beyond-loss.md §3, §7;
  research/10-reasoning-small-scale.md §6]`.
- **Procedural regeneration.** Every G3/G4/G5 item is freshly instantiated per
  scoring round from private seeds; answers are derivable offline from the
  seed; memorization is structurally useless `[research/05-long-context.md §3;
  research/10-reasoning-small-scale.md §6]`.
- **Truncation audits.** Counterfactual needle corruption (an honest
  full-context model's target loss must move when a remote needle is
  corrupted; a truncator's does not), per-position loss curves (truncators
  show a step at their window), and key-token-only scoring
  `[research/05-long-context.md §6.5]`.
- **Top-k contamination audits.** Canary GUIDs in all eval artifacts; n-gram
  overlap attestation against the declared corpus; Min-K%++; Oren
  order-permutation test (works down to 1.4B/1,000-example sets, so cheap at
  350M); guided-instruction probes. Membership-inference alone is near-chance
  and is never the sole detector `[research/04-evaluation-beyond-loss.md §3,
  §7–8]`.
- **Runtime verification, not good faith** — the ArchAgent rule: any invariant
  enforced only by documentation will be exploited; prompt hashes,
  output-position alignment, and state-byte accounting are checked at runtime
  `[research/05-long-context.md §6.4]`.
- **No best-of-N channels.** One accepted submission per hotkey (existing
  gating), every scored run published permanently, no retraction — the
  Leaderboard Illusion closure `[research/01-current-state-implementation.md §7;
  research/09-miner-metrics-leaderboards.md §1, §4]`.

### 2.5 Cost of the battery

The long-context battery at 3B costs ≈4.5–5 L40S-hours
`[research/05-long-context.md §7]`; at 350M the full G1–G8 eval (plus the G7
inference pass) is estimated at **1.5–3 pod-hours per submission** against the
6h training run — a 25–50% pod-cost increase, partially offset by IRT-based
item subsetting for intermediate feedback `[research/04-evaluation-beyond-loss.md §5]`.
This is the single largest operational cost of v3 and is priced into the
migration plan (§6).

---

## 3. Custom kernels and sandboxed execution

Miners may submit a **full source tree** — training loop, architecture, and
their own CUDA/Triton/TileLang kernels, A to Z — replacing the two-script,
128 KiB contract `[research/01-current-state-implementation.md §2]`. The
execution model follows appendix 08's concrete design
`[research/08-custom-kernels-sandbox.md §7]`:

**Sandbox model (per scored run).**
1. **Fresh single-tenant Lium pod per scored run** (today's pod-per-eval
   discipline, formalized), ECC on, `nvidia-smi -q` snapshot in the run
   manifest; JIT/autotune caches reset between runs (cross-run hygiene is a
   fairness control, not just security) `[research/08-custom-kernels-sandbox.md §3.2, §7.2]`.
2. **gVisor `runsc --nvproxy`** container (Kata+VFIO only if bare metal is ever
   available); NVIDIA Container Toolkit ≥ 1.17.8 with the
   `enable-cuda-compat` hook disabled (CVE-2025-23266/23267)
   `[research/08-custom-kernels-sandbox.md §3.1–3.2]`.
3. **Network: default-deny egress.** Today's "dataset URL + tokenizer only"
   posture `[research/01-current-state-implementation.md §4]` becomes fully
   offline: the pinned dataset, tokenizer, and all private eval assets are
   pre-staged or streamed by the harness from outside the sandbox
   `[research/08-custom-kernels-sandbox.md §7.2.3–7.2.4]`.
4. **Read-only root FS**, size-capped tmpfs scratch, no host mounts, no
   secrets in the container, cgroups v2 limits, per-phase timeouts
   (compile/train/eval) `[research/08-custom-kernels-sandbox.md §7.2.5]`.
5. Miner code runs in a **spawned subprocess**, never in the harness process;
   results return over a dedicated FD; harness/reference files are unreadable
   before miner code starts (popcorn pattern)
   `[research/08-custom-kernels-sandbox.md §1.1, §7.2.6]`.
6. **Digest-pinned base image** with the toolchain miners need: CUDA/nvcc,
   PyTorch, Triton, CUTLASS/CuTe, TileLang, FLA (+mamba-ssm, flash-attn).
   Vendored pure-source deps allowed via hash-locked manifest; **prebuilt
   binaries banned** (no `ctypes`/`cuModuleLoadData`/embedded cubins); all
   native code compiles from source in-sandbox within a counted, capped
   compile budget `[research/08-custom-kernels-sandbox.md §6, §7.1, §7.3]`.

**Cheat detection (the harness discipline that makes kernels scoreable).**
Secret seeds (delivered via env, unset immediately, Cantor-combined with
public seeds); deep-cloned inputs; fresh data per timed rep with per-rep
correctness rechecks; NaN/Inf guard buffers; determinism double-run; **pointer
poisoning**; thread/stream/timer audits (hybrid CUDA-event + full-sync timing,
ratio > 1.5× flags); identity+address checks on timing functions post-import;
static AST audit + kernelguard-style rules pre-execution; a dynamic-integrity
sample under profiling to prove claimed kernels actually executed on the timed
path. Assume a double-digit percentage of optimizing submissions attempt
something (SOL-ExecBench measured 14.5%) `[research/08-custom-kernels-sandbox.md
§1.1, §2, §7.4]`. Post-hoc: the organizer **re-runs prize winners plus a random
sample** on fresh pods with new seeds (MLPerf audit pattern), and all scored
submissions are open-sourced after the round for community audit
`[research/08-custom-kernels-sandbox.md §7.4.7]`.

**Attribution / ablation protocol.** End-to-end gain = architecture × kernels
× tuning. Submissions must expose kernels behind a documented, swappable
interface, enabling the organizer-run 2×2 matrix — reference architecture ±
submission kernels, submission architecture ± reference kernels. Cell B
isolates pure kernel contributions and must pass correctness under **hidden
shapes** (shape-specialized kernels get zero kernel credit). The headline
score remains the combined submission; kernel and architecture subscores are
published for credit, with a dedicated systems track kept as a display
leaderboard, not a separate emission pool `[research/08-custom-kernels-sandbox.md §5, §7.5]`.

**Tuning fairness.** AlgoPerf-style: either a declared, capped HP-search
budget with sweep GPU-hours added to a public total-cost ledger, or
on-the-clock tuning inside the 6h; µP/spectral parametrization is strongly
encouraged and its LR-stability gate is scored in G8
`[research/06-training-cost.md §4, §8.3; research/11-sample-efficiency-scaling.md §3]`.

---

## 4. Miner-reported metrics (two-zone system)

Every metric exists in exactly one of two zones, distinguished at the schema,
storage, and pixel level `[research/09-miner-metrics-leaderboards.md §7]`:

**Zone A — organizer-measured, verified, scored.** The fixed closed set behind
G1–G8 plus provenance/audit keys (`org.run.wall_clock_s`,
`org.run.tokens_seen` from the harness dataloader counter — fixing today's
hardcoded 2048 `[research/01-current-state-implementation.md §9]` —
`org.run.gpu_type`, container digest, dataset SHA, `org.eff.mfu`, intermediate
probes, contamination-audit outputs). Only Zone A keys enter the composite and
the lattice `[research/09-miner-metrics-leaderboards.md §7]`.

**Zone B — participant-reported, displayed, validated, never scored.** The
existing `prism_telemetry.report(...)` hook
`[research/01-current-state-implementation.md §6]` is generalized into a
schema'd envelope: `schema_version` pinned to the recipe version, hash-chained
reports (`prev_hash`), typed values (`scalar` / `series` / `histogram`), UCUM
units, display `direction` hints, names matching `miner.<group>.<name>`
(`org.*` from miner code is rejected at ingest), and cardinality caps (≤64
scalars, ≤16 series, ≤10k points/series, ≤1 MB per report call)
`[research/09-miner-metrics-leaderboards.md §3, §7]`.

**Validation before display** `[research/09-miner-metrics-leaderboards.md §2, §7]`:
token/step/wall-clock consistency against organizer ground truth; physics
ceilings (implied MFU above the GPU-class ceiling → fabricated); terminal
loss anchored to the organizer's own bpb within a tolerance band; hash-chain
integrity; cross-miner duplicate-series detection; cohort outliers (median ±
6·MAD). Verdicts: physics/ground-truth violations or a broken hash chain →
`quarantined` and routed to the agentic anti-cheat path as evidence;
statistical outliers → `flagged` ("atypical — under review"), **never
auto-zero**, because in an architecture competition outliers are the product.
The existing `missing_telemetry_hooks` hard violation is unchanged: the
required hooks become reserved Zone B keys with the same terminal semantics.

**Display rules** `[research/09-miner-metrics-leaderboards.md §5–7]`: submission
pages open on fixed-layout Zone A panels (score + CI, curves with organizer
probe overlays, efficiency panel with the MFU ceiling line, provenance,
contamination audit, badges); Zone B renders below a permanent
"Participant-reported · unverified · not used for scoring" banner with
per-metric status badges; leaderboard sorts/aggregates expose Zone A only;
Zone B numbers are unreachable from any sortable column, aggregation, or the
scoring path; a Zone B metric graduates to Zone A only when the organizer
re-measures it in a new recipe version and renames it `org.*`.

---

## 5. Frontier relevance and the milestone ladder

**Honest transfer framing.** The historical record: small-scale research
transfers when it is *mechanism-level*, *efficiency-angled*, *kernel-backed*,
and *iso-recipe-validated at ≥1B–3B* — Gated DeltaNet → Qwen3-Next, KDA → Kimi
Linear, MLA → the DeepSeek lineage, Muon → Kimi K2 are the confirmed cases;
pure-SSM replacements, early linear attention, RetNet, and most NAS winners
are the counter-examples `[research/13-frontier-transfer.md §1]`. What a
350M/6h result can honestly claim: better loss/compute constants, better
loss-vs-context scaling, better HP transfer, better throughput — the
properties measurable at this scale that historically transferred. What it
cannot claim: agency, instruction following, knowledge, reasoning RL — those
are scale + data + post-training, >70% of frontier capability
`[research/13-frontier-transfer.md §4]`. "Challenging Opus-class" means a
winning *block* ends up inside an Opus-class model via the ladder below, not
that a 350M winner beats anything; miner-facing docs must say so verbatim, in
the BabyLM/speedrun tradition where framing honesty built durable credibility
`[research/13-frontier-transfer.md §6–7.C]`.

**Milestone ladder** `[research/13-frontier-transfer.md §7.B]`:

| Milestone | Content | Cost / funding | Gate |
|---|---|---|---|
| **M0** | Competition win at 350M: composite victory under fixed budget, ≥3-seed re-run, recall probes passed, open code | existing | — |
| **M1** | Organizer/treasury-funded 1.5B replication, iso-recipe (~100B tokens) | ~$15–30K | gap holds within noise → preprint with miner as lead author (the real prize) |
| **M2** | 3B validation (~300B tokens), loss-vs-context curves to 64K+, FLA/vLLM-merged fused kernel, ablations, PostNAS-style graft demo into an open 2–8B model | ~$80–150K; treasury + cloud research credits | publishable scaling story + merged kernel → shop to labs |
| **M3** | 30B-A3B pilot (~1T tokens) | ~$1–2M; **partnership-gated only** | lab/ecosystem-fund sponsorship; treasury does not pretend to fund this |
| **M4** | Frontier flagship | out of the competition's hands | — |

The prize-release evidence gates for winners — multi-tier scaling runs with
challenger exponent ≥ baseline exponent, loss-vs-context curves, stability
report with µP transfer and zero retuning, an open fused kernel ("no kernel,
no top prize" — the adoption bottleneck), per-component ablations, and the
graft demo — are exactly the properties the 2024–2026 record says transferred
`[research/13-frontier-transfer.md §2, §7.A]`.

---

## 6. Migration plan

Each phase is a `prism-recipe` version bump (the pin rule: any parameter
change bumps the recipe version so old leaves stay unambiguous
`[research/02-current-state-docs-contracts.md §2]`); `SCORING_VERSION` bumps
2 → 3 at Phase 3, when the composite replaces `score_from_bpb` — the v1→v2
precedent of reallocating scoring inside the same lattice without a
chain-facing bundle change is preserved (bundle `protocol_version` stays 1)
`[research/02-current-state-docs-contracts.md §1, §5]`. Miner-facing docs
(`docs/external-miner/`, public `BaseIntelligence/prism` repo) update at every
phase per the repo's mandatory sync rule
`[research/02-current-state-docs-contracts.md §6]`.

- **Phase 1 — Foundations (recipe 1.3.0).** Organizer-side token/step/loss
  counters streamed off-pod (fixes the hardcoded `tokens_seen`); harness
  intermediate probes at fixed token checkpoints; private fresh-crawl val
  stream v0 replaces the published val cut; reference Transformer++ and hybrid
  baselines trained and published; G6 curve data collected. Scoring remains
  bpb (now on private data). Ships first: it is pure harness work, no
  composite, and immediately closes the public-val hole.
- **Phase 2 — Two-zone metrics + shadow composite (recipe 1.4.0).** Zone B
  schema ships (existing hooks become reserved keys); procedural generators
  published (public dev family); G1–G5+G8 measured and the composite computed
  and displayed **as a shadow** on dashboards; leaves still bpb. Purpose:
  calibrate anchors, gates, and mirror tolerances against live submissions
  before money rides on them.
- **Phase 3 — Composite becomes the score (SCORING_VERSION = 3, recipe
  2.0.0).** Gates + weighted geometric composite + LCB become the lattice;
  private mirror v1 + mirror-gap penalty live; sandbox v2 (gVisor, FD result
  channel, secret seeds, pointer poisoning, cheat battery); full source-tree
  submissions with custom kernels and the 2×2 attribution ablation;
  pre-registration hash-commit published. Top-of-epoch 3-seed re-runs begin.
- **Phase 4 — Efficiency + ladder (recipe 2.1.0).** G7 inference harness on
  the pod (TTFT/TPOT/throughput/state card/energy/quant probe) enters the
  composite at the §2.3 weights; M0→M1 ladder funding goes to governance;
  weights/anchors re-registered with the first full sensitivity annex.

Backwards compatibility: the architecture registry and training-only
submissions persist, but published archs are re-scored under the new recipe on
their next training-only run (recall gates apply — pure finite-state archs
that silently dropped retrieval can no longer win); epoch-close emission,
`max(own, owner)` credit, `Score(0)`/`NoScore` semantics, and the burn
fallback are untouched `[research/01-current-state-implementation.md §3, §7;
research/02-current-state-docs-contracts.md §1]`.

---

## 7. Risks and open questions

1. **Pod cost.** The battery adds an estimated 25–50% pod-time per submission
   on top of 6h training (§2.5); with eval concurrency defaulting to 1, master
   throughput, not epoch length, stays the bottleneck
   `[research/01-current-state-implementation.md §7, §9]`. Mitigations: IRT
   item subsetting, shared forward passes across G1/G5 loss metrics, budget
   re-tiering.
2. **Calibration risk.** At 350M, many tasks sit near the floor; each band
   needs ≥3 subtasks in the 20–80% discriminative window, re-tiered between
   rounds via generator knobs `[research/10-reasoning-small-scale.md §7;
   research/11-sample-efficiency-scaling.md §7]`.
3. **Mirror miscalibration.** A too-easy private mirror manufactures false
   "contamination" gaps for honest miners; difficulty-matching (human or
   reference-model solve rates) is required before the penalty goes live
   `[research/04-evaluation-beyond-loss.md §3]`.
4. **Goodharting the published form.** Competitors optimize the exact
   functional form published (Jigsaw lesson); mitigated by gates, the
   geometric mean, saturation tripwires, the sensitivity annex, and
   pre-registered weight-jitter stability — but never eliminated
   `[research/12-score-aggregation.md §2–4]`.
5. **gVisor/nvproxy compatibility** with pinned NVIDIA drivers on Lium is
   unverified; a fallback of hardening today's SSH-pod model (subprocess + FD
   + network deny + read-only FS, without runsc) ships Phase 3 if nvproxy
   fails `[research/08-custom-kernels-sandbox.md §3.2]`.
6. **Non-autoregressive submissions** (diffusion LMs) break the
   autoregressive-eval assumptions (TTFT, per-token loss); paradigm knobs and
   time-to-first-committed-token redefinitions exist
   `[research/07-inference-efficiency.md §5]` but v3 initially scopes to
   AR-compatible logits, with diffusion support an open work item.
7. **Single-tier blindness.** One 6h/350M slice cannot separate scaling
   exponent from offset; the fitted-frontier approach (3 tiers) is deferred
   to the M1/M2 ladder because in-competition multi-tier triples train cost
   `[research/11-sample-efficiency-scaling.md §1.3–1.4, §8;
   research/13-frontier-transfer.md §7.A]`. Open question: is a second
   in-competition tier (e.g., 100M) affordable later?
8. **Private-asset ops burden.** Fresh-crawl streams, mirror construction,
   seed rotation, and generator maintenance are recurring organizer work with
   governance (hash-commit) overhead `[research/04-evaluation-beyond-loss.md §7;
   research/10-reasoning-small-scale.md §6]`.
9. **Ladder funding.** M1 (~$15–30K) and M2 (~$80–150K/season) need a
   treasury/credits decision; M3 is partnership-gated and must not be promised
   `[research/13-frontier-transfer.md §5, §7.B]`.
10. **Emission drift.** Challenges.toml already diverges from older
    prism-10000 docs; v3 text changes must keep `docs/PRISM.md`,
    `PRISM_RECIPE.md`, external-miner mirrors, and the public repo coherent in
    one pass `[research/02-current-state-docs-contracts.md §6]`.

---

## 8. Appendix index

| File | Content |
|---|---|
| `research/01-current-state-implementation.md` | Codebase map: submissions, harness, bpb scoring, gates, emission, gaps |
| `research/02-current-state-docs-contracts.md` | Normative docs, recipe pins, CI doc gates, emission drift |
| `research/03-beyond-transformers.md` | 2023–2026 architecture survey (SSM, linear RNN, hybrids, memory, MoE, diffusion); ranked competition substrates |
| `research/04-evaluation-beyond-loss.md` | Why loss fails; small-model benchmark suite; memorization/contamination toolkit; iso-compute fairness; statistical rigor; recommended eval stack |
| `research/05-long-context.md` | Long-context benchmark landscape, extrapolation methodology, mechanistic probes, loss metrics, efficiency-at-length, concrete battery |
| `research/06-training-cost.md` | Cost-metrics taxonomy (why FLOPs lie), speedrun methodology, iso-compute protocols, µP fairness, hardware pinning, concrete measurement plan |
| `research/07-inference-efficiency.md` | TTFT/TPOT/throughput definitions, state cards, Pareto reporting, quantization robustness, gaming mitigations, concrete protocol |
| `research/08-custom-kernels-sandbox.md` | Kernel-competition prior art, cheat taxonomy, sandboxing stack, fair timing, 2×2 kernel attribution, concrete secure-execution design |
| `research/09-miner-metrics-leaderboards.md` | Trust models for self-reported metrics, telemetry validation, metric schema, anti-Goodhart leaderboard design, two-zone recommendation |
| `research/10-reasoning-small-scale.md` | What reasoning is measurable at 100M–3B; procedural generators; theory-of-computation probes; concrete reasoning battery |
| `research/11-sample-efficiency-scaling.md` | Scaling-law fitting, loss→downstream honesty, µP substrate, intermediate-checkpoint scoring, BabyLM lessons, tiered protocol |
| `research/12-score-aggregation.md` | Aggregation taxonomy, real leaderboard failures, manipulation-resistance ordering, the recommended steps 0–6 formula |
| `research/13-frontier-transfer.md` | Small→frontier transfer record, scaling-exponent evidence, honest capability gap, evidence package, M0–M4 ladder |
