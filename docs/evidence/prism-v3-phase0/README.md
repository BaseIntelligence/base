# Prism v3 Phase 0 — σ_seed and real MFU

> **Evidence, not spec.** Per [`AGENTS.md`](../../../AGENTS.md), nothing here is
> normative. The contract stays [`docs/PRISM.md`](../../PRISM.md) and the
> pre-registered anchor sets. Measured 2026-08-16 on Lium RTX 5090 pods.

## Why this run existed

The v3 dual-cap budget design is gated on two numbers nobody had measured:

1. **σ_seed** — *training*-seed variance. The existing bootstrap resamples
   **eval items only**, so it has never seen training variance at all. A seed
   change alone re-ranks NAS architectures at Kendall τ = 0.48 in the
   literature, which would make the LCB overconfident. If σ_seed exceeds
   typical inter-architecture gaps, **no emission rule can rescue the
   ranking** and runs must be seed-averaged — changing cost per submission and
   the subnet's economics.
2. **real MFU** — every FLOPs constant scales linearly with it. At 15 % real
   MFU the `3.0e18` cap becomes wall-bound and the design partly reverts to
   the status quo.

## What was actually run

| | |
|---|---|
| Baseline | reference Transformer++ (`d=1024, L=24, V=50257`) |
| Seeds | 1001, 1002, 1003 — **only the seed varied** |
| Hardware | 1 × RTX 5090 (32 GB), $0.65–0.69/pod-h |
| Train budget | **REDUCED: `3.0e17` FLOPs**, 1/10 of `TRAIN_FLOPS_CAP` |
| Eval | full battery, `battery_budget_s = 3600`, `truncated: false` |
| Scored against | anchors **v0** (`prereg 581643c7…`), `scoring_mode: benchmarks` |

> ### ⚠ This is a LOWER BOUND on σ_seed, not the full-budget number
>
> The σ_seed runs used a **reduced `3.0e17` training budget** (~6 % of
> `TRAIN_FLOPS_CAP`) to stay inside the spend ceiling. Seed dispersion at
> reduced budget is a valid *lower bound* — longer training generally
> **shrinks** dispersion on aggregate likelihood metrics, but it does not
> follow for the small-item accuracy metrics, and it says nothing about
> whether the gap between two *architectures* also grows. **The full-budget
> σ_seed is still unmeasured.**

## Headline results

### Real MFU — comfortably above the design assumption

**Measured MFU = 40.2 %** (σ = 0.0016, 0.40 % CV across seeds).

The design needed ≥ 20 % to stay FLOPs-bound inside the 5.0 h wall, and
feared 15 %. At 40.2 % a full `3.0e18` budget needs ≈ 2.5 h on 4 GPUs, so it
is **FLOPs-bound with real margin**. This retires the "design reverts to the
status quo" risk.

### FLOPs attestation — the analytic cross-check agrees to 1.1 %

| metric | value |
|---|---|
| `org.diag.flops_per_token_probe` | `2.221e9` |
| `org.diag.flops_attested` | `1.820e17` |
| `org.diag.flops_analytic_ratio` | **1.0113** |
| `org.diag.flops_analytic_gap` | **0.0112** (1.1 %, threshold 25 %) |
| `org.diag.flops_analytic_mismatch` | 0 |
| `org.diag.flops_probe_cv` | **0.0** (threshold 0.15) |
| `org.diag.flops_probe_samples` | 8 |
| `org.diag.flops_physically_possible` | 1 |

The dispatcher-counted number and the independent analytic model of the same
graph agree to **1.1 %**, and `bytes_seen / bytes_per_token × F_tok`
reproduces `flops_attested` to **0.00 %**. Probe CV is exactly 0 across all
three seeds — the probe is deterministic for a fixed secret, as designed.

Note `flops_attested`, `flops_per_token_probe` and `spend_fraction` are
**bit-identical across seeds**. That is correct, not a bug: the FLOPs cap
stops the run at a fixed compute quantity, so every seed spends the same
compute and differs only in *where* the weights land.

### σ_seed — the actual dispersion

Aggregate likelihood metrics are **stable**; small-item accuracy metrics are
**not**.

| metric | mean | σ_seed | CV |
|---|---|---|---|
| `org.g1.bits_per_byte_prose` | 2.00885 | **0.0123** | 0.61 % |
| `org.g1.bits_per_byte_math` | 2.46788 | 0.0136 | 0.55 % |
| `org.g1.bits_per_byte_fresh_crawl` | 1.81868 | 0.0044 | 0.24 % |
| `org.g1.bits_per_byte_key_token` | 0.230904 | 0.0018 | 0.77 % |
| `org.g1.bits_per_byte_code` | 3.19353 | 0.1752 | 5.49 % |
| `org.g6.auc_log_tokens` | 5.5394 | 0.0209 | 0.38 % |
| `org.g7.throughput_toks_s` | 115.689 | 3.218 | 2.78 % |
| `org.g2.lambada_strict_acc` | 0.0 | 0.0 | — (at floor) |
| `org.g3.passkey_acc` | 0.333 | 0.2887 | **86.6 %** |
| `org.g3.induction_acc` | 0.667 | 0.2887 | **43.3 %** |
| `org.g4.modular_acc` | 0.083 | 0.1443 | **173 %** |
| `org.g5.babilong_acc` | 0.100 | 0.1732 | **173 %** |
| `org.g2.boolq_acc` | 0.583 | 0.2887 | **49.5 %** |

The split is explained by item count, not by architecture: the unstable
metrics are measured on **1–2 clusters** (`passkey/f24` is a single cluster,
`mqar` two, `modular` two) while G1 bits/byte integrates over ~8 clusters of
many tokens each. These are reduced-budget eval caps, so the instability is
partly an artifact of this run's eval sizing — but see the next section for
why that still matters.

### The decisive finding: a hard eligibility gate flipped on seed alone

| seed | `g3` group score | `g3_ok` | composite |
|---|---|---|---|
| 1001 | **0.19792** | **false** (floor 0.25) | **0.0** |
| 1002 | ≥ 0.25 | true | 0.0 |
| 1003 | ≥ 0.25 | true | 0.330184 |

Same architecture, same budget, same code — **only the seed differed** — and
seed 1001 fell below the **hard G3 floor** and was forced to composite 0.
The driver is `org.g3.passkey_acc` swinging **0.0 → 0.5** on a single
cluster, plus `induction_acc` 0.5 → 1.0.

This is more important than the continuous σ_seed numbers. A gate is
**binary**: it does not degrade gracefully with variance, it flips. So even
though the *scored* likelihood surface is stable to 0.6 %, a seed change can
still move a submission between `Eligible` and `Ineligible` — i.e. between
full participation and **lattice 0**.

All three runs were `ineligible` regardless, for a reason unrelated to seed
variance: four keys the battery does not emit at this recipe version
(`org.g7.ttft_ms_32k`, `org.g7.tpot_ms_32k`, `org.g7.joules_per_token`,
`org.g8.mup_lr_stability`) tripped `missing_metric` under v0's completeness
gate. **The 0.0 vs 0.330 composite spread is therefore a gate artifact, not
a measurement of composite σ_seed** — composite-level σ_seed at full budget
remains unmeasured.

## Verdict

**For continuous likelihood ranking: yes, σ_seed is small enough.**
σ_seed(bits_per_byte_prose) = **0.0123 bpb**. The design's own reference
point is a **0.02-nat plateau** spanning ~88–236M body params, so seed noise
is *below* the plateau scale the competition must resolve — architecture
ranking on G1 is meaningful at this budget. G1 carries 0.25 composite weight
and is the largest single group.

**For the significance-gated emission rule on PR #168: not yet safe to switch
on.** Two blockers, both measured here:

1. **Gate instability.** The G3 floor flipped on seed alone. A significance
   gate sitting on top of eligibility inherits that binary instability, so
   the same submission can be paid or burned depending on its training seed.
   Either the small-item G3/G4/G5 metrics need enough items to stabilize, or
   the floors need a seed-averaged / CI-aware form before any gate is armed.
2. **The number that matters is still a lower bound.** These are reduced-budget
   single-baseline dispersions. The gate's threshold has to be set against
   *full-budget* σ_seed and against a measured *inter-architecture* gap, and
   neither exists yet — the second baseline (hybrid delta-net) produced no
   metrics in this run.

**Recommended next measurement**, in priority order: (a) full-budget σ_seed on
the same baseline, to convert the lower bound into the real number; (b) the
hybrid delta-net baseline, to get an inter-architecture gap to compare σ_seed
against — without it, "σ_seed is small" has no denominator; (c) eval item
counts for the 1–2-cluster metrics, since those, not training noise, are what
actually moved the gate.

## Cost and pod hygiene

Six pods were rented across the Phase-0 attempts (five 1-GPU at $0.60–0.69,
one 4-GPU at $0.50/GPU-h), all `prism-recipe-v9` on RTX 5090.

**Actual spend ≈ $13** (dominated by three ~1.4 h σ_seed pods plus the
4-GPU pod). Well inside the 8-pod-run ceiling: **6 pods, 0 remaining**.

One pod (`prism-11ff5c042c6f`, `fcea756d-…`) was found **orphaned and still
billing** — its orchestrator had been killed, so it was running unsupervised
with an 8 h `removal_scheduled_at`. It was terminated and all six pod ids
verified gone (`GET /pods/{id}` → 404, active pod list empty). Three
orphaned `prism-challenge serve` processes were also killed, since they hold
the Lium key and can rent.

## Reproducing

```bash
# validate wiring, rent nothing
./deploy/scripts/prism-phase0-seed-variance.sh --dry-run
# real run: requires explicit spend authorization
./deploy/scripts/prism-phase0-seed-variance.sh --confirm-spend
```

Raw per-seed `metrics.json`, `status.json` and harness logs for all six
submissions are preserved outside the repo (they contain full eval cluster
dumps, ~5 MB); the reduced numbers above are the committed record.
