# v3 G2 changes to route — anchor set + sibling-owned files

> Routing request for the Prism v3 work (`docs/spikes/prism-v3/`). Produced
> 2026-08-16. Not a research appendix and not normative — a work item.
>
> **Status: a request, not a change.** Everything in this file is in a
> sibling-owned path (`crates/prism-recipe/anchors/*.json`,
> `crates/prism-eval-store/src/finalize.rs`). Nothing here has been applied on
> this branch. The harness-side support each item needs **is** implemented and
> tested here — see the last column.
>
> Non-normative. `docs/PRISM.md` and the pre-registered anchor sets remain the
> contract.

## 1. G2 task set and weights for `anchors/v3.json`

### The problem, in one line

Four of G2's eight sub-metrics normalize to a **constant 0 for the entire
field** at this operating point, and G2's sub-metrics are **equal-weighted**
(`composite.rs::default_metric_weight` = 1 each, then weight-normalized within
the group). So those four carry **half of G2's 0.15 composite weight — 7.5 % of
the whole composite — while measuring nothing.**

Worse than dead weight: the composite is a weighted **geometric** mean, so an
axis pinned at 0 collapses `C` entirely unless the normalization floors it. The
four dead tasks therefore contribute noise-free zeros to a group whose remaining
range is already the narrowest in the battery.

### Which tasks and why

| Task | Metric key | Chance | Verdict at ~160 M / 6 h |
|---|---|---|---|
| Winogrande | `org.g2.winogrande_acc` | 0.50 | **At chance.** Separating two submissions needs ~**76 824** items; the set has **1 267**. Not fixable by raising the cap — the items do not exist. |
| OpenBookQA | `org.g2.obqa_acc` | 0.25 | **At chance.** |
| ARC-challenge | `org.g2.arc_challenge_acc` | 0.25 | **At/below floor** for the whole field. |
| BoolQ | `org.g2.boolq_acc` | 0.50 | **At/below floor**; majority-class answering sits at chance. |

At n=200 and p=0.5 the 95 % CI half-width on a *difference* is **±9.8 pp**
against FLOP-matched expected margins of 1–3 pp. No procedure — LCB, pairing,
bootstrap — recovers resolution the item count does not contain, and for
Winogrande the item count cannot contain it at any cap.

The four that do move (already at `PRISM_EVAL_G2_CAP_USABLE = 1000` on the base
branch): **LAMBADA-strict, HellaSwag, PIQA, ARC-easy**.

### Requested `v3.json` G2 block

Group weight **unchanged at 0.15** — this is a change to *what G2 measures*, not
to how much G2 counts. Retiring the four dead metrics automatically doubles each
surviving metric's share of G2 (equal weights over 4 instead of 8), which is the
intended effect: G2's weight now sits entirely on metrics with dynamic range.

```jsonc
"g2": {
  "weight": 0.15,
  "metrics": {
    "org.g2.lambada_strict_acc": { "kind": "accuracy", "chance": 0.0, "weight": 1.0 },
    "org.g2.hellaswag_acc":      { "kind": "accuracy", "chance": 0.25, "weight": 1.0 },
    "org.g2.piqa_acc":           { "kind": "accuracy", "chance": 0.5,  "weight": 1.0 },
    "org.g2.arc_easy_acc":       { "kind": "accuracy", "chance": 0.25, "weight": 1.0 }
    // REMOVED vs v2: arc_challenge_acc, winogrande_acc, boolq_acc, obqa_acc
  }
}
```

Effective per-metric weight inside G2: **0.25 each** (was 0.125), i.e. **3.75 %
of the composite each** (was 1.875 %).

### Two decisions to make explicitly, not by default

1. **Retire vs demote.** Above is *retire* (drop the metric). The alternative is
   *demote* — keep the metric at `"weight": 0.0` so the anchor set still declares
   it and the completeness gate still records the measurement. Demote preserves
   the observation for the saturation-tripwire record at zero scoring cost; it
   also keeps four metrics being measured that nobody is paid for, which is pod
   time. **Recommendation: retire, and pick the freed budget up as items on the
   four survivors** — resolution on axes that already move beats new families
   that cannot discriminate.
2. **Optional: replace a dead accuracy with a live likelihood.** Where accuracy
   is pinned at chance the **gold-answer NLL still moves continuously**. The
   harness now emits `g2.<task>.mean_gold_nll` for every task (observed only,
   never scored — implemented on this branch). A v3 set *could* score
   `org.g2.<task>.mean_gold_nll` for one or two of the retired tasks instead of
   dropping them. **Not recommended for v3**: the anchors would have to be
   measured first, and shipping an unmeasured `status: "placeholder"` NLL anchor
   is how you get a metric whose normalization is guesswork. File it for v4 once
   baselines have produced a distribution.

### Harness support — already implemented here

| Need | Where | State |
|---|---|---|
| Score only the surviving tasks | `harness/eval/common.py::eval_g2_tasks()` + `PRISM_EVAL_G2_TASKS` | **Done.** Default is *all* tasks so v0/v1/v2 sets keep scoring exactly what they declare; unknown names ignored; empty/unparseable falls back to the full set. |
| Per-task NLL for a future likelihood metric | `harness/eval/g2_downstream.py` → `g2.<task>.mean_gold_nll` | **Done**, observed-only. |
| Per-row bootstrap clusters | `g2/<task>#<i>` | Already on the base branch. |
| Test that the task list cannot drift from `TASKS` | `harness/tests/test_g2_task_selection.py` | **Done.** |

So the anchor-set edit is the *only* remaining step, plus setting
`PRISM_EVAL_G2_TASKS=lambada,hellaswag,piqa,arc_easy` on runs scored against
v3 so the retired tasks are not measured at all.

**Do not** set `PRISM_EVAL_G2_TASKS` on runs scored against v0/v1/v2: those sets
declare all eight metrics and would fail their own completeness gate.

## 2. `prism-eval-store` — the finalize-time contamination gate

Refusing to *persist* an unchecked composite is strictly stronger than refusing
to pay it, because it also keeps the unchecked number out of the positive-carry
set and off the public leaderboard. The policy predicate is implemented and
tested in `prism_competition::contamination`; only the call site is missing, and
it is in a sibling-owned file.

Requested insert in `prism_eval_store::finalize::finalize_composite`, immediately
after `submission_metrics` resolves:

```rust
if prism_competition::contamination::require_check()
    && !prism_competition::contamination::checked(metrics_v2)
{
    tracing::warn!(
        submission_id,
        reason = prism_competition::contamination::reason(metrics_v2),
        "refusing to score: mirror defence inert (no contamination evidence)"
    );
    return Err(FinalizeError::ContaminationUnchecked);
}
```

plus a variant on `FinalizeError`:

```rust
/// Scored run carried no contamination evidence while a check was required
/// (`PRISM_EVAL_REQUIRE_PRIVATE=1`, implied by `PRISM_EMISSION_MODE=sig`).
#[error("mirror defence inert: refusing to score an unchecked run")]
ContaminationUnchecked,
```

and a `prism-competition` path dependency in that crate's `Cargo.toml`.

**This is not blocking.** `sig`-mode emission already fail-closes on its own
(`SigContext::contamination_checked = false` ⇒ nothing allocated, full burn), so
the protected champion share cannot be paid on an unchecked round with or without
this insert. The insert adds leaderboard/carry hygiene, not emission safety.

## 3. Nothing else is requested

No changes needed in `prism-recipe/src/**`, `prismlib/**`, `prism-lium*`,
`prism-pipeline/src/composite.rs`, `prism-budget/`, `deploy/scripts/**`, or
`docs/PRISM_RECIPE.md`.
