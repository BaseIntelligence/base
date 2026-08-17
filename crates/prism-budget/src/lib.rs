//! Prism budget facts and the lexicographic **budget gates** of the v3
//! composite (`docs/PRISM_RECIPE.md` § budget currency).
//!
//! Split out of `prism-pipeline::composite` when the dual-cap gates landed:
//! that module was within a few lines of the 1500 non-test LOC cap, and the
//! budget gates are a self-contained concern with no dependency on scoring.
//! `prism-pipeline` re-exports every type here, so the public API and the
//! serialized JSON shape are unchanged (precedent: `resolved_pod_image` in
//! `prism-lium-harness`, `prism-competition` out of `prism-registry`).
//!
//! ## Why the budget is dual-capped
//!
//! The scoring currency is **attested FLOPs**, not wall-clock. Wall-clock
//! makes MFU a scored quantity: two identical architectures differ in score
//! by their kernel maturity (no `sm_120` `FlashAttention` cubins, Triton
//! version rent, a looped model paying ~`r`× per token). FLOPs measured from
//! the *realized* dispatch graph prices looping, `MoE` sparsity and vocabulary
//! size automatically, so the budget adapts to the architecture class
//! without any tier to declare or shop for.
//!
//! Wall-clock stays as an **anti-DoS bound** — a pure FLOPs cap does not
//! bound a 2 %-MFU implementation, which would hold a pod for days. So both
//! caps are live, whichever binds first stops the run, and
//! [`BudgetFacts::binding_cap`] records which one it was.
//!
//! ## Why underspend is a gate and not a discount
//!
//! Scoring against a compute-normalized reference invites deliberate
//! underspend: the compute-optimal frontier's slope is only ≈ −0.05..−0.10
//! nats per e-fold, so an architecture that saturates early gains more from
//! being judged against a weaker reference than it loses by training less.
//! [`GateFailure::SpendBelowFloor`] closes that: below the floor the run is
//! **ineligible**, not merely scaled, and no truncation correction can ever
//! be positive.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

/// Which cap stopped the run (recorded, and visible to the operator).
///
/// Serialized lowercase to match the harness's `org.diag.binding_cap`
/// string, so the Python and Rust sides speak one vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum BindingCap {
    /// The attested-FLOPs cap bound first — the intended case.
    Flops,
    /// The wall-clock safety bound bound first (MFU below target).
    Wall,
    /// The step cap bound first.
    Steps,
    /// Training ended on its own (miner stop condition) or the run predates
    /// FLOPs attestation.
    #[default]
    None,
}

impl BindingCap {
    /// Parse the harness `org.diag.binding_cap` string; unknown ⇒ `None`.
    #[must_use]
    pub fn parse(s: &str) -> Self {
        match s {
            "flops" => Self::Flops,
            "wall" => Self::Wall,
            "steps" => Self::Steps,
            _ => Self::None,
        }
    }

    /// Stable lowercase label (inverse of [`Self::parse`]).
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Flops => "flops",
            Self::Wall => "wall",
            Self::Steps => "steps",
            Self::None => "none",
        }
    }

    /// The recipe, not the miner, stopped the run.
    ///
    /// Underspend is only a cheat when the miner *chose* to stop
    /// (`None`). Hitting steps/wall/flops is spending the protocol budget.
    #[must_use]
    pub const fn protocol_bound(self) -> bool {
        matches!(self, Self::Flops | Self::Wall | Self::Steps)
    }
}

/// Lexicographic gate thresholds (research/12 §7 step 3).
///
/// `max_flops` and `min_spend_fraction` are `Option` so the byte-frozen
/// v0/v1/v2 anchor JSON — which predates the FLOPs currency — still parses
/// unchanged. Absent means "this anchor set does not gate on compute", which
/// is exactly true of those sets.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GateThresholds {
    /// Minimum G3 (recall probes) group score.
    pub g3_min: f64,
    /// Minimum G8 (stability) group score.
    pub g8_min: f64,
    /// Max per-axis clustered 95% CI half-width δ.
    pub ci_half_width_delta: f64,
    /// Fixed-recipe parameter budget.
    pub max_params: u64,
    /// Wall-clock safety bound (seconds). Not the currency — see module docs.
    pub max_wall_s: f64,
    /// Attested-FLOPs budget (the currency). `None` ⇒ ungated (v0/v1/v2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_flops: Option<f64>,
    /// Eligibility floor as a fraction of `max_flops`. `None` ⇒ no floor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_spend_fraction: Option<f64>,
}

/// Budget facts for the fixed-recipe gates (step 3).
///
/// Every FLOPs field is harness-attested: the miner reports none of them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct BudgetFacts {
    /// Trained parameter count.
    pub params: u64,
    /// Wall-clock seconds.
    pub wall_s: f64,
    /// Real training tokens consumed (recorded, not gated).
    pub tokens: u64,
    /// Attested FLOPs spent (`flops_per_token_probe × tokens_seen`).
    /// `None` for runs from before attestation existed — those skip the
    /// compute gates rather than failing them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flops_attested: Option<f64>,
    /// Which cap stopped the run.
    #[serde(default)]
    pub binding_cap: BindingCap,
}

/// One gate failure (structured; serialized for the dashboard/audit).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "gate", rename_all = "snake_case")]
pub enum GateFailure {
    /// A whole group's metrics are absent from the submission.
    MissingGroup {
        /// Group key (`g1`..`g8`).
        group: String,
    },
    /// A declared sub-metric is absent from the submission.
    MissingMetric {
        /// Metric key (`org.<group>.<name>`).
        key: String,
    },
    /// `g3 < g3_min` (recall floor).
    G3BelowFloor {
        /// Observed g3.
        g: f64,
        /// Required floor.
        floor: f64,
    },
    /// `g8 < g8_min` (stability floor).
    G8BelowFloor {
        /// Observed g8.
        g: f64,
        /// Required floor.
        floor: f64,
    },
    /// Parameter budget exceeded.
    ParamsOverBudget {
        /// Observed params.
        params: u64,
        /// Budget.
        max: u64,
    },
    /// Wall-clock safety bound exceeded.
    WallClockOverBudget {
        /// Observed seconds.
        wall_s: f64,
        /// Budget seconds.
        max_s: f64,
    },
    /// Attested-FLOPs budget exceeded — the currency gate.
    FlopsOverBudget {
        /// Attested FLOPs.
        flops: f64,
        /// Budget FLOPs.
        max: f64,
    },
    /// Attested spend below `min_spend_fraction × max_flops` — the
    /// underspend guard (see module docs).
    SpendBelowFloor {
        /// Attested FLOPs.
        flops: f64,
        /// Required floor in FLOPs.
        floor: f64,
        /// The fraction that produced the floor.
        fraction: f64,
    },
    /// An axis's clustered 95% CI half-width exceeds δ.
    CiHalfWidthTooWide {
        /// Group key.
        group: String,
        /// Observed half-width.
        half_width: f64,
        /// Pre-registered δ.
        delta: f64,
    },
}

impl GateFailure {
    /// Whether this failure is one of the budget gates.
    #[must_use]
    pub const fn is_budget(&self) -> bool {
        matches!(
            self,
            Self::ParamsOverBudget { .. }
                | Self::WallClockOverBudget { .. }
                | Self::FlopsOverBudget { .. }
                | Self::SpendBelowFloor { .. }
        )
    }

    /// Whether this failure is a completeness failure.
    #[must_use]
    pub const fn is_completeness(&self) -> bool {
        matches!(self, Self::MissingGroup { .. } | Self::MissingMetric { .. })
    }
}

/// Per-gate pass/fail flags (all true ⇔ eligible).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct GateReport {
    /// Every declared group/metric present.
    pub complete: bool,
    /// g3 floor passed.
    pub g3_ok: bool,
    /// g8 floor passed.
    pub g8_ok: bool,
    /// Budget gates passed (params, wall, FLOPs, spend floor).
    pub budgets_ok: bool,
    /// CI-sufficiency gate passed.
    pub ci_ok: bool,
}

impl GateReport {
    /// Derive the flags from a failure list.
    #[must_use]
    pub fn from_failures(failures: &[GateFailure]) -> Self {
        let has = |f: fn(&GateFailure) -> bool| failures.iter().any(f);
        Self {
            complete: !has(GateFailure::is_completeness),
            g3_ok: !has(|f| matches!(f, GateFailure::G3BelowFloor { .. })),
            g8_ok: !has(|f| matches!(f, GateFailure::G8BelowFloor { .. })),
            budgets_ok: !has(GateFailure::is_budget),
            ci_ok: !has(|f| matches!(f, GateFailure::CiHalfWidthTooWide { .. })),
        }
    }
}

/// Evaluate every budget gate, appending failures in a stable order:
/// params, wall, FLOPs over, spend floor.
///
/// The compute gates are skipped when the anchor set declares no
/// `max_flops` (v0/v1/v2) or when the run carries no attested FLOPs — a run
/// that predates attestation must not be failed for a measurement that did
/// not exist. That is deliberate: the migration order is **emit, then
/// declare**, because a gate declared before the harness emits its input
/// would make every in-flight submission ineligible.
pub fn check(facts: &BudgetFacts, gates: &GateThresholds, out: &mut Vec<GateFailure>) {
    if facts.params > gates.max_params {
        out.push(GateFailure::ParamsOverBudget {
            params: facts.params,
            max: gates.max_params,
        });
    }
    if facts.wall_s > gates.max_wall_s {
        out.push(GateFailure::WallClockOverBudget {
            wall_s: facts.wall_s,
            max_s: gates.max_wall_s,
        });
    }
    let (Some(max_flops), Some(spent)) = (gates.max_flops, facts.flops_attested) else {
        return;
    };
    if !spent.is_finite() || spent < 0.0 || !max_flops.is_finite() || max_flops <= 0.0 {
        return;
    }
    if spent > max_flops {
        out.push(GateFailure::FlopsOverBudget {
            flops: spent,
            max: max_flops,
        });
    }
    // Underspend floor. A *voluntary* early stop must not be a free win
    // (see module docs). A run that hit a protocol cap (steps / wall /
    // flops) spent as much as the recipe allowed at its batch size — Phase
    // 0 measured the reference Transformer++ at batch 8 × seq 512 stopping
    // at 20 006 steps with only 6.1 % of TRAIN_FLOPS_CAP. Rejecting that
    // is a quality-neutral batch-size failure, not an underspend attack.
    if let Some(fraction) = gates.min_spend_fraction {
        if fraction.is_finite() && fraction > 0.0 && !facts.binding_cap.protocol_bound() {
            let floor = max_flops * fraction;
            if spent < floor {
                out.push(GateFailure::SpendBelowFloor {
                    flops: spent,
                    floor,
                    fraction,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    /// v3-shaped thresholds: compute gated at 3.0e18 with a 0.5 floor.
    fn v3_gates() -> GateThresholds {
        GateThresholds {
            g3_min: 0.25,
            g8_min: 0.5,
            ci_half_width_delta: 0.05,
            max_params: 1_000_000_000,
            max_wall_s: 18_000.0,
            max_flops: Some(3.0e18),
            min_spend_fraction: Some(0.5),
        }
    }

    /// v0-shaped thresholds: no compute gate at all (byte-frozen sets).
    fn legacy_gates() -> GateThresholds {
        GateThresholds {
            max_flops: None,
            min_spend_fraction: None,
            ..v3_gates()
        }
    }

    fn facts(flops: Option<f64>) -> BudgetFacts {
        BudgetFacts {
            params: 300_000_000,
            wall_s: 15_000.0,
            tokens: 2_500_000_000,
            flops_attested: flops,
            binding_cap: BindingCap::Flops,
        }
    }

    fn check_vec(f: &BudgetFacts, g: &GateThresholds) -> Vec<GateFailure> {
        let mut out = Vec::new();
        check(f, g, &mut out);
        out
    }

    #[test]
    fn full_spend_inside_both_caps_passes() {
        let out = check_vec(&facts(Some(2.9e18)), &v3_gates());
        assert!(out.is_empty(), "{out:?}");
        assert!(GateReport::from_failures(&out).budgets_ok);
    }

    #[test]
    fn overspending_the_flops_cap_fails() {
        let out = check_vec(&facts(Some(3.5e18)), &v3_gates());
        assert!(
            out.iter()
                .any(|f| matches!(f, GateFailure::FlopsOverBudget { .. })),
            "{out:?}"
        );
        assert!(!GateReport::from_failures(&out).budgets_ok);
    }

    /// The underspend guard: stopping early must not be a free win.
    #[test]
    fn underspending_below_the_floor_fails() {
        // 1.4e18 is 47% of the cap — under the 50% floor. Voluntary
        // (binding_cap = none); a protocol-bound run at this spend is
        // eligible (see step_bound_reference_baseline_is_eligible).
        let mut f = facts(Some(1.4e18));
        f.binding_cap = BindingCap::None;
        let out = check_vec(&f, &v3_gates());
        match out.as_slice() {
            [GateFailure::SpendBelowFloor {
                flops,
                floor,
                fraction,
            }] => {
                assert!((*flops - 1.4e18).abs() < 1e12);
                assert!((*floor - 1.5e18).abs() < 1e12);
                assert!((*fraction - 0.5).abs() < f64::EPSILON);
            }
            other => panic!("expected exactly one spend-floor failure: {other:?}"),
        }
        // Exactly at the floor is allowed (the floor is inclusive).
        let mut at_floor = facts(Some(1.5e18));
        at_floor.binding_cap = BindingCap::None;
        assert!(check_vec(&at_floor, &v3_gates()).is_empty());
    }

    /// A wall-bound run that still cleared the floor stays eligible — that
    /// is the whole point of recording `binding_cap` instead of failing.
    #[test]
    fn wall_bound_run_above_the_floor_is_eligible() {
        let mut f = facts(Some(1.8e18));
        f.binding_cap = BindingCap::Wall;
        f.wall_s = 18_000.0;
        assert!(check_vec(&f, &v3_gates()).is_empty());
    }

    /// Phase 0 reference: batch 8 × seq 512, 20 006 steps, 1.82e17 FLOPs
    /// (6.1 % of the cap), `binding_cap = steps`. Must stay eligible —
    /// dual cap must not reject small-batch honest runs.
    #[test]
    fn step_bound_reference_baseline_is_eligible() {
        let mut f = facts(Some(1.82e17));
        f.binding_cap = BindingCap::Steps;
        f.tokens = 81_920_000;
        assert!(
            check_vec(&f, &v3_gates()).is_empty(),
            "step-capped reference spend must not trip SpendBelowFloor"
        );
        // The same spend with no protocol cap *is* voluntary underspend.
        f.binding_cap = BindingCap::None;
        assert!(check_vec(&f, &v3_gates())
            .iter()
            .any(|g| matches!(g, GateFailure::SpendBelowFloor { .. })));
    }

    /// Emit-then-declare: a pre-attestation run must not be failed for a
    /// measurement that did not exist when it ran.
    #[test]
    fn missing_attestation_skips_compute_gates() {
        assert!(check_vec(&facts(None), &v3_gates()).is_empty());
        // ...and a v0/v1/v2 anchor set gates nothing on compute even when
        // the run *does* carry attestation.
        assert!(check_vec(&facts(Some(1.0e16)), &legacy_gates()).is_empty());
    }

    #[test]
    fn non_finite_attestation_is_ignored_not_trusted() {
        for bad in [f64::NAN, f64::INFINITY, -1.0] {
            let out = check_vec(&facts(Some(bad)), &v3_gates());
            assert!(out.is_empty(), "{bad} produced {out:?}");
        }
    }

    #[test]
    fn param_and_wall_gates_still_fire() {
        let mut f = facts(Some(2.0e18));
        f.params = 2_000_000_000;
        f.wall_s = 20_000.0;
        let out = check_vec(&f, &v3_gates());
        assert!(out
            .iter()
            .any(|x| matches!(x, GateFailure::ParamsOverBudget { .. })));
        assert!(out
            .iter()
            .any(|x| matches!(x, GateFailure::WallClockOverBudget { .. })));
        // Order is stable: params before wall.
        assert!(matches!(out[0], GateFailure::ParamsOverBudget { .. }));
    }

    #[test]
    fn binding_cap_roundtrips_the_harness_vocabulary() {
        for (s, v) in [
            ("flops", BindingCap::Flops),
            ("wall", BindingCap::Wall),
            ("steps", BindingCap::Steps),
        ] {
            assert_eq!(BindingCap::parse(s), v);
            assert_eq!(v.label(), s);
        }
        assert_eq!(BindingCap::parse("nonsense"), BindingCap::None);
        assert_eq!(BindingCap::default(), BindingCap::None);
        let json = serde_json::to_string(&BindingCap::Flops).unwrap();
        assert_eq!(json, "\"flops\"");
    }

    /// The byte-frozen anchor sets predate `max_flops`, so their JSON must
    /// still parse — and must not gain the field on re-serialization.
    #[test]
    fn legacy_gate_json_parses_and_stays_clean() {
        let legacy = r#"{"g3_min":0.25,"g8_min":0.5,"ci_half_width_delta":0.05,
                         "max_params":350000000,"max_wall_s":21600.0}"#;
        let g: GateThresholds = serde_json::from_str(legacy).unwrap();
        assert_eq!(g.max_flops, None);
        assert_eq!(g.min_spend_fraction, None);
        let round = serde_json::to_string(&g).unwrap();
        assert!(!round.contains("max_flops"), "{round}");
        assert!(!round.contains("min_spend_fraction"), "{round}");
        // And a v3-shaped set round-trips its compute gates.
        let v3 = serde_json::to_string(&v3_gates()).unwrap();
        assert!(v3.contains("max_flops"));
        assert_eq!(
            serde_json::from_str::<GateThresholds>(&v3).unwrap(),
            v3_gates()
        );
    }

    #[test]
    fn gate_report_classifies_every_variant() {
        let all = vec![
            GateFailure::MissingGroup { group: "g1".into() },
            GateFailure::MissingMetric {
                key: "org.x".into(),
            },
            GateFailure::G3BelowFloor {
                g: 0.1,
                floor: 0.25,
            },
            GateFailure::G8BelowFloor { g: 0.1, floor: 0.5 },
            GateFailure::ParamsOverBudget { params: 2, max: 1 },
            GateFailure::WallClockOverBudget {
                wall_s: 2.0,
                max_s: 1.0,
            },
            GateFailure::FlopsOverBudget {
                flops: 2.0,
                max: 1.0,
            },
            GateFailure::SpendBelowFloor {
                flops: 1.0,
                floor: 2.0,
                fraction: 0.5,
            },
            GateFailure::CiHalfWidthTooWide {
                group: "g1".into(),
                half_width: 0.2,
                delta: 0.05,
            },
        ];
        let report = GateReport::from_failures(&all);
        assert_eq!(report, GateReport::default(), "every flag must be false");
        assert_eq!(all.iter().filter(|f| f.is_budget()).count(), 4);
        assert_eq!(all.iter().filter(|f| f.is_completeness()).count(), 2);
        // Serialized tag names are the audit contract.
        let json = serde_json::to_string(&all).unwrap();
        for tag in [
            "flops_over_budget",
            "spend_below_floor",
            "wall_clock_over_budget",
            "params_over_budget",
        ] {
            assert!(json.contains(tag), "missing tag {tag}");
        }
    }
}
