//! Versioned anchor sets for Prism v3 composite scoring
//! (`docs/spikes/prism-v3/research/12-score-aggregation.md` §7 step 0).
//!
//! Anchors (chance floors, reference bpb per domain, efficiency reference
//! points, gates, group weights, mirror tolerance, bootstrap parameters) are
//! **constants of the spec** — never functions of the current submission
//! field — and are hash-committed (pre-registered) before a round opens.
//!
//! This module is wired into the crate root as `pub mod anchors;` by the E3
//! integration pass; it is intentionally self-contained (serde + sha2 + hex
//! only, all already `prism-recipe` dependencies).
//!
//! The shipped v0 set is a **placeholder**: every numeric anchor is marked
//! `"status": "placeholder"` in the JSON and must be measured on the E6
//! reference baselines (Transformer++ / hybrid delta-net) before any
//! `PRISM_SCORING_MODE=composite` governance flip. G1 scored keys are
//! tokenizer-neutral `org.g1.bits_per_byte_*` (see harness
//! `eval/calibrate_anchors.py` to fill references from baseline METRICS).

#![allow(clippy::module_name_repetitions)]

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Embedded v0 anchor set (PLACEHOLDER values — see module docs).
pub const ANCHOR_SET_V0_JSON: &str = include_str!("../anchors/v0.json");

/// Embedded v1 anchor set (Prism **v2.1** additions; PLACEHOLDER values).
///
/// v1 = v0 plus two battery keys — `org.g7.reasoning_throughput`
/// (compute-normalized reasoning) and `org.g8.mup_scaling_slope` (local
/// scaling-exponent probe) — with identical group weights, gates, mirror
/// and bootstrap parameters. Selected at runtime via `PRISM_ANCHOR_VERSION`
/// (default 0); like every placeholder set it must be measured on the E6
/// baselines and pre-registered before scoring against it.
pub const ANCHOR_SET_V1_JSON: &str = include_str!("../anchors/v1.json");

/// Embedded v2 anchor set (Prism **v2.2** swap; PLACEHOLDER values).
///
/// v2 = v1 with `org.g2.lambada_acc` (4-way MC over random-word
/// distractors — saturated: 0.955 at 112M, 0.985 GPT-2 Large) replaced by
/// `org.g2.lambada_strict_acc` — the canonical LAMBADA protocol
/// (unconstrained greedy last-word exact match, chance ≈ 0, real headroom).
/// Same asset, same weights/gates/mirror/bootstrap; the harness emits both
/// keys so v0/v1 scoring is untouched.
pub const ANCHOR_SET_V2_JSON: &str = include_str!("../anchors/v2.json");

/// Embedded v3 anchor set (Prism **v3 measurement**; PLACEHOLDER values).
///
/// v3 = v2 with the measurement changes of the dual-cap budget:
///
/// - **removes** `org.g8.mup_scaling_slope` from the scored set. It is
///   confounded: for `L = E + A/N^α` the measured local slope is
///   `α·(1 − E/L)`, i.e. only ~30–56 % of `α`, so a model better *in level*
///   can look like it scales *worse*. The harness keeps emitting the key —
///   it is cheap and informative — but a key absent from the anchor set is
///   inert (the `unknown_metrics_are_ignored` path), which is precisely the
///   "telemetry, never scored" outcome.
/// - **adds** three byte-denominated / compute-milestone G6 keys
///   (`auc_log_bytes`, `bytes_to_bpb_threshold`, `bpb_at_half_budget`),
///   re-reading G6 as *data and compute* efficiency.
/// - **adds** the confirmation-tier `org.conf.*` keys under a `conf` group
///   at weight 0. That group is *structurally* inert, not merely
///   down-weighted: `prism_pipeline::composite` hardcodes
///   `GROUP_KEYS = [g1..g8]`, so a group outside that list is never
///   validated, normalized, or required.
/// - **adds** the compute gates `max_flops` + `min_spend_fraction`, and
///   lowers `max_wall_s` 21600 → 18000 because wall-clock is now only the
///   anti-DoS bound.
///
/// Every numeric is `placeholder` and must be measured on the E6 baselines
/// and pre-registered before any governance flip.
pub const ANCHOR_SET_V3_JSON: &str = include_str!("../anchors/v3.json");

/// Latest anchor-set version known to this build (v3 measurement set).
pub const LATEST_ANCHOR_VERSION: u16 = 3;

/// The anchor-set version live scoring defaults to (`PRISM_ANCHOR_VERSION`
/// absent). Stays 0 until v1+ anchors are measured + pre-registered.
///
/// This is load-bearing, not caution theatre: v1/v2/v3 are hash-committed
/// pre-registration artifacts, and v3 declares keys the battery does not
/// emit yet. Selecting it early would make every in-flight submission fail
/// the `MissingMetric` completeness gate — `Ineligible`, lattice 0.
pub const DEFAULT_ANCHOR_VERSION: u16 = 0;

/// Per-metric normalization descriptor (research/12 §7 step 1).
///
/// The same JSON shape is consumed by `prism-pipeline::composite`; the
/// embedded canonical JSON is the contract between the two crates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NormKind {
    /// Accuracy-like: `clip01((x - chance) / (1 - chance))`.
    Accuracy {
        /// Chance floor (e.g. `1/#choices`; 0 for generative probes).
        chance: f64,
    },
    /// bpb per domain, log-ratio between fixed anchors:
    /// `clip01((ln chance - ln x) / (ln chance - ln reference))`.
    BpbLogRatio {
        /// Chance-level bpb (`log2|V|` scaled to bytes).
        chance: f64,
        /// Pre-registered reference architecture bpb.
        reference: f64,
    },
    /// Efficiency, capped log-ratio to the reference recipe:
    /// `clip01(ln(x / reference) / ln(cap / reference))`. `cap < reference`
    /// encodes lower-better metrics (both logs negative).
    EfficiencyLogRatio {
        /// Reference operating point (score 0).
        reference: f64,
        /// Plausible extreme (score 1); below `reference` if lower-better.
        cap: f64,
    },
    /// Stability: already a bounded construction; `clip01(x)`.
    StabilityBounded,
}

fn default_metric_weight() -> f64 {
    1.0
}

/// One metric's anchor entry (normalization + provenance markers).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricAnchor {
    /// Normalization descriptor.
    #[serde(flatten)]
    pub norm: NormKind,
    /// Relative weight within the group (default 1 = equal mean).
    /// G5 uses unequal internal weights (ruler/babilong/natural/…).
    #[serde(default = "default_metric_weight")]
    pub weight: f64,
    /// `"placeholder"` until measured on the reference baselines.
    #[serde(default)]
    pub status: Option<String>,
    /// Free-form provenance note.
    #[serde(default)]
    pub note: Option<String>,
}

/// One composite group (G1–G8): weight plus its sub-metrics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroupAnchors {
    /// Composite weight `w_k` (Σ over groups = 1).
    pub weight: f64,
    /// Sub-metrics keyed `org.<group>.<name>`.
    pub metrics: BTreeMap<String, MetricAnchor>,
}

/// Lexicographic gate thresholds (research/12 §7 step 3).
///
/// Re-exported from `prism-budget` so the anchor registry and the scorer
/// parse **one** schema rather than two structs that must be kept in sync by
/// hand — the compute gates (`max_flops`, `min_spend_fraction`) have to mean
/// the same thing on both sides of the JSON.
pub use prism_budget::GateThresholds;

/// Mirror-gap (contamination) penalty parameters (§7 step 2.5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MirrorParams {
    /// Tolerance `τ_m`: gaps up to this are free.
    pub tau_m: f64,
    /// Groups the penalty applies to (`["g2", "g4"]`).
    pub groups: Vec<String>,
}

/// Clustered-bootstrap parameters (§7 steps 5–6).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BootstrapParams {
    /// Resample count B (≥ 1000).
    pub b: u32,
    /// One-sided 95% LCB z multiplier (1.645).
    pub lcb_z: f64,
}

/// A complete versioned anchor set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnchorSet {
    /// Anchor-set version.
    pub version: u16,
    /// `"placeholder"` until calibrated on the reference baselines.
    pub status: String,
    /// Issue date (YYYY-MM-DD).
    pub issued: String,
    /// Provenance / calibration notes.
    pub notes: String,
    /// Groups keyed `g1`..`g8`.
    pub groups: BTreeMap<String, GroupAnchors>,
    /// Gate thresholds.
    pub gates: GateThresholds,
    /// Mirror-gap penalty parameters.
    pub mirror: MirrorParams,
    /// Bootstrap parameters.
    pub bootstrap: BootstrapParams,
}

/// Anchor-set load errors.
#[derive(Debug, Error)]
pub enum AnchorError {
    /// Version not embedded in this build.
    #[error("unknown anchor-set version {0}")]
    UnknownVersion(u16),
    /// Embedded JSON failed to parse (build-time bug).
    #[error("anchor JSON parse: {0}")]
    Parse(#[from] serde_json::Error),
}

impl AnchorSet {
    /// Load an embedded anchor set by version.
    ///
    /// # Errors
    /// Unknown version, or embedded JSON fails to parse (build-time bug).
    pub fn load(version: u16) -> Result<Self, AnchorError> {
        Ok(serde_json::from_str(Self::canonical_json(version)?)?)
    }

    /// Load the latest embedded anchor set.
    ///
    /// # Errors
    /// Embedded JSON fails to parse (build-time bug).
    pub fn latest() -> Result<Self, AnchorError> {
        Self::load(LATEST_ANCHOR_VERSION)
    }

    /// Canonical JSON bytes for a version (the exact embedded file; the
    /// pre-registration hash is defined over these bytes).
    ///
    /// # Errors
    /// Unknown version.
    pub fn canonical_json(version: u16) -> Result<&'static str, AnchorError> {
        match version {
            0 => Ok(ANCHOR_SET_V0_JSON),
            1 => Ok(ANCHOR_SET_V1_JSON),
            2 => Ok(ANCHOR_SET_V2_JSON),
            3 => Ok(ANCHOR_SET_V3_JSON),
            v => Err(AnchorError::UnknownVersion(v)),
        }
    }

    /// Pre-registration hash: sha256 hex over the canonical JSON bytes.
    #[must_use]
    pub fn prereg_hash(&self) -> String {
        let canonical = Self::canonical_json(self.version).unwrap_or("");
        hex::encode(Sha256::digest(canonical.as_bytes()))
    }

    /// Pre-registration hash for a version without parsing the set.
    ///
    /// # Errors
    /// Unknown version.
    pub fn prereg_hash_for(version: u16) -> Result<String, AnchorError> {
        Ok(hex::encode(Sha256::digest(
            Self::canonical_json(version)?.as_bytes(),
        )))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    #[test]
    fn v0_loads_and_is_marked_placeholder() {
        let set = AnchorSet::load(0).expect("v0 parses");
        assert_eq!(set.version, 0);
        assert_eq!(set.status, "placeholder");
        assert_eq!(set.groups.len(), 8);
        for (gk, g) in &set.groups {
            assert!(!g.metrics.is_empty(), "{gk} has metrics");
            for (mk, m) in &g.metrics {
                assert_eq!(
                    m.status.as_deref(),
                    Some("placeholder"),
                    "{mk} must stay marked placeholder until measured"
                );
            }
        }
    }

    #[test]
    fn group_weights_sum_to_one() {
        let set = AnchorSet::load(0).expect("v0 parses");
        let sum: f64 = set.groups.values().map(|g| g.weight).sum();
        assert!((sum - 1.0).abs() < 1e-9, "weights sum to {sum}");
        let expected = [0.25, 0.15, 0.10, 0.15, 0.15, 0.075, 0.075, 0.05];
        for (i, w) in expected.iter().enumerate() {
            let key = format!("g{}", i + 1);
            let got = set.groups.get(&key).map_or(-1.0, |g| g.weight);
            assert!((got - w).abs() < 1e-12, "{key} weight {got} != {w}");
        }
    }

    #[test]
    fn gates_mirror_bootstrap_match_spec() {
        let set = AnchorSet::load(0).expect("v0 parses");
        assert!((set.gates.g3_min - 0.25).abs() < f64::EPSILON);
        assert!((set.gates.g8_min - 0.5).abs() < f64::EPSILON);
        assert!((set.gates.ci_half_width_delta - 0.05).abs() < f64::EPSILON);
        // v0 is byte-frozen at the pre-registration 350M cap; the live cap
        // raise to 1B lands in v2 only (see v2_swaps_saturated_mc_lambada...).
        assert_eq!(set.gates.max_params, 350_000_000);
        assert!((set.gates.max_wall_s - 21_600.0).abs() < f64::EPSILON);
        assert_eq!(
            set.mirror.groups,
            vec!["g2".to_string(), "g4".to_string(), "g5".to_string()]
        );
        let g5 = set.groups.get("g5").expect("g5");
        let ruler_w = g5.metrics.get("org.g5.ruler_acc").map(|m| m.weight);
        assert_eq!(ruler_w, Some(0.35));
        let lstar_w = g5.metrics.get("org.g5.lstar").map(|m| m.weight);
        assert_eq!(lstar_w, Some(0.10));
        assert!(set.mirror.tau_m > 0.0);
        assert!(set.bootstrap.b >= 1000);
        assert!((set.bootstrap.lcb_z - 1.645).abs() < f64::EPSILON);
    }

    #[test]
    fn prereg_hash_is_stable_sha256_hex() {
        let a = AnchorSet::prereg_hash_for(0).expect("hash v0");
        let b = AnchorSet::load(0).expect("v0 parses").prereg_hash();
        assert_eq!(a, b);
        assert_eq!(a.len(), 64);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn unknown_version_errors() {
        assert!(matches!(
            AnchorSet::load(99),
            Err(AnchorError::UnknownVersion(99))
        ));
        assert!(matches!(
            AnchorSet::prereg_hash_for(99),
            Err(AnchorError::UnknownVersion(99))
        ));
    }

    #[test]
    fn v1_is_v0_plus_the_two_v21_keys() {
        let v0 = AnchorSet::load(0).expect("v0 parses");
        let v1 = AnchorSet::load(1).expect("v1 parses");
        assert_eq!(v1.version, 1);
        assert_eq!(v1.status, "placeholder");
        assert_eq!(
            AnchorSet::latest().expect("latest").version,
            LATEST_ANCHOR_VERSION
        );
        assert_eq!(DEFAULT_ANCHOR_VERSION, 0, "live default stays v0");

        // Identical group weights, gates, mirror, bootstrap.
        for key in v0.groups.keys() {
            let (a, b) = (&v0.groups[key], &v1.groups[key]);
            assert!((a.weight - b.weight).abs() < 1e-12, "{key} weight moved");
        }
        assert_eq!(v0.gates, v1.gates);
        assert_eq!(v0.mirror, v1.mirror);
        assert_eq!(v0.bootstrap, v1.bootstrap);

        // Exactly two additions, both placeholder-marked.
        let keys = |s: &AnchorSet| -> Vec<String> {
            s.groups
                .values()
                .flat_map(|g| g.metrics.keys().cloned())
                .collect()
        };
        let (k0, k1) = (keys(&v0), keys(&v1));
        assert_eq!(k1.len(), k0.len() + 2);
        for added in ["org.g7.reasoning_throughput", "org.g8.mup_scaling_slope"] {
            assert!(k1.iter().any(|k| k == added), "{added} missing from v1");
            assert!(!k0.iter().any(|k| k == added), "{added} must not be in v0");
        }
        let slope = &v1.groups["g8"].metrics["org.g8.mup_scaling_slope"];
        assert_eq!(slope.status.as_deref(), Some("placeholder"));
        assert!(matches!(
            slope.norm,
            NormKind::EfficiencyLogRatio { reference, cap }
                if reference > 0.0 && cap > reference
        ));

        // Distinct canonical bytes ⇒ distinct pre-registration hashes.
        assert_ne!(
            AnchorSet::prereg_hash_for(0).expect("v0 hash"),
            AnchorSet::prereg_hash_for(1).expect("v1 hash")
        );
    }

    #[test]
    fn v2_swaps_saturated_mc_lambada_for_strict() {
        let v1 = AnchorSet::load(1).expect("v1 parses");
        let v2 = AnchorSet::load(2).expect("v2 parses");
        assert_eq!(v2.version, 2);
        assert_eq!(v2.status, "placeholder");
        // v2 must stay embedded and loadable however far LATEST advances, and
        // whatever LATEST points at must itself parse.
        assert!(AnchorSet::load(2).is_ok(), "v2 must stay loadable");
        assert!(
            AnchorSet::load(LATEST_ANCHOR_VERSION).is_ok(),
            "LATEST_ANCHOR_VERSION must name a set that parses"
        );
        assert_eq!(DEFAULT_ANCHOR_VERSION, 0, "live default stays v0");

        // Identical group weights, mirror, bootstrap — one key swap.
        for key in v1.groups.keys() {
            let (a, b) = (&v1.groups[key], &v2.groups[key]);
            assert!((a.weight - b.weight).abs() < 1e-12, "{key} weight moved");
        }
        assert_eq!(v1.mirror, v2.mirror);
        assert_eq!(v1.bootstrap, v2.bootstrap);

        // Gates differ in exactly ONE field: the intentional 350M -> 1B
        // parameter-cap raise (v0/v1 stay byte-frozen at 350M). Same spirit as
        // the G2 key swap above — assert the single difference, not equality.
        assert_eq!(v1.gates.max_params, 350_000_000, "v1 frozen at the old cap");
        assert_eq!(
            v2.gates.max_params,
            crate::MAX_PARAMS,
            "v2 tracks the live cap"
        );
        assert_eq!(v2.gates.max_params, 1_000_000_000);
        assert_eq!(
            v1.gates,
            GateThresholds {
                max_params: 350_000_000,
                ..v2.gates
            },
            "max_params is the ONLY gate difference v1 -> v2"
        );

        let g2_v1 = &v1.groups["g2"].metrics;
        let g2_v2 = &v2.groups["g2"].metrics;
        assert_eq!(g2_v1.len(), g2_v2.len(), "swap, not add/remove");
        assert!(g2_v1.contains_key("org.g2.lambada_acc"), "v1 keeps MC form");
        assert!(
            !g2_v2.contains_key("org.g2.lambada_acc"),
            "saturated MC out"
        );
        let strict = &g2_v2["org.g2.lambada_strict_acc"];
        assert_eq!(strict.status.as_deref(), Some("placeholder"));
        // Open-vocabulary exact match: chance floor is ~0.
        assert!(matches!(strict.norm, NormKind::Accuracy { chance } if chance == 0.0));
        // Every group KEY set outside G2 is inherited unchanged from v1
        // (G6 re-anchors values in place — no key rename; see
        // `v2_fixes_inverted_g6_auc_direction`).
        for (name, group) in &v1.groups {
            if name == "g2" {
                continue;
            }
            assert_eq!(
                group.metrics.keys().collect::<Vec<_>>(),
                v2.groups[name].metrics.keys().collect::<Vec<_>>(),
                "{name} keys moved"
            );
        }
        assert_ne!(
            AnchorSet::prereg_hash_for(1).expect("v1 hash"),
            AnchorSet::prereg_hash_for(2).expect("v2 hash")
        );
    }

    /// `org.g6.auc_log_tokens` was direction-inverted and inert in v0/v1:
    /// the anchor declared `reference 0.5 / cap 0.95` ("higher-better"),
    /// but the harness computes a MEAN CROSS-ENTROPY per decade of tokens
    /// (`eval/g6_curve.py`, lower-better, plausibly 3-5 nats), so every
    /// plausible submission clipped to 1.0 and half of G6's weight was a
    /// constant. v2 re-anchors it to the quantity actually computed.
    ///
    /// v0 and v1 are hash-committed pre-registration artifacts, so the bug
    /// stays byte-frozen there — this test pins both sides.
    #[test]
    fn v2_fixes_inverted_g6_auc_direction() {
        let auc_of = |v: u16| {
            let set = AnchorSet::load(v).expect("anchor set parses");
            match set.groups["g6"].metrics["org.g6.auc_log_tokens"].norm {
                NormKind::EfficiencyLogRatio { reference, cap } => (reference, cap),
                ref other => panic!("unexpected norm kind for v{v}: {other:?}"),
            }
        };

        // v0 / v1 keep the inverted anchor verbatim (pre-registration).
        for frozen in [0u16, 1u16] {
            let (reference, cap) = auc_of(frozen);
            assert!(
                (reference - 0.5).abs() < f64::EPSILON && (cap - 0.95).abs() < f64::EPSILON,
                "v{frozen} must stay byte-frozen at the pre-registered values"
            );
            assert!(cap > reference, "v{frozen} encoded higher-better");
        }

        // v2: lower-better (cap < reference) over a real mean-CE range.
        let (reference, cap) = auc_of(2);
        assert!(cap < reference, "v2 must encode lower-better for a mean CE");
        assert!(
            reference > 1.0 && cap > 1.0,
            "anchors must sit in the plausible nats/token range, not [0.5, 0.95]"
        );

        // The tokens-to-threshold sibling keeps its lower-better anchors;
        // censoring is fail-closed harness-side (CENSORED_TOKENS -> 0.0).
        let set = AnchorSet::load(2).expect("v2 parses");
        match set.groups["g6"].metrics["org.g6.tokens_to_threshold"].norm {
            NormKind::EfficiencyLogRatio { reference, cap } => {
                assert!(cap < reference, "tokens-to-threshold is lower-better");
            }
            ref other => panic!("unexpected norm kind: {other:?}"),
        }
        let g6_py = crate::HARNESS_FILES
            .iter()
            .find(|(path, _)| *path == "eval/g6_curve.py")
            .map(|(_, body)| *body)
            .expect("g6_curve.py is embedded");
        assert!(
            g6_py.contains("CENSORED_TOKENS"),
            "harness must fail-closed on right-censored curves"
        );
    }

    /// v3 is the **measurement** set: it drops the confounded scaling slope,
    /// adds byte/compute-denominated G6 keys, adds the inert confirmation
    /// tier, and adds the compute gates. Same spirit as
    /// `v2_swaps_saturated_mc_lambada_for_strict` — assert the exact
    /// differences, not equality.
    #[test]
    fn v3_drops_confounded_slope_and_adds_compute_keys() {
        let v2 = AnchorSet::load(2).expect("v2 parses");
        let v3 = AnchorSet::load(3).expect("v3 parses");
        assert_eq!(v3.version, 3);
        assert_eq!(v3.status, "placeholder");
        assert_eq!(LATEST_ANCHOR_VERSION, 3);
        assert_eq!(DEFAULT_ANCHOR_VERSION, 0, "live default stays v0");

        // Every scored group weight is inherited unchanged; the g1..g8
        // weights must still sum to 1 with the `conf` group excluded.
        for key in v2.groups.keys() {
            let (a, b) = (&v2.groups[key], &v3.groups[key]);
            assert!((a.weight - b.weight).abs() < 1e-12, "{key} weight moved");
        }
        assert_eq!(v2.mirror, v3.mirror);
        assert_eq!(v2.bootstrap, v3.bootstrap);
        let scored_sum: f64 = v3
            .groups
            .iter()
            .filter(|(k, _)| k.starts_with('g'))
            .map(|(_, g)| g.weight)
            .sum();
        assert!(
            (scored_sum - 1.0).abs() < 1e-9,
            "scored g1..g8 weights must still sum to 1, got {scored_sum}"
        );

        // Every other scored group's key set is inherited verbatim.
        for (name, group) in &v2.groups {
            if name == "g6" || name == "g8" {
                continue;
            }
            assert_eq!(
                group.metrics.keys().collect::<Vec<_>>(),
                v3.groups[name].metrics.keys().collect::<Vec<_>>(),
                "{name} keys moved"
            );
        }

        // Distinct canonical bytes ⇒ distinct pre-registration hashes.
        let hashes: Vec<String> = (0u16..=3)
            .map(|v| AnchorSet::prereg_hash_for(v).expect("hash"))
            .collect();
        for (i, a) in hashes.iter().enumerate() {
            for (j, b) in hashes.iter().enumerate() {
                assert!(i == j || a != b, "v{i} and v{j} share a prereg hash");
            }
        }
    }

    /// The confounded slope must be gone from the scored set — and still
    /// **emitted**, because demotion to telemetry is the decision, not
    /// deletion. A key absent from the anchor set is inert, not missing.
    #[test]
    fn v3_demotes_the_scaling_slope_to_telemetry() {
        let v2 = AnchorSet::load(2).expect("v2 parses");
        let v3 = AnchorSet::load(3).expect("v3 parses");
        let g8 = &v3.groups["g8"].metrics;
        assert!(
            !g8.contains_key("org.g8.mup_scaling_slope"),
            "the scaling slope is confounded (measured slope is only \
             α·(1 − E/L) ≈ 30–56% of α) and must not be scored in any form"
        );
        assert!(
            v2.groups["g8"]
                .metrics
                .contains_key("org.g8.mup_scaling_slope"),
            "v2 must stay byte-frozen WITH the slope"
        );
        assert_eq!(g8.len(), v2.groups["g8"].metrics.len() - 1, "removal only");
        // ...but the harness must still EMIT it: demoted to telemetry, not
        // deleted. A key absent from the anchor set is inert, not missing.
        let g8_py = crate::HARNESS_FILES
            .iter()
            .find(|(p, _)| *p == "eval/g8_stability.py")
            .map(|(_, b)| *b)
            .expect("g8_stability.py embedded");
        assert!(
            g8_py.contains("scaling_slope"),
            "slope must remain emitted as unscored telemetry"
        );
    }

    /// The three new G6 keys re-read the group as *data and compute*
    /// efficiency. The byte-denominated names are only honest because the
    /// probe curve now records bytes — asserted here, not assumed.
    #[test]
    fn v3_adds_byte_and_compute_denominated_g6_keys() {
        let v2 = AnchorSet::load(2).expect("v2 parses");
        let v3 = AnchorSet::load(3).expect("v3 parses");
        let g6 = &v3.groups["g6"].metrics;
        for added in [
            "org.g6.auc_log_bytes",
            "org.g6.bytes_to_bpb_threshold",
            "org.g6.bpb_at_half_budget",
        ] {
            let m = g6.get(added).unwrap_or_else(|| panic!("{added} missing"));
            assert_eq!(m.status.as_deref(), Some("placeholder"), "{added}");
            assert!(
                !v2.groups["g6"].metrics.contains_key(added),
                "{added} must not be in the byte-frozen v2"
            );
        }
        // Direction: `cap < reference` is the ONLY encoding of lower-better
        // (there is no direction flag) — the v0 auc bug got this backwards.
        for lower_better in ["org.g6.auc_log_bytes", "org.g6.bytes_to_bpb_threshold"] {
            match g6[lower_better].norm {
                NormKind::EfficiencyLogRatio { reference, cap } => assert!(
                    cap < reference,
                    "{lower_better} must encode lower-better (cap < reference), \
                     got reference={reference} cap={cap}"
                ),
                ref other => panic!("{lower_better}: unexpected norm {other:?}"),
            }
        }
        assert!(matches!(
            g6["org.g6.bpb_at_half_budget"].norm,
            NormKind::BpbLogRatio { chance, reference } if chance > reference
        ));
        // The token-denominated predecessor is superseded, and the byte form
        // is only honest because the probe curve now records bytes.
        assert!(
            !g6.contains_key("org.g6.auc_log_tokens"),
            "the tokenizer-dependent token form is superseded by auc_log_bytes"
        );
        let all = crate::HARNESS_FILES
            .iter()
            .map(|(_, c)| *c)
            .collect::<String>();
        for byte_marker in ["bytes_seen", "bytes_per_token", "probe_bits_per_byte"] {
            assert!(
                all.contains(byte_marker),
                "org.g6.auc_log_bytes requires the probe curve to carry \
                 {byte_marker} — otherwise the key name would be a lie"
            );
        }
    }

    /// The confirmation tier is a separate audit record, so it must be
    /// **structurally** inert rather than merely down-weighted.
    #[test]
    fn v3_confirmation_tier_is_structurally_inert() {
        let v3 = AnchorSet::load(3).expect("v3 parses");
        let conf = v3.groups.get("conf").expect("conf group present");
        assert!(
            conf.weight.abs() < f64::EPSILON,
            "the confirmation tier is a separate audit record, not part of \
             the Stage-1 composite"
        );
        assert!(
            !v3.groups.contains_key("g9"),
            "must NOT be named g9: composite hardcodes GROUP_KEYS = [g1..g8], \
             so a g9-looking group would read as a scored group that is silently \
             ignored"
        );
        for key in [
            "org.conf.isoflop_min_bpb",
            "org.conf.isoflop_convexity_r2",
            "org.conf.isoflop_argmin_nbody",
            "org.conf.advantage_growth",
        ] {
            let m = conf.metrics.get(key).unwrap_or_else(|| panic!("{key}"));
            assert_eq!(m.status.as_deref(), Some("placeholder"), "{key}");
        }
        // The two never-scorable statistics carry a degenerate normalizer, so
        // promoting them into a scored group collapses the geometric mean to
        // 0 instead of quietly ranking noise (argmin is ±17–47% in N; the
        // advantage-growth MDD 0.019–0.028 straddles the 0.013–0.065 signal).
        for never_scored in ["org.conf.isoflop_argmin_nbody", "org.conf.advantage_growth"] {
            match conf.metrics[never_scored].norm {
                NormKind::EfficiencyLogRatio { reference, cap } => assert!(
                    (reference - cap).abs() < f64::EPSILON,
                    "{never_scored} must keep a degenerate normalizer"
                ),
                ref other => panic!("{never_scored}: unexpected norm {other:?}"),
            }
        }
    }

    /// The compute gates are the currency: `max_flops` + the underspend
    /// floor, with the wall bound demoted. And every v3 numeric must still be
    /// a placeholder whose note states the measurement obligation.
    #[test]
    fn v3_adds_compute_gates_and_keeps_every_numeric_placeholder() {
        let v2 = AnchorSet::load(2).expect("v2 parses");
        let v3 = AnchorSet::load(3).expect("v3 parses");
        assert_eq!(v2.gates.max_flops, None, "v2 predates the FLOPs currency");
        assert_eq!(v2.gates.min_spend_fraction, None);
        assert_eq!(v3.gates.max_flops, Some(crate::TRAIN_FLOPS_CAP));
        assert_eq!(v3.gates.min_spend_fraction, Some(crate::MIN_SPEND_FRACTION));
        assert!(
            (v3.gates.max_wall_s - crate::TRAIN_HOURS_CAP * 3600.0).abs() < f64::EPSILON,
            "max_wall_s must track TRAIN_HOURS_CAP (now the anti-DoS bound)"
        );
        assert!(
            (v2.gates.max_wall_s - 21_600.0).abs() < f64::EPSILON,
            "v2 frozen at 6h"
        );
        // max_params is unchanged: under iso-FLOPs it is non-binding (the
        // 0.02-nat plateau spans ~88–236M body params), so it is a VRAM
        // parameter, not a scientific one.
        assert_eq!(v3.gates.max_params, v2.gates.max_params);
        assert_eq!(v3.gates.max_params, 1_000_000_000);
        assert_eq!(
            v2.gates,
            GateThresholds {
                max_wall_s: 21_600.0,
                max_flops: None,
                min_spend_fraction: None,
                ..v3.gates
            },
            "wall bound + the two compute gates are the ONLY gate differences"
        );

        // Every numeric is still a placeholder awaiting E6 measurement, and
        // every note says so.
        for (gk, g) in &v3.groups {
            for (mk, m) in &g.metrics {
                assert_eq!(
                    m.status.as_deref(),
                    Some("placeholder"),
                    "{gk}/{mk} must stay placeholder until measured on E6"
                );
                let note = m.note.as_deref().unwrap_or("");
                assert!(
                    note.contains("MUST") || note.contains("NEVER"),
                    "{gk}/{mk} note must state the measurement/scoring \
                     obligation, got {note:?}"
                );
            }
        }
        assert!(v3.notes.contains("MUST be measured"));
        assert!(v3.notes.contains("emit before declare"));
    }

    /// v0/v1/v2 are hash-committed pre-registration artifacts. Their bytes
    /// are the commitment, so this pins the exact hashes: any edit to those
    /// files — including a "harmless" reformat — breaks the commitment and
    /// must fail here rather than in an audit.
    #[test]
    fn v0_v1_v2_stay_byte_frozen() {
        for (version, want) in [
            (
                0u16,
                "581643c789faca19b9acb9980856aa9cac718c2a3cad279b8aa0bf89099de671",
            ),
            (
                1u16,
                "85998f13355171ff2e6632f1a730e627a4e10eeaa530149fa9e3af5cdd6323ef",
            ),
            (
                2u16,
                "6a6246c5f4c3cb25751df254b1b8e78017c66df72d79aef4aadbf0a036c2bc79",
            ),
        ] {
            let got = AnchorSet::prereg_hash_for(version).expect("hash");
            assert_eq!(
                got, want,
                "anchors/v{version}.json is a hash-committed pre-registration \
                 artifact and must stay byte-identical"
            );
        }
        // v3 is new, so it only has to be self-consistent and distinct.
        let v3 = AnchorSet::prereg_hash_for(3).expect("v3 hash");
        assert_eq!(v3.len(), 64);
        assert!(v3.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn roundtrip_preserves_canonical_semantics() {
        let set = AnchorSet::load(0).expect("v0 parses");
        let ser = serde_json::to_string(&set).expect("serialize");
        let de: AnchorSet = serde_json::from_str(&ser).expect("re-parse");
        assert_eq!(set, de);
    }
}
