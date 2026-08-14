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

/// Latest anchor-set version known to this build (v2.1 additions).
pub const LATEST_ANCHOR_VERSION: u16 = 1;

/// The anchor-set version live scoring defaults to (`PRISM_ANCHOR_VERSION`
/// absent). Stays 0 until v1 anchors are measured + pre-registered.
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
    /// Fixed-recipe wall-clock budget (seconds; 6h = 21600).
    pub max_wall_s: f64,
}

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
        assert_eq!(AnchorSet::latest().expect("latest").version, 1);
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
    fn roundtrip_preserves_canonical_semantics() {
        let set = AnchorSet::load(0).expect("v0 parses");
        let ser = serde_json::to_string(&set).expect("serialize");
        let de: AnchorSet = serde_json::from_str(&ser).expect("re-parse");
        assert_eq!(set, de);
    }
}
