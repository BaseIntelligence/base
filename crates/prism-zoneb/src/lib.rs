//! PRISM Zone B participant-reported metrics: contract types
//! (`docs/spikes/prism-v3/research/09-miner-metrics-leaderboards.md` §2–§3, §7).
//!
//! Every metric lives in exactly one of two zones. Zone A (`org.*`) is
//! organizer-measured and scored; Zone B (`miner.<group>.<name>`) is
//! participant-reported, displayed-but-labeled, validated, and **never
//! reaches the scoring path**. Miner-emitted `org.*` keys are rejected from
//! the payload at ingest and quarantine the report (kept as anti-cheat
//! evidence — no retraction).
//!
//! This crate holds only the contract: envelope shape, caps, verdicts, and
//! structured findings. The validation lattice lives in
//! `prism_pipeline::zone_b`; storage I/O lives in `prism-eval-store`.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Max distinct scalar metrics per report.
pub const MAX_SCALARS: usize = 64;
/// Max distinct series per report.
pub const MAX_SERIES: usize = 16;
/// Max points per series.
pub const MAX_SERIES_POINTS: usize = 10_000;
/// Max canonical payload bytes per report (1 MB).
pub const MAX_PAYLOAD_BYTES: usize = 1_048_576;

/// Display hint for a Zone B metric (`Higher`/`Lower` better, `Neutral`
/// unordered); never used for scoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    Higher,
    Lower,
    Neutral,
}

/// Typed metric payload (OTel-shaped; research/09 §3). `Series` aligns
/// `step[i]` ↔ `value[i]` (monotonic, no dup steps); `Histogram` is
/// explicit-bucket with `counts.len() == boundaries.len() + 1` (last count
/// is the +inf bucket).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MetricKind {
    /// One f64.
    Scalar { value: f64 },
    /// Per-step f64 arrays.
    Series { step: Vec<u64>, value: Vec<f64> },
    /// Bucketed distribution.
    Histogram {
        boundaries: Vec<f64>,
        counts: Vec<u64>,
    },
}

/// One Zone B metric entry: typed payload + optional UCUM `unit` (display
/// normalization only) + optional `direction` display hint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ZoneBMetric {
    #[serde(flatten)]
    pub kind: MetricKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub direction: Option<Direction>,
}

/// Zone B report envelope (`schema_version` is pinned to the recipe).
/// `prev_hash` is the miner-chained hash of the previous report's canonical
/// payload; `None` = master-chained at ingest (legacy `train_metrics` lift).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ZoneBEnvelope {
    pub schema_version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prev_hash: Option<String>,
    /// Metrics keyed `miner.<group>.<name>`.
    pub metrics: BTreeMap<String, ZoneBMetric>,
}

/// Report verdict (stored on the row; display badge): `Ok` clean, `Flagged`
/// atypical/unverifiable (never an auto-zero), `Quarantined` provable
/// dishonesty / tamper evidence. DB/API string form is the serde
/// `snake_case` tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Ok,
    Flagged,
    Quarantined,
}

/// One structured validation finding (serialized for the audit trail).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum VerdictReason {
    /// Miner-chained `prev_hash` (`got`) does not match the stored head
    /// (`expected`: previous `report_hash`, or the submission id at seq 0).
    HashChainMismatch { expected: String, got: String },
    /// Miner emitted an `org.*` `key` (organizer namespace is reserved).
    OrgNamespace { key: String },
    /// `claimed` tokens exceed the harness-side counter (`observed`).
    TokenCountExceedsObserved { claimed: u64, observed: u64 },
    /// A `step` index exceeds the recipe step `cap`.
    StepCapExceeded { step: u64, cap: u64 },
    /// `claimed` wall-clock exceeds `observed` × slack (see `zone_b`).
    WallClockExceedsObserved { claimed: f64, observed: f64 },
    /// Implied `mfu` above the physics `ceiling` for the pod `gpu`.
    MfuAboveCeiling { mfu: f64, ceiling: f64, gpu: String },
    /// `terminal` train loss (nats) outside the band around `expected`
    /// (`bpb · ln 2`).
    TerminalLossOutsideBand { terminal: f64, expected: f64 },
    /// Byte-identical series `name` already seen from another submission.
    DuplicateSeries { name: String },
    /// Series `name` steps are not non-decreasing.
    NonMonotonicSteps { name: String },
    /// Series `name` repeats `step`.
    DuplicateSteps { name: String, step: u64 },
    /// Non-finite value in `name` (defense in depth; JSON cannot carry one).
    NonFiniteValue { name: String },
    /// `value` outside cohort `median` ± k·`mad` for `name`.
    CohortOutlier {
        name: String,
        value: f64,
        median: f64,
        mad: f64,
    },
    /// A ground-truth `check` (`token_accounting` | `wall_clock` |
    /// `mfu_ceiling` | `terminal_loss`) could not run (e.g. Sim pods).
    GroundTruthUnavailable { check: String },
}

impl VerdictReason {
    /// Quarantine-class finding (provable dishonesty / tamper evidence).
    /// Flag-class is exactly {statistical outlier, non-finite, degraded
    /// ground truth}; everything else is quarantine-class.
    #[must_use]
    pub const fn is_quarantine(&self) -> bool {
        use VerdictReason as R;
        !matches!(
            self,
            R::CohortOutlier { .. } | R::NonFiniteValue { .. } | R::GroundTruthUnavailable { .. }
        )
    }
}

/// Hard ingest rejections: the report is malformed or over caps and nothing
/// is stored (distinct from verdicts, which always store the report).
#[derive(Debug, Error)]
pub enum IngestReject {
    /// Canonical payload over the 1 MB cap.
    #[error("payload too large")]
    PayloadTooLarge,
    /// Too many scalar keys.
    #[error("too many scalars: {0}")]
    TooManyScalars(usize),
    /// Too many series keys.
    #[error("too many series: {0}")]
    TooManySeries(usize),
    /// One series over the points cap.
    #[error("series too long: {0}")]
    SeriesTooLong(String),
    /// Metric name outside `miner.<group>.<name>`.
    #[error("invalid metric name: {0}")]
    InvalidName(String),
    /// Structurally malformed envelope.
    #[error("bad envelope: {0}")]
    BadEnvelope(String),
}

/// Organizer-side ground truth for one run (from `METRICS_JSON` v2).
#[derive(Debug, Clone, Default)]
pub struct GroundTruth {
    /// Harness-measured bpb on the frozen val cut.
    pub bpb: Option<f64>,
    /// Harness-measured train wall-clock seconds.
    pub wall_s: Option<f64>,
    /// Harness-counted train tokens (authoritative when `train_stream`).
    pub tokens_seen: Option<u64>,
    /// Model parameter count measured in-pod.
    pub n_params: Option<u64>,
    /// Pod GPU type label.
    pub gpu_type: Option<String>,
    /// Recipe step cap.
    pub max_steps: u64,
}

/// Cohort observations for one metric name, from **other** submissions:
/// terminal values (scalar, or last series point) and content hashes of
/// cohort series (duplicate detection).
#[derive(Debug, Clone, Default)]
pub struct CohortMetric {
    pub terminals: Vec<f64>,
    pub series_hashes: Vec<String>,
}

/// A validated report ready to persist. `seq` is the 0-based per-submission
/// chain sequence; `prev_hash` links to the previous `report_hash` (or the
/// submission id at `seq == 0`); `report_hash` is sha256 over the canonical
/// `payload` (stored immutably; served by the API).
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedReport {
    pub seq: u64,
    pub prev_hash: String,
    pub report_hash: String,
    pub payload: serde_json::Value,
    pub verdict: Verdict,
    pub reasons: Vec<VerdictReason>,
}

/// Metric-name rule: exactly `miner.<group>.<name>`, segments
/// `[a-z0-9_]+` (research/09 §7; deeper nesting is not v0).
#[must_use]
pub fn is_valid_metric_name(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("miner.") else {
        return false;
    };
    let mut p = rest.split('.');
    let (Some(a), Some(b), None) = (p.next(), p.next(), p.next()) else {
        return false;
    };
    [a, b].iter().all(|s| {
        !s.is_empty()
            && s.bytes()
                .all(|c| matches!(c, b'a'..=b'z' | b'0'..=b'9' | b'_'))
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn metric_name_rule() {
        assert!(is_valid_metric_name("miner.train.loss"));
        assert!(is_valid_metric_name("miner.sys.tokens_per_s2"));
        assert!(!is_valid_metric_name("miner.train"));
        assert!(!is_valid_metric_name("miner.a.b.c"));
        assert!(!is_valid_metric_name("org.eval.bpb"));
        assert!(!is_valid_metric_name("Miner.train.loss"));
        assert!(!is_valid_metric_name("miner.train..loss"));
        assert!(!is_valid_metric_name("miner.train.lo ss"));
    }

    #[test]
    fn verdict_reason_serde_tags() {
        let r = VerdictReason::MfuAboveCeiling {
            mfu: 0.9,
            ceiling: 0.75,
            gpu: "H100".to_string(),
        };
        let v = serde_json::to_value(&r).unwrap();
        assert_eq!(v["reason"], "mfu_above_ceiling");
        let back: VerdictReason = serde_json::from_value(v).unwrap();
        assert_eq!(back, r);
        assert!(r.is_quarantine());
        assert!(!VerdictReason::CohortOutlier {
            name: String::new(),
            value: 0.0,
            median: 0.0,
            mad: 1.0
        }
        .is_quarantine());
    }

    #[test]
    fn envelope_roundtrip() {
        let mut metrics = BTreeMap::new();
        metrics.insert(
            "miner.train.loss".to_string(),
            ZoneBMetric {
                kind: MetricKind::Series {
                    step: vec![0, 1],
                    value: vec![2.0, 1.0],
                },
                unit: Some("nats".to_string()),
                direction: Some(Direction::Lower),
            },
        );
        let env = ZoneBEnvelope {
            schema_version: "1.3.0".to_string(),
            prev_hash: None,
            metrics,
        };
        let s = serde_json::to_string(&env).unwrap();
        let back: ZoneBEnvelope = serde_json::from_str(&s).unwrap();
        assert_eq!(env, back);
        assert!(s.contains("\"kind\":\"series\""));
    }
}
