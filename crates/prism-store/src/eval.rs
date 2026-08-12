//! PRISM v3 eval persistence contract (migration 0017): composite runs,
//! Zone A metric/mirror rows, the anchor registry, pre-registration
//! hash-commits, and Zone B participant metric reports.
//!
//! The trait + records live beside [`crate::store::PrismStore`] so the API
//! and orchestrator share one import surface; the memory and Postgres impls
//! live in `prism-eval-store` (per-crate LOC cap — the same split as
//! `prism-pipeline` vs `prism-challenge`).
//!
//! Write semantics: an eval run is written once at finalize and replaced
//! wholesale on re-score (children cascade, mirroring `prism_telemetry`).
//! Zone B reports are append-only audit: flagged/quarantined rows stay
//! visible and retries continue the hash chain at `max(seq) + 1`.

use std::fmt::Debug;

use async_trait::async_trait;
use prism_zoneb::{PreparedReport, Verdict, VerdictReason};
use serde_json::Value;

use prism_store_types::StoreError;

/// `prism_eval_run` row: at most one per submission.
///
/// `run_id` and `created_at` are store-assigned on save; the input values
/// are ignored.
#[derive(Debug, Clone, PartialEq)]
pub struct EvalRunRecord {
    /// Run id (store-assigned).
    pub run_id: String,
    /// Owning submission.
    pub submission_id: String,
    /// Anchor-set version scored against.
    pub anchor_version: i32,
    /// sha256 hex over the canonical anchor JSON.
    pub prereg_hash: String,
    /// `shadow` | `composite` (`PRISM_SCORING_MODE` at finalize time).
    pub scoring_mode: String,
    /// `METRICS_JSON` v2 pod manifest snapshot.
    pub pod_manifest: Option<Value>,
    /// sha256 hex of the uploaded harness file set.
    pub harness_files_sha256: Option<String>,
    /// Miner subprocess ran in an empty netns.
    pub netns: Option<bool>,
    /// Battery tier label when reported.
    pub eval_tier: Option<String>,
    /// Authoritative `CompositeOutcome` serde blob (`status`-tagged).
    pub outcome_json: Value,
    /// Store-assigned creation timestamp (text).
    pub created_at: String,
}

/// `prism_eval_group` row: per-group point estimate + bootstrap CI.
#[derive(Debug, Clone, PartialEq)]
pub struct EvalGroupRecord {
    /// Group key `g1`..`g8`.
    pub grp: String,
    /// Point estimate after the mirror-gap penalty.
    pub g: f64,
    /// Clustered-bootstrap 2.5% (`None` = bootstrap did not run).
    pub ci_lo: Option<f64>,
    /// Clustered-bootstrap 97.5%.
    pub ci_hi: Option<f64>,
    /// Mirror-gap penalty deducted.
    pub mirror_penalty: f64,
}

/// `prism_eval_metric` row: one Zone A raw metric (`org.*` keys only).
#[derive(Debug, Clone, PartialEq)]
pub struct EvalMetricRecord {
    /// Metric key (`org.<group>.<name>`).
    pub key: String,
    /// Aggregate on the metric's natural scale.
    pub value: f64,
    /// Cluster id -> per-cluster value (bootstrap units).
    pub clusters: Option<Value>,
}

/// `prism_mirror_pair` row: public/private mirror for the contamination gap.
#[derive(Debug, Clone, PartialEq)]
pub struct MirrorPairRecord {
    /// Group the penalty applies to.
    pub grp: String,
    /// `org.*` key whose anchors normalize both sides.
    pub metric: String,
    /// Public-anchor aggregate.
    pub public_value: f64,
    /// Private-mirror aggregate.
    pub mirror_value: f64,
    /// Public per-cluster values.
    pub public_clusters: Option<Value>,
    /// Mirror per-cluster values.
    pub mirror_clusters: Option<Value>,
}

/// `prism_anchor_set` row: the versioned anchor registry.
///
/// `activated_at` is store-assigned on the `placeholder` -> `active` flip.
#[derive(Debug, Clone, PartialEq)]
pub struct AnchorSetRecord {
    /// Anchor-set version.
    pub version: i32,
    /// Canonical anchor set JSON.
    pub json: Value,
    /// sha256 hex over the canonical JSON.
    pub prereg_hash: String,
    /// `placeholder` | `active`.
    pub status: String,
    /// Activation timestamp (text), set by the store on the status flip.
    pub activated_at: Option<String>,
}

/// `prism_prereg` row: a pre-registration hash-commit (append-only).
///
/// `committed_at` is store-assigned.
#[derive(Debug, Clone, PartialEq)]
pub struct PreregRecord {
    /// Anchor-set version.
    pub version: i32,
    /// sha256 hex over the canonical anchor JSON.
    pub hash: String,
    /// Store-assigned commit timestamp (text).
    pub committed_at: String,
    /// Optional operator note.
    pub notes: Option<String>,
}

/// `prism_metric_report` row: one Zone B participant report (append-only).
#[derive(Debug, Clone, PartialEq)]
pub struct MetricReportRecord {
    /// Per-submission chain sequence (0-based).
    pub seq: i64,
    /// Envelope schema version (pinned to the recipe version).
    pub schema_version: String,
    /// Previous `report_hash`; the submission id when `seq = 0`.
    pub prev_hash: String,
    /// sha256 hex over the canonical payload.
    pub report_hash: String,
    /// Canonical Zone B payload (immutable).
    pub payload: Value,
    /// Validation verdict.
    pub verdict: Verdict,
    /// Structured verdict reasons.
    pub verdict_reasons: Vec<VerdictReason>,
    /// Store-assigned creation timestamp (text).
    pub created_at: String,
}

/// Eval persistence surface (Zone A composite runs + Zone B reports +
/// anchor registry). See the module docs for write semantics.
#[async_trait]
pub trait EvalStore: Send + Sync + Debug {
    /// Replace the eval run for `run.submission_id` wholesale (children
    /// cascade) and insert the new run + groups + metrics + mirrors.
    ///
    /// # Errors
    /// Backend fault.
    async fn save_eval_run(
        &self,
        run: &EvalRunRecord,
        groups: &[EvalGroupRecord],
        metrics: &[EvalMetricRecord],
        mirrors: &[MirrorPairRecord],
    ) -> Result<(), StoreError>;

    /// The eval run for one submission, when finalized.
    ///
    /// # Errors
    /// Backend fault.
    async fn eval_run(&self, submission_id: &str) -> Result<Option<EvalRunRecord>, StoreError>;

    /// Group rows for a run (`g1`..`g8` order not guaranteed).
    ///
    /// # Errors
    /// Backend fault.
    async fn eval_groups(&self, run_id: &str) -> Result<Vec<EvalGroupRecord>, StoreError>;

    /// Zone A raw metric rows for a run.
    ///
    /// # Errors
    /// Backend fault.
    async fn eval_metrics(&self, run_id: &str) -> Result<Vec<EvalMetricRecord>, StoreError>;

    /// Mirror-pair rows for a run.
    ///
    /// # Errors
    /// Backend fault.
    async fn eval_mirrors(&self, run_id: &str) -> Result<Vec<MirrorPairRecord>, StoreError>;

    /// Insert or refresh an anchor-set registry row (status flip sets
    /// `activated_at`).
    ///
    /// # Errors
    /// Backend fault.
    async fn upsert_anchor_set(&self, set: &AnchorSetRecord) -> Result<(), StoreError>;

    /// Every known anchor set, ascending version.
    ///
    /// # Errors
    /// Backend fault.
    async fn anchor_sets(&self) -> Result<Vec<AnchorSetRecord>, StoreError>;

    /// Append a pre-registration hash-commit (idempotent on `(version, hash)`).
    ///
    /// # Errors
    /// Backend fault.
    async fn record_prereg(&self, entry: &PreregRecord) -> Result<(), StoreError>;

    /// Every pre-registration commit, ascending version.
    ///
    /// # Errors
    /// Backend fault.
    async fn preregistrations(&self) -> Result<Vec<PreregRecord>, StoreError>;

    /// Append a validated Zone B report to the submission's chain.
    ///
    /// # Errors
    /// Backend fault (incl. duplicate `(submission_id, seq)` / `report_hash`).
    async fn insert_metric_report(
        &self,
        submission_id: &str,
        report: &PreparedReport,
    ) -> Result<(), StoreError>;

    /// The submission's Zone B chain, ascending `seq`.
    ///
    /// # Errors
    /// Backend fault.
    async fn metric_reports(
        &self,
        submission_id: &str,
    ) -> Result<Vec<MetricReportRecord>, StoreError>;

    /// The chain head (highest `seq`) for a submission.
    ///
    /// # Errors
    /// Backend fault.
    async fn latest_metric_report(
        &self,
        submission_id: &str,
    ) -> Result<Option<MetricReportRecord>, StoreError>;

    /// Every Zone B report NOT belonging to `exclude_submission` (cohort
    /// reference for MAD outlier detection).
    ///
    /// # Errors
    /// Backend fault.
    async fn cohort_reports(
        &self,
        exclude_submission: &str,
    ) -> Result<Vec<MetricReportRecord>, StoreError>;
}
