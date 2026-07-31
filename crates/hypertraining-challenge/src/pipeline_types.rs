//! Pipeline input/output types and errors.

use hypertraining_antinois::AntinoisReport;
use hypertraining_cluster::{SegmentResult, SegmentSeeds, Topology};
use hypertraining_eval::{EvalRun, EvalVerdict};
use thiserror::Error;

use crate::score::PipelineOutcome;

/// Inputs for one candidate sim evaluation against a champion baseline.
#[derive(Debug, Clone)]
pub struct SimPipelineInput<'a> {
    /// Candidate source text (anti-noise L1).
    pub cand_source: &'a str,
    /// Candidate compiled blob (anti-noise L2 / fingerprint).
    pub cand_compiled: &'a [u8],
    /// Champion source text.
    pub champ_source: &'a str,
    /// Champion compiled blob.
    pub champ_compiled: &'a [u8],
    /// Miner id for dedupe ledger.
    pub miner_id: &'a str,
    /// Segment index for dedupe.
    pub segment_index: u64,
    /// Token budget for sim segment.
    pub budget_tokens: u64,
    /// Shared segment seeds.
    pub seeds: SegmentSeeds,
    /// Topology (master == slot).
    pub topology: Topology,
    /// `PKey` partition id.
    pub pkey_id: u16,
    /// Optional sim noise amplitude.
    pub noise_ms: u32,
    /// Validator lock bytes (hermetic build).
    pub validator_lock: &'a [u8],
    /// Admitted source tree for build (path → bytes).
    pub admitted_files: Vec<(String, Vec<u8>)>,
    /// Champion wall-clock override; when `None`, measured via sim on champ fingerprint.
    pub t_champ_ms_override: Option<u64>,
    /// Paired val-loss runs for Guard 2 (champion).
    pub champ_loss: Vec<EvalRun>,
    /// Paired val-loss runs for Guard 2 (candidate).
    pub cand_loss: Vec<EvalRun>,
}

/// Full pipeline result before leaf mapping.
#[derive(Debug, Clone)]
pub struct SimPipelineResult {
    /// Anti-noise report.
    pub antinois: AntinoisReport,
    /// Candidate segment result.
    pub cand_segment: SegmentResult,
    /// Champion wall-clock used for Δ.
    pub t_champ_ms: u64,
    /// Eval verdict (guards 2–3).
    pub eval: EvalVerdict,
    /// Kernel gate passed.
    pub kernel_ok: bool,
    /// Build image digest.
    pub image_digest: String,
    /// Integer leaf score from pay (`0..=SCORE_MAX`).
    pub score_u64: u64,
    /// Pipeline outcome for [`crate::score::score_from_pipeline`].
    pub outcome: PipelineOutcome,
}

/// Pipeline failures that abort before a terminal outcome.
#[derive(Debug, Error)]
pub enum PipelineError {
    /// Hermetic build failed.
    #[error("build: {0}")]
    Build(String),
    /// Kernel / attestation gate failed.
    #[error("kernel: {0}")]
    Kernel(String),
    /// Anti-noise evaluation failed.
    #[error("antinois: {0}")]
    Antinois(String),
    /// Cluster sim failed.
    #[error("cluster: {0}")]
    Cluster(String),
    /// Eval guards failed to run.
    #[error("eval: {0}")]
    Eval(String),
    /// Invalid pipeline inputs.
    #[error("invalid input: {0}")]
    Invalid(String),
}

