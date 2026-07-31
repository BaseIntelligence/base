//! Miner submission domain: brief §7 JSON parse, policy checks, queue, fixture admit.
//!
//! HTTP handlers live in [`crate::routes`]. Unit tests must inject fixture trees —
//! never clone remote git.

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use hypertraining_cluster::Topology;
use hypertraining_kernel_gate::{
    validate_attestation, AccumulateDtype, AttestationError, AttestationPolicy,
    PrecisionAttestation, PrecisionFormat, ScalingRecipe,
};
use hypertraining_sealed::{admit, AdmitError, AdmitInput, SealedSurfaceV1};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::CHALLENGE_ID;

/// Opaque submission id (monotonic counter, hex).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SubmissionId(String);

impl SubmissionId {
    /// Borrow the id string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for SubmissionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Wire topology object (`tp`/`pp`/`ep`/`cp`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopologyWire {
    /// Tensor-parallel degree.
    pub tp: u32,
    /// Pipeline-parallel degree.
    pub pp: u32,
    /// Expert-parallel degree.
    pub ep: u32,
    /// Context-parallel degree.
    pub cp: u32,
}

impl TopologyWire {
    /// Convert to cluster [`Topology`] after axis checks.
    ///
    /// # Errors
    /// When any axis is zero.
    pub fn to_topology(self) -> Result<Topology, SubmissionError> {
        if self.tp == 0 || self.pp == 0 || self.ep == 0 || self.cp == 0 {
            return Err(SubmissionError::InvalidTopology);
        }
        Ok(Topology::new(self.tp, self.pp, self.ep, self.cp))
    }
}

/// Wire precision attestation (brief §7 field names / string enums).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrecisionAttestationWire {
    /// `fp8_e4m3` | `bf16` | `mixed`
    pub format: String,
    /// `fp32` (only accepted value under default policy)
    pub accumulate_dtype: String,
    /// Steps between accumulate flushes.
    pub accumulate_interval: u32,
    /// `delayed` | `current` | `block`
    pub scaling_recipe: String,
    /// TF32 flag (rejected when policy requires false).
    pub allow_tf32: bool,
}

impl PrecisionAttestationWire {
    /// Parse into domain attestation.
    ///
    /// # Errors
    /// Unknown enum strings.
    pub fn to_attestation(&self) -> Result<PrecisionAttestation, SubmissionError> {
        let format = parse_format(&self.format)?;
        let accumulate_dtype = parse_accumulate_dtype(&self.accumulate_dtype)?;
        let scaling_recipe = parse_scaling_recipe(&self.scaling_recipe)?;
        Ok(PrecisionAttestation {
            format,
            accumulate_dtype,
            accumulate_interval: self.accumulate_interval,
            scaling_recipe,
            allow_tf32: self.allow_tf32,
        })
    }
}

/// Miner POST body (brief §7).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubmissionRequest {
    /// Git remote URL (not fetched in unit tests).
    pub repo_url: String,
    /// Commit SHA (hex).
    pub commit_sha: String,
    /// Tree SHA (hex).
    pub tree_sha: String,
    /// Parallelism topology.
    pub topology: TopologyWire,
    /// Binding precision attestation.
    pub precision_attestation: PrecisionAttestationWire,
}

/// Accepted / queued submission snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueuedSubmission {
    /// Assigned id.
    pub id: SubmissionId,
    /// Original request.
    pub request: SubmissionRequest,
    /// Parsed topology.
    pub topology: Topology,
    /// Parsed attestation (policy-validated).
    pub attestation: PrecisionAttestation,
}

/// Successful HTTP accept payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SubmissionAccepted {
    /// Always `accepted`.
    pub status: &'static str,
    /// Queue id.
    pub submission_id: String,
    /// Challenge id pin.
    pub challenge_id: &'static str,
}

/// Submission / validation failures (map to HTTP 400).
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SubmissionError {
    /// Required string field empty or whitespace-only.
    #[error("missing or empty field: {0}")]
    EmptyField(&'static str),
    /// Topology axis is zero.
    #[error("topology axes must be >= 1")]
    InvalidTopology,
    /// Unknown precision format string.
    #[error("unknown precision format: {0}")]
    UnknownFormat(String),
    /// Unknown accumulate dtype string.
    #[error("unknown accumulate_dtype: {0}")]
    UnknownAccumulateDtype(String),
    /// Unknown scaling recipe string.
    #[error("unknown scaling_recipe: {0}")]
    UnknownScalingRecipe(String),
    /// Attestation policy rejection.
    #[error("precision_attestation: {0}")]
    Attestation(#[from] AttestationError),
    /// Sealed-surface admission failed.
    #[error("admission: {0}")]
    Admission(String),
}

/// In-memory submission queue + attestation policy.
#[derive(Debug)]
pub struct SubmissionService {
    policy: AttestationPolicy,
    next_id: AtomicU64,
    queue: Mutex<VecDeque<QueuedSubmission>>,
}

impl Default for SubmissionService {
    fn default() -> Self {
        Self::new(AttestationPolicy::default())
    }
}

impl SubmissionService {
    /// Build with an explicit attestation policy.
    #[must_use]
    pub fn new(policy: AttestationPolicy) -> Self {
        Self {
            policy,
            next_id: AtomicU64::new(1),
            queue: Mutex::new(VecDeque::new()),
        }
    }

    /// Borrow policy.
    #[must_use]
    pub fn policy(&self) -> &AttestationPolicy {
        &self.policy
    }

    /// Validate brief §7 body and enqueue (no network).
    ///
    /// # Errors
    /// [`SubmissionError`] on schema/policy failure.
    pub fn accept(&self, request: SubmissionRequest) -> Result<QueuedSubmission, SubmissionError> {
        validate_nonempty(&request.repo_url, "repo_url")?;
        validate_nonempty(&request.commit_sha, "commit_sha")?;
        validate_nonempty(&request.tree_sha, "tree_sha")?;
        let topology = request.topology.to_topology()?;
        let attestation = request.precision_attestation.to_attestation()?;
        validate_attestation(&attestation, &self.policy)?;

        let id = SubmissionId(format!(
            "ht-sub-{:016x}",
            self.next_id.fetch_add(1, Ordering::Relaxed)
        ));
        let queued = QueuedSubmission {
            id,
            request,
            topology,
            attestation,
        };
        if let Ok(mut q) = self.queue.lock() {
            q.push_back(queued.clone());
        }
        Ok(queued)
    }

    /// Accept and build the HTTP 202 body.
    ///
    /// # Errors
    /// See [`Self::accept`].
    pub fn accept_response(
        &self,
        request: SubmissionRequest,
    ) -> Result<SubmissionAccepted, SubmissionError> {
        let q = self.accept(request)?;
        Ok(SubmissionAccepted {
            status: "accepted",
            submission_id: q.id.as_str().to_owned(),
            challenge_id: CHALLENGE_ID,
        })
    }

    /// Run sealed admission against an **injected** fixture tree (no git clone).
    ///
    /// # Errors
    /// [`SubmissionError::Admission`] when sealed checks fail.
    pub fn admit_with_fixture_tree(
        &self,
        changed_paths: &[String],
        file_contents: &BTreeMap<String, Vec<u8>>,
        manifest: &SealedSurfaceV1,
    ) -> Result<(), SubmissionError> {
        let input = AdmitInput {
            changed_paths,
            file_contents,
            manifest,
        };
        admit(&input).map_err(|e: AdmitError| SubmissionError::Admission(e.to_string()))
    }

    /// Number of queued submissions (test helper).
    #[must_use]
    pub fn queued_len(&self) -> usize {
        self.queue.lock().map_or(0, |q| q.len())
    }
}

fn validate_nonempty(value: &str, field: &'static str) -> Result<(), SubmissionError> {
    if value.trim().is_empty() {
        return Err(SubmissionError::EmptyField(field));
    }
    Ok(())
}

fn parse_format(s: &str) -> Result<PrecisionFormat, SubmissionError> {
    match s.trim().to_ascii_lowercase().as_str() {
        "fp8_e4m3" => Ok(PrecisionFormat::Fp8E4m3),
        "bf16" => Ok(PrecisionFormat::Bf16),
        "mixed" => Ok(PrecisionFormat::Mixed),
        other => Err(SubmissionError::UnknownFormat(other.to_owned())),
    }
}

fn parse_accumulate_dtype(s: &str) -> Result<AccumulateDtype, SubmissionError> {
    match s.trim().to_ascii_lowercase().as_str() {
        "fp32" => Ok(AccumulateDtype::Fp32),
        other => Err(SubmissionError::UnknownAccumulateDtype(other.to_owned())),
    }
}

fn parse_scaling_recipe(s: &str) -> Result<ScalingRecipe, SubmissionError> {
    match s.trim().to_ascii_lowercase().as_str() {
        "delayed" => Ok(ScalingRecipe::Delayed),
        "current" => Ok(ScalingRecipe::Current),
        "block" => Ok(ScalingRecipe::Block),
        other => Err(SubmissionError::UnknownScalingRecipe(other.to_owned())),
    }
}

/// Example valid body for docs / tests (`fp8_e4m3` path).
#[must_use]
pub fn example_valid_request() -> SubmissionRequest {
    SubmissionRequest {
        repo_url: "https://example.invalid/miner/fork.git".into(),
        commit_sha: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into(),
        tree_sha: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".into(),
        topology: TopologyWire {
            tp: 4,
            pp: 2,
            ep: 8,
            cp: 1,
        },
        precision_attestation: PrecisionAttestationWire {
            format: "fp8_e4m3".into(),
            accumulate_dtype: "fp32".into(),
            accumulate_interval: 128,
            scaling_recipe: "delayed".into(),
            allow_tf32: false,
        },
    }
}
