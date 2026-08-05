//! Miner submission accept path (two-script PRISM contract).

use std::collections::VecDeque;
use std::sync::Mutex;

use base64::Engine;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Opaque submission id (hex sha256).
pub type SubmissionId = String;

/// Miner submit body — architecture + training sources (agent-challenge-like UX).
///
/// Prefer ZIP (`application/zip` or `zip_base64`); raw source fields remain
/// accepted for local/CI convenience.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmissionRequest {
    /// Miner hotkey hex (64 hex chars = 32 bytes).
    pub miner_hotkey: String,
    /// `architecture.py` source (must define `build_model`).
    #[serde(default)]
    pub architecture_py: String,
    /// `training.py` source (must define `train`).
    #[serde(default)]
    pub training_py: String,
    /// Optional base64-encoded ZIP containing `architecture.py` + `training.py`.
    #[serde(default)]
    pub zip_base64: Option<String>,
    /// Optional human label.
    #[serde(default)]
    pub label: Option<String>,
}

/// Accepted response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmissionAccepted {
    pub submission_id: SubmissionId,
    pub status: String,
}

/// Queued work item.
#[derive(Debug, Clone)]
pub struct QueuedSubmission {
    pub id: SubmissionId,
    pub request: SubmissionRequest,
}

/// Submit validation errors.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SubmissionError {
    #[error("empty field: {0}")]
    EmptyField(&'static str),
    #[error("invalid miner_hotkey hex (need 64 hex chars)")]
    InvalidHotkey,
    #[error("architecture_py must contain build_model")]
    MissingBuildModel,
    #[error("training_py must contain train")]
    MissingTrain,
}

/// In-memory queue service (v1).
#[derive(Debug, Default)]
pub struct SubmissionService {
    queue: Mutex<VecDeque<QueuedSubmission>>,
}

impl SubmissionService {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate + enqueue.
    ///
    /// # Errors
    /// Schema / contract failures.
    pub fn accept(&self, req: SubmissionRequest) -> Result<SubmissionAccepted, SubmissionError> {
        validate(&req)?;
        let id = submission_id(&req);
        let accepted = SubmissionAccepted {
            submission_id: id.clone(),
            status: "accepted".into(),
        };
        self.queue
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push_back(QueuedSubmission { id, request: req });
        Ok(accepted)
    }

    /// Pop next work item.
    pub fn pop(&self) -> Option<QueuedSubmission> {
        self.queue
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pop_front()
    }

    /// Queue length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.queue
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    /// Empty?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Expand optional `zip_base64` into source fields (mutates `req`).
///
/// # Errors
/// ZIP / decode failures.
pub fn expand_zip_fields(req: &mut SubmissionRequest) -> Result<(), String> {
    let Some(b64) = req
        .zip_base64
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    else {
        return Ok(());
    };
    let zip = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .map_err(|e| format!("zip_base64: {e}"))?;
    let (arch, train) = prism_recipe::sources_from_zip(&zip).map_err(|e| e.to_string())?;
    req.architecture_py = arch;
    req.training_py = train;
    Ok(())
}

/// Validate contract-shape of an incoming submission (pre-queue).
///
/// # Errors
/// Schema/contract failures.
pub fn validate(req: &SubmissionRequest) -> Result<(), SubmissionError> {
    if req.miner_hotkey.trim().is_empty() {
        return Err(SubmissionError::EmptyField("miner_hotkey"));
    }
    if req.architecture_py.trim().is_empty() {
        return Err(SubmissionError::EmptyField("architecture_py"));
    }
    if req.training_py.trim().is_empty() {
        return Err(SubmissionError::EmptyField("training_py"));
    }
    let hk = req.miner_hotkey.trim();
    if hk.len() != 64 || !hk.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(SubmissionError::InvalidHotkey);
    }
    if !req.architecture_py.contains("build_model") {
        return Err(SubmissionError::MissingBuildModel);
    }
    if !req.training_py.contains("def train") && !req.training_py.contains("train(") {
        return Err(SubmissionError::MissingTrain);
    }
    Ok(())
}

/// Deterministic submission id (sha256 of the contract bytes).
#[must_use]
pub fn submission_id(req: &SubmissionRequest) -> SubmissionId {
    let mut h = Sha256::new();
    h.update(req.miner_hotkey.as_bytes());
    h.update(req.architecture_py.as_bytes());
    h.update(req.training_py.as_bytes());
    hex::encode(h.finalize())
}

/// Fixture request for tests (contract-conformant incl. telemetry hooks).
#[must_use]
pub fn example_valid_request() -> SubmissionRequest {
    SubmissionRequest {
        miner_hotkey: "11".repeat(32),
        architecture_py: "import torch\ndef build_model(ctx):\n    return torch.nn.Linear(8, 8)\n"
            .into(),
        training_py: concat!(
            "import prism_telemetry\n",
            "def train(model, ctx):\n",
            "    prism_telemetry.report(loss=1.0, step=1)\n",
            "    prism_telemetry.finish_evaluation()\n",
            "    return {'loss': 1.0}\n",
        )
        .into(),
        zip_base64: None,
        label: Some("tiny".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_valid() {
        let s = SubmissionService::new();
        let a = s.accept(example_valid_request()).unwrap();
        assert_eq!(a.status, "accepted");
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn rejects_missing_build_model() {
        let mut r = example_valid_request();
        r.architecture_py = "x = 1".into();
        let e = SubmissionService::new().accept(r).unwrap_err();
        assert!(matches!(e, SubmissionError::MissingBuildModel));
    }
}
