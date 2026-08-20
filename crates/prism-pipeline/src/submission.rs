//! Miner submission accept path (recipe 2.0 AutoModel patch ZIP; legacy 1.x
//! JSON / `arch_id` behind `PRISM_ALLOW_LEGACY_INTAKE`).

use std::collections::VecDeque;
use std::sync::Mutex;

use base64::Engine;
use prism_store::{Stage, SubmissionState};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Opaque submission id (hex sha256).
pub type SubmissionId = String;

/// Miner submit body (AutoModel ZIP/JSON fields, or transitional 1.x sources).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SubmissionRequest {
    pub miner_hotkey: String,
    #[serde(default)]
    pub architecture_py: String,
    #[serde(default)]
    pub training_py: String,
    #[serde(default)]
    pub zip_base64: Option<String>,
    #[serde(default)]
    pub arch_id: Option<String>,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub automodel_base: Option<String>,
    #[serde(default)]
    pub automodel_patch: Option<String>,
    #[serde(default)]
    pub prism_toml: Option<String>,
    /// Packed applied AutoModel tree (set by expand; not on the wire).
    ///
    /// Custom deps: a miner ships `requirements.txt` / `pyproject.toml` by
    /// adding it in their `automodel.patch` (it becomes a touched file that
    /// slim delivery keeps); the pod harness installs any manifest found in
    /// the tree (network-on install phase, see `prismlib/deps.py`).
    #[serde(skip)]
    pub packed_tree: Option<Vec<u8>>,
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
    #[error("arch_id must match arch_<16 lowercase hex>")]
    InvalidArchId,
    #[error("architecture_py must be empty when arch_id is set (pulled from registry)")]
    ArchIdWithSource,
    #[error("unsupported_layout: {0}")]
    UnsupportedLayout(String),
}

impl SubmissionError {
    #[must_use]
    pub fn code(&self) -> &'static str {
        match self {
            Self::UnsupportedLayout(_) => "unsupported_layout",
            _ => "invalid_request",
        }
    }
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

fn trim_nonempty(s: Option<&str>) -> Option<&str> {
    s.map(str::trim).filter(|t| !t.is_empty())
}

#[must_use]
pub fn is_automodel_request(req: &SubmissionRequest) -> bool {
    req.packed_tree.is_some()
        || (trim_nonempty(req.automodel_base.as_deref()).is_some()
            && req
                .automodel_patch
                .as_deref()
                .is_some_and(|s| !s.is_empty()))
}

/// Expand ZIP / JSON AutoModel fields; reject legacy ZIP layouts under 2.0.
pub fn expand_zip_fields(req: &mut SubmissionRequest) -> Result<(), String> {
    let zip = match trim_nonempty(req.zip_base64.as_deref()) {
        Some(b64) => Some(
            base64::engine::general_purpose::STANDARD
                .decode(b64)
                .map_err(|e| format!("zip_base64: {e}"))?,
        ),
        None => None,
    };
    match prism_automodel::expand_automodel(
        req.automodel_base.as_deref(),
        req.automodel_patch.as_deref(),
        req.prism_toml.as_deref(),
        zip.as_deref(),
    ) {
        Ok(Some(exp)) => {
            req.automodel_base = Some(exp.pin_id);
            req.automodel_patch = Some(exp.patch_text);
            req.packed_tree = Some(exp.packed_tree);
            req.architecture_py = exp.architecture_py;
            req.training_py = exp.training_py;
            req.arch_id = None;
            Ok(())
        }
        Ok(None) => Ok(()),
        Err(e) => Err(e.to_string()),
    }
}

pub fn tree_blob_for(req: &SubmissionRequest) -> Result<Option<Vec<u8>>, String> {
    if let Some(blob) = &req.packed_tree {
        return Ok(Some(blob.clone()));
    }
    match source_tree(req)? {
        Some(t) => t.pack_blob().map(Some).map_err(|e| e.to_string()),
        None => Ok(None),
    }
}

pub fn source_tree(req: &SubmissionRequest) -> Result<Option<prism_recipe::SourceTree>, String> {
    if req.packed_tree.is_some() || is_automodel_request(req) {
        return Ok(None);
    }
    let Some(b64) = trim_nonempty(req.zip_base64.as_deref()) else {
        return Ok(None);
    };
    let zip = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .map_err(|e| format!("zip_base64: {e}"))?;
    tree_from_zip_bytes(&zip)
}

fn tree_from_zip_bytes(zip: &[u8]) -> Result<Option<prism_recipe::SourceTree>, String> {
    match prism_recipe::probe_zip_kind(zip).map_err(|e| e.to_string())? {
        prism_recipe::ZipKind::Tree => prism_recipe::tree_from_zip(zip)
            .map(Some)
            .map_err(|e| e.to_string()),
        _ => Ok(None),
    }
}

pub fn validate(req: &SubmissionRequest) -> Result<(), SubmissionError> {
    if req.miner_hotkey.trim().is_empty() {
        return Err(SubmissionError::EmptyField("miner_hotkey"));
    }
    let hk = req.miner_hotkey.trim();
    if hk.len() != 64 || !hk.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(SubmissionError::InvalidHotkey);
    }
    if is_automodel_request(req) {
        if req.arch_id.as_deref().is_some_and(|s| !s.trim().is_empty()) {
            return Err(SubmissionError::UnsupportedLayout(
                "arch_id is not accepted under recipe ≥ 2.0".into(),
            ));
        }
        return Ok(());
    }
    if req.arch_id.as_deref().is_some_and(|s| !s.trim().is_empty())
        && !prism_automodel::allow_legacy_intake()
    {
        return Err(SubmissionError::UnsupportedLayout(
            "training-only arch_id submissions are rejected under recipe ≥ 2.0".into(),
        ));
    }
    if req.training_py.trim().is_empty() {
        return Err(SubmissionError::EmptyField("training_py"));
    }
    if !req.training_py.contains("def train") && !req.training_py.contains("train(") {
        return Err(SubmissionError::MissingTrain);
    }
    match req.arch_id.as_deref().map(str::trim) {
        None | Some("") => {
            if req.architecture_py.trim().is_empty() {
                return Err(SubmissionError::EmptyField("architecture_py"));
            }
            if !req.architecture_py.contains("build_model") {
                return Err(SubmissionError::MissingBuildModel);
            }
        }
        Some(id) => {
            if !is_arch_id(id) {
                return Err(SubmissionError::InvalidArchId);
            }
            if !req.architecture_py.trim().is_empty() {
                return Err(SubmissionError::ArchIdWithSource);
            }
        }
    }
    Ok(())
}

/// `arch_<16 lowercase hex>` shape.
#[must_use]
pub fn is_arch_id(s: &str) -> bool {
    s.len() == 21
        && s.starts_with("arch_")
        && s[5..]
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
}

/// Registry id for an architecture source (`arch_` + first 16 hex of sha256).
#[must_use]
pub fn arch_id_for(architecture_py: &str) -> String {
    let digest = hex::encode(Sha256::digest(architecture_py.as_bytes()));
    format!("arch_{}", &digest[..16])
}

/// Full sha256 hex digest of an architecture source (registry unique key).
#[must_use]
pub fn arch_digest(architecture_py: &str) -> String {
    hex::encode(Sha256::digest(architecture_py.as_bytes()))
}

/// Gating key: `prism`, or `prism:train:<arch_id>` for training-only (fits CHECK).
#[must_use]
pub fn gating_key(arch_id: Option<&str>) -> String {
    match arch_id {
        Some(id) => format!("prism:train:{id}"),
        None => "prism".to_owned(),
    }
}

/// Deterministic submission id (`sha256(pin‖0x00‖patch)` for AutoModel; legacy hotkey+sources).
#[must_use]
pub fn submission_id(req: &SubmissionRequest) -> SubmissionId {
    if let (Some(pin), Some(patch)) = (
        trim_nonempty(req.automodel_base.as_deref()),
        req.automodel_patch.as_deref(),
    ) {
        return prism_automodel::submission_id_for_patch(pin, patch.as_bytes());
    }
    let mut h = Sha256::new();
    h.update(req.miner_hotkey.as_bytes());
    if let Ok(Some(tree)) = source_tree(req) {
        h.update(b"prism-tree-v1\0");
        h.update(tree.canonical_sha256().as_bytes());
    } else {
        h.update(req.architecture_py.as_bytes());
        h.update(req.training_py.as_bytes());
    }
    hex::encode(h.finalize())
}

/// Queued row for an accepted submission (`miner_coldkey` from metagraph when live).
#[must_use]
pub fn queued_row(
    req: &SubmissionRequest,
    id: SubmissionId,
    miner_coldkey: Option<String>,
    tree_blob: Option<Vec<u8>>,
    epoch: u64,
    netuid: u16,
    now_ms: u64,
) -> SubmissionState {
    SubmissionState {
        id,
        miner_hotkey: req.miner_hotkey.trim().to_owned(),
        miner_coldkey,
        epoch,
        netuid,
        status: Stage::Queued,
        architecture_py: req.architecture_py.clone(),
        training_py: req.training_py.clone(),
        tree_blob,
        label: req.label.clone(),
        pod_id: None,
        pod_provider: None,
        receipt: None,
        metrics_json: None,
        bpb: None,
        arch_id: req.arch_id.clone(),
        review: None,
        similarity: None,
        final_score: None,
        retry_count: 0,
        error_detail: None,
        created_at_ms: now_ms,
        updated_at_ms: now_ms,
    }
}

/// Legacy 1.x JSON two-script fixture (requires `PRISM_ALLOW_LEGACY_INTAKE=1`
/// under recipe ≥ 2.0). Prefer [`example_automodel_request`] for 2.0 paths.
#[must_use]
pub fn example_valid_request() -> SubmissionRequest {
    example_legacy_request()
}

/// Legacy 1.x JSON two-script fixture (requires `PRISM_ALLOW_LEGACY_INTAKE=1`).
#[must_use]
pub fn example_legacy_request() -> SubmissionRequest {
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
        label: Some("tiny".into()),
        ..Default::default()
    }
}

/// Recipe 2.0 fixture pin + `happy.patch` ZIP (expanded).
///
/// # Panics
/// When the vendored fixture ZIP / patch is missing or fails to apply.
#[must_use]
#[allow(clippy::expect_used)]
pub fn example_automodel_request() -> SubmissionRequest {
    // Fixture pin is CI/Sim-only; live rejects it without this env.
    std::env::set_var(prism_automodel::ENV_FIXTURE, "1");
    let zip = prism_automodel::fixture_automodel_zip().expect("fixture zip");
    let mut req = SubmissionRequest {
        miner_hotkey: "11".repeat(32),
        zip_base64: Some(base64::engine::general_purpose::STANDARD.encode(zip)),
        label: Some("automodel-fixture".into()),
        ..Default::default()
    };
    expand_zip_fields(&mut req).expect("automodel fixture expand");
    req
}

#[cfg(test)]
mod tests {
    use super::*;

    fn allow_legacy() {
        std::env::set_var(prism_automodel::ALLOW_LEGACY_ENV, "1");
    }

    fn zip_b64(files: &[(&str, &[u8])]) -> String {
        use std::io::Write;
        let mut buf = std::io::Cursor::new(Vec::new());
        {
            let mut w = zip::ZipWriter::new(&mut buf);
            let opts = zip::write::SimpleFileOptions::default();
            for (n, b) in files {
                w.start_file(*n, opts).unwrap();
                w.write_all(b).unwrap();
            }
            w.finish().unwrap();
        }
        base64::engine::general_purpose::STANDARD.encode(buf.into_inner())
    }

    #[test]
    fn gating_key_fits_challenge_check() {
        let key = gating_key(Some("arch_d50dcfef6eaf5f04"));
        assert_eq!(key, "prism:train:arch_d50dcfef6eaf5f04");
        assert!(key.len() <= 64);
    }

    #[test]
    fn accepts_automodel_fixture() {
        let req = example_automodel_request();
        assert!(req.packed_tree.is_some());
        validate(&req).unwrap();
        let a = SubmissionService::new().accept(req.clone()).unwrap();
        assert_eq!(a.status, "accepted");
        let want = prism_automodel::submission_id_for_patch(
            req.automodel_base.as_deref().unwrap(),
            req.automodel_patch.as_deref().unwrap().as_bytes(),
        );
        assert_eq!(a.submission_id, want);
        assert!(tree_blob_for(&req).unwrap().is_some());
    }

    #[test]
    fn rejects_legacy_zip_layout() {
        std::env::remove_var(prism_automodel::ALLOW_LEGACY_ENV);
        let mut req = SubmissionRequest {
            zip_base64: Some(zip_b64(&[
                (
                    "architecture.py",
                    b"import torch\ndef build_model(ctx):\n    return torch.nn.Linear(8, 8)\n",
                ),
                ("training.py", b"def train(model, ctx):\n    return {}\n"),
            ])),
            ..example_legacy_request()
        };
        let err = expand_zip_fields(&mut req).unwrap_err();
        assert!(err.contains("unsupported_layout"), "{err}");
    }

    #[test]
    fn rejects_arch_id_without_allow() {
        std::env::remove_var(prism_automodel::ALLOW_LEGACY_ENV);
        let mut req = example_legacy_request();
        req.architecture_py.clear();
        req.arch_id = Some("arch_d50dcfef6eaf5f04".into());
        let err = validate(&req).unwrap_err();
        assert_eq!(err.code(), "unsupported_layout");
    }

    #[test]
    fn legacy_json_two_script_still_validates() {
        let req = example_legacy_request();
        validate(&req).unwrap();
        assert!(SubmissionService::new().accept(req).is_ok());
    }

    #[test]
    fn arch_id_ok_when_legacy_allowed() {
        allow_legacy();
        let mut req = example_legacy_request();
        req.architecture_py.clear();
        req.arch_id = Some("arch_d50dcfef6eaf5f04".into());
        validate(&req).unwrap();
    }
}
