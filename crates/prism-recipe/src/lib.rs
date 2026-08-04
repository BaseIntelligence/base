//! Normative PRISM recipe (`docs/PRISM_RECIPE.md`).
//!
//! PRISM is a **recipe-contract challenge**: the miner submits custom code
//! (`architecture.py` defining `build_model(ctx)` and `training.py` defining
//! `train(model, ctx)`) that runs **inside the operator-owned harness** on a
//! Lium pod. The harness supplies everything the contract promises — "all the
//! data" the miner can request:
//!
//! - the pinned pretraining shard (fineweb-edu sample, pinned URL + SHA-256),
//! - seed, tokenizer, device, step/time caps,
//! - a frozen val stream and its tokenization,
//! - the exact BPB lattice (val CE / ln 2).
//!
//! The same objects are exposed to miners over `GET /v1/recipe` and
//! `GET /v1/recipe/baseline` so they can run the harness locally before
//! submitting. Nothing secret: there are no held-out private rows — the val
//! stream is part of the published pin (ephemeral-attempt safety comes from
//! the seed lattice + similarity review, not data secrecy).
//!
//! The harness itself is a single Python file embedded at build time
//! ([`HARNESS_PY`]) uploaded to the pod over SSH by `prism-lium`. No `PyPI` packages at
//! runtime: it only uses stdlib + `torch` + `transformers` + `datasets`
//! preinstalled in the pinned pod image.

#![forbid(unsafe_code)]

mod zip_submit;

pub use zip_submit::{sources_from_zip, ZipSubmitError};

/// Harness source executed inside the pod (uploaded over SSH, no secrets).
pub const HARNESS_PY: &str = include_str!("../harness/prism_harness.py");

/// Baseline submission: `architecture.py`.
pub const BASELINE_ARCHITECTURE_PY: &str = include_str!("../baseline/architecture.py");

/// Baseline submission: `training.py`.
pub const BASELINE_TRAINING_PY: &str = include_str!("../baseline/training.py");

/// Recipe semantic version (surfaced through the API). Bumped to 1.0.2 for
/// the ≤350M parameter hard cap enforced in-pod after `build_model`.
pub const RECIPE_VERSION: &str = "1.0.2";

/// Maximum model parameters allowed after `build_model` (350M).
pub const MAX_PARAMS: u64 = 350_000_000;

/// Pinned pretraining shard: fineweb-edu sample-10BT, single parquet shard.
///
/// Pin triple: URL + sha256 + license. The harness verifies the SHA-256
/// before training; a mismatch fails the eval `ChallengeInternal`.
pub const DATASET_URL: &str =
    "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/010_00000.parquet";

/// Canonical dataset ref (HF dataset id).
pub const DATASET_REF: &str = "HuggingFaceFW/fineweb-edu@sample/10BT";

/// Expected byte length of the pinned shard (operator-verified; the harness
/// still trust-verifies cryptographically via [`dataset_sha256`]).
#[must_use]
pub fn dataset_len_bytes() -> u64 {
    2_152_798_864
}

/// Expected SHA-256 of the pinned shard (hex lowercase). Operator-verified
/// by downloading the pin once and hashing; the upload path re-verifies.
///
/// See `docs/PRISM_RECIPE.md` for the update ceremony.
pub const DATASET_SHA256: &str = "e5a2eae25f057f0856a10bfae314c6ca8ea8bb08456d2131e9e89b2b8305e2f6";

/// Overrideable SHA-256 (`PRISM_DATASET_SHA256`); defaults to the build pin.
#[must_use]
pub fn dataset_sha256() -> String {
    std::env::var("PRISM_DATASET_SHA256")
        .ok()
        .map(|v| v.trim().to_owned())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| DATASET_SHA256.into())
}

/// Train wall clock cap per submission (seconds). **User goal: up to 6h.**
pub const TRAIN_HOURS_CAP: f64 = 6.0;

/// Pod lifetime cap total (seconds): train cap + bootstrap margin (1h).
pub const POD_LIFETIME_HOURS_CAP: f64 = 7.0;

/// Steps hard stop inside `train` (belt + clock). Config overrides only down.
pub const MAX_TRAIN_STEPS: u32 = 20_000;

/// Seed lattice (deterministic across miners; recipes share seed space).
pub const RECIPE_SEED: u64 = 0x0050_5249_534D;

/// Val stream rows frozen into evaluation (first N validated texts).
pub const VAL_ROWS: usize = 256;

/// Train rows from the pinned shard (`ctx.train_iter()`).
pub const TRAIN_ROWS: usize = 2_048;

/// Minimal sanity constraints every submission must satisfy **before** any
/// pod is rented — string-level only, the real check happens in-pod.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum ContractError {
    /// architecture.py missing `def build_model(`.
    #[error("architecture_py must define build_model(ctx)")]
    MissingBuildModel,
    /// training.py missing `def train(`.
    #[error("training_py must define train(model, ctx)")]
    MissingTrain,
    /// `architecture_py` too large.
    #[error("architecture_py exceeds contract size ({0} bytes > {MAX_SOURCE_BYTES})")]
    ArchitectureTooLarge(usize),
    /// `training_py` too large.
    #[error("training_py exceeds contract size ({0} bytes > {MAX_SOURCE_BYTES})")]
    TrainingTooLarge(usize),
}

/// Per-source byte cap (schema only).
pub const MAX_SOURCE_BYTES: usize = 128 * 1024;

/// Validate contract shape (cheap local screen; in-pod check still authoritative).
///
/// # Errors
/// [`ContractError`] variants.
pub fn check_contract(architecture_py: &str, training_py: &str) -> Result<(), ContractError> {
    if architecture_py.len() > MAX_SOURCE_BYTES {
        return Err(ContractError::ArchitectureTooLarge(architecture_py.len()));
    }
    if training_py.len() > MAX_SOURCE_BYTES {
        return Err(ContractError::TrainingTooLarge(training_py.len()));
    }
    if !architecture_py.contains("def build_model(") {
        return Err(ContractError::MissingBuildModel);
    }
    if !training_py.contains("def train(") {
        return Err(ContractError::MissingTrain);
    }
    Ok(())
}

/// SHA-256 hex of the recipe contract tuple surfaced via the API.
#[must_use]
pub fn recipe_pin_hex() -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(RECIPE_VERSION.as_bytes());
    h.update(MAX_PARAMS.to_le_bytes());
    h.update(DATASET_URL.as_bytes());
    h.update(dataset_sha256().as_bytes());
    h.update(HARNESS_PY.as_bytes());
    hex::encode(h.finalize())
}

/// Public recipe descriptor (`GET /v1/recipe` body minus volatile bits).
#[derive(Debug, Clone, serde::Serialize)]
pub struct RecipeDescriptor {
    /// Semver of the recipe contract.
    pub version: &'static str,
    /// Pinned dataset URL (download+verify by the miner locally).
    pub dataset_url: &'static str,
    /// Canonical HF ref string.
    pub dataset_ref: &'static str,
    /// Pinned dataset SHA-256.
    pub dataset_sha256: String,
    /// Expected size hint (bytes).
    pub dataset_len_bytes: u64,
    /// Train wall seconds cap.
    pub train_hours_cap: f64,
    /// Pod lifetime cap hours.
    pub pod_lifetime_hours_cap: f64,
    /// Hard step stop.
    pub max_train_steps: u32,
    /// Seed lattice.
    pub seed: u64,
    /// Row caps.
    pub val_rows: usize,
    /// Train rows.
    pub train_rows: usize,
    /// Per-source byte cap.
    pub max_source_bytes: usize,
    /// Maximum model parameters after `build_model`.
    pub max_params: u64,
    /// Recipe contract pin (sha256 hex over the tuple).
    pub pin_hex: String,
}

/// Build the public descriptor (deterministic).
#[must_use]
pub fn descriptor() -> RecipeDescriptor {
    RecipeDescriptor {
        version: RECIPE_VERSION,
        dataset_url: DATASET_URL,
        dataset_ref: DATASET_REF,
        dataset_sha256: dataset_sha256(),
        dataset_len_bytes: dataset_len_bytes(),
        train_hours_cap: TRAIN_HOURS_CAP,
        pod_lifetime_hours_cap: POD_LIFETIME_HOURS_CAP,
        max_train_steps: MAX_TRAIN_STEPS,
        seed: RECIPE_SEED,
        val_rows: VAL_ROWS,
        train_rows: TRAIN_ROWS,
        max_source_bytes: MAX_SOURCE_BYTES,
        max_params: MAX_PARAMS,
        pin_hex: recipe_pin_hex(),
    }
}

/// Validate the pinned dataset locally (download+hash) — no network in prod
/// callers; only run from tests or the operator CLI.
///
/// # Errors
/// Transport / hash mismatch.
pub fn verify_dataset_sha256(bytes: &[u8]) -> Result<(), String> {
    use sha2::{Digest, Sha256};
    let got = hex::encode(Sha256::digest(bytes));
    if got != dataset_sha256() {
        return Err(format!(
            "dataset sha256 mismatch: got {got}, want {}",
            dataset_sha256()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn harness_embedded_and_nonempty() {
        assert!(HARNESS_PY.contains("build_model"));
        assert!(HARNESS_PY.contains("METRICS_JSON="));
        assert!(HARNESS_PY.contains("fineweb") || HARNESS_PY.contains("DATASET_URL"));
    }

    #[test]
    fn baseline_satisfies_contract() {
        check_contract(BASELINE_ARCHITECTURE_PY, BASELINE_TRAINING_PY).expect("baseline");
    }

    #[test]
    fn contract_rejects_missing_train() {
        let err = check_contract(BASELINE_ARCHITECTURE_PY, "x = 1").unwrap_err();
        assert!(matches!(err, ContractError::MissingTrain));
    }

    #[test]
    fn caps_match_user_goal() {
        assert!((TRAIN_HOURS_CAP - 6.0).abs() < f64::EPSILON);
        let a = POD_LIFETIME_HOURS_CAP;
        let b = TRAIN_HOURS_CAP;
        assert!(a > b);
        assert_eq!(MAX_PARAMS, 350_000_000);
        assert!(HARNESS_PY.contains("350000000") || HARNESS_PY.contains("PRISM_MAX_PARAMS"));
        assert!(HARNESS_PY.contains("parameter cap"));
    }

    #[test]
    fn descriptor_pin_stable() {
        let a = recipe_pin_hex();
        let b = recipe_pin_hex();
        assert_eq!(a, b);
        assert_eq!(a.len(), 64);
    }
}
