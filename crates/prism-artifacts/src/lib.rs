//! Master-side Prism checkpoint parking + **secure receive hook**.
//!
//! Trust model: the Lium pod is untrusted (miner code ran there). Master
//! **pulls** a tar over SSH, then stages through [`receive_tar_bytes`] which
//! enforces the FP32×2×1.5 size budget ([`checkpoint_byte_budget`]),
//! path-traversal refusal, filename allowlist, and a hashed
//! [`ArtifactReceipt`]. Top-model publish must call [`verify_parked`].
//!
//! Operators may re-stage via `POST /v1/admin/artifacts/{id}/receive` (admin
//! bearer, fail-closed) — never an open upload from the pod.

#![forbid(unsafe_code)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]

mod http;
mod receive;

pub use http::{artifact_get_route, artifact_receive_route, ArtifactReceiveState};
pub use receive::{
    checkpoint_byte_budget, receive_bytes, receive_tar_bytes, resolve_checkpoint_budget,
    validate_submission_id, verify_parked, write_sim_checkpoint, ArtifactReceipt, ReceiveSource,
    ALLOWED_FILENAMES, BF16_BYTES_PER_PARAM, CHECKPOINT_OVERHEAD_DEN, CHECKPOINT_OVERHEAD_NUM,
    FP32_BYTES_PER_PARAM, MAX_CHECKPOINT_BYTES, POD_WORKDIR, RECEIPT_FILE, RECIPE_MAX_PARAMS,
    STATE_DICT_TYING_FACTOR,
};

use std::path::PathBuf;

/// Resolve the master-side artifact root (`PRISM_ARTIFACT_DIR` or default).
#[must_use]
pub fn artifact_root() -> PathBuf {
    std::env::var("PRISM_ARTIFACT_DIR")
        .ok()
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| PathBuf::from("/var/lib/prism/artifacts"))
}

/// Ensure [`artifact_root`] exists and is writable by this process.
///
/// Live failure mode (2026-08-09 private e2e): challenge runs as uid 65532
/// (`base`) while `/var/lib/prism/artifacts` was missing / root-owned →
/// `create_dir_all` → `Permission denied`. Call before SSH harvest so the
/// error names the path and the operator knobs (`PRISM_ARTIFACT_DIR`,
/// image/volume ownership) instead of a bare `mkdir: …`.
pub fn ensure_artifact_root() -> Result<PathBuf, prism_lium_types::LiumError> {
    let root = artifact_root();
    std::fs::create_dir_all(&root).map_err(|e| {
        prism_lium_types::LiumError::Exec(format!(
            "mkdir {}: {e} (PRISM_ARTIFACT_DIR must exist and be writable by the challenge user)",
            root.display()
        ))
    })?;
    let probe = root.join(".prism_write_probe");
    match std::fs::write(&probe, b"ok") {
        Ok(()) => {
            let _ = std::fs::remove_file(&probe);
            Ok(root)
        }
        Err(e) => Err(prism_lium_types::LiumError::Exec(format!(
            "artifact root {} not writable: {e} (fix volume ownership or PRISM_ARTIFACT_DIR)",
            root.display()
        ))),
    }
}

/// Park directory for one submission's harvested weights.
#[must_use]
pub fn artifact_dir_for(submission_id: &str) -> PathBuf {
    artifact_root().join(submission_id)
}

/// Path of the parked checkpoint (may not exist yet).
#[must_use]
pub fn checkpoint_path_for(submission_id: &str) -> PathBuf {
    artifact_dir_for(submission_id).join("checkpoint.pt")
}

/// Write `MANIFEST.json` next to a harvested primary file (legacy helper).
pub fn write_manifest(
    primary: &std::path::Path,
    bytes: &[u8],
) -> Result<String, prism_lium_types::LiumError> {
    receive::write_manifest_inner(primary, bytes)
}
