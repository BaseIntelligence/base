//! Master-side Prism checkpoint parking + **secure receive hook**.
//!
//! Trust model: the Lium pod is untrusted (miner code ran there). Master
//! **pulls** a tar over SSH, then stages through [`receive_tar_bytes`] which
//! enforces size caps, path-traversal refusal, filename allowlist, and a
//! hashed [`ArtifactReceipt`]. Top-model publish must call [`verify_parked`].
//!
//! Operators may re-stage via `POST /v1/admin/artifacts/{id}/receive` (admin
//! bearer, fail-closed) — never an open upload from the pod.

#![forbid(unsafe_code)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]

mod http;
mod receive;

pub use http::{artifact_get_route, artifact_receive_route};
pub use receive::{
    receive_bytes, receive_tar_bytes, validate_submission_id, verify_parked, write_sim_checkpoint,
    ArtifactReceipt, ReceiveSource, ALLOWED_FILENAMES, MAX_CHECKPOINT_BYTES, POD_WORKDIR,
    RECEIPT_FILE,
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
