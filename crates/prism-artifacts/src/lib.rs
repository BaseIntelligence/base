//! Master-side Prism checkpoint parking (paths + Sim stubs).
//!
//! Live SSH harvest stays in `prism-lium` (needs the SSH client); this crate
//! holds the shared layout so registry / playground / orchestrator agree on
//! `$PRISM_ARTIFACT_DIR/<submission_id>/checkpoint.pt`.

#![forbid(unsafe_code)]
#![allow(clippy::missing_errors_doc)]

use std::path::PathBuf;

use sha2::{Digest, Sha256};

use prism_lium_types::LiumError;

/// Remote workdir used by the v3 harness (`PRISM_WORKDIR` default).
pub const POD_WORKDIR: &str = "/tmp/prism_eval";

/// Max packed checkpoint bytes accepted over SSH (fail-closed above this).
pub const MAX_CHECKPOINT_BYTES: usize = 2 * 1024 * 1024 * 1024; // 2 GiB

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

/// Write a tiny deterministic stub checkpoint for Sim (CI / local e2e).
pub fn write_sim_checkpoint(dest_dir: &std::path::Path, seed: &[u8]) -> Result<PathBuf, LiumError> {
    std::fs::create_dir_all(dest_dir).map_err(|e| LiumError::Exec(format!("mkdir: {e}")))?;
    let mut h = Sha256::new();
    h.update(b"prism-sim-checkpoint-v1");
    h.update(seed);
    let dig = h.finalize();
    let path = dest_dir.join("checkpoint.pt");
    let mut body = b"PRISM_SIM_CKPT\n".to_vec();
    body.extend_from_slice(&dig);
    std::fs::write(&path, &body).map_err(|e| LiumError::Exec(format!("write: {e}")))?;
    let manifest = serde_json::json!({
        "sha256": hex::encode(dig),
        "bytes": body.len(),
        "path": "checkpoint.pt",
        "sim": true,
    });
    std::fs::write(
        dest_dir.join("MANIFEST.json"),
        serde_json::to_vec_pretty(&manifest).unwrap_or_default(),
    )
    .map_err(|e| LiumError::Exec(format!("manifest: {e}")))?;
    Ok(path)
}

/// Write `MANIFEST.json` next to a harvested primary file.
pub fn write_manifest(primary: &std::path::Path, bytes: &[u8]) -> Result<String, LiumError> {
    let mut h = Sha256::new();
    h.update(bytes);
    let sha = hex::encode(h.finalize());
    let manifest = serde_json::json!({
        "sha256": sha,
        "bytes": bytes.len(),
        "path": primary.file_name().and_then(|s| s.to_str()),
    });
    let dir = primary
        .parent()
        .ok_or_else(|| LiumError::Exec("checkpoint has no parent".into()))?;
    std::fs::write(
        dir.join("MANIFEST.json"),
        serde_json::to_vec_pretty(&manifest).unwrap_or_default(),
    )
    .map_err(|e| LiumError::Exec(format!("manifest: {e}")))?;
    Ok(sha)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn sim_checkpoint_is_deterministic() {
        let dir = std::env::temp_dir().join(format!("prism-art-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let a = write_sim_checkpoint(&dir, b"sub-1").unwrap();
        let bytes_a = std::fs::read(&a).unwrap();
        let b = write_sim_checkpoint(&dir, b"sub-1").unwrap();
        assert_eq!(bytes_a, std::fs::read(b).unwrap());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
