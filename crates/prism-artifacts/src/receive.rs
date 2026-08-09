//! Secure stage/verify for master-parked checkpoints.

use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use prism_lium_types::LiumError;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Remote workdir used by the v3 harness (`PRISM_WORKDIR` default).
pub const POD_WORKDIR: &str = "/tmp/prism_eval";

/// Max packed / uploaded checkpoint bytes (fail-closed above this).
pub const MAX_CHECKPOINT_BYTES: usize = 2 * 1024 * 1024 * 1024; // 2 GiB

/// Receipt filename written next to the checkpoint after a successful stage.
pub const RECEIPT_FILE: &str = "RECEIPT.json";

/// Exact filenames allowed inside a park directory (plus shard pattern).
pub const ALLOWED_FILENAMES: &[&str] = &[
    "checkpoint.pt",
    "checkpoint.pt.index",
    "MANIFEST.json",
    RECEIPT_FILE,
];

/// Provenance recorded on a staged receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReceiveSource {
    /// Master-initiated SSH pull from the eval pod.
    SshHarvest,
    /// Operator `POST /v1/admin/artifacts/.../receive`.
    AdminUpload,
    /// Deterministic Sim stub (CI / local e2e).
    Sim,
}

impl ReceiveSource {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SshHarvest => "ssh_harvest",
            Self::AdminUpload => "admin_upload",
            Self::Sim => "sim",
        }
    }
}

/// Hashed receipt written after a successful secure receive.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArtifactReceipt {
    /// Submission id (validated; park directory name).
    pub submission_id: String,
    /// sha256 of the primary checkpoint bytes.
    pub sha256: String,
    /// Primary file size in bytes.
    pub bytes: usize,
    /// Primary relative filename (`checkpoint.pt` or `.index`).
    pub path: String,
    /// `ssh_harvest` | `admin_upload` | `sim`.
    pub source: String,
    /// Unix seconds when staged on master.
    pub staged_at_unix: u64,
}

/// Refuse path traversal / odd ids before joining under `PRISM_ARTIFACT_DIR`.
pub fn validate_submission_id(id: &str) -> Result<(), LiumError> {
    if id.is_empty() || id.len() > 128 {
        return Err(LiumError::Integrity("submission_id length".into()));
    }
    if !id
        .bytes()
        .all(|b| b.is_ascii_hexdigit() || b == b'-' || b == b'_')
    {
        return Err(LiumError::Integrity(
            "submission_id must be hex / [A-Za-z0-9_-]".into(),
        ));
    }
    if id.contains("..") || id.contains('/') || id.contains('\\') {
        return Err(LiumError::Integrity("submission_id path refuse".into()));
    }
    Ok(())
}

fn is_allowed_member(name: &str) -> bool {
    if ALLOWED_FILENAMES.contains(&name) {
        return true;
    }
    // Sharded torch: checkpoint-00001.pt etc.
    let Some(rest) = name.strip_prefix("checkpoint-") else {
        return false;
    };
    let Some(stem) = rest.strip_suffix(".pt") else {
        return false;
    };
    !stem.is_empty() && stem.bytes().all(|b| b.is_ascii_digit())
}

fn refuse_member_path(name: &str) -> Result<(), LiumError> {
    if name.is_empty() || name.contains('\0') {
        return Err(LiumError::Integrity("empty tar member".into()));
    }
    if name.starts_with('/') || name.starts_with('\\') {
        return Err(LiumError::Integrity("absolute tar member refused".into()));
    }
    let p = Path::new(name);
    if p.components().any(|c| {
        matches!(c, Component::ParentDir | Component::RootDir) || matches!(c, Component::Prefix(_))
    }) {
        return Err(LiumError::Integrity(format!(
            "tar member path refuse: {name}"
        )));
    }
    // Flat park only — no nested dirs.
    if p.components().count() != 1 {
        return Err(LiumError::Integrity(format!(
            "nested tar member refused: {name}"
        )));
    }
    let file = p
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| LiumError::Integrity("non-utf8 tar member".into()))?;
    if !is_allowed_member(file) {
        return Err(LiumError::Integrity(format!(
            "unexpected tar member: {file}"
        )));
    }
    Ok(())
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    hex::encode(h.finalize())
}

pub(crate) fn write_manifest_inner(primary: &Path, bytes: &[u8]) -> Result<String, LiumError> {
    let sha = sha256_hex(bytes);
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

fn write_receipt(dir: &Path, receipt: &ArtifactReceipt) -> Result<(), LiumError> {
    let body = serde_json::to_vec_pretty(receipt)
        .map_err(|e| LiumError::Exec(format!("receipt encode: {e}")))?;
    std::fs::write(dir.join(RECEIPT_FILE), body)
        .map_err(|e| LiumError::Exec(format!("receipt: {e}")))?;
    Ok(())
}

fn primary_in_dir(dir: &Path) -> Result<PathBuf, LiumError> {
    let ckpt = dir.join("checkpoint.pt");
    if ckpt.is_file() {
        return Ok(ckpt);
    }
    let index = dir.join("checkpoint.pt.index");
    if index.is_file() {
        return Ok(index);
    }
    Err(LiumError::Integrity(
        "park has no checkpoint.pt / checkpoint.pt.index".into(),
    ))
}

/// Post-extract audit: no symlinks, only allowlisted names, flat layout.
fn audit_park_dir(dir: &Path) -> Result<PathBuf, LiumError> {
    let meta = std::fs::metadata(dir).map_err(|e| LiumError::Exec(format!("stat park: {e}")))?;
    if !meta.is_dir() {
        return Err(LiumError::Integrity("park is not a directory".into()));
    }
    for ent in std::fs::read_dir(dir).map_err(|e| LiumError::Exec(format!("readdir: {e}")))? {
        let ent = ent.map_err(|e| LiumError::Exec(format!("readdir ent: {e}")))?;
        let name = ent.file_name();
        let name = name
            .to_str()
            .ok_or_else(|| LiumError::Integrity("non-utf8 park entry".into()))?;
        let ft = ent
            .file_type()
            .map_err(|e| LiumError::Exec(format!("filetype: {e}")))?;
        if ft.is_symlink() {
            return Err(LiumError::Integrity(format!(
                "symlink refused in park: {name}"
            )));
        }
        if ft.is_dir() {
            return Err(LiumError::Integrity(format!(
                "nested dir refused in park: {name}"
            )));
        }
        if !is_allowed_member(name) {
            return Err(LiumError::Integrity(format!(
                "unexpected park file: {name}"
            )));
        }
    }
    primary_in_dir(dir)
}

fn finalize_park(
    dest_dir: &Path,
    submission_id: &str,
    source: ReceiveSource,
) -> Result<(PathBuf, ArtifactReceipt), LiumError> {
    let primary = audit_park_dir(dest_dir)?;
    let bytes = std::fs::read(&primary).map_err(|e| LiumError::Exec(format!("read: {e}")))?;
    if bytes.is_empty() {
        return Err(LiumError::Integrity("empty checkpoint".into()));
    }
    if bytes.len() > MAX_CHECKPOINT_BYTES {
        return Err(LiumError::Integrity(format!(
            "checkpoint exceeds {MAX_CHECKPOINT_BYTES} bytes"
        )));
    }
    let sha = write_manifest_inner(&primary, &bytes)?;
    let path_name = primary
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("checkpoint.pt")
        .to_owned();
    let receipt = ArtifactReceipt {
        submission_id: submission_id.to_owned(),
        sha256: sha,
        bytes: bytes.len(),
        path: path_name,
        source: source.as_str().to_owned(),
        staged_at_unix: now_unix(),
    };
    write_receipt(dest_dir, &receipt)?;
    Ok((primary, receipt))
}

/// Stage raw checkpoint bytes (admin upload / single-file path).
pub fn receive_bytes(
    submission_id: &str,
    dest_dir: &Path,
    filename: &str,
    bytes: &[u8],
    expected_sha256: Option<&str>,
    source: ReceiveSource,
) -> Result<(PathBuf, ArtifactReceipt), LiumError> {
    validate_submission_id(submission_id)?;
    refuse_member_path(filename)?;
    if filename == "MANIFEST.json" || filename == RECEIPT_FILE {
        return Err(LiumError::Integrity(
            "cannot upload manifest/receipt as primary".into(),
        ));
    }
    if bytes.is_empty() {
        return Err(LiumError::Integrity("empty upload".into()));
    }
    if bytes.len() > MAX_CHECKPOINT_BYTES {
        return Err(LiumError::Integrity(format!(
            "upload exceeds {MAX_CHECKPOINT_BYTES} bytes"
        )));
    }
    let got = sha256_hex(bytes);
    if let Some(want) = expected_sha256 {
        let want = want.trim().to_ascii_lowercase();
        if want.len() != 64 || !want.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(LiumError::Integrity("X-Prism-Sha256 must be 64 hex".into()));
        }
        if got != want {
            return Err(LiumError::Integrity("sha256 mismatch".into()));
        }
    }
    // Fresh park — wipe prior contents (fail-closed re-stage).
    if dest_dir.exists() {
        std::fs::remove_dir_all(dest_dir).map_err(|e| LiumError::Exec(format!("wipe: {e}")))?;
    }
    std::fs::create_dir_all(dest_dir).map_err(|e| LiumError::Exec(format!("mkdir: {e}")))?;
    let primary = dest_dir.join(filename);
    std::fs::write(&primary, bytes).map_err(|e| LiumError::Exec(format!("write: {e}")))?;
    finalize_park(dest_dir, submission_id, source)
}

/// List tar members, refuse unsafe paths, extract, audit, write receipt.
pub fn receive_tar_bytes(
    submission_id: &str,
    dest_dir: &Path,
    packed: &[u8],
    source: ReceiveSource,
) -> Result<(PathBuf, ArtifactReceipt), LiumError> {
    validate_submission_id(submission_id)?;
    if packed.is_empty() {
        return Err(LiumError::Integrity("empty checkpoint harvest".into()));
    }
    if packed.len() > MAX_CHECKPOINT_BYTES {
        return Err(LiumError::Integrity(format!(
            "checkpoint pack exceeds {MAX_CHECKPOINT_BYTES} bytes"
        )));
    }
    // List members first (fail-closed before extract).
    let list = Command::new("tar")
        .args(["-t"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write as _;
            if let Some(mut stdin) = child.stdin.take() {
                stdin.write_all(packed)?;
            }
            child.wait_with_output()
        })
        .map_err(|e| LiumError::Exec(format!("tar -t: {e}")))?;
    if !list.status.success() {
        return Err(LiumError::Exec(format!(
            "tar -t: {}",
            String::from_utf8_lossy(&list.stderr)
        )));
    }
    let listing = String::from_utf8_lossy(&list.stdout);
    let mut saw_primary = false;
    for line in listing.lines() {
        let name = line.trim().trim_end_matches('/');
        if name.is_empty() {
            continue;
        }
        refuse_member_path(name)?;
        if name == "checkpoint.pt" || name == "checkpoint.pt.index" {
            saw_primary = true;
        }
    }
    if !saw_primary {
        return Err(LiumError::Integrity(
            "tar has no checkpoint.pt / checkpoint.pt.index".into(),
        ));
    }
    if dest_dir.exists() {
        std::fs::remove_dir_all(dest_dir).map_err(|e| LiumError::Exec(format!("wipe: {e}")))?;
    }
    std::fs::create_dir_all(dest_dir).map_err(|e| LiumError::Exec(format!("mkdir: {e}")))?;
    let extract = Command::new("tar")
        .args(["-x", "-C"])
        .arg(dest_dir)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write as _;
            if let Some(mut stdin) = child.stdin.take() {
                stdin.write_all(packed)?;
            }
            child.wait_with_output()
        })
        .map_err(|e| LiumError::Exec(format!("tar -x: {e}")))?;
    if !extract.status.success() {
        let _ = std::fs::remove_dir_all(dest_dir);
        return Err(LiumError::Exec(format!(
            "tar extract: {}",
            String::from_utf8_lossy(&extract.stderr)
        )));
    }
    match finalize_park(dest_dir, submission_id, source) {
        Ok(v) => Ok(v),
        Err(e) => {
            let _ = std::fs::remove_dir_all(dest_dir);
            Err(e)
        }
    }
}

/// Verify an existing park: `RECEIPT.json` present and sha256 matches bytes.
pub fn verify_parked(submission_id: &str) -> Result<ArtifactReceipt, LiumError> {
    validate_submission_id(submission_id)?;
    let dir = crate::artifact_dir_for(submission_id);
    let receipt_path = dir.join(RECEIPT_FILE);
    let raw = std::fs::read(&receipt_path).map_err(|_| {
        LiumError::Integrity(format!(
            "missing {RECEIPT_FILE}: secure receive incomplete for {submission_id}"
        ))
    })?;
    let receipt: ArtifactReceipt = serde_json::from_slice(&raw)
        .map_err(|e| LiumError::Integrity(format!("receipt decode: {e}")))?;
    if receipt.submission_id != submission_id {
        return Err(LiumError::Integrity(
            "receipt submission_id mismatch".into(),
        ));
    }
    refuse_member_path(&receipt.path)?;
    let primary = dir.join(&receipt.path);
    // Ensure resolved path stays under park dir.
    let base = dir
        .canonicalize()
        .map_err(|e| LiumError::Exec(format!("canon park: {e}")))?;
    let canon = primary
        .canonicalize()
        .map_err(|e| LiumError::Integrity(format!("canon primary: {e}")))?;
    if !canon.starts_with(&base) {
        return Err(LiumError::Integrity("primary escaped park dir".into()));
    }
    let bytes = std::fs::read(&canon).map_err(|e| LiumError::Exec(format!("read: {e}")))?;
    if bytes.len() != receipt.bytes {
        return Err(LiumError::Integrity("receipt bytes mismatch".into()));
    }
    let got = sha256_hex(&bytes);
    if got != receipt.sha256 {
        return Err(LiumError::Integrity("receipt sha256 mismatch".into()));
    }
    Ok(receipt)
}

/// Write a tiny deterministic stub checkpoint for Sim (CI / local e2e).
///
/// `dest_dir` must be `$PRISM_ARTIFACT_DIR/<submission_id>` (hex id).
pub fn write_sim_checkpoint(dest_dir: &Path, seed: &[u8]) -> Result<PathBuf, LiumError> {
    let submission_id = dest_dir
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| LiumError::Integrity("park dir missing name".into()))?;
    validate_submission_id(submission_id)?;
    let mut h = Sha256::new();
    h.update(b"prism-sim-checkpoint-v1");
    h.update(seed);
    let dig = h.finalize();
    let mut body = b"PRISM_SIM_CKPT\n".to_vec();
    body.extend_from_slice(&dig);
    let (path, _) = receive_bytes(
        submission_id,
        dest_dir,
        "checkpoint.pt",
        &body,
        None,
        ReceiveSource::Sim,
    )?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "prism-art-{}-{}-{}",
            name,
            std::process::id(),
            now_unix()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn rejects_path_traversal_id() {
        assert!(validate_submission_id("../etc").is_err());
        assert!(validate_submission_id("a/b").is_err());
        assert!(validate_submission_id("").is_err());
    }

    fn with_artifact_root<T>(root: &Path, f: impl FnOnce() -> T) -> T {
        // Safety: unit tests are single-threaded per process for this env key.
        std::env::set_var("PRISM_ARTIFACT_DIR", root);
        let out = f();
        std::env::remove_var("PRISM_ARTIFACT_DIR");
        out
    }

    #[test]
    fn sim_checkpoint_is_deterministic() {
        let dir = tmp("sim");
        let sid = "aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899";
        let park = dir.join(sid);
        with_artifact_root(&dir, || {
            let a = write_sim_checkpoint(&park, b"sub-1").unwrap();
            let bytes_a = std::fs::read(&a).unwrap();
            let b = write_sim_checkpoint(&park, b"sub-1").unwrap();
            assert_eq!(bytes_a, std::fs::read(b).unwrap());
            let r = verify_parked(sid).unwrap();
            assert_eq!(r.source, "sim");
        });
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn receive_bytes_requires_matching_sha() {
        let dir = tmp("sha");
        let sid = "11223344556677889900aabbccddeeff11223344556677889900aabbccddeeff";
        let park = dir.join(sid);
        let body = b"weights-v1";
        let good = sha256_hex(body);
        receive_bytes(
            sid,
            &park,
            "checkpoint.pt",
            body,
            Some(&good),
            ReceiveSource::AdminUpload,
        )
        .unwrap();
        let err = receive_bytes(
            sid,
            &park,
            "checkpoint.pt",
            body,
            Some("0000000000000000000000000000000000000000000000000000000000000000"),
            ReceiveSource::AdminUpload,
        )
        .unwrap_err();
        assert!(format!("{err}").contains("sha256"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tar_stages_and_refuses_bad_members() {
        let dir = tmp("tar");
        let sid = "99aabbccddeeff00112233445566778899aabbccddeeff001122334455667788";
        let park = dir.join(sid);
        let staging = dir.join("stage");
        std::fs::create_dir_all(&staging).unwrap();
        std::fs::write(staging.join("checkpoint.pt"), b"ok").unwrap();
        let packed = Command::new("tar")
            .args(["-c", "-C"])
            .arg(&staging)
            .arg("checkpoint.pt")
            .output()
            .unwrap();
        assert!(packed.status.success());
        with_artifact_root(&dir, || {
            let (path, receipt) =
                receive_tar_bytes(sid, &park, &packed.stdout, ReceiveSource::SshHarvest).unwrap();
            assert!(path.ends_with("checkpoint.pt"));
            assert_eq!(receipt.source, "ssh_harvest");
            verify_parked(sid).unwrap();
        });
        assert!(refuse_member_path("../checkpoint.pt").is_err());
        assert!(refuse_member_path("/tmp/x").is_err());
        assert!(refuse_member_path("evil.bin").is_err());
        assert!(refuse_member_path("checkpoint-00001.pt").is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
