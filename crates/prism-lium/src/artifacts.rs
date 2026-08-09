//! SSH harvest of trained checkpoints (pod → master).
//!
//! Pull is master-initiated; staging goes through `prism_artifacts::receive_tar_bytes`
//! (size cap, tar member allowlist, path-traversal refuse, hashed receipt).

use std::path::{Path, PathBuf};

use tracing::info;

use crate::ssh::{ssh_exec_bytes, SshTarget};
use crate::LiumError;
use prism_artifacts::{receive_tar_bytes, ReceiveSource, MAX_CHECKPOINT_BYTES, POD_WORKDIR};

/// SSH-tar `checkpoint.pt` (or sharded index) from the pod into `dest_dir`.
pub async fn harvest_checkpoint_ssh(
    target: &SshTarget,
    private_key: &Path,
    dest_dir: &Path,
    submission_id: &str,
    ssh_attempts: u32,
    ssh_retry_secs: u64,
) -> Result<PathBuf, LiumError> {
    let remote = format!(
        "set -e; cd {POD_WORKDIR}; \
         if [ -f checkpoint.pt ]; then tar -c checkpoint.pt; \
         elif [ -f checkpoint.pt.index ]; then tar -c checkpoint.pt.index checkpoint-*.pt 2>/dev/null; \
         else echo 'no checkpoint' >&2; exit 2; fi"
    );
    let packed = ssh_exec_bytes(
        target,
        private_key,
        &remote,
        ssh_attempts,
        ssh_retry_secs,
        600,
    )
    .await?;
    if packed.len() > MAX_CHECKPOINT_BYTES {
        return Err(LiumError::Integrity(format!(
            "checkpoint pack exceeds {MAX_CHECKPOINT_BYTES} bytes"
        )));
    }
    let sid = submission_id.to_owned();
    let dest = dest_dir.to_path_buf();
    let (primary, receipt) = tokio::task::spawn_blocking(move || {
        receive_tar_bytes(&sid, &dest, &packed, ReceiveSource::SshHarvest)
    })
    .await
    .map_err(|e| LiumError::Exec(format!("receive join: {e}")))??;
    info!(
        path = %primary.display(),
        bytes = receipt.bytes,
        sha = %receipt.sha256,
        source = %receipt.source,
        "secure receive: harvested prism checkpoint"
    );
    Ok(primary)
}
