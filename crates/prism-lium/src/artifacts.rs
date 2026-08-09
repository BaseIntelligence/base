//! SSH harvest of trained checkpoints (pod → master).

use std::path::{Path, PathBuf};

use tracing::info;

use crate::ssh::{ssh_exec_bytes, SshTarget};
use crate::LiumError;
use prism_artifacts::{receive_tar_bytes, resolve_checkpoint_budget, ReceiveSource, POD_WORKDIR};

/// SSH-tar `checkpoint.pt` (or sharded index) from the pod into `dest_dir`.
///
/// `n_params` should be the harness-measured count. When `None`, falls back to
/// `prism_recipe::max_params()` so older harness builds still harvest.
pub async fn harvest_checkpoint_ssh(
    target: &SshTarget,
    private_key: &Path,
    dest_dir: &Path,
    submission_id: &str,
    ssh_attempts: u32,
    ssh_retry_secs: u64,
    n_params: Option<u64>,
) -> Result<PathBuf, LiumError> {
    let _ = prism_artifacts::ensure_artifact_root()?;
    let (max_bytes, used_params) = resolve_checkpoint_budget(n_params, prism_recipe::max_params())?;
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
    if packed.len() > max_bytes {
        return Err(LiumError::Integrity(format!(
            "checkpoint pack exceeds budget (packed={} max={max_bytes} bytes; n_params={used_params} FP32×2×1.5)",
            packed.len()
        )));
    }
    let sid = submission_id.to_owned();
    let dest = dest_dir.to_path_buf();
    let (primary, receipt) = tokio::task::spawn_blocking(move || {
        receive_tar_bytes(&sid, &dest, &packed, ReceiveSource::SshHarvest, max_bytes)
    })
    .await
    .map_err(|e| LiumError::Exec(format!("receive join: {e}")))??;
    info!(path = %primary.display(), bytes = receipt.bytes, sha = %receipt.sha256, source = %receipt.source, n_params = used_params, max_bytes, "secure receive: harvested prism checkpoint");
    Ok(primary)
}
