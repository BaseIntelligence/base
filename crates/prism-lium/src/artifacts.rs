//! SSH harvest of trained checkpoints (pod → master). Paths/Sim stubs live
//! in `prism-artifacts`.

use std::path::{Path, PathBuf};

use tracing::info;

use crate::ssh::{ssh_exec_bytes, SshTarget};
use crate::LiumError;
use prism_artifacts::{write_manifest, MAX_CHECKPOINT_BYTES, POD_WORKDIR};

/// SSH-tar `checkpoint.pt` (or sharded index) from the pod into `dest_dir`.
pub async fn harvest_checkpoint_ssh(
    target: &SshTarget,
    private_key: &Path,
    dest_dir: &Path,
    ssh_attempts: u32,
    ssh_retry_secs: u64,
) -> Result<PathBuf, LiumError> {
    std::fs::create_dir_all(dest_dir).map_err(|e| LiumError::Exec(format!("mkdir: {e}")))?;
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
    if packed.is_empty() {
        return Err(LiumError::Exec("empty checkpoint harvest".into()));
    }
    if packed.len() > MAX_CHECKPOINT_BYTES {
        return Err(LiumError::Exec(format!(
            "checkpoint pack exceeds {MAX_CHECKPOINT_BYTES} bytes"
        )));
    }
    {
        use tokio::io::AsyncWriteExt as _;
        let mut child = tokio::process::Command::new("tar")
            .args(["-x", "-C"])
            .arg(dest_dir)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| LiumError::Exec(format!("tar spawn: {e}")))?;
        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(&packed)
                .await
                .map_err(|e| LiumError::Exec(format!("tar stdin: {e}")))?;
        }
        let out = child
            .wait_with_output()
            .await
            .map_err(|e| LiumError::Exec(format!("tar wait: {e}")))?;
        if !out.status.success() {
            return Err(LiumError::Exec(format!(
                "tar extract: {}",
                String::from_utf8_lossy(&out.stderr)
            )));
        }
    }
    let ckpt = dest_dir.join("checkpoint.pt");
    let primary = if ckpt.is_file() {
        ckpt
    } else {
        let index = dest_dir.join("checkpoint.pt.index");
        if !index.is_file() {
            return Err(LiumError::Exec(
                "harvest produced no checkpoint file".into(),
            ));
        }
        index
    };
    let bytes = std::fs::read(&primary).map_err(|e| LiumError::Exec(format!("read: {e}")))?;
    let sha = write_manifest(&primary, &bytes)?;
    info!(
        path = %primary.display(),
        bytes = bytes.len(),
        %sha,
        "harvested prism checkpoint"
    );
    Ok(primary)
}
